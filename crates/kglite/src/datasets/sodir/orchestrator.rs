//! Top-level refresh orchestrator — drives the index, the ArcGIS
//! client, the fetcher and the preprocessor.
//!
//! `refresh_csvs` runs two passes (ported from the Python `wrapper.py`
//! `_refresh_csvs`):
//!
//! 1. **classify** (sequential, no network) — per dataset, decide
//!    skip / probe / fetch / user-supplied / unfetchable.
//! 2. **execute** (concurrent) — run probes + fetches across a bounded
//!    pool of scoped worker threads that pull from a shared queue. The
//!    `ArcGISClient`'s Arc-shared rate gate is the real throughput
//!    gate (global across the per-worker clones); concurrency only
//!    overlaps latency. Results flow back over an `mpsc` channel; the
//!    main thread owns all `index` mutation and flushes the index
//!    after every completion so a Ctrl-C never loses progress.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;

use crate::datasets::sodir::catalog;
use crate::datasets::sodir::client::ArcGISClient;
use crate::datasets::sodir::error::Result;
use crate::datasets::sodir::fetch;
use crate::datasets::sodir::index::{self, Action, DatasetEntry, SodirIndex};
use crate::datasets::sodir::layout::Workdir;
use crate::datasets::sodir::preprocess::{self, PreprocessReport};

/// Outcome of a CSV refresh pass — every needed stem lands in exactly
/// one bucket.
#[derive(Debug, Clone, Default)]
pub struct RefreshReport {
    /// Datasets downloaded fresh.
    pub fetched: Vec<String>,
    /// Datasets probed and found unchanged.
    pub unchanged: Vec<String>,
    /// User-supplied CSVs (not in the REST catalog).
    pub user_supplied: Vec<String>,
    /// Datasets left as-is (within cooldown).
    pub cached: Vec<String>,
    /// Blueprint datasets absent from the catalog and not pre-supplied.
    pub unfetchable: Vec<String>,
    /// Per-dataset fetch failures `(stem, message)`.
    pub errors: Vec<(String, String)>,
}

/// `refresh_csvs` + `preprocess::apply` combined.
#[derive(Debug, Clone, Default)]
pub struct FetchAllReport {
    pub refresh: RefreshReport,
    pub preprocess: PreprocessReport,
}

/// What a single execute task produced.
enum ExecResult {
    Fetched { rows: u64, elapsed: f64 },
    Unchanged,
}

/// Refresh the CSVs for `needed`, mutating `index` in place. Returns a
/// per-dataset classification report. Sets `last_full_check_iso` when
/// a cooldown sweep ran.
pub fn refresh_csvs(
    workdir: &Workdir,
    client: &ArcGISClient,
    needed: &[String],
    index: &mut SodirIndex,
    index_cooldown_days: i64,
    dataset_cooldown_days: i64,
    concurrency: usize,
) -> Result<RefreshReport> {
    let mut report = RefreshReport::default();
    let sweep_due = index::sweep_due(index.last_full_check_iso.as_deref(), index_cooldown_days);

    // ── Pass 1: classify (no network) ──
    let mut stems: Vec<String> = needed.to_vec();
    stems.sort();
    stems.dedup();

    let mut work: Vec<(String, Action)> = Vec::new();
    for stem in &stems {
        let csv_path = workdir.csv_path(stem);
        if !catalog::is_known(stem) {
            if csv_path.is_file() {
                report.user_supplied.push(stem.clone());
                index.datasets.entry(stem.clone()).or_insert_with(|| {
                    DatasetEntry::user_supplied(
                        stem,
                        index::quick_row_count(&csv_path),
                        &index::now_iso(),
                    )
                });
            } else {
                report.unfetchable.push(stem.clone());
            }
            continue;
        }
        match index::decide_action(
            index.datasets.get(stem),
            &csv_path,
            sweep_due,
            dataset_cooldown_days,
        ) {
            Action::Skip => report.cached.push(stem.clone()),
            action => work.push((stem.clone(), action)),
        }
    }

    // Largest datasets first — the long pole (wellbore, seismic_*)
    // should not be tail-of-queue. Fresh runs have row_count 0
    // everywhere → stable alpha order.
    work.sort_by_key(|item| std::cmp::Reverse(size_hint(index, &item.0)));

    // Snapshot each item's prior row count now — the workers can't
    // borrow `index` (the main thread owns its mutation), so the count
    // `execute_one` compares against on a probe is captured up front.
    let work: Vec<(String, Action, Option<u64>)> = work
        .into_iter()
        .map(|(stem, action)| {
            let prior = index.datasets.get(&stem).map(|e| e.row_count);
            (stem, action, prior)
        })
        .collect();

    // ── Pass 2: execute across a bounded scoped worker pool ──
    //
    // Workers pull from the sorted `work` list via a shared atomic
    // cursor (largest-first order preserved), run the probe/fetch, and
    // send each result back over `mpsc`. The main thread does *all*
    // `index` mutation and the per-completion `index::save`. `abort`
    // makes workers stop picking new work once the main thread hits a
    // fatal error (a `save` or catalog-resolution failure); in-flight
    // fetches finish, but no new ones start — matching the fail-fast
    // semantics of the prior JoinSet drop.
    if !work.is_empty() {
        let n_workers = concurrency.max(1);
        let next = AtomicUsize::new(0);
        let abort = AtomicBool::new(false);

        let outcome: Result<()> = std::thread::scope(|scope| {
            let (tx, rx) = mpsc::channel::<(String, Result<ExecResult>)>();
            for _ in 0..n_workers {
                let client = client.clone();
                let tx = tx.clone();
                let next = &next;
                let abort = &abort;
                let work = &work;
                scope.spawn(move || {
                    loop {
                        if abort.load(Ordering::Relaxed) {
                            break;
                        }
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some((stem, action, prior_count)) = work.get(i) else {
                            break;
                        };
                        let csv_path = workdir.csv_path(stem);
                        let result = execute_one(&client, stem, *action, &csv_path, *prior_count);
                        // Receiver hung up (main aborted) — stop.
                        if tx.send((stem.clone(), result)).is_err() {
                            break;
                        }
                    }
                });
            }
            // Drop the main thread's sender so `rx` closes once every
            // worker's clone is dropped (all work drained).
            drop(tx);

            for (stem, result) in rx {
                match result {
                    Ok(ExecResult::Fetched { rows, elapsed }) => {
                        match build_fetched_entry(&stem, rows, elapsed) {
                            Ok(entry) => {
                                index.datasets.insert(stem.clone(), entry);
                                report.fetched.push(stem);
                            }
                            Err(e) => {
                                abort.store(true, Ordering::Relaxed);
                                return Err(e);
                            }
                        }
                    }
                    Ok(ExecResult::Unchanged) => {
                        if let Some(entry) = index.datasets.get_mut(&stem) {
                            entry.count_checked_at_iso = index::now_iso();
                        }
                        report.unchanged.push(stem);
                    }
                    Err(e) => report.errors.push((stem, e.to_string())),
                }
                // Flush after every completion so a Ctrl-C never loses
                // progress — the next run resumes from here.
                if let Err(e) = index::save(workdir, index) {
                    abort.store(true, Ordering::Relaxed);
                    return Err(e);
                }
            }
            Ok(())
        });
        outcome?;
    }

    if sweep_due {
        index.last_full_check_iso = Some(index::now_iso());
    }
    Ok(report)
}

/// Run one classified dataset: a probe (which upgrades to a fetch when
/// the remote count drifted) or a direct fetch.
fn execute_one(
    client: &ArcGISClient,
    stem: &str,
    action: Action,
    csv_path: &std::path::Path,
    prior_count: Option<u64>,
) -> Result<ExecResult> {
    let mut action = action;
    if action == Action::Probe {
        let remote = fetch::count(client, stem)?;
        if prior_count == Some(remote) {
            return Ok(ExecResult::Unchanged);
        }
        action = Action::Fetch; // count drifted — download fresh
    }
    debug_assert_eq!(action, Action::Fetch);
    let t0 = std::time::Instant::now();
    let rows = fetch::fetch_to_csv(client, stem, csv_path)?;
    Ok(ExecResult::Fetched {
        rows: rows as u64,
        elapsed: t0.elapsed().as_secs_f64(),
    })
}

/// Build the index entry for a freshly fetched dataset. Resolving the
/// catalog metadata can only fail for a stem that isn't in the catalog
/// — impossible here (classify already gated on `catalog::is_known`) —
/// so this is defensive; a failure is fatal (fail-fast) exactly as the
/// prior inline `?` was.
fn build_fetched_entry(stem: &str, rows: u64, elapsed: f64) -> Result<DatasetEntry> {
    let (base, layer_id) = catalog::resolve(stem)?;
    let kind = catalog::kind_of(stem)?;
    Ok(DatasetEntry::fetched(
        kind.as_str(),
        layer_id,
        base,
        stem,
        rows,
        elapsed,
        &index::now_iso(),
    ))
}

/// Best-known dataset size for scheduling — the prior fetch's row
/// count, or 0 (unknown → sorts last).
fn size_hint(index: &SodirIndex, stem: &str) -> u64 {
    index.datasets.get(stem).map(|e| e.row_count).unwrap_or(0)
}

/// Full refresh: ensure the workdir, load the index, refresh every
/// needed CSV, persist the index, then run the FK preprocessing.
pub fn fetch_all(
    workdir: &Workdir,
    needed: &[String],
    index_cooldown_days: i64,
    dataset_cooldown_days: i64,
    concurrency: usize,
) -> Result<FetchAllReport> {
    workdir.ensure_dirs()?;
    let client = ArcGISClient::new()?;
    let mut index = index::load(workdir)?;
    let refresh = refresh_csvs(
        workdir,
        &client,
        needed,
        &mut index,
        index_cooldown_days,
        dataset_cooldown_days,
        concurrency,
    )?;
    index::save(workdir, &index)?;
    let preprocess = preprocess::apply(&workdir.csv_dir())?;
    Ok(FetchAllReport {
        refresh,
        preprocess,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_hint_uses_prior_row_count() {
        let mut idx = SodirIndex::default();
        idx.datasets.insert(
            "field".to_string(),
            DatasetEntry::fetched(
                "layer",
                7100,
                "http://x",
                "field",
                87,
                1.0,
                &index::now_iso(),
            ),
        );
        assert_eq!(size_hint(&idx, "field"), 87);
        assert_eq!(size_hint(&idx, "missing"), 0);
    }
}
