//! Per-filing dispatch planning — read `processed/filing_index.csv`,
//! apply scope filters (companies / year range / form types), and
//! group filings by [`SecFormBucket`] so the binding's downstream
//! batch fetcher loop iterates a clean plan.
//!
//! The CSV-reading + filtering + bucketing is generic across every
//! binding; this module is the canonical implementation. The binding
//! still owns the **execution** step (calling the appropriate fetcher
//! per bucket) because each binding has its own progress idiom and
//! its own runtime layer around the core `fetch_*_filing` functions.
//!
//! Lifted from the Python wrapper's
//! `_dispatch_per_filing_fetches` in the 2026-05-25 binding prep.
//! The execution half stays in the wrapper for now (see
//! `docs/internal/consider-for-future.md` → "SEC dispatch execution
//! lift" for the deferred follow-up).

use std::collections::HashMap;

use crate::datasets::sec::buckets::SecFormBucket;
use crate::datasets::sec::error::{Result, SecError};
use crate::datasets::sec::layout::Workdir;

/// One filing's identifying tuple: the issuer CIK, the SEC accession
/// number (dashed form, e.g. `"0000320193-25-000123"`), and the
/// primary-document filename. All three are needed by the various
/// per-filing fetchers (Form 4 / 13F / 8-K / etc.) — some use all
/// three, some use just `(cik, accession)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilingTask {
    pub cik: u64,
    pub accession_dashed: String,
    pub primary_document: String,
}

/// Filter spec for the dispatch plan. None / empty means "no filter
/// on this axis" — see [`prepare_dispatch_plan`] for the semantics.
#[derive(Debug, Clone)]
pub struct DispatchScope {
    /// Restrict to these CIKs. `None` = all CIKs in the index.
    pub companies: Option<Vec<u64>>,
    /// Lower bound (inclusive) for the filed-year. The caller
    /// resolves the `(year_range or detailed-window)` choice.
    pub year_lo: u16,
    /// Upper bound (inclusive) for the filed-year.
    pub year_hi: u16,
}

/// Grouped filings ready for per-bucket batch fetching. The
/// `by_bucket` map has one entry per `SecFormBucket` that had at
/// least one matching filing in the index; buckets with zero
/// matches are omitted, not present-with-empty-vec.
#[derive(Debug, Clone, Default)]
pub struct DispatchPlan {
    pub by_bucket: HashMap<SecFormBucket, Vec<FilingTask>>,
}

impl DispatchPlan {
    /// True when no bucket has any filings (the binding can
    /// short-circuit the execution step).
    pub fn is_empty(&self) -> bool {
        self.by_bucket.values().all(|v| v.is_empty())
    }

    /// Total filings across all buckets — convenient for verbose
    /// progress prints in the binding.
    pub fn total_filings(&self) -> usize {
        self.by_bucket.values().map(|v| v.len()).sum()
    }

    /// Sorted list of distinct CIKs across every bucket. Used by
    /// the XBRL `fetch_company_facts_batch` path which fetches
    /// per-CIK (not per-filing).
    pub fn distinct_ciks(&self) -> Vec<u64> {
        let mut set = std::collections::BTreeSet::new();
        for tasks in self.by_bucket.values() {
            for t in tasks {
                set.insert(t.cik);
            }
        }
        set.into_iter().collect()
    }
}

/// Read `<workdir>/processed/filing_index.csv`, apply the scope
/// filters, group surviving rows by their form bucket via
/// [`SecFormBucket::from_form_string`].
///
/// CSV-row rules:
///
/// - Missing or unparseable `cik` field → skipped silently
/// - Missing `filed_date` or non-numeric year prefix → skipped
/// - Year outside `[scope.year_lo, scope.year_hi]` → skipped
/// - Missing or empty `accession_number` → skipped
/// - `form_type` with no matching bucket → skipped (the binding's
///   `_resolve_fetch_buckets` already filtered active buckets;
///   anything unknown here is genuinely out of scope)
///
/// Returns `Ok(plan)` even when the CSV is missing (the binding can
/// check `plan.is_empty()` to decide whether to skip the dispatch).
/// Returns `Err` only on I/O errors actually reading the existing
/// file or on malformed CSV structure.
pub fn prepare_dispatch_plan(workdir: &Workdir, scope: &DispatchScope) -> Result<DispatchPlan> {
    let csv_path = workdir.processed_csv("filing_index");
    if !csv_path.is_file() {
        // Cold workdir — the wrapper hasn't run extraction yet. Empty
        // plan is the right answer; the caller short-circuits.
        return Ok(DispatchPlan::default());
    }

    let cik_filter: Option<std::collections::HashSet<u64>> = scope
        .companies
        .as_ref()
        .filter(|v| !v.is_empty())
        .map(|v| v.iter().copied().collect());

    let mut rdr = csv::Reader::from_path(&csv_path).map_err(|e| {
        SecError::Io(std::io::Error::other(format!(
            "failed to open filing_index.csv: {e}"
        )))
    })?;

    let mut plan = DispatchPlan::default();
    for record in rdr.deserialize::<FilingIndexRow>() {
        let Ok(row) = record else {
            continue; // skip malformed rows silently — matches Python wrapper
        };
        let Ok(cik) = row.cik.parse::<u64>() else {
            continue;
        };
        if let Some(ref allowed) = cik_filter {
            if !allowed.contains(&cik) {
                continue;
            }
        }
        if row.filed_date.len() < 4 {
            continue;
        }
        let Ok(year) = row.filed_date[..4].parse::<u16>() else {
            continue;
        };
        if year < scope.year_lo || year > scope.year_hi {
            continue;
        }
        if row.accession_number.is_empty() {
            continue;
        }
        let Some(bucket) = SecFormBucket::from_form_string(&row.form_type) else {
            continue;
        };
        plan.by_bucket.entry(bucket).or_default().push(FilingTask {
            cik,
            accession_dashed: row.accession_number,
            primary_document: row.primary_document,
        });
    }

    Ok(plan)
}

// The CSV schema is the orchestrator's `filing_index.csv` writeout;
// see `crates/kglite/src/datasets/sec/extract/orchestrator.rs`.
// Extra fields in the file are tolerated — `serde(deny_unknown_fields)`
// is deliberately NOT set so adding columns to the index doesn't break
// dispatch.
#[derive(serde::Deserialize)]
struct FilingIndexRow {
    cik: String,
    filed_date: String,
    form_type: String,
    #[serde(default)]
    primary_document: String,
    accession_number: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn make_workdir_with_index(rows: &str) -> tempfile::TempDir {
        let tmp = tempdir().unwrap();
        let processed = tmp.path().join("processed");
        fs::create_dir_all(&processed).unwrap();
        let csv = processed.join("filing_index.csv");
        let mut body = String::from("cik,filed_date,form_type,primary_document,accession_number\n");
        body.push_str(rows);
        fs::write(&csv, body).unwrap();
        tmp
    }

    #[test]
    fn missing_csv_returns_empty_plan() {
        let tmp = tempdir().unwrap();
        let wd = Workdir::new(tmp.path().to_path_buf());
        let plan = prepare_dispatch_plan(
            &wd,
            &DispatchScope {
                companies: None,
                year_lo: 2020,
                year_hi: 2024,
            },
        )
        .unwrap();
        assert!(plan.is_empty());
    }

    #[test]
    fn groups_by_bucket() {
        let tmp = make_workdir_with_index(
            "320193,2023-01-15,4,doc.xml,0000320193-23-000001\n\
             789019,2023-02-20,8-K,form8k.htm,0000789019-23-000002\n\
             1234,2023-03-10,SC 13D,sc13d.htm,0000001234-23-000003\n",
        );
        let wd = Workdir::new(tmp.path().to_path_buf());
        let plan = prepare_dispatch_plan(
            &wd,
            &DispatchScope {
                companies: None,
                year_lo: 2020,
                year_hi: 2024,
            },
        )
        .unwrap();
        assert_eq!(plan.by_bucket.len(), 3);
        assert_eq!(plan.by_bucket[&SecFormBucket::Form4].len(), 1);
        assert_eq!(plan.by_bucket[&SecFormBucket::Form8k].len(), 1);
        assert_eq!(plan.by_bucket[&SecFormBucket::Sc13d].len(), 1);
    }

    #[test]
    fn cik_filter_restricts() {
        let tmp = make_workdir_with_index(
            "320193,2023-01-15,4,doc.xml,acc1\n\
             789019,2023-01-15,4,doc.xml,acc2\n\
             1234,2023-01-15,4,doc.xml,acc3\n",
        );
        let wd = Workdir::new(tmp.path().to_path_buf());
        let plan = prepare_dispatch_plan(
            &wd,
            &DispatchScope {
                companies: Some(vec![320193, 1234]),
                year_lo: 2020,
                year_hi: 2024,
            },
        )
        .unwrap();
        assert_eq!(plan.by_bucket[&SecFormBucket::Form4].len(), 2);
    }

    #[test]
    fn year_range_filter() {
        let tmp = make_workdir_with_index(
            "320193,2018-06-01,4,doc.xml,old\n\
             320193,2023-06-01,4,doc.xml,recent\n\
             320193,2026-06-01,4,doc.xml,future\n",
        );
        let wd = Workdir::new(tmp.path().to_path_buf());
        let plan = prepare_dispatch_plan(
            &wd,
            &DispatchScope {
                companies: None,
                year_lo: 2020,
                year_hi: 2024,
            },
        )
        .unwrap();
        let tasks = &plan.by_bucket[&SecFormBucket::Form4];
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].accession_dashed, "recent");
    }

    #[test]
    fn unknown_form_skipped() {
        let tmp = make_workdir_with_index(
            "320193,2023-01-15,4,doc.xml,acc1\n\
             789019,2023-01-15,UNKNOWN-99,whatever.htm,acc2\n",
        );
        let wd = Workdir::new(tmp.path().to_path_buf());
        let plan = prepare_dispatch_plan(
            &wd,
            &DispatchScope {
                companies: None,
                year_lo: 2020,
                year_hi: 2024,
            },
        )
        .unwrap();
        assert_eq!(plan.by_bucket.len(), 1);
        assert_eq!(plan.by_bucket[&SecFormBucket::Form4].len(), 1);
    }

    #[test]
    fn distinct_ciks_dedup_across_buckets() {
        let tmp = make_workdir_with_index(
            "320193,2023-01-15,4,doc.xml,acc1\n\
             320193,2023-02-15,8-K,doc.htm,acc2\n\
             789019,2023-03-15,4,doc.xml,acc3\n",
        );
        let wd = Workdir::new(tmp.path().to_path_buf());
        let plan = prepare_dispatch_plan(
            &wd,
            &DispatchScope {
                companies: None,
                year_lo: 2020,
                year_hi: 2024,
            },
        )
        .unwrap();
        assert_eq!(plan.distinct_ciks(), vec![320193, 789019]);
    }

    #[test]
    fn malformed_rows_skipped() {
        let tmp = make_workdir_with_index(
            "abc,2023-01-15,4,doc.xml,acc1\n\
             320193,,4,doc.xml,acc2\n\
             320193,2023-01-15,4,doc.xml,\n\
             320193,2023-01-15,4,doc.xml,valid_acc\n",
        );
        let wd = Workdir::new(tmp.path().to_path_buf());
        let plan = prepare_dispatch_plan(
            &wd,
            &DispatchScope {
                companies: None,
                year_lo: 2020,
                year_hi: 2024,
            },
        )
        .unwrap();
        let tasks = &plan.by_bucket[&SecFormBucket::Form4];
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].accession_dashed, "valid_acc");
    }
}
