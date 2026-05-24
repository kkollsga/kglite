//! PyO3 bindings exposing the SEC EDGAR loader as `kglite._sec_internal`.
//!
//! Two surfaces, kept deliberately thin:
//!
//! 1. **Fetch helpers** (`fetch_raw`, `fetch_form4_batch`,
//!    `fetch_13f_batch`, `fetch_filing_batch`, `fetch_exhibit21_batch`)
//!    download SEC documents into `raw/` under a single SecClient that
//!    enforces the 10 req/s SEC rate limit. The Python wrapper invokes
//!    these in the order dictated by form-type dependencies.
//! 2. **Feature extraction** is exposed as ONE function:
//!    `extract_all_py(workdir, *, force, cik_list, form_types, year_range)`.
//!    It calls the Rust orchestrator `kglite_core::datasets::sec::run_all` which
//!    dispatches every form-specific extractor and emits the
//!    info-row CSVs in `processed/`.
//!
//! The fetch surface stays multi-function (each batch fetcher exists
//! so the Python wrapper can dispatch by form-type before/after the
//! main idx + submissions fetch). Extraction is a single call because
//! the dispatch happens inside Rust now — every form module is wired
//! into the orchestrator's loop, so callers don't need (and shouldn't
//! have) a per-form Python binding.

use pyo3::exceptions::{PyIOError, PyKeyboardInterrupt, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::wrap_pyfunction;

use kglite_core::datasets::sec::{
    fetch_company_tickers, fetch_exhibit21_attachment, fetch_filing_primary_doc,
    fetch_quarterly_master_idx, fetch_submissions_bulk, run_all, SecClient, SecError, SliceSpec,
    Workdir, YearRange,
};

// Workdir args cross the PyO3 boundary as `String`, not `PathBuf` —
// PyO3's `PathBuf` extraction routes through CPython's filesystem
// codec, which crashed on macOS once pyarrow's native libraries were
// loaded. `Workdir::new` takes `impl Into<PathBuf>`, so a `String`
// slots in with no body changes.

/// Build a SliceSpec from optional Python args — thin delegate to
/// `SliceSpec::from_optional_filters` (lifted to core in 0.10.1).
fn build_slice(
    cik_list: Option<Vec<u64>>,
    form_types: Option<Vec<String>>,
    year_range: Option<(u16, u16)>,
) -> SliceSpec {
    SliceSpec::from_optional_filters(cik_list, form_types, year_range)
}

// ─────────────────────────── fetch surface ───────────────────────────

/// A per-item fetch future, yielding that item's
/// `(downloaded, skipped)` delta. Boxed + `'static` (the closures own
/// cloned `SecClient` / `Workdir` handles) so `run_batch` can stay a
/// plain generic without lifetime gymnastics.
type FetchDeltaFut = std::pin::Pin<Box<dyn std::future::Future<Output = (usize, usize)>>>;

/// Fire one `kglite.progress`-schema event into the optional Python
/// `progress` callback (`{"kind","phase","label","total","current",
/// "unit","elapsed_s"}`). A pending SIGINT surfaces as `Err` so Ctrl+C
/// aborts a long fetch; callback errors are swallowed — a broken
/// progress UI must not kill the download.
fn fire_event(
    py: Python<'_>,
    progress: Option<&Py<PyAny>>,
    build: impl FnOnce(&Bound<'_, PyDict>) -> PyResult<()>,
) -> PyResult<()> {
    py.check_signals()?;
    if let Some(cb) = progress {
        let d = PyDict::new(py);
        build(&d)?;
        let _ = cb.call1(py, (d,));
    }
    Ok(())
}

/// Drive a sequential per-filing fetch loop under one `SecClient` +
/// tokio runtime, emitting `start` / `update` / `complete` progress
/// events. `fetch_one` runs one item and reports its
/// `(downloaded, skipped)` delta. Shared by every `fetch_*_batch`
/// binding so the rate-limited loop + progress plumbing live once.
#[allow(clippy::too_many_arguments)]
fn run_batch<T, F>(
    py: Python<'_>,
    user_agent: &str,
    workdir: String,
    batch: Vec<T>,
    phase: &str,
    label: &str,
    unit: &str,
    progress: Option<&Py<PyAny>>,
    fetch_one: F,
) -> PyResult<(usize, usize)>
where
    T: Send,
    F: Fn(SecClient, Workdir, T) -> FetchDeltaFut + Send,
{
    let client = SecClient::new(user_agent).map_err(map_err)?;
    let wd = Workdir::new(workdir);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime: {e}")))?;

    let total = batch.len();
    let started = std::time::Instant::now();
    fire_event(py, progress, |d| {
        d.set_item("kind", "start")?;
        d.set_item("phase", phase)?;
        d.set_item("label", label)?;
        d.set_item("unit", unit)?;
        d.set_item("total", total)?;
        Ok(())
    })?;

    // Release the GIL for the rate-limited network loop, re-acquiring
    // it only to fire each progress event. Without this the GIL stays
    // held for the whole batch, starving a Jupyter kernel's IOPub
    // thread — progress output can't flush until the call returns.
    let outcome: Result<(usize, usize), ()> = py.detach(|| {
        rt.block_on(async move {
            let mut downloaded = 0usize;
            let mut skipped = 0usize;
            for (i, item) in batch.into_iter().enumerate() {
                let (dl, sk) = fetch_one(client.clone(), wd.clone(), item).await;
                downloaded += dl;
                skipped += sk;
                let current = i + 1;
                let fired = Python::attach(|py| {
                    fire_event(py, progress, |ev| {
                        ev.set_item("kind", "update")?;
                        ev.set_item("phase", phase)?;
                        ev.set_item("current", current)?;
                        Ok(())
                    })
                    .is_ok()
                });
                if !fired {
                    return Err(());
                }
            }
            Ok((downloaded, skipped))
        })
    });
    let (downloaded, skipped) = match outcome {
        Ok(totals) => totals,
        Err(()) => return Err(PyKeyboardInterrupt::new_err("fetch interrupted")),
    };

    let _ = fire_event(py, progress, |d| {
        d.set_item("kind", "complete")?;
        d.set_item("phase", phase)?;
        d.set_item("elapsed_s", started.elapsed().as_secs_f64())?;
        Ok(())
    });
    Ok((downloaded, skipped))
}

/// Fetch the `raw/` tier — quarterly master.idx files for the shallow
/// window plus the nightly bulk submissions.zip and company_tickers.json.
#[pyfunction]
#[pyo3(signature = (
    workdir, *,
    user_agent,
    years,
    current_year,
    current_quarter,
    force_refetch=false,
    staleness_hours=24,
))]
#[allow(clippy::too_many_arguments)]
fn fetch_raw(
    py: Python<'_>,
    workdir: String,
    user_agent: &str,
    years: u16,
    current_year: u16,
    current_quarter: u8,
    force_refetch: bool,
    staleness_hours: u64,
) -> PyResult<Py<PyDict>> {
    let client = SecClient::new(user_agent).map_err(map_err)?;
    let wd = Workdir::new(workdir);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime: {e}")))?;

    let (idx_dl, idx_sk, submissions_dl, tickers_dl) = rt.block_on(async {
        let mut idx_dl = 0;
        let mut idx_sk = 0;
        if years > 0 {
            let start_year = current_year.saturating_sub(years - 1).max(1993);
            let range = YearRange::new(start_year, current_year);
            let r = fetch_quarterly_master_idx(&client, &wd, range, current_year, current_quarter)
                .await
                .map_err(map_err)?;
            idx_dl = r.0;
            idx_sk = r.1;
        }
        let submissions_dl = fetch_submissions_bulk(&client, &wd, staleness_hours, force_refetch)
            .await
            .map_err(map_err)?;
        let tickers_dl = fetch_company_tickers(&client, &wd, force_refetch)
            .await
            .map_err(map_err)?;
        Ok::<_, PyErr>((idx_dl, idx_sk, submissions_dl, tickers_dl))
    })?;

    let d = PyDict::new(py);
    d.set_item("master_idx_downloaded", idx_dl)?;
    d.set_item("master_idx_cached", idx_sk)?;
    d.set_item("submissions_downloaded", submissions_dl)?;
    d.set_item("company_tickers_downloaded", tickers_dl)?;
    Ok(d.into())
}

/// Batch-fetch Form 3/4/5 ownership XMLs. Returns (downloaded, skipped).
#[pyfunction]
#[pyo3(signature = (workdir, *, user_agent, batch, progress=None))]
fn fetch_form4_batch(
    py: Python<'_>,
    workdir: String,
    user_agent: &str,
    batch: Vec<(u64, String, String)>,
    progress: Option<Py<PyAny>>,
) -> PyResult<(usize, usize)> {
    use kglite_core::datasets::sec::fetch_form4_filing;
    run_batch(
        py,
        user_agent,
        workdir,
        batch,
        "ownership",
        "Form 3/4/5 ownership",
        "filing",
        progress.as_ref(),
        |client, wd, (cik, accession, primary_doc)| {
            Box::pin(async move {
                match fetch_form4_filing(&client, &wd, cik, &accession, &primary_doc).await {
                    Ok(true) => (1, 0),
                    _ => (0, 1),
                }
            })
        },
    )
}

/// Batch-fetch 13F info tables. Returns (downloaded, skipped).
#[pyfunction]
#[pyo3(signature = (workdir, *, user_agent, batch, progress=None))]
fn fetch_13f_batch(
    py: Python<'_>,
    workdir: String,
    user_agent: &str,
    batch: Vec<(u64, String)>,
    progress: Option<Py<PyAny>>,
) -> PyResult<(usize, usize)> {
    use kglite_core::datasets::sec::fetch_13f_info_table;
    run_batch(
        py,
        user_agent,
        workdir,
        batch,
        "form13f",
        "13F info tables",
        "filing",
        progress.as_ref(),
        |client, wd, (cik, accession)| {
            Box::pin(async move {
                match fetch_13f_info_table(&client, &wd, cik, &accession).await {
                    Ok(true) => (1, 0),
                    _ => (0, 1),
                }
            })
        },
    )
}

/// Batch-fetch any filing's primary document. `phase`/`label` name
/// the progress bar — the same fetcher backs 8-K, SC 13D/G, DEF 14A
/// and Form 144. Returns (downloaded, skipped).
#[pyfunction]
#[pyo3(signature = (workdir, *, user_agent, batch, phase="filings", label="Filings", progress=None))]
fn fetch_filing_batch(
    py: Python<'_>,
    workdir: String,
    user_agent: &str,
    batch: Vec<(u64, String, String)>,
    phase: &str,
    label: &str,
    progress: Option<Py<PyAny>>,
) -> PyResult<(usize, usize)> {
    run_batch(
        py,
        user_agent,
        workdir,
        batch,
        phase,
        label,
        "filing",
        progress.as_ref(),
        |client, wd, (cik, accession, primary_doc)| {
            Box::pin(async move {
                match fetch_filing_primary_doc(&client, &wd, cik, &accession, &primary_doc).await {
                    Ok(true) => (1, 0),
                    _ => (0, 1),
                }
            })
        },
    )
}

/// Batch-fetch per-company submission JSON. For sliced runs this
/// avoids the ~1 GB bulk submissions.zip download + the 528K-entry
/// central-directory parse at extract time. Returns (downloaded,
/// skipped).
#[pyfunction]
#[pyo3(signature = (workdir, *, user_agent, ciks, force_refetch=false, progress=None))]
fn fetch_company_submissions_batch(
    py: Python<'_>,
    workdir: String,
    user_agent: &str,
    ciks: Vec<u64>,
    force_refetch: bool,
    progress: Option<Py<PyAny>>,
) -> PyResult<(usize, usize)> {
    use kglite_core::datasets::sec::fetch_company_submission;
    run_batch(
        py,
        user_agent,
        workdir,
        ciks,
        "submissions",
        "Company submissions",
        "company",
        progress.as_ref(),
        move |client, wd, cik| {
            Box::pin(async move {
                match fetch_company_submission(&client, &wd, cik, force_refetch).await {
                    Ok(true) => (1, 0),
                    _ => (0, 1),
                }
            })
        },
    )
}

/// Batch-fetch XBRL company-facts JSON. Takes a list of CIKs; the
/// company-facts API returns every tagged financial fact a company
/// has reported in one JSON document. Returns (downloaded, skipped).
#[pyfunction]
#[pyo3(signature = (workdir, *, user_agent, ciks, force_refetch=false, progress=None))]
fn fetch_company_facts_batch(
    py: Python<'_>,
    workdir: String,
    user_agent: &str,
    ciks: Vec<u64>,
    force_refetch: bool,
    progress: Option<Py<PyAny>>,
) -> PyResult<(usize, usize)> {
    use kglite_core::datasets::sec::fetch_company_facts;
    run_batch(
        py,
        user_agent,
        workdir,
        ciks,
        "company_facts",
        "XBRL company facts",
        "company",
        progress.as_ref(),
        move |client, wd, cik| {
            Box::pin(async move {
                match fetch_company_facts(&client, &wd, cik, force_refetch).await {
                    Ok(true) => (1, 0),
                    _ => (0, 1),
                }
            })
        },
    )
}

/// Batch-fetch Exhibit 21 attachments. Returns (downloaded, skipped) —
/// `downloaded` counts attachment files, not 10-K filings.
#[pyfunction]
#[pyo3(signature = (workdir, *, user_agent, batch, progress=None))]
fn fetch_exhibit21_batch(
    py: Python<'_>,
    workdir: String,
    user_agent: &str,
    batch: Vec<(u64, String)>,
    progress: Option<Py<PyAny>>,
) -> PyResult<(usize, usize)> {
    run_batch(
        py,
        user_agent,
        workdir,
        batch,
        "exhibit21",
        "Exhibit 21",
        "filing",
        progress.as_ref(),
        |client, wd, (cik, accession)| {
            Box::pin(async move {
                match fetch_exhibit21_attachment(&client, &wd, cik, &accession).await {
                    Ok(n) if n > 0 => (n, 0),
                    _ => (0, 1),
                }
            })
        },
    )
}

// ───────────────────────── extract surface (thin) ─────────────────────────

/// Single feature-extraction entry point. Calls `run_all` which
/// dispatches every form-specific extractor in turn and emits the
/// info-row CSVs in `processed/` (purchase, sale, holding, role,
/// corporate_event, subsidiary, ...). Identity tables (company,
/// person, security, manager) populate as a side-effect.
///
/// Returns a flat dict with row counts per form for telemetry +
/// total_rows, total_identity_counts.
#[pyfunction]
#[pyo3(signature = (workdir, *, force=false, cik_list=None, form_types=None, year_range=None))]
fn extract_all_py(
    py: Python<'_>,
    workdir: String,
    force: bool,
    cik_list: Option<Vec<u64>>,
    form_types: Option<Vec<String>>,
    year_range: Option<(u16, u16)>,
) -> PyResult<Py<PyDict>> {
    let wd = Workdir::new(workdir);
    let slice = build_slice(cik_list, form_types, year_range);
    let report = run_all(&wd, &slice, force).map_err(map_err)?;
    let d = PyDict::new(py);
    d.set_item("extracted_at", report.extracted_at.clone())?;
    d.set_item("total_rows", report.total_rows())?;
    d.set_item("submission_parse_errors", report.submission_parse_errors)?;
    d.set_item("distinct_sic_codes", report.distinct_sic_codes)?;
    // Bottleneck-detection timings (milliseconds).
    d.set_item("total_ms", report.total_ms)?;
    d.set_item("identity_ms", report.identity_ms)?;
    // Identity counts.
    d.set_item("companies", report.identity_counts.companies)?;
    d.set_item("people", report.identity_counts.people)?;
    d.set_item("securities", report.identity_counts.securities)?;
    d.set_item("managers", report.identity_counts.managers)?;
    // Per-form row counts. Useful for the wrapper print() lines + tests.
    macro_rules! form_rows {
        ($name:ident) => {{
            let inner = PyDict::new(py);
            inner.set_item("files_read", report.$name.files_read)?;
            inner.set_item("parse_errors", report.$name.parse_errors)?;
            inner.set_item("rows_written", report.$name.rows_written)?;
            inner.set_item("duration_ms", report.$name.duration_ms)?;
            d.set_item(stringify!($name), inner)?;
        }};
    }
    form_rows!(form3);
    form_rows!(form4);
    form_rows!(form5);
    form_rows!(form144);
    form_rows!(form13f);
    form_rows!(schedule13);
    form_rows!(def14a);
    form_rows!(eightk);
    form_rows!(ten_k);
    form_rows!(ten_q);
    form_rows!(s1);
    form_rows!(s3);
    form_rows!(s4);
    form_rows!(prospectus);
    form_rows!(formd);
    form_rows!(npx);
    form_rows!(xbrl);
    Ok(d.into())
}

// ─────────────────────── storage-mode planning ───────────────────────

/// Estimate the built graph's resident size in GB for the given scope.
#[pyfunction]
#[pyo3(signature = (
    years, detailed, cik_count=None,
    include_subsidiaries=true, include_xbrl_metrics=true, include_8k_events=true,
))]
fn predict_graph_size_gb(
    years: u32,
    detailed: u32,
    cik_count: Option<usize>,
    include_subsidiaries: bool,
    include_xbrl_metrics: bool,
    include_8k_events: bool,
) -> f64 {
    kglite_core::datasets::sec::predict_graph_size_gb(
        years,
        detailed,
        cik_count,
        include_subsidiaries,
        include_xbrl_metrics,
        include_8k_events,
    )
}

/// Pick a storage backend for an estimated graph size.
#[pyfunction]
fn pick_storage_mode(predicted_gb: f64) -> &'static str {
    kglite_core::datasets::sec::pick_storage_mode(predicted_gb)
}

// ─────────────────────── graph location helpers ───────────────────────

/// Path to the workdir's expected blueprint output dir for the given
/// mode. Returns the path as a `String` (see the module note on
/// `PathBuf` PyO3 args).
#[pyfunction]
fn graph_dir(workdir: String, mode: &str) -> PyResult<String> {
    let m: kglite_core::datasets::sec::StorageMode =
        mode.parse().map_err(|e: String| PyValueError::new_err(e))?;
    Ok(Workdir::new(workdir)
        .graph_dir(m)
        .to_string_lossy()
        .into_owned())
}

/// True if a built graph for `mode` already exists in `workdir/graph/{mode}/`.
#[pyfunction]
fn graph_exists(workdir: String, mode: &str) -> PyResult<bool> {
    let m: kglite_core::datasets::sec::StorageMode =
        mode.parse().map_err(|e: String| PyValueError::new_err(e))?;
    Ok(Workdir::new(workdir).graph_exists(m))
}

fn map_err(e: SecError) -> PyErr {
    match &e {
        SecError::Io(_) => PyIOError::new_err(format!("{e}")),
        SecError::MissingUserAgent => PyValueError::new_err(format!("{e}")),
        _ => PyRuntimeError::new_err(format!("{e}")),
    }
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "_sec_internal")?;
    // Fetch
    m.add_function(wrap_pyfunction!(fetch_raw, &m)?)?;
    m.add_function(wrap_pyfunction!(fetch_form4_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(fetch_13f_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(fetch_filing_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(fetch_exhibit21_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(fetch_company_facts_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(fetch_company_submissions_batch, &m)?)?;
    // Extract (single entry point)
    m.add_function(wrap_pyfunction!(extract_all_py, &m)?)?;
    // Storage-mode planning
    m.add_function(wrap_pyfunction!(predict_graph_size_gb, &m)?)?;
    m.add_function(wrap_pyfunction!(pick_storage_mode, &m)?)?;
    // Graph location helpers
    m.add_function(wrap_pyfunction!(graph_dir, &m)?)?;
    m.add_function(wrap_pyfunction!(graph_exists, &m)?)?;
    parent.add_submodule(&m)?;
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("kglite._sec_internal", m)?;
    Ok(())
}
