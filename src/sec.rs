//! PyO3 bindings exposing the SEC EDGAR loader as `kglite._sec_internal`.
//!
//! Two surfaces, kept deliberately thin:
//!
//! 1. **Fetch helpers** (`fetch_raw`, `fetch_fsnds`, `fetch_form4_batch`,
//!    `fetch_13f_batch`, `fetch_filing_batch`, `fetch_exhibit21_batch`)
//!    download SEC documents into `raw/` under a single SecClient that
//!    enforces the 10 req/s SEC rate limit. The Python wrapper invokes
//!    these in the order dictated by form-type dependencies.
//! 2. **Feature extraction** is exposed as ONE function:
//!    `extract_all_py(workdir, *, force, cik_list, form_types, year_range)`.
//!    It calls the Rust orchestrator `kglite_sec::run_all` which
//!    dispatches every form-specific extractor and emits the
//!    info-row CSVs in `processed/`.
//!
//! The fetch surface stays multi-function (each batch fetcher exists
//! so the Python wrapper can dispatch by form-type before/after the
//! main idx + submissions fetch). Extraction is a single call because
//! the dispatch happens inside Rust now — every form module is wired
//! into the orchestrator's loop, so callers don't need (and shouldn't
//! have) a per-form Python binding.

use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::wrap_pyfunction;

use kglite_sec::{
    fetch_company_tickers, fetch_exhibit21_attachment, fetch_filing_primary_doc,
    fetch_fsnds_quarterly, fetch_quarterly_master_idx, fetch_submissions_bulk, run_all, SecClient,
    SecError, SliceSpec, Workdir, YearRange,
};
use std::path::PathBuf;

/// Build a SliceSpec from optional Python args. Empty / None args
/// produce an unrestricted slice.
fn build_slice(
    cik_list: Option<Vec<u64>>,
    form_types: Option<Vec<String>>,
    year_range: Option<(u16, u16)>,
) -> SliceSpec {
    let mut s = SliceSpec::default();
    if let Some(ciks) = cik_list {
        if !ciks.is_empty() {
            s = s.with_cik_list(ciks);
        }
    }
    if let Some(forms) = form_types {
        if !forms.is_empty() {
            s = s.with_form_types(forms);
        }
    }
    if let Some((lo, hi)) = year_range {
        s = s.with_year_range(lo, hi);
    }
    s
}

// ─────────────────────────── fetch surface ───────────────────────────

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
    workdir: PathBuf,
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

/// Fetch FSNDS NUM.tsv for one (year, quarter). Bulk path; not rate
/// limited. Returns `true` if newly downloaded, `false` if cached.
#[pyfunction]
#[pyo3(signature = (workdir, *, user_agent, year, quarter, force_refetch=false))]
fn fetch_fsnds(
    workdir: PathBuf,
    user_agent: &str,
    year: u16,
    quarter: u8,
    force_refetch: bool,
) -> PyResult<bool> {
    let client = SecClient::new(user_agent).map_err(map_err)?;
    let wd = Workdir::new(workdir);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime: {e}")))?;
    rt.block_on(async {
        fetch_fsnds_quarterly(&client, &wd, year, quarter, force_refetch)
            .await
            .map_err(map_err)
    })
}

/// Batch-fetch Form 4 XMLs. Returns (downloaded, skipped).
#[pyfunction]
#[pyo3(signature = (workdir, *, user_agent, batch))]
fn fetch_form4_batch(
    workdir: PathBuf,
    user_agent: &str,
    batch: Vec<(u64, String, String)>,
) -> PyResult<(usize, usize)> {
    use kglite_sec::fetch_form4_filing;
    let client = SecClient::new(user_agent).map_err(map_err)?;
    let wd = Workdir::new(workdir);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime: {e}")))?;
    let mut downloaded = 0;
    let mut skipped = 0;
    rt.block_on(async {
        for (cik, accession, primary_doc) in batch {
            match fetch_form4_filing(&client, &wd, cik, &accession, &primary_doc).await {
                Ok(true) => downloaded += 1,
                Ok(false) => skipped += 1,
                Err(_) => skipped += 1,
            }
        }
    });
    Ok((downloaded, skipped))
}

/// Batch-fetch 13F info tables.
#[pyfunction]
#[pyo3(signature = (workdir, *, user_agent, batch))]
fn fetch_13f_batch(
    workdir: PathBuf,
    user_agent: &str,
    batch: Vec<(u64, String)>,
) -> PyResult<(usize, usize)> {
    use kglite_sec::fetch_13f_info_table;
    let client = SecClient::new(user_agent).map_err(map_err)?;
    let wd = Workdir::new(workdir);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime: {e}")))?;
    let mut downloaded = 0;
    let mut skipped = 0;
    rt.block_on(async {
        for (cik, accession) in batch {
            match fetch_13f_info_table(&client, &wd, cik, &accession).await {
                Ok(true) => downloaded += 1,
                Ok(false) => skipped += 1,
                Err(_) => skipped += 1,
            }
        }
    });
    Ok((downloaded, skipped))
}

/// Batch-fetch any filing's primary document.
#[pyfunction]
#[pyo3(signature = (workdir, *, user_agent, batch))]
fn fetch_filing_batch(
    workdir: PathBuf,
    user_agent: &str,
    batch: Vec<(u64, String, String)>,
) -> PyResult<(usize, usize)> {
    let client = SecClient::new(user_agent).map_err(map_err)?;
    let wd = Workdir::new(workdir);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime: {e}")))?;
    let mut downloaded = 0;
    let mut skipped = 0;
    rt.block_on(async {
        for (cik, accession, primary_doc) in batch {
            match fetch_filing_primary_doc(&client, &wd, cik, &accession, &primary_doc).await {
                Ok(true) => downloaded += 1,
                Ok(false) => skipped += 1,
                Err(_) => skipped += 1,
            }
        }
    });
    Ok((downloaded, skipped))
}

/// Batch-fetch per-company submission JSON. For sliced runs this
/// avoids the ~1 GB bulk submissions.zip download + the 528K-entry
/// central-directory parse at extract time. Returns (downloaded,
/// skipped).
#[pyfunction]
#[pyo3(signature = (workdir, *, user_agent, ciks, force_refetch=false))]
fn fetch_company_submissions_batch(
    workdir: PathBuf,
    user_agent: &str,
    ciks: Vec<u64>,
    force_refetch: bool,
) -> PyResult<(usize, usize)> {
    use kglite_sec::fetch_company_submission;
    let client = SecClient::new(user_agent).map_err(map_err)?;
    let wd = Workdir::new(workdir);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime: {e}")))?;
    let mut downloaded = 0;
    let mut skipped = 0;
    rt.block_on(async {
        for cik in ciks {
            match fetch_company_submission(&client, &wd, cik, force_refetch).await {
                Ok(true) => downloaded += 1,
                Ok(false) => skipped += 1,
                Err(_) => skipped += 1,
            }
        }
    });
    Ok((downloaded, skipped))
}

/// Batch-fetch XBRL company-facts JSON. Takes a list of CIKs; the
/// company-facts API returns every tagged financial fact a company
/// has reported in one JSON document. Returns (downloaded, skipped).
#[pyfunction]
#[pyo3(signature = (workdir, *, user_agent, ciks, force_refetch=false))]
fn fetch_company_facts_batch(
    workdir: PathBuf,
    user_agent: &str,
    ciks: Vec<u64>,
    force_refetch: bool,
) -> PyResult<(usize, usize)> {
    use kglite_sec::fetch_company_facts;
    let client = SecClient::new(user_agent).map_err(map_err)?;
    let wd = Workdir::new(workdir);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime: {e}")))?;
    let mut downloaded = 0;
    let mut skipped = 0;
    rt.block_on(async {
        for cik in ciks {
            match fetch_company_facts(&client, &wd, cik, force_refetch).await {
                Ok(true) => downloaded += 1,
                Ok(false) => skipped += 1,
                Err(_) => skipped += 1,
            }
        }
    });
    Ok((downloaded, skipped))
}

/// Batch-fetch Exhibit 21 attachments.
#[pyfunction]
#[pyo3(signature = (workdir, *, user_agent, batch))]
fn fetch_exhibit21_batch(
    workdir: PathBuf,
    user_agent: &str,
    batch: Vec<(u64, String)>,
) -> PyResult<(usize, usize)> {
    let client = SecClient::new(user_agent).map_err(map_err)?;
    let wd = Workdir::new(workdir);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime: {e}")))?;
    let mut downloaded = 0;
    let mut skipped = 0;
    rt.block_on(async {
        for (cik, accession) in batch {
            match fetch_exhibit21_attachment(&client, &wd, cik, &accession).await {
                Ok(n) if n > 0 => downloaded += n,
                _ => skipped += 1,
            }
        }
    });
    Ok((downloaded, skipped))
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
    workdir: PathBuf,
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

// ─────────────────────── graph location helpers ───────────────────────

/// Path to the workdir's expected blueprint output dir for the given mode.
#[pyfunction]
fn graph_dir(workdir: PathBuf, mode: &str) -> PyResult<PathBuf> {
    let m: kglite_sec::StorageMode = mode.parse().map_err(|e: String| PyValueError::new_err(e))?;
    Ok(Workdir::new(workdir).graph_dir(m))
}

/// True if a built graph for `mode` already exists in `workdir/graph/{mode}/`.
#[pyfunction]
fn graph_exists(workdir: PathBuf, mode: &str) -> PyResult<bool> {
    let m: kglite_sec::StorageMode = mode.parse().map_err(|e: String| PyValueError::new_err(e))?;
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
    m.add_function(wrap_pyfunction!(fetch_fsnds, &m)?)?;
    m.add_function(wrap_pyfunction!(fetch_form4_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(fetch_13f_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(fetch_filing_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(fetch_exhibit21_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(fetch_company_facts_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(fetch_company_submissions_batch, &m)?)?;
    // Extract (single entry point)
    m.add_function(wrap_pyfunction!(extract_all_py, &m)?)?;
    // Graph location helpers
    m.add_function(wrap_pyfunction!(graph_dir, &m)?)?;
    m.add_function(wrap_pyfunction!(graph_exists, &m)?)?;
    parent.add_submodule(&m)?;
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("kglite._sec_internal", m)?;
    Ok(())
}
