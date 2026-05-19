//! pyo3 bindings exposing the SEC EDGAR loader as `kglite._sec_internal`.
//!
//! The pure-Rust loader lives in `kglite-sec` (sibling crate). This
//! file wraps a small surface of it — `fetch_raw` and
//! `extract_processed` — for the Python `kglite.datasets.sec.SEC.open()`
//! lifecycle to call. Graph build itself stays in Python:
//! `kglite.from_blueprint(...)` reads the CSVs we produce.
//!
//! The Rust loader is async; we spin up a single-threaded tokio
//! runtime per call so Python callers see plain blocking functions.

use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::wrap_pyfunction;

use kglite_sec::{
    extract_companies_and_filings, fetch_company_tickers, fetch_quarterly_master_idx,
    fetch_submissions_bulk, SecClient, SecError, Workdir, YearRange,
};
use std::path::PathBuf;

/// Fetch the `raw/` tier — quarterly master.idx files for the shallow
/// window plus the nightly bulk submissions.zip and company_tickers.json.
///
/// `years` = how many years back to fetch master.idx for. `0` skips
/// the shallow fetch entirely. Returns a dict with download statistics.
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

    // Single-threaded runtime per call. Cheap to construct; the
    // crate's parallelism is bounded by the 10 req/s rate limit
    // anyway, so a multi-thread runtime gains us nothing.
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
    d.set_item("master_idx_skipped", idx_sk)?;
    d.set_item("submissions_zip_fetched", submissions_dl)?;
    d.set_item("company_tickers_fetched", tickers_dl)?;
    Ok(d.into())
}

/// Extract `processed/` CSVs (company.csv, filing.csv) from the raw
/// cache. Returns a dict with extraction stats.
#[pyfunction]
#[pyo3(signature = (
    workdir, *,
    years,
    current_year,
    force=false,
))]
fn extract_processed(
    py: Python<'_>,
    workdir: PathBuf,
    years: u16,
    current_year: u16,
    force: bool,
) -> PyResult<Py<PyDict>> {
    let wd = Workdir::new(workdir);
    let start_year = current_year
        .saturating_sub(years.saturating_sub(1))
        .max(1993);
    let range = YearRange::new(start_year, current_year);
    let report = extract_companies_and_filings(&wd, range, force).map_err(map_err)?;

    let d = PyDict::new(py);
    d.set_item("companies_written", report.companies_written)?;
    d.set_item("filings_from_submissions", report.filings_from_submissions)?;
    d.set_item("filings_from_master_idx", report.filings_from_master_idx)?;
    d.set_item("master_idx_files_read", report.master_idx_files_read)?;
    d.set_item("master_idx_parse_errors", report.master_idx_parse_errors)?;
    d.set_item("submission_parse_errors", report.submission_parse_errors)?;
    Ok(d.into())
}

/// Path to the workdir's expected blueprint output dir for the given
/// mode. Pure path arithmetic — does not touch the filesystem. Used by
/// the Python wrapper to find where to write/load the .kgl.
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
    m.add_function(wrap_pyfunction!(fetch_raw, &m)?)?;
    m.add_function(wrap_pyfunction!(extract_processed, &m)?)?;
    m.add_function(wrap_pyfunction!(graph_dir, &m)?)?;
    m.add_function(wrap_pyfunction!(graph_exists, &m)?)?;
    parent.add_submodule(&m)?;
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("kglite._sec_internal", &m)?;
    Ok(())
}
