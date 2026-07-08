//! PyO3 bindings exposing the Sodir FactMaps loader as
//! `kglite._sodir_internal`.
//!
//! A thin surface over the pure-Rust `kglite-sodir` crate: refresh the
//! CSV cache, merge blueprints, and a few graph-location helpers. The
//! graph build itself stays a Python `from_blueprint` call — the
//! `kglite-sodir` crate cannot depend back on the main `kglite` crate.
//!
//! Workdir / path arguments cross the boundary as `String`, never
//! `PathBuf`: PyO3's `PathBuf` extraction routes through CPython's
//! filesystem codec, which has crashed on macOS once pyarrow's native
//! libraries are loaded. Plain `String` extraction is unaffected.

use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::wrap_pyfunction;

use kglite_core::api::datasets::sodir::{
    datasets_used_by_blueprint, fetch_all, SodirError, Workdir,
};

fn map_err(e: SodirError) -> PyErr {
    match &e {
        SodirError::Io(_) => PyIOError::new_err(format!("{e}")),
        SodirError::UnknownStem(_) | SodirError::Decode(_) => PyValueError::new_err(format!("{e}")),
        _ => PyRuntimeError::new_err(format!("{e}")),
    }
}

fn parse_json(s: &str, what: &str) -> PyResult<serde_json::Value> {
    serde_json::from_str(s).map_err(|e| PyValueError::new_err(format!("{what}: {e}")))
}

/// Refresh the CSV cache for every dataset the blueprint references,
/// then run the FK preprocessing. `blueprint_json` is the already-
/// merged blueprint. Returns a flat report dict.
#[pyfunction]
#[pyo3(signature = (
    workdir, blueprint_json, *,
    index_cooldown_days=14,
    dataset_cooldown_days=30,
    concurrency=10,
))]
fn refresh(
    py: Python<'_>,
    workdir: String,
    blueprint_json: String,
    index_cooldown_days: i64,
    dataset_cooldown_days: i64,
    concurrency: usize,
) -> PyResult<Py<PyDict>> {
    let wd = Workdir::new(workdir);
    let blueprint = parse_json(&blueprint_json, "blueprint")?;
    let needed = datasets_used_by_blueprint(&blueprint);

    // `fetch_all` is synchronous now (backed by the shared blocking
    // `DatasetClient`); its own scoped worker pool overlaps the network
    // latency. Release the GIL for the whole download so other Python
    // threads (e.g. a Jupyter kernel's IOPub) can run while it blocks —
    // same treatment `sec::run_batch` got in Phase 2. The closure returns
    // the plain `Result`; we build the `PyErr` *after* `detach` returns,
    // since error mapping may touch Python. Nothing captured crosses the
    // `Ungil`/`Send` bound problematically — `Workdir` and `Vec<String>`
    // are both `Send + Sync`.
    let report = py
        .detach(|| {
            fetch_all(
                &wd,
                &needed,
                index_cooldown_days,
                dataset_cooldown_days,
                concurrency,
            )
        })
        .map_err(map_err)?;

    let d = PyDict::new(py);
    d.set_item("fetched", report.refresh.fetched)?;
    d.set_item("unchanged", report.refresh.unchanged)?;
    d.set_item("user_supplied", report.refresh.user_supplied)?;
    d.set_item("cached", report.refresh.cached)?;
    d.set_item("unfetchable", report.refresh.unfetchable)?;
    d.set_item("errors", report.refresh.errors)?;

    let pp = PyDict::new(py);
    pp.set_item("petreg_licence_pk", report.preprocess.petreg_licence_pk)?;
    pp.set_item("seismic_progress_fk", report.preprocess.seismic_progress_fk)?;
    pp.set_item("chrono_parent_fk", report.preprocess.chrono_parent_fk)?;
    pp.set_item("announced_block_fk", report.preprocess.announced_block_fk)?;
    d.set_item("preprocess", pp)?;

    Ok(d.into())
}

/// Deep-merge a base blueprint with an optional complement, returning
/// the merged blueprint as a JSON string. Base wins on leaf collisions
/// unless `complement_overrides`.
#[pyfunction]
#[pyo3(signature = (base_json, complement_json=None, complement_overrides=false))]
fn merge_blueprint(
    base_json: String,
    complement_json: Option<String>,
    complement_overrides: bool,
) -> PyResult<String> {
    // Engine logic lives in `kglite_core::api::datasets::sodir::merge_blueprint_json`
    // (lifted from this file in 0.10.1). This wrapper only adapts the
    // String → PyErr boundary.
    kglite_core::api::datasets::sodir::merge_blueprint_json(
        &base_json,
        complement_json.as_deref(),
        complement_overrides,
    )
    .map_err(PyRuntimeError::new_err)
}

/// The dataset stems a blueprint references (CSV filename stems).
#[pyfunction]
fn datasets_for_blueprint(blueprint_json: String) -> PyResult<Vec<String>> {
    let bp = parse_json(&blueprint_json, "blueprint")?;
    Ok(datasets_used_by_blueprint(&bp))
}

/// Path to the workdir's `graph/` directory.
#[pyfunction]
fn graph_dir(workdir: String) -> PyResult<String> {
    Ok(Workdir::new(workdir)
        .graph_dir()
        .to_string_lossy()
        .into_owned())
}

/// True if a disk-mode graph already exists in the workdir.
#[pyfunction]
fn graph_exists(workdir: String) -> PyResult<bool> {
    Ok(Workdir::new(workdir).graph_exists())
}

/// Age in days of the disk-mode graph metadata, or `None` if no graph
/// has been built. Thin delegate to `Workdir::disk_graph_age_days`
/// (lifted to core in 0.10.1).
#[pyfunction]
fn disk_graph_age_days(workdir: String) -> PyResult<Option<f64>> {
    Ok(Workdir::new(workdir).disk_graph_age_days())
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "_sodir_internal")?;
    m.add_function(wrap_pyfunction!(refresh, &m)?)?;
    m.add_function(wrap_pyfunction!(merge_blueprint, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets_for_blueprint, &m)?)?;
    m.add_function(wrap_pyfunction!(graph_dir, &m)?)?;
    m.add_function(wrap_pyfunction!(graph_exists, &m)?)?;
    m.add_function(wrap_pyfunction!(disk_graph_age_days, &m)?)?;
    parent.add_submodule(&m)?;
    let sys = py.import("sys")?;
    sys.getattr("modules")?
        .set_item("kglite._sodir_internal", m)?;
    Ok(())
}
