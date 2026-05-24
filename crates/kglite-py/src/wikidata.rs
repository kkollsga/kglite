//! PyO3 bindings exposing the Wikidata dump fetcher as
//! `kglite._wikidata_internal`.
//!
//! The `kglite-wikidata` crate owns the dump cache lifecycle only —
//! resumable download + staleness. The N-Triples → graph build stays
//! a Python `KnowledgeGraph.load_ntriples(...)` call, since that
//! loader already lives in this crate.
//!
//! Path arguments cross as `String`, not `PathBuf` (see `src/sodir.rs`
//! for the rationale).

use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;

use kglite_core::datasets::wikidata::{WikidataError, Workdir};

fn map_err(e: WikidataError) -> PyErr {
    match &e {
        WikidataError::Io(_) => PyIOError::new_err(format!("{e}")),
        _ => PyRuntimeError::new_err(format!("{e}")),
    }
}

fn runtime() -> PyResult<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime: {e}")))
}

/// Ensure the local Wikidata dump exists, downloading or resuming as
/// needed. Returns `(dump_path, remote_last_modified_iso)`.
#[pyfunction]
#[pyo3(signature = (workdir, *, cooldown_days=31, verbose=true))]
fn ensure_dump(
    workdir: String,
    cooldown_days: i64,
    verbose: bool,
) -> PyResult<(String, Option<String>)> {
    let wd = Workdir::new(workdir);
    let rt = runtime()?;
    let (path, mtime) = rt
        .block_on(kglite_core::datasets::wikidata::ensure_dump(
            &wd,
            cooldown_days,
            verbose,
        ))
        .map_err(map_err)?;
    Ok((
        path.to_string_lossy().into_owned(),
        mtime.map(|m| m.to_rfc3339()),
    ))
}

/// The remote dump's `Last-Modified` as an ISO string, or `None` if
/// the dump server is unreachable.
#[pyfunction]
fn remote_last_modified() -> PyResult<Option<String>> {
    let rt = runtime()?;
    Ok(rt
        .block_on(kglite_core::datasets::wikidata::remote_last_modified())
        .map(|m| m.to_rfc3339()))
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "_wikidata_internal")?;
    m.add_function(wrap_pyfunction!(ensure_dump, &m)?)?;
    m.add_function(wrap_pyfunction!(remote_last_modified, &m)?)?;
    parent.add_submodule(&m)?;
    let sys = py.import("sys")?;
    sys.getattr("modules")?
        .set_item("kglite._wikidata_internal", m)?;
    Ok(())
}
