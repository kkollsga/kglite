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

use chrono::DateTime;
use std::path::Path;

use kglite_core::api::datasets::wikidata::{
    decide as decide_freshness, CacheDecision, FreshnessInputs, WikidataError, Workdir,
};

fn map_err(e: WikidataError) -> PyErr {
    match &e {
        WikidataError::Io(_) => PyIOError::new_err(format!("{e}")),
        _ => PyRuntimeError::new_err(format!("{e}")),
    }
}

/// Ensure the local Wikidata dump exists, downloading or resuming as
/// needed. Returns `(dump_path, remote_last_modified_iso)`.
///
/// The core fetch is synchronous (shared `DatasetClient`) and runs
/// directly on the calling thread — no tokio runtime. Because a full
/// dump is multi-GB, the GIL is released for the entire download via
/// `py.detach` (mirroring `sec::run_batch`'s Phase-2 treatment) so other
/// Python threads keep running while it blocks. The Rust-side progress
/// `eprintln!`s inside the wikidata client need no GIL, so they keep
/// working. The closure returns the plain `Result`; the `PyErr` is built
/// *after* `detach` returns, since error mapping may touch Python.
#[pyfunction]
#[pyo3(signature = (workdir, *, cooldown_days=31, verbose=true))]
fn ensure_dump(
    py: Python<'_>,
    workdir: String,
    cooldown_days: i64,
    verbose: bool,
) -> PyResult<(String, Option<String>)> {
    let wd = Workdir::new(workdir);
    let (path, mtime) = py
        .detach(|| kglite_core::api::datasets::wikidata::ensure_dump(&wd, cooldown_days, verbose))
        .map_err(map_err)?;
    Ok((
        path.to_string_lossy().into_owned(),
        mtime.map(|m| m.to_rfc3339()),
    ))
}

/// The remote dump's `Last-Modified` as an ISO string, or `None` if
/// the dump server is unreachable. A live HEAD against the dump server;
/// the GIL is released for the round-trip (same rationale as
/// `ensure_dump`).
#[pyfunction]
fn remote_last_modified(py: Python<'_>) -> PyResult<Option<String>> {
    let mtime = py.detach(kglite_core::api::datasets::wikidata::remote_last_modified);
    Ok(mtime.map(|m| m.to_rfc3339()))
}

/// Run the cache-freshness decision tree. Returns
/// `(action, reason)` where `action` is one of `"build"`, `"load"`,
/// `"rebuild"`. Lifted from the Python wrapper's `open()` body so
/// every binding shares the same comparisons.
#[pyfunction]
#[pyo3(signature = (
    *,
    force_rebuild,
    graph_meta_path,
    source_meta_path,
    cooldown_days,
    remote_mtime_iso=None,
))]
fn decide_cache_freshness(
    force_rebuild: bool,
    graph_meta_path: &str,
    source_meta_path: &str,
    cooldown_days: i64,
    remote_mtime_iso: Option<&str>,
) -> PyResult<(&'static str, String)> {
    let remote_mtime = match remote_mtime_iso {
        None => None,
        Some(iso) => Some(
            DateTime::parse_from_rfc3339(iso)
                .map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "invalid remote_mtime_iso: {e}"
                    ))
                })?
                .with_timezone(&chrono::Utc),
        ),
    };
    let decision = decide_freshness(FreshnessInputs {
        force_rebuild,
        graph_meta_path: Path::new(graph_meta_path),
        source_meta_path: Path::new(source_meta_path),
        cooldown_days,
        remote_mtime,
    });
    Ok(match decision {
        CacheDecision::Build { reason } => ("build", reason.to_string()),
        CacheDecision::Load { reason } => ("load", reason),
        CacheDecision::Rebuild { reason } => ("rebuild", reason),
    })
}

pub fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "_wikidata_internal")?;
    m.add_function(wrap_pyfunction!(ensure_dump, &m)?)?;
    m.add_function(wrap_pyfunction!(remote_last_modified, &m)?)?;
    m.add_function(wrap_pyfunction!(decide_cache_freshness, &m)?)?;
    parent.add_submodule(&m)?;
    let sys = py.import("sys")?;
    sys.getattr("modules")?
        .set_item("kglite._wikidata_internal", m)?;
    Ok(())
}
