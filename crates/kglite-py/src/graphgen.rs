//! PyO3 binding for the bundled synthetic-graph generator
//! (`kglite_core::graphgen`). The thin Rust surface just streams the CSVs +
//! `manifest.json`; the Python `kglite.graphgen(...)` wrapper adds scale
//! resolution and the in-memory (`out=None` → `KnowledgeGraph`) convenience.

use kglite_core::api::{graphgen as core_graphgen, GraphGenConfig};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;

/// Stream the synthetic org/social graph as one CSV per type + `manifest.json`
/// into `out` (created if needed), in bounded memory. Returns
/// `{'nodes', 'edges', 'out'}`.
#[pyfunction]
#[pyo3(signature = (out, persons, knows_per=8, seed=1234, zipf=true, zipf_exp=1.6))]
fn graphgen_to_dir(
    py: Python<'_>,
    out: &str,
    persons: u64,
    knows_per: u64,
    seed: u64,
    zipf: bool,
    zipf_exp: f64,
) -> PyResult<Py<PyAny>> {
    let cfg = GraphGenConfig {
        persons,
        knows_per,
        seed,
        zipf,
        zipf_exp,
    };
    // Release the GIL — this is pure CPU + disk I/O, no Python objects touched.
    let stats = py
        .detach(|| core_graphgen(&cfg, Path::new(out)))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("graphgen: {e}")))?;
    let d = PyDict::new(py);
    d.set_item("nodes", stats.nodes)?;
    d.set_item("edges", stats.edges)?;
    d.set_item("out", stats.out_dir.to_string_lossy())?;
    Ok(d.into())
}

pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(graphgen_to_dir, m)?)?;
    Ok(())
}
