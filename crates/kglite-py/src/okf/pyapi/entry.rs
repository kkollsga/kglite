//! Public Python functions for OKF ingestion: `build`.

use pyo3::prelude::*;
use std::path::PathBuf;

use crate::graph::KnowledgeGraph;
use crate::okf::{BuildOptions, Dialect};

/// Build a KnowledgeGraph from an OKF bundle directory.
///
/// `dialect`: `"okf"` (default — strict markdown links) or `"loose"`/`"obsidian"`
/// (also resolve `[[wikilinks]]`, tolerate missing `type`). `with_body`: store
/// each concept's markdown body as a `body` property (off by default — bodies are
/// read on demand via the `file_path` pointer).
#[pyfunction]
#[pyo3(signature = (path, *, dialect=None, with_body=false, embed=false))]
pub fn build(
    py: Python<'_>,
    path: PathBuf,
    dialect: Option<String>,
    with_body: bool,
    embed: bool,
) -> PyResult<KnowledgeGraph> {
    let opts = BuildOptions {
        dialect: Dialect::parse(dialect.as_deref()),
        with_body,
        embed,
    };
    py.detach(|| crate::okf::build(&path, &opts))
        .map(KnowledgeGraph::from_arc)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}
