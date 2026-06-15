//! Public Python functions for OKF ingestion: `build`.

use pyo3::prelude::*;
use std::path::PathBuf;

use crate::graph::KnowledgeGraph;
use crate::okf::{BuildOptions, Dialect};

/// Build a KnowledgeGraph from an OKF bundle directory.
///
/// `dialect`: `"okf"` (default — strict markdown links) or `"loose"`/`"obsidian"`
/// (also resolve `[[wikilinks]]`, tolerate missing `type`). `require_frontmatter`
/// (default `True`): ingest only `.md` files with YAML frontmatter — the
/// structured-knowledge vs plain-markdown discriminator; set `False` to ingest
/// every `.md`. `respect_skip` (default `True`): honor the `kg_skip: true`
/// frontmatter marker that opts a file out of the sweep; set `False` to ingest
/// skip-marked files anyway. `with_body`: store each concept's markdown body as a
/// `body` property (off by default — bodies are read on demand via the
/// `file_path` pointer).
#[pyfunction]
#[pyo3(signature = (path, *, dialect=None, require_frontmatter=true, respect_skip=true, with_body=false, embed=false))]
pub fn build(
    py: Python<'_>,
    path: PathBuf,
    dialect: Option<String>,
    require_frontmatter: bool,
    respect_skip: bool,
    with_body: bool,
    embed: bool,
) -> PyResult<KnowledgeGraph> {
    let opts = BuildOptions {
        dialect: Dialect::parse(dialect.as_deref()),
        require_frontmatter,
        respect_skip,
        with_body,
        embed,
    };
    py.detach(|| crate::okf::build(&path, &opts))
        .map(KnowledgeGraph::from_arc)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Read a concept's markdown body on demand (frontmatter stripped).
///
/// Pairs with partial ingestion: the graph stores each concept's `file_path`;
/// pass that path (joined with the bundle root) here to fetch the prose when an
/// agent has narrowed to a single concept.
#[pyfunction]
pub fn source(path: PathBuf) -> PyResult<String> {
    crate::okf::read_body(&path).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}
