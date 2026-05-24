//! Public Python functions: build, read_manifest, repo_tree.
//!
//! These are the *only* Python-callable entry points for code_tree.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

use crate::graph::KnowledgeGraph;

/// Parse a directory into a KnowledgeGraph.
#[pyfunction]
#[pyo3(signature = (src_dir, *, save_to=None, verbose=false, include_tests=true, max_loc_per_file=None))]
pub fn build(
    py: Python<'_>,
    src_dir: PathBuf,
    save_to: Option<PathBuf>,
    verbose: bool,
    include_tests: bool,
    max_loc_per_file: Option<usize>,
) -> PyResult<KnowledgeGraph> {
    py.detach(|| {
        crate::code_tree::builder::run_with_options(
            &src_dir,
            verbose,
            include_tests,
            save_to.as_deref(),
            max_loc_per_file,
        )
    })
    .map(KnowledgeGraph::from_arc)
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Read a project manifest and return a dict of project metadata.
#[pyfunction]
pub fn read_manifest<'py>(
    py: Python<'py>,
    project_root: PathBuf,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let Some(info) = crate::code_tree::manifest::read_manifest(&project_root) else {
        return Ok(None);
    };
    let d = PyDict::new(py);
    d.set_item("name", info.name)?;
    d.set_item("version", info.version)?;
    d.set_item("description", info.description)?;
    d.set_item("languages", info.languages)?;
    d.set_item("authors", info.authors)?;
    d.set_item("license", info.license)?;
    d.set_item("repository_url", info.repository_url)?;
    d.set_item("manifest_path", info.manifest_path)?;
    d.set_item("build_system", info.build_system)?;
    let src_roots: Vec<String> = info
        .source_roots
        .iter()
        .map(|r| r.path.to_string_lossy().to_string())
        .collect();
    d.set_item("source_roots", src_roots)?;
    let test_roots: Vec<String> = info
        .test_roots
        .iter()
        .map(|r| r.path.to_string_lossy().to_string())
        .collect();
    d.set_item("test_roots", test_roots)?;
    Ok(Some(d))
}

/// Clone a GitHub repo and build its KnowledgeGraph.
#[pyfunction]
#[pyo3(signature = (
    repo,
    *,
    save_to=None,
    clone_to=None,
    branch=None,
    token=None,
    verbose=false,
    include_tests=true,
    max_loc_per_file=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn repo_tree(
    py: Python<'_>,
    repo: String,
    save_to: Option<PathBuf>,
    clone_to: Option<PathBuf>,
    branch: Option<String>,
    token: Option<String>,
    verbose: bool,
    include_tests: bool,
    max_loc_per_file: Option<usize>,
) -> PyResult<KnowledgeGraph> {
    py.detach(|| {
        crate::code_tree::repo::clone_and_build(
            &repo,
            save_to.as_deref(),
            clone_to.as_deref(),
            branch.as_deref(),
            token.as_deref(),
            verbose,
            include_tests,
            max_loc_per_file,
        )
    })
    .map(KnowledgeGraph::from_arc)
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}
