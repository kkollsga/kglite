//! Public Python functions: build, read_manifest, repo_tree.
//!
//! These are the *only* Python-callable entry points for code_tree.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::{Path, PathBuf};

use crate::graph::KnowledgeGraph;

/// Map a file path to its `code_tree` language identifier, or `None`
/// if no parser handles the file. Used by the MCP watch callback to
/// decide whether a filesystem event is graph-relevant. Wraps
/// `kglite::api::language_for_path` for the Python side.
#[pyfunction]
pub fn language_for_path(path: &str) -> Option<&'static str> {
    kglite_core::api::code_tree::language_for_path(Path::new(path))
}

/// Parse a directory into a KnowledgeGraph.
///
/// Set ``include_docs=True`` to also ingest the repo's markdown as ``:Doc``
/// nodes and link them to the code they describe
/// (``(:Doc)-[:MENTIONS]->(:Function|:Class|…)`` and
/// ``(:Doc)-[:DOCUMENTS]->(:Doc|:File)``). Off by default (code-only graph).
///
/// Pass ``rev`` (a git tag / branch / SHA) to build the codebase as it existed
/// at that revision instead of the working tree. The revision's tracked files
/// are materialized into a tempdir via ``git archive`` — ``HEAD`` and the
/// working tree are never touched, and uncommitted changes are excluded. The
/// git root is auto-resolved from ``src_dir`` (override with ``repo_root``);
/// a bad rev or non-git directory raises a clear error. The built graph's
/// ``describe()`` records the revision it represents.
///
/// Pass ``revs`` (a list of git revspecs, oldest → newest) to merge N
/// revisions into ONE multi-rev graph: one node per entity across revs, each
/// node carrying native list props ``revs: [str]`` (revisions it appears in) +
/// ``rev_fp: [int]`` (per-rev shape fingerprint), and each edge carrying
/// ``revs: [str]``. Ordinary properties report the newest rev an entity appears
/// in (newest-wins). Unscoped queries span ALL revs (an over-count trap) — scope
/// with ``WHERE 'v1' IN n.revs``; use ``CALL rev_diff({from:'v1', to:'v2'})``
/// for deltas. ``describe()`` lists the loaded revs and teaches the scoping
/// idiom. ``rev`` and ``revs`` are mutually exclusive.
#[pyfunction]
#[pyo3(signature = (src_dir, *, save_to=None, verbose=false, include_tests=true, max_loc_per_file=None, include_docs=false, rev=None, revs=None, repo_root=None))]
#[allow(clippy::too_many_arguments)]
pub fn build(
    py: Python<'_>,
    src_dir: PathBuf,
    save_to: Option<PathBuf>,
    verbose: bool,
    include_tests: bool,
    max_loc_per_file: Option<usize>,
    include_docs: bool,
    rev: Option<String>,
    revs: Option<Vec<String>>,
    repo_root: Option<PathBuf>,
) -> PyResult<KnowledgeGraph> {
    if rev.is_some() && revs.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "build(): `rev` and `revs` are mutually exclusive — pass one git \
             revision as `rev=`, or a list of revisions as `revs=[...]` to merge \
             into a multi-rev graph.",
        ));
    }
    py.detach(|| match (rev, revs) {
        (_, Some(revs)) => crate::code_tree::rev::build_code_tree_revs(
            &src_dir,
            &revs,
            repo_root.as_deref(),
            verbose,
            include_tests,
            save_to.as_deref(),
            max_loc_per_file,
            include_docs,
        ),
        (Some(rev), None) => crate::code_tree::rev::archive_and_build(
            &src_dir,
            &rev,
            repo_root.as_deref(),
            verbose,
            include_tests,
            save_to.as_deref(),
            max_loc_per_file,
            include_docs,
        ),
        (None, None) => crate::code_tree::builder::run_with_options(
            &src_dir,
            verbose,
            include_tests,
            save_to.as_deref(),
            max_loc_per_file,
            include_docs,
        ),
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
///
/// Set ``include_docs=True`` to also ingest the repo's markdown as ``:Doc``
/// nodes linked to the code they describe (see :func:`build`).
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
    include_docs=false,
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
    include_docs: bool,
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
            include_docs,
        )
    })
    .map(KnowledgeGraph::from_arc)
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}
