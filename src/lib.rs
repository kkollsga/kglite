// src/lib.rs

// Phase A.2 / C2 — crate-wide allow for clippy::result_large_err.
// `KgError` is intentionally rich (16 variants spanning Cypher /
// schema / IO / argument validation) so its size pushes past clippy's
// default 128-byte threshold. Boxing the error variant in every
// `Result<T, KgError>` would add an allocation per error path for no
// real benefit — error paths aren't hot. Standard pattern for crates
// with a unified typed error.
#![allow(clippy::result_large_err)]

// mimalloc as the global allocator. samply profile of the N-Triples
// build showed libsystem_malloc accounting for ~32% of loader-thread
// CPU time. mimalloc is consistently faster than macOS's default
// allocator on small-object-heavy workloads (Strings, HashMaps, Vecs
// in the parser hot loop). Pure Rust dependency — no system dep, just
// a slightly larger build artifact.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use pyo3::prelude::*;
mod code_tree;
mod datatypes;
mod error;
mod error_py;
mod graph;
mod mcp_tools;
mod sec;
mod sodir;
mod wikidata;
use graph::io::file::load_file;
use graph::pyapi::blueprint::from_blueprint_rust;
use graph::pyapi::result_view::{ResultIter, ResultView};
use graph::{KnowledgeGraph, Transaction};

/// Curated Rust-side façade for downstream binaries (notably
/// `kglite-mcp-server`). This module is the **only** stable Rust API
/// kglite promises to keep — the underlying `pub mod graph` /
/// `pub mod code_tree` are public for tooling but their internals can
/// move between minor releases. New consumers should import from
/// `kglite::api::*`; existing breakage there is a semver concern.
///
/// The Python API (`#[pymethods]` on `KnowledgeGraph`, etc.) is
/// independent — it stays as the wheel's primary surface.
pub mod api {
    pub use crate::code_tree::builder::run_with_options as build_code_tree;
    pub use crate::datatypes::Value;
    // Per-variant carriers for the Value enum's compound shapes. Phase A.1
    // added `Value::Node` / `Relationship` / `Path` carrying these struct
    // types; downstream Rust consumers (kglite-bolt-server's value adapter,
    // and future Arrow/Polars exporters) want to pattern-match into them
    // without re-deriving accessors.
    pub use crate::datatypes::values::{NodeValue, PathValue, RelValue};
    pub use crate::graph::dir_graph::DirGraph;
    #[cfg(feature = "fastembed")]
    pub use crate::graph::embedder::fastembed::FastEmbedAdapter;
    pub use crate::graph::embedder::Embedder;
    pub use crate::graph::explore::{explore_markdown, ExploreOptions};
    pub use crate::graph::introspection::describe::compute_description;
    pub use crate::graph::introspection::schema_overview::compute_schema;
    pub use crate::graph::introspection::SchemaOverview;
    pub use crate::graph::introspection::{ConnectionDetail, CypherDetail, FluentDetail};
    pub use crate::graph::io::file::{load_file, save_graph};
    pub use crate::graph::{KnowledgeGraph, SourceLocation, SourceLookup};

    /// Cypher parser + planner + executor surface. Downstream Rust
    /// consumers (notably `kglite-mcp-server`) build their own
    /// parse → rewrite_text_score → optimize → execute pipeline using
    /// these items; the Python boundary in
    /// `src/graph/pyapi/kg_core.rs::cypher` is the canonical example.
    pub mod cypher {
        pub use crate::graph::languages::cypher::ast::OutputFormat;
        pub use crate::graph::languages::cypher::executor::write::execute_mutable;
        pub use crate::graph::languages::cypher::executor::CypherExecutor;
        pub use crate::graph::languages::cypher::is_mutation_query;
        pub use crate::graph::languages::cypher::parser::parse_cypher;
        pub use crate::graph::languages::cypher::planner;
        pub use crate::graph::languages::cypher::planner::mark_lazy_eligibility;
        pub use crate::graph::languages::cypher::planner::schema_check::validate_schema;
        pub use crate::graph::languages::cypher::planner::simplification::rewrite_text_score;
        pub use crate::graph::languages::cypher::result::CypherResult;
    }
}

/// Read-only accessor for the underlying [`DirGraph`] of a
/// [`api::KnowledgeGraph`]. The struct field is private; this method
/// gives downstream Rust binaries a stable handle to plug into the
/// planner / executor surface in [`api::cypher`].
impl crate::graph::KnowledgeGraph {
    pub fn dir(&self) -> &std::sync::Arc<crate::graph::dir_graph::DirGraph> {
        &self.inner
    }
}

#[pyfunction]
fn load(py: Python<'_>, path: String) -> PyResult<KnowledgeGraph> {
    py.detach(|| load_file(&path))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
}

/// Names of every Cypher optimizer pass, in execution order. Useful for
/// the `disabled_passes=` kwarg of `KnowledgeGraph.cypher()` and for
/// bisection scripts. The list is the source of truth — names that
/// aren't here will be rejected by `cypher(..., disabled_passes=[...])`.
#[pyfunction]
fn cypher_pass_names() -> Vec<String> {
    graph::languages::cypher::planner::all_pass_names()
}

#[pymodule]
fn kglite(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(from_blueprint_rust, m)?)?;
    m.add_function(wrap_pyfunction!(cypher_pass_names, m)?)?;
    m.add_class::<KnowledgeGraph>()?;
    m.add_class::<Transaction>()?;
    m.add_class::<ResultView>()?;
    m.add_class::<ResultIter>()?;
    // Phase A.2 / C1 — typed exception class hierarchy. Every kglite
    // error surfaces as `kglite.KgError` or a more specific subclass.
    error_py::register(py, m)?;
    code_tree::pyapi::register(py, m)?;
    mcp_tools::register(py, m)?;
    sec::register(py, m)?;
    sodir::register(py, m)?;
    wikidata::register(py, m)?;
    Ok(())
}
