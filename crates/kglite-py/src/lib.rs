// src/lib.rs

// Phase A.2 / C2 — crate-wide allow for clippy::result_large_err.
// `KgError` is intentionally rich (16 variants spanning Cypher /
// schema / IO / argument validation) so its size pushes past clippy's
// default 128-byte threshold. Boxing the error variant in every
// `Result<T, KgError>` would add an allocation per error path for no
// real benefit — error paths aren't hot. Standard pattern for crates
// with a unified typed error.
#![allow(clippy::result_large_err)]
// kglite-py is the PyO3 wrapper over the kglite engine crate
// (aliased as `kglite_core` in this crate's source — see
// Cargo.toml for the `package = "kglite"` indirection). The
// local module shims (`graph/mod.rs`, `graph/languages/mod.rs`,
// `graph/embedder/mod.rs`, `code_tree/mod.rs`, `datatypes/mod.rs`)
// pull in the engine's content via `pub use kglite_core::*::*;`
// glob re-exports + add the pyo3-only submodules. clippy flags
// the legitimate shadowing pattern under
// `hidden_glob_reexports`. The `unused_imports` allows the same
// pattern in nested shims.
#![allow(hidden_glob_reexports)]
#![allow(unused_imports)]

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
mod error_py;
mod graph;
mod mcp_tools;
mod sec;
mod sodir;
mod wikidata;

// The pyo3 wrapper depends on the kglite engine for everything
// non-Python. Re-export the engine's `error` module so existing
// `crate::error::*` paths in pyapi/, error_py.rs, the datatypes
// shims, etc. resolve unchanged.
pub use kglite_core::error;

use graph::pyapi::blueprint::from_blueprint_rust;
use graph::pyapi::result_view::{ResultIter, ResultView};
use graph::{KnowledgeGraph, Transaction};
use kglite_core::graph::io::file::load_file;

/// Curated Rust-side façade for downstream binaries (notably
/// `kglite-mcp-server`). This module is the **only** stable Rust
/// API the kglite-py wrapper promises to keep — the underlying
/// `pub mod graph` / `pub mod code_tree` are public for tooling
/// but their internals can move between minor releases. New
/// consumers should import from `kglite::api::*` (or
/// `kglite_core::api::*` from inside this crate's source, where
/// the dep is aliased); breakage there is a semver concern.
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
    // Typed error surface — Phase A.2 added KgError + KgErrorCode for the
    // Python boundary; Phase C.6 (bolt-server) consumes them to map onto
    // Neo4j `Neo.ClientError.*` wire codes via `BoltError::Query`.
    pub use crate::error::{KgError, KgErrorCode};
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
    ///
    /// **For new consumers, prefer [`session`]** — it bundles the
    /// canonical pipeline + transaction CoW into a single surface
    /// so future drift between bindings is impossible. This raw
    /// `cypher` re-export stays public for callers that need to
    /// reach into specific passes (planner introspection,
    /// custom-disabled-pass sets, etc.).
    pub mod cypher {
        pub use crate::graph::languages::cypher::ast::CypherQuery;
        pub use crate::graph::languages::cypher::ast::OutputFormat;
        pub use crate::graph::languages::cypher::executor::write::execute_mutable;
        pub use crate::graph::languages::cypher::executor::CypherExecutor;
        pub use crate::graph::languages::cypher::generate_explain_result;
        pub use crate::graph::languages::cypher::is_mutation_query;
        pub use crate::graph::languages::cypher::parser::parse_cypher;
        pub use crate::graph::languages::cypher::planner;
        pub use crate::graph::languages::cypher::planner::mark_lazy_eligibility;
        pub use crate::graph::languages::cypher::planner::schema_check::validate_schema;
        pub use crate::graph::languages::cypher::planner::simplification::rewrite_text_score;
        pub use crate::graph::languages::cypher::result::CypherResult;
    }

    /// Canonical query + transaction surface. Single source of truth
    /// for the Cypher pipeline (parse → validate → rewrite → optimize
    /// → execute) and the snapshot/working CoW transaction model.
    /// All bindings (pyapi, mcp-server, bolt-server, future Go/TS/JVM)
    /// wrap this module's types and free functions.
    ///
    /// See `docs/explanation/session.md` for the operator-facing
    /// guide and `bolt_implementation.md` Phase E for the rationale.
    pub mod session {
        pub use crate::graph::session::{
            execute_mut, execute_read, CommitOutcome, ExecuteOptions, ExecuteOutcome, Session,
            Transaction,
        };
    }
}

/// Read-only accessor for the underlying [`DirGraph`] of a
/// [`api::KnowledgeGraph`]. The struct field is private; this method
/// gives downstream Rust binaries a stable handle to plug into the
/// planner / executor surface in [`api::cypher`].
impl crate::graph::KnowledgeGraph {
    pub fn dir(&self) -> &std::sync::Arc<kglite_core::graph::dir_graph::DirGraph> {
        &self.inner
    }
}

#[pyfunction]
fn load(py: Python<'_>, path: String) -> PyResult<KnowledgeGraph> {
    py.detach(|| load_file(&path))
        .map(KnowledgeGraph::from_arc)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
}

/// Names of every Cypher optimizer pass, in execution order. Useful for
/// the `disabled_passes=` kwarg of `KnowledgeGraph.cypher()` and for
/// bisection scripts. The list is the source of truth — names that
/// aren't here will be rejected by `cypher(..., disabled_passes=[...])`.
#[pyfunction]
fn cypher_pass_names() -> Vec<String> {
    kglite_core::graph::languages::cypher::planner::all_pass_names()
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
