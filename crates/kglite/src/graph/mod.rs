// crates/kglite/src/graph/mod.rs
//
// Pure-Rust core of the graph engine. The `KnowledgeGraph` /
// `Transaction` / `ResultView` / `ResultIter` Python-facing structs
// + PyO3 helpers stay in the kglite-py wrapper crate (workspace
// root pre-G.4) because they carry `#[pyclass]` / `#[pymethods]`
// attributes. This module declares the engine submodules + a few
// shared types those wrappers reference.

pub mod advisories;
pub mod algorithms;
pub mod blueprint;
pub mod cdc;
pub mod constraints;
pub mod core;
pub(crate) mod cow;
pub mod diagnostics;
pub mod dir_graph;
pub mod durability;
pub(crate) mod durable_io;
pub(crate) mod edge_embedding_generation;
pub(crate) mod edge_embeddings;
pub mod embedder;
pub mod embedding_carry;
pub(crate) mod embedding_hints;
pub mod embedding_inventory;
pub(crate) mod embedding_quantization;
pub(crate) mod embedding_store;
pub(crate) mod embedding_validation;
pub mod embeddings;
pub mod explore;
pub mod features;
pub mod handle;
pub mod index_freshness;
pub mod introspection;
pub mod io;
pub mod languages;
pub(crate) mod memory_release;
pub mod mutation;
pub mod ontology;
pub(crate) mod parallel;
pub mod property_types;
pub mod property_vector_indexes;
pub mod quantized;
pub mod recipes;
pub mod schema;
pub mod schema_json;
pub mod session;
pub mod skills;
pub mod storage;
pub mod tables;
#[cfg(test)]
mod test_scope;
pub mod text_indexes;
pub(crate) mod text_scan;
pub mod value_type;
pub mod vector_buf;
pub mod wal;

#[cfg(test)]
#[path = "advisories_tests.rs"]
mod advisories_tests;

#[cfg(test)]
#[path = "value_byte_identity_tests.rs"]
mod value_byte_identity_tests;

// Declared here rather than beside the other edge-embedding test modules in
// `edge_embeddings.rs`: these cases drive `Session::open_durable` and the WAL,
// not the embedding store's own seam, so nothing in them is `super::`-scoped.
#[cfg(test)]
#[path = "edge_embedding_durable_recovery_tests.rs"]
mod edge_embedding_durable_recovery_tests;

// Re-export DirGraph at the graph-mod top level — matches the
// path the executor / planner / blueprint code uses
// (`crate::graph::DirGraph`). Actual definition lives in
// `dir_graph` and is re-exported by `schema` too.
pub use dir_graph::DirGraph;

/// The fluent cursor's valid-time context, set by `date()` and carried
/// through the chain. Every fluent step that reads declared elements —
/// `select`, `traverse`, `expand`, `where_connected`, `where_orphans`,
/// `degrees`, `relationships`, `compare`, `to_subgraph`, `save_subset` —
/// resolves it into
/// the filter a
/// `FOR VALID_TIME AS OF` statement runs under (`api::fluent::FluentFilter`).
#[derive(Clone, Debug, Default)]
pub enum TemporalContext {
    /// The graph's valid-time default (the cursor's default), resolved at
    /// each step: today's UTC date unless `DirGraph::valid_time_default`
    /// says every version or a fixed day.
    #[default]
    Today,
    /// Valid at one day, under each declaration's convention.
    At(chrono::NaiveDate),
    /// Valid at some point of `[start, end]`.
    During(chrono::NaiveDate, chrono::NaiveDate),
    /// No temporal filtering — show everything regardless of
    /// validity dates.
    All,
}

impl TemporalContext {
    pub fn is_all(&self) -> bool {
        matches!(self, TemporalContext::All)
    }
}

/// Resolved code-entity location returned by
/// `KnowledgeGraph::source_location`. All optional fields mirror
/// what code-graph builders (e.g. codingest) store on the node — graphs built from
/// non-code-tree sources may have fewer populated.
#[derive(Debug, Clone)]
pub struct SourceLocation {
    pub type_name: String,
    pub name: String,
    pub qualified_name: String,
    pub file_path: Option<String>,
    pub line_number: Option<i64>,
    pub end_line: Option<i64>,
    pub signature: Option<String>,
}

/// Outcome of a `KnowledgeGraph::source_location` lookup.
#[derive(Debug, Clone)]
pub enum SourceLookup {
    Found(SourceLocation),
    /// Multiple code entities matched the given (name, node_type).
    /// The payload lists each match's qualified_name so the caller
    /// can ask the agent to disambiguate.
    Ambiguous(Vec<String>),
    NotFound,
}

// (The canonical `resolve_noderefs` lives in `graph::session`; it is the
// one re-exported as `kglite::api::session::resolve_noderefs`. A dead
// duplicate here was removed when the hard seal surfaced it.)
