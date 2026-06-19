// crates/kglite/src/graph/mod.rs
//
// Pure-Rust core of the graph engine. The `KnowledgeGraph` /
// `Transaction` / `ResultView` / `ResultIter` Python-facing structs
// + PyO3 helpers stay in the kglite-py wrapper crate (workspace
// root pre-G.4) because they carry `#[pyclass]` / `#[pymethods]`
// attributes. This module declares the engine submodules + a few
// shared types those wrappers reference.

pub mod algorithms;
pub mod blueprint;
pub mod core;
pub mod dir_graph;
pub mod embedder;
pub mod embedding_carry;
pub mod explore;
pub mod features;
pub mod handle;
pub mod introspection;
pub mod io;
pub mod languages;
pub mod mutation;
pub mod schema;
pub mod session;
pub mod storage;
pub mod wal;

// Re-export DirGraph at the graph-mod top level — matches the
// path the executor / planner / blueprint code uses
// (`crate::graph::DirGraph`). Actual definition lives in
// `dir_graph` and is re-exported by `schema` too.
pub use dir_graph::DirGraph;

/// Temporal context for automatic date filtering on select /
/// traverse / collect. Set via `KnowledgeGraph::date()` (Python:
/// `g.date(...)`). Carried through clone (fluent API chaining).
#[derive(Clone, Debug, Default)]
pub enum TemporalContext {
    /// Use today's date (default). Resolved at query time.
    #[default]
    Today,
    /// Point-in-time: valid_from <= date AND (valid_to IS NULL OR valid_to >= date).
    At(chrono::NaiveDate),
    /// Range overlap: valid_from <= end AND (valid_to IS NULL OR valid_to >= start).
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
/// what `code_tree` stores on the node — graphs built from
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
