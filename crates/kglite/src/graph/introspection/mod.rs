//! Schema introspection — exploring graph structure.
//!
//! All functions take `&DirGraph` and return Rust structs. PyO3 conversion
//! happens in `graph::pyapi`.
//!
//! Submodules (Phase 9 split of the monolithic `introspection.rs`):
//! - [`capabilities`] — TypeCapabilities + endpoint-type discovery
//! - [`connectivity`] — type connectivity triples + neighbors_from_triples
//! - [`schema_overview`] — schema / property / neighbors / sample / join candidates
//! - [`describe`] — XML describe() entry point + inventory builders + XML writers
//! - [`topics`] — tier-3 Cypher + Fluent topic detail writers
//! - [`bug_report`] / [`debugging`] / [`reporting`] — Phase 7 helpers (pre-existing)

pub mod bug_report;
pub mod capabilities;
pub mod connectivity;
pub mod debugging;
pub mod describe;
pub mod reporting;
pub mod schema_overview;
pub mod topics;

use crate::datatypes::values::Value;
use crate::graph::schema::DirGraph;
use std::collections::HashMap;

pub use connectivity::{compute_type_connectivity, derive_edge_counts_from_triples};
pub use describe::{compute_description, mcp_quickstart};
pub use schema_overview::{
    compute_connection_type_stats, compute_neighbors_schema, compute_property_stats, compute_schema,
};

// ── Return types ────────────────────────────────────────────────────────────

/// Statistics about a connection type: count, source/target node types, property names.
pub struct ConnectionTypeStats {
    pub connection_type: String,
    pub count: usize,
    pub source_types: Vec<String>,
    pub target_types: Vec<String>,
    pub property_names: Vec<String>,
}

/// Summary of a node type: count and property schemas with types.
pub struct NodeTypeOverview {
    pub count: usize,
    pub properties: HashMap<String, String>,
}

/// Complete schema summary: all node types, connection types, indexes, and totals.
pub struct SchemaOverview {
    pub node_types: Vec<(String, NodeTypeOverview)>,
    pub connection_types: Vec<ConnectionTypeStats>,
    pub indexes: Vec<String>,
    pub node_count: usize,
    pub edge_count: usize,
}

/// Per-property statistics: data type, non-null count, unique count, and optional value list.
pub struct PropertyStatInfo {
    pub property_name: String,
    pub type_string: String,
    pub non_null: usize,
    pub unique: usize,
    pub values: Option<Vec<Value>>,
    /// One example value when `values` is None (high-cardinality property).
    /// Lets the schema output show what a property *looks like* even when
    /// the full set is too large to enumerate. Added in 0.9.30 for
    /// operator-reported friction where the agent had to guess the value
    /// shape of properties like `file_path` from the name alone.
    pub sample: Option<Value>,
}

/// A single neighbor connection: edge type, connected node type, and count.
#[derive(Clone)]
pub struct NeighborConnection {
    pub connection_type: String,
    pub other_type: String,
    pub count: usize,
}

/// Grouped neighbor connections for a node type: incoming and outgoing edges.
#[derive(Clone)]
pub struct NeighborsSchema {
    pub outgoing: Vec<NeighborConnection>,
    pub incoming: Vec<NeighborConnection>,
}

/// Scale classification for adaptive describe output.
pub enum GraphScale {
    /// 0-15 core types: full inline detail.
    Small,
    /// 16-200 core types: compact inventory listing.
    Medium,
    /// 201-5000 core types: top-N types with summary.
    Large,
    /// 5001+ core types: statistical summary, search required.
    Extreme,
}

/// Classify graph scale by core type count (excluding supporting types).
pub fn graph_scale(graph: &DirGraph) -> GraphScale {
    let core_count = graph
        .type_indices
        .keys()
        .filter(|nt| !graph.parent_types.contains_key(*nt))
        .count();
    match core_count {
        0..=15 => GraphScale::Small,
        16..=200 => GraphScale::Medium,
        201..=5000 => GraphScale::Large,
        _ => GraphScale::Extreme,
    }
}

/// Level of Cypher documentation requested via `describe(cypher=...)`.
pub enum CypherDetail {
    /// No Cypher docs (default).
    Off,
    /// Tier 2: compact reference listing — all clauses, operators, functions, procedures.
    Overview,
    /// Tier 3: detailed docs with params and examples for specific topics.
    Topics(Vec<String>),
}

/// Level of fluent API documentation requested via `describe(fluent=...)`.
pub enum FluentDetail {
    /// No fluent docs (default).
    Off,
    /// Compact reference: all methods grouped by area with 1-line descriptions.
    Overview,
    /// Detailed docs with params and examples for specific topics.
    Topics(Vec<String>),
}

/// Level of connection documentation requested via `describe(connections=...)`.
pub enum ConnectionDetail {
    /// No standalone connection docs (default — connections shown in inventory).
    Off,
    /// Overview: all connection types with count, endpoints, property names.
    Overview,
    /// Deep-dive: specific connection types with per-pair counts, property stats, samples.
    Topics(Vec<String>),
}
