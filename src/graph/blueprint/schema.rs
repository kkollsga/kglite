//! Serde types for the blueprint JSON schema.
//!
//! See docs/guides/blueprints.md for the user-facing spec. These structs
//! are lenient: unknown fields are allowed and missing fields default to
//! empty where sensible, matching the behaviour of the old Python loader.

use indexmap::IndexMap;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Default)]
pub struct Blueprint {
    #[serde(default)]
    pub settings: Settings,
    /// Node specs, in blueprint-JSON order. Iteration order matters because
    /// the FK-edge phase writes parallel edges on the *first* call per
    /// connection type (then dedupes on subsequent calls). Alphabetical
    /// order would produce different edge counts than the Python loader.
    #[serde(default)]
    pub nodes: IndexMap<String, NodeSpec>,
    /// Optional ordered pipeline of post-load compute primitives.
    /// 0.9.47+: each `ComputeOp` runs after the 5 existing load phases.
    /// Vec order = execution order; later ops can reference types
    /// produced by earlier ops.
    #[serde(default)]
    pub compute: Vec<ComputeOp>,
}

#[derive(Debug, Deserialize, Default)]
pub struct Settings {
    #[serde(default, alias = "root")]
    pub input_root: Option<String>,
    #[serde(default)]
    pub output_path: Option<String>,
    #[serde(default, alias = "output")]
    pub output_file: Option<String>,
}

impl Settings {
    /// Compute the absolute output path from `output_path` + `output_file`,
    /// falling back to `input_root / output_file`. Returns None if no output
    /// was configured.
    pub fn resolved_output(&self, input_root: &std::path::Path) -> Option<PathBuf> {
        let output_file = self.output_file.as_ref()?;
        let base = match &self.output_path {
            Some(p) => std::path::PathBuf::from(p),
            None => input_root.to_path_buf(),
        };
        Some(base.join(output_file))
    }
}

#[derive(Debug, Deserialize, Default)]
pub struct NodeSpec {
    #[serde(default)]
    pub csv: Option<String>,
    #[serde(default)]
    pub pk: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub parent: Option<String>,
    #[serde(default)]
    pub parent_fk: Option<String>,
    #[serde(default)]
    pub properties: IndexMap<String, String>,
    #[serde(default)]
    pub skipped: Vec<String>,
    #[serde(default)]
    pub filter: IndexMap<String, serde_json::Value>,
    #[serde(default)]
    pub connections: Connections,
    #[serde(default)]
    pub sub_nodes: IndexMap<String, NodeSpec>,
    #[serde(default)]
    pub timeseries: Option<TimeseriesSpec>,
}

#[derive(Debug, Deserialize, Default)]
pub struct Connections {
    #[serde(default)]
    pub fk_edges: IndexMap<String, FkEdge>,
    #[serde(default)]
    pub junction_edges: IndexMap<String, JunctionEdge>,
}

#[derive(Debug, Deserialize)]
pub struct FkEdge {
    pub target: String,
    pub fk: String,
}

#[derive(Debug, Deserialize)]
pub struct JunctionEdge {
    pub csv: String,
    pub source_fk: String,
    pub target: String,
    pub target_fk: String,
    #[serde(default)]
    pub properties: Vec<String>,
    #[serde(default)]
    pub property_types: IndexMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum TimeKey {
    Single(String),
    Composite(IndexMap<String, String>),
}

#[derive(Debug, Deserialize)]
pub struct TimeseriesSpec {
    pub time_key: TimeKey,
    #[serde(default)]
    pub channels: IndexMap<String, String>,
    #[serde(default)]
    pub resolution: Option<String>,
    #[serde(default)]
    pub units: IndexMap<String, String>,
}

// ─── compute pipeline (0.9.47) ────────────────────────────────────────────

/// One operation in the blueprint's `compute:` pipeline. Each variant
/// is a named primitive with a fixed shape — no free-form DSL, no
/// user-defined functions, no graph traversal in expressions.
/// Cypher handles the post-build dynamic side; this layer handles
/// declarative graph shaping.
///
/// K2 ships the type + serde parsing + validation; per-variant
/// fields become "read" as K3-K6 wire each primitive's executor.
#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum ComputeOp {
    /// Add or overwrite properties on an existing node type via
    /// row-level expressions. Schema gains the new properties.
    Derive {
        from: String,
        set: IndexMap<String, String>,
    },
    /// Copy nodes matching a predicate from one type to another (or
    /// drop non-matching rows in place if `into` is omitted). The
    /// predicate is a row-level boolean expression.
    Filter {
        from: String,
        #[serde(rename = "where")]
        where_expr: String,
        #[serde(default)]
        into: Option<String>,
    },
    /// Synthesise a doubly-linked-list edge between consecutive nodes
    /// of a type, grouped by composite key and ordered by a property.
    /// Used for temporal walks (NEXT_TX per insider, NEXT_QUARTER
    /// per fund/security HOLDS series).
    Chain {
        from: String,
        group_by: Vec<String>,
        order_by: String,
        edge: String,
    },
    /// Synthesise `:Date` nodes for the closed range `[start, end]`
    /// plus chain + hierarchy edges, then link source-type date
    /// columns to the matching Date node.
    Calendar {
        #[serde(rename = "type", default = "default_calendar_type")]
        node_type: String,
        start: String,
        end: String,
        #[serde(default = "default_next_day_edge")]
        next_edge: String,
        #[serde(default)]
        in_month_edge: Option<String>,
        #[serde(default)]
        in_quarter_edge: Option<String>,
        #[serde(default)]
        in_year_edge: Option<String>,
        #[serde(default)]
        links: Vec<CalendarLink>,
    },
    /// Group source nodes by a composite key, evaluate per-group
    /// aggregate expressions, emit one summary node per group plus
    /// optional FK edges to the group-key target types.
    Aggregate {
        from: String,
        group_by: Vec<String>,
        into: String,
        agg: IndexMap<String, String>,
        #[serde(default)]
        edges: Vec<AggregateEdge>,
    },
}

#[derive(Debug, Deserialize, Clone)]
pub struct CalendarLink {
    pub from: String,
    pub date_col: String,
    pub edge: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AggregateEdge {
    pub to: String,
    pub fk: String,
    pub edge: String,
}

fn default_calendar_type() -> String {
    "Date".to_string()
}
fn default_next_day_edge() -> String {
    "NEXT_DAY".to_string()
}

/// Load a blueprint from a file path.
pub fn load_blueprint_file(path: &std::path::Path) -> Result<Blueprint, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("Blueprint file not found: {}: {}", path.display(), e))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("Invalid blueprint JSON: {}", e))
}
