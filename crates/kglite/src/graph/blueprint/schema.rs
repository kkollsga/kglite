//! Serde types for the blueprint JSON schema.
//!
//! See docs/python/guides/blueprints.md for the user-facing spec. These structs
//! are lenient: missing fields default to empty where sensible, matching the
//! behaviour of the old Python loader, and an unrecognised field never fails
//! the parse — blueprints in the wild carry stray keys and must keep building.
//!
//! Leniency is not silence, though. Each spec that a user hand-writes captures
//! its unrecognised keys in an `extra` map, and
//! [`super::validation::unknown_key_warnings`] turns them into build-report
//! warnings with a near-miss hint. A dropped `"lables"` otherwise costs every
//! label it carried and reports success. The `ACCEPTED_*_KEYS` lists below
//! feed only that hint — `extra` already knows the key is unrecognised — and
//! `accepted_key_lists_match_the_structs` keeps them in step with the fields.

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
    /// Path to an ontology declaration document (JSON), resolved relative
    /// to the blueprint file (config sits with config; CSVs resolve against
    /// `input_root`). Installed and audited as a final build phase: warn-
    /// level violations land in the build report, error-level violations
    /// fail the build after the full report — no output file is written.
    #[serde(default)]
    pub ontology: Option<String>,
    /// Keys at the top level of the blueprint that this struct does not read.
    #[serde(flatten)]
    pub extra: IndexMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Default)]
pub struct Settings {
    #[serde(default, alias = "root")]
    pub input_root: Option<String>,
    #[serde(default)]
    pub output_path: Option<String>,
    #[serde(default, alias = "output")]
    pub output_file: Option<String>,
    /// Drop unpromoted provisional stub nodes (edges to a node that no
    /// CSV provided) at the end of the build. Default `false` — stubs
    /// are kept so no edge is lost; opt in to discard dangling refs.
    #[serde(default)]
    pub auto_purge: bool,
    /// Keys under `settings` that this struct does not read.
    #[serde(flatten)]
    pub extra: IndexMap<String, serde_json::Value>,
}

/// Keys the blueprint's top level reads. Hint source only — see the module
/// header.
pub const ACCEPTED_BLUEPRINT_KEYS: &[&str] = &["settings", "nodes", "compute", "ontology"];

/// Keys `settings` reads, including the `root` / `output` aliases.
pub const ACCEPTED_SETTINGS_KEYS: &[&str] = &[
    "input_root",
    "root",
    "output_path",
    "output_file",
    "output",
    "auto_purge",
];

/// Keys a node spec (and a `sub_nodes` entry) reads.
pub const ACCEPTED_NODE_KEYS: &[&str] = &[
    "csv",
    "pk",
    "title",
    "parent",
    "parent_fk",
    "properties",
    "labels",
    "skipped",
    "filter",
    "connections",
    "sub_nodes",
    "timeseries",
];

/// Keys an `fk_edges` entry reads.
pub const ACCEPTED_FK_EDGE_KEYS: &[&str] =
    &["target", "fk", "properties", "property_types", "rename"];

/// Keys a `junction_edges` entry reads.
pub const ACCEPTED_JUNCTION_EDGE_KEYS: &[&str] = &[
    "csv",
    "source_fk",
    "target",
    "target_type_column",
    "target_fk",
    "properties",
    "property_types",
    "rename",
];

/// `"Disease"` or `["Disease", "Phenotype"]` — both land as a list, so the
/// loader has one shape to read.
fn string_or_string_list<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(String),
        Many(Vec<String>),
    }
    Ok(match OneOrMany::deserialize(deserializer)? {
        OneOrMany::One(s) => vec![s],
        OneOrMany::Many(v) => v,
    })
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

#[derive(Debug, Deserialize, Default, Clone)]
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
    /// Secondary labels stamped on every node of this type. The type name is
    /// the primary label and is never restamped, so listing it here is a
    /// no-op rather than a duplicate.
    #[serde(default)]
    pub labels: Vec<String>,
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
    /// Keys on this node spec that this struct does not read.
    #[serde(flatten)]
    pub extra: IndexMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct Connections {
    #[serde(default)]
    pub fk_edges: IndexMap<String, FkEdge>,
    #[serde(default)]
    pub junction_edges: IndexMap<String, JunctionEdge>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct FkEdge {
    pub target: String,
    pub fk: String,
    /// Columns of the *source* node's CSV to attach to the edge, taken from
    /// the same row the FK value came from. Listing a column here does not
    /// keep it off the node — `skipped` is what does that.
    #[serde(default)]
    pub properties: Vec<String>,
    #[serde(default)]
    pub property_types: IndexMap<String, String>,
    /// CSV column → edge property name, with the same rules as
    /// [`JunctionEdge::rename`].
    #[serde(default)]
    pub rename: IndexMap<String, String>,
    /// Keys on this fk_edge that this struct does not read.
    #[serde(flatten)]
    pub extra: IndexMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct JunctionEdge {
    pub csv: String,
    pub source_fk: String,
    /// The node type(s) this relationship points at. A JSON string is the
    /// one-type form; a list is the union form, for a relationship whose
    /// range is an abstract class — without it such a relation needs one
    /// relationship name per concrete type, which no query and no ontology
    /// `range` declaration can put back together.
    #[serde(deserialize_with = "string_or_string_list")]
    pub target: Vec<String>,
    pub target_fk: String,
    /// CSV column naming each row's target type, for the union form. Its
    /// values must be among `target`; a row naming anything else is skipped
    /// with a build warning rather than routed by guess. Without it, a union
    /// `target` is resolved by probing the declared types for the row's
    /// target id. Routing only — the column becomes an edge property just as
    /// any other does, by being listed in `properties`.
    #[serde(default)]
    pub target_type_column: Option<String>,
    #[serde(default)]
    pub properties: Vec<String>,
    #[serde(default)]
    pub property_types: IndexMap<String, String>,
    /// CSV column → edge property name. Keys must be listed in `properties`
    /// and refer to CSV columns (`property_types` stays keyed by the CSV
    /// name); fk columns are not renamable. This is the rename facility
    /// `property_types` was never — see `validation::
    /// unknown_property_type_warnings`.
    #[serde(default)]
    pub rename: IndexMap<String, String>,
    /// Keys on this junction_edge that this struct does not read.
    #[serde(flatten)]
    pub extra: IndexMap<String, serde_json::Value>,
}

impl FkEdge {
    /// A `target` + `fk` edge with nothing else declared — the shape the
    /// loader synthesises for a node spec's implicit `parent` edge.
    pub fn plain(target: String, fk: String) -> Self {
        FkEdge {
            target,
            fk,
            properties: vec![],
            property_types: IndexMap::new(),
            rename: IndexMap::new(),
            extra: IndexMap::new(),
        }
    }
}

impl JunctionEdge {
    /// Property-less edge over a compute-pipeline output CSV.
    pub fn computed(csv: String, source_fk: String, target: String, target_fk: String) -> Self {
        JunctionEdge {
            csv,
            source_fk,
            target: vec![target],
            target_fk,
            target_type_column: None,
            properties: vec![],
            property_types: IndexMap::new(),
            rename: IndexMap::new(),
            extra: IndexMap::new(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum TimeKey {
    Single(String),
    Composite(IndexMap<String, String>),
}

#[derive(Debug, Deserialize, Clone)]
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
