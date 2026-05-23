//! Schema / property / neighbors / sample / join-candidate computation.

use crate::datatypes::values::Value;
use crate::graph::schema::{DirGraph, InternedKey, NodeData};
use crate::graph::storage::GraphRead;
use petgraph::Direction;
use std::collections::{HashMap, HashSet};

use super::capabilities::discover_endpoint_types_batch;
use super::connectivity::derive_edge_counts_from_triples;
use super::{
    ConnectionTypeStats, NeighborConnection, NeighborsSchema, NodeTypeOverview, PropertyStatInfo,
    SchemaOverview,
};

// ── Core functions ──────────────────────────────────────────────────────────

/// Compute per-connection-type stats.
///
/// Fast path: uses connection_type_metadata + cached edge counts (O(types)).
/// Fallback: scans all edges (O(edges)) for pre-metadata graphs.
pub fn compute_connection_type_stats(graph: &DirGraph) -> Vec<ConnectionTypeStats> {
    // Fast path: use metadata (already has source/target types) + cached counts
    if !graph.connection_type_metadata.is_empty() {
        let counts = graph.get_edge_type_counts();
        let mut result: Vec<ConnectionTypeStats> = graph
            .connection_type_metadata
            .iter()
            .map(|(conn_type, info)| {
                let mut source_types: Vec<String> = info.source_types.iter().cloned().collect();
                source_types.sort();
                let mut target_types: Vec<String> = info.target_types.iter().cloned().collect();
                target_types.sort();
                let mut property_names: Vec<String> = info.property_types.keys().cloned().collect();
                property_names.sort();
                ConnectionTypeStats {
                    connection_type: conn_type.clone(),
                    count: counts.get(conn_type).copied().unwrap_or(0),
                    source_types,
                    target_types,
                    property_names,
                }
            })
            .collect();
        result.sort_by(|a, b| a.connection_type.cmp(&b.connection_type));

        // Post-process: resolve empty source/target types.
        // Prefer type connectivity triples (instant) over edge scan.
        let has_empty = result
            .iter()
            .any(|ct| ct.source_types.is_empty() && ct.target_types.is_empty() && ct.count > 0);
        if has_empty {
            let triples_guard = graph.type_connectivity_cache.read().unwrap();
            if let Some(triples) = triples_guard.as_ref() {
                // Derive endpoints from cached triples — zero I/O
                let derived = derive_edge_counts_from_triples(triples);
                for ct in &mut result {
                    if ct.source_types.is_empty() && ct.target_types.is_empty() {
                        if let Some((src, tgt)) = derived.endpoints.get(&ct.connection_type) {
                            let mut src_vec: Vec<String> = src.iter().cloned().collect();
                            src_vec.sort();
                            let mut tgt_vec: Vec<String> = tgt.iter().cloned().collect();
                            tgt_vec.sort();
                            ct.source_types = src_vec;
                            ct.target_types = tgt_vec;
                        }
                    }
                }
            } else {
                // No cached triples — fall back to bounded edge scan
                let discovered = discover_endpoint_types_batch(graph, 1_000_000);
                for ct in &mut result {
                    if ct.source_types.is_empty() && ct.target_types.is_empty() {
                        if let Some((src, tgt)) = discovered.get(&ct.connection_type) {
                            let mut src_vec: Vec<String> = src.iter().cloned().collect();
                            src_vec.sort();
                            let mut tgt_vec: Vec<String> = tgt.iter().cloned().collect();
                            tgt_vec.sort();
                            ct.source_types = src_vec;
                            ct.target_types = tgt_vec;
                        }
                    }
                }
            }
        }

        return result;
    }

    // Fallback: scan all edges (pre-metadata graphs)
    struct Accum {
        count: usize,
        sources: HashSet<String>,
        targets: HashSet<String>,
        props: HashSet<String>,
    }
    let mut stats: HashMap<String, Accum> = HashMap::new();

    let g = &graph.graph;
    for edge_ref in g.edge_references() {
        let edge_data = edge_ref.weight();
        let entry = stats
            .entry(edge_data.connection_type_str(&graph.interner).to_string())
            .or_insert_with(|| Accum {
                count: 0,
                sources: HashSet::new(),
                targets: HashSet::new(),
                props: HashSet::new(),
            });
        entry.count += 1;

        if let Some(source_node) = graph.get_node(edge_ref.source()) {
            entry
                .sources
                .insert(source_node.node_type_str(&graph.interner).to_string());
        }
        if let Some(target_node) = graph.get_node(edge_ref.target()) {
            entry
                .targets
                .insert(target_node.node_type_str(&graph.interner).to_string());
        }
        for key in edge_data.property_keys(&graph.interner) {
            entry.props.insert(key.to_string());
        }
    }

    let mut result: Vec<ConnectionTypeStats> = stats
        .into_iter()
        .map(|(conn_type, acc)| {
            let mut source_types: Vec<String> = acc.sources.into_iter().collect();
            source_types.sort();
            let mut target_types: Vec<String> = acc.targets.into_iter().collect();
            target_types.sort();
            let mut property_names: Vec<String> = acc.props.into_iter().collect();
            property_names.sort();
            ConnectionTypeStats {
                connection_type: conn_type,
                count: acc.count,
                source_types,
                target_types,
                property_names,
            }
        })
        .collect();
    result.sort_by(|a, b| a.connection_type.cmp(&b.connection_type));
    result
}

/// Set of node types that participate in at least one edge (as source or target).
pub(super) fn compute_connected_types(conn_stats: &[ConnectionTypeStats]) -> HashSet<String> {
    let mut connected = HashSet::new();
    for ct in conn_stats {
        for s in &ct.source_types {
            connected.insert(s.clone());
        }
        for t in &ct.target_types {
            connected.insert(t.clone());
        }
    }
    connected
}

/// Set of unordered (TypeA, TypeB) pairs directly connected by at least one edge type.
pub(super) fn compute_connected_type_pairs(
    conn_stats: &[ConnectionTypeStats],
) -> HashSet<(String, String)> {
    let mut pairs = HashSet::new();
    for ct in conn_stats {
        for s in &ct.source_types {
            for t in &ct.target_types {
                // Store both orderings so lookup is direction-independent
                pairs.insert((s.clone(), t.clone()));
                pairs.insert((t.clone(), s.clone()));
            }
        }
    }
    pairs
}

/// A candidate join between two disconnected types based on property value overlap.
pub(super) struct JoinCandidate {
    pub(super) left_type: String,
    pub(super) left_prop: String,
    pub(super) left_unique: usize,
    pub(super) right_type: String,
    pub(super) right_prop: String,
    pub(super) right_unique: usize,
    pub(super) overlap: usize,
}

/// Check whether two property type strings are compatible for join candidate comparison.
/// Metadata types use Rust names: "String", "Int64", "Float64", "UniqueId", etc.
pub(super) fn types_compatible(left: &str, right: &str) -> bool {
    let is_str = |t: &str| {
        t.eq_ignore_ascii_case("string")
            || t.eq_ignore_ascii_case("uniqueid")
            || t.eq_ignore_ascii_case("str")
    };
    let is_num = |t: &str| {
        t.eq_ignore_ascii_case("int64")
            || t.eq_ignore_ascii_case("float64")
            || t.eq_ignore_ascii_case("int")
            || t.eq_ignore_ascii_case("float")
    };
    (is_str(left) && is_str(right)) || (is_num(left) && is_num(right))
}

/// Sample up to `max` unique non-null values from a type's property.
pub(super) fn sample_unique_values(
    graph: &DirGraph,
    node_type: &str,
    property: &str,
    max: usize,
) -> HashSet<String> {
    let mut unique = HashSet::new();
    let Some(indices) = graph.type_indices.get(node_type) else {
        return unique;
    };
    let key = InternedKey::from_str(property);
    let backend = &graph.graph;
    for idx in indices.iter() {
        if unique.len() >= max {
            break;
        }
        if let Some(val) = backend.get_node_property(idx, key) {
            if !is_null_value(&val) {
                let s = match &val {
                    Value::String(s) => s.clone(),
                    Value::Int64(n) => n.to_string(),
                    Value::Float64(f) => f.to_string(),
                    Value::UniqueId(id) => id.to_string(),
                    _ => format!("{:?}", val),
                };
                unique.insert(s);
            }
        }
    }
    unique
}

/// Insert a `(type, prop)` sample into the cache if not already present.
/// Stores `None` for empty results to avoid resampling.
pub(super) fn populate_sample(
    cache: &mut HashMap<(String, String), Option<HashSet<String>>>,
    graph: &DirGraph,
    node_type: &str,
    property: &str,
    max: usize,
) {
    let key = (node_type.to_string(), property.to_string());
    if cache.contains_key(&key) {
        return;
    }
    let vals = sample_unique_values(graph, node_type, property, max);
    cache.insert(key, if vals.is_empty() { None } else { Some(vals) });
}

/// Find join candidates between disconnected core type pairs.
///
/// Performance note: samples each (type, property) at most once by memoising
/// into `sample_cache`. Without this, a property shared across N types gets
/// resampled O(N²) times — which was 6× slower on columnar-backed graphs
/// (where each property read clones through the column store).
pub(super) fn compute_join_candidates(
    graph: &DirGraph,
    connected_pairs: &HashSet<(String, String)>,
    max_candidates: usize,
    max_sample: usize,
) -> Vec<JoinCandidate> {
    // Collect core types (exclude supporting types)
    let mut core_types: Vec<&str> = graph
        .type_indices
        .keys()
        .filter(|nt| !graph.parent_types.contains_key(*nt))
        .collect();
    core_types.sort();

    let mut candidates: Vec<JoinCandidate> = Vec::new();
    // Memoise sampled values per (type, property). `None` means "already sampled
    // and found empty" so we don't resample.
    let mut sample_cache: HashMap<(String, String), Option<HashSet<String>>> = HashMap::new();

    // Check all unordered pairs of disconnected core types
    'outer: for i in 0..core_types.len() {
        if candidates.len() >= max_candidates * 3 {
            break; // Early exit: we have enough raw candidates
        }
        for j in (i + 1)..core_types.len() {
            if candidates.len() >= max_candidates * 3 {
                break 'outer;
            }
            let left = core_types[i];
            let right = core_types[j];

            // Skip already-connected pairs
            if connected_pairs.contains(&(left.to_string(), right.to_string())) {
                continue;
            }

            let left_meta = match graph.node_type_metadata.get(left) {
                Some(m) => m,
                None => continue,
            };
            let right_meta = match graph.node_type_metadata.get(right) {
                Some(m) => m,
                None => continue,
            };

            // Find shared property names with compatible types.
            // Sort by property name for deterministic candidate ordering — HashMap
            // iteration order otherwise depends on RandomState seed and changes
            // describe() output between processes.
            let mut props: Vec<(&String, &String)> = left_meta.iter().collect();
            props.sort_by(|a, b| a.0.cmp(b.0));
            for (prop, left_type) in props {
                let Some(right_type) = right_meta.get(prop) else {
                    continue;
                };
                if !types_compatible(left_type, right_type) {
                    continue;
                }
                // Populate cache for both sides, then read — avoids simultaneous
                // immutable+mutable borrows on `sample_cache`.
                populate_sample(&mut sample_cache, graph, left, prop, max_sample);
                if sample_cache
                    .get(&(left.to_string(), prop.clone()))
                    .is_none_or(|v| v.is_none())
                {
                    continue;
                }
                populate_sample(&mut sample_cache, graph, right, prop, max_sample);
                let left_vals = match sample_cache.get(&(left.to_string(), prop.clone())) {
                    Some(Some(v)) => v,
                    _ => continue,
                };
                let right_vals = match sample_cache.get(&(right.to_string(), prop.clone())) {
                    Some(Some(v)) => v,
                    _ => continue,
                };
                let overlap = left_vals.intersection(right_vals).count();
                if overlap > 0 {
                    candidates.push(JoinCandidate {
                        left_type: left.to_string(),
                        left_prop: prop.clone(),
                        left_unique: left_vals.len(),
                        right_type: right.to_string(),
                        right_prop: prop.clone(),
                        right_unique: right_vals.len(),
                        overlap,
                    });
                }
            }
        }
    }

    // Sort by overlap descending; break ties on (left_type, right_type, left_prop)
    // for deterministic output across processes.
    candidates.sort_by(|a, b| {
        b.overlap
            .cmp(&a.overlap)
            .then_with(|| a.left_type.cmp(&b.left_type))
            .then_with(|| a.right_type.cmp(&b.right_type))
            .then_with(|| a.left_prop.cmp(&b.left_prop))
    });
    candidates.truncate(max_candidates);
    candidates
}

/// All node-type names (Neo4j "labels"), sorted alphabetically.
///
/// Phase A.3 — single source of truth for `db.labels()` and any other
/// caller that needs a deterministic enumeration of node types. Pulls
/// from both `type_indices` (types with live nodes) and
/// `node_type_metadata` (types declared via schema validation but no
/// live nodes yet), matching the existing `get_node_types()` semantics.
pub(crate) fn collect_labels(graph: &DirGraph) -> Vec<String> {
    let mut labels = graph.get_node_types();
    labels.sort();
    labels
}

/// Index kind in KGLite's terminology — surfaces via `db.indexes()`.
///
/// KGLite has three distinct index kinds where Neo4j collapses two of them
/// into `PROPERTY`. We expose the distinction because index advisors and
/// query planners need it: an equality index can't serve a range query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IndexKind {
    /// Hash-based equality lookup (one property per index).
    Equality,
    /// Multi-property equality lookup (conjunctive filters).
    Composite,
    /// B-Tree range lookup (supports comparison operators).
    Range,
}

impl IndexKind {
    /// Neo4j-compatible `type` column value for `db.indexes()`.
    ///
    /// Equality + Composite both map to `"PROPERTY"` (Neo4j convention);
    /// `Range` is a KGLite-specific value documented in CYPHER.md.
    pub(crate) fn neo4j_type(self) -> &'static str {
        match self {
            IndexKind::Equality | IndexKind::Composite => "PROPERTY",
            IndexKind::Range => "RANGE",
        }
    }
}

/// One index entry surfaced by `db.indexes()`.
///
/// Field shape mirrors Neo4j's `db.indexes()` minimal subset:
/// `name, type, entityType, labelsOrTypes, properties, state`. Degenerate
/// columns (`uniqueness`, `populationPercent`, `indexProvider`) are
/// deferred until a Bolt client demands them — see Phase A.3 plan.
#[derive(Debug, Clone)]
pub(crate) struct IndexInfo {
    /// Stable string ID — `"<node_type>.<property>"` for equality/range,
    /// `"<node_type>.(<p1>,<p2>,...)"` for composite.
    pub name: String,
    pub kind: IndexKind,
    /// Always `"NODE"` today; relationship indexes not yet supported.
    pub entity_type: &'static str,
    /// Node types this index covers — always a single-element vec today.
    pub labels_or_types: Vec<String>,
    /// Indexed property names, in definition order. Length ≥ 2 for composite.
    pub properties: Vec<String>,
    /// Always `"ONLINE"` — KGLite indexes are atomic; no POPULATING state.
    pub state: &'static str,
}

/// All indexes installed on the graph, in deterministic order.
///
/// Phase A.3 — single source of truth for `db.indexes()` and the
/// `compute_schema()` formatted string list. Walks all three index
/// stores (`property_indices`, `composite_indices`, `range_indices`)
/// and produces structured rows. Sorted by `name` so the output is
/// stable across runs and storage modes.
pub(crate) fn collect_indexes_structured(graph: &DirGraph) -> Vec<IndexInfo> {
    let mut out: Vec<IndexInfo> = Vec::new();

    for (node_type, property) in graph.property_indices.keys() {
        out.push(IndexInfo {
            name: format!("{node_type}.{property}"),
            kind: IndexKind::Equality,
            entity_type: "NODE",
            labels_or_types: vec![node_type.clone()],
            properties: vec![property.clone()],
            state: "ONLINE",
        });
    }
    for (node_type, properties) in graph.composite_indices.keys() {
        out.push(IndexInfo {
            name: format!("{node_type}.({})", properties.join(",")),
            kind: IndexKind::Composite,
            entity_type: "NODE",
            labels_or_types: vec![node_type.clone()],
            properties: properties.clone(),
            state: "ONLINE",
        });
    }
    for (node_type, property) in graph.range_indices.keys() {
        out.push(IndexInfo {
            name: format!("{node_type}.{property}"),
            kind: IndexKind::Range,
            entity_type: "NODE",
            labels_or_types: vec![node_type.clone()],
            properties: vec![property.clone()],
            state: "ONLINE",
        });
    }

    out.sort_by(|a, b| {
        a.name
            .cmp(&b.name)
            .then_with(|| (a.kind as u8).cmp(&(b.kind as u8)))
    });
    out
}

/// All connection-type names (Neo4j "relationship types"), sorted alphabetically.
///
/// Phase A.3 — single source of truth for `db.relationshipTypes()`. Unions
/// two sources to match Neo4j semantics ("types that currently exist in
/// the graph"):
///   1. `connection_type_metadata` — types declared via `add_connections`
///      (always populated for these; carries source/target/property schemas).
///   2. `get_edge_type_counts` — live edge scan (populated for types added
///      via raw `CREATE ()-[:T]->()` cypher, which doesn't upsert metadata
///      for fresh types).
pub(crate) fn collect_relationship_types(graph: &DirGraph) -> Vec<String> {
    let mut types: HashSet<String> = graph.connection_type_metadata.keys().cloned().collect();
    types.extend(graph.get_edge_type_counts().into_keys());
    let mut out: Vec<String> = types.into_iter().collect();
    out.sort();
    out
}

/// Full schema overview: node types, connection types, indexes, totals.
pub fn compute_schema(graph: &DirGraph) -> SchemaOverview {
    // Node types from type_indices
    let mut node_types: Vec<(String, NodeTypeOverview)> = graph
        .type_indices
        .iter()
        .map(|(nt, indices)| {
            let properties = graph
                .node_type_metadata
                .get(nt)
                .cloned()
                .unwrap_or_default();
            (
                nt.to_string(),
                NodeTypeOverview {
                    count: indices.len(),
                    properties,
                },
            )
        })
        .collect();
    node_types.sort_by(|a, b| a.0.cmp(&b.0));

    // Connection types via edge scan
    let connection_types = compute_connection_type_stats(graph);

    // Indexes — formatted from the structured helper that also feeds
    // `db.indexes()`. String shape preserved to keep schema() Python API
    // tests green:
    //   - Equality: "Type.prop"
    //   - Composite: "Type.(p1, p2)"
    //   - Range: "Type.prop [range]"
    let mut indexes: Vec<String> = collect_indexes_structured(graph)
        .into_iter()
        .map(|idx| match idx.kind {
            IndexKind::Equality => format!("{}.{}", idx.labels_or_types[0], idx.properties[0]),
            IndexKind::Composite => {
                format!("{}.({})", idx.labels_or_types[0], idx.properties.join(", "))
            }
            IndexKind::Range => format!("{}.{} [range]", idx.labels_or_types[0], idx.properties[0]),
        })
        .collect();
    indexes.sort();

    SchemaOverview {
        node_types,
        connection_types,
        indexes,
        node_count: graph.graph.node_count(),
        edge_count: graph.graph.edge_count(),
    }
}

pub(super) fn is_null_value(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::Float64(f) => f.is_nan(),
        _ => false,
    }
}

pub(super) fn value_type_name(v: &Value) -> &'static str {
    match v {
        Value::String(_) => "str",
        Value::Int64(_) => "int",
        Value::Float64(_) => "float",
        Value::Boolean(_) => "bool",
        Value::DateTime(_) => "datetime",
        Value::UniqueId(_) => "uniqueid",
        Value::Point { .. } => "point",
        Value::Duration { .. } => "duration",
        Value::Null => "unknown",
        Value::NodeRef(_) => "noderef",
        // Phase A.1 — these typically appear in query results, not as
        // stored properties, but classify defensively for introspection.
        Value::List(_) => "list",
        Value::Map(_) => "map",
        Value::Node(_) => "node",
        Value::Relationship(_) => "relationship",
        Value::Path(_) => "path",
    }
}

/// Compact display string for a Value (used in agent description `vals` attributes).
///
/// `truncate_at = Some(n)` truncates string values longer than `n` chars to
/// `n - 3` chars + `"..."`. `None` (or `Some(0)`) emits the value unchanged —
/// the escape hatch for callers who pass `describe(sample_truncate=None)`.
pub(super) fn value_display_compact(v: &Value, truncate_at: Option<usize>) -> String {
    match v {
        Value::String(s) => match truncate_at {
            Some(n) if n >= 4 && s.chars().count() > n => {
                let truncated: String = s.chars().take(n - 3).collect();
                format!("{}...", truncated)
            }
            _ => s.clone(),
        },
        Value::Int64(i) => i.to_string(),
        Value::Float64(f) => format!("{}", f),
        Value::Boolean(b) => {
            if *b {
                "true"
            } else {
                "false"
            }
        }
        .to_string(),
        Value::DateTime(d) => d.to_string(),
        Value::UniqueId(u) => u.to_string(),
        Value::Point { lat, lon } => format!("({},{})", lat, lon),
        Value::Duration {
            months,
            days,
            seconds,
        } => format!("dur(M={},D={},S={})", months, days, seconds),
        Value::NodeRef(idx) => format!("node#{}", idx),
        Value::Null => String::new(),
        // Phase A.1 — collection / graph-entity variants delegate to
        // format_value; truncation applies only to the String variant.
        Value::List(_)
        | Value::Map(_)
        | Value::Node(_)
        | Value::Relationship(_)
        | Value::Path(_) => crate::datatypes::values::format_value(v),
    }
}

/// Property stats for one node type.
/// `max_values`: include `values` list when unique count ≤ this threshold (0 = never).
/// `sample_size`: when Some(n), sample n evenly-spaced nodes instead of scanning all.
///   Sampled non_null counts are scaled to the full population.
pub fn compute_property_stats(
    graph: &DirGraph,
    node_type: &str,
    max_values: usize,
    sample_size: Option<usize>,
) -> Result<Vec<PropertyStatInfo>, String> {
    let node_indices = graph
        .type_indices
        .get(node_type)
        .ok_or_else(|| format!("Node type '{}' not found", node_type))?;

    let total_nodes = node_indices.len();

    // Per-property accumulator
    // Cap value_set at max_values+1 to avoid cloning every value when there are
    // thousands of unique values. We only need the set for small-cardinality props.
    // Cap at max_values+1: we need one extra to detect "too many unique values".
    // When capped, unique count is a lower bound (max_values+1) and values = None.
    let value_cap = if max_values > 0 {
        max_values + 1
    } else {
        usize::MAX // still need unique counts even when not reporting values
    };

    struct PropAccum {
        non_null: usize,
        value_set: HashSet<Value>,
        value_cap: usize,
        first_type: Option<&'static str>,
    }
    impl PropAccum {
        fn new(cap: usize) -> Self {
            Self {
                non_null: 0,
                value_set: HashSet::new(),
                value_cap: cap,
                first_type: None,
            }
        }
        fn add(&mut self, v: &Value) {
            if !is_null_value(v) {
                self.non_null += 1;
                if self.value_set.len() < self.value_cap {
                    self.value_set.insert(v.clone());
                }
                if self.first_type.is_none() {
                    self.first_type = Some(value_type_name(v));
                }
            }
        }
    }

    // Determine which nodes to scan (all or sampled)
    let (scan_indices, sample_count): (Vec<petgraph::graph::NodeIndex>, usize) = match sample_size {
        Some(n) if n > 0 && n < total_nodes => {
            let step = total_nodes / n;
            let sampled: Vec<_> = (0..n).filter_map(|i| node_indices.get(i * step)).collect();
            let count = sampled.len();
            (sampled, count)
        }
        _ => {
            // No sampling — scan all nodes
            (node_indices.to_vec(), total_nodes)
        }
    };

    // Single pass: accumulate stats for all properties simultaneously
    let mut accum: HashMap<String, PropAccum> = HashMap::new();
    // Pre-insert built-in fields so they appear even when all null
    accum.insert("title".to_string(), PropAccum::new(value_cap));
    accum.insert("id".to_string(), PropAccum::new(value_cap));

    // When sampling, pre-populate property keys from TypeSchema (knows ALL keys)
    if sample_size.is_some() {
        if let Some(schema) = graph.type_schemas.get(node_type) {
            for slot_key in schema.iter() {
                if let Some(key_str) = graph.interner.try_resolve(slot_key.1) {
                    accum
                        .entry(key_str.to_string())
                        .or_insert_with(|| PropAccum::new(value_cap));
                }
            }
        }
    }

    for &idx in &scan_indices {
        if let Some(node) = graph.get_node(idx) {
            accum
                .entry("id".to_string())
                .or_insert_with(|| PropAccum::new(value_cap))
                .add(&node.id());
            accum
                .entry("title".to_string())
                .or_insert_with(|| PropAccum::new(value_cap))
                .add(&node.title());
            for (key, value) in node.property_iter(&graph.interner) {
                accum
                    .entry(key.to_string())
                    .or_insert_with(|| PropAccum::new(value_cap))
                    .add(value);
            }
        }
    }

    // When sampling, scale non_null counts to the full population
    let scale_factor = if sample_count < total_nodes && sample_count > 0 {
        total_nodes as f64 / sample_count as f64
    } else {
        1.0
    };

    // Build ordered property list: type, title, id, then remaining sorted
    let mut results = Vec::new();

    // "type" is always synthetic
    results.push(PropertyStatInfo {
        property_name: "type".to_string(),
        type_string: "str".to_string(),
        non_null: total_nodes,
        unique: 1,
        values: Some(vec![Value::String(node_type.to_string())]),
        sample: None,
    });

    // Canonical order for remaining: title, id first, then sorted discovered
    let builtins = ["title", "id"];
    let mut discovered: Vec<String> = accum
        .keys()
        .filter(|k| !builtins.contains(&k.as_str()))
        .cloned()
        .collect();
    discovered.sort();

    let ordered: Vec<String> = builtins
        .iter()
        .map(|s| s.to_string())
        .chain(discovered)
        .collect();

    let metadata = graph.node_type_metadata.get(node_type);

    for prop_name in &ordered {
        if let Some(pa) = accum.remove(prop_name) {
            let type_string = metadata
                .and_then(|meta| meta.get(prop_name))
                .cloned()
                .unwrap_or_else(|| pa.first_type.unwrap_or("unknown").to_string());

            let unique = pa.value_set.len();
            let non_null = (pa.non_null as f64 * scale_factor).round() as usize;
            let (values, sample) = if max_values > 0 && unique <= max_values && unique > 0 {
                let mut vals: Vec<Value> = pa.value_set.into_iter().collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                (Some(vals), None)
            } else if unique > 0 {
                // 0.9.30: too many distinct values to enumerate, but
                // pick one as a sample so the schema XML can still
                // show what the property *looks like*. Closes the
                // operator-reported friction where high-cardinality
                // properties (file_path with hundreds of values,
                // docstring with thousands) showed only `unique=N`
                // and forced the agent to guess value shape from the
                // property name. HashSet iteration order isn't
                // deterministic, but for a sample value this is
                // acceptable — the contract is "one real value",
                // not "the same value every time."
                let sample = pa.value_set.into_iter().next();
                (None, sample)
            } else {
                (None, None)
            };

            results.push(PropertyStatInfo {
                property_name: prop_name.clone(),
                type_string,
                non_null,
                unique,
                values,
                sample,
            });
        }
    }

    Ok(results)
}

/// Connection topology for one node type: outgoing and incoming grouped by (conn_type, other_type).
pub fn compute_neighbors_schema(
    graph: &DirGraph,
    node_type: &str,
) -> Result<NeighborsSchema, String> {
    let node_indices = graph
        .type_indices
        .get(node_type)
        .ok_or_else(|| format!("Node type '{}' not found", node_type))?;

    let mut outgoing: HashMap<(String, String), usize> = HashMap::new();
    let mut incoming: HashMap<(String, String), usize> = HashMap::new();

    let g = &graph.graph;
    for node_idx in node_indices.iter() {
        for edge_ref in g.edges_directed(node_idx, Direction::Outgoing) {
            if let Some(target_node) = graph.get_node(edge_ref.target()) {
                let key = (
                    edge_ref
                        .weight()
                        .connection_type_str(&graph.interner)
                        .to_string(),
                    target_node.node_type_str(&graph.interner).to_string(),
                );
                *outgoing.entry(key).or_insert(0) += 1;
            }
        }
        for edge_ref in g.edges_directed(node_idx, Direction::Incoming) {
            if let Some(source_node) = graph.get_node(edge_ref.source()) {
                let key = (
                    edge_ref
                        .weight()
                        .connection_type_str(&graph.interner)
                        .to_string(),
                    source_node.node_type_str(&graph.interner).to_string(),
                );
                *incoming.entry(key).or_insert(0) += 1;
            }
        }
    }

    let mut outgoing_list: Vec<NeighborConnection> = outgoing
        .into_iter()
        .map(|((ct, ot), count)| NeighborConnection {
            connection_type: ct,
            other_type: ot,
            count,
        })
        .collect();
    outgoing_list.sort_by(|a, b| {
        (&a.connection_type, &a.other_type).cmp(&(&b.connection_type, &b.other_type))
    });

    let mut incoming_list: Vec<NeighborConnection> = incoming
        .into_iter()
        .map(|((ct, ot), count)| NeighborConnection {
            connection_type: ct,
            other_type: ot,
            count,
        })
        .collect();
    incoming_list.sort_by(|a, b| {
        (&a.connection_type, &a.other_type).cmp(&(&b.connection_type, &b.other_type))
    });

    Ok(NeighborsSchema {
        outgoing: outgoing_list,
        incoming: incoming_list,
    })
}

/// Pre-compute neighbor schemas for ALL types in a single pass over edges.
/// Much faster than calling `compute_neighbors_schema` per type in `describe()`.
pub fn compute_all_neighbors_schemas(graph: &DirGraph) -> HashMap<String, NeighborsSchema> {
    // Key: (source_type, conn_type, target_type) → count
    let mut edge_counts: HashMap<(String, String, String), usize> = HashMap::new();

    let g = &graph.graph;
    for edge_ref in g.edge_references() {
        if let (Some(source), Some(target)) = (
            graph.get_node(edge_ref.source()),
            graph.get_node(edge_ref.target()),
        ) {
            let conn_type = edge_ref
                .weight()
                .connection_type_str(&graph.interner)
                .to_string();
            let key = (
                source.node_type_str(&graph.interner).to_string(),
                conn_type,
                target.node_type_str(&graph.interner).to_string(),
            );
            *edge_counts.entry(key).or_insert(0) += 1;
        }
    }

    let mut result: HashMap<String, NeighborsSchema> = HashMap::new();
    for ((src_type, conn_type, tgt_type), count) in &edge_counts {
        // Outgoing for src_type
        let schema = result
            .entry(src_type.clone())
            .or_insert_with(|| NeighborsSchema {
                outgoing: Vec::new(),
                incoming: Vec::new(),
            });
        schema.outgoing.push(NeighborConnection {
            connection_type: conn_type.clone(),
            other_type: tgt_type.clone(),
            count: *count,
        });

        // Incoming for tgt_type
        let schema = result
            .entry(tgt_type.clone())
            .or_insert_with(|| NeighborsSchema {
                outgoing: Vec::new(),
                incoming: Vec::new(),
            });
        schema.incoming.push(NeighborConnection {
            connection_type: conn_type.clone(),
            other_type: src_type.clone(),
            count: *count,
        });
    }

    // Sort each type's lists for deterministic output
    for schema in result.values_mut() {
        schema.outgoing.sort_by(|a, b| {
            (&a.connection_type, &a.other_type).cmp(&(&b.connection_type, &b.other_type))
        });
        schema.incoming.sort_by(|a, b| {
            (&a.connection_type, &a.other_type).cmp(&(&b.connection_type, &b.other_type))
        });
    }

    result
}

/// Return first N nodes of a type for quick inspection.
pub fn compute_sample<'a>(
    graph: &'a DirGraph,
    node_type: &str,
    n: usize,
) -> Result<Vec<&'a NodeData>, String> {
    let node_indices = graph
        .type_indices
        .get(node_type)
        .ok_or_else(|| format!("Node type '{}' not found", node_type))?;

    let mut result = Vec::with_capacity(n.min(node_indices.len()));
    for idx in node_indices.iter().take(n) {
        if let Some(node) = graph.get_node(idx) {
            result.push(node);
        }
    }
    Ok(result)
}
