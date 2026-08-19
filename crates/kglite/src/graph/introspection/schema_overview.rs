//! Schema / property / neighbors / sample / join-candidate computation.

use crate::datatypes::values::Value;
use crate::graph::constraints::{ConstraintKind, EntityKind};
use crate::graph::property_types::DeclaredType;
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::column_store::TypedColumn;
use crate::graph::storage::GraphRead;
use petgraph::Direction;
use rustc_hash::FxHashMap;
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
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
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
                let mut property_names: Vec<String> = info
                    .property_types
                    .keys()
                    .filter(|k| !crate::graph::schema::is_reserved_provenance_key(k))
                    .cloned()
                    .collect();
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

        if let Some(source_node) = graph.node_view(edge_ref.source()) {
            entry
                .sources
                .insert(source_node.node_type_str(&graph.interner).to_string());
        }
        if let Some(target_node) = graph.node_view(edge_ref.target()) {
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
            let mut property_names: Vec<String> = acc
                .props
                .into_iter()
                .filter(|k| !crate::graph::schema::is_reserved_provenance_key(k))
                .collect();
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

/// One constraint entry surfaced by `SHOW CONSTRAINTS` / `CALL db.constraints()`.
///
/// Field shape mirrors Neo4j's `SHOW CONSTRAINTS` subset KGLite can answer:
/// `name, type, entityType, labelsOrTypes, properties, propertyType`. Neo4j also
/// returns `id` and `ownedIndex`; KGLite has no equivalent state for either (a
/// unique constraint *is* its index rather than owning a separate one), so they
/// are omitted rather than filled with invented values. Documented in CYPHER.md.
#[derive(Debug, Clone)]
pub(crate) struct ConstraintInfo {
    /// The name its author gave it, or the canonical `Label.property` /
    /// `Label.(a, b)` descriptor when it was declared without one.
    pub name: String,
    pub kind: ConstraintKind,
    /// Whether the row describes a node constraint or a relationship one.
    /// Every row the collector below emits is a node row today — the
    /// relationship stores do not exist yet — but the `type` and `entityType`
    /// columns are rendered from this rather than from a hard-coded `"NODE"`,
    /// so the relationship half is a matter of building the row, not of
    /// rewriting the vocabulary.
    pub entity: EntityKind,
    pub labels_or_types: Vec<String>,
    /// Constrained property names, in declaration order.
    pub properties: Vec<String>,
    /// The declared type, for a `NODE_PROPERTY_TYPE` row; `None` for every
    /// other kind — which is exactly what Neo4j 5 puts in the `propertyType`
    /// column, so a ported script's result handling reads unchanged.
    pub property_type: Option<DeclaredType>,
}

impl ConstraintInfo {
    /// Neo4j-compatible `type` column value, using Neo4j 5's `ConstraintType`
    /// spellings so a ported script's result handling reads unchanged.
    pub(crate) fn neo4j_type(&self) -> &'static str {
        match (self.entity, self.kind) {
            (EntityKind::Node, ConstraintKind::Unique) => "UNIQUENESS",
            (EntityKind::Node, ConstraintKind::NodeKey) => "NODE_KEY",
            (EntityKind::Node, ConstraintKind::NotNull) => "NODE_PROPERTY_EXISTENCE",
            (EntityKind::Node, ConstraintKind::PropertyType) => "NODE_PROPERTY_TYPE",
            // Neo4j 5's relationship spellings. `RELATIONSHIP_UNIQUENESS`
            // breaks the node side's `UNIQUENESS`/`NODE_KEY` asymmetry — it is
            // Neo4j's naming, not a transcription slip.
            (EntityKind::Relationship, ConstraintKind::Unique) => "RELATIONSHIP_UNIQUENESS",
            (EntityKind::Relationship, ConstraintKind::NodeKey) => "RELATIONSHIP_KEY",
            (EntityKind::Relationship, ConstraintKind::NotNull) => {
                "RELATIONSHIP_PROPERTY_EXISTENCE"
            }
            (EntityKind::Relationship, ConstraintKind::PropertyType) => {
                "RELATIONSHIP_PROPERTY_TYPE"
            }
        }
    }

    /// Neo4j-compatible `entityType` column value.
    pub(crate) fn entity_type(&self) -> &'static str {
        self.entity.keyword()
    }
}

/// Every declared constraint on the graph, in deterministic order.
///
/// Single source of truth for `SHOW CONSTRAINTS` and `CALL db.constraints()`,
/// the same one-collector/two-surfaces arrangement
/// [`collect_indexes_structured`] gives the index listings.
///
/// A unique constraint reports as `NODE_KEY` when every property in its tuple is
/// also required — that is what a node key is — and the presence half is then
/// *not* listed again as a separate `NODE_PROPERTY_EXISTENCE` row, matching
/// Neo4j, where a node key is one constraint rather than two.
pub(crate) fn collect_constraints_structured(graph: &DirGraph) -> Vec<ConstraintInfo> {
    let mut out: Vec<ConstraintInfo> = Vec::new();
    // Properties already accounted for by a NODE_KEY row, so the presence pass
    // below does not report them twice.
    let mut covered: HashSet<(String, String)> = HashSet::new();

    for (node_type, properties) in graph.list_unique_constraints() {
        let kind = graph.unique_kind_for(&node_type, &properties);
        if kind == ConstraintKind::NodeKey {
            for property in &properties {
                covered.insert((node_type.clone(), property.clone()));
            }
        }
        out.push(ConstraintInfo {
            name: constraint_name(graph, &node_type, &properties),
            kind,
            entity: EntityKind::Node,
            labels_or_types: vec![node_type.clone()],
            properties,
            property_type: None,
        });
    }

    for (node_type, property) in graph.list_not_null_constraints() {
        if covered.contains(&(node_type.clone(), property.clone())) {
            continue;
        }
        let properties = vec![property];
        out.push(ConstraintInfo {
            name: constraint_name(graph, &node_type, &properties),
            kind: ConstraintKind::NotNull,
            entity: EntityKind::Node,
            labels_or_types: vec![node_type.clone()],
            properties,
            property_type: None,
        });
    }

    // A declared property type is its own constraint, never folded into the
    // rows above: a property may be UNIQUE, NOT NULL *and* typed, and Neo4j
    // reports each as a separate row. Listing them is also what makes an
    // unnamed one droppable — `DROP CONSTRAINT` resolves a canonical descriptor
    // through this collector.
    for (node_type, property, declared) in graph.list_property_type_constraints() {
        let properties = vec![property];
        out.push(ConstraintInfo {
            name: constraint_name(graph, &node_type, &properties),
            kind: ConstraintKind::PropertyType,
            entity: EntityKind::Node,
            labels_or_types: vec![node_type.clone()],
            properties,
            property_type: Some(declared),
        });
    }

    out.sort_by(|a, b| {
        a.name
            .cmp(&b.name)
            .then_with(|| a.neo4j_type().cmp(b.neo4j_type()))
    });
    out
}

/// The author's name for a constraint when one was registered, else the
/// canonical descriptor — the same fallback rule index names use.
fn constraint_name(graph: &DirGraph, node_type: &str, properties: &[String]) -> String {
    graph
        .name_for_constraint(node_type, properties)
        .map(str::to_string)
        .unwrap_or_else(|| crate::graph::constraints::descriptor(node_type, properties))
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
    types.extend(graph.get_edge_type_counts().keys().cloned());
    let mut out: Vec<String> = types.into_iter().collect();
    out.sort();
    out
}

/// All property keys declared anywhere in the graph (node + relationship
/// property names), sorted and de-duplicated.
///
/// Single source of truth for `db.propertyKeys()` (Neo4j-compatible). Unions
/// `node_type_metadata` (per-type `prop → declared_type`) with each
/// `connection_type_metadata` entry's `property_types`. Mirrors how
/// `collect_labels`/`collect_relationship_types` feed their `db.*` procedures.
pub(crate) fn collect_property_keys(graph: &DirGraph) -> Vec<String> {
    let mut keys: HashSet<String> = HashSet::new();
    for props in graph.node_type_metadata.values() {
        keys.extend(props.keys().cloned());
    }
    for info in graph.connection_type_metadata.values() {
        keys.extend(info.property_types.keys().cloned());
    }
    let mut out: Vec<String> = keys.into_iter().collect();
    out.sort();
    out
}

/// Full schema overview: node types, connection types, indexes, totals.
pub fn compute_schema(graph: &DirGraph) -> SchemaOverview {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
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
        Value::Timestamp(_) => "timestamp",
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
        Value::Timestamp(d) => d.to_string(),
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
/// Per-property accumulator for [`compute_property_stats`].
///
/// `value_set` is capped at `value_cap` so a high-cardinality property does not
/// clone every value; when capped, `unique` is a lower bound and `values` is
/// reported as `None`.
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

    /// Whether this accumulator still needs the *value* of a row, or only
    /// whether the row holds one.
    ///
    /// Once the distinct set is capped and the first non-null value has named
    /// the property's type, the only thing left to learn from a row is that it
    /// is non-null — which every column shape can answer from its null byte,
    /// without building the `Value` the row loop had to build. On a
    /// high-cardinality string column that is the difference between one heap
    /// allocation per row and none.
    #[inline]
    fn needs_value(&self) -> bool {
        self.value_set.len() < self.value_cap || self.first_type.is_none()
    }

    /// Fold one whole column into this accumulator, in `rows` order.
    ///
    /// The column-major half of the scan: `rows` is visited once per property
    /// instead of every property being visited once per row, so the per-`(row,
    /// property)` map probe of the row loop is hoisted to one probe per
    /// property and the column's dispatch is resolved once for the whole walk.
    /// Visit order within a property is `rows` order — the same order the row
    /// loop saw them in — so `first_type` (the only order-sensitive field here)
    /// is unchanged.
    ///
    /// The arms differ only in how cheaply they can answer "non-null":
    /// `Mixed` lends its `Value`, `Float64` must be *read* because a NaN counts
    /// as null here ([`is_null_value`]) while its null byte says otherwise, and
    /// every other shape can answer from the null byte alone once the
    /// accumulator stops needing values.
    ///
    /// Returns whether any row yielded a value — which is exactly whether the
    /// row loop's `row_properties` would have emitted this key for any node in
    /// the scan, and therefore whether the property belongs in the output at
    /// all. It is **not** `non_null > 0`: an all-NaN column yields values that
    /// count as null, and the row loop reported it.
    fn add_column(&mut self, col: &TypedColumn, rows: &[u32]) -> bool {
        let mut yielded = false;
        match col {
            // A `Mixed` column already holds `Value`s: borrow rather than
            // clone, which matters because this is where a list property lands
            // and `get` would copy the whole list per row.
            TypedColumn::Mixed { .. } => {
                for &row in rows {
                    if let Some(value) = col.get_ref(row) {
                        yielded = true;
                        self.add(value);
                    }
                }
            }
            // NaN is null to `is_null_value` but not to the null byte, so this
            // column can never take the presence-only shortcut. Reading it
            // allocates nothing.
            TypedColumn::Float64 { .. } => {
                for &row in rows {
                    if let Some(value) = col.get(row) {
                        yielded = true;
                        self.add(&value);
                    }
                }
            }
            _ => {
                for &row in rows {
                    if self.needs_value() {
                        if let Some(value) = col.get(row) {
                            yielded = true;
                            self.add(&value);
                        }
                    } else if col.is_present(row) {
                        yielded = true;
                        self.non_null += 1;
                    }
                }
            }
        }
        yielded
    }
}

/// Accumulator keys for the two synthetic built-in fields.
///
/// `InternedKey` is the FNV hash of the name, so a property *literally* named
/// `id`/`title` hashes to the same key and folds into the same accumulator —
/// exactly what the previous `HashMap<String, _>` keying did (both wrote to the
/// `"id"` entry).
fn builtin_accum_keys() -> (InternedKey, InternedKey) {
    (InternedKey::from_str("id"), InternedKey::from_str("title"))
}

// Test hook: force `accumulate_property_values` down its row-major route.
//
// The column-major path has to produce byte-identical stats, and the only way
// to assert that is to run the *same* fixture both ways in one process. This
// is the decline switch the equivalence test flips — not a configuration knob;
// it does not exist outside `cfg(test)`.
#[cfg(test)]
thread_local! {
    static FORCE_ROW_MAJOR_STATS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f` with the column-major stats path declined.
#[cfg(test)]
pub(crate) fn with_row_major_stats<R>(f: impl FnOnce() -> R) -> R {
    FORCE_ROW_MAJOR_STATS.set(true);
    let out = f();
    FORCE_ROW_MAJOR_STATS.set(false);
    out
}

/// The rows `scan_indices` occupy in their type's column store, or `None` when
/// the store cannot answer the scan column-major.
///
/// Three things disqualify a store, all of them because the *row* read resolves
/// through more than its dense columns and a column walk would silently drop
/// what the extra sources contribute (`ColumnStore::row_properties`): an mmap
/// base (mapped mode), an overflow bag (built by the disk loader), and any node
/// in the scan whose properties are still inline rather than columnar. A
/// tombstoned row yields no properties at all through the row route, so it is
/// dropped here rather than declining the whole scan.
fn columnar_scan_rows(
    graph: &DirGraph,
    store: &crate::graph::storage::ColumnStore,
    scan_indices: &[petgraph::graph::NodeIndex],
) -> Option<Vec<u32>> {
    #[cfg(test)]
    if FORCE_ROW_MAJOR_STATS.get() {
        return None;
    }
    if store.has_mmap_base() || store.has_overflow() {
        return None;
    }
    let mut rows = Vec::with_capacity(scan_indices.len());
    for &idx in scan_indices {
        let node = graph.node_view(idx)?;
        let row = node.data().properties.columnar_row_id()?;
        if !store.is_tombstoned(row) {
            rows.push(row);
        }
    }
    Some(rows)
}

/// Single pass over `scan_indices`, folding id / title / every stored property
/// into `accum`.
///
/// Keyed by [`InternedKey`] (a `Copy` u64) rather than `String`: the row loop
/// runs once per property per node, so string keying cost one allocation plus a
/// string hash per property per node (~650k allocations for a 50k-node × 13-prop
/// type). Names are resolved once, after the scan, in
/// [`compute_property_stats`].
///
/// **Column-major where the store allows it** ([`columnar_scan_rows`]): the
/// stored properties are folded one whole column at a time
/// ([`PropAccum::add_column`]), which turns the row loop's per-`(row, property)`
/// accumulator probe into one probe per property and lets a capped property
/// answer from its null byte instead of materialising a `Value` per row. The
/// identity fields still come from the node walk — they resolve through
/// `NodeView`, which prefers an inline value over the store's column.
///
/// Reads through [`NodeView`](crate::graph::storage::NodeView): the previous
/// `NodeData::property_iter` route yielded **nothing** for columnar storage, so
/// on a saved graph this pass contributed no values at all and the stats
/// degraded to whatever the `type_schemas` pre-seed supplied — keys with a zero
/// non-null count (D1 defect 1). `property_pairs` is the same complete route as
/// `property_pairs_named`, minus the per-key `String`.
fn accumulate_property_values(
    graph: &DirGraph,
    node_type: &str,
    scan_indices: &[petgraph::graph::NodeIndex],
    value_cap: usize,
    accum: &mut FxHashMap<InternedKey, PropAccum>,
) {
    let (id_key, title_key) = builtin_accum_keys();
    let store = graph.graph.column_store(InternedKey::from_str(node_type));
    // Resolved before the walk, never mid-walk: a scan that discovered an
    // inline node halfway through would have to re-visit the rows it had
    // already folded column-major.
    let column_major = store
        .and_then(|store| columnar_scan_rows(graph, store, scan_indices).map(|rows| (store, rows)));

    for &idx in scan_indices {
        let Some(node) = graph.node_view(idx) else {
            continue;
        };
        accum
            .entry(id_key)
            .or_insert_with(|| PropAccum::new(value_cap))
            .add(&node.id());
        accum
            .entry(title_key)
            .or_insert_with(|| PropAccum::new(value_cap))
            .add(&node.title());
        if column_major.is_none() {
            for (key, value) in node.property_pairs() {
                accum
                    .entry(key)
                    .or_insert_with(|| PropAccum::new(value_cap))
                    .add(&value);
            }
        }
    }

    let Some((store, rows)) = column_major else {
        return;
    };
    for (slot, key) in store.schema().iter() {
        let Some(col) = store.column(slot as usize) else {
            continue;
        };
        // A key no row in the scan carries must stay *out* of the map: the row
        // loop only ever inserted a key it had a value for, and an empty entry
        // would add a property row to `describe`'s output.
        match accum.entry(key) {
            std::collections::hash_map::Entry::Occupied(mut existing) => {
                existing.get_mut().add_column(col, &rows);
            }
            std::collections::hash_map::Entry::Vacant(vacant) => {
                let mut fresh = PropAccum::new(value_cap);
                if fresh.add_column(col, &rows) {
                    vacant.insert(fresh);
                }
            }
        }
    }
}

/// Resolve the scan's interned accumulator keys back to property names.
///
/// Keys the interner cannot resolve are dropped, matching the
/// `property_pairs_named` contract the scan used to enforce inline; the two
/// built-ins resolve to their fixed names whether or not a real property shares
/// the key.
fn resolve_accum_names(
    graph: &DirGraph,
    interned: FxHashMap<InternedKey, PropAccum>,
) -> HashMap<String, PropAccum> {
    let (id_key, title_key) = builtin_accum_keys();
    let mut out = HashMap::with_capacity(interned.len());
    for (key, accum) in interned {
        let name = if key == id_key {
            "id"
        } else if key == title_key {
            "title"
        } else if let Some(name) = graph.interner.try_resolve(key) {
            name
        } else {
            continue;
        };
        out.insert(name.to_string(), accum);
    }
    out
}

pub fn compute_property_stats(
    graph: &DirGraph,
    node_type: &str,
    max_values: usize,
    sample_size: Option<usize>,
) -> Result<Vec<PropertyStatInfo>, String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
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

    // Single pass: accumulate stats for all properties simultaneously.
    // Interned keys throughout — names are resolved once, after the scan.
    let (id_key, title_key) = builtin_accum_keys();
    let mut interned_accum: FxHashMap<InternedKey, PropAccum> = FxHashMap::default();
    // Pre-insert built-in fields so they appear even when all null
    interned_accum.insert(title_key, PropAccum::new(value_cap));
    interned_accum.insert(id_key, PropAccum::new(value_cap));

    // When sampling, pre-populate property keys from TypeSchema (knows ALL keys)
    if sample_size.is_some() {
        if let Some(schema) = graph.type_schemas.get(node_type) {
            for slot_key in schema.iter() {
                interned_accum
                    .entry(slot_key.1)
                    .or_insert_with(|| PropAccum::new(value_cap));
            }
        }
    }

    accumulate_property_values(
        graph,
        node_type,
        &scan_indices,
        value_cap,
        &mut interned_accum,
    );
    let mut accum = resolve_accum_names(graph, interned_accum);

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
        approx: false, // every node has exactly this type — always exact
    });

    // Whether we scanned only a subset: `unique`/`values` are then observations
    // over the sample, never a proven exhaustive count.
    let sampled = sample_count < total_nodes;

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
            // The distinct-value set is capped at `value_cap`; hitting it means
            // `unique` is a lower bound (there may be more distinct values we
            // stopped counting). Either that or a subset scan makes stats approx.
            let capped = pa.value_cap != usize::MAX && unique >= pa.value_cap;
            let approx = sampled || capped;
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
                approx,
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
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let node_indices = graph
        .type_indices
        .get(node_type)
        .ok_or_else(|| format!("Node type '{}' not found", node_type))?;

    let mut outgoing: HashMap<(String, String), usize> = HashMap::new();
    let mut incoming: HashMap<(String, String), usize> = HashMap::new();

    let g = &graph.graph;
    for node_idx in node_indices.iter() {
        for edge_ref in g.edges_directed(node_idx, Direction::Outgoing) {
            if let Some(target_node) = graph.node_view(edge_ref.target()) {
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
            if let Some(source_node) = graph.node_view(edge_ref.source()) {
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
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    // Key: (source_type, conn_type, target_type) → count
    let mut edge_counts: HashMap<(String, String, String), usize> = HashMap::new();

    let g = &graph.graph;
    for edge_ref in g.edge_references() {
        if let (Some(source), Some(target)) = (
            graph.node_view(edge_ref.source()),
            graph.node_view(edge_ref.target()),
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
///
/// Yields [`NodeView`]s, not `&NodeData`: a bare `NodeData` reference hands the
/// caller one replica of a columnar type's column store, which is how the
/// sample block in `describe()` came to enumerate zero properties for saved
/// graphs. The returned views borrow the disk arena, so they must be consumed
/// inside the caller's read pass.
pub fn compute_sample<'a>(
    graph: &'a DirGraph,
    node_type: &str,
    n: usize,
) -> Result<Vec<crate::graph::storage::NodeView<'a>>, String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let node_indices = graph
        .type_indices
        .get(node_type)
        .ok_or_else(|| format!("Node type '{}' not found", node_type))?;

    let mut result = Vec::with_capacity(n.min(node_indices.len()));
    for idx in node_indices.iter().take(n) {
        if let Some(node) = graph.node_view(idx) {
            result.push(node);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod column_major_stats_tests {
    use super::*;
    use crate::datatypes::DataFrame;
    use crate::graph::dir_graph::DirGraph;

    /// A fixture whose columns cover every arm `PropAccum::add_column`
    /// distinguishes, plus the shapes the row loop resolves differently:
    /// low- and high-cardinality strings (capped vs enumerated distinct sets),
    /// ints, floats **including a NaN** (null to `is_null_value`, non-null to
    /// the null byte), booleans, a sparse column (nulls in the middle), a
    /// `Mixed` column holding lists, and a property literally named `id`.
    fn wide_fixture(n: i64) -> DirGraph {
        let mut graph = DirGraph::new();
        let columns: Vec<String> = [
            "key",
            "label",
            "bucket",
            "unique_text",
            "count",
            "ratio",
            "flag",
            "sparse",
            "vec",
            "id",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let rows: Vec<Vec<Value>> = (0..n)
            .map(|i| {
                vec![
                    Value::Int64(i),
                    Value::String(format!("Item_{i}")),
                    Value::String(format!("bucket_{}", i % 3)),
                    Value::String(format!("text-{i}-{}", i * 7)),
                    Value::Int64(i * 10),
                    if i % 11 == 0 {
                        Value::Float64(f64::NAN)
                    } else {
                        Value::Float64(i as f64 / 3.0)
                    },
                    Value::Boolean(i % 2 == 0),
                    if i % 4 == 0 {
                        Value::Null
                    } else {
                        Value::String(format!("s{i}"))
                    },
                    Value::List(vec![Value::Int64(i), Value::Int64(i + 1)]),
                    Value::String(format!("shadow-{i}")),
                ]
            })
            .collect();
        let df = DataFrame::from_cypher_rows(columns, rows).unwrap();
        crate::graph::mutation::maintain::add_nodes(
            &mut graph,
            df,
            "Item".to_string(),
            "key".to_string(),
            Some("label".to_string()),
            None,
        )
        .unwrap();
        graph.enable_columnar();
        graph
    }

    /// The Track-H shape: many wide, high-cardinality string properties, the
    /// case whose per-row `String` materialisation the column-major path is
    /// meant to remove. Shared with the release A/B probe.
    pub(super) fn wide_probe_fixture(n: i64) -> DirGraph {
        let mut graph = DirGraph::new();
        let names = [
            "key",
            "label",
            "citation",
            "summary",
            "section",
            "court",
            "docket",
            "para",
            "keywords",
            "source_url",
            "decided",
            "pages",
        ];
        let columns: Vec<String> = names.iter().map(|s| s.to_string()).collect();
        let rows: Vec<Vec<Value>> = (0..n)
            .map(|i| {
                vec![
                    Value::Int64(i),
                    Value::String(format!("Decision {i}")),
                    Value::String(format!("HR-{}-{i}-A", 1900 + (i % 120))),
                    Value::String(format!(
                        "A summary of decision {i} running to a realistic width for a \
                         law-shaped corpus, with clause {} and reference {}.",
                        i % 37,
                        i * 13
                    )),
                    Value::String(format!("section_{}", i % 24)),
                    Value::String(format!("court_{}", i % 9)),
                    Value::String(format!("{}-{:06}", i % 4, i)),
                    Value::String(format!("paragraph {} of {}", i % 60, 60)),
                    Value::String(format!("kw_{},kw_{},kw_{}", i % 11, i % 17, i % 23)),
                    Value::String(format!("https://example.invalid/decisions/{i}")),
                    Value::Int64(1900 + (i % 120)),
                    Value::Int64(i % 400),
                ]
            })
            .collect();
        let df = DataFrame::from_cypher_rows(columns, rows).unwrap();
        crate::graph::mutation::maintain::add_nodes(
            &mut graph,
            df,
            "Item".to_string(),
            "key".to_string(),
            Some("label".to_string()),
            None,
        )
        .unwrap();
        graph.enable_columnar();
        graph
    }

    /// Everything the two routes must agree on, in a comparable form.
    /// `sample` is deliberately excluded: it is documented as "one real value,
    /// not the same value every time" and is drawn from a `HashSet` whose
    /// iteration order is randomised per process.
    fn comparable(stats: &[PropertyStatInfo]) -> Vec<(String, String, usize, usize, bool, bool)> {
        stats
            .iter()
            .map(|s| {
                (
                    s.property_name.clone(),
                    s.type_string.clone(),
                    s.non_null,
                    s.unique,
                    s.approx,
                    s.values.is_some(),
                )
            })
            .collect()
    }

    fn both_routes(graph: &DirGraph, max_values: usize, sample: Option<usize>) {
        let column_major = compute_property_stats(graph, "Item", max_values, sample).unwrap();
        let row_major =
            with_row_major_stats(|| compute_property_stats(graph, "Item", max_values, sample))
                .unwrap();
        assert_eq!(
            comparable(&column_major),
            comparable(&row_major),
            "column-major stats diverged from the row loop (max_values={max_values}, \
             sample={sample:?})"
        );
        // Enumerated value sets are a stronger claim than the counts: where a
        // property is under the cap, the two routes must report the *same
        // values*, sorted identically.
        for (col, row) in column_major.iter().zip(row_major.iter()) {
            assert_eq!(
                col.values, row.values,
                "enumerated values diverged for {}",
                col.property_name
            );
        }
    }

    #[test]
    fn column_major_stats_match_the_row_loop() {
        let graph = wide_fixture(64);
        // 16 is `describe`'s own threshold: `bucket` stays under it (3 distinct)
        // while `unique_text` blows past it, so one call exercises both the
        // enumerated and the capped-and-presence-counted paths.
        both_routes(&graph, 16, None);
        // max_values = 0 keeps every distinct set uncapped — the arm where the
        // presence shortcut must never engage.
        both_routes(&graph, 0, None);
        both_routes(&graph, 1024, None);
        both_routes(&graph, 16, Some(8));
    }

    #[test]
    fn column_major_stats_match_the_row_loop_after_writes() {
        let mut graph = wide_fixture(48);
        let params = std::collections::HashMap::new();
        let opts = crate::graph::session::ExecuteOptions::eager(&params);
        // A differing-length string `SET` lands in the `Str` column's
        // relocation overlay rather than its byte arena — the one shape whose
        // rows are not a straight slice walk.
        crate::graph::session::execute_mut(
            &mut graph,
            "MATCH (n:Item) WHERE n.count < 100 SET n.bucket = 'relocated-to-a-much-longer-value'",
            &opts,
        )
        .unwrap();
        // A key the store's schema never saw grows a column back-filled with
        // nulls, so most rows are absent for it.
        crate::graph::session::execute_mut(
            &mut graph,
            "MATCH (n:Item) WHERE n.count < 30 SET n.late = 'added'",
            &opts,
        )
        .unwrap();
        both_routes(&graph, 16, None);

        // Deleting rows tombstones them; a tombstoned row yields no properties
        // through either route.
        crate::graph::session::execute_mut(
            &mut graph,
            "MATCH (n:Item) WHERE n.count > 400 DELETE n",
            &opts,
        )
        .unwrap();
        both_routes(&graph, 16, None);
        both_routes(&graph, 0, None);
    }

    /// The equivalence test is only worth its runtime if the two routes can
    /// actually disagree — i.e. if the fixture really takes the column-major
    /// path. Proven by the one observable difference between them: the row
    /// loop reads through `NodeView::property_pairs`, the column path does not.
    #[test]
    fn the_fixture_takes_the_column_major_path() {
        let graph = wide_fixture(8);
        let store = graph
            .graph
            .column_store(InternedKey::from_str("Item"))
            .expect("fixture must be columnar");
        let indices: Vec<_> = graph.type_indices.get("Item").unwrap().to_vec();
        assert_eq!(
            columnar_scan_rows(&graph, store, &indices).map(|rows| rows.len()),
            Some(8),
            "the fixture must qualify for the column-major path, or the \
             equivalence test compares the row loop with itself"
        );
        assert!(
            with_row_major_stats(|| columnar_scan_rows(&graph, store, &indices)).is_none(),
            "the decline hook must actually decline"
        );
    }
}

/// In-process A/B probe for the column-major stats path.
///
/// Both routes run against the same fixture in the same binary, so the
/// comparison is immune to the build-to-build variance a two-wheel A/B carries
/// — the only difference between the two timings is
/// [`columnar_scan_rows`]'s answer. Release profile only; a debug reading is
/// invalid (project performance protocol) and the probe says so rather than
/// producing a number nobody should quote.
///
/// `cargo test -p kglite --release --lib -- --ignored --nocapture stats_column_major_ab`
#[cfg(test)]
mod column_major_stats_probe {
    use super::column_major_stats_tests::wide_probe_fixture;
    use super::{compute_property_stats, with_row_major_stats};
    use std::time::Instant;

    fn min_of(rounds: usize, mut f: impl FnMut()) -> f64 {
        let mut best = f64::MAX;
        for _ in 0..rounds {
            let start = Instant::now();
            f();
            best = best.min(start.elapsed().as_secs_f64() * 1e3);
        }
        best
    }

    #[test]
    #[ignore = "perf probe — release profile only"]
    fn stats_column_major_ab() {
        if cfg!(debug_assertions) {
            panic!("run this probe with --release; a debug-profile number is invalid");
        }
        for nodes in [5_000i64, 50_000] {
            let graph = wide_probe_fixture(nodes);
            // Warm the caches both ways before either is timed.
            let _ = compute_property_stats(&graph, "Item", 16, None).unwrap();
            let _ = with_row_major_stats(|| compute_property_stats(&graph, "Item", 16, None));
            let rounds = if nodes <= 5_000 { 40 } else { 10 };
            let row = min_of(rounds, || {
                with_row_major_stats(|| compute_property_stats(&graph, "Item", 16, None)).unwrap();
            });
            let col = min_of(rounds, || {
                compute_property_stats(&graph, "Item", 16, None).unwrap();
            });
            println!(
                "compute_property_stats  n={nodes:>6}  row-major {row:8.3} ms  \
                 column-major {col:8.3} ms  speedup {:.2}x",
                row / col
            );
        }
    }
}

#[cfg(test)]
mod constraint_row_vocabulary_tests {
    use super::*;

    fn info(entity: EntityKind, kind: ConstraintKind) -> ConstraintInfo {
        ConstraintInfo {
            name: "c".to_string(),
            kind,
            entity,
            labels_or_types: vec!["T".to_string()],
            properties: vec!["p".to_string()],
            property_type: None,
        }
    }

    /// The `type` column is Neo4j's `ConstraintType` vocabulary, and its two
    /// halves are not a prefix apart: the node uniqueness spelling is bare
    /// `UNIQUENESS` while the relationship one is `RELATIONSHIP_UNIQUENESS`.
    /// A ported script matching on these strings sees the same words Neo4j
    /// gave it.
    #[test]
    fn each_kind_reports_under_its_entity_s_neo4j_type() {
        for (entity, kind, expected) in [
            (EntityKind::Node, ConstraintKind::Unique, "UNIQUENESS"),
            (EntityKind::Node, ConstraintKind::NodeKey, "NODE_KEY"),
            (
                EntityKind::Node,
                ConstraintKind::NotNull,
                "NODE_PROPERTY_EXISTENCE",
            ),
            (
                EntityKind::Node,
                ConstraintKind::PropertyType,
                "NODE_PROPERTY_TYPE",
            ),
            (
                EntityKind::Relationship,
                ConstraintKind::Unique,
                "RELATIONSHIP_UNIQUENESS",
            ),
            (
                EntityKind::Relationship,
                ConstraintKind::NodeKey,
                "RELATIONSHIP_KEY",
            ),
            (
                EntityKind::Relationship,
                ConstraintKind::NotNull,
                "RELATIONSHIP_PROPERTY_EXISTENCE",
            ),
            (
                EntityKind::Relationship,
                ConstraintKind::PropertyType,
                "RELATIONSHIP_PROPERTY_TYPE",
            ),
        ] {
            assert_eq!(
                info(entity, kind).neo4j_type(),
                expected,
                "{entity:?} {kind:?}"
            );
        }
    }

    #[test]
    fn the_entity_type_column_reads_from_the_entity() {
        assert_eq!(
            info(EntityKind::Node, ConstraintKind::Unique).entity_type(),
            "NODE"
        );
        assert_eq!(
            info(EntityKind::Relationship, ConstraintKind::Unique).entity_type(),
            "RELATIONSHIP"
        );
    }

    /// Every row the collector emits today is a node row — the relationship
    /// stores do not exist yet — and `SHOW CONSTRAINTS` must not start
    /// claiming otherwise until they do.
    #[test]
    fn the_collector_still_emits_only_node_rows() {
        let mut graph = DirGraph::new();
        graph
            .create_not_null_constraint("Person", "email")
            .expect("declaration");
        let rows = collect_constraints_structured(&graph);
        assert!(!rows.is_empty());
        assert!(rows.iter().all(|row| row.entity == EntityKind::Node));
    }
}
