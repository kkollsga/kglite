//! Native graph-merge: fold one in-memory graph into another in place.
//!
//! [`extend_graph`] merges every node and edge of a read-only *source*
//! graph into a mutable *target* graph, reusing the exact bulk-load
//! machinery (`maintain::add_nodes` / `maintain::add_connections`) so
//! conflict handling, schema merging, id-indexing and edge dedup all
//! behave identically to a CSV round-trip — without the round-trip.
//!
//! ## Why route through `add_nodes` / `add_connections`
//!
//! Those two functions are the single source of truth for:
//! - `ConflictHandling` dispatch (`update` / `replace` / `skip` /
//!   `preserve` / `sum`),
//! - `TypeSchema` / interner extension for new properties,
//! - eager id-index rebuild (`build_id_index`),
//! - edge dedup keyed on `(connection_type, src, tgt)`, plus a declared
//!   temporal `from` bound, with per-mode
//!   property merge.
//!
//! Re-implementing any of that here would risk drift. Instead we
//! materialise the source's nodes/edges into `DataFrame`s grouped the
//! way those functions expect, then call them. The cost is one
//! `DataFrame` build per `(node_type)` and per
//! `(connection_type, source_type, target_type)` group — `O(nodes2 +
//! edges2)` total, with id-index lookups (not scans) for matching.
//!
//! ## Semantics (see the Python `extend` docstring for the user-facing
//! contract)
//!
//! - **Node identity** is `(node_type, id)`, matching the id-index used
//!   by `add_nodes`. `id` is the canonical integer node id in every
//!   storage mode (post-0.10.10). Conflicts resolve per
//!   `conflict_handling`.
//! - **Secondary labels** (multi-label) are *unioned*
//!   onto the matched/created target node via
//!   [`DirGraph::add_node_label`] — idempotent, never removes a label.
//! - **Edges** dedup exactly as `add_connections` does: an edge with the
//!   same `(connection_type, src, tgt)` — plus the same `from` bound when
//!   the type carries a temporal declaration — that already exists in the
//!   target is *not* duplicated; its properties merge per
//!   `conflict_handling`. This is the defensible choice over petgraph's
//!   raw parallel-edge capability — a merge that silently doubled every
//!   shared edge would be surprising. Parallel edges the *source* itself
//!   carries between one pair follow `add_connections`' ownership rule
//!   (`maintain::source_owns_its_edges`): for a connection type the target
//!   holds no edge of from that source node type, every one is copied; for
//!   one it does, they fold onto one edge per key, as a re-load would.
//! - **Constraints**: every edge group is judged against the target's
//!   relationship constraints before the first node or edge is written, so a
//!   refusal leaves the target as it was.
//! - **Property schemas** merge through the same `upsert_node_type_metadata`
//!   / `type_schemas` extension path `add_nodes` uses.
//! - **Declarations**: the source's temporal declarations and spatial
//!   configs are copied onto the target. The target's own declaration of the
//!   same key wins; a temporal one that conflicts with it, or that the
//!   merged rows refuse, is not copied and is named in `errors`.
//!
//! ## v1 scope limits
//!
//! - Both graphs must be in-memory `Default` storage. Mapped/Disk are
//!   rejected with a clear error (callers should export/import instead).
//! - The source graph is **never mutated** — it is read through
//!   `GraphRead` only.
//! - Embedding stores are **not** merged (the caller surfaces a warning
//!   when the source has any). Re-run `set_embeddings` / `add_embeddings`
//!   after the merge.

use crate::datatypes::{DataFrame, Value};
use crate::graph::features::temporal;
use crate::graph::features::temporal::merge_start_key;
use crate::graph::introspection::reporting::{ConnectionOperationReport, NodeOperationReport};
use crate::graph::mutation::batch::ConflictHandling;
use crate::graph::mutation::edge_props::resolve_edge_property_columns;
use crate::graph::mutation::maintain::{
    add_connections_with_initial_load, add_nodes, gate_node_rows, parse_conflict_mode,
    source_owns_its_edges, InitialLoad,
};
use crate::graph::mutation::rel_constraint_gate::{ConnectionBatchGate, RowFolding};
use crate::graph::schema::DirGraph;
use crate::graph::storage::GraphRead;
use std::collections::{BTreeMap, HashMap, HashSet};

/// Combined report for an `extend` merge.
#[derive(Debug, Clone)]
pub struct ExtendReport {
    pub nodes_created: usize,
    pub nodes_updated: usize,
    pub nodes_skipped: usize,
    pub edges_created: usize,
    /// Source edges merged into an existing edge of the same type between the
    /// same endpoints, per `conflict_handling`.
    pub edges_updated: usize,
    pub edges_skipped: usize,
    pub node_types_merged: usize,
    pub connection_types_merged: usize,
    pub labels_unioned: usize,
    pub processing_time_ms: f64,
    pub errors: Vec<String>,
}

/// Per-type node rows accumulated from the source graph, ready to be
/// turned into a `DataFrame` for `add_nodes`. Each row is keyed by
/// property name; the canonical `id` and `title` ride in dedicated
/// fields so an absent property in one row doesn't shift columns.
struct NodeGroup {
    /// All property column names seen across this type's source nodes.
    columns: Vec<String>,
    column_set: std::collections::HashSet<String>,
    /// One entry per source node: (id, title, property map).
    rows: Vec<(Value, Value, HashMap<String, Value>)>,
}

impl NodeGroup {
    fn new() -> Self {
        NodeGroup {
            columns: Vec::new(),
            column_set: std::collections::HashSet::new(),
            rows: Vec::new(),
        }
    }

    fn note_column(&mut self, name: &str) {
        if self.column_set.insert(name.to_string()) {
            self.columns.push(name.to_string());
        }
    }
}

/// Per-(conn_type, source_type, target_type) edge rows.
struct EdgeGroup {
    source_type: String,
    target_type: String,
    columns: Vec<String>,
    column_set: std::collections::HashSet<String>,
    /// (source_id, target_id, property map).
    rows: Vec<(Value, Value, HashMap<String, Value>)>,
}

impl EdgeGroup {
    fn new(source_type: String, target_type: String) -> Self {
        EdgeGroup {
            source_type,
            target_type,
            columns: Vec::new(),
            column_set: std::collections::HashSet::new(),
            rows: Vec::new(),
        }
    }

    fn note_column(&mut self, name: &str) {
        if self.column_set.insert(name.to_string()) {
            self.columns.push(name.to_string());
        }
    }
}

fn snapshot_node_row(
    source: &DirGraph,
    mut title: Value,
    mut properties: HashMap<String, Value>,
) -> (Value, HashMap<String, Value>) {
    crate::graph::session::snapshot_property_values(
        &source.graph,
        std::iter::once(&mut title).chain(properties.values_mut()),
    );
    properties.remove("id");
    properties.remove("title");
    (title, properties)
}

fn snapshot_edge_properties(
    source: &DirGraph,
    mut properties: HashMap<String, Value>,
) -> HashMap<String, Value> {
    crate::graph::session::snapshot_property_values(&source.graph, properties.values_mut());
    properties
}

/// Merge `source` into `target` in place. See module docs for full
/// semantics. `source` is read-only.
///
/// Errors when either graph is not the in-memory `Default` backend, when a
/// relationship constraint on the target refuses an edge group (before
/// anything is written), or when a routed `add_nodes` / `add_connections`
/// call fails (the error string is propagated unchanged).
pub fn extend_graph(
    target: &mut DirGraph,
    source: &DirGraph,
    conflict_handling: Option<String>,
) -> Result<ExtendReport, String> {
    let start = std::time::Instant::now();

    // Scope limit: in-memory Default backend only on BOTH sides.
    if target.graph.is_mapped() || target.graph.is_disk() {
        return Err(scope_error("target"));
    }
    if source.graph.is_mapped() || source.graph.is_disk() {
        return Err(scope_error("source"));
    }

    // ---- Pass 1: collect source nodes grouped by node_type ----
    //
    // We also remember, per source NodeIndex, its (node_type, id) so the
    // label-union pass can resolve the *target* node after merge.
    let mut node_groups: BTreeMap<String, NodeGroup> = BTreeMap::new();
    // (node_type, id, secondary_label_names) for the label-union pass.
    let mut label_carriers: Vec<(String, Value, Vec<String>)> = Vec::new();

    for idx in source.graph.node_indices() {
        let Some(node) = source.graph.node_view(idx) else {
            continue;
        };
        let node_type = node.node_type_str(&source.interner).to_string();
        let id = node.id().into_owned();
        let props = node.properties_cloned(&source.interner);
        let (title, props) = snapshot_node_row(source, node.title().into_owned(), props);

        let group = node_groups
            .entry(node_type.clone())
            .or_insert_with(NodeGroup::new);
        for k in props.keys() {
            group.note_column(k);
        }
        group.rows.push((id.clone(), title, props));

        // Secondary labels (everything beyond the primary type).
        let labels = source.node_labels(idx);
        if labels.len() > 1 {
            let secondaries: Vec<String> = labels
                .iter()
                .skip(1)
                .map(|k| source.interner.resolve(*k).to_string())
                .collect();
            label_carriers.push((node_type, id, secondaries));
        }
    }

    // Validate the complete merge vocabulary before the first routed write.
    // Per-group validation inside add_nodes/add_connections is too late here:
    // a collision in a later node/edge group would otherwise leave earlier
    // groups committed to the target.
    let mut incoming_names = vec![
        "id".to_string(),
        "title".to_string(),
        "src_id".to_string(),
        "tgt_id".to_string(),
        "__provisional__".to_string(),
        "updated_at".to_string(),
    ];
    for (node_type, group) in &node_groups {
        incoming_names.push(node_type.clone());
        incoming_names.extend(group.columns.iter().cloned());
    }
    for (_, _, labels) in &label_carriers {
        incoming_names.extend(labels.iter().cloned());
    }
    for edge_idx in source.graph.edge_indices() {
        let Some(edge) = source.graph.edge_weight(edge_idx) else {
            continue;
        };
        incoming_names.push(edge.connection_type_str(&source.interner).to_string());
        incoming_names.extend(edge.properties_cloned(&source.interner).into_keys());
        if let Some((src_idx, tgt_idx)) = source.graph.edge_endpoints(edge_idx) {
            for idx in [src_idx, tgt_idx] {
                if let Some(node) = source.graph.node_view(idx) {
                    incoming_names.push(node.node_type_str(&source.interner).to_string());
                }
            }
        }
    }
    target
        .interner
        .validate_names(incoming_names.iter().map(String::as_str))
        .map_err(|err| err.to_string())?;

    let edge_groups = collect_edge_groups(source);
    let mut report = ExtendReport {
        nodes_created: 0,
        nodes_updated: 0,
        nodes_skipped: 0,
        edges_created: 0,
        edges_updated: 0,
        edges_skipped: 0,
        node_types_merged: node_groups.len(),
        connection_types_merged: edge_groups
            .keys()
            .map(|(conn_type, _, _)| conn_type)
            .collect::<HashSet<_>>()
            .len(),
        labels_unioned: 0,
        processing_time_ms: 0.0,
        errors: Vec::new(),
    };

    // The temporal declarations go in before any row, so the edge merge — and
    // the gate below, which must model it — key declared relationship types on
    // their `from` bound. They are validated once the edges have landed, or
    // withdrawn when the merge fails before then.
    let adopted = temporal::adopt_declarations(target, temporal::list(source), &mut report.errors);
    let merged = merge_rows(
        target,
        source,
        node_groups,
        label_carriers,
        edge_groups,
        &conflict_handling,
        &mut report,
    );
    if let Err(e) = merged {
        temporal::withdraw_adopted(target, adopted);
        return Err(e);
    }
    temporal::settle_adopted(target, adopted, &mut report.errors);

    report.processing_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(report)
}

/// `(connection_type, source node type, target node type)` — one
/// `add_connections` call each.
type EdgeGroupKey = (String, String, String);

/// Collect the source's edges grouped by [`EdgeGroupKey`], in key order so a
/// refusal names the same group on every run. The endpoint *node types* are
/// resolved in the source graph so `add_connections` can id-index-match them
/// in the target.
fn collect_edge_groups(source: &DirGraph) -> BTreeMap<EdgeGroupKey, EdgeGroup> {
    let mut edge_groups: BTreeMap<EdgeGroupKey, EdgeGroup> = BTreeMap::new();
    for edge_idx in source.graph.edge_indices() {
        let Some(edge) = source.graph.edge_weight(edge_idx) else {
            continue;
        };
        let Some((src_idx, tgt_idx)) = source.graph.edge_endpoints(edge_idx) else {
            continue;
        };
        let conn_type = edge.connection_type_str(&source.interner).to_string();
        let (Some(src_node), Some(tgt_node)) = (
            source.graph.node_view(src_idx),
            source.graph.node_view(tgt_idx),
        ) else {
            continue;
        };
        let source_type = src_node.node_type_str(&source.interner).to_string();
        let target_type = tgt_node.node_type_str(&source.interner).to_string();
        let src_id = src_node.id().into_owned();
        let tgt_id = tgt_node.id().into_owned();
        let props = snapshot_edge_properties(source, edge.properties_cloned(&source.interner));

        let group = edge_groups
            .entry((conn_type, source_type.clone(), target_type.clone()))
            .or_insert_with(|| EdgeGroup::new(source_type, target_type));
        for k in props.keys() {
            group.note_column(k);
        }
        group.rows.push((src_id, tgt_id, props));
    }
    edge_groups
}

/// Write the source's rows into the target: every node and edge group is
/// judged against the target's constraints first, then the nodes, labels and
/// edges land. A refusal therefore writes nothing — not the node groups, and
/// not a group that sorts before the refused one.
fn merge_rows(
    target: &mut DirGraph,
    source: &DirGraph,
    node_groups: BTreeMap<String, NodeGroup>,
    label_carriers: Vec<(String, Value, Vec<String>)>,
    edge_groups: BTreeMap<EdgeGroupKey, EdgeGroup>,
    conflict_handling: &Option<String>,
    report: &mut ExtendReport,
) -> Result<(), String> {
    let conflict_mode = parse_conflict_mode(conflict_handling.as_deref())?;
    // Whether the target has edges of a connection type from a group's source
    // type (`maintain::source_owns_its_edges`) is decided before any group
    // lands: the first group of a pair registers it, so re-detecting per group
    // would keep the parallel edges of whichever group ran first and fold
    // every later group's. Node writes register no connection type, so the
    // gate and the merge see the same answer.
    let owned: HashSet<(String, String)> = edge_groups
        .iter()
        .filter(|((conn_type, _, _), group)| {
            source_owns_its_edges(target, conn_type, &group.source_type)
        })
        .map(|((conn_type, _, _), group)| (conn_type.clone(), group.source_type.clone()))
        .collect();
    gate_node_groups(target, &node_groups)?;
    gate_edge_groups(target, &edge_groups, &owned, conflict_mode)?;
    write_node_groups(target, source, node_groups, conflict_handling, report)?;
    union_labels(target, label_carriers, report);
    copy_spatial_configs(target, source);
    merge_edge_groups(target, edge_groups, &owned, conflict_handling, report)
}

/// Judge every node group against the target's node constraints, as its
/// `add_nodes` call will. Node types are independent, so each alone is exact.
fn gate_node_groups(
    target: &mut DirGraph,
    node_groups: &BTreeMap<String, NodeGroup>,
) -> Result<(), String> {
    for (node_type, group) in node_groups {
        gate_node_rows(target, node_type, "id", "title", || {
            build_node_dataframe(group)
        })?;
    }
    Ok(())
}

/// Judge every edge group against the target's relationship constraints,
/// under the regime its `add_connections` call will run in, before anything
/// is written. Rows are resolved against the target as it stands: an
/// endpoint the node pass has yet to create cannot hold a stored edge, so its
/// row is judged as a create, keyed by ids. Groups never share an endpoint
/// pair (the key includes both node types), so judging each alone is exact.
fn gate_edge_groups(
    target: &mut DirGraph,
    edge_groups: &BTreeMap<EdgeGroupKey, EdgeGroup>,
    owned: &HashSet<(String, String)>,
    conflict_mode: ConflictHandling,
) -> Result<(), String> {
    if !target.has_rel_constraints() {
        return Ok(());
    }
    for ((conn_type, source_type, target_type), group) in edge_groups {
        if !target.type_has_rel_constraints(conn_type) {
            continue;
        }
        let df = build_edge_dataframe(group)?;
        let mut matched = Vec::new();
        let mut deferred = Vec::new();
        for (row, (src_id, tgt_id, _)) in group.rows.iter().enumerate() {
            match (
                target.lookup_by_id_normalized(source_type, src_id),
                target.lookup_by_id_normalized(target_type, tgt_id),
            ) {
                (Some(src), Some(tgt)) => matched.push((row, src, tgt)),
                _ => deferred.push((row, src_id.clone(), tgt_id.clone())),
            }
        }
        let is_owned = owned.contains(&(conn_type.clone(), source_type.clone()));
        let start_key = merge_start_key(target, conn_type, Some(source_type));
        ConnectionBatchGate {
            connection_type: conn_type,
            rows: &df,
            property_columns: &resolve_edge_property_columns(&df, "src_id", "tgt_id", None, None),
            matched: &matched,
            deferred: &deferred,
            conflict_mode,
            folding: RowFolding::for_load(is_owned),
            start_key: start_key.as_ref(),
        }
        .run(target)?;
    }
    Ok(())
}

/// Route each node group through `add_nodes`.
fn write_node_groups(
    target: &mut DirGraph,
    source: &DirGraph,
    node_groups: BTreeMap<String, NodeGroup>,
    conflict_handling: &Option<String>,
    report: &mut ExtendReport,
) -> Result<(), String> {
    for (node_type, group) in node_groups {
        // Carry the source's id/title field aliases for a type the target
        // doesn't already have, so `MATCH (n {originalIdCol: ...})` keeps
        // resolving after the merge. For an already-present type the
        // target's own alias is authoritative — leave it.
        let target_has_type = target.type_indices.get(&node_type).is_some();
        if !target_has_type {
            if let Some(alias) = source.id_field_aliases.get(&node_type) {
                target
                    .id_field_aliases_mut()
                    .insert(node_type.clone(), alias.clone());
            }
            if let Some(alias) = source.title_field_aliases.get(&node_type) {
                target
                    .title_field_aliases_mut()
                    .insert(node_type.clone(), alias.clone());
            }
        }

        let df = build_node_dataframe(&group)?;
        let r: NodeOperationReport = add_nodes(
            target,
            df,
            node_type,
            "id".to_string(),
            // Always carry the title across (the column is always present).
            Some("title".to_string()),
            conflict_handling.clone(),
        )?;
        report.nodes_created += r.nodes_created;
        report.nodes_updated += r.nodes_updated;
        report.nodes_skipped += r.nodes_skipped;
        report.errors.extend(r.errors);
    }
    Ok(())
}

/// Union the source's secondary labels onto the target nodes. Runs after the
/// node pass so every carrier's target node exists; the id index (rebuilt by
/// `add_nodes`) makes each lookup O(1).
fn union_labels(
    target: &mut DirGraph,
    label_carriers: Vec<(String, Value, Vec<String>)>,
    report: &mut ExtendReport,
) {
    for (node_type, id, secondaries) in label_carriers {
        if let Some(target_idx) = target.lookup_by_id(&node_type, &id) {
            for label in secondaries {
                let key = target.interner.get_or_intern(&label);
                if target.add_node_label(target_idx, key) {
                    report.labels_unioned += 1;
                }
            }
        }
    }
}

/// Route each source edge group through `add_connections`, tallying the
/// outcome into `report`, under the ownership `owned` decided before any
/// group landed (see [`merge_rows`]).
fn merge_edge_groups(
    target: &mut DirGraph,
    edge_groups: BTreeMap<EdgeGroupKey, EdgeGroup>,
    owned: &HashSet<(String, String)>,
    conflict_handling: &Option<String>,
    report: &mut ExtendReport,
) -> Result<(), String> {
    for ((conn_type, _, _), group) in edge_groups {
        let df = build_edge_dataframe(&group)?;
        let initial_load =
            InitialLoad::Preset(owned.contains(&(conn_type.clone(), group.source_type.clone())));
        let r: ConnectionOperationReport = add_connections_with_initial_load(
            target,
            df,
            conn_type,
            group.source_type,
            "src_id".to_string(),
            group.target_type,
            "tgt_id".to_string(),
            None,
            None,
            conflict_handling.clone(),
            initial_load,
        )?;
        report.edges_created += r.connections_created;
        report.edges_updated += r.connections_updated;
        report.edges_skipped += r.connections_skipped;
        report.errors.extend(r.errors);
    }
    Ok(())
}

/// Carry the source's spatial configs onto node types the target has none
/// for; a type the target already configures keeps its own.
fn copy_spatial_configs(target: &mut DirGraph, source: &DirGraph) {
    let mut configs: Vec<_> = source
        .spatial_configs
        .iter()
        .filter(|(node_type, _)| !target.spatial_configs.contains_key(*node_type))
        .map(|(node_type, config)| (node_type.clone(), config.clone()))
        .collect();
    configs.sort_by(|a, b| a.0.cmp(&b.0));
    for (node_type, config) in configs {
        target.set_spatial_config(&node_type, config);
    }
}

fn scope_error(which: &str) -> String {
    format!(
        "extend() requires both graphs to use in-memory (Default) storage, but the {which} \
         graph is mapped/disk-backed. Merge by exporting one graph and re-importing it into the \
         other (e.g. export to CSV / a blueprint, then add_nodes / add_connections), or rebuild \
         both in memory before extending."
    )
}

/// Build a node `DataFrame` with columns `[id, title, <props...>]`.
/// Property cells absent on a given row are filled with `Value::Null`,
/// matching how `add_nodes` treats missing values (skip-on-null).
fn build_node_dataframe(group: &NodeGroup) -> Result<DataFrame, String> {
    let mut columns = Vec::with_capacity(group.columns.len() + 2);
    columns.push("id".to_string());
    columns.push("title".to_string());
    columns.extend(group.columns.iter().cloned());

    let rows: Vec<Vec<Value>> = group
        .rows
        .iter()
        .map(|(id, title, props)| {
            let mut row = Vec::with_capacity(columns.len());
            row.push(id.clone());
            row.push(title.clone());
            for col in &group.columns {
                row.push(props.get(col).cloned().unwrap_or(Value::Null));
            }
            row
        })
        .collect();

    DataFrame::from_cypher_rows(columns, rows)
}

/// Build an edge `DataFrame` with columns `[src_id, tgt_id, <props...>]`.
fn build_edge_dataframe(group: &EdgeGroup) -> Result<DataFrame, String> {
    let mut columns = Vec::with_capacity(group.columns.len() + 2);
    columns.push("src_id".to_string());
    columns.push("tgt_id".to_string());
    columns.extend(group.columns.iter().cloned());

    let rows: Vec<Vec<Value>> = group
        .rows
        .iter()
        .map(|(src, tgt, props)| {
            let mut row = Vec::with_capacity(columns.len());
            row.push(src.clone());
            row.push(tgt.clone());
            for col in &group.columns {
                row.push(props.get(col).cloned().unwrap_or(Value::Null));
            }
            row
        })
        .collect();

    DataFrame::from_cypher_rows(columns, rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::schema::InternedKey;

    #[test]
    fn collision_preflight_keeps_extend_target_unchanged() {
        let mut source = DirGraph::new();
        let frame = DataFrame::from_cypher_rows(
            vec!["id".into(), "title".into(), "late_collision".into()],
            vec![vec![
                Value::Int64(1),
                Value::String("one".into()),
                Value::Int64(7),
            ]],
        )
        .unwrap();
        add_nodes(
            &mut source,
            frame,
            "SourceNode".into(),
            "id".into(),
            Some("title".into()),
            None,
        )
        .unwrap();

        let mut target = DirGraph::new();
        target
            .interner
            .try_register(
                InternedKey::from_str("late_collision"),
                "conflicting-existing-name",
            )
            .unwrap();
        let version_before = target.version;
        let interner_before: Vec<_> = target
            .interner
            .iter()
            .map(|(key, name)| (key, name.to_string()))
            .collect();

        let error = extend_graph(&mut target, &source, None).unwrap_err();
        assert!(error.contains("hash collision"));
        assert_eq!(target.graph.node_count(), 0);
        assert_eq!(target.version, version_before);
        assert!(target.type_indices.is_empty());
        assert_eq!(
            target
                .interner
                .iter()
                .map(|(key, name)| (key, name.to_string()))
                .collect::<Vec<_>>(),
            interner_before
        );
    }
}
