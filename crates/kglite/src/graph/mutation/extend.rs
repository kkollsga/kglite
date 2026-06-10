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
//! - edge dedup keyed on `(connection_type, src, tgt)` with per-mode
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
//! - **Secondary labels** (multi-label, since 0.10.5) are *unioned*
//!   onto the matched/created target node via
//!   [`DirGraph::add_node_label`] — idempotent, never removes a label.
//! - **Edges** dedup exactly as `add_connections` does: an edge with the
//!   same `(connection_type, src, tgt)` that already exists in the
//!   target is *not* duplicated; its properties merge per
//!   `conflict_handling`. This is the defensible choice over petgraph's
//!   raw parallel-edge capability — a merge that silently doubled every
//!   shared edge would be surprising. Genuinely parallel edges that the
//!   *source* itself carries between the same pair are preserved only up
//!   to one per `(type, src, tgt)` after the merge, matching
//!   `add_connections`' within-batch consolidation.
//! - **Property schemas** merge through the same `upsert_node_type_metadata`
//!   / `type_schemas` extension path `add_nodes` uses.
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
use crate::graph::introspection::reporting::{ConnectionOperationReport, NodeOperationReport};
use crate::graph::mutation::maintain::{add_connections, add_nodes};
use crate::graph::schema::DirGraph;
use crate::graph::storage::GraphRead;
use std::collections::HashMap;

/// Combined report for an `extend` merge.
#[derive(Debug, Clone)]
pub struct ExtendReport {
    pub nodes_created: usize,
    pub nodes_updated: usize,
    pub nodes_skipped: usize,
    pub edges_created: usize,
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

/// Merge `source` into `target` in place. See module docs for full
/// semantics. `source` is read-only.
///
/// Errors when either graph is not the in-memory `Default` backend, or
/// when a routed `add_nodes` / `add_connections` call fails (the error
/// string is propagated unchanged).
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
    let mut node_groups: HashMap<String, NodeGroup> = HashMap::new();
    // (node_type, id, secondary_label_names) for the label-union pass.
    let mut label_carriers: Vec<(String, Value, Vec<String>)> = Vec::new();

    for idx in source.graph.node_indices() {
        let Some(node) = source.graph.node_weight(idx) else {
            continue;
        };
        let node_type = node.node_type_str(&source.interner).to_string();
        let id = node.id().into_owned();
        let title = node.title().into_owned();
        let props = node.properties_cloned(&source.interner);

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

    // ---- Pass 2: route each node group through add_nodes ----
    let mut report = ExtendReport {
        nodes_created: 0,
        nodes_updated: 0,
        nodes_skipped: 0,
        edges_created: 0,
        edges_skipped: 0,
        node_types_merged: node_groups.len(),
        connection_types_merged: 0,
        labels_unioned: 0,
        processing_time_ms: 0.0,
        errors: Vec::new(),
    };

    for (node_type, group) in node_groups {
        // Carry the source's id/title field aliases for a type the target
        // doesn't already have, so `MATCH (n {originalIdCol: ...})` keeps
        // resolving after the merge. For an already-present type the
        // target's own alias is authoritative — leave it.
        let target_has_type = target.type_indices.get(&node_type).is_some();
        if !target_has_type {
            if let Some(alias) = source.id_field_aliases.get(&node_type) {
                target
                    .id_field_aliases
                    .insert(node_type.clone(), alias.clone());
            }
            if let Some(alias) = source.title_field_aliases.get(&node_type) {
                target
                    .title_field_aliases
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

    // ---- Pass 3: union secondary labels onto target nodes ----
    //
    // Done after node merge so every carrier's target node exists. Uses
    // the id-index (rebuilt by add_nodes) for O(1) lookups.
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

    // ---- Pass 4: collect + route source edges ----
    //
    // Group key: (connection_type, source_type, target_type). The source
    // and target *node types* are resolved from the edge endpoints in the
    // source graph so add_connections can id-index-match them in target.
    let mut edge_groups: HashMap<(String, String, String), EdgeGroup> = HashMap::new();

    for edge_idx in source.graph.edge_indices() {
        let Some(edge) = source.graph.edge_weight(edge_idx) else {
            continue;
        };
        let Some((src_idx, tgt_idx)) = source.graph.edge_endpoints(edge_idx) else {
            continue;
        };
        let conn_type = edge.connection_type_str(&source.interner).to_string();
        let (Some(src_node), Some(tgt_node)) = (
            source.graph.node_weight(src_idx),
            source.graph.node_weight(tgt_idx),
        ) else {
            continue;
        };
        let source_type = src_node.node_type_str(&source.interner).to_string();
        let target_type = tgt_node.node_type_str(&source.interner).to_string();
        let src_id = src_node.id().into_owned();
        let tgt_id = tgt_node.id().into_owned();
        let props = edge.properties_cloned(&source.interner);

        let group = edge_groups
            .entry((conn_type.clone(), source_type.clone(), target_type.clone()))
            .or_insert_with(|| EdgeGroup::new(source_type, target_type));
        for k in props.keys() {
            group.note_column(k);
        }
        group.rows.push((src_id, tgt_id, props));
    }

    report.connection_types_merged = edge_groups
        .keys()
        .map(|(ct, _, _)| ct.clone())
        .collect::<std::collections::HashSet<_>>()
        .len();

    for ((conn_type, _, _), group) in edge_groups {
        let df = build_edge_dataframe(&group)?;
        let r: ConnectionOperationReport = add_connections(
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
        )?;
        report.edges_created += r.connections_created;
        report.edges_skipped += r.connections_skipped;
        report.errors.extend(r.errors);
    }

    report.processing_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(report)
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
