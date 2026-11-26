// src/graph/subgraph.rs
//! Subgraph extraction and selection expansion operations

use crate::datatypes::values::Value;
use crate::graph::core::fluent_filter::FluentFilter;
use crate::graph::schema::{
    ColumnarRow, CurrentSelection, DirGraph, EdgeData, InternedKey, NodeData, PropertyStorage,
    SchemaInstall,
};
use crate::graph::storage::column_store::ColumnStore;
use crate::graph::storage::{GraphRead, GraphWrite};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::Direction;
use rustc_hash::FxHashMap;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Expand the current selection by N hops using BFS.
///
/// This function takes all currently selected nodes and expands the selection
/// to include all nodes within `hops` distance from any selected node.
/// The expansion considers edges in both directions (undirected). Under
/// `valid_time` a hop is followed only when the relationship and the node it
/// reaches are visible.
pub fn expand_selection(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    hops: usize,
    valid_time: Option<&FluentFilter>,
) -> Result<(), String> {
    let valid_time = valid_time.filter(|filter| !filter.is_empty());
    let level_idx = selection.get_level_count().saturating_sub(1);
    let level = selection
        .get_level(level_idx)
        .ok_or_else(|| "No active selection level".to_string())?;

    // Start with current selection
    let mut frontier: HashSet<NodeIndex> = level.iter_node_indices().collect();
    let mut visited = frontier.clone();

    // BFS expansion for N hops
    let g = &graph.graph;
    for _ in 0..hops {
        let mut next_frontier = HashSet::new();

        for &node in &frontier {
            let Some(filter) = valid_time else {
                for neighbor in g.neighbors_undirected(node) {
                    if visited.insert(neighbor) {
                        next_frontier.insert(neighbor);
                    }
                }
                continue;
            };
            let outgoing = g
                .edges_directed(node, Direction::Outgoing)
                .map(|e| (e.target(), e));
            let incoming = g
                .edges_directed(node, Direction::Incoming)
                .map(|e| (e.source(), e));
            for (far, edge) in outgoing.chain(incoming) {
                let conn = edge.connection_type();
                if !visited.contains(&far)
                    && filter.admits_hop(graph, edge.id(), conn, edge.source(), far)?
                {
                    visited.insert(far);
                    next_frontier.insert(far);
                }
            }
        }

        // If no new nodes were found, stop early
        if next_frontier.is_empty() {
            break;
        }

        frontier = next_frontier;
    }

    // Update selection with expanded nodes
    let level_mut = selection
        .get_level_mut(level_idx)
        .ok_or_else(|| "Failed to get mutable selection level".to_string())?;

    level_mut.selections.clear();
    level_mut.add_selection(None, visited.into_iter().collect());

    valid_time.map_or(Ok(()), FluentFilter::finish)
}

/// Extract a subgraph containing only the selected nodes and edges between them.
///
/// This creates an independent copy of the graph containing only the nodes
/// in the current selection and all edges that connect those nodes — under
/// `valid_time`, those whose own interval is visible (the nodes are the
/// selection, which the chain's steps already filtered).
pub fn extract_subgraph(
    source: &DirGraph,
    selection: &CurrentSelection,
    valid_time: Option<&FluentFilter>,
) -> Result<DirGraph, String> {
    let level_idx = selection.get_level_count().saturating_sub(1);
    let level = selection
        .get_level(level_idx)
        .ok_or_else(|| "No active selection level".to_string())?;
    let nodes = level.get_all_nodes();
    let Some(filter) = valid_time.filter(|filter| !filter.is_empty()) else {
        return copy_induced_subgraph(source, &nodes, |_| true).map(|(graph, _)| graph);
    };
    let visible = |edge: EdgeIndex| {
        let (Some((from, _)), Some(weight)) = (
            source.graph.edge_endpoints(edge),
            source.graph.edge_weight(edge),
        ) else {
            return false;
        };
        filter
            .admits_edge(source, edge, weight.connection_type, from)
            .unwrap_or(false)
    };
    let (graph, _) = copy_induced_subgraph(source, &nodes, visible)?;
    filter.finish()?;
    Ok(graph)
}

/// A fresh graph holding `nodes` (copied in the order given) and every
/// relationship between two of them that `keep_edge` accepts, with the
/// source's secondary labels, ontology, schema and type metadata
/// ([`clone_subset_metadata`]). The map takes each copied
/// source node to its new index. Temporal declarations, embeddings, text and
/// vector indexes, spatial config and id indexes are not copied. The copy's
/// column stores hold only the copied nodes' rows.
pub(crate) fn copy_induced_subgraph(
    source: &DirGraph,
    nodes: &[NodeIndex],
    mut keep_edge: impl FnMut(EdgeIndex) -> bool,
) -> Result<(DirGraph, HashMap<NodeIndex, NodeIndex>), String> {
    let mut new_graph = DirGraph::new();
    // Before the inserts: they encode rows against the interner and type
    // schemas, and only add to the type metadata.
    clone_subset_metadata(&mut new_graph, source);

    let index_map = copy_nodes(source, nodes, &mut new_graph);

    // Copy the relationships between copied nodes. A node absent from
    // `index_map` was not copied, so neither are its relationships.
    let mut rel_types = HashSet::new();
    for &old_source_idx in nodes {
        let Some(&new_source) = index_map.get(&old_source_idx) else {
            continue;
        };
        for edge in source.graph.edges(old_source_idx) {
            let Some(&new_target) = index_map.get(&edge.target()) else {
                continue;
            };
            if !keep_edge(edge.id()) {
                continue;
            }
            // Edge properties are already interned.
            let mut properties = edge.weight().properties.clone();
            crate::graph::session::snapshot_property_values(
                &source.graph,
                properties.iter_mut().map(|(_, value)| value),
            );
            // `connection_type()`: each Disk `weight()` parks another copy.
            rel_types.insert(edge.connection_type());
            let edge_data = EdgeData::new_interned(edge.connection_type(), properties);
            GraphWrite::add_edge(&mut new_graph.graph, new_source, new_target, edge_data);
        }
    }

    retain_subset_types(&mut new_graph, &rel_types);

    // Carry secondary labels: buckets are keyed above the storage backend
    // (labels.rs), so the node copy does not move them — pre-2026-08-26 this
    // silently dropped every label from save_subset / extract_subgraph.
    // Copied through index_map (never re-derived: a manual label must survive
    // even where an ontology could not explain it), then sorted, since
    // `nodes` need not be in index order.
    for (label, bucket) in &source.secondary_label_index {
        let mut copied: Vec<NodeIndex> = bucket
            .iter()
            .filter_map(|old_idx| index_map.get(old_idx).copied())
            .collect();
        if copied.is_empty() {
            continue;
        }
        copied.sort_unstable();
        new_graph.secondary_label_index.insert(*label, copied);
        new_graph.has_secondary_labels = true;
    }
    // The declared ontology travels with the copy (type-level metadata,
    // like the schema install below).
    new_graph.ontology = Arc::clone(&source.ontology);
    // Derives the write gate with the closures: a copy that dropped it would
    // accept writes its source refuses.
    new_graph.rebuild_ontology_closures();

    // Copy schema definition if present. The subgraph's nodes are a subset of a
    // graph that already satisfied these constraints, so installing them cannot
    // find a duplicate the source did not have — but surface the error rather
    // than discard it, so a genuine inconsistency in the source is not laundered
    // into a silently unconstrained copy.
    if let Some(schema) = source.get_schema() {
        new_graph
            // The target is a fresh graph, so merge and replace coincide; merge
            // states the intent (install this schema) without also asserting
            // that everything unnamed should be withdrawn.
            .set_schema(schema.clone(), SchemaInstall::Merge)
            .map_err(|violation| format!("subgraph schema install failed: {violation}"))?;
    }

    Ok((new_graph, index_map))
}

/// One node type's share of a copy: the kept nodes' rows in the source
/// store, gathered in one pass per column when every kept node of the type
/// reads its id, title and properties from that store alone.
struct TypeCopy<'a> {
    key: InternedKey,
    store: Option<&'a ColumnStore>,
    rows: Vec<u32>,
    new_indices: Vec<NodeIndex>,
}

/// Copy `nodes` into `dest` in the order given; the map takes each copied
/// source node to its new index.
///
/// The copy's column stores hold only the copied rows and never share a
/// store or column `Arc` with the source: a mapped or disk column is
/// file-backed, and the first write into a shared one would clone every row
/// of the type. A type whose kept nodes all live in one store with nothing
/// behind its columns (no mmap base or overflow bag) is gathered column by
/// column ([`ColumnStore::gather_rows`]), keeping each column's stored kind;
/// any other type — a Disk store, a node holding its id, title or properties
/// inline — is copied node by node through [`copy_node`].
fn copy_nodes(
    source: &DirGraph,
    nodes: &[NodeIndex],
    dest: &mut DirGraph,
) -> HashMap<NodeIndex, NodeIndex> {
    let mut types: Vec<TypeCopy<'_>> = Vec::new();
    let mut slot_of: FxHashMap<InternedKey, usize> = FxHashMap::default();
    // Each kept node with its type's position in `types`. A gathered type's
    // nodes take the gathered rows in this order; one node that cannot be
    // gathered sends its whole type to the per-node copy.
    let mut plan: Vec<(NodeIndex, usize)> = Vec::with_capacity(nodes.len());
    for &old_idx in nodes {
        let Some(node) = source.graph.node_view(old_idx) else {
            continue;
        };
        let key = node.node_type();
        let slot = *slot_of.entry(key).or_insert_with(|| {
            let store = source
                .graph
                .column_store(key)
                .map(|store| &**store)
                .filter(|store| store.columns_cover_rows() && !store.has_overflow());
            types.push(TypeCopy {
                key,
                store,
                rows: Vec::new(),
                new_indices: Vec::new(),
            });
            types.len() - 1
        });
        let entry = &mut types[slot];
        let gathered_row = entry.store.and_then(|store| {
            let data = node.data();
            let (node_store, row) = node.column_row()?;
            (std::ptr::eq(node_store, store)
                && matches!(data.id_ref(), Value::Null)
                && matches!(data.title_ref(), Value::Null))
            .then_some(row)
        });
        match gathered_row {
            Some(row) => entry.rows.push(row),
            None => entry.store = None,
        }
        plan.push((old_idx, slot));
    }

    for entry in &types {
        if let Some(store) = entry.store {
            install_gathered_store(source, dest, entry.key, store, &entry.rows);
        }
    }

    let mut index_map: HashMap<NodeIndex, NodeIndex> = HashMap::with_capacity(plan.len());
    let mut next_row = vec![0u32; types.len()];
    for (old_idx, slot) in plan {
        let entry = &mut types[slot];
        let new_idx = if entry.store.is_some() {
            let row = next_row[slot];
            next_row[slot] += 1;
            let idx = GraphWrite::add_node(
                &mut dest.graph,
                NodeData {
                    ident: None,
                    node_type: entry.key,
                    properties: PropertyStorage::Columnar(ColumnarRow::new(row)),
                },
            );
            // A no-op on the heap backends; the per-node route's step, kept
            // so both routes build the same node.
            GraphWrite::update_row_id(&mut dest.graph, idx, row);
            idx
        } else {
            copy_node(source, dest, old_idx)
        };
        index_map.insert(old_idx, new_idx);
        entry.new_indices.push(new_idx);
    }
    for entry in types {
        dest.type_indices
            .entry_or_default(source.interner.resolve(entry.key).to_string())
            .extend(entry.new_indices);
    }
    index_map
}

/// Install `rows` of `store` as `node_type`'s store in `dest`, with the
/// bookkeeping the per-node insert does per row: the type schema holds the
/// store's keys, a key the type metadata lacks is registered from its first
/// non-null value, and node references in `Mixed` cells are snapshotted.
fn install_gathered_store(
    source: &DirGraph,
    dest: &mut DirGraph,
    node_type: InternedKey,
    store: &ColumnStore,
    rows: &[u32],
) {
    let mut gathered = store
        .gather_rows(rows)
        .expect("a store is gathered only when it has no mmap base or overflow bag");
    crate::graph::session::snapshot_property_values(
        &source.graph,
        gathered.heterogeneous_cells_mut(),
    );
    let type_name = source.interner.resolve(node_type);
    let keys: Vec<InternedKey> = gathered.schema().iter().map(|(_, key)| key).collect();
    dest.ensure_type_schema_keys(type_name, &keys);
    let known = dest.node_type_metadata.get(type_name);
    let missing: HashMap<String, String> = gathered
        .schema()
        .iter()
        .filter(|(_, key)| {
            known.is_none_or(|props| !props.contains_key(source.interner.resolve(*key)))
        })
        .filter_map(|(slot, key)| {
            let value =
                (0..gathered.row_count()).find_map(|row| gathered.get_by_slot(row, slot))?;
            Some((
                source.interner.resolve(key).to_string(),
                value.type_name().to_string(),
            ))
        })
        .collect();
    if !missing.is_empty() {
        dest.upsert_node_type_metadata(type_name, missing);
    }
    GraphWrite::install_column_store(&mut dest.graph, node_type, Arc::new(gathered));
}

/// Copy one node through the routed insert, read through `GraphRead` — the
/// route for a node [`copy_nodes`] cannot gather. `dest` has no schema
/// installed yet ([`copy_induced_subgraph`] installs it after the rows), so
/// an `auto_timestamp` type is not re-stamped: the copy keeps the source's
/// provenance values.
fn copy_node(source: &DirGraph, dest: &mut DirGraph, old_idx: NodeIndex) -> NodeIndex {
    let node = source
        .graph
        .node_view(old_idx)
        .expect("copy_nodes planned only nodes with a view");
    let node_type = node.node_type_str(&source.interner);
    let mut id = node.id().into_owned();
    let mut title = node.title().into_owned();
    let mut properties = node.property_pairs();
    crate::graph::session::snapshot_property_values(
        &source.graph,
        [&mut id, &mut title]
            .into_iter()
            .chain(properties.iter_mut().map(|(_, value)| value)),
    );
    let properties: HashMap<String, Value> = properties
        .into_iter()
        .map(|(key, value)| (source.interner.resolve(key).to_string(), value))
        .collect();
    dest.insert_node_routed(id, title, node_type, properties)
}

/// Copy the graph-level metadata a subset needs to be self-contained on
/// reload: the interner and type schemas the rows are encoded against, the
/// source's node and relationship type metadata (every property the type has,
/// not only those the kept rows carry — the schema check reads it), the
/// alias/tier maps `describe()` and property resolution read, and the caller's
/// user-schema version. Both subset copies — [`copy_induced_subgraph`] and the
/// streaming disk writer — take it from here and narrow it with
/// [`retain_subset_types`] once the copy is built, so both save the same
/// metadata.
///
/// The version carries because a subset of a graph at user-schema version N
/// is still at version N — only which rows came along changed — so a
/// migration runner pointed at the subset must not re-run migrations `1..=N`.
pub(crate) fn clone_subset_metadata(dest: &mut DirGraph, source: &DirGraph) {
    dest.interner = source.interner.clone();
    dest.type_schemas = source.type_schemas.clone();
    dest.node_type_metadata = source.node_type_metadata.clone();
    dest.connection_type_metadata = source.connection_type_metadata.clone();
    dest.id_field_aliases = source.id_field_aliases.clone();
    dest.title_field_aliases = source.title_field_aliases.clone();
    dest.parent_types = source.parent_types.clone();
    dest.user_schema_version = source.user_schema_version;
}

/// Narrow the metadata [`clone_subset_metadata`] copied to the types the
/// subset holds: the node types with a copied node and `rel_types`, the
/// relationship types of the copied edges. A type the subset has none of is
/// not listed by it.
pub(crate) fn retain_subset_types(
    dest: &mut DirGraph,
    rel_types: &HashSet<crate::graph::schema::InternedKey>,
) {
    let present = |node_type: &str| dest.type_indices.contains_key(node_type);
    let node_type_metadata = dest
        .node_type_metadata
        .iter()
        .filter(|(node_type, _)| present(node_type))
        .map(|(node_type, props)| (node_type.clone(), props.clone()))
        .collect();
    let id_field_aliases = dest
        .id_field_aliases
        .iter()
        .filter(|(node_type, _)| present(node_type))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let title_field_aliases = dest
        .title_field_aliases
        .iter()
        .filter(|(node_type, _)| present(node_type))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let parent_types = dest
        .parent_types
        .iter()
        .filter(|(node_type, _)| present(node_type))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let rel_names: HashSet<&str> = rel_types
        .iter()
        .map(|key| dest.interner.resolve(*key))
        .collect();
    let connection_type_metadata = dest
        .connection_type_metadata
        .iter()
        .filter(|(rel_type, _)| rel_names.contains(rel_type.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    dest.node_type_metadata = Arc::new(node_type_metadata);
    dest.id_field_aliases = Arc::new(id_field_aliases);
    dest.title_field_aliases = Arc::new(title_field_aliases);
    dest.parent_types = Arc::new(parent_types);
    dest.connection_type_metadata = Arc::new(connection_type_metadata);
}

/// Get summary statistics about the subgraph that would be extracted.
///
/// Returns the number of nodes and edges that would be included.
pub fn get_subgraph_stats(
    source: &DirGraph,
    selection: &CurrentSelection,
) -> Result<SubgraphStats, String> {
    let level_idx = selection.get_level_count().saturating_sub(1);
    let level = selection
        .get_level(level_idx)
        .ok_or_else(|| "No active selection level".to_string())?;

    let nodes = level.get_all_nodes();
    let node_set: HashSet<NodeIndex> = nodes.iter().copied().collect();

    // Count edges between selected nodes
    let mut edge_count = 0;
    let mut connection_types: HashMap<String, usize> = HashMap::new();
    let mut node_types: HashMap<String, usize> = HashMap::new();

    // Count node types
    for &node_idx in &nodes {
        if let Some(node) = source.graph.node_view(node_idx) {
            *node_types
                .entry(node.node_type_str(&source.interner).to_string())
                .or_insert(0) += 1;
        }
    }

    // Count edges and connection types
    for &source_idx in &nodes {
        for edge in source.graph.edges(source_idx) {
            if node_set.contains(&edge.target()) {
                edge_count += 1;
                let conn_type = source.interner.resolve(edge.connection_type());
                *connection_types.entry(conn_type.to_string()).or_insert(0) += 1;
            }
        }
    }

    Ok(SubgraphStats {
        node_count: nodes.len(),
        edge_count,
        node_types,
        connection_types,
    })
}

/// Statistics about a potential subgraph extraction
#[derive(Debug, Clone)]
pub struct SubgraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub node_types: HashMap<String, usize>,
    pub connection_types: HashMap<String, usize>,
}

#[cfg(test)]
#[path = "subgraph_tests.rs"]
mod tests;
