// src/graph/subgraph.rs
//! Subgraph extraction and selection expansion operations

use crate::graph::schema::{CurrentSelection, DirGraph, EdgeData};
use crate::graph::storage::{GraphRead, GraphWrite};
use petgraph::graph::NodeIndex;
use std::collections::{HashMap, HashSet};

/// Expand the current selection by N hops using BFS.
///
/// This function takes all currently selected nodes and expands the selection
/// to include all nodes within `hops` distance from any selected node.
/// The expansion considers edges in both directions (undirected).
pub fn expand_selection(
    graph: &DirGraph,
    selection: &mut CurrentSelection,
    hops: usize,
) -> Result<(), String> {
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
            // Add all neighbors (both directions)
            for neighbor in g.neighbors_undirected(node) {
                // Only add if not already visited
                if visited.insert(neighbor) {
                    next_frontier.insert(neighbor);
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

    Ok(())
}

/// Extract a subgraph containing only the selected nodes and edges between them.
///
/// This creates an independent copy of the graph containing only the nodes
/// in the current selection and all edges that connect those nodes.
pub fn extract_subgraph(
    source: &DirGraph,
    selection: &CurrentSelection,
) -> Result<DirGraph, String> {
    let level_idx = selection.get_level_count().saturating_sub(1);
    let level = selection
        .get_level(level_idx)
        .ok_or_else(|| "No active selection level".to_string())?;

    let nodes = level.get_all_nodes();
    let node_set: HashSet<NodeIndex> = nodes.iter().copied().collect();

    let mut new_graph = DirGraph::new();

    // Copy interner so the subgraph can resolve InternedKeys from compact storage
    new_graph.interner = source.interner.clone();

    // Copy type schemas so compact property storage works correctly
    new_graph.type_schemas = source.type_schemas.clone();

    // Map from old node indices to new node indices
    let mut index_map: HashMap<NodeIndex, NodeIndex> = HashMap::with_capacity(nodes.len());

    // Copy selected nodes
    for &old_idx in &nodes {
        if let Some(node_data) = source.graph.node_weight(old_idx) {
            // Add to new graph (single clone instead of double)
            let new_idx = GraphWrite::add_node(&mut new_graph.graph, node_data.clone());
            index_map.insert(old_idx, new_idx);

            // Update type indices
            new_graph
                .type_indices
                .entry_or_default(node_data.node_type_str(&source.interner).to_string())
                .push(new_idx);
        }
    }

    // Copy edges between selected nodes
    for &old_source_idx in &nodes {
        for edge in source.graph.edges(old_source_idx) {
            let old_target_idx = edge.target();

            // Only copy edge if target is also in selection
            if node_set.contains(&old_target_idx) {
                if let (Some(&new_source), Some(&new_target)) = (
                    index_map.get(&old_source_idx),
                    index_map.get(&old_target_idx),
                ) {
                    // Clone edge data (properties are already interned)
                    let edge_data = EdgeData::new_interned(
                        edge.weight().connection_type,
                        edge.weight().properties.clone(),
                    );
                    GraphWrite::add_edge(&mut new_graph.graph, new_source, new_target, edge_data);
                }
            }
        }
    }

    // Copy schema definition if present
    if let Some(schema) = source.get_schema() {
        new_graph.set_schema(schema.clone());
    }

    Ok(new_graph)
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
        if let Some(node) = source.graph.node_weight(node_idx) {
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
                let conn_type = edge.weight().connection_type_str(&source.interner);
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
