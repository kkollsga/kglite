use super::*;
use crate::datatypes::values::Value;
use crate::graph::schema::{DirGraph, EdgeData, NodeData};
use crate::graph::storage::GraphWrite;
use std::collections::HashMap;

/// Build a linear graph: A -> B -> C -> D -> E
fn build_chain_graph() -> (DirGraph, Vec<petgraph::graph::NodeIndex>) {
    let mut graph = DirGraph::new();
    let mut indices = Vec::new();
    for i in 0..5 {
        let node = NodeData::new(
            Value::Int64(i),
            Value::String(format!("Node_{}", i)),
            "Chain".to_string(),
            HashMap::new(),
            &mut graph.interner,
        );
        let idx = graph.graph.add_node(node);
        graph
            .type_indices
            .entry_or_default("Chain".to_string())
            .push(idx);
        indices.push(idx);
    }
    for i in 0..4 {
        let edge = EdgeData::new("NEXT".to_string(), HashMap::new(), &mut graph.interner);
        graph.graph.add_edge(indices[i], indices[i + 1], edge);
    }
    (graph, indices)
}

/// Build a triangle graph: A -- B -- C -- A
fn build_triangle_graph() -> (DirGraph, Vec<petgraph::graph::NodeIndex>) {
    let mut graph = DirGraph::new();
    let mut indices = Vec::new();
    for i in 0..3 {
        let node = NodeData::new(
            Value::Int64(i),
            Value::String(format!("N_{}", i)),
            "Node".to_string(),
            HashMap::new(),
            &mut graph.interner,
        );
        let idx = graph.graph.add_node(node);
        graph
            .type_indices
            .entry_or_default("Node".to_string())
            .push(idx);
        indices.push(idx);
    }
    // A->B, B->C, C->A
    let pairs = [(0, 1), (1, 2), (2, 0)];
    for (from, to) in pairs {
        let edge = EdgeData::new("LINK".to_string(), HashMap::new(), &mut graph.interner);
        graph.graph.add_edge(indices[from], indices[to], edge);
    }
    (graph, indices)
}

/// Build two disconnected components: {A, B} and {C, D}
fn build_disconnected_graph() -> (DirGraph, Vec<petgraph::graph::NodeIndex>) {
    let mut graph = DirGraph::new();
    let mut indices = Vec::new();
    for i in 0..4 {
        let node = NodeData::new(
            Value::Int64(i),
            Value::String(format!("N_{}", i)),
            "Node".to_string(),
            HashMap::new(),
            &mut graph.interner,
        );
        let idx = graph.graph.add_node(node);
        graph
            .type_indices
            .entry_or_default("Node".to_string())
            .push(idx);
        indices.push(idx);
    }
    // Component 1: A-B
    let edge_ab = EdgeData::new("LINK".to_string(), HashMap::new(), &mut graph.interner);
    graph.graph.add_edge(indices[0], indices[1], edge_ab);
    // Component 2: C-D
    let edge_cd = EdgeData::new("LINK".to_string(), HashMap::new(), &mut graph.interner);
    graph.graph.add_edge(indices[2], indices[3], edge_cd);
    (graph, indices)
}

// ========================================================================
// shortest_path
// ========================================================================

#[test]
fn test_shortest_path_adjacent() {
    let (graph, indices) = build_chain_graph();
    let result = shortest_path(&graph, indices[0], indices[1], None, None, None);
    assert!(result.is_some());
    let path = result.unwrap();
    assert_eq!(path.cost, 1);
    assert_eq!(path.path.len(), 2);
}

#[test]
fn test_shortest_path_multi_hop() {
    let (graph, indices) = build_chain_graph();
    let result = shortest_path(&graph, indices[0], indices[4], None, None, None);
    assert!(result.is_some());
    let path = result.unwrap();
    assert_eq!(path.cost, 4);
    assert_eq!(path.path.len(), 5);
}

#[test]
fn test_shortest_path_same_node() {
    let (graph, indices) = build_chain_graph();
    let result = shortest_path(&graph, indices[0], indices[0], None, None, None);
    assert!(result.is_some());
    let path = result.unwrap();
    assert_eq!(path.cost, 0);
    assert_eq!(path.path.len(), 1);
}

#[test]
fn test_shortest_path_not_found() {
    let (graph, indices) = build_disconnected_graph();
    let result = shortest_path(&graph, indices[0], indices[2], None, None, None);
    assert!(result.is_none());
}

#[test]
fn test_shortest_path_reverse_direction() {
    // BFS is undirected, so B -> A should find a path even though edge is A -> B
    let (graph, indices) = build_chain_graph();
    let result = shortest_path(&graph, indices[4], indices[0], None, None, None);
    assert!(result.is_some());
    assert_eq!(result.unwrap().cost, 4);
}

// ========================================================================
// all_paths
// ========================================================================

#[test]
fn test_all_paths_basic() {
    let (graph, indices) = build_chain_graph();
    let paths = all_paths(&graph, indices[0], indices[2], 5, None, None, None, None);
    assert!(!paths.is_empty());
    // There should be a path of length 2: A -> B -> C
    assert!(paths.iter().any(|p| p.len() == 3));
}

#[test]
fn test_all_paths_limited_hops() {
    let (graph, indices) = build_chain_graph();
    // With max_hops=1, can only reach adjacent node
    let paths = all_paths(&graph, indices[0], indices[2], 1, None, None, None, None);
    assert!(paths.is_empty()); // Can't reach C in 1 hop
}

#[test]
fn test_all_paths_triangle() {
    let (graph, indices) = build_triangle_graph();
    let paths = all_paths(&graph, indices[0], indices[2], 3, None, None, None, None);
    // Multiple paths possible in a triangle
    assert!(!paths.is_empty());
}

#[test]
fn test_all_paths_max_results() {
    let (graph, indices) = build_triangle_graph();
    // Triangle has multiple paths — limit to 1
    let paths = all_paths(&graph, indices[0], indices[2], 3, Some(1), None, None, None);
    assert_eq!(paths.len(), 1);
}

#[test]
fn test_all_paths_max_results_none_unlimited() {
    let (graph, indices) = build_triangle_graph();
    let limited = all_paths(&graph, indices[0], indices[2], 3, Some(1), None, None, None);
    let unlimited = all_paths(&graph, indices[0], indices[2], 3, None, None, None, None);
    assert!(unlimited.len() >= limited.len());
}

#[test]
fn test_shortest_path_connection_type_filter() {
    // Build graph with two edge types: A -NEXT-> B -NEXT-> C and A -SKIP-> C
    let mut graph = DirGraph::new();
    let mut indices = Vec::new();
    for i in 0..3 {
        let node = NodeData::new(
            Value::Int64(i),
            Value::String(format!("Node_{}", i)),
            "Test".to_string(),
            HashMap::new(),
            &mut graph.interner,
        );
        let idx = graph.graph.add_node(node);
        graph
            .type_indices
            .entry_or_default("Test".to_string())
            .push(idx);
        indices.push(idx);
    }
    let edge1 = EdgeData::new("NEXT".to_string(), HashMap::new(), &mut graph.interner);
    graph.graph.add_edge(indices[0], indices[1], edge1);
    let edge2 = EdgeData::new("NEXT".to_string(), HashMap::new(), &mut graph.interner);
    graph.graph.add_edge(indices[1], indices[2], edge2);
    let edge3 = EdgeData::new("SKIP".to_string(), HashMap::new(), &mut graph.interner);
    graph.graph.add_edge(indices[0], indices[2], edge3);

    // Without filter: shortest path is A->C via SKIP (1 hop)
    let result = shortest_path(&graph, indices[0], indices[2], None, None, None);
    assert_eq!(result.unwrap().cost, 1);

    // With NEXT filter: must go A->B->C (2 hops)
    let next_only = vec!["NEXT".to_string()];
    let result = shortest_path(&graph, indices[0], indices[2], Some(&next_only), None, None);
    assert_eq!(result.unwrap().cost, 2);

    // With SKIP filter: A->C (1 hop)
    let skip_only = vec!["SKIP".to_string()];
    let result = shortest_path(&graph, indices[0], indices[2], Some(&skip_only), None, None);
    assert_eq!(result.unwrap().cost, 1);
}

// ========================================================================
// connected_components / weakly_connected_components
// ========================================================================

#[test]
fn test_weakly_connected_components_connected() {
    let (graph, _) = build_chain_graph();
    let components = weakly_connected_components(&graph, None).unwrap();
    assert_eq!(components.len(), 1);
    assert_eq!(components[0].len(), 5);
}

#[test]
fn test_weakly_connected_components_disconnected() {
    let (graph, _) = build_disconnected_graph();
    let components = weakly_connected_components(&graph, None).unwrap();
    assert_eq!(components.len(), 2);
    // Sorted by size descending, both have 2 nodes
    assert_eq!(components[0].len(), 2);
    assert_eq!(components[1].len(), 2);
}

#[test]
fn test_weakly_connected_components_empty() {
    let graph = DirGraph::new();
    let components = weakly_connected_components(&graph, None).unwrap();
    assert!(components.is_empty());
}

// ========================================================================
// are_connected
// ========================================================================

#[test]
fn test_are_connected_true() {
    let (graph, indices) = build_chain_graph();
    assert!(are_connected(&graph, indices[0], indices[4]));
}

#[test]
fn test_are_connected_false() {
    let (graph, indices) = build_disconnected_graph();
    assert!(!are_connected(&graph, indices[0], indices[2]));
}

// ========================================================================
// node_degree
// ========================================================================

#[test]
fn test_node_degree() {
    let (graph, indices) = build_chain_graph();
    // First node: 1 outgoing edge
    assert_eq!(node_degree(&graph, indices[0]), 1);
    // Middle node: 1 outgoing + 1 incoming
    assert_eq!(node_degree(&graph, indices[2]), 2);
    // Last node: 1 incoming
    assert_eq!(node_degree(&graph, indices[4]), 1);
}

// ========================================================================
// Centrality algorithms
// ========================================================================

#[test]
fn test_betweenness_centrality_chain() {
    let (graph, indices) = build_chain_graph();
    let results = betweenness_centrality(&graph, false, None, None, None).unwrap();
    assert_eq!(results.len(), 5);
    // Middle node (index 2) should have highest betweenness in a chain
    let middle_score = results
        .iter()
        .find(|r| r.node_idx == indices[2])
        .unwrap()
        .score;
    let end_score = results
        .iter()
        .find(|r| r.node_idx == indices[0])
        .unwrap()
        .score;
    assert!(middle_score > end_score);
}

#[test]
fn test_betweenness_centrality_with_sampling() {
    let (graph, indices) = build_chain_graph();
    // With sample_size, stride-based sampling should still find the middle node
    let results = betweenness_centrality(&graph, false, Some(3), None, None).unwrap();
    assert_eq!(results.len(), 5);
    // Middle node should still have a non-zero betweenness score
    let middle_score = results
        .iter()
        .find(|r| r.node_idx == indices[2])
        .unwrap()
        .score;
    assert!(
        middle_score > 0.0,
        "Middle node should have non-zero betweenness with sampling"
    );
}

#[test]
fn test_degree_centrality() {
    let (graph, indices) = build_chain_graph();
    let results = degree_centrality(&graph, false, None, None).unwrap();
    assert_eq!(results.len(), 5);
    // Middle nodes should have degree 2, end nodes degree 1
    let middle = results.iter().find(|r| r.node_idx == indices[2]).unwrap();
    let end = results.iter().find(|r| r.node_idx == indices[0]).unwrap();
    assert_eq!(middle.score, 2.0);
    assert_eq!(end.score, 1.0);
}

#[test]
fn test_pagerank_basic() {
    let (graph, _) = build_triangle_graph();
    let results = pagerank(&graph, 0.85, 100, 1e-6, None, None).unwrap();
    assert_eq!(results.len(), 3);
    // All nodes in a symmetric triangle should have roughly equal PageRank
    let scores: Vec<f64> = results.iter().map(|r| r.score).collect();
    let diff = (scores[0] - scores[2]).abs();
    assert!(
        diff < 0.01,
        "Triangle nodes should have similar PageRank: {:?}",
        scores
    );
}

#[test]
fn test_closeness_centrality_chain() {
    let (graph, indices) = build_chain_graph();
    let results = closeness_centrality(&graph, false, None, None, None).unwrap();
    assert_eq!(results.len(), 5);
    // Middle node should have highest closeness
    let middle = results
        .iter()
        .find(|r| r.node_idx == indices[2])
        .unwrap()
        .score;
    let end = results
        .iter()
        .find(|r| r.node_idx == indices[0])
        .unwrap()
        .score;
    assert!(middle > end);
}

#[test]
fn test_pagerank_empty_graph() {
    let graph = DirGraph::new();
    let results = pagerank(&graph, 0.85, 100, 1e-6, None, None).unwrap();
    assert!(results.is_empty());
}

// ========================================================================
// get_node_info / get_path_connections
// ========================================================================

#[test]
fn test_get_node_info() {
    let (graph, indices) = build_chain_graph();
    let info = get_node_info(&graph, indices[0]);
    assert!(info.is_some());
    let info = info.unwrap();
    assert_eq!(info.node_type, "Chain");
    assert_eq!(info.title, "Node_0");
}

#[test]
fn test_get_path_connections() {
    let (graph, indices) = build_chain_graph();
    let path = vec![indices[0], indices[1], indices[2]];
    let connections = get_path_connections(&graph, &path);
    assert_eq!(connections.len(), 2);
    assert_eq!(connections[0], Some("NEXT".to_string()));
    assert_eq!(connections[1], Some("NEXT".to_string()));
}
