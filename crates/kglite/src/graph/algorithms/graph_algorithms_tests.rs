use super::*;
use crate::datatypes::values::Value;
use crate::graph::schema::{DirGraph, EdgeData, InternedKey, NodeData};
use crate::graph::storage::GraphWrite;
use std::collections::{HashMap, HashSet};

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
    let result = shortest_path(&graph, indices[0], indices[1], &PathOptions::default());
    assert!(result.is_some());
    let path = result.unwrap();
    assert_eq!(path.cost, 1);
    assert_eq!(path.path.len(), 2);
}

#[test]
fn test_shortest_path_multi_hop() {
    let (graph, indices) = build_chain_graph();
    let result = shortest_path(&graph, indices[0], indices[4], &PathOptions::default());
    assert!(result.is_some());
    let path = result.unwrap();
    assert_eq!(path.cost, 4);
    assert_eq!(path.path.len(), 5);
}

#[test]
fn test_shortest_path_same_node() {
    let (graph, indices) = build_chain_graph();
    let result = shortest_path(&graph, indices[0], indices[0], &PathOptions::default());
    assert!(result.is_some());
    let path = result.unwrap();
    assert_eq!(path.cost, 0);
    assert_eq!(path.path.len(), 1);
}

#[test]
fn test_shortest_path_not_found() {
    let (graph, indices) = build_disconnected_graph();
    let result = shortest_path(&graph, indices[0], indices[2], &PathOptions::default());
    assert!(result.is_none());
}

#[test]
fn test_shortest_path_reverse_direction() {
    // BFS is undirected, so B -> A should find a path even though edge is A -> B
    let (graph, indices) = build_chain_graph();
    let result = shortest_path(&graph, indices[4], indices[0], &PathOptions::default());
    assert!(result.is_some());
    assert_eq!(result.unwrap().cost, 4);
}

// ========================================================================
// all_paths
// ========================================================================

#[test]
fn test_all_paths_basic() {
    let (graph, indices) = build_chain_graph();
    let paths = all_paths(
        &graph,
        indices[0],
        indices[2],
        &AllPathsOptions::default().with_max_hops(5),
    );
    assert!(!paths.is_empty());
    // There should be a path of length 2: A -> B -> C
    assert!(paths.iter().any(|p| p.len() == 3));
}

#[test]
fn test_all_paths_limited_hops() {
    let (graph, indices) = build_chain_graph();
    // With max_hops=1, can only reach adjacent node
    let paths = all_paths(
        &graph,
        indices[0],
        indices[2],
        &AllPathsOptions::default().with_max_hops(1),
    );
    assert!(paths.is_empty()); // Can't reach C in 1 hop
}

#[test]
fn test_all_paths_triangle() {
    let (graph, indices) = build_triangle_graph();
    let paths = all_paths(
        &graph,
        indices[0],
        indices[2],
        &AllPathsOptions::default().with_max_hops(3),
    );
    // Multiple paths possible in a triangle
    assert!(!paths.is_empty());
}

#[test]
fn test_all_paths_max_results() {
    let (graph, indices) = build_triangle_graph();
    // Triangle has multiple paths — limit to 1
    let paths = all_paths(
        &graph,
        indices[0],
        indices[2],
        &AllPathsOptions::default()
            .with_max_hops(3)
            .with_max_results(1),
    );
    assert_eq!(paths.len(), 1);
}

#[test]
fn test_all_paths_max_results_none_unlimited() {
    let (graph, indices) = build_triangle_graph();
    let limited = all_paths(
        &graph,
        indices[0],
        indices[2],
        &AllPathsOptions::default()
            .with_max_hops(3)
            .with_max_results(1),
    );
    let unlimited = all_paths(
        &graph,
        indices[0],
        indices[2],
        &AllPathsOptions::default().with_max_hops(3),
    );
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
    let result = shortest_path(&graph, indices[0], indices[2], &PathOptions::default());
    assert_eq!(result.unwrap().cost, 1);

    // With NEXT filter: must go A->B->C (2 hops)
    let next_only = vec!["NEXT".to_string()];
    let result = shortest_path(
        &graph,
        indices[0],
        indices[2],
        &PathOptions::default().with_connection_types(&next_only),
    );
    assert_eq!(result.unwrap().cost, 2);

    // With SKIP filter: A->C (1 hop)
    let skip_only = vec!["SKIP".to_string()];
    let result = shortest_path(
        &graph,
        indices[0],
        indices[2],
        &PathOptions::default().with_connection_types(&skip_only),
    );
    assert_eq!(result.unwrap().cost, 1);
}

// ========================================================================
// connected_components / weakly_connected_components
// ========================================================================

#[test]
fn test_weakly_connected_components_connected() {
    let (graph, _) = build_chain_graph();
    let components =
        weakly_connected_components(&graph, crate::graph::algorithms::Interrupt::default())
            .unwrap();
    assert_eq!(components.len(), 1);
    assert_eq!(components[0].len(), 5);
}

#[test]
fn test_weakly_connected_components_disconnected() {
    let (graph, _) = build_disconnected_graph();
    let components =
        weakly_connected_components(&graph, crate::graph::algorithms::Interrupt::default())
            .unwrap();
    assert_eq!(components.len(), 2);
    // Sorted by size descending, both have 2 nodes
    assert_eq!(components[0].len(), 2);
    assert_eq!(components[1].len(), 2);
}

#[test]
fn test_weakly_connected_components_empty() {
    let graph = DirGraph::new();
    let components =
        weakly_connected_components(&graph, crate::graph::algorithms::Interrupt::default())
            .unwrap();
    assert!(components.is_empty());
}

/// Two Person pairs joined only via a shared Company:
///   P0-[:KNOWS]-P1, P2-[:KNOWS]-P3, and P0,P2 -[:WORKS_AT]-> C0.
/// Whole-graph WCC sees one component (WORKS_AT bridges everything);
/// scoping to {node_type: Person, relationship: KNOWS} must split into the
/// two KNOWS pairs and exclude the Company entirely.
fn build_two_type_graph() -> DirGraph {
    let mut graph = DirGraph::new();
    let mut persons = Vec::new();
    for i in 0..4 {
        let node = NodeData::new(
            Value::Int64(i),
            Value::String(format!("P{i}")),
            "Person".to_string(),
            HashMap::new(),
            &mut graph.interner,
        );
        let idx = graph.graph.add_node(node);
        graph
            .type_indices
            .entry_or_default("Person".to_string())
            .push(idx);
        persons.push(idx);
    }
    let company = NodeData::new(
        Value::Int64(100),
        Value::String("C0".to_string()),
        "Company".to_string(),
        HashMap::new(),
        &mut graph.interner,
    );
    let c0 = graph.graph.add_node(company);
    graph
        .type_indices
        .entry_or_default("Company".to_string())
        .push(c0);

    let knows =
        |g: &mut DirGraph| EdgeData::new("KNOWS".to_string(), HashMap::new(), &mut g.interner);
    let e = knows(&mut graph);
    graph.graph.add_edge(persons[0], persons[1], e);
    let e = knows(&mut graph);
    graph.graph.add_edge(persons[2], persons[3], e);
    let e = EdgeData::new("WORKS_AT".to_string(), HashMap::new(), &mut graph.interner);
    graph.graph.add_edge(persons[0], c0, e);
    let e = EdgeData::new("WORKS_AT".to_string(), HashMap::new(), &mut graph.interner);
    graph.graph.add_edge(persons[2], c0, e);
    graph
}

#[test]
fn test_wcc_unscoped_bridges_via_other_edge_type() {
    let graph = build_two_type_graph();
    let components =
        weakly_connected_components(&graph, crate::graph::algorithms::Interrupt::default())
            .unwrap();
    // WORKS_AT connects both Person pairs through C0 → one component of 5.
    assert_eq!(components.len(), 1);
    assert_eq!(components[0].len(), 5);
}

#[test]
fn test_wcc_scoped_to_node_type_and_relationship() {
    let graph = build_two_type_graph();
    let node_types = ["Person".to_string()];
    let rel_types = [InternedKey::from_str("KNOWS")];
    let components = weakly_connected_components_scoped(
        &graph,
        Some(&node_types),
        Some(&rel_types),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    // Two KNOWS pairs, Company excluded → two components of 2.
    assert_eq!(components.len(), 2);
    assert_eq!(components[0].len(), 2);
    assert_eq!(components[1].len(), 2);
}

// ========================================================================
// coreness (k-core) + clustering coefficient
// ========================================================================

fn intersection_count_gt_hashset_oracle(a: &[u32], b: &[u32], gt: u32) -> u64 {
    let eligible: HashSet<u32> = a.iter().copied().filter(|value| *value > gt).collect();
    b.iter()
        .filter(|value| **value > gt && eligible.contains(value))
        .count() as u64
}

#[test]
fn intersection_count_gt_matches_hashset_at_boundaries() {
    let cases: &[(&[u32], &[u32], u32)] = &[
        (&[], &[], 0),
        (&[], &[1], 0),
        (&[0, 1, 2], &[0, 1, 2], 0),
        (&[0, 1, 2], &[0, 1, 2], 1),
        (&[0, 1, 2], &[0, 1, 2], 2),
        (&[0, 2, 4, 6, 8], &[1, 2, 3, 4, 7, 8], 3),
        (&[1, 3, 5, 7, 9], &[0, 1, 4, 5, 8, 9], 0),
        (
            &[u32::MAX - 1, u32::MAX],
            &[u32::MAX - 1, u32::MAX],
            u32::MAX - 1,
        ),
        (&[u32::MAX], &[u32::MAX], u32::MAX),
    ];

    for &(a, b, gt) in cases {
        assert_eq!(
            intersection_count_gt(a, b, gt),
            intersection_count_gt_hashset_oracle(a, b, gt),
            "a={a:?}, b={b:?}, gt={gt}"
        );
    }
}

#[test]
fn test_coreness_triangle_all_two() {
    let (graph, _) = build_triangle_graph();
    let mut scores = coreness_scoped(
        &graph,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    scores.sort_by_key(|(n, _)| n.index());
    assert_eq!(scores.len(), 3);
    assert!(
        scores.iter().all(|(_, c)| *c == 2),
        "triangle coreness should all be 2: {scores:?}"
    );
}

#[test]
fn test_coreness_chain_all_one() {
    let (graph, _) = build_chain_graph();
    let scores = coreness_scoped(
        &graph,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    assert_eq!(scores.len(), 5);
    assert!(
        scores.iter().all(|(_, c)| *c == 1),
        "path coreness should all be 1: {scores:?}"
    );
}

#[test]
fn test_coreness_streaming_pre_cancel_drops_partial_degree_buffer_safely() {
    static CANCELLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

    let (graph, _) = build_triangle_graph();
    let result = coreness_scoped_streaming(
        &graph,
        None,
        None,
        crate::graph::algorithms::Interrupt {
            deadline: None,
            cancel: Some(&CANCELLED),
        },
    );

    assert_eq!(result.unwrap_err(), algorithm_timeout_err());
}

#[test]
fn test_clustering_triangle_all_one() {
    let (graph, _) = build_triangle_graph();
    let scores = clustering_coefficient_scoped(
        &graph,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    assert_eq!(scores.len(), 3);
    assert!(
        scores.iter().all(|(_, c)| (*c - 1.0).abs() < 1e-9),
        "triangle clustering coefficient should all be 1.0: {scores:?}"
    );
}

#[test]
fn test_clustering_chain_all_zero() {
    let (graph, _) = build_chain_graph();
    let scores = clustering_coefficient_scoped(
        &graph,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    // A path has no triangles → every coefficient is 0.
    assert!(
        scores.iter().all(|(_, c)| *c == 0.0),
        "path clustering should all be 0: {scores:?}"
    );
}

#[test]
fn test_coreness_scoped_to_relationship() {
    // Person/KNOWS subgraph is two disjoint single edges → coreness 1 each;
    // the bridging Company (WORKS_AT) must be excluded.
    let graph = build_two_type_graph();
    let node_types = ["Person".to_string()];
    let rel_types = [InternedKey::from_str("KNOWS")];
    let scores = coreness_scoped(
        &graph,
        Some(&node_types),
        Some(&rel_types),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    assert_eq!(scores.len(), 4); // 4 Persons, Company excluded
    assert!(scores.iter().all(|(_, c)| *c == 1));
}

#[test]
fn test_wcc_scoped_relationship_only_induces_subgraph() {
    let graph = build_two_type_graph();
    // No node_type → universe is nodes incident to a KNOWS edge (the 4 Persons).
    let rel_types = [InternedKey::from_str("KNOWS")];
    let components = weakly_connected_components_scoped(
        &graph,
        None,
        Some(&rel_types),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    assert_eq!(components.len(), 2);
    assert_eq!(components.iter().map(|c| c.len()).sum::<usize>(), 4);
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
    let results =
        betweenness_centrality(&graph, &CentralityOptions::default().with_normalized(false))
            .unwrap();
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
    let results = betweenness_centrality(
        &graph,
        &CentralityOptions::default()
            .with_normalized(false)
            .with_sample_size(3),
    )
    .unwrap();
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
fn test_betweenness_rejects_zero_sample_size() {
    let (graph, _) = build_chain_graph();
    let err = betweenness_centrality(&graph, &CentralityOptions::default().with_sample_size(0))
        .unwrap_err();
    assert_eq!(err, "sample_size must be greater than 0");
}

#[test]
fn test_degree_centrality() {
    let (graph, indices) = build_chain_graph();
    let results = degree_centrality(
        &graph,
        &DegreeCentralityOptions::default().with_normalized(false),
    )
    .unwrap();
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
    let results = pagerank(&graph, &PagerankOptions::default()).unwrap();
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
    let results =
        closeness_centrality(&graph, &CentralityOptions::default().with_normalized(false)).unwrap();
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
fn test_closeness_rejects_zero_sample_size() {
    let (graph, _) = build_chain_graph();
    let err = closeness_centrality(&graph, &CentralityOptions::default().with_sample_size(0))
        .unwrap_err();
    assert_eq!(err, "sample_size must be greater than 0");
}

#[test]
fn test_centrality_scope_restricts_nodes_and_edges() {
    // Chain 0-1-2-3-4. Scope to {1,2,3}: edges 0-1 and 3-4 leave scope and are
    // dropped, leaving the sub-chain 1-2-3.
    let (graph, indices) = build_chain_graph();
    let scope: std::collections::HashSet<_> =
        [indices[1], indices[2], indices[3]].into_iter().collect();

    let deg = degree_centrality(
        &graph,
        &DegreeCentralityOptions::default()
            .with_normalized(false)
            .with_scope(&scope),
    )
    .unwrap();
    assert_eq!(deg.len(), 3, "only scoped nodes returned");
    let score_of = |idx| deg.iter().find(|r| r.node_idx == idx).unwrap().score;
    // Within the sub-chain, the middle node (2) has degree 2; the ends (1,3) have 1.
    assert_eq!(score_of(indices[2]), 2.0);
    assert_eq!(score_of(indices[1]), 1.0);
    assert_eq!(score_of(indices[3]), 1.0);

    // Excluded nodes never appear in any scoped algorithm's output.
    let pr = pagerank(&graph, &PagerankOptions::default().with_scope(&scope)).unwrap();
    let pr_nodes: std::collections::HashSet<_> = pr.iter().map(|r| r.node_idx).collect();
    assert_eq!(pr_nodes, scope);
}

#[test]
fn test_pagerank_empty_graph() {
    let graph = DirGraph::new();
    let results = pagerank(&graph, &PagerankOptions::default()).unwrap();
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

// ========================================================================
// multilevel Louvain + hierarchy
// ========================================================================

/// Two triangles {A,B,C} and {D,E,F}, each fully connected, joined by a single
/// bridge edge C--D. Classic community-structure fixture.
fn build_two_triangle_bridge() -> (DirGraph, Vec<petgraph::graph::NodeIndex>) {
    let mut graph = DirGraph::new();
    let mut indices = Vec::new();
    for i in 0..6 {
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
    // triangle 0-1-2, triangle 3-4-5, bridge 2-3
    let pairs = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)];
    for (from, to) in pairs {
        let edge = EdgeData::new("LINK".to_string(), HashMap::new(), &mut graph.interner);
        graph.graph.add_edge(indices[from], indices[to], edge);
    }
    (graph, indices)
}

#[test]
fn test_modularity_uses_complete_filtered_scoped_weighted_graph() {
    let mut graph = DirGraph::new();
    let nodes: Vec<_> = (0..4)
        .map(|i| {
            graph.graph.add_node(NodeData::new(
                Value::Int64(i),
                Value::String(format!("M_{i}")),
                "Node".to_string(),
                HashMap::new(),
                &mut graph.interner,
            ))
        })
        .collect();
    // Scoped LINK edges: two parallel internal edges (weights 2 + 1), one
    // cross-community edge (1), and one self-loop (4). The OTHER edge and the
    // LINK to out-of-scope node 3 are deliberate high-weight noise.
    for (source, target, edge_type, weight) in [
        (0, 1, "LINK", 2.0),
        (0, 1, "LINK", 1.0),
        (1, 2, "LINK", 1.0),
        (2, 2, "LINK", 4.0),
        (0, 2, "OTHER", 100.0),
        (0, 3, "LINK", 100.0),
    ] {
        let properties = HashMap::from([("weight".to_string(), Value::Float64(weight))]);
        let edge = EdgeData::new(edge_type.to_string(), properties, &mut graph.interner);
        graph.graph.add_edge(nodes[source], nodes[target], edge);
    }

    let community = [0, 0, 1, 0];
    let node_exists = [true, true, true, false];
    let connection_types = ["LINK".to_string()];
    let modularity = compute_modularity(
        &graph,
        &community,
        &node_exists,
        Some("weight"),
        Some(&connection_types),
    );

    // m=8; internal=(3,4); community degree sums=(7,9).
    let expected = 3.0 / 8.0 - (7.0_f64 / 16.0).powi(2) + 4.0 / 8.0 - (9.0_f64 / 16.0).powi(2);
    assert!((modularity - expected).abs() < 1e-15);
}

/// Two disconnected LINK triangles plus edges that the public community
/// options must exclude from both partitioning and the reported modularity.
fn build_filtered_scoped_two_triangles() -> (DirGraph, Vec<petgraph::graph::NodeIndex>) {
    let mut graph = DirGraph::new();
    let nodes: Vec<_> = (0..7)
        .map(|i| {
            graph.graph.add_node(NodeData::new(
                Value::Int64(i),
                Value::String(format!("F_{i}")),
                "Node".to_string(),
                HashMap::new(),
                &mut graph.interner,
            ))
        })
        .collect();

    for (source, target, edge_type, weight) in [
        (0, 1, "LINK", 1.0),
        (1, 2, "LINK", 1.0),
        (0, 2, "LINK", 1.0),
        (3, 4, "LINK", 1.0),
        (4, 5, "LINK", 1.0),
        (3, 5, "LINK", 1.0),
        // Excluded by connection type despite lying inside the node scope.
        (2, 3, "OTHER", 100.0),
        // Excluded because node 6 is outside the requested scope.
        (0, 6, "LINK", 100.0),
    ] {
        let properties = HashMap::from([("weight".to_string(), Value::Float64(weight))]);
        let edge = EdgeData::new(edge_type.to_string(), properties, &mut graph.interner);
        graph.graph.add_edge(nodes[source], nodes[target], edge);
    }
    (graph, nodes)
}

#[test]
fn test_public_community_modularity_honors_filters_and_scope() {
    let (graph, nodes) = build_filtered_scoped_two_triangles();
    let scope: NodeScope = nodes[..6].iter().copied().collect();
    let connection_types = ["LINK".to_string()];
    let community_options = CommunityOptions::default()
        .with_weight_property("weight")
        .with_connection_types(&connection_types)
        .with_scope(&scope);

    let louvain = louvain_communities(&graph, &community_options).unwrap();
    let leiden = leiden_communities(&graph, &community_options).unwrap();
    let label_options = LabelPropagationOptions::default()
        .with_connection_types(&connection_types)
        .with_scope(&scope);
    let label_propagation = label_propagation(&graph, &label_options).unwrap();

    // Each triangle has internal weight 3 and degree sum 6, while total
    // included weight is 6: 2 * (3/6 - (6/12)^2) = 0.5.
    for (name, result) in [
        ("Louvain", louvain),
        ("Leiden", leiden),
        ("label propagation", label_propagation),
    ] {
        assert_eq!(result.assignments.len(), 6, "{name} scope");
        assert_eq!(result.num_communities, 2, "{name} partition");
        assert!(
            (result.modularity - 0.5).abs() < 1e-15,
            "{name} modularity: {}",
            result.modularity
        );
    }
}

fn community_of(result: &CommunityResult, idx: petgraph::graph::NodeIndex) -> usize {
    result
        .assignments
        .iter()
        .find(|a| a.node_idx == idx)
        .map(|a| a.community_id)
        .expect("node assigned")
}

#[test]
fn test_louvain_multilevel_two_communities() {
    let (graph, ix) = build_two_triangle_bridge();
    let r = louvain_communities(&graph, &CommunityOptions::default()).unwrap();
    assert_eq!(r.num_communities, 2, "two triangles → two communities");
    assert!(
        r.modularity > 0.0,
        "positive modularity, got {}",
        r.modularity
    );
    // triangle members share a community, distinct across triangles
    assert_eq!(community_of(&r, ix[0]), community_of(&r, ix[1]));
    assert_eq!(community_of(&r, ix[0]), community_of(&r, ix[2]));
    assert_eq!(community_of(&r, ix[3]), community_of(&r, ix[4]));
    assert_eq!(community_of(&r, ix[3]), community_of(&r, ix[5]));
    assert_ne!(community_of(&r, ix[0]), community_of(&r, ix[3]));
}

#[test]
fn test_louvain_exposes_hierarchy_levels() {
    let (graph, _) = build_two_triangle_bridge();
    let r = louvain_communities(&graph, &CommunityOptions::default()).unwrap();
    assert!(!r.levels.is_empty(), "hierarchy levels present");
    // last level == flat assignments (best partition)
    assert_eq!(r.levels.last().unwrap().len(), r.assignments.len());
    // every level assigns all 6 nodes
    for level in &r.levels {
        assert_eq!(level.len(), 6);
    }
}

#[test]
fn test_louvain_deterministic() {
    let (graph, _) = build_two_triangle_bridge();
    let a = louvain_communities(&graph, &CommunityOptions::default()).unwrap();
    let b = louvain_communities(&graph, &CommunityOptions::default()).unwrap();
    assert_eq!(a.num_communities, b.num_communities);
    let ca: Vec<usize> = a.assignments.iter().map(|x| x.community_id).collect();
    let cb: Vec<usize> = b.assignments.iter().map(|x| x.community_id).collect();
    assert_eq!(ca, cb, "deterministic across runs");
}

#[test]
fn test_louvain_empty_and_isolated() {
    // empty
    let g = DirGraph::new();
    let r = louvain_communities(&g, &CommunityOptions::default()).unwrap();
    assert_eq!(r.num_communities, 0);
    assert!(r.levels.is_empty());
    // isolated nodes (no edges) → each its own community, modularity 0
    let mut g3 = DirGraph::new();
    for i in 0..3 {
        let node = NodeData::new(
            Value::Int64(i),
            Value::String(format!("I_{}", i)),
            "Node".to_string(),
            HashMap::new(),
            &mut g3.interner,
        );
        let idx = g3.graph.add_node(node);
        g3.type_indices
            .entry_or_default("Node".to_string())
            .push(idx);
    }
    let r3 = louvain_communities(&g3, &CommunityOptions::default()).unwrap();
    assert_eq!(r3.num_communities, 3);
    assert_eq!(r3.modularity, 0.0);
}

// ========================================================================
// Leiden
// ========================================================================

/// Assert every multi-node community in `result` is a connected subgraph
/// (Leiden's well-connectedness guarantee). Rebuilds an undirected adjacency
/// from the graph and BFSes within each community.
fn assert_all_communities_connected(graph: &DirGraph, result: &CommunityResult) {
    use std::collections::{HashMap, HashSet, VecDeque};

    let mut adj: HashMap<petgraph::graph::NodeIndex, Vec<petgraph::graph::NodeIndex>> =
        HashMap::new();
    for e in graph.graph.edge_references() {
        adj.entry(e.source()).or_default().push(e.target());
        adj.entry(e.target()).or_default().push(e.source());
    }
    let mut groups: HashMap<usize, Vec<petgraph::graph::NodeIndex>> = HashMap::new();
    for a in &result.assignments {
        groups.entry(a.community_id).or_default().push(a.node_idx);
    }
    for (cid, members) in &groups {
        if members.len() <= 1 {
            continue;
        }
        let set: HashSet<_> = members.iter().copied().collect();
        let mut seen: HashSet<_> = HashSet::new();
        let mut q = VecDeque::new();
        q.push_back(members[0]);
        seen.insert(members[0]);
        while let Some(u) = q.pop_front() {
            if let Some(ns) = adj.get(&u) {
                for &v in ns {
                    if set.contains(&v) && seen.insert(v) {
                        q.push_back(v);
                    }
                }
            }
        }
        assert_eq!(
            seen.len(),
            members.len(),
            "community {cid} is disconnected (Leiden must not produce that)"
        );
    }
}

#[test]
fn test_leiden_two_communities() {
    let (graph, ix) = build_two_triangle_bridge();
    let r = leiden_communities(&graph, &CommunityOptions::default()).unwrap();
    assert_eq!(r.num_communities, 2);
    assert!(r.modularity > 0.0);
    assert_eq!(community_of(&r, ix[0]), community_of(&r, ix[1]));
    assert_eq!(community_of(&r, ix[3]), community_of(&r, ix[5]));
    assert_ne!(community_of(&r, ix[0]), community_of(&r, ix[3]));
}

#[test]
fn test_leiden_communities_well_connected() {
    let (graph, _) = build_two_triangle_bridge();
    let r = leiden_communities(&graph, &CommunityOptions::default()).unwrap();
    assert_all_communities_connected(&graph, &r);

    // also on a chain and a triangle — the invariant must always hold
    let (chain, _) = build_chain_graph();
    let rc = leiden_communities(&chain, &CommunityOptions::default()).unwrap();
    assert_all_communities_connected(&chain, &rc);
}

#[test]
fn test_leiden_deterministic() {
    let (graph, _) = build_two_triangle_bridge();
    let a = leiden_communities(&graph, &CommunityOptions::default()).unwrap();
    let b = leiden_communities(&graph, &CommunityOptions::default()).unwrap();
    let ca: Vec<usize> = a.assignments.iter().map(|x| x.community_id).collect();
    let cb: Vec<usize> = b.assignments.iter().map(|x| x.community_id).collect();
    assert_eq!(ca, cb);
}

#[test]
fn test_leiden_hierarchy_and_modularity_vs_louvain() {
    let (graph, _) = build_two_triangle_bridge();
    let lei = leiden_communities(&graph, &CommunityOptions::default()).unwrap();
    let lou = louvain_communities(&graph, &CommunityOptions::default()).unwrap();
    assert!(!lei.levels.is_empty());
    assert_eq!(lei.levels.last().unwrap().len(), lei.assignments.len());
    // Leiden modularity should be competitive with Louvain (≥ within fp slack).
    assert!(
        lei.modularity >= lou.modularity - 1e-9,
        "leiden {} should be >= louvain {}",
        lei.modularity,
        lou.modularity
    );
}

#[test]
fn test_leiden_empty_and_isolated() {
    let g = DirGraph::new();
    let r = leiden_communities(&g, &CommunityOptions::default()).unwrap();
    assert_eq!(r.num_communities, 0);
    assert!(r.levels.is_empty());
}

// ============================================================================
// EdgeDir: direction-aware expansion, the directed scoped adjacency, and the
// batch API's restricted-universe contract (S2).
// ============================================================================

/// Mirrors `tests/test_shortest_path_python_parity.py`'s fixture:
///
/// ```text
/// P0 -KNOWS-> P1 <-KNOWS- P3        P2 -LIVES_IN-> C(4) <-LIVES_IN- P3
/// P1 -KNOWS-> P4 -KNOWS-> P3
/// ```
///
/// So P0→P3 is 2 hops undirected (through the *backwards* KNOWS P3→P1) but 3
/// hops following outgoing edges, and P2→P3 exists only through the City.
fn build_direction_fixture() -> (DirGraph, Vec<petgraph::graph::NodeIndex>) {
    let mut graph = DirGraph::new();
    let mut idx = Vec::new();
    for i in 0..5 {
        let node = NodeData::new(
            Value::Int64(i),
            Value::String(format!("P{}", i)),
            "Person".to_string(),
            HashMap::new(),
            &mut graph.interner,
        );
        let n = graph.graph.add_node(node);
        graph
            .type_indices
            .entry_or_default("Person".to_string())
            .push(n);
        idx.push(n);
    }
    let city = NodeData::new(
        Value::Int64(10),
        Value::String("Oslo".to_string()),
        "City".to_string(),
        HashMap::new(),
        &mut graph.interner,
    );
    let city_idx = graph.graph.add_node(city);
    graph
        .type_indices
        .entry_or_default("City".to_string())
        .push(city_idx);
    idx.push(city_idx);

    for (from, to) in [(0usize, 1usize), (3, 1), (1, 4), (4, 3)] {
        let edge = EdgeData::new("KNOWS".to_string(), HashMap::new(), &mut graph.interner);
        graph.graph.add_edge(idx[from], idx[to], edge);
    }
    for person in [2usize, 3] {
        let edge = EdgeData::new("LIVES_IN".to_string(), HashMap::new(), &mut graph.interner);
        graph.graph.add_edge(idx[person], city_idx, edge);
    }
    (graph, idx)
}

#[test]
fn test_edge_dir_default_is_undirected() {
    assert_eq!(EdgeDir::default(), EdgeDir::Any);
    assert_eq!(PathOptions::default().direction, EdgeDir::Any);
    assert_eq!(AllPathsOptions::default().direction, EdgeDir::Any);
}

#[test]
fn test_shortest_path_direction_changes_the_answer() {
    let (graph, idx) = build_direction_fixture();
    let undirected = shortest_path(&graph, idx[0], idx[3], &PathOptions::default()).unwrap();
    assert_eq!(undirected.cost, 2);
    assert_eq!(undirected.path, vec![idx[0], idx[1], idx[3]]);

    let outgoing = shortest_path(
        &graph,
        idx[0],
        idx[3],
        &PathOptions::default().with_direction(EdgeDir::Outgoing),
    )
    .unwrap();
    assert_eq!(outgoing.cost, 3);
    assert_eq!(outgoing.path, vec![idx[0], idx[1], idx[4], idx[3]]);

    // Nothing points at P0, so an incoming-only walk leaves it immediately.
    assert!(shortest_path(
        &graph,
        idx[0],
        idx[3],
        &PathOptions::default().with_direction(EdgeDir::Incoming),
    )
    .is_none());
}

#[test]
fn test_incoming_expansion_is_the_mirror_of_outgoing() {
    let (graph, idx) = build_direction_fixture();
    let out = shortest_path(
        &graph,
        idx[0],
        idx[3],
        &PathOptions::default().with_direction(EdgeDir::Outgoing),
    )
    .unwrap();
    let inc = shortest_path(
        &graph,
        idx[3],
        idx[0],
        &PathOptions::default().with_direction(EdgeDir::Incoming),
    )
    .unwrap();
    let mut reversed = inc.path.clone();
    reversed.reverse();
    assert_eq!(reversed, out.path);
    assert_eq!(inc.cost, out.cost);
}

#[test]
fn test_shortest_path_directed_delegates_to_outgoing() {
    let (graph, idx) = build_direction_fixture();
    let via_delegate = shortest_path_directed(&graph, idx[0], idx[3], &PathOptions::default());
    let via_option = shortest_path(
        &graph,
        idx[0],
        idx[3],
        &PathOptions::default().with_direction(EdgeDir::Outgoing),
    );
    assert_eq!(
        via_delegate.map(|r| r.path),
        via_option.map(|r| r.path),
        "shortest_path_directed must be shortest_path(EdgeDir::Outgoing)"
    );
}

#[test]
fn test_cost_with_matches_path_with_across_the_matrix() {
    let (graph, idx) = build_direction_fixture();
    let knows = ["KNOWS".to_string()];
    for direction in [EdgeDir::Any, EdgeDir::Outgoing, EdgeDir::Incoming] {
        for conn in [None, Some(&knows[..])] {
            for (s, t) in [(0usize, 3usize), (3, 0), (2, 3), (0, 2)] {
                let mut opts = PathOptions::default().with_direction(direction);
                if let Some(c) = conn {
                    opts = opts.with_connection_types(c);
                }
                let by_path = shortest_path(&graph, idx[s], idx[t], &opts).map(|r| r.cost);
                let by_cost = shortest_path_cost_with(&graph, idx[s], idx[t], &opts);
                let by_batch = shortest_path_cost_batch_with(&graph, &[(idx[s], idx[t])], &opts)[0];
                let connected = are_connected_with(&graph, idx[s], idx[t], &opts);
                assert_eq!(by_path, by_cost, "{:?} {:?} {}->{}", direction, conn, s, t);
                assert_eq!(by_path, by_batch, "{:?} {:?} {}->{}", direction, conn, s, t);
                assert_eq!(
                    by_path.is_some(),
                    connected,
                    "{:?} {:?} {}->{}",
                    direction,
                    conn,
                    s,
                    t
                );
            }
        }
    }
}

#[test]
fn test_cost_with_defaults_match_the_sealed_fns() {
    let (graph, idx) = build_direction_fixture();
    for (s, t) in [(0usize, 3usize), (2, 3), (0, 0), (0, 2)] {
        assert_eq!(
            shortest_path_cost(&graph, idx[s], idx[t]),
            shortest_path_cost_with(&graph, idx[s], idx[t], &PathOptions::default())
        );
    }
    let pairs: Vec<_> = [(0usize, 3usize), (2, 3), (0, 0)]
        .iter()
        .map(|&(s, t)| (idx[s], idx[t]))
        .collect();
    assert_eq!(
        shortest_path_cost_batch(&graph, &pairs),
        shortest_path_cost_batch_with(&graph, &pairs, &PathOptions::default())
    );
}

#[test]
fn test_batch_honours_connection_types() {
    let (graph, idx) = build_direction_fixture();
    let knows = vec!["KNOWS".to_string()];
    // P2 reaches P3 only through the City.
    assert_eq!(
        shortest_path_cost_batch(&graph, &[(idx[2], idx[3])]),
        vec![Some(2)]
    );
    assert_eq!(
        shortest_path_cost_batch_with(
            &graph,
            &[(idx[2], idx[3])],
            &PathOptions::default().with_connection_types(&knows),
        ),
        vec![None]
    );
}

#[test]
fn test_batch_endpoint_outside_restricted_universe_answers_none() {
    let (graph, idx) = build_direction_fixture();
    let knows = vec!["KNOWS".to_string()];
    let opts = PathOptions::default().with_connection_types(&knows);
    // P2 has no KNOWS edge at all — it is not in the KNOWS universe. The pair
    // answers None; the pairs beside it in the same call still answer.
    assert_eq!(
        shortest_path_cost_batch_with(
            &graph,
            &[(idx[2], idx[0]), (idx[0], idx[1]), (idx[2], idx[2])],
            &opts,
        ),
        vec![None, Some(1), Some(0)]
    );
}

#[test]
fn test_batch_via_types_exempts_the_endpoints_only() {
    let (graph, idx) = build_direction_fixture();
    let persons = vec!["Person".to_string()];
    let opts = PathOptions::default().with_via_types(&persons);
    // P2→P3 needs the City as an *intermediate*: blocked.
    assert_eq!(
        shortest_path_cost_batch_with(&graph, &[(idx[2], idx[3])], &opts),
        vec![None]
    );
    // ...but the City may still be an endpoint.
    assert_eq!(
        shortest_path_cost_batch_with(&graph, &[(idx[2], idx[5])], &opts),
        vec![Some(1)]
    );
    // ...and that agrees with the single-pair member.
    assert_eq!(
        shortest_path_cost_with(&graph, idx[2], idx[5], &opts),
        Some(1)
    );
}

#[test]
fn test_batch_direction() {
    let (graph, idx) = build_direction_fixture();
    let pairs = [(idx[0], idx[3])];
    assert_eq!(
        shortest_path_cost_batch_with(&graph, &pairs, &PathOptions::default()),
        vec![Some(2)]
    );
    assert_eq!(
        shortest_path_cost_batch_with(
            &graph,
            &pairs,
            &PathOptions::default().with_direction(EdgeDir::Outgoing)
        ),
        vec![Some(3)]
    );
    assert_eq!(
        shortest_path_cost_batch_with(
            &graph,
            &pairs,
            &PathOptions::default().with_direction(EdgeDir::Incoming)
        ),
        vec![None]
    );
}

#[test]
fn test_scoped_directed_adjacency_orients_each_edge_once() {
    let (graph, idx) = build_direction_fixture();
    let nodes: Vec<_> = idx.clone();
    let (out_nodes, out_adj) = build_scoped_adjacency_over(
        &graph,
        nodes.clone(),
        None,
        EdgeDir::Outgoing,
        Interrupt::default(),
    )
    .unwrap();
    let (_, in_adj) = build_scoped_adjacency_over(
        &graph,
        nodes.clone(),
        None,
        EdgeDir::Incoming,
        Interrupt::default(),
    )
    .unwrap();
    let (_, any_adj) =
        build_scoped_adjacency_over(&graph, nodes, None, EdgeDir::Any, Interrupt::default())
            .unwrap();
    assert_eq!(out_nodes.len(), idx.len());

    // P0 has one outgoing KNOWS and nothing incoming.
    assert_eq!(out_adj[0], vec![1u32]);
    assert!(in_adj[0].is_empty());
    assert_eq!(any_adj[0], vec![1u32]);

    // Incoming is the transpose of outgoing, and Any is their union.
    for (u, outs) in out_adj.iter().enumerate() {
        for &v in outs {
            assert!(
                in_adj[v as usize].contains(&(u as u32)),
                "{}→{} missing from the incoming transpose",
                u,
                v
            );
        }
    }
    for (u, links) in any_adj.iter().enumerate() {
        for &v in links {
            assert!(
                out_adj[u].contains(&v) || in_adj[u].contains(&v),
                "Any link {}→{} appears in neither directed build",
                u,
                v
            );
        }
    }
}

#[test]
fn test_weighted_honours_direction_and_filters() {
    let (graph, idx) = build_direction_fixture();
    let knows = ["KNOWS".to_string()];
    // Every edge weighs 1.0 (no weight property present), so the weighted
    // answer must track the hop count exactly — including under direction.
    for direction in [EdgeDir::Any, EdgeDir::Outgoing, EdgeDir::Incoming] {
        for conn in [None, Some(&knows[..])] {
            let mut opts = PathOptions::default().with_direction(direction);
            if let Some(c) = conn {
                opts = opts.with_connection_types(c);
            }
            let hops = shortest_path_cost_with(&graph, idx[0], idx[3], &opts);
            let weight = shortest_path_cost_weighted(&graph, idx[0], idx[3], "missing", &opts);
            assert_eq!(
                hops.map(|h| h as f64),
                weight,
                "{:?} {:?}: weighted diverged from unweighted on unit weights",
                direction,
                conn
            );
        }
    }
}

#[test]
fn test_weighted_via_types_no_longer_silently_dropped() {
    let (graph, idx) = build_direction_fixture();
    let persons = vec!["Person".to_string()];
    // P2→P3 routes only through the City; via_types=['Person'] must kill it.
    assert_eq!(
        shortest_path_cost_weighted(&graph, idx[2], idx[3], "missing", &PathOptions::default()),
        Some(2.0)
    );
    assert_eq!(
        shortest_path_cost_weighted(
            &graph,
            idx[2],
            idx[3],
            "missing",
            &PathOptions::default().with_via_types(&persons),
        ),
        None
    );
}

#[test]
fn test_all_paths_direction() {
    let (graph, idx) = build_direction_fixture();
    let undirected = all_paths(
        &graph,
        idx[0],
        idx[3],
        &AllPathsOptions::default().with_max_hops(5),
    );
    let mut lens: Vec<usize> = undirected.iter().map(|p| p.len() - 1).collect();
    lens.sort_unstable();
    assert_eq!(lens, vec![2, 3]);

    let directed = all_paths(
        &graph,
        idx[0],
        idx[3],
        &AllPathsOptions::default()
            .with_max_hops(5)
            .with_direction(EdgeDir::Outgoing),
    );
    let lens: Vec<usize> = directed.iter().map(|p| p.len() - 1).collect();
    assert_eq!(lens, vec![3]);
}

// ============================================================================
// shortest_path_costs_from: the one-to-many member (S3).
// ============================================================================

/// `(node index position, hops)` pairs, sorted, for stable comparison.
fn costs_by_position(
    idx: &[petgraph::graph::NodeIndex],
    costs: &[(petgraph::graph::NodeIndex, usize)],
) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = costs
        .iter()
        .map(|&(node, d)| (idx.iter().position(|&n| n == node).unwrap(), d))
        .collect();
    out.sort_unstable();
    out
}

#[test]
fn test_costs_from_matches_the_pair_finder_on_every_node() {
    // The whole point of the API: one BFS answering what N
    // `shortest_path_cost_with` calls answer one pair at a time.
    let (graph, idx) = build_direction_fixture();
    let knows = vec!["KNOWS".to_string()];
    let persons = vec!["Person".to_string()];

    for direction in [EdgeDir::Any, EdgeDir::Outgoing, EdgeDir::Incoming] {
        for filter in 0..3 {
            let mut opts = PathOptions::default().with_direction(direction);
            match filter {
                1 => opts = opts.with_connection_types(&knows),
                2 => opts = opts.with_via_types(&persons),
                _ => {}
            }
            for &source in &idx {
                let costs = shortest_path_costs_from(&graph, source, &opts, None).unwrap();
                let map: HashMap<_, _> = costs.iter().copied().collect();
                for &target in &idx {
                    assert_eq!(
                        map.get(&target).copied(),
                        shortest_path_cost_with(&graph, source, target, &opts),
                        "{:?} filter {} {:?} -> {:?}",
                        direction,
                        filter,
                        source,
                        target
                    );
                }
            }
        }
    }
}

#[test]
fn test_costs_from_reports_the_source_at_zero_and_omits_unreachable() {
    let (graph, idx) = build_direction_fixture();
    let costs = shortest_path_costs_from(&graph, idx[0], &PathOptions::default(), None).unwrap();
    assert_eq!(
        costs_by_position(&idx, &costs),
        vec![(0, 0), (1, 1), (2, 4), (3, 2), (4, 2), (5, 3)]
    );
    // Distances come back non-decreasing, source first.
    assert_eq!(costs[0], (idx[0], 0));
    assert!(costs.windows(2).all(|w| w[0].1 <= w[1].1));

    // Nothing points at P0, so an incoming-only walk reaches only itself.
    let inbound = shortest_path_costs_from(
        &graph,
        idx[0],
        &PathOptions::default().with_direction(EdgeDir::Incoming),
        None,
    )
    .unwrap();
    assert_eq!(inbound, vec![(idx[0], 0)]);
}

#[test]
fn test_costs_from_direction_changes_the_answer() {
    let (graph, idx) = build_direction_fixture();
    let outgoing = shortest_path_costs_from(
        &graph,
        idx[0],
        &PathOptions::default().with_direction(EdgeDir::Outgoing),
        None,
    )
    .unwrap();
    // Following arrows: P0->P1->P4->P3->City; P2 is unreachable (its only edge
    // points *into* the City).
    assert_eq!(
        costs_by_position(&idx, &outgoing),
        vec![(0, 0), (1, 1), (3, 3), (4, 2), (5, 4)]
    );
}

#[test]
fn test_costs_from_max_hops_boundary() {
    let (graph, idx) = build_direction_fixture();
    let opts = PathOptions::default();

    // Some(0) is the degenerate bound: the source and nothing else.
    assert_eq!(
        shortest_path_costs_from(&graph, idx[0], &opts, Some(0)).unwrap(),
        vec![(idx[0], 0)]
    );
    assert_eq!(
        costs_by_position(
            &idx,
            &shortest_path_costs_from(&graph, idx[0], &opts, Some(1)).unwrap()
        ),
        vec![(0, 0), (1, 1)]
    );
    assert_eq!(
        costs_by_position(
            &idx,
            &shortest_path_costs_from(&graph, idx[0], &opts, Some(2)).unwrap()
        ),
        vec![(0, 0), (1, 1), (3, 2), (4, 2)]
    );
    // The eccentricity of P0 is 4, so any cap at or above it is unbounded.
    let unbounded = shortest_path_costs_from(&graph, idx[0], &opts, None).unwrap();
    for cap in [4usize, 5, 100] {
        assert_eq!(
            costs_by_position(
                &idx,
                &shortest_path_costs_from(&graph, idx[0], &opts, Some(cap)).unwrap()
            ),
            costs_by_position(&idx, &unbounded),
            "cap {cap}"
        );
    }
    // Every cap truncates the *unbounded* answer rather than changing it.
    for cap in 0..=5usize {
        let capped = shortest_path_costs_from(&graph, idx[0], &opts, Some(cap)).unwrap();
        let expected: Vec<_> = unbounded
            .iter()
            .copied()
            .filter(|&(_, d)| d <= cap)
            .collect();
        assert_eq!(
            costs_by_position(&idx, &capped),
            costs_by_position(&idx, &expected),
            "cap {cap}"
        );
    }
}

#[test]
fn test_costs_from_via_types_gates_the_middle_not_the_ends() {
    let (graph, idx) = build_direction_fixture();
    let persons = vec!["Person".to_string()];
    let costs = shortest_path_costs_from(
        &graph,
        idx[0],
        &PathOptions::default().with_via_types(&persons),
        None,
    )
    .unwrap();
    // The City is still *reported* (it is a path end, and ends are exempt),
    // but it is never expanded — so P2, which is only reachable through it,
    // drops out. This is exactly what the pair finder answers per target.
    assert_eq!(
        costs_by_position(&idx, &costs),
        vec![(0, 0), (1, 1), (3, 2), (4, 2), (5, 3)]
    );
}

#[test]
fn test_costs_from_connection_types_filter() {
    let (graph, idx) = build_direction_fixture();
    let knows = vec!["KNOWS".to_string()];
    let costs = shortest_path_costs_from(
        &graph,
        idx[0],
        &PathOptions::default().with_connection_types(&knows),
        None,
    )
    .unwrap();
    assert_eq!(
        costs_by_position(&idx, &costs),
        vec![(0, 0), (1, 1), (3, 2), (4, 2)]
    );
}

#[test]
fn test_costs_from_source_outside_the_graph_is_empty_not_a_panic() {
    let (graph, _idx) = build_direction_fixture();
    let outside = petgraph::graph::NodeIndex::new(9_999);
    assert_eq!(
        shortest_path_costs_from(&graph, outside, &PathOptions::default(), None).unwrap(),
        vec![]
    );
    // Same on the filtered path, which takes a different expansion.
    let knows = vec!["KNOWS".to_string()];
    assert_eq!(
        shortest_path_costs_from(
            &graph,
            outside,
            &PathOptions::default().with_connection_types(&knows),
            Some(3)
        )
        .unwrap(),
        vec![]
    );
}

#[test]
fn test_costs_from_isolated_source_and_empty_graph() {
    let graph = DirGraph::new();
    assert_eq!(
        shortest_path_costs_from(
            &graph,
            petgraph::graph::NodeIndex::new(0),
            &PathOptions::default(),
            None
        )
        .unwrap(),
        vec![]
    );

    let (graph, idx) = build_disconnected_graph();
    // build_disconnected_graph: 0-1 and 2-3, nothing between.
    let costs = shortest_path_costs_from(&graph, idx[0], &PathOptions::default(), None).unwrap();
    assert_eq!(costs_by_position(&idx, &costs), vec![(0, 0), (1, 1)]);
}

#[test]
fn test_costs_from_expired_deadline_errors_rather_than_truncating() {
    let (graph, idx) = build_direction_fixture();
    let expired = Interrupt::from_deadline(Some(
        std::time::Instant::now() - std::time::Duration::from_secs(1),
    ));
    // A partial map that silently drops its far half is a wrong answer, so
    // this member errors where the pair finders answer `None`.
    let mut sources = Vec::new();
    for &source in &idx {
        sources.push(shortest_path_costs_from(
            &graph,
            source,
            &PathOptions::default().with_interrupt(expired),
            None,
        ));
    }
    // The graph is tiny (under the 1024-pop check interval), so the searches
    // may complete; what must never happen is a *silently truncated* Ok.
    for (source, result) in idx.iter().zip(&sources) {
        match result {
            Err(msg) => assert!(msg.contains("timed out"), "unexpected error: {msg}"),
            Ok(costs) => assert_eq!(
                costs.len(),
                shortest_path_costs_from(&graph, *source, &PathOptions::default(), None)
                    .unwrap()
                    .len()
            ),
        }
    }
}

#[test]
fn test_eccentricity_agrees_with_costs_from() {
    // The lift: eccentricity is the max distance of the same single-source
    // search, over its own scoped (undirected, unfiltered) subgraph.
    let (graph, idx) = build_direction_fixture();
    let eccs = eccentricity_scoped(&graph, None, None, Interrupt::default()).unwrap();
    let by_node: HashMap<_, _> = eccs.iter().copied().collect();
    for &source in &idx {
        let costs =
            shortest_path_costs_from(&graph, source, &PathOptions::default(), None).unwrap();
        let furthest = costs.iter().map(|&(_, d)| d).max().unwrap_or(0) as i64;
        assert_eq!(by_node[&source], furthest, "{:?}", source);
    }
}

// ========================================================================
// S4: the bidirectional (meet-in-the-middle) pair finder.
//
// `shortest_path` / `shortest_path_directed` / `shortest_path_cost{,_with}`
// no longer run the one-sided BFS that used to live in `reconstruct_path_bfs`.
// That loop survives here as `one_sided_reference` — the oracle the randomized
// cross-check measures the new engine against.
//
// The cross-check asserts LENGTH equality and path VALIDITY, never sequence
// equality: when several shortest paths tie, meeting in the middle picks a
// different one from the one-sided scan, and both answers are correct.
// ========================================================================

/// The deleted one-sided BFS, behaviour-for-behaviour: a parent map doubling
/// as the visited set, `via_types` gating everything but the endpoints, and
/// the first touch of the target winning.
fn one_sided_reference(
    graph: &DirGraph,
    source: petgraph::graph::NodeIndex,
    target: petgraph::graph::NodeIndex,
    options: &PathOptions,
) -> Option<Vec<petgraph::graph::NodeIndex>> {
    use std::collections::VecDeque;

    if source == target {
        return Some(vec![source]);
    }
    let via_set: Option<HashSet<&str>> = options
        .via_types
        .map(|vt| vt.iter().map(|s| s.as_str()).collect());
    let interned = intern_connection_types(options.connection_types);
    let connection_types = interned.as_deref();

    let mut parent: HashMap<usize, u32> = HashMap::new();
    let mut queue: VecDeque<usize> = VecDeque::new();
    let source_idx = source.index();
    let target_idx = target.index();
    parent.insert(source_idx, source_idx as u32);
    queue.push_back(source_idx);

    while let Some(current_idx) = queue.pop_front() {
        let current = petgraph::graph::NodeIndex::new(current_idx);
        for neighbor in filtered_neighbors(graph, current, options.direction, connection_types) {
            let neighbor_idx = neighbor.index();
            if parent.contains_key(&neighbor_idx) {
                continue;
            }
            if neighbor_idx != target_idx && !node_passes_via_filter(graph, neighbor, &via_set) {
                continue;
            }
            parent.insert(neighbor_idx, current_idx as u32);
            if neighbor_idx == target_idx {
                let mut path = Vec::new();
                let mut node_idx = target_idx;
                while node_idx != source_idx {
                    path.push(petgraph::graph::NodeIndex::new(node_idx));
                    node_idx = parent[&node_idx] as usize;
                }
                path.push(source);
                path.reverse();
                return Some(path);
            }
            queue.push_back(neighbor_idx);
        }
    }
    None
}

/// Is there an edge `from -> to` the query is allowed to walk? Written against
/// the raw edge lists and the interner rather than the production neighbour
/// helpers, so the validity check cannot be satisfied by the same bug that
/// produced the path.
fn admissible_edge(
    graph: &DirGraph,
    from: petgraph::graph::NodeIndex,
    to: petgraph::graph::NodeIndex,
    options: &PathOptions,
) -> bool {
    let type_ok = |key: InternedKey| match options.connection_types {
        None => true,
        Some(types) => types.iter().any(|t| t == graph.interner.resolve(key)),
    };
    let forward = graph
        .graph
        .edges_directed(from, petgraph::Direction::Outgoing)
        .any(|e| e.target() == to && type_ok(e.connection_type()));
    let backward = graph
        .graph
        .edges_directed(from, petgraph::Direction::Incoming)
        .any(|e| e.source() == to && type_ok(e.connection_type()));
    match options.direction {
        EdgeDir::Any => forward || backward,
        EdgeDir::Outgoing => forward,
        EdgeDir::Incoming => backward,
    }
}

/// A returned path must genuinely be a path: right endpoints, no revisits,
/// every consecutive pair joined by an edge the filters admit, and every
/// *intermediate* node admitted by `via_types` (the ends are exempt).
fn assert_valid_path(
    graph: &DirGraph,
    path: &[petgraph::graph::NodeIndex],
    source: petgraph::graph::NodeIndex,
    target: petgraph::graph::NodeIndex,
    options: &PathOptions,
    ctx: &str,
) {
    assert_eq!(path.first().copied(), Some(source), "{ctx}: bad start");
    assert_eq!(path.last().copied(), Some(target), "{ctx}: bad end");
    let unique: HashSet<_> = path.iter().copied().collect();
    assert_eq!(unique.len(), path.len(), "{ctx}: revisits a node: {path:?}");
    for window in path.windows(2) {
        assert!(
            admissible_edge(graph, window[0], window[1], options),
            "{ctx}: no admissible edge {:?} -> {:?} in {path:?}",
            window[0],
            window[1]
        );
    }
    if let Some(via) = options.via_types {
        if path.len() > 2 {
            for &node in &path[1..path.len() - 1] {
                let view = graph.graph.node_view(node).expect("node exists");
                let node_type = view.node_type_str(&graph.interner);
                assert!(
                    via.iter().any(|v| v == node_type),
                    "{ctx}: intermediate {node:?} is {node_type}, not in {via:?}"
                );
            }
        }
    }
}

/// The termination rule [`bidirectional_bfs`] deliberately does **not** use:
/// expand *both* frontiers one level per round, then compare the visited sets
/// and report `forward_level + backward_level`. It is the textbook shape of
/// the bidirectional off-by-one — correct on even-length shortest paths, one
/// hop too long on odd-length ones, because the meeting happens on the middle
/// *edge* and the round-robin counter cannot express a half-round.
fn naive_round_robin_length(
    graph: &DirGraph,
    source: petgraph::graph::NodeIndex,
    target: petgraph::graph::NodeIndex,
) -> Option<usize> {
    if source == target {
        return Some(0);
    }
    let mut seen_f: HashSet<petgraph::graph::NodeIndex> = HashSet::from([source]);
    let mut seen_b: HashSet<petgraph::graph::NodeIndex> = HashSet::from([target]);
    let mut frontier_f = vec![source];
    let mut frontier_b = vec![target];
    let mut levels = 0usize;

    while !frontier_f.is_empty() && !frontier_b.is_empty() {
        for (frontier, seen) in [
            (&mut frontier_f, &mut seen_f),
            (&mut frontier_b, &mut seen_b),
        ] {
            let mut next = Vec::new();
            for &u in frontier.iter() {
                for w in filtered_neighbors(graph, u, EdgeDir::Any, None) {
                    if seen.insert(w) {
                        next.push(w);
                    }
                }
            }
            *frontier = next;
        }
        levels += 2;
        if seen_f.intersection(&seen_b).next().is_some() {
            return Some(levels);
        }
    }
    None
}

/// A bare line of `hops + 1` nodes: 0 - 1 - ... - hops, all `NEXT` edges
/// pointing forward, all nodes of type `Line`.
fn build_line_graph(hops: usize) -> (DirGraph, Vec<petgraph::graph::NodeIndex>) {
    let mut graph = DirGraph::new();
    let mut indices = Vec::new();
    for i in 0..=hops {
        let node = NodeData::new(
            Value::Int64(i as i64),
            Value::String(format!("L{i}")),
            "Line".to_string(),
            HashMap::new(),
            &mut graph.interner,
        );
        let idx = graph.graph.add_node(node);
        graph
            .type_indices
            .entry_or_default("Line".to_string())
            .push(idx);
        indices.push(idx);
    }
    for i in 0..hops {
        let edge = EdgeData::new("NEXT".to_string(), HashMap::new(), &mut graph.interner);
        graph.graph.add_edge(indices[i], indices[i + 1], edge);
    }
    (graph, indices)
}

#[test]
fn test_edge_dir_reversed_is_an_involution() {
    for dir in [EdgeDir::Any, EdgeDir::Outgoing, EdgeDir::Incoming] {
        assert_eq!(dir.reversed().reversed(), dir);
    }
    assert_eq!(EdgeDir::Any.reversed(), EdgeDir::Any);
    assert_eq!(EdgeDir::Outgoing.reversed(), EdgeDir::Incoming);
    assert_eq!(EdgeDir::Incoming.reversed(), EdgeDir::Outgoing);
}

#[test]
fn test_bidirectional_beats_the_naive_level_counter_off_by_one() {
    // The adversarial case for the level-synchronised termination rule: an
    // ODD-length shortest path, where the two half-searches meet on the
    // middle EDGE rather than on a shared middle node. The round-robin
    // implementation must overshoot by exactly one hop; ours must not.
    for hops in [3usize, 5, 7] {
        let (graph, idx) = build_line_graph(hops);
        let (source, target) = (idx[0], idx[hops]);

        assert_eq!(
            naive_round_robin_length(&graph, source, target),
            Some(hops + 1),
            "the trap must be real: round-robin termination should overshoot \
             a {hops}-hop path by one"
        );
        assert_eq!(
            shortest_path_cost(&graph, source, target),
            Some(hops),
            "{hops}-hop line: bidirectional must not inherit the overshoot"
        );
        let result = shortest_path(&graph, source, target, &PathOptions::default())
            .expect("the line is connected");
        assert_eq!(result.cost, hops);
        assert_eq!(result.path.len(), hops + 1);
        assert_valid_path(
            &graph,
            &result.path,
            source,
            target,
            &PathOptions::default(),
            &format!("odd line {hops}"),
        );
        // The line has exactly one path, so here the sequence IS pinnable.
        assert_eq!(result.path, idx);
    }

    // Non-vacuity of the trap's shape: the naive rule is right on EVEN
    // lengths, so the test above is detecting the odd/even asymmetry, not a
    // reference implementation that is simply broken everywhere.
    for hops in [2usize, 4, 6] {
        let (graph, idx) = build_line_graph(hops);
        assert_eq!(
            naive_round_robin_length(&graph, idx[0], idx[hops]),
            Some(hops),
            "round-robin termination is correct on even lengths"
        );
        assert_eq!(shortest_path_cost(&graph, idx[0], idx[hops]), Some(hops));
    }
}

/// Deterministic xorshift64*. The cross-check must reproduce exactly from its
/// seed when it fails, so: no thread RNG, no external dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Random G(n, m): `n` nodes split over two node types, `m` directed edges
/// split over two relationship types. Self-loops are rejected; parallel edges
/// and a→b/b→a pairs are kept, because they are exactly the shapes whose
/// duplicate neighbour entries the allocation-free expansion no longer
/// deduplicates.
fn build_random_graph(
    rng: &mut Rng,
    n: usize,
    m: usize,
) -> (DirGraph, Vec<petgraph::graph::NodeIndex>) {
    let mut graph = DirGraph::new();
    let mut indices = Vec::with_capacity(n);
    for i in 0..n {
        let node_type = if rng.below(2) == 0 { "T0" } else { "T1" };
        let node = NodeData::new(
            Value::Int64(i as i64),
            Value::String(format!("N{i}")),
            node_type.to_string(),
            HashMap::new(),
            &mut graph.interner,
        );
        let idx = graph.graph.add_node(node);
        graph
            .type_indices
            .entry_or_default(node_type.to_string())
            .push(idx);
        indices.push(idx);
    }
    for _ in 0..m {
        let a = rng.below(n);
        let b = rng.below(n);
        if a == b {
            continue;
        }
        let rel = if rng.below(2) == 0 { "R0" } else { "R1" };
        let edge = EdgeData::new(rel.to_string(), HashMap::new(), &mut graph.interner);
        graph.graph.add_edge(indices[a], indices[b], edge);
    }
    (graph, indices)
}

#[test]
fn test_bidirectional_matches_one_sided_on_random_graphs() {
    let rel_r0 = vec!["R0".to_string()];
    let via_t0 = vec!["T0".to_string()];
    let configs: Vec<(&str, PathOptions)> = vec![
        ("default", PathOptions::default()),
        (
            "outgoing",
            PathOptions::default().with_direction(EdgeDir::Outgoing),
        ),
        (
            "incoming",
            PathOptions::default().with_direction(EdgeDir::Incoming),
        ),
        (
            "rel=R0",
            PathOptions::default().with_connection_types(&rel_r0),
        ),
        ("via=T0", PathOptions::default().with_via_types(&via_t0)),
        (
            "rel=R0+outgoing",
            PathOptions::default()
                .with_connection_types(&rel_r0)
                .with_direction(EdgeDir::Outgoing),
        ),
        (
            "via=T0+incoming",
            PathOptions::default()
                .with_via_types(&via_t0)
                .with_direction(EdgeDir::Incoming),
        ),
        (
            "rel=R0+via=T0",
            PathOptions::default()
                .with_connection_types(&rel_r0)
                .with_via_types(&via_t0),
        ),
    ];

    let mut checks = 0usize;
    let mut reachable = 0usize;
    let mut deepest = 0usize;
    for seed in 1..=10u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        // A spread of densities: sparse graphs exercise the "one frontier runs
        // dry" exit, dense ones the tie-breaking between equal-length paths.
        let (n, m) = match seed % 4 {
            0 => (20, 24),
            1 => (12, 30),
            2 => (30, 31), // very sparse: long paths, and frontiers that run dry
            _ => (14, 16),
        };
        let (graph, idx) = build_random_graph(&mut rng, n, m);

        for (label, options) in &configs {
            for &source in &idx {
                for &target in &idx {
                    let ctx = format!("seed {seed} [{label}] {source:?}->{target:?}");
                    let expected = one_sided_reference(&graph, source, target, options);
                    let got = shortest_path(&graph, source, target, options);
                    checks += 1;

                    match (&expected, &got) {
                        (None, None) => {}
                        (Some(want), Some(result)) => {
                            reachable += 1;
                            deepest = deepest.max(result.cost);
                            assert_eq!(
                                result.path.len(),
                                want.len(),
                                "{ctx}: length disagreement, one-sided {want:?} vs \
                                 bidirectional {:?}",
                                result.path
                            );
                            assert_eq!(result.cost, want.len() - 1, "{ctx}: cost/path mismatch");
                            assert_valid_path(&graph, &result.path, source, target, options, &ctx);
                            assert_eq!(
                                shortest_path_cost_with(&graph, source, target, options),
                                Some(result.cost),
                                "{ctx}: the length-only member disagrees with the path"
                            );
                            assert!(
                                are_connected_with(&graph, source, target, options),
                                "{ctx}: are_connected_with denies a path it returned"
                            );
                        }
                        _ => panic!(
                            "{ctx}: reachability disagreement — one-sided {expected:?}, \
                             bidirectional {got:?}"
                        ),
                    }
                }
            }
        }
    }
    assert!(
        checks > 10_000,
        "cross-check corpus shrank to {checks} cases"
    );
    assert!(
        reachable > checks / 10,
        "only {reachable}/{checks} pairs were connected — the fixtures went \
         too sparse to exercise the meeting rule"
    );
    // Both odd and even path lengths, deep enough that the two frontiers
    // actually take several rounds to meet. A corpus of 1- and 2-hop pairs
    // would never exercise the termination rule at all.
    assert!(
        deepest >= 5,
        "deepest shortest path in the corpus was {deepest} hops — too shallow \
         to exercise multi-round meeting"
    );
}

#[test]
fn test_bidirectional_directed_agrees_with_the_transpose_query() {
    // The backward frontier must expand the REVERSE of the query direction.
    // If it did not, an outgoing query would search out of the target and
    // silently answer the wrong question — which on a symmetric fixture is
    // invisible, so the fixture is deliberately asymmetric.
    let mut rng = Rng(0xC0FF_EE12_3456_789A);
    let (graph, idx) = build_random_graph(&mut rng, 16, 22);
    for &source in &idx {
        for &target in &idx {
            let out = shortest_path_cost_with(
                &graph,
                source,
                target,
                &PathOptions::default().with_direction(EdgeDir::Outgoing),
            );
            let inn = shortest_path_cost_with(
                &graph,
                target,
                source,
                &PathOptions::default().with_direction(EdgeDir::Incoming),
            );
            assert_eq!(
                out, inn,
                "{source:?}->{target:?} outgoing vs transposed incoming"
            );
        }
    }
}
