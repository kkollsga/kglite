use super::*;
use crate::datatypes::values::Value;
use crate::graph::schema::{DirGraph, EdgeData, InternedKey, NodeData};
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
    let result = shortest_path(
        &graph,
        indices[0],
        indices[1],
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    );
    assert!(result.is_some());
    let path = result.unwrap();
    assert_eq!(path.cost, 1);
    assert_eq!(path.path.len(), 2);
}

#[test]
fn test_shortest_path_multi_hop() {
    let (graph, indices) = build_chain_graph();
    let result = shortest_path(
        &graph,
        indices[0],
        indices[4],
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    );
    assert!(result.is_some());
    let path = result.unwrap();
    assert_eq!(path.cost, 4);
    assert_eq!(path.path.len(), 5);
}

#[test]
fn test_shortest_path_same_node() {
    let (graph, indices) = build_chain_graph();
    let result = shortest_path(
        &graph,
        indices[0],
        indices[0],
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    );
    assert!(result.is_some());
    let path = result.unwrap();
    assert_eq!(path.cost, 0);
    assert_eq!(path.path.len(), 1);
}

#[test]
fn test_shortest_path_not_found() {
    let (graph, indices) = build_disconnected_graph();
    let result = shortest_path(
        &graph,
        indices[0],
        indices[2],
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    );
    assert!(result.is_none());
}

#[test]
fn test_shortest_path_reverse_direction() {
    // BFS is undirected, so B -> A should find a path even though edge is A -> B
    let (graph, indices) = build_chain_graph();
    let result = shortest_path(
        &graph,
        indices[4],
        indices[0],
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    );
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
        5,
        None,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
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
        1,
        None,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
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
        3,
        None,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
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
        3,
        Some(1),
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
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
        3,
        Some(1),
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    );
    let unlimited = all_paths(
        &graph,
        indices[0],
        indices[2],
        3,
        None,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
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
    let result = shortest_path(
        &graph,
        indices[0],
        indices[2],
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    );
    assert_eq!(result.unwrap().cost, 1);

    // With NEXT filter: must go A->B->C (2 hops)
    let next_only = vec!["NEXT".to_string()];
    let result = shortest_path(
        &graph,
        indices[0],
        indices[2],
        Some(&next_only),
        None,
        crate::graph::algorithms::Interrupt::default(),
    );
    assert_eq!(result.unwrap().cost, 2);

    // With SKIP filter: A->C (1 hop)
    let skip_only = vec!["SKIP".to_string()];
    let result = shortest_path(
        &graph,
        indices[0],
        indices[2],
        Some(&skip_only),
        None,
        crate::graph::algorithms::Interrupt::default(),
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
    let results = betweenness_centrality(
        &graph,
        false,
        None,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    )
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
        false,
        Some(3),
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
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
fn test_degree_centrality() {
    let (graph, indices) = build_chain_graph();
    let results = degree_centrality(
        &graph,
        false,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
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
    let results = pagerank(
        &graph,
        0.85,
        100,
        1e-6,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
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
    let results = closeness_centrality(
        &graph,
        false,
        None,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
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
fn test_centrality_scope_restricts_nodes_and_edges() {
    // Chain 0-1-2-3-4. Scope to {1,2,3}: edges 0-1 and 3-4 leave scope and are
    // dropped, leaving the sub-chain 1-2-3.
    let (graph, indices) = build_chain_graph();
    let scope: std::collections::HashSet<_> =
        [indices[1], indices[2], indices[3]].into_iter().collect();

    let deg = degree_centrality(
        &graph,
        false,
        None,
        Some(&scope),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    assert_eq!(deg.len(), 3, "only scoped nodes returned");
    let score_of = |idx| deg.iter().find(|r| r.node_idx == idx).unwrap().score;
    // Within the sub-chain, the middle node (2) has degree 2; the ends (1,3) have 1.
    assert_eq!(score_of(indices[2]), 2.0);
    assert_eq!(score_of(indices[1]), 1.0);
    assert_eq!(score_of(indices[3]), 1.0);

    // Excluded nodes never appear in any scoped algorithm's output.
    let pr = pagerank(
        &graph,
        0.85,
        100,
        1e-6,
        None,
        Some(&scope),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    let pr_nodes: std::collections::HashSet<_> = pr.iter().map(|r| r.node_idx).collect();
    assert_eq!(pr_nodes, scope);
}

#[test]
fn test_pagerank_empty_graph() {
    let graph = DirGraph::new();
    let results = pagerank(
        &graph,
        0.85,
        100,
        1e-6,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
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
    let r = louvain_communities(
        &graph,
        None,
        1.0,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
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
    let r = louvain_communities(
        &graph,
        None,
        1.0,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
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
    let a = louvain_communities(
        &graph,
        None,
        1.0,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    let b = louvain_communities(
        &graph,
        None,
        1.0,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    assert_eq!(a.num_communities, b.num_communities);
    let ca: Vec<usize> = a.assignments.iter().map(|x| x.community_id).collect();
    let cb: Vec<usize> = b.assignments.iter().map(|x| x.community_id).collect();
    assert_eq!(ca, cb, "deterministic across runs");
}

#[test]
fn test_louvain_empty_and_isolated() {
    // empty
    let g = DirGraph::new();
    let r = louvain_communities(
        &g,
        None,
        1.0,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
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
    let r3 = louvain_communities(
        &g3,
        None,
        1.0,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
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
    let r = leiden_communities(
        &graph,
        None,
        1.0,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    assert_eq!(r.num_communities, 2);
    assert!(r.modularity > 0.0);
    assert_eq!(community_of(&r, ix[0]), community_of(&r, ix[1]));
    assert_eq!(community_of(&r, ix[3]), community_of(&r, ix[5]));
    assert_ne!(community_of(&r, ix[0]), community_of(&r, ix[3]));
}

#[test]
fn test_leiden_communities_well_connected() {
    let (graph, _) = build_two_triangle_bridge();
    let r = leiden_communities(
        &graph,
        None,
        1.0,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    assert_all_communities_connected(&graph, &r);

    // also on a chain and a triangle — the invariant must always hold
    let (chain, _) = build_chain_graph();
    let rc = leiden_communities(
        &chain,
        None,
        1.0,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    assert_all_communities_connected(&chain, &rc);
}

#[test]
fn test_leiden_deterministic() {
    let (graph, _) = build_two_triangle_bridge();
    let a = leiden_communities(
        &graph,
        None,
        1.0,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    let b = leiden_communities(
        &graph,
        None,
        1.0,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    let ca: Vec<usize> = a.assignments.iter().map(|x| x.community_id).collect();
    let cb: Vec<usize> = b.assignments.iter().map(|x| x.community_id).collect();
    assert_eq!(ca, cb);
}

#[test]
fn test_leiden_hierarchy_and_modularity_vs_louvain() {
    let (graph, _) = build_two_triangle_bridge();
    let lei = leiden_communities(
        &graph,
        None,
        1.0,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    let lou = louvain_communities(
        &graph,
        None,
        1.0,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
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
    let r = leiden_communities(
        &g,
        None,
        1.0,
        None,
        None,
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    assert_eq!(r.num_communities, 0);
    assert!(r.levels.is_empty());
}
