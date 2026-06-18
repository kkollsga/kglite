// src/graph/graph_algorithms.rs
//! Graph algorithms module providing path finding and connectivity analysis.

use crate::datatypes::values::Value;
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::GraphRead;
use petgraph::algo::kosaraju_scc;
use petgraph::graph::NodeIndex;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

// Centrality algorithms moved to the sibling `centrality` module to keep this
// file under the god-file ceiling. Re-exported so existing
// `graph_algorithms::{betweenness_centrality, pagerank, degree_centrality,
// closeness_centrality, CentralityResult}` paths keep resolving.
pub use super::centrality::*;

/// Standard timeout error message for graph algorithms.
/// Mirrors the MATCH timeout text in `cypher::executor::mod::check_deadline`,
/// adapted for procedure context (no anchor hint — large graphs may simply
/// not converge within the default 20s).
pub fn algorithm_timeout_err() -> String {
    "CALL procedure timed out. Pass timeout_ms=N to cypher() to extend, \
     or timeout_ms=0 to disable the deadline. Scope to a subgraph with \
     {node_type: '...', where: '...'} to run on fewer nodes."
        .to_string()
}

// ============================================================================
// Path Filtering Helpers
// ============================================================================

/// Pre-intern connection type strings into InternedKeys for fast comparison.
pub(crate) fn intern_connection_types(
    connection_types: Option<&[String]>,
) -> Option<Vec<InternedKey>> {
    connection_types.map(|types| types.iter().map(|t| InternedKey::from_str(t)).collect())
}

/// Optional subgraph scope for the centrality / community procedures: the set
/// of node indices the algorithm is allowed to consider. `None` means the whole
/// graph (the unscoped fast path — every loop below short-circuits the scope
/// check on the `None` discriminant, so there is no per-edge cost when absent).
/// Built in the Cypher CALL dispatcher from `{node_type, where}` so an analysis
/// can run e.g. PageRank over non-test, non-external functions only.
pub(crate) type NodeScope = std::collections::HashSet<NodeIndex>;

/// The working node set, honoring an optional subgraph scope. Preserves graph
/// (index) order so compact-index mappings stay deterministic.
pub(crate) fn scoped_node_set(graph: &DirGraph, scope: Option<&NodeScope>) -> Vec<NodeIndex> {
    let g = &graph.graph;
    match scope {
        Some(s) => g.node_indices().filter(|n| s.contains(n)).collect(),
        None => g.node_indices().collect(),
    }
}

/// True when an edge lies within scope — both endpoints in the set, or no scope.
#[inline]
pub(crate) fn edge_in_scope(scope: Option<&NodeScope>, src: NodeIndex, tgt: NodeIndex) -> bool {
    match scope {
        Some(s) => s.contains(&src) && s.contains(&tgt),
        None => true,
    }
}

/// Get undirected neighbors filtered by edge connection type.
/// When connection_types is None, returns all neighbors (equivalent to
/// neighbors_undirected).
///
/// Both branches deduplicate — petgraph's `neighbors_undirected` walks
/// every incident edge so parallel edges and a→b/b→a pairs each appear
/// twice; the filtered branch concatenates Outgoing + Incoming and has
/// the same property. Without dedup, undirected `shortestPath` and
/// `all_paths` over a bidirectional pair surfaced duplicate
/// (A, B) / (B, A) entries during enumeration (B4) — the visited
/// bitmap downstream caught most cases for `shortestPath`, but
/// `all_paths` paid wasted DFS work per duplicate.
///
/// Sort + dedup is faster than a presence-set probe for the typical
/// small-degree case (n ≲ 32) because the in-place comparison fits in
/// cache; insertion order is not load-bearing for any caller (BFS,
/// DFS path enumeration use set-membership, not order).
fn filtered_neighbors_undirected(
    graph: &DirGraph,
    node: NodeIndex,
    connection_types: Option<&[InternedKey]>,
) -> Vec<NodeIndex> {
    use petgraph::Direction;
    let g = &graph.graph;
    let mut neighbors: Vec<NodeIndex> = match connection_types {
        None => g.neighbors_undirected(node).collect(),
        Some(types) => {
            let mut n = Vec::new();
            for edge in g.edges_directed(node, Direction::Outgoing) {
                if types.iter().any(|t| *t == edge.connection_type()) {
                    n.push(edge.target());
                }
            }
            for edge in g.edges_directed(node, Direction::Incoming) {
                if types.iter().any(|t| *t == edge.connection_type()) {
                    n.push(edge.source());
                }
            }
            n
        }
    };
    if neighbors.len() > 1 {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    neighbors
}

/// Get directed (outgoing only) neighbors filtered by edge connection type.
fn filtered_neighbors_outgoing(
    graph: &DirGraph,
    node: NodeIndex,
    connection_types: Option<&[InternedKey]>,
) -> Vec<NodeIndex> {
    use petgraph::Direction;
    let g = &graph.graph;
    match connection_types {
        None => g.neighbors_directed(node, Direction::Outgoing).collect(),
        Some(types) => g
            .edges_directed(node, Direction::Outgoing)
            .filter(|e| types.iter().any(|t| *t == e.connection_type()))
            .map(|e| e.target())
            .collect(),
    }
}

/// Check if a node passes the via_types filter.
/// Source and target should be excluded from this check by the caller.
fn node_passes_via_filter(
    graph: &DirGraph,
    node: NodeIndex,
    via_types: &Option<HashSet<&str>>,
) -> bool {
    match via_types {
        None => true,
        Some(types) => {
            if let Some(node_data) = graph.graph.node_weight(node) {
                types.contains(node_data.node_type_str(&graph.interner))
            } else {
                false
            }
        }
    }
}

/// Result of a path finding operation
#[derive(Debug, Clone)]
pub struct PathResult {
    /// The path as a sequence of node indices
    pub path: Vec<NodeIndex>,
    /// The total cost/length of the path
    pub cost: usize,
}

/// Information about a node in a path (for Python output)
#[derive(Debug, Clone)]
pub struct PathNodeInfo {
    pub node_type: String,
    pub title: String,
    pub id: Value,
}

/// Find the shortest path between two nodes using undirected BFS.
/// This treats the graph as undirected, finding connections in either direction.
/// Returns None if no path exists.
///
/// # Arguments
/// * `connection_types` - Only traverse edges of these types (None = all)
/// * `via_types` - Only traverse through nodes of these types (None = all)
pub fn shortest_path(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
    connection_types: Option<&[String]>,
    via_types: Option<&[String]>,
    deadline: Option<Instant>,
) -> Option<PathResult> {
    let via_set: Option<HashSet<&str>> =
        via_types.map(|vt| vt.iter().map(|s| s.as_str()).collect());
    let interned = intern_connection_types(connection_types);
    let path = reconstruct_path_bfs(
        graph,
        source,
        target,
        interned.as_deref(),
        &via_set,
        deadline,
    )?;
    let cost = path.len().saturating_sub(1);

    Some(PathResult { path, cost })
}

/// Find the shortest path LENGTH between two nodes using undirected BFS.
/// Only returns the hop count, avoiding parent tracking and path reconstruction.
/// Uses level-by-level BFS to avoid per-node distance tracking.
pub fn shortest_path_cost(graph: &DirGraph, source: NodeIndex, target: NodeIndex) -> Option<usize> {
    if source == target {
        return Some(0);
    }

    let node_bound = graph.graph.node_bound();
    let mut visited: Vec<bool> = vec![false; node_bound];

    let target_idx = target.index();

    // Level-by-level BFS using two alternating vectors (avoids VecDeque overhead)
    let mut current_level: Vec<usize> = vec![source.index()];
    let mut next_level: Vec<usize> = Vec::new();
    visited[source.index()] = true;
    let mut depth: usize = 0;

    while !current_level.is_empty() {
        depth += 1;
        next_level.clear();

        for &current_idx in &current_level {
            let current = NodeIndex::new(current_idx);

            for neighbor in {
                let g = &graph.graph;
                g.neighbors_undirected(current)
            } {
                let neighbor_idx = neighbor.index();
                if !visited[neighbor_idx] {
                    if neighbor_idx == target_idx {
                        return Some(depth);
                    }
                    visited[neighbor_idx] = true;
                    next_level.push(neighbor_idx);
                }
            }
        }

        std::mem::swap(&mut current_level, &mut next_level);
    }

    None
}

/// Batch shortest path cost — reuses visited Vec and adjacency list across multiple pairs.
/// Much faster than calling shortest_path_cost N times for large graphs.
pub fn shortest_path_cost_batch(
    graph: &DirGraph,
    pairs: &[(NodeIndex, NodeIndex)],
) -> Vec<Option<usize>> {
    let node_bound = graph.graph.node_bound();

    // Pre-build undirected adjacency list ONCE for all queries
    let nodes: Vec<NodeIndex> = {
        let g = &graph.graph;
        g.node_indices().collect()
    };
    let n = nodes.len();
    let mut node_to_idx = vec![usize::MAX; node_bound];
    for (i, &node) in nodes.iter().enumerate() {
        node_to_idx[node.index()] = i;
    }

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for edge in {
        let g = &graph.graph;
        g.edge_references()
    } {
        let src_i = node_to_idx[edge.source().index()];
        let tgt_i = node_to_idx[edge.target().index()];
        if src_i != usize::MAX && tgt_i != usize::MAX {
            adj[src_i].push(tgt_i);
            adj[tgt_i].push(src_i);
        }
    }
    // Dedup undirected adjacency (handles bidirectional edges A→B + B→A)
    for neighbors in &mut adj {
        neighbors.sort_unstable();
        neighbors.dedup();
    }

    // Reusable visited array — cleared between queries
    let mut visited: Vec<bool> = vec![false; n];
    let mut current_level: Vec<usize> = Vec::new();
    let mut next_level: Vec<usize> = Vec::new();

    let mut results = Vec::with_capacity(pairs.len());

    for &(source, target) in pairs {
        if source == target {
            results.push(Some(0));
            continue;
        }

        let src_i = node_to_idx[source.index()];
        let tgt_i = node_to_idx[target.index()];
        if src_i == usize::MAX || tgt_i == usize::MAX {
            results.push(None);
            continue;
        }

        // Clear visited (only reset nodes we actually touched)
        // Use a generation counter instead of clearing — much faster
        // But for simplicity, track touched nodes
        let mut touched: Vec<usize> = Vec::new();

        current_level.clear();
        current_level.push(src_i);
        visited[src_i] = true;
        touched.push(src_i);
        let mut depth: usize = 0;
        let mut found = false;

        'bfs: while !current_level.is_empty() {
            depth += 1;
            next_level.clear();

            for &current_idx in &current_level {
                for &neighbor_idx in &adj[current_idx] {
                    if !visited[neighbor_idx] {
                        if neighbor_idx == tgt_i {
                            found = true;
                            break 'bfs;
                        }
                        visited[neighbor_idx] = true;
                        touched.push(neighbor_idx);
                        next_level.push(neighbor_idx);
                    }
                }
            }

            std::mem::swap(&mut current_level, &mut next_level);
        }

        results.push(if found { Some(depth) } else { None });

        // Reset only touched nodes (much faster than clearing entire array)
        for &idx in &touched {
            visited[idx] = false;
        }
    }

    results
}

/// Reconstruct path using BFS.
///
/// Phase A.3 / 0.9.53 perf fix: switched from `Vec<bool> + Vec<u32>` of
/// `node_bound` capacity to `HashMap` for parent tracking. The old
/// implementation allocated 500 KB (Vec<bool>) + 2 MB (Vec<u32>) per
/// call on a 500 K-node graph regardless of actual BFS scope. For a
/// shallow lookup that visits ~16 nodes the per-call alloc + init
/// (~30 ms) dominated the operation; this fix moved d=1 shortestPath
/// latency from 37 µs → 4 µs on a 500K-node fixture.
///
/// The HashMap also doubles as the visited set: presence ⇔ visited.
/// Deep BFS does pay slightly more per-node (~100 ns hash insert vs
/// ~5 ns array write), but the per-call alloc savings dominate for
/// any realistic graph + traversal shape; deep paths on small graphs
/// stay fast because the HashMap doesn't preallocate `node_bound`.
fn reconstruct_path_bfs(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
    connection_types: Option<&[InternedKey]>,
    via_types: &Option<HashSet<&str>>,
    deadline: Option<Instant>,
) -> Option<Vec<NodeIndex>> {
    use std::collections::{HashMap, VecDeque};

    if source == target {
        return Some(vec![source]);
    }

    let mut parent: HashMap<usize, u32> = HashMap::with_capacity(64);
    let mut queue: VecDeque<usize> = VecDeque::with_capacity(64);

    let source_idx = source.index();
    let target_idx = target.index();

    parent.insert(source_idx, source_idx as u32);
    queue.push_back(source_idx);

    let mut visit_count = 0u32;

    while let Some(current_idx) = queue.pop_front() {
        // Periodic timeout check (every 1000 nodes)
        visit_count += 1;
        if visit_count.is_multiple_of(1000) {
            if let Some(dl) = deadline {
                if Instant::now() > dl {
                    return None;
                }
            }
        }

        let current = NodeIndex::new(current_idx);

        // Check all neighbors (both directions for undirected path finding)
        let neighbors = filtered_neighbors_undirected(graph, current, connection_types);
        for neighbor in neighbors {
            let neighbor_idx = neighbor.index();

            if parent.contains_key(&neighbor_idx) {
                continue;
            }
            // Apply via_types filter (skip if not target and doesn't match)
            if neighbor_idx != target_idx && !node_passes_via_filter(graph, neighbor, via_types) {
                continue;
            }

            parent.insert(neighbor_idx, current_idx as u32);
            if neighbor_idx == target_idx {
                // Found target - reconstruct path
                let mut path = Vec::with_capacity(16);
                let mut node_idx = target_idx;

                while node_idx != source_idx {
                    path.push(NodeIndex::new(node_idx));
                    node_idx = parent[&node_idx] as usize;
                }
                path.push(source);
                path.reverse();
                return Some(path);
            }
            queue.push_back(neighbor_idx);
        }
    }

    None // No path found
}

/// Directed BFS shortest path — only follows outgoing edges.
/// Used by Cypher shortestPath() which respects edge direction.
///
/// # Arguments
/// * `connection_types` - Only traverse edges of these types (None = all)
/// * `via_types` - Only traverse through nodes of these types (None = all)
pub fn shortest_path_directed(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
    connection_types: Option<&[String]>,
    via_types: Option<&[String]>,
    deadline: Option<Instant>,
) -> Option<PathResult> {
    use std::collections::VecDeque;

    if source == target {
        return Some(PathResult {
            path: vec![source],
            cost: 0,
        });
    }

    let via_set: Option<HashSet<&str>> =
        via_types.map(|vt| vt.iter().map(|s| s.as_str()).collect());
    let interned = intern_connection_types(connection_types);

    let node_bound = graph.graph.node_bound();
    let mut visited: Vec<bool> = vec![false; node_bound];
    let mut parent: Vec<u32> = vec![u32::MAX; node_bound];
    let mut queue = VecDeque::with_capacity(node_bound / 4);

    let source_idx = source.index();
    let target_idx = target.index();

    queue.push_back(source_idx);
    visited[source_idx] = true;

    let mut visit_count = 0u32;

    while let Some(current_idx) = queue.pop_front() {
        // Periodic timeout check
        visit_count += 1;
        if visit_count.is_multiple_of(1000) {
            if let Some(dl) = deadline {
                if Instant::now() > dl {
                    return None;
                }
            }
        }

        let current = NodeIndex::new(current_idx);

        // Only follow outgoing edges
        let neighbors = filtered_neighbors_outgoing(graph, current, interned.as_deref());
        for neighbor in neighbors {
            let neighbor_idx = neighbor.index();

            if !visited[neighbor_idx] {
                // Apply via_types filter (skip if not target and doesn't match)
                if neighbor_idx != target_idx && !node_passes_via_filter(graph, neighbor, &via_set)
                {
                    continue;
                }

                visited[neighbor_idx] = true;
                parent[neighbor_idx] = current_idx as u32;
                queue.push_back(neighbor_idx);

                if neighbor_idx == target_idx {
                    let mut path = Vec::with_capacity(16);
                    let mut node_idx = target_idx;

                    while node_idx != source_idx {
                        path.push(NodeIndex::new(node_idx));
                        node_idx = parent[node_idx] as usize;
                    }
                    path.push(source);
                    path.reverse();

                    let cost = path.len().saturating_sub(1);
                    return Some(PathResult { path, cost });
                }
            }
        }
    }

    None
}

/// Find all paths between two nodes up to a maximum number of hops.
/// Warning: This can be expensive for graphs with many paths!
///
/// # Arguments
/// * `max_results` - Stop after finding this many paths (prevents OOM on dense graphs)
/// * `connection_types` - Only traverse edges of these types (None = all)
/// * `via_types` - Only traverse through nodes of these types (None = all)
#[allow(clippy::too_many_arguments)]
pub fn all_paths(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
    max_hops: usize,
    max_results: Option<usize>,
    connection_types: Option<&[String]>,
    via_types: Option<&[String]>,
    deadline: Option<Instant>,
) -> Vec<Vec<NodeIndex>> {
    let via_set: Option<HashSet<&str>> =
        via_types.map(|vt| vt.iter().map(|s| s.as_str()).collect());
    let interned = intern_connection_types(connection_types);
    let mut results = Vec::new();
    let mut current_path = vec![source];
    let mut visited = HashSet::new();
    visited.insert(source);

    find_all_paths_recursive(
        graph,
        source,
        target,
        max_hops,
        &mut current_path,
        &mut visited,
        &mut results,
        max_results,
        interned.as_deref(),
        &via_set,
        deadline,
    );

    results
}

#[allow(clippy::only_used_in_recursion, clippy::too_many_arguments)]
fn find_all_paths_recursive(
    graph: &DirGraph,
    current: NodeIndex,
    target: NodeIndex,
    remaining_hops: usize,
    current_path: &mut Vec<NodeIndex>,
    visited: &mut HashSet<NodeIndex>,
    results: &mut Vec<Vec<NodeIndex>>,
    max_results: Option<usize>,
    connection_types: Option<&[InternedKey]>,
    via_types: &Option<HashSet<&str>>,
    deadline: Option<Instant>,
) {
    // Early termination when result limit is hit
    if let Some(max) = max_results {
        if results.len() >= max {
            return;
        }
    }

    // Timeout check at each recursive entry
    if let Some(dl) = deadline {
        if Instant::now() > dl {
            return;
        }
    }

    if current == target {
        results.push(current_path.clone());
        return;
    }

    if remaining_hops == 0 {
        return;
    }

    // Explore all neighbors (undirected), filtered by connection type
    let neighbors = filtered_neighbors_undirected(graph, current, connection_types);
    for neighbor in neighbors {
        // Check limit before exploring deeper
        if let Some(max) = max_results {
            if results.len() >= max {
                return;
            }
        }

        if !visited.contains(&neighbor) {
            // Apply via_types filter (skip if not target and doesn't match)
            if neighbor != target && !node_passes_via_filter(graph, neighbor, via_types) {
                continue;
            }

            visited.insert(neighbor);
            current_path.push(neighbor);

            find_all_paths_recursive(
                graph,
                neighbor,
                target,
                remaining_hops - 1,
                current_path,
                visited,
                results,
                max_results,
                connection_types,
                via_types,
                deadline,
            );

            current_path.pop();
            visited.remove(&neighbor);
        }
    }
}

/// Find all strongly connected components in the graph.
/// Returns a vector of components, each component is a vector of node indices.
pub fn connected_components(graph: &DirGraph) -> Vec<Vec<NodeIndex>> {
    // For disk mode, fall back to weakly_connected_components since
    // kosaraju_scc requires petgraph trait bounds.
    if GraphRead::is_disk(&graph.graph) {
        return weakly_connected_components(graph, None)
            .expect("weakly_connected_components with deadline=None cannot time out");
    }
    kosaraju_scc(graph.graph.as_stable_digraph())
}

/// Find weakly connected components (treating graph as undirected).
/// This is often more useful for knowledge graphs.
/// Uses Union-Find (disjoint set) for optimal performance — O(E * α(V)) ≈ O(E).
///
/// @procedure: connected_components
/// @procedure: weakly_connected_components
pub fn weakly_connected_components(
    graph: &DirGraph,
    deadline: Option<Instant>,
) -> Result<Vec<Vec<NodeIndex>>, String> {
    weakly_connected_components_scoped(graph, None, None, deadline)
}

/// Weakly connected components, optionally scoped to a node-type universe
/// and/or a set of relationship types.
///
/// - `node_types`: when `Some`, the component universe is restricted to
///   nodes of those types — a node of an excluded type never appears, even
///   as a singleton. When `None`, the universe is every node, *unless*
///   `rel_types` is `Some`, in which case it is the set of nodes incident to
///   at least one matching edge (the subgraph induced by those edges).
/// - `rel_types`: when `Some`, only edges of those types union their
///   endpoints; all other edges are ignored. When `None`, every edge unions.
///
/// `weakly_connected_components_scoped(g, Some(&["Person"]), Some(&[knows]), …)`
/// is the "components of the Person/KNOWS subgraph" query — the single-
/// relationship projection a graph-algorithm library would operate on.
/// Unknown node-type names contribute no nodes (they are skipped, not an
/// error) so a multi-type request degrades gracefully.
pub fn weakly_connected_components_scoped(
    graph: &DirGraph,
    node_types: Option<&[String]>,
    rel_types: Option<&[InternedKey]>,
    deadline: Option<Instant>,
) -> Result<Vec<Vec<NodeIndex>>, String> {
    let edge_matches = |key: InternedKey| -> bool {
        match rel_types {
            Some(keys) => keys.contains(&key),
            None => true,
        }
    };

    // Node universe (see doc-comment for the three cases).
    let nodes: Vec<NodeIndex> = if let Some(types) = node_types {
        let mut v = Vec::new();
        for t in types {
            if let Some(type_nodes) = graph.type_indices.get(t.as_str()) {
                v.extend(type_nodes.iter());
            }
        }
        v
    } else if rel_types.is_some() {
        let mut seen: HashSet<NodeIndex> = HashSet::new();
        for edge in {
            let g = &graph.graph;
            g.edge_references()
        } {
            if edge_matches(edge.connection_type()) {
                seen.insert(edge.source());
                seen.insert(edge.target());
            }
        }
        seen.into_iter().collect()
    } else {
        let g = &graph.graph;
        g.node_indices().collect()
    };

    let n = nodes.len();

    if n == 0 {
        return Ok(Vec::new());
    }

    // Use node_bound() not node_count() — StableDiGraph indices can have gaps
    let bound = graph.graph.node_bound();

    // Build compact index mapping: graph NodeIndex → contiguous 0..n.
    // usize::MAX marks a node outside the universe (skipped during union).
    let mut node_to_idx = vec![usize::MAX; bound];
    for (i, &node) in nodes.iter().enumerate() {
        node_to_idx[node.index()] = i;
    }

    // Union-Find with path compression + union by rank
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<u8> = vec![0; n];

    // Find with path compression (iterative)
    #[inline]
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]]; // path halving
            x = parent[x];
        }
        x
    }

    // Union by rank
    #[inline]
    fn union(parent: &mut [usize], rank: &mut [u8], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra == rb {
            return;
        }
        if rank[ra] < rank[rb] {
            parent[ra] = rb;
        } else if rank[ra] > rank[rb] {
            parent[rb] = ra;
        } else {
            parent[rb] = ra;
            rank[ra] += 1;
        }
    }

    // Process all edges — single pass, no adjacency list needed.
    // Periodic deadline check (every ~1M edges, negligible overhead via bitmask).
    let mut edge_counter: usize = 0;
    for edge in {
        let g = &graph.graph;
        g.edge_references()
    } {
        edge_counter += 1;
        if edge_counter & 0xFFFFF == 0 {
            if let Some(dl) = deadline {
                if Instant::now() > dl {
                    return Err(algorithm_timeout_err());
                }
            }
        }
        if !edge_matches(edge.connection_type()) {
            continue;
        }
        let src_i = node_to_idx[edge.source().index()];
        let tgt_i = node_to_idx[edge.target().index()];
        // Skip edges touching a node outside the universe (e.g. a KNOWS edge
        // to a node type not in `node_types`).
        if src_i == usize::MAX || tgt_i == usize::MAX {
            continue;
        }
        union(&mut parent, &mut rank, src_i, tgt_i);
    }

    // Collect components by root
    let mut component_map: HashMap<usize, Vec<NodeIndex>> = HashMap::new();
    for (i, &node) in nodes.iter().enumerate() {
        let root = find(&mut parent, i);
        component_map.entry(root).or_default().push(node);
    }

    let mut components: Vec<Vec<NodeIndex>> = component_map.into_values().collect();

    // Sort components by size (largest first)
    components.sort_by_key(|b| std::cmp::Reverse(b.len()));

    Ok(components)
}

/// Build the undirected adjacency of a *scoped* subgraph — same scoping rules
/// as [`weakly_connected_components_scoped`]: `node_types` sets the vertex
/// universe (nodes of other types are excluded), `rel_types` limits which edge
/// types contribute an (undirected) link. Returns the universe node list and
/// per-vertex sorted, de-duplicated neighbour lists in compact indices
/// (`0..n`), with self-loops dropped. Shared by the coreness and
/// clustering-coefficient procedures.
fn build_scoped_undirected_adjacency(
    graph: &DirGraph,
    node_types: Option<&[String]>,
    rel_types: Option<&[InternedKey]>,
    deadline: Option<Instant>,
) -> Result<(Vec<NodeIndex>, Vec<Vec<u32>>), String> {
    let edge_matches = |key: InternedKey| -> bool {
        match rel_types {
            Some(keys) => keys.contains(&key),
            None => true,
        }
    };

    let nodes: Vec<NodeIndex> = scoped_universe(graph, node_types, rel_types);

    let n = nodes.len();
    let bound = graph.graph.node_bound();
    let mut node_to_idx = vec![u32::MAX; bound];
    for (i, &node) in nodes.iter().enumerate() {
        node_to_idx[node.index()] = i as u32;
    }

    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut counter = 0usize;
    for edge in {
        let g = &graph.graph;
        g.edge_references()
    } {
        counter += 1;
        if counter & 0xFFFFF == 0 {
            if let Some(dl) = deadline {
                if Instant::now() > dl {
                    return Err(algorithm_timeout_err());
                }
            }
        }
        if !edge_matches(edge.connection_type()) {
            continue;
        }
        let s = node_to_idx[edge.source().index()];
        let t = node_to_idx[edge.target().index()];
        if s == u32::MAX || t == u32::MAX || s == t {
            continue;
        }
        adj[s as usize].push(t);
        adj[t as usize].push(s);
    }
    for list in adj.iter_mut() {
        list.sort_unstable();
        list.dedup();
    }
    Ok((nodes, adj))
}

/// k-core decomposition (coreness): the largest `k` such that each node belongs
/// to a maximal subgraph where every vertex has degree ≥ `k`. Computed over the
/// scoped undirected subgraph via the O(V+E) Batagelj–Zaversnik peeling.
/// Returns `(node, coreness)` per node. Filter `WHERE coreness >= k` for the
/// k-core itself.
pub fn coreness_scoped(
    graph: &DirGraph,
    node_types: Option<&[String]>,
    rel_types: Option<&[InternedKey]>,
    deadline: Option<Instant>,
) -> Result<Vec<(NodeIndex, i64)>, String> {
    // Disk/mapped: stream the scoped neighbours (bounded memory) instead of
    // materialising the whole adjacency. In-memory keeps the materialised path.
    if graph.graph.is_disk() || graph.graph.is_mapped() {
        return coreness_scoped_streaming(graph, node_types, rel_types, deadline);
    }
    let (nodes, adj) = build_scoped_undirected_adjacency(graph, node_types, rel_types, deadline)?;
    let n = nodes.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut deg: Vec<u32> = adj.iter().map(|a| a.len() as u32).collect();
    let max_deg = deg.iter().copied().max().unwrap_or(0) as usize;

    // Bin counts of vertices per degree, turned into bin start offsets.
    let mut bin = vec![0usize; max_deg + 2];
    for &d in &deg {
        bin[d as usize] += 1;
    }
    let mut start = 0usize;
    for slot in bin.iter_mut().take(max_deg + 1) {
        let count = *slot;
        *slot = start;
        start += count;
    }

    // `vert` lists vertices ordered by degree; `pos` is each vertex's index in
    // `vert`. (Bin offsets are consumed as vertices are placed.)
    let mut vert = vec![0usize; n];
    let mut pos = vec![0usize; n];
    {
        let mut binc = bin.clone();
        for v in 0..n {
            let d = deg[v] as usize;
            pos[v] = binc[d];
            vert[pos[v]] = v;
            binc[d] += 1;
        }
    }

    let mut core = vec![0i64; n];
    for i in 0..n {
        let v = vert[i];
        core[v] = deg[v] as i64;
        // Iterating `adj[v]` immutably while mutating the separate bookkeeping
        // vectors (vert/pos/bin/deg) is fine — there are no self-loops, so
        // `deg[v]` is never touched inside the loop.
        let dv = deg[v];
        for &nbr in &adj[v] {
            let u = nbr as usize;
            if deg[u] > dv {
                let du = deg[u] as usize;
                let pu = pos[u];
                let pw = bin[du];
                let w = vert[pw];
                if u != w {
                    vert[pu] = w;
                    vert[pw] = u;
                    pos[u] = pw;
                    pos[w] = pu;
                }
                bin[du] += 1;
                deg[u] -= 1;
            }
        }
    }

    Ok(nodes.into_iter().zip(core).collect())
}

/// Bounded-memory k-core for mapped/disk: the same Batagelj–Zaversnik peeling as
/// `coreness_scoped`, but the per-node neighbour lists are streamed on demand from
/// the CSR (`DedupNeighborSource`) instead of materialising the whole O(edges)
/// adjacency. Resident state is O(nodes) (deg/bin/vert/pos/core + index map);
/// edges stay on mmap. Two streaming sweeps: one to count degrees, one for the
/// peeling. Produces results identical to the materialised path.
fn coreness_scoped_streaming(
    graph: &DirGraph,
    node_types: Option<&[String]>,
    rel_types: Option<&[InternedKey]>,
    deadline: Option<Instant>,
) -> Result<Vec<(NodeIndex, i64)>, String> {
    let nodes = scoped_universe(graph, node_types, rel_types);
    let src = DedupNeighborSource::new(graph, nodes, rel_types.map(|k| k.to_vec()));
    let n = src.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut buf: Vec<u32> = Vec::new();

    // Pass 1: distinct-neighbour degree per node.
    let mut deg: Vec<u32> = vec![0; n];
    for (v, d) in deg.iter_mut().enumerate() {
        if v & 0xFFFFF == 0 {
            if let Some(dl) = deadline {
                if Instant::now() > dl {
                    return Err(algorithm_timeout_err());
                }
            }
        }
        src.neighbors_deduped(v, &mut buf);
        *d = buf.len() as u32;
    }
    let max_deg = deg.iter().copied().max().unwrap_or(0) as usize;

    // Bin counts of vertices per degree, turned into bin start offsets.
    let mut bin = vec![0usize; max_deg + 2];
    for &d in &deg {
        bin[d as usize] += 1;
    }
    let mut start = 0usize;
    for slot in bin.iter_mut().take(max_deg + 1) {
        let count = *slot;
        *slot = start;
        start += count;
    }

    // `vert` lists vertices ordered by degree; `pos` is each vertex's index in
    // `vert`.
    let mut vert = vec![0usize; n];
    let mut pos = vec![0usize; n];
    {
        let mut binc = bin.clone();
        for v in 0..n {
            let d = deg[v] as usize;
            pos[v] = binc[d];
            vert[pos[v]] = v;
            binc[d] += 1;
        }
    }

    // Pass 2: peel in degree order, re-streaming each vertex's neighbours once.
    let mut core = vec![0i64; n];
    for i in 0..n {
        if i & 0xFFFFF == 0 {
            if let Some(dl) = deadline {
                if Instant::now() > dl {
                    return Err(algorithm_timeout_err());
                }
            }
        }
        let v = vert[i];
        core[v] = deg[v] as i64;
        let dv = deg[v];
        src.neighbors_deduped(v, &mut buf);
        for &nbr in &buf {
            let u = nbr as usize;
            if deg[u] > dv {
                let du = deg[u] as usize;
                let pu = pos[u];
                let pw = bin[du];
                let w = vert[pw];
                if u != w {
                    vert[pu] = w;
                    vert[pw] = u;
                    pos[u] = pw;
                    pos[w] = pu;
                }
                bin[du] += 1;
                deg[u] -= 1;
            }
        }
    }

    Ok(src.nodes.into_iter().zip(core).collect())
}

/// Local clustering coefficient per node over the scoped undirected subgraph:
/// `2 * (links among neighbours) / (k * (k-1))`, where `k` is the node's
/// degree. Nodes with degree < 2 get `0.0`. Returns `(node, coefficient)`.
pub fn clustering_coefficient_scoped(
    graph: &DirGraph,
    node_types: Option<&[String]>,
    rel_types: Option<&[InternedKey]>,
    deadline: Option<Instant>,
) -> Result<Vec<(NodeIndex, f64)>, String> {
    let (nodes, adj) = build_scoped_undirected_adjacency(graph, node_types, rel_types, deadline)?;
    let n = nodes.len();
    let mut out = Vec::with_capacity(n);

    for v in 0..n {
        if v & 0xFFFF == 0 {
            if let Some(dl) = deadline {
                if Instant::now() > dl {
                    return Err(algorithm_timeout_err());
                }
            }
        }
        let nbrs = &adj[v];
        let k = nbrs.len();
        if k < 2 {
            out.push((nodes[v], 0.0));
            continue;
        }
        // Count links among neighbours: for each neighbour a, count its
        // neighbours that are also neighbours of v and have a higher index
        // (so each link is counted once). Both lists are sorted → linear merge.
        let mut links: u64 = 0;
        for &a in nbrs {
            links += intersection_count_gt(&adj[a as usize], nbrs, a);
        }
        let kf = k as f64;
        out.push((nodes[v], (2.0 * links as f64) / (kf * (kf - 1.0))));
    }
    Ok(out)
}

/// Count elements common to two sorted slices that are strictly greater than
/// `gt`. Linear merge.
fn intersection_count_gt(a: &[u32], b: &[u32], gt: u32) -> u64 {
    let (mut i, mut j) = (0usize, 0usize);
    let mut count = 0u64;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                if a[i] > gt {
                    count += 1;
                }
                i += 1;
                j += 1;
            }
        }
    }
    count
}

/// Get node info for building Python-friendly path output
pub fn get_node_info(graph: &DirGraph, node_idx: NodeIndex) -> Option<PathNodeInfo> {
    let node = graph.get_node(node_idx)?;
    let node_title = node.title();
    let title_str = match &*node_title {
        Value::String(s) => s.clone(),
        _ => format!("{:?}", &*node_title),
    };
    Some(PathNodeInfo {
        node_type: node.node_type_str(&graph.interner).to_string(),
        title: title_str,
        id: node.id().into_owned(),
    })
}

/// Get information about what connection types link nodes in a path
pub fn get_path_connections(graph: &DirGraph, path: &[NodeIndex]) -> Vec<Option<String>> {
    // Pre-allocate with exact size (one connection per edge = path.len() - 1)
    let mut connections = Vec::with_capacity(path.len().saturating_sub(1));

    for window in path.windows(2) {
        let from = window[0];
        let to = window[1];

        // Find edge between these nodes (either direction)
        let conn_type = graph
            .graph
            .edges(from)
            .find(|e| e.target() == to)
            .map(|e| e.weight().connection_type_str(&graph.interner).to_string())
            .or_else(|| {
                graph
                    .graph
                    .edges(to)
                    .find(|e| e.target() == from)
                    .map(|e| e.weight().connection_type_str(&graph.interner).to_string())
            });

        connections.push(conn_type);
    }

    connections
}

/// Check if two nodes are connected (directly or indirectly)
pub fn are_connected(graph: &DirGraph, source: NodeIndex, target: NodeIndex) -> bool {
    shortest_path(graph, source, target, None, None, None).is_some()
}

/// Calculate the degree (number of connections) for a node
pub fn node_degree(graph: &DirGraph, node: NodeIndex) -> usize {
    let g = &graph.graph;
    g.edges(node).count()
        + g.neighbors_directed(node, petgraph::Direction::Incoming)
            .count()
}

// ============================================================================
// Community Detection
// ============================================================================

#[derive(Debug, Clone)]
pub struct CommunityAssignment {
    pub node_idx: NodeIndex,
    pub community_id: usize,
}

#[derive(Debug)]
pub struct CommunityResult {
    /// Flat partition — the best (final/coarsest) level for hierarchical
    /// algorithms, or the sole partition for single-level ones.
    pub assignments: Vec<CommunityAssignment>,
    pub num_communities: usize,
    pub modularity: f64,
    /// Hierarchical levels, finest → coarsest (`levels.last() == assignments`
    /// for multilevel algorithms). Empty for single-level algorithms, in which
    /// case consumers treat `assignments` as the only level (level 0).
    pub levels: Vec<Vec<CommunityAssignment>>,
}

// ── Shared community-detection primitives (Louvain + Leiden) ──────────────

/// Compact weighted adjacency list: `adj[i]` = neighbours of compact node `i` as
/// `(neighbor_idx, weight)` pairs (self-loops allowed after aggregation).
type Adjacency = Vec<Vec<(usize, f64)>>;

/// One Leiden coarsening step's result: `(refined_partition, k_ref, next_init)`,
/// or `None` when converged / nothing left to coarsen.
type RefineStep = Option<(Vec<usize>, usize, Vec<usize>)>;

/// Build a compact undirected weighted adjacency list from the graph.
///
/// Returns `(nodes, adj, total_weight)` where `adj[i]` is the deduped neighbour
/// list of compact node `i` (parallel edges summed) and `total_weight` (m) is the
/// sum of edge weights, each edge counted once. Shared by every community
/// algorithm so the connection-type filtering and weight resolution live once.
fn build_weighted_adjacency(
    graph: &DirGraph,
    weight_property: Option<&str>,
    connection_types: Option<&[String]>,
    scope: Option<&NodeScope>,
) -> (Vec<NodeIndex>, Adjacency, f64) {
    let nodes: Vec<NodeIndex> = scoped_node_set(graph, scope);
    let n = nodes.len();
    let bound = graph.graph.node_bound();
    let mut node_to_idx = vec![0usize; bound];
    for (i, &node) in nodes.iter().enumerate() {
        node_to_idx[node.index()] = i;
    }

    let interned_ct = intern_connection_types(connection_types);
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut total_weight = 0.0f64;
    for edge in {
        let g = &graph.graph;
        g.edge_references()
    } {
        if let Some(ref types) = interned_ct {
            if !types.iter().any(|t| *t == edge.connection_type()) {
                continue;
            }
        }
        if !edge_in_scope(scope, edge.source(), edge.target()) {
            continue;
        }
        let w = edge_weight(graph, edge.id(), weight_property);
        let src_i = node_to_idx[edge.source().index()];
        let tgt_i = node_to_idx[edge.target().index()];
        adj[src_i].push((tgt_i, w));
        adj[tgt_i].push((src_i, w));
        total_weight += w;
    }
    // Dedup weighted adjacency: merge duplicate neighbours by summing weights.
    for neighbors in &mut adj {
        neighbors.sort_unstable_by_key(|&(idx, _)| idx);
        neighbors.dedup_by(|a, b| {
            if a.0 == b.0 {
                b.1 += a.1;
                true
            } else {
                false
            }
        });
    }
    (nodes, adj, total_weight)
}

/// Per-node weighted adjacency for the community-detection inner loops, abstracted
/// so level 0 can be served either from a materialised in-memory adjacency or
/// **streamed from the CSR** (disk/mapped) without ever holding O(edges) on the
/// heap. The local-move / aggregate / refine primitives are generic over this, so
/// there is one implementation of each algorithm.
///
/// Resident state for the streaming source is O(nodes) (an index map); the edges
/// stay on mmap and are read on demand. Correctness matches the materialised path:
/// parallel edges sum naturally in `comm_weight`, and self-loops are skipped by the
/// existing `neighbor == i` guard.
trait NeighborSource {
    /// Number of (compact) nodes — community ids live in `0..len()`.
    fn len(&self) -> usize;
    /// Total edge weight `m` (each edge counted once).
    fn total_weight(&self) -> f64;
    /// Invoke `f(peer_compact_idx, weight)` for every neighbour of compact node `i`
    /// (both directions; parallel edges yielded separately; self-loops included —
    /// callers skip `peer == i`).
    fn for_each_neighbor(&self, i: usize, f: impl FnMut(usize, f64));
}

/// Materialised adjacency source — wraps a `Vec<Vec<(usize,f64)>>` (the in-memory
/// path and every aggregated level). Zero-cost: `for_each_neighbor` just iterates
/// the slice.
struct MaterializedAdj<'a> {
    adj: &'a [Vec<(usize, f64)>],
    m: f64,
}

impl NeighborSource for MaterializedAdj<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.adj.len()
    }
    #[inline]
    fn total_weight(&self) -> f64 {
        self.m
    }
    #[inline]
    fn for_each_neighbor(&self, i: usize, mut f: impl FnMut(usize, f64)) {
        for &(j, w) in &self.adj[i] {
            f(j, w);
        }
    }
}

/// Streaming adjacency source for mapped/disk graphs — reads each node's incident
/// edges straight from the CSR (`edges_directed`) on demand. Holds only the
/// compact node list + reverse index (O(nodes)); never materialises the O(edges)
/// adjacency. Used for level 0 only; after the first aggregation the super-graph is
/// small and runs through `MaterializedAdj`.
struct CsrSource<'a> {
    graph: &'a DirGraph,
    nodes: Vec<NodeIndex>,
    node_to_idx: Vec<usize>,
    interned_ct: Option<Vec<InternedKey>>,
    weight_property: Option<&'a str>,
    /// Fast path: unweighted + at most one connection type → use the CSR
    /// peer-walk (`iter_peers_filtered`, no `EdgeData` materialisation, weight
    /// implicitly 1.0). `fast_conn` is the single interned type (or `None` for
    /// no filter). Weighted / multi-type fall back to `edges_directed`.
    use_fast: bool,
    fast_conn: Option<u64>,
    m: f64,
}

impl<'a> CsrSource<'a> {
    fn new(
        graph: &'a DirGraph,
        weight_property: Option<&'a str>,
        connection_types: Option<&[String]>,
    ) -> Self {
        let nodes: Vec<NodeIndex> = graph.graph.node_indices().collect();
        let bound = graph.graph.node_bound();
        let mut node_to_idx = vec![0usize; bound];
        for (i, &node) in nodes.iter().enumerate() {
            node_to_idx[node.index()] = i;
        }
        let interned_ct = intern_connection_types(connection_types);
        let multi_type = connection_types.is_some_and(|c| c.len() > 1);
        let use_fast = weight_property.is_none() && !multi_type;
        let fast_conn = if use_fast {
            interned_ct
                .as_ref()
                .and_then(|v| v.first())
                .map(|k| k.as_u64())
        } else {
            None
        };
        // m = total matching edge weight (each edge once), one streaming pass.
        let mut m = 0.0f64;
        for edge in graph.graph.edge_references() {
            if let Some(ref types) = interned_ct {
                if !types.iter().any(|t| *t == edge.connection_type()) {
                    continue;
                }
            }
            m += edge_weight(graph, edge.id(), weight_property);
        }
        Self {
            graph,
            nodes,
            node_to_idx,
            interned_ct,
            weight_property,
            use_fast,
            fast_conn,
            m,
        }
    }
}

impl NeighborSource for CsrSource<'_> {
    fn len(&self) -> usize {
        self.nodes.len()
    }
    fn total_weight(&self) -> f64 {
        self.m
    }
    fn for_each_neighbor(&self, i: usize, mut f: impl FnMut(usize, f64)) {
        use petgraph::Direction;
        let node = self.nodes[i];
        if self.use_fast {
            // CSR peer-walk: no EdgeData materialisation, weight 1.0. On disk
            // this is a direct offsets/targets read (forward + reverse CSR).
            for dir in [Direction::Outgoing, Direction::Incoming] {
                for (peer, _eid) in self
                    .graph
                    .graph
                    .iter_peers_filtered(node, dir, self.fast_conn)
                {
                    f(self.node_to_idx[peer.index()], 1.0);
                }
            }
            return;
        }
        // Weighted / multi-type: materialise edges to read weights + filter.
        for dir in [Direction::Outgoing, Direction::Incoming] {
            for edge in self.graph.graph.edges_directed(node, dir) {
                if let Some(ref types) = self.interned_ct {
                    if !types.iter().any(|t| *t == edge.connection_type()) {
                        continue;
                    }
                }
                let peer = match dir {
                    Direction::Outgoing => edge.target(),
                    Direction::Incoming => edge.source(),
                };
                let w = edge_weight(self.graph, edge.id(), self.weight_property);
                f(self.node_to_idx[peer.index()], w);
            }
        }
    }
}

/// The scoped vertex universe shared by the unweighted scoped algorithms
/// (k-core, clustering coefficient). `node_types` sets the universe (other types
/// excluded); else if `edge_types` is set, the endpoints of matching edges; else
/// all nodes. Extracted so the materialised builder and the streaming source
/// agree on the universe.
fn scoped_universe(
    graph: &DirGraph,
    node_types: Option<&[String]>,
    edge_types: Option<&[InternedKey]>,
) -> Vec<NodeIndex> {
    if let Some(types) = node_types {
        let mut v = Vec::new();
        for t in types {
            if let Some(tn) = graph.type_indices.get(t.as_str()) {
                v.extend(tn.iter());
            }
        }
        v
    } else if let Some(keys) = edge_types {
        let mut seen: HashSet<NodeIndex> = HashSet::new();
        for edge in graph.graph.edge_references() {
            if keys.contains(&edge.connection_type()) {
                seen.insert(edge.source());
                seen.insert(edge.target());
            }
        }
        seen.into_iter().collect()
    } else {
        graph.graph.node_indices().collect()
    }
}

/// Streaming source of **deduplicated** undirected neighbours for the unweighted
/// algorithms (label propagation, k-core) on mapped/disk graphs — fills a
/// reusable buffer with the sorted, de-duped, in-universe compact neighbour
/// indices of a node, read on demand from the CSR (O(nodes) resident, edges stay
/// on mmap). Distinct from `CsrSource` (which sums parallel-edge weights for
/// modularity); these algorithms need *distinct* neighbours. `node_to_idx` uses
/// `u32::MAX` for out-of-universe nodes.
struct DedupNeighborSource<'a> {
    graph: &'a DirGraph,
    nodes: Vec<NodeIndex>,
    node_to_idx: Vec<u32>,
    edge_types: Option<Vec<InternedKey>>,
}

impl<'a> DedupNeighborSource<'a> {
    fn new(
        graph: &'a DirGraph,
        nodes: Vec<NodeIndex>,
        edge_types: Option<Vec<InternedKey>>,
    ) -> Self {
        let bound = graph.graph.node_bound();
        let mut node_to_idx = vec![u32::MAX; bound];
        for (i, &node) in nodes.iter().enumerate() {
            node_to_idx[node.index()] = i as u32;
        }
        Self {
            graph,
            nodes,
            node_to_idx,
            edge_types,
        }
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Fill `buf` with node `v`'s sorted, de-duped, in-universe compact
    /// neighbour indices (both directions; self-loops and out-of-universe peers
    /// dropped). When there's no edge-type filter the cheap CSR peer-walk
    /// (`iter_peers_filtered`) is used; otherwise edges are materialised to read
    /// their type.
    fn neighbors_deduped(&self, v: usize, buf: &mut Vec<u32>) {
        use petgraph::Direction;
        buf.clear();
        let node = self.nodes[v];
        match &self.edge_types {
            None => {
                for dir in [Direction::Outgoing, Direction::Incoming] {
                    for (peer, _e) in self.graph.graph.iter_peers_filtered(node, dir, None) {
                        let p = self.node_to_idx[peer.index()];
                        if p != u32::MAX && p as usize != v {
                            buf.push(p);
                        }
                    }
                }
            }
            Some(types) => {
                for dir in [Direction::Outgoing, Direction::Incoming] {
                    for edge in self.graph.graph.edges_directed(node, dir) {
                        if !types.contains(&edge.connection_type()) {
                            continue;
                        }
                        let peer = match dir {
                            Direction::Outgoing => edge.target(),
                            Direction::Incoming => edge.source(),
                        };
                        let p = self.node_to_idx[peer.index()];
                        if p != u32::MAX && p as usize != v {
                            buf.push(p);
                        }
                    }
                }
            }
        }
        buf.sort_unstable();
        buf.dedup();
    }
}

/// One Louvain local-moving phase over a compact weighted adjacency list.
///
/// Each node starts in its own community and greedily moves to the neighbouring
/// community with the largest modularity gain until no node moves (or
/// `max_iterations`). Returns the (not-yet-renumbered) community id per node.
/// Deterministic: nodes scanned in index order, `> best_delta` tie-break keeps
/// the lowest community.
///
/// Self-loops (`neighbor == i`, produced by aggregation) count toward a node's
/// degree but are excluded when scoring moves — they are the node's own internal
/// weight, not a link to another node. The original graph is loop-free, so this
/// exclusion never fires at level 0 and behaviour there is unchanged.
fn local_move<S: NeighborSource>(
    src: &S,
    init: Option<&[usize]>,
    resolution: f64,
    deadline: Option<Instant>,
) -> Result<Vec<usize>, String> {
    let n = src.len();
    let total_weight = src.total_weight();
    // Start from the given partition (Leiden: previous level's communities) or
    // from singletons (Louvain / level 0). Community ids stay in `0..n`.
    let mut community: Vec<usize> = match init {
        Some(p) => p.to_vec(),
        None => (0..n).collect(),
    };
    if n == 0 || total_weight == 0.0 {
        return Ok(community);
    }

    let mut degree: Vec<f64> = vec![0.0; n];
    for (i, d) in degree.iter_mut().enumerate() {
        src.for_each_neighbor(i, |_, w| *d += w);
    }
    // sigma_tot[c] = sum of degrees of nodes currently in community c.
    let mut sigma_tot: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        sigma_tot[community[i]] += degree[i];
    }

    let m = total_weight;
    let two_m = 2.0 * m;
    let inv_m = 1.0 / m;
    let resolution_over_two_m_sq = resolution / (two_m * two_m);

    let mut comm_weight: Vec<f64> = vec![0.0; n];
    let mut touched_comms: Vec<usize> = Vec::with_capacity(64);

    let max_iterations = 100;
    for _ in 0..max_iterations {
        if let Some(dl) = deadline {
            if Instant::now() > dl {
                return Err(algorithm_timeout_err());
            }
        }

        let mut improved = false;
        for i in 0..n {
            let current_community = community[i];
            let k_i = degree[i];
            let k_i_res = k_i * resolution_over_two_m_sq;

            touched_comms.clear();
            src.for_each_neighbor(i, |neighbor, w| {
                if neighbor == i {
                    return; // self-loop: internal weight, not a move target
                }
                let c = community[neighbor];
                if comm_weight[c] == 0.0 {
                    touched_comms.push(c);
                }
                comm_weight[c] += w;
            });

            let k_i_in_current = comm_weight[current_community];
            let mut best_community = current_community;
            let mut best_delta = 0.0f64;
            for &cand_community in &touched_comms {
                if cand_community == current_community {
                    continue;
                }
                let k_i_in_cand = comm_weight[cand_community];
                let sigma_cand = sigma_tot[cand_community];
                let sigma_curr = sigma_tot[current_community] - k_i;
                let gain_add = k_i_in_cand * inv_m - sigma_cand * k_i_res;
                let loss_remove = k_i_in_current * inv_m - sigma_curr * k_i_res;
                let delta = gain_add - loss_remove;
                if delta > best_delta {
                    best_delta = delta;
                    best_community = cand_community;
                }
            }

            for &c in &touched_comms {
                comm_weight[c] = 0.0;
            }

            if best_community != current_community {
                sigma_tot[current_community] -= k_i;
                sigma_tot[best_community] += k_i;
                community[i] = best_community;
                improved = true;
            }
        }

        if !improved {
            break;
        }
    }
    Ok(community)
}

/// Renumber arbitrary community ids to a contiguous `0..k` range in
/// first-appearance order (deterministic). Returns `(renumbered, k)`.
fn renumber_communities(community: &[usize]) -> (Vec<usize>, usize) {
    let mut id_map: HashMap<usize, usize> = HashMap::new();
    let renumbered: Vec<usize> = community
        .iter()
        .map(|&c| {
            let next = id_map.len();
            *id_map.entry(c).or_insert(next)
        })
        .collect();
    let k = id_map.len();
    (renumbered, k)
}

/// Collapse each community into a super-node, producing the next-level weighted
/// adjacency. `community` must be contiguous `0..k`. Inter-community weights sum
/// into a single super-edge; intra-community weight becomes a self-loop whose
/// weight equals the community's internal weight ×2 — exactly the degree the
/// super-node must carry so modularity is preserved across levels. The total
/// edge weight `m` is invariant under aggregation, so callers keep the original.
fn aggregate_graph<S: NeighborSource>(src: &S, community: &[usize], k: usize) -> Adjacency {
    let mut acc: Vec<HashMap<usize, f64>> = vec![HashMap::new(); k];
    for i in 0..src.len() {
        let ci = community[i];
        src.for_each_neighbor(i, |j, w| {
            let cj = community[j];
            *acc[ci].entry(cj).or_insert(0.0) += w;
        });
    }
    let mut new_adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); k];
    for (ci, entry) in acc.into_iter().enumerate() {
        let mut row: Vec<(usize, f64)> = entry.into_iter().collect();
        row.sort_unstable_by_key(|&(idx, _)| idx); // determinism (HashMap order)
        new_adj[ci] = row;
    }
    new_adj
}

/// Multilevel Louvain driver: `local_move` → `aggregate` until no further
/// merging. Returns one partition per level in **original** node-index space
/// (`0..n0`), finest (level 0) → coarsest (last). Reuses [`local_move`],
/// [`renumber_communities`], [`aggregate_graph`].
fn louvain_levels<S: NeighborSource>(
    level0: &S,
    resolution: f64,
    deadline: Option<Instant>,
) -> Result<Vec<Vec<usize>>, String> {
    let n0 = level0.len();
    let m = level0.total_weight();
    let mut levels: Vec<Vec<usize>> = Vec::new();
    let mut node_to_super: Vec<usize> = (0..n0).collect();

    // ── Level 0 — served by `level0` (streamed from CSR on disk/mapped, or a
    //    MaterializedAdj in-memory). This is the only level that can be O(edges)
    //    big, so it's the one we keep off the heap on disk/mapped. ──
    let moved = local_move(level0, None, resolution, deadline)?;
    let (partition, k) = renumber_communities(&moved);
    for s in node_to_super.iter_mut() {
        *s = partition[*s];
    }
    levels.push(node_to_super.clone());
    if k == n0 {
        // No merging — communities already optimal (e.g. all singletons).
        return Ok(levels);
    }
    let mut adj = aggregate_graph(level0, &partition, k);

    // ── Levels ≥1 — the aggregated super-graph is small, always materialised. ──
    loop {
        let src = MaterializedAdj { adj: &adj, m };
        let moved = local_move(&src, None, resolution, deadline)?;
        let (partition, k) = renumber_communities(&moved);
        if k == adj.len() {
            break;
        }
        for s in node_to_super.iter_mut() {
            *s = partition[*s];
        }
        levels.push(node_to_super.clone());
        adj = aggregate_graph(&src, &partition, k);
    }
    Ok(levels)
}

/// Empty result for a graph with no nodes.
fn empty_community_result() -> CommunityResult {
    CommunityResult {
        assignments: Vec::new(),
        num_communities: 0,
        modularity: 0.0,
        levels: Vec::new(),
    }
}

/// Assemble a [`CommunityResult`] from per-level partitions (compact node-index
/// space, contiguous ids, finest → coarsest). Modularity is computed for the
/// best (last) level on the **original** graph — an independent check that does
/// not depend on the aggregation bookkeeping. Shared by Louvain and Leiden.
fn build_community_result(
    graph: &DirGraph,
    nodes: &[NodeIndex],
    levels: &[Vec<usize>],
    total_weight: f64,
    weight_property: Option<&str>,
) -> CommunityResult {
    let best = levels.last().expect("at least one level");
    let bound = graph.graph.node_bound();
    let mut community_bound: Vec<usize> = vec![0; bound];
    let mut node_exists: Vec<bool> = vec![false; bound];
    for (i, &node) in nodes.iter().enumerate() {
        community_bound[node.index()] = best[i];
        node_exists[node.index()] = true;
    }
    let num_communities = best.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    let modularity = if total_weight == 0.0 {
        0.0
    } else {
        compute_modularity(
            graph,
            &community_bound,
            &node_exists,
            total_weight,
            weight_property,
        )
    };
    let levels_out: Vec<Vec<CommunityAssignment>> = levels
        .iter()
        .map(|lc| {
            lc.iter()
                .enumerate()
                .map(|(i, &c)| CommunityAssignment {
                    node_idx: nodes[i],
                    community_id: c,
                })
                .collect()
        })
        .collect();
    let assignments = levels_out.last().cloned().unwrap_or_default();
    CommunityResult {
        assignments,
        num_communities,
        modularity,
        levels: levels_out,
    }
}

/// Multilevel Louvain modularity optimisation for community detection.
///
/// Builds a weighted adjacency, then runs the full multilevel loop
/// (`local_move` → `aggregate` → repeat), returning the best partition plus the
/// hierarchy of levels (`CommunityResult.levels`, finest → coarsest). Unlike the
/// prior single-level implementation this finds higher-modularity partitions and
/// exposes a community hierarchy.
///
/// @procedure: louvain
/// @procedure: louvain_communities
pub fn louvain_communities(
    graph: &DirGraph,
    weight_property: Option<&str>,
    resolution: f64,
    connection_types: Option<&[String]>,
    scope: Option<&NodeScope>,
    deadline: Option<Instant>,
) -> Result<CommunityResult, String> {
    // Unscoped disk/mapped: stream level 0 from the CSR (O(nodes) heap) — the
    // bounded-memory path for whole-graph runs. In-memory, or any *scoped* run:
    // materialise the (scope-bounded) adjacency. build_weighted_adjacency reads
    // through GraphRead (scoped_node_set / edge_in_scope), exactly like
    // connected_components' scoped path, so scoping works on every storage mode.
    let (nodes, levels, m) =
        if (graph.graph.is_disk() || graph.graph.is_mapped()) && scope.is_none() {
            let nodes: Vec<NodeIndex> = graph.graph.node_indices().collect();
            if nodes.is_empty() {
                return Ok(empty_community_result());
            }
            let src = CsrSource::new(graph, weight_property, connection_types);
            let m = src.total_weight();
            let levels = louvain_levels(&src, resolution, deadline)?;
            (nodes, levels, m)
        } else {
            let (nodes, adj, m) =
                build_weighted_adjacency(graph, weight_property, connection_types, scope);
            if nodes.is_empty() {
                return Ok(empty_community_result());
            }
            let src = MaterializedAdj { adj: &adj, m };
            let levels = louvain_levels(&src, resolution, deadline)?;
            (nodes, levels, m)
        };
    Ok(build_community_result(
        graph,
        &nodes,
        &levels,
        m,
        weight_property,
    ))
}

// ── Leiden ────────────────────────────────────────────────────────────────

/// Refine a partition by splitting each community into its **connected
/// components** (using only in-community edges). Returns `(refined, k_ref)` with
/// contiguous ids; every refined sub-community is therefore connected. This is
/// the property Louvain lacks — a local-move community can be internally
/// disconnected. Deterministic: components discovered by index-ordered BFS over
/// the (sorted) adjacency. Self-loops are ignored.
fn refine_connected<S: NeighborSource>(src: &S, partition: &[usize]) -> (Vec<usize>, usize) {
    let n = src.len();
    let mut refined = vec![usize::MAX; n];
    let mut next_id = 0usize;
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..n {
        if refined[start] != usize::MAX {
            continue;
        }
        let comm = partition[start];
        let id = next_id;
        next_id += 1;
        refined[start] = id;
        stack.push(start);
        while let Some(u) = stack.pop() {
            src.for_each_neighbor(u, |v, _w| {
                if v != u && partition[v] == comm && refined[v] == usize::MAX {
                    refined[v] = id;
                    stack.push(v);
                }
            });
        }
    }
    (refined, next_id)
}

/// Multilevel **Leiden** driver. Like [`louvain_levels`] but inserts a
/// refinement step: each level's local-move partition `P` is split into
/// connected components (`refine_connected`), aggregation is done on those
/// components, and the next level's local move *starts from* `P` (so it can only
/// re-merge components along real edges). The net effect is the same hierarchy
/// shape as Louvain but with a **well-connectedness guarantee** on the merged
/// communities — Leiden's defining property over Louvain.
///
/// This is a *deterministic* Leiden variant: the refinement guarantees connected
/// communities (the headline fix) but omits the reference implementation's
/// randomised within-community modularity sub-refinement, keeping results
/// reproducible — a kglite value. Returns per-level partitions in original
/// node-index space, finest → coarsest.
fn leiden_levels<S: NeighborSource>(
    level0: &S,
    resolution: f64,
    deadline: Option<Instant>,
) -> Result<Vec<Vec<usize>>, String> {
    let n0 = level0.len();
    let m = level0.total_weight();
    let mut levels: Vec<Vec<usize>> = Vec::new();
    let mut node_to_super: Vec<usize> = (0..n0).collect();

    // One coarsening step shared by level 0 (`src` = `level0`) and levels ≥1
    // (`src` = a MaterializedAdj over the small super-graph). Records this level's
    // partition, then returns the refined component partition for aggregation, or
    // `None` if we've converged / can't coarsen further.
    fn step<S: NeighborSource>(
        src: &S,
        n0: usize,
        node_to_super: &[usize],
        init: Option<&[usize]>,
        levels: &mut Vec<Vec<usize>>,
        resolution: f64,
        deadline: Option<Instant>,
    ) -> Result<RefineStep, String> {
        let moved = local_move(src, init, resolution, deadline)?;
        let (p, _kp) = renumber_communities(&moved);
        let raw_level: Vec<usize> = (0..n0).map(|o| p[node_to_super[o]]).collect();
        let (level_norm, _) = renumber_communities(&raw_level);
        if levels.last() == Some(&level_norm) {
            return Ok(None); // partition stable → converged
        }
        levels.push(level_norm);
        let (refined, k_ref) = refine_connected(src, &p);
        if k_ref == src.len() {
            return Ok(None); // can't coarsen further
        }
        // Each refined component starts in its P-community next round, so local
        // move can only re-merge components along real edges (connectivity guard).
        let mut next_init = vec![0usize; k_ref];
        for s in 0..src.len() {
            next_init[refined[s]] = p[s];
        }
        Ok(Some((refined, k_ref, next_init)))
    }

    // ── Level 0 (streamed for disk/mapped) ──
    let Some((refined, k_ref, next_init)) = step(
        level0,
        n0,
        &node_to_super,
        None,
        &mut levels,
        resolution,
        deadline,
    )?
    else {
        if levels.is_empty() {
            levels.push((0..n0).collect());
        }
        return Ok(levels);
    };
    let mut adj = aggregate_graph(level0, &refined, k_ref);
    for s in node_to_super.iter_mut() {
        *s = refined[*s];
    }
    let mut init = next_init;

    // ── Levels ≥1 (materialised super-graph) ──
    loop {
        let src = MaterializedAdj { adj: &adj, m };
        match step(
            &src,
            n0,
            &node_to_super,
            Some(&init),
            &mut levels,
            resolution,
            deadline,
        )? {
            None => break,
            Some((refined, k_ref, next_init)) => {
                adj = aggregate_graph(&src, &refined, k_ref);
                for s in node_to_super.iter_mut() {
                    *s = refined[*s];
                }
                init = next_init;
            }
        }
    }
    Ok(levels)
}

/// Leiden community detection — multilevel modularity optimisation with a
/// refinement phase that guarantees **well-connected communities** (Louvain can
/// return internally-disconnected ones). Same parameters and output shape as
/// [`louvain_communities`], including the `CommunityResult.levels` hierarchy.
///
/// @procedure: leiden
/// @procedure: leiden_communities
pub fn leiden_communities(
    graph: &DirGraph,
    weight_property: Option<&str>,
    resolution: f64,
    connection_types: Option<&[String]>,
    scope: Option<&NodeScope>,
    deadline: Option<Instant>,
) -> Result<CommunityResult, String> {
    let (nodes, levels, m) =
        if (graph.graph.is_disk() || graph.graph.is_mapped()) && scope.is_none() {
            let nodes: Vec<NodeIndex> = graph.graph.node_indices().collect();
            if nodes.is_empty() {
                return Ok(empty_community_result());
            }
            let src = CsrSource::new(graph, weight_property, connection_types);
            let m = src.total_weight();
            let levels = leiden_levels(&src, resolution, deadline)?;
            (nodes, levels, m)
        } else {
            let (nodes, adj, m) =
                build_weighted_adjacency(graph, weight_property, connection_types, scope);
            if nodes.is_empty() {
                return Ok(empty_community_result());
            }
            let src = MaterializedAdj { adj: &adj, m };
            let levels = leiden_levels(&src, resolution, deadline)?;
            (nodes, levels, m)
        };
    Ok(build_community_result(
        graph,
        &nodes,
        &levels,
        m,
        weight_property,
    ))
}

/// Label propagation for community detection.
///
/// Each node adopts the most frequent label among its neighbors.
/// Converges when no node changes its label.
/// Optimized with pre-built adjacency list and Vec-based label counting.
///
/// @procedure: label_propagation
pub fn label_propagation(
    graph: &DirGraph,
    max_iterations: usize,
    connection_types: Option<&[String]>,
    scope: Option<&NodeScope>,
    deadline: Option<Instant>,
) -> Result<CommunityResult, String> {
    // Unscoped disk/mapped: stream the deduped neighbours (bounded memory). A
    // *scoped* run falls through to the materialised path below — the scoped
    // subgraph is bounded and scoped_node_set reads through GraphRead, so it
    // works on every storage mode (see louvain_communities).
    if (graph.graph.is_disk() || graph.graph.is_mapped()) && scope.is_none() {
        return label_propagation_streaming(graph, max_iterations, connection_types, deadline);
    }

    let nodes: Vec<NodeIndex> = scoped_node_set(graph, scope);
    let n = nodes.len();

    if n == 0 {
        return Ok(empty_community_result());
    }

    // Build compact index mapping
    let bound = graph.graph.node_bound();
    let mut node_to_idx = vec![0usize; bound];
    for (i, &node) in nodes.iter().enumerate() {
        node_to_idx[node.index()] = i;
    }

    // Pre-build undirected adjacency list (both directions)
    let interned_ct = intern_connection_types(connection_types);
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for edge in {
        let g = &graph.graph;
        g.edge_references()
    } {
        if let Some(ref types) = interned_ct {
            if !types.iter().any(|t| *t == edge.connection_type()) {
                continue;
            }
        }
        if !edge_in_scope(scope, edge.source(), edge.target()) {
            continue;
        }
        let src_i = node_to_idx[edge.source().index()];
        let tgt_i = node_to_idx[edge.target().index()];
        adj[src_i].push(tgt_i);
        adj[tgt_i].push(src_i);
    }
    // Dedup undirected adjacency (handles bidirectional edges A→B + B→A)
    for neighbors in &mut adj {
        neighbors.sort_unstable();
        neighbors.dedup();
    }

    // Initialize: each node gets a unique label (0..n)
    let mut labels: Vec<usize> = (0..n).collect();

    // Vec-based label counting (reused across iterations)
    // label_count[label] = count for that label among neighbors
    // We use a sparse approach: track which labels were touched and reset only those
    let mut label_count: Vec<usize> = vec![0; n];
    let mut touched_labels: Vec<usize> = Vec::with_capacity(64);

    for _ in 0..max_iterations {
        // Timeout check each iteration — error rather than return partial.
        if let Some(dl) = deadline {
            if Instant::now() > dl {
                return Err(algorithm_timeout_err());
            }
        }

        let mut changed = false;

        for i in 0..n {
            let neighbors = &adj[i];
            if neighbors.is_empty() {
                continue; // isolated node keeps its label
            }

            // Count neighbor labels using Vec (O(1) per access)
            touched_labels.clear();
            for &neighbor in neighbors {
                let lbl = labels[neighbor];
                if label_count[lbl] == 0 {
                    touched_labels.push(lbl);
                }
                label_count[lbl] += 1;
            }

            // Find most frequent label
            let mut best_label = labels[i];
            let mut best_count = 0;
            for &lbl in &touched_labels {
                if label_count[lbl] > best_count {
                    best_count = label_count[lbl];
                    best_label = lbl;
                }
            }

            // Reset counts for next node (only touched entries)
            for &lbl in &touched_labels {
                label_count[lbl] = 0;
            }

            if best_label != labels[i] {
                labels[i] = best_label;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    // label propagation is single-level: `levels` empty ⇒ consumers treat
    // `assignments` as the only level.
    Ok(label_prop_result(graph, &nodes, &labels))
}

/// Build the single-level `CommunityResult` shared by both the in-memory and the
/// streaming label-propagation paths: renumber labels contiguously, emit one
/// assignment per node, and compute modularity over the bound-sized label array.
fn label_prop_result(graph: &DirGraph, nodes: &[NodeIndex], labels: &[usize]) -> CommunityResult {
    let bound = graph.graph.node_bound();
    let mut labels_bound: Vec<usize> = vec![0; bound];
    let mut node_exists: Vec<bool> = vec![false; bound];
    for (i, &node) in nodes.iter().enumerate() {
        labels_bound[node.index()] = labels[i];
        node_exists[node.index()] = true;
    }

    // Renumber labels to be contiguous in first-seen order.
    let mut id_map: HashMap<usize, usize> = HashMap::new();
    for &lbl in labels {
        let next_id = id_map.len();
        id_map.entry(lbl).or_insert(next_id);
    }

    let assignments: Vec<CommunityAssignment> = nodes
        .iter()
        .enumerate()
        .map(|(i, &idx)| CommunityAssignment {
            node_idx: idx,
            community_id: *id_map.get(&labels[i]).unwrap(),
        })
        .collect();

    let total_weight = graph.graph.edge_count() as f64;
    let num_communities = id_map.len();
    let modularity = compute_modularity(graph, &labels_bound, &node_exists, total_weight, None);

    CommunityResult {
        assignments,
        num_communities,
        modularity,
        levels: Vec::new(),
    }
}

/// Bounded-memory label propagation for mapped/disk: identical semantics to the
/// in-memory path (each node adopts the most frequent label among its distinct
/// neighbours; isolated nodes keep their own label; deterministic first-seen tie
/// break) but neighbour lists are streamed on demand from the CSR
/// (`DedupNeighborSource`) each iteration rather than materialised. Resident
/// state is O(nodes); edges stay on mmap.
fn label_propagation_streaming(
    graph: &DirGraph,
    max_iterations: usize,
    connection_types: Option<&[String]>,
    deadline: Option<Instant>,
) -> Result<CommunityResult, String> {
    let nodes: Vec<NodeIndex> = graph.graph.node_indices().collect();
    if nodes.is_empty() {
        return Ok(empty_community_result());
    }
    let interned_ct = intern_connection_types(connection_types);
    let src = DedupNeighborSource::new(graph, nodes, interned_ct);
    let n = src.len();

    // Each node starts with a unique label (its compact index).
    let mut labels: Vec<usize> = (0..n).collect();
    let mut label_count: Vec<usize> = vec![0; n];
    let mut touched_labels: Vec<usize> = Vec::with_capacity(64);
    let mut buf: Vec<u32> = Vec::new();

    for _ in 0..max_iterations {
        if let Some(dl) = deadline {
            if Instant::now() > dl {
                return Err(algorithm_timeout_err());
            }
        }

        let mut changed = false;
        for i in 0..n {
            src.neighbors_deduped(i, &mut buf);
            if buf.is_empty() {
                continue; // isolated node keeps its label
            }

            touched_labels.clear();
            for &neighbor in &buf {
                let lbl = labels[neighbor as usize];
                if label_count[lbl] == 0 {
                    touched_labels.push(lbl);
                }
                label_count[lbl] += 1;
            }

            let mut best_label = labels[i];
            let mut best_count = 0;
            for &lbl in &touched_labels {
                if label_count[lbl] > best_count {
                    best_count = label_count[lbl];
                    best_label = lbl;
                }
            }

            for &lbl in &touched_labels {
                label_count[lbl] = 0;
            }

            if best_label != labels[i] {
                labels[i] = best_label;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    Ok(label_prop_result(graph, &src.nodes, &labels))
}

/// Get edge weight from a property, or 1.0 if not specified.
pub(crate) fn edge_weight(
    graph: &DirGraph,
    edge_id: petgraph::graph::EdgeIndex,
    weight_property: Option<&str>,
) -> f64 {
    if let Some(prop) = weight_property {
        let g = &graph.graph;
        if let Some(edge_data) = g.edge_weight(edge_id) {
            if let Some(val) = edge_data.get_property(prop) {
                return crate::graph::core::value_operations::value_to_f64(val).unwrap_or(1.0);
            }
        }
    }
    1.0
}

/// Result of a weighted path-finding operation. Distinct from [`PathResult`]
/// (which carries an integer hop count) so the f64 weight survives a round
/// trip through the Python layer without coercion.
#[derive(Debug, Clone)]
pub struct WeightedPathResult {
    pub path: Vec<NodeIndex>,
    pub weight: f64,
}

/// Dijkstra-based weighted shortest path. Treats the graph as undirected
/// (mirrors [`shortest_path`]) and reads `weight_property` from each edge,
/// defaulting to 1.0 when the property is absent or non-numeric — the same
/// fallback Louvain uses for its weighted-adjacency build.
///
/// Returns `None` when no path exists, when the deadline expires, or when
/// any traversed edge has a negative weight (Dijkstra requires non-negative
/// edges; the procedure errs on the side of returning no path rather than
/// silently producing wrong answers).
pub fn shortest_path_weighted(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
    weight_property: &str,
    connection_types: Option<&[String]>,
    via_types: Option<&[String]>,
    deadline: Option<Instant>,
) -> Option<WeightedPathResult> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    if source == target {
        return Some(WeightedPathResult {
            path: vec![source],
            weight: 0.0,
        });
    }

    let via_set: Option<HashSet<&str>> =
        via_types.map(|vt| vt.iter().map(|s| s.as_str()).collect());
    let interned = intern_connection_types(connection_types);
    let conn_filter = interned.as_deref();

    let node_bound = graph.graph.node_bound();
    let mut dist: Vec<f64> = vec![f64::INFINITY; node_bound];
    let mut parent: Vec<u32> = vec![u32::MAX; node_bound];
    dist[source.index()] = 0.0;

    // Min-heap keyed on (distance, node_index). f64 isn't Ord, so wrap in a
    // newtype that flips the order to make BinaryHeap a min-heap.
    #[derive(PartialEq)]
    struct State(f64, usize);
    impl Eq for State {}
    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            other
                .0
                .partial_cmp(&self.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| self.1.cmp(&other.1))
        }
    }
    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut heap: BinaryHeap<State> = BinaryHeap::new();
    heap.push(State(0.0, source.index()));

    let mut visit_count = 0u32;
    while let Some(State(d, current_idx)) = heap.pop() {
        // Stale entry — already processed via a shorter path.
        if d > dist[current_idx] {
            continue;
        }
        if current_idx == target.index() {
            // Reconstruct path
            let mut path = Vec::with_capacity(16);
            let mut idx = current_idx;
            while idx != source.index() {
                path.push(NodeIndex::new(idx));
                idx = parent[idx] as usize;
            }
            path.push(source);
            path.reverse();
            return Some(WeightedPathResult { path, weight: d });
        }

        visit_count += 1;
        if visit_count.is_multiple_of(1000) {
            if let Some(dl) = deadline {
                if Instant::now() > dl {
                    return None;
                }
            }
        }

        let current = NodeIndex::new(current_idx);
        for edge in graph
            .graph
            .edges_directed(current, petgraph::Direction::Outgoing)
            .chain(
                graph
                    .graph
                    .edges_directed(current, petgraph::Direction::Incoming),
            )
        {
            if let Some(types) = conn_filter {
                if !types.iter().any(|t| *t == edge.connection_type()) {
                    continue;
                }
            }
            let neighbor = if edge.source() == current {
                edge.target()
            } else {
                edge.source()
            };
            let n_idx = neighbor.index();
            if n_idx != target.index() && !node_passes_via_filter(graph, neighbor, &via_set) {
                continue;
            }
            let w = edge_weight(graph, edge.id(), Some(weight_property));
            if w < 0.0 {
                // Dijkstra is invalid with negative weights — abort.
                return None;
            }
            let next = d + w;
            if next < dist[n_idx] {
                dist[n_idx] = next;
                parent[n_idx] = current_idx as u32;
                heap.push(State(next, n_idx));
            }
        }
    }
    None
}

/// Lightweight variant of [`shortest_path_weighted`] that returns only the
/// total weight without reconstructing the path.
pub fn shortest_path_cost_weighted(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
    weight_property: &str,
    connection_types: Option<&[String]>,
    via_types: Option<&[String]>,
    deadline: Option<Instant>,
) -> Option<f64> {
    shortest_path_weighted(
        graph,
        source,
        target,
        weight_property,
        connection_types,
        via_types,
        deadline,
    )
    .map(|r| r.weight)
}

/// Sum of edge weights for all nodes in a community.
/// Compute Newman modularity: Q = (1/2m) * sum [ A_ij - k_i*k_j/(2m) ] * delta(c_i, c_j)
fn compute_modularity(
    graph: &DirGraph,
    community: &[usize],
    node_exists: &[bool],
    total_weight: f64,
    weight_property: Option<&str>,
) -> f64 {
    if total_weight == 0.0 {
        return 0.0;
    }

    let two_m = 2.0 * total_weight;
    let mut q = 0.0f64;

    // Compute degree (sum of edge weights) for each node
    let g = &graph.graph;
    let bound = g.node_bound();
    let mut degrees: Vec<f64> = vec![0.0; bound];
    for node_idx in g.node_indices() {
        let i = node_idx.index();
        if !node_exists[i] {
            continue;
        }
        for edge in g.edges(node_idx) {
            degrees[i] += edge_weight(graph, edge.id(), weight_property);
        }
        for edge in g.edges_directed(node_idx, petgraph::Direction::Incoming) {
            degrees[i] += edge_weight(graph, edge.id(), weight_property);
        }
    }

    // Sum over all edges
    for edge in {
        let g = &graph.graph;
        g.edge_references()
    } {
        let u = edge.source().index();
        let v = edge.target().index();
        let w = edge_weight(graph, edge.id(), weight_property);

        if community[u] == community[v] {
            q += w - degrees[u] * degrees[v] / two_m;
        }
    }

    q / two_m
}

#[cfg(test)]
#[path = "graph_algorithms_tests.rs"]
mod tests;
