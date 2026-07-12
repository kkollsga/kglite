// src/graph/graph_algorithms.rs
//! Graph algorithms module providing path finding and connectivity analysis.

use super::Interrupt;
use crate::datatypes::values::Value;
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::GraphRead;
use petgraph::algo::kosaraju_scc;
use petgraph::graph::NodeIndex;
use std::collections::{HashMap, HashSet};

// Centrality algorithms moved to the sibling `centrality` module to keep this
// file under the god-file ceiling. Re-exported so existing
// `graph_algorithms::{betweenness_centrality, pagerank, degree_centrality,
// closeness_centrality, CentralityResult}` paths keep resolving.
pub use super::centrality::*;
// Community detection lives in a sibling module, but the historical
// `graph_algorithms::*` paths remain the public compatibility surface.
pub use super::community::*;
use super::community::{scoped_universe, DedupNeighborSource};

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
    deadline: Interrupt,
) -> Option<PathResult> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
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

/// Enumerate ALL shortest paths between two anchored endpoints — the
/// `allShortestPaths(...)` Cypher form. Unlike [`shortest_path`] (one
/// minimal path), this returns every path of the minimal length.
/// Undirected. Capped at `max_paths` to bound pathological fan-out;
/// honours `deadline`.
pub fn all_shortest_paths(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
    connection_types: Option<&[String]>,
    deadline: Interrupt,
    max_paths: usize,
) -> Vec<PathResult> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    all_shortest_paths_impl(
        graph,
        source,
        target,
        connection_types,
        deadline,
        max_paths,
        false,
    )
}

/// Directed variant of [`all_shortest_paths`] — follows outgoing edges
/// only (mirrors [`shortest_path_directed`]).
pub fn all_shortest_paths_directed(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
    connection_types: Option<&[String]>,
    deadline: Interrupt,
    max_paths: usize,
) -> Vec<PathResult> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    all_shortest_paths_impl(
        graph,
        source,
        target,
        connection_types,
        deadline,
        max_paths,
        true,
    )
}

fn all_shortest_paths_impl(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
    connection_types: Option<&[String]>,
    deadline: Interrupt,
    max_paths: usize,
    directed: bool,
) -> Vec<PathResult> {
    use std::collections::HashMap;

    if source == target {
        return vec![PathResult {
            path: vec![source],
            cost: 0,
        }];
    }

    let interned = intern_connection_types(connection_types);
    let interned_ref = interned.as_deref();

    // Level-synchronous BFS recording EVERY minimal-distance predecessor
    // of each node (a predecessor DAG), so all shortest paths can be
    // reconstructed. Frontier nodes are at `level - 1`; their newly seen
    // neighbours land at `level`.
    let mut dist: HashMap<NodeIndex, usize> = HashMap::new();
    let mut preds: HashMap<NodeIndex, Vec<NodeIndex>> = HashMap::new();
    dist.insert(source, 0);
    let mut frontier = vec![source];
    let mut level = 0usize;
    let mut found = false;
    let mut visit_count = 0u32;

    while !frontier.is_empty() && !found {
        level += 1;
        let mut next: Vec<NodeIndex> = Vec::new();
        for &u in &frontier {
            visit_count += 1;
            if visit_count.is_multiple_of(1000) && deadline.exceeded() {
                return Vec::new();
            }
            let neighbors = if directed {
                filtered_neighbors_outgoing(graph, u, interned_ref)
            } else {
                filtered_neighbors_undirected(graph, u, interned_ref)
            };
            for v in neighbors {
                match dist.get(&v).copied() {
                    None => {
                        dist.insert(v, level);
                        preds.entry(v).or_default().push(u);
                        if v == target {
                            found = true;
                        }
                        next.push(v);
                    }
                    // Another equally-short predecessor seen this level.
                    Some(dv) if dv == level => {
                        preds.entry(v).or_default().push(u);
                    }
                    _ => {}
                }
            }
        }
        frontier = next;
    }

    let Some(&d) = dist.get(&target) else {
        return Vec::new();
    };

    // Back-track target → source over the predecessor DAG, enumerating
    // every distinct minimal path. Capped to bound fan-out.
    let mut results: Vec<PathResult> = Vec::new();
    let mut stack: Vec<Vec<NodeIndex>> = vec![vec![target]];
    while let Some(path_rev) = stack.pop() {
        if results.len() >= max_paths {
            break;
        }
        let head = *path_rev.last().expect("path_rev is never empty");
        if head == source {
            let mut p = path_rev.clone();
            p.reverse();
            results.push(PathResult { path: p, cost: d });
            continue;
        }
        if let Some(ps) = preds.get(&head) {
            for &pnode in ps {
                let mut np = path_rev.clone();
                np.push(pnode);
                stack.push(np);
            }
        }
    }

    results
}

/// Find the shortest path LENGTH between two nodes using undirected BFS.
/// Only returns the hop count, avoiding parent tracking and path reconstruction.
/// Uses level-by-level BFS to avoid per-node distance tracking.
pub fn shortest_path_cost(graph: &DirGraph, source: NodeIndex, target: NodeIndex) -> Option<usize> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
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
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
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
    deadline: Interrupt,
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
        if visit_count.is_multiple_of(1000) && deadline.exceeded() {
            {
                return None;
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
    deadline: Interrupt,
) -> Option<PathResult> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
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
        if visit_count.is_multiple_of(1000) && deadline.exceeded() {
            {
                return None;
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
    deadline: Interrupt,
) -> Vec<Vec<NodeIndex>> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
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
    deadline: Interrupt,
) {
    // Early termination when result limit is hit
    if let Some(max) = max_results {
        if results.len() >= max {
            return;
        }
    }

    // Timeout check at each recursive entry
    if deadline.exceeded() {
        {
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
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    // For disk mode, fall back to weakly_connected_components since
    // kosaraju_scc requires petgraph trait bounds.
    if GraphRead::is_disk(&graph.graph) {
        return weakly_connected_components(graph, Interrupt::default())
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
    deadline: Interrupt,
) -> Result<Vec<Vec<NodeIndex>>, String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
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
    deadline: Interrupt,
) -> Result<Vec<Vec<NodeIndex>>, String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
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
        if edge_counter & 0xFFFFF == 0 && deadline.exceeded() {
            {
                return Err(algorithm_timeout_err());
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
    deadline: Interrupt,
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
        if counter & 0xFFFFF == 0 && deadline.exceeded() {
            {
                return Err(algorithm_timeout_err());
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
    deadline: Interrupt,
) -> Result<Vec<(NodeIndex, i64)>, String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
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
    deadline: Interrupt,
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
        if v & 0xFFFFF == 0 && deadline.exceeded() {
            {
                return Err(algorithm_timeout_err());
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
        if i & 0xFFFFF == 0 && deadline.exceeded() {
            {
                return Err(algorithm_timeout_err());
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

/// Dependency-frontier / "ready set": over a DAG on edge type `E`, return the
/// nodes whose dependencies are all satisfied (the nodes ready to be worked
/// next). A node's **dependencies** are its outgoing-`E` neighbours — so
/// `(task)-[:DEPENDS_ON]->(dependency)` reads naturally: `task` is ready once
/// every `dependency` it points to is in the `done` set. A node already in
/// `done` is excluded (it's finished, not "ready"); a node with no
/// dependencies (a root) is ready as soon as it isn't done.
///
/// `done` is precomputed by the caller (the CALL dispatcher evaluates the
/// `done` predicate per node). `node_types` limits which nodes are *emitted*;
/// dependencies are followed regardless of type. Returns
/// `(node, dependency_count)` where the count is how many `E`-dependencies the
/// ready node had (all satisfied). General graph op (build ordering,
/// scheduling, dataflow), not a Task concept.
pub fn ready_set_scoped(
    graph: &DirGraph,
    node_types: Option<&[String]>,
    rel_types: Option<&[InternedKey]>,
    done: &HashSet<NodeIndex>,
    deadline: Interrupt,
) -> Result<Vec<(NodeIndex, i64)>, String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    // Candidate nodes to emit: union of the requested types, or every node.
    let candidates: Vec<NodeIndex> = match node_types {
        Some(types) => {
            let mut v = Vec::new();
            for t in types {
                if let Some(idxs) = graph.type_indices.get(t.as_str()) {
                    v.extend(idxs.iter());
                }
            }
            v
        }
        None => graph.graph.node_indices().collect(),
    };

    let mut ready = Vec::new();
    for (i, node) in candidates.into_iter().enumerate() {
        if i & 0xFFFF == 0 && deadline.exceeded() {
            return Err("Query interrupted".to_string());
        }
        // Already done → not part of the ready frontier.
        if done.contains(&node) {
            continue;
        }
        let deps = filtered_neighbors_outgoing(graph, node, rel_types);
        if deps.iter().all(|d| done.contains(d)) {
            ready.push((node, deps.len() as i64));
        }
    }
    Ok(ready)
}

/// Local clustering coefficient per node over the scoped undirected subgraph:
/// `2 * (links among neighbours) / (k * (k-1))`, where `k` is the node's
/// degree. Nodes with degree < 2 get `0.0`. Returns `(node, coefficient)`.
pub fn clustering_coefficient_scoped(
    graph: &DirGraph,
    node_types: Option<&[String]>,
    rel_types: Option<&[InternedKey]>,
    deadline: Interrupt,
) -> Result<Vec<(NodeIndex, f64)>, String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let (nodes, adj) = build_scoped_undirected_adjacency(graph, node_types, rel_types, deadline)?;
    let n = nodes.len();
    let mut out = Vec::with_capacity(n);

    for v in 0..n {
        if v & 0xFFFF == 0 && deadline.exceeded() {
            {
                return Err(algorithm_timeout_err());
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

/// Global triangle count + transitivity over the scoped undirected subgraph.
///
/// Returns `(triangles, transitivity)`:
/// - `triangles` — the number of distinct triangles (3-cliques).
/// - `transitivity` — the global clustering coefficient
///   `3 * triangles / connected_triples` (a connected triple is a path of
///   length 2), in `[0, 1]`; `0.0` when there are no connected triples.
///
/// Shares the adjacency build + sorted-neighbour intersection counting with
/// [`clustering_coefficient_scoped`]. The per-node "edges among my
/// neighbours" count is summed across all nodes; since each triangle is seen
/// once at each of its three corners, that raw sum is `3 * triangles` — so
/// `triangles = sum / 3`, and dividing the sum by the connected-triple count
/// yields transitivity directly (the factor of 3 cancels).
pub fn triangle_count_scoped(
    graph: &DirGraph,
    node_types: Option<&[String]>,
    rel_types: Option<&[InternedKey]>,
    deadline: Interrupt,
) -> Result<(u64, f64), String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let (nodes, adj) = build_scoped_undirected_adjacency(graph, node_types, rel_types, deadline)?;
    let n = nodes.len();
    // `link_sum` = Σ_v (edges among v's neighbours) = 3 × triangles.
    // `triple_sum` = Σ_v C(deg(v), 2) = number of connected triples.
    let mut link_sum: u64 = 0;
    let mut triple_sum: u64 = 0;
    for (v, nbrs) in adj.iter().enumerate().take(n) {
        if v & 0xFFFF == 0 && deadline.exceeded() {
            return Err(algorithm_timeout_err());
        }
        let k = nbrs.len() as u64;
        if k < 2 {
            continue;
        }
        triple_sum += k * (k - 1) / 2;
        for &a in nbrs {
            link_sum += intersection_count_gt(&adj[a as usize], nbrs, a);
        }
    }
    let triangles = link_sum / 3;
    let transitivity = if triple_sum > 0 {
        link_sum as f64 / triple_sum as f64
    } else {
        0.0
    };
    Ok((triangles, transitivity))
}

/// Maximum scoped-subgraph size for the all-pairs eccentricity / diameter
/// procedures. They run a BFS from every node — O(V·(V+E)) — so they are a
/// small/medium-graph feature; beyond this the procedure errors with guidance
/// to scope down rather than churning for minutes.
const MAX_ECCENTRICITY_NODES: usize = 20_000;

/// Per-node eccentricity over the scoped undirected subgraph: the greatest
/// shortest-path distance from a node to any other node in its connected
/// component. Returns `(node, eccentricity)`; an isolated node has
/// eccentricity 0. Distances ignore unreachable nodes, so the result is
/// well-defined on a disconnected graph (unlike NetworkX, which errors).
///
/// All-pairs (a BFS per node, O(V·(V+E))) — capped at
/// [`MAX_ECCENTRICITY_NODES`] scoped nodes; narrow with
/// `{node_type, relationship}` for larger graphs.
pub fn eccentricity_scoped(
    graph: &DirGraph,
    node_types: Option<&[String]>,
    rel_types: Option<&[InternedKey]>,
    deadline: Interrupt,
) -> Result<Vec<(NodeIndex, i64)>, String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    use std::collections::VecDeque;
    let (nodes, adj) = build_scoped_undirected_adjacency(graph, node_types, rel_types, deadline)?;
    let n = nodes.len();
    if n > MAX_ECCENTRICITY_NODES {
        return Err(format!(
            "eccentricity/diameter is an all-pairs O(V·(V+E)) computation; the scoped \
             subgraph has {n} nodes (cap {MAX_ECCENTRICITY_NODES}). Narrow it with \
             {{node_type, relationship}} scoping, or compute on a smaller subgraph."
        ));
    }
    let mut out = Vec::with_capacity(n);
    // Generation-stamped visited markers avoid an O(n) reset per source.
    let mut seen = vec![0u32; n];
    let mut dist = vec![0u32; n];
    let mut queue: VecDeque<u32> = VecDeque::with_capacity(64);
    let mut generation = 0u32;
    for (s, &node) in nodes.iter().enumerate() {
        if s & 0x3FF == 0 && deadline.exceeded() {
            return Err(algorithm_timeout_err());
        }
        generation += 1;
        seen[s] = generation;
        dist[s] = 0;
        queue.clear();
        queue.push_back(s as u32);
        let mut ecc = 0u32;
        while let Some(u) = queue.pop_front() {
            let du = dist[u as usize];
            for &w in &adj[u as usize] {
                if seen[w as usize] != generation {
                    seen[w as usize] = generation;
                    dist[w as usize] = du + 1;
                    ecc = ecc.max(du + 1);
                    queue.push_back(w);
                }
            }
        }
        out.push((node, ecc as i64));
    }
    Ok(out)
}

/// Graph diameter over the scoped undirected subgraph: the greatest
/// eccentricity (i.e. the longest shortest path within any connected
/// component). `0` for an empty or edgeless subgraph. Same all-pairs cost +
/// node cap as [`eccentricity_scoped`].
pub fn diameter_scoped(
    graph: &DirGraph,
    node_types: Option<&[String]>,
    rel_types: Option<&[InternedKey]>,
    deadline: Interrupt,
) -> Result<i64, String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let eccs = eccentricity_scoped(graph, node_types, rel_types, deadline)?;
    Ok(eccs.iter().map(|(_, e)| *e).max().unwrap_or(0))
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
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let node = graph.get_node(node_idx)?;
    let node_title = node.title();
    let title_str = match &*node_title {
        Value::String(s) => s.clone(),
        _ => format!("{:?}", *node_title),
    };
    Some(PathNodeInfo {
        node_type: node.node_type_str(&graph.interner).to_string(),
        title: title_str,
        id: node.id().into_owned(),
    })
}

/// Get information about what connection types link nodes in a path
pub fn get_path_connections(graph: &DirGraph, path: &[NodeIndex]) -> Vec<Option<String>> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
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
    shortest_path(graph, source, target, None, None, Interrupt::default()).is_some()
}

/// Calculate the degree (number of connections) for a node
pub fn node_degree(graph: &DirGraph, node: NodeIndex) -> usize {
    let g = &graph.graph;
    g.edges(node).count()
        + g.neighbors_directed(node, petgraph::Direction::Incoming)
            .count()
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
    deadline: Interrupt,
) -> Option<WeightedPathResult> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
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
        if visit_count.is_multiple_of(1000) && deadline.exceeded() {
            {
                return None;
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
    deadline: Interrupt,
) -> Option<f64> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
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
pub(super) fn compute_modularity(
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
