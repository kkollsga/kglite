//! Graph algorithms module providing path finding and connectivity analysis.

use super::Interrupt;
use crate::datatypes::values::Value;
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::GraphRead;
use petgraph::algo::kosaraju_scc;
use petgraph::graph::NodeIndex;
use petgraph::Direction;
use std::collections::{HashMap, HashSet, VecDeque};

// Centrality algorithms live in the sibling `centrality` module (god-file
// ceiling); re-exported so existing `graph_algorithms::…` paths keep resolving.
use super::bidirectional::bidirectional_bfs;
pub use super::centrality::*;
// Community detection likewise: sibling module, same compatibility re-export.
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

/// Undirected neighbours filtered by connection type (`None` = every neighbour).
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

/// Get directed (incoming only) neighbors filtered by edge connection type —
/// the twin of [`filtered_neighbors_outgoing`], walking the graph backwards.
/// `edges_directed(_, Incoming)` is O(degree) in every storage backend (the
/// in-memory petgraph keeps both adjacency chains; mapped/disk CSR carries a
/// reverse index), so an incoming expansion costs what an outgoing one does.
fn filtered_neighbors_incoming(
    graph: &DirGraph,
    node: NodeIndex,
    connection_types: Option<&[InternedKey]>,
) -> Vec<NodeIndex> {
    use petgraph::Direction;
    let g = &graph.graph;
    match connection_types {
        None => g.neighbors_directed(node, Direction::Incoming).collect(),
        Some(types) => g
            .edges_directed(node, Direction::Incoming)
            .filter(|e| types.iter().any(|t| *t == e.connection_type()))
            .map(|e| e.source())
            .collect(),
    }
}

/// Which way a path finder is allowed to walk each edge.
///
/// Default is [`EdgeDir::Any`] — every path finder in this module has always
/// been undirected and stays that way unless a caller asks otherwise.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum EdgeDir {
    /// Traverse edges in both directions (the historical behaviour).
    #[default]
    Any,
    /// Follow outgoing edges only (source → target).
    Outgoing,
    /// Follow incoming edges only (target → source).
    Incoming,
}

impl EdgeDir {
    /// The direction a search walking *backwards* from the target must use to
    /// retrace the edges a forward search walks. `Any` is its own reverse.
    ///
    /// This is what makes [`bidirectional_bfs`] correct on a directed query:
    /// the backward frontier must expand the transpose, or it would search for
    /// paths *out of* the target instead of *into* it.
    #[inline]
    fn reversed(self) -> Self {
        match self {
            EdgeDir::Any => EdgeDir::Any,
            EdgeDir::Outgoing => EdgeDir::Incoming,
            EdgeDir::Incoming => EdgeDir::Outgoing,
        }
    }
}

/// Neighbour expansion for one BFS/DFS step, honouring both the direction and
/// the connection-type filter. The single dispatch point for the three
/// `filtered_neighbors_*` helpers.
#[inline]
fn filtered_neighbors(
    graph: &DirGraph,
    node: NodeIndex,
    direction: EdgeDir,
    connection_types: Option<&[InternedKey]>,
) -> Vec<NodeIndex> {
    match direction {
        EdgeDir::Any => filtered_neighbors_undirected(graph, node, connection_types),
        EdgeDir::Outgoing => filtered_neighbors_outgoing(graph, node, connection_types),
        EdgeDir::Incoming => filtered_neighbors_incoming(graph, node, connection_types),
    }
}

/// Feed every neighbour of `node` to `sink`, honouring direction and the
/// connection-type filter — the allocation-free expansion the *search* engines
/// use, as opposed to [`filtered_neighbors`], which materialises a deduplicated
/// `Vec` for the DFS enumerators that need one.
///
/// The unfiltered arms hand petgraph's own iterator straight to the sink: no
/// `Vec`, no sort, no dedup. That matters because those three cost more than
/// the traversal itself on the default (unfiltered) path — a path-returning
/// call was ~8× slower than the length-only call purely from this per-node
/// allocate/sort/dedup, even though a BFS discards duplicates for free against
/// its own visited set.
///
/// Duplicates therefore reach `sink`: `neighbors_undirected` yields one entry
/// per incident edge, so a parallel edge or an a→b/b→a pair repeats. Every
/// caller here is a BFS whose seen-check already rejects them. The
/// deduplicating [`filtered_neighbors_undirected`] stays in place for
/// [`all_paths`] and [`all_shortest_paths_impl`], where a repeat is not free
/// (wasted DFS branches, and duplicated predecessor-DAG entries respectively).
#[inline]
fn expand_neighbors_into(
    graph: &DirGraph,
    node: NodeIndex,
    direction: EdgeDir,
    connection_types: Option<&[InternedKey]>,
    sink: &mut dyn FnMut(u32),
) {
    let g = &graph.graph;
    match (connection_types, direction) {
        (None, EdgeDir::Any) => {
            for neighbor in g.neighbors_undirected(node) {
                sink(neighbor.index() as u32);
            }
        }
        (None, EdgeDir::Outgoing) => {
            for neighbor in g.neighbors_directed(node, Direction::Outgoing) {
                sink(neighbor.index() as u32);
            }
        }
        (None, EdgeDir::Incoming) => {
            for neighbor in g.neighbors_directed(node, Direction::Incoming) {
                sink(neighbor.index() as u32);
            }
        }
        (Some(_), _) => {
            for neighbor in filtered_neighbors(graph, node, direction, connection_types) {
                sink(neighbor.index() as u32);
            }
        }
    }
}

/// Whether a node passes the via_types filter.
/// Source and target should be excluded from this check by the caller.
fn node_passes_via_filter(
    graph: &DirGraph,
    node: NodeIndex,
    via_types: &Option<HashSet<&str>>,
) -> bool {
    match via_types {
        None => true,
        Some(types) => {
            if let Some(node_data) = graph.graph.node_view(node) {
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
    pub path: Vec<NodeIndex>,
    pub cost: usize,
}

/// Information about a node in a path (for Python output)
#[derive(Debug, Clone)]
pub struct PathNodeInfo {
    pub node_type: String,
    pub title: String,
    pub id: Value,
}

/// The shared edge-type / via-type / direction / interrupt knobs the path
/// finders in this module take — [`shortest_path`], the `shortest_path_cost*`
/// family, [`shortest_path_weighted`], [`are_connected_with`] and
/// [`shortest_path_costs_from`]. The endpoints (and, for the weighted finders,
/// the weight property) stay positional as primary inputs.
/// Construct via [`PathOptions::default`] then the `with_*` builders.
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct PathOptions<'a> {
    /// Only traverse edges of these connection types (`None` = all edges).
    pub connection_types: Option<&'a [String]>,
    /// Only route through nodes of these types (`None` = any node).
    ///
    /// This — not the endpoint node types the bindings take, which are only an
    /// **id namespace** for the lookup — is what restricts which node types a
    /// path may pass through.
    pub via_types: Option<&'a [String]>,
    /// Which way edges may be walked ([`EdgeDir::Any`] = undirected, the
    /// default and the historical behaviour of every finder here).
    pub direction: EdgeDir,
    /// Deadline + cooperative-cancellation bundle.
    pub interrupt: Interrupt,
}

impl<'a> PathOptions<'a> {
    pub fn with_connection_types(mut self, connection_types: &'a [String]) -> Self {
        self.connection_types = Some(connection_types);
        self
    }
    pub fn with_via_types(mut self, via_types: &'a [String]) -> Self {
        self.via_types = Some(via_types);
        self
    }
    pub fn with_direction(mut self, direction: EdgeDir) -> Self {
        self.direction = direction;
        self
    }
    pub fn with_interrupt(mut self, interrupt: Interrupt) -> Self {
        self.interrupt = interrupt;
        self
    }
}

/// Tunable options for [`all_paths`] — the single-path [`PathOptions`] knobs
/// plus the path-enumeration bounds (`max_hops`, `max_results`) that only make
/// sense when enumerating every path.
#[derive(Clone)]
#[non_exhaustive]
pub struct AllPathsOptions<'a> {
    /// Maximum path length (hop count) to search (default `5`).
    pub max_hops: usize,
    /// Stop after finding this many paths (`None` = unlimited); bounds OOM on
    /// dense graphs.
    pub max_results: Option<usize>,
    /// Only traverse edges of these connection types (`None` = all edges).
    pub connection_types: Option<&'a [String]>,
    /// Only route through nodes of these types (`None` = any node).
    pub via_types: Option<&'a [String]>,
    /// Which way edges may be walked ([`EdgeDir::Any`] = undirected, default).
    pub direction: EdgeDir,
    /// Deadline + cooperative-cancellation bundle.
    pub interrupt: Interrupt,
}

impl Default for AllPathsOptions<'_> {
    fn default() -> Self {
        Self {
            max_hops: 5,
            max_results: None,
            connection_types: None,
            via_types: None,
            direction: EdgeDir::Any,
            interrupt: Interrupt::default(),
        }
    }
}

impl<'a> AllPathsOptions<'a> {
    pub fn with_max_hops(mut self, max_hops: usize) -> Self {
        self.max_hops = max_hops;
        self
    }
    pub fn with_max_results(mut self, max_results: usize) -> Self {
        self.max_results = Some(max_results);
        self
    }
    pub fn with_connection_types(mut self, connection_types: &'a [String]) -> Self {
        self.connection_types = Some(connection_types);
        self
    }
    pub fn with_via_types(mut self, via_types: &'a [String]) -> Self {
        self.via_types = Some(via_types);
        self
    }
    pub fn with_direction(mut self, direction: EdgeDir) -> Self {
        self.direction = direction;
        self
    }
    pub fn with_interrupt(mut self, interrupt: Interrupt) -> Self {
        self.interrupt = interrupt;
        self
    }
}

/// Find the shortest path between two nodes, by [`bidirectional_bfs`].
/// By default (`PathOptions::direction == EdgeDir::Any`) the graph is treated
/// as undirected, finding connections in either direction; set
/// [`PathOptions::direction`] to follow outgoing or incoming edges only.
/// Returns None if no path exists.
///
/// Where several shortest paths tie, *which* one is returned is unspecified —
/// it always was, and meeting in the middle makes a different arbitrary
/// choice than a one-sided scan. Callers needing all of them want
/// [`all_shortest_paths`].
pub fn shortest_path(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
    options: &PathOptions,
) -> Option<PathResult> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let PathOptions {
        connection_types,
        via_types,
        direction,
        interrupt: deadline,
    } = *options;
    let via_set: Option<HashSet<&str>> =
        via_types.map(|vt| vt.iter().map(|s| s.as_str()).collect());
    let interned = intern_connection_types(connection_types);
    let path = bidirectional_path(
        graph,
        source,
        target,
        interned.as_deref(),
        &via_set,
        direction,
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

/// Find the shortest path LENGTH between two nodes — the hop count of the
/// path [`shortest_path`] would return, from the same [`bidirectional_bfs`].
///
/// Unfiltered and undirected. [`shortest_path_cost_with`] is the same search
/// with a [`PathOptions`] — connection/via-type filters, direction, deadline.
pub fn shortest_path_cost(graph: &DirGraph, source: NodeIndex, target: NodeIndex) -> Option<usize> {
    shortest_path_cost_with(graph, source, target, &PathOptions::default())
}

/// [`shortest_path_cost`] honouring a [`PathOptions`]: connection-type and
/// via-type filters, [`EdgeDir`], and the deadline. Answers the same question
/// [`shortest_path`] does, without reconstructing the path.
pub fn shortest_path_cost_with(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
    options: &PathOptions,
) -> Option<usize> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    if source == target {
        return Some(0);
    }
    let PathOptions {
        connection_types,
        via_types,
        direction,
        interrupt: deadline,
    } = *options;

    let via_set: Option<HashSet<&str>> =
        via_types.map(|vt| vt.iter().map(|s| s.as_str()).collect());
    let interned = intern_connection_types(connection_types);
    // The same meet-in-the-middle search `shortest_path` runs, with the
    // reconstructed sequence discarded: a separate hop-count-only BFS would
    // save only the O(hops) reconstruction, at the cost of a second copy of
    // the termination rule to get wrong.
    bidirectional_path(
        graph,
        source,
        target,
        interned.as_deref(),
        &via_set,
        direction,
        deadline,
    )
    .map(|path| path.len().saturating_sub(1))
}

/// Batch shortest path cost — reuses visited Vec and adjacency list across multiple pairs.
/// Much faster than calling shortest_path_cost N times for large graphs.
///
/// Unfiltered and undirected; [`shortest_path_cost_batch_with`] takes the
/// filters and direction.
pub fn shortest_path_cost_batch(
    graph: &DirGraph,
    pairs: &[(NodeIndex, NodeIndex)],
) -> Vec<Option<usize>> {
    shortest_path_cost_batch_with(graph, pairs, &PathOptions::default())
}

/// [`shortest_path_cost_batch`] honouring a [`PathOptions`].
///
/// The adjacency is built once over the restricted universe: `via_types` picks
/// the vertices, `connection_types` picks the edges, and `direction` decides
/// whether each edge contributes one link or two. The query endpoints are
/// added to that universe even when `via_types` excludes their type, and are
/// then barred from serving as an intermediate hop — the same "endpoints are
/// exempt from `via_types`, the middle is not" rule the single-pair members
/// follow, so the whole family answers one question.
///
/// A pair whose endpoint has no surviving edge answers `None` — the same "no
/// path" a disconnected pair gets. It is never an error.
///
/// Returns every pair's answer in input order. On deadline expiry the pairs
/// not yet answered come back as `None`.
pub fn shortest_path_cost_batch_with(
    graph: &DirGraph,
    pairs: &[(NodeIndex, NodeIndex)],
    options: &PathOptions,
) -> Vec<Option<usize>> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let PathOptions {
        connection_types,
        via_types,
        direction,
        interrupt: deadline,
    } = *options;

    let interned = intern_connection_types(connection_types);

    // Vertex universe: whatever `via_types` admits, plus the query endpoints
    // (which may be of an excluded type — `via_types` gates the middle of a
    // path, not its ends). `via_ok` remembers which is which.
    let mut nodes = scoped_universe(graph, via_types, interned.as_deref());
    let bound = graph.graph.node_bound();
    let mut in_universe = vec![false; bound];
    for &node in &nodes {
        in_universe[node.index()] = true;
    }
    // `via_types == None` means "any node may be an intermediate"; otherwise
    // only the nodes the scope admitted may be, and the endpoints appended
    // below may not.
    let via_unrestricted = via_types.is_none();
    let via_ok_by_node = in_universe.clone();
    for &(source, target) in pairs {
        for endpoint in [source, target] {
            if endpoint.index() < bound && !in_universe[endpoint.index()] {
                in_universe[endpoint.index()] = true;
                nodes.push(endpoint);
            }
        }
    }

    let Ok((nodes, adj)) =
        build_scoped_adjacency_over(graph, nodes, interned.as_deref(), direction, deadline)
    else {
        // A timed-out adjacency build answers every pair `None` rather than
        // erroring: this API's contract is "a distance or no distance".
        return vec![None; pairs.len()];
    };

    let n = nodes.len();
    let mut node_to_idx = vec![u32::MAX; bound];
    let mut via_ok = vec![true; n];
    for (i, &node) in nodes.iter().enumerate() {
        node_to_idx[node.index()] = i as u32;
        via_ok[i] = via_unrestricted || via_ok_by_node[node.index()];
    }

    let mut visited: Vec<bool> = vec![false; n];
    let mut current_level: Vec<usize> = Vec::new();
    let mut next_level: Vec<usize> = Vec::new();

    let mut results = Vec::with_capacity(pairs.len());

    for &(source, target) in pairs {
        if source == target {
            results.push(Some(0));
            continue;
        }
        if deadline.exceeded() {
            results.push(None);
            continue;
        }

        let src_i = node_to_idx[source.index()];
        let tgt_i = node_to_idx[target.index()];
        if src_i == u32::MAX || tgt_i == u32::MAX {
            results.push(None);
            continue;
        }
        let (src_i, tgt_i) = (src_i as usize, tgt_i as usize);

        // Reset only the nodes this query touched (much faster than clearing
        // the whole array between pairs).
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
                for &neighbor in &adj[current_idx] {
                    let neighbor_idx = neighbor as usize;
                    if visited[neighbor_idx] {
                        continue;
                    }
                    if neighbor_idx == tgt_i {
                        found = true;
                        break 'bfs;
                    }
                    if !via_ok[neighbor_idx] {
                        continue;
                    }
                    visited[neighbor_idx] = true;
                    touched.push(neighbor_idx);
                    next_level.push(neighbor_idx);
                }
            }

            std::mem::swap(&mut current_level, &mut next_level);
        }

        results.push(if found { Some(depth) } else { None });

        for &idx in &touched {
            visited[idx] = false;
        }
    }

    results
}

/// Reusable scratch for a *generation-stamped* single-source BFS over a fixed
/// vertex space (`0..n`).
///
/// The generation counter is what makes the all-pairs eccentricity loop cheap:
/// bumping `generation` invalidates every `seen` entry in O(1), so n
/// back-to-back searches never pay an O(n) reset each. A caller that runs a
/// single search (`shortest_path_costs_from`) simply allocates one and drops it.
struct BfsScratch {
    /// Per-vertex "reached in generation g" marker; `0` is never a live
    /// generation, so a freshly allocated buffer starts fully unvisited.
    seen: Vec<u32>,
    /// Hop distance from the current source, valid only where
    /// `seen[v] == generation`.
    dist: Vec<u32>,
    queue: VecDeque<u32>,
    generation: u32,
}

impl BfsScratch {
    fn new(n: usize) -> Self {
        Self {
            seen: vec![0u32; n],
            dist: vec![0u32; n],
            queue: VecDeque::with_capacity(64),
            generation: 0,
        }
    }
}

/// One single-source breadth-first search, abstracted over how neighbours are
/// produced and what is done with each node reached. The shared engine behind
/// [`shortest_path_costs_from`] (which expands over the live graph, honouring
/// [`PathOptions`]) and [`eccentricity_scoped`] (which expands over a compact
/// scoped adjacency it built once for all its sources).
///
/// * `expand(u, sink)` feeds every neighbour of `u` to `sink`; duplicates are
///   harmless (the `seen` stamp filters them).
/// * `visit(v, dist)` is called exactly once per vertex the search reaches,
///   in non-decreasing distance order, **including `start` at distance 0**.
///   Returning `false` reports the vertex but stops the search expanding
///   *through* it — that is how `via_types` gates the middle of a path while
///   leaving its ends exempt, exactly as the pair finders do.
/// * `max_hops` stops the search after that many levels (`Some(0)` visits only
///   `start`).
///
/// Returns `Err(algorithm_timeout_err())` when `deadline` expires mid-search.
fn single_source_bfs(
    scratch: &mut BfsScratch,
    start: u32,
    max_hops: Option<usize>,
    deadline: Interrupt,
    mut expand: impl FnMut(u32, &mut dyn FnMut(u32)),
    mut visit: impl FnMut(u32, u32) -> bool,
) -> Result<(), String> {
    scratch.generation += 1;
    let generation = scratch.generation;
    let start_idx = start as usize;
    scratch.seen[start_idx] = generation;
    scratch.dist[start_idx] = 0;
    scratch.queue.clear();
    if visit(start, 0) {
        scratch.queue.push_back(start);
    }

    let mut popped = 0u32;
    while let Some(u) = scratch.queue.pop_front() {
        popped = popped.wrapping_add(1);
        if popped.is_multiple_of(1024) && deadline.exceeded() {
            return Err(algorithm_timeout_err());
        }
        let du = scratch.dist[u as usize];
        if max_hops.is_some_and(|cap| du as usize >= cap) {
            // FIFO order means every remaining entry is at least this deep.
            break;
        }
        let BfsScratch {
            seen, dist, queue, ..
        } = &mut *scratch;
        expand(u, &mut |w| {
            let wi = w as usize;
            if seen[wi] != generation {
                seen[wi] = generation;
                dist[wi] = du + 1;
                if visit(w, du + 1) {
                    queue.push_back(w);
                }
            }
        });
    }
    Ok(())
}

/// Hop distance from **one** source to every node it can reach — the
/// one-to-many member of the shortest-path family, answering in a single BFS
/// what N [`shortest_path_cost_with`] calls would answer one pair at a time.
///
/// Returns `(node, hops)` in non-decreasing distance order, `source` itself
/// first at distance `0`. Unreachable nodes are simply absent. `max_hops`
/// bounds the search (`Some(0)` returns only the source); `None` walks the
/// whole reachable component, so a caller exposing this to users should bound
/// it.
///
/// The [`PathOptions`] knobs mean exactly what they mean everywhere else in
/// the family: `connection_types` limits which edge types may be walked,
/// `direction` limits their orientation, and `via_types` limits which node
/// types a path may pass **through**. A node whose type `via_types` excludes
/// is still *reported* (with its distance) but is never expanded — the same
/// "endpoints are exempt from `via_types`, the middle is not" rule the pair
/// finders follow, generalised to a search where every reached node is a
/// potential endpoint.
///
/// Unlike the pair finders (which answer `None` on a deadline, indistinguishable
/// from "no path"), this returns `Err(algorithm_timeout_err())`: a partial map
/// silently missing its far half is a wrong answer, not a missing one.
pub fn shortest_path_costs_from(
    graph: &DirGraph,
    source: NodeIndex,
    options: &PathOptions,
    max_hops: Option<usize>,
) -> Result<Vec<(NodeIndex, usize)>, String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let bound = graph.graph.node_bound();
    if source.index() >= bound {
        return Ok(Vec::new());
    }
    let PathOptions {
        connection_types,
        via_types,
        direction,
        interrupt: deadline,
    } = *options;

    let mut out: Vec<(NodeIndex, usize)> = Vec::new();
    let mut scratch = BfsScratch::new(bound);
    let start = source.index() as u32;

    let via_set: Option<HashSet<&str>> =
        via_types.map(|vt| vt.iter().map(|s| s.as_str()).collect());
    let interned = intern_connection_types(connection_types);
    single_source_bfs(
        &mut scratch,
        start,
        max_hops,
        deadline,
        |u, sink| {
            expand_neighbors_into(
                graph,
                NodeIndex::new(u as usize),
                direction,
                interned.as_deref(),
                sink,
            )
        },
        |v, d| {
            let node = NodeIndex::new(v as usize);
            out.push((node, d as usize));
            // The source is always expandable; every other node must pass
            // `via_types` to serve as an intermediate hop. An absent filter
            // waves everything through, so the unfiltered search pays only
            // this discriminant check.
            d == 0 || node_passes_via_filter(graph, node, &via_set)
        },
    )?;
    Ok(out)
}

/// [`bidirectional_bfs`] driven over the live graph — the single search behind
/// [`shortest_path`], [`shortest_path_directed`] and
/// [`shortest_path_cost_with`]. The backward frontier walks
/// [`EdgeDir::reversed`], so a directed query still asks "which paths lead
/// *into* the target".
fn bidirectional_path(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
    connection_types: Option<&[InternedKey]>,
    via_types: &Option<HashSet<&str>>,
    direction: EdgeDir,
    deadline: Interrupt,
) -> Option<Vec<NodeIndex>> {
    let source_id = u32::try_from(source.index()).ok()?;
    let target_id = u32::try_from(target.index()).ok()?;
    let backward = direction.reversed();
    let path = bidirectional_bfs(
        source_id,
        target_id,
        |u, sink| {
            expand_neighbors_into(
                graph,
                NodeIndex::new(u as usize),
                direction,
                connection_types,
                sink,
            )
        },
        |u, sink| {
            expand_neighbors_into(
                graph,
                NodeIndex::new(u as usize),
                backward,
                connection_types,
                sink,
            )
        },
        |w| node_passes_via_filter(graph, NodeIndex::new(w as usize), via_types),
        deadline,
    )?;
    Some(
        path.into_iter()
            .map(|idx| NodeIndex::new(idx as usize))
            .collect(),
    )
}

/// Directed BFS shortest path — only follows outgoing edges. A thin
/// delegate over [`shortest_path`] with [`EdgeDir::Outgoing`]; kept as a
/// named entry point because the Cypher `shortestPath()` executor dispatches
/// on its own `EdgeDirection` and reads better naming the directed case.
pub fn shortest_path_directed(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
    options: &PathOptions,
) -> Option<PathResult> {
    shortest_path(
        graph,
        source,
        target,
        &options.clone().with_direction(EdgeDir::Outgoing),
    )
}

/// Find all paths between two nodes up to a maximum number of hops.
/// Warning: This can be expensive for graphs with many paths!
pub fn all_paths(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
    options: &AllPathsOptions,
) -> Vec<Vec<NodeIndex>> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let AllPathsOptions {
        max_hops,
        max_results,
        connection_types,
        via_types,
        direction,
        interrupt: deadline,
    } = *options;
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
        direction,
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
    direction: EdgeDir,
    deadline: Interrupt,
) {
    if let Some(max) = max_results {
        if results.len() >= max {
            return;
        }
    }

    if deadline.exceeded() {
        return;
    }

    if current == target {
        results.push(current_path.clone());
        return;
    }

    if remaining_hops == 0 {
        return;
    }

    let neighbors = filtered_neighbors(graph, current, direction, connection_types);
    for neighbor in neighbors {
        if let Some(max) = max_results {
            if results.len() >= max {
                return;
            }
        }

        if !visited.contains(&neighbor) {
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
                direction,
                deadline,
            );

            current_path.pop();
            visited.remove(&neighbor);
        }
    }
}

/// Find all strongly connected components in the graph.
pub fn connected_components(graph: &DirGraph) -> Vec<Vec<NodeIndex>> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    // A forked backend has no single `StableDiGraph` to hand petgraph — its
    // nodes are base⊕overlay — so it takes the same generic `GraphRead`
    // traversal the disk backend already uses. Same result, one dispatch per
    // node instead of a direct petgraph walk.
    if GraphRead::is_disk(&graph.graph) || graph.graph.is_forked() {
        return strongly_connected_components(&graph.graph);
    }
    kosaraju_scc(graph.graph.as_stable_digraph())
}

/// Kosaraju's algorithm over the storage abstraction. Disk graphs cannot
/// implement petgraph's borrowing traversal traits because their CSR reads
/// materialise through a query arena, so the disk path uses the equivalent
/// two-pass traversal through [`GraphRead`].
fn strongly_connected_components(graph: &impl GraphRead) -> Vec<Vec<NodeIndex>> {
    let nodes: Vec<NodeIndex> = graph.node_indices().collect();
    let mut visited = HashSet::with_capacity(nodes.len());
    let mut finished = Vec::with_capacity(nodes.len());

    for &root in &nodes {
        if visited.contains(&root) {
            continue;
        }
        let mut stack = vec![(root, false)];
        while let Some((node, expanded)) = stack.pop() {
            if expanded {
                finished.push(node);
                continue;
            }
            if !visited.insert(node) {
                continue;
            }
            stack.push((node, true));
            for neighbor in graph.neighbors_directed(node, Direction::Outgoing) {
                if !visited.contains(&neighbor) {
                    stack.push((neighbor, false));
                }
            }
        }
    }

    visited.clear();
    let mut components = Vec::new();
    for root in finished.into_iter().rev() {
        if !visited.insert(root) {
            continue;
        }
        let mut component = Vec::new();
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            component.push(node);
            for neighbor in graph.neighbors_directed(node, Direction::Incoming) {
                if visited.insert(neighbor) {
                    stack.push(neighbor);
                }
            }
        }
        components.push(component);
    }
    components
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

    #[inline]
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]]; // path halving
            x = parent[x];
        }
        x
    }

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
            return Err(algorithm_timeout_err());
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

    let mut component_map: HashMap<usize, Vec<NodeIndex>> = HashMap::new();
    for (i, &node) in nodes.iter().enumerate() {
        let root = find(&mut parent, i);
        component_map.entry(root).or_default().push(node);
    }

    let mut components: Vec<Vec<NodeIndex>> = component_map.into_values().collect();

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
    let nodes = scoped_universe(graph, node_types, rel_types);
    build_scoped_adjacency_over(graph, nodes, rel_types, EdgeDir::Any, deadline)
}

/// The adjacency build behind [`build_scoped_undirected_adjacency`], over an
/// explicit vertex universe and with a direction — the *directed sibling* the
/// batch shortest-path API needs, generalised rather than duplicated.
///
/// `direction` decides how many links each surviving edge contributes:
/// [`EdgeDir::Any`] adds both `s → t` and `t → s`, [`EdgeDir::Outgoing`] only
/// `s → t`, [`EdgeDir::Incoming`] only `t → s`. Edges whose endpoints are not
/// both in `nodes` are dropped, as are self-loops. Neighbour lists come back
/// sorted and de-duplicated in compact indices (`0..nodes.len()`).
///
/// Taking the universe as a parameter is what lets the batch API add its query
/// endpoints to a `via_types`-restricted universe: an endpoint is allowed to
/// *be* a path end without being allowed as an intermediate hop (the caller
/// enforces the second half with its own via mask), which is exactly the
/// single-pair members' rule.
fn build_scoped_adjacency_over(
    graph: &DirGraph,
    nodes: Vec<NodeIndex>,
    rel_types: Option<&[InternedKey]>,
    direction: EdgeDir,
    deadline: Interrupt,
) -> Result<(Vec<NodeIndex>, Vec<Vec<u32>>), String> {
    let edge_matches = |key: InternedKey| -> bool {
        match rel_types {
            Some(keys) => keys.contains(&key),
            None => true,
        }
    };

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
            return Err(algorithm_timeout_err());
        }
        if !edge_matches(edge.connection_type()) {
            continue;
        }
        let s = node_to_idx[edge.source().index()];
        let t = node_to_idx[edge.target().index()];
        if s == u32::MAX || t == u32::MAX || s == t {
            continue;
        }
        match direction {
            EdgeDir::Any => {
                adj[s as usize].push(t);
                adj[t as usize].push(s);
            }
            EdgeDir::Outgoing => adj[s as usize].push(t),
            EdgeDir::Incoming => adj[t as usize].push(s),
        }
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
    let mut pos = Vec::with_capacity(n);
    {
        let mut binc = bin.clone();
        for (v, &degree) in deg.iter().enumerate() {
            let d = degree as usize;
            let position = binc[d];
            pos.push(position);
            vert[position] = v;
            binc[d] += 1;
        }
    }

    for i in 0..n {
        let v = vert[i];
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

    Ok(nodes
        .into_iter()
        .zip(deg)
        .map(|(node, core)| (node, i64::from(core)))
        .collect())
}

/// Bounded-memory k-core for mapped/disk: the same Batagelj–Zaversnik peeling as
/// `coreness_scoped`, but the per-node neighbour lists are streamed on demand from
/// the CSR (`DedupNeighborSource`) instead of materialising the whole O(edges)
/// adjacency. Resident state is O(nodes) (deg/bin/vert/pos + index map);
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
    let mut deg: Vec<u32> = Vec::with_capacity(n);
    for v in 0..n {
        if v & 0xFFFFF == 0 && deadline.exceeded() {
            return Err(algorithm_timeout_err());
        }
        src.neighbors_deduped(v, &mut buf);
        deg.push(buf.len() as u32);
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
    let mut pos = Vec::with_capacity(n);
    {
        let mut binc = bin.clone();
        for (v, &degree) in deg.iter().enumerate() {
            let d = degree as usize;
            let position = binc[d];
            pos.push(position);
            vert[position] = v;
            binc[d] += 1;
        }
    }

    // Pass 2: peel in degree order, re-streaming each vertex's neighbours once.
    for i in 0..n {
        if i & 0xFFFFF == 0 && deadline.exceeded() {
            return Err(algorithm_timeout_err());
        }
        let v = vert[i];
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

    Ok(src
        .nodes
        .into_iter()
        .zip(deg)
        .map(|(node, core)| (node, i64::from(core)))
        .collect())
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
            return Err(algorithm_timeout_err());
        }
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
            return Err(algorithm_timeout_err());
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
    // One scratch for all n searches: the generation stamp is what keeps the
    // per-source cost O(V+E) instead of O(n) reset + O(V+E).
    let mut scratch = BfsScratch::new(n);
    for (s, &node) in nodes.iter().enumerate() {
        if s & 0x3FF == 0 && deadline.exceeded() {
            return Err(algorithm_timeout_err());
        }
        let mut ecc = 0u32;
        single_source_bfs(
            &mut scratch,
            s as u32,
            None,
            deadline,
            |u, sink| {
                for &w in &adj[u as usize] {
                    sink(w);
                }
            },
            |_v, d| {
                ecc = ecc.max(d);
                true
            },
        )?;
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
/// `gt`. Binary-searches to the eligible suffixes, then uses a linear merge.
fn intersection_count_gt(a: &[u32], b: &[u32], gt: u32) -> u64 {
    let mut i = a.partition_point(|&value| value <= gt);
    let mut j = b.partition_point(|&value| value <= gt);
    let mut count = 0u64;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
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
    let node = graph.node_view(node_idx)?;
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
    let mut connections = Vec::with_capacity(path.len().saturating_sub(1));

    for window in path.windows(2) {
        let from = window[0];
        let to = window[1];

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

/// Check if two nodes are connected (directly or indirectly), undirected and
/// unfiltered. [`are_connected_with`] takes the filters and direction.
pub fn are_connected(graph: &DirGraph, source: NodeIndex, target: NodeIndex) -> bool {
    are_connected_with(graph, source, target, &PathOptions::default())
}

/// [`are_connected`] honouring a [`PathOptions`] — "is there a path at all",
/// answered by the same search [`shortest_path_cost_with`] runs, with the
/// distance discarded.
pub fn are_connected_with(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
    options: &PathOptions,
) -> bool {
    let _arena_guard = graph.graph.begin_query(); // disk arena guard (owned; no-op on memory/mapped)
    shortest_path_cost_with(graph, source, target, options).is_some()
}

pub fn node_degree(graph: &DirGraph, node: NodeIndex) -> usize {
    let _arena_guard = graph.graph.begin_query(); // disk arena guard (owned; no-op on memory/mapped)
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

/// Dijkstra-based weighted shortest path. Undirected by default and directed
/// on [`PathOptions::direction`], exactly as [`shortest_path`]; reads
/// `weight_property` from each edge, defaulting to 1.0 when the property is
/// absent or non-numeric — the same fallback Louvain uses for its
/// weighted-adjacency build.
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
    options: &PathOptions,
) -> Option<WeightedPathResult> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let PathOptions {
        connection_types,
        via_types,
        direction,
        interrupt: deadline,
    } = *options;
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
            return None;
        }

        let current = NodeIndex::new(current_idx);
        // Direction is the same neighbour-set swap the unweighted finders do:
        // Any relaxes both incident chains, Outgoing/Incoming just one. No
        // allocation and no separate Dijkstra — so `direction` is honoured
        // with `weight_property`, never silently ignored.
        let out_edges = matches!(direction, EdgeDir::Any | EdgeDir::Outgoing).then(|| {
            graph
                .graph
                .edges_directed(current, petgraph::Direction::Outgoing)
        });
        let in_edges = matches!(direction, EdgeDir::Any | EdgeDir::Incoming).then(|| {
            graph
                .graph
                .edges_directed(current, petgraph::Direction::Incoming)
        });
        for edge in out_edges
            .into_iter()
            .flatten()
            .chain(in_edges.into_iter().flatten())
        {
            if let Some(types) = conn_filter {
                if !types.iter().any(|t| *t == edge.connection_type()) {
                    continue;
                }
            }
            let neighbor = match direction {
                EdgeDir::Outgoing => edge.target(),
                EdgeDir::Incoming => edge.source(),
                EdgeDir::Any => {
                    if edge.source() == current {
                        edge.target()
                    } else {
                        edge.source()
                    }
                }
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

/// [`shortest_path_weighted`] returning only the total weight. The path is
/// still reconstructed and then dropped — this saves the caller a field, not
/// the search any work.
pub fn shortest_path_cost_weighted(
    graph: &DirGraph,
    source: NodeIndex,
    target: NodeIndex,
    weight_property: &str,
    options: &PathOptions,
) -> Option<f64> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    shortest_path_weighted(graph, source, target, weight_property, options).map(|r| r.weight)
}

/// Compute standard Newman modularity (`resolution = 1`) over the filtered
/// induced subgraph. Each edge contributes once to `m` and internal weight,
/// and once to each endpoint's community degree (twice for a self-loop).
pub(super) fn compute_modularity(
    graph: &DirGraph,
    community: &[usize],
    node_exists: &[bool],
    weight_property: Option<&str>,
    connection_types: Option<&[String]>,
) -> f64 {
    let community_count = graph
        .graph
        .node_indices()
        .filter(|node| node_exists.get(node.index()).copied().unwrap_or(false))
        .map(|node| community[node.index()])
        .max()
        .map_or(0, |max_id| max_id + 1);
    if community_count == 0 {
        return 0.0;
    }

    let interned_ct = intern_connection_types(connection_types);
    let mut internal_weight = vec![0.0f64; community_count];
    let mut degree_sum = vec![0.0f64; community_count];
    let mut total_weight = 0.0f64;
    for edge in graph.graph.edge_references() {
        let u = edge.source().index();
        let v = edge.target().index();
        if !node_exists.get(u).copied().unwrap_or(false)
            || !node_exists.get(v).copied().unwrap_or(false)
        {
            continue;
        }
        if let Some(ref types) = interned_ct {
            if !types
                .iter()
                .any(|edge_type| *edge_type == edge.connection_type())
            {
                continue;
            }
        }
        let w = edge_weight(graph, edge.id(), weight_property);
        let cu = community[u];
        let cv = community[v];
        total_weight += w;
        degree_sum[cu] += w;
        degree_sum[cv] += w;
        if cu == cv {
            internal_weight[cu] += w;
        }
    }
    if total_weight == 0.0 {
        return 0.0;
    }

    let two_m = 2.0 * total_weight;
    internal_weight
        .iter()
        .zip(degree_sum.iter())
        .map(|(&internal, &degree)| internal / total_weight - (degree / two_m).powi(2))
        .sum()
}

#[cfg(test)]
#[path = "graph_algorithms_tests.rs"]
mod tests;
