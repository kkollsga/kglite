//! Community-detection algorithms and their shared adjacency strategies.

use super::graph_algorithms::{
    algorithm_timeout_err, compute_modularity, edge_in_scope, edge_weight, intern_connection_types,
    scoped_node_set, NodeScope,
};
use super::Interrupt;
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;
use std::collections::{HashMap, HashSet};

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

/// Tunable options for the modularity-optimizing community detectors
/// [`louvain_communities`] and [`leiden_communities`] — they share the same
/// knob shape (weighted, resolution-parameterized, hierarchical). Construct
/// via [`CommunityOptions::default`] then the `with_*` builders. Label
/// propagation has a different shape ([`LabelPropagationOptions`]).
#[derive(Clone)]
#[non_exhaustive]
pub struct CommunityOptions<'a> {
    /// Edge property read as the weight (`None` = every edge weighs `1.0`).
    pub weight_property: Option<&'a str>,
    /// Resolution parameter; higher values yield more, smaller communities
    /// (default `1.0`).
    pub resolution: f64,
    /// Only traverse edges of these connection types (`None` = all edges).
    pub connection_types: Option<&'a [String]>,
    /// Restrict the community universe to this node set (`None` = whole graph).
    pub scope: Option<&'a NodeScope>,
    /// Deadline + cooperative-cancellation bundle.
    pub interrupt: Interrupt,
}

impl Default for CommunityOptions<'_> {
    fn default() -> Self {
        Self {
            weight_property: None,
            resolution: 1.0,
            connection_types: None,
            scope: None,
            interrupt: Interrupt::default(),
        }
    }
}

impl<'a> CommunityOptions<'a> {
    /// Read edge weights from the given property.
    pub fn with_weight_property(mut self, weight_property: &'a str) -> Self {
        self.weight_property = Some(weight_property);
        self
    }
    /// Set the resolution parameter.
    pub fn with_resolution(mut self, resolution: f64) -> Self {
        self.resolution = resolution;
        self
    }
    /// Restrict traversal to the given connection types.
    pub fn with_connection_types(mut self, connection_types: &'a [String]) -> Self {
        self.connection_types = Some(connection_types);
        self
    }
    /// Restrict the community universe to the given node set.
    pub fn with_scope(mut self, scope: &'a NodeScope) -> Self {
        self.scope = Some(scope);
        self
    }
    /// Set the deadline + cancellation bundle.
    pub fn with_interrupt(mut self, interrupt: Interrupt) -> Self {
        self.interrupt = interrupt;
        self
    }
}

/// Tunable options for [`label_propagation`]. Unweighted and iteration-bounded
/// (no resolution / weight knobs — a different shape from [`CommunityOptions`]).
#[derive(Clone)]
#[non_exhaustive]
pub struct LabelPropagationOptions<'a> {
    /// Maximum propagation sweeps before stopping (default `100`).
    pub max_iterations: usize,
    /// Only traverse edges of these connection types (`None` = all edges).
    pub connection_types: Option<&'a [String]>,
    /// Restrict the community universe to this node set (`None` = whole graph).
    pub scope: Option<&'a NodeScope>,
    /// Deadline + cooperative-cancellation bundle.
    pub interrupt: Interrupt,
}

impl Default for LabelPropagationOptions<'_> {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            connection_types: None,
            scope: None,
            interrupt: Interrupt::default(),
        }
    }
}

impl<'a> LabelPropagationOptions<'a> {
    /// Set the maximum sweep count.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }
    /// Restrict traversal to the given connection types.
    pub fn with_connection_types(mut self, connection_types: &'a [String]) -> Self {
        self.connection_types = Some(connection_types);
        self
    }
    /// Restrict the community universe to the given node set.
    pub fn with_scope(mut self, scope: &'a NodeScope) -> Self {
        self.scope = Some(scope);
        self
    }
    /// Set the deadline + cancellation bundle.
    pub fn with_interrupt(mut self, interrupt: Interrupt) -> Self {
        self.interrupt = interrupt;
        self
    }
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
pub(super) fn scoped_universe(
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
pub(super) struct DedupNeighborSource<'a> {
    graph: &'a DirGraph,
    pub(super) nodes: Vec<NodeIndex>,
    node_to_idx: Vec<u32>,
    edge_types: Option<Vec<InternedKey>>,
}

impl<'a> DedupNeighborSource<'a> {
    pub(super) fn new(
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

    pub(super) fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Fill `buf` with node `v`'s sorted, de-duped, in-universe compact
    /// neighbour indices (both directions; self-loops and out-of-universe peers
    /// dropped). When there's no edge-type filter the cheap CSR peer-walk
    /// (`iter_peers_filtered`) is used; otherwise edges are materialised to read
    /// their type.
    pub(super) fn neighbors_deduped(&self, v: usize, buf: &mut Vec<u32>) {
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
    deadline: Interrupt,
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
        if deadline.exceeded() {
            {
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
    deadline: Interrupt,
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
    options: &CommunityOptions,
) -> Result<CommunityResult, String> {
    // Direct GraphRead traversal outside the executor: hold the disk arena
    // guard while node/edge weights are borrowed (owned counter handle;
    // no-op on memory/mapped backends). Results returned are owned.
    let _arena_guard = graph.graph.begin_query();
    let CommunityOptions {
        weight_property,
        resolution,
        connection_types,
        scope,
        interrupt: deadline,
    } = *options;
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
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
    deadline: Interrupt,
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
        deadline: Interrupt,
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
    options: &CommunityOptions,
) -> Result<CommunityResult, String> {
    // Direct GraphRead traversal outside the executor: hold the disk arena
    // guard while node/edge weights are borrowed (owned counter handle;
    // no-op on memory/mapped backends). Results returned are owned.
    let _arena_guard = graph.graph.begin_query();
    let CommunityOptions {
        weight_property,
        resolution,
        connection_types,
        scope,
        interrupt: deadline,
    } = *options;
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
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
    options: &LabelPropagationOptions,
) -> Result<CommunityResult, String> {
    // Direct GraphRead traversal outside the executor: hold the disk arena
    // guard while node/edge weights are borrowed (owned counter handle;
    // no-op on memory/mapped backends). Results returned are owned.
    let _arena_guard = graph.graph.begin_query();
    let LabelPropagationOptions {
        max_iterations,
        connection_types,
        scope,
        interrupt: deadline,
    } = *options;
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
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
        if deadline.exceeded() {
            {
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
    deadline: Interrupt,
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
        if deadline.exceeded() {
            {
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
