//! Centrality algorithms — betweenness (Brandes), PageRank, degree, and
//! closeness. Split out of `graph_algorithms.rs` to keep that file under the
//! god-file ceiling; re-exported from `graph_algorithms` so existing
//! `graph_algorithms::betweenness_centrality` / `CentralityResult` paths keep
//! working.

use super::graph_algorithms::{
    algorithm_timeout_err, edge_in_scope, intern_connection_types, scoped_node_set, NodeScope,
};
use super::Interrupt;
use crate::graph::schema::DirGraph;
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;

/// Result of centrality calculation
#[derive(Debug, Clone)]
pub struct CentralityResult {
    pub node_idx: NodeIndex,
    pub score: f64,
}

/// Tunable options for [`pagerank`]. Construct via
/// [`PagerankOptions::default`] then the `with_*` builders, e.g.
/// `PagerankOptions::default().with_damping_factor(0.9)`. `#[non_exhaustive]`
/// so new knobs can be added without breaking callers.
#[derive(Clone)]
#[non_exhaustive]
pub struct PagerankOptions<'a> {
    /// Probability of following a link each iteration (default `0.85`).
    pub damping_factor: f64,
    /// Maximum power-iteration count before returning (default `100`).
    pub max_iterations: usize,
    /// Convergence threshold on the per-iteration score delta (default `1e-6`).
    pub tolerance: f64,
    /// Only traverse edges of these connection types (`None` = all edges).
    pub connection_types: Option<&'a [String]>,
    /// Restrict the scoring universe to this node set (`None` = whole graph).
    pub scope: Option<&'a NodeScope>,
    /// Deadline + cooperative-cancellation bundle.
    pub interrupt: Interrupt,
}

impl Default for PagerankOptions<'_> {
    fn default() -> Self {
        Self {
            damping_factor: 0.85,
            max_iterations: 100,
            tolerance: 1e-6,
            connection_types: None,
            scope: None,
            interrupt: Interrupt::default(),
        }
    }
}

impl<'a> PagerankOptions<'a> {
    /// Set the damping factor (probability of following a link).
    pub fn with_damping_factor(mut self, damping_factor: f64) -> Self {
        self.damping_factor = damping_factor;
        self
    }
    /// Set the maximum power-iteration count.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }
    /// Set the convergence tolerance.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }
    /// Restrict traversal to the given connection types.
    pub fn with_connection_types(mut self, connection_types: &'a [String]) -> Self {
        self.connection_types = Some(connection_types);
        self
    }
    /// Restrict the scoring universe to the given node set.
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

/// Tunable options for [`betweenness_centrality`] and
/// [`closeness_centrality`] — the two centrality measures that share the same
/// knob shape (both support sampling). Construct via
/// [`CentralityOptions::default`] then the `with_*` builders.
#[derive(Clone)]
#[non_exhaustive]
pub struct CentralityOptions<'a> {
    /// Normalize scores to a comparable range (default `true`).
    pub normalized: bool,
    /// Sample this many source nodes for a faster approximation on large
    /// graphs (`None` = exact, use every node).
    pub sample_size: Option<usize>,
    /// Only traverse edges of these connection types (`None` = all edges).
    pub connection_types: Option<&'a [String]>,
    /// Restrict the scoring universe to this node set (`None` = whole graph).
    pub scope: Option<&'a NodeScope>,
    /// Deadline + cooperative-cancellation bundle.
    pub interrupt: Interrupt,
}

impl Default for CentralityOptions<'_> {
    fn default() -> Self {
        Self {
            normalized: true,
            sample_size: None,
            connection_types: None,
            scope: None,
            interrupt: Interrupt::default(),
        }
    }
}

impl<'a> CentralityOptions<'a> {
    /// Toggle score normalization.
    pub fn with_normalized(mut self, normalized: bool) -> Self {
        self.normalized = normalized;
        self
    }
    /// Set the source-node sample size (approximate mode).
    pub fn with_sample_size(mut self, sample_size: usize) -> Self {
        self.sample_size = Some(sample_size);
        self
    }
    /// Restrict traversal to the given connection types.
    pub fn with_connection_types(mut self, connection_types: &'a [String]) -> Self {
        self.connection_types = Some(connection_types);
        self
    }
    /// Restrict the scoring universe to the given node set.
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

/// Tunable options for [`degree_centrality`]. Degree is exact and O(1) per
/// node, so it has no `sample_size` knob (unlike [`CentralityOptions`]).
#[derive(Clone)]
#[non_exhaustive]
pub struct DegreeCentralityOptions<'a> {
    /// Normalize by `(n - 1)` for values in `[0, 1]` (default `true`).
    pub normalized: bool,
    /// Only count edges of these connection types (`None` = all edges).
    pub connection_types: Option<&'a [String]>,
    /// Restrict the scoring universe to this node set (`None` = whole graph).
    pub scope: Option<&'a NodeScope>,
    /// Deadline + cooperative-cancellation bundle.
    pub interrupt: Interrupt,
}

impl Default for DegreeCentralityOptions<'_> {
    fn default() -> Self {
        Self {
            normalized: true,
            connection_types: None,
            scope: None,
            interrupt: Interrupt::default(),
        }
    }
}

impl<'a> DegreeCentralityOptions<'a> {
    /// Toggle score normalization.
    pub fn with_normalized(mut self, normalized: bool) -> Self {
        self.normalized = normalized;
        self
    }
    /// Restrict counting to the given connection types.
    pub fn with_connection_types(mut self, connection_types: &'a [String]) -> Self {
        self.connection_types = Some(connection_types);
        self
    }
    /// Restrict the scoring universe to the given node set.
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

/// Calculate betweenness centrality for all nodes in the graph.
///
/// Betweenness centrality measures how often a node lies on the shortest path
/// between other pairs of nodes. Higher values indicate nodes that are more
/// important as "bridges" in the network.
///
/// Uses Brandes' algorithm for efficiency: O(V * E) for unweighted graphs.
/// Optimized to use Vec instead of HashMap for O(1) direct indexing.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `normalized` - If true, normalize scores by 2/((n-1)*(n-2)) for directed graphs
/// * `sample_size` - Optional number of source nodes to sample (for large graphs)
///
/// @procedure: betweenness
/// @procedure: betweenness_centrality
pub fn betweenness_centrality(
    graph: &DirGraph,
    options: &CentralityOptions,
) -> Result<Vec<CentralityResult>, String> {
    let CentralityOptions {
        normalized,
        sample_size,
        connection_types,
        scope,
        interrupt: deadline,
    } = *options;
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, Ordering};

    let nodes: Vec<NodeIndex> = scoped_node_set(graph, scope);
    let n = nodes.len();

    if n <= 2 {
        return Ok(nodes
            .iter()
            .map(|&idx| CentralityResult {
                node_idx: idx,
                score: 0.0,
            })
            .collect());
    }

    // Use Vec-based index mapping for O(1) lookup (vs HashMap)
    let bound = graph.graph.node_bound();
    let mut node_to_idx = vec![0usize; bound];
    for (i, &node) in nodes.iter().enumerate() {
        node_to_idx[node.index()] = i;
    }

    // Pre-build undirected adjacency list for BFS.
    // Betweenness treats edges as undirected so that nodes bridging
    // communities are detected regardless of edge direction.
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

    // Determine which nodes to use as sources
    // Use stride-based sampling to ensure even coverage across the graph,
    // avoiding bias from sequential selection (e.g. first k nodes being
    // Module/Class containers with no outgoing edges of the filtered type).
    let source_indices: Vec<usize> = if let Some(k) = sample_size {
        let k = k.min(n);
        if k == n {
            (0..n).collect()
        } else {
            let step = n as f64 / k as f64;
            (0..k).map(|i| (i as f64 * step) as usize).collect()
        }
    } else {
        (0..n).collect()
    };

    // Parallel vs sequential Brandes' algorithm
    let use_parallel = n >= 4096;

    // Shared timeout flag for the parallel path: rayon closures can't bubble
    // up Result, so we set this on deadline expiry and check after the join.
    let timed_out = AtomicBool::new(false);

    let mut betweenness: Vec<f64> = if use_parallel {
        use rayon::prelude::*;

        let adj_ref = &adj;
        let deadline_ref = &deadline;
        let timed_out_ref = &timed_out;
        let num_threads = rayon::current_num_threads();
        let chunk_size = (source_indices.len() / num_threads).max(1);

        // Thread-local accumulation + reduction (avoids write conflicts on shared array)
        source_indices
            .par_chunks(chunk_size)
            .map(|chunk| {
                // Thread-local data structures (allocated once per thread)
                let mut local_betweenness: Vec<f64> = vec![0.0; n];
                let mut stack: Vec<usize> = Vec::with_capacity(n);
                let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
                let mut sigma: Vec<f64> = vec![0.0; n];
                let mut dist: Vec<i64> = vec![-1; n];
                let mut delta: Vec<f64> = vec![0.0; n];
                let mut queue: VecDeque<usize> = VecDeque::with_capacity(n);

                for (local_counter, &s_idx) in chunk.iter().enumerate() {
                    // Periodic timeout check (every 10 sources within this chunk)
                    if local_counter % 10 == 0 {
                        if timed_out_ref.load(Ordering::Relaxed) {
                            break;
                        }
                        if deadline_ref.exceeded() {
                            {
                                timed_out_ref.store(true, Ordering::Relaxed);
                                break;
                            }
                        }
                    }

                    // Reset only stack/queue
                    stack.clear();
                    queue.clear();

                    // Initialize source
                    sigma[s_idx] = 1.0;
                    dist[s_idx] = 0;
                    queue.push_back(s_idx);

                    // BFS phase
                    while let Some(v_idx) = queue.pop_front() {
                        stack.push(v_idx);
                        let v_dist = dist[v_idx];

                        for &w_idx in &adj_ref[v_idx] {
                            let d = dist[w_idx];
                            if d < 0 {
                                dist[w_idx] = v_dist + 1;
                                queue.push_back(w_idx);
                                sigma[w_idx] += sigma[v_idx];
                                pred[w_idx].push(v_idx);
                            } else if d == v_dist + 1 {
                                sigma[w_idx] += sigma[v_idx];
                                pred[w_idx].push(v_idx);
                            }
                        }
                    }

                    // Accumulation phase + sparse reset
                    while let Some(w_idx) = stack.pop() {
                        for &v_idx in &pred[w_idx] {
                            let contribution = (sigma[v_idx] / sigma[w_idx]) * (1.0 + delta[w_idx]);
                            delta[v_idx] += contribution;
                        }
                        if w_idx != s_idx {
                            local_betweenness[w_idx] += delta[w_idx];
                        }
                        pred[w_idx].clear();
                        sigma[w_idx] = 0.0;
                        dist[w_idx] = -1;
                        delta[w_idx] = 0.0;
                    }
                }

                local_betweenness
            })
            .reduce(
                || vec![0.0; n],
                |mut a, b| {
                    for i in 0..n {
                        a[i] += b[i];
                    }
                    a
                },
            )
    } else {
        // Sequential path (n < 4096): reuses pre-allocated buffers across iterations
        let mut betweenness: Vec<f64> = vec![0.0; n];
        let mut stack: Vec<usize> = Vec::with_capacity(n);
        let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut sigma: Vec<f64> = vec![0.0; n];
        let mut dist: Vec<i64> = vec![-1; n];
        let mut delta: Vec<f64> = vec![0.0; n];
        let mut queue: VecDeque<usize> = VecDeque::with_capacity(n);

        for (source_counter, &s_idx) in source_indices.iter().enumerate() {
            // Periodic timeout check (every 10 source nodes)
            if source_counter.is_multiple_of(10) && deadline.exceeded() {
                {
                    return Err(algorithm_timeout_err());
                }
            }

            stack.clear();
            queue.clear();

            sigma[s_idx] = 1.0;
            dist[s_idx] = 0;
            queue.push_back(s_idx);

            while let Some(v_idx) = queue.pop_front() {
                stack.push(v_idx);
                let v_dist = dist[v_idx];

                for &w_idx in &adj[v_idx] {
                    let d = dist[w_idx];
                    if d < 0 {
                        dist[w_idx] = v_dist + 1;
                        queue.push_back(w_idx);
                        sigma[w_idx] += sigma[v_idx];
                        pred[w_idx].push(v_idx);
                    } else if d == v_dist + 1 {
                        sigma[w_idx] += sigma[v_idx];
                        pred[w_idx].push(v_idx);
                    }
                }
            }

            while let Some(w_idx) = stack.pop() {
                for &v_idx in &pred[w_idx] {
                    let contribution = (sigma[v_idx] / sigma[w_idx]) * (1.0 + delta[w_idx]);
                    delta[v_idx] += contribution;
                }
                if w_idx != s_idx {
                    betweenness[w_idx] += delta[w_idx];
                }
                pred[w_idx].clear();
                sigma[w_idx] = 0.0;
                dist[w_idx] = -1;
                delta[w_idx] = 0.0;
            }
        }

        betweenness
    };

    // Surface deadline expiry from the parallel rayon path (set via timed_out flag).
    if timed_out.load(Ordering::Relaxed) {
        return Err(algorithm_timeout_err());
    }

    // Undirected BFS counts each (s,t) pair twice, so halve raw scores.
    for score in betweenness.iter_mut() {
        *score /= 2.0;
    }

    // Normalize if requested
    // For undirected graphs: 2 / ((n-1)*(n-2))
    if normalized && n > 2 {
        let scale = 2.0 / ((n - 1) as f64 * (n - 2) as f64);
        for score in betweenness.iter_mut() {
            *score *= scale;
        }
    }

    // If we sampled, scale up the scores
    if let Some(k) = sample_size {
        if k < n {
            let scale = n as f64 / k as f64;
            for score in betweenness.iter_mut() {
                *score *= scale;
            }
        }
    }

    // Convert to sorted results
    let mut results: Vec<CentralityResult> = nodes
        .iter()
        .enumerate()
        .map(|(i, &node_idx)| CentralityResult {
            node_idx,
            score: betweenness[i],
        })
        .collect();

    // Sort by score descending
    results.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(results)
}

/// Calculate PageRank centrality for all nodes in the graph.
///
/// PageRank measures the importance of nodes based on the structure of incoming links.
/// Originally developed by Google for ranking web pages.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `damping_factor` - Probability of following a link (typically 0.85)
/// * `max_iterations` - Maximum number of iterations (default: 100)
/// * `tolerance` - Convergence threshold (default: 1e-6)
///
/// @procedure: pagerank
pub fn pagerank(
    graph: &DirGraph,
    options: &PagerankOptions,
) -> Result<Vec<CentralityResult>, String> {
    let PagerankOptions {
        damping_factor,
        max_iterations,
        tolerance,
        connection_types,
        scope,
        interrupt: deadline,
    } = *options;
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let nodes: Vec<NodeIndex> = scoped_node_set(graph, scope);
    let n = nodes.len();

    if n == 0 {
        return Ok(Vec::new());
    }

    // Use Vec-based index mapping for O(1) lookup (vs HashMap)
    let bound = graph.graph.node_bound();
    let mut node_to_idx = vec![0usize; bound];
    for (i, &node) in nodes.iter().enumerate() {
        node_to_idx[node.index()] = i;
    }

    // Pre-build reverse adjacency list: for each target j, store list of source indices.
    // Pull-based formulation: each target reads from its in-neighbors independently,
    // enabling rayon parallelization (no write conflicts on new_pr[j]).
    let interned_ct = intern_connection_types(connection_types);
    let mut in_adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut out_degrees: Vec<usize> = vec![0; n];
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
        in_adj[tgt_i].push(src_i);
        out_degrees[src_i] += 1;
    }

    // Initialize PageRank scores (uniform distribution)
    let mut pr: Vec<f64> = vec![1.0 / n as f64; n];
    let mut new_pr: Vec<f64> = vec![0.0; n];

    // Precompute inverse out-degree (multiply instead of divide in hot loop)
    let inv_out_degrees: Vec<f64> = out_degrees
        .iter()
        .map(|&d| {
            if d > 0 {
                damping_factor / d as f64
            } else {
                0.0
            }
        })
        .collect();

    // Identify dangling nodes (no outgoing links) — store as bitmask for fast sum
    let is_dangling: Vec<bool> = out_degrees.iter().map(|&d| d == 0).collect();

    let teleport = (1.0 - damping_factor) / n as f64;
    let inv_n = 1.0 / n as f64;
    let use_parallel = n >= 4096;

    // Iterative computation
    for _iteration in 0..max_iterations {
        // Timeout check each iteration — error rather than return partial.
        // Half-converged PageRank scores are misleading; an explicit error
        // tells the caller to extend timeout_ms or scope the graph.
        if deadline.exceeded() {
            {
                return Err(algorithm_timeout_err());
            }
        }

        // Calculate dangling node contribution
        let dangling_sum: f64 = if use_parallel {
            use rayon::prelude::*;
            (0..n)
                .into_par_iter()
                .filter(|&i| is_dangling[i])
                .map(|i| pr[i])
                .sum()
        } else {
            (0..n).filter(|&i| is_dangling[i]).map(|i| pr[i]).sum()
        };
        let base_score = teleport + damping_factor * dangling_sum * inv_n;

        // Pull-based PageRank: each target j computes its own score independently.
        // No write conflicts → parallelizable with rayon.
        if use_parallel {
            use rayon::prelude::*;
            new_pr.par_iter_mut().enumerate().for_each(|(j, score)| {
                let mut s = base_score;
                for &src in &in_adj[j] {
                    s += inv_out_degrees[src] * pr[src];
                }
                *score = s;
            });
        } else {
            for j in 0..n {
                let mut s = base_score;
                for &src in &in_adj[j] {
                    s += inv_out_degrees[src] * pr[src];
                }
                new_pr[j] = s;
            }
        }

        // Check for convergence (L1 norm)
        let diff: f64 = if use_parallel {
            use rayon::prelude::*;
            pr.par_iter()
                .zip(new_pr.par_iter())
                .map(|(a, b)| (a - b).abs())
                .sum()
        } else {
            pr.iter()
                .zip(new_pr.iter())
                .map(|(a, b)| (a - b).abs())
                .sum()
        };

        std::mem::swap(&mut pr, &mut new_pr);

        if diff < tolerance {
            break;
        }
    }

    // Convert to results and sort by score
    let mut results: Vec<CentralityResult> = nodes
        .iter()
        .enumerate()
        .map(|(i, &node_idx)| CentralityResult {
            node_idx,
            score: pr[i],
        })
        .collect();

    results.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(results)
}

/// Calculate degree centrality for all nodes.
///
/// Simply counts the number of connections each node has.
/// Optionally normalized by (n-1) to get values between 0 and 1.
///
/// @procedure: degree
/// @procedure: degree_centrality
pub fn degree_centrality(
    graph: &DirGraph,
    options: &DegreeCentralityOptions,
) -> Result<Vec<CentralityResult>, String> {
    let DegreeCentralityOptions {
        normalized,
        connection_types,
        scope,
        interrupt: deadline,
    } = *options;
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let nodes: Vec<NodeIndex> = scoped_node_set(graph, scope);
    let n = nodes.len();

    if n == 0 {
        return Ok(Vec::new());
    }

    let scale = if normalized && n > 1 {
        1.0 / (n - 1) as f64
    } else {
        1.0
    };

    // Compute all degrees in a single pass over edges instead of per-node traversal.
    // Periodic deadline check (every ~1M edges) keeps overhead negligible while
    // ensuring 863M-edge scans on Wikidata-scale graphs honor the 20s timeout.
    let interned_ct = intern_connection_types(connection_types);
    let bound = graph.graph.node_bound();
    let mut degrees = vec![0usize; bound];
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
        if let Some(ref types) = interned_ct {
            if !types.iter().any(|t| *t == edge.connection_type()) {
                continue;
            }
        }
        if !edge_in_scope(scope, edge.source(), edge.target()) {
            continue;
        }
        degrees[edge.source().index()] += 1; // out-degree
        degrees[edge.target().index()] += 1; // in-degree
    }

    let mut results: Vec<CentralityResult> = nodes
        .iter()
        .map(|&node_idx| CentralityResult {
            node_idx,
            score: degrees[node_idx.index()] as f64 * scale,
        })
        .collect();

    results.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(results)
}

/// Calculate closeness centrality for all nodes.
///
/// Closeness centrality measures how close a node is to all other nodes.
/// Defined as the reciprocal of the sum of shortest path distances.
///
/// Note: For disconnected graphs, only reachable nodes are considered.
/// Optimized to use Vec instead of HashMap for O(1) direct indexing.
///
/// * `sample_size` - Optional number of source nodes to sample (for large graphs).
///   Uses stride-based selection for even coverage.
///
/// @procedure: closeness
/// @procedure: closeness_centrality
pub fn closeness_centrality(
    graph: &DirGraph,
    options: &CentralityOptions,
) -> Result<Vec<CentralityResult>, String> {
    let CentralityOptions {
        normalized,
        sample_size,
        connection_types,
        scope,
        interrupt: deadline,
    } = *options;
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    use std::sync::atomic::{AtomicBool, Ordering};

    let nodes: Vec<NodeIndex> = scoped_node_set(graph, scope);
    let n = nodes.len();

    if n == 0 {
        return Ok(Vec::new());
    }

    // Use Vec-based index mapping for O(1) lookup (vs HashMap)
    let bound = graph.graph.node_bound();
    let mut node_to_idx = vec![0usize; bound];
    for (i, &node) in nodes.iter().enumerate() {
        node_to_idx[node.index()] = i;
    }

    // Pre-build incoming adjacency list: for closeness centrality on directed graphs,
    // we BFS via incoming edges (convention: d(v, u) = how easy for v to reach u)
    let interned_ct = intern_connection_types(connection_types);
    let mut adj_incoming: Vec<Vec<usize>> = vec![Vec::new(); n];
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
        // For incoming BFS from node u: follow edges pointing INTO u
        // edge: src -> tgt, so tgt has incoming edge from src
        adj_incoming[tgt_i].push(src_i);
    }
    // Dedup incoming adjacency (handles duplicate edges)
    for neighbors in &mut adj_incoming {
        neighbors.sort_unstable();
        neighbors.dedup();
    }

    // Determine which nodes to use as sources.
    // Stride-based sampling ensures even coverage across the graph.
    let source_indices: Vec<usize> = if let Some(k) = sample_size {
        let k = k.min(n);
        if k == n {
            (0..n).collect()
        } else {
            let step = n as f64 / k as f64;
            (0..k).map(|i| (i as f64 * step) as usize).collect()
        }
    } else {
        (0..n).collect()
    };

    // Parallel path: each source BFS is independent, no shared accumulator
    let use_parallel = source_indices.len() >= 4096;

    // Shared timeout flag for the parallel rayon path; checked after the join.
    let timed_out = AtomicBool::new(false);

    if use_parallel {
        use rayon::prelude::*;

        let adj_ref = &adj_incoming;
        let deadline_ref = &deadline;
        let nodes_ref = &nodes;
        let timed_out_ref = &timed_out;

        let mut results: Vec<CentralityResult> = source_indices
            .par_iter()
            .enumerate()
            .map(|(i, &s_idx)| {
                let source = nodes_ref[s_idx];

                // Periodic timeout check (every 100 sources)
                if i % 100 == 0 {
                    if timed_out_ref.load(Ordering::Relaxed) {
                        return CentralityResult {
                            node_idx: source,
                            score: 0.0,
                        };
                    }
                    if deadline_ref.exceeded() {
                        {
                            timed_out_ref.store(true, Ordering::Relaxed);
                            return CentralityResult {
                                node_idx: source,
                                score: 0.0,
                            };
                        }
                    }
                }

                // Thread-local BFS data structures
                let mut dist: Vec<i64> = vec![-1; n];
                let mut current_level: Vec<usize> = Vec::with_capacity(n / 4);
                let mut next_level: Vec<usize> = Vec::with_capacity(n / 4);
                let mut touched: Vec<usize> = Vec::with_capacity(n / 4);

                // Initialize source
                current_level.push(s_idx);
                dist[s_idx] = 0;
                touched.push(s_idx);
                let mut depth: i64 = 0;

                // Level-based BFS via incoming edges
                while !current_level.is_empty() {
                    depth += 1;
                    next_level.clear();

                    for &current_idx in &current_level {
                        for &neighbor_idx in &adj_ref[current_idx] {
                            if dist[neighbor_idx] < 0 {
                                dist[neighbor_idx] = depth;
                                next_level.push(neighbor_idx);
                                touched.push(neighbor_idx);
                            }
                        }
                    }

                    std::mem::swap(&mut current_level, &mut next_level);
                }

                // Calculate closeness from touched nodes only
                let reachable = touched.len();
                let total_distance: i64 = touched.iter().map(|&idx| dist[idx]).sum();

                if reachable > 1 && total_distance > 0 {
                    let closeness = (reachable - 1) as f64 / total_distance as f64;
                    let score = if normalized {
                        closeness * (reachable - 1) as f64 / (n - 1) as f64
                    } else {
                        closeness
                    };
                    CentralityResult {
                        node_idx: source,
                        score,
                    }
                } else {
                    CentralityResult {
                        node_idx: source,
                        score: 0.0,
                    }
                }
            })
            .collect();

        if timed_out.load(Ordering::Relaxed) {
            return Err(algorithm_timeout_err());
        }

        results.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        return Ok(results);
    }

    // Sequential path: reuses pre-allocated buffers across iterations
    let mut results = Vec::with_capacity(source_indices.len());
    let mut dist: Vec<i64> = vec![-1; n];
    let mut current_level: Vec<usize> = Vec::with_capacity(n);
    let mut next_level: Vec<usize> = Vec::with_capacity(n);
    let mut touched: Vec<usize> = Vec::with_capacity(n);

    for (i, &s_idx) in source_indices.iter().enumerate() {
        let source = nodes[s_idx];

        // Periodic timeout check (every 10 source nodes)
        if i.is_multiple_of(10) && deadline.exceeded() {
            {
                return Err(algorithm_timeout_err());
            }
        }

        // Sparse reset from previous iteration (only visited nodes)
        for &idx in &touched {
            dist[idx] = -1;
        }
        touched.clear();
        current_level.clear();

        // Initialize source
        current_level.push(s_idx);
        dist[s_idx] = 0;
        touched.push(s_idx);
        let mut depth: i64 = 0;

        // Level-based BFS via incoming edges
        while !current_level.is_empty() {
            depth += 1;
            next_level.clear();

            for &current_idx in &current_level {
                for &neighbor_idx in &adj_incoming[current_idx] {
                    if dist[neighbor_idx] < 0 {
                        dist[neighbor_idx] = depth;
                        next_level.push(neighbor_idx);
                        touched.push(neighbor_idx);
                    }
                }
            }

            std::mem::swap(&mut current_level, &mut next_level);
        }

        // Calculate closeness from touched nodes only (not all N)
        let reachable = touched.len();
        let total_distance: i64 = touched.iter().map(|&idx| dist[idx]).sum();

        if reachable > 1 && total_distance > 0 {
            let closeness = (reachable - 1) as f64 / total_distance as f64;

            let score = if normalized {
                closeness * (reachable - 1) as f64 / (n - 1) as f64
            } else {
                closeness
            };

            results.push(CentralityResult {
                node_idx: source,
                score,
            });
        } else {
            results.push(CentralityResult {
                node_idx: source,
                score: 0.0,
            });
        }
    }

    results.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(results)
}
