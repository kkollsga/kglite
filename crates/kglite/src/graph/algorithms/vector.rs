// Vector search module for embedding-based similarity queries.
// Operates on the current graph selection for filtered vector search.

use super::hnsw::{HnswIndex, HnswMetric};
use crate::graph::schema::{CurrentSelection, DirGraph, EmbeddingStore};
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;
use std::collections::{BinaryHeap, HashSet};

/// Distance metric for vector similarity search.
#[derive(Clone, Copy, Debug)]
pub enum DistanceMetric {
    Cosine,
    DotProduct,
    Euclidean,
    Poincare,
}

impl DistanceMetric {
    /// Parse the Cypher-facing metric name (`'cosine'`, `'dot_product'`,
    /// `'euclidean'`, `'poincare'`). Single source of truth so every
    /// `vector_score` / `text_score` call site agrees on the spelling.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "cosine" => Some(DistanceMetric::Cosine),
            "dot_product" => Some(DistanceMetric::DotProduct),
            "euclidean" => Some(DistanceMetric::Euclidean),
            "poincare" => Some(DistanceMetric::Poincare),
            _ => None,
        }
    }
}

/// A single vector search result: node index + similarity score.
#[derive(Clone, Debug)]
pub struct VectorSearchResult {
    pub node_idx: NodeIndex,
    pub score: f32,
}

/// Tunable options for [`vector_search`]. The selection, embedding property,
/// and query vector are positional (primary inputs); the ranking knobs live
/// here. Construct via [`VectorSearchOptions::default`] then the `with_*`
/// builders, e.g. `VectorSearchOptions::default().with_top_k(20)`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct VectorSearchOptions {
    /// Number of results to return (default `10`).
    pub top_k: usize,
    /// Distance metric (default [`DistanceMetric::Cosine`]).
    pub metric: DistanceMetric,
    /// Force a full exact scan, bypassing any HNSW index (default `false`).
    pub exact: bool,
}

impl Default for VectorSearchOptions {
    fn default() -> Self {
        Self {
            top_k: 10,
            metric: DistanceMetric::Cosine,
            exact: false,
        }
    }
}

impl VectorSearchOptions {
    /// Set the number of results to return.
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }
    /// Set the distance metric.
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }
    /// Force an exact full scan (bypass HNSW).
    pub fn with_exact(mut self, exact: bool) -> Self {
        self.exact = exact;
        self
    }
}

/// Threshold for switching to parallel search via rayon.
const PARALLEL_THRESHOLD: usize = 10_000;

/// Minimum candidate count before an HNSW index is auto-used. Below this a
/// brute-force scan is both faster (no index overhead) and exact, so there's no
/// reason to risk approximate recall.
const HNSW_AUTO_MIN: usize = 256;

/// Over-fetch factor for HNSW: fetch `top_k * this` candidates so that, after
/// dropping any that fall outside the (possibly filtered) selection, `top_k`
/// survive. If too few survive, the caller falls back to an exact scan.
const HNSW_OVERSAMPLE: usize = 4;

/// Perform vector search over the current selection.
///
/// Gets candidate nodes from the selection's current level, computes similarity
/// for each candidate that has an embedding, and returns top-k results sorted
/// by score (descending for cosine/dot, ascending for euclidean).
///
/// When `exact` is false and the store carries an HNSW index whose metric
/// supports it (cosine/dot/euclidean) and the selection covers enough of the
/// store, search dispatches through the index (approximate, much faster on large
/// stores). `exact = true` always forces the full linear scan. The Poincaré
/// metric, small/heavily-filtered selections, and stores without an index all
/// fall back to the (norm-accelerated) exact scan.
pub fn vector_search(
    graph: &DirGraph,
    selection: &CurrentSelection,
    embedding_property: &str,
    query_vector: &[f32],
    options: &VectorSearchOptions,
) -> Result<Vec<VectorSearchResult>, String> {
    // Direct GraphRead traversal outside the executor: hold the disk arena
    // guard while node/edge weights are borrowed (owned counter handle;
    // no-op on memory/mapped backends). Results returned are owned.
    let _arena_guard = graph.graph.begin_query();
    let VectorSearchOptions {
        top_k,
        metric,
        exact,
    } = *options;
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let level_count = selection.get_level_count();
    if level_count == 0 {
        return Ok(Vec::new());
    }

    let candidates: Vec<NodeIndex> = selection
        .get_level(level_count - 1)
        .map(|l| l.get_all_nodes())
        .unwrap_or_default();

    if candidates.is_empty() || top_k == 0 {
        return Ok(Vec::new());
    }

    // Fast path: check if first candidate's type has an embedding store (common after type_filter)
    let first_type: Option<&str> = graph
        .graph
        .node_weight(candidates[0])
        .map(|n| n.node_type_str(&graph.interner));

    let single_type = first_type.and_then(|ft| {
        let key = (ft.to_string(), embedding_property.to_string());
        graph.embeddings.get(&key).map(|store| (ft, store))
    });

    let results = if let Some((node_type, store)) = single_type {
        // Validate query vector dimension
        if query_vector.len() != store.dimension {
            return Err(format!(
                "Query vector dimension {} does not match embedding dimension {} for '{}.{}'",
                query_vector.len(),
                store.dimension,
                node_type,
                embedding_property
            ));
        }

        let scorer = Scorer::new(metric, query_vector);

        // HNSW fast path: use the index when allowed, supported, and the
        // selection covers enough of the store. Returns None (→ exact fallback)
        // if a selective filter left fewer than top_k survivors.
        let hnsw_result = if exact {
            None
        } else {
            store.index.as_ref().and_then(|idx| {
                let eligible = HnswMetric::from_distance(metric).is_some()
                    && candidates.len() >= HNSW_AUTO_MIN
                    && candidates.len().saturating_mul(2) >= store.len();
                if eligible {
                    hnsw_search(store, idx, &candidates, query_vector, top_k, &scorer)
                } else {
                    None
                }
            })
        };

        match hnsw_result {
            Some(r) => r,
            None if candidates.len() > PARALLEL_THRESHOLD => {
                parallel_search(&candidates, store, query_vector, top_k, &scorer)
            }
            None => sequential_search(&candidates, store, query_vector, top_k, &scorer),
        }
    } else {
        // Multi-type path: group by node type
        let scorer = Scorer::new(metric, query_vector);
        let mut heap = MinHeap::with_capacity(top_k);

        for &node_idx in &candidates {
            let node_type = match graph.graph.node_weight(node_idx) {
                Some(n) => n.node_type_str(&graph.interner),
                None => continue,
            };

            let key = (node_type.to_string(), embedding_property.to_string());
            let store = match graph.embeddings.get(&key) {
                Some(s) => s,
                None => continue,
            };

            if query_vector.len() != store.dimension {
                return Err(format!(
                    "Query vector dimension {} does not match embedding dimension {} for '{}.{}'",
                    query_vector.len(),
                    store.dimension,
                    node_type,
                    embedding_property
                ));
            }

            if let Some((embedding, norm)) = store.get_embedding_with_norm(node_idx.index()) {
                let score = scorer.score(query_vector, embedding, norm);
                heap.push_if_better(node_idx, score, top_k);
            }
        }

        heap.into_sorted_results()
    };

    Ok(results)
}

/// HNSW-backed top-k over a single store, restricted to `candidates` (the
/// selection). Fetches an over-sampled candidate set from the index, drops any
/// whose node falls outside the selection, then re-scores the survivors with the
/// shared `Scorer` so the returned scores are on the exact same scale as a
/// brute-force scan (the ANN step only narrows *which* nodes are scored, never
/// changes the score formula).
///
/// Returns `None` to signal "fall back to an exact scan" when a selective filter
/// leaves fewer than `top_k` survivors — guaranteeing correctness when the
/// filter is tight enough that the index's over-fetch wasn't sufficient.
fn hnsw_search(
    store: &EmbeddingStore,
    idx: &HnswIndex,
    candidates: &[NodeIndex],
    query: &[f32],
    top_k: usize,
    scorer: &Scorer,
) -> Option<Vec<VectorSearchResult>> {
    let whole_store = candidates.len() >= store.len();
    // Membership test for the filtered case; skipped when the selection is the
    // whole store (every slot trivially qualifies).
    let membership: Option<HashSet<usize>> = if whole_store {
        None
    } else {
        Some(candidates.iter().map(|n| n.index()).collect())
    };

    let query_norm = dot_product(query, query).sqrt();
    let k_fetch = top_k
        .saturating_mul(HNSW_OVERSAMPLE)
        .min(store.len())
        .max(top_k);
    let ef = k_fetch.max(idx.params().ef_search);
    let raw = idx.search(
        query,
        query_norm,
        k_fetch,
        Some(ef),
        &store.data,
        &store.norms,
    );

    let mut heap = MinHeap::with_capacity(top_k);
    for (slot, _dist) in raw {
        let node_raw = store.slot_to_node[slot as usize];
        if let Some(set) = &membership {
            if !set.contains(&node_raw) {
                continue;
            }
        }
        let start = slot as usize * store.dimension;
        let emb = &store.data[start..start + store.dimension];
        let norm = store.norms[slot as usize];
        let score = scorer.score(query, emb, norm);
        heap.push_if_better(NodeIndex::new(node_raw), score, top_k);
    }

    let results = heap.into_sorted_results();
    // Whole-store: the index's recall is the only limiter and that's the ANN
    // contract — accept it. Filtered: if the over-fetch didn't survive the
    // filter down to top_k, bail to an exact scan for a correct result.
    if !whole_store && results.len() < top_k {
        return None;
    }
    Some(results)
}

// ─── Similarity Functions ──────────────────────────────────────────────────────

type SimilarityFn = fn(&[f32], &[f32]) -> f32;

/// A query-bound scorer. Built once per search (the query vector is constant
/// across all candidates), then applied to each candidate alongside its cached
/// L2 norm.
///
/// Cosine is special-cased: with the query's norm precomputed once and each
/// stored vector's norm cached in the `EmbeddingStore`, the per-candidate work
/// collapses from "dot + two norm sweeps + sqrt" to a single dot product and a
/// divide. Every other metric needs the raw vectors (dot, euclidean) or their
/// magnitudes recomputed per pair (Poincaré is non-linear in the norms), so
/// they fall through to the plain kernel and the cached norm is ignored.
///
/// Build once per query via [`Scorer::new`] (the query vector is constant across
/// all candidates), then call [`Scorer::score`] per candidate. Shared by the
/// fluent `vector_search` path and the Cypher `vector_score` / `text_score`
/// scalar function so all cosine scoring benefits from the cached norm.
#[derive(Clone, Copy)]
pub struct Scorer {
    kind: ScorerKind,
}

#[derive(Clone, Copy)]
enum ScorerKind {
    Cosine { query_norm: f32 },
    Generic(SimilarityFn),
}

impl Scorer {
    pub fn new(metric: DistanceMetric, query: &[f32]) -> Self {
        let kind = match metric {
            // dot_product(q, q) reuses the cosine kernel's accumulator layout,
            // so query_norm matches the norm cosine_similarity would compute inline.
            DistanceMetric::Cosine => ScorerKind::Cosine {
                query_norm: dot_product(query, query).sqrt(),
            },
            DistanceMetric::DotProduct => ScorerKind::Generic(dot_product),
            DistanceMetric::Euclidean => ScorerKind::Generic(neg_euclidean_distance),
            DistanceMetric::Poincare => ScorerKind::Generic(neg_poincare_distance),
        };
        Scorer { kind }
    }

    /// Score a candidate. `emb_norm` is the candidate's cached L2 norm
    /// (`EmbeddingStore::get_embedding_with_norm`); it is consumed only by the
    /// cosine path and ignored by the others.
    #[inline]
    pub fn score(&self, query: &[f32], emb: &[f32], emb_norm: f32) -> f32 {
        match self.kind {
            ScorerKind::Cosine { query_norm } => {
                let denom = query_norm * emb_norm;
                if denom > 0.0 {
                    dot_product(query, emb) / denom
                } else {
                    0.0
                }
            }
            ScorerKind::Generic(f) => f(query, emb),
        }
    }
}

/// Cosine similarity between two f32 slices.
/// Uses 4 independent accumulators per metric for instruction-level parallelism,
/// with chunks_exact(8) for LLVM auto-vectorization (SSE2/AVX2/NEON).
/// Returns similarity in [-1.0, 1.0].
///
/// Standalone SIMD util exercised by this module's tests; the live vector
/// scoring path inlines its own distance math (`vector_score`), so this is
/// not currently wired into production — kept (and tested) for reuse.
#[allow(dead_code)]
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // 4 independent accumulators break the loop-carried dependency chain,
    // allowing the CPU to pipeline multiply-add operations.
    let (mut dot0, mut dot1, mut dot2, mut dot3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut na0, mut na1, mut na2, mut na3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut nb0, mut nb1, mut nb2, mut nb3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);
    let a_rem = a_chunks.remainder();
    let b_rem = b_chunks.remainder();

    for (ac, bc) in a_chunks.zip(b_chunks) {
        dot0 += ac[0] * bc[0];
        dot1 += ac[1] * bc[1];
        dot2 += ac[2] * bc[2];
        dot3 += ac[3] * bc[3];
        na0 += ac[0] * ac[0];
        na1 += ac[1] * ac[1];
        na2 += ac[2] * ac[2];
        na3 += ac[3] * ac[3];
        nb0 += bc[0] * bc[0];
        nb1 += bc[1] * bc[1];
        nb2 += bc[2] * bc[2];
        nb3 += bc[3] * bc[3];

        dot0 += ac[4] * bc[4];
        dot1 += ac[5] * bc[5];
        dot2 += ac[6] * bc[6];
        dot3 += ac[7] * bc[7];
        na0 += ac[4] * ac[4];
        na1 += ac[5] * ac[5];
        na2 += ac[6] * ac[6];
        na3 += ac[7] * ac[7];
        nb0 += bc[4] * bc[4];
        nb1 += bc[5] * bc[5];
        nb2 += bc[6] * bc[6];
        nb3 += bc[7] * bc[7];
    }
    for (av, bv) in a_rem.iter().zip(b_rem.iter()) {
        dot0 += av * bv;
        na0 += av * av;
        nb0 += bv * bv;
    }

    let dot = (dot0 + dot1) + (dot2 + dot3);
    let norm_a = (na0 + na1) + (na2 + na3);
    let norm_b = (nb0 + nb1) + (nb2 + nb3);

    let denom = (norm_a * norm_b).sqrt();
    if denom > 0.0 {
        dot / denom
    } else {
        0.0
    }
}

/// Dot product similarity.
/// Uses 4 independent accumulators for instruction-level parallelism.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);
    let a_rem = a_chunks.remainder();
    let b_rem = b_chunks.remainder();

    for (ac, bc) in a_chunks.zip(b_chunks) {
        s0 += ac[0] * bc[0];
        s1 += ac[1] * bc[1];
        s2 += ac[2] * bc[2];
        s3 += ac[3] * bc[3];
        s0 += ac[4] * bc[4];
        s1 += ac[5] * bc[5];
        s2 += ac[6] * bc[6];
        s3 += ac[7] * bc[7];
    }
    for (av, bv) in a_rem.iter().zip(b_rem.iter()) {
        s0 += av * bv;
    }

    (s0 + s1) + (s2 + s3)
}

/// Negative Euclidean distance (higher = more similar).
/// Uses 4 independent accumulators for instruction-level parallelism.
#[inline]
pub fn neg_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);
    let a_rem = a_chunks.remainder();
    let b_rem = b_chunks.remainder();

    for (ac, bc) in a_chunks.zip(b_chunks) {
        let d0 = ac[0] - bc[0];
        let d1 = ac[1] - bc[1];
        let d2 = ac[2] - bc[2];
        let d3 = ac[3] - bc[3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
        let d4 = ac[4] - bc[4];
        let d5 = ac[5] - bc[5];
        let d6 = ac[6] - bc[6];
        let d7 = ac[7] - bc[7];
        s0 += d4 * d4;
        s1 += d5 * d5;
        s2 += d6 * d6;
        s3 += d7 * d7;
    }
    for (av, bv) in a_rem.iter().zip(b_rem.iter()) {
        let d = av - bv;
        s0 += d * d;
    }

    -((s0 + s1) + (s2 + s3)).sqrt()
}

/// Negative Poincaré distance (higher = more similar).
///
/// Computes the hyperbolic distance in the Poincaré ball model:
///   d(u,v) = acosh(1 + 2 * ||u-v||² / ((1-||u||²)(1-||v||²)))
///
/// Negated so that higher values indicate greater similarity, consistent with
/// the other metrics. Vectors must lie inside the unit ball (||x|| < 1).
///
/// Based on Nickel & Kiela (2017), "Poincaré Embeddings for Learning
/// Hierarchical Representations". Particularly effective for data with latent
/// hierarchical structure (taxonomies, ontologies, org charts).
///
/// Uses 4 independent accumulators for instruction-level parallelism.
#[inline]
pub fn neg_poincare_distance(a: &[f32], b: &[f32]) -> f32 {
    // Compute ||a||², ||b||², and ||a-b||² in a single pass.
    let (mut na0, mut na1, mut na2, mut na3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut nb0, mut nb1, mut nb2, mut nb3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut d0, mut d1, mut d2, mut d3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);
    let a_rem = a_chunks.remainder();
    let b_rem = b_chunks.remainder();

    for (ac, bc) in a_chunks.zip(b_chunks) {
        na0 += ac[0] * ac[0];
        na1 += ac[1] * ac[1];
        na2 += ac[2] * ac[2];
        na3 += ac[3] * ac[3];
        nb0 += bc[0] * bc[0];
        nb1 += bc[1] * bc[1];
        nb2 += bc[2] * bc[2];
        nb3 += bc[3] * bc[3];
        let dd0 = ac[0] - bc[0];
        let dd1 = ac[1] - bc[1];
        let dd2 = ac[2] - bc[2];
        let dd3 = ac[3] - bc[3];
        d0 += dd0 * dd0;
        d1 += dd1 * dd1;
        d2 += dd2 * dd2;
        d3 += dd3 * dd3;

        na0 += ac[4] * ac[4];
        na1 += ac[5] * ac[5];
        na2 += ac[6] * ac[6];
        na3 += ac[7] * ac[7];
        nb0 += bc[4] * bc[4];
        nb1 += bc[5] * bc[5];
        nb2 += bc[6] * bc[6];
        nb3 += bc[7] * bc[7];
        let dd4 = ac[4] - bc[4];
        let dd5 = ac[5] - bc[5];
        let dd6 = ac[6] - bc[6];
        let dd7 = ac[7] - bc[7];
        d0 += dd4 * dd4;
        d1 += dd5 * dd5;
        d2 += dd6 * dd6;
        d3 += dd7 * dd7;
    }
    for (av, bv) in a_rem.iter().zip(b_rem.iter()) {
        na0 += av * av;
        nb0 += bv * bv;
        let dd = av - bv;
        d0 += dd * dd;
    }

    let norm_a_sq = (na0 + na1) + (na2 + na3);
    let norm_b_sq = (nb0 + nb1) + (nb2 + nb3);
    let diff_sq = (d0 + d1) + (d2 + d3);

    // Clamp norms to stay inside the Poincaré ball (||x|| < 1).
    // Embeddings exactly on the boundary would produce infinite distance.
    let alpha = (1.0 - norm_a_sq).max(1e-7); // 1 - ||a||²
    let beta = (1.0 - norm_b_sq).max(1e-7); // 1 - ||b||²

    // γ = 1 + 2 * ||a-b||² / ((1-||a||²)(1-||b||²))
    let gamma = 1.0 + 2.0 * diff_sq / (alpha * beta);

    // Clamp γ ≥ 1 for numerical stability (acosh domain).
    let gamma = gamma.max(1.0);

    // acosh(γ) = ln(γ + √(γ²-1))
    let dist = (gamma + (gamma * gamma - 1.0).sqrt()).ln();

    -dist
}

// ─── Top-K Min-Heap ────────────────────────────────────────────────────────────

/// Wrapper for min-heap that keeps the top-k highest-scoring results.
struct MinHeap {
    heap: BinaryHeap<ScoredNode>,
}

/// Node with score, ordered so BinaryHeap acts as a min-heap (lowest score at top).
struct ScoredNode {
    score: f32,
    node_idx: NodeIndex,
}

impl PartialEq for ScoredNode {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for ScoredNode {}

impl PartialOrd for ScoredNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reversed: lower score = higher priority in the heap (min-heap)
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl MinHeap {
    fn with_capacity(cap: usize) -> Self {
        MinHeap {
            heap: BinaryHeap::with_capacity(cap + 1),
        }
    }

    #[inline]
    fn push_if_better(&mut self, node_idx: NodeIndex, score: f32, top_k: usize) {
        if self.heap.len() < top_k {
            self.heap.push(ScoredNode { score, node_idx });
        } else if let Some(min) = self.heap.peek() {
            if score > min.score {
                self.heap.pop();
                self.heap.push(ScoredNode { score, node_idx });
            }
        }
    }

    fn into_sorted_results(self) -> Vec<VectorSearchResult> {
        let mut results: Vec<VectorSearchResult> = self
            .heap
            .into_vec()
            .into_iter()
            .map(|sn| VectorSearchResult {
                node_idx: sn.node_idx,
                score: sn.score,
            })
            .collect();
        // Sort descending by score
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }
}

// ─── Search Implementations ────────────────────────────────────────────────────

fn sequential_search(
    candidates: &[NodeIndex],
    store: &EmbeddingStore,
    query: &[f32],
    top_k: usize,
    scorer: &Scorer,
) -> Vec<VectorSearchResult> {
    let mut heap = MinHeap::with_capacity(top_k);

    for &node_idx in candidates {
        if let Some((embedding, norm)) = store.get_embedding_with_norm(node_idx.index()) {
            let score = scorer.score(query, embedding, norm);
            heap.push_if_better(node_idx, score, top_k);
        }
    }

    heap.into_sorted_results()
}

fn parallel_search(
    candidates: &[NodeIndex],
    store: &EmbeddingStore,
    query: &[f32],
    top_k: usize,
    scorer: &Scorer,
) -> Vec<VectorSearchResult> {
    use rayon::prelude::*;

    let chunk_size = (candidates.len() / rayon::current_num_threads()).max(1024);

    let per_thread_results: Vec<Vec<VectorSearchResult>> = candidates
        .par_chunks(chunk_size)
        .map(|chunk| sequential_search(chunk, store, query, top_k, scorer))
        .collect();

    // Merge per-thread top-k results
    let mut heap = MinHeap::with_capacity(top_k);
    for thread_results in per_thread_results {
        for result in thread_results {
            heap.push_if_better(result.node_idx, result.score, top_k);
        }
    }

    heap.into_sorted_results()
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_large_vector() {
        // Test with >8 elements to exercise chunked path
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..100).map(|i| (i * 2) as f32).collect();
        let sim = cosine_similarity(&a, &b);
        assert!(sim > 0.99); // Nearly parallel vectors
    }

    #[test]
    fn test_dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dp = dot_product(&a, &b);
        assert!((dp - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_neg_euclidean_distance_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let d = neg_euclidean_distance(&a, &b);
        assert!(d.abs() < 1e-6); // Distance 0 → -0.0
    }

    #[test]
    fn test_neg_euclidean_distance_basic() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        let d = neg_euclidean_distance(&a, &b);
        assert!((d + 5.0).abs() < 1e-6); // -sqrt(9+16) = -5.0
    }

    #[test]
    fn test_min_heap_top_k() {
        let mut heap = MinHeap::with_capacity(3);
        let scores = [0.5, 0.9, 0.1, 0.8, 0.3, 0.95, 0.2];

        for (i, &score) in scores.iter().enumerate() {
            heap.push_if_better(NodeIndex::new(i), score, 3);
        }

        let results = heap.into_sorted_results();
        assert_eq!(results.len(), 3);
        assert!((results[0].score - 0.95).abs() < 1e-6);
        assert!((results[1].score - 0.9).abs() < 1e-6);
        assert!((results[2].score - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_store_basic() {
        let mut store = EmbeddingStore::new(3);
        store.set_embedding(0, &[1.0, 2.0, 3.0]);
        store.set_embedding(5, &[4.0, 5.0, 6.0]);

        assert_eq!(store.len(), 2);
        assert_eq!(store.get_embedding(0), Some([1.0, 2.0, 3.0].as_slice()));
        assert_eq!(store.get_embedding(5), Some([4.0, 5.0, 6.0].as_slice()));
        assert_eq!(store.get_embedding(1), None);
    }

    #[test]
    fn test_embedding_store_replace() {
        let mut store = EmbeddingStore::new(2);
        store.set_embedding(0, &[1.0, 2.0]);
        store.set_embedding(0, &[3.0, 4.0]);

        assert_eq!(store.len(), 1);
        assert_eq!(store.get_embedding(0), Some([3.0, 4.0].as_slice()));
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_poincare_identical_vectors() {
        let a = vec![0.3, 0.2, 0.1];
        let score = neg_poincare_distance(&a, &a);
        assert!(
            (score - 0.0).abs() < 1e-5,
            "identical vectors should have distance 0, got {}",
            score
        );
    }

    #[test]
    fn test_poincare_origin_to_point() {
        let origin = vec![0.0, 0.0, 0.0];
        let point = vec![0.5, 0.0, 0.0];
        let score = neg_poincare_distance(&origin, &point);
        // d(0, x) = acosh(1 + 2*||x||² / (1 * (1-||x||²)))
        // = acosh(1 + 2*0.25 / 0.75) = acosh(1 + 0.6667) = acosh(1.6667)
        let expected = -((1.6667f32 + (1.6667f32 * 1.6667f32 - 1.0).sqrt()).ln());
        assert!(
            (score - expected).abs() < 0.01,
            "got {}, expected {}",
            score,
            expected
        );
    }

    #[test]
    fn test_poincare_distance_increases_near_boundary() {
        let origin = vec![0.0, 0.0, 0.0];
        let near = vec![0.1, 0.0, 0.0];
        let mid = vec![0.5, 0.0, 0.0];
        let far = vec![0.9, 0.0, 0.0];

        let score_near = neg_poincare_distance(&origin, &near);
        let score_mid = neg_poincare_distance(&origin, &mid);
        let score_far = neg_poincare_distance(&origin, &far);

        // Closer points have higher (less negative) scores
        assert!(
            score_near > score_mid,
            "near {} should > mid {}",
            score_near,
            score_mid
        );
        assert!(
            score_mid > score_far,
            "mid {} should > far {}",
            score_mid,
            score_far
        );
    }

    #[test]
    fn test_poincare_symmetry() {
        let a = vec![0.3, 0.2, 0.1];
        let b = vec![0.1, 0.4, 0.2];
        let d_ab = neg_poincare_distance(&a, &b);
        let d_ba = neg_poincare_distance(&b, &a);
        assert!(
            (d_ab - d_ba).abs() < 1e-6,
            "should be symmetric: {} vs {}",
            d_ab,
            d_ba
        );
    }

    #[test]
    fn test_poincare_large_vector() {
        // Test with >8 elements to exercise chunked path
        let a = vec![0.1; 16];
        let b = vec![0.2; 16];
        let score = neg_poincare_distance(&a, &b);
        assert!(score < 0.0, "different vectors should have negative score");
        assert!(score.is_finite(), "score should be finite");
    }

    /// The cached-norm cosine path (Scorer) must match the standalone
    /// `cosine_similarity` kernel within floating-point epsilon, across vector
    /// shapes (chunked + remainder), through the real EmbeddingStore norm cache.
    #[test]
    fn test_scorer_cosine_matches_kernel() {
        let cases: Vec<(Vec<f32>, Vec<f32>)> = vec![
            (vec![1.0, 2.0, 3.0, 4.0], vec![4.0, 3.0, 2.0, 1.0]),
            (vec![0.1; 16], vec![0.2; 16]),
            (
                (0..100).map(|i| i as f32).collect(),
                (0..100).map(|i| (i as f32 * 0.37).sin()).collect(),
            ),
            (vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]),
            (vec![1.0, 2.0, 3.0], vec![-1.0, -2.0, -3.0]),
            // Non-multiple-of-8 length to exercise the remainder loop.
            (vec![0.5, 1.5, 2.5, 3.5, 4.5], vec![5.5, 4.5, 3.5, 2.5, 1.5]),
        ];
        for (q, v) in cases {
            let mut store = EmbeddingStore::new(q.len());
            store.set_embedding(0, &v);
            let (emb, norm) = store.get_embedding_with_norm(0).unwrap();

            let scorer = Scorer::new(DistanceMetric::Cosine, &q);
            let got = scorer.score(&q, emb, norm);
            let expected = cosine_similarity(&q, &v);
            assert!(
                (got - expected).abs() < 1e-5,
                "cosine parity failed: scorer={}, kernel={}",
                got,
                expected
            );
        }
    }

    #[test]
    fn test_scorer_cosine_zero_vectors() {
        // Zero query or zero stored vector → 0.0, matching cosine_similarity.
        let mut store = EmbeddingStore::new(3);
        store.set_embedding(0, &[0.0, 0.0, 0.0]);
        let (emb, norm) = store.get_embedding_with_norm(0).unwrap();
        let scorer = Scorer::new(DistanceMetric::Cosine, &[1.0, 2.0, 3.0]);
        assert_eq!(scorer.score(&[1.0, 2.0, 3.0], emb, norm), 0.0);

        store.set_embedding(0, &[1.0, 2.0, 3.0]);
        let (emb, norm) = store.get_embedding_with_norm(0).unwrap();
        let scorer = Scorer::new(DistanceMetric::Cosine, &[0.0, 0.0, 0.0]);
        assert_eq!(scorer.score(&[0.0, 0.0, 0.0], emb, norm), 0.0);
    }

    #[test]
    fn test_scorer_generic_metrics_match_kernels() {
        // Non-cosine metrics route through Generic and ignore the cached norm,
        // so Scorer::score must equal the bare kernel.
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![4.0, 3.0, 2.0, 1.0];
        let mut store = EmbeddingStore::new(4);
        store.set_embedding(0, &v);
        let (emb, norm) = store.get_embedding_with_norm(0).unwrap();

        let dot = Scorer::new(DistanceMetric::DotProduct, &q);
        assert!((dot.score(&q, emb, norm) - dot_product(&q, &v)).abs() < 1e-6);
        let euc = Scorer::new(DistanceMetric::Euclidean, &q);
        assert!((euc.score(&q, emb, norm) - neg_euclidean_distance(&q, &v)).abs() < 1e-6);
        let poi = Scorer::new(DistanceMetric::Poincare, &q);
        assert!((poi.score(&q, emb, norm) - neg_poincare_distance(&q, &v)).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_store_norm_cache() {
        let mut store = EmbeddingStore::new(3);
        store.set_embedding(0, &[3.0, 4.0, 0.0]); // norm 5
        store.set_embedding(7, &[0.0, 0.0, 0.0]); // norm 0
        let (_, n0) = store.get_embedding_with_norm(0).unwrap();
        let (_, n7) = store.get_embedding_with_norm(7).unwrap();
        assert!((n0 - 5.0).abs() < 1e-6);
        assert_eq!(n7, 0.0);

        // Replace must update the cached norm in place.
        store.set_embedding(0, &[5.0, 12.0, 0.0]); // norm 13
        let (_, n0b) = store.get_embedding_with_norm(0).unwrap();
        assert!((n0b - 13.0).abs() < 1e-6);

        // rebuild_norms (the post-load / post-compaction path) reproduces them.
        store.norms.clear();
        store.rebuild_norms();
        let (_, n0c) = store.get_embedding_with_norm(0).unwrap();
        let (_, n7c) = store.get_embedding_with_norm(7).unwrap();
        assert!((n0c - 13.0).abs() < 1e-6);
        assert_eq!(n7c, 0.0);
    }

    #[test]
    fn test_poincare_numerical_stability_near_boundary() {
        // Vectors very close to boundary (norm ~0.999)
        let a = vec![0.999, 0.0, 0.0];
        let b = vec![0.0, 0.999, 0.0];
        let score = neg_poincare_distance(&a, &b);
        assert!(
            score.is_finite(),
            "should not produce infinity near boundary"
        );
        assert!(score < 0.0, "should be negative");
    }
}
