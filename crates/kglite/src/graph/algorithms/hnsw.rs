//! Hand-rolled HNSW (Hierarchical Navigable Small World) index for approximate
//! nearest-neighbour search — Malkov & Yashunin (2016), "Efficient and robust
//! approximate nearest neighbor search using Hierarchical Navigable Small World
//! graphs".
//!
//! This module is deliberately decoupled from the graph: it operates over a flat
//! `&[f32]` vector buffer (the same contiguous layout as
//! [`EmbeddingStore::data`](crate::graph::schema::EmbeddingStore)), the matching
//! per-vector cached L2 norms, a dimension, and a metric. A node is just a *slot*
//! `0..n` into that buffer, so the index stores only topology (per-node level +
//! per-layer neighbour lists + entry point) — never a copy of the vectors. That
//! keeps it cheap to persist and lets it sit alongside an `EmbeddingStore`
//! sharing the very same buffer.
//!
//! Supported metrics are cosine / dot-product / Euclidean (see [`HnswMetric`]);
//! Poincaré is intentionally excluded (its distance is non-linear in the vector
//! norms, so the triangle-inequality-ish navigation HNSW relies on degrades) and
//! stays on the brute-force path.

use super::vector::{dot_product, neg_euclidean_distance, DistanceMetric};
use serde::{Deserialize, Serialize};

/// Metric subset HNSW navigates over. A strict subset of [`DistanceMetric`] —
/// Poincaré has no entry here on purpose.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HnswMetric {
    Cosine,
    Dot,
    Euclidean,
}

impl HnswMetric {
    /// Map a query-time [`DistanceMetric`] onto the HNSW-navigable subset.
    /// Returns `None` for Poincaré (caller falls back to brute force).
    pub fn from_distance(metric: DistanceMetric) -> Option<Self> {
        match metric {
            DistanceMetric::Cosine => Some(HnswMetric::Cosine),
            DistanceMetric::DotProduct => Some(HnswMetric::Dot),
            DistanceMetric::Euclidean => Some(HnswMetric::Euclidean),
            DistanceMetric::Poincare => None,
        }
    }
}

/// Build/search tuning. Defaults follow the common HNSW recommendation
/// (`M=16`, `ef_construction=200`) which gives high recall on typical embedding
/// dimensionalities without an unreasonable graph fan-out.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct HnswParams {
    /// Max neighbours per node on layers > 0. Layer 0 allows `2*m` (`m0`).
    pub m: usize,
    /// Search width while inserting (larger → better graph, slower build).
    pub ef_construction: usize,
    /// Default search width at query time (larger → better recall, slower query).
    pub ef_search: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        HnswParams {
            m: 16,
            ef_construction: 200,
            ef_search: 64,
        }
    }
}

/// A deterministic, seedable PRNG (SplitMix64) used only for HNSW level
/// assignment. Deterministic so a build is reproducible (and so tests are
/// stable); the project bans `Math.random`-style nondeterminism in
/// reproducibility-sensitive paths for the same reason.
struct SplitMix64(u64);

impl SplitMix64 {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A float in `(0, 1)` (strictly positive so `ln` is finite).
    #[inline]
    fn unit(&mut self) -> f64 {
        let v = (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
        if v <= 0.0 {
            f64::MIN_POSITIVE
        } else {
            v
        }
    }
}

/// (slot id, distance-to-target). Ordered by distance so it can drive both a
/// min-heap (via `Reverse`) and a max-heap. Smaller distance = closer.
#[derive(Clone, Copy)]
struct Cand {
    id: u32,
    dist: f32,
}

impl PartialEq for Cand {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cand {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Total order over finite distances; NaN treated as equal (shouldn't occur).
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Distance context: bundles the vector buffer + cached norms + metric so the
/// inner loops don't thread four params each. Distances are "smaller = closer".
struct DistCtx<'a> {
    data: &'a [f32],
    norms: &'a [f32],
    dim: usize,
    metric: HnswMetric,
}

impl<'a> DistCtx<'a> {
    #[inline]
    fn vec(&self, id: u32) -> &[f32] {
        let s = id as usize * self.dim;
        &self.data[s..s + self.dim]
    }

    /// Distance between two stored slots.
    #[inline]
    fn dist_ids(&self, a: u32, b: u32) -> f32 {
        let va = self.vec(a);
        let vb = self.vec(b);
        match self.metric {
            HnswMetric::Cosine => {
                let denom = self.norms[a as usize] * self.norms[b as usize];
                if denom > 0.0 {
                    1.0 - dot_product(va, vb) / denom
                } else {
                    1.0
                }
            }
            HnswMetric::Dot => -dot_product(va, vb),
            // neg_euclidean_distance returns -‖a-b‖; negate back to a true distance.
            HnswMetric::Euclidean => -neg_euclidean_distance(va, vb),
        }
    }

    /// Distance between an external query (norm precomputed) and a stored slot.
    #[inline]
    fn dist_query(&self, query: &[f32], query_norm: f32, b: u32) -> f32 {
        let vb = self.vec(b);
        match self.metric {
            HnswMetric::Cosine => {
                let denom = query_norm * self.norms[b as usize];
                if denom > 0.0 {
                    1.0 - dot_product(query, vb) / denom
                } else {
                    1.0
                }
            }
            HnswMetric::Dot => -dot_product(query, vb),
            HnswMetric::Euclidean => -neg_euclidean_distance(query, vb),
        }
    }
}

/// An HNSW index over `n` slots. Stores topology only; vectors live in the
/// caller's buffer (an `EmbeddingStore`). Serializable so it can ride along in
/// the `.kgl` embeddings section.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HnswIndex {
    params: HnswParams,
    metric: HnswMetric,
    dim: usize,
    /// Number of slots inserted.
    len: usize,
    /// `node_levels[slot]` = top layer this node participates in.
    node_levels: Vec<u8>,
    /// `links[slot][layer]` = neighbour slot ids. Outer indexed by slot, middle
    /// by layer (`0..=node_levels[slot]`), inner the adjacency list.
    links: Vec<Vec<Vec<u32>>>,
    /// Entry point (slot id) into the top layer; `None` only when empty.
    entry_point: Option<u32>,
    max_level: usize,
    /// Seed used for level assignment — kept so incremental inserts after a
    /// reload continue the same deterministic sequence if desired.
    seed: u64,
    /// Insert counter feeding the level PRNG (so reloads are reproducible).
    insert_counter: u64,
}

impl HnswIndex {
    /// Maximum neighbours at a given layer (`2*m` at layer 0, `m` above).
    #[inline]
    fn m_max(&self, layer: usize) -> usize {
        if layer == 0 {
            self.params.m * 2
        } else {
            self.params.m
        }
    }

    /// Number of indexed slots.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> HnswMetric {
        self.metric
    }

    pub fn params(&self) -> HnswParams {
        self.params
    }

    /// Build an index over slots `0..n` of `data` (a flat `n*dim` buffer) with
    /// matching `norms` (length `n`; used by cosine, ignored otherwise).
    /// Insertion order is slot order, which — paired with the seeded level PRNG
    /// — makes the build deterministic.
    pub fn build(
        data: &[f32],
        norms: &[f32],
        dim: usize,
        metric: HnswMetric,
        params: HnswParams,
        seed: u64,
    ) -> Self {
        let n = data.len().checked_div(dim).unwrap_or(0);
        let mut index = HnswIndex {
            params,
            metric,
            dim,
            len: 0,
            node_levels: Vec::with_capacity(n),
            links: Vec::with_capacity(n),
            entry_point: None,
            max_level: 0,
            seed,
            insert_counter: 0,
        };
        let ctx = DistCtx {
            data,
            norms,
            dim,
            metric,
        };
        for slot in 0..n as u32 {
            index.insert_with_ctx(slot, &ctx);
        }
        index
    }

    /// Insert a single slot incrementally. `data`/`norms`/`dim` must describe the
    /// same buffer the index was built over (extended to include `slot`).
    pub fn insert(&mut self, slot: u32, data: &[f32], norms: &[f32], dim: usize) {
        debug_assert_eq!(dim, self.dim, "dimension mismatch on incremental insert");
        let ctx = DistCtx {
            data,
            norms,
            dim,
            metric: self.metric,
        };
        self.insert_with_ctx(slot, &ctx);
    }

    /// Draw a level from the exponential distribution `floor(-ln(U) * mL)`,
    /// `mL = 1/ln(M)` — the standard HNSW assignment.
    fn random_level(&mut self) -> usize {
        let mut rng =
            SplitMix64(self.seed ^ self.insert_counter.wrapping_mul(0x2545_F491_4F6C_DD1D));
        self.insert_counter += 1;
        let m_l = 1.0 / (self.params.m as f64).max(2.0).ln();
        (-rng.unit().ln() * m_l).floor() as usize
    }

    fn insert_with_ctx(&mut self, slot: u32, ctx: &DistCtx) {
        let level = self.random_level();

        // Ensure per-node storage exists up to `slot`.
        let need = slot as usize + 1;
        if self.node_levels.len() < need {
            self.node_levels.resize(need, 0);
            self.links.resize(need, Vec::new());
        }
        self.node_levels[slot as usize] = level as u8;
        self.links[slot as usize] = vec![Vec::new(); level + 1];
        self.len += 1;

        // First node ever → it's the entry point, nothing to link.
        let entry = match self.entry_point {
            Some(e) => e,
            None => {
                self.entry_point = Some(slot);
                self.max_level = level;
                return;
            }
        };

        let df = |id: u32| ctx.dist_ids(slot, id);

        // Phase 1: greedy-descend from the top layer down to `level+1` with ef=1.
        let mut ep = vec![entry];
        let top = self.max_level;
        if top > level {
            for lc in (level + 1..=top).rev() {
                let w = self.search_layer(ctx, &ep, 1, lc, &df);
                if let Some(best) = w.into_iter().min() {
                    ep = vec![best.id];
                }
            }
        }

        // Phase 2: from min(top, level) down to 0, connect.
        let start = top.min(level);
        for lc in (0..=start).rev() {
            let w = self.search_layer(ctx, &ep, self.params.ef_construction, lc, &df);
            let m_max = self.m_max(lc);
            let selected = self.select_neighbors(ctx, slot, &w, self.params.m);

            // Bidirectional links.
            self.links[slot as usize][lc] = selected.clone();
            for &e in &selected {
                self.links[e as usize][lc].push(slot);
                // Prune the neighbour if it now exceeds m_max.
                if self.links[e as usize][lc].len() > m_max {
                    let cands: Vec<Cand> = self.links[e as usize][lc]
                        .iter()
                        .map(|&id| Cand {
                            id,
                            dist: ctx.dist_ids(e, id),
                        })
                        .collect();
                    let pruned = self.select_neighbors(ctx, e, &cands, m_max);
                    self.links[e as usize][lc] = pruned;
                }
            }

            // Carry the full candidate set down as the next layer's entry points.
            ep = w.iter().map(|c| c.id).collect();
            if ep.is_empty() {
                ep = vec![entry];
            }
        }

        // New top layer → this node becomes the entry point.
        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(slot);
        }
    }

    /// HNSW SEARCH-LAYER (algorithm 2). `df(id)` yields the distance from the
    /// target (a node during insert, or an external query during search) to
    /// `id`. Returns up to `ef` nearest candidates on `layer`.
    fn search_layer(
        &self,
        _ctx: &DistCtx,
        entry_points: &[u32],
        ef: usize,
        layer: usize,
        df: &impl Fn(u32) -> f32,
    ) -> Vec<Cand> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut visited = std::collections::HashSet::with_capacity(ef * 4);
        // candidates: min-heap (nearest popped first).
        let mut candidates: BinaryHeap<Reverse<Cand>> = BinaryHeap::new();
        // w: max-heap (farthest popped first), the running result set bounded to ef.
        let mut w: BinaryHeap<Cand> = BinaryHeap::new();

        for &e in entry_points {
            if visited.insert(e) {
                let c = Cand { id: e, dist: df(e) };
                candidates.push(Reverse(c));
                w.push(c);
            }
        }
        while w.len() > ef {
            w.pop();
        }

        while let Some(Reverse(c)) = candidates.pop() {
            let farthest = w.peek().map(|f| f.dist).unwrap_or(f32::INFINITY);
            if c.dist > farthest && w.len() >= ef {
                break;
            }
            // Snapshot neighbours (immutable borrow released before recursion-free loop).
            let neighbours = match self.links.get(c.id as usize).and_then(|l| l.get(layer)) {
                Some(n) => n,
                None => continue,
            };
            for &e in neighbours {
                if visited.insert(e) {
                    let d = df(e);
                    let farthest = w.peek().map(|f| f.dist).unwrap_or(f32::INFINITY);
                    if d < farthest || w.len() < ef {
                        let cand = Cand { id: e, dist: d };
                        candidates.push(Reverse(cand));
                        w.push(cand);
                        if w.len() > ef {
                            w.pop();
                        }
                    }
                }
            }
        }

        w.into_vec()
    }

    /// HNSW neighbour-selection heuristic (algorithm 4). Picks up to `m`
    /// candidates that are each closer to `base` than to any already-picked
    /// neighbour — favouring spread-out links over a tight cluster, which is
    /// what gives HNSW its long-range connectivity. Falls back to filling with
    /// the next-closest leftovers if the heuristic under-fills.
    fn select_neighbors(
        &self,
        ctx: &DistCtx,
        base: u32,
        candidates: &[Cand],
        m: usize,
    ) -> Vec<u32> {
        let mut sorted: Vec<Cand> = candidates
            .iter()
            .copied()
            .filter(|c| c.id != base)
            .collect();
        sorted.sort_unstable();

        let mut result: Vec<u32> = Vec::with_capacity(m);
        let mut deferred: Vec<u32> = Vec::new();
        for c in &sorted {
            if result.len() >= m {
                break;
            }
            let closer_to_base = result.iter().all(|&r| ctx.dist_ids(c.id, r) > c.dist);
            if closer_to_base {
                result.push(c.id);
            } else {
                deferred.push(c.id);
            }
        }
        // Backfill from deferred (closest-first, already sorted) if under-filled.
        for id in deferred {
            if result.len() >= m {
                break;
            }
            result.push(id);
        }
        result
    }

    /// Approximate top-`k` search for an external query vector. `ef` is the
    /// search width (clamped to at least `k`); pass `None` for the configured
    /// default. Returns `(slot, distance)` ascending by distance (closer first);
    /// callers map distance back to a similarity score via the shared `Scorer`.
    pub fn search(
        &self,
        query: &[f32],
        query_norm: f32,
        k: usize,
        ef: Option<usize>,
        data: &[f32],
        norms: &[f32],
    ) -> Vec<(u32, f32)> {
        if self.len == 0 || k == 0 {
            return Vec::new();
        }
        let ctx = DistCtx {
            data,
            norms,
            dim: self.dim,
            metric: self.metric,
        };
        let ef = ef.unwrap_or(self.params.ef_search).max(k);

        let entry = match self.entry_point {
            Some(e) => e,
            None => return Vec::new(),
        };
        let df = |id: u32| ctx.dist_query(query, query_norm, id);

        // Greedy-descend the upper layers with ef=1.
        let mut ep = vec![entry];
        for lc in (1..=self.max_level).rev() {
            let w = self.search_layer(&ctx, &ep, 1, lc, &df);
            if let Some(best) = w.into_iter().min() {
                ep = vec![best.id];
            }
        }

        // Full-width search on layer 0.
        let mut w = self.search_layer(&ctx, &ep, ef, 0, &df);
        w.sort_unstable();
        w.truncate(k);
        w.into_iter().map(|c| (c.id, c.dist)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic gaussian-ish vectors via the same SplitMix64 (no rng dep).
    fn make_data(n: usize, dim: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut rng = SplitMix64(seed);
        let mut data = Vec::with_capacity(n * dim);
        for _ in 0..n * dim {
            // Box-Muller-ish: just map two uniforms to a centered value.
            let u = rng.unit() as f32;
            let v = rng.unit() as f32;
            data.push((u - 0.5) * 2.0 + (v - 0.5));
        }
        let mut norms = Vec::with_capacity(n);
        for i in 0..n {
            let s = i * dim;
            let nn: f32 = data[s..s + dim].iter().map(|x| x * x).sum::<f32>().sqrt();
            norms.push(nn);
        }
        (data, norms)
    }

    fn brute_topk(
        data: &[f32],
        norms: &[f32],
        dim: usize,
        metric: HnswMetric,
        query: &[f32],
        qnorm: f32,
        k: usize,
    ) -> Vec<u32> {
        let n = data.len() / dim;
        let ctx = DistCtx {
            data,
            norms,
            dim,
            metric,
        };
        let mut all: Vec<Cand> = (0..n as u32)
            .map(|id| Cand {
                id,
                dist: ctx.dist_query(query, qnorm, id),
            })
            .collect();
        all.sort_unstable();
        all.truncate(k);
        all.into_iter().map(|c| c.id).collect()
    }

    fn recall_at_k(metric: HnswMetric, n: usize, dim: usize, k: usize) -> f64 {
        let (data, norms) = make_data(n, dim, 0xABCD);
        let index = HnswIndex::build(&data, &norms, dim, metric, HnswParams::default(), 42);
        assert_eq!(index.len(), n);

        // Use stored vectors as queries (their own norm is in `norms`).
        let mut hits = 0usize;
        let mut total = 0usize;
        let n_queries = 50.min(n);
        for q in 0..n_queries {
            let qs = q * dim;
            let query = &data[qs..qs + dim];
            let qnorm = norms[q];
            let truth = brute_topk(&data, &norms, dim, metric, query, qnorm, k);
            let got: Vec<u32> = index
                .search(query, qnorm, k, Some(100), &data, &norms)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let truth_set: std::collections::HashSet<u32> = truth.into_iter().collect();
            for g in got {
                if truth_set.contains(&g) {
                    hits += 1;
                }
            }
            total += k;
        }
        hits as f64 / total as f64
    }

    #[test]
    fn test_recall_cosine() {
        let r = recall_at_k(HnswMetric::Cosine, 2000, 32, 10);
        assert!(r > 0.90, "cosine recall@10 too low: {}", r);
    }

    #[test]
    fn test_recall_euclidean() {
        let r = recall_at_k(HnswMetric::Euclidean, 2000, 32, 10);
        assert!(r > 0.90, "euclidean recall@10 too low: {}", r);
    }

    #[test]
    fn test_recall_dot() {
        let r = recall_at_k(HnswMetric::Dot, 2000, 32, 10);
        // Dot-product is not a true metric; recall is typically a touch lower.
        assert!(r > 0.85, "dot recall@10 too low: {}", r);
    }

    #[test]
    fn test_empty_and_single() {
        let index = HnswIndex::build(&[], &[], 4, HnswMetric::Cosine, HnswParams::default(), 1);
        assert!(index.is_empty());
        assert!(index
            .search(&[1.0, 0.0, 0.0, 0.0], 1.0, 5, None, &[], &[])
            .is_empty());

        let data = vec![1.0, 0.0, 0.0, 0.0];
        let norms = vec![1.0];
        let index = HnswIndex::build(
            &data,
            &norms,
            4,
            HnswMetric::Cosine,
            HnswParams::default(),
            1,
        );
        assert_eq!(index.len(), 1);
        let res = index.search(&[1.0, 0.0, 0.0, 0.0], 1.0, 5, None, &data, &norms);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, 0);
    }

    #[test]
    fn test_k_larger_than_n() {
        let (data, norms) = make_data(5, 8, 7);
        let index = HnswIndex::build(
            &data,
            &norms,
            8,
            HnswMetric::Cosine,
            HnswParams::default(),
            3,
        );
        let qs = &data[0..8];
        let res = index.search(qs, norms[0], 100, None, &data, &norms);
        assert_eq!(res.len(), 5, "k>n should return all n");
    }

    #[test]
    fn test_incremental_matches_build_recall() {
        // Insert one slot at a time; recall should stay high (same algorithm).
        let (data, norms) = make_data(1500, 24, 0x1234);
        let mut index = HnswIndex {
            params: HnswParams::default(),
            metric: HnswMetric::Cosine,
            dim: 24,
            len: 0,
            node_levels: Vec::new(),
            links: Vec::new(),
            entry_point: None,
            max_level: 0,
            seed: 99,
            insert_counter: 0,
        };
        for slot in 0..1500u32 {
            index.insert(slot, &data, &norms, 24);
        }
        assert_eq!(index.len(), 1500);

        let mut hits = 0;
        for q in 0..40 {
            let qs = q * 24;
            let query = &data[qs..qs + 24];
            let truth = brute_topk(&data, &norms, 24, HnswMetric::Cosine, query, norms[q], 10);
            let got: std::collections::HashSet<u32> = index
                .search(query, norms[q], 10, Some(100), &data, &norms)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            for t in truth {
                if got.contains(&t) {
                    hits += 1;
                }
            }
        }
        let recall = hits as f64 / (40 * 10) as f64;
        assert!(recall > 0.90, "incremental recall too low: {}", recall);
    }

    #[test]
    fn test_deterministic_build() {
        let (data, norms) = make_data(300, 16, 55);
        let a = HnswIndex::build(
            &data,
            &norms,
            16,
            HnswMetric::Cosine,
            HnswParams::default(),
            7,
        );
        let b = HnswIndex::build(
            &data,
            &norms,
            16,
            HnswMetric::Cosine,
            HnswParams::default(),
            7,
        );
        assert_eq!(a.node_levels, b.node_levels);
        assert_eq!(a.links, b.links);
        assert_eq!(a.entry_point, b.entry_point);
    }

    #[test]
    fn test_metric_subset_mapping() {
        assert_eq!(
            HnswMetric::from_distance(DistanceMetric::Cosine),
            Some(HnswMetric::Cosine)
        );
        assert_eq!(
            HnswMetric::from_distance(DistanceMetric::DotProduct),
            Some(HnswMetric::Dot)
        );
        assert_eq!(
            HnswMetric::from_distance(DistanceMetric::Euclidean),
            Some(HnswMetric::Euclidean)
        );
        assert_eq!(HnswMetric::from_distance(DistanceMetric::Poincare), None);
    }
}
