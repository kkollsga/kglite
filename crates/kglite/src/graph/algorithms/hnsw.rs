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
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::RwLock;

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

impl HnswParams {
    pub(crate) fn validate(&self) -> Result<(), &'static str> {
        if self.m < 2 || self.ef_construction == 0 || self.ef_search == 0 {
            return Err("HNSW tuning parameters are outside their valid range");
        }
        self.m
            .checked_mul(2)
            .map(|_| ())
            .ok_or("HNSW layer-zero degree bound overflows usize")
    }
}

/// A deterministic, seedable PRNG (SplitMix64) used only for HNSW level
/// assignment. The seeded levels are reproducible for a given seed (the
/// concurrent build's link graph is not — see `build`), which keeps the
/// per-slot layer structure stable and tests on it deterministic.
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

    /// Validate deserialized topology against its owning embedding store.
    ///
    /// Persistence treats an HNSW index as a rebuildable cache, so callers
    /// must reject an error here and retain the store without the index.
    pub(crate) fn validate_for_store(
        &self,
        data: &[f32],
        norms: &[f32],
        dimension: usize,
    ) -> Result<(), &'static str> {
        self.validate_store_header(data, norms, dimension)?;
        let Some(entry_point) = self.validate_canonical_state()? else {
            return Ok(());
        };
        self.validate_link_topology(entry_point)
    }

    fn validate_store_header(
        &self,
        data: &[f32],
        norms: &[f32],
        dimension: usize,
    ) -> Result<(), &'static str> {
        if dimension == 0 || self.dim != dimension {
            return Err("HNSW dimension does not match its embedding store");
        }
        self.params.validate()?;
        if self.len > u32::MAX as usize {
            return Err("HNSW topology has more slots than its u32 identifiers can address");
        }
        let covered_data_len = self
            .len
            .checked_mul(self.dim)
            .ok_or("HNSW vector cardinality overflows usize")?;
        // A **prefix**, not an equality: an index covers slots `0..len` of a
        // store that may have grown past it since the build (the catch-up
        // delta). What is never sound is topology addressing slots the store
        // does not hold — that is a vector read out of bounds.
        if data.len() < covered_data_len {
            return Err("HNSW topology covers more vectors than its embedding store holds");
        }
        if norms.len() < self.len {
            return Err("HNSW topology covers more norms than its embedding store holds");
        }
        if self.node_levels.len() != self.len {
            return Err("HNSW node-level cardinality does not match its length");
        }
        if self.links.len() != self.len {
            return Err("HNSW link cardinality does not match its length");
        }
        Ok(())
    }

    fn validate_canonical_state(&self) -> Result<Option<usize>, &'static str> {
        if self.len == 0 {
            return if self.entry_point.is_none() && self.max_level == 0 && self.insert_counter == 0
            {
                Ok(None)
            } else {
                Err("empty HNSW topology has non-canonical state")
            };
        }
        if self.insert_counter != self.len as u64 {
            return Err("HNSW insert counter does not match its length");
        }

        let entry_point =
            self.entry_point
                .ok_or("non-empty HNSW topology has no entry point")? as usize;
        if entry_point >= self.len {
            return Err("HNSW entry point is outside the topology");
        }
        Ok(Some(entry_point))
    }

    fn validate_link_topology(&self, entry_point: usize) -> Result<(), &'static str> {
        let layer_zero_degree = self.params.m * 2;
        let mut observed_max_level = 0usize;
        let mut unique_neighbors = HashSet::new();
        for (slot, (&node_level, layers)) in self.node_levels.iter().zip(&self.links).enumerate() {
            let node_level = node_level as usize;
            observed_max_level = observed_max_level.max(node_level);
            if layers.len() != node_level + 1 {
                return Err("HNSW node layer count does not match its declared level");
            }
            for (layer, neighbors) in layers.iter().enumerate() {
                unique_neighbors.clear();
                let degree_bound = if layer == 0 {
                    layer_zero_degree
                } else {
                    self.params.m
                };
                if neighbors.len() > degree_bound {
                    return Err("HNSW layer exceeds its degree bound");
                }
                for &neighbor in neighbors {
                    if !unique_neighbors.insert(neighbor) {
                        return Err("HNSW layer contains a duplicate neighbor");
                    }
                    let neighbor = neighbor as usize;
                    if neighbor >= self.len {
                        return Err("HNSW neighbor is outside the topology");
                    }
                    if (self.node_levels[neighbor] as usize) < layer {
                        return Err("HNSW neighbor does not participate in its linked layer");
                    }
                    if neighbor == slot {
                        return Err("HNSW node links to itself");
                    }
                }
            }
        }
        if self.max_level != observed_max_level {
            return Err("HNSW maximum level does not match its topology");
        }
        if self.node_levels[entry_point] as usize != self.max_level {
            return Err("HNSW entry point does not participate in the maximum layer");
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn corrupt_entry_point_for_test(&mut self) {
        self.entry_point = Some(self.len as u32);
    }

    fn validate_incremental_state(&self) -> Result<(), &'static str> {
        self.params.validate()?;
        if self.len > u32::MAX as usize
            || self.node_levels.len() != self.len
            || self.links.len() != self.len
            || self.insert_counter != self.len as u64
        {
            return Err("HNSW incremental topology cardinalities are inconsistent");
        }
        match self.entry_point {
            None if self.len == 0 && self.max_level == 0 => Ok(()),
            Some(entry) if self.len > 0 && (entry as usize) < self.len => {
                if self.node_levels[entry as usize] as usize == self.max_level {
                    Ok(())
                } else {
                    Err("HNSW entry point does not participate in the maximum layer")
                }
            }
            _ => Err("HNSW incremental entry-point state is inconsistent"),
        }
    }

    /// Build an index over slots `0..n` of `data` (a flat `n*dim` buffer) with
    /// matching `norms` (length `n`; used by cosine, ignored otherwise).
    ///
    /// Inserts run **concurrently** (rayon): the per-slot level assignment is
    /// deterministic (seeded), but the vectors are immutable during the build —
    /// only the link graph mutates — so each insert reads the growing graph
    /// through per-node read locks and writes only its own + its neighbours'
    /// link lists, never holding two link locks at once (deadlock-free). The
    /// resulting graph differs run-to-run (concurrency), but recall is
    /// statistically equivalent to a sequential build; the index is a
    /// rebuildable cache, so bit-for-bit reproducibility isn't a contract.
    ///
    /// "Statistically equivalent" is load-bearing and does not come for free:
    /// slots are claimed in order from a shared counter, and a slot merges its
    /// chosen neighbours into its own list instead of assigning them. Both are
    /// there so the link graph stays *navigable*, not merely connected — see
    /// the comments at each site, and
    /// `concurrent_build_leaves_no_one_way_island`.
    pub fn build(
        data: &[f32],
        norms: &[f32],
        dim: usize,
        metric: HnswMetric,
        params: HnswParams,
        seed: u64,
    ) -> Self {
        let n = data.len().checked_div(dim).unwrap_or(0);
        if n == 0 {
            return HnswIndex {
                params,
                metric,
                dim,
                len: 0,
                node_levels: Vec::new(),
                links: Vec::new(),
                entry_point: None,
                max_level: 0,
                seed,
                insert_counter: 0,
            };
        }

        // Deterministic level per slot — the same sequence the sequential
        // `insert` path would assign (insert_counter == slot).
        let node_levels: Vec<u8> = (0..n as u64)
            .map(|i| level_for(seed, i, params.m) as u8)
            .collect();
        // Per-node link store, behind a lock each (only the graph mutates).
        let links: Vec<RwLock<Vec<Vec<u32>>>> = node_levels
            .iter()
            .map(|&lvl| RwLock::new(vec![Vec::new(); lvl as usize + 1]))
            .collect();
        // Slot 0 seeds the entry point; taller nodes take over as they land.
        let ep_state = RwLock::new((0u32, node_levels[0] as usize));
        let ctx = DistCtx {
            data,
            norms,
            dim,
            metric,
        };

        // Claim slots from one shared counter rather than letting rayon split
        // `1..n` into contiguous per-thread blocks. The block split starts every
        // worker at a *different* slot — and a slot is a position in the vector
        // buffer, so the workers seed several mutually invisible regions of the
        // graph at once and the losers of the entry-point race stay one-way
        // islands (measured on a 600-point ramp: slots 0..4 reachable only
        // through slot 75, the second block's first slot). Dynamic claiming avoids those fixed
        // distant starts. It bounds the number of concurrent inserts, not the
        // span of slot IDs: a slow insert can lag behind faster workers.
        let next_slot = AtomicU32::new(1);
        let workers = rayon::current_num_threads().max(1);
        (0..workers).into_par_iter().for_each(|_| loop {
            let slot = next_slot.fetch_add(1, Ordering::Relaxed);
            if slot >= n as u32 {
                break;
            }
            insert_concurrent(slot, &ctx, &params, &node_levels, &links, &ep_state);
        });

        let links: Vec<Vec<Vec<u32>>> = links
            .into_iter()
            .map(|l| l.into_inner().unwrap_or_default())
            .collect();
        let (entry_point, max_level) = *ep_state.read().unwrap();
        HnswIndex {
            params,
            metric,
            dim,
            len: n,
            node_levels,
            links,
            entry_point: Some(entry_point),
            max_level,
            seed,
            insert_counter: n as u64,
        }
    }

    /// Insert a single slot incrementally. `data`/`norms`/`dim` must describe the
    /// same buffer the index was built over (extended to include `slot`).
    pub fn insert(
        &mut self,
        slot: u32,
        data: &[f32],
        norms: &[f32],
        dim: usize,
    ) -> Result<(), String> {
        self.validate_incremental_state().map_err(str::to_string)?;
        if dim != self.dim {
            return Err("dimension mismatch on incremental insert".to_string());
        }
        if dim == 0 {
            return Err("incremental insert requires a non-zero dimension".to_string());
        }
        if slot as usize != self.len {
            return Err("incremental insert slot must be the next contiguous slot".to_string());
        }
        if !data.len().is_multiple_of(dim) {
            return Err("incremental insert vector buffer has a partial vector".to_string());
        }
        let vector_count = data.len() / dim;
        if norms.len() != vector_count {
            return Err("incremental insert vector and norm cardinalities differ".to_string());
        }
        if (slot as usize) >= vector_count {
            return Err("incremental insert buffers do not contain the new slot".to_string());
        }
        let ctx = DistCtx {
            data,
            norms,
            dim,
            metric: self.metric,
        };
        self.insert_with_ctx(slot, &ctx);
        Ok(())
    }

    /// Draw the next level for the sequential `insert` path (advances the
    /// per-index insert counter). Delegates to the shared [`level_for`] so the
    /// sequential and concurrent builds assign identical levels for a seed.
    fn random_level(&mut self) -> usize {
        let lvl = level_for(self.seed, self.insert_counter, self.params.m);
        self.insert_counter += 1;
        lvl
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

        self.link_slot(slot, level, entry, ctx);

        // New top layer → this node becomes the entry point.
        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(slot);
        }
    }

    /// Recompute one slot's outgoing links for the vector currently at that
    /// slot — the in-place-update half of catch-up.
    ///
    /// HNSW has no update primitive: a vector replaced in place leaves
    /// topology that was built for the *old* vector, so the node stays
    /// findable but through the wrong neighbourhood. Scores never go wrong
    /// (this index stores no vectors — every distance is read from the
    /// caller's live buffer), so the damage is bounded to recall, which is why
    /// re-linking is a legitimate repair where a rebuild is not affordable.
    ///
    /// **The node's level and the index's length do not move**, so back-links
    /// pointing at it from other nodes' lists stay valid — and they are
    /// deliberately *not* swept out, which is what keeps this O(one insert)
    /// instead of O(corpus): a stale back-link is an extra edge into a node
    /// that is still there, never a dangling reference. Its old outgoing links
    /// are likewise left in place until each layer overwrites them, so the
    /// search that computes the replacement still has a navigable graph.
    pub fn relink(
        &mut self,
        slot: u32,
        data: &[f32],
        norms: &[f32],
        dim: usize,
    ) -> Result<(), String> {
        self.validate_incremental_state().map_err(str::to_string)?;
        if dim != self.dim {
            return Err("dimension mismatch on relink".to_string());
        }
        if dim == 0 {
            return Err("relink requires a non-zero dimension".to_string());
        }
        if (slot as usize) >= self.len {
            return Err("relink slot is outside the topology".to_string());
        }
        if !data.len().is_multiple_of(dim) {
            return Err("relink vector buffer has a partial vector".to_string());
        }
        let vector_count = data.len() / dim;
        if norms.len() != vector_count {
            return Err("relink vector and norm cardinalities differ".to_string());
        }
        if self.len > vector_count {
            return Err("relink buffers do not cover the topology".to_string());
        }
        let ctx = DistCtx {
            data,
            norms,
            dim,
            metric: self.metric,
        };
        let level = self.node_levels[slot as usize] as usize;
        let entry = self
            .entry_point
            .ok_or("relink on a topology with no entry point")?;
        self.link_slot(slot, level, entry, &ctx);
        Ok(())
    }

    /// Connect `slot` — whose level and per-layer link storage are already
    /// sized — into the graph, descending from `entry`. Shared by the
    /// sequential insert and by [`Self::relink`], so the two cannot drift on
    /// neighbour selection or degree pruning.
    fn link_slot(&mut self, slot: u32, level: usize, entry: u32, ctx: &DistCtx) {
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
            let selected = select_neighbors(ctx, slot, &w, self.params.m);

            // Bidirectional links.
            self.links[slot as usize][lc] = selected.clone();
            for &e in &selected {
                // Guarded because `relink` re-runs this over a slot some
                // neighbours already point at; a duplicate neighbour is a
                // topology-validation failure, not a harmless repeat.
                if !self.links[e as usize][lc].contains(&slot) {
                    self.links[e as usize][lc].push(slot);
                }
                // Prune the neighbour if it now exceeds m_max.
                if self.links[e as usize][lc].len() > m_max {
                    let cands: Vec<Cand> = self.links[e as usize][lc]
                        .iter()
                        .map(|&id| Cand {
                            id,
                            dist: ctx.dist_ids(e, id),
                        })
                        .collect();
                    let pruned = select_neighbors(ctx, e, &cands, m_max);
                    self.links[e as usize][lc] = pruned;
                }
            }

            // Carry the full candidate set down as the next layer's entry points.
            ep = w.iter().map(|c| c.id).collect();
            if ep.is_empty() {
                ep = vec![entry];
            }
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

        let mut visited = FxHashSet::with_capacity_and_hasher(ef * 4, Default::default());
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

// ─── Shared / concurrent-build free functions ───────────────────────────────

/// Level for a slot from the exponential distribution `floor(-ln(U) * mL)`,
/// `mL = 1/ln(M)`. Seeded by `(seed, counter)` so the sequential `insert` path
/// (counter == insert order) and the concurrent `build` (counter == slot)
/// assign identical levels for a given seed.
fn level_for(seed: u64, counter: u64, m: usize) -> usize {
    let mut rng = SplitMix64(seed ^ counter.wrapping_mul(0x2545_F491_4F6C_DD1D));
    let m_l = 1.0 / (m as f64).max(2.0).ln();
    (-rng.unit().ln() * m_l).floor() as usize
}

/// HNSW neighbour-selection heuristic (algorithm 4). Picks up to `m` candidates
/// each closer to `base` than to any already-picked neighbour — favouring
/// spread-out links over a tight cluster (HNSW's long-range connectivity).
/// Backfills with the next-closest leftovers if the heuristic under-fills.
/// Pure over `ctx` (distance only) — no link access — so it is shared by the
/// sequential and concurrent build paths.
fn select_neighbors(ctx: &DistCtx, base: u32, candidates: &[Cand], m: usize) -> Vec<u32> {
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
    for id in deferred {
        if result.len() >= m {
            break;
        }
        result.push(id);
    }
    result
}

/// SEARCH-LAYER over a concurrent (lock-guarded) link store — the build-time
/// twin of `HnswIndex::search_layer`. Reads each visited node's neighbour list
/// under a brief read lock (cloned, then released), so it never holds a lock
/// while computing distances. Used only during the one-time concurrent build,
/// where the per-node clone is negligible; the query path keeps the
/// borrow-only method (no clone).
fn search_layer_locked(
    links: &[RwLock<Vec<Vec<u32>>>],
    entry_points: &[u32],
    ef: usize,
    layer: usize,
    df: &impl Fn(u32) -> f32,
) -> Vec<Cand> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let mut visited = FxHashSet::with_capacity_and_hasher(ef * 4, Default::default());
    let mut candidates: BinaryHeap<Reverse<Cand>> = BinaryHeap::new();
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
        // Clone this node's layer neighbours under a brief read lock.
        let neighbours: Vec<u32> = match links.get(c.id as usize) {
            Some(lock) => lock.read().unwrap().get(layer).cloned().unwrap_or_default(),
            None => continue,
        };
        for e in neighbours {
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

/// Merge links received during search before enforcing the degree bound.
/// Replacing the list would lose reciprocal edges from concurrent insertions.
fn merge_selected_links(
    links: &mut Vec<u32>,
    selected: &[u32],
    ctx: &DistCtx,
    slot: u32,
    m_max: usize,
) {
    for &id in selected {
        if !links.contains(&id) {
            links.push(id);
        }
    }
    if links.len() > m_max {
        let candidates: Vec<Cand> = links
            .iter()
            .map(|&id| Cand {
                id,
                dist: ctx.dist_ids(slot, id),
            })
            .collect();
        *links = select_neighbors(ctx, slot, &candidates, m_max);
    }
}

/// Insert one slot into the concurrent build. Mirrors `insert_with_ctx` but
/// over the lock-guarded link store, taking a snapshot of the entry point /
/// max level. Lock discipline: at most one link write lock is held at a time
/// (own node, then each neighbour in turn), and distance computation reads only
/// the immutable vector data — so there is no lock nesting and no deadlock.
fn insert_concurrent(
    slot: u32,
    ctx: &DistCtx,
    params: &HnswParams,
    node_levels: &[u8],
    links: &[RwLock<Vec<Vec<u32>>>],
    ep_state: &RwLock<(u32, usize)>,
) {
    let level = node_levels[slot as usize] as usize;
    let df = |id: u32| ctx.dist_ids(slot, id);
    let (entry, top) = *ep_state.read().unwrap();

    // Phase 1: greedy-descend the layers above `level` with ef=1.
    let mut ep = vec![entry];
    if top > level {
        for lc in (level + 1..=top).rev() {
            let w = search_layer_locked(links, &ep, 1, lc, &df);
            if let Some(best) = w.into_iter().min() {
                ep = vec![best.id];
            }
        }
    }

    // Phase 2: connect from min(top, level) down to 0.
    let start = top.min(level);
    for lc in (0..=start).rev() {
        let w = search_layer_locked(links, &ep, params.ef_construction, lc, &df);
        let m_max = if lc == 0 { params.m * 2 } else { params.m };
        let selected = select_neighbors(ctx, slot, &w, params.m);

        // Other inserts may already have supplied reciprocal links while this
        // slot was searching. Preserve them when publishing its own selection.
        {
            let mut g = links[slot as usize].write().unwrap();
            if lc < g.len() {
                merge_selected_links(&mut g[lc], &selected, ctx, slot, m_max);
            }
        }
        // Bidirectional links + prune, one neighbour lock at a time.
        for &e in &selected {
            let mut eg = links[e as usize].write().unwrap();
            if lc >= eg.len() {
                continue; // defensive: e doesn't participate in this layer
            }
            // Another insertion can already have added this reciprocal edge:
            // unlike the sequential builder, all concurrently built slots are
            // visible from the start. Keep each adjacency list set-like.
            if !eg[lc].contains(&slot) {
                eg[lc].push(slot);
            }
            if eg[lc].len() > m_max {
                let cands: Vec<Cand> = eg[lc]
                    .iter()
                    .map(|&id| Cand {
                        id,
                        dist: ctx.dist_ids(e, id),
                    })
                    .collect();
                eg[lc] = select_neighbors(ctx, e, &cands, m_max);
            }
        }

        ep = w.iter().map(|c| c.id).collect();
        if ep.is_empty() {
            ep = vec![entry];
        }
    }

    // Took a new top layer → become the entry point.
    if level > top {
        let mut g = ep_state.write().unwrap();
        if level > g.1 {
            *g = (slot, level);
        }
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

    fn valid_index() -> (HnswIndex, Vec<f32>, Vec<f32>) {
        let (data, norms) = make_data(40, 8, 0x51A7);
        let index = HnswIndex::build(
            &data,
            &norms,
            8,
            HnswMetric::Cosine,
            HnswParams::default(),
            19,
        );
        (index, data, norms)
    }

    fn empty_index() -> HnswIndex {
        HnswIndex::build(&[], &[], 8, HnswMetric::Cosine, HnswParams::default(), 1)
    }

    #[test]
    fn persisted_validation_accepts_valid_topology() {
        let (index, data, norms) = valid_index();
        assert_eq!(index.validate_for_store(&data, &norms, 8), Ok(()));
    }

    #[test]
    fn persisted_validation_accepts_repeated_concurrent_topology() {
        let (data, norms) = make_data(500, 16, 11);
        for attempt in 0..10 {
            let index = HnswIndex::build(
                &data,
                &norms,
                16,
                HnswMetric::Cosine,
                HnswParams::default(),
                7,
            );
            assert_eq!(
                index.validate_for_store(&data, &norms, 16),
                Ok(()),
                "attempt {attempt}"
            );
        }
    }

    /// An index may cover a *prefix* of its store — that is the whole
    /// catch-up delta — so extra vectors past its length are not a mismatch.
    #[test]
    fn persisted_validation_accepts_a_store_that_has_grown_past_the_index() {
        let (index, data, norms) = valid_index();
        let (extra_data, extra_norms) = make_data(5, 8, 0xBEEF);
        let mut data = data;
        let mut norms = norms;
        data.extend_from_slice(&extra_data);
        norms.extend_from_slice(&extra_norms);
        assert_eq!(index.validate_for_store(&data, &norms, 8), Ok(()));
    }

    #[test]
    fn incremental_insert_extends_the_topology_and_keeps_it_valid() {
        let (mut index, data, norms) = valid_index();
        let (extra_data, extra_norms) = make_data(3, 8, 0x5EED);
        let mut data = data;
        let mut norms = norms;
        data.extend_from_slice(&extra_data);
        norms.extend_from_slice(&extra_norms);

        for slot in 40..43u32 {
            index.insert(slot, &data, &norms, 8).expect("append");
        }
        assert_eq!(index.len(), 43);
        assert_eq!(index.validate_for_store(&data, &norms, 8), Ok(()));

        // The appended vectors are findable, which is what catch-up buys.
        let query = &data[42 * 8..43 * 8];
        let hits = index.search(query, norms[42], 1, None, &data, &norms);
        assert_eq!(hits[0].0, 42);
    }

    /// Re-linking a slot whose vector was replaced in place must leave a
    /// topology that still satisfies every invariant — no duplicate
    /// neighbours, no degree-bound violation, no orphaned level — and must find
    /// the *new* vector where the stale topology would have hidden it.
    #[test]
    fn relink_repairs_an_in_place_replacement_without_breaking_the_topology() {
        let (mut index, mut data, mut norms) = valid_index();
        let (replacement, replacement_norms) = make_data(1, 8, 0xC0FFEE);
        data[7 * 8..8 * 8].copy_from_slice(&replacement);
        norms[7] = replacement_norms[0];

        index.relink(7, &data, &norms, 8).expect("relink");
        assert_eq!(index.len(), 40, "re-linking adds no slot");
        assert_eq!(index.validate_for_store(&data, &norms, 8), Ok(()));

        let hits = index.search(&replacement, norms[7], 1, None, &data, &norms);
        assert_eq!(hits[0].0, 7, "the replaced vector is found at its own slot");
    }

    /// Re-linking is idempotent: running it twice must not double a back-link
    /// (a duplicate neighbour is a validation failure, not a harmless repeat).
    #[test]
    fn repeated_relink_never_duplicates_a_neighbour() {
        let (mut index, data, norms) = valid_index();
        for _ in 0..3 {
            index.relink(11, &data, &norms, 8).expect("relink");
        }
        assert_eq!(index.validate_for_store(&data, &norms, 8), Ok(()));
    }

    #[test]
    fn relink_rejects_arguments_it_cannot_honour() {
        let (mut index, data, norms) = valid_index();
        assert!(index.relink(40, &data, &norms, 8).is_err(), "outside");
        assert!(index.relink(0, &data, &norms, 4).is_err(), "dimension");
        assert!(
            index.relink(0, &data[..8], &norms[..1], 8).is_err(),
            "buffers do not cover the topology"
        );
    }

    #[test]
    fn persisted_validation_rejects_len_dimension_overflow() {
        let (mut index, data, norms) = valid_index();
        index.len = usize::MAX;
        assert!(index.validate_for_store(&data, &norms, 8).is_err());
    }

    #[test]
    fn persisted_validation_rejects_data_cardinality() {
        let (index, mut data, norms) = valid_index();
        data.pop();
        assert!(index.validate_for_store(&data, &norms, 8).is_err());
    }

    #[test]
    fn persisted_validation_rejects_norm_cardinality() {
        let (index, data, mut norms) = valid_index();
        norms.pop();
        assert!(index.validate_for_store(&data, &norms, 8).is_err());
    }

    #[test]
    fn persisted_validation_rejects_node_level_length() {
        let (mut index, data, norms) = valid_index();
        index.node_levels.pop();
        assert!(index.validate_for_store(&data, &norms, 8).is_err());
    }

    #[test]
    fn persisted_validation_rejects_link_length() {
        let (mut index, data, norms) = valid_index();
        index.links.pop();
        assert!(index.validate_for_store(&data, &norms, 8).is_err());
    }

    #[test]
    fn persisted_validation_rejects_entry_point() {
        let (mut index, data, norms) = valid_index();
        index.entry_point = Some(index.len as u32);
        assert!(index.validate_for_store(&data, &norms, 8).is_err());
    }

    #[test]
    fn persisted_validation_rejects_neighbor_id() {
        let (mut index, data, norms) = valid_index();
        index.links[0][0] = vec![index.len as u32];
        assert!(index.validate_for_store(&data, &norms, 8).is_err());
    }

    #[test]
    fn persisted_validation_rejects_layer_count() {
        let (mut index, data, norms) = valid_index();
        index.links[0].push(Vec::new());
        assert!(index.validate_for_store(&data, &norms, 8).is_err());
    }

    #[test]
    fn persisted_validation_rejects_max_level() {
        let (mut index, data, norms) = valid_index();
        index.max_level += 1;
        assert!(index.validate_for_store(&data, &norms, 8).is_err());
    }

    #[test]
    fn persisted_validation_rejects_layer_degree() {
        let (mut index, data, norms) = valid_index();
        index.links[0][0] = vec![1; index.m_max(0) + 1];
        assert!(index.validate_for_store(&data, &norms, 8).is_err());
    }

    #[test]
    fn persisted_validation_rejects_m_below_minimum() {
        let (mut index, data, norms) = valid_index();
        index.params.m = 1;
        assert!(index.validate_for_store(&data, &norms, 8).is_err());
    }

    #[test]
    fn persisted_validation_rejects_zero_construction_width() {
        let (mut index, data, norms) = valid_index();
        index.params.ef_construction = 0;
        assert!(index.validate_for_store(&data, &norms, 8).is_err());
    }

    #[test]
    fn persisted_validation_rejects_zero_search_width() {
        let (mut index, data, norms) = valid_index();
        index.params.ef_search = 0;
        assert!(index.validate_for_store(&data, &norms, 8).is_err());
    }

    #[test]
    fn persisted_validation_rejects_layer_zero_degree_overflow() {
        let (mut index, data, norms) = valid_index();
        index.params.m = usize::MAX;
        assert!(index.validate_for_store(&data, &norms, 8).is_err());
    }

    #[test]
    fn persisted_validation_rejects_noncanonical_empty_state() {
        let mut index = empty_index();
        index.insert_counter = 1;
        assert!(index.validate_for_store(&[], &[], 8).is_err());
    }

    #[test]
    fn persisted_validation_rejects_empty_layer_zero_degree_overflow() {
        let mut index = empty_index();
        index.params.m = usize::MAX;
        assert!(index.validate_for_store(&[], &[], 8).is_err());
    }

    #[test]
    fn persisted_validation_rejects_insert_counter() {
        let (mut index, data, norms) = valid_index();
        index.insert_counter -= 1;
        assert!(index.validate_for_store(&data, &norms, 8).is_err());
    }

    #[test]
    fn persisted_validation_rejects_duplicate_neighbor() {
        let (mut index, data, norms) = valid_index();
        index.links[0][0] = vec![1, 1];
        assert!(index.validate_for_store(&data, &norms, 8).is_err());
    }

    #[test]
    fn persisted_validation_rejects_self_link() {
        let (mut index, data, norms) = valid_index();
        index.links[0][0] = vec![0];
        assert!(index.validate_for_store(&data, &norms, 8).is_err());
    }

    #[test]
    fn persisted_validation_rejects_neighbor_missing_layer() {
        let (mut index, data, norms) = valid_index();
        let upper_slot = index
            .node_levels
            .iter()
            .position(|&level| level > 0)
            .expect("fixture must contain an upper-layer node");
        let layer = index.node_levels[upper_slot] as usize;
        let lower_slot = index
            .node_levels
            .iter()
            .position(|&level| (level as usize) < layer)
            .expect("fixture must contain a lower-layer node");
        index.links[upper_slot][layer] = vec![lower_slot as u32];
        assert!(index.validate_for_store(&data, &norms, 8).is_err());
    }

    #[test]
    fn incremental_insert_checks_dimension_in_release_builds() {
        let mut index = empty_index();
        let before = format!("{index:?}");
        assert!(index.insert(0, &[0.0; 4], &[0.0], 4).is_err());
        assert_eq!(format!("{index:?}"), before);
    }

    #[test]
    fn incremental_insert_checks_complete_vector_cardinality_in_release_builds() {
        let mut index = empty_index();
        let before = format!("{index:?}");
        assert!(index.insert(0, &[0.0; 7], &[0.0], 8).is_err());
        assert_eq!(format!("{index:?}"), before);
    }

    #[test]
    fn incremental_insert_checks_norm_cardinality_in_release_builds() {
        let mut index = empty_index();
        let before = format!("{index:?}");
        assert!(index.insert(0, &[0.0; 8], &[], 8).is_err());
        assert_eq!(format!("{index:?}"), before);
    }

    #[test]
    fn incremental_insert_checks_slot_cardinality_in_release_builds() {
        let mut index = empty_index();
        let before = format!("{index:?}");
        assert!(index.insert(1, &[0.0; 16], &[0.0; 2], 8).is_err());
        assert_eq!(format!("{index:?}"), before);
    }

    #[test]
    fn incremental_insert_checks_current_parameters_before_mutation() {
        let (mut index, mut data, mut norms) = valid_index();
        index.params.ef_construction = 0;
        data.extend_from_slice(&[0.0; 8]);
        norms.push(0.0);
        let before = format!("{index:?}");
        assert!(index.insert(40, &data, &norms, 8).is_err());
        assert_eq!(format!("{index:?}"), before);
    }

    #[test]
    fn incremental_insert_checks_current_topology_before_mutation() {
        let (mut index, mut data, mut norms) = valid_index();
        index.links.pop();
        data.extend_from_slice(&[0.0; 8]);
        norms.push(0.0);
        let before = format!("{index:?}");
        assert!(index.insert(40, &data, &norms, 8).is_err());
        assert_eq!(format!("{index:?}"), before);
    }

    #[test]
    fn incremental_insert_success_is_immediately_searchable() {
        let mut index = empty_index();
        let data = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let norms = [1.0];
        index.insert(0, &data, &norms, 8).unwrap();

        assert_eq!(index.validate_for_store(&data, &norms, 8), Ok(()));
        assert_eq!(
            index.search(&data, norms[0], 1, None, &data, &norms),
            vec![(0, 0.0)]
        );
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
            index.insert(slot, &data, &norms, 24).unwrap();
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
    fn test_deterministic_levels_concurrent_build() {
        // The build is now concurrent (rayon), so the link graph differs
        // run-to-run — but the level assignment is seeded and must be identical,
        // and both builds must reach the same len. (Recall stability across
        // builds is covered by the recall tests, which call `build`.)
        let (data, norms) = make_data(400, 16, 55);
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
        assert_eq!(
            a.node_levels, b.node_levels,
            "seeded levels must be deterministic"
        );
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), 400);
        // Every node's links are bounded by m_max at each layer (valid graph).
        for (slot, layers) in a.links.iter().enumerate() {
            for (lc, nbrs) in layers.iter().enumerate() {
                let m_max = if lc == 0 { a.params.m * 2 } else { a.params.m };
                assert!(
                    nbrs.len() <= m_max,
                    "node {} layer {} over m_max: {} > {}",
                    slot,
                    lc,
                    nbrs.len(),
                    m_max
                );
            }
        }
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

    /// A ramp of collinear points: slot `i` sits at `(i, 1)`, so slot index *is*
    /// position and the exact top-5 for a query at the origin is `0..5`. The
    /// degenerate geometry is the point — it makes a one-way edge visible as a
    /// stranded prefix instead of hiding in ANN noise.
    fn ramp(n: usize) -> (Vec<f32>, Vec<f32>) {
        let mut data = Vec::with_capacity(n * 2);
        for i in 0..n {
            data.push(i as f32);
            data.push(1.0);
        }
        let norms = (0..n).map(|i| ((i * i) as f32 + 1.0).sqrt()).collect();
        (data, norms)
    }

    /// Slots reachable from the entry point by following layer-0 links.
    fn layer_zero_reach(index: &HnswIndex) -> usize {
        let mut seen = vec![false; index.len];
        let start = index
            .entry_point
            .expect("a non-empty index has an entry point") as usize;
        seen[start] = true;
        let mut stack = vec![start];
        let mut count = 1;
        while let Some(slot) = stack.pop() {
            for &e in &index.links[slot][0] {
                if !seen[e as usize] {
                    seen[e as usize] = true;
                    count += 1;
                    stack.push(e as usize);
                }
            }
        }
        count
    }

    #[test]
    fn concurrent_selection_preserves_received_links() {
        let (data, norms) = ramp(4);
        let ctx = DistCtx {
            data: &data,
            norms: &norms,
            dim: 2,
            metric: HnswMetric::Euclidean,
        };
        // Slot 3 linked back while slot 0 was searching. Publishing the search
        // selection must retain that edge and avoid duplicating its slot 1 edge.
        let mut links = vec![3, 1];
        merge_selected_links(&mut links, &[1, 2], &ctx, 0, 4);
        assert_eq!(links, vec![3, 1, 2]);
    }

    /// Real scheduling must preserve navigation through the collinear ramp.
    /// One-way edges are diagnostic only: pruning each endpoint independently
    /// can legitimately remove a reciprocal edge in either build strategy.
    #[test]
    fn concurrent_build_preserves_reachability_and_recall() {
        const N: usize = 600;
        let (data, norms) = ramp(N);
        for threads in [2, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("build a thread pool");

            for round in 0..8 {
                let index = pool.install(|| {
                    HnswIndex::build(
                        &data,
                        &norms,
                        2,
                        HnswMetric::Euclidean,
                        HnswParams::default(),
                        7,
                    )
                });
                let one_way = (0..N)
                    .flat_map(|slot| {
                        index.links[slot][0]
                            .iter()
                            .map(move |&e| (slot as u32, e as usize))
                    })
                    .filter(|&(slot, e)| !index.links[e][0].contains(&slot))
                    .count();
                assert_eq!(
                layer_zero_reach(&index),
                N,
                "{threads} threads, round {round}: layer 0 strands slots; {one_way} one-way edges"
            );
                let hits = index
                    .search(&[0.0, 1.0], 1.0, 5, None, &data, &norms)
                    .iter()
                    .filter(|(slot, _)| (*slot as usize) < 5)
                    .count();
                assert!(
                hits >= 4,
                "{threads} threads, round {round}: recall@5 is {hits}/5; {one_way} one-way edges, top-5 = {:?}",
                index.search(&[0.0, 1.0], 1.0, 5, None, &data, &norms)
            );
            }
        }
    }
}
