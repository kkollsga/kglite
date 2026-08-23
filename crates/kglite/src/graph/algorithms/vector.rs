// Embedding similarity search over the current graph selection.

use super::hnsw::{HnswIndex, HnswMetric};
use crate::graph::schema::{CurrentSelection, DirGraph, EmbeddingStore};
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;
use std::borrow::Cow;
use std::collections::{BTreeSet, BinaryHeap, HashSet};

#[derive(Clone, Copy, Debug)]
pub enum DistanceMetric {
    Cosine,
    DotProduct,
    Euclidean,
    Poincare,
}

impl DistanceMetric {
    /// Single source of truth for the Cypher-facing metric spelling, so
    /// every `vector_score` / `text_score` call site agrees on it.
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

#[derive(Clone, Debug)]
pub struct VectorSearchResult {
    pub node_idx: NodeIndex,
    pub score: f32,
}

/// Ranking knobs for [`vector_search`]. Construct via
/// [`VectorSearchOptions::default`] then the `with_*` builders, e.g.
/// `VectorSearchOptions::default().with_top_k(20)`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct VectorSearchOptions {
    /// Number of results to return (default `10`).
    pub top_k: usize,
    /// Distance metric (default [`DistanceMetric::Cosine`]).
    pub metric: DistanceMetric,
    /// Force a full exact scan, bypassing any HNSW index (default `false`).
    pub exact: bool,
    /// Resolve the metric from embedding stores represented by the selection.
    /// Set through [`Self::with_stored_metric`]; explicit metrics remain the
    /// default for direct Rust callers.
    use_stored_metric: bool,
}

impl Default for VectorSearchOptions {
    fn default() -> Self {
        Self {
            top_k: 10,
            metric: DistanceMetric::Cosine,
            exact: false,
            use_stored_metric: false,
        }
    }
}

impl VectorSearchOptions {
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self.use_stored_metric = false;
        self
    }
    /// Use the unique stored metric represented by the current selection.
    /// Falls back to cosine when no selected embedding store supplies a metric
    /// and rejects selections whose contributing stores disagree.
    pub fn with_stored_metric(mut self) -> Self {
        self.use_stored_metric = true;
        self
    }
    pub fn with_exact(mut self, exact: bool) -> Self {
        self.exact = exact;
        self
    }
}

/// Candidate count above which the scan fans out over rayon.
const PARALLEL_THRESHOLD: usize = 10_000;

/// Minimum candidate count before an HNSW index is auto-used. Below this a
/// brute-force scan is both faster (no index overhead) and exact, so there's no
/// reason to risk approximate recall.
///
/// Calibrated 2026-08-22 (release, `tests/benchmarks/bench_vector_index.py`,
/// cosine, top_k=10, min-of-200-rounds, three agreeing runs). The previous
/// value of 256 sat *below* the measured crossover on both corpora, so the
/// index was auto-used in a band where it was strictly dominated — slower
/// *and* approximate:
///
/// | n   | dim | low-rank corpus | Gaussian corpus |
/// |-----|-----|-----------------|-----------------|
/// | 256 |  64 | 1.04x slower    | 1.15x slower    |
/// | 256 | 384 | 1.04x slower    | 1.13x slower    |
/// | 300 | 384 | 1.03x slower    | 1.15x slower    |
/// | 400 | 384 | 1.09x *faster*  | 1.14x slower    |
/// | 500 | 384 | 1.18x *faster*  | 1.10x slower    |
///
/// 400 is the largest value at which no measured cell prefers the index and
/// the smallest at which no measured cell prefers the scan, on the low-rank
/// corpus that represents real embeddings (`test_bench_vector_index.py`
/// documents why independent Gaussian noise is the adversarial case, not the
/// workload). Raising it further to chase the Gaussian crossover (~600 at
/// d=64, ~750 at d=384) would cost the representative corpus 1.18-1.39x.
///
/// The threshold is deliberately *not* dimension-aware: the representative
/// corpus crosses over at ~350 at both d=64 and d=384, so a `f(dim)` gate
/// would encode a dependence the measurement does not show. Above this band
/// the index wins by 2.7-9.8x at every size and dimension swept up to 50k.
const HNSW_AUTO_MIN: usize = 400;

/// Over-fetch factor for HNSW: fetch `top_k * this` candidates so that, after
/// dropping any that fall outside the (possibly filtered) selection, `top_k`
/// survive. If too few survive, the caller falls back to an exact scan.
const HNSW_OVERSAMPLE: usize = 4;

fn parse_stored_metric(metric: &str) -> Result<DistanceMetric, String> {
    DistanceMetric::from_name(metric).ok_or_else(|| {
        format!(
            "Embedding store uses unknown metric '{metric}'. Expected cosine, dot_product, euclidean, or poincare."
        )
    })
}

fn resolve_metric_from_nodes(
    graph: &DirGraph,
    nodes: impl Iterator<Item = NodeIndex>,
    embedding_property: &str,
) -> Result<DistanceMetric, String> {
    let mut seen_types = BTreeSet::new();
    let mut metrics = BTreeSet::new();
    for node in nodes {
        let Some(node_type_key) = GraphRead::node_type_of(&graph.graph, node) else {
            continue;
        };
        if seen_types.contains(&node_type_key) {
            continue;
        }
        let node_type = graph.interner.resolve(node_type_key);
        let key = (node_type.to_string(), embedding_property.to_string());
        let Some(store) = graph.embeddings.get(&key) else {
            seen_types.insert(node_type_key);
            continue;
        };
        if store.get_embedding(node.index()).is_none() {
            continue;
        }
        seen_types.insert(node_type_key);
        metrics.insert(store.metric.as_deref().unwrap_or("cosine"));
    }

    if metrics.len() > 1 {
        return Err(format!(
            "Selected embedding stores use multiple stored metrics ({}); pass metric= explicitly",
            metrics.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    parse_stored_metric(metrics.into_iter().next().unwrap_or("cosine"))
}

/// A selection that was narrowed contributes its current level. A selection
/// that was **never** narrowed ([`CurrentSelection::never_selected`]) means
/// "the whole graph" — the same rule `get_nodes()` applies — so a caller who
/// never selected gets a search instead of a silent `[]`. A selection a query
/// emptied is a real empty result and returns an empty slice.
///
/// Whole-graph candidates come back in node-index order, which is the order an
/// embedding store's slots are filled in for a freshly embedded type — so a
/// whole-graph search over a single embedded type still satisfies the
/// `ordered_whole_store` coverage proof and rides the HNSW fast path.
fn selection_candidates<'a>(
    graph: &DirGraph,
    selection: &'a CurrentSelection,
) -> Cow<'a, [NodeIndex]> {
    let level = selection
        .get_level(selection.get_level_count().saturating_sub(1))
        .filter(|level| level.node_count() > 0);

    match level {
        // A normal select/filter/set-operation level has one group: borrow its
        // contiguous slice so the common path does not clone O(N) candidates.
        // Multi-parent traversal levels stay on the flattened owned shape.
        Some(level) => {
            let mut groups = level.iter_groups();
            match (groups.next(), groups.next()) {
                (Some((_, nodes)), None) => Cow::Borrowed(nodes.as_slice()),
                _ => Cow::Owned(level.get_all_nodes()),
            }
        }
        None if selection.never_selected() => {
            Cow::Owned(GraphRead::node_indices(&graph.graph).collect())
        }
        None => Cow::Owned(Vec::new()),
    }
}

/// How a search resolves the store(s) behind `embedding_property`.
struct StoreRouting<'a> {
    /// The one store the whole search can run against, when routing proved
    /// that cannot drop rows. `None` sends the search down the per-candidate
    /// multi-store scan.
    single: Option<(&'a str, &'a EmbeddingStore)>,
    /// The candidate slice is exactly this store's slots, in slot order — an
    /// O(1)-after-the-zip proof of whole-store coverage.
    ordered_whole_store: bool,
}

/// The single node type carrying `embedding_property`, if exactly one does.
///
/// Costs O(#stores) — a handful of map keys, independent of graph size.
fn unique_store_for<'a>(
    graph: &'a DirGraph,
    embedding_property: &str,
) -> Option<(&'a str, &'a EmbeddingStore)> {
    let mut stores = graph
        .embeddings
        .iter()
        .filter(|(key, _)| key.1 == *embedding_property)
        .map(|(key, store)| (key.0.as_str(), store));
    let only = stores.next()?;
    stores.next().is_none().then_some(only)
}

/// Decide whether one store can serve the whole search.
///
/// The invariant the single-store path needs is **store uniqueness for this
/// embedding property**, not type homogeneity of the selection: `node_to_slot`
/// is keyed by the global node index, so a candidate of a foreign type simply
/// misses the store and is skipped — exactly what the multi-store scan does
/// with it. Only a *second* type carrying the same property makes the
/// single-store path lossy (it would silently drop that type's rows), and that
/// case keeps the homogeneity proof.
///
/// Routing on uniqueness also *removes* work from the common call: proving
/// homogeneity costs O(#candidates) backend type reads, proving uniqueness
/// costs O(#stores) map keys. The whole-graph search on a graph with a second,
/// un-embedded type used to fail the homogeneity test on its first foreign
/// node and fall to the scan-only path, never reaching the index at all.
fn route_stores<'a>(
    graph: &'a DirGraph,
    candidates: &[NodeIndex],
    embedding_property: &str,
) -> StoreRouting<'a> {
    let first_type = candidates
        .first()
        .and_then(|&node| GraphRead::node_type_of(&graph.graph, node));
    let unique = unique_store_for(graph, embedding_property);
    // With two or more stores the shape checks run against the store the first
    // candidate's type owns, which is the only one the single-store path could
    // have used.
    let shape_store = unique.or_else(|| {
        first_type.and_then(|node_type| {
            let node_type = graph.interner.resolve(node_type);
            let key = (node_type.to_string(), embedding_property.to_string());
            graph.embeddings.get(&key).map(|store| (node_type, store))
        })
    });

    // Type selections preserve embedding insertion order, so exact slot
    // identity proves coverage without backend type reads or a membership set.
    let ordered_whole_store = shape_store.is_some_and(|(_, store)| {
        candidates.len() == store.len()
            && candidates
                .iter()
                .zip(&store.slot_to_node)
                .all(|(candidate, &stored)| candidate.index() == stored)
    });

    // Two or more types carry this property: looking at the first row alone
    // would drop the others from a union selection, so non-contiguous shapes
    // prove type homogeneity with granular primary-type reads first.
    let single = unique.or_else(|| {
        shape_store.filter(|_| {
            ordered_whole_store
                || first_type.is_some_and(|expected| {
                    candidates.iter().all(|&candidate| {
                        GraphRead::node_type_of(&graph.graph, candidate)
                            .is_none_or(|node_type| node_type == expected)
                    })
                })
        })
    });

    StoreRouting {
        single,
        ordered_whole_store,
    }
}

/// Whether "every node in the graph" covers every slot of `store`.
///
/// Removing a node does not prune its embedding, so a store can hold slots for
/// nodes that no longer exist; those are *not* covered by the whole graph and
/// the caller must fall back to an explicit membership set. Costs O(#store)
/// existence probes rather than the O(#candidates) set build that proof would
/// otherwise need — which is the entire point of the whole-graph fast path.
fn whole_graph_covers_store(graph: &DirGraph, store: &EmbeddingStore) -> bool {
    let bound = GraphRead::node_bound(&graph.graph);
    if GraphRead::node_count(&graph.graph) == bound {
        // No freed slots, so every index below the bound is a live node.
        return store.slot_to_node.iter().all(|&node| node < bound);
    }
    store.slot_to_node.iter().all(|&node| {
        node < bound && GraphRead::node_type_of(&graph.graph, NodeIndex::new(node)).is_some()
    })
}

/// The auto-use gate: `Some(covered)` when the index should serve this search,
/// `None` to scan. `covered` is the number of store slots the selection
/// actually contains.
///
/// The gate is stated in *covered slots*, never in raw candidate count. Once
/// the single-store path admits candidates of foreign types (which it must —
/// see [`route_stores`]) the candidate count stops being a coverage measure: a
/// selection of one embedded node beside 100k foreign ones would clear a gate
/// it covers 1/100 000 of, spend the index walk, and then have almost every
/// result filtered back out.
///
/// The candidate-count conditions are kept as a cheap *necessary* pre-filter —
/// `covered` can never exceed `candidates.len()` — so the O(#candidates) probe
/// stays off the reject path, and off the proven-coverage path entirely.
fn hnsw_covered_slots(
    store: &EmbeddingStore,
    candidates: &[NodeIndex],
    coverage_proven: bool,
) -> Option<usize> {
    if candidates.len() < HNSW_AUTO_MIN || candidates.len().saturating_mul(2) < store.len() {
        return None;
    }
    let covered = if coverage_proven {
        store.len()
    } else {
        candidates
            .iter()
            .filter(|node| store.node_to_slot.contains_key(&node.index()))
            .count()
    };
    (covered >= HNSW_AUTO_MIN && covered.saturating_mul(2) >= store.len()).then_some(covered)
}

/// Top-k similarity over the candidates of `selection` (see
/// [`selection_candidates`]), scored highest-first.
///
/// The HNSW index serves the search when `exact` is false, its metric matches
/// the request (cosine/dot/euclidean), and the selection covers enough of the
/// store — "enough" measured in store slots the selection actually contains,
/// not in candidates (see [`hnsw_covered_slots`]). The Poincaré metric,
/// `exact = true`, small or heavily-filtered selections, and stores without an
/// index all fall back to the (norm-accelerated) exact scan.
///
/// The index is reachable whenever *one* node type carries
/// `embedding_property` — the selection may span any number of other types (see
/// [`route_stores`]) — or, when two or more types carry it, whenever the
/// selection is that single type. A selection spanning two embedded types is
/// scored by scan, so no type's rows can be dropped.
pub fn vector_search(
    graph: &DirGraph,
    selection: &CurrentSelection,
    embedding_property: &str,
    query_vector: &[f32],
    options: &VectorSearchOptions,
) -> Result<Vec<VectorSearchResult>, String> {
    let VectorSearchOptions {
        top_k,
        metric,
        exact,
        use_stored_metric,
    } = *options;
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena, which must run under a DiskQueryGuard (arena protocol in
    // disk/graph.rs, enforced by a debug assert); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    if top_k == 0 {
        return Ok(Vec::new());
    }

    let candidates = selection_candidates(graph, selection);
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let StoreRouting {
        single: single_type,
        ordered_whole_store,
    } = route_stores(graph, candidates.as_ref(), embedding_property);

    let metric = if use_stored_metric {
        match single_type {
            Some((_, store)) => parse_stored_metric(store.metric.as_deref().unwrap_or("cosine"))?,
            None => {
                resolve_metric_from_nodes(graph, candidates.iter().copied(), embedding_property)?
            }
        }
    } else {
        metric
    };

    let results = if let Some((node_type, store)) = single_type {
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

        // Whole-store coverage, proven without an O(#candidates) membership
        // set: either the candidates *are* the store's slots in slot order, or
        // the selection was never narrowed and is therefore every live node.
        let coverage_proven = ordered_whole_store
            || (selection.never_selected() && whole_graph_covers_store(graph, store));

        let hnsw_result = if exact {
            None
        } else {
            store.index.as_ref().and_then(|idx| {
                if HnswMetric::from_distance(metric) != Some(idx.metric()) {
                    return None;
                }
                let covered = hnsw_covered_slots(store, candidates.as_ref(), coverage_proven)?;
                debug_assert!(covered >= HNSW_AUTO_MIN);
                hnsw_search(
                    store,
                    idx,
                    candidates.as_ref(),
                    coverage_proven,
                    query_vector,
                    top_k,
                    &scorer,
                )
            })
        };

        let rows = match hnsw_result {
            Some(r) => r,
            None if candidates.len() > PARALLEL_THRESHOLD => {
                parallel_search(&candidates, store, query_vector, top_k, &scorer)
            }
            None => sequential_search(&candidates, store, query_vector, top_k, &scorer),
        };
        // The mistake `unmatched_store_error` reconstructs can only surface as
        // an empty result, so the walk never costs a populated search anything.
        if rows.is_empty() {
            if let Some(err) =
                unmatched_store_error(graph, candidates.as_ref(), node_type, embedding_property)
            {
                return Err(err);
            }
        }
        rows
    } else {
        let scorer = Scorer::new(metric, query_vector);
        let mut heap = MinHeap::with_capacity(top_k);
        let mut cached_type = None;
        let mut cached_store = None;
        // A selection spanning embedded and un-embedded types is a supported
        // partial result, so a per-type miss only skips; "no type matched at
        // all" is the caller mistake `missing_store_error` raises. Both are
        // computed on the cache-miss branch only.
        let mut matched_a_store = false;
        let mut unmatched_types: Vec<&str> = Vec::new();

        for &node_idx in candidates.iter() {
            let node_type = match GraphRead::node_type_of(&graph.graph, node_idx) {
                Some(node_type) => node_type,
                None => continue,
            };

            if cached_type != Some(node_type) {
                let node_type_name = graph.interner.resolve(node_type);
                let key = (node_type_name.to_string(), embedding_property.to_string());
                cached_store = graph.embeddings.get(&key);
                cached_type = Some(node_type);
                if cached_store.is_some() {
                    matched_a_store = true;
                } else if !unmatched_types.contains(&node_type_name) {
                    unmatched_types.push(node_type_name);
                }
            }
            let store = match cached_store {
                Some(s) => s,
                None => continue,
            };

            if query_vector.len() != store.dimension {
                let node_type = graph.interner.resolve(node_type);
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

        if !matched_a_store && !unmatched_types.is_empty() {
            return Err(missing_store_error(
                graph,
                &unmatched_types,
                embedding_property,
            ));
        }

        heap.into_sorted_results()
    };

    Ok(results)
}

/// The error a search raises when *no* selected node type carries the store.
///
/// The silent-`[]` case this replaces was unrecoverable from the outside: the
/// caller could not tell "nothing is similar" from "you named a store that does
/// not exist". So the message names the store that was looked up, the types
/// that were asked for it, and — through the shared probe next to
/// [`store_name`](crate::graph::embeddings::store_name) — the column that would
/// have worked.
///
/// A selection where *some* type has the store never reaches here; those rows
/// are a legitimate partial result.
fn missing_store_error(graph: &DirGraph, node_types: &[&str], store: &str) -> String {
    let text_column = crate::graph::embeddings::text_column_of(store).unwrap_or(store);
    let types = node_types
        .iter()
        .map(|node_type| format!("'{node_type}'"))
        .collect::<Vec<_>>()
        .join(", ");
    let plural = if node_types.len() == 1 {
        "type"
    } else {
        "types"
    };
    let hint = crate::graph::embeddings::unknown_column_hint(
        graph,
        node_types,
        text_column,
        "vector_search()",
    );
    let hint = if hint.is_empty() {
        " Call set_embeddings()/embed_texts() first, or list_embeddings() to see \
         what is embedded."
            .to_string()
    } else {
        hint
    };
    format!(
        "vector_search('{text_column}'): no embedding store '{store}' on node {plural} {types}.{hint}"
    )
}

/// The caller-mistake error for a single-store search that scored nothing:
/// `Some` when *no* candidate's type is `store_type`, the one type carrying the
/// store.
///
/// The multi-store scan raises this from its own per-candidate walk. The
/// single-store path has no such walk — it looks nodes up in the store
/// directly — so it reconstructs the answer here, on the empty result that is
/// the only outcome the mistake can produce.
fn unmatched_store_error(
    graph: &DirGraph,
    candidates: &[NodeIndex],
    store_type: &str,
    embedding_property: &str,
) -> Option<String> {
    let mut unmatched: Vec<&str> = Vec::new();
    for &node_idx in candidates {
        let Some(node_type) = GraphRead::node_type_of(&graph.graph, node_idx) else {
            continue;
        };
        let node_type = graph.interner.resolve(node_type);
        if node_type == store_type {
            return None;
        }
        if !unmatched.contains(&node_type) {
            unmatched.push(node_type);
        }
    }
    (!unmatched.is_empty()).then(|| missing_store_error(graph, &unmatched, embedding_property))
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
///
/// Exact whole-store coverage is deliberately factored into
/// [`store_is_fully_selected`]. Any future fast path that scans the embedding
/// store contiguously instead of walking the selection must use the same gate.
fn hnsw_search(
    store: &EmbeddingStore,
    idx: &HnswIndex,
    candidates: &[NodeIndex],
    coverage_proven: bool,
    query: &[f32],
    top_k: usize,
    scorer: &Scorer,
) -> Option<Vec<VectorSearchResult>> {
    // Shapes routing could not prove without allocating build membership once
    // and reuse it for both the coverage test and the HNSW result filter.
    let membership: Option<HashSet<usize>> = if coverage_proven {
        None
    } else {
        let selected: HashSet<usize> = candidates.iter().map(|n| n.index()).collect();
        if store_is_fully_selected(store, |node| selected.contains(&node)) {
            None
        } else {
            Some(selected)
        }
    };
    let whole_store = membership.is_none();

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
    // Whole-store recall is the ANN contract and is accepted; a filtered
    // shortfall is not.
    if !whole_store && results.len() < top_k {
        return None;
    }
    Some(results)
}

/// Whether a selection contains every node represented by `store`.
///
/// Cardinality alone is insufficient: a same-sized mixed selection can omit
/// embedded nodes and replace them with unrelated, unembedded, or duplicate
/// nodes. Enumerating `slot_to_node` makes coverage exact. Callers supply their
/// existing membership structure, so Cypher's node-to-row map needs no extra
/// allocation and fluent HNSW can reuse its filter set.
pub(crate) fn store_is_fully_selected(
    store: &EmbeddingStore,
    contains_node: impl Fn(usize) -> bool,
) -> bool {
    store.slot_to_node.iter().copied().all(contains_node)
}

// ─── Similarity Functions ──────────────────────────────────────────────────────

type SimilarityFn = fn(&[f32], &[f32]) -> f32;

/// A query-bound scorer: built once per query via [`Scorer::new`] (the query
/// vector is constant across all candidates), then [`Scorer::score`] per
/// candidate alongside its cached L2 norm. Shared by the fluent
/// `vector_search` path and the Cypher `vector_score` / `text_score` scalar
/// function so all cosine scoring benefits from the cached norm.
///
/// Cosine is special-cased: with the query's norm precomputed once and each
/// stored vector's norm cached in the `EmbeddingStore`, the per-candidate work
/// collapses from "dot + two norm sweeps + sqrt" to a single dot product and a
/// divide. Every other metric needs the raw vectors (dot, euclidean) or their
/// magnitudes recomputed per pair (Poincaré is non-linear in the norms), so
/// they fall through to the plain kernel and the cached norm is ignored.
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

/// Cosine similarity between two f32 slices, in [-1.0, 1.0].
/// Uses 4 independent accumulators per metric for instruction-level parallelism,
/// with `as_chunks::<8>()` for LLVM auto-vectorization (SSE2/AVX2/NEON).
///
/// Standalone SIMD util exercised by this module's tests; production cosine
/// scoring goes through [`Scorer`]'s cached-norm kernel instead, so this has
/// no production caller — kept, and tested, as that kernel's parity oracle.
#[allow(dead_code)]
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot0, mut dot1, mut dot2, mut dot3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut na0, mut na1, mut na2, mut na3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut nb0, mut nb1, mut nb2, mut nb3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

    let (a_chunks, a_rem) = a.as_chunks::<8>();
    let (b_chunks, b_rem) = b.as_chunks::<8>();

    for (ac, bc) in a_chunks.iter().zip(b_chunks) {
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

    let (a_chunks, a_rem) = a.as_chunks::<8>();
    let (b_chunks, b_rem) = b.as_chunks::<8>();

    for (ac, bc) in a_chunks.iter().zip(b_chunks) {
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

    let (a_chunks, a_rem) = a.as_chunks::<8>();
    let (b_chunks, b_rem) = b.as_chunks::<8>();

    for (ac, bc) in a_chunks.iter().zip(b_chunks) {
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

    let (a_chunks, a_rem) = a.as_chunks::<8>();
    let (b_chunks, b_rem) = b.as_chunks::<8>();

    for (ac, bc) in a_chunks.iter().zip(b_chunks) {
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
    let alpha = (1.0 - norm_a_sq).max(1e-7);
    let beta = (1.0 - norm_b_sq).max(1e-7);

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

    let mut heap = MinHeap::with_capacity(top_k);
    for thread_results in per_thread_results {
        for result in thread_results {
            heap.push_if_better(result.node_idx, result.score, top_k);
        }
    }

    heap.into_sorted_results()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::Value;
    use crate::graph::algorithms::hnsw::HnswParams;
    use crate::graph::schema::NodeData;
    use crate::graph::storage::GraphWrite;
    use std::collections::HashMap;

    fn selection_of(nodes: Vec<NodeIndex>) -> CurrentSelection {
        let mut selection = CurrentSelection::new();
        selection
            .get_level_mut(0)
            .expect("initial selection level")
            .add_selection(None, nodes);
        selection
    }

    /// A selection a query *emptied*: zero nodes, but a recorded plan step —
    /// the shape a filter that matched nothing leaves behind.
    fn emptied_by_query() -> CurrentSelection {
        let mut selection = selection_of(Vec::new());
        selection.add_plan_step(crate::graph::schema::PlanStep::new(
            "FILTER",
            Some("Doc"),
            0,
        ));
        selection
    }

    /// Two docs of one type, both embedded, plus an untyped-store node type so
    /// the multi-store paths have something to skip.
    fn two_doc_graph() -> (DirGraph, Vec<NodeIndex>) {
        let mut graph = DirGraph::new();
        let mut docs = Vec::new();
        let mut store = EmbeddingStore::with_metric(2, "cosine");
        for id in 0..2 {
            let node = NodeData::new(
                Value::Int64(id as i64),
                Value::String(format!("Doc {id}")),
                "Doc".to_string(),
                HashMap::new(),
                &mut graph.interner,
            );
            let idx = GraphWrite::add_node(&mut graph.graph, node);
            graph
                .type_indices
                .entry_or_default("Doc".to_string())
                .push(idx);
            store.set_embedding(idx.index(), &[1.0 - id as f32, id as f32]);
            docs.push(idx);
        }
        graph
            .embeddings
            .insert(("Doc".to_string(), "summary_emb".to_string()), store);
        (graph, docs)
    }

    #[test]
    fn never_selected_candidates_are_the_whole_graph_in_index_order() {
        let (graph, docs) = two_doc_graph();
        let virgin = CurrentSelection::new();
        assert!(virgin.never_selected());
        assert_eq!(
            selection_candidates(&graph, &virgin).as_ref(),
            docs.as_slice(),
            "a never-narrowed selection resolves to every node, in index order \
             (which is what keeps the HNSW whole-store coverage proof valid)"
        );

        // A query that matched nothing is a real empty result, not "everything".
        let emptied = emptied_by_query();
        assert!(!emptied.never_selected());
        assert!(selection_candidates(&graph, &emptied).is_empty());

        // An explicit selection is unchanged.
        assert_eq!(
            selection_candidates(&graph, &selection_of(docs.clone())).as_ref(),
            docs.as_slice()
        );
    }

    #[test]
    fn never_selected_search_covers_the_graph_and_an_emptied_query_stays_empty() {
        let (graph, docs) = two_doc_graph();
        let options = VectorSearchOptions::default()
            .with_top_k(2)
            .with_metric(DistanceMetric::Cosine);

        let whole = vector_search(
            &graph,
            &CurrentSelection::new(),
            "summary_emb",
            &[1.0, 0.0],
            &options,
        )
        .expect("never-selected search");
        assert_eq!(
            whole.iter().map(|r| r.node_idx).collect::<Vec<_>>(),
            vec![docs[0], docs[1]]
        );

        let emptied = vector_search(
            &graph,
            &emptied_by_query(),
            "summary_emb",
            &[1.0, 0.0],
            &options,
        )
        .expect("emptied search");
        assert!(emptied.is_empty());

        // An empty graph has nothing to search either way.
        let empty_graph = DirGraph::new();
        assert!(vector_search(
            &empty_graph,
            &CurrentSelection::new(),
            "summary_emb",
            &[1.0, 0.0],
            &options,
        )
        .expect("empty-graph search")
        .is_empty());
    }

    /// One embedded `Doc` and one un-embedded `Note`, so a selection can span a
    /// type that has the store and one that does not.
    fn doc_and_note_graph() -> (DirGraph, NodeIndex, NodeIndex) {
        let (mut graph, docs) = two_doc_graph();
        let note = NodeData::new(
            Value::Int64(0),
            Value::String("Note 0".to_string()),
            "Note".to_string(),
            HashMap::new(),
            &mut graph.interner,
        );
        let note = GraphWrite::add_node(&mut graph.graph, note);
        graph
            .type_indices
            .entry_or_default("Note".to_string())
            .push(note);
        (graph, docs[0], note)
    }

    /// The silent-`[]` bug: the Python surface mints `{column}_emb`, so a caller
    /// who passes the *store* name searches `summary_emb_emb` — a key nothing
    /// can ever hold. Every node was skipped and the empty list read as "no
    /// matches". It must name the column that would have worked.
    #[test]
    fn a_store_no_selected_type_has_raises_and_names_the_text_column() {
        let (graph, docs) = two_doc_graph();
        let options = VectorSearchOptions::default().with_top_k(2);

        let err = vector_search(
            &graph,
            &selection_of(docs),
            "summary_emb_emb",
            &[1.0, 0.0],
            &options,
        )
        .expect_err("a store no selected type has is a caller mistake, not an empty result");
        assert!(err.contains("summary_emb_emb"), "{err}");
        assert!(err.contains("Did you mean 'summary'?"), "{err}");
        assert!(
            err.contains("vector_search() takes the text column"),
            "{err}"
        );
    }

    /// An unknown column has no suffix story, so the error falls back to the
    /// columns the selected types actually have embedded.
    #[test]
    fn an_unknown_column_raises_listing_the_embedded_columns() {
        let (graph, docs) = two_doc_graph();
        let options = VectorSearchOptions::default().with_top_k(2);

        let err = vector_search(
            &graph,
            &selection_of(docs),
            "nope_emb",
            &[1.0, 0.0],
            &options,
        )
        .expect_err("an unknown column is a caller mistake");
        assert!(err.contains("'nope_emb'"), "{err}");
        assert!(err.contains("summary"), "{err}");
    }

    /// A near-miss gets the generic did-you-mean, over the type's own columns.
    #[test]
    fn a_near_miss_column_gets_a_suggestion() {
        let (graph, docs) = two_doc_graph();
        let options = VectorSearchOptions::default().with_top_k(2);

        let err = vector_search(
            &graph,
            &selection_of(docs),
            "summry_emb",
            &[1.0, 0.0],
            &options,
        )
        .expect_err("a near-miss column is still a caller mistake");
        assert!(err.contains("Did you mean 'summary'?"), "{err}");
    }

    /// The supported heterogeneous case: some selected types carry the store and
    /// some do not. That is a partial result, never an error — the raise fires
    /// only when *nothing* in the selection could have matched.
    #[test]
    fn a_selection_only_partly_covered_by_the_store_still_returns_its_rows() {
        let (graph, doc, note) = doc_and_note_graph();
        let options = VectorSearchOptions::default().with_top_k(5);

        let rows = vector_search(
            &graph,
            &selection_of(vec![doc, note]),
            "summary_emb",
            &[1.0, 0.0],
            &options,
        )
        .expect("a type without the store is skipped, not fatal");
        assert_eq!(
            rows.iter().map(|r| r.node_idx).collect::<Vec<_>>(),
            vec![doc]
        );
    }

    /// The same selection narrowed to the un-embedded type alone: now nothing
    /// matched, so it is the mistake case again.
    #[test]
    fn a_selection_of_only_unembedded_types_raises() {
        let (graph, _doc, note) = doc_and_note_graph();
        let options = VectorSearchOptions::default().with_top_k(5);

        let err = vector_search(
            &graph,
            &selection_of(vec![note]),
            "summary_emb",
            &[1.0, 0.0],
            &options,
        )
        .expect_err("no selected type has the store");
        assert!(err.contains("'Note'"), "{err}");
    }

    #[test]
    fn hnsw_mixed_selection_never_returns_unselected_store_nodes() {
        // DOCS - OMITTED_DOCS must stay above HNSW_AUTO_MIN: the auto-use gate
        // counts *store-covered* candidates, so a smaller embedded set would
        // route the mixed case to the scan and leave the membership filter
        // this test exists for unexercised.
        const DOCS: usize = 520;
        const OMITTED_DOCS: usize = 32;
        const UNEMBEDDED: usize = 520;

        let mut graph = DirGraph::new();
        let mut docs = Vec::with_capacity(DOCS);
        for id in 0..DOCS {
            let node = NodeData::new(
                Value::Int64(id as i64),
                Value::String(format!("Doc {id}")),
                "Doc".to_string(),
                HashMap::new(),
                &mut graph.interner,
            );
            let idx = GraphWrite::add_node(&mut graph.graph, node);
            graph
                .type_indices
                .entry_or_default("Doc".to_string())
                .push(idx);
            docs.push(idx);
        }

        let mut unembedded = Vec::with_capacity(UNEMBEDDED);
        for id in 0..UNEMBEDDED {
            let node = NodeData::new(
                Value::Int64(id as i64),
                Value::String(format!("Other {id}")),
                "Doc".to_string(),
                HashMap::new(),
                &mut graph.interner,
            );
            let idx = GraphWrite::add_node(&mut graph.graph, node);
            graph
                .type_indices
                .entry_or_default("Doc".to_string())
                .push(idx);
            unembedded.push(idx);
        }

        let mut store = EmbeddingStore::with_metric(2, "euclidean");
        for (id, &node) in docs.iter().enumerate() {
            store.set_embedding(node.index(), &[id as f32, 0.0]);
        }
        store
            .build_index(DistanceMetric::Euclidean, HnswParams::default(), 7)
            .expect("build HNSW index");
        graph
            .embeddings
            .insert(("Doc".to_string(), "summary_emb".to_string()), store);

        let selected_docs = docs[OMITTED_DOCS..].to_vec();
        let mut mixed_nodes = selected_docs.clone();
        mixed_nodes.extend(unembedded);
        assert!(mixed_nodes.len() >= DOCS);
        let store = graph
            .embeddings
            .get(&("Doc".to_string(), "summary_emb".to_string()))
            .expect("inserted embedding store");
        let mixed_store_members: HashSet<usize> =
            mixed_nodes.iter().map(|node| node.index()).collect();
        assert!(!store_is_fully_selected(store, |node| {
            mixed_store_members.contains(&node)
        }));
        let whole_store_members: HashSet<usize> = docs.iter().map(|node| node.index()).collect();
        assert!(store_is_fully_selected(store, |node| {
            whole_store_members.contains(&node)
        }));
        let duplicate_candidates = vec![docs[OMITTED_DOCS]; DOCS];
        let duplicate_nodes: HashSet<usize> = duplicate_candidates
            .iter()
            .map(|node| node.index())
            .collect();
        assert!(
            !store_is_fully_selected(store, |node| duplicate_nodes.contains(&node)),
            "duplicate candidates cannot stand in for omitted store nodes"
        );
        let mixed_members: HashSet<NodeIndex> = mixed_nodes.iter().copied().collect();
        let query = [0.0, 0.0];
        let options = VectorSearchOptions::default()
            .with_top_k(5)
            .with_metric(DistanceMetric::Euclidean);

        let mixed = vector_search(
            &graph,
            &selection_of(mixed_nodes.clone()),
            "summary_emb",
            &query,
            &options,
        )
        .expect("mixed approximate search");
        assert_eq!(mixed.len(), 5);
        assert!(
            mixed
                .iter()
                .all(|result| mixed_members.contains(&result.node_idx)),
            "mixed HNSW results escaped the current selection: {:?}",
            mixed
                .iter()
                .map(|result| result.node_idx)
                .collect::<Vec<_>>()
        );

        let mixed_exact = vector_search(
            &graph,
            &selection_of(mixed_nodes),
            "summary_emb",
            &query,
            &options.clone().with_exact(true),
        )
        .expect("mixed exact search");
        assert_eq!(mixed_exact.len(), 5);
        assert_eq!(mixed_exact[0].node_idx, docs[OMITTED_DOCS]);
        assert!(mixed_exact
            .iter()
            .all(|result| mixed_members.contains(&result.node_idx)));

        let filtered_members: HashSet<NodeIndex> = selected_docs.iter().copied().collect();
        let filtered = vector_search(
            &graph,
            &selection_of(selected_docs.clone()),
            "summary_emb",
            &query,
            &options,
        )
        .expect("filtered approximate search");
        assert_eq!(filtered.len(), 5);
        assert!(filtered
            .iter()
            .all(|result| filtered_members.contains(&result.node_idx)));

        let exact = vector_search(
            &graph,
            &selection_of(selected_docs),
            "summary_emb",
            &query,
            &options.clone().with_exact(true),
        )
        .expect("filtered exact search");
        assert_eq!(exact.len(), 5);
        assert_eq!(exact[0].node_idx, docs[OMITTED_DOCS]);

        let whole = vector_search(
            &graph,
            &selection_of(docs.clone()),
            "summary_emb",
            &query,
            &options,
        )
        .expect("whole-store approximate search");
        let whole_members: HashSet<NodeIndex> = docs.iter().copied().collect();
        assert_eq!(whole.len(), 5);
        assert!(whole
            .iter()
            .all(|result| whole_members.contains(&result.node_idx)));

        let whole_exact = vector_search(
            &graph,
            &selection_of(docs.clone()),
            "summary_emb",
            &query,
            &options.with_exact(true),
        )
        .expect("whole-store exact search");
        assert_eq!(whole_exact.len(), 5);
        assert_eq!(whole_exact[0].node_idx, docs[0]);
        assert!(whole_exact
            .iter()
            .all(|result| whole_members.contains(&result.node_idx)));
    }

    /// `count` nodes of `node_type`, embedded into `store` when `embed`.
    fn push_nodes(
        graph: &mut DirGraph,
        node_type: &str,
        count: usize,
        store: Option<&mut EmbeddingStore>,
    ) -> Vec<NodeIndex> {
        let mut store = store;
        let mut nodes = Vec::with_capacity(count);
        for id in 0..count {
            let node = NodeData::new(
                Value::Int64(id as i64),
                Value::String(format!("{node_type} {id}")),
                node_type.to_string(),
                HashMap::new(),
                &mut graph.interner,
            );
            let idx = GraphWrite::add_node(&mut graph.graph, node);
            graph
                .type_indices
                .entry_or_default(node_type.to_string())
                .push(idx);
            if let Some(store) = store.as_deref_mut() {
                store.set_embedding(idx.index(), &[id as f32, 1.0]);
            }
            nodes.push(idx);
        }
        nodes
    }

    /// An embedded `Doc` type beside an un-embedded `Note` type — the shape a
    /// whole-graph search on any realistic multi-type graph has.
    fn one_embedded_type_graph(docs: usize, notes: usize) -> (DirGraph, Vec<NodeIndex>) {
        let mut graph = DirGraph::new();
        let mut store = EmbeddingStore::with_metric(2, "euclidean");
        let doc_nodes = push_nodes(&mut graph, "Doc", docs, Some(&mut store));
        push_nodes(&mut graph, "Note", notes, None);
        store
            .build_index(DistanceMetric::Euclidean, HnswParams::default(), 7)
            .expect("build HNSW index");
        graph
            .embeddings
            .insert(("Doc".to_string(), "summary_emb".to_string()), store);
        (graph, doc_nodes)
    }

    /// The routing bug: whole-graph search on a multi-type graph never reached
    /// the index, because eligibility was proven by *type homogeneity* — which
    /// the first `Note` node kills — instead of by store uniqueness.
    #[test]
    fn one_store_routes_a_mixed_candidate_set_to_that_store() {
        let (graph, docs) = one_embedded_type_graph(600, 600);
        let whole_graph: Vec<NodeIndex> = GraphRead::node_indices(&graph.graph).collect();
        assert!(whole_graph.len() > docs.len(), "the graph is mixed");

        let routing = route_stores(&graph, &whole_graph, "summary_emb");
        let (node_type, store) = routing
            .single
            .expect("one type carries the property, so one store can serve any candidate set");
        assert_eq!(node_type, "Doc");
        assert!(
            !routing.ordered_whole_store,
            "the mixed candidate slice is not the store's slot order"
        );

        // Never-narrowed candidates are every live node, so they cover the
        // store without an O(#candidates) membership set...
        assert!(whole_graph_covers_store(&graph, store));
        // ...and the covered-slot gate then admits the index.
        assert_eq!(
            hnsw_covered_slots(store, &whole_graph, true),
            Some(store.len())
        );
    }

    /// The never-drop guarantee: two embedded types keep the homogeneity proof,
    /// so a union of both is scored by scan rather than by one type's index.
    #[test]
    fn two_embedded_types_keep_the_homogeneity_fallback() {
        let mut graph = DirGraph::new();
        let mut docs_store = EmbeddingStore::with_metric(2, "euclidean");
        let mut notes_store = EmbeddingStore::with_metric(2, "euclidean");
        let docs = push_nodes(&mut graph, "Doc", 8, Some(&mut docs_store));
        let notes = push_nodes(&mut graph, "Note", 8, Some(&mut notes_store));
        graph
            .embeddings
            .insert(("Doc".to_string(), "summary_emb".to_string()), docs_store);
        graph
            .embeddings
            .insert(("Note".to_string(), "summary_emb".to_string()), notes_store);

        let mut mixed = docs.clone();
        mixed.extend(notes);
        assert!(
            route_stores(&graph, &mixed, "summary_emb").single.is_none(),
            "a union of two embedded types must not be served by one of the \
             two stores — that silently drops the other type's rows"
        );
        // A homogeneous selection of either type is still eligible.
        assert_eq!(
            route_stores(&graph, &docs, "summary_emb")
                .single
                .map(|s| s.0),
            Some("Doc")
        );
    }

    /// The coverage gate is stated in store slots, not candidates: admitting
    /// mixed candidates makes the raw candidate count an overstatement.
    #[test]
    fn the_auto_use_gate_counts_covered_slots_not_candidates() {
        let (graph, docs) = one_embedded_type_graph(600, 4_000);
        let store = graph
            .embeddings
            .get(&("Doc".to_string(), "summary_emb".to_string()))
            .expect("inserted store");

        let mut one_doc_among_notes: Vec<NodeIndex> = GraphRead::node_indices(&graph.graph)
            .skip(docs.len())
            .collect();
        one_doc_among_notes.push(docs[0]);
        assert!(one_doc_among_notes.len() > HNSW_AUTO_MIN);
        assert!(one_doc_among_notes.len() > store.len());
        assert_eq!(
            hnsw_covered_slots(store, &one_doc_among_notes, false),
            None,
            "4000 candidates covering one of 600 slots must not engage the index"
        );

        // The same shape, but genuinely covering the store, is admitted.
        let mut most_docs_among_notes = one_doc_among_notes.clone();
        most_docs_among_notes.extend_from_slice(&docs[1..]);
        assert_eq!(
            hnsw_covered_slots(store, &most_docs_among_notes, false),
            Some(store.len())
        );

        // And a genuinely small selection is still excluded by size alone.
        assert_eq!(hnsw_covered_slots(store, &docs[..64], false), None);
    }

    /// A store slot whose node is gone voids the whole-graph coverage proof.
    ///
    /// **This assertion was inverted in R2**, where it pinned the *defect*:
    /// deleting a node did not prune its embedding, so the ordinary
    /// `DELETE` path produced exactly this state and the proof had to defend
    /// against it. `detach_delete_nodes` now prunes (see the test below), so
    /// the only way left to reach a dangling slot is a removal *beneath* that
    /// chokepoint — a raw backend `remove_node`, which in production is only
    /// the undo of a node a statement created, and no statement can embed a
    /// node it just created. The proof stays because it is cheap and this
    /// construction is still expressible.
    #[test]
    fn a_dangling_store_slot_voids_the_whole_graph_coverage_proof() {
        let (mut graph, docs) = one_embedded_type_graph(8, 2);
        let store_key = ("Doc".to_string(), "summary_emb".to_string());
        assert!(whole_graph_covers_store(
            &graph,
            graph.embeddings.get(&store_key).expect("store")
        ));

        // Beneath the chokepoint on purpose: this is the residual case.
        GraphWrite::remove_node(&mut graph.graph, docs[3]).expect("remove an embedded node");
        assert!(
            !whole_graph_covers_store(&graph, graph.embeddings.get(&store_key).expect("store")),
            "the store still holds a slot for a node the graph no longer has"
        );
    }

    /// The deletion chokepoint prunes the node's vector, so the store never
    /// holds a slot for a node the graph no longer has.
    #[test]
    fn deleting_an_embedded_node_prunes_its_vector() {
        let (mut graph, docs) = one_embedded_type_graph(8, 2);
        let store_key = ("Doc".to_string(), "summary_emb".to_string());
        let doomed = docs[3];

        crate::graph::mutation::maintain::detach_delete_nodes(&mut graph, &HashSet::from([doomed]));

        let store = graph.embeddings.get(&store_key).expect("store");
        assert_eq!(store.len(), 7, "the deleted node gave up its slot");
        assert_eq!(store.get_embedding(doomed.index()), None);
        assert_eq!(store.validate_shape(), Ok(()));
        assert!(
            whole_graph_covers_store(&graph, store),
            "every remaining slot belongs to a live node"
        );
    }

    /// The ghost: `StableDiGraph` hands the deleted node's index to the next
    /// node created, and an un-pruned store made that node inherit the
    /// deleted one's vector — a 1.0-similarity top hit for a node of any
    /// type that was never embedded. Pinned on both dispatch paths.
    #[test]
    fn a_node_reusing_a_deleted_slot_inherits_no_vector() {
        for exact in [true, false] {
            let (mut graph, docs) = one_embedded_type_graph(600, 2);
            let doomed = docs[7];
            let doomed_vector = graph
                .embeddings
                .get(&("Doc".to_string(), "summary_emb".to_string()))
                .expect("store")
                .get_embedding(doomed.index())
                .expect("the doomed node is embedded")
                .to_vec();

            crate::graph::mutation::maintain::detach_delete_nodes(
                &mut graph,
                &HashSet::from([doomed]),
            );
            // Any type will do — the store is keyed by the global node index,
            // so a `Note` landing on the freed slot inherits just as readily.
            let heir = push_nodes(&mut graph, "Note", 1, None)[0];
            assert_eq!(heir, doomed, "the freed index is reused");

            // Rebuild the index the prune dropped, so the non-exact arm
            // exercises the HNSW path rather than falling back to the scan.
            let store = graph
                .embeddings
                .get_mut(&("Doc".to_string(), "summary_emb".to_string()))
                .expect("store");
            store
                .build_index(DistanceMetric::Euclidean, HnswParams::default(), 7)
                .expect("rebuild HNSW index");

            let rows = vector_search(
                &graph,
                &CurrentSelection::new(),
                "summary_emb",
                &doomed_vector,
                &VectorSearchOptions::default()
                    .with_top_k(5)
                    .with_metric(DistanceMetric::Euclidean)
                    .with_exact(exact),
            )
            .expect("whole-graph search");

            assert!(
                !rows.iter().any(|r| r.node_idx == heir),
                "exact={exact}: the node that reused the slot returned {rows:?}"
            );
        }
    }

    /// End-to-end: the whole-graph search on a mixed graph returns exactly the
    /// exact scan's answer, index or not.
    #[test]
    fn whole_graph_search_on_a_mixed_graph_matches_the_exact_scan() {
        let (graph, docs) = one_embedded_type_graph(600, 600);
        let options = VectorSearchOptions::default()
            .with_top_k(5)
            .with_metric(DistanceMetric::Euclidean);
        let query = [0.0, 1.0];

        let approximate = vector_search(
            &graph,
            &CurrentSelection::new(),
            "summary_emb",
            &query,
            &options,
        )
        .expect("whole-graph search");
        let exact = vector_search(
            &graph,
            &CurrentSelection::new(),
            "summary_emb",
            &query,
            &options.clone().with_exact(true),
        )
        .expect("whole-graph exact search");
        assert_eq!(
            approximate.iter().map(|r| r.node_idx).collect::<Vec<_>>(),
            exact.iter().map(|r| r.node_idx).collect::<Vec<_>>()
        );
        assert_eq!(
            exact.iter().map(|r| r.node_idx).collect::<Vec<_>>(),
            docs[..5].to_vec()
        );
    }

    /// Routing on store uniqueness must not swallow the caller mistake the
    /// multi-store scan raises: a selection of types that do not carry the
    /// store still errors instead of returning an empty list.
    #[test]
    fn a_single_store_graph_still_raises_when_no_selected_type_has_it() {
        let (graph, doc, note) = doc_and_note_graph();
        let options = VectorSearchOptions::default().with_top_k(5);

        let err = vector_search(
            &graph,
            &selection_of(vec![note]),
            "summary_emb",
            &[1.0, 0.0],
            &options,
        )
        .expect_err("the only store belongs to a type the selection excludes");
        assert!(err.contains("'Note'"), "{err}");

        // A selection that does include the store's type is a normal result.
        let rows = vector_search(
            &graph,
            &selection_of(vec![doc, note]),
            "summary_emb",
            &[1.0, 0.0],
            &options,
        )
        .expect("partial coverage is a supported result");
        assert_eq!(
            rows.iter().map(|r| r.node_idx).collect::<Vec<_>>(),
            vec![doc]
        );
    }

    #[test]
    fn hnsw_index_is_not_used_for_a_different_requested_metric() {
        const DOCS: usize = 320;
        let mut graph = DirGraph::new();
        let mut docs = Vec::with_capacity(DOCS);
        let mut store = EmbeddingStore::with_metric(2, "cosine");
        for id in 0..DOCS {
            let node = NodeData::new(
                Value::Int64(id as i64),
                Value::String(format!("Doc {id}")),
                "Doc".to_string(),
                HashMap::new(),
                &mut graph.interner,
            );
            let idx = GraphWrite::add_node(&mut graph.graph, node);
            graph
                .type_indices
                .entry_or_default("Doc".to_string())
                .push(idx);
            docs.push(idx);
            let embedding = match id {
                0 => [1.0, 0.0],
                1 => [100.0, 100.0],
                _ => [1.0, 0.1],
            };
            store.set_embedding(idx.index(), &embedding);
        }
        store
            .build_index(DistanceMetric::Cosine, HnswParams::default(), 7)
            .unwrap();
        graph
            .embeddings
            .insert(("Doc".to_string(), "summary_emb".to_string()), store);

        let options = VectorSearchOptions::default()
            .with_top_k(1)
            .with_metric(DistanceMetric::DotProduct);
        let automatic = vector_search(
            &graph,
            &selection_of(docs.clone()),
            "summary_emb",
            &[1.0, 0.0],
            &options,
        )
        .unwrap();
        let exact = vector_search(
            &graph,
            &selection_of(docs),
            "summary_emb",
            &[1.0, 0.0],
            &options.with_exact(true),
        )
        .unwrap();
        assert_eq!(automatic[0].node_idx, exact[0].node_idx);
        assert_eq!(exact[0].node_idx, NodeIndex::new(1));
    }

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
        assert!(sim > 0.99);
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
