"""Gated HNSW build/search benchmarks with exact-search accuracy oracles.

The 10k x 128 corpus is the practical iteration gate.  The 50k x 128 cells
exercise scaling, but are also marked ``slow`` because building their index is
intentionally expensive.  Run either surface explicitly::

    pytest tests/benchmarks/test_bench_vector_index.py -m benchmark -v
    pytest tests/benchmarks/test_bench_vector_index.py -m "benchmark and not slow" -v

Corpus construction and embedding ingestion happen in the module-scoped
fixture.  Only ``build_vector_index`` is inside the build timing, and the
search timings operate on an already-built index.  The accuracy checks compare
``exact=True`` with an independent NumPy brute-force oracle, then compare HNSW
with that exact ordering outside every timed search region.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
import pandas as pd
import pytest

import kglite

DIMENSION = 128
TOP_K = 10
QUERY_COUNT = 20
RECALL_FLOOR = 0.90
SEARCH_ROUNDS = 100
SEARCH_WARMUP_ROUNDS = 20
LATENT_DIMENSION = 8
SELECTOR_VECTORS = 2_048
FIXED_SEARCH_VECTORS = 10_000
MULTI_TYPE_VECTORS_PER_TYPE = 2_048
SELECTOR_WIDTHS = ((8, 64), (8, 200), (16, 64), (16, 200), (32, 64), (32, 200))
VISITED_WIDTH_TOP_K = (16, 32, 64, 128)
#: Corpus for the auto-selection guard: inside the band the engine's
#: ``HNSW_AUTO_MIN`` gate must exclude, at the widest dimension swept.
AUTO_GATE_VECTORS = 300
AUTO_GATE_DIMENSION = 384
#: Seen from both sides on 2026-08-22 (release): with the gate correct the two
#: arms are the same code path and measure 0.987-1.000x; with the gate mutated
#: to admit the index at this size the same cell measures 1.10-1.17x.  1.05
#: sits between the two clusters with ~5% margin on each side.
AUTO_GATE_MAX_RATIO = 1.05
#: Un-embedded nodes placed beside the embedded store for the whole-graph cell.
WHOLE_GRAPH_FOREIGN_NODES = 10_000
#: Whole-graph multi-type guard: the index must beat the exact scan by at least
#: this factor.  Stated as a speedup floor, not a never-slower tolerance —
#: when the routing is broken both arms run the same scan and the ratio is
#: ~1.0, so only a floor below 1.0 can catch it.  Seen from both sides on
#: 2026-08-22 (release): routed correctly the cell measures 0.35-0.38x, with
#: the routing reverted to the type-homogeneity predicate it measures
#: 0.99-1.00x.  0.60 is the geometric midpoint, ~1.6x of margin on each side.
WHOLE_GRAPH_AUTO_MAX_RATIO = 0.60


@dataclass
class VectorCorpus:
    """Graph plus independent query/truth data shared by one size's cells."""

    graph: kglite.KnowledgeGraph
    vectors: np.ndarray
    selected_ids: frozenset[int]
    query_ids: tuple[int, ...]
    _exact: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    _oracle: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    _norms: np.ndarray | None = None

    def query(self, query_id: int) -> np.ndarray:
        return self.vectors[query_id]

    def exact_ids(self, query_id: int, top_k: int = TOP_K) -> list[int]:
        key = (query_id, top_k)
        if key not in self._exact:
            rows = self.graph.select("Doc").vector_search(
                "summary",
                self.query(query_id),
                top_k=top_k,
                exact=True,
            )
            self._exact[key] = [int(row["id"]) for row in rows]
        return self._exact[key]

    def oracle_ids(self, query_id: int, top_k: int = TOP_K) -> list[int]:
        """Independent NumPy brute-force cosine top-k, including order."""
        key = (query_id, top_k)
        if key not in self._oracle:
            if self._norms is None:
                self._norms = np.linalg.norm(self.vectors, axis=1)
            query = self.query(query_id)
            scores = (self.vectors @ query) / (self._norms * np.linalg.norm(query))
            # Random projected vectors have no exact score ties.  Stable sort
            # nevertheless makes the oracle's tie behavior explicit.
            order = np.argsort(-scores, kind="stable")[:top_k]
            self._oracle[key] = [int(node_id) for node_id in order]
        return self._oracle[key]

    def approximate_ids(self, query_id: int, top_k: int = TOP_K) -> list[int]:
        rows = self.graph.select("Doc").vector_search(
            "summary",
            self.query(query_id),
            top_k=top_k,
        )
        return [int(row["id"]) for row in rows]


@dataclass(frozen=True)
class MultiTypeExactCorpus:
    """Preselected multi-store exact-scan corpus and independent truth."""

    selection: kglite.KnowledgeGraph
    query: np.ndarray
    expected_ids: tuple[int, ...]
    expected_scores: tuple[float, ...]
    selected_ids: frozenset[int]


def _build_corpus(n: int, seed: int = 20_260_805) -> VectorCorpus:
    """Create deterministic, structured embeddings without Python float bloat.

    Dense semantic embeddings occupy a much lower-dimensional manifold than
    their storage width.  Projecting eight latent features into 128 dimensions
    preserves that property; independent 128-D Gaussian noise is an adversarial
    nearest-neighbour corpus whose default HNSW recall degrades with corpus size
    and is not representative of the product workload.
    """
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n, LATENT_DIMENSION), dtype=np.float32)
    projection = rng.standard_normal((LATENT_DIMENSION, DIMENSION), dtype=np.float32)
    vectors = np.asarray(latent @ projection, dtype=np.float32)
    graph = kglite.KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": np.arange(n, dtype=np.int64),
                "title": [f"n{i}" for i in range(n)],
                "summary": [f"text {i}" for i in range(n)],
            }
        ),
        "Doc",
        "id",
        "title",
    )
    report = graph.set_embeddings("Doc", "summary", dict(enumerate(vectors)), metric="cosine")
    assert report == {"embeddings_stored": n, "dimension": DIMENSION, "skipped": 0}

    # Span the corpus rather than sampling a scheduling-friendly prefix.  The
    # concurrent HNSW build is deliberately nondeterministic, so accuracy is
    # gated on aggregate recall, never identical approximate result order.
    query_ids = tuple(int(i) for i in np.linspace(0, n - 1, QUERY_COUNT, dtype=np.int64))
    return VectorCorpus(graph, vectors, frozenset(range(n)), query_ids)


def _build_multi_type_exact_corpus(
    n_per_type: int = MULTI_TYPE_VECTORS_PER_TYPE,
    seed: int = 20_260_807,
) -> MultiTypeExactCorpus:
    """Build two contiguous stores selected as one multi-type exact scan."""
    total = 2 * n_per_type
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((total, LATENT_DIMENSION), dtype=np.float32)
    projection = rng.standard_normal((LATENT_DIMENSION, DIMENSION), dtype=np.float32)
    vectors = np.asarray(latent @ projection, dtype=np.float32)
    graph = kglite.KnowledgeGraph()

    for type_offset, node_type in ((0, "Doc"), (n_per_type, "Note")):
        node_ids = np.arange(type_offset, type_offset + n_per_type, dtype=np.int64)
        graph.add_nodes(
            pd.DataFrame(
                {
                    "id": node_ids,
                    "title": [f"{node_type.lower()}-{node_id}" for node_id in node_ids],
                    "summary": [f"text {node_id}" for node_id in node_ids],
                }
            ),
            node_type,
            "id",
            "title",
        )
        stored = graph.set_embeddings(
            node_type,
            "summary",
            {int(node_id): vectors[int(node_id)] for node_id in range(type_offset, type_offset + n_per_type)},
            metric="cosine",
        )
        assert stored == {"embeddings_stored": n_per_type, "dimension": DIMENSION, "skipped": 0}

    query_id = n_per_type + n_per_type // 2
    query = vectors[query_id]
    norms = np.linalg.norm(vectors, axis=1)
    scores = (vectors @ query) / (norms * np.linalg.norm(query))
    expected_order = np.argsort(-scores, kind="stable")[:TOP_K]
    expected_ids = tuple(int(node_id) for node_id in expected_order)
    expected_scores = tuple(float(scores[node_id]) for node_id in expected_order)
    selected_ids = frozenset(range(total))
    selection = graph.select("Doc").union(graph.select("Note"))
    assert len(selection) == total
    assert expected_ids[0] == query_id
    return MultiTypeExactCorpus(selection, query, expected_ids, expected_scores, selected_ids)


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(10_000, id="10k-x-128"),
        pytest.param(50_000, id="50k-x-128", marks=pytest.mark.slow),
    ],
)
def vector_corpus(request) -> VectorCorpus:
    return _build_corpus(request.param)


@pytest.fixture
def selector_corpus() -> VectorCorpus:
    """Return an index-free corpus for one selector-width build cell."""
    return _build_corpus(SELECTOR_VECTORS, seed=20_260_806)


@pytest.fixture(scope="module")
def fixed_search_corpus() -> VectorCorpus:
    corpus = _build_corpus(FIXED_SEARCH_VECTORS, seed=20_260_808)
    corpus.graph.build_vector_index("Doc", "summary", m=16, ef_construction=200, ef_search=64)
    assert corpus.graph.has_vector_index("Doc", "summary")
    return corpus


@pytest.fixture(scope="module")
def multi_type_exact_corpus() -> MultiTypeExactCorpus:
    return _build_multi_type_exact_corpus()


def _ensure_index(corpus: VectorCorpus) -> None:
    if not corpus.graph.has_vector_index("Doc", "summary"):
        corpus.graph.build_vector_index("Doc", "summary")


def _assert_search_quality(
    corpus: VectorCorpus,
    *,
    top_k: int = TOP_K,
    recall_floor: float = RECALL_FLOOR,
) -> float:
    """Assert result shape/membership and aggregate recall against exact scan."""
    hits = 0
    for query_id in corpus.query_ids:
        exact = corpus.exact_ids(query_id, top_k)
        oracle = corpus.oracle_ids(query_id, top_k)
        approximate = corpus.approximate_ids(query_id, top_k)

        assert exact == oracle
        assert len(exact) == top_k
        assert len(set(exact)) == top_k
        # A stored vector is its own exact cosine nearest neighbour.
        assert exact[0] == query_id

        assert len(approximate) == top_k
        assert len(set(approximate)) == top_k
        assert set(approximate) <= corpus.selected_ids
        hits += len(set(exact) & set(approximate))

    recall = hits / (len(corpus.query_ids) * top_k)
    # Match the existing Rust HNSW cosine/euclidean floor.  Aggregate recall
    # absorbs normal topology variance from concurrent index construction.
    assert recall > recall_floor, f"HNSW recall@{top_k} too low: {recall:.3f}"
    return recall


@pytest.mark.benchmark
def test_bench_hnsw_build(benchmark, vector_corpus):
    """Build only: nodes and embeddings are prepared outside the timer."""

    result = benchmark.pedantic(
        vector_corpus.graph.build_vector_index,
        args=("Doc", "summary"),
        rounds=1,
        iterations=1,
    )
    assert result["indexed"] == len(vector_corpus.selected_ids)
    assert result["metric"] == "cosine"
    recall = _assert_search_quality(vector_corpus)
    benchmark.extra_info["recall_at_10"] = recall
    benchmark.extra_info["dimension"] = DIMENSION
    benchmark.extra_info["vectors"] = len(vector_corpus.selected_ids)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("m", "ef_construction"),
    SELECTOR_WIDTHS,
    ids=[f"m-{m}-ef-{ef}" for m, ef in SELECTOR_WIDTHS],
)
def test_bench_hnsw_selector_widths(
    benchmark,
    selector_corpus: VectorCorpus,
    m: int,
    ef_construction: int,
) -> None:
    """Build matrix for neighbour selection at narrow/default/wide bounds."""
    result = benchmark.pedantic(
        selector_corpus.graph.build_vector_index,
        args=("Doc", "summary"),
        kwargs={"m": m, "ef_construction": ef_construction, "ef_search": 64},
        rounds=1,
        iterations=1,
    )
    assert result["indexed"] == len(selector_corpus.selected_ids)
    assert result["metric"] == "cosine"
    assert result["m"] == m
    recall = _assert_search_quality(selector_corpus)
    benchmark.extra_info["m"] = m
    benchmark.extra_info["ef_construction"] = ef_construction
    benchmark.extra_info["recall_at_10"] = recall
    benchmark.extra_info["vectors"] = len(selector_corpus.selected_ids)


@pytest.mark.benchmark
def test_bench_hnsw_search(benchmark, vector_corpus):
    """Approximate search only: index build and exact truth stay untimed."""
    _ensure_index(vector_corpus)
    recall = _assert_search_quality(vector_corpus)
    query_id = vector_corpus.query_ids[len(vector_corpus.query_ids) // 2]

    rows = benchmark.pedantic(
        vector_corpus.graph.select("Doc").vector_search,
        args=("summary", vector_corpus.query(query_id)),
        kwargs={"top_k": TOP_K},
        rounds=SEARCH_ROUNDS,
        iterations=1,
        warmup_rounds=SEARCH_WARMUP_ROUNDS,
    )
    result_ids = [int(row["id"]) for row in rows]
    assert len(result_ids) == TOP_K
    assert len(set(result_ids)) == TOP_K
    assert set(result_ids) <= vector_corpus.selected_ids
    benchmark.extra_info["recall_at_10"] = recall
    benchmark.extra_info["dimension"] = DIMENSION
    benchmark.extra_info["vectors"] = len(vector_corpus.selected_ids)


@pytest.mark.benchmark
@pytest.mark.parametrize("top_k", VISITED_WIDTH_TOP_K, ids=[f"top-{top_k}" for top_k in VISITED_WIDTH_TOP_K])
def test_bench_hnsw_fixed_topology_visited_width(
    benchmark,
    fixed_search_corpus: VectorCorpus,
    top_k: int,
) -> None:
    """Search one fixed topology while top-k drives ef/visited-set width."""
    recall = _assert_search_quality(fixed_search_corpus, top_k=top_k)
    query_id = fixed_search_corpus.query_ids[len(fixed_search_corpus.query_ids) // 2]
    expected = fixed_search_corpus.oracle_ids(query_id, top_k)
    search = fixed_search_corpus.graph.select("Doc").vector_search
    fixed_rows = search("summary", fixed_search_corpus.query(query_id), top_k=top_k)
    fixed_ids = [int(row["id"]) for row in fixed_rows]
    fixed_scores = [float(row["score"]) for row in fixed_rows]

    rows = benchmark.pedantic(
        search,
        args=("summary", fixed_search_corpus.query(query_id)),
        kwargs={"top_k": top_k},
        rounds=SEARCH_ROUNDS,
        iterations=1,
        warmup_rounds=SEARCH_WARMUP_ROUNDS,
    )
    actual = [int(row["id"]) for row in rows]
    actual_scores = [float(row["score"]) for row in rows]
    timed_recall = len(set(actual) & set(expected)) / top_k
    assert actual == fixed_ids
    assert actual_scores == fixed_scores
    assert len(actual) == top_k
    assert len(set(actual)) == top_k
    assert set(actual) <= fixed_search_corpus.selected_ids
    assert timed_recall > RECALL_FLOOR, f"fixed-topology HNSW recall@{top_k} too low: {timed_recall:.3f}"
    benchmark.extra_info["aggregate_recall"] = recall
    benchmark.extra_info["timed_query_recall"] = timed_recall
    benchmark.extra_info["effective_ef_floor"] = max(64, top_k * 4)
    benchmark.extra_info["vectors"] = len(fixed_search_corpus.selected_ids)


@pytest.mark.benchmark
def test_bench_cypher_fused_hnsw_whole_store(benchmark, vector_corpus):
    """Fused Cypher top-k over every node represented by the HNSW store."""
    _ensure_index(vector_corpus)
    query_id = vector_corpus.query_ids[len(vector_corpus.query_ids) // 2]
    expected = vector_corpus.oracle_ids(query_id)
    query = (
        "MATCH (n:Doc) RETURN n.id AS id, vector_score(n, 'summary_emb', $query) AS score ORDER BY score DESC LIMIT 10"
    )
    params = {"query": vector_corpus.query(query_id).tolist()}

    def search() -> list[dict]:
        return vector_corpus.graph.cypher(query, params=params).to_list()

    rows = benchmark.pedantic(
        search,
        rounds=SEARCH_ROUNDS,
        iterations=1,
        warmup_rounds=SEARCH_WARMUP_ROUNDS,
    )
    actual = [int(row["id"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    recall = len(set(actual) & set(expected)) / TOP_K

    assert len(actual) == TOP_K
    assert len(set(actual)) == TOP_K
    assert set(actual) <= vector_corpus.selected_ids
    assert actual[0] == expected[0]
    assert scores == sorted(scores, reverse=True)
    assert recall > RECALL_FLOOR, f"fused Cypher HNSW recall@{TOP_K} too low: {recall:.3f}"
    benchmark.extra_info["recall_at_10"] = recall
    benchmark.extra_info["dimension"] = DIMENSION
    benchmark.extra_info["vectors"] = len(vector_corpus.selected_ids)


@pytest.mark.benchmark
def test_bench_exact_vector_search(benchmark, vector_corpus):
    """Exact scan control for the same stored-vector query and selection."""
    query_id = vector_corpus.query_ids[len(vector_corpus.query_ids) // 2]
    expected = vector_corpus.oracle_ids(query_id)

    rows = benchmark.pedantic(
        vector_corpus.graph.select("Doc").vector_search,
        args=("summary", vector_corpus.query(query_id)),
        kwargs={"top_k": TOP_K, "exact": True},
        rounds=SEARCH_ROUNDS,
        iterations=1,
        warmup_rounds=SEARCH_WARMUP_ROUNDS,
    )
    actual = [int(row["id"]) for row in rows]
    assert actual == expected
    assert set(expected) <= vector_corpus.selected_ids
    benchmark.extra_info["dimension"] = DIMENSION
    benchmark.extra_info["vectors"] = len(vector_corpus.selected_ids)
    benchmark.extra_info["selection_shape"] = "single-type-whole-store-contiguous"


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("metric_mode", "metric_kwargs"),
    [
        pytest.param("omitted-stored", {}, id="omitted-stored-metric"),
        pytest.param("explicit-cosine", {"metric": "cosine"}, id="explicit-cosine"),
    ],
)
def test_bench_exact_vector_search_multi_type(
    benchmark,
    multi_type_exact_corpus: MultiTypeExactCorpus,
    metric_mode: str,
    metric_kwargs: dict[str, str],
) -> None:
    """Compare stored-metric resolution with explicit cosine on one exact scan."""
    rows = benchmark.pedantic(
        multi_type_exact_corpus.selection.vector_search,
        args=("summary", multi_type_exact_corpus.query),
        kwargs={"top_k": TOP_K, "exact": True, **metric_kwargs},
        rounds=SEARCH_ROUNDS,
        iterations=1,
        warmup_rounds=SEARCH_WARMUP_ROUNDS,
    )
    actual = tuple(int(row["id"]) for row in rows)
    actual_scores = tuple(float(row["score"]) for row in rows)
    assert actual == multi_type_exact_corpus.expected_ids
    assert actual_scores == pytest.approx(multi_type_exact_corpus.expected_scores, rel=1e-5, abs=1e-5)
    assert len(set(actual)) == TOP_K
    assert set(actual) <= multi_type_exact_corpus.selected_ids
    assert rows[0]["type"] == "Note"
    benchmark.extra_info["dimension"] = DIMENSION
    benchmark.extra_info["vectors"] = len(multi_type_exact_corpus.selected_ids)
    benchmark.extra_info["selection_shape"] = "multi-type-two-contiguous-stores"
    benchmark.extra_info["embedding_stores"] = 2
    benchmark.extra_info["stored_metrics"] = "cosine,cosine"
    benchmark.extra_info["metric_mode"] = metric_mode
    benchmark.extra_info["metric_argument"] = metric_kwargs.get("metric", "<omitted>")


@pytest.fixture(scope="module")
def whole_graph_multi_type_corpus() -> VectorCorpus:
    """One embedded type beside an un-embedded one, searched with no selection.

    The shape every realistic whole-graph search has, and the one no cell
    covered before: the corpus is a single embedded store, but the graph also
    holds nodes of another type, so the candidate set the search resolves is
    heterogeneous.
    """
    corpus = _build_corpus(FIXED_SEARCH_VECTORS, seed=20_260_822)
    corpus.graph.add_nodes(
        pd.DataFrame(
            {
                "id": np.arange(WHOLE_GRAPH_FOREIGN_NODES, dtype=np.int64),
                "title": [f"note{i}" for i in range(WHOLE_GRAPH_FOREIGN_NODES)],
                "body": [f"note text {i}" for i in range(WHOLE_GRAPH_FOREIGN_NODES)],
            }
        ),
        "Note",
        "id",
        "title",
    )
    corpus.graph.build_vector_index("Doc", "summary")
    assert corpus.graph.has_vector_index("Doc", "summary")
    return corpus


@pytest.mark.benchmark
def test_bench_whole_graph_multi_type_rides_the_index(benchmark, whole_graph_multi_type_corpus) -> None:
    """A whole-graph search on a multi-type graph must reach the index.

    Both arms are the identical whole-graph call, so the ratio measures the
    engine's routing decision and nothing else.  This cell is stated as a
    *speedup floor* rather than the usual "never slower" tolerance for a
    reason: when routing is broken both arms run the same exact scan and land
    at ~1.00x, which a never-slower assertion reads as a pass.  Only a floor
    below 1.0 can go red on the defect it exists for — pinned by mutation on
    2026-08-22 (release, see the commit that added it).
    """
    corpus = whole_graph_multi_type_corpus
    query_id = corpus.query_ids[len(corpus.query_ids) // 2]
    query = corpus.query(query_id)
    # No `.select(...)`: the never-narrowed selection *is* the whole graph.
    search = corpus.graph.vector_search

    exact_s, auto_s = _interleaved_mins(
        lambda: search("summary", query, top_k=TOP_K, exact=True),
        lambda: search("summary", query, top_k=TOP_K),
    )
    ratio = auto_s / exact_s

    rows = benchmark.pedantic(
        search,
        args=("summary", query),
        kwargs={"top_k": TOP_K},
        rounds=SEARCH_ROUNDS,
        iterations=1,
        warmup_rounds=SEARCH_WARMUP_ROUNDS,
    )
    actual = [int(row["id"]) for row in rows]
    assert len(actual) == TOP_K
    assert all(row["type"] == "Doc" for row in rows), "an un-embedded type leaked into the results"

    # The whole-graph exact answer is the single-type exact answer: the foreign
    # type is skipped, never scored, never dropped from the ranking.
    exact_whole = [int(row["id"]) for row in search("summary", query, top_k=TOP_K, exact=True)]
    assert exact_whole == corpus.exact_ids(query_id)
    recall = len(set(exact_whole) & set(actual)) / TOP_K
    assert recall > RECALL_FLOOR, f"whole-graph auto-selected recall@{TOP_K} too low: {recall:.3f}"

    assert ratio <= WHOLE_GRAPH_AUTO_MAX_RATIO, (
        f"whole-graph multi-type search is {ratio:.2f}x the exact scan at "
        f"n={FIXED_SEARCH_VECTORS} d={DIMENSION} beside "
        f"{WHOLE_GRAPH_FOREIGN_NODES} un-embedded nodes "
        f"(auto {auto_s * 1e3:.4f} ms vs exact {exact_s * 1e3:.4f} ms) — "
        f"the index is not being used for a whole-graph search"
    )
    benchmark.extra_info["auto_over_exact_ratio"] = ratio
    benchmark.extra_info["exact_min_ms"] = exact_s * 1e3
    benchmark.extra_info["recall_at_10"] = recall
    benchmark.extra_info["dimension"] = DIMENSION
    benchmark.extra_info["vectors"] = FIXED_SEARCH_VECTORS
    benchmark.extra_info["foreign_nodes"] = WHOLE_GRAPH_FOREIGN_NODES
    benchmark.extra_info["selection_shape"] = "whole-graph-one-store-among-two-types"


def _timed_once(call) -> float:
    start = time.perf_counter()
    call()
    return time.perf_counter() - start


def _interleaved_mins(first, second, *, passes: int = 3) -> tuple[float, float]:
    """Best-case wall clock for two calls, measured in alternating passes.

    Alternating cancels the monotonic warming drift that biases whichever arm
    a single sequential A-then-B comparison happens to run first — visible as a
    ~10% gap between two arms that are, after the size gate excludes the index,
    literally the same code path.
    """
    for _ in range(SEARCH_WARMUP_ROUNDS):
        first()
        second()
    best = [float("inf"), float("inf")]
    for _ in range(passes):
        for index, call in enumerate((first, second)):
            for _ in range(SEARCH_ROUNDS):
                best[index] = min(best[index], _timed_once(call))
    return best[0], best[1]


@pytest.fixture(scope="module")
def auto_gate_corpus() -> VectorCorpus:
    """Adversarial corpus sized inside the band ``HNSW_AUTO_MIN`` must exclude.

    Independent Gaussian vectors, deliberately unlike the low-rank corpus every
    other cell uses: this is the ANN worst case, where the index's fixed walk
    cost is largest relative to a scan this short.  Measured 2026-08-22 in
    release, the index loses 1.13-1.15x here across three runs, so the size gate
    is the only thing keeping the fast path correct *and* fast at this size.
    """
    rng = np.random.default_rng(20_260_822)
    vectors = np.asarray(rng.standard_normal((AUTO_GATE_VECTORS, AUTO_GATE_DIMENSION)), dtype=np.float32)
    graph = kglite.KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": np.arange(AUTO_GATE_VECTORS, dtype=np.int64),
                "title": [f"n{i}" for i in range(AUTO_GATE_VECTORS)],
                "summary": [f"text {i}" for i in range(AUTO_GATE_VECTORS)],
            }
        ),
        "Doc",
        "id",
        "title",
    )
    stored = graph.set_embeddings("Doc", "summary", dict(enumerate(vectors)), metric="cosine")
    assert stored["embeddings_stored"] == AUTO_GATE_VECTORS
    graph.build_vector_index("Doc", "summary")
    assert graph.has_vector_index("Doc", "summary")
    query_ids = tuple(int(i) for i in np.linspace(0, AUTO_GATE_VECTORS - 1, QUERY_COUNT, dtype=np.int64))
    return VectorCorpus(graph, vectors, frozenset(range(AUTO_GATE_VECTORS)), query_ids)


@pytest.mark.benchmark
def test_bench_auto_selection_never_loses_to_exact(benchmark, auto_gate_corpus: VectorCorpus) -> None:
    """Auto-selection must not be slower than the exact scan it replaces.

    An index exists, so ``exact=False`` is a real dispatch decision.  Both arms
    go through the identical Python call, so the ratio measures the engine's
    choice and nothing else.  When the size gate is set correctly this cell
    compares the exact path with itself and lands at ~1.00x; it goes red if the
    gate is lowered back into the band where the HNSW walk costs more than the
    scan.
    """
    query_id = auto_gate_corpus.query_ids[len(auto_gate_corpus.query_ids) // 2]
    query = auto_gate_corpus.query(query_id)
    search = auto_gate_corpus.graph.select("Doc").vector_search

    exact_s, auto_s = _interleaved_mins(
        lambda: search("summary", query, top_k=TOP_K, exact=True),
        lambda: search("summary", query, top_k=TOP_K),
    )
    ratio = auto_s / exact_s

    rows = benchmark.pedantic(
        search,
        args=("summary", query),
        kwargs={"top_k": TOP_K},
        rounds=SEARCH_ROUNDS,
        iterations=1,
        warmup_rounds=SEARCH_WARMUP_ROUNDS,
    )
    actual = [int(row["id"]) for row in rows]
    assert len(actual) == TOP_K
    assert set(actual) <= auto_gate_corpus.selected_ids

    # Recall against the engine's own exact scan, not the NumPy oracle: at
    # d=384 this corpus has cosine scores that tie to within f32 rounding, so
    # oracle-vs-engine ordering differences below 1e-7 are accumulation order,
    # not a wrong answer, and are not this cell's subject.
    hits = 0
    for probe_id in auto_gate_corpus.query_ids:
        exact_ids = set(auto_gate_corpus.exact_ids(probe_id))
        hits += len(exact_ids & set(auto_gate_corpus.approximate_ids(probe_id)))
    recall = hits / (len(auto_gate_corpus.query_ids) * TOP_K)
    assert recall > RECALL_FLOOR, f"auto-selected recall@{TOP_K} too low: {recall:.3f}"
    assert ratio <= AUTO_GATE_MAX_RATIO, (
        f"auto-selected search is {ratio:.2f}x the exact scan at "
        f"n={AUTO_GATE_VECTORS} d={AUTO_GATE_DIMENSION} "
        f"(auto {auto_s * 1e3:.4f} ms vs exact {exact_s * 1e3:.4f} ms) — "
        f"HNSW_AUTO_MIN is below the measured crossover"
    )
    benchmark.extra_info["auto_over_exact_ratio"] = ratio
    benchmark.extra_info["exact_min_ms"] = exact_s * 1e3
    benchmark.extra_info["recall_at_10"] = recall
    benchmark.extra_info["dimension"] = AUTO_GATE_DIMENSION
    benchmark.extra_info["vectors"] = AUTO_GATE_VECTORS
    benchmark.extra_info["distribution"] = "independent-gaussian"
