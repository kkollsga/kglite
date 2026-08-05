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


@dataclass
class VectorCorpus:
    """Graph plus independent query/truth data shared by one size's cells."""

    graph: kglite.KnowledgeGraph
    vectors: np.ndarray
    selected_ids: frozenset[int]
    query_ids: tuple[int, ...]
    _exact: dict[int, list[int]] = field(default_factory=dict)
    _oracle: dict[int, list[int]] = field(default_factory=dict)
    _norms: np.ndarray | None = None

    def query(self, query_id: int) -> np.ndarray:
        return self.vectors[query_id]

    def exact_ids(self, query_id: int) -> list[int]:
        if query_id not in self._exact:
            rows = self.graph.select("Doc").vector_search(
                "summary",
                self.query(query_id),
                top_k=TOP_K,
                exact=True,
            )
            self._exact[query_id] = [int(row["id"]) for row in rows]
        return self._exact[query_id]

    def oracle_ids(self, query_id: int) -> list[int]:
        """Independent NumPy brute-force cosine top-k, including order."""
        if query_id not in self._oracle:
            if self._norms is None:
                self._norms = np.linalg.norm(self.vectors, axis=1)
            query = self.query(query_id)
            scores = (self.vectors @ query) / (self._norms * np.linalg.norm(query))
            # Random projected vectors have no exact score ties.  Stable sort
            # nevertheless makes the oracle's tie behavior explicit.
            order = np.argsort(-scores, kind="stable")[:TOP_K]
            self._oracle[query_id] = [int(node_id) for node_id in order]
        return self._oracle[query_id]

    def approximate_ids(self, query_id: int) -> list[int]:
        rows = self.graph.select("Doc").vector_search(
            "summary",
            self.query(query_id),
            top_k=TOP_K,
        )
        return [int(row["id"]) for row in rows]


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


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(10_000, id="10k-x-128"),
        pytest.param(50_000, id="50k-x-128", marks=pytest.mark.slow),
    ],
)
def vector_corpus(request) -> VectorCorpus:
    return _build_corpus(request.param)


def _ensure_index(corpus: VectorCorpus) -> None:
    if not corpus.graph.has_vector_index("Doc", "summary"):
        corpus.graph.build_vector_index("Doc", "summary")


def _assert_search_quality(corpus: VectorCorpus) -> float:
    """Assert result shape/membership and aggregate recall against exact scan."""
    hits = 0
    for query_id in corpus.query_ids:
        exact = corpus.exact_ids(query_id)
        oracle = corpus.oracle_ids(query_id)
        approximate = corpus.approximate_ids(query_id)

        assert exact == oracle
        assert len(exact) == TOP_K
        assert len(set(exact)) == TOP_K
        # A stored vector is its own exact cosine nearest neighbour.
        assert exact[0] == query_id

        assert len(approximate) == TOP_K
        assert len(set(approximate)) == TOP_K
        assert set(approximate) <= corpus.selected_ids
        hits += len(set(exact) & set(approximate))

    recall = hits / (len(corpus.query_ids) * TOP_K)
    # Match the existing Rust HNSW cosine/euclidean floor.  Aggregate recall
    # absorbs normal topology variance from concurrent index construction.
    assert recall > RECALL_FLOOR, f"HNSW recall@{TOP_K} too low: {recall:.3f}"
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
