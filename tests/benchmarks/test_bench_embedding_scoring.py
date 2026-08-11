"""Perf gate for the embedding ingest + query surface (the Java-embeddings program).

Two cells matter for the program's perf gate:

* **Ingest throughput** — ``set_embeddings`` over a 100k x 384 corpus (the
  packed float path). Each round ingests into a *fresh* graph, so this is a
  once-per-event cost (Performance protocol item 4a) and is reported by the
  **mean** of first-writes, not ``min``.
* **Scoring latency** — a whole-store exact scan scoring every node with a
  **raw query vector** through ``vector_score`` and ``text_score``. The
  docs-blind demo measured a hand-rolled 64-dim dot product at ~1.66 s over
  95k notes; the native path scores 100k x 384 and should be far below that.
  Deterministic, tens-of-ms, so reported by ``min`` (protocol item 4).

Run explicitly (release build required — debug timings are invalid)::

    make dev-release  # or: uv run --no-sync maturin develop --release
    pytest tests/benchmarks/test_bench_embedding_scoring.py -m benchmark -v -s
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import kglite

DIMENSION = 384
INGEST_N = 100_000
SCORING_N = 100_000
LATENT_DIMENSION = 16
TOP_K = 10
SEARCH_ROUNDS = 60
SEARCH_WARMUP_ROUNDS = 10
INGEST_ROUNDS = 5


def _vectors(n: int, seed: int) -> np.ndarray:
    """Deterministic low-rank float32 corpus (dense-embedding manifold shape)."""
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n, LATENT_DIMENSION), dtype=np.float32)
    projection = rng.standard_normal((LATENT_DIMENSION, DIMENSION), dtype=np.float32)
    return np.asarray(latent @ projection, dtype=np.float32)


def _nodes_frame(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": np.arange(n, dtype=np.int64),
            "title": [f"n{i}" for i in range(n)],
            "summary": [f"text {i}" for i in range(n)],
        }
    )


@pytest.fixture(scope="module")
def scoring_corpus():
    vectors = _vectors(SCORING_N, seed=20_260_811)
    graph = kglite.KnowledgeGraph()
    graph.add_nodes(_nodes_frame(SCORING_N), "Doc", "id", "title")
    report = graph.set_embeddings("Doc", "summary", dict(enumerate(vectors)), metric="cosine")
    assert report == {"embeddings_stored": SCORING_N, "dimension": DIMENSION, "skipped": 0}
    query = vectors[SCORING_N // 2].tolist()
    return graph, query


@pytest.mark.benchmark
def test_bench_set_embeddings_ingest_100k_384(benchmark):
    """Ingest throughput: set_embeddings over 100k x 384, fresh graph per round."""
    vectors = _vectors(INGEST_N, seed=20_260_812)
    mapping = dict(enumerate(vectors))
    frame = _nodes_frame(INGEST_N)

    def setup():
        graph = kglite.KnowledgeGraph()
        graph.add_nodes(frame, "Doc", "id", "title")
        return (graph,), {}

    def ingest(graph):
        return graph.set_embeddings("Doc", "summary", mapping, metric="cosine")

    report = benchmark.pedantic(ingest, setup=setup, rounds=INGEST_ROUNDS, iterations=1)
    assert report == {"embeddings_stored": INGEST_N, "dimension": DIMENSION, "skipped": 0}
    mean_s = benchmark.stats.stats.mean
    benchmark.extra_info["statistic"] = "mean-of-first-writes"
    benchmark.extra_info["dimension"] = DIMENSION
    benchmark.extra_info["vectors"] = INGEST_N
    benchmark.extra_info["throughput_vectors_per_s"] = INGEST_N / mean_s


@pytest.mark.benchmark
def test_bench_vector_score_scan_100k_384(benchmark, scoring_corpus):
    """Whole-store exact scan scoring every node with a raw query vector."""
    graph, query = scoring_corpus
    q = "MATCH (n:Doc) RETURN n.id AS id, vector_score(n, 'summary_emb', $q) AS score ORDER BY score DESC LIMIT 10"
    params = {"q": query}

    def scan():
        return graph.cypher(q, params=params).to_list()

    rows = benchmark.pedantic(scan, rounds=SEARCH_ROUNDS, iterations=1, warmup_rounds=SEARCH_WARMUP_ROUNDS)
    assert len(rows) == TOP_K
    assert rows[0]["id"] == SCORING_N // 2  # a stored vector is its own nearest neighbour
    scores = [float(r["score"]) for r in rows]
    assert scores == sorted(scores, reverse=True)
    benchmark.extra_info["statistic"] = "min"
    benchmark.extra_info["dimension"] = DIMENSION
    benchmark.extra_info["vectors"] = SCORING_N
    benchmark.extra_info["scoring_fn"] = "vector_score"


@pytest.mark.benchmark
def test_bench_text_score_scan_100k_384(benchmark, scoring_corpus):
    """Same scan through text_score with a raw vector (Phase 1's rewrite path)."""
    graph, query = scoring_corpus
    q = "MATCH (n:Doc) RETURN n.id AS id, text_score(n, 'summary', $q) AS score ORDER BY score DESC LIMIT 10"
    params = {"q": query}

    def scan():
        return graph.cypher(q, params=params).to_list()

    rows = benchmark.pedantic(scan, rounds=SEARCH_ROUNDS, iterations=1, warmup_rounds=SEARCH_WARMUP_ROUNDS)
    assert len(rows) == TOP_K
    assert rows[0]["id"] == SCORING_N // 2
    benchmark.extra_info["statistic"] = "min"
    benchmark.extra_info["dimension"] = DIMENSION
    benchmark.extra_info["vectors"] = SCORING_N
    benchmark.extra_info["scoring_fn"] = "text_score"
