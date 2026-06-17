"""Recall + latency report for the HNSW vector index vs the exact scan.

Not part of the gated benchmark baselines — ANN is a recall-vs-latency
trade-off, not a regression gate. Run on demand:

    python tests/benchmarks/bench_vector_index.py

Queries are *stored* vectors (a realistic nearest-neighbour structure, unlike
random vectors which are a near-worst case for any ANN in high dimensions).
"""

from __future__ import annotations

import random
import time

import pandas as pd

import kglite


def _build(n: int, d: int, seed: int = 1):
    rng = random.Random(seed)
    rows = {
        "id": list(range(n)),
        "title": [f"n{i}" for i in range(n)],
        "summary": [f"t{i}" for i in range(n)],
    }
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame(rows), "Doc", "id", "title")
    emb = {i: [rng.gauss(0, 1) for _ in range(d)] for i in range(n)}
    g.set_embeddings("Doc", "summary", emb, metric="cosine")
    return g, emb


def _min_ms(fn, rounds: int = 25, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def report(n: int, d: int, k: int = 10, n_queries: int = 30) -> None:
    g, emb = _build(n, d)
    queries = [emb[i] for i in range(0, min(n, n_queries * 50), 50)][:n_queries]

    def exact_one(q):
        return g.select("Doc").vector_search("summary", q, top_k=k, exact=True)

    def hnsw_one(q):
        return g.select("Doc").vector_search("summary", q, top_k=k)

    exact_ms = _min_ms(lambda: exact_one(queries[0]))
    t0 = time.perf_counter()
    g.build_vector_index("Doc", "summary")
    build_s = time.perf_counter() - t0
    hnsw_ms = _min_ms(lambda: hnsw_one(queries[0]))

    # recall@k over the query set
    hits = 0
    for q in queries:
        truth = {r["id"] for r in exact_one(q)}
        got = {r["id"] for r in hnsw_one(q)}
        hits += len(truth & got)
    recall = hits / (len(queries) * k)

    speedup = exact_ms / hnsw_ms if hnsw_ms else float("inf")
    print(
        f"n={n:>7} d={d:>4} k={k}: "
        f"exact {exact_ms:7.3f} ms | hnsw {hnsw_ms:7.3f} ms | "
        f"{speedup:5.1f}x | recall@{k} {recall:.3f} | build {build_s:5.1f}s"
    )


if __name__ == "__main__":
    print("HNSW vector index — recall + latency (stored-vector queries)")
    for n, d in [(10_000, 128), (50_000, 128), (100_000, 256)]:
        report(n, d)
