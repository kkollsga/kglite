"""Recall + latency report for the HNSW vector index vs the exact scan.

Not part of the gated benchmark baselines — ANN is a recall-vs-latency
trade-off, not a regression gate. Run on demand::

    python tests/benchmarks/bench_vector_index.py            # size x dim sweep
    python tests/benchmarks/bench_vector_index.py --legacy   # 3-cell recall report

The sweep answers one question: *is auto-selected HNSW ever slower than the
exact scan it replaces?*  The exact scan is a contiguous, SIMD-friendly fused
dot product; the HNSW walk is pointer chasing, so a crossover somewhere above
``HNSW_AUTO_MIN`` (400) is plausible a priori and has to be measured rather than
argued.  For each ``size x dim`` cell the sweep reports the exact-scan latency,
the auto-selected latency (same call, ``exact=False``), recall@10 of auto
against exact, and the index build time.

Methodology (CLAUDE.md "Performance protocol"):

* Sub-millisecond cells, so ``min`` over many rounds is the reported statistic;
  the median is printed alongside so a heavy tail (min < ~0.7x median) is
  visible instead of being silently reported as a rate.
* Every timed query is a *stored* vector — a realistic nearest-neighbour
  structure.  Vectors are a low-rank projection (8 latent features widened to
  ``d``), matching the corpus rationale in ``test_bench_vector_index.py``:
  independent Gaussian noise is an adversarial ANN corpus, not the product
  workload.  ``--dist gaussian`` measures that worst case explicitly.
* An unchanged-path CONTROL cell (a fixed 10k x 64 exact scan) is measured at
  sweep start and again at sweep end.  If the control moves, the instrument
  moved and the sweep's verdict is not trustworthy.

``vector_search`` takes the **text column** ("summary"), not the store name
("summary_emb").  Passing the store name derives ``summary_emb_emb`` and raises.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import random
import statistics
import time

import numpy as np
import pandas as pd

import kglite

LATENT_DIMENSION = 8
TOP_K = 10
RECALL_QUERIES = 20
TIMED_QUERIES = 3
SWEEP_SIZES = (1_000, 5_000, 10_000, 20_000, 50_000)
SWEEP_DIMENSIONS = (64, 128, 384)
CONTROL_SIZE = 10_000
CONTROL_DIMENSION = 64
#: A cell whose index build exceeds this budget disables every larger cell at
#: the same dimension — the sweep is a latency probe, not a build benchmark.
BUILD_BUDGET_S = 300.0
#: Auto-HNSW is "slower than exact" only past this ratio; below it the two paths
#: are the same speed within the noise this harness can resolve.
SLOWER_RATIO = 1.1
RECALL_FLOOR = 0.90


def _rounds_for(n: int) -> int:
    if n <= 10_000:
        return 200
    if n <= 20_000:
        return 100
    return 50


def _vectors(n: int, d: int, *, dist: str, seed: int) -> np.ndarray:
    """Deterministic corpus: low-rank projection (default) or raw Gaussian."""
    rng = np.random.default_rng(seed)
    if dist == "gaussian":
        return np.asarray(rng.standard_normal((n, d)), dtype=np.float32)
    latent = rng.standard_normal((n, LATENT_DIMENSION), dtype=np.float32)
    projection = rng.standard_normal((LATENT_DIMENSION, d), dtype=np.float32)
    return np.asarray(latent @ projection, dtype=np.float32)


def _graph_with(vectors: np.ndarray) -> kglite.KnowledgeGraph:
    n = len(vectors)
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
    assert report["embeddings_stored"] == n, report
    return graph


def _stats_ms(fn, rounds: int, warmup: int = 20) -> tuple[float, float]:
    """Return (min, median) wall-clock milliseconds over ``rounds`` calls."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return min(samples), statistics.median(samples)


def _timed_search(graph, vectors: np.ndarray, query_ids, *, exact: bool, rounds: int) -> tuple[float, float]:
    """Median-of-per-query mins, so one lucky query cannot carry the cell."""
    mins, medians = [], []
    for query_id in query_ids:
        query = vectors[query_id]
        lo, mid = _stats_ms(
            lambda q=query: graph.select("Doc").vector_search("summary", q, top_k=TOP_K, exact=exact),
            rounds,
        )
        mins.append(lo)
        medians.append(mid)
    return statistics.median(mins), statistics.median(medians)


@dataclass
class Cell:
    n: int
    d: int
    exact_ms: float
    exact_median_ms: float
    auto_ms: float
    auto_median_ms: float
    recall: float
    build_s: float

    @property
    def ratio(self) -> float:
        """auto / exact — >1 means the index lost to the scan it replaced."""
        return self.auto_ms / self.exact_ms if self.exact_ms else float("nan")

    def row(self) -> str:
        tail = "  <tail>" if self.auto_ms < 0.7 * self.auto_median_ms else ""
        verdict = "SLOWER" if self.ratio > SLOWER_RATIO else "ok"
        return (
            f"{self.n:>7} {self.d:>5} | {self.exact_ms:9.4f} {self.exact_median_ms:9.4f} | "
            f"{self.auto_ms:9.4f} {self.auto_median_ms:9.4f} | {1 / self.ratio:6.2f}x | "
            f"{self.recall:.3f} | {self.build_s:7.1f}s | {verdict}{tail}"
        )


def measure_cell(n: int, d: int, *, dist: str, seed: int) -> Cell:
    vectors = _vectors(n, d, dist=dist, seed=seed)
    graph = _graph_with(vectors)
    recall_ids = [int(i) for i in np.linspace(0, n - 1, RECALL_QUERIES, dtype=np.int64)]
    timed_ids = [int(i) for i in np.linspace(0, n - 1, TIMED_QUERIES + 2, dtype=np.int64)[1:-1]]
    rounds = _rounds_for(n)

    t0 = time.perf_counter()
    build = graph.build_vector_index("Doc", "summary")
    build_s = time.perf_counter() - t0
    assert build["indexed"] == n, build

    # Both paths measured on the same post-build graph: the exact scan is the
    # path auto-selection would have taken, in the state it would have taken it.
    exact_ms, exact_median = _timed_search(graph, vectors, timed_ids, exact=True, rounds=rounds)
    auto_ms, auto_median = _timed_search(graph, vectors, timed_ids, exact=False, rounds=rounds)

    hits = 0
    for query_id in recall_ids:
        query = vectors[query_id]
        truth = {r["id"] for r in graph.select("Doc").vector_search("summary", query, top_k=TOP_K, exact=True)}
        got = {r["id"] for r in graph.select("Doc").vector_search("summary", query, top_k=TOP_K)}
        hits += len(truth & got)
    recall = hits / (len(recall_ids) * TOP_K)

    return Cell(n, d, exact_ms, exact_median, auto_ms, auto_median, recall, build_s)


def _control(graph, vectors: np.ndarray) -> float:
    """Unchanged-path drift meter: fixed exact scan, no index involved."""
    query_ids = [int(i) for i in np.linspace(0, len(vectors) - 1, TIMED_QUERIES + 2, dtype=np.int64)[1:-1]]
    return _timed_search(graph, vectors, query_ids, exact=True, rounds=_rounds_for(len(vectors)))[0]


def sweep(*, dist: str, sizes, dims, seed: int = 20_260_822) -> None:
    control_vectors = _vectors(CONTROL_SIZE, CONTROL_DIMENSION, dist="structured", seed=7)
    control_graph = _graph_with(control_vectors)
    control_start = _control(control_graph, control_vectors)
    print(f"control (exact {CONTROL_SIZE}x{CONTROL_DIMENSION}, no index) start: {control_start:.4f} ms")
    print(
        f"\n{'n':>7} {'dim':>5} | {'exact min':>9} {'exact med':>9} | "
        f"{'auto min':>9} {'auto med':>9} | {'speedup':>7} | recall | {'build':>8} | verdict"
    )

    cells: list[Cell] = []
    skip_over: dict[int, int] = {}
    for d in dims:
        for n in sizes:
            if d in skip_over and n > skip_over[d]:
                print(f"{n:>7} {d:>5} | skipped (build budget exceeded at n={skip_over[d]})")
                continue
            cell = measure_cell(n, d, dist=dist, seed=seed + n + d)
            cells.append(cell)
            print(cell.row(), flush=True)
            if cell.build_s > BUILD_BUDGET_S:
                skip_over[d] = n

    control_end = _control(control_graph, control_vectors)
    drift = (control_end / control_start - 1.0) * 100.0
    print(f"\ncontrol end: {control_end:.4f} ms  (drift {drift:+.1f}%)")

    losers = [c for c in cells if c.ratio > SLOWER_RATIO and c.recall >= RECALL_FLOOR]
    if losers:
        print(f"\nVERDICT: auto-HNSW is >{SLOWER_RATIO}x slower than exact at {len(losers)} cell(s):")
        for cell in losers:
            print(f"  n={cell.n} d={cell.d}: {cell.ratio:.2f}x slower (recall {cell.recall:.3f})")
    else:
        worst = max(cells, key=lambda c: c.ratio)
        print(
            f"\nVERDICT: auto-HNSW never >{SLOWER_RATIO}x slower than exact "
            f"(worst cell n={worst.n} d={worst.d} at {worst.ratio:.2f}x)."
        )
    low_recall = [c for c in cells if c.recall < RECALL_FLOOR]
    if low_recall:
        print(f"recall below {RECALL_FLOOR} at: " + ", ".join(f"n={c.n} d={c.d} ({c.recall:.3f})" for c in low_recall))


def _legacy_build(n: int, d: int, seed: int = 1):
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


def report(n: int, d: int, k: int = 10, n_queries: int = 30) -> None:
    g, emb = _legacy_build(n, d)
    queries = [emb[i] for i in range(0, min(n, n_queries * 50), 50)][:n_queries]

    def exact_one(q):
        return g.select("Doc").vector_search("summary", q, top_k=k, exact=True)

    def hnsw_one(q):
        return g.select("Doc").vector_search("summary", q, top_k=k)

    exact_ms = _stats_ms(lambda: exact_one(queries[0]), rounds=25, warmup=3)[0]
    t0 = time.perf_counter()
    g.build_vector_index("Doc", "summary")
    build_s = time.perf_counter() - t0
    hnsw_ms = _stats_ms(lambda: hnsw_one(queries[0]), rounds=25, warmup=3)[0]

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", action="store_true", help="run the original 3-cell recall report")
    parser.add_argument("--dist", choices=("structured", "gaussian"), default="structured")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SWEEP_SIZES))
    parser.add_argument("--dims", type=int, nargs="+", default=list(SWEEP_DIMENSIONS))
    args = parser.parse_args()

    if args.legacy:
        print("HNSW vector index — recall + latency (stored-vector queries)")
        for n, d in [(10_000, 128), (50_000, 128), (100_000, 256)]:
            report(n, d)
        return

    print(f"HNSW auto-selection vs exact scan — size x dim sweep ({args.dist} vectors, top_k={TOP_K}, cosine)")
    sweep(dist=args.dist, sizes=args.sizes, dims=args.dims)


if __name__ == "__main__":
    main()
