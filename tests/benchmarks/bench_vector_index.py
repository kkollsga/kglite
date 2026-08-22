"""Recall + latency report for the HNSW vector index vs the exact scan.

Not part of the gated benchmark baselines — ANN is a recall-vs-latency
trade-off, not a regression gate. Run on demand::

    python tests/benchmarks/bench_vector_index.py                  # size x dim sweep
    python tests/benchmarks/bench_vector_index.py --recall-sweep   # ef_search sweep
    python tests/benchmarks/bench_vector_index.py --legacy         # 3-cell recall report

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

The ``--recall-sweep`` mode answers a different question: *what does the default
``ef_search`` cost in recall, and what does raising it buy?*  It walks
``ef_search x size x dim`` on both corpora, reporting recall@10 against the exact
scan and the auto-path latency at every cell.  ``ef_search`` is a build-time
default stored in the index, so each ef value is a fresh ``build_vector_index``
call; index construction inserts concurrently (rayon), so two builds of the same
corpus are not bit-identical graphs and their recall differs slightly.  That
build-to-build spread is the sweep's own noise floor for recall, so designated
cells are built twice and both recalls are reported (``VARIANCE_CELLS``) — an
ef effect smaller than that spread is not an effect.
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

#: --recall-sweep axes.  ef_search 64 is the shipped default (HnswParams::default).
RECALL_EF_VALUES = (64, 128, 256)
RECALL_SWEEP_SIZES = (20_000, 50_000, 100_000)
RECALL_SWEEP_DIMENSIONS = (128, 384)
RECALL_SWEEP_DISTS = ("structured", "gaussian")
#: ``(dist, n, d, ef)`` cells built twice so build-to-build recall spread is
#: measured rather than assumed.  One is the stop-rule cell (the largest, widest
#: structured cell at the default ef); one is the worst-recall adversarial cell.
VARIANCE_CELLS = (
    ("structured", 100_000, 384, 64),
    ("gaussian", 50_000, 384, 64),
)


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


@dataclass
class RecallCell:
    """One ``(corpus, n, d, ef_search)`` point of the recall sweep."""

    dist: str
    n: int
    d: int
    ef: int
    recall: float
    auto_ms: float
    auto_median_ms: float
    exact_ms: float
    build_s: float
    #: Recall of an independent second build of the same corpus at the same ef —
    #: the build-to-build spread, measured only at ``VARIANCE_CELLS``.
    recall_rebuild: float | None = None

    @property
    def speedup(self) -> float:
        return self.exact_ms / self.auto_ms if self.auto_ms else float("nan")

    def row(self) -> str:
        tail = "  <tail>" if self.auto_ms < 0.7 * self.auto_median_ms else ""
        spread = ""
        if self.recall_rebuild is not None:
            spread = f"  rebuild {self.recall_rebuild:.3f} (delta {self.recall_rebuild - self.recall:+.3f})"
        floor = "" if self.recall >= RECALL_FLOOR else "  BELOW-FLOOR"
        return (
            f"{self.n:>7} {self.d:>5} {self.ef:>4} | {self.recall:.3f} | "
            f"{self.auto_ms:9.4f} {self.auto_median_ms:9.4f} | {self.exact_ms:9.4f} | "
            f"{self.speedup:6.2f}x | {self.build_s:7.1f}s{floor}{tail}{spread}"
        )


def _exact_truth(graph, vectors: np.ndarray, recall_ids) -> list[set]:
    """Exact top-``TOP_K`` id sets — computed once per corpus, reused per ef."""
    return [
        {r["id"] for r in graph.select("Doc").vector_search("summary", vectors[i], top_k=TOP_K, exact=True)}
        for i in recall_ids
    ]


def _recall_against(graph, vectors: np.ndarray, recall_ids, truth: list[set]) -> float:
    hits = 0
    for query_id, want in zip(recall_ids, truth):
        got = {r["id"] for r in graph.select("Doc").vector_search("summary", vectors[query_id], top_k=TOP_K)}
        hits += len(want & got)
    return hits / (len(recall_ids) * TOP_K)


def _build_index(graph, n: int, ef: int) -> float:
    """(Re)build the index at ``ef_search=ef``; returns build seconds."""
    graph.drop_vector_index("Doc", "summary")
    t0 = time.perf_counter()
    report = graph.build_vector_index("Doc", "summary", ef_search=ef)
    build_s = time.perf_counter() - t0
    assert report["indexed"] == n, report
    return build_s


def measure_recall_cells(n: int, d: int, *, dist: str, efs, seed: int) -> list[RecallCell]:
    """Every ef value measured against ONE corpus and ONE exact-truth set."""
    vectors = _vectors(n, d, dist=dist, seed=seed)
    graph = _graph_with(vectors)
    recall_ids = [int(i) for i in np.linspace(0, n - 1, RECALL_QUERIES, dtype=np.int64)]
    timed_ids = [int(i) for i in np.linspace(0, n - 1, TIMED_QUERIES + 2, dtype=np.int64)[1:-1]]
    rounds = _rounds_for(n)

    truth = _exact_truth(graph, vectors, recall_ids)
    exact_ms, _ = _timed_search(graph, vectors, timed_ids, exact=True, rounds=rounds)

    cells = []
    for ef in efs:
        build_s = _build_index(graph, n, ef)
        recall = _recall_against(graph, vectors, recall_ids, truth)
        auto_ms, auto_median = _timed_search(graph, vectors, timed_ids, exact=False, rounds=rounds)
        cell = RecallCell(dist, n, d, ef, recall, auto_ms, auto_median, exact_ms, build_s)
        if (dist, n, d, ef) in VARIANCE_CELLS:
            _build_index(graph, n, ef)
            cell.recall_rebuild = _recall_against(graph, vectors, recall_ids, truth)
        cells.append(cell)
    return cells


def recall_sweep(*, efs, sizes, dims, dists, seed: int = 20_260_823) -> None:
    control_vectors = _vectors(CONTROL_SIZE, CONTROL_DIMENSION, dist="structured", seed=7)
    control_graph = _graph_with(control_vectors)
    control_start = _control(control_graph, control_vectors)
    print(f"control (exact {CONTROL_SIZE}x{CONTROL_DIMENSION}, no index) start: {control_start:.4f} ms")

    cells: list[RecallCell] = []
    for dist in dists:
        label = "low-rank / clustered (realistic)" if dist == "structured" else "independent Gaussian (adversarial)"
        print(f"\n=== {dist} corpus — {label} ===")
        print(
            f"{'n':>7} {'dim':>5} {'ef':>4} | recall | {'auto min':>9} {'auto med':>9} | "
            f"{'exact min':>9} | {'speedup':>7} | {'build':>8}"
        )
        for d in dims:
            for n in sizes:
                for cell in measure_recall_cells(n, d, dist=dist, efs=efs, seed=seed + n + d):
                    cells.append(cell)
                    print(cell.row(), flush=True)

    control_end = _control(control_graph, control_vectors)
    drift = (control_end / control_start - 1.0) * 100.0
    print(f"\ncontrol end: {control_end:.4f} ms  (drift {drift:+.1f}%)")

    default_ef = min(efs)
    structured = [c for c in cells if c.dist == "structured" and c.ef == default_ef]
    if structured:
        worst = min(structured, key=lambda c: c.recall)
        holds = worst.recall >= RECALL_FLOOR
        print(
            f"\nSTOP RULE (clustered corpus, ef_search={default_ef}, floor {RECALL_FLOOR}): "
            f"{'HOLDS' if holds else 'BREACHED'} — worst cell n={worst.n} d={worst.d} "
            f"recall {worst.recall:.3f}"
        )
    for cell in cells:
        if cell.recall_rebuild is not None:
            print(
                f"build-to-build spread at {cell.dist} n={cell.n} d={cell.d} ef={cell.ef}: "
                f"{cell.recall:.3f} vs {cell.recall_rebuild:.3f} "
                f"(delta {cell.recall_rebuild - cell.recall:+.3f})"
            )


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
    parser.add_argument(
        "--recall-sweep",
        action="store_true",
        help="sweep recall@10 and latency across ef_search x size x dim on both corpora",
    )
    parser.add_argument("--dist", choices=("structured", "gaussian"), default="structured")
    parser.add_argument("--efs", type=int, nargs="+", default=list(RECALL_EF_VALUES))
    parser.add_argument("--sizes", type=int, nargs="+", default=None)
    parser.add_argument("--dims", type=int, nargs="+", default=None)
    args = parser.parse_args()

    if args.legacy:
        print("HNSW vector index — recall + latency (stored-vector queries)")
        for n, d in [(10_000, 128), (50_000, 128), (100_000, 256)]:
            report(n, d)
        return

    if args.recall_sweep:
        sizes = args.sizes or list(RECALL_SWEEP_SIZES)
        dims = args.dims or list(RECALL_SWEEP_DIMENSIONS)
        print(
            f"HNSW recall@{TOP_K} vs ef_search — ef x size x dim sweep "
            f"(both corpora, top_k={TOP_K}, cosine, m=16, ef_construction=200)"
        )
        recall_sweep(efs=args.efs, sizes=sizes, dims=dims, dists=list(RECALL_SWEEP_DISTS))
        return

    sizes = args.sizes or list(SWEEP_SIZES)
    dims = args.dims or list(SWEEP_DIMENSIONS)
    print(f"HNSW auto-selection vs exact scan — size x dim sweep ({args.dist} vectors, top_k={TOP_K}, cosine)")
    sweep(dist=args.dist, sizes=sizes, dims=dims)


if __name__ == "__main__":
    main()
