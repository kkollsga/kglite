"""Benchmarks for memory management: spill, unspill, vacuum, and save.

Compares fully heap-resident columns against columns spilled to disk under
`set_memory_limit`. There is no shape to switch: properties live in per-type
columns from the first node, and the only axis here is where those columns'
bytes sit.
Run with: pytest tests/benchmarks/test_bench_memory.py -m benchmark -v -s
"""

import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from kglite import KnowledgeGraph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_graph(n=5000):
    """Build a graph with n nodes and multiple property types."""
    graph = KnowledgeGraph()
    nodes = pd.DataFrame(
        {
            "nid": list(range(n)),
            "name": [f"Node_{i}" for i in range(n)],
            "value": [float(i) for i in range(n)],
            "category": [f"cat_{i % 50}" for i in range(n)],
            "score": [float(i * 0.1) for i in range(n)],
            "flag": [i % 2 == 0 for i in range(n)],
        }
    )
    graph.add_nodes(nodes, "Item", "nid", "name")

    edges = pd.DataFrame(
        {
            "from_id": [i % n for i in range(n * 2)],
            "to_id": [(i * 7 + 13) % n for i in range(n * 2)],
            "weight": [float(i % 100) for i in range(n * 2)],
        }
    )
    graph.add_connections(edges, "LINKS", "Item", "from_id", "Item", "to_id", columns=["weight"])
    return graph


@pytest.fixture
def graph_5k():
    """5000-node graph, columns heap-resident.

    Was two fixtures — `graph_5k` ("compact storage") and `graph_5k`
    ("columnar, heap-backed") — with identical bodies, because the shapes they
    named were never distinguishable by anything a fixture could do. One now.
    """
    return _build_graph(5000)


@pytest.fixture
def bench_graph_1k():
    """1000-node graph — the shape the retired tracked cell measured."""
    return _build_graph(1000)


@pytest.fixture
def graph_5k_spilled(tmp_path):
    """5000-node graph (columnar, spilled to disk).

    A `save()` is the enforcement point that applies the limit; there is no
    regime switch to call.
    """
    g = _build_graph(5000)
    g.set_memory_limit(1024)  # force full spill
    g.save(str(tmp_path / "spill-trigger.kgl"))
    return g


# ---------------------------------------------------------------------------
# Unspill
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_unspill_5k(benchmark, graph_5k_spilled, tmp_path):
    """Time to move spilled data back to heap (5000 nodes).

    The re-spill runs in `setup`, not in the measured body: it is a `save()`
    now, so folding it into the timed call would report file I/O as unspill
    cost.
    """

    def setup():
        graph_5k_spilled.set_memory_limit(1024)
        graph_5k_spilled.save(str(tmp_path / "respill.kgl"))

    benchmark.pedantic(graph_5k_spilled.unspill, setup=setup, rounds=20, iterations=1)


# ---------------------------------------------------------------------------
# Query performance: heap vs spilled
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_query_where_heap_5k(benchmark, graph_5k):
    """Filtered query on heap-resident columns (5000 nodes)."""
    benchmark(
        graph_5k.cypher,
        "MATCH (n:Item) WHERE n.value > 4000 RETURN n.title, n.value",
    )


@pytest.mark.benchmark
def test_bench_query_where_spilled_5k(benchmark, graph_5k_spilled):
    """Filtered query on spilled columns (5000 nodes)."""
    benchmark(
        graph_5k_spilled.cypher,
        "MATCH (n:Item) WHERE n.value > 4000 RETURN n.title, n.value",
    )


@pytest.mark.benchmark
def test_bench_query_match_heap_5k(benchmark, graph_5k):
    """Simple MATCH on heap-resident columns (5000 nodes)."""
    benchmark(
        graph_5k.cypher,
        "MATCH (n:Item) RETURN n.title, n.value LIMIT 100",
    )


@pytest.mark.benchmark
def test_bench_query_match_spilled_5k(benchmark, graph_5k_spilled):
    """Simple MATCH on spilled columns (5000 nodes)."""
    benchmark(
        graph_5k_spilled.cypher,
        "MATCH (n:Item) RETURN n.title, n.value LIMIT 100",
    )


@pytest.mark.benchmark
def test_bench_query_aggregation_heap_5k(benchmark, graph_5k):
    """Aggregation on heap-resident columns."""
    benchmark(
        graph_5k.cypher,
        "MATCH (n:Item) RETURN count(n) AS cnt, avg(n.value) AS avg_val",
    )


@pytest.mark.benchmark
def test_bench_query_aggregation_spilled_5k(benchmark, graph_5k_spilled):
    """Aggregation on spilled columns."""
    benchmark(
        graph_5k_spilled.cypher,
        "MATCH (n:Item) RETURN count(n) AS cnt, avg(n.value) AS avg_val",
    )


# ---------------------------------------------------------------------------
# Vacuum and consolidation
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_vacuum_5k(benchmark):
    """Vacuum after deleting 60% of nodes.

    Was a `columnar` / `no_columnar` pair whose bodies were character-identical
    — the "baseline comparison" arm never built a different graph, so the two
    cells reported the same operation under two names.
    """

    def run():
        g = _build_graph(5000)
        g.set_auto_vacuum(None)
        g.cypher("MATCH (n:Item) WHERE n.value < 3000 DETACH DELETE n")
        g.vacuum()

    benchmark(run)


@pytest.mark.benchmark
def test_bench_unspill_rebuild_1k(benchmark, bench_graph_1k):
    """One full consolidation pass over a heap-resident graph's columns.

    `unspill()` is the public route to the rebuild `save()` and `vacuum()` also
    run. The cell lived in `test_bench_core.py` as `test_bench_columnar_enable`
    and timed a `disable_columnar()` / `enable_columnar()` round trip; both are
    gone and the operation is not the same one, so the tracked cell was retired
    rather than renamed over an anchor value that measured something else. It
    is untracked here until a release capture can baseline it on both
    platforms.
    """
    benchmark(bench_graph_1k.unspill)


# ---------------------------------------------------------------------------
# Save: heap vs spilled
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_save_kgl_heap_5k(benchmark, graph_5k, tmp_path):
    """Save a `.kgl` from heap-resident columns."""
    counter = [0]

    def save():
        graph_5k.save(str(tmp_path / f"save_{counter[0]}.kgl"))
        counter[0] += 1

    benchmark(save)


@pytest.mark.benchmark
def test_bench_save_kgl_spilled_5k(benchmark, graph_5k_spilled, tmp_path):
    """Save a `.kgl` from spilled columns."""
    counter = [0]

    def save():
        graph_5k_spilled.save(str(tmp_path / f"save_{counter[0]}.kgl"))
        counter[0] += 1

    benchmark(save)


# ---------------------------------------------------------------------------
# Variable-length k-hop peak memory (part-6 program, phase V1)
# ---------------------------------------------------------------------------
#
# Not a benchmark: this is a LAW, and its units are ratios, not bytes. A
# multi-seed k-hop query's intermediate rows are `seeds x |reachable|`, so the
# peak grows with the seed count even though the *answer* — the union of the
# reachable sets — barely moves. That is the +3 GB the part-6 eval reported,
# fully attributed (~300 B/row, no leak).
#
# Measured on this shape at 30k nodes, release build, before V4: 25 seeds reach
# 15 847 nodes for 27.9 MB of peak; 50 seeds reach 19 268 (1.22x) for 54.7 MB
# (1.96x). Peak tracked the seeds, not the targets, and this landed `xfail`.
#
# V4 made the expansion deduplicate targets globally for a distinct-only
# consumer, so the pair-shaped rows are never built: the same two sizes on the
# same fixture now cost 4.6 MB -> 6.6 MB (1.44x) release, 5.2 -> 7.2 MB (1.39x)
# debug. The law is a plain test from here — a regression back to the row-shaped
# implementation lands at 1.96x and cannot hide under the 1.8x ceiling.


#: 2x the seeds may cost at most this much more peak. The union of reachable
#: sets grows ~1.22x between the two sizes, so a frontier-shaped
#: implementation lands near there; today's row-shaped one lands at ~1.96x.
#: 1.8x sits clear of both, so neither verdict is a coin flip.
MAX_SEED_GROWTH_FACTOR = 1.8

_KHOP_PEAK_PROBE = textwrap.dedent(
    """
    import resource, sys
    import pandas as pd
    import kglite

    def peak_mb():
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux kilobytes.
        return v / (1024 * 1024) if sys.platform == "darwin" else v / 1024

    node_count, seed_count, attachments = 30_000, int(sys.argv[1]), 2

    repeated = list(range(attachments))
    src, dst = [], []
    state = 20_260_821
    for new in range(attachments, node_count):
        chosen = []
        while len(chosen) < attachments:
            state = (state * 1_103_515_245 + 12_345) & 0x7FFF_FFFF
            candidate = repeated[state % len(repeated)]
            if candidate not in chosen:
                chosen.append(candidate)
        for target in chosen:
            src.append(new)
            dst.append(target)
            repeated.append(target)
        repeated.extend([new] * attachments)
    src, dst = src + dst, dst + src

    graph = kglite.KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame({"pid": list(range(node_count)), "name": [f"P{i}" for i in range(node_count)]}),
        "Person", "pid", "name",
    )
    graph.add_connections(pd.DataFrame({"s": src, "d": dst}), "KNOWS", "Person", "s", "Person", "d")

    ids = [(i * 197 + 13) % node_count for i in range(seed_count)]
    graph.cypher("MATCH (p:Person) RETURN count(p) AS n").to_list()

    before = peak_mb()
    rows = graph.cypher(
        "MATCH (p:Person)-[:KNOWS*1..3]->(f:Person) WHERE p.id IN $ids RETURN count(DISTINCT f) AS reached",
        params={"ids": ids},
    ).to_list()
    print(peak_mb() - before, rows[0]["reached"])
    """
)


def _khop_peak_delta_mb(seed_count: int) -> tuple[float, int]:
    """Peak-RSS delta (MB) and reached-node count for one seed count.

    A fresh subprocess per size, for the reason `test_unwind_scope_narrowing`
    documents: run both in one process and the allocator's retained arenas from
    the first size absorb the second, understating the growth — a failure in
    the reassuring direction.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _KHOP_PEAK_PROBE, str(seed_count)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == 0, f"probe at {seed_count} seeds failed (rc={proc.returncode})\nstderr:\n{proc.stderr}"
    delta, reached = proc.stdout.strip().splitlines()[-1].split()
    return float(delta), int(reached)


@pytest.mark.skipif(sys.platform == "win32", reason="peak-RSS probe uses resource.getrusage (POSIX-only)")
def test_khop3_peak_memory_scales_with_targets_not_seeds():
    """Doubling the seeds must not double the peak.

    The seeds only decide how many BFS roots there are; the answer is the union
    of what they reach, which grows far more slowly. An implementation whose
    memory follows the *targets* satisfies this; one whose memory follows the
    *rows* does not.
    """
    small_peak, small_reached = _khop_peak_delta_mb(25)
    large_peak, large_reached = _khop_peak_delta_mb(50)

    # The law is only meaningful while the reachable sets stay comparable: if
    # 2x the seeds really did reach 2x the nodes, 2x the peak would be correct.
    target_growth = large_reached / small_reached
    assert target_growth < 1.5, (
        f"fixture drift: 50 seeds reach {target_growth:.2f}x the nodes 25 do "
        f"({small_reached} -> {large_reached}). The law compares seed growth "
        "against target growth and needs them to differ."
    )
    # Guard against a small side so close to zero that any large side passes.
    assert small_peak >= 1.0, f"peak delta at 25 seeds was {small_peak:.1f} MB — too small to form a ratio from"

    seed_growth = large_peak / small_peak
    assert seed_growth <= MAX_SEED_GROWTH_FACTOR, (
        f"2x the seeds cost {seed_growth:.2f}x the peak "
        f"({small_peak:.1f} MB -> {large_peak:.1f} MB) while reaching only "
        f"{target_growth:.2f}x the nodes. Peak is following the seed count, "
        f"not the reachable set (ceiling {MAX_SEED_GROWTH_FACTOR}x)."
    )
