"""Benchmarks for memory management: spill, unspill, vacuum, and save.

Compares fully heap-resident columns against columns spilled to disk under
`set_memory_limit`. There is no shape to switch: properties live in per-type
columns from the first node, and the only axis here is where those columns'
bytes sit.
Run with: pytest tests/benchmarks/test_bench_memory.py -m benchmark -v -s
"""

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
