"""Benchmarks for memory management: columnar enable/disable, spill, unspill, vacuum.

Compares fully in-memory vs part-memory/part-disk (spilled) performance.
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
    """5000-node graph (compact storage)."""
    return _build_graph(5000)


@pytest.fixture
def graph_5k_columnar():
    """5000-node graph (columnar, heap-backed)."""
    return _build_graph(5000)


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
def test_bench_query_where_heap_5k(benchmark, graph_5k_columnar):
    """Filtered query on heap-backed columnar (5000 nodes)."""
    benchmark(
        graph_5k_columnar.cypher,
        "MATCH (n:Item) WHERE n.value > 4000 RETURN n.title, n.value",
    )


@pytest.mark.benchmark
def test_bench_query_where_spilled_5k(benchmark, graph_5k_spilled):
    """Filtered query on spilled columnar (5000 nodes)."""
    benchmark(
        graph_5k_spilled.cypher,
        "MATCH (n:Item) WHERE n.value > 4000 RETURN n.title, n.value",
    )


@pytest.mark.benchmark
def test_bench_query_match_heap_5k(benchmark, graph_5k_columnar):
    """Simple MATCH on heap-backed columnar (5000 nodes)."""
    benchmark(
        graph_5k_columnar.cypher,
        "MATCH (n:Item) RETURN n.title, n.value LIMIT 100",
    )


@pytest.mark.benchmark
def test_bench_query_match_spilled_5k(benchmark, graph_5k_spilled):
    """Simple MATCH on spilled columnar (5000 nodes)."""
    benchmark(
        graph_5k_spilled.cypher,
        "MATCH (n:Item) RETURN n.title, n.value LIMIT 100",
    )


@pytest.mark.benchmark
def test_bench_query_aggregation_heap_5k(benchmark, graph_5k_columnar):
    """Aggregation on heap-backed columnar."""
    benchmark(
        graph_5k_columnar.cypher,
        "MATCH (n:Item) RETURN count(n) AS cnt, avg(n.value) AS avg_val",
    )


@pytest.mark.benchmark
def test_bench_query_aggregation_spilled_5k(benchmark, graph_5k_spilled):
    """Aggregation on spilled columnar."""
    benchmark(
        graph_5k_spilled.cypher,
        "MATCH (n:Item) RETURN count(n) AS cnt, avg(n.value) AS avg_val",
    )


# ---------------------------------------------------------------------------
# Vacuum with columnar rebuild
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_vacuum_columnar_5k(benchmark):
    """Vacuum + columnar rebuild after deleting 60% of nodes."""

    def run():
        g = _build_graph(5000)
        g.set_auto_vacuum(None)
        g.cypher("MATCH (n:Item) WHERE n.value < 3000 DETACH DELETE n")
        g.vacuum()

    benchmark(run)


@pytest.mark.benchmark
def test_bench_vacuum_no_columnar_5k(benchmark):
    """Vacuum without columnar (baseline comparison)."""

    def run():
        g = _build_graph(5000)
        g.set_auto_vacuum(None)
        g.cypher("MATCH (n:Item) WHERE n.value < 3000 DETACH DELETE n")
        g.vacuum()

    benchmark(run)


# ---------------------------------------------------------------------------
# Save benchmarks: heap vs spilled
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_save_v3_heap_5k(benchmark, graph_5k_columnar, tmp_path):
    """Save v3 .kgl from heap-backed columnar."""
    counter = [0]

    def save():
        graph_5k_columnar.save(str(tmp_path / f"v3_{counter[0]}.kgl"))
        counter[0] += 1

    benchmark(save)


@pytest.mark.benchmark
def test_bench_save_v3_spilled_5k(benchmark, graph_5k_spilled, tmp_path):
    """Save v3 .kgl from spilled columnar."""
    counter = [0]

    def save():
        graph_5k_spilled.save(str(tmp_path / f"v3_{counter[0]}.kgl"))
        counter[0] += 1

    benchmark(save)
