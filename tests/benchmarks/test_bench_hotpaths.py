"""Hot-path benchmarks for the Phase 3 optimisation candidates.

Each benchmark isolates one suspected hot path so before/after deltas
attribute cleanly to a single change:

- DISTINCT keying (per-row Debug-format String allocation)
- RETURN n materialization on wide schemas (full property-map clone)
- UNWIND expansion (per-item row clone)
- WHERE with property access (per-row alias resolution)
- ORDER BY + LIMIT over expressions

Sized at 50k nodes so per-row costs dominate fixed overheads while a
full round stays comfortably under pytest-benchmark's calibration
budget. Run with: make bench (or pytest tests/benchmarks/ -m benchmark).
"""

import pandas as pd
import pytest

from kglite import KnowledgeGraph

N = 50_000

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hot_graph():
    """50k nodes, 12 properties each (wide schema), 100k edges.

    high_card has ~N/2 distinct values (stress dedup keying);
    mid_card has 1000; low_card has 10 (GROUP BY shape).
    """
    graph = KnowledgeGraph()

    nodes = pd.DataFrame(
        {
            "nid": list(range(N)),
            "name": [f"Node_{i}" for i in range(N)],
            "high_card": [f"hc_{i % (N // 2)}" for i in range(N)],
            "mid_card": [f"mc_{i % 1000}" for i in range(N)],
            "low_card": [f"lc_{i % 10}" for i in range(N)],
            "value": [float(i) for i in range(N)],
            "rank_val": [(i * 7919) % N for i in range(N)],
            "flag": [i % 2 == 0 for i in range(N)],
            "p1": [f"a{i % 97}" for i in range(N)],
            "p2": [f"b{i % 89}" for i in range(N)],
            "p3": [float(i % 83) for i in range(N)],
            "p4": [i % 79 for i in range(N)],
        }
    )
    graph.add_nodes(nodes, "Item", "nid", "name")

    edges = pd.DataFrame(
        {
            "from_id": [i % N for i in range(2 * N)],
            "to_id": [(i * 7 + 13) % N for i in range(2 * N)],
            "weight": [float(i % 100) for i in range(2 * N)],
        }
    )
    graph.add_connections(edges, "LINKS", "Item", "from_id", "Item", "to_id", columns=["weight"])

    return graph


# ---------------------------------------------------------------------------
# DISTINCT keying
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_count_distinct_high_card(benchmark, hot_graph):
    """count(DISTINCT prop) with ~25k distinct string values."""
    benchmark(hot_graph.cypher, "MATCH (n:Item) RETURN count(DISTINCT n.high_card)")


@pytest.mark.benchmark
def test_bench_collect_distinct_mid_card(benchmark, hot_graph):
    """collect(DISTINCT prop) with 1000 distinct values."""
    benchmark(hot_graph.cypher, "MATCH (n:Item) RETURN size(collect(DISTINCT n.mid_card))")


@pytest.mark.benchmark
def test_bench_return_distinct_rows(benchmark, hot_graph):
    """RETURN DISTINCT over two columns (row-level dedup)."""
    benchmark(
        hot_graph.cypher,
        "MATCH (n:Item) RETURN DISTINCT n.mid_card, n.low_card",
    )


# ---------------------------------------------------------------------------
# RETURN n materialization (wide schema)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_return_whole_nodes_wide(benchmark, hot_graph):
    """RETURN n materializes all 12 properties x 5k nodes."""
    benchmark(hot_graph.cypher, "MATCH (n:Item) WHERE n.rank_val < 5000 RETURN n")


@pytest.mark.benchmark
def test_bench_return_two_props(benchmark, hot_graph):
    """Same selectivity, projecting 2 properties instead of the node."""
    benchmark(
        hot_graph.cypher,
        "MATCH (n:Item) WHERE n.rank_val < 5000 RETURN n.name, n.value",
    )


@pytest.mark.benchmark
def test_bench_collect_nodes(benchmark, hot_graph):
    """collect(n) per group — node materialization inside aggregation."""
    benchmark(
        hot_graph.cypher,
        "MATCH (n:Item) WHERE n.rank_val < 5000 RETURN n.low_card, size(collect(n))",
    )


# ---------------------------------------------------------------------------
# UNWIND expansion
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_unwind_literal_x_rows(benchmark, hot_graph):
    """5k matched rows x UNWIND 10 — per-item row clone stress."""
    benchmark(
        hot_graph.cypher,
        "MATCH (n:Item) WHERE n.rank_val < 5000 UNWIND [1,2,3,4,5,6,7,8,9,10] AS x RETURN count(x)",
    )


@pytest.mark.benchmark
def test_bench_unwind_range_aggregate(benchmark, hot_graph):
    """Pure UNWIND throughput: 100k items into an aggregate."""
    benchmark(hot_graph.cypher, "UNWIND range(1, 100000) AS x RETURN sum(x)")


# ---------------------------------------------------------------------------
# WHERE / alias resolution
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_where_multi_prop(benchmark, hot_graph):
    """WHERE touching three properties per row (alias-resolution heavy)."""
    benchmark(
        hot_graph.cypher,
        "MATCH (n:Item) WHERE n.value > 25000.0 AND n.p4 < 40 AND n.flag RETURN count(n)",
    )


@pytest.mark.benchmark
def test_bench_order_by_expr_limit(benchmark, hot_graph):
    """ORDER BY on an expression with LIMIT (top-K path)."""
    benchmark(
        hot_graph.cypher,
        "MATCH (n:Item) RETURN n.name, n.value ORDER BY n.value DESC LIMIT 25",
    )
