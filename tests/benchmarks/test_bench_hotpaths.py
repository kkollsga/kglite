"""Hot-path benchmarks for the Phase 3 optimisation candidates.

Each benchmark isolates one suspected hot path so before/after deltas
attribute cleanly to a single change:

- DISTINCT keying (per-row Debug-format String allocation)
- RETURN n materialization on wide schemas (full property-map clone)
- UNWIND expansion (per-item row clone)
- WHERE with property access (per-row alias resolution)
- ORDER BY + LIMIT over expressions
- atomic-checkpoint selection for trivial in-memory mutations

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


# ---------------------------------------------------------------------------
# Mutation checkpoint selection
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_single_node_create_delete_large_graph(benchmark, hot_graph):
    """A trivial write must not deep-clone the unrelated 50k-node graph."""

    def cycle():
        hot_graph.cypher("CREATE (:Scratch {id: 9999999})")
        hot_graph.cypher("MATCH (n:Scratch {id: 9999999}) DELETE n")

    benchmark(cycle)


@pytest.mark.benchmark
def test_bench_cartesian_node_scans_limit(benchmark, hot_graph):
    """LIMIT 20 must cap node-only cartesian expansion before materialization."""
    benchmark(
        hot_graph.cypher,
        "MATCH (a:Item), (b:Item) RETURN a.nid, b.nid LIMIT 20",
    )


@pytest.mark.benchmark
def test_bench_two_hop_global_count(benchmark, hot_graph):
    """A pure count must stream exact path cardinality without building rows."""
    benchmark(
        hot_graph.cypher,
        "MATCH (a:Item)-[:LINKS]->(b:Item)-[:LINKS]->(c:Item) RETURN count(*) AS paths",
    )


@pytest.mark.benchmark
def test_bench_fixed_path_relationship_materialization(benchmark, hot_graph):
    """Exact relationship lookup for an anchored fixed-length path."""
    benchmark(
        hot_graph.cypher,
        "MATCH p=(a:Item {nid: 1})-[:LINKS]->(b:Item)-[:LINKS]->(c:Item) "
        "RETURN sum(size(relationships(p))) AS relationships",
    )


@pytest.mark.benchmark
def test_bench_variable_path_relationship_materialization(benchmark, hot_graph):
    """Relationship-unique trail tracking for an anchored variable path."""
    benchmark(
        hot_graph.cypher,
        "MATCH p=(a:Item {nid: 1})-[:LINKS*1..3]->(b:Item) RETURN sum(size(relationships(p))) AS relationships",
    )


# ---------------------------------------------------------------------------
# Point lookup with a live fluent clone (Arc shared) + rel-heavy named-var
# match — the two perf landmines fixed in the opencypher-contract branch.
# ---------------------------------------------------------------------------


@pytest.fixture
def prop_edge_graph():
    """20k nodes / 100k edges with 5 string props per edge.

    Sized so a per-edge property-map clone in the matcher (the old
    `MatchBinding::Edge { properties }` fill for named edge vars)
    dominates the query time if it ever comes back.
    """
    n_nodes, n_edges = 20_000, 100_000
    graph = KnowledgeGraph()
    nodes = pd.DataFrame(
        {
            "nid": list(range(n_nodes)),
            "name": [f"N_{i}" for i in range(n_nodes)],
        }
    )
    graph.add_nodes(nodes, "PN", "nid", "name")
    edges = pd.DataFrame(
        {
            "src": [i % n_nodes for i in range(n_edges)],
            "dst": [(i * 7 + 13) % n_nodes for i in range(n_edges)],
            "p1": [f"alpha_{i % 50}" for i in range(n_edges)],
            "p2": [f"beta_{i % 40}" for i in range(n_edges)],
            "p3": [f"gamma_{i % 30}" for i in range(n_edges)],
            "p4": [f"delta_{i % 20}" for i in range(n_edges)],
            "p5": [f"epsilon_{i % 10}" for i in range(n_edges)],
        }
    )
    graph.add_connections(edges, "PR", "PN", "src", "PN", "dst", columns=["p1", "p2", "p3", "p4", "p5"])
    return graph


@pytest.mark.benchmark
def test_bench_node_lookup_while_arc_shared(benchmark, hot_graph):
    """node() point lookup with a fresh fluent clone sharing the Arc.

    Each round re-shares the Arc (select) then does the point lookup —
    if node() ever regresses to `Arc::make_mut`, every round deep-copies
    the whole 50k-node graph and this jumps by ~4 orders of magnitude.
    """
    hot_graph.build_id_indices(["Item"])

    def share_then_lookup():
        holder = hot_graph.select("Item")
        got = hot_graph.node("Item", 1234)
        assert got is not None
        del holder

    benchmark(share_then_lookup)


@pytest.mark.benchmark
def test_bench_named_rel_match_count(benchmark, prop_edge_graph):
    """MATCH with a *named* edge variable over 100k property-heavy edges.

    count(r) needs r bound but never reads its properties — the binding
    must stay index-only (no per-edge property-map clone in the matcher).
    """
    result = benchmark(prop_edge_graph.cypher, "MATCH (a:PN)-[r:PR]->(b:PN) RETURN count(r) AS c")
    assert result[0]["c"] == 100_000
