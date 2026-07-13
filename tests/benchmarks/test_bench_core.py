"""Core benchmarks using pytest-benchmark for historical tracking.

These benchmarks measure the key operations and are tracked over time.
Run with: make bench-save (to save a baseline) or make bench-compare (to compare).
"""

import pandas as pd
import pytest

from kglite import KnowledgeGraph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bench_graph():
    """Graph with 1000 nodes and 2000 edges for benchmarking."""
    graph = KnowledgeGraph()

    nodes = pd.DataFrame(
        {
            "nid": list(range(1000)),
            "name": [f"Node_{i}" for i in range(1000)],
            "value": [float(i) for i in range(1000)],
            "category": [f"cat_{i % 10}" for i in range(1000)],
        }
    )
    graph.add_nodes(nodes, "Item", "nid", "name")

    edges = pd.DataFrame(
        {
            "from_id": [i % 1000 for i in range(2000)],
            "to_id": [(i * 7 + 13) % 1000 for i in range(2000)],
            "weight": [float(i % 100) for i in range(2000)],
        }
    )
    graph.add_connections(edges, "LINKS", "Item", "from_id", "Item", "to_id", columns=["weight"])

    return graph


@pytest.fixture(scope="module")
def grouped_count_graph():
    """10k+10k nodes and 30k edges for grouped-count top-k regressions.

    Both endpoints intentionally repeat their grouping property across many
    nodes. This keeps the benchmark honest: the fast path must aggregate by
    the resolved property value, not by node identity.
    """
    graph = KnowledgeGraph()
    n = 10_000
    graph.add_nodes(
        pd.DataFrame(
            {
                "sid": list(range(n)),
                "name": [f"Source_{i}" for i in range(n)],
                "bucket": [f"source_bucket_{i % 100}" for i in range(n)],
            }
        ),
        "Source",
        "sid",
        "name",
    )
    graph.add_nodes(
        pd.DataFrame(
            {
                "gid": list(range(n)),
                "name": [f"Group_{i}" for i in range(n)],
                "bucket": [f"target_bucket_{i % 100}" for i in range(n)],
            }
        ),
        "Group",
        "gid",
        "name",
    )
    graph.add_connections(
        pd.DataFrame(
            {
                "source": [i % n for i in range(3 * n)],
                "target": [(i * 13 + (i // n) * 997 + 7) % n for i in range(3 * n)],
            }
        ),
        "RELATES_TO",
        "Source",
        "source",
        "Group",
        "target",
    )
    return graph


@pytest.fixture(scope="module")
def wide_edge_count_graph():
    """One million homogeneous edges, matching the reported legal graph scale."""
    graph = KnowledgeGraph()
    node_count = 20_000
    edge_count = 1_000_000
    graph.add_nodes(
        pd.DataFrame(
            {
                "nid": list(range(node_count)),
                "name": [f"Node_{i}" for i in range(node_count)],
            }
        ),
        "Item",
        "nid",
        "name",
    )
    graph.add_connections(
        pd.DataFrame(
            {
                "source": [i % node_count for i in range(edge_count)],
                "target": [(i * 13 + 7) % node_count for i in range(edge_count)],
            }
        ),
        "LINKS",
        "Item",
        "source",
        "Item",
        "target",
    )
    return graph


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_bench_add_nodes(benchmark):
    """Bulk node insertion (1000 nodes)."""
    graph = KnowledgeGraph()
    nodes = pd.DataFrame(
        {
            "nid": list(range(1000)),
            "name": [f"Node_{i}" for i in range(1000)],
            "value": [float(i) for i in range(1000)],
        }
    )

    benchmark(graph.add_nodes, nodes, "Item", "nid", "name")


@pytest.mark.benchmark
def test_bench_add_connections(benchmark):
    """Bulk edge insertion (2000 edges)."""
    graph = KnowledgeGraph()
    nodes = pd.DataFrame(
        {
            "nid": list(range(1000)),
            "name": [f"Node_{i}" for i in range(1000)],
        }
    )
    graph.add_nodes(nodes, "Item", "nid", "name")

    edges = pd.DataFrame(
        {
            "from_id": [i % 1000 for i in range(2000)],
            "to_id": [(i * 7 + 13) % 1000 for i in range(2000)],
            "weight": [float(i % 100) for i in range(2000)],
        }
    )

    benchmark(graph.add_connections, edges, "LINKS", "Item", "from_id", "Item", "to_id", columns=["weight"])


@pytest.mark.benchmark
def test_bench_cypher_match(benchmark, bench_graph):
    """Simple MATCH...RETURN query."""
    benchmark(bench_graph.cypher, "MATCH (n:Item) RETURN n.title, n.value LIMIT 100")


@pytest.mark.benchmark
def test_bench_cypher_match_materialized(benchmark, bench_graph):
    """Simple MATCH consumed into Python rows (includes lazy materialization)."""

    def query_and_consume():
        return bench_graph.cypher("MATCH (n:Item) RETURN n.title, n.value LIMIT 100").to_list()

    benchmark(query_and_consume)


@pytest.mark.benchmark
def test_bench_cypher_where(benchmark, bench_graph):
    """Filtered MATCH...WHERE...RETURN query."""
    benchmark(bench_graph.cypher, "MATCH (n:Item) WHERE n.value > 500 RETURN n.title, n.value")


@pytest.mark.benchmark
def test_bench_grouped_count_top_k_target_property(benchmark, grouped_count_graph):
    """User shape: count incoming rows, group on target property, order + limit."""

    def query_and_consume():
        return grouped_count_graph.cypher(
            "MATCH (s:Source)-[:RELATES_TO]->(g:Group) "
            "RETURN g.bucket AS bucket, count(s) AS uses "
            "ORDER BY uses DESC LIMIT 10"
        ).to_list()

    result = benchmark(query_and_consume)
    assert len(result) == 10
    assert all(row["uses"] == 300 for row in result)


@pytest.mark.benchmark
def test_bench_grouped_count_top_k_source_property(benchmark, grouped_count_graph):
    """User shape: count outgoing rows, group on source property, order + limit."""

    def query_and_consume():
        return grouped_count_graph.cypher(
            "MATCH (s:Source)-[:RELATES_TO]->(g:Group) "
            "RETURN s.bucket AS bucket, count(g) AS uses "
            "ORDER BY uses DESC LIMIT 10"
        ).to_list()

    result = benchmark(query_and_consume)
    assert len(result) == 10
    assert all(row["uses"] == 300 for row in result)


@pytest.mark.benchmark
def test_bench_untyped_edge_count_1m(benchmark, wide_edge_count_graph):
    """Wide `MATCH ()-[r]->()` count used by graph inventory interfaces."""

    def query_and_consume():
        return wide_edge_count_graph.cypher("MATCH ()-[r]->() RETURN count(r) AS edges").to_list()

    result = benchmark(query_and_consume)
    assert result == [{"edges": 1_000_000}]


@pytest.mark.benchmark
def test_bench_two_edge_distinct_filtered_path(benchmark, grouped_count_graph):
    """Two-edge filtered path with DISTINCT+LIMIT, mirroring legal graph lookups."""

    def query_and_consume():
        return grouped_count_graph.cypher(
            "MATCH (g:Group)<-[:RELATES_TO]-(s:Source)-[:RELATES_TO]->(peer:Group) "
            "WHERE g.name CONTAINS 'Group_1' "
            "RETURN DISTINCT peer.bucket AS bucket LIMIT 20"
        ).to_list()

    result = benchmark(query_and_consume)
    assert 1 <= len(result) <= 20


@pytest.mark.benchmark
def test_bench_traversal(benchmark, bench_graph):
    """Multi-hop traversal via fluent API."""
    benchmark(bench_graph.select("Item").where({"id": 0}).traverse, "LINKS")


@pytest.mark.benchmark
def test_bench_shortest_path(benchmark, bench_graph):
    """Shortest path computation."""
    benchmark(bench_graph.cypher, "MATCH p = shortestPath((a:Item {id: 0})-[*]-(b:Item {id: 500})) RETURN length(p)")


# ---------------------------------------------------------------------------
# Columnar storage benchmarks
# ---------------------------------------------------------------------------


@pytest.fixture
def bench_graph_columnar():
    """Graph with 1000 nodes using columnar storage."""
    graph = KnowledgeGraph()
    nodes = pd.DataFrame(
        {
            "nid": list(range(1000)),
            "name": [f"Node_{i}" for i in range(1000)],
            "value": [float(i) for i in range(1000)],
            "category": [f"cat_{i % 10}" for i in range(1000)],
        }
    )
    graph.add_nodes(nodes, "Item", "nid", "name")

    edges = pd.DataFrame(
        {
            "from_id": [i % 1000 for i in range(2000)],
            "to_id": [(i * 7 + 13) % 1000 for i in range(2000)],
            "weight": [float(i % 100) for i in range(2000)],
        }
    )
    graph.add_connections(edges, "LINKS", "Item", "from_id", "Item", "to_id", columns=["weight"])
    graph.enable_columnar()
    return graph


@pytest.mark.benchmark
def test_bench_columnar_enable(benchmark, bench_graph):
    """Time to convert from compact to columnar storage."""

    def enable():
        bench_graph.disable_columnar()
        bench_graph.enable_columnar()

    benchmark(enable)


@pytest.mark.benchmark
def test_bench_columnar_cypher_where(benchmark, bench_graph_columnar):
    """Filtered MATCH...WHERE with columnar storage."""
    benchmark(bench_graph_columnar.cypher, "MATCH (n:Item) WHERE n.value > 500 RETURN n.title, n.value")


@pytest.mark.benchmark
def test_bench_columnar_cypher_match(benchmark, bench_graph_columnar):
    """Simple MATCH...RETURN with columnar storage."""
    benchmark(bench_graph_columnar.cypher, "MATCH (n:Item) RETURN n.title, n.value LIMIT 100")


@pytest.mark.benchmark
def test_bench_columnar_save_kgl(benchmark, bench_graph_columnar, tmp_path):
    """Save columnar graph as standard .kgl file.

    fsync=False: this tracks columnar *serialization + write* throughput, the
    thing kglite controls. The fsync durability barrier (default in save()) is a
    fixed OS-level cost orthogonal to serialization — including it would make a
    µs-scale bench dominated by ms-scale disk-flush latency.
    """
    path = str(tmp_path / "bench.kgl")
    benchmark(lambda: bench_graph_columnar.save(path, fsync=False))


@pytest.mark.benchmark
def test_bench_save_v3(benchmark, bench_graph_columnar, tmp_path):
    """Save columnar graph as a .kgl file (fsync=False — see save_kgl bench)."""
    counter = [0]

    def save():
        bench_graph_columnar.save(str(tmp_path / f"v3_{counter[0]}.kgl"), fsync=False)
        counter[0] += 1

    benchmark(save)


# ---------------------------------------------------------------------------
# Value::Node projection benchmarks (Phase A.1 → Phase C.4 Bolt consumer)
# ---------------------------------------------------------------------------
#
# Phase A.1 (shipped in 0.10.0) added Value::Node / Relationship / Path / List
# / Map variants. `RETURN n` no longer collapses to a title string — it
# materializes a full {id, labels, properties} structure. The Bolt server
# (Phase C.4) routes this over PackStream as a Node struct, so any
# regression in projection cost shows up in both Python `cypher()` and Bolt
# PULL.
#
# These benchmarks are the pre-Bolt baseline for that path. Captured to
# `tests/benchmarks/baselines/<version>.json` on the next release commit
# via `make refresh-release-constants`. Phase B itself doesn't ship a
# release.


@pytest.fixture
def node_projection_graph():
    """10k Person nodes + ~30k KNOWS edges — sized so projection cost
    dominates over query planning."""
    graph = KnowledgeGraph()
    n = 10_000
    nodes = pd.DataFrame(
        {
            "pid": list(range(n)),
            "name": [f"P{i}" for i in range(n)],
            "age": [20 + (i % 60) for i in range(n)],
            "city": [f"city_{i % 100}" for i in range(n)],
        }
    )
    graph.add_nodes(nodes, "Person", "pid", "name")

    edges = pd.DataFrame(
        {
            "s": [i % n for i in range(3 * n)],
            "d": [(i * 13 + 7) % n for i in range(3 * n)],
        }
    )
    graph.add_connections(edges, "KNOWS", "Person", "s", "Person", "d")
    return graph


@pytest.mark.benchmark
def test_bench_return_node_10k(benchmark, node_projection_graph):
    """RETURN n over 10k nodes — eager Value::Node projection.

    Drives the projection path shared between Python `cypher()` and the
    Bolt server's RECORD emission (Phase C.4). Regressions here are
    visible everywhere downstream of A.1.
    """
    benchmark(node_projection_graph.cypher, "MATCH (n:Person) RETURN n")


@pytest.mark.benchmark
def test_bench_return_node_rel_node_100(benchmark, node_projection_graph):
    """Multi-binding projection: `a`, `r`, `b` LIMIT 100.

    Exercises Node + Relationship + Node materialization in the same
    record — the typical shape of a Bolt PULL response for graph
    visualization clients (Neo4j Browser, BloodHound).
    """
    benchmark(
        node_projection_graph.cypher,
        "MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a, r, b LIMIT 100",
    )
