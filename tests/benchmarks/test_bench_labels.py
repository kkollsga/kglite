"""Secondary-label benchmark cells for the label-hardening program.

Outside the frozen core CI harness (like test_bench_fused_scans.py). Four
concerns, mapped to the program's phases:

- bulk ``add_label`` scaling (quadratic before the bulk stamp path landed)
- big-bucket matching: ``MATCH (n:Label)`` candidate list vs ``WHERE n:Label``
  per-row membership test
- ``labels(n)`` projection vs distinct-label count L (the every-bucket scan)
- the global fusion bail: the same aggregate / spatial-join query with zero
  secondary labels (fused — also the capture's control cell) vs one secondary
  label anywhere in the graph (fusion disabled)
"""

import pandas as pd
import pytest

from kglite import KnowledgeGraph

# Default addopts deselect '-m benchmark'; CI's Python matrix installs no
# pytest-benchmark, so unmarked cells error at collection there.
pytestmark = pytest.mark.benchmark

BULK_SMALL = 10_000
BULK_LARGE = 50_000
BUCKET_NODES = 20_000
LABELS_NODES = 5_000
AGG_SOURCES = 50_000
AGG_GROUPS = 16
SPATIAL_AREAS = 50
SPATIAL_POINTS = 2_000


def _flat_graph(n: int) -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(n),
                "name": [f"n_{i}" for i in range(n)],
                "cat": [f"cat_{i % AGG_GROUPS}" for i in range(n)],
            }
        ),
        "P",
        "id",
        "name",
        columns=["cat"],
    )
    return graph


def _explain_ops(graph: KnowledgeGraph, query: str) -> list[str]:
    return [row["operation"] for row in graph.cypher(f"EXPLAIN {query}").to_list()]


# ── bulk add_label scaling ────────────────────────────────────────────────


@pytest.mark.parametrize("n", [BULK_SMALL, BULK_LARGE])
def test_bench_add_label_bulk(benchmark, n):
    """Stamp one label onto every node of a type through ``add_label``."""

    def setup():
        return (_flat_graph(n),), {}

    def stamp(graph):
        result = graph.add_label("P", list(range(n)), "VIP")
        assert result["labelled"] == n

    benchmark.pedantic(stamp, setup=setup, rounds=5, iterations=1)


# ── big-bucket matching ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def bucket_graph() -> KnowledgeGraph:
    graph = _flat_graph(BUCKET_NODES)
    graph.add_label("P", list(range(BUCKET_NODES)), "VIP")
    return graph


def test_bench_match_secondary_count(benchmark, bucket_graph):
    """Bare secondary-label match: one hash probe + candidate list."""
    query = "MATCH (n:VIP) RETURN count(n) AS c"
    assert bucket_graph.cypher(query).to_list()[0]["c"] == BUCKET_NODES
    benchmark(lambda: bucket_graph.cypher(query))


def test_bench_where_label_filter(benchmark, bucket_graph):
    """Per-row ``node_has_label`` over a graph-sized bucket."""
    query = "MATCH (n:P) WHERE n:VIP RETURN count(n) AS c"
    assert bucket_graph.cypher(query).to_list()[0]["c"] == BUCKET_NODES
    benchmark(lambda: bucket_graph.cypher(query))


# ── labels(n) projection vs distinct-label count ──────────────────────────


@pytest.mark.parametrize("distinct_labels", [1, 10, 100])
def test_bench_labels_projection(benchmark, distinct_labels):
    """``RETURN labels(n)`` iterates every bucket per node; cost moves with
    both the bucket count L and the per-bucket length N/L."""
    graph = _flat_graph(LABELS_NODES)
    for label_index in range(distinct_labels):
        ids = list(range(label_index, LABELS_NODES, distinct_labels))
        graph.add_label("P", ids, f"L{label_index}")
    query = "MATCH (n:P) RETURN labels(n) AS l"
    rows = graph.cypher(query).to_list()
    assert len(rows) == LABELS_NODES and len(rows[0]["l"]) == 2
    benchmark(lambda: graph.cypher(query))


# ── fusion bail: edge aggregate (fuse_match_return_aggregate) ─────────────
# The flat `MATCH (n:P) RETURN n.cat, count(*)` shape is served by
# fuse_node_scan_aggregate, which stays on for multi-label graphs; the
# global has_secondary_labels bail lives in the *edge*-aggregate fusions,
# so the cells use the relationship shape.

AGG_QUERY = "MATCH (s:P)-[:E]->(g:G) RETURN g.name AS name, count(s) AS k"


def _edge_agg_graph() -> KnowledgeGraph:
    graph = _flat_graph(AGG_SOURCES)
    graph.add_nodes(
        pd.DataFrame(
            {
                "gid": range(AGG_GROUPS),
                "name": [f"group_{i}" for i in range(AGG_GROUPS)],
            }
        ),
        "G",
        "gid",
        "name",
    )
    graph.add_connections(
        pd.DataFrame(
            {
                "source": range(AGG_SOURCES),
                "target": [i % AGG_GROUPS for i in range(AGG_SOURCES)],
            }
        ),
        "E",
        "P",
        "source",
        "G",
        "target",
    )
    return graph


@pytest.fixture(scope="module")
def agg_graph_no_labels() -> KnowledgeGraph:
    return _edge_agg_graph()


@pytest.fixture(scope="module")
def agg_graph_one_label() -> KnowledgeGraph:
    graph = _edge_agg_graph()
    # One label on one node flips has_secondary_labels for the whole graph.
    graph.add_label("P", [0], "Tagged")
    return graph


def test_bench_agg_group_no_labels(benchmark, agg_graph_no_labels):
    """Control cell: unchanged fused path on a label-free graph."""
    ops = " ".join(_explain_ops(agg_graph_no_labels, AGG_QUERY))
    assert "Fused" in ops, f"expected fused plan, got: {ops}"
    benchmark(lambda: agg_graph_no_labels.cypher(AGG_QUERY))


def test_bench_agg_group_one_label(benchmark, agg_graph_one_label):
    """Same query with one unrelated secondary label in the graph. Under the
    pre-P9 global bail this ran unfused (baseline: 23.4 ms vs 329 us — 71x);
    the finer gate keeps it fused, so the two arms should now match."""
    ops = " ".join(_explain_ops(agg_graph_one_label, AGG_QUERY))
    assert "Fused" in ops, f"expected fused plan under the finer gate, got: {ops}"
    benchmark(lambda: agg_graph_one_label.cypher(AGG_QUERY))


# ── fusion bail: spatial join ─────────────────────────────────────────────

SPATIAL_QUERY = "MATCH (a:Area), (p:Pt) WHERE contains(a, p) RETURN count(*) AS c"


def _spatial_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    # 50 unit-square areas in a row; 2k points spread over the first 40.
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(SPATIAL_AREAS),
                "name": [f"area_{i}" for i in range(SPATIAL_AREAS)],
                "wkt": [f"POLYGON(({i} 0, {i + 1} 0, {i + 1} 1, {i} 1, {i} 0))" for i in range(SPATIAL_AREAS)],
            }
        ),
        "Area",
        "id",
        "name",
        column_types={"wkt": "geometry"},
    )
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(SPATIAL_POINTS),
                "name": [f"pt_{i}" for i in range(SPATIAL_POINTS)],
                "lat": [0.5] * SPATIAL_POINTS,
                "lon": [(i % 40) + 0.5 for i in range(SPATIAL_POINTS)],
            }
        ),
        "Pt",
        "id",
        "name",
        column_types={"lat": "location.lat", "lon": "location.lon"},
    )
    return graph


def test_bench_spatial_join_no_labels(benchmark):
    graph = _spatial_graph()
    ops = " ".join(_explain_ops(graph, SPATIAL_QUERY))
    assert "Spatial" in ops, f"expected spatial-join plan, got: {ops}"
    assert graph.cypher(SPATIAL_QUERY).to_list()[0]["c"] == SPATIAL_POINTS
    benchmark(lambda: graph.cypher(SPATIAL_QUERY))


def test_bench_spatial_join_one_label(benchmark):
    """Pre-P9 baseline: 12.8 ms unfused vs 387 us fused (33x); the finer
    gate keeps an unrelated label from disabling the spatial join."""
    graph = _spatial_graph()
    graph.add_label("Area", [SPATIAL_AREAS - 1], "Tagged")
    ops = " ".join(_explain_ops(graph, SPATIAL_QUERY))
    assert "Spatial" in ops, f"expected spatial-join plan under the finer gate, got: {ops}"
    assert graph.cypher(SPATIAL_QUERY).to_list()[0]["c"] == SPATIAL_POINTS
    benchmark(lambda: graph.cypher(SPATIAL_QUERY))
