"""Focused fused-scan benchmarks kept outside the frozen core CI harness."""

import pandas as pd
import pytest

from kglite import KnowledgeGraph

FUSED_TYPED_SOURCE_COUNT = 50_000
FUSED_TYPED_ORACLE_SOURCE_COUNT = 64
FUSED_TYPED_GROUP_COUNT = 128
FUSED_TYPED_HEAVY_SOURCES = 16
FUSED_TYPED_TOP_K = 10


def _build_fused_typed_scan_graph(source_count: int) -> KnowledgeGraph:
    """Build typed candidates with deterministic, unique high-degree leaders."""
    assert source_count >= FUSED_TYPED_HEAVY_SOURCES
    graph = KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame(
            {
                "sid": range(source_count),
                "name": [f"Source_{source}" for source in range(source_count)],
                "bucket": [f"bucket_{source}" for source in range(source_count)],
                "eligible": [source < FUSED_TYPED_HEAVY_SOURCES or source % 2 == 0 for source in range(source_count)],
            }
        ),
        "ScanSource",
        "sid",
        "name",
        columns=["bucket", "eligible"],
    )
    graph.add_nodes(
        pd.DataFrame(
            {
                "gid": range(FUSED_TYPED_GROUP_COUNT),
                "name": [f"Group_{group}" for group in range(FUSED_TYPED_GROUP_COUNT)],
            }
        ),
        "ScanGroup",
        "gid",
        "name",
    )

    sources = list(range(source_count))
    targets = [source % FUSED_TYPED_GROUP_COUNT for source in sources]
    for source in range(FUSED_TYPED_HEAVY_SOURCES):
        desired_degree = 33 - source
        sources.extend([source] * (desired_degree - 1))
        targets.extend((source + offset) % FUSED_TYPED_GROUP_COUNT for offset in range(1, desired_degree))
    graph.add_connections(
        pd.DataFrame({"source": sources, "target": targets}),
        "SCAN_EDGE",
        "ScanSource",
        "source",
        "ScanGroup",
        "target",
    )

    expected_edges = source_count + sum(32 - source for source in range(FUSED_TYPED_HEAVY_SOURCES))
    assert graph.shape == (source_count + FUSED_TYPED_GROUP_COUNT, expected_edges)
    assert graph.node_type_counts() == {
        "ScanGroup": FUSED_TYPED_GROUP_COUNT,
        "ScanSource": source_count,
    }
    return graph


@pytest.fixture(scope="module")
def fused_typed_scan_graph():
    """50k-candidate graph reserved for benchmark-marked cells."""
    return _build_fused_typed_scan_graph(FUSED_TYPED_SOURCE_COUNT)


@pytest.fixture(scope="module")
def fused_typed_oracle_graph():
    """Bounded graph retaining the benchmark's planner and executor shapes."""
    return _build_fused_typed_scan_graph(FUSED_TYPED_ORACLE_SOURCE_COUNT)


FUSED_TYPED_PROPERTY_QUERY = (
    "MATCH (s:ScanSource)-[:SCAN_EDGE]->(g:ScanGroup) "
    "RETURN s.bucket AS bucket, count(g) AS uses "
    "ORDER BY uses DESC LIMIT 10"
)
FUSED_TYPED_NODE_QUERY = (
    "MATCH (s:ScanSource {eligible: true})-[:SCAN_EDGE]->(g:ScanGroup) "
    "RETURN s AS source, count(g) AS uses "
    "ORDER BY uses DESC LIMIT 10"
)


def _assert_fused_match_return_aggregate(graph: KnowledgeGraph, query: str) -> list[dict]:
    operations = [row["operation"] for row in graph.cypher(f"EXPLAIN {query}").to_list()]
    disabled_operations = [
        row["operation"]
        for row in graph.cypher(
            f"EXPLAIN {query}",
            disabled_passes=["fuse_match_return_aggregate"],
        ).to_list()
    ]
    assert any("FusedMatchReturnAggregate" in operation for operation in operations), operations
    assert not any("FusedMatchReturnAggregate" in operation for operation in disabled_operations), disabled_operations

    optimized = graph.cypher(query).to_list()
    disabled = graph.cypher(query, disabled_passes=["fuse_match_return_aggregate"]).to_list()
    assert optimized == disabled
    return optimized


def _assert_fused_typed_property_rows(rows: list[dict]) -> None:
    assert rows == [{"bucket": f"bucket_{source}", "uses": 33 - source} for source in range(FUSED_TYPED_TOP_K)]


def _assert_fused_typed_node_rows(rows: list[dict]) -> None:
    assert [(row["source"]["id"], row["uses"]) for row in rows] == [
        (source, 33 - source) for source in range(FUSED_TYPED_TOP_K)
    ]


def test_fused_typed_scan_benchmark_oracles(fused_typed_oracle_graph):
    property_rows = _assert_fused_match_return_aggregate(fused_typed_oracle_graph, FUSED_TYPED_PROPERTY_QUERY)
    _assert_fused_typed_property_rows(property_rows)

    node_rows = _assert_fused_match_return_aggregate(fused_typed_oracle_graph, FUSED_TYPED_NODE_QUERY)
    _assert_fused_typed_node_rows(node_rows)


@pytest.mark.benchmark
def test_bench_fused_typed_property_group_scan(benchmark, fused_typed_scan_graph):
    untimed_rows = _assert_fused_match_return_aggregate(fused_typed_scan_graph, FUSED_TYPED_PROPERTY_QUERY)
    _assert_fused_typed_property_rows(untimed_rows)

    rows = benchmark(lambda: fused_typed_scan_graph.cypher(FUSED_TYPED_PROPERTY_QUERY).to_list())
    _assert_fused_typed_property_rows(rows)
    benchmark.extra_info["typed_candidates"] = FUSED_TYPED_SOURCE_COUNT
    benchmark.extra_info["target_materialization"] = "TypeNodesRef::to_vec property group"


@pytest.mark.benchmark
def test_bench_fused_typed_property_filtered_node_top_k(benchmark, fused_typed_scan_graph):
    untimed_rows = _assert_fused_match_return_aggregate(fused_typed_scan_graph, FUSED_TYPED_NODE_QUERY)
    _assert_fused_typed_node_rows(untimed_rows)

    rows = benchmark(lambda: fused_typed_scan_graph.cypher(FUSED_TYPED_NODE_QUERY).to_list())
    _assert_fused_typed_node_rows(rows)
    benchmark.extra_info["typed_candidates"] = FUSED_TYPED_SOURCE_COUNT
    benchmark.extra_info["target_materialization"] = "TypeNodesRef::to_vec node top-k"
