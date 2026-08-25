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


# ---------------------------------------------------------------------------
# Streaming aggregate group cap — `push_limit_into_aggregate`'s
# `group_limit_hint`, read by the streaming pipeline since 0.16.10.
# ---------------------------------------------------------------------------

GROUP_CAP_NODES = 200_000
GROUP_CAP_DISTINCT = 100_000
GROUP_CAP_LIMIT = 5

#: The cap has to keep *firing* on the streaming path. Measured 0.51x
#: (release, macOS arm64, 2026-08-25, two agreeing runs); before the fix the
#: streaming aggregate stamped the hint and never read it, which reads as
#: 1.00x. The bound sits between the two, so the cell goes red when the
#: streaming aggregate stops honouring the cap — not when the machine drifts,
#: since both arms are the same query on the same graph, interleaved.
GROUP_CAP_MAX_RATIO = 0.75

#: The projecting `WITH` is load-bearing twice over. It keeps the query off
#: `fuse_node_scan_aggregate`, which absorbs `MATCH (n:Ev) RETURN n.g,
#: count(*) LIMIT 5` whole and consults no hint; and it makes the group key a
#: plain variable, so the group set is keyed by resolved value rather than by
#: `NodeIndex` — the only shape where capping the group set is sound (see
#: `stream::aggregate::apply`).
GROUP_CAP_QUERY = f"MATCH (n:Ev) WITH n.g AS gg, n.w AS w RETURN gg AS g, count(*) AS c LIMIT {GROUP_CAP_LIMIT}"
GROUP_CAP_NOHINT = ["push_limit_into_aggregate"]


def _build_group_cap_graph(nodes: int, distinct: int) -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame(
            {
                "nid": range(nodes),
                "name": [f"E{i}" for i in range(nodes)],
                "g": [f"g_{i % distinct}" for i in range(nodes)],
                "w": [i % 7 for i in range(nodes)],
            }
        ),
        "Ev",
        "nid",
        "name",
        columns=["g", "w"],
    )
    return graph


@pytest.fixture(scope="module")
def group_cap_graph() -> KnowledgeGraph:
    return _build_group_cap_graph(GROUP_CAP_NODES, GROUP_CAP_DISTINCT)


def _interleaved_mins(first, second, rounds: int, warmup: int) -> tuple[float, float]:
    """Min of `rounds` alternating samples of each arm.

    Alternating rather than running one arm to completion first: a thermal or
    scheduler excursion then lands on both arms instead of on whichever one
    happened to be running, which is what the ratio below depends on.
    """
    import time

    for _ in range(warmup):
        first()
        second()
    first_best = float("inf")
    second_best = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        first()
        first_best = min(first_best, time.perf_counter() - start)
        start = time.perf_counter()
        second()
        second_best = min(second_best, time.perf_counter() - start)
    return first_best, second_best


@pytest.mark.benchmark
def test_bench_streaming_group_cap(benchmark, group_cap_graph):
    """The streaming aggregate must stop opening groups once the LIMIT is met."""
    operations = [row["operation"] for row in group_cap_graph.cypher(f"EXPLAIN {GROUP_CAP_QUERY}").to_list()]
    assert not any(operation.startswith("Fused") for operation in operations), operations

    def capped():
        return group_cap_graph.cypher(GROUP_CAP_QUERY).to_list()

    def uncapped():
        return group_cap_graph.cypher(GROUP_CAP_QUERY, disabled_passes=GROUP_CAP_NOHINT).to_list()

    expected = uncapped()
    rows_per_group = GROUP_CAP_NODES // GROUP_CAP_DISTINCT
    assert len(expected) == GROUP_CAP_LIMIT
    assert {row["c"] for row in expected} == {rows_per_group}
    assert capped() == expected, "the cap changed the answer"

    capped_s, uncapped_s = _interleaved_mins(capped, uncapped, rounds=9, warmup=2)
    ratio = capped_s / uncapped_s

    rows = benchmark(capped)
    assert rows == expected

    assert ratio <= GROUP_CAP_MAX_RATIO, (
        f"the streaming group cap is not firing: capped {capped_s * 1e3:.2f} ms "
        f"vs uncapped {uncapped_s * 1e3:.2f} ms = {ratio:.3f}x"
    )
    benchmark.extra_info["statistic"] = "min (both arms, interleaved)"
    benchmark.extra_info["nodes"] = GROUP_CAP_NODES
    benchmark.extra_info["distinct_groups"] = GROUP_CAP_DISTINCT
    benchmark.extra_info["capped_over_uncapped"] = ratio
    benchmark.extra_info["capped_ms"] = capped_s * 1e3
    benchmark.extra_info["uncapped_ms"] = uncapped_s * 1e3
