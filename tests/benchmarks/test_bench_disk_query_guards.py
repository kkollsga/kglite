"""Release-mode coverage for disk query-guard overhead.

The disk backend protects arena-backed materializations with a query guard.
These cells cover four shapes where redundant or nested guards are visible:

* ``degrees()`` performs guarded node-info and adjacency reads for every node;
* an in-memory ``degrees()`` control catches backend-neutral regressions;
* same-node shortest-path length makes the fixed guard cost large relative to
  the algorithm work; and
* fluent ``update()`` sends all selected nodes through the bulk property
  updater, whose validation currently performs a guarded read per node.

Run after building the extension in release mode::

    uv run --no-sync maturin develop --release
    .venv/bin/python -m pytest \
        tests/benchmarks/test_bench_disk_query_guards.py -m benchmark -v

The mutation cell uses a fresh copy of the published template for every timed
round. Copying, loading, and selection construction happen in ``setup`` and
are therefore outside the measurement.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph

DISK_GUARD_NODES = 10_000
ORACLE_NODES = 32
PATH_CALLS_PER_SAMPLE = 64
READ_ROUNDS = 100
READ_WARMUP_ROUNDS = 20
MUTATION_ROUNDS = 5
MUTATION_WARMUP_ROUNDS = 1


def _populate_ring(graph: KnowledgeGraph, node_count: int) -> None:
    """Populate ``graph`` with a deterministic directed ring."""
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(node_count),
                "title": [f"Node {node_id}" for node_id in range(node_count)],
                "guard_marker": [0] * node_count,
            }
        ),
        "Node",
        "id",
        "title",
    )
    graph.add_connections(
        pd.DataFrame(
            {
                "source": range(node_count),
                "target": [(node_id + 1) % node_count for node_id in range(node_count)],
            }
        ),
        "NEXT",
        "Node",
        "source",
        "Node",
        "target",
    )


def _publish_ring(root: Path, node_count: int) -> None:
    """Publish a deterministic directed ring as a disk graph."""
    graph = KnowledgeGraph(storage="disk", path=str(root))
    _populate_ring(graph, node_count)
    graph.save(str(root), fsync=False)


@pytest.fixture(scope="module")
def disk_guard_template(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("disk_guard") / "template"
    _publish_ring(root, DISK_GUARD_NODES)
    return root


@pytest.fixture(scope="module")
def disk_guard_graph(disk_guard_template) -> KnowledgeGraph:
    return kglite.load(str(disk_guard_template))


@pytest.fixture(scope="module")
def memory_guard_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    _populate_ring(graph, DISK_GUARD_NODES)
    return graph


@pytest.fixture(scope="module")
def disk_guard_oracle_graph(tmp_path_factory) -> KnowledgeGraph:
    root = tmp_path_factory.mktemp("disk_guard_oracle") / "graph"
    _publish_ring(root, ORACLE_NODES)
    return kglite.load(str(root))


def test_disk_guard_result_oracle(disk_guard_oracle_graph):
    """A small, unmarked exact oracle for every result contract below."""
    graph = disk_guard_oracle_graph
    assert graph.shortest_path_length("Node", 7, "Node", 7) == 0

    degrees = graph.select("Node").degrees()
    assert degrees == {f"Node {node_id}": 2 for node_id in range(ORACLE_NODES)}

    report = graph.select("Node").update({"guard_marker": 1})
    assert report["nodes_updated"] == ORACLE_NODES
    rows = report["graph"].cypher("MATCH (n:Node) WHERE n.guard_marker = 1 RETURN count(n) AS count").to_list()
    assert rows == [{"count": ORACLE_NODES}]


@pytest.mark.benchmark
def test_bench_disk_degrees_10k(benchmark, disk_guard_graph):
    """Degree materialization for all 10k ring nodes."""
    selected = disk_guard_graph.select("Node")
    expected = {f"Node {node_id}": 2 for node_id in range(DISK_GUARD_NODES)}

    result = benchmark.pedantic(
        selected.degrees,
        rounds=READ_ROUNDS,
        iterations=1,
        warmup_rounds=READ_WARMUP_ROUNDS,
    )

    assert result == expected


@pytest.mark.benchmark
def test_bench_memory_degrees_10k(benchmark, memory_guard_graph):
    """In-memory control for the same 10k-node degree materialization."""
    selected = memory_guard_graph.select("Node")
    expected = {f"Node {node_id}": 2 for node_id in range(DISK_GUARD_NODES)}

    result = benchmark.pedantic(
        selected.degrees,
        rounds=READ_ROUNDS,
        iterations=1,
        warmup_rounds=READ_WARMUP_ROUNDS,
    )

    assert result == expected


@pytest.mark.benchmark
def test_bench_disk_same_node_shortest_path_length(benchmark, disk_guard_graph):
    """Repeated fixed-cost path queries, amplified above Python call noise."""

    def repeated_same_node_path() -> int | float | None:
        for _ in range(PATH_CALLS_PER_SAMPLE):
            result = disk_guard_graph.shortest_path_length("Node", 5_000, "Node", 5_000)
            if result != 0:
                return result
        return 0

    result = benchmark.pedantic(
        repeated_same_node_path,
        rounds=READ_ROUNDS,
        iterations=1,
        warmup_rounds=READ_WARMUP_ROUNDS,
    )

    assert result == 0


@pytest.mark.benchmark
@pytest.mark.slow
def test_bench_disk_update_10k(benchmark, disk_guard_template, tmp_path):
    """One-property bulk update over 10k nodes from an identical disk state."""
    round_number = 0

    def setup():
        nonlocal round_number
        round_number += 1
        root = tmp_path / f"round-{round_number}"
        shutil.copytree(disk_guard_template, root)
        selected = kglite.load(str(root)).select("Node")
        return (selected,), {}

    def update(selected: KnowledgeGraph):
        return selected.update({"guard_marker": 1})

    result = benchmark.pedantic(
        update,
        setup=setup,
        rounds=MUTATION_ROUNDS,
        iterations=1,
        warmup_rounds=MUTATION_WARMUP_ROUNDS,
    )

    assert result["nodes_updated"] == DISK_GUARD_NODES
    rows = result["graph"].cypher("MATCH (n:Node) WHERE n.guard_marker = 1 RETURN count(n) AS count").to_list()
    assert rows == [{"count": DISK_GUARD_NODES}]
