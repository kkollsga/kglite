"""Filtered-scan and simple-aggregate cells over a 100k-node graph.

The scan cells cover the fused `MATCH (n:T) WHERE ... RETURN count/sum/min/max`
path: a typed int / float / string compare reaching the column filter, and the
aggregate tail. The two control cells run paths the scan work does not touch
(id-index point lookups with a 1-hop expand, and a 2-hop traversal), each long
enough (several ms) to sit well above the timer's noise floor, so a control that
moves means the machine moved.

This file is outside `test_bench_core.py`, the only file `make bench-check` and
the CI perf gate compare against a baseline, so adding cells here cannot trip
that gate. Release profile only for any timing; each cell asserts its result
against plain Python over the inserted data.
"""

import pandas as pd
import pytest

from kglite import KnowledgeGraph

N = 100_000
LOOKUPS = 100
HOP_ANCHORS = 200


def _year(i: int) -> int:
    return 1950 + (i * 7) % 80


def _score(i: int) -> float:
    return (i % 1000) / 1000.0


def _kind(i: int) -> str:
    return f"kind-{i % 20}"


YEARS = [_year(i) for i in range(N)]
SCORES = [_score(i) for i in range(N)]


@pytest.fixture(scope="module")
def scan_graph():
    graph = KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame(
            {
                "pid": range(N),
                "name": [f"Paper_{i}" for i in range(N)],
                "year": YEARS,
                "score": SCORES,
                "kind": [_kind(i) for i in range(N)],
            }
        ),
        "Paper",
        "pid",
        "name",
        columns=["year", "score", "kind"],
    )
    src = [i for i in range(N) for _ in range(2)]
    dst = [(i * 31 + k * 17 + 1) % N for i in range(N) for k in range(2)]
    graph.add_connections(
        pd.DataFrame({"src": src, "dst": dst}),
        "CITES",
        "Paper",
        "src",
        "Paper",
        "dst",
    )
    assert graph.shape == (N, 2 * N)
    return graph


def _scalar(graph, query: str):
    rows = graph.cypher(query).to_list()
    assert len(rows) == 1
    return next(iter(rows[0].values()))


FILTER_CELLS = {
    "int_gt": ("n.year > 2000", sum(1 for y in YEARS if y > 2000)),
    "float_lt": ("n.score < 0.5", sum(1 for s in SCORES if s < 0.5)),
    "and_int_float": (
        "n.year > 2000 AND n.score < 0.5",
        sum(1 for y, s in zip(YEARS, SCORES) if y > 2000 and s < 0.5),
    ),
    "string_eq": ("n.kind = 'kind-7'", sum(1 for i in range(N) if _kind(i) == "kind-7")),
}


@pytest.mark.benchmark
@pytest.mark.parametrize("cell", list(FILTER_CELLS))
def test_bench_scan_filter_count(benchmark, scan_graph, cell):
    predicate, expected = FILTER_CELLS[cell]
    query = f"MATCH (n:Paper) WHERE {predicate} RETURN count(n) AS c"
    assert _scalar(scan_graph, query) == expected
    assert benchmark(lambda: _scalar(scan_graph, query)) == expected
    benchmark.extra_info["nodes"] = N


AGG_CELLS = {
    "sum_int": ("sum(n.year)", sum(YEARS)),
    "min_int": ("min(n.year)", min(YEARS)),
    "max_int": ("max(n.year)", max(YEARS)),
    "sum_float": ("sum(n.score)", None),  # float sum order: compared with approx below
    "min_float": ("min(n.score)", min(SCORES)),
    "max_float": ("max(n.score)", max(SCORES)),
}


@pytest.mark.benchmark
@pytest.mark.parametrize("cell", list(AGG_CELLS))
def test_bench_scan_aggregate(benchmark, scan_graph, cell):
    expr, expected = AGG_CELLS[cell]
    query = f"MATCH (n:Paper) RETURN {expr} AS v"
    if expected is None:
        expected = pytest.approx(sum(SCORES), rel=1e-9)
    assert _scalar(scan_graph, query) == expected
    assert benchmark(lambda: _scalar(scan_graph, query)) == expected
    benchmark.extra_info["nodes"] = N


# --- control cells: paths the scan work leaves alone ------------------------------------------


def _cites(i: int) -> list[int]:
    return [(i * 31 + k * 17 + 1) % N for k in range(2)]


@pytest.mark.benchmark
def test_bench_control_lookup_expand_batch(benchmark, scan_graph):
    ids = [(j * 997) % N for j in range(LOOKUPS)]
    expected = sum(2 for _ in ids)

    def run() -> int:
        total = 0
        for i in ids:
            total += len(
                scan_graph.cypher("MATCH (a:Paper {pid: $i})-[:CITES]->(b) RETURN b.pid AS p", params={"i": i})
            )
        return total

    assert run() == expected
    assert benchmark(run) == expected
    benchmark.extra_info["lookups"] = LOOKUPS


@pytest.mark.benchmark
def test_bench_control_two_hop_count(benchmark, scan_graph):
    ids = [(j * 499) % N for j in range(HOP_ANCHORS)]
    expected = sum(len(_cites(b)) for i in ids for b in _cites(i))

    def run() -> int:
        total = 0
        for i in ids:
            total += _scalar(
                scan_graph,
                f"MATCH (a:Paper {{pid: {i}}})-[:CITES]->(b)-[:CITES]->(c) RETURN count(c) AS c",
            )
        return total

    assert run() == expected
    assert benchmark(run) == expected
    benchmark.extra_info["anchors"] = HOP_ANCHORS
