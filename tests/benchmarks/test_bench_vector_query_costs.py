"""Two bug-shaped perf signals from the docs-blind evaluation, characterized.

Both are single-comparison signals from the Java demo; measurement is what
turns them from signal into confirmed-defect-or-cleared (Performance protocol).

**1. List-subscript cost.** ``n.list[i]`` was ~22 us/element in the demo,
~150x a scalar read, directly on the vector-query path (``reduce(i IN
range(0,W) | ... n.emb[i] ...)``). The diagnostic is a *width sweep*: if the
per-element cost grows with the list width ``W``, subscript is cloning the
whole list per access (O(W) per element, O(W^2) per row). A scalar-reread
control isolates that from reduce-loop overhead.

**2. Property-width traversal tax.** A 3-hop count-only query went 11 ms ->
70 ms (~6x) when unrelated scalar properties were added to nodes. The
diagnostic is a *property-width sweep* on a fixed traversal that reads no
properties: if count-only traversal time scales with property count, the
traversal is materializing properties it never uses.

Release build required. Run explicitly::

    pytest tests/benchmarks/test_bench_vector_query_costs.py -m benchmark -v -s
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import kglite

# ---- List-subscript width sweep -------------------------------------------

SUBSCRIPT_NODES = 2_000
SUBSCRIPT_WIDTHS = (16, 64, 256, 384)
SUBSCRIPT_ROUNDS = 30
SUBSCRIPT_WARMUP = 5


@pytest.fixture(scope="module", params=SUBSCRIPT_WIDTHS, ids=[f"w{w}" for w in SUBSCRIPT_WIDTHS])
def subscript_graph(request):
    """One node type whose `emb` list is exactly `width` elements long.

    The stored list length is the swept variable — that is what makes this a
    valid test of a per-access whole-list clone (O(width) per element ->
    O(width) per-element cost).  A scalar control property rides along.
    """
    width = request.param
    rng = np.random.default_rng(20_260_811)
    emb = rng.standard_normal((SUBSCRIPT_NODES, width), dtype=np.float32)
    graph = kglite.KnowledgeGraph()
    rows = [
        {"id": i, "title": f"n{i}", "emb": emb[i].tolist(), "scalar": float(emb[i, 0])} for i in range(SUBSCRIPT_NODES)
    ]
    # CREATE via UNWIND so `emb` lands as a native Value::List property.
    graph.cypher(
        "UNWIND $rows AS row CREATE (:Note {id: row.id, title: row.title, emb: row.emb, scalar: row.scalar})",
        params={"rows": rows},
    )
    assert graph.cypher("MATCH (n:Note) RETURN count(n) AS c").scalar() == SUBSCRIPT_NODES
    return width, graph


@pytest.mark.benchmark
def test_bench_list_subscript_width(benchmark, subscript_graph):
    """reduce over n.emb[i] for i in 0..width-1 over a width-length stored list."""
    width, graph = subscript_graph
    q = (
        f"MATCH (n:Note) "
        f"WITH n, reduce(s = 0.0, i IN range(0, {width - 1}) | s + n.emb[i]) AS score "
        f"RETURN n.id AS id, score ORDER BY score DESC LIMIT 5"
    )

    def run():
        return graph.cypher(q).to_list()

    rows = benchmark.pedantic(run, rounds=SUBSCRIPT_ROUNDS, iterations=1, warmup_rounds=SUBSCRIPT_WARMUP)
    assert len(rows) == 5
    total_s = benchmark.stats.stats.min
    accesses = SUBSCRIPT_NODES * width
    benchmark.extra_info["statistic"] = "min"
    benchmark.extra_info["stored_list_len"] = width
    benchmark.extra_info["nodes"] = SUBSCRIPT_NODES
    benchmark.extra_info["us_per_element"] = total_s * 1e6 / accesses


@pytest.mark.benchmark
def test_bench_scalar_reread_width(benchmark, subscript_graph):
    """Control: reduce rereading one scalar property `width` times (no subscript)."""
    width, graph = subscript_graph
    q = (
        f"MATCH (n:Note) "
        f"WITH n, reduce(s = 0.0, i IN range(0, {width - 1}) | s + n.scalar) AS score "
        f"RETURN n.id AS id, score ORDER BY score DESC LIMIT 5"
    )

    def run():
        return graph.cypher(q).to_list()

    rows = benchmark.pedantic(run, rounds=SUBSCRIPT_ROUNDS, iterations=1, warmup_rounds=SUBSCRIPT_WARMUP)
    assert len(rows) == 5
    total_s = benchmark.stats.stats.min
    accesses = SUBSCRIPT_NODES * width
    benchmark.extra_info["statistic"] = "min"
    benchmark.extra_info["stored_list_len"] = width
    benchmark.extra_info["nodes"] = SUBSCRIPT_NODES
    benchmark.extra_info["us_per_element"] = total_s * 1e6 / accesses


# ---- Property-width traversal tax -----------------------------------------

TRAVERSAL_NODES = 3_000
FANOUT = 3
TRAVERSAL_WIDTHS = (0, 8, 32, 64)
TRAVERSAL_ROUNDS = 30
TRAVERSAL_WARMUP = 5


def _build_traversal_graph(prop_width: int) -> kglite.KnowledgeGraph:
    """N nodes, each with `prop_width` extra scalar props the query never reads."""
    ids = np.arange(TRAVERSAL_NODES, dtype=np.int64)
    frame = {"id": ids, "title": [f"n{i}" for i in ids]}
    rng = np.random.default_rng(20_260_812)
    for k in range(prop_width):
        frame[f"e{k}"] = rng.standard_normal(TRAVERSAL_NODES).astype(np.float64)
    graph = kglite.KnowledgeGraph()
    graph.add_nodes(pd.DataFrame(frame), "N", "id", "title")
    # Ring + fanout edges: node i -> (i+1..i+FANOUT) mod N. Bounded 3-hop count.
    src, dst = [], []
    for i in range(TRAVERSAL_NODES):
        for k in range(1, FANOUT + 1):
            src.append(i)
            dst.append((i + k) % TRAVERSAL_NODES)
    graph.add_connections(pd.DataFrame({"src": src, "dst": dst}), "R", "N", "src", "N", "dst")
    return graph


@pytest.fixture(scope="module", params=TRAVERSAL_WIDTHS, ids=[f"p{w}" for w in TRAVERSAL_WIDTHS])
def traversal_graph(request):
    return request.param, _build_traversal_graph(request.param)


@pytest.mark.benchmark
def test_bench_traversal_property_width(benchmark, traversal_graph):
    """3-hop count-only traversal; property count is the swept variable."""
    prop_width, graph = traversal_graph
    q = "MATCH (a:N)-[:R]->(b)-[:R]->(c)-[:R]->(d) RETURN count(*) AS c"

    def run():
        return graph.cypher(q).scalar()

    count = benchmark.pedantic(run, rounds=TRAVERSAL_ROUNDS, iterations=1, warmup_rounds=TRAVERSAL_WARMUP)
    assert count == TRAVERSAL_NODES * FANOUT**3
    benchmark.extra_info["statistic"] = "min"
    benchmark.extra_info["property_width"] = prop_width
    benchmark.extra_info["nodes"] = TRAVERSAL_NODES
    benchmark.extra_info["three_hop_paths"] = count
