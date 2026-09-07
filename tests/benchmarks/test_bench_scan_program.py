"""Deep-scan program cells that cannot sit in the core CI harness.

Outside the frozen core harness (like test_bench_labels.py). CI's perf gate
copies `test_bench_core.py` alone and benchmarks it on the published 0.13.2
wheel on the same runner, gating every cell in it at 20% (leg 1). A core cell
must therefore both run *unmodified* under 0.13.2 and be within 20% of it.
These two are not, so they live here and run under `make bench`:

- ``param_list_conversion`` isolates per-element parameter binding, and the
  current build is ~2.1x slower there than 0.13.2 (85 us vs 40/39 us on a
  1 000-element list, macOS/M4 2026-09-08) — the cost of 0.17.0's strict
  parameter validation, deliberate and not a regression to gate against a
  three-year-old wheel.
- ``hop1_deg3_mapped`` measures ~17% slower than 0.13.2 (12.3 ms vs
  10.52/10.34 ms on the same fixture, same machine): longitudinal drift that
  predates this program, and a cell that close to the 20% line is a coin-flip
  on a shared runner.

The fixtures are duplicated from the core harness rather than imported: the
core file must stay self-contained because CI copies it out of the checkout on
its own.
"""

import pandas as pd
import pytest

from kglite import KnowledgeGraph

# Default addopts deselect '-m benchmark'; CI's Python matrix installs no
# pytest-benchmark, so unmarked cells error at collection there.
pytestmark = pytest.mark.benchmark

HOP1_NODES = 100_000
HOP1_DEGREE = 3


@pytest.fixture
def bench_graph():
    """Graph with 1000 nodes and 2000 edges — the core harness's shape."""
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


def _hop1_frames():
    """100k `Person` nodes and 300k `KNOWS` edges with **uncorrelated** endpoints.

    Byte-identical to the core harness's generator, deliberately: this cell is
    only readable next to `test_bench_hop1_deg3_memory`, and the two must be
    measuring the same graph. The endpoint draw is an inlined LCG so the
    fixture is identical on every interpreter, and uncorrelated endpoints are
    the point — a row order that already groups by source is the fast case.
    """
    src: list[int] = []
    dst: list[int] = []
    state = 20_260_907
    for _ in range(HOP1_NODES * HOP1_DEGREE):
        state = (state * 1_103_515_245 + 12_345) & 0x7FFF_FFFF
        src.append(state % HOP1_NODES)
        state = (state * 1_103_515_245 + 12_345) & 0x7FFF_FFFF
        dst.append(state % HOP1_NODES)
    nodes = pd.DataFrame(
        {
            "pid": list(range(HOP1_NODES)),
            "name": [f"P{i}" for i in range(HOP1_NODES)],
            "city": [f"city{i % 50}" for i in range(HOP1_NODES)],
        }
    )
    return nodes, pd.DataFrame({"s": src, "d": dst})


@pytest.fixture(scope="module")
def hop1_graph_mapped():
    nodes, edges = _hop1_frames()
    graph = KnowledgeGraph(storage="mapped")
    graph.add_nodes(nodes, "Person", "pid", "name")
    graph.add_connections(edges, "KNOWS", "Person", "s", "Person", "d")
    return graph


@pytest.mark.benchmark
def test_bench_param_list_conversion(benchmark, bench_graph):
    """Per-element cost of binding a list parameter, with no graph work.

    `size($ids)` touches no node, so essentially the whole cell is
    `convert_query_value` running once per element — the only cell that
    isolates it. Every parameterised query in the Python binding pays this per
    element, and a per-element cost added there is invisible to the cells that
    pass no parameters (`cypher_match*`) and diluted in the ones that also
    traverse (`exists_fixed_hop` carries 50 elements against ~16 us of engine
    work, so an 8% move there is a 19% move here).

    Regression rationale: an unconditional `pd.NaT` type-name probe ahead of
    the integer arm cost +17% on a 1 000-element list before it was moved
    into the datetime arm it only ever applied to.
    """
    ids = list(range(1000))

    def query_and_consume():
        return bench_graph.cypher("RETURN size($ids) AS n", params={"ids": ids}).to_list()

    result = benchmark(query_and_consume)
    assert result[0]["n"] == 1000


@pytest.mark.benchmark
def test_bench_hop1_deg3_mapped(benchmark, hop1_graph_mapped):
    """Cross-mode control for the core harness's `hop1_deg3_memory`.

    In-memory is the core product and must not be the slower of the two — the
    2.3x inversion that reached 0.17.0 is what both cells exist to catch, so
    read this number next to that one.
    """

    def query_and_consume():
        return hop1_graph_mapped.cypher("MATCH (a:Person)-[:KNOWS]->(b) RETURN count(*) AS c").to_list()

    result = benchmark(query_and_consume)
    assert result == [{"c": HOP1_NODES * HOP1_DEGREE}]
