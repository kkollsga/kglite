"""Edge insertion with many property columns — the `add_connections` property path.

codingest measured ~94% of their control-edge load inside `add_connections`
(their own builder lever is exhausted). Nothing in the benchmark suite covered
that shape: `test_bench_core.py::test_bench_add_connections` inserts 2,000 edges
carrying **one** property column, so the per-cell property work is a rounding
error next to the fixed per-call cost. This file supplies the missing cell.

The measured shape
------------------

* 60,000 edges over 20,000 nodes — enough rows that per-cell work dominates.
* 10 property columns of mixed type (float / int / str / bool), which is the
  codingest control-edge width.
* One of those columns is ``None`` in every row. It is not decoration: an
  all-null column must reach the engine and store **nothing** (see
  ``maintain.rs::all_null_edge_property_column_stores_nothing``), and a
  regression that starts storing nulls would show up here as *cheaper* per-cell
  work in the wide cell while widening every consumer's property set. Pairing
  the timing cell with that golden is what makes the pair meaningful.

Three cells, deliberately:

``wide``      60k edges × 10 property columns — the cell under test.
``narrow``    the same 60k edges with no property columns at all. Isolates the
              fixed per-call/per-edge cost from the per-cell property cost, so
              a change to the property path can be read as a change in
              ``wide - narrow`` rather than in a number that is mostly endpoint
              resolution and edge creation.
``control``   the *node* bulk-insert path over the same corpus. `add_nodes`
              already resolves its property columns once per call, so this cell
              is untouched by any change to the connection path — it is the
              machine-drift meter CLAUDE.md's Performance protocol item 8
              requires. A control that moves means the instrument moved.

Runbook
-------

Release profile only (CLAUDE.md Performance protocol — a debug number is
invalid evidence, not a weak one)::

    uv run --no-sync maturin develop --release
    pytest tests/benchmarks/test_bench_edge_property_columns.py -m benchmark \\
        --benchmark-json=<scratch>/edge-props-run-a.json
    sleep 30
    pytest tests/benchmarks/test_bench_edge_property_columns.py -m benchmark \\
        --benchmark-json=<scratch>/edge-props-run-b.json

Read ``min`` per cell, and require the two runs to agree before believing
either. These cells are milliseconds, not microseconds, so ``min`` is the right
statistic (no once-per-event or heavy-tail exemption applies).

Every cell drives ``benchmark.pedantic`` with an explicit round count rather
than plain ``benchmark(fn)``. Auto-calibration would time the first call to
size the round count and then schedule that many rounds of a multi-millisecond
bulk load; ``test_bench_write_scaling.py`` records that pattern costing five
minutes from a single test. ``-m benchmark`` is exempt from the 120 s pytest
hang ceiling, so nothing would interrupt it.

Each round must start from a graph with no edges of the type: the first
`add_connections` for a connection type takes the initial-load path
(``skip_existence_check``), and every later one pays a per-edge existence
lookup instead. Reusing one graph across rounds would measure the update path
after round 1 and read as a bimodal distribution. Hence ``setup=``, whose cost
pytest-benchmark excludes from the timing.
"""

import pandas as pd
import pytest

from kglite import KnowledgeGraph

N_NODES = 20_000
N_EDGES = 60_000

ROUNDS = 15
WARMUP_ROUNDS = 2


def _edge_frame() -> pd.DataFrame:
    """Edges plus their property columns.

    Mixed width and type on purpose: the per-cell cost is a ``Value``
    materialization plus a key write, and strings allocate where ints do not.
    ``unused`` is ``None`` in every row — see the module docstring.
    """
    # Every (src, tgt) pair distinct. `src` cycles with period N_NODES and so
    # does `i * 7919`, so without the block term rows i and i + N_NODES would
    # be the *same* edge — 20k distinct pairs inserted three times each, which
    # is not the fan-out shape this cell claims to measure.
    src = [i % N_NODES for i in range(N_EDGES)]
    tgt = [(i * 7919 + 13 + (i // N_NODES) * 6661) % N_NODES for i in range(N_EDGES)]
    return pd.DataFrame(
        {
            "src": src,
            "tgt": tgt,
            "weight": [float(i % 997) for i in range(N_EDGES)],
            "score": [float(i % 31) / 7.0 for i in range(N_EDGES)],
            "confidence": [float(i % 101) / 101.0 for i in range(N_EDGES)],
            "rank": [i % 50 for i in range(N_EDGES)],
            "depth": [i % 7 for i in range(N_EDGES)],
            "kind": [f"kind_{i % 12}" for i in range(N_EDGES)],
            "label": [f"label_{i % 500}" for i in range(N_EDGES)],
            "detail": [f"detail text for edge {i}" for i in range(N_EDGES)],
            "is_primary": [(i % 3) == 0 for i in range(N_EDGES)],
            "unused": [None] * N_EDGES,
        }
    )


def _node_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nid": list(range(N_NODES)),
            "name": [f"Node_{i}" for i in range(N_NODES)],
            "bucket": [f"bucket_{i % 64}" for i in range(N_NODES)],
        }
    )


# Built once. `add_connections` / `add_nodes` never mutate their input frame,
# and rebuilding 60k rows of Python objects per round would dwarf the setup
# budget for no gain.
EDGES_WIDE = _edge_frame()
EDGES_NARROW = EDGES_WIDE[["src", "tgt"]]
NODES = _node_frame()

PROPERTY_COLUMNS = [c for c in EDGES_WIDE.columns if c not in ("src", "tgt")]


def _graph_with_nodes() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.add_nodes(NODES, "Item", "nid", "name")
    return graph


def test_edge_frame_shape_oracle():
    """The corpus is what the timing cells claim it is.

    Runs in the default suite (no timing, valid in any profile). Without it a
    silently-narrowed frame — a dropped column, a renamed one — would turn the
    wide cell into a second narrow cell and certify health unconditionally.
    """
    assert len(PROPERTY_COLUMNS) == 10, PROPERTY_COLUMNS
    assert "unused" in PROPERTY_COLUMNS
    assert EDGES_WIDE["unused"].isna().all()
    assert len(EDGES_WIDE) == N_EDGES
    assert list(EDGES_NARROW.columns) == ["src", "tgt"]
    # No repeated edges — see `_edge_frame`.
    assert len(set(zip(EDGES_WIDE["src"], EDGES_WIDE["tgt"], strict=True))) == N_EDGES

    # And the all-null column must not become an edge property. This is the
    # Python-side face of the engine golden in
    # `maintain.rs::all_null_edge_property_column_stores_nothing`.
    graph = KnowledgeGraph()
    graph.add_nodes(pd.DataFrame({"nid": [1, 2], "name": ["a", "b"]}), "Item", "nid", "name")
    graph.add_connections(
        pd.DataFrame({"src": [1], "tgt": [2], "weight": [1.5], "unused": [None]}),
        "LINKS",
        "Item",
        "src",
        "Item",
        "tgt",
    )
    keys = graph.cypher("MATCH ()-[r:LINKS]->() RETURN keys(r) AS k").to_dicts()[0]["k"]
    assert "weight" in keys
    assert "unused" not in keys, "an all-None column must store nothing"


@pytest.mark.benchmark
def test_bench_add_connections_wide_properties(benchmark):
    """60k edges × 10 property columns — the cell C2b exists to move."""

    def run(graph: KnowledgeGraph):
        graph.add_connections(EDGES_WIDE, "LINKS", "Item", "src", "Item", "tgt")

    benchmark.pedantic(
        run,
        setup=lambda: ((_graph_with_nodes(),), {}),
        rounds=ROUNDS,
        warmup_rounds=WARMUP_ROUNDS,
    )


@pytest.mark.benchmark
def test_bench_add_connections_no_properties(benchmark):
    """The same 60k edges with no property columns — the fixed-cost floor."""

    def run(graph: KnowledgeGraph):
        graph.add_connections(EDGES_NARROW, "LINKS", "Item", "src", "Item", "tgt")

    benchmark.pedantic(
        run,
        setup=lambda: ((_graph_with_nodes(),), {}),
        rounds=ROUNDS,
        warmup_rounds=WARMUP_ROUNDS,
    )


@pytest.mark.benchmark
def test_bench_add_nodes_control(benchmark):
    """Unchanged-path control: the node bulk-insert over the same corpus.

    `add_nodes` already resolves and interns its property columns once per
    call, so nothing in the connection path can move this number. If it moves,
    the machine moved — re-measure rather than bisect.
    """

    def run():
        graph = KnowledgeGraph()
        graph.add_nodes(NODES, "Item", "nid", "name")

    benchmark.pedantic(run, rounds=ROUNDS, warmup_rounds=WARMUP_ROUNDS)
