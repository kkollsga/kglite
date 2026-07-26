"""Does the cost of one write depend on how big the graph is?

For an embedded primary database the answer has to be no. It used to be yes:
statement atomicity was bought with a whole-graph clone taken before every
mutating statement, so a single `SET` on a million-node graph deep-copied a
million `NodeData` values — property strings and all — before touching
anything. The undo journal (``crates/kglite/src/graph/storage/undo.rs``)
replaced that with O(changes) capture.

These benchmarks measure the shape of that curve rather than an absolute
number. Read them as ratios across `size`: flat means decoupled, linear means
the clone is back. A same-order reading across 1k → 1M is the pass condition;
the absolute microseconds are machine-specific and deliberately not asserted,
since a hard threshold here would just be a flaky duplicate of `bench-check`.

Deliberately **not** in ``test_bench_core.py``: that file is the tracked set
compared against ``baselines/current*.json`` by ``make bench-check``, and on
Linux with ``--require-exact-set``. Adding entries there would break the gate
until every platform baseline was recaptured.

Run with:

    uv run --no-sync maturin develop --release
    .venv/bin/python -m pytest tests/benchmarks/test_bench_write_scaling.py \\
        -m benchmark --benchmark-min-rounds=100 \\
        --benchmark-warmup=on --benchmark-warmup-iterations=20 -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from kglite import KnowledgeGraph

# 1k / 100k / 1M spans three orders of magnitude, which is enough for a linear
# term to be unmissable and small enough that the fixtures build in seconds.
SIZES = [1_000, 100_000, 1_000_000]

# Sizes for the one benchmark here that is *expected* to be O(V) per call, and
# which therefore stops at 100k. 1k -> 100k already moves ~20 us -> ~2,100 us,
# which shows the linear term unambiguously without a third decade.
LINEAR_MARKER_SIZES = [1_000, 100_000]

# The two benchmarks below whose per-call cost is O(V) drive `benchmark.pedantic`
# with an explicit round count rather than plain `benchmark(fn)`.
#
# This is not tidiness — plain `benchmark(fn)` hangs on these shapes. Auto-
# calibration times the *first* call to size the round count, and for these
# statements the first call is ~1000x cheaper than every call after it: the
# fixture leaves the type's id index warm, the first call invalidates it, and
# every later call pays a full rebuild. Calibration therefore concludes the
# function is microseconds-fast and schedules thousands of rounds of a
# multi-millisecond call. Observed while writing this file: 12,686 rounds of a
# 24 ms call = 5 minutes, from one test. `--benchmark-min-rounds` cannot save
# you, it only raises the floor. And a `-m benchmark` run is exempt from the
# 120 s pytest hang ceiling, so nothing interrupts it.
#
# `pedantic` skips calibration entirely and honours these numbers, so runtime is
# bounded no matter what the caller passes on the command line.
OV_ROUNDS = 20
OV_WARMUP_ROUNDS = 1


def _graph(size: int, *, primary_key: bool = True) -> KnowledgeGraph:
    """`size` nodes of one type, each with a string and a numeric property.

    The string property matters: a `Compact` `NodeData` clone allocates per
    `Value::String`, so it is what made the old checkpoint expensive in bytes
    rather than merely in count.

    A primary key is declared by default, and that is load-bearing for what
    these benchmarks measure. Without one, `CREATE` invalidates the whole
    type's id index (`id_indices.remove`), so the *next* lookup by id rebuilds
    it — an O(V) cost per statement that has nothing to do with the rollback
    checkpoint but would swamp it. Measured at 1M nodes: 34,486 us without a
    declared key, 4.1 us with one. `test_bench_id_index_invalidation_on_create`
    keeps that separate cost visible.
    """
    graph = KnowledgeGraph()
    if primary_key:
        graph.define_schema({"nodes": {"Item": {"primary_key": "id"}}})
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(size),
                "name": [f"item-{i}" for i in range(size)],
                "qty": [i % 977 for i in range(size)],
            }
        ),
        "Item",
        "id",
        "name",
    )
    # Warm the id index so the measurement is the write, not a first-touch
    # index build.
    graph.cypher("MATCH (n:Item {id: 0}) RETURN n.id")
    return graph


@pytest.fixture(scope="module")
def scaled_graphs() -> dict[int, KnowledgeGraph]:
    return {size: _graph(size) for size in SIZES}


@pytest.fixture(scope="module")
def scaled_graphs_no_pk() -> dict[int, KnowledgeGraph]:
    return {size: _graph(size, primary_key=False) for size in LINEAR_MARKER_SIZES}


@pytest.mark.benchmark
@pytest.mark.parametrize("size", SIZES)
def test_bench_single_set_by_id(benchmark, scaled_graphs, size):
    """One `SET` on one node, by primary key. The headline number."""
    graph = scaled_graphs[size]
    counter = iter(range(1, 1 << 30))

    def write():
        graph.cypher("MATCH (n:Item {id: 7}) SET n.qty = $v", params={"v": next(counter)})

    benchmark(write)
    assert graph.cypher("MATCH (n:Item {id: 7}) RETURN n.qty AS q").to_list()[0]["q"] > 0


@pytest.mark.benchmark
@pytest.mark.parametrize("size", SIZES)
def test_bench_single_create(benchmark, scaled_graphs, size):
    """One `CREATE` of one node.

    Already covered by the checkpoint-free fast path before this work, so this
    is the control: it was flat before and must stay flat.
    """
    graph = scaled_graphs[size]
    ids = iter(range(10_000_000, 1 << 30))

    def write():
        graph.cypher("CREATE (:Item {id: $i, name: 'x'})", params={"i": next(ids)})

    benchmark(write)


@pytest.mark.benchmark
@pytest.mark.parametrize("size", SIZES)
def test_bench_multi_clause_write(benchmark, scaled_graphs, size):
    """A two-clause write — the shape no fast path ever covered.

    A `SET` followed by a `CREATE` cannot be proven infallible after its first
    write, so before the journal this always paid the full clone. It is the
    clearest before/after of the sprint.
    """
    graph = scaled_graphs[size]
    ids = iter(range(20_000_000, 1 << 30))

    def write():
        graph.cypher(
            "MATCH (n:Item {id: 11}) SET n.qty = 1 CREATE (:Item {id: $i, name: 'y'})",
            params={"i": next(ids)},
        )

    benchmark(write)


@pytest.mark.benchmark
@pytest.mark.parametrize("size", SIZES)
def test_bench_single_delete(benchmark, scaled_graphs, size):
    """A create-then-delete pair, so the graph size stays roughly constant.

    Covers the delete-side journal capture (bucket positions, incident-edge
    detach) -- but read it with care, because it is **not** a clean measurement
    of that capture. `DETACH DELETE` invalidates the type's id index, so the
    trailing `CREATE`'s uniqueness probe rebuilds it, and that pre-existing
    O(V) cost dominates the number. It is here for the delete path's
    correctness-under-load rather than as evidence about the journal; the flat
    shapes above are the evidence.
    """
    graph = scaled_graphs[size]
    ids = iter(range(30_000_000, 1 << 30))

    def write():
        node_id = next(ids)
        graph.cypher("CREATE (:Item {id: $i, name: 'z'})", params={"i": node_id})
        graph.cypher(
            "MATCH (n:Item {id: $i}) DETACH DELETE n CREATE (:Item {id: $j, name: 'w'})",
            params={"i": node_id, "j": node_id + (1 << 24)},
        )

    benchmark.pedantic(write, rounds=OV_ROUNDS, iterations=1, warmup_rounds=OV_WARMUP_ROUNDS)


@pytest.mark.benchmark
@pytest.mark.parametrize("size", LINEAR_MARKER_SIZES)
def test_bench_id_index_invalidation_on_create(benchmark, scaled_graphs_no_pk, size):
    """The separate O(V) cost this file exists to keep from being mistaken for
    the checkpoint.

    On a type with no declared primary key, `CREATE` drops the type's whole id
    index rather than inserting into it, so the next `MATCH ... {id: ...}` pays
    a full rebuild. This benchmark is expected to scale with `size` -- it is a
    marker for a known open issue, not a regression guard. Compare against
    `test_bench_multi_clause_write`, which runs the same statement shape on a
    type that *does* declare a key and stays flat across all three decades.

    Capped at `LINEAR_MARKER_SIZES` for termination; see that constant.
    """
    graph = scaled_graphs_no_pk[size]
    ids = iter(range(40_000_000, 1 << 30))

    def write():
        graph.cypher(
            "MATCH (n:Item {id: 11}) SET n.qty = 1 CREATE (:Item {id: $i, name: 'k'})",
            params={"i": next(ids)},
        )

    benchmark.pedantic(write, rounds=OV_ROUNDS, iterations=1, warmup_rounds=OV_WARMUP_ROUNDS)
