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

import kglite
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


# ── durability levels: what does a commit actually cost? ─────────────
#
# `durable` picks what a committed mutation survives, and the levels differ
# only in whether each commit is barriered:
#
#   "full"   — barrier per commit (survives power loss)
#   "normal" — log written, no barrier (survives the process dying)
#   "off"    — no log
#
# The point of measuring all three side by side is to attribute the cost
# rather than assume it. Reading the code says the barrier should dominate
# "full" and that "normal" should land near "off", because a WAL frame is one
# postcard encode, one CRC32, and one `write` — but that is a *hypothesis
# derived from reading*, and the whole reason these cells exist is that it has
# to be measured before anyone quotes a number.
#
# Read `normal - off` as the true cost of logging, and `full - normal` as the
# true cost of the barrier. Both are per-commit and neither should scale with
# graph size, which is why one fixed size is enough here.
#
# Not in `test_bench_core.py` on purpose: everything in this file is outside
# the `make bench-check` tracked set, so adding cells here cannot break the
# gate. Absolute numbers are machine- and device-specific — a barrier is
# storage-hardware latency, so these are meaningless across machines and must
# never be compared against a number captured elsewhere.

#: 1k matches the scale the competitive single-insert comparison uses. The
#: per-commit cost is independent of graph size, so a second decade would add
#: runtime without adding information.
DURABILITY_BENCH_SIZE = 1_000

DURABILITY_LEVELS = ["full", "normal", "off"]


def _persisted_graph(path, level: str) -> KnowledgeGraph:
    """A file-backed graph of `DURABILITY_BENCH_SIZE` nodes, opened at `level`.

    Built and checkpointed with logging off, then reopened at the level under
    test, so the fixture's own bulk load never lands in the measurement.
    """
    seed = kglite.open(str(path), durable="off")
    seed.define_schema({"nodes": {"Item": {"primary_key": "id"}}})
    seed.add_nodes(
        pd.DataFrame(
            {
                "id": range(DURABILITY_BENCH_SIZE),
                "name": [f"item-{i}" for i in range(DURABILITY_BENCH_SIZE)],
            }
        ),
        "Item",
        "id",
        "name",
    )
    seed.save()
    seed.close()

    graph = kglite.open(str(path), durable=level)
    # Warm the id index so the measurement is the write, not a first touch.
    graph.cypher("MATCH (n:Item {id: 0}) RETURN n.id")
    return graph


@pytest.mark.benchmark
@pytest.mark.parametrize("level", DURABILITY_LEVELS)
def test_bench_single_create_by_durability_level(benchmark, tmp_path, level):
    """One `CREATE` of one node, per durability level. The headline cell."""
    graph = _persisted_graph(tmp_path / "bench.kgl", level)
    ids = iter(range(10_000_000, 1 << 30))

    def write():
        graph.cypher("CREATE (:Item {id: $i, name: 'x'})", params={"i": next(ids)})

    benchmark(write)


@pytest.mark.benchmark
@pytest.mark.parametrize("level", ["full", "normal"])
def test_bench_explicit_sync(benchmark, tmp_path, level):
    """`sync()` — the on-demand barrier.

    Under `"normal"` this is a real barrier and should cost about what
    `full`'s per-commit barrier costs; under `"full"` it returns immediately
    because every commit was already barriered. That gap is the number a
    caller needs to decide how often to call it.
    """
    graph = _persisted_graph(tmp_path / "bench.kgl", level)
    ids = iter(range(60_000_000, 1 << 30))

    def commit_then_sync():
        graph.cypher("CREATE (:Item {id: $i, name: 'x'})", params={"i": next(ids)})
        graph.sync()

    benchmark(commit_then_sync)


# ── diagnostic: what is the non-durable commit actually spending? ────
#
# `durable="off"` is the engine's own per-commit cost with no durability
# excuse, and reading the commit path turns up three candidate explanations
# that reading alone cannot rank:
#
#   1. The Cypher plan cache is keyed on graph `version`, which is bumped
#      before the lookup — so a write can never hit it and re-runs the full
#      parse + optimizer pipeline every time.
#   2. `Arc::make_mut` deep-clones the whole graph when a second
#      `Arc<DirGraph>` is alive; a lazy `ResultView` above the 32-cell
#      materialisation budget holds one.
#   3. The statement rollback checkpoint, when it falls off its skip
#      whitelist.
#
# The pair below discriminates (2) from the rest without changing a line of
# engine code: same statement, once with the result dropped immediately and
# once with it held across the next write. If `result_held` is dramatically
# worse, it is the Arc clone; if the two are close, the cost is elsewhere and
# (1) is the next suspect.
#
# This is a diagnostic, not a regression guard — it exists to attribute a
# cost, and its conclusion belongs in an issue rather than in a threshold.


@pytest.mark.benchmark
@pytest.mark.parametrize("held", [False, True], ids=["result_dropped", "result_held"])
def test_bench_create_with_lazy_result_alive(benchmark, held):
    """One `CREATE`, with and without a lazy `ResultView` pinning the graph."""
    graph = _graph(DURABILITY_BENCH_SIZE)
    ids = iter(range(50_000_000, 1 << 30))
    # Wide enough to exceed the eager-materialisation budget, so the view
    # stays lazy and keeps its own `Arc<DirGraph>` alive.
    pinned = graph.cypher("MATCH (n:Item) RETURN n.id AS i, n.name AS nm") if held else None

    def write():
        graph.cypher("CREATE (:Item {id: $i, name: 'x'})", params={"i": next(ids)})

    benchmark(write)
    # Keep `pinned` referenced past the measurement, or the interpreter is
    # free to collect it early and quietly turn this into the other case.
    assert pinned is None or pinned is not None
