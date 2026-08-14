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

import os

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


# ── the shape gap: fresh graphs are not the graphs users write to ────
#
# Every fixture above builds a graph and never persists it, so every cell above
# measures the `Compact` (row) write path. That is the gap this whole file
# missed: `save()` consolidates each type into a columnar master
# `Arc<ColumnStore>`, and from then on a single-row `SET` takes a completely
# different route — one that deep-clones the touched type's whole store per
# statement, because the undo journal holds the second `Arc` handle and
# `Arc::make_mut` therefore forks. The cost is O(rows_of_type × cols) per
# statement and it never amortises.
#
# A full write-perf program ran against this file and reported flat curves,
# which were true and beside the point: fresh-only fixtures cannot see a cost
# that only exists after a save. Measured on the 0.15.14 wheel, single-row SET
# at 50k x 12:
#
#     fresh  4.3 us  |  post-save  328 us  |  mapped ~3,020 us
#
# The three cells below close that gap. They are the Phase-3 verification
# targets of the shape-convergence program (`dev-docs/plans/`): after the
# cell-grained undo journal lands, post-save must come back within 2x of fresh
# or the program stops for diagnosis.
#
# Not tracked by `make bench-check`, like everything else in this file — the
# gate collects `test_bench_core.py` only, and on Linux it additionally runs
# `--require-exact-set`, which *errors* on a benchmark present in the run but
# absent from the baseline. New cells therefore belong here, not there, until a
# release-boundary rebaseline promotes them.

#: The Phase-0 grid's headline cell. Wide enough that the per-statement store
#: clone is unmistakable (12 int columns x 50k rows ~ 6 MB copied to write one
#: cell), small enough that the fixture builds in well under a second.
SHAPE_SIZE = 50_000
SHAPE_COLS = 12


def _wide_graph(size: int = SHAPE_SIZE, cols: int = SHAPE_COLS) -> KnowledgeGraph:
    """`size` nodes with `cols` int properties, plus id and name.

    Int columns on purpose: they are what a `ColumnStore` types and packs, so
    the clone they drive is the cheapest-per-row version of the defect. A
    string-heavy shape would report a larger number for reasons that are about
    strings rather than about the write path.
    """
    graph = KnowledgeGraph()
    graph.define_schema({"nodes": {"Item": {"primary_key": "id"}}})
    data: dict = {"id": range(size), "name": [f"item-{i}" for i in range(size)]}
    for c in range(cols):
        data[f"p{c}"] = [(i + c) % 977 for i in range(size)]
    graph.add_nodes(pd.DataFrame(data), "Item", "id", "name")
    graph.cypher("MATCH (n:Item {id: 0}) RETURN n.id")
    return graph


@pytest.mark.benchmark
def test_bench_wide_set_fresh(benchmark):
    """Single-row `SET` on a never-saved wide graph — the control.

    Same statement, same shape, same size as the two cells below; the only
    difference is that this graph has never been through a consolidation pass.
    Without it the post-save number has nothing to be a ratio *of*, and a
    machine-wide slowdown would be indistinguishable from a regression. The
    ratio is the point: it used to separate two storage shapes and now reports
    only what consolidation itself costs a later write.
    """
    graph = _wide_graph()
    counter = iter(range(1, 1 << 30))

    def write():
        graph.cypher("MATCH (n:Item {id: 7}) SET n.p0 = $v", params={"v": next(counter)})

    benchmark(write)
    assert graph.graph_info()["columnar_total_rows"] == SHAPE_SIZE, (
        "the control must carry its rows in the column store like the cells it anchors"
    )


@pytest.mark.benchmark
def test_bench_wide_set_after_save(benchmark, tmp_path):
    """Single-row `SET` after `save()` — the cell the fresh fixtures hid.

    `save()` runs the consolidation pass that rebuilds every column store, so
    the type ends up behind a freshly built master `Arc<ColumnStore>`. The graph
    handle is kept and written through afterwards, so the timed statement is an
    ordinary point write on an ordinary application graph — one that has been
    checkpointed once.
    """
    graph = _wide_graph()
    graph.save(str(tmp_path / "wide.kgl"))
    assert graph.graph_info()["columnar_total_rows"] == SHAPE_SIZE, (
        "save() must have consolidated the type, or this cell measures nothing"
    )
    counter = iter(range(1, 1 << 30))

    def write():
        graph.cypher("MATCH (n:Item {id: 7}) SET n.p0 = $v", params={"v": next(counter)})

    benchmark(write)


@pytest.mark.benchmark
def test_bench_wide_set_mapped(benchmark, tmp_path):
    """Single-row `SET` on a mapped-mode graph — the worst arm, and a contract
    defect as well as a cost.

    Mapped mode holds columns in mmap-backed storage, so the per-statement
    store clone copies them onto the *heap* (`MmapOrVec::clone` always yields
    the `Heap` variant). The write is therefore both the slowest of the three
    and the one that silently defeats `set_memory_limit`; the contract half is
    pinned separately by
    `tests/test_memory_management.py::TestSpillContractAcrossWrites`.

    The fixture is built and saved through a plain handle first, then reopened
    with `storage="mapped"`, so the bulk load never lands in the measurement.
    """
    path = tmp_path / "mapped.kgl"
    seed = _wide_graph()
    seed.save(str(path))
    seed.close()

    graph = kglite.open(str(path), storage="mapped")
    assert graph.graph_info()["storage_mode"] == "mapped"
    # Warm the id index so the measurement is the write, not a first touch.
    graph.cypher("MATCH (n:Item {id: 0}) RETURN n.id")
    counter = iter(range(1, 1 << 30))

    def write():
        graph.cypher("MATCH (n:Item {id: 7}) SET n.p0 = $v", params={"v": next(counter)})

    benchmark(write)


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
# ── correction, 2026-07-27: what `off` is, and what it is not ────────
#
# A competitive benchmark read the `off` cell at 1.8 us and called it
# impossible, on the grounds that it beat the tracked `cypher_match` READ
# baseline of 6.2 us and a write must do strictly more work than a read.
# Both halves of that are wrong, and the correction is worth keeping because
# the cell invites the mistake:
#
#   * The comparison is not like-for-like. `test_bench_cypher_match` is
#     `MATCH (n:Item) RETURN n.title, n.value LIMIT 100` *without* `.to_list()`
#     — a lazy 100-row scan, tracked at 5.42 us min / 6.22 us mean. Its
#     materialized sibling is 19.08 us min. A single-node insert with a warm
#     plan cache producing no rows is simply less work than planning and
#     scanning 100 rows. 1.8 us (a min) against 6.2 us (a mean) compares two
#     different statistics of two different workloads.
#   * Nothing is being skipped. At `off`, `logs()` is false, so `setup_durable`
#     never runs (`lib.rs:416-418`), the backend is never wrapped in
#     `RecordingGraph`, and `flush_wal` returns immediately on
#     `durable.is_none()` (`graph/mod.rs:405-407`). The mutation itself is
#     applied in place and synchronously at every level; there is no write
#     buffer, no deferred commit, no background thread, and no coalescing
#     anywhere in `wal.rs`. At `off` the write is a petgraph insert and index
#     bookkeeping, full stop.
#
# So the number is real. What was unsound is the FRAMING: `off` is not a
# cheaper commit, it is **no commit at all**, and there is no per-commit
# durable work at `off` to measure. Its durability boundary is `save()`.
# Labelling it a "durability level" alongside two levels that do write to disk
# invites reading 1.8 us as "what a commit costs at off".
#
# Two cells below now hold that framing in place rather than leaving it to a
# comment: `test_bench_unlogged_write_control` shows a plain in-memory graph
# produces the same number (so file-backing buys nothing at `off`), and
# `test_durable_off_loses_unsaved_writes` is the ordinary test proving what the
# 1.8 us did *not* buy. The `wal_bytes` guard on the headline cell pins each
# level to the configuration it claims.
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


def _wal_bytes(path) -> int:
    """Size of the `<path>-wal` sidecar, or 0 when no log exists.

    `wal_path` is `<checkpoint>-wal` (`wal.rs:520-524`). This is the only
    Python-visible window onto the WAL — `DurableState` exposes no getter for
    `wal`, `next_lsn` or `level` — and it is enough to prove which level a cell
    actually ran at.
    """
    try:
        return os.path.getsize(str(path) + "-wal")
    except FileNotFoundError:
        return 0


@pytest.mark.benchmark
@pytest.mark.parametrize("level", DURABILITY_LEVELS)
def test_bench_single_create_by_durability_level(benchmark, tmp_path, level):
    """One `CREATE` of one node, per durability level. The headline cell.

    The trailing assertion is a vacuity guard, not a correctness test. A cell
    that silently ran at the wrong level would report a plausible number under
    a wrong label, and the 2026-07-27 methodology note is emphatic that this is
    how benchmark harness defects actually present — every one caught in that
    run was found because a number contradicted the configuration it claimed,
    never because it looked implausible. A config-key collision there nearly
    erased the headline finding while the table looked entirely normal.

    At `off` the sidecar is never created, so a non-zero reading means the log
    is on and the number is not an `off` number.
    """
    path = tmp_path / "bench.kgl"
    graph = _persisted_graph(path, level)
    ids = iter(range(10_000_000, 1 << 30))

    def write():
        graph.cypher("CREATE (:Item {id: $i, name: 'x'})", params={"i": next(ids)})

    benchmark(write)

    logged = _wal_bytes(path)
    if level == "off":
        assert logged == 0, f"durable='off' must write no WAL; found {logged} bytes"
    else:
        assert logged > 0, f"durable='{level}' must write a WAL; sidecar is empty or absent"


@pytest.mark.benchmark
def test_bench_unlogged_write_control(benchmark):
    """The same `CREATE` on a plain in-memory graph — the control for `off`.

    `durable="off"` opens no log and never wraps the backend, so a file-backed
    graph at `off` should be indistinguishable from one that was never given a
    path. This cell is what turns that from a claim into a reading: if it
    differs materially from `test_bench_single_create_by_durability_level
    [off]`, then something about file-backing costs per-write time and the
    `off` row means something other than what it says.

    It also gives the `off` number a name that cannot be misread. There is no
    per-commit durable work at `off` to measure — the honest cost of durability
    there is `save()` amortised over N writes, and `save()` is O(graph)
    (tracked separately as `test_bench_columnar_save_kgl`, 300 us min at 1k and
    `fsync=False`).
    """
    graph = _graph(DURABILITY_BENCH_SIZE)
    ids = iter(range(70_000_000, 1 << 30))

    def write():
        graph.cypher("CREATE (:Item {id: $i, name: 'x'})", params={"i": next(ids)})

    benchmark(write)


def test_durable_off_loses_unsaved_writes(tmp_path):
    """What the 1.8 us at `durable="off"` did not buy.

    Not a benchmark — an ordinary test, and deliberately so. The `off` cell's
    speed is only interpretable next to the guarantee it declines, and a
    comment saying "off loses your data" is worth less than a test that
    demonstrates it. Reopening is the closest in-process stand-in for the
    process dying: no `save()`, so no checkpoint, so nothing to recover from.
    """
    path = tmp_path / "off.kgl"
    graph = _persisted_graph(path, "off")
    graph.cypher("CREATE (:Item {id: 999000, name: 'lost'})")
    assert graph.cypher("MATCH (n:Item {id: 999000}) RETURN count(n) AS c").scalar() == 1

    # The first handle is deliberately NOT closed. `close()` performs a full
    # save, which would persist the very write this test exists to lose — and
    # an assertion taken after it would be vacuous. Read the file as it stands
    # on disk instead: the last checkpoint is `_persisted_graph`'s `seed.save()`
    # from before the write. `lock=False` because the live handle still holds
    # the writer lease.
    on_disk = kglite.open(str(path), durable="off", lock=False)
    survived = on_disk.cypher("MATCH (n:Item {id: 999000}) RETURN count(n) AS c").scalar()

    assert survived == 0, (
        "durable='off' wrote to a checkpoint it should not have; the speed of "
        "the 'off' benchmark cell is only meaningful because this write is lost"
    )
    assert graph.cypher("MATCH (n:Item {id: 999000}) RETURN count(n) AS c").scalar() == 1


# ── repaired 2026-07-27: `sync()` is timed alone, not after a create ──
#
# This cell used to time `CREATE` + `sync()` together, and at `full` it read
# **1439 us — below the same file's create-only cost at `full`**. Create+sync
# cannot be cheaper than create, so the pair was reported as an impossible
# measurement. Reading the source says the cell was not measuring an ordering
# at all:
#
#   * `sync()` at `full` is a **hard early return** (`kg_core.rs:711-713`). It
#     touches no state — no flush, no barrier, no flag — because every commit
#     was already barriered. `SyncMode` is fixed at `Wal::open` and never
#     mutated afterwards, and nothing anywhere caches, batches or amortises
#     across commits.
#   * `Wal::append` is unconditional (`wal.rs:704-711`): no size threshold, no
#     timer, no coalescing window. So at `full`, `CREATE` + `sync()` performs
#     *exactly* the same work as `CREATE` alone.
#
# The old cell therefore duplicated `test_bench_single_create_by_durability_
# level[full]` by construction and could never carry information — and 1439 vs
# ~3400 us is F_FULLFSYNC variance between two runs of identical work, not an
# ordering. A device-level barrier is the noisiest thing this suite measures.
#
# The repair is to move the `CREATE` into an untimed `pedantic` setup so the
# timed region is the barrier and nothing else. `full` should now read ~0
# (pinning the early return) and `normal` should read one barrier — which is
# the number a caller actually needs to decide how often to call it, and was
# previously buried under a create.
#
# `off` is absent from the parametrisation because `sync()` raises `ValueError`
# there (`kg_core.rs:693-701`) rather than silently doing nothing.


@pytest.mark.benchmark
@pytest.mark.parametrize("level", ["full", "normal"])
def test_bench_explicit_sync(benchmark, tmp_path, level):
    """`sync()` alone — the on-demand barrier, with the commit untimed.

    Under `"normal"` this is one real `F_FULLFSYNC`; under `"full"` it is an
    early return. The gap between the two arms is the cost of taking a
    power-safe point on demand, and it is what makes `"normal"` adoptable
    rather than merely fast.

    See the comment above before changing the shape of this cell — folding the
    `CREATE` back into the timed region is what made its predecessor
    uninterpretable.
    """
    graph = _persisted_graph(tmp_path / "bench.kgl", level)
    ids = iter(range(60_000_000, 1 << 30))

    def setup():
        graph.cypher("CREATE (:Item {id: $i, name: 'x'})", params={"i": next(ids)})
        return (), {}

    benchmark.pedantic(graph.sync, setup=setup, rounds=OV_ROUNDS, iterations=1, warmup_rounds=OV_WARMUP_ROUNDS)


# ── the lazy-`ResultView` diagnostic moved, 2026-07-27 ───────────────
#
# A `test_bench_create_with_lazy_result_alive` pair used to sit here, aimed at
# the `Arc::make_mut` whole-graph clone that a held `ResultView` forces. It
# reported **no difference**, and that null was wrong twice over:
#
#   1. **Under-powered.** It ran only at `DURABILITY_BENCH_SIZE` (1k), where
#      the clone is ~1000x too small to separate from a write. The clone is
#      O(V+E); at 100k it is ~5 ms, at 1M ~28 ms.
#   2. **Structurally blind.** The reference was taken ONCE, outside the
#      measurement. The clone fires on the first write after a second
#      `Arc<DirGraph>` appears and once only — so `min`, `p50` and `p95` all
#      saw post-clone rounds and read healthy. Only `max` could see it.
#
# It has been rewritten and moved to `test_bench_fast_write_path.py::
# test_bench_first_write_after_reference`, which takes the reference in an
# untimed per-round `setup` and runs at 1k and 100k. Consolidated rather than
# duplicated: that file also owns the `journal_covers` cells, and both defects
# are answers to "why did this one write cost milliseconds".
#
# Left here as a signpost because a null result that was believed for a while
# is worth being able to trace.
