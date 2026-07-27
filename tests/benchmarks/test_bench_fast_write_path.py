"""Does an ordinary user action silently move every later write onto a
whole-graph clone?

Two independent defects live here. Both make a single write O(V+E) instead of
O(changes), both are invisible at small scale, and both were found by an
external competitive benchmark rather than by this suite — because nothing in
this suite was shaped to see them.

────────────────────────────────────────────────────────────────────────────
Defect A — the ``journal_covers`` vetoes (fixed in PR #86, ``59478b43``)
────────────────────────────────────────────────────────────────────────────

Until that fix, ``journal_covers`` had two independent veto terms, each
permanently tripped by an ordinary user action, each sending every subsequent
mutating statement through ``fork_transaction()`` — a full copy of the node
and edge stores — for the rest of the session:

* **``save()``** calls ``enable_columnar()``, which permanently populates
  ``DirGraph.column_stores``. The old gate required that field to be *empty*.
  Measured at 100k (2026-07-27): a single insert went **0.0062 ms → 3.0931 ms
  and stayed there** — a ~500× permanent degradation from one ``save()``. A
  second ``save()`` did not clear it. ``min ≈ mean``, so *every* write paid.
* **any user index** vetoed the same path. Measured at 100k, ``durable=False``,
  no ``save()``: **0.0127 ms → 15.22 ms**, reported as a 6–10× range rather
  than a point estimate because the indexed cells were 17% noisy (two takes of
  identical code gave 21.18 and 37.77 ms).

With neither trigger, the same 100k insert costs **3.668 ms against SQLite's
3.666 ms — exact parity**. So this was not an architectural deficit; it was a
gate that excluded every realistic application graph.

After PR #86 ``journal_covers`` (``rollback.rs:195-198``) has **exactly one**
term left — ``graph.graph.supports_undo_journal()``. Neither ``save()`` nor any
index family vetoes it any more. The only remaining way to force the clone is
the *backend*: ``Memory`` journals, ``Recording`` forwards to its inner
backend, and ``Mapped`` / ``Disk`` return ``false``
(``storage/backend.rs:110-116``). A reintroduced veto is therefore a new
conjunct in a one-conjunct predicate — cheap to add by accident, and these
cells are what would notice.

⚠ **The statement shape is load-bearing — see the STATEMENT SHAPE comment
below.** A bare single-node ``CREATE`` cannot see this defect at all, in
either direction, because it never opens a checkpoint in the first place.

**Why the existing suite could not see it.** The Rust guard
(``rollback_tests.rs::journalled_statements_copy_zero_nodes``) ran only on
``seeded()`` — a graph with no column stores and no indexes, which cannot take
the clone path *no matter what the gate says*. PR #86 added ``seeded_columnar``
and ``seeded_indexed`` arms, and those are the authoritative, deterministic
guard: they assert ``backend_clone_nodes() == 0``, an exact oracle for *which
path ran*. This file cannot do that — ``backend_clone_nodes`` is ``#[cfg(test)]
pub(crate)`` on a thread-local (``storage/backend.rs:42``) and is not reachable
from Python by any route. **These benchmarks are the cost/scaling complement,
not a replacement**: they measure that the fast path is still *cheap* and still
*flat*, and they cover the one combination the Rust arms do not — saved **and**
indexed at once, through a real ``save()`` rather than a bare
``enable_columnar()``.

────────────────────────────────────────────────────────────────────────────
Defect B — ``Arc::make_mut`` deep clone on a merely-held reference
────────────────────────────────────────────────────────────────────────────

Holding a returned result view across a write triggers one full graph clone.
Measured at 100k (2026-07-27): mean 0.0561 ms, p50 0.0044 ms, **max 5.17 ms**,
with full recovery once the reference is released; ``freeze()`` held behaves
the same (max 5.83 ms); ~28 ms at 1M. No threading, no snapshot API, no
explicit copy — merely keeping the ``ResultView`` a query returned.

See the measurement-trap comment above that section. It is the reason this
defect survived a benchmark suite that already had a cell pointed at it.

────────────────────────────────────────────────────────────────────────────
How to read these, and the scale rule that governs the sizes
────────────────────────────────────────────────────────────────────────────

Like ``test_bench_write_scaling.py``, this file asserts no timing threshold —
the signal is a **ratio**, and there are two of them:

1. **Across variants at one size.** ``saved`` / ``indexed`` / ``saved_indexed``
   must each stay close to ``fresh``. A variant that is orders of magnitude
   worse has had its fast path vetoed.
2. **Across sizes within one variant.** Every variant must be flat from 1k to
   100k. A reintroduced O(V+E) term shows up as *divergence between the sizes*,
   which is a far more robust detector than any absolute number: it survives a
   change of machine, and it cannot be explained away as thermal noise.

Nothing here is in the ``make bench-check`` tracked set. That gate runs
``tests/benchmarks/test_bench_core.py`` only (see ``Makefile:85``), so a new
file cannot perturb it — and, for defect B specifically, that gate compares on
``--metric min``, which is structurally blind to it (again, see the trap
comment).

Run with::

    uv run --no-sync maturin develop --release
    .venv/bin/python -m pytest tests/benchmarks/test_bench_fast_write_path.py \\
        -m benchmark -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from kglite import KnowledgeGraph

# ⚠ THE 100k CELL IS THE ONLY ONE THAT CAN SEE DEFECT A. DO NOT "SIMPLIFY"
# THIS DOWN TO 1k.
#
# A fixed per-operation cost masks any multiplicative defect below the scale at
# which the defect exceeds it. In the 2026-07-27 competitive run the index
# penalty was *completely invisible* at 1k — 3.597 ms with no index vs 3.759 ms
# with one, well inside noise — because the 3.4 ms F_FULLFSYNC floor swamped a
# defect that is unmissable two decades up. A suite run only at small scale
# would have certified this write path as healthy, which is exactly what
# happened.
#
# 1k is kept as the *control*, not as the measurement: its job is to be the
# denominator that makes the 100k divergence legible. Deleting it is as wrong
# as deleting 100k.
#
# 1M is deliberately omitted. The defect is already unambiguous at two decades,
# and four variants x 1M nodes is minutes of fixture build for no added signal.
SIZES = [1_000, 100_000]

#: The four corners of the ``journal_covers`` gate. ``saved_indexed`` is the
#: one a real application actually is, and the one combination the Rust-side
#: arms never build.
VARIANTS = ["fresh", "saved", "indexed", "saved_indexed"]

# Every cell here drives `benchmark.pedantic` with an explicit round count
# rather than plain `benchmark(fn)`.
#
# This is a termination requirement, not a style choice. `pytest-benchmark`
# auto-calibration times the first call to size the round count, and this file
# exists precisely to compare a world where a write costs ~6 us against a world
# where it costs ~15-38 ms. Calibration performed in the cheap world and then
# applied in the expensive one schedules thousands of rounds of a
# multi-millisecond call; `test_bench_write_scaling.py` records a real instance
# of that pattern costing 5 minutes from a single test. A `-m benchmark` run is
# exempt from the 120 s pytest hang ceiling (`tests/conftest.py`), so nothing
# would interrupt it.
#
# `pedantic` skips calibration entirely, so runtime is bounded in BOTH worlds:
# ~2 ms per cell when healthy, ~2 s per cell when the defect is back.
ROUNDS = 50
WARMUP_ROUNDS = 5

#: Rounds for the reference-held cells (defect B). Lower, because each round
#: forces a whole-graph copy by construction when the defect is present, and
#: because a pinned reference keeps the pre-clone copy alive.
REF_ROUNDS = 20
REF_WARMUP_ROUNDS = 2


# ⚠ STATEMENT SHAPE — THE SINGLE EASIEST WAY TO SILENTLY DISABLE THIS FILE.
#
# `session/execute.rs:390-394` opens a `StatementCheckpoint` only when
# `can_skip_rollback_checkpoint(...)` is false. That whitelist
# (`execute.rs:309-337`) returns true — meaning NO CHECKPOINT IS OPENED AT ALL,
# `StatementCheckpoint::None` — for a query that is exactly one `Clause::Create`
# with a single-node pattern, on a backend with
# `supports_checkpoint_free_mutation()` (`backend.rs:94-96`: `Memory` only).
#
# So `CREATE (:Item {id: 1})` on an in-memory graph takes the same free path
# before and after PR #86, on every variant. A benchmark built on it reads
# identical everywhere and certifies health unconditionally — the exact
# failure this file was written to end.
#
# `MERGE` is used instead: still a one-node insert, but not on the whitelist,
# so it opens a real checkpoint and is therefore sensitive to which one opens.
# The other non-whitelisted shapes are `SET`, a multi-element `CREATE`, and an
# edge `CREATE` (they are what makes `rollback_tests.rs::ZERO_COPY_QUERIES`
# non-vacuous — note its first entry, a bare `CREATE`, satisfies the assertion
# trivially for this same reason).
#
# `test_bench_checkpoint_free_create_control` below pins the whitelist itself,
# so that if it ever narrows, the cost shows up somewhere rather than nowhere.
INSERT_STATEMENT = "MERGE (n:Item {id: $i}) ON CREATE SET n.name = 'x', n.code = 'c', n.qty = 1"


def _base_graph(size: int) -> KnowledgeGraph:
    """`size` nodes of one type, with a primary key and an indexable property.

    The declared primary key is load-bearing and is not incidental tidiness.
    Without one, `CREATE` invalidates the whole type's id index, so the next
    lookup by id rebuilds it — an O(V) cost per statement that has nothing to
    do with the rollback checkpoint but is large enough to swamp it entirely
    (34,486 us vs 4.1 us at 1M, per `test_bench_write_scaling.py`). Measuring
    the checkpoint through that noise would be measuring the wrong thing.

    The string property matters too: a `Compact` `NodeData` clone allocates per
    `Value::String`, so strings are what make a whole-graph clone expensive in
    bytes rather than merely in node count.
    """
    graph = KnowledgeGraph()
    graph.define_schema({"nodes": {"Item": {"primary_key": "id"}}})
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(size),
                "name": [f"item-{i}" for i in range(size)],
                "code": [f"code-{i}" for i in range(size)],
                "qty": [i % 977 for i in range(size)],
            }
        ),
        "Item",
        "id",
        "name",
        columns=["code", "qty"],
    )
    # Warm the id index so the measurement is the write, not a first-touch
    # index build landing in an arbitrary sample.
    graph.cypher("MATCH (n:Item {id: 0}) RETURN n.id")
    return graph


def _variant_graph(size: int, variant: str, tmp_dir) -> KnowledgeGraph:
    """A `_base_graph` pushed into one corner of the `journal_covers` gate.

    ⚠ The save here is a **plain `KnowledgeGraph().save()`, never
    `kglite.open()`**, and that is the single most important decision in this
    file. `kglite.open()` defaults to `durable="full"`, which puts an
    F_FULLFSYNC barrier — measured at 3.37 ms on this machine — on every commit.
    The clone this file is trying to detect costs ~3.06 ms at 100k. Opening the
    graph durably would therefore bury the entire signal underneath a fixed
    cost of the same magnitude, and the cell would read "healthy" in both
    worlds. Durability cost is measured in `test_bench_write_scaling.py`, on
    purpose, separately.

    `fsync=False` for the same reason: this save exists to flip
    `column_stores` from empty to populated, not to benchmark a disk flush.
    """
    graph = _base_graph(size)
    if variant in ("indexed", "saved_indexed"):
        # A secondary property index — a lookup structure, not a constraint.
        # Two of them, matching the shape the competitive run measured.
        graph.create_index("Item", "code")
        graph.create_index("Item", "qty")
    if variant in ("saved", "saved_indexed"):
        graph.save(str(tmp_dir / f"veto-{variant}-{size}.kgl"), fsync=False)

    # Vacuity guards. The 2026-07-27 run's own methodology note earned this:
    # every harness defect it caught was found because a number was
    # inconsistent with the configuration it *claimed* to represent, never
    # because a number looked implausible. A `saved` cell that silently isn't
    # columnar, or an `indexed` cell whose index didn't take, would report a
    # healthy fast path under a label promising otherwise — indistinguishable
    # from the fix working. These are cheap and they fail loudly.
    expect_columnar = variant in ("saved", "saved_indexed")
    assert graph.is_columnar is expect_columnar, (
        f"{variant} must{'' if expect_columnar else ' not'} own column stores; "
        "save() populates DirGraph.column_stores via enable_columnar() "
        "(io/file.rs:1495) and that is what the old gate vetoed on"
    )
    expect_indexed = variant in ("indexed", "saved_indexed")
    assert graph.has_index("Item", "code") is expect_indexed
    assert graph.has_index("Item", "qty") is expect_indexed
    return graph


@pytest.fixture(scope="module")
def veto_graphs(tmp_path_factory) -> dict[tuple[int, str], KnowledgeGraph]:
    """One graph per (size, variant) corner.

    Module-scoped: eight fixtures, two of them 100k nodes, are seconds of build
    time that must not be repeated per cell. `tmp_path_factory` rather than
    `tmp_path` because the latter is function-scoped and cannot be reached from
    here — and because nothing in this file may write outside a pytest-managed
    temporary directory.
    """
    tmp_dir = tmp_path_factory.mktemp("veto")
    return {(size, variant): _variant_graph(size, variant, tmp_dir) for size in SIZES for variant in VARIANTS}


@pytest.mark.benchmark
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("size", SIZES)
def test_bench_insert_after_veto_trigger(benchmark, veto_graphs, size, variant):
    """One node inserted, on a graph in each corner of the `journal_covers` gate.

    The headline cell. Defends: 100k insert **3.668 ms with neither trigger**
    vs **21–38 ms with two secondary indexes**, and **0.0062 → 3.0931 ms
    permanently after a single `save()`** (measured 2026-07-27).

    Read `saved` / `indexed` / `saved_indexed` against `fresh` at the same
    size, and each variant's 100k against its own 1k. Both ratios flat means
    the journal path still covers real application graphs.

    Uses `MERGE`, not `CREATE` — see the STATEMENT SHAPE comment above. This is
    not interchangeable.
    """
    graph = veto_graphs[(size, variant)]
    ids = iter(range(10_000_000, 1 << 30))

    def write():
        graph.cypher(INSERT_STATEMENT, params={"i": next(ids)})

    benchmark.pedantic(write, rounds=ROUNDS, iterations=1, warmup_rounds=WARMUP_ROUNDS)


@pytest.mark.benchmark
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("size", SIZES)
def test_bench_checkpoint_free_create_control(benchmark, veto_graphs, size, variant):
    """A bare single-node `CREATE` — the statement that opens no checkpoint.

    This cell is **expected to be flat and fast in every variant**, including
    the ones where the defect would be present, because
    `can_skip_rollback_checkpoint` gives it `StatementCheckpoint::None`
    outright. It is here for two reasons, neither of them regression-gating:

    1. It documents, executably, why the obvious insert benchmark cannot see
       defect A — the next person to reach for `CREATE` here finds the answer
       next to the code rather than re-deriving it from a null result.
    2. It is the tripwire for the whitelist *narrowing*. If the checkpoint-free
       path stops covering single-node `CREATE`, this cell moves and
       `test_bench_insert_after_veto_trigger` does not, which localises the
       change immediately.

    A divergence between this cell and the `MERGE` cell in the `fresh` variant
    is the cost of opening a journal checkpoint at all — worth knowing, and
    currently unmeasured anywhere else.
    """
    graph = veto_graphs[(size, variant)]
    ids = iter(range(40_000_000, 1 << 30))

    def write():
        graph.cypher("CREATE (:Item {id: $i, name: 'x', code: 'c', qty: 1})", params={"i": next(ids)})

    benchmark.pedantic(write, rounds=ROUNDS, iterations=1, warmup_rounds=WARMUP_ROUNDS)


@pytest.mark.benchmark
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("size", SIZES)
def test_bench_update_after_veto_trigger(benchmark, veto_graphs, size, variant):
    """One `SET` by primary key, in each corner of the gate.

    The update half of the same defect: `update_by_id` measured **3.986 ms with
    no index vs 43.6 ms with indexes** at 100k (2026-07-27). `SET` is the
    shape most exposed to the columnar veto, because a columnar graph routes
    the write through the master column store and the journal has to be able to
    reverse that sweep.

    `qty` is written rather than `name` deliberately: `execute_set`'s columnar
    fast path only fires for a property that is neither `title` nor `name`, so
    writing `name` would quietly exercise the per-node fallback — the write
    path that was never at risk. `rollback_tests.rs::
    the_columnar_fixture_writes_through_the_master_store` pins the same
    precondition on the Rust side.
    """
    graph = veto_graphs[(size, variant)]
    counter = iter(range(1, 1 << 30))

    def write():
        graph.cypher("MATCH (n:Item {id: 7}) SET n.qty = $v", params={"v": next(counter)})

    benchmark.pedantic(write, rounds=ROUNDS, iterations=1, warmup_rounds=WARMUP_ROUNDS)
    assert graph.cypher("MATCH (n:Item {id: 7}) RETURN n.qty AS q").to_list()[0]["q"] > 0


# ── defect B: the whole-graph clone a held reference forces ──────────
#
# ⚠⚠ THIS SECTION DELIBERATELY DEPARTS FROM THIS PROJECT'S `min` CONVENTION.
# DO NOT "FIX" IT BACK. READ THIS FIRST.
#
# CLAUDE.md's performance protocol says to trust `min` over `median` for
# sub-millisecond benchmarks, because median pulls upward with system load.
# That guidance is correct in general and WRONG HERE, and this comment exists
# because the obvious "cleanup" silently disables the benchmark.
#
# The clone fires on the FIRST write after a second `Arc<DirGraph>` appears,
# and once only — afterwards the graph is uniquely owned again and every
# subsequent write is back to normal speed. The measured 100k signature is
# exactly that: mean 0.0561 ms, p50 0.0044 ms, max 5.17 ms. So over N rounds
# with a single reference taken once:
#
#   * `min`  sees round 2..N  -> healthy. Blind by construction.
#   * `p50`  sees round N/2   -> healthy. Blind by construction.
#   * `p95`  is healthy for any N > 20. Blind by construction.
#   * only `max` sees it, and `max` is the one statistic nobody gates on
#     because it is where genuine noise lands.
#
# `test_bench_write_scaling.py` already had a cell aimed at this defect
# (`test_bench_create_with_lazy_result_alive`) and it reported NO DIFFERENCE —
# doubly blinded, by the averaging above and by running only at 1k where the
# clone is ~1000x too small to see. It has been removed and consolidated here;
# a cell that cannot see the defect it was written for is worse than no cell,
# because it certifies health.
#
# The fix is structural rather than statistical: `pedantic(setup=...)` runs
# `setup` before EVERY round and does not time it, so each round takes a fresh
# reference and each timed call is therefore a genuine first-write-after-a-
# reference. Every round pays the clone, which makes mean, min and p50 all
# meaningful again and needs no special-casing downstream.
#
# The invariant to preserve if you ever touch this: **the reference must be
# acquired inside `setup`, never once outside the measurement.** Hoisting it
# out is the one edit that turns this back into a benchmark that always passes.


def _wide_result(graph: KnowledgeGraph):
    """A lazy `ResultView` that keeps its own `Arc<DirGraph>` alive.

    Every clause of this query is chosen against the laziness rules in
    `planner/fusion/aggregate.rs:1733-1819` and `result_view.rs:112,203-217`,
    because a view that materialises eagerly drops its `Arc` and silently
    becomes the `holder="none"` case under a name claiming otherwise:

    * **>32 cells.** `EAGER_MATERIALISE_MAX_CELLS = 32`, applied as
      `rows * columns`. Two columns x `LIMIT 100` = 200 cells, comfortably past
      it while staying cheap to build.
    * **No standalone `WHERE`, no `WITH` / `UNWIND` / `CALL` / `ORDER BY`, one
      `RETURN`, not `DISTINCT`.** Any of those disqualifies laziness outright.
    * **Every `RETURN` item is a `PropertyAccess`.** Bare `RETURN n` is not one
      and would force eager materialisation; `n.name` and `n.qty` are.
    * **`streaming=True`** (the default) — `streaming=False` guarantees no pin.

    Sessions, transactions and frozen views set `lazy_eligible: false`
    explicitly, so they never produce a lazy view; `freeze()` pins the `Arc`
    directly instead, which is the `holder="frozen"` arm.
    """
    return graph.cypher("MATCH (n:Item) RETURN n.name, n.qty LIMIT 100")


@pytest.mark.benchmark
@pytest.mark.parametrize("holder", ["none", "result_view", "frozen"])
@pytest.mark.parametrize("size", SIZES)
def test_bench_first_write_after_reference(benchmark, veto_graphs, size, holder):
    """The first write after a reference to the graph appears.

    Defends (measured 2026-07-27, 100k): holding a returned `ResultView` across
    a write costs **max 5.17 ms** against a 0.0044 ms p50, with full recovery
    on release; `freeze()` held is the same at **max 5.83 ms**; ~28 ms at 1M.
    `holder="none"` is the control and must stay at the ordinary write cost.

    Read this as `result_view` / `frozen` against `none` at the same size, and
    each against itself across sizes — the clone is O(V+E), so a present defect
    shows a ~100x gap between 1k and 100k while the control stays flat.

    Every round re-acquires the reference in an untimed `setup`; see the long
    comment above for why that is load-bearing and not refactorable.

    A bare single-node `CREATE` is deliberately correct here, unlike in the
    defect-A cells. `Arc::make_mut` fires in `get_graph_mut`
    (`kg_core.rs:1569` -> `handle.rs:630`) *before* the checkpoint whitelist is
    ever consulted, so the cheapest possible statement still triggers the
    clone — which isolates defect B from checkpoint cost rather than summing
    the two.
    """
    graph = veto_graphs[(size, "fresh")]
    ids = iter(range(20_000_000, 1 << 30))

    def setup():
        if holder == "result_view":
            reference = _wide_result(graph)
        elif holder == "frozen":
            reference = graph.freeze()
        else:
            reference = None
        return (reference,), {}

    def write(reference):
        graph.cypher("CREATE (:Item {id: $i, name: 'y', code: 'c', qty: 1})", params={"i": next(ids)})
        # Returned, not merely closed over, so the reference is unambiguously
        # live for the whole timed call and cannot be collected early — which
        # would quietly turn this into the `none` case.
        return reference

    benchmark.pedantic(
        write,
        setup=setup,
        rounds=REF_ROUNDS,
        iterations=1,
        warmup_rounds=REF_WARMUP_ROUNDS,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("size", SIZES)
def test_bench_write_recovers_after_reference_released(benchmark, veto_graphs, size):
    """Steady-state writes with no reference held — the recovery control.

    The competitive run's key attribution: after the reference is released,
    write cost returns fully to baseline. That is what separates defect B (one
    write, then recovery) from defect A (every write, forever). Without this
    cell a future reader cannot tell which of the two a regression in
    `test_bench_first_write_after_reference` represents.

    Expect this to match `test_bench_insert_after_veto_trigger[*-fresh]`.
    """
    graph = veto_graphs[(size, "fresh")]
    ids = iter(range(30_000_000, 1 << 30))
    # Taken and dropped before the measurement, so the graph has definitely
    # been through the clone-and-recover cycle by the time timing starts.
    _wide_result(graph)
    graph.cypher("CREATE (:Item {id: 29999999, name: 'settle', code: 'c', qty: 1})")

    def write():
        graph.cypher("CREATE (:Item {id: $i, name: 'z', code: 'c', qty: 1})", params={"i": next(ids)})

    benchmark.pedantic(write, rounds=ROUNDS, iterations=1, warmup_rounds=WARMUP_ROUNDS)
