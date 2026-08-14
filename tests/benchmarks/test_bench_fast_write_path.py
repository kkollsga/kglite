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

After PR #86 ``journal_covers`` (``rollback.rs``) has **exactly one** term left
— ``graph.graph.supports_undo_journal()``. Neither ``save()`` nor any index
family vetoes it any more. The only remaining way to force the clone is the
*backend*: ``Memory`` **and ``Mapped``** journal (both wrap the same heap
``StableDiGraph``), ``Recording`` forwards to its inner backend, and only
``Disk`` returns ``false`` — it has no petgraph and therefore no
``NodeIndex`` identity for an ``UndoEntry`` to name
(``storage/backend.rs::supports_undo_journal``). A reintroduced veto is
therefore a new conjunct in a one-conjunct predicate — cheap to add by
accident, and these cells are what would notice.

*(Corrected 2026-08-10. This paragraph previously claimed ``Mapped`` returned
``false``. It has returned ``true`` since 2026-07-30, and
``rollback_tests.rs::mapped_statements_copy_zero_nodes`` — itself the
inversion of an arm that used to pin the opposite — is the executable proof.
The stale claim mattered: it named the mapped backend as a known-slow write
path, which is exactly the kind of "already understood" that stops a
measurement being taken.)*

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
indexed at once, through a real ``save()``.

────────────────────────────────────────────────────────────────────────────
Defect B — ``Arc::make_mut`` deep clone on a merely-held reference
────────────────────────────────────────────────────────────────────────────

Holding a returned result view across a write triggers one full graph clone.
Measured at 100k (2026-07-27): mean 0.0561 ms, p50 0.0044 ms, **max 5.17 ms**,
with full recovery once the reference is released; ``freeze()`` held behaves
the same (max 5.83 ms); ~28 ms at 1M. No threading, no snapshot API, no
explicit copy — merely keeping the ``ResultView`` a query returned.

**Re-pinned 2026-08-10** (release, two agreeing runs, D2 Phase 0). Mean µs of
the timed first-write, every round against a freshly acquired reference:

===============  =========  ===========  ============
holder            1k         100k         1M
===============  =========  ===========  ============
``none``            2.8          2.8           3.0
``dropped_view``    3.1          3.0           3.4
``result_view``    36.6      3,462.9      36,338.5
``frozen``         37.5      3,354.6      36,211.3
``session``        38.1      3,265.0      36,303.5
``transaction``    38.7      3,357.6      36,404.9
===============  =========  ===========  ============

Three things that table says and the 2026-07-26 one could not:

* **All four holders cost the same.** ``session`` and ``transaction`` were
  never measured before and are the two an application acquires *deliberately*;
  they pay the identical fork. There is no "just don't hold a ResultView"
  work-around.
* **``dropped_view`` isolates it to the hold.** Building the view and releasing
  it costs +0.3 µs at every size. Everything else is the ``Arc``.
* **The control got 2.6x faster while the cliff got worse.** ``none`` went
  7.3/7.8/8.1 µs (2026-07-26) to 2.8/2.8/3.0, so the *ratio* went 3.8x/278x/
  3,400x to **13x/1,240x/12,100x**. Removing per-write cost elsewhere made this
  the dominant term, not a smaller one.

Memory is the other half: at 1M the fork grew process peak RSS by **668.8 MB**
in a single round — a second whole copy of the graph — while the ``none`` and
``dropped_view`` arms immediately before it grew it by 0.0 MB.

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

import gc
import itertools
import math
import os
import sys

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

#: Sizes for the **defect-B** (held-reference) cells only.
#:
#: 1M is deliberately omitted from ``SIZES`` above and deliberately present
#: here, and the two decisions do not conflict — they are about different
#: defects. Defect A is a *multiplier* on every write, unambiguous at two
#: decades, and four variants x 1M nodes buys no signal for minutes of fixture
#: build. Defect B is a *whole-graph copy*, so its cost is the graph itself:
#: 27.6 ms at 1M against 2.2 ms at 100k (2026-07-26). 1M is where it lives, and
#: a program that proposes to remove it has to be measured where it is largest.
REF_SIZES = [1_000, 100_000, 1_000_000]

#: The four corners of the ``journal_covers`` gate. ``saved_indexed`` is the
#: one a real application actually is, and the one combination the Rust-side
#: arms never build.
VARIANTS = ["fresh", "saved", "indexed", "saved_indexed"]

#: Every ordinary way a Python caller ends up holding a second
#: ``Arc<DirGraph>``, plus two controls.
#:
#: * ``none`` — no reference at all. The floor.
#: * ``dropped_view`` — the same view the ``result_view`` arm holds, built and
#:   **released** inside the untimed setup. This is the control that separates
#:   *building* the view from *holding* it: without it, a ``result_view`` cell
#:   that got slower could equally be a query-planner regression, and the two
#:   are indistinguishable from the ``none`` arm alone.
#: * ``result_view`` — the accidental case: a lazy view a query returned.
#: * ``frozen`` — ``freeze()``, the deliberate snapshot.
#: * ``session`` — ``g.session()`` pins the source graph's ``Arc`` for the
#:   session object's whole lifetime (``pyapi/session.rs::from_arc``).
#: * ``transaction`` — ``g.begin()`` holds a snapshot until commit/rollback
#:   (``kg_core.rs``, core ``session/transaction.rs``).
#:
#: ``session`` and ``transaction`` were unmeasured anywhere before 2026-08-10,
#: and they are the two holders an application acquires *on purpose* — i.e. the
#: ones a user cannot be told to simply stop doing.
REF_HOLDERS = ["none", "dropped_view", "result_view", "frozen", "session", "transaction"]

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

# Module-level, NOT per-test. The defect-B cells are parametrized over `holder`
# but every parametrization writes to the SAME shared `(size, "fresh")` graph,
# so a per-test `iter(range(BASE, ...))` hands the second parametrization ids
# the first one already inserted -- a `duplicate primary key` failure that only
# appears when the cells run together, never in isolation. A shared counter is
# immune to that however these cells are parametrized in future.
_FRESH_GRAPH_IDS = itertools.count(20_000_000)


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
    # because a number looked implausible. A cell whose fixture silently
    # carries no column rows, or an `indexed` cell whose index didn't take,
    # would report a healthy fast path under a label promising otherwise —
    # indistinguishable from the fix working. These are cheap and they fail
    # loudly. The columnar half no longer discriminates the variants (every
    # graph owns stores from its first node), so it asserts the rows exist.
    assert graph.graph_info()["columnar_total_rows"] == size, (
        f"{variant} must carry its rows in the column store, or this cell "
        "measures a code path the cost does not live on"
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


@pytest.fixture(scope="module")
def ref_graphs(veto_graphs) -> dict[int, KnowledgeGraph]:
    """One `fresh` graph per :data:`REF_SIZES`, for the defect-B cells.

    The 1k and 100k entries are the *same objects* the defect-A cells use — a
    second 100k build would be seconds of fixture time for an identical graph.
    Only 1M is built here, and only in the `fresh` variant: defect B is
    ``Arc::make_mut`` in ``get_graph_mut``, which fires before the checkpoint
    whitelist is consulted, so the ``journal_covers`` corners are irrelevant to
    it and building four of them at 1M would measure nothing extra.
    """
    graphs = {size: veto_graphs[(size, "fresh")] for size in REF_SIZES if size in SIZES}
    for size in REF_SIZES:
        if size not in graphs:
            graphs[size] = _base_graph(size)
    return graphs


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
    return graph.cypher(WIDE_QUERY)


#: The query :func:`_wide_result` runs, hoisted to a constant so the
#: shape guard below can assert on the text a benchmark round actually uses
#: rather than on a copy of it.
WIDE_QUERY = "MATCH (n:Item) RETURN n.name, n.qty LIMIT 100"

#: ``result_view.rs::EAGER_MATERIALISE_MAX_CELLS``. A view at or under this
#: many ``rows x columns`` materialises and **drops** its ``Arc``, silently
#: becoming the ``holder="none"`` case under a name promising otherwise.
EAGER_MATERIALISE_MAX_CELLS = 32


def _acquire_reference(graph: KnowledgeGraph, holder: str):
    """Take (or deliberately decline to take) a second reference to `graph`.

    **One definition, two callers**, and that is the point: the benchmark's
    untimed `setup` and the non-benchmark guard below both go through here, so
    the guard is testing the acquisition the benchmark performs rather than a
    lookalike written next to it. A hoist that made the benchmark vacuous would
    have to be made *here* to escape the guard, and here is the one place the
    guard is looking.
    """
    if holder == "none":
        return None
    if holder == "dropped_view":
        # Built and released before the timed call: same construction cost,
        # no surviving `Arc`. CPython drops it at zero refcount, immediately.
        _wide_result(graph)
        return None
    if holder == "result_view":
        return _wide_result(graph)
    if holder == "frozen":
        return graph.freeze()
    if holder == "session":
        return graph.session()
    if holder == "transaction":
        return graph.begin()
    raise ValueError(f"unknown holder {holder!r}")


#: Holders that must yield a live Python object. The two that must not are
#: `none` and `dropped_view`; asserting both directions is what makes the
#: control a control rather than an untested label.
PINNING_HOLDERS = [h for h in REF_HOLDERS if h not in ("none", "dropped_view")]


def _peak_rss_mb() -> float:
    """Process peak RSS in MB. `ru_maxrss` is bytes on macOS, KiB on Linux.

    ⚠ **`ru_maxrss` is a monotone process high-water mark**, so
    `rss_peak_growth_mb` is *not* per-cell memory. It reads non-zero only for
    the first cell that pushes the process past its previous peak, and zero for
    every later cell that stays under it — including cells that allocated
    exactly as much.

    That is why the order of :data:`REF_HOLDERS` is load-bearing rather than
    alphabetical: `none` and `dropped_view` run **first** at each size and
    establish the peak without holding anything, so the first pinning holder's
    growth is attributable to the fork and nothing else. Measured 2026-08-10 at
    1M: `none` and `dropped_view` grew the peak by 0.0 MB, and `result_view` —
    the very next cell, differing only in that it kept the view — grew it by
    **668.8 MB**, i.e. a whole second copy of the graph. The three holders after
    it read 0.0 MB because the peak was already there, not because they were
    free.

    Do not "fix" this by resetting between cells; there is no way to reset
    `ru_maxrss`. Read the first pinning arm, or run one cell per process.
    """
    try:
        import resource
    except ImportError:
        # Windows has no `resource`. Deliberately NaN rather than 0.0: a zero
        # would read as "the fork allocated nothing", which is the reassuring
        # direction and the one lie this file must not tell. NaN propagates
        # through the subtraction and shows up as `nan` in `extra_info`.
        #
        # The import is function-local, not module-scope, because
        # `tests/test_test_suite_portability.py::
        # test_no_module_scope_posix_only_imports` is a **default-suite** guard:
        # a POSIX-only import at module scope aborts collection for the entire
        # suite on Windows. It caught this exact line on 2026-08-10, one commit
        # after it was written.
        return math.nan
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


@pytest.mark.benchmark
@pytest.mark.parametrize("holder", REF_HOLDERS)
@pytest.mark.parametrize("size", REF_SIZES)
def test_bench_first_write_after_reference(benchmark, ref_graphs, size, holder):
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

    **Memory is recorded alongside time**, as `extra_info`. The fork's cost is
    not only latency: a copy of the whole graph is also a transient allocation
    the size of the graph, and at 1M nodes that is the difference between a
    process that fits and one that does not. `ru_maxrss` is a process *peak*,
    so `rss_peak_growth_mb` is a floor on the largest live footprint the cell
    reached, not a per-round figure — read it across holders at one size, where
    the only difference between the arms is whether a reference was held.
    """
    graph = ref_graphs[size]
    ids = _FRESH_GRAPH_IDS

    def setup():
        return (_acquire_reference(graph, holder),), {}

    def write(reference):
        graph.cypher("CREATE (:Item {id: $i, name: 'y', code: 'c', qty: 1})", params={"i": next(ids)})
        # Returned, not merely closed over, so the reference is unambiguously
        # live for the whole timed call and cannot be collected early — which
        # would quietly turn this into the `none` case.
        return reference

    rss_before = _peak_rss_mb()
    benchmark.pedantic(
        write,
        setup=setup,
        rounds=REF_ROUNDS,
        iterations=1,
        warmup_rounds=REF_WARMUP_ROUNDS,
    )
    benchmark.extra_info["holder"] = holder
    benchmark.extra_info["nodes"] = size
    benchmark.extra_info["rss_peak_before_mb"] = round(rss_before, 1)
    rss_after = _peak_rss_mb()
    benchmark.extra_info["rss_peak_after_mb"] = round(rss_after, 1)
    benchmark.extra_info["rss_peak_growth_mb"] = round(rss_after - rss_before, 1)

    # Non-vacuity. A cell whose write silently no-opped, or whose reference was
    # collected before the timed call, would read as a spectacular improvement.
    assert graph.cypher("MATCH (n:Item) RETURN count(n) AS c").to_list()[0]["c"] > size


@pytest.mark.benchmark
@pytest.mark.parametrize("size", REF_SIZES)
def test_bench_write_recovers_after_reference_released(benchmark, ref_graphs, size):
    """Steady-state writes with no reference held — the recovery control.

    The competitive run's key attribution: after the reference is released,
    write cost returns fully to baseline. That is what separates defect B (one
    write, then recovery) from defect A (every write, forever). Without this
    cell a future reader cannot tell which of the two a regression in
    `test_bench_first_write_after_reference` represents.

    Expect this to match `test_bench_insert_after_veto_trigger[*-fresh]`.

    Distinct from the `dropped_view` arm of the cell above, which is the
    *per-round* control (a view built and released inside every setup). This
    one is the *steady-state* control: one clone-and-recover cycle, then plain
    writes with nothing held at all. Together they separate three costs that a
    single arm would sum — building the view, holding it, and the write itself.
    """
    graph = ref_graphs[size]
    ids = _FRESH_GRAPH_IDS
    # Taken and dropped before the measurement, so the graph has definitely
    # been through the clone-and-recover cycle by the time timing starts.
    _wide_result(graph)
    graph.cypher("CREATE (:Item {id: $i, name: 'settle', code: 'c', qty: 1})", params={"i": next(ids)})

    def write():
        graph.cypher("CREATE (:Item {id: $i, name: 'z', code: 'c', qty: 1})", params={"i": next(ids)})

    benchmark.pedantic(write, rounds=ROUNDS, iterations=1, warmup_rounds=WARMUP_ROUNDS)
    benchmark.extra_info["nodes"] = size


# ── guards: the defect-B cells cannot go vacuous ─────────────────────
#
# Everything below is NON-benchmark and runs in the default suite (testpaths
# includes `tests/`, and only `-m benchmark` is deselected), inside the 120 s
# hang ceiling. The fixtures here are small on purpose: these tests check
# *structure*, never cost, so a thousand nodes proves what a million would.
#
# They exist because the defect-B cells have three independent ways to keep
# reporting numbers while measuring nothing, and none of them is detectable by
# reading the output:
#
#   1. the reference gets hoisted out of `setup` — every round after the first
#      measures the recovered write, and the cell reads "fixed";
#   2. `_wide_result` stops being lazy (a `LIMIT` edit, an extra clause) — the
#      view materialises, drops its `Arc`, and every `result_view` round
#      silently becomes the `none` control;
#   3. a holder stops holding — an API change makes `session()` or `begin()`
#      copy instead of pin, and its arm becomes a second `none` control.
#
# Each has a test. None of them asserts a timing.


GUARD_NODES = 1_000
GUARD_ROUNDS = 5


@pytest.fixture
def guard_graph() -> KnowledgeGraph:
    """A small `_base_graph`, function-scoped so no test can see another's
    writes. Deliberately *not* `veto_graphs`: that fixture builds eight graphs
    including two at 100k, which is benchmark-scale setup no default-suite test
    should trigger."""
    return _base_graph(GUARD_NODES)


def _distinct_reference_count(acquire, graph: KnowledgeGraph, rounds: int) -> int:
    """Drive `acquire` once per round exactly as `pedantic(setup=...)` does,
    and count how many *distinct live objects* it produced.

    Every reference is retained in `held` for the duration. That is what makes
    `id()` a sound identity test: CPython reuses the address of a freed object,
    so a per-round acquisition whose result was dropped between rounds could
    hand back the same `id()` three times and look hoisted. Holding them all
    makes distinct objects provably distinct addresses.
    """
    held = []
    for _ in range(rounds):
        reference = acquire(graph)
        held.append(reference)
        graph.cypher("CREATE (:Item {id: $i, name: 'g', code: 'c', qty: 1})", params={"i": next(_FRESH_GRAPH_IDS)})
    return len({id(reference) for reference in held})


@pytest.mark.parametrize("holder", PINNING_HOLDERS)
def test_every_round_acquires_a_fresh_reference(guard_graph, holder):
    """The hoisting detector, pointed at the real acquisition helper.

    `_acquire_reference` is what the benchmark's `setup` calls, so a one-line
    edit that hoists the reference out of the per-round path has to go through
    this function to take effect — and this test then sees `1` where it
    requires `GUARD_ROUNDS`.
    """
    distinct = _distinct_reference_count(lambda g: _acquire_reference(g, holder), guard_graph, GUARD_ROUNDS)
    assert distinct == GUARD_ROUNDS, (
        f"holder={holder!r} produced {distinct} distinct references over {GUARD_ROUNDS} rounds; "
        "the benchmark's `setup` must take a NEW reference every round or every round after "
        "the first measures the already-recovered write"
    )


def test_the_hoisting_detector_fires_on_a_hoisted_acquirer():
    """Proof the detector above can go red — the non-vacuity requirement.

    A gate that has never been shown failing is a gate nobody has tested. This
    drives the same counter with an acquirer that hoists its reference (the
    exact one-line mistake the comment above the defect-B section warns about)
    and requires the counter to report `1`, i.e. *invalid*.

    Both directions are asserted in one place on purpose: if a future refactor
    made `_distinct_reference_count` always return `rounds` — by dropping the
    `held` list, say — this test fails immediately, while the guard above would
    keep passing forever.
    """
    graph = _base_graph(GUARD_NODES)
    hoisted = _wide_result(graph)

    def hoisting_acquire(_graph):
        return hoisted

    distinct = _distinct_reference_count(hoisting_acquire, graph, GUARD_ROUNDS)
    assert distinct == 1, (
        "the detector must report a hoisted reference as a single distinct object; "
        f"it reported {distinct}, so it cannot detect the hoist it exists to detect"
    )


def test_the_controls_hold_no_reference(guard_graph):
    """`none` and `dropped_view` must yield nothing to hold.

    The other half of the previous test: a control that silently started
    pinning would make every arm look equally slow, and the cell would report
    "no defect" for the one reason nobody would check.
    """
    assert _acquire_reference(guard_graph, "none") is None
    assert _acquire_reference(guard_graph, "dropped_view") is None


def test_the_wide_result_stays_past_the_eager_materialise_cutoff(guard_graph):
    """`_wide_result` must produce a view the engine keeps lazy.

    A lazy view and an eagerly-materialised one are **semantically
    indistinguishable from Python** — that is the whole point of the fork, and
    it is why this test asserts the *rule's inputs* rather than its outcome.
    `EAGER_MATERIALISE_MAX_CELLS` is applied as `rows * columns`
    (`result_view.rs`), so the two checkable inputs are the cell count and the
    absence of any clause that disqualifies laziness outright. A `LIMIT 100` ->
    `LIMIT 10` edit is the realistic trapdoor and it fails here.

    What this cannot check is the `Arc` itself; the benchmark's
    `rss_peak_growth_mb` on the `result_view` arm is the observable that does,
    and it needs benchmark scale to be legible.
    """
    view = _wide_result(guard_graph)
    cells = len(view.columns) * len(view.to_list())
    assert cells > EAGER_MATERIALISE_MAX_CELLS, (
        f"the held view is {cells} cells, at or under the {EAGER_MATERIALISE_MAX_CELLS}-cell "
        "eager-materialise cutoff; it would drop its Arc and the result_view arm would be a "
        "second copy of the `none` control"
    )
    assert view.columns == ["n.name", "n.qty"], "every RETURN item must stay a PropertyAccess"
    disqualifying = (" WITH ", " UNWIND ", " CALL ", " ORDER BY ", " WHERE ", "DISTINCT")
    for clause in disqualifying:
        assert clause not in f" {WIDE_QUERY} ", f"{clause.strip()!r} disqualifies laziness (planner/fusion)"


@pytest.mark.parametrize("holder", PINNING_HOLDERS)
def test_a_held_reference_keeps_its_pre_write_values(guard_graph, holder):
    """The semantic invariant the whole D2 program must preserve.

    Today this passes for the *expensive* reason — the write forks the graph,
    so the holder is left looking at an untouched copy. Structural sharing must
    make it pass for a cheap reason instead. If a future change makes the
    holder observe the write, this is the test that says the program broke its
    own contract rather than merely got faster.

    The write is a `SET` on a node the holder can see, not a `CREATE`: an
    inserted node is invisible to a `LIMIT 100` view whether or not isolation
    holds, so a `CREATE` would pass this test on a graph with no isolation at
    all.
    """
    reference = _acquire_reference(guard_graph, holder)
    before = guard_graph.cypher("MATCH (n:Item {id: 5}) RETURN n.qty AS q").to_list()[0]["q"]
    sentinel = before + 424_242

    guard_graph.cypher("MATCH (n:Item {id: 5}) SET n.qty = $v", params={"v": sentinel})

    assert guard_graph.cypher("MATCH (n:Item {id: 5}) RETURN n.qty AS q").to_list()[0]["q"] == sentinel, (
        "the graph must see its own write"
    )

    if holder == "result_view":
        seen = {row["n.name"]: row["n.qty"] for row in reference.to_list()}["item-5"]
    else:
        seen = reference.cypher("MATCH (n:Item {id: 5}) RETURN n.qty AS q").to_list()[0]["q"]
    assert seen == before, f"holder={holder!r} must still see the pre-write value {before}, saw {seen}"


# ── the memory half of the same defect ──────────────────────────────────
#
# The timing cells above answer "how long does the first write take". This one
# answers the question a long-running process actually cares about: **does
# holding a view make the graph's memory grow, and does it come back.**
#
# Before D2 each fork allocated a second whole graph — measured at Phase 0 as
# **+668.8 MB of process peak at 1M**, in the cell immediately after two
# controls that grew it by 0.0 MB. So the acceptance bounds below are not
# invented: they are "an order of magnitude under one graph copy", and the
# defect they exclude is 668.8 MB.


def _current_rss_mb() -> float:
    """Resident set size *now* — not the high-water mark.

    `resource.getrusage(...).ru_maxrss` is monotonic, so it can show a peak but
    can never show a graph settling back. `ps` is the portable way to read the
    live value on both platforms this repo tests on, and one subprocess per
    measurement point is irrelevant next to a 1M-node build.
    """
    import subprocess

    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, check=True)
    return int(out.stdout.strip()) / 1024


def _settle() -> float:
    """Drop Python-side references and read the settled RSS."""
    gc.collect()
    return _current_rss_mb()


#: Rounds per sequence. The plan asks for 20; the number matters because the
#: failure this detects is *per-round accumulation*, which one round cannot see.
RSS_ROUNDS = 20

#: One 1M-node graph copy, measured at Phase 0 (2026-08-10). Every bound below
#: is stated as a fraction of it so the numbers keep their meaning if the
#: fixture changes.
ONE_GRAPH_COPY_MB = 668.8

#: Ids for the sequences below, module-scoped because `ref_graphs` is: both
#: parametrizations write into the *same* 1M graph, and a per-test counter would
#: make the second one collide with the first's primary keys.
_RSS_IDS = itertools.count(70_000_000)


@pytest.mark.benchmark
@pytest.mark.parametrize("holder", ["result_view", "frozen"])
def test_held_view_writes_do_not_grow_a_graph_copy_per_write(ref_graphs, holder):
    """Three RSS sequences at 1M: dropped-per-round, continuously held, released.

    * **dropped** — take a view, write, drop it, x20. A regression looks like
      growth proportional to *rounds*: every round forks and never folds back.
    * **held** — one view alive across 20 writes. A regression looks like growth
      proportional to *writes*: a graph copy per write, the pre-D2 shape.
    * **released** — drop the view, write once more. A regression looks like no
      settling: the overlay never folds, so the base stays pinned forever.

    Not `benchmark.pedantic`: this measures bytes, not seconds, and pytest-
    benchmark's rounds would confound both.

    ⚠ **Only the first parametrization to run measures a cold allocator.** The
    two arms share a process and a 1M graph, so once one of them has caused a
    large allocation the arena stays with the process and a later arm can
    allocate the same bytes again without RSS moving. Proven, not assumed:
    under a mutation that restores whole-graph-clone semantics, `result_view`
    (which runs first) reads **+331.1 MB** settled and fails, while `frozen`
    reads **+0.0 MB** and passes on the arena the first arm left behind. That is
    the same hazard `_peak_rss_mb`'s docstring describes, and it has the same
    remedy: to red-team the second arm, run one cell per process. Do not "fix"
    it by resetting between arms — there is nothing to reset.
    """
    graph = ref_graphs[1_000_000]

    def write() -> None:
        graph.cypher("CREATE (:Item {id: $i, name: 'x', code: 'c', qty: 1})", params={"i": next(_RSS_IDS)})

    def acquire():
        return _wide_result(graph) if holder == "result_view" else graph.freeze()

    # Warm: the very first write after the fixture build allocates arenas that
    # belong to no sequence. Measure from a settled floor.
    write()
    baseline = _settle()

    # ── sequence 1: a fresh view per round, dropped before the next ──
    dropped_peak = baseline
    for _ in range(RSS_ROUNDS):
        ref = acquire()
        write()
        del ref
        dropped_peak = max(dropped_peak, _current_rss_mb())
    dropped_settled = _settle()

    # ── sequence 2: one view held across every write ──
    pin = acquire()
    held_peak = dropped_settled
    for _ in range(RSS_ROUNDS):
        write()
        held_peak = max(held_peak, _current_rss_mb())
    held_settled = _current_rss_mb()

    # ── sequence 3: release, then one more write to fold back ──
    del pin
    write()
    released_settled = _settle()

    report = {
        "baseline_mb": round(baseline, 1),
        "dropped_peak_growth_mb": round(dropped_peak - baseline, 1),
        "dropped_settled_growth_mb": round(dropped_settled - baseline, 1),
        "held_peak_growth_mb": round(held_peak - dropped_settled, 1),
        "held_settled_growth_mb": round(held_settled - dropped_settled, 1),
        "released_settled_growth_mb": round(released_settled - dropped_settled, 1),
        "one_graph_copy_mb": ONE_GRAPH_COPY_MB,
    }
    print(f"\nRSS[{holder}] {report}")

    # Acceptance, all as fractions of one graph copy (668.8 MB at 1M):
    #
    # 1. Twenty take-write-drop rounds must not accumulate. Each round forks and
    #    the next write folds it back, so the settled cost is the 20 nodes
    #    created — kilobytes. A tenth of one copy is a wide margin over that and
    #    an order of magnitude under the defect.
    assert dropped_settled - baseline < ONE_GRAPH_COPY_MB * 0.1, (
        f"twenty fork/fold rounds settled +{dropped_settled - baseline:.1f} MB; "
        "that is per-round accumulation, not O(changes)"
    )
    # 2. Twenty writes under one held view must cost one overlay, not twenty
    #    graphs. The pre-D2 behaviour would be a copy per write.
    assert held_peak - dropped_settled < ONE_GRAPH_COPY_MB * 0.5, (
        f"twenty writes under a held view peaked +{held_peak - dropped_settled:.1f} MB; "
        f"a graph copy is {ONE_GRAPH_COPY_MB} MB, so this is growing storage per write"
    )
    # 3. Releasing the view and writing once must not grow it further — that
    #    write is where compaction fires.
    assert released_settled <= held_peak + ONE_GRAPH_COPY_MB * 0.05, (
        f"the fold-back write grew RSS to +{released_settled - dropped_settled:.1f} MB; "
        "compaction should release the overlay, not duplicate it"
    )
