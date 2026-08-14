"""What does a write against a saved (columnar) type cost, and does it scale
with the size of the type rather than the size of the write?

The gap this file fills. `test_bench_fast_write_path.py` measures a **one-row**
`MERGE` across the four corners of the `journal_covers` gate, and
`test_bench_write_scaling.py` measures inserts as the graph grows. Neither
measures the shapes that carry the remaining columnar cost: **one `SET` /
`REMOVE` / `MERGE` statement writing R rows into a type that already has N
nodes**, on a graph that has been through `save()`.

────────────────────────────────────────────────────────────────────────────
The shape, and why it is worse than "R times a one-row write"
────────────────────────────────────────────────────────────────────────────

**One mechanism, once per statement** — as of the D1 Phase-3 ownership change
(measured 2026-08-10). The storage backend is now the sole owner of the column
stores: `PropertyStorage::Columnar` carries only a `row_id`, no node holds a
handle, and the O(N_type) end-of-clause re-point sweep and the per-row
node-private fork are both **deleted**.

What is left is the rollback pre-image. The first property write of a statement
against a columnar type hands the statement's undo journal an `Arc::clone` of
the type's master store — `capture_property_pre_image`
(`storage/impls.rs`) on the fallback route, `write_column_master`
(`languages/cypher/executor/columnar_write.rs`) on the master route. The
journal's handle is then what forces the following `Arc::make_mut` to
**deep-clone the whole `ColumnStore`**, so the pre-statement image stays
pristine for rollback. That clone is O(|store|), i.e. proportional to
N_type x columns, and it is paid **once per statement**: the journal declines a
second clone, so later rows in the same statement mutate the fork in place.

The cost profile that follows, and what each axis now measures:

* A one-row `SET` / `REMOVE` / `MERGE` against a 100k x 14-property saved type
  costs ~1.2 ms, essentially all of it that one clone (59% of samples in
  `_platform_memmove`, 2026-08-10 profile).
* Adding rows is nearly free by comparison — 500 rows cost the one clone plus
  500 rows of ordinary work.
* `MERGE` no longer compounds. It still issues one `SET` clause per row, but
  the *statement's* journal entry is shared across those clauses, so R rows pay
  one clone rather than R. `merge[100000-50-saved]` fell 73.0 ms -> 1.27 ms.
* Memory no longer grows with rows at all. The pre-Phase-3 node-private route
  retained ~44-59 MB per row; the measured per-row growth is now **zero** on
  every route (master, title, MERGE), which is why this file no longer carries
  a memory-budget skip guard.

────────────────────────────────────────────────────────────────────────────
How to read it
────────────────────────────────────────────────────────────────────────────

Two axes, and the interesting number is a **ratio**, never an absolute:

* **Across the row count at one `size`.** Linear in rows is the ceiling now,
  not the floor: the fixed per-statement clone is amortised over rows, so a
  500-row cell should be *sub*-linear against its own R = 1 cell. Growth that
  is faster than linear means a per-row term has come back.
* **Across `SIZES` at one row count.** This is the diagnostic axis, and it is
  the one still carrying cost. Writing R rows should not care how many nodes
  the type already has; it does, because the pre-image clone copies the whole
  store. The single-row `SET` cell isolates that term best — at R = 1 there is
  one row of real work and everything else on the clock is the clone.

The `SET` cells carry a third axis, `target`, and it is not cosmetic: writing a
genuine store column (`c1`) and writing the promoted node **title** (`name`) are
two different routes through the same node — `write_column_master` versus the
`GraphWrite::set_node_property` fallback. Before D1 Phase 3 they differed by
more than an order of magnitude on a saved graph; they now converge, because
both end at the same once-per-statement store fork. Keeping both cells is what
makes that convergence visible rather than assumed. See `SET_TARGETS` and the
cell's docstring. The `MERGE` cells write `name`, so they sit on the title
route.

`fresh` is the control, and doubles as the in-memory regression control the
ownership program must not regress by more than 5%. It has no column stores at
all, so it cannot take the master-write path and should stay flat across
`SIZES` no matter what happens to `saved` / `reloaded`. If `fresh` ever tracks
them, the cost is not columnar and this file's premise is wrong — treat that as
a finding, not as noise.

`saved` (`save()` in place) and `reloaded` (`save()` then `kglite.load()`) are
both consolidated but arrive there by different routes — an in-place rebuild
versus `attach_portable_column_stores` on the load path. They are measured separately
so that "after save" and "after save and reload" are answered independently
rather than assumed identical.

⚠ This file asserts no timing threshold, deliberately: wall time here is
sensitive to machine state, and the project's doctrine is that a timing gate on
a shared runner flakes in both directions. It is a measurement harness for a
change that has not landed yet, and the honest reading of a benchmark with no
assertion is "read the numbers", not "this is guarded". The *correctness* of
the columnar path is guarded in Rust
(`rollback_tests.rs::a_columnar_set_journals_one_pre_image_per_changed_node`
and the mapped arm beside it), which is where a deterministic assertion
belongs.

Nothing here is in the `make bench-check` tracked set.

Run with::

    uv run --no-sync maturin develop --release
    .venv/bin/python -m pytest tests/benchmarks/test_bench_merge_columnar.py \\
        -m benchmark -v
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph

#: Existing nodes in the written type. The diagnostic axis: the cost of writing
#: R rows must not depend on N, and two decades of N make a dependency obvious.
SIZES = [1_000, 100_000]

#: Rows merged by the single measured statement. 1 is the degenerate case the
#: other files already cover, and is included so this file's own numbers can be
#: compared against theirs; the rest is where per-row compounding would show if
#: it ever came back (it was there before D1 Phase 3 and is not there now).
MERGE_ROWS = [1, 50, 500]

#: Rows written by the single measured `SET` / `REMOVE` statement. R = 1 is the
#: load-bearing cell: one statement, one row of real work, so everything else on
#: the clock is the once-per-statement rollback pre-image clone of the whole
#: store. R = 500 shows that the clone really is amortised over rows.
WRITE_ROWS = [1, 500]

#: `fresh` owns no column stores, so it cannot take the master-write path —
#: the control that separates "columnar cost" from "MERGE is just slow".
VARIANTS = ["fresh", "saved"]

#: The `SET` / `REMOVE` cells add the save+reload arm, because the headline
#: claim under test is stated as "before save versus after save and reload".
#: Reload reaches columnar storage by a different code path than an in-place
#: `save()`, so it is measured rather than assumed equivalent.
WRITE_VARIANTS = ["fresh", "saved", "reloaded"]

#: Wide enough that `|store|` is a visible term rather than a rounding error.
#: The pre-image clone copies every column, so a narrow type understates it:
#: measured 2026-08-10 at N = 100k, a one-row `SET` costs 409 us against a
#: 1-extra-column type and 866 us against this 12-column one.
EXTRA_COLUMNS = 12

ROUNDS = 5
WARMUP_ROUNDS = 1

#: The `SET` / `REMOVE` cells are cheaper per round than a 500-row MERGE, so
#: they can afford more rounds; `min` over 20 is a far steadier reading than
#: `min` over 5 for a shape that lands in the tens-of-microseconds range on the
#: `fresh` arm.
WRITE_BENCH_ROUNDS = 20
WRITE_WARMUP_ROUNDS = 3

#: The property the REMOVE cells clear and the setup step restores. It is one of
#: the `EXTRA_COLUMNS`, i.e. a genuine column of the store rather than the
#: inline-promoted `name`.
REMOVE_PROP = "c0"

#: There is deliberately **no memory-budget skip guard** here any more.
#:
#: D1 Phase 0 (2026-08-09) added one, because the node-private write route
#: retained ~44-59 MB of peak RSS *per row* and `merge[100000-500-saved]` was
#: SIGKILLed on a 17 GB machine — three cells skipped behind a predicted-peak
#: budget. Phase 4 (2026-08-10) re-measured every one of them with `ru_maxrss`
#: after the ownership change: **zero** per-row growth on all three, and on the
#: master and title routes besides. `merge[100000-500-saved]`, predicted at
#: ~34 GiB, completes in ~2 ms with no measurable RSS delta.
#:
#: The guard was a *static* prediction (`rows x 2.5 x store_bytes`), so it could
#: not clear itself when the mechanism it predicted was deleted — it would have
#: gone on hiding three now-cheap cells indefinitely. Deleting it is the
#: rebaseline. If a future change reintroduces per-row store retention, these
#: cells are the ones that will show it, which is why they must run.


#: Where a `SET` lands, and the reason the SET cell has this extra axis:
#: `add_nodes(..., "id", "name")` promotes `name` to the node **title**, so a
#: `SET n.name` writes the inline title field *and* the `name` store column via
#: the `GraphWrite::set_node_property` fallback, while `SET n.c1` goes through
#: `write_column_master`. Two routes, two pre-image capture sites
#: (`storage/impls.rs::capture_property_pre_image` and
#: `columnar_write.rs::write_column_master`) — measured separately so a
#: regression in one is not hidden by the other. Note the MERGE cells below
#: write `name`, i.e. they sit on the `title` route.
SET_TARGETS = {"column": "c1", "title": "name"}


def _frame(rows: int, offset: int) -> pd.DataFrame:
    data = {
        "id": range(offset, offset + rows),
        "name": [f"item-{i}" for i in range(rows)],
    }
    for c in range(EXTRA_COLUMNS):
        data[f"c{c}"] = [f"v{c}-{i}" for i in range(rows)]
    return pd.DataFrame(data)


def _variant_graph(size: int, variant: str, tmp_dir) -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.define_schema({"nodes": {"Item": {"primary_key": "id"}}})
    graph.add_nodes(_frame(size, 0), "Item", "id", "name")
    if variant in ("saved", "reloaded"):
        # save() runs the consolidation pass over DirGraph's column stores.
        # fsync=False: this save exists to reach that pass, not to benchmark a
        # disk flush.
        path = str(tmp_dir / f"merge-{variant}-{size}.kgl")
        graph.save(path, fsync=False)
        if variant == "reloaded":
            # The other route to a consolidated store: attach_portable_column_stores
            # on the load path, rather than a rebuild in place.
            graph = kglite.load(path)

    # Vacuity guard, in the spirit of the one in test_bench_fast_write_path.py:
    # a cell whose fixture silently carries no column rows would report a
    # healthy number under a label promising the opposite. It no longer
    # *discriminates* the variants — all three own stores now — so it asserts
    # the rows are there rather than which shape they are in.
    assert graph.graph_info()["columnar_total_rows"] == size, (
        f"{variant} must carry its rows in the column store; "
        "without them this cell measures a code path the cost does not live on"
    )
    return graph


@pytest.fixture(scope="module")
def merge_graphs(tmp_path_factory) -> dict[tuple[int, str], KnowledgeGraph]:
    """One graph per (size, variant). Module-scoped — the 100k builds are
    seconds each and must not be repeated per cell."""
    tmp_dir = tmp_path_factory.mktemp("merge-columnar")
    variants = sorted(set(VARIANTS) | set(WRITE_VARIANTS))
    return {(size, variant): _variant_graph(size, variant, tmp_dir) for size in SIZES for variant in variants}


@pytest.mark.benchmark
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("rows", MERGE_ROWS)
@pytest.mark.parametrize("size", SIZES)
def test_bench_merge_rows_into_columnar_type(benchmark, merge_graphs, size, rows, variant):
    """One `MERGE` statement covering `rows` rows, against a type with `size`
    existing nodes.

    The cell the sprint's Phase 0 found missing. Read it two ways — across
    `rows` at fixed `size` (sub-linear is correct now that one statement pays
    one store clone; growth faster than linear means a per-row term is back),
    and across `size` at fixed `rows` (growth here is the pre-image clone, and
    is the term D1 left on the table).

    Every merged row is a genuine ON MATCH: the ids are drawn from the rows the
    fixture already created, so the statement never inserts and the number is
    the update path rather than a mix of insert and update.
    """
    graph = merge_graphs[(size, variant)]

    # ON MATCH, not ON CREATE — ids in [0, rows) already exist in every fixture
    # (all SIZES are >= max(MERGE_ROWS)). Merging new ids would measure inserts
    # and grow the fixture between rounds, making later rounds a different
    # experiment from earlier ones.
    ids = list(range(rows))

    statement = "UNWIND $ids AS i MERGE (n:Item {id: i}) ON MATCH SET n.name = 'merged'"

    def merge():
        return graph.cypher(statement, params={"ids": ids})

    benchmark.pedantic(merge, rounds=ROUNDS, iterations=1, warmup_rounds=WARMUP_ROUNDS)

    benchmark.extra_info["variant"] = variant
    benchmark.extra_info["existing_nodes"] = size
    benchmark.extra_info["merged_rows"] = rows
    benchmark.extra_info["columns"] = EXTRA_COLUMNS + 2


@pytest.mark.benchmark
@pytest.mark.parametrize("variant", WRITE_VARIANTS)
@pytest.mark.parametrize("target", sorted(SET_TARGETS))
@pytest.mark.parametrize("rows", WRITE_ROWS)
@pytest.mark.parametrize("size", SIZES)
def test_bench_set_rows_on_columnar_type(benchmark, merge_graphs, size, rows, target, variant):
    """One `SET` clause covering `rows` rows, against a type with `size` nodes.

    The headline cell of the ownership program, and the `target` axis is why it
    is two cells rather than one — the two write *destinations* take different
    routes through the same node:

    * `column` writes `c1`, a genuine store column, so it takes the master-write
      path (`set_via_column_master` -> `write_column_master`), which journals the
      pre-image itself.
    * `title` writes `name`, which `add_nodes` promoted to the node title, so it
      writes the inline title field *and* the `name` column through the
      `GraphWrite::set_node_property` fallback, whose pre-image is journalled by
      `capture_property_pre_image`.

    Two routes, two journalling sites, one shared consequence — the first write
    of the statement forks the whole store. Measuring only one of them would let
    a regression in the other hide; before D1 Phase 3 they differed by 30x.

    `fresh` is the before-save arm and the in-memory regression control;
    `saved` and `reloaded` are the two after-save arms. Read `saved / fresh` and
    `reloaded / fresh` at each size: a ratio flat across `SIZES` would mean the
    graph-size term is not there, and would falsify the program's premise.
    """
    graph = merge_graphs[(size, variant)]

    ids = list(range(rows))
    prop = SET_TARGETS[target]
    statement = f"UNWIND $ids AS i MATCH (n:Item {{id: i}}) SET n.{prop} = 'set'"

    def run_set():
        return graph.cypher(statement, params={"ids": ids})

    # Non-vacuity: a statement that matched nothing would report a beautifully
    # flat number for a code path never entered.
    run_set()
    written = graph.cypher(f"MATCH (n:Item {{id: 0}}) RETURN n.{prop} AS p").to_list()
    assert written and written[0]["p"] == "set", f"SET did not land on {variant}/{size}/{target}: {written!r}"

    benchmark.pedantic(run_set, rounds=WRITE_BENCH_ROUNDS, iterations=1, warmup_rounds=WRITE_WARMUP_ROUNDS)

    benchmark.extra_info["variant"] = variant
    benchmark.extra_info["existing_nodes"] = size
    benchmark.extra_info["written_rows"] = rows
    benchmark.extra_info["columns"] = EXTRA_COLUMNS + 2
    benchmark.extra_info["clause"] = "SET"
    benchmark.extra_info["target"] = target


@pytest.mark.benchmark
@pytest.mark.parametrize("variant", WRITE_VARIANTS)
@pytest.mark.parametrize("rows", WRITE_ROWS)
@pytest.mark.parametrize("size", SIZES)
def test_bench_remove_rows_on_columnar_type(benchmark, merge_graphs, size, rows, variant):
    """One `REMOVE` clause covering `rows` rows, against a type with `size` nodes.

    `execute_remove` is the second columnar write path, and its master-side
    write is a different one — a `Null` into the store rather than a value —
    so it is measured rather than inferred from `SET`.

    Each round is preceded by an **untimed** `setup` that puts the property
    back. Without it every round after the first would be removing an absent
    property, which is a different and much cheaper statement: the cell would
    look healthy for the wrong reason.
    """
    graph = merge_graphs[(size, variant)]
    ids = list(range(rows))
    restore = f"UNWIND $ids AS i MATCH (n:Item {{id: i}}) SET n.{REMOVE_PROP} = 'restored'"
    statement = f"UNWIND $ids AS i MATCH (n:Item {{id: i}}) REMOVE n.{REMOVE_PROP}"

    def setup():
        graph.cypher(restore, params={"ids": ids})

    def run_remove():
        return graph.cypher(statement, params={"ids": ids})

    probe = f"MATCH (n:Item {{id: 0}}) RETURN n.{REMOVE_PROP} AS p"

    # Non-vacuity: prove the property is there before, and gone after.
    setup()
    before = graph.cypher(probe).to_list()
    assert before and before[0]["p"] == "restored", (
        f"setup did not restore {REMOVE_PROP} on {variant}/{size}: {before!r}"
    )
    run_remove()
    after = graph.cypher(probe).to_list()
    assert not after or after[0].get("p") is None, (
        f"REMOVE left {REMOVE_PROP}={after!r} on {variant}/{size}; the cell would time a no-op"
    )

    benchmark.pedantic(
        run_remove,
        setup=setup,
        rounds=WRITE_BENCH_ROUNDS,
        iterations=1,
        warmup_rounds=WRITE_WARMUP_ROUNDS,
    )

    benchmark.extra_info["variant"] = variant
    benchmark.extra_info["existing_nodes"] = size
    benchmark.extra_info["written_rows"] = rows
    benchmark.extra_info["columns"] = EXTRA_COLUMNS + 2
    benchmark.extra_info["clause"] = "REMOVE"
