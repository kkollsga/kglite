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

Two mechanisms, one per statement and one per row.

**Per clause — the handle-refresh sweep.** Every node of a columnar type holds
its own strong `Arc<ColumnStore>` handle, so a master write forks the store and
`refresh_columnar_node_handles`
(`languages/cypher/executor/columnar_write.rs`) then re-points **every** node in
`type_indices[type]` at the fork. That is O(N_type) per `SET` / `REMOVE`
clause, *independent of how many rows the clause wrote* — a one-row `SET`
against a 100k type pays 100k re-points. It fires only on a saved graph,
because only a saved graph has column stores.

**Per row — MERGE's fan-out.** `MERGE` issues one `SET` clause per row
(`executor/write.rs`), each with a single-row `ResultSet`, and
`touched_columnar_types` is **local to each `execute_set` call**. So R rows pay
R sweeps; and because each sweep returns the master's strong count to `1 + N`,
the next row's `Arc::make_mut` deep-copies the whole `ColumnStore` again
(`ColumnStore::clone` copies `Vec<TypedColumn>`, whose heap arm is a `Vec<T>`).
The MERGE cost is therefore **O(R x (N + |store|))** — not the O(R x N) the
backlog originally recorded. The store term is what makes wide types
disproportionately expensive, and it is invisible in any cell that writes a
single row.

The root of both is that the node handle is an *owning* handle. The remedy
being pursued is **not** a `Weak` handle — that was evaluated and rejected,
because `Arc::make_mut` on the node's own handle is a load-bearing node-private
copy-on-write fork (disk write staging in `storage/disk/graph.rs` and the
`enable_columnar` drift check in `dir_graph/mod.rs` both depend on it), and a
`Weak` cannot fork. The chosen design instead makes the **storage backend the
sole owner** of the column stores: `PropertyStorage::Columnar` keeps only
`row_id`, every read resolves through a `GraphRead` accessor, and with no node
handle there is nothing to refresh — the sweep is deleted rather than made
cheaper.

────────────────────────────────────────────────────────────────────────────
How to read it
────────────────────────────────────────────────────────────────────────────

Two axes, and the interesting number is a **ratio**, never an absolute:

* **Across the row count at one `size`.** Linear in rows is the floor — R rows
  is R rows of work. Growing *faster* than linearly is the per-row compounding
  above.
* **Across `SIZES` at one row count.** This is the diagnostic axis. Writing R
  rows should not care how many nodes the type already has. If the 100k column
  is materially slower than the 1k column at the same R, the sweep is being
  paid. The single-row `SET` cell isolates it best: at R = 1 there is one row of
  real work and everything else on the clock is the sweep.

The `SET` cells carry a third axis, `target`, and it is not cosmetic: writing a
genuine store column (`c1`) and writing the promoted node **title** (`name`) are
two different routes through the same node, and they differ by more than an
order of magnitude on a saved graph. See `SET_TARGETS` and the cell's docstring.
The `MERGE` cells write `name`, so they sit on the title route.

`fresh` is the control, and doubles as the in-memory regression control the
ownership program must not regress by more than 5%. It has no column stores at
all, so it cannot take the master-write path and should stay flat across
`SIZES` no matter what happens to `saved` / `reloaded`. If `fresh` ever tracks
them, the cost is not columnar and this file's premise is wrong — treat that as
a finding, not as noise.

`saved` (`save()` in place) and `reloaded` (`save()` then `kglite.load()`) are
both columnar but arrive there by different routes — `enable_columnar()` versus
`attach_portable_column_stores` on the load path. They are measured separately
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
#: compared against theirs; the rest is where the compounding shows.
MERGE_ROWS = [1, 50, 500]

#: Rows written by the single measured `SET` / `REMOVE` statement. R = 1 is the
#: load-bearing cell: one clause, one row of real work, so everything else on
#: the clock is the per-clause O(N_type) handle-refresh sweep. R = 500 shows
#: whether the clause cost is amortised over rows (it should be — unlike MERGE,
#: a multi-row `SET` is one clause and therefore one sweep).
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
#: The fork copies every column, so a narrow type would show the sweep cost and
#: hide the deep-copy cost — and the deep copy is the half the backlog missed.
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

#: Peak resident memory a single measured statement may be *predicted* to reach
#: before the cell is skipped instead of run. This exists because the cost below
#: is not only a latency cost: during the D1 Phase-0 capture (2026-08-09) the
#: pre-existing `merge[100000-500-saved]` cell was SIGKILLed on a 17 GB machine.
#: A killed process yields no baseline at all, so the guard converts an
#: un-measurable cell into a named skip that states the predicted figure. It is
#: a finding about the code under test, not a harness workaround — when the
#: node-private fork goes away the prediction collapses and the skips clear
#: themselves, which is the evidence Phase 4 should look for.
MEMORY_BUDGET_BYTES = 6 * 1024**3

#: Whole-store copies a *node-private* write retains per row. Measured
#: 2026-08-09 at `size=100_000` (store 25.7 MB, peak RSS via `ru_maxrss`):
#: `MERGE … ON MATCH SET n.name` peaked at ~44 MB/row and a bare `SET n.name`
#: at ~59 MB/row, i.e. 1.7x-2.3x the store per row. 2.5 is the conservative
#: bound the guard predicts with. Writes that take the *master* path
#: (`SET n.c1`, `REMOVE n.c0`) showed no per-row growth at all — 500 rows cost
#: the same peak RSS as 1 — so they are not guarded.
NODE_PRIVATE_STORE_COPIES_PER_ROW = 2.5


def _skip_if_predicted_to_exhaust_memory(graph: KnowledgeGraph, rows: int, *, node_private: bool) -> None:
    """Skip, loudly, a cell whose single statement would not fit in RAM.

    Only the node-private route grows with `rows`; a master-path write does not,
    so passing `node_private=False` is a no-op rather than a smaller budget.
    """
    if not node_private:
        return
    store_bytes = graph.graph_info().get("columnar_heap_bytes") or 0
    predicted = rows * NODE_PRIVATE_STORE_COPIES_PER_ROW * store_bytes
    if predicted > MEMORY_BUDGET_BYTES:
        pytest.skip(
            f"predicted peak ~{predicted / 1024**3:.1f} GiB > {MEMORY_BUDGET_BYTES / 1024**3:.0f} GiB budget: "
            f"{rows} rows x ~{NODE_PRIVATE_STORE_COPIES_PER_ROW} node-private copies of a "
            f"{store_bytes / 1024**2:.1f} MiB store. This cell is unmeasurable, not fast."
        )


#: Where a `SET` lands, and the reason the SET cell has this extra axis:
#: `add_nodes(..., "id", "name")` promotes `name` to the node **title**, which
#: on a columnar node is *not* in the column store. Writing it therefore takes
#: `PropertyStorage::insert` → `Arc::make_mut` on the node's own handle (a
#: node-private deep copy of the store, per row), while writing `c1` takes the
#: master-write path the sweep belongs to. Conflating the two would attribute
#: one path's cost to the other. Note the existing MERGE cells below write
#: `name`, i.e. they sit on the `title` route.
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
        # save() is what populates DirGraph.column_stores via enable_columnar().
        # fsync=False: this save exists to flip the storage shape, not to
        # benchmark a disk flush.
        path = str(tmp_dir / f"merge-{variant}-{size}.kgl")
        graph.save(path, fsync=False)
        if variant == "reloaded":
            # The other route into columnar storage: attach_portable_column_stores
            # on the load path, rather than enable_columnar() in place.
            graph = kglite.load(path)

    # Vacuity guard, in the spirit of the one in test_bench_fast_write_path.py:
    # a `saved` cell that silently is not columnar would report a healthy number
    # under a label promising the opposite — indistinguishable from a fix.
    expect_columnar = variant != "fresh"
    assert graph.is_columnar is expect_columnar, (
        f"{variant} must{'' if expect_columnar else ' not'} own column stores; "
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
    `rows` at fixed `size` (linear is the floor), and across `size` at fixed
    `rows` (flat is correct; growth is the per-row fork and sweep).

    Every merged row is a genuine ON MATCH: the ids are drawn from the rows the
    fixture already created, so the statement never inserts and the number is
    the update path rather than a mix of insert and update.
    """
    graph = merge_graphs[(size, variant)]

    # `ON MATCH SET n.name` writes the promoted title, i.e. the node-private
    # route — see SET_TARGETS. That is what makes this cell's memory grow with
    # `rows`, and what makes the 100k x 500 corner unrunnable today.
    _skip_if_predicted_to_exhaust_memory(graph, rows, node_private=variant != "fresh")

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
      path (`set_via_column_master`) once per row and the O(N_type)
      handle-refresh sweep once per clause.
    * `title` writes `name`, which `add_nodes` promoted to the node title and is
      therefore **not** in the store. On a columnar node that write goes through
      `PropertyStorage::insert`, i.e. `Arc::make_mut` on the node's *own* handle
      — a node-private deep copy of the whole store, **per row**.

    Measuring only one of them would misreport the program by a large factor in
    either direction, so both are here and each cell says which it is.

    `fresh` is the before-save arm and the in-memory regression control;
    `saved` and `reloaded` are the two after-save arms. Read `saved / fresh` and
    `reloaded / fresh` at each size: a ratio flat across `SIZES` would mean the
    graph-size term is not there, and would falsify the program's premise.
    """
    graph = merge_graphs[(size, variant)]
    _skip_if_predicted_to_exhaust_memory(graph, rows, node_private=target == "title" and variant != "fresh")

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

    `REMOVE` is the second caller of the sweep (`execute_remove`), and its
    master-side write is a different one — a `Null` store on the master rather
    than a value write — so it is measured rather than inferred from `SET`.

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
