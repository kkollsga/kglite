"""What does one multi-row `MERGE` cost against a saved (columnar) type?

The gap this file fills. `test_bench_fast_write_path.py` measures a **one-row**
`MERGE` across the four corners of the `journal_covers` gate, and
`test_bench_write_scaling.py` measures inserts as the graph grows. Neither
measures the shape that carries the remaining columnar cost: **one statement
merging R rows into a type that already has N nodes.**

────────────────────────────────────────────────────────────────────────────
The shape, and why it is worse than "R times a one-row MERGE"
────────────────────────────────────────────────────────────────────────────

`MERGE` issues one `SET` clause per row (`executor/write.rs`), and each clause
calls `execute_set` with a single-row `ResultSet`. Two costs compound:

* `touched_columnar_types` is **local to each `execute_set` call**, so the
  end-of-clause handle-refresh sweep re-points every node of the type at the
  fork. After the sweep the master's strong count is back to `1 + N`, which
  means the **next** row's `Arc::make_mut` forks the whole `ColumnStore` again.
* `ColumnStore::clone` is a genuine deep copy — `columns: self.columns.clone()`
  over `Vec<TypedColumn>` whose heap arm is a `Vec<T>`.

So the cost is **O(R x (N + |store|))**: R full column-store deep copies *plus*
R O(N) sweeps — not the O(R x N) the backlog originally recorded. The store
term is what makes wide types disproportionately expensive, and it is invisible
in any cell that merges a single row.

Every node holding its own strong `Arc<ColumnStore>` handle is the root, and it
is pinned as intended design by
`rollback_tests.rs::every_node_shares_the_master_column_store_handle`. Changing
it to `Weak` is the bounded fix — the first write of a statement still forks
(the journal's `prior` clone keeps the strong count at 2), while every
subsequent write mutates in place, collapsing R forks + R sweeps to one each.

────────────────────────────────────────────────────────────────────────────
How to read it
────────────────────────────────────────────────────────────────────────────

Two axes, and the interesting number is a **ratio**, never an absolute:

* **Across `MERGE_ROWS` at one `size`.** Linear in rows is the floor — R rows
  is R rows of work. Growing *faster* than linearly is the compounding above.
* **Across `SIZES` at one row count.** This is the diagnostic axis. Merging R
  rows should not care how many nodes the type already has. If the 100k column
  is materially slower than the 1k column at the same R, the per-row sweep and
  fork are being paid, and that is exactly what the weak-handle change removes.

`fresh` is the control. It has no column stores at all, so it cannot take the
master-write path and should stay flat across `SIZES` no matter what happens to
`saved`. If `fresh` ever tracks `saved`, the cost is not columnar and this
file's premise is wrong — treat that as a finding, not as noise.

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

from kglite import KnowledgeGraph

#: Existing nodes in the merged type. The diagnostic axis: the cost of merging
#: R rows must not depend on N, and two decades of N make a dependency obvious.
SIZES = [1_000, 100_000]

#: Rows merged by the single measured statement. 1 is the degenerate case the
#: other files already cover, and is included so this file's own numbers can be
#: compared against theirs; the rest is where the compounding shows.
MERGE_ROWS = [1, 50, 500]

#: `fresh` owns no column stores, so it cannot take the master-write path —
#: the control that separates "columnar cost" from "MERGE is just slow".
VARIANTS = ["fresh", "saved"]

#: Wide enough that `|store|` is a visible term rather than a rounding error.
#: The fork copies every column, so a narrow type would show the sweep cost and
#: hide the deep-copy cost — and the deep copy is the half the backlog missed.
EXTRA_COLUMNS = 12

ROUNDS = 5
WARMUP_ROUNDS = 1


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
    if variant == "saved":
        # save() is what populates DirGraph.column_stores via enable_columnar().
        # fsync=False: this save exists to flip the storage shape, not to
        # benchmark a disk flush.
        graph.save(str(tmp_dir / f"merge-{variant}-{size}.kgl"), fsync=False)

    # Vacuity guard, in the spirit of the one in test_bench_fast_write_path.py:
    # a `saved` cell that silently is not columnar would report a healthy number
    # under a label promising the opposite — indistinguishable from a fix.
    expect_columnar = variant == "saved"
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
    return {(size, variant): _variant_graph(size, variant, tmp_dir) for size in SIZES for variant in VARIANTS}


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
