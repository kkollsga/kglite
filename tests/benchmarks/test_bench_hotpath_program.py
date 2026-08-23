"""Before/after oracles for the perf hot-path program.

**These are program-lifecycle cells, not gate cells.** They exist to pin the
program's ten hot-path findings at branch HEAD so each later phase can show
what it moved, and they assert nothing about time.
The release that ships the program may retire them (a defect that is gone needs
no standing cell) or promote the survivors into `test_bench_core.py` with a
baseline recapture. Neither is decided here.

Deliberately **not** in `test_bench_core.py`: that file is the tracked set
`make bench-check` collects and compares against `baselines/current*.json`, and
on Linux with `--require-exact-set`, which *errors* on a benchmark present in
the run but absent from the baseline. `bench-check` names `test_bench_core.py`
explicitly (see the Makefile target), so nothing here can reach the gate.

Each cell carries a structural (non-vacuity) assertion — that the fixture really
has 200 node types, that the `SET` really landed, that the mapped graph really
came back mapped. Those are wanted and must stay: every benchmark defect this
project has caught presented as a *plausible* number under a wrong label, never
as an implausible one. No cell asserts on a duration.

Run with::

    uv run --no-sync maturin develop --release --no-default-features \\
        --features abi3,python-extension
    .venv/bin/python -m pytest tests/benchmarks/test_bench_hotpath_program.py \\
        -m benchmark --benchmark-min-rounds=100 \\
        --benchmark-warmup=on --benchmark-warmup-iterations=20 -v

The `benchmark` marker is exempt from the 120 s pytest hang ceiling
(`tests/conftest.py::pytest_collection_modifyitems`), but no fixture here needs
the exemption: the most expensive one (1M nodes) builds in ~0.4 s and the whole
file runs in well under a minute. The O(V)-per-call cells drive
`benchmark.pedantic` with explicit round counts rather than plain
`benchmark(fn)`, because auto-calibration times the *first* call and several of
these shapes have a first call that is orders of magnitude cheaper than every
call after it (the id-index-invalidation trap documented at length in
`test_bench_write_scaling.py`).

Pre-program baseline
--------------------

Captured 2026-08-14 on branch `refactor/shape-convergence` at HEAD a97256e7,
release profile (`--no-default-features --features abi3,python-extension`),
macOS arm64 (Darwin 25.3.0). **Machine state: not idle** — 5 user sessions,
1-minute load average 1.7-2.3 throughout, an editor and a browser resident.
Recorded per the longitudinal-capture doctrine (CLAUDE.md, Performance protocol
item 7/9): these are cross-session comparison inputs, so the state they were
taken under is part of the number.

`min`, both runs, ~60 s apart. Both are given so the spread is visible rather
than averaged away; every cell agrees within 6% between them, and the two
noisiest (mapped SET, 12%; large-type delete, 3.7%) are called out below.
Whole file runs in 13.4 s including all fixtures.

===================================== ========= ========= ==============
cell                                    run 1     run 2      scan says
===================================== ========= ========= ==============
wide_schema_stmt_overhead (SET)        225.8 us  226.0 us   ~226 us
wide_schema_match_control                1.95 us   1.94 us  (control)
incremental_add_nodes_append            10.03 ms  10.33 ms  10.2 ms
single_delete_from_large_type            3.89 ms   4.03 ms  4.0 ms
single_delete_from_small_type            6.92 us   6.96 us  7.8 us
mapped_set_many_types                  257.4 us  288.5 us   319.6 us
memory_set_many_types (control)         17.87 us  16.29 us  17.0 us
wide_sparse_properties_fn               33.82 ms  33.77 ms  23.6 ms
wide_dense_keys_fn                      62.06 ms  61.36 ms  45.7 ms
no_index_mass_set                       45.69 ms  44.79 ms  ~45 ms
unchanged_save_1m                       48.56 ms  46.72 ms  ~45 ms
create_wide_seed                         4.18 ms   4.28 ms  5.25 ms
fused_property_access                    4.85 ms   4.69 ms  (slope)
fused_constant_control                   2.49 ms   2.50 ms  (control)
scan_clean_string_column               426.0 us  428.7 us   (control)
scan_relocated_string_column           597.1 us  601.2 us   +29%
===================================== ========= ========= ==============

The derived quantities the phase targets are stated in, since several of these
cells are only meaningful as a pair:

* **schema-shell clone** (P2, SET - MATCH): **223.8 / 224.1 us**. The scan's
  225.6 us, reproduced to within 1%.
* **mapped spill tax** (P4, mapped / memory): **14.4x / 17.7x**.
* **delete scaling** (P3, 1M-type / 1k-type): **562x / 579x**.
* **mass-SET per row** (P6): **457 / 448 ns/row** over 100k rows. The scan's
  450 ns/row, to within 2%.
* **CREATE per node** (P5): **836 / 857 ns/node** over a 5k batch.
* **property-access slope** (P7, (property - constant) / 200,000 accesses):
  **11.8 / 10.9 ns per access**, i.e. 47.1 / 43.7 ns per 4-access row.
* **string-relocation tax** (P4, relocated / clean): **+40.2% / +40.2%**, from
  one single-row write.

Where these disagree with the scan, and why it matters
-----------------------------------------------------

Six write-path cells land on the scan's numbers within a few percent, which is
the evidence that the fixtures reproduce the shapes the scan profiled. Four
diverge, and a later phase reading only the scan would draw the wrong
conclusion from each:

* `wide_sparse_properties_fn` 33.8 ms vs 23.6, and `wide_dense_keys_fn` 61.7 vs
  45.7 — both ~1.4x the scan. The write cells on the same machine, in the same
  session, matched exactly, so this is not machine drift; the read cells were
  measured here as statement `min` over 15 rounds and the scan quoted its own
  probe's statistic. **Use the numbers in this table as P5/P8's baseline, not
  the scan's** — a phase measuring against 23.6 ms would credit itself with a
  30% win it did not earn.
* `mapped_set_many_types` 257-289 us vs 319.6, and `create_wide_seed` 836-857
  ns/node vs 1.05 us — this file's fixtures are somewhat smaller than the
  scan's (100 types x 100 rows; a 1k-row seed). Same defect, smaller absolute
  number. P4/P5 targets scale accordingly.
* The relocation tax reads **+40%** here against the scan's +29%. The
  relocated arm agrees between the two measurements; the *clean* control is
  faster here (426 us vs the probe's 478), so the ratio widens. That is the
  expected direction for a `min`-over-2300-rounds control against a small-n
  mean, and it makes the tax bigger, not smaller.

One open question for P7, flagged rather than resolved: at 10.9-11.8 ns per
access, `fused_property_access` sits inside the scan's "~10-14 ns/row/property"
band for the *unhoisted* fused shapes, but well under the plan's stated
"21-25 -> <=15 ns/row" target. An aggregate over a bare `MATCH (n:T)` may
already be taking the hoisted route (`fused_match.rs:531-573`), in which case
this pair is a **control** for P7 rather than its subject, and P7 needs a cell
on one of the three shapes that demonstrably re-resolves per row. Confirm which
before reading a flat result here as a failure to improve.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph

# ---------------------------------------------------------------------------
# Round counts for the `pedantic` cells.
#
# Plain `benchmark(fn)` is used only where the call is cheap AND its cost does
# not depend on how many times it has already run. Everything else is pedantic
# with an explicit budget, so total runtime is bounded no matter what
# `--benchmark-min-rounds` the caller passes.
# ---------------------------------------------------------------------------

#: ~4-11 ms per call (append, delete, CREATE batch).
MS_ROUNDS, MS_WARMUP = 25, 2

#: ~35-60 ms per call (mass SET, 1M save, properties/keys over 20k).
TENS_MS_ROUNDS, TENS_MS_WARMUP = 15, 2

#: ~2-5 ms per call, read-only and repeatable (the fused-shape pair).
READ_ROUNDS, READ_WARMUP = 40, 3

#: The CREATE cell rebuilds its graph in an untimed setup every round; 10 is
#: enough for a steady `min` and keeps the rebuild budget trivial.
CREATE_ROUNDS, CREATE_WARMUP = 10, 1


def _wal_bytes(path) -> int:
    """Size of the `<path>-wal` sidecar, or 0 when no log exists.

    The only Python-visible window onto the WAL, and enough to prove a cell ran
    at the durability level it claims. Same helper as
    `test_bench_write_scaling.py`; duplicated rather than shared because these
    two files have no other coupling and a benchmark-support import between them
    would be one.
    """
    try:
        return os.path.getsize(str(path) + "-wal")
    except FileNotFoundError:
        return 0


# ===========================================================================
# Scan #1 — the per-statement rollback checkpoint deep-clones the schema shell
# ===========================================================================
#
# `schema_shell` (rollback.rs) is a full `DirGraph::clone` minus ten parked
# fields, so its cost is O(types x properties) and independent of node count.
# The scan measured ~22 ns per schema cell: 0.9 us at 1 type x 3 props, 225.6 us
# at 200 x 50.
#
# The measurement is a **difference**, which is why there are two cells. A
# `MATCH` that binds nothing takes no checkpoint; the otherwise identical `SET`
# takes one and then rolls it back over zero changes. Everything else — parse,
# plan-cache hit, the failed index probe — is common to both, so
# `SET - MATCH` isolates the shell clone and nothing else. Reading the SET cell
# alone would attribute the whole statement to the defect.

#: 200 node types x 50 declared columns = 10,000 schema cells, the scan's
#: headline shape. `id` and `name` are two of the 50, so the type carries 48
#: further `p*` columns.
SCHEMA_TYPES = 200
SCHEMA_COLS = 50

#: Nodes per type. 200 x 50 = 10,000 nodes total, matching the scan's "10k
#: graph". The node count is deliberately small: the defect is node-count-
#: independent, and a larger graph would only add fixture time.
SCHEMA_NODES_PER_TYPE = 50


@pytest.fixture(scope="module")
def wide_schema_graph() -> KnowledgeGraph:
    """10k nodes across `SCHEMA_TYPES` types of `SCHEMA_COLS` columns each.

    Builds in ~0.3 s.
    """
    graph = KnowledgeGraph()
    for t in range(SCHEMA_TYPES):
        data: dict = {
            "id": range(SCHEMA_NODES_PER_TYPE),
            "name": [f"x{i}" for i in range(SCHEMA_NODES_PER_TYPE)],
        }
        for c in range(SCHEMA_COLS - 2):
            data[f"p{c}"] = [i + c for i in range(SCHEMA_NODES_PER_TYPE)]
        graph.add_nodes(pd.DataFrame(data), f"T{t}", "id", "name")
    # Warm the plan cache and the id index so neither lands in a sample.
    graph.cypher("MATCH (n:T0 {id: 0}) RETURN n.p0")
    return graph


def _assert_wide_schema(graph: KnowledgeGraph) -> None:
    info = graph.graph_info()
    assert info["type_count"] == SCHEMA_TYPES, (
        f"the shell clone is O(types x properties); with {info['type_count']} types "
        "instead of 200 this cell measures a different shape than the scan did"
    )
    assert info["node_count"] == SCHEMA_TYPES * SCHEMA_NODES_PER_TYPE
    assert graph.cypher("MATCH (n:T0 {id: -1}) RETURN count(n) AS c").scalar() == 0, (
        "both cells depend on the pattern binding nothing; a match would add "
        "real write work to the SET arm and destroy the difference"
    )


@pytest.mark.benchmark
def test_bench_wide_schema_stmt_overhead(benchmark, wide_schema_graph):
    """A `SET` that matches no node, on a 200-type x 50-column schema.

    Zero rows are written, so essentially all of this is the rollback
    checkpoint's schema-shell clone plus its drop. Subtract
    `test_bench_wide_schema_match_control` to get the shell cost alone; that
    difference is P2's target (226 us -> <=5 us).
    """
    _assert_wide_schema(wide_schema_graph)
    counter = iter(range(1, 1 << 30))

    def write():
        wide_schema_graph.cypher("MATCH (n:T0 {id: -1}) SET n.p0 = $v", params={"v": next(counter)})

    benchmark(write)
    benchmark.extra_info["types"] = SCHEMA_TYPES
    benchmark.extra_info["columns_per_type"] = SCHEMA_COLS


@pytest.mark.benchmark
def test_bench_wide_schema_match_control(benchmark, wide_schema_graph):
    """The same non-matching pattern with no write clause — the control.

    Identical parse, plan-cache lookup and failed id probe; no checkpoint. It is
    what makes the cell above a measurement of the shell rather than of a
    statement, and it must stay flat across the program: a control that moves
    means the instrument moved.
    """
    _assert_wide_schema(wide_schema_graph)

    def read():
        wide_schema_graph.cypher("MATCH (n:T0 {id: -1}) RETURN n.p0")

    benchmark(read)
    benchmark.extra_info["types"] = SCHEMA_TYPES
    benchmark.extra_info["columns_per_type"] = SCHEMA_COLS


# ===========================================================================
# Scan #2 — incremental `add_nodes` rebuilds the type's id index twice per call
# ===========================================================================

#: Existing rows in the appended-to type. The defect is O(existing), ~51 us per
#: 1k rows, so 200k is where 10 appended rows cost 10 ms.
APPEND_EXISTING = 200_000

#: Rows per appended batch — a streaming-ingest shape, not a bulk load.
APPEND_ROWS = 10


@pytest.fixture(scope="module")
def append_graph() -> KnowledgeGraph:
    """One type of `APPEND_EXISTING` rows with a declared primary key.

    The key is declared on purpose: without one the type has no id index to
    rebuild, and the cell would measure a different (cheaper, and separately
    tracked) path. Builds in ~0.1 s.
    """
    graph = KnowledgeGraph()
    graph.define_schema({"nodes": {"Item": {"primary_key": "id"}}})
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(APPEND_EXISTING),
                "name": [f"n{i}" for i in range(APPEND_EXISTING)],
                "v": [i % 977 for i in range(APPEND_EXISTING)],
            }
        ),
        "Item",
        "id",
        "name",
    )
    graph.cypher("MATCH (n:Item {id: 0}) RETURN n.id")
    return graph


@pytest.mark.benchmark
def test_bench_incremental_add_nodes_append(benchmark, append_graph):
    """`APPEND_ROWS` new rows appended to a type that already has 200k.

    P3's target: 10.2 ms -> <=0.5 ms. The frame is built in an untimed
    `setup` so the measurement is the append and not pandas.

    The graph grows by `APPEND_ROWS` per round — 270 rows over the cell's
    budget, 0.14% of the type, which is below the resolution of what is being
    measured. Ids are drawn from a disjoint high range so no round ever hits the
    uniqueness path instead of the insert path.
    """
    before = append_graph.graph_info()["node_count"]
    blocks = iter(range(1_000_000, 1 << 29))
    last: dict = {}

    def setup():
        last["base"] = next(blocks) * APPEND_ROWS
        frame = pd.DataFrame(
            {
                "id": range(last["base"], last["base"] + APPEND_ROWS),
                "name": [f"a{i}" for i in range(APPEND_ROWS)],
                "v": [1] * APPEND_ROWS,
            }
        )
        return (frame,), {}

    def append(frame):
        append_graph.add_nodes(frame, "Item", "id", "name")

    benchmark.pedantic(append, setup=setup, rounds=MS_ROUNDS, iterations=1, warmup_rounds=MS_WARMUP)

    # Round-count *independent* on purpose. Asserting `before + rounds * rows`
    # ties the guard to how many times pytest-benchmark chose to call the
    # function, which `--benchmark-disable` (one call) and `--benchmark-skip`
    # both change -- the cell would then fail for a reason that has nothing to
    # do with the graph. What must be true either way is that the final round
    # inserted its own batch, and that the type only ever grew.
    after = append_graph.graph_info()["node_count"]
    assert after > before and (after - before) % APPEND_ROWS == 0, (
        f"the type must have grown by whole batches of {APPEND_ROWS}; {before} -> {after}"
    )
    landed = append_graph.cypher(
        "MATCH (n:Item) WHERE n.id >= $lo AND n.id < $hi RETURN count(n) AS c",
        params={"lo": last["base"], "hi": last["base"] + APPEND_ROWS},
    ).scalar()
    assert landed == APPEND_ROWS, (
        f"the last round inserted {landed} of {APPEND_ROWS} rows; some rounds upserted "
        "or were rejected, which is a different code path from the append being measured"
    )
    benchmark.extra_info["existing_rows"] = APPEND_EXISTING
    benchmark.extra_info["appended_rows"] = APPEND_ROWS


# ===========================================================================
# Scan #3 — single-node DETACH DELETE costs O(size of its type)
# ===========================================================================
#
# `retain_in_type` walks the whole bucket with SipHash set probes, so deleting
# one node from a 1M-row type costs 4.0 ms while the same statement against a
# 1k-row type in the *same graph* costs 7.8 us. Both types live in one fixture
# because that pairing is the evidence: the cost is scoped to the deleted node's
# type, not to the graph, which is what identifies the bucket walk as the
# mechanism rather than anything global.
#
# Pre-existing (identical on the 0.15.14 wheel), and quadratic in a delete loop.

DELETE_BIG = 1_000_000
DELETE_SMALL = 1_000


@pytest.fixture(scope="module")
def delete_graph() -> KnowledgeGraph:
    """A 1M-row type and a 1k-row type in one graph. Builds in ~0.4 s."""
    graph = KnowledgeGraph()
    graph.define_schema({"nodes": {"Big": {"primary_key": "id"}, "Small": {"primary_key": "id"}}})
    graph.add_nodes(
        pd.DataFrame({"id": range(DELETE_BIG), "name": [f"b{i}" for i in range(DELETE_BIG)]}),
        "Big",
        "id",
        "name",
    )
    graph.add_nodes(
        pd.DataFrame({"id": range(DELETE_SMALL), "name": [f"s{i}" for i in range(DELETE_SMALL)]}),
        "Small",
        "id",
        "name",
    )
    graph.cypher("MATCH (n:Big {id: 0}) RETURN n.id")
    graph.cypher("MATCH (n:Small {id: 0}) RETURN n.id")
    return graph


def _delete_cell(benchmark, graph: KnowledgeGraph, label: str, ids) -> None:
    """One `DETACH DELETE` per round, each round a different pre-created node.

    Every round targets a node the fixture already built — no per-round rebuild,
    which is what keeps a 1M fixture affordable. The trailing assertion proves
    the last round actually removed something: a `DETACH DELETE` that bound
    nothing would report a fast, flat, entirely meaningless number.
    """
    last: dict = {}

    def delete():
        last["id"] = next(ids)
        graph.cypher(f"MATCH (n:{label} {{id: $i}}) DETACH DELETE n", params={"i": last["id"]})

    benchmark.pedantic(delete, rounds=MS_ROUNDS, iterations=1, warmup_rounds=MS_WARMUP)

    probe = f"MATCH (n:{label} {{id: $i}}) RETURN count(n) AS c"
    assert graph.cypher(probe, params={"i": last["id"]}).scalar() == 0, (
        f"the last round did not delete {label}.id={last['id']}; this cell timed a no-op"
    )


@pytest.mark.benchmark
def test_bench_single_delete_from_large_type(benchmark, delete_graph):
    """One `DETACH DELETE` from a 1M-row type. P3's target: 4.0 ms -> <=100 us."""
    assert delete_graph.cypher("MATCH (n:Big) RETURN count(n) AS c").scalar() > DELETE_BIG * 0.99
    _delete_cell(benchmark, delete_graph, "Big", iter(range(DELETE_BIG - 1, 0, -1)))
    benchmark.extra_info["type_rows"] = DELETE_BIG


@pytest.mark.benchmark
def test_bench_single_delete_from_small_type(benchmark, delete_graph):
    """The same statement against a 1k-row type in the same graph — the control.

    Its ratio to the cell above is the scaling term. It should stay roughly
    where it is: P3 removes the term, so the *large* cell must fall to meet this
    one rather than this one rising.
    """
    _delete_cell(benchmark, delete_graph, "Small", iter(range(DELETE_SMALL - 1, 0, -1)))
    benchmark.extra_info["type_rows"] = DELETE_SMALL


# ===========================================================================
# Scan #4 — mapped mode re-runs the full spill pass on every statement
# ===========================================================================
#
# `heap_bytes` counts unspillable tombstones, so the trigger sees total > 0 =
# limit forever and the per-type loop (Vec + sort + create_dir_all) runs on
# every statement. The cost is therefore O(types) per statement rather than
# O(rows), which is why the fixture is wide in types and small in nodes.

MANY_TYPES = 100
MANY_TYPES_ROWS = 100


def _many_types_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    for t in range(MANY_TYPES):
        graph.add_nodes(
            pd.DataFrame(
                {
                    "id": range(MANY_TYPES_ROWS),
                    "name": [f"n{i}" for i in range(MANY_TYPES_ROWS)],
                    "v": list(range(MANY_TYPES_ROWS)),
                }
            ),
            f"T{t}",
            "id",
            "name",
        )
    return graph


@pytest.fixture(scope="module")
def memory_many_types_graph() -> KnowledgeGraph:
    graph = _many_types_graph()
    graph.cypher("MATCH (n:T0 {id: 0}) RETURN n.id")
    return graph


@pytest.fixture(scope="module")
def mapped_many_types_graph(tmp_path_factory):
    """The same graph, saved and reopened `storage="mapped"`.

    `durable="off"` is explicit and load-bearing. `kglite.open()` resolves an
    unspecified `durable=` to full fsync, which would make this a measurement of
    the WAL barrier rather than of the spill pass, and its control
    (`memory_many_types_graph`) is a plain `KnowledgeGraph` with no log at all —
    anything else makes the two numbers incomparable. The `_wal_bytes` guard
    below turns that from an intention into a reading.

    The bulk load happens through a plain handle before the reopen, so it never
    lands in the measurement.
    """
    path = tmp_path_factory.mktemp("hotpath-mapped") / "many.kgl"
    seed = _many_types_graph()
    seed.save(str(path), fsync=False)
    seed.close()

    graph = kglite.open(str(path), storage="mapped", durable="off")
    graph.cypher("MATCH (n:T0 {id: 0}) RETURN n.id")
    yield graph, path
    graph.close()


@pytest.mark.benchmark
def test_bench_mapped_set_many_types(benchmark, mapped_many_types_graph):
    """A single-row `SET` on a mapped graph with 100 node types.

    P4's target: 319.6 us -> <=25 us, i.e. within touching distance of the
    memory control below.
    """
    graph, path = mapped_many_types_graph
    info = graph.graph_info()
    assert info["storage_mode"] == "mapped", (
        f"reopen landed in {info['storage_mode']!r}; a memory graph here would "
        "report a healthy number for a code path the defect does not live on"
    )
    assert info["type_count"] == MANY_TYPES, "the spill pass is per type; the type count is the shape"
    counter = iter(range(1, 1 << 30))

    def write():
        graph.cypher("MATCH (n:T0 {id: 7}) SET n.v = $x", params={"x": next(counter)})

    benchmark(write)

    assert _wal_bytes(path) == 0, "durable='off' must write no WAL; this cell is timing a log"
    assert graph.cypher("MATCH (n:T0 {id: 7}) RETURN n.v AS v").scalar() > 0
    benchmark.extra_info["types"] = MANY_TYPES
    benchmark.extra_info["storage"] = "mapped"


@pytest.mark.benchmark
def test_bench_memory_set_many_types(benchmark, memory_many_types_graph):
    """The same statement on the same shape in memory mode — the control.

    `maybe_spill` is a no-op in memory mode, so this is the cell above minus the
    defect. The gap between them (19x at capture) is what P4 closes.
    """
    graph = memory_many_types_graph
    assert graph.graph_info()["type_count"] == MANY_TYPES
    counter = iter(range(1, 1 << 30))

    def write():
        graph.cypher("MATCH (n:T0 {id: 7}) SET n.v = $x", params={"x": next(counter)})

    benchmark(write)
    assert graph.cypher("MATCH (n:T0 {id: 7}) RETURN n.v AS v").scalar() > 0
    benchmark.extra_info["types"] = MANY_TYPES
    benchmark.extra_info["storage"] = "memory"


# ===========================================================================
# Scan #5 (+ the `keys(n)` mention) — the columnar-completion pass per node
# ===========================================================================
#
# `helpers.rs` walks the *declared* type metadata for every materialized node
# and calls the full `resolve_node_property` for keys the node does not have.
# So the cost tracks declared columns, not populated ones — which is why the
# two cells below run on two different shapes, each the shape its scan number
# was taken on:
#
#   * `properties(n)` on 30 declared / 5 populated (scan item #5, 23.6 ms) —
#     the sparse shape, where the 25 absent columns are pure waste.
#   * `keys(n)` on 30 declared / 30 populated (the honourable mention, 45.7 ms)
#     — the dense shape, where the defect is that names are answered by cloning
#     every VALUE.
#
# Running both functions on both shapes would double the cell count for numbers
# neither scan item is about.

FN_NODES = 20_000
FN_DECLARED = 30
FN_POPULATED = 5

#: Id of the one dense row that declares the full column set. Chosen far outside
#: the measured range so no cell can collide with it.
FN_SEED_ID = 1 << 30


@pytest.fixture(scope="module")
def sparse_wide_graph() -> KnowledgeGraph:
    """20k nodes carrying 5 of the type's 30 declared columns.

    The declaration comes from one seeded dense row written *first*: property
    columns are declared by what has been written to the type, so a type whose
    every row carries 5 columns declares 5, and the sparse shape needs something
    to have declared the other 25. That row is 1 node in 20,001 (0.005% of the
    measured work) and is what makes this a 30-declared type rather than a
    5-declared one — verified by the assertion in the cell.
    """
    graph = KnowledgeGraph()
    seed: dict = {"id": [FN_SEED_ID], "name": ["seed"]}
    for c in range(FN_DECLARED):
        seed[f"p{c}"] = [c]
    graph.add_nodes(pd.DataFrame(seed), "T", "id", "name")

    data: dict = {"id": range(FN_NODES), "name": [f"n{i}" for i in range(FN_NODES)]}
    for c in range(FN_POPULATED):
        data[f"p{c}"] = [i + c for i in range(FN_NODES)]
    graph.add_nodes(pd.DataFrame(data), "T", "id", "name")
    return graph


@pytest.fixture(scope="module")
def dense_wide_graph() -> KnowledgeGraph:
    """20k nodes carrying all 30 of the type's declared columns."""
    graph = KnowledgeGraph()
    data: dict = {"id": range(FN_NODES), "name": [f"n{i}" for i in range(FN_NODES)]}
    for c in range(FN_DECLARED):
        data[f"p{c}"] = [i + c for i in range(FN_NODES)]
    graph.add_nodes(pd.DataFrame(data), "T", "id", "name")
    return graph


@pytest.mark.benchmark
def test_bench_wide_sparse_properties_fn(benchmark, sparse_wide_graph):
    """`count(properties(n))` over a 30-declared / 5-populated type.

    P5's target: 3.5x the 0.15.14 wheel -> <=1.3x. `count()` consumes the map
    without shipping it to Python, so the number is the completion pass and the
    map build rather than PyO3 marshalling.
    """
    ordinary = sparse_wide_graph.cypher("MATCH (n:T {id: 3}) RETURN keys(n) AS k").to_list()[0]["k"]
    seeded = sparse_wide_graph.cypher(f"MATCH (n:T {{id: {FN_SEED_ID}}}) RETURN keys(n) AS k").to_list()[0]["k"]
    # `keys` includes the synthetic `title`/`type` alongside `id`/`name`.
    assert len(ordinary) == FN_POPULATED + 4, f"measured rows must be sparse; got {ordinary}"
    assert len(seeded) == FN_DECLARED + 4, (
        f"the type must declare {FN_DECLARED} columns for the completion pass to have "
        f"25 absent ones to resolve; the seed row carries {len(seeded)} keys"
    )

    def read():
        return sparse_wide_graph.cypher("MATCH (n:T) RETURN count(properties(n)) AS c").scalar()

    assert read() == FN_NODES + 1
    benchmark.pedantic(read, rounds=TENS_MS_ROUNDS, iterations=1, warmup_rounds=TENS_MS_WARMUP)
    benchmark.extra_info["declared"] = FN_DECLARED
    benchmark.extra_info["populated"] = FN_POPULATED


@pytest.mark.benchmark
def test_bench_wide_dense_keys_fn(benchmark, dense_wide_graph):
    """`count(keys(n))` over a 30-declared / 30-populated type.

    P8's target: `keys(n)` returns *names*, and cloning every value to produce
    them is the whole of the honourable mention. Dense on purpose — the cost
    being measured is per populated value, so the sparse shape would understate
    it by 6x.
    """
    assert len(dense_wide_graph.cypher("MATCH (n:T {id: 3}) RETURN keys(n) AS k").to_list()[0]["k"]) == FN_DECLARED + 4

    def read():
        return dense_wide_graph.cypher("MATCH (n:T) RETURN count(keys(n)) AS c").scalar()

    assert read() == FN_NODES
    benchmark.pedantic(read, rounds=TENS_MS_ROUNDS, iterations=1, warmup_rounds=TENS_MS_WARMUP)
    benchmark.extra_info["declared"] = FN_DECLARED
    benchmark.extra_info["populated"] = FN_DECLARED


# ===========================================================================
# Scan #6 — index maintenance runs in full on a graph with zero indexes
# ===========================================================================

MASS_SET_ROWS = 100_000


@pytest.fixture(scope="module")
def no_index_graph() -> KnowledgeGraph:
    """100k rows, no declared primary key and no secondary index.

    No `define_schema` call: a declared key installs the identity index, and the
    point of this cell is a type with *nothing* to maintain. The assertion in
    the cell pins that.
    """
    graph = KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(MASS_SET_ROWS),
                "name": [f"n{i}" for i in range(MASS_SET_ROWS)],
                "x": [0] * MASS_SET_ROWS,
            }
        ),
        "T",
        "id",
        "name",
    )
    return graph


@pytest.mark.benchmark
def test_bench_no_index_mass_set(benchmark, no_index_graph):
    """`MATCH (n:T) SET n.x = $v` over 100k rows on an index-free type.

    Reported as a statement total; divide by `MASS_SET_ROWS` for the per-row
    figure the scan quotes (450 ns/row, of which the actual cell write is 10.4%).
    P6's target: <=300 ns/row.

    The written value varies per round so no round can be short-circuited as an
    unchanged write.
    """
    assert no_index_graph.graph_info()["property_index_count"] == 0, (
        "this cell measures index maintenance on a type that has no indexes; "
        "with an index present it measures ordinary index work instead"
    )
    assert no_index_graph.graph_info()["composite_index_count"] == 0
    counter = iter(range(1, 1 << 30))
    last: dict = {}

    def write():
        last["v"] = next(counter)
        no_index_graph.cypher("MATCH (n:T) SET n.x = $v", params={"v": last["v"]})

    benchmark.pedantic(write, rounds=TENS_MS_ROUNDS, iterations=1, warmup_rounds=TENS_MS_WARMUP)

    assert no_index_graph.cypher("MATCH (n:T {id: 5}) RETURN n.x AS x").scalar() == last["v"], (
        "the last round did not write; a SET matching nothing would report a "
        "beautiful number for a statement that did no work"
    )
    benchmark.extra_info["rows"] = MASS_SET_ROWS


# ===========================================================================
# Scan #7c — `save()`'s drift-check map is the top self symbol of a 1M save
# ===========================================================================

SAVE_ROWS = 1_000_000


@pytest.fixture(scope="module")
def save_graph() -> KnowledgeGraph:
    """1M nodes, one property column. Builds in ~0.35 s."""
    graph = KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame({"id": range(SAVE_ROWS), "name": [f"n{i}" for i in range(SAVE_ROWS)]}),
        "One",
        "id",
        "name",
    )
    return graph


@pytest.mark.benchmark
def test_bench_unchanged_save_1m(benchmark, save_graph, tmp_path):
    """Repeated `save()` of a 1M-node graph that never changes between saves.

    Nothing is dirty, so this is the floor cost of taking a checkpoint — and the
    scan found `RandomState` at 14.9% self time inside it (the drift-check
    `next_row` map), i.e. 17-19% of every save on this project's own graphs.
    P4's target: -10% or better.

    `fsync=False` deliberately. A device barrier is the noisiest thing this
    suite can measure and it is not what P4 touches; including it would bury a
    10% CPU win under storage-hardware latency.
    """
    path = str(tmp_path / "unchanged.kgl")
    save_graph.save(path, fsync=False)
    assert os.path.getsize(path) > 0

    def save():
        save_graph.save(path, fsync=False)

    benchmark.pedantic(save, rounds=TENS_MS_ROUNDS, iterations=1, warmup_rounds=TENS_MS_WARMUP)

    assert save_graph.graph_info()["node_count"] == SAVE_ROWS, (
        "the graph must be unchanged across every save; a mutation would make "
        "this a measurement of dirty-page writeback instead"
    )
    benchmark.extra_info["rows"] = SAVE_ROWS


# ===========================================================================
# Scan #8 — `ensure_type_metadata` materializes a full pairs Vec per created node
# ===========================================================================
#
# The cost is O(type columns) per created node — ~12.9 ns per node per column —
# to answer a `contains_key` that `register_property_types` already answered
# upstream. So the shape that shows it is a *narrow* CREATE into a *wide* type:
# each new node carries 2 properties, and pays for 30.

CREATE_SEED_ROWS = 1_000
CREATE_SEED_COLS = 30
CREATE_BATCH = 5_000


def _create_seed_graph() -> KnowledgeGraph:
    """A type of `CREATE_SEED_COLS` columns, seeded with `CREATE_SEED_ROWS` rows.

    Builds in ~4 ms, which is why the cell can afford to rebuild it every round.
    """
    graph = KnowledgeGraph()
    data: dict = {"id": range(CREATE_SEED_ROWS), "name": [f"s{i}" for i in range(CREATE_SEED_ROWS)]}
    for c in range(CREATE_SEED_COLS - 2):
        data[f"p{c}"] = [i + c for i in range(CREATE_SEED_ROWS)]
    graph.add_nodes(pd.DataFrame(data), "Item", "id", "name")
    return graph


@pytest.mark.benchmark
def test_bench_create_wide_seed(benchmark, tmp_path):
    """`UNWIND ... CREATE` of `CREATE_BATCH` 2-property nodes into a 30-column type.

    P5's target: -10% at 30 columns. Divide by `CREATE_BATCH` for the per-node
    figure the scan quotes.

    A **fresh graph per round**, built in an untimed `setup`. Keeping one graph
    would grow the type 50x over the cell's budget, so late rounds would measure
    a different shape from early ones; rebuilding costs ~4 ms of untimed setup
    and makes every round identical. The price is that each round pays one cold
    plan-compile for the statement, which at ~5 ms of timed work is under 1%.
    """
    state: dict = {}

    def setup():
        state["graph"] = _create_seed_graph()
        return (state["graph"],), {}

    def create(graph):
        graph.cypher(
            "UNWIND range($lo, $hi) AS i CREATE (:Item {id: i, name: 'x'})",
            params={"lo": 10_000_000, "hi": 10_000_000 + CREATE_BATCH - 1},
        )

    benchmark.pedantic(create, setup=setup, rounds=CREATE_ROUNDS, iterations=1, warmup_rounds=CREATE_WARMUP)

    final = state["graph"]
    assert final.cypher("MATCH (n:Item) RETURN count(n) AS c").scalar() == CREATE_SEED_ROWS + CREATE_BATCH, (
        "the last round must have created exactly CREATE_BATCH nodes into the seeded type"
    )
    benchmark.extra_info["seed_columns"] = CREATE_SEED_COLS
    benchmark.extra_info["created"] = CREATE_BATCH


# ===========================================================================
# Scan #10 — the resolution hoist stops at one fused shape
# ===========================================================================
#
# Two cells whose only difference is whether the summed expression reads
# properties. Their difference over (rows x 4 accesses) is the per-access slope
# that P7 targets; neither cell alone carries the information, because the row
# loop, the aggregate and the result plumbing are common to both.

FUSED_ROWS = 50_000
FUSED_PROPS = ("a", "b", "c", "d")


@pytest.fixture(scope="module")
def fused_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    data: dict = {"id": range(FUSED_ROWS), "name": [f"n{i}" for i in range(FUSED_ROWS)]}
    for c in FUSED_PROPS:
        data[c] = [ord(c) + i for i in range(FUSED_ROWS)]
    graph.add_nodes(pd.DataFrame(data), "T", "id", "name")
    return graph


@pytest.mark.benchmark
def test_bench_fused_property_access(benchmark, fused_graph):
    """`sum(n.a + n.b + n.c + n.d)` over 50k rows — four property reads per row."""
    expr = " + ".join(f"n.{p}" for p in FUSED_PROPS)

    def read():
        return fused_graph.cypher(f"MATCH (n:T) RETURN sum({expr}) AS s").scalar()

    total = read()
    assert total is not None and total > 0
    benchmark.pedantic(read, rounds=READ_ROUNDS, iterations=1, warmup_rounds=READ_WARMUP)
    benchmark.extra_info["rows"] = FUSED_ROWS
    benchmark.extra_info["accesses_per_row"] = len(FUSED_PROPS)


@pytest.mark.benchmark
def test_bench_fused_constant_control(benchmark, fused_graph):
    """`sum(1 + 2 + 3 + 4)` over the same 50k rows — the control.

    Same scan, same aggregate, same row count, no property resolution. The
    difference between this and the cell above, divided by 200,000 accesses, is
    the number P7 moves.
    """

    def read():
        return fused_graph.cypher("MATCH (n:T) RETURN sum(1 + 2 + 3 + 4) AS s").scalar()

    assert read() == FUSED_ROWS * 10, "the control must still visit every row"
    benchmark.pedantic(read, rounds=READ_ROUNDS, iterations=1, warmup_rounds=READ_WARMUP)
    benchmark.extra_info["rows"] = FUSED_ROWS
    benchmark.extra_info["accesses_per_row"] = 0


# ===========================================================================
# Scan #7b — the `Str` `relocated` overlay map taxes every later scan
# ===========================================================================
#
# A `SET` writing a string of a *different length* cannot be patched in place,
# so the new value goes into a `relocated` overlay keyed by row — a SipHash map
# probed on every subsequent read of that column, and never compacted. One
# single-row write therefore makes every later full scan of a 50k column ~29%
# slower, permanently.
#
# Two graphs, identical but for that one write, so the pair isolates the overlay
# from everything else about the column.

STR_ROWS = 50_000

#: A value present on exactly one row, so the scan is a full column walk with a
#: single survivor — the predicate cost is the measurement, not the result set.
STR_NEEDLE = "val-042000"

#: Longer than the values it replaces, which is what forces relocation rather
#: than an in-place patch.
STR_REPLACEMENT = "a-considerably-longer-replacement-string"

STR_SCAN = "MATCH (n:T) WHERE n.s = $needle RETURN count(n) AS c"


def _string_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.define_schema({"nodes": {"T": {"primary_key": "id"}}})
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(STR_ROWS),
                "name": [f"n{i}" for i in range(STR_ROWS)],
                "s": [f"val-{i:06d}" for i in range(STR_ROWS)],
            }
        ),
        "T",
        "id",
        "name",
    )
    graph.cypher("MATCH (n:T {id: 0}) RETURN n.id")
    return graph


@pytest.fixture(scope="module")
def clean_string_graph() -> KnowledgeGraph:
    return _string_graph()


@pytest.fixture(scope="module")
def relocated_string_graph() -> KnowledgeGraph:
    graph = _string_graph()
    graph.cypher("MATCH (n:T {id: 0}) SET n.s = $v", params={"v": STR_REPLACEMENT})
    return graph


@pytest.mark.benchmark
def test_bench_scan_clean_string_column(benchmark, clean_string_graph):
    """Equality scan of a 50k `Str` column that has never been written — the control."""

    def read():
        return clean_string_graph.cypher(STR_SCAN, params={"needle": STR_NEEDLE}).scalar()

    assert read() == 1
    benchmark(read)
    benchmark.extra_info["rows"] = STR_ROWS
    benchmark.extra_info["relocated_rows"] = 0


@pytest.mark.benchmark
def test_bench_scan_relocated_string_column(benchmark, relocated_string_graph):
    """The same scan after **one** differing-length `SET`. P4's target: +29% -> <=5%.

    The two assertions are the whole guard. The first proves the write landed
    (so the overlay exists); the second proves the scan still finds its single
    survivor (so the two cells are answering the same question and the ratio
    between them is a ratio of like work).
    """
    written = relocated_string_graph.cypher("MATCH (n:T {id: 0}) RETURN n.s AS s").scalar()
    assert written == STR_REPLACEMENT, (
        f"the relocating SET did not land (n.s = {written!r}); without it this cell is a duplicate of its own control"
    )

    def read():
        return relocated_string_graph.cypher(STR_SCAN, params={"needle": STR_NEEDLE}).scalar()

    assert read() == 1
    benchmark(read)
    benchmark.extra_info["rows"] = STR_ROWS
    benchmark.extra_info["relocated_rows"] = 1
