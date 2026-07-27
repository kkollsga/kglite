"""Why is deleting so much more expensive than inserting?

The 2026-07-27 competitive benchmark left exactly one finding **deliberately
unattributed**, and this file is its instrument. At 100k nodes, with no index
and at matched durability, in the very same configuration where an insert hit
**exact parity with SQLite (3.668 ms vs 3.666 ms)**:

| operation               | kglite     | SQLite    | ratio |
|-------------------------|-----------:|----------:|------:|
| `delete_comment`        |  10.09 ms  | 5.462 ms  | 1.8x  |
| `cascade_delete_issue`  |  24.71 ms  | 5.476 ms  | 4.5x  |

Against SQLite's `wal_normal` rung the gap is 2.6-6.1x. Inserts were at parity;
deletes were not. Nothing in the suite measured a delete shape across sizes, so
the *scaling* of that gap is simply unknown — which is why it stayed a mystery
rather than becoming a bug report.

**This file deliberately does not encode a cause.** The standing hypothesis at
the time of writing was `check_auto_vacuum` being reachable on the delete path
and not the insert path, and the probe was never run. A benchmark built around
that hypothesis would measure the hypothesis; these cells measure the *shapes*
instead — leaf delete, `DETACH DELETE` with edges, cascade — across three sizes,
so the scaling is visible whatever the cause turns out to be, including a cause
nobody has thought of yet.

Two candidate costs *are* separated, because both are known to exist and would
otherwise be summed into one uninterpretable number:

* **the id-index rebuild.** A delete invalidates the type's id index, so the
  next lookup by id pays an O(V) rebuild. `test_bench_write_scaling.py`
  measured that term alone at 34,486 us vs 4.1 us at 1M. Its cost lands on
  whoever queries next, not on the deleting statement, so it is split into its
  own cell rather than left to contaminate the others.
* **auto-vacuum.** `check_auto_vacuum` (`dir_graph/mod.rs:1901-1922`) is
  called only from `after_mutation` (`kglite-py/src/graph/mod.rs:361-376`) and
  only when `nodes_deleted > 0 || relationships_deleted > 0` — so it is, in
  fact, structurally unreachable from an insert. It fires when tombstones
  exceed a hard floor of 100 **and** the fragmentation ratio exceeds a
  threshold, and `vacuum()` then rebuilds column stores and reindexes. See
  `test_bench_bulk_delete_by_auto_vacuum` for what that cell can and cannot
  tell you.

**Every other cell here pins `set_auto_vacuum(None)`.** Not to hide the cost,
but because an auto-vacuum firing on an arbitrary round turns a per-delete
measurement into a bimodal one, and pytest-benchmark reports the blend. Leaving
it enabled would produce the sort of plausible, mildly-elevated,
completely-uninterpretable number this project's own methodology note warns
about.

Nothing here is in the `make bench-check` tracked set — that gate runs
`tests/benchmarks/test_bench_core.py` only (`Makefile:85`).

Run with::

    uv run --no-sync maturin develop --release
    .venv/bin/python -m pytest tests/benchmarks/test_bench_delete_scaling.py \\
        -m benchmark -v
"""

from __future__ import annotations

import pandas as pd
import pytest

from kglite import KnowledgeGraph

# Three decades, because this defect is unexplained and the *slope* is the
# entire finding. Two points give a ratio; three distinguish "linear in graph
# size" from "linear in edges touched" from "flat with a large constant" — and
# the answer decides where anyone looks next. 1k is also the scale at which the
# competitive run found deletes to be *fine*, so it doubles as the control that
# localises the problem to scale rather than to the delete path as such.
SIZES = [1_000, 10_000, 100_000]

#: Comments per issue in the fixture, and therefore edges detached by a cascade.
COMMENTS_PER_ISSUE = 10

#: Victims pre-created per shape, one consumed per round. Must exceed
#: ROUNDS + WARMUP_ROUNDS or a cell runs out mid-measurement.
VICTIM_POOL = 80

# Explicit rounds rather than plain `benchmark(fn)`. Auto-calibration times the
# first call to size the round count, and a delete's first call is not
# representative: the fixture leaves the id index warm and the first delete
# invalidates it. `test_bench_write_scaling.py:46-63` records that exact
# pattern scheduling 12,686 rounds of a 24 ms call — five minutes from one
# test, uninterruptible because `-m benchmark` is exempt from the 120 s hang
# ceiling (`tests/conftest.py:34`).
#
# The pool size caps rounds independently: each round destroys a victim, so
# rounds must stay under VICTIM_POOL.
ROUNDS = 50
WARMUP_ROUNDS = 5

#: Id-space partitions, kept far apart so a victim id can never collide with a
#: base-graph id and turn a delete into a silent no-op — which would read as a
#: very fast delete rather than as an error.
LEAF_BASE = 10_000_000
DETACH_BASE = 20_000_000
CASCADE_ISSUE_BASE = 30_000_000
CASCADE_COMMENT_BASE = 40_000_000
PROBE_BASE = 50_000_000
BULK_BASE = 60_000_000


def _issue_comment_graph(size: int) -> KnowledgeGraph:
    """`size` Comments hanging off `size // COMMENTS_PER_ISSUE` Issues.

    Mirrors the application the competitive benchmark implemented three times
    (kglite / SQLite / ArcadeDB), so the numbers here are comparable to the
    table in this file's docstring rather than merely internally consistent.

    Primary keys are declared on both types. Without one, `CREATE` invalidates
    the whole type's id index and the next id lookup rebuilds it — an O(V) cost
    per statement that has nothing to do with deletion but is large enough to
    swamp it (34,486 us vs 4.1 us at 1M).
    """
    graph = KnowledgeGraph()
    graph.define_schema(
        {
            "nodes": {
                "Issue": {"primary_key": "id"},
                "Comment": {"primary_key": "id"},
            }
        }
    )
    issue_count = max(1, size // COMMENTS_PER_ISSUE)
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(issue_count),
                "title": [f"issue-{i}" for i in range(issue_count)],
                "state": ["open" if i % 3 else "closed" for i in range(issue_count)],
            }
        ),
        "Issue",
        "id",
        "title",
        columns=["state"],
    )
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(size),
                "body": [f"comment body {i}" for i in range(size)],
            }
        ),
        "Comment",
        "id",
        "body",
    )
    graph.add_connections(
        pd.DataFrame({"src": range(size), "dst": [i % issue_count for i in range(size)]}),
        "ON",
        "Comment",
        "src",
        "Issue",
        "dst",
    )
    return graph


def _add_victims(graph: KnowledgeGraph, base: int, *, attached: bool) -> None:
    """A pool of throwaway Comments, optionally wired to an Issue.

    Pre-created in the fixture rather than in a `pedantic` setup so the timed
    region is a delete and nothing else. A setup that created the victim would
    also warm the id index the delete then invalidates, which quietly changes
    what the following round measures.
    """
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(base, base + VICTIM_POOL),
                "body": [f"victim body {i}" for i in range(VICTIM_POOL)],
            }
        ),
        "Comment",
        "id",
        "body",
    )
    if attached:
        graph.add_connections(
            pd.DataFrame({"src": range(base, base + VICTIM_POOL), "dst": [0] * VICTIM_POOL}),
            "ON",
            "Comment",
            "src",
            "Issue",
            "dst",
        )


def _add_cascade_victims(graph: KnowledgeGraph) -> None:
    """`VICTIM_POOL` Issues, each carrying `COMMENTS_PER_ISSUE` Comments."""
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(CASCADE_ISSUE_BASE, CASCADE_ISSUE_BASE + VICTIM_POOL),
                "title": [f"victim issue {i}" for i in range(VICTIM_POOL)],
                "state": ["open"] * VICTIM_POOL,
            }
        ),
        "Issue",
        "id",
        "title",
        columns=["state"],
    )
    total = VICTIM_POOL * COMMENTS_PER_ISSUE
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(CASCADE_COMMENT_BASE, CASCADE_COMMENT_BASE + total),
                "body": [f"cascade body {i}" for i in range(total)],
            }
        ),
        "Comment",
        "id",
        "body",
    )
    graph.add_connections(
        pd.DataFrame(
            {
                "src": range(CASCADE_COMMENT_BASE, CASCADE_COMMENT_BASE + total),
                "dst": [CASCADE_ISSUE_BASE + (i // COMMENTS_PER_ISSUE) for i in range(total)],
            }
        ),
        "ON",
        "Comment",
        "src",
        "Issue",
        "dst",
    )


@pytest.fixture(scope="module")
def delete_graphs() -> dict[int, KnowledgeGraph]:
    """One graph per size, pre-loaded with every shape's victim pool.

    Module-scoped: three graphs, one of them 100k nodes plus 100k edges, is
    seconds of build time that must not repeat per cell.

    `set_auto_vacuum(None)` is the load-bearing line — see the module
    docstring. Without it a vacuum fires on an arbitrary round once tombstones
    pass 100, rebuilding column stores and reindexing, and every cell in this
    file reports a blend of two populations.
    """
    graphs = {}
    for size in SIZES:
        graph = _issue_comment_graph(size)
        _add_victims(graph, LEAF_BASE, attached=False)
        _add_victims(graph, DETACH_BASE, attached=True)
        _add_cascade_victims(graph)
        graph.set_auto_vacuum(None)
        graphs[size] = graph
    return graphs


@pytest.mark.benchmark
@pytest.mark.parametrize("size", SIZES)
def test_bench_delete_leaf(benchmark, delete_graphs, size):
    """Delete one Comment that has no edges at all.

    The floor of the delete path: node removal, tombstone, index maintenance,
    and nothing else. If this scales with graph size then the cost is in the
    delete machinery itself and has nothing to do with edges — which would rule
    out the whole "detaching relationships is expensive" family of explanations
    in a single reading.

    Defends the unattributed 100x-of-insert gap: `delete_comment` measured
    **10.09 ms at 100k against a 3.668 ms insert in the same cell**
    (2026-07-27).
    """
    graph = delete_graphs[size]
    ids = iter(range(LEAF_BASE, LEAF_BASE + VICTIM_POOL))

    def delete():
        graph.cypher("MATCH (c:Comment {id: $i}) DELETE c", params={"i": next(ids)})

    benchmark.pedantic(delete, rounds=ROUNDS, iterations=1, warmup_rounds=WARMUP_ROUNDS)


@pytest.mark.benchmark
@pytest.mark.parametrize("size", SIZES)
def test_bench_detach_delete_one_edge(benchmark, delete_graphs, size):
    """`DETACH DELETE` one Comment that owns exactly one edge.

    Read against `test_bench_delete_leaf` at the same size: the difference is
    the per-edge detach cost, isolated at one edge so it cannot be confused
    with the cascade cell's fan-out. If leaf and detach diverge *with size*,
    the edge-detach path is doing something proportional to the graph rather
    than to the node's degree — the single most useful thing this file can
    establish.
    """
    graph = delete_graphs[size]
    ids = iter(range(DETACH_BASE, DETACH_BASE + VICTIM_POOL))

    def delete():
        graph.cypher("MATCH (c:Comment {id: $i}) DETACH DELETE c", params={"i": next(ids)})

    benchmark.pedantic(delete, rounds=ROUNDS, iterations=1, warmup_rounds=WARMUP_ROUNDS)


@pytest.mark.benchmark
@pytest.mark.parametrize("size", SIZES)
def test_bench_cascade_delete_issue(benchmark, delete_graphs, size):
    """Delete an Issue together with its `COMMENTS_PER_ISSUE` Comments.

    The worst cell in the competitive table: **24.71 ms at 100k against
    SQLite's 5.476 ms, a 4.5x gap**, in a configuration where inserts were at
    parity (2026-07-27).

    Eleven nodes and ten edges per round. Against `test_bench_detach_delete_
    one_edge` this says whether cascade cost is simply 11x a single delete
    (a constant-factor story, uninteresting) or superlinear in the number of
    entities removed per statement (a real defect).
    """
    graph = delete_graphs[size]
    ids = iter(range(CASCADE_ISSUE_BASE, CASCADE_ISSUE_BASE + VICTIM_POOL))

    def delete():
        graph.cypher(
            "MATCH (i:Issue {id: $i})<-[:ON]-(c:Comment) DETACH DELETE i, c",
            params={"i": next(ids)},
        )

    benchmark.pedantic(delete, rounds=ROUNDS, iterations=1, warmup_rounds=WARMUP_ROUNDS)


@pytest.mark.benchmark
@pytest.mark.parametrize("size", SIZES)
def test_bench_delete_then_id_lookup(benchmark, delete_graphs, size):
    """A delete plus the id lookup that pays for it.

    The deleting statement invalidates the type's id index; the *rebuild* is
    paid by whoever looks up by id next. Splitting it out matters because the
    two costs land on different statements and scale differently, and an
    application's real per-delete cost is this cell, not `test_bench_delete_
    leaf`.

    Read the difference between this and `test_bench_delete_leaf` as the
    rebuild term. It is expected to scale with size — that is a known open
    issue (`test_bench_write_scaling.py::test_bench_id_index_invalidation_on_
    create`), not a regression — so the question this cell answers is how much
    of the unexplained delete gap it accounts for. If it accounts for all of
    it, the mystery is solved and it was never about deletion.
    """
    graph = delete_graphs[size]
    ids = iter(range(PROBE_BASE, PROBE_BASE + VICTIM_POOL))
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(PROBE_BASE, PROBE_BASE + VICTIM_POOL),
                "body": [f"probe body {i}" for i in range(VICTIM_POOL)],
            }
        ),
        "Comment",
        "id",
        "body",
    )

    def delete_then_lookup():
        graph.cypher("MATCH (c:Comment {id: $i}) DELETE c", params={"i": next(ids)})
        graph.cypher("MATCH (c:Comment {id: 0}) RETURN c.id")

    benchmark.pedantic(delete_then_lookup, rounds=ROUNDS, iterations=1, warmup_rounds=WARMUP_ROUNDS)


# ── the auto-vacuum discriminator ────────────────────────────────────
#
# Read the caveat before quoting this cell. `check_auto_vacuum` fires only when
# BOTH conditions hold (`dir_graph/mod.rs:1901-1922`):
#
#     tombstones = node_bound() - node_count() > 100
#     tombstones / node_bound() > threshold          (default 0.3)
#
# A per-round single delete produces one tombstone per round, so at the default
# threshold a bounded benchmark never crosses either condition and the on/off
# pair reads *identical*. That null result would look like evidence that
# auto-vacuum is not the cost — while actually being evidence that the
# benchmark never reached it. Under-powered nulls are how this defect stayed
# unexplained in the first place.
#
# So this cell does two things differently: it deletes in a batch large enough
# to clear the 100-tombstone floor within a few rounds, and it uses an
# artificially low threshold so the ratio condition is reachable at all.
#
# WHAT IT MEASURES: whether the vacuum path, once entered, is expensive enough
# to explain a delete gap. That is the open question.
# WHAT IT DOES NOT MEASURE: the cost of the DEFAULT configuration. At
# threshold=0.3 a real application vacuums rarely, and this cell says nothing
# about how rarely. Do not quote it as "auto-vacuum costs X per delete".

#: Comments removed per round — comfortably over the 100-tombstone floor, so
#: the ratio condition is the only thing left gating the vacuum.
BULK_DELETE_BATCH = 150

#: Low enough that the ratio condition is satisfiable within a bounded run.
#: The default is 0.3; see the caveat above before comparing the two.
PROBE_VACUUM_THRESHOLD = 0.001

#: Few rounds: each round deletes BULK_DELETE_BATCH nodes and a fired vacuum
#: rebuilds column stores and reindexes the whole graph.
BULK_ROUNDS = 6


@pytest.mark.benchmark
@pytest.mark.parametrize("auto_vacuum", [None, PROBE_VACUUM_THRESHOLD], ids=["vacuum_off", "vacuum_on"])
@pytest.mark.parametrize("size", SIZES)
def test_bench_bulk_delete_by_auto_vacuum(benchmark, size, auto_vacuum):
    """A batched delete with auto-vacuum reachable, versus disabled.

    The discriminating pair for the standing hypothesis, run without asserting
    it. Builds its own graph rather than sharing `delete_graphs`, because a
    fired vacuum remaps every `NodeIndex` and rebuilds the column stores — it
    would leave a shared fixture in a different state for whichever cell ran
    next, making results depend on collection order.

    Read `vacuum_on` against `vacuum_off` at the same size — **only against
    each other, never against the single-delete cells above.** The `WHERE c.id
    IN $ids` predicate is a label scan, so each round carries an O(V) term that
    the point-lookup cells do not. Both arms pay it identically, so it cancels
    in the comparison this cell exists to make, but it makes the absolute
    number meaningless on its own. (The scan is deliberate: an `UNWIND` +
    id-match spelling would be cheaper, and is also the exact shape of a fixed
    0.11.2 bug where a bare point lookup silently returned nothing once the
    list passed ~64 entries. A cell that quietly deleted zero nodes would be a
    very fast delete benchmark and a completely false one.)

    A large gap means the vacuum path is worth investigating as the delete
    cost; a small one retires the hypothesis and points elsewhere. Either way
    the answer is recorded rather than assumed.
    """
    graph = _issue_comment_graph(size)
    graph.set_auto_vacuum(auto_vacuum)
    total = BULK_DELETE_BATCH * (BULK_ROUNDS + 2)
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(BULK_BASE, BULK_BASE + total),
                "body": [f"bulk body {i}" for i in range(total)],
            }
        ),
        "Comment",
        "id",
        "body",
    )
    batches = iter(
        [
            list(range(BULK_BASE + n * BULK_DELETE_BATCH, BULK_BASE + (n + 1) * BULK_DELETE_BATCH))
            for n in range(BULK_ROUNDS + 2)
        ]
    )

    def delete_batch():
        graph.cypher(
            "MATCH (c:Comment) WHERE c.id IN $ids DETACH DELETE c",
            params={"ids": next(batches)},
        )

    benchmark.pedantic(delete_batch, rounds=BULK_ROUNDS, iterations=1, warmup_rounds=1)
