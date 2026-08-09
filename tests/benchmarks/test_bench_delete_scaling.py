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
* **auto-vacuum.** `check_auto_vacuum` (`dir_graph/mod.rs:1914-1943`) is
  called only from `after_mutation` (`kglite-py/src/graph/mod.rs:361-376`) and
  only when `nodes_deleted > 0 || relationships_deleted > 0` — so it is, in
  fact, structurally unreachable from an insert. It fires when tombstones
  exceed a hard floor of 100 **and** the fragmentation ratio exceeds a
  threshold, and `vacuum()` then rebuilds column stores and reindexes.

**Every per-delete cell here pins `set_auto_vacuum(None)`.** Not to hide the
cost, but because an auto-vacuum firing on an arbitrary round turns a
per-delete measurement into a bimodal one, and pytest-benchmark reports the
blend. Leaving it enabled would produce the sort of plausible,
mildly-elevated, completely-uninterpretable number this project's own
methodology note warns about.

Two auto-vacuum cells exist, and they answer different questions
-----------------------------------------------------------------

1. ``test_bench_forced_threshold_vacuum_cost`` — the original probe. It pins
   ``set_auto_vacuum(0.001)``, i.e. it *forces* the vacuum to be reachable.
   **The ~18x figure people quote from this file came from here, and it is a
   forced-trigger number, not a default-configuration number.** It says how
   expensive the vacuum path is once entered; it says nothing at all about how
   often a default-configured graph enters it.

2. ``test_bench_default_auto_vacuum_lifecycle`` — added for backlog item C3.
   The **default** ``0.3`` threshold, an off control, and a delete sequence
   long enough to genuinely cross the threshold several times. This is the cell
   that can answer "what does auto-vacuum cost a real application".

Nothing here is in the `make bench-check` tracked set — that gate runs
`tests/benchmarks/test_bench_core.py` only (`Makefile:85`).

Runbook — the two-run release protocol (backlog C3)
---------------------------------------------------

Correctness (valid in any profile, runs in the default suite; records no
timing). 1k and 100k run by default; the 1M arm holds ~1 GB and is opt-in::

    .venv/bin/python -m pytest tests/benchmarks/test_bench_delete_scaling.py \\
        -k oracle -v
    KGLITE_VACUUM_LIFECYCLE_1M=1 .venv/bin/python -m pytest \\
        tests/benchmarks/test_bench_delete_scaling.py -k oracle -v

Timing. **Release profile only** (CLAUDE.md Performance protocol), on an
otherwise-idle machine, with a thermal settle between the two runs::

    uv run --no-sync maturin develop --release
    .venv/bin/python -m pytest tests/benchmarks/test_bench_delete_scaling.py \\
        -m benchmark -k lifecycle \\
        --benchmark-json=/tmp/vacuum-lifecycle-run-a.json
    sleep 30
    .venv/bin/python -m pytest tests/benchmarks/test_bench_delete_scaling.py \\
        -m benchmark -k lifecycle \\
        --benchmark-json=/tmp/vacuum-lifecycle-run-b.json
    .venv/bin/python tests/benchmarks/test_bench_delete_scaling.py \\
        /tmp/vacuum-lifecycle-run-a.json /tmp/vacuum-lifecycle-run-b.json

The report prints, per size, the ``default_on/off`` total-delete ratio, the
worst threshold-crossing batch, and the off-control p99 — then the verdict
(exit 0 = PROCEED, exit 1 = RETIRE/ABORT):

* **PROCEED to a remedy phase** — **both** runs show a total ratio >= 1.10 at
  some size, **or** both runs show the 1M threshold-crossing batch at >= 100 ms
  *and* >= 10x the 1M off-control p99. Only then may `crates/` be touched, and
  only the profiled root.
* **RETIRE with no source change** — neither criterion reproduces in both runs.
  Record "not reproduced" and close C3; the stale ~18x concern was the forced
  ``0.001`` artifact described above.
* **ABORT (invalid, not green)** — the default arm fired zero vacuums, the two
  arms disagree on final counts or query results, or a size is missing. A
  sequence that never crossed the threshold measured nothing; re-run, do not
  interpret.

Timing runs are release-only by construction: a debug-profile number is invalid
evidence and must be discarded, not reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

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


# ── FORCED-TRIGGER vacuum cost (threshold 0.001, NOT the default) ────
#
# Read the caveat before quoting this cell. `check_auto_vacuum` fires only when
# BOTH conditions hold (`dir_graph/mod.rs:1914-1943`):
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
# WHAT IT MEASURES: **forced-trigger cost** — how expensive the vacuum path is
# once entered, with the entry condition artificially forced open.
# WHAT IT DOES NOT MEASURE: the cost of the DEFAULT configuration. At
# threshold=0.3 a real application vacuums rarely, and this cell says nothing
# about how rarely. Do not quote it as "auto-vacuum costs X per delete", and do
# not quote its ratio as a default-configuration overhead — that mislabelling
# is exactly what backlog item C3 was opened to correct. The default-threshold
# question is answered by `test_bench_default_auto_vacuum_lifecycle` below.

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
@pytest.mark.parametrize("auto_vacuum", [None, PROBE_VACUUM_THRESHOLD], ids=["vacuum_off", "forced_trigger"])
@pytest.mark.parametrize("size", SIZES)
def test_bench_forced_threshold_vacuum_cost(benchmark, size, auto_vacuum):
    """FORCED-TRIGGER vacuum cost at threshold 0.001 — **not** the default.

    Renamed from `test_bench_bulk_delete_by_auto_vacuum` so the label carries
    the caveat: every number this cell produces is the cost of a vacuum that
    was *forced* to be reachable. The ~18x figure quoted from this file comes
    from here. It is a property of the vacuum path, not of a default-configured
    graph, and it is not evidence about steady-state delete cost.

    The discriminating pair for the standing hypothesis, run without asserting
    it. Builds its own graph rather than sharing `delete_graphs`, because a
    fired vacuum remaps every `NodeIndex` and rebuilds the column stores — it
    would leave a shared fixture in a different state for whichever cell ran
    next, making results depend on collection order.

    Read `forced_trigger` against `vacuum_off` at the same size — **only against
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


# ── DEFAULT-THRESHOLD lifecycle (backlog C3) ─────────────────────────
#
# The cell above forces the trigger open. This one leaves the engine's own
# default in place and asks the question a user actually has: over a long-lived
# delete workload, what does auto-vacuum cost?
#
# The design problem is that "leave the default alone and delete some rows" is
# an under-powered null — a bounded benchmark at threshold 0.3 never crosses,
# both arms read identical, and the identical reading looks like evidence of no
# cost while actually being evidence of no measurement. So the sequence is
# constructed to cross the threshold *repeatedly*, and the crossings are
# asserted rather than assumed. A run whose default arm fired zero vacuums is
# INVALID, not green, and the report aborts on it.
#
# How the crossing schedule is derived (`check_auto_vacuum`, tombstones must
# exceed 100 AND `tombstones / node_bound > threshold`):
#
#   fixture      = S Comments + S/10 Issues        -> node_bound = 1.1 * S
#   batch        = S/200 Comments, DETACH DELETEd  -> +S/200 tombstones each
#   ratio at b   = (S/200 * b) / (1.1 * S)         =  b / 220     <- S cancels
#
# `b/220 > 0.3` first holds at b = 67 (0.30454...), and it does not hold at
# b = 66 (exactly 0.3, and the comparison is strict `>`). A vacuum resets
# node_bound to node_count, so the next crossing is recomputed against the
# smaller graph; the same arithmetic gives b = 113 and b = 146 within 160
# batches. **The S term cancels, so the schedule is identical at 1k, 100k and
# 1M** — which is what makes a single literal `EXPECTED_FIRE_BATCHES` a real
# assertion across three decades rather than three separate recordings.
#
# That derivation also pins the default threshold behaviourally. No vacuum at
# b=66 means the effective threshold is >= 0.3; a vacuum at b=67 means it is
# < 0.30455. There is no Python getter for `auto_vacuum_threshold`
# (`graph_info()` does not report it), so this band *is* the assertion that the
# default is 0.3 — see `test_default_auto_vacuum_threshold_band_oracle`.
#
# The delete spelling is `UNWIND $ids AS i MATCH (c:Comment {id: i})`, i.e. a
# primary-key point lookup per victim, not the label scan the forced cell uses.
# Both arms pay whatever the spelling costs, so it cancels in the ratio — but it
# does not cancel in *sensitivity*: a shared O(V) scan per batch would sit in
# the denominator of the on/off ratio and dilute the vacuum signal toward 1.0,
# biasing the stop rule toward RETIRE. A point-lookup batch is O(batch), so the
# ratio is as sharp as this workload allows.

#: Three decades. 1M is the size the stop rule's second criterion is written
#: against; it peaks at roughly 1 GB RSS, which is why the oracle needs an
#: opt-in for it (see `LIFECYCLE_1M_ENV`).
LIFECYCLE_SIZES = [1_000, 100_000, 1_000_000]

#: Comments removed per batch, as a divisor of the graph size. Fixing the
#: *fraction* rather than the count is what makes the crossing schedule
#: scale-invariant.
LIFECYCLE_STEP_DIVISOR = 200

#: 160 batches removes 80% of the Comments. Chosen for two reasons: it spans
#: three threshold crossings (so the cost of a crossing is sampled more than
#: once), and it yields 160 per-batch latency samples per arm, enough for a p99
#: that is not simply the maximum.
LIFECYCLE_BATCHES = 160

#: Literal, not recomputed from the model above — mutate it and the oracle must
#: go red. 1-indexed batch numbers at which the default arm vacuums.
EXPECTED_FIRE_BATCHES = (67, 113, 146)

#: The band the observed schedule pins the engine default into. Both bounds are
#: load-bearing: the lower comes from *no* vacuum at batch 66, the upper from a
#: vacuum at batch 67.
DEFAULT_THRESHOLD_BAND = (0.3, 67 / 220)

#: Literal expected end state per size, for both arms. `node_capacity` and
#: `node_tombstones` are the two figures that legitimately differ between the
#: arms — everything else must match exactly, because a vacuum is a storage
#: compaction and must not be observable as a semantic change.
LIFECYCLE_EXPECTED = {
    1_000: {
        "comments": 200,
        "issues": 100,
        "node_count": 300,
        "edge_count": 200,
        "on_capacity": 370,
        "on_tombstones": 70,
        "off_capacity": 1_100,
        "off_tombstones": 800,
    },
    100_000: {
        "comments": 20_000,
        "issues": 10_000,
        "node_count": 30_000,
        "edge_count": 20_000,
        "on_capacity": 37_000,
        "on_tombstones": 7_000,
        "off_capacity": 110_000,
        "off_tombstones": 80_000,
    },
    1_000_000: {
        "comments": 200_000,
        "issues": 100_000,
        "node_count": 300_000,
        "edge_count": 200_000,
        "on_capacity": 370_000,
        "on_tombstones": 70_000,
        "off_capacity": 1_100_000,
        "off_tombstones": 800_000,
    },
}

#: Set to 1 to include the 1M arm in the correctness (oracle) run. Off by
#: default because the oracle runs in the *default* pytest suite, where a ~1 GB
#: fixture is not a reasonable standing cost.
LIFECYCLE_1M_ENV = "KGLITE_VACUUM_LIFECYCLE_1M"

LIFECYCLE_DELETE = "UNWIND $ids AS i MATCH (c:Comment {id: i}) DETACH DELETE c"

#: Parity probe run against both arms at the end of the sequence. Ordered, so
#: the digest is a statement about content rather than about iteration order —
#: which a vacuum *does* change, and is allowed to.
LIFECYCLE_PARITY_QUERY = (
    "MATCH (i:Issue)<-[:ON]-(c:Comment) RETURN i.id AS issue, count(c) AS comments ORDER BY issue LIMIT 25"
)

#: Stop-rule constants (backlog C3). Literals, so the report cannot drift from
#: the rule it claims to implement.
STOP_RATIO = 1.10
STOP_SPIKE_MS = 100.0
STOP_SPIKE_FACTOR = 10.0
STOP_SPIKE_SIZE = 1_000_000


def _lifecycle_batches(size: int) -> list[list[int]]:
    """Every batch's victim ids, materialised up front.

    Built before the sequence starts so the timed region contains engine work
    only. A `list(range(...))` per batch inside the loop would put Python list
    construction — 5,000 ints per batch at 1M — inside the measurement.
    """
    step = size // LIFECYCLE_STEP_DIVISOR
    return [list(range(n * step, (n + 1) * step)) for n in range(LIFECYCLE_BATCHES)]


def _result_digest(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _run_lifecycle(size: int, *, auto_vacuum_off: bool) -> dict:
    """One arm of the lifecycle: build, delete `LIFECYCLE_BATCHES` batches, observe.

    Returns the record both the oracle and the timing cell consume. The timing
    is collected here rather than by `benchmark.pedantic` around the whole
    sequence because the sequence is destructive (it cannot be repeated without
    a rebuild) and because the per-batch `graph_info()` probes — which are how
    the crossings are *proven* — must stay outside the measured region.

    `auto_vacuum_off=False` deliberately calls nothing: leaving `set_auto_vacuum`
    untouched is what makes this the DEFAULT arm rather than an arm configured
    to the value we believe the default to be.
    """
    graph = _issue_comment_graph(size)
    if auto_vacuum_off:
        graph.set_auto_vacuum(None)

    batches = _lifecycle_batches(size)
    step = size // LIFECYCLE_STEP_DIVISOR
    latencies_ms: list[float] = []
    fires: list[int] = []
    tombstones_after: list[int] = []
    capacity_after: list[int] = []
    previous_tombstones = graph.graph_info()["node_tombstones"]
    total_s = 0.0

    for number, ids in enumerate(batches, start=1):
        started = time.perf_counter()
        graph.cypher(LIFECYCLE_DELETE, params={"ids": ids})
        elapsed = time.perf_counter() - started
        total_s += elapsed
        latencies_ms.append(elapsed * 1_000.0)

        info = graph.graph_info()
        # A vacuum is the only thing that can *reduce* the tombstone count: a
        # delete-only sequence adds one tombstone per removed node and never
        # reclaims a slot otherwise. So a drop is the trigger event, observed
        # rather than inferred.
        if info["node_tombstones"] < previous_tombstones:
            fires.append(number)
        previous_tombstones = info["node_tombstones"]
        tombstones_after.append(info["node_tombstones"])
        capacity_after.append(info["node_capacity"])

    final = graph.graph_info()
    return {
        "size": size,
        "arm": "off" if auto_vacuum_off else "default_on",
        "step": step,
        "batches": LIFECYCLE_BATCHES,
        "fires": fires,
        "latencies_ms": latencies_ms,
        "total_delete_s": total_s,
        "tombstones_after": tombstones_after,
        "capacity_after": capacity_after,
        "node_count": final["node_count"],
        "node_capacity": final["node_capacity"],
        "node_tombstones": final["node_tombstones"],
        "edge_count": final["edge_count"],
        "comments": graph.cypher("MATCH (c:Comment) RETURN count(c) AS n").to_list()[0]["n"],
        "issues": graph.cypher("MATCH (i:Issue) RETURN count(i) AS n").to_list()[0]["n"],
        "parity_rows": graph.cypher(LIFECYCLE_PARITY_QUERY).to_list(),
    }


def _lifecycle_sizes_for_oracle() -> list[int]:
    if os.environ.get(LIFECYCLE_1M_ENV) == "1":
        return LIFECYCLE_SIZES
    return [size for size in LIFECYCLE_SIZES if size < 1_000_000]


# --------------------------------------------------------------------------
# Correctness mode — no timing recorded, valid in any build profile.
# --------------------------------------------------------------------------


#: Both arms per size, built on first use and shared by the four oracles.
#:
#: A module-scoped *fixture* would build every size the moment the first oracle
#: ran, so with the 1M opt-in enabled the 1k test would carry ~50 s of unrelated
#: fixture work — and a single test is what the 120 s hang ceiling applies to.
#: Caching per size keeps each test's cost proportional to its own parameter.
_LIFECYCLE_CACHE: dict[tuple[int, str], dict] = {}


def lifecycle_runs(size: int) -> dict[str, dict]:
    if (size, "off") not in _LIFECYCLE_CACHE:
        for off in (False, True):
            record = _run_lifecycle(size, auto_vacuum_off=off)
            _LIFECYCLE_CACHE[(size, record["arm"])] = record
    return {arm: _LIFECYCLE_CACHE[(size, arm)] for arm in ("default_on", "off")}


@pytest.mark.parametrize("size", _lifecycle_sizes_for_oracle())
def test_default_auto_vacuum_actually_fires_oracle(size):
    """The default arm must cross the threshold — three times, on schedule.

    This is the assertion that makes every other lifecycle number meaningful.
    A sequence that never triggered a vacuum has measured the absence of a
    measurement, and the C3 stop rule explicitly calls such a run invalid
    rather than a pass.
    """
    on = lifecycle_runs(size)["default_on"]
    off = lifecycle_runs(size)["off"]

    assert tuple(on["fires"]) == EXPECTED_FIRE_BATCHES
    assert off["fires"] == [], "auto-vacuum fired with set_auto_vacuum(None)"

    # Immediately after a fire the graph is fully compacted; the off arm never
    # reclaims anything, so its tombstone count is exactly the running total.
    for number in EXPECTED_FIRE_BATCHES:
        index = number - 1
        assert on["tombstones_after"][index] == 0
        assert on["capacity_after"][index] == off["capacity_after"][index] - off["tombstones_after"][index]
    for index in range(LIFECYCLE_BATCHES):
        assert off["tombstones_after"][index] == off["step"] * (index + 1)
        assert off["capacity_after"][index] == LIFECYCLE_EXPECTED[size]["off_capacity"]


@pytest.mark.parametrize("size", _lifecycle_sizes_for_oracle())
def test_default_auto_vacuum_threshold_band_oracle(size):
    """The observed schedule pins the engine default into [0.3, 0.30455).

    There is no way to read `auto_vacuum_threshold` back from Python, so the
    default is asserted through the behaviour it produces. The band is narrow
    enough that only 0.3 sits in it at any sane precision — change the engine
    default and this goes red, which is the point.
    """
    on = lifecycle_runs(size)["default_on"]
    first_fire = on["fires"][0]
    off = lifecycle_runs(size)["off"]

    # Ratio the engine saw on the batch *before* the first fire (no vacuum) and
    # on the firing batch itself, read off the off-control's untouched counters.
    ratio_at_no_fire = off["tombstones_after"][first_fire - 2] / off["capacity_after"][first_fire - 2]
    ratio_at_fire = off["tombstones_after"][first_fire - 1] / off["capacity_after"][first_fire - 1]

    assert ratio_at_no_fire <= DEFAULT_THRESHOLD_BAND[0]
    assert ratio_at_fire > DEFAULT_THRESHOLD_BAND[0]
    assert math.isclose(ratio_at_fire, DEFAULT_THRESHOLD_BAND[1], rel_tol=1e-9)


@pytest.mark.parametrize("size", _lifecycle_sizes_for_oracle())
def test_auto_vacuum_is_semantically_invisible_oracle(size):
    """A vacuum compacts storage; it must not change a single visible answer.

    Counts and query results are compared against the off control *and* against
    literal expectations, so a bug that broke both arms identically still goes
    red.
    """
    on = lifecycle_runs(size)["default_on"]
    off = lifecycle_runs(size)["off"]
    expected = LIFECYCLE_EXPECTED[size]

    for key in ("node_count", "edge_count", "comments", "issues"):
        assert on[key] == off[key], key
        assert on[key] == expected[key], key

    assert on["parity_rows"] == off["parity_rows"]
    assert len(on["parity_rows"]) == 25
    assert _result_digest(on["parity_rows"]) == _result_digest(off["parity_rows"])


@pytest.mark.parametrize("size", _lifecycle_sizes_for_oracle())
def test_auto_vacuum_tombstone_bookkeeping_oracle(size):
    """The storage counters the stop rule reads must themselves be consistent.

    `node_capacity`, `node_count` and `node_tombstones` are three views of two
    numbers; if they ever disagree, every "did it fire?" reading in this file is
    built on sand.
    """
    expected = LIFECYCLE_EXPECTED[size]
    for arm, capacity_key, tombstone_key in (
        ("default_on", "on_capacity", "on_tombstones"),
        ("off", "off_capacity", "off_tombstones"),
    ):
        record = lifecycle_runs(size)[arm]
        assert record["node_capacity"] == expected[capacity_key], arm
        assert record["node_tombstones"] == expected[tombstone_key], arm
        assert record["node_capacity"] - record["node_count"] == record["node_tombstones"], arm

    # The compacted arm must carry strictly less dead space than the control —
    # otherwise the vacuums fired and reclaimed nothing.
    assert lifecycle_runs(size)["default_on"]["node_tombstones"] < lifecycle_runs(size)["off"]["node_tombstones"]


# --------------------------------------------------------------------------
# Timing mode — release profile only. Never interpret a debug-profile number.
# --------------------------------------------------------------------------


def _percentile(samples: list[float], fraction: float) -> float:
    """Nearest-rank percentile. No numpy dependency in the benchmark suite."""
    ordered = sorted(samples)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


@pytest.mark.benchmark
@pytest.mark.parametrize("arm_off", [False, True], ids=["default_on", "vacuum_off"])
@pytest.mark.parametrize("size", LIFECYCLE_SIZES)
def test_bench_default_auto_vacuum_lifecycle(benchmark, size, arm_off):
    """DEFAULT-threshold delete lifecycle, versus an auto-vacuum-off control.

    The C3 measurement. Read `default_on` against `vacuum_off` at the same
    size — never against the forced-`0.001` cell above, which measures a
    different thing on a different configuration.

    `benchmark.pedantic(rounds=1)` is deliberate and not a mistake to be
    "fixed" by raising the round count: the sequence deletes 80% of the graph,
    so a second round would run against a different (mostly empty) graph and
    would not cross the threshold at all. Repetition for this cell is the
    **two-run release protocol** in the module docstring, not rounds within a
    run — which is also why the report demands agreement between two JSON
    files before it will say PROCEED.

    Everything the stop rule needs is written to `extra_info`; `stats.min` is
    the same figure including the untimed bookkeeping probes and is not what
    the report reads.
    """
    record: dict = {}

    def run_sequence():
        record.update(_run_lifecycle(size, auto_vacuum_off=arm_off))

    benchmark.pedantic(run_sequence, rounds=1, iterations=1, warmup_rounds=0)

    expected = LIFECYCLE_EXPECTED[size]
    # Untimed contract checks: a timing sample from a sequence that did not do
    # the expected work is not a slow number, it is a wrong one.
    assert record["node_count"] == expected["node_count"]
    assert record["edge_count"] == expected["edge_count"]
    assert record["comments"] == expected["comments"]
    if arm_off:
        assert record["fires"] == []
    else:
        assert tuple(record["fires"]) == EXPECTED_FIRE_BATCHES

    latencies = record["latencies_ms"]
    trigger_ms = [latencies[number - 1] for number in record["fires"]]
    benchmark.extra_info.update(
        {
            "vacuum_lifecycle": True,
            "vacuum_size": size,
            "vacuum_arm": record["arm"],
            "vacuum_batches": record["batches"],
            "vacuum_step": record["step"],
            "vacuum_fires": record["fires"],
            "vacuum_total_delete_s": record["total_delete_s"],
            "vacuum_max_trigger_ms": max(trigger_ms) if trigger_ms else None,
            "vacuum_p99_ms": _percentile(latencies, 0.99),
            "vacuum_median_ms": _percentile(latencies, 0.50),
            "vacuum_max_ms": max(latencies),
            "vacuum_parity_digest": _result_digest(record["parity_rows"]),
            "vacuum_final_counts": {
                "node_count": record["node_count"],
                "node_capacity": record["node_capacity"],
                "node_tombstones": record["node_tombstones"],
                "edge_count": record["edge_count"],
                "comments": record["comments"],
                "issues": record["issues"],
            },
            "vacuum_latencies_ms": latencies,
        }
    )


# --------------------------------------------------------------------------
# Stop-rule report: `python <this file> run_a.json run_b.json`
# --------------------------------------------------------------------------


def _load_lifecycle_run(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells: dict[tuple[int, str], dict] = {}
    for entry in payload.get("benchmarks", []):
        info = entry.get("extra_info") or {}
        if not info.get("vacuum_lifecycle"):
            continue
        cells[(info["vacuum_size"], info["vacuum_arm"])] = info
    return {"cells": cells, "path": path}


def _lifecycle_verdict(run: dict) -> tuple[list[dict], list[str]]:
    """Per-size stop-rule figures for one run, plus any invalidating problems."""
    rows: list[dict] = []
    problems: list[str] = []
    sizes = sorted({size for size, _ in run["cells"]})
    if not sizes:
        problems.append(f"no lifecycle cells in {run['path'].name}")
    for size in sizes:
        on = run["cells"].get((size, "default_on"))
        off = run["cells"].get((size, "off"))
        if on is None or off is None:
            problems.append(f"{run['path'].name}: size {size} is missing an arm")
            continue
        if not on["vacuum_fires"]:
            problems.append(
                f"{run['path'].name}: size {size} default arm fired no vacuum "
                "— the sequence never crossed the threshold (INVALID, not green)"
            )
            continue
        if off["vacuum_fires"]:
            problems.append(f"{run['path'].name}: size {size} off control fired a vacuum")
        # `node_capacity`/`node_tombstones` are *expected* to differ — that is
        # what a vacuum does. Every other count is a semantic fact and must match.
        semantic = ("node_count", "edge_count", "comments", "issues")
        if any(on["vacuum_final_counts"][key] != off["vacuum_final_counts"][key] for key in semantic):
            problems.append(f"{run['path'].name}: size {size} arms disagree on final counts")
        if on["vacuum_parity_digest"] != off["vacuum_parity_digest"]:
            problems.append(f"{run['path'].name}: size {size} arms disagree on query results")
        rows.append(
            {
                "size": size,
                "ratio": on["vacuum_total_delete_s"] / off["vacuum_total_delete_s"],
                "on_total_s": on["vacuum_total_delete_s"],
                "off_total_s": off["vacuum_total_delete_s"],
                "max_trigger_ms": on["vacuum_max_trigger_ms"],
                "off_p99_ms": off["vacuum_p99_ms"],
                "fires": on["vacuum_fires"],
            }
        )
    return rows, problems


def _report_lifecycle(paths: list[Path]) -> int:
    runs = [_load_lifecycle_run(path) for path in paths]
    per_run, problems = [], []
    for run in runs:
        rows, run_problems = _lifecycle_verdict(run)
        per_run.append({row["size"]: row for row in rows})
        problems.extend(run_problems)

    for path, rows in zip(paths, per_run, strict=True):
        print(f"\n== {path.name}")
        print(
            f"  {'size':>9} {'on total s':>11} {'off total s':>12} {'ratio':>7} "
            f"{'max trigger ms':>15} {'off p99 ms':>11}  fires"
        )
        for size in sorted(rows):
            row = rows[size]
            print(
                f"  {size:>9} {row['on_total_s']:>11.4f} {row['off_total_s']:>12.4f} "
                f"{row['ratio']:>6.3f}x {row['max_trigger_ms']:>15.2f} "
                f"{row['off_p99_ms']:>11.2f}  {row['fires']}"
            )

    if problems:
        print("\nABORT (invalid, not green):")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    if len(runs) < 2:
        print("\nOnly one run supplied — the stop rule needs two. No verdict.")
        return 1

    print("\n== stop rule (backlog C3)")
    shared = sorted(set(per_run[0]) & set(per_run[1]))
    ratio_hit = False
    for size in shared:
        ratios = [run[size]["ratio"] for run in per_run]
        agreed = all(ratio >= STOP_RATIO for ratio in ratios)
        ratio_hit = ratio_hit or agreed
        print(
            f"  (a) size {size:>9}: ratio {ratios[0]:.3f}x / {ratios[1]:.3f}x "
            f"-> {'>= ' if agreed else '< '}{STOP_RATIO:.2f} in both runs"
        )

    spike_hit = False
    if STOP_SPIKE_SIZE in shared:
        spikes = [run[STOP_SPIKE_SIZE]["max_trigger_ms"] for run in per_run]
        p99s = [run[STOP_SPIKE_SIZE]["off_p99_ms"] for run in per_run]
        spike_hit = all(
            spike >= STOP_SPIKE_MS and spike >= STOP_SPIKE_FACTOR * p99 for spike, p99 in zip(spikes, p99s, strict=True)
        )
        for spike, p99 in zip(spikes, p99s, strict=True):
            print(
                f"  (b) size {STOP_SPIKE_SIZE}: trigger {spike:.2f} ms vs off p99 "
                f"{p99:.2f} ms ({spike / p99:.1f}x); needs >= {STOP_SPIKE_MS:.0f} ms "
                f"and >= {STOP_SPIKE_FACTOR:.0f}x"
            )
    else:
        print(f"  (b) size {STOP_SPIKE_SIZE} absent from both runs — criterion not evaluated")

    if ratio_hit or spike_hit:
        print("\nPROCEED: profile the auto-vacuum path before changing any source.")
        return 0
    print("\nRETIRE: neither criterion reproduced in two runs. Record 'not reproduced',")
    print("close C3, and leave the forced-0.001 cell labelled as forced-trigger cost.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="C3 default auto-vacuum stop-rule report")
    parser.add_argument("runs", nargs="+", type=Path, help="two --benchmark-json files from release runs")
    args = parser.parse_args()
    return _report_lifecycle(args.runs)


if __name__ == "__main__":
    sys.exit(main())
