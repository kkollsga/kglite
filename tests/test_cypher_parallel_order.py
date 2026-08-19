"""Row order under the parallel runtime.

The four golden-order files this mirrors — ``test_cypher_top_k_ordering.py``,
``test_cypher_mixed_type_ordering.py``, ``test_cypher_order_by_after_aggregate.py``
and ``test_v0_9_02_nulls_ordering.py`` — pin order against hand-derived
expectations on 4-to-64-node fixtures. Adding a ``parallel=True`` axis *there*
would be vacuous: every one of those fixtures is three to four orders of
magnitude below the row gate, so both sides of the comparison would be the same
sequential run. This file carries the parallel axis instead, at fixture sizes
where the executor actually fans out, and covers the same four invariants:

* fused top-K selection and its tie-break,
* mixed-type ordering (the total-order fallback),
* ``ORDER BY`` after an aggregate, including first-seen group order,
* NULLS FIRST/LAST placement.

Each test is a differential — parallel against serial — plus an absolute
assertion wherever the expected order is derivable, so a bug that moved *both*
paths the same way is still caught.
"""

import pandas as pd
import pytest

from kglite import KnowledgeGraph

# Above the compiled cost class's gate, so the candidate scan fans out.
ROWS = 200_137
GROUPS = 7


@pytest.fixture(scope="module")
def graph() -> KnowledgeGraph:
    frame = pd.DataFrame(
        {
            "nid": list(range(ROWS)),
            "name": [f"Item_{i:07d}" for i in range(ROWS)],
            # Dense ranks with deliberate ties, so a tie-break is exercised.
            "score": [float(i % 500) for i in range(ROWS)],
            "grp": [f"g{i % GROUPS}" for i in range(ROWS)],
            # Two in every thousand rows carry no score at all.
            "maybe": [None if i % 500 == 0 else float(i % 97) for i in range(ROWS)],
            # A genuinely mixed-type column — ints, strings and NULLs in one
            # column, which is what forces the engine's total-order fallback
            # rather than a typed column compare. Bulk-loaded through an
            # object-dtype column so the fixture stays cheap; building it with
            # per-row SET (as the small golden file does) would not scale to
            # 200k rows.
            "mixed": [None if i % 401 == 0 else f"s{i % 89:03d}" for i in range(ROWS)],
        }
    )
    graph = KnowledgeGraph()
    graph.add_nodes(frame, "Item", "nid", "name")
    # `add_nodes` coerces an object-dtype column to one type, so the mixed
    # column has to be made mixed *after* the load — the same reason the small
    # golden file builds its mixed column with `SET`. One statement, 200 rows.
    graph.cypher("MATCH (n:Item) WHERE n.nid < 200 SET n.mixed = n.nid")
    return graph


def both(graph: KnowledgeGraph, query: str) -> list:
    """Run `query` with the parallel runtime off and on; assert they agree
    exactly, in order, and return the rows."""
    serial = graph.cypher(query, parallel=False).to_list()
    parallel = graph.cypher(query, parallel=True).to_list()
    assert serial == parallel, f"parallel diverged from serial on `{query}`"
    assert serial, f"no rows for `{query}` — the comparison would be vacuous"
    return serial


# ── top-K ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("direction", ["ASC", "DESC"])
@pytest.mark.parametrize("limit", [1, 5, 50])
def test_top_k_selection_and_order(graph, direction, limit):
    query = (
        f"MATCH (n:Item) WHERE n.score > 10 RETURN n.name AS nm, n.score AS s "
        f"ORDER BY n.score {direction}, n.nid ASC LIMIT {limit}"
    )
    rows = both(graph, query)
    assert len(rows) == limit
    scores = [row["s"] for row in rows]
    assert scores == sorted(scores, reverse=direction == "DESC")
    # The full sort's prefix — the bounded heap must select exactly it.
    full = f"MATCH (n:Item) WHERE n.score > 10 RETURN n.name AS nm ORDER BY n.score {direction}, n.nid ASC"
    assert [row["nm"] for row in rows] == [row["nm"] for row in graph.cypher(full, parallel=True).to_list()[:limit]]


# ── mixed-type ordering ─────────────────────────────────────────────────────


def test_mixed_type_column_orders_identically(graph):
    """A column holding integers, strings and NULLs falls back to the engine's
    total order across type classes; partitioning must not perturb it."""
    query = "MATCH (n:Item) WHERE n.nid < 5000 RETURN n.name AS nm, n.mixed AS m ORDER BY n.mixed ASC, n.nid ASC"
    rows = both(graph, query)
    kinds = {type(row["m"]).__name__ for row in rows}
    assert len(kinds) > 2, f"fixture is not mixed-type: {kinds}"
    # The type classes must come back contiguously — the total order groups by
    # class before comparing within one, so a partition boundary that leaked
    # would interleave them.
    boundaries = sum(1 for a, b in zip(rows, rows[1:]) if type(a["m"]).__name__ != type(b["m"]).__name__)
    assert boundaries == len(kinds) - 1, (
        f"type classes are not contiguous under ORDER BY: {boundaries} transitions for {len(kinds)} classes"
    )


# ── ORDER BY after an aggregate, and first-seen group order ─────────────────


def test_order_by_after_aggregate(graph):
    query = "MATCH (n:Item) WHERE n.score < 400 RETURN n.grp AS g, count(*) AS c ORDER BY c DESC, g ASC"
    rows = both(graph, query)
    assert len(rows) == GROUPS
    counts = [row["c"] for row in rows]
    assert counts == sorted(counts, reverse=True)


def test_unordered_group_emission_is_first_seen(graph):
    """No ORDER BY: groups come back in the order the scan first met them.
    That is the invariant partitioned aggregation has to reconstruct, and it is
    checked here against the absolute expected order, not just against serial.
    """
    query = "MATCH (n:Item) RETURN n.grp AS g, count(*) AS c"
    rows = both(graph, query)
    assert [row["g"] for row in rows] == [f"g{i}" for i in range(GROUPS)]


# ── NULLS ordering ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("direction", "nulls"),
    [("ASC", "FIRST"), ("ASC", "LAST"), ("DESC", "FIRST"), ("DESC", "LAST")],
)
def test_nulls_placement(graph, direction, nulls):
    query = (
        "MATCH (n:Item) WHERE n.nid < 3000 RETURN n.name AS nm, n.maybe AS m "
        f"ORDER BY n.maybe {direction} NULLS {nulls}, n.nid ASC LIMIT 20"
    )
    rows = both(graph, query)
    got_nulls = [row["m"] is None for row in rows]
    if nulls == "FIRST":
        assert got_nulls[0], "NULLS FIRST did not put a NULL first"
    else:
        assert not got_nulls[0], "NULLS LAST put a NULL first"


# ── unordered scan order ────────────────────────────────────────────────────


def test_unordered_scan_preserves_bucket_order(graph):
    """The load-bearing one: with no ORDER BY at all, the surviving candidates
    must come back in candidate order. This is what partitioning the scan could
    break silently."""
    query = "MATCH (n:Item) WHERE n.score = 42 RETURN n.nid AS id"
    rows = both(graph, query)
    ids = [row["id"] for row in rows]
    assert ids == sorted(ids), "the partitioned scan reordered its candidates"


# ── aggregation tie-breaks ──────────────────────────────────────────────────


def test_min_max_over_a_mixed_type_column_per_group(graph):
    """`min`/`max` fall back to the engine's total order across type classes.
    Evaluated per group and now evaluated *across* groups in parallel, so this
    pins that the per-group answer is unchanged by the fan-out."""
    query = (
        "MATCH (n:Item) WHERE n.nid < 5000 "
        "RETURN n.grp AS g, min(n.mixed) AS lo, max(n.mixed) AS hi, collect(n.mixed) AS all"
    )
    rows = both(graph, query)
    assert len(rows) == GROUPS
    kinds = {type(row["lo"]).__name__ for row in rows} | {type(row["hi"]).__name__ for row in rows}
    assert kinds, "no min/max values at all"


def test_mode_first_seen_winner_is_per_group(graph):
    """`mode`'s tie-break is "first-seen wins", and that state is per group —
    across-group fan-out must not touch it."""
    query = "MATCH (n:Item) WHERE n.nid < 20000 RETURN n.grp AS g, mode(n.score) AS m"
    rows = both(graph, query)
    assert len(rows) == GROUPS


def test_median_and_percentile_per_group(graph):
    """Whole-multiset aggregates stay sequential *within* a group; only the
    groups themselves are evaluated in parallel."""
    query = (
        "MATCH (n:Item) RETURN n.grp AS g, median(n.score) AS md, "
        "percentile_cont(n.score, 0.9) AS p90, percentile_disc(n.score, 0.25) AS p25"
    )
    rows = both(graph, query)
    assert len(rows) == GROUPS
    assert all(row["md"] is not None for row in rows)


def test_collect_preserves_row_order_within_each_group(graph):
    """The strongest order assertion available: `collect` exposes a group's row
    order directly. `nid` ascends with scan order, so each group's collected
    ids must be ascending."""
    query = "MATCH (n:Item) WHERE n.nid < 30000 RETURN n.grp AS g, collect(n.nid) AS ids"
    rows = both(graph, query)
    assert len(rows) == GROUPS
    for row in rows:
        ids = row["ids"]
        assert ids == sorted(ids), f"group {row['g']} collected out of row order"
