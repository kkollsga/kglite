"""Opt-in parallel Cypher runtime — differential and surface tests.

The contract this file pins:

* ``parallel=True`` is a **hint**. It never changes an answer, never changes
  row order, and never fails a query that would otherwise succeed.
* It is off by default, on every surface.
* Storage modes that cannot fan out (disk, spatial graphs) serve the query
  serially rather than refusing it, so portable code runs unchanged.

The fixture is sized above the interpreted-cost-class row gate on purpose —
below it the executor stays sequential and every assertion here would compare
two identical serial runs.
"""

import pandas as pd
import pytest

from kglite import KnowledgeGraph

# Above `PARALLEL_MIN_ROWS_INTERPRETED` (20_000) with margin. The queries below
# all carry one interpreted expression (`toUpper`), which is what puts them on
# that side of the gate; the compiled side needs 200k rows and belongs in the
# benchmark file, not a unit test.
ROWS = 25_000

# Every aggregate the fused node-scan operator serves, plus the grouped and
# filtered shapes. Group order is part of the comparison: `to_list()` preserves
# row order, and first-seen group order is a documented invariant.
QUERIES = [
    "MATCH (n:Item) RETURN toUpper(n.cat) AS c, count(*) AS n",
    "MATCH (n:Item) RETURN toUpper(n.cat) AS c, sum(n.value) AS s",
    "MATCH (n:Item) RETURN toUpper(n.cat) AS c, avg(n.value) AS a",
    "MATCH (n:Item) RETURN toUpper(n.cat) AS c, min(n.value) AS lo, max(n.value) AS hi",
    "MATCH (n:Item) RETURN toUpper(n.cat) AS c, count(DISTINCT n.value) AS d",
    "MATCH (n:Item) WHERE n.value > 500 RETURN toUpper(n.cat) AS c, count(*) AS n",
    "MATCH (n:Item) RETURN toUpper(n.cat) AS c, count(*) AS n ORDER BY c",
]


def _frame(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nid": list(range(rows)),
            "name": [f"Item_{i}" for i in range(rows)],
            "value": [float(i % 1000) for i in range(rows)],
            "cat": [f"cat_{i % 7}" for i in range(rows)],
        }
    )


# The candidate scan's compiled cost class (a bulk-loaded graph puts property
# matchers on a `ColumnStore`, which is the compiled route) needs 200_000
# candidates before it fans out. Scan+filter shapes therefore get their own,
# larger fixture; reusing the 25k one would compare two serial runs.
SCAN_ROWS = 200_137

# Scan + filter shapes, spanning both cost classes and the candidate sources
# the scan can draw from: compiled equality and range predicates, interpreted
# text predicates, an IN set, an id anchor, and a multi-label pattern.
SCAN_QUERIES = [
    "MATCH (n:Item {cat: 'cat_3'}) RETURN n.name AS nm",
    "MATCH (n:Item) WHERE n.cat = 'cat_3' RETURN n.name AS nm",
    "MATCH (n:Item) WHERE n.value > 500 RETURN n.name AS nm",
    "MATCH (n:Item) WHERE n.value >= 100 AND n.value < 200 RETURN n.name AS nm",
    "MATCH (n:Item) WHERE n.name STARTS WITH 'Item_19' RETURN n.name AS nm",
    "MATCH (n:Item) WHERE n.name CONTAINS '999' RETURN n.name AS nm",
    "MATCH (n:Item) WHERE n.cat IN ['cat_1', 'cat_5'] RETURN n.name AS nm",
    "MATCH (n:Item) WHERE n.cat = 'cat_2' RETURN count(*) AS n",
]


def _build(storage: str | None = None) -> KnowledgeGraph:
    graph = KnowledgeGraph(storage=storage) if storage else KnowledgeGraph()
    graph.add_nodes(_frame(ROWS), "Item", "nid", "name")
    return graph


@pytest.fixture(scope="module")
def graph() -> KnowledgeGraph:
    return _build()


@pytest.mark.parametrize("query", QUERIES)
def test_parallel_matches_serial(graph, query):
    """The differential axis: identical values AND identical row order."""
    serial = graph.cypher(query, parallel=False).to_list()
    parallel = graph.cypher(query, parallel=True).to_list()
    assert serial == parallel
    assert serial, "fixture produced no rows — the comparison would be vacuous"


def test_parallel_is_off_by_default(graph):
    """Omitting the kwarg must be exactly `parallel=False`."""
    query = QUERIES[0]
    assert graph.cypher(query).to_list() == graph.cypher(query, parallel=False).to_list()


def test_parallel_rejects_a_non_bool(graph):
    with pytest.raises(TypeError):
        graph.cypher(QUERIES[0], parallel="yes")


def test_parallel_is_keyword_only(graph):
    """`cypher` is keyword-only after `query`; a positional would silently land
    on `to_df` if that ever changed."""
    with pytest.raises(TypeError):
        graph.cypher(QUERIES[0], True)


@pytest.mark.parametrize("storage", ["memory", "mapped"])
def test_parallel_matches_serial_across_storage_modes(storage):
    graph = _build(storage)
    query = QUERIES[0]
    assert graph.cypher(query, parallel=True).to_list() == graph.cypher(query, parallel=False).to_list()


def test_disk_mode_serves_parallel_requests_serially(tmp_path):
    """Disk cannot fan out yet — its node materialisation parks into a shared
    query arena. `parallel=True` must be *ignored*, not refused: the flag is a
    hint, and refusing it would break code that runs against all three modes.
    """
    graph = KnowledgeGraph(storage="disk", path=str(tmp_path / "g"))
    graph.add_nodes(_frame(ROWS), "Item", "nid", "name")
    query = QUERIES[0]
    assert graph.cypher(query, parallel=True).to_list() == graph.cypher(query, parallel=False).to_list()


@pytest.fixture(scope="module")
def scan_graph() -> KnowledgeGraph:
    """Above the *compiled* cost class's row gate — see `SCAN_ROWS`."""
    graph = KnowledgeGraph()
    graph.add_nodes(_frame(SCAN_ROWS), "Item", "nid", "name")
    return graph


@pytest.mark.parametrize("query", SCAN_QUERIES)
def test_parallel_scan_filter_matches_serial(scan_graph, query):
    """Scan + filter, in order.

    Row order is the point: bucket order of an un-``ORDER BY``'d MATCH is a
    documented invariant, and partitioning the candidate scan is exactly the
    change that could reorder it while a set comparison stayed green. These are
    list comparisons.
    """
    serial = scan_graph.cypher(query, parallel=False).to_list()
    parallel = scan_graph.cypher(query, parallel=True).to_list()
    assert serial == parallel
    assert serial, "fixture produced no rows — the comparison would be vacuous"


def test_parallel_scan_filter_id_anchor_is_unaffected(scan_graph):
    """An id-anchored lookup never reaches the partitioned scan (it resolves
    through the index), and must be identical regardless."""
    query = "MATCH (n:Item {nid: 4242}) RETURN n.name AS nm"
    assert scan_graph.cypher(query, parallel=True).to_list() == scan_graph.cypher(query, parallel=False).to_list()


# Aggregation shapes that route to the *materialized* grouping path — the
# streaming pipeline declines `collect`, `median`, `percentile_*`, `std` and
# `mode`, which is exactly what sends them here. Q4 parallelises the per-group
# evaluation across groups, so these exercise it.
AGG_QUERIES = [
    "MATCH (n:Item) RETURN n.cat AS c, collect(n.value) AS vals",
    "MATCH (n:Item) RETURN n.cat AS c, collect(DISTINCT n.value) AS vals",
    "MATCH (n:Item) RETURN n.cat AS c, median(n.value) AS m",
    "MATCH (n:Item) RETURN n.cat AS c, mode(n.value) AS mo",
    "MATCH (n:Item) RETURN n.cat AS c, percentile_cont(n.value, 0.9) AS p90",
    "MATCH (n:Item) RETURN n.cat AS c, percentile_disc(n.value, 0.5) AS p50",
    "MATCH (n:Item) RETURN n.cat AS c, std(n.value) AS sd, variance(n.value) AS vr",
    "MATCH (n:Item) RETURN n.cat AS c, collect(n.value) AS vals, count(*) AS k",
    "MATCH (n:Item) WHERE n.value > 100 RETURN n.cat AS c, collect(n.value) AS vals",
    "MATCH (n:Item) RETURN n.cat AS c, collect(n.value) AS vals ORDER BY c DESC",
    "MATCH (n:Item) WITH n.cat AS c, collect(n.value) AS vals RETURN c, size(vals) AS k",
]


@pytest.mark.parametrize("query", AGG_QUERIES)
def test_parallel_aggregation_matches_serial(scan_graph, query):
    """Grouped aggregation, in order.

    `collect` is the order-sensitive one — it concatenates its group's values
    in row order — so this is a list comparison of a list-valued column, not a
    set comparison.
    """
    serial = scan_graph.cypher(query, parallel=False).to_list()
    parallel = scan_graph.cypher(query, parallel=True).to_list()
    assert serial == parallel
    assert serial, "fixture produced no rows — the comparison would be vacuous"


def test_parallel_aggregation_group_emission_order(scan_graph):
    """No ORDER BY: groups come back first-seen. `cat_i` is assigned by
    `nid % 7`, so first-seen order is `cat_0 … cat_6` — asserted absolutely,
    not just against the serial run."""
    query = "MATCH (n:Item) RETURN n.cat AS c, collect(n.value) AS vals"
    for parallel in (False, True):
        rows = scan_graph.cypher(query, parallel=parallel).to_list()
        assert [row["c"] for row in rows] == [f"cat_{i}" for i in range(7)]
