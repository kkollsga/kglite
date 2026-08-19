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
