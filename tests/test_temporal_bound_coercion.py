"""ISO-text bounds of a declared type are stored as dates, not as text.

A declaration accepts a bound written as an ISO string, so a reload that
omits ``column_types`` used to write ``String`` beside the ``Date`` cells
already stored and record the property as ``String``. The loaders now coerce
a declared bound column whose cells all parse; one that does not is left for
the row check to refuse.

A ``half_open`` interval whose ``from`` is a timestamp later than the date
``to``'s midnight is inverted and refused, not classed empty.
"""

import datetime as dt
import warnings

import pandas as pd
import pytest

import kglite

pytestmark = pytest.mark.parity

MODES = ["memory", "mapped", "disk"]


def _graph(mode, tmp_path):
    if mode == "memory":
        return kglite.KnowledgeGraph()
    if mode == "mapped":
        return kglite.KnowledgeGraph(storage="mapped")
    return kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "g"))


def _declared_graph(mode, tmp_path, convention="closed"):
    graph = _graph(mode, tmp_path)
    first = pd.DataFrame(
        {
            "id": [1, 2],
            "vf": [dt.date(2000, 1, 1), dt.date(2005, 1, 1)],
            "vt": [dt.date(2004, 1, 1), dt.date(2009, 1, 1)],
        }
    )
    graph.add_nodes(first, "Status", "id", column_types={"vf": "date", "vt": "date"})
    graph.cypher(f"CALL db.temporal.declare({{node: 'Status', from: 'vf', to: 'vt', convention: '{convention}'}})")
    return graph


def _bounds(graph, label="Status"):
    rows = graph.cypher(
        f"FOR VALID_TIME ALL MATCH (s:{label}) RETURN s.id AS id, s.vf AS vf, s.vt AS vt ORDER BY id"
    ).to_list()
    return [(r["id"], r["vf"], r["vt"]) for r in rows]


@pytest.mark.parametrize("mode", MODES)
def test_text_bounds_of_a_reload_are_stored_as_dates(mode, tmp_path):
    graph = _declared_graph(mode, tmp_path)
    reload = pd.DataFrame({"id": [3, 4], "vf": ["2010-01-01", "2011-06-01"], "vt": ["2010-12-31", None]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = graph.add_nodes(reload, "Status", "id")
    assert not report.get("has_errors"), report.get("errors")
    assert _bounds(graph)[2:] == [
        (3, dt.date(2010, 1, 1), dt.date(2010, 12, 31)),
        (4, dt.date(2011, 6, 1), None),
    ]
    assert graph.schema()["node_types"]["Status"]["properties"]["vf"] == "DateTime"


@pytest.mark.parametrize("mode", MODES)
def test_text_bounds_with_a_time_part_are_stored_as_timestamps(mode, tmp_path):
    graph = _declared_graph(mode, tmp_path)
    reload = pd.DataFrame({"id": [3], "vf": ["2010-01-01T08:30:00"], "vt": ["2010-01-01T17:00:00"]})
    graph.add_nodes(reload, "Status", "id")
    stored = _bounds(graph)[2]
    assert stored == (3, dt.datetime(2010, 1, 1, 8, 30), dt.datetime(2010, 1, 1, 17, 0))


@pytest.mark.parametrize("mode", MODES)
def test_an_unparsable_text_bound_is_still_refused(mode, tmp_path):
    graph = _declared_graph(mode, tmp_path)
    reload = pd.DataFrame({"id": [3], "vf": ["not a date"], "vt": [None]})
    with pytest.raises(kglite.ArgumentError, match=r"row 0 \(0-based\) of the load"):
        graph.add_nodes(reload, "Status", "id")
    assert len(_bounds(graph)) == 2


@pytest.mark.parametrize("mode", MODES)
def test_text_edge_bounds_are_stored_as_dates(mode, tmp_path):
    graph = _declared_graph(mode, tmp_path)
    graph.add_nodes(pd.DataFrame({"id": [10]}), "Co", "id")
    graph.cypher(
        "MATCH (s:Status {id: 2}), (c:Co {id: 10}) "
        "CREATE (s)-[:OP {vf: date('2005-01-01'), vt: date('2006-01-01')}]->(c)"
    )
    graph.cypher("CALL db.temporal.declare({relationship: 'OP', from: 'vf', to: 'vt', convention: 'closed'})")
    edges = pd.DataFrame({"s": [1], "t": [10], "vf": ["2001-01-01"], "vt": ["2002-01-01"]})
    graph.add_relationships(edges, "OP", "Status", "s", "Co", "t", columns=["vf", "vt"])
    rows = graph.cypher("FOR VALID_TIME ALL MATCH ()-[r:OP]->() RETURN r.vf AS vf, r.vt AS vt ORDER BY vf").to_list()
    assert [(r["vf"], r["vt"]) for r in rows] == [
        (dt.date(2001, 1, 1), dt.date(2002, 1, 1)),
        (dt.date(2005, 1, 1), dt.date(2006, 1, 1)),
    ]


@pytest.mark.parametrize("mode", MODES)
def test_half_open_timestamp_from_after_a_date_to_midnight_is_refused_as_inverted(mode, tmp_path):
    graph = _declared_graph(mode, tmp_path, convention="half_open")
    row = pd.DataFrame({"id": [3], "vf": [pd.Timestamp("2011-01-01 12:00")], "vt": [pd.Timestamp("2011-01-01")]})
    with pytest.raises(kglite.ArgumentError, match="is after the to bound"):
        graph.add_nodes(row, "Status", "id")
    assert len(_bounds(graph)) == 2


@pytest.mark.parametrize("mode", MODES)
def test_closed_timestamp_from_within_a_date_to_day_is_valid(mode, tmp_path):
    graph = _declared_graph(mode, tmp_path, convention="closed")
    graph.cypher("CREATE (:Status {id: 3, vf: datetime('2011-01-01T12:00:00'), vt: date('2011-01-01')})")
    counts = graph.cypher("CALL db.temporal.declarations() YIELD empty_rows RETURN empty_rows").to_list()
    assert counts == [{"empty_rows": 0}]
