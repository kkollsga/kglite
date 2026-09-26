"""`date({year, month, day})` and `datetime({...})`: openCypher's map form.

Only strings were accepted, so building a date from integers needed
`date(toString(y) + '-01-01')`. `point()` took its map form in 0.18.1; the
temporal constructors now do too.
"""

from __future__ import annotations

import datetime as dt

import pytest

import kglite


@pytest.fixture
def graph() -> kglite.KnowledgeGraph:
    return kglite.KnowledgeGraph()


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("date({year: 2010, month: 6, day: 30})", dt.date(2010, 6, 30)),
        ("date({year: 2010})", dt.date(2010, 1, 1)),
        ("date({year: 2010, month: 2})", dt.date(2010, 2, 1)),
        ("date({year: null, month: 1})", None),
        ("date({year: 2010, month: 6, day: 30}) = date('2010-06-30')", True),
        (
            "datetime({year: 2010, month: 1, day: 2, hour: 3, minute: 4, second: 5, millisecond: 6})",
            dt.datetime(2010, 1, 2, 3, 4, 5, 6000),
        ),
        ("datetime({year: 2010}) = datetime('2010-01-01T00:00:00')", True),
    ],
)
def test_map_form_builds_the_value(graph, expression, expected) -> None:
    assert graph.cypher(f"RETURN {expression} AS v").to_list() == [{"v": expected}]


def test_map_form_takes_row_values(graph) -> None:
    rows = graph.cypher("UNWIND [1990, 2000] AS y RETURN date({year: y, month: 1, day: 1}) AS d").to_list()
    assert rows == [{"d": dt.date(1990, 1, 1)}, {"d": dt.date(2000, 1, 1)}]


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("date({year: 2010, month: 2, day: 30})", "2010-02-30 is not a valid date"),
        ("date({year: 2010, mnth: 2})", "unknown key 'mnth'"),
        ("date({year: 2010.5})", "year must be an integer"),
        ("date({month: 1})", "missing 'year'"),
        ("date({year: 2010, hour: 1})", "unknown key 'hour'"),
        ("datetime({year: 2010, timezone: 'Z'})", "unknown key 'timezone'"),
        ("datetime({year: 2010, hour: 25})", "is not a valid datetime"),
    ],
)
def test_map_form_refuses_what_it_cannot_build(graph, expression, message) -> None:
    with pytest.raises(kglite.CypherExecutionError, match=message):
        graph.cypher(f"RETURN {expression} AS v").to_list()
