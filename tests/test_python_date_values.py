"""Dates reach Python as `datetime.date`, and date columns as `datetime64`.

A date value used to come back as its ISO string (`'2010-01-02'`) while a
datetime came back as `datetime.datetime`, so a date could not be compared,
sorted or subtracted as one without parsing it, and a `to_df()` date column was
a string column. Now a date is a `datetime.date` on every route —
`cypher()`/`to_list()`, rows, `collect()`, fluent `to_df()`, nested in lists
and maps — and a date column in a frame is `datetime64[ns]`. `str(d)` or
`d.isoformat()` gives the old text.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

import kglite

D = dt.date(2010, 1, 2)


@pytest.fixture
def graph() -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.cypher(
        "CREATE (a:M {n: 1, d: date('2010-01-02'), t: datetime('2010-01-02T03:04:05')})"
        "-[:R {on: date('2011-05-06')}]->(:M {n: 2})"
    ).to_list()
    return g


def test_a_stored_date_round_trips_as_a_date(graph) -> None:
    row = graph.cypher("MATCH (m:M {n: 1}) RETURN m.d AS d, m.t AS t, date('2020-01-01') AS lit").to_list()[0]
    assert row == {"d": D, "t": dt.datetime(2010, 1, 2, 3, 4, 5), "lit": dt.date(2020, 1, 1)}
    assert type(row["d"]) is dt.date and type(row["lit"]) is dt.date


def test_relationship_properties_nested_values_and_collect(graph) -> None:
    assert graph.cypher("MATCH ()-[r:R]->() RETURN r.on AS on").to_list() == [{"on": dt.date(2011, 5, 6)}]
    nested = graph.cypher("RETURN [date('2010-01-02'), 'x'] AS l, {a: date('2010-01-02')} AS m").to_list()
    assert nested == [{"l": [D, "x"], "m": {"a": D}}]
    collected = graph.select("M").where({"n": 1}).collect().to_list()
    assert collected[0]["d"] == D


def test_null_dates_stay_none(graph) -> None:
    rows = graph.cypher("MATCH (m:M) RETURN m.d AS d ORDER BY m.n").to_list()
    assert rows == [{"d": D}, {"d": None}]


def test_a_date_parameter_round_trips(graph) -> None:
    rows = graph.cypher("MATCH (m:M) WHERE m.d = $d RETURN m.d AS d", params={"d": D}).to_list()
    assert rows == [{"d": D}]
    assert graph.cypher("RETURN $d AS d", params={"d": D}).to_list() == [{"d": D}]


def test_to_df_types_date_columns_as_datetime64(graph) -> None:
    df = graph.cypher("MATCH (m:M) RETURN m.d AS d, m.t AS t ORDER BY m.n").to_df()
    assert str(df["d"].dtype) == "datetime64[ns]"
    assert df["d"].iloc[0] == pd.Timestamp(2010, 1, 2) and pd.isna(df["d"].iloc[1])
    # A datetime column keeps the dtype it always had.
    assert df["t"].dtype.kind == "M"
    fluent = graph.select("M").to_df()
    assert str(fluent["d"].dtype) == "datetime64[ns]"


def test_mixed_columns_keep_each_value(graph) -> None:
    df = graph.cypher("UNWIND [date('2010-01-02'), 'x'] AS v RETURN v").to_df()
    assert df["v"].dtype == object and list(df["v"]) == [D, "x"]
    df = graph.cypher("UNWIND [date('2010-01-02'), 3] AS v RETURN v").to_df()
    assert df["v"].dtype == object and list(df["v"]) == [D, 3]
    df = graph.cypher("UNWIND [date('2010-01-02'), datetime('2010-01-02T05:00:00')] AS v RETURN v").to_df()
    assert str(df["v"].dtype) == "datetime64[ns]"
    assert list(df["v"]) == [pd.Timestamp(2010, 1, 2), pd.Timestamp(2010, 1, 2, 5)]


def test_the_user_test_shapes(graph) -> None:
    # probe2: a returned date, a relationship date, a literal and `.year`.
    row = graph.cypher(
        "MATCH (a:M {n: 1})-[r:R]->() RETURN a.d AS vt, r.on AS od, date('2020-01-01') AS lit, a.d.year AS y"
    ).to_list()[0]
    assert [type(v) for v in row.values()] == [dt.date, dt.date, dt.date, int]
    # probe3: a returned date fed straight back into an equality finds its row.
    returned = graph.cypher("MATCH (m:M {n: 1}) RETURN m.d AS d").to_list()[0]["d"]
    again = graph.cypher("MATCH (m:M) WHERE m.d = $v RETURN m.n AS n", params={"v": returned}).to_list()
    assert again == [{"n": 1}]
    assert graph.select("M").where({"d": returned}).len() == 1
