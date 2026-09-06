"""Declared text grammar and native timestamp values preserve their input meaning."""

import csv
import datetime as dt
import json
import math

import pandas as pd
import pytest

import kglite
from tests.value_assertions import assert_rows_equal, assert_value_equal

SCALARS = [
    (
        "int",
        [
            "9007199254740993.0",
            "9223372036854775807.0",
            "-9223372036854775808.0",
            "9007199254740993.1",
            "9223372036854775808.0",
            " +001.000e+3 ",
            "bad",
            "",
        ],
        [9007199254740993, 2**63 - 1, -(2**63), None, None, 1000, None, None],
    ),
    ("float", [" 1.5 ", "bad", "", "2e3"], [1.5, None, None, 2000.0]),
    ("bool", [" TRUE ", " no ", "bad", ""], [True, False, None, None]),
    (
        "date",
        [" 2025-01-02 ", "2025-01-02T03:04:05", "1609459200000", "2025/01/02", "bad", ""],
        ["2025-01-02", "2025-01-02", "2021-01-01", None, None, None],
    ),
]


def blueprint_graph(tmp_path, kind, values, route):
    path = tmp_path / f"{kind}-{route}.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "v"])
        writer.writerows(enumerate(values))
    file = {"format": route}
    if route == "csv":
        file["path"] = path.name
    spec = {
        "settings": {"root": str(tmp_path)},
        "files": {"rows": file},
        "nodes": {"N": {"file": "rows", "pk": "id", "properties": {"v": kind}}},
    }
    spec_path = tmp_path / f"{kind}-{route}.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    kwargs = {"frames": {"rows": pd.DataFrame({"id": range(len(values)), "v": values})}} if route == "frame" else {}
    return kglite.from_blueprint(spec_path, save=False, **kwargs)


@pytest.mark.parametrize("kind,values,expected", SCALARS)
def test_declared_csv_frame_absolute_values(tmp_path, kind, values, expected):
    for route in ["csv", "frame"]:
        graph = blueprint_graph(tmp_path, kind, values, route)
        actual = graph.cypher("MATCH(n:N) RETURN n.v AS v ORDER BY n.id").to_list()
        assert_rows_equal(actual, [{"v": v} for v in expected], order="ordered")


@pytest.mark.parametrize("route", ["nodes", "edges"])
@pytest.mark.parametrize("kind,values,expected", SCALARS[:3])
def test_direct_declared_scalar_values(route, kind, values, expected):
    graph = kglite.KnowledgeGraph()
    if route == "nodes":
        graph.add_nodes(pd.DataFrame({"id": range(len(values)), "v": values}), "N", "id", column_types={"v": kind})
        query = "MATCH(n:N) RETURN n.id AS id,n.v AS v ORDER BY id"
    else:
        graph.add_nodes(pd.DataFrame({"id": range(len(values) + 1)}), "N", "id")
        frame = pd.DataFrame({"src": range(len(values)), "dst": [len(values)] * len(values), "v": values})
        graph.add_connections(frame, "R", "N", "src", "N", "dst", column_types={"v": kind})
        query = "MATCH(n:N)-[r:R]->() RETURN n.id AS id,r.v AS v ORDER BY id"
    assert_rows_equal(
        graph.cypher(query).to_list(), [{"id": i, "v": v} for i, v in enumerate(expected)], order="ordered"
    )


def test_direct_date_aliases_remain_separate_from_blueprint_grammar():
    graph = kglite.KnowledgeGraph()
    values = ["2025/01/02", "02-01-2025", "01/02/2025"]
    graph.add_nodes(pd.DataFrame({"id": range(3), "v": values}), "N", "id", column_types={"v": "date"})
    assert graph.cypher("MATCH(n:N) RETURN n.v AS v").column("v") == ["2025-01-02"] * 3


@pytest.mark.parametrize("downcast", [False, True])
def test_native_float_int64_upper_boundary_never_saturates(downcast):
    graph = kglite.KnowledgeGraph()
    values = [float(2**63), float(-(2**63)), 1.0] if downcast else [float(2**63), float(-(2**63)), 1.5]
    options = {"nullable_int_downcast": True} if downcast else {"column_types": {"v": "int64"}}
    graph.add_nodes(pd.DataFrame({"id": range(3), "v": values}), "N", "id", **options)
    expected = values if downcast else [None, -(2**63), None]
    assert_value_equal(graph.cypher("MATCH(n:N) RETURN n.v AS v ORDER BY n.id").column("v"), expected)


class NaiveZone(dt.tzinfo):
    def utcoffset(self, value):
        return None


class SeasonalZone(dt.tzinfo):
    def utcoffset(self, value):
        return dt.timedelta(hours=2 if value.month >= 6 else 1)

    def dst(self, value):
        return dt.timedelta(0)


@pytest.mark.parametrize(
    "value,expected",
    [
        (
            dt.datetime(2025, 1, 2, 0, 30, 5, 123456, tzinfo=dt.timezone(dt.timedelta(hours=2))),
            dt.datetime(2025, 1, 1, 22, 30, 5, 123456),
        ),
        (dt.datetime(2025, 1, 2, 0, 30, tzinfo=NaiveZone()), dt.datetime(2025, 1, 2, 0, 30)),
        (dt.datetime(2025, 7, 2, 0, 30, tzinfo=SeasonalZone()), dt.datetime(2025, 7, 1, 22, 30)),
    ],
)
@pytest.mark.parametrize("route", ["nodes", "edges"])
def test_scalar_nested_timestamp_same_utc_instant(value, expected, route):
    graph = kglite.KnowledgeGraph()
    frame = pd.DataFrame(
        {
            "id": [1],
            "v": pd.Series([value], dtype=object),
            "m": [{"v": value}],
            "l": [[value]],
            "day": pd.Series([value], dtype=object),
        }
    )
    types = {"v": "timestamp", "m": "map", "l": "list", "day": "date"}
    if route == "nodes":
        graph.add_nodes(frame, "N", "id", column_types=types)
        query = "MATCH(n:N) RETURN n.v AS v,n.m.v AS m,n.l[0] AS l,n.day AS day"
    else:
        graph.add_nodes(pd.DataFrame({"id": [1, 2]}), "N", "id")
        frame["src"], frame["dst"] = [1], [2]
        graph.add_connections(frame, "R", "N", "src", "N", "dst", column_types=types)
        query = "MATCH()-[n:R]->() RETURN n.v AS v,n.m.v AS m,n.l[0] AS l,n.day AS day"
    assert_rows_equal(
        graph.cypher(query).to_list(),
        [{"v": expected, "m": expected, "l": expected, "day": value.date().isoformat()}],
        order="ordered",
    )


def test_csv_frame_nan_text_observation(tmp_path):
    # Both use the declared floating grammar; actual pandas missing markers stay NULL.
    for route in ["csv", "frame"]:
        graph = blueprint_graph(tmp_path, "float", ["NaN"], route)
        assert math.isnan(graph.cypher("MATCH(n:N) RETURN n.v AS v").scalar())


def test_declared_native_integer_out_of_range_does_not_round_back():
    values = [2**63, -(2**63) - 1, 2**63 - 1]
    graph = kglite.KnowledgeGraph()
    frame = pd.DataFrame({"id": range(3), "v": pd.Series(values, dtype=object)})
    graph.add_nodes(frame, "N", "id", column_types={"v": "int64"})
    assert_value_equal(graph.cypher("MATCH(n:N) RETURN n.v AS v ORDER BY n.id").column("v"), [None, None, 2**63 - 1])
