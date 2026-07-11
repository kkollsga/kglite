"""Docker-free contract tests for the independently authored oracle core."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from cypher_conformance import (  # noqa: E402
    FIXTURE_BUILDERS,
    QueryCase,
    QueryResult,
    _canonical,
    _normalise,
    run_conformance,
)


class FakeResult:
    def __init__(self, columns, rows):
        self.columns = list(columns)
        self._rows = list(rows)

    def to_list(self):
        return list(self._rows)


class FakeGraph:
    def __init__(self, name, responses):
        self.name = name
        self.responses = responses

    def cypher(self, query, params=None):
        response = self.responses[query]
        if isinstance(response, BaseException):
            raise response
        columns, rows = response
        return FakeResult(columns, rows)


class FakeNeoResult:
    def __init__(self, columns, rows):
        self._columns = columns
        self._rows = rows

    def keys(self):
        return self._columns

    def __iter__(self):
        return iter(self._rows)


class FakeSession:
    def __init__(self, driver):
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def run(self, query, **params):
        self.driver.runs.append((self.driver.active, query, params))
        response = self.driver.responses[self.driver.active][query]
        if isinstance(response, BaseException):
            raise response
        columns, rows = response
        return FakeNeoResult(columns, rows)


class FakeDriver:
    def __init__(self, responses):
        self.responses = responses
        self.active = None
        self.databases = []
        self.runs = []

    def session(self, *, database):
        self.databases.append(database)
        return FakeSession(self)


def _runner(cases, graphs, neo=None, **kwargs):
    neo = neo or {name: graph.responses for name, graph in graphs.items()}
    driver = FakeDriver(neo)
    activations = []

    def export(graph):
        activations.append(graph.name)
        driver.active = graph.name

    report = run_conformance(
        cases,
        builders={name: (lambda graph=graph: graph) for name, graph in graphs.items()},
        driver=driver,
        database="contract-db",
        export_graph=export,
        **kwargs,
    )
    return report, driver, activations


def test_all_differential_fixtures_have_plain_builders():
    assert set(FIXTURE_BUILDERS) == {
        "small_graph",
        "social_graph",
        "file_imports_graph",
        "multi_label_graph",
    }
    assert all(not hasattr(builder, "_pytestfixturefunction") for builder in FIXTURE_BUILDERS.values())


def test_fixture_is_restored_across_interleaved_transitions():
    response = {"RETURN 1 AS n": (("n",), [{"n": 1}])}
    graphs = {name: FakeGraph(name, response) for name in ("small", "social")}
    cases = [
        QueryCase("a", "small", "RETURN 1 AS n"),
        QueryCase("b", "social", "RETURN 1 AS n"),
        QueryCase("c", "small", "RETURN 1 AS n"),
    ]

    report, driver, activations = _runner(cases, graphs)

    assert report.ok
    assert activations == ["small", "social", "small"]
    assert [active for active, _, _ in driver.runs] == activations
    assert report.counters["fixture_activation"] == 3


def test_database_is_used_for_every_query_and_side_effect_probe():
    responses = {
        "CREATE ()": ((), []),
        "MATCH (n) RETURN count(n) AS n": (("n",), [{"n": 1}]),
    }
    graph = FakeGraph("g", responses)
    case = QueryCase("write", "g", "CREATE ()", side_effect_query="MATCH (n) RETURN count(n) AS n")

    report, driver, _ = _runner([case], {"g": graph})

    assert report.ok
    assert driver.databases == ["contract-db", "contract-db"]
    assert report.counters["pass_side_effect"] == 1


class Node(dict):
    def __init__(self, labels, **properties):
        super().__init__(properties)
        self.labels = labels


class Relationship(dict):
    type = "KNOWS"
    start_node = object()
    end_node = object()


class PathValue:
    pass


PathValue.__name__ = "Path"


class Duration:
    months = 2
    days = 3
    seconds = 4
    nanoseconds = 5


class Point:
    srid = 4326
    x = 10.5
    y = 59.9
    z = None


def test_typed_canonicalization_covers_nested_graph_temporal_and_spatial_values():
    node = Node({"Person"}, name="Ada", active=True)
    rel = Relationship(since=2020)
    path = PathValue()
    path.nodes = [node, Node({"Person"}, name="Bob")]
    path.relationships = [rel]

    value = {
        "scalar": [None, True, 1, 1.0, "1"],
        "node": node,
        "rel": rel,
        "path": path,
        "date": dt.date(2024, 1, 2),
        "duration": Duration(),
        "point": Point(),
    }
    canonical = _canonical(value)
    rendered = repr(canonical)

    assert "'null'" in rendered
    assert "'bool'" in rendered
    assert "'integer'" in rendered
    assert "'float'" in rendered
    assert "'node'" in rendered
    assert "'relationship'" in rendered
    assert "'path'" in rendered
    assert "'date'" in rendered
    assert "'duration'" in rendered
    assert "'point'" in rendered


def test_kglite_graph_dictionaries_match_driver_graph_values():
    kg_node = {"id": 7, "labels": ["Person"], "properties": {"name": "Ada"}}
    kg_rel = {"id": 9, "start": 7, "end": 8, "type": "KNOWS", "properties": {"since": 2020}}
    assert _canonical(kg_node) == _canonical(Node({"Person"}, name="Ada"))
    assert _canonical(kg_rel) == _canonical(Relationship(since=2020))


def test_normalisation_preserves_columns_and_duplicate_rows():
    result = QueryResult(("b", "a"), [{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    columns, rows = _normalise(result, False)
    assert columns == ("b", "a")
    assert len(rows) == 2
    assert rows[0] == rows[1]


@pytest.mark.parametrize(
    ("ordered", "expected_ok"),
    [(True, False), (False, True)],
)
def test_explicit_order_metadata_controls_row_comparison(ordered, expected_ok):
    kg = FakeGraph("g", {"Q": (("n",), [{"n": 1}, {"n": 2}])})
    neo = {"g": {"Q": (("n",), [{"n": 2}, {"n": 1}])}}
    report, _, _ = _runner([QueryCase("q", "g", "Q", ordered=ordered)], {"g": kg}, neo)
    assert report.ok is expected_ok


def test_column_order_mismatch_is_a_failure_even_when_maps_match():
    kg = FakeGraph("g", {"Q": (("a", "b"), [{"a": 1, "b": 2}])})
    neo = {"g": {"Q": (("b", "a"), [{"a": 1, "b": 2}])}}
    report, _, _ = _runner([QueryCase("q", "g", "Q")], {"g": kg}, neo)
    assert not report.ok
    assert report.counters["fail_result"] == 1


class SyntaxFailure(Exception):
    code = "Neo.ClientError.Statement.SyntaxError"


def test_expected_error_categories_are_compared_on_both_engines():
    kg = FakeGraph("g", {"BAD": SyntaxFailure("cannot parse")})
    neo = {"g": {"BAD": SyntaxFailure("different wording")}}
    case = QueryCase("bad", "g", "BAD", expected_error="syntax")
    report, _, _ = _runner([case], {"g": kg}, neo)
    assert report.ok
    assert report.counters["pass_expected_error"] == 1


def test_side_effect_mismatch_is_reported_separately():
    responses = {"WRITE": ((), []), "SNAPSHOT": (("n",), [{"n": 1}])}
    graph = FakeGraph("g", responses)
    neo = {"g": {"WRITE": ((), []), "SNAPSHOT": (("n",), [{"n": 2}])}}
    case = QueryCase("write", "g", "WRITE", side_effect_query="SNAPSHOT")
    report, _, _ = _runner([case], {"g": graph}, neo)
    assert not report.ok
    assert report.counters["pass_result"] == 1
    assert report.counters["fail_side_effect"] == 1


def test_skip_reasons_have_distinct_counters():
    response = {"RETURN 1": (("1",), [{"1": 1}])}
    graph = FakeGraph("g", response)
    cases = [
        QueryCase("intentional", "g", "RETURN 1"),
        QueryCase("extension", "g", "CALL kglite.special()"),
        QueryCase("missing", "absent", "RETURN 1"),
    ]
    report, _, activations = _runner(
        cases,
        {"g": graph},
        intentional_divergences={"intentional": "documented"},
    )
    assert report.ok
    assert activations == []
    assert report.counters["skip_intentional_divergence"] == 1
    assert report.counters["skip_kglite_extension"] == 1
    assert report.counters["skip_missing_fixture"] == 1
