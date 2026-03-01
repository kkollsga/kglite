"""Tests for new Cypher features: keys(), math, auto-coercion, datetime, scientific notation, DateTime accessors."""

import pytest
import math
import pandas as pd
from kglite import KnowledgeGraph


@pytest.fixture
def graph():
    g = KnowledgeGraph()
    people = pd.DataFrame({
        'person_id': [1, 2],
        'name': ['Alice', 'Bob'],
        'age': [30, 25],
        'active': [True, False],
    })
    g.add_nodes(people, 'Person', 'person_id', 'name')

    edges = pd.DataFrame({
        'from_id': [1],
        'to_id': [2],
        'since': ['2020-01-15'],
        'weight': [0.8],
    })
    g.add_connections(edges, 'KNOWS', 'Person', 'from_id', 'Person', 'to_id',
                      columns=['since', 'weight'])
    return g


# ── keys() ──────────────────────────────────────────────────

class TestKeys:
    def test_keys_node(self, graph):
        rows = graph.cypher("MATCH (n:Person {name: 'Alice'}) RETURN keys(n) AS k")
        assert len(rows) == 1
        keys = rows[0]['k']
        # keys() returns a list (parsed from JSON list string by output layer)
        assert 'id' in keys
        assert 'title' in keys
        assert 'type' in keys
        assert 'age' in keys

    def test_keys_relationship(self, graph):
        rows = graph.cypher(
            "MATCH ()-[r:KNOWS]->() RETURN keys(r) AS k"
        )
        assert len(rows) == 1
        keys = rows[0]['k']
        assert 'type' in keys
        assert 'since' in keys
        assert 'weight' in keys


# ── Math functions ──────────────────────────────────────────

class TestMathFunctions:
    def test_log(self, graph):
        rows = graph.cypher("RETURN log(1) AS a, log(2.718281828) AS b")
        assert rows[0]['a'] == pytest.approx(0.0)
        assert rows[0]['b'] == pytest.approx(1.0, abs=1e-5)

    def test_log_negative_returns_null(self, graph):
        rows = graph.cypher("RETURN log(-1) AS v")
        assert rows[0]['v'] is None

    def test_log10(self, graph):
        rows = graph.cypher("RETURN log10(100) AS v")
        assert rows[0]['v'] == pytest.approx(2.0)

    def test_exp(self, graph):
        rows = graph.cypher("RETURN exp(0) AS a, exp(1) AS b")
        assert rows[0]['a'] == pytest.approx(1.0)
        assert rows[0]['b'] == pytest.approx(math.e, abs=1e-10)

    def test_pow(self, graph):
        rows = graph.cypher("RETURN pow(2, 10) AS v")
        assert rows[0]['v'] == pytest.approx(1024.0)

    def test_pi(self, graph):
        rows = graph.cypher("RETURN pi() AS v")
        assert rows[0]['v'] == pytest.approx(math.pi)

    def test_rand(self, graph):
        rows = graph.cypher("RETURN rand() AS v")
        val = rows[0]['v']
        assert isinstance(val, float)
        assert 0.0 <= val < 1.0


# ── String auto-coercion ───────────────────────────────────

class TestStringAutoCoercion:
    def test_substring_on_date(self, graph):
        rows = graph.cypher("RETURN substring(date('2020-06-15'), 0, 4) AS y")
        assert rows[0]['y'] == '2020'

    def test_left_on_date(self, graph):
        rows = graph.cypher("RETURN left(date('2020-06-15'), 7) AS v")
        assert rows[0]['v'] == '2020-06'

    def test_right_on_integer(self, graph):
        rows = graph.cypher("RETURN right(12345, 2) AS v")
        assert rows[0]['v'] == '45'

    def test_reverse_on_integer(self, graph):
        rows = graph.cypher("RETURN reverse(123) AS v")
        assert rows[0]['v'] == '321'

    def test_trim_on_boolean(self, graph):
        rows = graph.cypher("RETURN trim(true) AS v")
        assert rows[0]['v'] == 'true'

    def test_split_on_date(self, graph):
        rows = graph.cypher("RETURN split(date('2020-06-15'), '-') AS v")
        assert '2020' in rows[0]['v']

    def test_replace_on_date(self, graph):
        rows = graph.cypher("RETURN replace(date('2020-06-15'), '-', '/') AS v")
        assert rows[0]['v'] == '2020/06/15'


# ── datetime() alias ───────────────────────────────────────

class TestDatetimeAlias:
    def test_datetime_equals_date(self, graph):
        rows = graph.cypher(
            "RETURN date('2020-01-15') AS d1, datetime('2020-01-15') AS d2"
        )
        assert rows[0]['d1'] == rows[0]['d2']


# ── Scientific notation ────────────────────────────────────

class TestScientificNotation:
    def test_1e6(self, graph):
        rows = graph.cypher("RETURN 1e6 AS v")
        assert rows[0]['v'] == pytest.approx(1000000.0)

    def test_1_5e3(self, graph):
        rows = graph.cypher("RETURN 1.5e3 AS v")
        assert rows[0]['v'] == pytest.approx(1500.0)

    def test_2e_minus_3(self, graph):
        rows = graph.cypher("RETURN 2e-3 AS v")
        assert rows[0]['v'] == pytest.approx(0.002)

    def test_scientific_in_expression(self, graph):
        rows = graph.cypher("RETURN 1e6 / 1e3 AS v")
        assert rows[0]['v'] == pytest.approx(1000.0)


# ── DateTime property accessors ─────────────────────────────

class TestDateTimeAccessors:
    def test_year(self, graph):
        rows = graph.cypher(
            "WITH date('2020-06-15') AS d RETURN d.year AS y"
        )
        assert rows[0]['y'] == 2020

    def test_month(self, graph):
        rows = graph.cypher(
            "WITH date('2020-06-15') AS d RETURN d.month AS m"
        )
        assert rows[0]['m'] == 6

    def test_day(self, graph):
        rows = graph.cypher(
            "WITH date('2020-06-15') AS d RETURN d.day AS da"
        )
        assert rows[0]['da'] == 15

    def test_all_components(self, graph):
        rows = graph.cypher(
            "WITH date('2023-12-25') AS d RETURN d.year AS y, d.month AS m, d.day AS da"
        )
        assert rows[0]['y'] == 2023
        assert rows[0]['m'] == 12
        assert rows[0]['da'] == 25
