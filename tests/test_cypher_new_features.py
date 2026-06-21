"""Tests for new Cypher features: keys(), math, auto-coercion, datetime, scientific notation, DateTime accessors."""

import math

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph


@pytest.fixture
def graph():
    g = KnowledgeGraph()
    people = pd.DataFrame(
        {
            "person_id": [1, 2],
            "name": ["Alice", "Bob"],
            "age": [30, 25],
            "active": [True, False],
        }
    )
    g.add_nodes(people, "Person", "person_id", "name")

    edges = pd.DataFrame(
        {
            "from_id": [1],
            "to_id": [2],
            "since": ["2020-01-15"],
            "weight": [0.8],
        }
    )
    g.add_connections(edges, "KNOWS", "Person", "from_id", "Person", "to_id", columns=["since", "weight"])
    return g


# ── keys() ──────────────────────────────────────────────────


class TestKeys:
    def test_keys_node(self, graph):
        rows = graph.cypher("MATCH (n:Person {name: 'Alice'}) RETURN keys(n) AS k")
        assert len(rows) == 1
        keys = rows[0]["k"]
        # keys() returns a list (parsed from JSON list string by output layer)
        assert "id" in keys
        assert "title" in keys
        assert "type" in keys
        assert "age" in keys

    def test_keys_relationship(self, graph):
        rows = graph.cypher("MATCH ()-[r:KNOWS]->() RETURN keys(r) AS k")
        assert len(rows) == 1
        keys = rows[0]["k"]
        assert "type" in keys
        assert "since" in keys
        assert "weight" in keys


# ── Math functions ──────────────────────────────────────────


class TestMathFunctions:
    def test_log(self, graph):
        rows = graph.cypher("RETURN log(1) AS a, log(2.718281828) AS b")
        assert rows[0]["a"] == pytest.approx(0.0)
        assert rows[0]["b"] == pytest.approx(1.0, abs=1e-5)

    def test_log_negative_returns_null(self, graph):
        rows = graph.cypher("RETURN log(-1) AS v")
        assert rows[0]["v"] is None

    def test_log10(self, graph):
        rows = graph.cypher("RETURN log10(100) AS v")
        assert rows[0]["v"] == pytest.approx(2.0)

    def test_exp(self, graph):
        rows = graph.cypher("RETURN exp(0) AS a, exp(1) AS b")
        assert rows[0]["a"] == pytest.approx(1.0)
        assert rows[0]["b"] == pytest.approx(math.e, abs=1e-10)

    def test_pow(self, graph):
        rows = graph.cypher("RETURN pow(2, 10) AS v")
        assert rows[0]["v"] == pytest.approx(1024.0)

    def test_pi(self, graph):
        rows = graph.cypher("RETURN pi() AS v")
        assert rows[0]["v"] == pytest.approx(math.pi)

    def test_rand(self, graph):
        rows = graph.cypher("RETURN rand() AS v")
        val = rows[0]["v"]
        assert isinstance(val, float)
        assert 0.0 <= val < 1.0


# ── String auto-coercion ───────────────────────────────────


class TestStringAutoCoercion:
    def test_substring_on_date(self, graph):
        rows = graph.cypher("RETURN substring(date('2020-06-15'), 0, 4) AS y")
        assert rows[0]["y"] == "2020"

    def test_left_on_date(self, graph):
        rows = graph.cypher("RETURN left(date('2020-06-15'), 7) AS v")
        assert rows[0]["v"] == "2020-06"

    def test_right_on_integer(self, graph):
        rows = graph.cypher("RETURN right(12345, 2) AS v")
        assert rows[0]["v"] == "45"

    def test_reverse_on_integer(self, graph):
        rows = graph.cypher("RETURN reverse(123) AS v")
        assert rows[0]["v"] == "321"

    def test_trim_on_boolean(self, graph):
        rows = graph.cypher("RETURN trim(true) AS v")
        assert rows[0]["v"] == "true"

    def test_split_on_date(self, graph):
        rows = graph.cypher("RETURN split(date('2020-06-15'), '-') AS v")
        assert "2020" in rows[0]["v"]

    def test_replace_on_date(self, graph):
        rows = graph.cypher("RETURN replace(date('2020-06-15'), '-', '/') AS v")
        assert rows[0]["v"] == "2020/06/15"


# ── datetime() alias ───────────────────────────────────────


class TestDatetimeAlias:
    def test_datetime_of_bare_date_is_midnight_timestamp(self, graph):
        # 0.12 Cluster 1: datetime() now yields a real timestamp (midnight
        # for a bare date), whereas date() stays date-only.
        import datetime

        rows = graph.cypher("RETURN date('2020-01-15') AS d1, datetime('2020-01-15') AS d2")
        assert rows[0]["d1"] == "2020-01-15"
        assert rows[0]["d2"] == datetime.datetime(2020, 1, 15, 0, 0, 0)


# ── Scientific notation ────────────────────────────────────


class TestScientificNotation:
    def test_1e6(self, graph):
        rows = graph.cypher("RETURN 1e6 AS v")
        assert rows[0]["v"] == pytest.approx(1000000.0)

    def test_1_5e3(self, graph):
        rows = graph.cypher("RETURN 1.5e3 AS v")
        assert rows[0]["v"] == pytest.approx(1500.0)

    def test_2e_minus_3(self, graph):
        rows = graph.cypher("RETURN 2e-3 AS v")
        assert rows[0]["v"] == pytest.approx(0.002)

    def test_scientific_in_expression(self, graph):
        rows = graph.cypher("RETURN 1e6 / 1e3 AS v")
        assert rows[0]["v"] == pytest.approx(1000.0)


# ── DateTime property accessors ─────────────────────────────


class TestDateTimeAccessors:
    def test_year(self, graph):
        rows = graph.cypher("WITH date('2020-06-15') AS d RETURN d.year AS y")
        assert rows[0]["y"] == 2020

    def test_month(self, graph):
        rows = graph.cypher("WITH date('2020-06-15') AS d RETURN d.month AS m")
        assert rows[0]["m"] == 6

    def test_day(self, graph):
        rows = graph.cypher("WITH date('2020-06-15') AS d RETURN d.day AS da")
        assert rows[0]["da"] == 15

    def test_all_components(self, graph):
        rows = graph.cypher("WITH date('2023-12-25') AS d RETURN d.year AS y, d.month AS m, d.day AS da")
        assert rows[0]["y"] == 2023
        assert rows[0]["m"] == 12
        assert rows[0]["da"] == 25


class TestFormatCsv:
    """Tests for FORMAT CSV output."""

    def test_basic_csv(self, graph):
        result = graph.cypher("MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age FORMAT CSV")
        assert isinstance(result, str)
        lines = result.strip().split("\n")
        assert lines[0] == "n.name,n.age"
        assert lines[1] == "Bob,25"
        assert lines[2] == "Alice,30"

    def test_csv_with_nulls(self, graph):
        result = graph.cypher("MATCH (n:Person) RETURN n.name, n.missing FORMAT CSV")
        lines = result.strip().split("\n")
        assert lines[0] == "n.name,n.missing"
        # Null renders as empty field
        assert lines[1].endswith(",")

    def test_csv_quoting(self):
        g = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["Hello, World"], "note": ['He said "hi"']})
        g.add_nodes(df, "Thing", "id", "name")
        result = g.cypher("MATCH (n:Thing) RETURN n.name, n.note FORMAT CSV")
        lines = result.strip().split("\n")
        assert '"Hello, World"' in lines[1]
        assert '""hi""' in lines[1]

    def test_csv_empty_result(self, graph):
        result = graph.cypher("MATCH (n:Person {name: 'Nobody'}) RETURN n.name FORMAT CSV")
        assert isinstance(result, str)
        lines = result.strip().split("\n")
        assert len(lines) == 1  # header only

    def test_csv_case_insensitive(self, graph):
        result = graph.cypher("MATCH (n:Person) RETURN n.name format csv")
        assert isinstance(result, str)
        assert "n.name" in result

    def test_csv_with_aggregation(self, graph):
        result = graph.cypher("MATCH (n:Person) RETURN count(n) AS cnt FORMAT CSV")
        lines = result.strip().split("\n")
        assert lines[0] == "cnt"
        assert lines[1] == "2"

    def test_csv_overrides_to_df(self, graph):
        """FORMAT CSV takes precedence — result is str even with to_df=True."""
        result = graph.cypher("MATCH (n:Person) RETURN n.name FORMAT CSV", to_df=True)
        assert isinstance(result, str)


# ── 0.8.20: properties() / start_node() / end_node() ────────────


@pytest.fixture
def stats_graph():
    g = KnowledgeGraph()
    people = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Carol", "Dan", "Eve"],
            "age": [10, 20, 30, 40, 50],
        }
    )
    g.add_nodes(people, "Person", "person_id", "name")
    edges = pd.DataFrame(
        {
            "from_id": [1, 2, 3],
            "to_id": [2, 3, 4],
            "weight": [1.5, 2.5, 3.5],
        }
    )
    g.add_connections(edges, "KNOWS", "Person", "from_id", "Person", "to_id", columns=["weight"])
    return g


class TestPropertiesFunction:
    def test_properties_node(self, stats_graph):
        rows = list(stats_graph.cypher("MATCH (n:Person {name: 'Alice'}) RETURN properties(n) AS p"))
        assert len(rows) == 1
        p = rows[0]["p"]
        assert p["title"] == "Alice"
        assert p["type"] == "Person"
        assert p["age"] == 10
        assert p["id"] == 1

    def test_properties_edge(self, stats_graph):
        rows = list(stats_graph.cypher("MATCH (a {name: 'Alice'})-[r:KNOWS]->() RETURN properties(r) AS p"))
        assert len(rows) == 1
        p = rows[0]["p"]
        assert p["type"] == "KNOWS"
        assert p["weight"] == 1.5

    def test_properties_unbound_returns_null(self, stats_graph):
        rows = list(stats_graph.cypher("RETURN properties(x) AS p"))
        assert rows[0]["p"] is None


class TestStartEndNode:
    def test_start_end_node(self, stats_graph):
        rows = list(
            stats_graph.cypher("MATCH (a {name: 'Alice'})-[r:KNOWS]->(b) RETURN start_node(r) AS s, end_node(r) AS e")
        )
        assert len(rows) == 1
        assert rows[0]["s"] == "Alice"
        assert rows[0]["e"] == "Bob"


# ── 0.8.20: percentile / median / variance ─────────────────────


class TestPercentileAggregates:
    def test_percentile_cont_median(self, stats_graph):
        rows = list(stats_graph.cypher("MATCH (n:Person) RETURN percentile_cont(n.age, 0.5) AS p"))
        assert rows[0]["p"] == 30.0

    def test_percentile_cont_quartile(self, stats_graph):
        rows = list(stats_graph.cypher("MATCH (n:Person) RETURN percentile_cont(n.age, 0.25) AS p"))
        # On [10,20,30,40,50], p=0.25 → rank=1.0 → values[1] = 20
        assert rows[0]["p"] == 20.0

    def test_percentile_cont_interp(self, stats_graph):
        # On [10,20,30,40,50], p=0.1 → rank=0.4 → 10 + (20-10)*0.4 = 14
        rows = list(stats_graph.cypher("MATCH (n:Person) RETURN percentile_cont(n.age, 0.1) AS p"))
        assert abs(rows[0]["p"] - 14.0) < 1e-9

    def test_percentile_disc_median(self, stats_graph):
        rows = list(stats_graph.cypher("MATCH (n:Person) RETURN percentile_disc(n.age, 0.5) AS p"))
        assert rows[0]["p"] == 30.0

    def test_percentile_invalid_p(self, stats_graph):
        with pytest.raises(kglite.KgError, match="between 0 and 1"):
            list(stats_graph.cypher("MATCH (n:Person) RETURN percentile_cont(n.age, 1.5) AS p"))

    def test_median(self, stats_graph):
        rows = list(stats_graph.cypher("MATCH (n:Person) RETURN median(n.age) AS m"))
        assert rows[0]["m"] == 30.0

    def test_median_even_count(self):
        g = KnowledgeGraph()
        g.add_nodes(pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "v": [10, 30]}), "T", "id", "name")
        rows = list(g.cypher("MATCH (n:T) RETURN median(n.v) AS m"))
        assert rows[0]["m"] == 20.0  # average of 10, 30

    def test_variance(self, stats_graph):
        # Sample variance of [10,20,30,40,50] with n-1 denominator = 250
        rows = list(stats_graph.cypher("MATCH (n:Person) RETURN variance(n.age) AS v"))
        assert rows[0]["v"] == 250.0

    def test_var_samp_alias(self, stats_graph):
        rows = list(stats_graph.cypher("MATCH (n:Person) RETURN var_samp(n.age) AS v"))
        assert rows[0]["v"] == 250.0


# ── 0.8.20: reduce ─────────────────────────────────────────────


class TestReduce:
    def test_reduce_sum(self):
        g = KnowledgeGraph()
        rows = list(g.cypher("RETURN reduce(s = 0, x IN [1,2,3,4,5] | s + x) AS total"))
        assert rows[0]["total"] == 15

    def test_reduce_concat(self):
        g = KnowledgeGraph()
        rows = list(g.cypher('RETURN reduce(s = "", x IN ["a","b","c"] | s + x) AS r'))
        assert rows[0]["r"] == "abc"

    def test_reduce_empty_list(self):
        g = KnowledgeGraph()
        rows = list(g.cypher("RETURN reduce(s = 42, x IN [] | s + x) AS r"))
        assert rows[0]["r"] == 42

    def test_reduce_in_match(self, stats_graph):
        rows = list(
            stats_graph.cypher(
                "MATCH (n:Person) WITH collect(n.age) AS ages RETURN reduce(s = 0, x IN ages | s + x) AS total"
            )
        )
        assert rows[0]["total"] == 150  # 10+20+30+40+50

    def test_reduce_max(self):
        g = KnowledgeGraph()
        rows = list(
            g.cypher("RETURN reduce(m = 0, x IN [5, 3, 8, 1, 7] | CASE WHEN x > m THEN x ELSE m END) AS max_val")
        )
        assert rows[0]["max_val"] == 8


# ── 0.8.20: text predicates ────────────────────────────────────


class TestTextEditDistance:
    def test_basic(self):
        g = KnowledgeGraph()
        rows = list(g.cypher("RETURN text_edit_distance('kitten', 'sitting') AS d"))
        assert rows[0]["d"] == 3

    def test_identical(self):
        g = KnowledgeGraph()
        rows = list(g.cypher("RETURN text_edit_distance('hello', 'hello') AS d"))
        assert rows[0]["d"] == 0

    def test_empty_string(self):
        g = KnowledgeGraph()
        rows = list(g.cypher("RETURN text_edit_distance('', 'abc') AS d"))
        assert rows[0]["d"] == 3

    def test_unicode(self):
        g = KnowledgeGraph()
        rows = list(g.cypher("RETURN text_edit_distance('café', 'cafe') AS d"))
        assert rows[0]["d"] == 1

    def test_null_returns_null(self):
        g = KnowledgeGraph()
        rows = list(g.cypher("RETURN text_edit_distance(null, 'abc') AS d"))
        assert rows[0]["d"] is None


class TestTextNormalize:
    def test_basic(self):
        g = KnowledgeGraph()
        rows = list(g.cypher("RETURN text_normalize('  Hello, World!  ') AS s"))
        assert rows[0]["s"] == "hello world"

    def test_collapse_whitespace(self):
        g = KnowledgeGraph()
        rows = list(g.cypher("RETURN text_normalize('a  b\\tc\\nd') AS s"))
        assert rows[0]["s"] == "a b c d"

    def test_keep_alphanumeric(self):
        g = KnowledgeGraph()
        rows = list(g.cypher("RETURN text_normalize('Abc-123_xyz!') AS s"))
        assert rows[0]["s"] == "abc123xyz"


class TestTextJaccard:
    def test_basic(self):
        g = KnowledgeGraph()
        # {a,b,c} vs {b,c,d} → intersection {b,c}=2, union {a,b,c,d}=4 → 0.5
        rows = list(g.cypher("RETURN text_jaccard('a b c', 'b c d') AS j"))
        assert rows[0]["j"] == 0.5

    def test_identical(self):
        g = KnowledgeGraph()
        rows = list(g.cypher("RETURN text_jaccard('hello world', 'hello world') AS j"))
        assert rows[0]["j"] == 1.0

    def test_disjoint(self):
        g = KnowledgeGraph()
        rows = list(g.cypher("RETURN text_jaccard('a b', 'c d') AS j"))
        assert rows[0]["j"] == 0.0

    def test_custom_separator(self):
        g = KnowledgeGraph()
        rows = list(g.cypher("RETURN text_jaccard('a,b,c', 'b,c,d', ',') AS j"))
        assert rows[0]["j"] == 0.5


class TestTextNgrams:
    def test_basic(self):
        g = KnowledgeGraph()
        rows = list(g.cypher("RETURN text_ngrams('hello', 3) AS g"))
        assert rows[0]["g"] == ["hel", "ell", "llo"]

    def test_n_equals_length(self):
        g = KnowledgeGraph()
        rows = list(g.cypher("RETURN text_ngrams('abc', 3) AS g"))
        assert rows[0]["g"] == ["abc"]

    def test_n_too_large(self):
        g = KnowledgeGraph()
        rows = list(g.cypher("RETURN text_ngrams('ab', 5) AS g"))
        assert rows[0]["g"] == []


class TestTextContainsAny:
    def test_list_form(self):
        g = KnowledgeGraph()
        rows = list(g.cypher("RETURN text_contains_any('hello world', ['foo', 'world']) AS r"))
        assert rows[0]["r"] is True

    def test_list_no_match(self):
        g = KnowledgeGraph()
        rows = list(g.cypher("RETURN text_contains_any('hello world', ['foo', 'bar']) AS r"))
        assert rows[0]["r"] is False

    def test_variadic_form(self):
        g = KnowledgeGraph()
        rows = list(g.cypher("RETURN text_contains_any('hello world', 'foo', 'world') AS r"))
        assert rows[0]["r"] is True

    def test_empty_list(self):
        g = KnowledgeGraph()
        rows = list(g.cypher("RETURN text_contains_any('hello world', []) AS r"))
        assert rows[0]["r"] is False


class TestTextStartsWithAny:
    def test_list_form(self):
        g = KnowledgeGraph()
        rows = list(g.cypher("RETURN text_starts_with_any('foobar', ['foo', 'bar']) AS r"))
        assert rows[0]["r"] is True

    def test_list_no_match(self):
        g = KnowledgeGraph()
        rows = list(g.cypher("RETURN text_starts_with_any('xyz', ['foo', 'bar']) AS r"))
        assert rows[0]["r"] is False

    def test_variadic_form(self):
        g = KnowledgeGraph()
        rows = list(g.cypher("RETURN text_starts_with_any('foobar', 'xyz', 'foo') AS r"))
        assert rows[0]["r"] is True


class TestTextPredicatesIntegration:
    """Use lexical predicates against real graph rows."""

    def test_fuzzy_dedup(self):
        g = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice Smith", "Alice  Smith", "Bob Jones"]})
        g.add_nodes(df, "Person", "id", "name")
        # Find pairs with edit-distance ≤ 2 on normalized names
        rows = list(
            g.cypher(
                "MATCH (a:Person), (b:Person) WHERE a.id < b.id "
                "WITH a, b, text_edit_distance(text_normalize(a.title), text_normalize(b.title)) AS d "
                "WHERE d <= 2 RETURN a.title, b.title, d"
            )
        )
        # Alice Smith and Alice  Smith normalize to the same string
        assert any(r["d"] == 0 for r in rows)
