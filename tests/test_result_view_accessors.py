"""Tests for ResultView.one() / .scalar() / .column() and KnowledgeGraph.exists().

These are the row/column/scalar conveniences added on top of the lazy
ResultView, plus the O(1) existence check that mirrors node().
"""

import pytest

# ----------------------------------------------------------------------------
# ResultView.one()
# ----------------------------------------------------------------------------


class TestResultViewOne:
    def test_one_non_empty(self, small_graph):
        row = small_graph.cypher("MATCH (n:Person) RETURN n.name AS name, n.age AS age ORDER BY n.age").one()
        assert isinstance(row, dict)
        assert row == {"name": "Alice", "age": 28}

    def test_one_empty_is_none(self, small_graph):
        row = small_graph.cypher("MATCH (n:Person) WHERE n.age > 1000 RETURN n.name AS name").one()
        assert row is None

    def test_one_matches_first_row(self, small_graph):
        result = small_graph.cypher("MATCH (n:Person) RETURN n.name AS name ORDER BY n.name")
        assert result.one() == result[0]


# ----------------------------------------------------------------------------
# ResultView.scalar()
# ----------------------------------------------------------------------------


class TestResultViewScalar:
    def test_scalar_aggregate(self, small_graph):
        n = small_graph.cypher("MATCH (n:Person) RETURN count(n) AS c").scalar()
        assert n == 3

    def test_scalar_single_column(self, small_graph):
        name = small_graph.cypher("MATCH (n:Person) RETURN n.name AS name ORDER BY n.age").scalar()
        assert name == "Alice"

    def test_scalar_multi_column_returns_first(self, small_graph):
        """Multi-column result returns the FIRST column by RETURN order."""
        result = small_graph.cypher("MATCH (n:Person) RETURN n.name AS name, n.age AS age ORDER BY n.age")
        assert result.columns[0] == "name"
        assert result.scalar() == "Alice"

    def test_scalar_empty_is_none(self, small_graph):
        v = small_graph.cypher("MATCH (n:Person) WHERE n.age > 1000 RETURN n.name AS name").scalar()
        assert v is None


# ----------------------------------------------------------------------------
# ResultView.column()
# ----------------------------------------------------------------------------


class TestResultViewColumn:
    def test_column_happy_path(self, small_graph):
        names = small_graph.cypher("MATCH (n:Person) RETURN n.name AS name ORDER BY n.age").column("name")
        assert names == ["Alice", "Bob", "Charlie"]

    def test_column_second_column(self, small_graph):
        ages = small_graph.cypher("MATCH (n:Person) RETURN n.name AS name, n.age AS age ORDER BY n.age").column("age")
        assert ages == [28, 35, 42]

    def test_column_empty_result(self, small_graph):
        col = small_graph.cypher("MATCH (n:Person) WHERE n.age > 1000 RETURN n.name AS name").column("name")
        assert col == []

    def test_column_unknown_name_raises(self, small_graph):
        result = small_graph.cypher("MATCH (n:Person) RETURN n.name AS name, n.age AS age")
        with pytest.raises(KeyError) as exc:
            result.column("nope")
        msg = str(exc.value)
        # Lists available columns to guide the caller.
        assert "name" in msg
        assert "age" in msg


# ----------------------------------------------------------------------------
# KnowledgeGraph.exists()
# ----------------------------------------------------------------------------


class TestExists:
    def test_exists_hit(self, small_graph):
        assert small_graph.exists("Person", 1) is True

    def test_exists_miss(self, small_graph):
        assert small_graph.exists("Person", 99999) is False

    def test_exists_wrong_type(self, small_graph):
        # A real id but the wrong node type → no match.
        assert small_graph.exists("Company", 1) is False

    def test_exists_matches_node(self, small_graph):
        """exists() agrees with node() for both hit and miss."""
        assert small_graph.exists("Person", 2) == (small_graph.node("Person", 2) is not None)
        assert small_graph.exists("Person", 12345) == (small_graph.node("Person", 12345) is not None)

    def test_exists_id_coercion_matches_node(self, small_graph):
        """Id coercion is identical to node(): int and str-int normalize the same."""
        # node() resolves "2" (str) the same way as 2 (int); exists() must agree.
        assert small_graph.exists("Person", 2) is True
        assert small_graph.exists("Person", "2") == (small_graph.node("Person", "2") is not None)
