"""Tests for fluent API parity features: regex, NOT, offset, filter_any, group_by, has_connection."""

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph


@pytest.fixture
def graph():
    g = KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
                "city": ["Oslo", "Bergen", "Oslo", "Trondheim", "Bergen"],
                "age": [30, 25, 35, 28, 32],
            }
        ),
        "Person",
        "id",
        "name",
    )

    g.cypher("""
        MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
        CREATE (a)-[:KNOWS]->(b)
    """)
    g.cypher("""
        MATCH (a:Person {name: 'Alice'}), (c:Person {name: 'Charlie'})
        CREATE (a)-[:KNOWS]->(c)
    """)
    g.cypher("""
        MATCH (b:Person {name: 'Bob'}), (d:Person {name: 'Diana'})
        CREATE (b)-[:WORKS_WITH]->(d)
    """)
    return g


# ============================================================================
# Regex operator in filter()
# ============================================================================


class TestRegexFilter:
    def test_regex_basic(self, graph):
        r = graph.select("Person").where({"name": {"regex": "^[A-C].*"}}).titles()
        assert sorted(r) == ["Alice", "Bob", "Charlie"]

    def test_regex_case_sensitive(self, graph):
        r = graph.select("Person").where({"name": {"regex": "^alice"}}).titles()
        assert not r  # empty selection

    def test_regex_case_insensitive(self, graph):
        r = graph.select("Person").where({"name": {"regex": "(?i)^alice"}}).titles()
        assert r == ["Alice"]

    def test_regex_end_anchor(self, graph):
        r = graph.select("Person").where({"name": {"regex": "e$"}}).titles()
        assert sorted(r) == ["Alice", "Charlie", "Eve"]

    def test_regex_invalid_pattern(self, graph):
        with pytest.raises(kglite.KgError, match="Invalid regex"):
            graph.select("Person").where({"name": {"regex": "[invalid"}})

    def test_regex_tilde_operator(self, graph):
        r = graph.select("Person").where({"name": {"=~": "^B.*"}}).titles()
        assert r == ["Bob"]

    # `regex` and `=~` were synonyms until 0.16.6. They are not: `=~` is
    # Cypher's operator and Cypher's `=~` matches the whole value, while
    # `regex` keeps the search semantics FLUENT.md documents. The pair below
    # is the difference, on one pattern.
    def test_tilde_operator_matches_the_whole_value(self, graph):
        assert not graph.select("Person").where({"name": {"=~": "e$"}}).titles()
        assert sorted(graph.select("Person").where({"name": {"=~": ".*e$"}}).titles()) == [
            "Alice",
            "Charlie",
            "Eve",
        ]

    def test_tilde_operator_anchors_an_alternation_as_a_unit(self, graph):
        r = graph.select("Person").where({"name": {"=~": "Bob|Eve"}}).titles()
        assert sorted(r) == ["Bob", "Eve"]
        # `^Bob|Eve$` would also accept a name merely *starting* with Bob.
        assert not graph.select("Person").where({"name": {"=~": "Bo"}}).titles()

    def test_tilde_operator_invalid_pattern_quotes_the_users_pattern(self, graph):
        with pytest.raises(kglite.KgError, match=r"Invalid regex pattern '\[invalid'"):
            graph.select("Person").where({"name": {"=~": "[invalid"}})


# ============================================================================
# NOT logic in filter()
# ============================================================================


class TestNotFilter:
    def test_not_contains(self, graph):
        r = graph.select("Person").where({"city": {"not_contains": "erg"}}).titles()
        assert sorted(r) == ["Alice", "Charlie", "Diana"]

    def test_not_in(self, graph):
        r = graph.select("Person").where({"city": {"not_in": ["Oslo", "Bergen"]}}).titles()
        assert r == ["Diana"]

    def test_not_starts_with(self, graph):
        r = graph.select("Person").where({"name": {"not_starts_with": "A"}}).titles()
        assert sorted(r) == ["Bob", "Charlie", "Diana", "Eve"]

    def test_not_ends_with(self, graph):
        r = graph.select("Person").where({"name": {"not_ends_with": "e"}}).titles()
        assert sorted(r) == ["Bob", "Diana"]

    def test_not_regex(self, graph):
        r = graph.select("Person").where({"name": {"not_regex": "^[A-C].*"}}).titles()
        assert sorted(r) == ["Diana", "Eve"]


# ============================================================================
# offset() for pagination
# ============================================================================


class TestOffset:
    def test_offset_basic(self, graph):
        r = graph.select("Person").sort("name").offset(2).titles()
        assert r == ["Charlie", "Diana", "Eve"]

    def test_offset_with_max_nodes(self, graph):
        """offset + max_nodes = pagination."""
        r = graph.select("Person").sort("name").offset(1).limit(2).titles()
        assert r == ["Bob", "Charlie"]

    def test_offset_zero(self, graph):
        r = graph.select("Person").sort("name").offset(0).titles()
        assert len(r) == 5

    def test_offset_exceeds_count(self, graph):
        r = graph.select("Person").sort("name").offset(100).titles()
        assert not r  # empty selection


# ============================================================================
# filter_any() — OR logic
# ============================================================================


class TestFilterAny:
    def test_filter_any_basic(self, graph):
        r = (
            graph.select("Person")
            .where_any(
                [
                    {"city": "Oslo"},
                    {"city": "Bergen"},
                ]
            )
            .titles()
        )
        assert sorted(r) == ["Alice", "Bob", "Charlie", "Eve"]

    def test_filter_any_mixed_conditions(self, graph):
        r = (
            graph.select("Person")
            .where_any(
                [
                    {"age": {">=": 35}},
                    {"name": {"starts_with": "E"}},
                ]
            )
            .titles()
        )
        assert sorted(r) == ["Charlie", "Eve"]

    def test_filter_any_after_filter(self, graph):
        """filter_any can chain after filter (AND then OR)."""
        r = (
            graph.select("Person")
            .where({"age": {">=": 28}})
            .where_any(
                [
                    {"city": "Oslo"},
                    {"city": "Bergen"},
                ]
            )
            .titles()
        )
        assert sorted(r) == ["Alice", "Charlie", "Eve"]

    def test_filter_any_single_condition(self, graph):
        r = graph.select("Person").where_any([{"city": "Trondheim"}]).titles()
        assert r == ["Diana"]

    def test_filter_any_empty_raises(self, graph):
        with pytest.raises(kglite.KgError, match="at least one"):
            graph.select("Person").where_any([])


# ============================================================================
# count() and statistics() with group_by
# ============================================================================


class TestGroupBy:
    def test_count_group_by(self, graph):
        r = graph.select("Person").count(group_by="city")
        assert r == {"Oslo": 2, "Bergen": 2, "Trondheim": 1}

    def test_count_group_by_with_filter(self, graph):
        r = graph.select("Person").where({"age": {">=": 30}}).count(group_by="city")
        assert r == {"Oslo": 2, "Bergen": 1}

    def test_statistics_group_by(self, graph):
        r = graph.select("Person").statistics("age", group_by="city")
        assert set(r.keys()) == {"Oslo", "Bergen", "Trondheim"}
        assert r["Oslo"]["count"] == 2
        assert r["Oslo"]["mean"] == 32.5
        assert r["Bergen"]["count"] == 2
        assert r["Trondheim"]["count"] == 1

    def test_statistics_group_by_has_std(self, graph):
        r = graph.select("Person").statistics("age", group_by="city")
        assert "std" in r["Oslo"]
        assert "std" not in r["Trondheim"]  # only 1 value, no std


# ============================================================================
# has_connection() — existence filter
# ============================================================================


class TestHasConnection:
    def test_has_connection_any(self, graph):
        r = graph.select("Person").where_connected("KNOWS").titles()
        assert sorted(r) == ["Alice", "Bob", "Charlie"]

    def test_has_connection_outgoing(self, graph):
        r = graph.select("Person").where_connected("KNOWS", direction="outgoing").titles()
        assert r == ["Alice"]

    def test_has_connection_incoming(self, graph):
        r = graph.select("Person").where_connected("KNOWS", direction="incoming").titles()
        assert sorted(r) == ["Bob", "Charlie"]

    def test_has_connection_works_with(self, graph):
        r = graph.select("Person").where_connected("WORKS_WITH").titles()
        assert sorted(r) == ["Bob", "Diana"]

    def test_has_connection_nonexistent_type(self, graph):
        r = graph.select("Person").where_connected("DISLIKES").titles()
        assert not r  # empty selection

    def test_has_connection_invalid_direction(self, graph):
        with pytest.raises(kglite.KgError, match="Invalid direction"):
            graph.select("Person").where_connected("KNOWS", direction="sideways")

    def test_has_connection_chained(self, graph):
        """has_connection filters, then further filter works."""
        r = graph.select("Person").where_connected("KNOWS").where({"age": {">=": 30}}).titles()
        assert sorted(r) == ["Alice", "Charlie"]
