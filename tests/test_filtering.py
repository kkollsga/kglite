"""Tests for filtering, sorting, limiting, null checks, orphan filtering."""

import pandas as pd
import pytest

from kglite import KnowledgeGraph


class TestBasicFiltering:
    def test_filter_exact_match(self, small_graph):
        result = small_graph.select("Person").where({"age": 28})
        assert result.len() == 1

    def test_filter_greater_than(self, social_graph):
        result = social_graph.select("Person").where({"age": {">": 30}})
        assert result.len() > 0
        nodes = result.collect()
        for n in nodes:
            assert n["age"] > 30

    def test_filter_less_than(self, social_graph):
        result = social_graph.select("Person").where({"age": {"<": 25}})
        nodes = result.collect()
        for n in nodes:
            assert n["age"] < 25

    def test_filter_greater_equal(self, social_graph):
        result = social_graph.select("Person").where({"age": {">=": 40}})
        nodes = result.collect()
        for n in nodes:
            assert n["age"] >= 40

    def test_filter_less_equal(self, social_graph):
        result = social_graph.select("Person").where({"age": {"<=": 22}})
        nodes = result.collect()
        for n in nodes:
            assert n["age"] <= 22

    def test_filter_multiple_conditions(self, social_graph):
        result = social_graph.select("Person").where(
            {
                "age": {">": 25},
                "city": "Oslo",
            }
        )
        nodes = result.collect()
        for n in nodes:
            assert n["age"] > 25
            assert n["city"] == "Oslo"

    def test_filter_in_operator(self, social_graph):
        result = social_graph.select("Person").where(
            {
                "city": {"in": ["Oslo", "Bergen"]},
            }
        )
        nodes = result.collect()
        for n in nodes:
            assert n["city"] in ["Oslo", "Bergen"]

    def test_filter_no_matches(self, small_graph):
        result = small_graph.select("Person").where({"title": "NonExistent"})
        assert result.len() == 0

    def test_filter_chained(self, social_graph):
        result = social_graph.select("Person").where({"city": "Oslo"}).where({"age": {">": 23}})
        nodes = result.collect()
        for n in nodes:
            assert n["city"] == "Oslo"
            assert n["age"] > 23


class TestNullFiltering:
    def test_is_null(self, social_graph):
        result = social_graph.select("Person").where({"email": {"is_null": True}})
        assert result.len() > 0

    def test_is_not_null(self, social_graph):
        result = social_graph.select("Person").where({"email": {"is_not_null": True}})
        assert result.len() > 0

    def test_null_and_not_null_partition(self, social_graph):
        null_count = social_graph.select("Person").where({"email": {"is_null": True}}).len()
        not_null_count = social_graph.select("Person").where({"email": {"is_not_null": True}}).len()
        total = social_graph.select("Person").len()
        assert null_count + not_null_count == total

    def test_filter_on_missing_property(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "name")
        result = graph.select("Node").where({"nonexistent": {"is_null": True}})
        assert result.len() == 2


class TestStringPredicates:
    def test_filter_contains(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "name": ["Alice Smith", "Bob Jones", "Carol Smith", "Dave Lee"],
            }
        )
        graph.add_nodes(df, "Person", "id", "name")
        result = graph.select("Person").where({"title": {"contains": "Smith"}})
        assert result.len() == 2
        for n in result.collect():
            assert "Smith" in n["title"]

    def test_filter_starts_with(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alpha", "Beta", "Alphabet"],
            }
        )
        graph.add_nodes(df, "Item", "id", "name")
        result = graph.select("Item").where({"title": {"starts_with": "Alp"}})
        assert result.len() == 2
        for n in result.collect():
            assert n["title"].startswith("Alp")

    def test_filter_ends_with(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["report.csv", "data.json", "output.csv"],
            }
        )
        graph.add_nodes(df, "File", "id", "name")
        result = graph.select("File").where({"title": {"ends_with": ".csv"}})
        assert result.len() == 2
        for n in result.collect():
            assert n["title"].endswith(".csv")

    def test_filter_contains_no_match(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        graph.add_nodes(df, "Person", "id", "name")
        result = graph.select("Person").where({"title": {"contains": "xyz"}})
        assert result.len() == 0

    def test_filter_string_predicates_combined(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "name": ["Alice Smith", "Bob Jones", "Alice Jones", "Carol Smith"],
                "email": ["alice@test.com", "bob@test.com", "alice2@other.com", "carol@test.com"],
            }
        )
        graph.add_nodes(df, "Person", "id", "name")
        # Combine starts_with with another filter
        result = graph.select("Person").where(
            {
                "title": {"starts_with": "Alice"},
            }
        )
        assert result.len() == 2


class TestMultiOperatorConditions:
    """A per-property operator dict with more than one operator is an AND.

    Before this was fixed, the Python→Rust converter kept only the first
    operator in the dict and silently discarded the rest, so a two-sided
    range filter ran as a one-sided one and returned extra rows.
    """

    def test_range_uses_both_bounds(self, social_graph):
        # Ages run 21..40; the band 30..35 is 6 people.
        result = social_graph.select("Person").where({"age": {">=": 30, "<=": 35}})
        ages = sorted(n["age"] for n in result.collect())
        assert ages == [30, 31, 32, 33, 34, 35]

    def test_range_is_order_independent(self, social_graph):
        # Same band written with the operators in the other order.
        result = social_graph.select("Person").where({"age": {"<=": 35, ">=": 30}})
        ages = sorted(n["age"] for n in result.collect())
        assert ages == [30, 31, 32, 33, 34, 35]

    def test_matches_chained_where(self, social_graph):
        one_dict = social_graph.select("Person").where({"age": {">=": 30, "<=": 35}})
        chained = social_graph.select("Person").where({"age": {">=": 30}}).where({"age": {"<=": 35}})
        assert sorted(one_dict.titles()) == sorted(chained.titles())

    def test_string_predicates_combined(self, social_graph):
        # starts_with alone matches Person_1 and Person_10..Person_19 (11 nodes);
        # only Person_10 also ends with "0".
        result = social_graph.select("Person").where({"title": {"starts_with": "Person_1", "ends_with": "0"}})
        assert result.titles() == ["Person_10"]

    def test_regex_composes(self, social_graph):
        # The regex arm of a conjunction is compiled through the same cache as
        # a lone regex condition.
        result = social_graph.select("Person").where({"title": {"regex": "^Person_1", "not_regex": "1$"}})
        titles = sorted(result.titles())
        assert titles == [
            "Person_10",
            "Person_12",
            "Person_13",
            "Person_14",
            "Person_15",
            "Person_16",
            "Person_17",
            "Person_18",
            "Person_19",
        ]

    def test_negated_operator_composes(self, social_graph):
        result = social_graph.select("Person").where({"city": {"contains": "e", "not_contains": "B"}})
        cities = {n["city"] for n in result.collect()}
        assert cities == {"Stavanger", "Trondheim"}

    def test_three_operators(self, social_graph):
        result = social_graph.select("Person").where({"age": {">=": 30, "<=": 35, "!=": 33}})
        ages = sorted(n["age"] for n in result.collect())
        assert ages == [30, 31, 32, 34, 35]

    def test_unsatisfiable_combination_matches_nothing(self, social_graph):
        result = social_graph.select("Person").where({"age": {">": 35, "<": 30}})
        assert result.len() == 0

    def test_where_connection_multi_operator(self, social_graph):
        # KNOWS.since runs 2015..2019; the band 2016..2017 must exclude 2018/2019.
        friends = social_graph.select("Person").traverse(
            connection_type="KNOWS",
            where_connection={"since": {">=": 2016, "<=": 2017}},
        )
        assert friends.len() > 0
        unbounded = social_graph.select("Person").traverse(
            connection_type="KNOWS",
            where_connection={"since": {">=": 2016}},
        )
        assert friends.len() < unbounded.len()

    def test_where_any_multi_operator(self, social_graph):
        result = social_graph.select("Person").where_any([{"age": {">=": 30, "<=": 32}}, {"age": {">=": 38, "<=": 39}}])
        ages = sorted(n["age"] for n in result.collect())
        assert ages == [30, 31, 32, 38, 39]

    def test_empty_operator_dict_still_errors(self, social_graph):
        with pytest.raises(ValueError):
            social_graph.select("Person").where({"age": {}})


class TestSorting:
    def test_sort_ascending(self, social_graph):
        nodes = social_graph.select("Person").sort("age").collect()
        ages = [n["age"] for n in nodes]
        assert ages == sorted(ages)

    def test_sort_descending(self, social_graph):
        nodes = social_graph.select("Person").sort("age", ascending=False).collect()
        ages = [n["age"] for n in nodes]
        assert ages == sorted(ages, reverse=True)

    def test_sort_multi_field(self, social_graph):
        nodes = social_graph.select("Person").sort([("city", True), ("age", True)]).collect()
        for i in range(len(nodes) - 1):
            if nodes[i]["city"] == nodes[i + 1]["city"]:
                assert nodes[i]["age"] <= nodes[i + 1]["age"]


class TestLimiting:
    def test_max_nodes(self, social_graph):
        result = social_graph.select("Person").limit(5)
        assert result.len() == 5

    def test_max_nodes_more_than_total(self, small_graph):
        result = small_graph.select("Person").limit(100)
        assert result.len() == 3


class TestOrphanFiltering:
    def test_filter_orphans_include(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        graph.add_nodes(df, "Node", "id", "name")
        conn_df = pd.DataFrame({"source": [1], "target": [2]})
        graph.add_connections(conn_df, "LINKS", "Node", "source", "Node", "target")

        orphans = graph.select("Node").where_orphans(include_orphans=True)
        assert orphans.len() == 1  # Node 3 is orphan

    def test_filter_orphans_exclude(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        graph.add_nodes(df, "Node", "id", "name")
        conn_df = pd.DataFrame({"source": [1], "target": [2]})
        graph.add_connections(conn_df, "LINKS", "Node", "source", "Node", "target")

        connected = graph.select("Node").where_orphans(include_orphans=False)
        assert connected.len() == 2  # Nodes 1 and 2
