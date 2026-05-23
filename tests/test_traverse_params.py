"""Tests for traverse() parameter enhancements: target_type, where alias, where_connection alias."""

import pandas as pd
import pytest

import kglite


@pytest.fixture
def graph():
    """Graph with multiple node types connected by shared connection types."""
    g = kglite.KnowledgeGraph()

    # People
    df_people = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "title": ["Alice", "Bob", "Carol"],
        }
    )
    g.add_nodes(df_people, "Person", "id", "title")

    # Companies
    df_companies = pd.DataFrame(
        {
            "id": [10, 20],
            "title": ["Acme", "Globex"],
            "city": ["Oslo", "Bergen"],
        }
    )
    g.add_nodes(df_companies, "Company", "id", "title")

    # Schools
    df_schools = pd.DataFrame(
        {
            "id": [100, 200],
            "title": ["NTNU", "UiO"],
            "city": ["Trondheim", "Oslo"],
        }
    )
    g.add_nodes(df_schools, "School", "id", "title")

    # AFFILIATED_WITH connections — Person → Company and Person → School
    df_aff = pd.DataFrame(
        {
            "person_id": [1, 1, 2, 2, 3],
            "target_id": [10, 100, 20, 200, 10],
            "target_type": ["Company", "School", "Company", "School", "Company"],
            "role": ["employee", "student", "contractor", "alumnus", "employee"],
        }
    )
    # Person → Company
    g.add_connections(
        df_aff[df_aff["target_type"] == "Company"],
        "AFFILIATED_WITH",
        source_type="Person",
        source_id_field="person_id",
        target_type="Company",
        target_id_field="target_id",
        columns=["role"],
    )
    # Person → School
    g.add_connections(
        df_aff[df_aff["target_type"] == "School"],
        "AFFILIATED_WITH",
        source_type="Person",
        source_id_field="person_id",
        target_type="School",
        target_id_field="target_id",
        columns=["role"],
    )

    # Products (separate type so RATED edges don't collide with AFFILIATED_WITH)
    df_products = pd.DataFrame(
        {
            "id": [1000, 2000],
            "title": ["Widget", "Gadget"],
        }
    )
    g.add_nodes(df_products, "Product", "id", "title")

    # RATED connections with edge properties — Person → Product
    df_rated = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "product_id": [1000, 2000, 1000],
            "score": [5, 3, 1],
        }
    )
    g.add_connections(
        df_rated,
        "RATED",
        source_type="Person",
        source_id_field="person_id",
        target_type="Product",
        target_id_field="product_id",
        columns=["score"],
    )

    return g


class TestTargetType:
    """target_type parameter for traverse()."""

    def test_target_type_string(self, graph):
        """target_type='Company' filters to Company nodes only."""
        result = (
            graph.select("Person")
            .where({"title": "Alice"})
            .traverse("AFFILIATED_WITH", target_type="Company")
            .collect()
        )
        titles = [r["title"] for r in result]
        assert "Acme" in titles
        assert "NTNU" not in titles

    def test_target_type_string_school(self, graph):
        """target_type='School' filters to School nodes only."""
        result = (
            graph.select("Person").where({"title": "Alice"}).traverse("AFFILIATED_WITH", target_type="School").collect()
        )
        titles = [r["title"] for r in result]
        assert "NTNU" in titles
        assert "Acme" not in titles

    def test_target_type_list(self, graph):
        """target_type=['Company', 'School'] includes both types."""
        result = (
            graph.select("Person")
            .where({"title": "Alice"})
            .traverse("AFFILIATED_WITH", target_type=["Company", "School"])
            .collect()
        )
        titles = [r["title"] for r in result]
        assert "Acme" in titles
        assert "NTNU" in titles

    def test_target_type_no_match(self, graph):
        """target_type='NonExistent' returns empty result."""
        result = (
            graph.select("Person")
            .where({"title": "Alice"})
            .traverse("AFFILIATED_WITH", target_type="NonExistent")
            .collect()
        )
        assert len(result) == 0

    def test_target_type_none_returns_all(self, graph):
        """Without target_type, all target types are returned."""
        result = graph.select("Person").where({"title": "Alice"}).traverse("AFFILIATED_WITH").collect()
        titles = [r["title"] for r in result]
        assert "Acme" in titles
        assert "NTNU" in titles

    def test_target_type_with_where(self, graph):
        """target_type + where= combined filtering."""
        result = (
            graph.select("Person").traverse("AFFILIATED_WITH", target_type="Company", where={"city": "Oslo"}).collect()
        )
        titles = [r["title"] for r in result]
        assert "Acme" in titles
        assert "Globex" not in titles

    def test_target_type_incoming(self, graph):
        """target_type with direction='incoming'."""
        result = (
            graph.select("Company")
            .where({"title": "Acme"})
            .traverse("AFFILIATED_WITH", direction="incoming", target_type="Person")
            .collect()
        )
        titles = [r["title"] for r in result]
        assert "Alice" in titles
        assert "Carol" in titles


class TestWhereAlias:
    """where= alias for filter_target=."""

    def test_where_same_as_filter_target(self, graph):
        """where= produces same results as filter_target=."""
        r1 = graph.select("Person").traverse("AFFILIATED_WITH", filter_target={"city": "Oslo"}).collect()
        r2 = graph.select("Person").traverse("AFFILIATED_WITH", where={"city": "Oslo"}).collect()
        assert len(r1) == len(r2)
        titles1 = sorted([r["title"] for r in r1])
        titles2 = sorted([r["title"] for r in r2])
        assert titles1 == titles2

    def test_both_where_and_filter_target_raises(self, graph):
        """Using both where= and filter_target= raises ValueError."""
        with pytest.raises(kglite.KgError, match="alias"):
            graph.select("Person").traverse(
                "AFFILIATED_WITH",
                filter_target={"city": "Oslo"},
                where={"city": "Oslo"},
            )


class TestWhereConnectionAlias:
    """where_connection= alias for filter_connection=."""

    def test_where_connection_same_as_filter_connection(self, graph):
        """where_connection= produces same results as filter_connection=."""
        r1 = graph.select("Person").traverse("RATED", filter_connection={"score": {">": 3}}).collect()
        r2 = graph.select("Person").traverse("RATED", where_connection={"score": {">": 3}}).collect()
        assert len(r1) == len(r2)
        titles1 = sorted([r["title"] for r in r1])
        titles2 = sorted([r["title"] for r in r2])
        assert titles1 == titles2

    def test_both_where_connection_and_filter_connection_raises(self, graph):
        """Using both where_connection= and filter_connection= raises ValueError."""
        with pytest.raises(kglite.KgError, match="alias"):
            graph.select("Person").traverse(
                "RATED",
                filter_connection={"score": {">": 3}},
                where_connection={"score": {">": 3}},
            )

    def test_where_connection_filters_edges(self, graph):
        """where_connection= actually filters by edge properties."""
        result = graph.select("Person").traverse("RATED", where_connection={"score": {">": 3}}).collect()
        # Only Alice (score=5) passes score > 3
        titles = [r["title"] for r in result]
        assert "Widget" in titles
        # Bob's rating of Gadget (score=3) doesn't pass > 3
        assert len(result) == 1


class TestTargetTypeEdgeCases:
    """Edge cases for target_type."""

    def test_target_type_empty_list_same_as_none(self, graph):
        """target_type=[] treated as no filter."""
        r1 = graph.select("Person").where({"title": "Alice"}).traverse("AFFILIATED_WITH").collect()
        r2 = graph.select("Person").where({"title": "Alice"}).traverse("AFFILIATED_WITH", target_type=[]).collect()
        assert len(r1) == len(r2)

    def test_target_type_invalid_type_raises(self, graph):
        """target_type=123 raises typed kglite.KgError (was TypeError pre-A.2)."""
        with pytest.raises(kglite.KgError, match="string or list"):
            graph.select("Person").traverse("AFFILIATED_WITH", target_type=123)
