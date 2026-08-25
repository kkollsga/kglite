"""Tests for range indexes / B-Tree indexes (Phase 4).

Range indexes enable efficient range queries (>, >=, <, <=, BETWEEN)
using BTreeMap under the hood.
"""

import os
import tempfile

import pandas as pd
import pytest

import kglite


@pytest.fixture
def graph():
    """Graph with Person nodes having age property."""
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        {
            "id": list(range(100)),
            "title": [f"Person_{i}" for i in range(100)],
            "age": list(range(20, 120)),  # ages 20-119
        }
    )
    g.add_nodes(df, "Person", "id", "title")
    return g


class TestCreateDropRangeIndex:
    """Create and drop range indexes."""

    def test_create_range_index(self, graph):
        result = graph.create_range_index("Person", "age")
        assert result["created"] is True
        assert result["unique_values"] == 100  # 100 distinct ages

    def test_create_range_index_no_matching_nodes(self, graph):
        result = graph.create_range_index("NonExistent", "age")
        assert result["created"] is True
        assert result["unique_values"] == 0

    def test_drop_range_index(self, graph):
        graph.create_range_index("Person", "age")
        assert graph.drop_range_index("Person", "age") is True

    def test_drop_nonexistent_range_index(self, graph):
        assert graph.drop_range_index("Person", "age") is False


class TestRangeIndexQueries:
    """Range queries using filter() with range conditions."""

    def test_greater_than(self, graph):
        graph.create_range_index("Person", "age")
        result = graph.where({"type": "Person"}).where({"age": {">": 110}}).collect()
        ages = sorted(n["age"] for n in result)
        assert all(a > 110 for a in ages)
        assert len(result) == 9  # ages 111-119

    def test_greater_than_equals(self, graph):
        graph.create_range_index("Person", "age")
        result = graph.where({"type": "Person"}).where({"age": {">=": 115}}).collect()
        ages = sorted(n["age"] for n in result)
        assert all(a >= 115 for a in ages)
        assert len(result) == 5  # ages 115-119

    def test_less_than(self, graph):
        graph.create_range_index("Person", "age")
        result = graph.where({"type": "Person"}).where({"age": {"<": 25}}).collect()
        ages = sorted(n["age"] for n in result)
        assert all(a < 25 for a in ages)
        assert len(result) == 5  # ages 20-24

    def test_less_than_equals(self, graph):
        graph.create_range_index("Person", "age")
        result = graph.where({"type": "Person"}).where({"age": {"<=": 22}}).collect()
        ages = sorted(n["age"] for n in result)
        assert all(a <= 22 for a in ages)
        assert len(result) == 3  # ages 20-22

    def test_between(self, graph):
        graph.create_range_index("Person", "age")
        result = graph.where({"type": "Person"}).where({"age": {"between": [50, 55]}}).collect()
        ages = sorted(n["age"] for n in result)
        assert all(50 <= a <= 55 for a in ages)
        assert len(result) == 6  # ages 50-55

    def test_two_sided_range_in_one_dict(self, graph):
        """Both operators of a one-dict range apply, index present or not."""
        graph.create_range_index("Person", "age")
        indexed = graph.where({"type": "Person"}).where({"age": {">=": 50, "<=": 55}}).collect()
        ages = sorted(n["age"] for n in indexed)
        assert ages == [50, 51, 52, 53, 54, 55]

        graph.drop_range_index("Person", "age")
        scanned = graph.where({"type": "Person"}).where({"age": {">=": 50, "<=": 55}}).collect()
        assert sorted(n["age"] for n in scanned) == ages

    def test_range_query_without_index_still_works(self, graph):
        """Range queries should work without an index (fallback to scan)."""
        result = graph.where({"type": "Person"}).where({"age": {">": 110}}).collect()
        assert len(result) == 9


class TestRangeIndexMaintenance:
    """Range indexes should stay up-to-date with mutations."""

    def test_index_updated_on_create(self, graph):
        graph.create_range_index("Person", "age")
        graph.cypher("CREATE (n:Person {id: 200, title: 'New', age: 200})")

        result = graph.where({"type": "Person"}).where({"age": {">": 150}}).collect()
        assert len(result) == 1
        assert result[0]["age"] == 200

    def test_index_updated_on_set(self, graph):
        graph.create_range_index("Person", "age")
        graph.cypher("MATCH (n:Person) WHERE n.title = 'Person_0' SET n.age = 999")

        result = graph.where({"type": "Person"}).where({"age": {">": 500}}).collect()
        assert len(result) == 1
        assert result[0]["age"] == 999

    def test_index_updated_on_delete(self, graph):
        graph.create_range_index("Person", "age")
        graph.set_auto_vacuum(None)  # disable to avoid index rebuild

        before = graph.where({"type": "Person"}).where({"age": {"<=": 25}}).collect()
        graph.cypher("MATCH (n:Person) WHERE n.age = 20 DETACH DELETE n")
        after = graph.where({"type": "Person"}).where({"age": {"<=": 25}}).collect()

        assert len(after) == len(before) - 1


class TestRangeIndexPersistence:
    """Range indexes should persist through save/load."""

    def test_range_index_survives_save_load(self, graph):
        graph.create_range_index("Person", "age")

        with tempfile.NamedTemporaryFile(suffix=".kglite", delete=False) as f:
            path = f.name

        try:
            graph.save(path)
            g2 = kglite.load(path)

            # Range index should work after reload
            result = g2.where({"type": "Person"}).where({"age": {">": 110}}).collect()
            assert len(result) == 9
        finally:
            os.unlink(path)


class TestRangeIndexInSchema:
    """Range indexes should appear in schema() output."""

    def test_range_index_in_schema(self, graph):
        graph.create_range_index("Person", "age")
        schema = graph.schema()
        # Range indexes should be listed
        assert any("[range]" in idx for idx in schema["indexes"])
