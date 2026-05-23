"""Tests for edge cases: empty graphs, nulls, special characters, boundary values."""

import os
import tempfile

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph


class TestEmptyGraph:
    def test_empty_graph_type_filter(self):
        graph = KnowledgeGraph()
        result = graph.select("NonExistent")
        assert result.len() == 0

    def test_empty_graph_get_nodes(self):
        graph = KnowledgeGraph()
        result = graph.select("Any")
        assert len(result.collect()) == 0

    def test_empty_graph_schema(self):
        graph = KnowledgeGraph()
        schema = graph.schema_text()
        assert isinstance(schema, str)

    def test_empty_graph_connected_components(self):
        graph = KnowledgeGraph()
        components = graph.connected_components()
        assert len(components) == 0


class TestSingleNode:
    def test_single_node_operations(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["Solo"]})
        graph.add_nodes(df, "Node", "id", "name")
        assert graph.select("Node").len() == 1
        nodes = graph.select("Node").collect()
        assert nodes[0]["title"] == "Solo"


class TestSpecialCharacters:
    def test_special_chars_in_properties(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1],
                "name": ["O'Brien & Co."],
                "desc": ['Has "quotes" and <brackets>'],
            }
        )
        graph.add_nodes(df, "Node", "id", "name")
        node = graph.select("Node").collect()[0]
        assert node["title"] == "O'Brien & Co."
        assert '"quotes"' in node["desc"]

    def test_unicode_in_properties(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1],
                "name": ["Test"],
                "content": ["Hello World"],
            }
        )
        graph.add_nodes(df, "Node", "id", "name")
        node = graph.select("Node").collect()[0]
        assert node["content"] is not None

    def test_very_long_strings(self):
        graph = KnowledgeGraph()
        long_str = "x" * 10000
        df = pd.DataFrame({"id": [1], "name": ["LongNode"], "data": [long_str]})
        graph.add_nodes(df, "Node", "id", "name")
        node = graph.select("Node").collect()[0]
        assert len(node["data"]) == 10000


class TestNumericEdgeCases:
    def test_zero_values(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"], "value": [0]})
        graph.add_nodes(df, "Node", "id", "name")
        result = graph.select("Node").where({"value": 0})
        assert result.len() == 1

    def test_negative_values(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"], "value": [-100]})
        graph.add_nodes(df, "Node", "id", "name")
        result = graph.select("Node").where({"value": {"<": 0}})
        assert result.len() == 1

    def test_float_precision(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"], "value": [3.14159265358979]})
        graph.add_nodes(df, "Node", "id", "name")
        node = graph.select("Node").collect()[0]
        assert abs(node["value"] - 3.14159265358979) < 1e-10


class TestManyNodeTypes:
    def test_many_types(self):
        graph = KnowledgeGraph()
        for i in range(20):
            df = pd.DataFrame({"id": [1], "name": [f"Node_{i}"]})
            graph.add_nodes(df, f"Type_{i}", "id", "name")

        for i in range(20):
            assert graph.select(f"Type_{i}").len() == 1


class TestDeepTraversal:
    def test_deep_chain(self):
        graph = KnowledgeGraph()
        # Create a chain: A -> B -> C -> D -> E
        for i in range(5):
            df = pd.DataFrame({"id": [i], "name": [f"Node_{i}"]})
            graph.add_nodes(df, "Chain", "id", "name")
        for i in range(4):
            conn = pd.DataFrame({"from_id": [i], "to_id": [i + 1]})
            graph.add_connections(conn, "NEXT", "Chain", "from_id", "Chain", "to_id")

        # Traverse 4 hops from start — each hop accumulates all reachable nodes
        start = graph.select("Chain").where({"id": 0})
        hop1 = start.traverse("NEXT")
        assert hop1.len() >= 1
        hop2 = hop1.traverse("NEXT")
        assert hop2.len() >= 1
        hop3 = hop2.traverse("NEXT")
        assert hop3.len() >= 1
        hop4 = hop3.traverse("NEXT")
        assert hop4.len() >= 1


# ====================================================================
# Migrated from pytest/test_edge_cases.py
# ====================================================================


class TestEmptyInputs:
    """Test behavior with empty inputs."""

    def test_empty_dataframe_add_nodes(self):
        """Test adding empty DataFrame."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [], "name": []})
        report = graph.add_nodes(df, "EmptyType", "id", "name")
        assert report["nodes_created"] == 0

    def test_empty_dataframe_add_connections(self):
        """Test adding empty connections DataFrame."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"]})
        graph.add_nodes(df, "Node", "id", "name")

        conn_df = pd.DataFrame({"source": [], "target": []})
        report = graph.add_connections(conn_df, "LINKS", "Node", "source", "Node", "target")
        assert report["connections_created"] == 0


class TestNullHandling:
    """Test null/None value handling."""

    def test_null_values_in_dataframe(self):
        """Test handling of None/NaN values."""
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["A", "B", "C"],
                "optional_field": ["value", None, "other"],
            }
        )
        graph.add_nodes(df, "Node", "id", "name")

        result = graph.select("Node").where({"optional_field": {"is_null": True}})
        assert result.count() == 1

        result = graph.select("Node").where({"optional_field": {"is_not_null": True}})
        assert result.count() == 2

    def test_filter_on_missing_property(self):
        """Test filtering on property that doesn't exist."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "name")

        result = graph.select("Node").where({"nonexistent": "value"})
        assert result.count() == 0


class TestBoundaryConditions:
    """Test boundary conditions."""

    def test_self_referential_connection(self):
        """Test connection from node to itself."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["Self"]})
        graph.add_nodes(df, "Node", "id", "name")

        conn_df = pd.DataFrame({"source": [1], "target": [1]})
        report = graph.add_connections(conn_df, "SELF_REF", "Node", "source", "Node", "target")
        assert report["connections_created"] == 1

    def test_duplicate_connections(self):
        """Test adding duplicate connections."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "name")

        conn_df = pd.DataFrame({"source": [1, 1], "target": [2, 2]})
        report = graph.add_connections(conn_df, "LINK", "Node", "source", "Node", "target")
        assert report["connections_created"] == 2


class TestSetOperationsEdgeCases:
    """Test edge cases in set operations."""

    def test_union_with_self(self):
        """Test union of selection with itself."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        graph.add_nodes(df, "Node", "id", "name")

        selection = graph.select("Node")
        result = selection.union(selection)
        assert result.count() == 3  # No duplicates

    def test_union_with_empty(self):
        """Test union with empty selection."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "name")

        all_nodes = graph.select("Node")
        empty = graph.select("NonExistent")

        result = all_nodes.union(empty)
        assert result.count() == 2

    def test_intersection_with_empty(self):
        """Test intersection with empty selection."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "name")

        all_nodes = graph.select("Node")
        empty = graph.select("NonExistent")

        result = all_nodes.intersection(empty)
        assert result.count() == 0

    def test_difference_with_self(self):
        """Test difference of selection with itself."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        graph.add_nodes(df, "Node", "id", "name")

        selection = graph.select("Node")
        result = selection.difference(selection)
        assert result.count() == 0  # A - A = empty


class TestSaveLoadEdgeCases:
    """Test edge cases in save/load operations."""

    def test_save_empty_graph(self):
        """Test saving empty graph."""
        graph = KnowledgeGraph()

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name

        try:
            graph.save(path)
            loaded = kglite.load(path)
            assert loaded.select("AnyType").count() == 0
        finally:
            os.unlink(path)

    def test_save_load_preserves_all_data(self):
        """Test that save/load preserves all node and connection properties."""
        graph = KnowledgeGraph()

        df = pd.DataFrame(
            {
                "id": [1, 2],
                "name": ["Node1", "Node2"],
                "int_prop": [100, 200],
                "float_prop": [1.5, 2.5],
                "str_prop": ["hello", "world"],
            }
        )
        graph.add_nodes(df, "Test", "id", "name")

        conn_df = pd.DataFrame({"source": [1], "target": [2], "weight": [0.75]})
        graph.add_connections(
            conn_df,
            "RELATES",
            "Test",
            "source",
            "Test",
            "target",
            columns=["weight"],
        )

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name

        try:
            graph.save(path)
            loaded = kglite.load(path)

            nodes = loaded.select("Test").collect().to_df()
            assert len(nodes) == 2

            node1 = nodes[nodes["id"] == 1].iloc[0]
            assert node1["int_prop"] == 100
            assert node1["float_prop"] == 1.5
            assert node1["str_prop"] == "hello"
        finally:
            os.unlink(path)


class TestTraversalEdgeCases:
    """Test edge cases in graph traversal."""

    def test_traverse_no_connections(self):
        """Test traversing when no connections of that type exist raises ValueError."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "name")

        with pytest.raises(kglite.KgError):
            graph.select("Node").traverse("NONEXISTENT")

    def test_traverse_with_existing_connection_type(self):
        """Test traversal with existing connection type but no matches from selection."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2, 3, 4], "name": ["A", "B", "C", "D"]})
        graph.add_nodes(df, "Node", "id", "name")

        conn_df = pd.DataFrame({"source": [1], "target": [2]})
        graph.add_connections(conn_df, "LINK", "Node", "source", "Node", "target")

        result = graph.select("Node").where({"title": "D"}).traverse("LINK")
        count = result.count()
        total = sum(count.values()) if isinstance(count, dict) else count
        assert total == 0


class TestPathFindingEdgeCases:
    """Test edge cases in path finding."""

    def test_shortest_path_same_node(self):
        """Test shortest path from node to itself."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"]})
        graph.add_nodes(df, "Node", "id", "name")

        result = graph.shortest_path("Node", 1, "Node", 1)
        assert result is not None
        assert len(result["path"]) == 1
        assert result["length"] == 0

    def test_shortest_path_no_path(self):
        """Test shortest path when no path exists."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        graph.add_nodes(df, "Node", "id", "name")

        result = graph.shortest_path("Node", 1, "Node", 2)
        assert result is None


class TestCalculationsEdgeCases:
    """Test edge cases in calculations."""

    def test_calculate_empty_selection(self):
        """Test calculation on empty selection raises ValueError."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"], "value": [10, 20]})
        graph.add_nodes(df, "Node", "id", "name")

        result = graph.select("Node").where({"name": "NonExistent"})

        with pytest.raises(kglite.KgError):
            result.calculate("sum(value)")

    def test_calculate_with_null_values(self):
        """Test calculation with null values in data."""
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["A", "B", "C"],
                "value": [10.0, None, 30.0],
            }
        )
        graph.add_nodes(df, "Node", "id", "name")

        result = graph.select("Node").calculate("sum(value)")
        if isinstance(result, dict):
            actual_value = list(result.values())[0] if result else 0
            assert actual_value == 40.0
        else:
            assert result == 40.0


class TestBatchUpdateEdgeCases:
    """Test edge cases in batch updates."""

    def test_update_empty_selection(self):
        """Test updating empty selection raises ValueError."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"]})
        graph.add_nodes(df, "Node", "id", "name")

        selection = graph.select("Node").where({"name": "NonExistent"})

        with pytest.raises(kglite.KgError):
            selection.update({"new_prop": "value"})

    def test_update_existing_property(self):
        """Test updating existing property via update (returns dict with graph)."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"], "category": ["old", "old"]})
        graph.add_nodes(df, "Node", "id", "name")

        selection = graph.select("Node")
        result = selection.update({"category": "new"}, keep_selection=False)

        assert "graph" in result
        assert result["nodes_updated"] == 2
        updated_graph = result["graph"]

        nodes = updated_graph.select("Node").collect().to_df()
        for _, node in nodes.iterrows():
            assert node.get("category") == "new"

        original_nodes = graph.select("Node").collect().to_df()
        for _, node in original_nodes.iterrows():
            assert node.get("category") == "old"

    def test_update_only_selected_nodes(self):
        """Test that update only affects selected nodes."""
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["A", "B", "C"],
                "status": ["active", "inactive", "active"],
                "processed": [False, False, False],
            }
        )
        graph.add_nodes(df, "Node", "id", "name")

        selection = graph.select("Node").where({"status": "active"})
        result = selection.update({"processed": True}, keep_selection=False)

        assert result["nodes_updated"] == 2
        updated_graph = result["graph"]

        nodes = updated_graph.select("Node").collect().to_df()
        active_processed = nodes[nodes["status"] == "active"]["processed"].tolist()
        inactive_processed = nodes[nodes["status"] == "inactive"]["processed"].tolist()

        assert all(p is True for p in active_processed)
        assert all(p is False for p in inactive_processed)


class TestExportEdgeCases:
    """Test edge cases in export functionality."""

    def test_export_empty_graph(self):
        """Test exporting empty graph."""
        graph = KnowledgeGraph()

        with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
            path = f.name

        try:
            graph.export(path, "graphml")
            assert os.path.exists(path)
            with open(path, "r") as f:
                content = f.read()
                assert "<?xml" in content
                assert "graphml" in content
        finally:
            os.unlink(path)


class TestSchemaValidationEdgeCases:
    """Test edge cases in schema validation."""

    def test_validate_empty_schema(self):
        """Test validating with empty schema."""
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"]})
        graph.add_nodes(df, "Node", "id", "name")

        graph.define_schema({"nodes": {}, "connections": {}})

        errors = graph.validate_schema()
        assert len(errors) == 0

    def test_validate_missing_required_field(self):
        """Test validation catches missing required field."""
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "name": ["A", "B"],
                "required_field": ["value", None],
            }
        )
        graph.add_nodes(df, "Node", "id", "name")

        graph.define_schema(
            {
                "nodes": {"Node": {"required": ["required_field"]}},
                "connections": {},
            }
        )

        errors = graph.validate_schema()
        assert len(errors) == 1


class TestQueryExplainEdgeCases:
    """Test edge cases in query explain."""

    def test_explain_empty_query(self):
        """Test explain on empty selection."""
        graph = KnowledgeGraph()

        result = graph.select("NonExistent")
        explain = result.explain()
        assert explain is not None
        assert "SELECT" in explain or "select" in explain.lower()
