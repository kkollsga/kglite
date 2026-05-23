"""
Tests for the Connector API (bulk loading from data sources).

This tests the standardized interface for loading nodes and connections
from external data sources like factpages_py.
"""

import pandas as pd
import pytest

from kglite import KnowledgeGraph


class TestNodeTypesProperty:
    """Tests for the node_types property."""

    def test_empty_graph_has_no_types(self):
        """Empty graph should have no node types."""
        graph = KnowledgeGraph()
        assert graph.node_types == []

    def test_node_types_after_adding_nodes(self):
        """node_types should return all loaded types."""
        graph = KnowledgeGraph()

        df1 = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        graph.add_nodes(df1, "Person", "id", "name")

        df2 = pd.DataFrame({"id": [100], "name": ["Corp"]})
        graph.add_nodes(df2, "Company", "id", "name")

        types = set(graph.node_types)
        assert types == {"Person", "Company"}

    def test_node_types_excludes_schema_nodes(self):
        """node_types should not include internal SchemaNode type."""
        graph = KnowledgeGraph()

        df = pd.DataFrame({"id": [1], "name": ["Test"]})
        graph.add_nodes(df, "TestType", "id", "name")

        # SchemaNode is used internally but should not appear
        assert "SchemaNode" not in graph.node_types
        assert "TestType" in graph.node_types


class TestAddNodesBulk:
    """Tests for add_nodes_bulk method."""

    def test_add_multiple_node_types(self):
        """Should add multiple node types in one call."""
        graph = KnowledgeGraph()

        node_specs = [
            {
                "node_type": "Field",
                "unique_id_field": "field_id",
                "node_title_field": "name",
                "data": pd.DataFrame({"field_id": [1, 2, 3], "name": ["Ekofisk", "Troll", "Snorre"]}),
            },
            {
                "node_type": "Wellbore",
                "unique_id_field": "wb_id",
                "node_title_field": "name",
                "data": pd.DataFrame({"wb_id": [10, 20], "name": ["Well-1", "Well-2"]}),
            },
        ]

        stats = graph.add_nodes_bulk(node_specs)

        assert stats["Field"] == 3
        assert stats["Wellbore"] == 2
        assert set(graph.node_types) == {"Field", "Wellbore"}

    def test_missing_required_field_raises_error(self):
        """Should raise error if node spec is missing required fields."""
        graph = KnowledgeGraph()

        # Missing 'node_type'
        bad_spec = [
            {"unique_id_field": "id", "node_title_field": "name", "data": pd.DataFrame({"id": [1], "name": ["Test"]})}
        ]

        with pytest.raises(KeyError):
            graph.add_nodes_bulk(bad_spec)


class TestAddConnectionsBulk:
    """Tests for add_connections_bulk method."""

    def test_add_multiple_connection_types(self):
        """Should add multiple connection types in one call."""
        graph = KnowledgeGraph()

        # Add nodes first
        graph.add_nodes(pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]}), "Person", "id", "name")
        graph.add_nodes(pd.DataFrame({"id": [100, 200], "name": ["Corp1", "Corp2"]}), "Company", "id", "name")

        conn_specs = [
            {
                "source_type": "Person",
                "target_type": "Person",
                "connection_name": "KNOWS",
                "data": pd.DataFrame({"source_id": [1, 2], "target_id": [2, 3]}),
            },
            {
                "source_type": "Person",
                "target_type": "Company",
                "connection_name": "WORKS_AT",
                "data": pd.DataFrame({"source_id": [1, 2, 3], "target_id": [100, 100, 200]}),
            },
        ]

        stats = graph.add_connections_bulk(conn_specs)

        assert stats["KNOWS"] == 2
        assert stats["WORKS_AT"] == 3

    def test_requires_source_id_column(self):
        """Should raise error if source_id column is missing."""
        graph = KnowledgeGraph()
        graph.add_nodes(pd.DataFrame({"id": [1], "name": ["A"]}), "Person", "id", "name")

        bad_conn = [
            {
                "source_type": "Person",
                "target_type": "Person",
                "connection_name": "KNOWS",
                "data": pd.DataFrame(
                    {
                        "wrong_source": [1],  # Wrong column name
                        "target_id": [1],
                    }
                ),
            }
        ]

        # Phase A.2 / C6 — typed kglite.KgError (was ValueError pre-A.2).
        import kglite

        with pytest.raises(kglite.KgError) as exc_info:
            graph.add_connections_bulk(bad_conn)
        assert "source_id" in str(exc_info.value)


class TestAddConnectionsFromSource:
    """Tests for add_connections_from_source method (auto-filter)."""

    def test_filters_to_loaded_types(self):
        """Should only add connections where both types are loaded."""
        graph = KnowledgeGraph()

        # Only load Person, NOT Company
        graph.add_nodes(pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]}), "Person", "id", "name")

        all_connections = [
            {
                "source_type": "Person",
                "target_type": "Person",
                "connection_name": "KNOWS",
                "data": pd.DataFrame({"source_id": [1, 2], "target_id": [2, 3]}),
            },
            {
                "source_type": "Person",
                "target_type": "Company",  # Company NOT loaded
                "connection_name": "WORKS_AT",
                "data": pd.DataFrame({"source_id": [1, 2, 3], "target_id": [100, 100, 200]}),
            },
            {
                "source_type": "Company",  # Company NOT loaded
                "target_type": "Company",
                "connection_name": "PARTNERS_WITH",
                "data": pd.DataFrame({"source_id": [100], "target_id": [200]}),
            },
        ]

        stats = graph.add_connections_from_source(all_connections)

        # Only KNOWS should be loaded (Person->Person)
        assert "KNOWS" in stats
        assert stats["KNOWS"] == 2

        # WORKS_AT and PARTNERS_WITH should be filtered out
        assert "WORKS_AT" not in stats
        assert "PARTNERS_WITH" not in stats

    def test_all_connections_loaded_when_types_exist(self):
        """All connections should load if all types exist."""
        graph = KnowledgeGraph()

        # Load both types
        graph.add_nodes(pd.DataFrame({"id": [1, 2], "name": ["A", "B"]}), "Person", "id", "name")
        graph.add_nodes(pd.DataFrame({"id": [100], "name": ["Corp"]}), "Company", "id", "name")

        connections = [
            {
                "source_type": "Person",
                "target_type": "Company",
                "connection_name": "WORKS_AT",
                "data": pd.DataFrame({"source_id": [1, 2], "target_id": [100, 100]}),
            }
        ]

        stats = graph.add_connections_from_source(connections)
        assert stats["WORKS_AT"] == 2


class TestCompleteWorkflow:
    """End-to-end test of the connector workflow."""

    def test_factpages_style_workflow(self):
        """Test workflow similar to factpages_py integration."""
        graph = KnowledgeGraph()

        # Simulate data source providing node specs
        node_configs = {
            "field": {"node_type": "Field", "unique_id_field": "fldNpdidField", "node_title_field": "fldName"},
            "wellbore": {
                "node_type": "Wellbore",
                "unique_id_field": "wlbNpdidWellbore",
                "node_title_field": "wlbWellboreName",
            },
        }

        node_data = {
            "field": pd.DataFrame(
                {
                    "fldNpdidField": [1, 2, 3],
                    "fldName": ["Ekofisk", "Troll", "Snorre"],
                    "fldStatus": ["PRODUCING", "PRODUCING", "PRODUCING"],
                }
            ),
            "wellbore": pd.DataFrame(
                {
                    "wlbNpdidWellbore": [10, 20, 30, 40],
                    "wlbWellboreName": ["EK-1", "EK-2", "TR-1", "SN-1"],
                    "wlbPurpose": ["PRODUCTION", "PRODUCTION", "EXPLORATION", "PRODUCTION"],
                }
            ),
        }

        # Build node specs
        node_specs = [
            {
                "node_type": cfg["node_type"],
                "unique_id_field": cfg["unique_id_field"],
                "node_title_field": cfg["node_title_field"],
                "data": node_data[table],
            }
            for table, cfg in node_configs.items()
        ]

        # Bulk load nodes
        node_stats = graph.add_nodes_bulk(node_specs)
        assert node_stats["Field"] == 3
        assert node_stats["Wellbore"] == 4

        # Simulate connector providing all connections
        connections = [
            {
                "source_type": "Wellbore",
                "target_type": "Field",
                "connection_name": "DRILLED_ON",
                "data": pd.DataFrame({"source_id": [10, 20, 30, 40], "target_id": [1, 1, 2, 3]}),
            }
        ]

        # Auto-load connections
        conn_stats = graph.add_connections_from_source(connections)
        assert conn_stats["DRILLED_ON"] == 4

        # Verify graph state
        assert set(graph.node_types) == {"Field", "Wellbore"}
