"""Tests for export_csv: organized CSV directory export with blueprint."""

import json
import os
import shutil
import tempfile

import pandas as pd
import pytest

from kglite import KnowledgeGraph


@pytest.fixture
def export_dir():
    """Create a temporary directory for export, cleaned up after test."""
    d = tempfile.mkdtemp(prefix="kglite_export_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def graph_with_subnodes():
    """Graph with parent/child types for sub-node nesting tests."""
    graph = KnowledgeGraph()

    wells = pd.DataFrame(
        {
            "well_id": ["W1", "W2"],
            "name": ["Well Alpha", "Well Beta"],
            "depth": [3000, 4500],
        }
    )
    graph.add_nodes(wells, "Well", "well_id", "name")

    logs = pd.DataFrame(
        {
            "log_id": ["L1", "L2", "L3"],
            "name": ["GR Log 1", "GR Log 2", "DT Log 1"],
            "tool": ["GR", "GR", "DT"],
            "well_fk": ["W1", "W1", "W2"],
        }
    )
    graph.add_nodes(logs, "Log", "log_id", "name")
    graph.set_parent_type("Log", "Well")

    edges = pd.DataFrame(
        {
            "from_id": ["L1", "L2", "L3"],
            "to_id": ["W1", "W1", "W2"],
        }
    )
    graph.add_connections(edges, "OF_WELL", "Log", "from_id", "Well", "to_id")

    return graph


class TestExportCsvBasic:
    def test_export_creates_directory_structure(self, small_graph, export_dir):
        out = os.path.join(export_dir, "out")
        result = small_graph.export_csv(out)

        assert os.path.isdir(os.path.join(out, "nodes"))
        assert os.path.isfile(os.path.join(out, "nodes", "Person.csv"))
        assert os.path.isfile(os.path.join(out, "blueprint.json"))
        assert result["files_written"] >= 2  # at least 1 node CSV + blueprint

    def test_export_returns_summary(self, small_graph, export_dir):
        out = os.path.join(export_dir, "out")
        result = small_graph.export_csv(out)

        assert result["output_dir"] == out
        assert "Person" in result["nodes"]
        assert result["nodes"]["Person"] == 3
        assert "KNOWS" in result["connections"]
        assert result["connections"]["KNOWS"] == 3
        assert result["files_written"] > 0

    def test_node_csv_has_properties(self, small_graph, export_dir):
        out = os.path.join(export_dir, "out")
        small_graph.export_csv(out)

        csv_path = os.path.join(out, "nodes", "Person.csv")
        df = pd.read_csv(csv_path, dtype=str)

        assert "id" in df.columns
        assert "title" in df.columns
        assert "age" in df.columns
        assert "city" in df.columns
        assert len(df) == 3

    def test_connection_csv_has_endpoints(self, small_graph, export_dir):
        out = os.path.join(export_dir, "out")
        small_graph.export_csv(out)

        csv_path = os.path.join(out, "connections", "KNOWS.csv")
        df = pd.read_csv(csv_path, dtype=str)

        assert "source_id" in df.columns
        assert "source_type" in df.columns
        assert "target_id" in df.columns
        assert "target_type" in df.columns
        assert "since" in df.columns
        assert len(df) == 3

    def test_node_ids_are_user_facing(self, small_graph, export_dir):
        """Node CSV should use the user-facing ID, not internal graph indices."""
        out = os.path.join(export_dir, "out")
        small_graph.export_csv(out)

        csv_path = os.path.join(out, "nodes", "Person.csv")
        df = pd.read_csv(csv_path)

        # The small_graph uses person_id 1,2,3 as IDs
        ids = set(df["id"].tolist())
        assert ids == {1, 2, 3}

    def test_connection_ids_match_node_ids(self, small_graph, export_dir):
        """Connection source_id/target_id should match node CSV id values."""
        out = os.path.join(export_dir, "out")
        small_graph.export_csv(out)

        nodes_df = pd.read_csv(os.path.join(out, "nodes", "Person.csv"))
        edges_df = pd.read_csv(os.path.join(out, "connections", "KNOWS.csv"))

        node_ids = set(nodes_df["id"].tolist())
        source_ids = set(edges_df["source_id"].tolist())
        target_ids = set(edges_df["target_id"].tolist())

        assert source_ids.issubset(node_ids)
        assert target_ids.issubset(node_ids)


class TestExportCsvSubnodes:
    def test_subnodes_nested_under_parent(self, graph_with_subnodes, export_dir):
        out = os.path.join(export_dir, "out")
        graph_with_subnodes.export_csv(out)

        # Well CSV at top level
        assert os.path.isfile(os.path.join(out, "nodes", "Well.csv"))
        # Log CSV nested under Well/
        assert os.path.isfile(os.path.join(out, "nodes", "Well", "Log.csv"))

    def test_subnode_csv_content(self, graph_with_subnodes, export_dir):
        out = os.path.join(export_dir, "out")
        graph_with_subnodes.export_csv(out)

        df = pd.read_csv(os.path.join(out, "nodes", "Well", "Log.csv"), dtype=str)
        assert len(df) == 3
        assert "tool" in df.columns


class TestExportCsvSelection:
    def test_export_with_type_filter(self, social_graph, export_dir):
        out = os.path.join(export_dir, "out")
        result = social_graph.select("Company").export_csv(out)

        assert "Company" in result["nodes"]
        assert "Person" not in result["nodes"]
        assert os.path.isfile(os.path.join(out, "nodes", "Company.csv"))
        assert not os.path.isfile(os.path.join(out, "nodes", "Person.csv"))

    def test_export_selection_connections_both_endpoints(self, social_graph, export_dir):
        """Only connections where both endpoints are selected should be exported."""
        out = os.path.join(export_dir, "out")
        result = social_graph.select("Person").export_csv(out)

        # KNOWS edges (Person -> Person) should be present
        assert "KNOWS" in result["connections"]
        # WORKS_AT (Person -> Company) should NOT be present because Company isn't selected
        assert "WORKS_AT" not in result["connections"]

    def test_export_full_graph_override(self, social_graph, export_dir):
        """selection_only=False exports the full graph even with active selection."""
        out = os.path.join(export_dir, "out")
        social_graph.select("Company")
        result = social_graph.export_csv(out, selection_only=False)

        assert "Company" in result["nodes"]
        assert "Person" in result["nodes"]


class TestExportCsvBlueprint:
    def test_blueprint_always_generated(self, small_graph, export_dir):
        out = os.path.join(export_dir, "out")
        small_graph.export_csv(out)

        bp_path = os.path.join(out, "blueprint.json")
        assert os.path.isfile(bp_path)

        with open(bp_path, encoding="utf-8") as f:
            bp = json.load(f)

        assert "settings" in bp
        assert "nodes" in bp
        assert "Person" in bp["nodes"]

    def test_blueprint_node_structure(self, small_graph, export_dir):
        out = os.path.join(export_dir, "out")
        small_graph.export_csv(out)

        with open(os.path.join(out, "blueprint.json"), encoding="utf-8") as f:
            bp = json.load(f)

        person = bp["nodes"]["Person"]
        assert person["pk"] == "id"
        assert person["title"] == "title"
        assert "csv" in person
        assert person["csv"].startswith("nodes/")

    def test_blueprint_has_connections(self, small_graph, export_dir):
        out = os.path.join(export_dir, "out")
        small_graph.export_csv(out)

        with open(os.path.join(out, "blueprint.json"), encoding="utf-8") as f:
            bp = json.load(f)

        person = bp["nodes"]["Person"]
        assert "connections" in person
        junctions = person["connections"]["junction_edges"]
        assert "KNOWS" in junctions

    def test_blueprint_subnode_has_parent(self, graph_with_subnodes, export_dir):
        out = os.path.join(export_dir, "out")
        graph_with_subnodes.export_csv(out)

        with open(os.path.join(out, "blueprint.json"), encoding="utf-8") as f:
            bp = json.load(f)

        assert "Log" in bp["nodes"]
        assert bp["nodes"]["Log"]["parent"] == "Well"


class TestExportCsvVerbose:
    def test_verbose_does_not_crash(self, small_graph, export_dir):
        """Verbose mode should complete without error and return correct results.

        Note: Rust println! output can't be captured by pytest's capsys,
        so we verify it doesn't crash and still returns correct results.
        """
        out = os.path.join(export_dir, "out")
        result = small_graph.export_csv(out, verbose=True)
        assert result["nodes"]["Person"] == 3


class TestExportCsvEdgeCases:
    def test_empty_graph(self, export_dir):
        graph = KnowledgeGraph()
        out = os.path.join(export_dir, "out")
        result = graph.export_csv(out)

        assert result["files_written"] == 1  # just blueprint.json
        assert len(result["nodes"]) == 0
        assert len(result["connections"]) == 0

    def test_graph_with_no_connections(self, export_dir):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"pk": [1, 2], "name": ["A", "B"]})
        graph.add_nodes(df, "Thing", "pk", "name")

        out = os.path.join(export_dir, "out")
        result = graph.export_csv(out)

        assert result["nodes"]["Thing"] == 2
        assert len(result["connections"]) == 0
        assert not os.path.isdir(os.path.join(out, "connections"))

    def test_special_characters_in_values(self, export_dir):
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "pk": [1],
                "name": ['O\'Brien & "Co"'],
                "note": ["has, commas"],
            }
        )
        graph.add_nodes(df, "Entity", "pk", "name")

        out = os.path.join(export_dir, "out")
        graph.export_csv(out)

        result_df = pd.read_csv(os.path.join(out, "nodes", "Entity.csv"), dtype=str)
        assert len(result_df) == 1
        assert result_df["title"].iloc[0] == 'O\'Brien & "Co"'

    def test_multiple_node_types(self, social_graph, export_dir):
        out = os.path.join(export_dir, "out")
        result = social_graph.export_csv(out)

        assert "Person" in result["nodes"]
        assert "Company" in result["nodes"]
        assert os.path.isfile(os.path.join(out, "nodes", "Person.csv"))
        assert os.path.isfile(os.path.join(out, "nodes", "Company.csv"))
