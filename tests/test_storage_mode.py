"""Tests for the storage mode parameter on KnowledgeGraph."""

import pandas as pd
import pytest

from kglite import KnowledgeGraph


class TestStorageModeConstruction:
    """Phase 0: constructor accepts storage parameter."""

    def test_default_mode_no_arg(self):
        graph = KnowledgeGraph()
        assert graph.select("Item").len() == 0

    def test_default_mode_explicit_none(self):
        graph = KnowledgeGraph(storage=None)
        assert graph.select("Item").len() == 0

    def test_mapped_mode(self):
        graph = KnowledgeGraph(storage="mapped")
        assert graph.select("Item").len() == 0

    def test_invalid_mode_raises(self):
        # Phase A.2 / C6 — typed kglite.ArgumentError (was ValueError pre-A.2).
        import kglite

        with pytest.raises(kglite.ArgumentError, match="Unknown storage mode"):
            KnowledgeGraph(storage="invalid")


class TestMappedModeBasicOperations:
    """Mapped mode graph should behave identically to default for basic ops."""

    @pytest.fixture
    def mapped_graph(self):
        graph = KnowledgeGraph(storage="mapped")
        nodes = pd.DataFrame(
            {
                "nid": list(range(100)),
                "name": [f"Node_{i}" for i in range(100)],
                "value": [float(i) for i in range(100)],
            }
        )
        graph.add_nodes(nodes, "Item", "nid", "name")

        edges = pd.DataFrame(
            {
                "from_id": [i % 100 for i in range(200)],
                "to_id": [(i * 7 + 3) % 100 for i in range(200)],
            }
        )
        graph.add_connections(edges, "LINKS", "Item", "from_id", "Item", "to_id")
        return graph

    def test_node_count(self, mapped_graph):
        assert mapped_graph.select("Item").len() == 100

    def test_edge_count(self, mapped_graph):
        result = mapped_graph.cypher("MATCH ()-[r:LINKS]->() RETURN count(r) AS cnt").to_df()
        assert result["cnt"][0] == 200

    def test_cypher_match(self, mapped_graph):
        result = mapped_graph.cypher("MATCH (n:Item) RETURN count(n) AS cnt").to_df()
        assert result["cnt"][0] == 100

    def test_cypher_where(self, mapped_graph):
        result = mapped_graph.cypher("MATCH (n:Item) WHERE n.value > 50 RETURN count(n) AS cnt").to_df()
        assert result["cnt"][0] == 49

    def test_cypher_traversal(self, mapped_graph):
        result = mapped_graph.cypher("MATCH (a:Item)-[:LINKS]->(b:Item) RETURN count(*) AS cnt").to_df()
        assert result["cnt"][0] == 200

    def test_select_and_traverse(self, mapped_graph):
        sel = mapped_graph.select("Item").where({"id": 0})
        assert sel.len() == 1
        traversed = sel.traverse("LINKS")
        assert traversed.len() > 0

    def test_properties_accessible(self, mapped_graph):
        result = mapped_graph.cypher("MATCH (n:Item {id: 42}) RETURN n.title, n.value").to_df()
        assert result["n.title"][0] == "Node_42"
        assert result["n.value"][0] == 42.0


class TestMappedModeSaveLoad:
    """Mapped mode graph should survive save/load round-trip."""

    def test_save_load_roundtrip(self, tmp_path):
        graph = KnowledgeGraph(storage="mapped")
        nodes = pd.DataFrame(
            {
                "nid": list(range(50)),
                "name": [f"N_{i}" for i in range(50)],
            }
        )
        graph.add_nodes(nodes, "Thing", "nid", "name")

        path = str(tmp_path / "mapped_test.kgl")
        graph.save(path)

        from kglite import load

        loaded = load(path)
        assert loaded.select("Thing").len() == 50
        result = loaded.cypher("MATCH (n:Thing) RETURN count(n) AS cnt").to_df()
        assert result["cnt"][0] == 50
