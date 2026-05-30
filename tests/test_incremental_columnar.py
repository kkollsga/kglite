"""Tests for incremental columnar insertion in mapped mode (Phase 2A-2B)."""

import pandas as pd

from kglite import KnowledgeGraph


class TestIncrementalColumnar:
    """Nodes added in mapped mode should use columnar storage from the start."""

    def test_properties_accessible_after_add_nodes(self):
        graph = KnowledgeGraph(storage="mapped")
        df = pd.DataFrame(
            {
                "nid": list(range(50)),
                "name": [f"Node_{i}" for i in range(50)],
                "score": [float(i * 10) for i in range(50)],
            }
        )
        graph.add_nodes(df, "Item", "nid", "name")

        # Properties should be readable via Cypher
        r = graph.cypher("MATCH (n:Item {id: 25}) RETURN n.score").to_df()
        assert r["n.score"][0] == 250.0

    def test_multiple_node_types(self):
        graph = KnowledgeGraph(storage="mapped")
        df1 = pd.DataFrame({"nid": [1, 2], "name": ["A", "B"], "age": [30, 40]})
        df2 = pd.DataFrame({"cid": [10, 20], "title": ["X", "Y"], "size": [100, 200]})
        graph.add_nodes(df1, "Person", "nid", "name")
        graph.add_nodes(df2, "Company", "cid", "title")

        r1 = graph.cypher("MATCH (n:Person {id: 1}) RETURN n.age").to_df()
        assert r1["n.age"][0] == 30

        r2 = graph.cypher("MATCH (n:Company {id: 10}) RETURN n.size").to_df()
        assert r2["n.size"][0] == 100

    def test_incremental_add_nodes_same_type(self):
        graph = KnowledgeGraph(storage="mapped")
        df1 = pd.DataFrame({"nid": [1, 2], "name": ["A", "B"], "value": [10, 20]})
        df2 = pd.DataFrame({"nid": [3, 4], "name": ["C", "D"], "value": [30, 40]})
        graph.add_nodes(df1, "Item", "nid", "name")
        graph.add_nodes(df2, "Item", "nid", "name")

        assert graph.select("Item").len() == 4
        r = graph.cypher("MATCH (n:Item {id: 3}) RETURN n.value").to_df()
        assert r["n.value"][0] == 30

    def test_schema_extension(self):
        """Second batch adds a new property column — schema should extend."""
        graph = KnowledgeGraph(storage="mapped")
        df1 = pd.DataFrame({"nid": [1], "name": ["A"], "x": [10]})
        graph.add_nodes(df1, "Item", "nid", "name")

        df2 = pd.DataFrame({"nid": [2], "name": ["B"], "x": [20], "y": [30]})
        graph.add_nodes(df2, "Item", "nid", "name")

        # Both x and y should be accessible
        r = graph.cypher("MATCH (n:Item {id: 2}) RETURN n.x, n.y").to_df()
        assert r["n.x"][0] == 20
        assert r["n.y"][0] == 30

    def test_cypher_where_filter(self):
        graph = KnowledgeGraph(storage="mapped")
        df = pd.DataFrame(
            {
                "nid": list(range(100)),
                "name": [f"N{i}" for i in range(100)],
                "val": [float(i) for i in range(100)],
            }
        )
        graph.add_nodes(df, "Item", "nid", "name")
        r = graph.cypher("MATCH (n:Item) WHERE n.val > 90 RETURN count(n) AS c").to_df()
        assert r["c"][0] == 9

    def test_is_columnar_in_mapped_mode(self):
        graph = KnowledgeGraph(storage="mapped")
        df = pd.DataFrame({"nid": [1], "name": ["A"]})
        graph.add_nodes(df, "Item", "nid", "name")
        assert graph.is_columnar

    def test_default_mode_not_columnar(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"nid": [1], "name": ["A"]})
        graph.add_nodes(df, "Item", "nid", "name")
        assert not graph.is_columnar

    def test_save_load_roundtrip_preserves_data(self, tmp_path):
        graph = KnowledgeGraph(storage="mapped")
        df = pd.DataFrame(
            {
                "nid": list(range(20)),
                "name": [f"N{i}" for i in range(20)],
                "score": [float(i * 5) for i in range(20)],
            }
        )
        graph.add_nodes(df, "Item", "nid", "name")

        path = str(tmp_path / "test_col.kgl")
        graph.save(path)

        from kglite import load

        loaded = load(path)
        assert loaded.select("Item").len() == 20
        r = loaded.cypher("MATCH (n:Item {id: 10}) RETURN n.score").to_df()
        assert r["n.score"][0] == 50.0


class TestNTriplesColumnar:
    """N-Triples loader in mapped mode should produce columnar nodes."""

    def test_ntriples_properties_in_mapped_mode(self):
        graph = KnowledgeGraph(storage="mapped")
        graph.load_ntriples(
            "tests/data/sample_wikidata.nt",
            languages=["en"],
        )
        # Cross-mode parity (0.11.0): id is the integer; query the string Q-code via nid
        r = graph.cypher('MATCH (n {nid: "Q42"}) RETURN n.description, n.P1082').to_df()
        assert r["n.description"][0] == "English author and humourist"
        assert r["n.P1082"][0] == 42

    def test_ntriples_edges_in_mapped_mode(self):
        graph = KnowledgeGraph(storage="mapped")
        graph.load_ntriples(
            "tests/data/sample_wikidata.nt",
            languages=["en"],
        )
        # Cross-mode parity (0.11.0): id is the integer; query the string Q-code via nid
        r = graph.cypher('MATCH (n {nid: "Q42"})-[:P27]->(m) RETURN m.title').to_df()
        assert len(r) == 1
        assert r["m.title"][0] == "United Kingdom"
