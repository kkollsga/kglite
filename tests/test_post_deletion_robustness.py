"""Post-deletion robustness tests.

Tests that ALL graph operations work correctly AFTER nodes have been deleted
via Cypher DETACH DELETE, both WITH tombstones still present and after
vacuum/reindex. These test the StableDiGraph migration correctness.

Legacy DiGraph code had critical bugs:
- Vec[node_count] allocations would panic (index out of bounds) with tombstones
- node_count() vs node_bound() mismatch caused ID collisions
- Stale index entries pointed to deleted nodes
"""

import json
import os
import tempfile

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph


@pytest.fixture
def chain_graph():
    """Linear chain: A -> B -> C -> D -> E with properties.

    5 Person nodes connected in a chain. Useful for testing path
    algorithms where deleting a middle node breaks the chain.
    """
    graph = KnowledgeGraph()
    people = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Dave", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "city": ["Oslo", "Bergen", "Oslo", "Bergen", "Oslo"],
            "salary": [50000, 60000, 70000, 80000, 90000],
        }
    )
    graph.add_nodes(people, "Person", "person_id", "name")

    edges = pd.DataFrame(
        {
            "from_id": [1, 2, 3, 4],
            "to_id": [2, 3, 4, 5],
            "weight": [1.0, 2.0, 3.0, 4.0],
        }
    )
    graph.add_connections(edges, "KNOWS", "Person", "from_id", "Person", "to_id", columns=["weight"])
    return graph


@pytest.fixture
def chain_with_deletion(chain_graph):
    """Chain graph with middle node (Charlie, id=3) deleted — tombstone present."""
    chain_graph.cypher("MATCH (p:Person {name: 'Charlie'}) DETACH DELETE p")
    return chain_graph


# ============================================================================
# 1. Graph Algorithms After DELETE
# ============================================================================


class TestGraphAlgorithmsAfterDelete:
    """Graph algorithms with tombstones present — tests the node_bound fix."""

    def test_shortest_path_with_tombstone(self, chain_with_deletion):
        """shortest_path should work with tombstones (Vec[node_bound] fix)."""
        # Path A->E should still find a route (not through deleted Charlie)
        # But since Charlie is deleted, A->B and D->E chains are disconnected
        path = chain_with_deletion.shortest_path(
            source_type="Person",
            source_id=1,
            target_type="Person",
            target_id=2,
        )
        assert path is not None  # A->B direct path still exists

    def test_shortest_path_broken_by_deletion(self, chain_with_deletion):
        """Deleting a middle node should break the path."""
        # A-B-[deleted]-D-E: no path from A to D anymore
        path = chain_with_deletion.shortest_path(
            source_type="Person",
            source_id=1,
            target_type="Person",
            target_id=4,
        )
        assert path is None  # Path broken by deletion

    def test_shortest_path_length_with_tombstone(self, chain_with_deletion):
        """shortest_path_length should not panic with tombstones."""
        length = chain_with_deletion.shortest_path_length(
            source_type="Person",
            source_id=1,
            target_type="Person",
            target_id=2,
        )
        assert length == 1

    def test_shortest_path_ids_with_tombstone(self, chain_with_deletion):
        """shortest_path_ids should work with tombstones."""
        ids = chain_with_deletion.shortest_path_ids(
            source_type="Person",
            source_id=1,
            target_type="Person",
            target_id=2,
        )
        assert ids is not None
        assert len(ids) == 2

    def test_shortest_path_indices_with_tombstone(self, chain_with_deletion):
        """shortest_path_indices should work with tombstones."""
        indices = chain_with_deletion.shortest_path_indices(
            source_type="Person",
            source_id=1,
            target_type="Person",
            target_id=2,
        )
        assert indices is not None

    def test_all_paths_with_tombstone(self, chain_with_deletion):
        """all_paths should work — not panic on Vec indexing."""
        paths = chain_with_deletion.all_paths(
            source_type="Person",
            source_id=4,
            target_type="Person",
            target_id=5,
            max_hops=3,
        )
        assert len(paths) >= 1  # D->E still exists

    def test_connected_components_with_tombstone(self, chain_with_deletion):
        """connected_components should correctly split after deletion."""
        components = chain_with_deletion.connected_components()
        # With Charlie deleted, we should have at least 2 disconnected components:
        # {A, B} and {D, E}
        person_components = []
        for comp in components:
            person_nodes = [n for n in comp if n.get("type") == "Person"]
            if person_nodes:
                person_components.append(person_nodes)
        assert len(person_components) >= 2

    def test_are_connected_with_tombstone(self, chain_with_deletion):
        """are_connected should return False for nodes split by deletion."""
        result = chain_with_deletion.are_connected(
            source_type="Person",
            source_id=1,
            target_type="Person",
            target_id=5,
        )
        assert result is False  # Chain broken

    def test_are_connected_same_component(self, chain_with_deletion):
        """are_connected should return True for nodes in same surviving component."""
        result = chain_with_deletion.are_connected(
            source_type="Person",
            source_id=4,
            target_type="Person",
            target_id=5,
        )
        assert result is True  # D->E still connected

    def test_centrality_with_tombstone(self, chain_with_deletion):
        """Centrality algorithms should not crash with tombstones."""
        result = chain_with_deletion.betweenness_centrality()
        assert result is not None

    def test_pagerank_with_tombstone(self, chain_with_deletion):
        """PageRank should not crash with tombstones."""
        result = chain_with_deletion.pagerank()
        assert result is not None

    def test_degree_centrality_with_tombstone(self, chain_with_deletion):
        """Degree centrality should not crash with tombstones."""
        result = chain_with_deletion.degree_centrality()
        assert result is not None

    def test_algorithms_after_heavy_deletion(self):
        """Graph algorithms on a graph with 50% nodes deleted."""
        graph = KnowledgeGraph()
        people = pd.DataFrame(
            {
                "person_id": list(range(1, 21)),
                "name": [f"P_{i}" for i in range(1, 21)],
                "age": [20 + i for i in range(1, 21)],
            }
        )
        graph.add_nodes(people, "Person", "person_id", "name")
        edges = pd.DataFrame(
            {
                "from_id": list(range(1, 20)),
                "to_id": list(range(2, 21)),
            }
        )
        graph.add_connections(edges, "KNOWS", "Person", "from_id", "Person", "to_id")

        # Delete every other node
        for i in range(2, 21, 2):
            graph.cypher(f"MATCH (p:Person {{name: 'P_{i}'}}) DETACH DELETE p")

        # All these should work, not crash
        components = graph.connected_components()
        assert len(components) >= 1

        result = graph.betweenness_centrality()
        assert result is not None

        result = graph.pagerank()
        assert result is not None


# ============================================================================
# 2. Traversal After DELETE
# ============================================================================


class TestTraversalAfterDelete:
    """Traversal operations with tombstones present."""

    def test_traverse_from_surviving_node(self, chain_with_deletion):
        """Traverse from a node whose neighbor was deleted."""
        bob = chain_with_deletion.select("Person").where({"title": "Bob"})
        # Bob -> Charlie was deleted, so traversal should return nothing
        friends = bob.traverse(connection_type="KNOWS", direction="outgoing")
        assert friends.len() == 0

    def test_traverse_intact_edge(self, chain_with_deletion):
        """Traverse along an edge that still exists."""
        dave = chain_with_deletion.select("Person").where({"title": "Dave"})
        friends = dave.traverse(connection_type="KNOWS", direction="outgoing")
        assert friends.len() == 1  # Dave -> Eve

    def test_traverse_incoming_after_delete(self, chain_with_deletion):
        """Incoming traversal should skip deleted source nodes."""
        eve = chain_with_deletion.select("Person").where({"title": "Eve"})
        known_by = eve.traverse(connection_type="KNOWS", direction="incoming")
        assert known_by.len() == 1  # Only Dave knows Eve now

    def test_traverse_chain_after_delete(self, chain_with_deletion):
        """Multi-hop traversal should stop at deletion boundary."""
        alice = chain_with_deletion.select("Person").where({"title": "Alice"})
        hop1 = alice.traverse(connection_type="KNOWS", direction="outgoing")
        assert hop1.len() == 1  # Only Bob
        hop2 = hop1.traverse(connection_type="KNOWS", direction="outgoing")
        assert hop2.len() == 0  # Bob -> Charlie deleted


# ============================================================================
# 3. Filter After DELETE
# ============================================================================


class TestFilterAfterDelete:
    """Filtering operations with tombstones present (no vacuum/reindex)."""

    def test_type_filter_with_tombstone(self, chain_with_deletion):
        """type_filter should exclude deleted nodes."""
        filtered = chain_with_deletion.select("Person")
        assert filtered.len() == 4  # 5 - 1 deleted

    def test_property_filter_with_tombstone(self, chain_with_deletion):
        """filter() should work correctly with tombstones."""
        oslo_people = chain_with_deletion.select("Person").where({"city": "Oslo"})
        # Alice and Eve are from Oslo, Charlie was deleted
        assert oslo_people.len() == 2

    def test_filter_for_deleted_node(self, chain_with_deletion):
        """Filtering for a deleted node's properties should return empty."""
        result = chain_with_deletion.select("Person").where({"title": "Charlie"})
        assert result.len() == 0


# ============================================================================
# 4. Pattern Matching After DELETE
# ============================================================================


class TestPatternMatchingAfterDelete:
    """Pattern matching with tombstones present."""

    def test_match_pattern_with_tombstone(self, chain_with_deletion):
        """match_pattern should skip deleted nodes."""
        result = chain_with_deletion.match_pattern("(:Person)-[:KNOWS]->(:Person)")
        # Only intact edges: A->B, D->E (Charlie's edges were detach-deleted)
        assert len(result) == 2

    def test_match_pattern_deleted_node_type(self, chain_with_deletion):
        """Pattern matching for deleted node's properties returns nothing."""
        result = chain_with_deletion.match_pattern("(:Person {name: 'Charlie'})-[:KNOWS]->(:Person)")
        assert len(result) == 0


# ============================================================================
# 5. Add Nodes After DELETE (ID generation correctness)
# ============================================================================


class TestAddNodesAfterDelete:
    """Adding new nodes after deletion — tests node_bound ID generation."""

    def test_add_nodes_after_delete(self, chain_with_deletion):
        """add_nodes should work after deletion, no ID collision."""
        new_people = pd.DataFrame(
            {
                "person_id": [10, 11],
                "name": ["Frank", "Grace"],
                "age": [28, 33],
            }
        )
        chain_with_deletion.add_nodes(new_people, "Person", "person_id", "name")
        filtered = chain_with_deletion.select("Person")
        assert filtered.len() == 6  # 4 surviving + 2 new

    def test_add_nodes_data_intact(self, chain_with_deletion):
        """New nodes added after deletion should have correct data."""
        new_people = pd.DataFrame(
            {
                "person_id": [10],
                "name": ["Frank"],
                "age": [28],
            }
        )
        chain_with_deletion.add_nodes(new_people, "Person", "person_id", "name")
        frank = chain_with_deletion.node("Person", 10)
        assert frank is not None
        assert frank["title"] == "Frank"
        assert frank["age"] == 28

    def test_cypher_create_after_delete(self, chain_with_deletion):
        """Cypher CREATE after DELETE should generate valid IDs."""
        chain_with_deletion.cypher("CREATE (n:Person {name: 'Frank', age: 28})")
        assert chain_with_deletion.last_mutation_stats["nodes_created"] == 1

        # Verify node exists and has no collision with surviving nodes
        check = chain_with_deletion.cypher("MATCH (n:Person {name: 'Frank'}) RETURN n.age")
        assert len(check) == 1
        assert check[0]["n.age"] == 28

    def test_add_connections_after_delete(self, chain_with_deletion):
        """add_connections should work after deletion."""
        new_edges = pd.DataFrame(
            {
                "from_id": [1],
                "to_id": [4],
            }
        )
        chain_with_deletion.add_connections(new_edges, "KNOWS", "Person", "from_id", "Person", "to_id")
        # Now A->D should be connected
        path = chain_with_deletion.shortest_path(
            source_type="Person",
            source_id=1,
            target_type="Person",
            target_id=5,
        )
        assert path is not None  # A->D->E


# ============================================================================
# 6. Cypher MATCH After DELETE
# ============================================================================


class TestCypherAfterDelete:
    """Cypher queries with tombstones present (no vacuum)."""

    def test_match_all_after_delete(self, chain_with_deletion):
        """MATCH all nodes should skip deleted nodes."""
        result = chain_with_deletion.cypher("MATCH (n:Person) RETURN count(*) AS cnt")
        assert result[0]["cnt"] == 4

    def test_match_deleted_node_returns_empty(self, chain_with_deletion):
        """MATCH for a deleted node should return empty."""
        result = chain_with_deletion.cypher("MATCH (n:Person {name: 'Charlie'}) RETURN n.name")
        assert len(result) == 0

    def test_match_with_relationship_after_delete(self, chain_with_deletion):
        """MATCH relationship pattern should skip deleted endpoints."""
        result = chain_with_deletion.cypher("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name")
        names = [(r["a.name"], r["b.name"]) for r in result]
        # Only A->B and D->E should remain
        assert ("Alice", "Bob") in names
        assert ("Dave", "Eve") in names
        assert len(names) == 2

    def test_where_on_deleted_property(self, chain_with_deletion):
        """WHERE clause should correctly filter with tombstones."""
        result = chain_with_deletion.cypher("MATCH (n:Person) WHERE n.age > 30 RETURN n.name ORDER BY n.name")
        names = [r["n.name"] for r in result]
        # Charlie (35) deleted, Dave (40) and Eve (45) remain
        assert "Charlie" not in names
        assert "Dave" in names
        assert "Eve" in names


# ============================================================================
# 7. Export After DELETE
# ============================================================================


class TestExportAfterDelete:
    """Export operations with tombstones present."""

    def test_export_graphml_with_tombstone(self, chain_with_deletion):
        """GraphML export should skip deleted nodes."""
        with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
            path = f.name
        try:
            # Export requires an active selection
            filtered = chain_with_deletion.select("Person")
            filtered.export(path, format="graphml")
            assert os.path.exists(path)
            content = open(path, encoding="utf-8").read()
            assert "Alice" in content
            assert "Charlie" not in content  # Deleted node should not appear
        finally:
            os.unlink(path)

    def test_export_d3_with_tombstone(self, chain_with_deletion):
        """D3 JSON export should skip deleted nodes."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            filtered = chain_with_deletion.select("Person")
            filtered.export(path, format="d3")
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            node_names = [n.get("title", n.get("name", "")) for n in data["nodes"]]
            assert "Charlie" not in node_names
        finally:
            os.unlink(path)

    def test_export_string_with_tombstone(self, chain_with_deletion):
        """export_string should work with tombstones."""
        filtered = chain_with_deletion.select("Person")
        result = filtered.export_string(format="json")
        assert result is not None
        assert len(result) > 0
        assert "Charlie" not in result


# ============================================================================
# 8. Statistics/Calculations After DELETE
# ============================================================================


class TestStatisticsAfterDelete:
    """Statistics and calculations with tombstones present."""

    def test_statistics_with_tombstone(self, chain_with_deletion):
        """statistics() should compute on surviving nodes only."""
        filtered = chain_with_deletion.select("Person")
        stats = filtered.statistics("age", 0)
        assert stats is not None
        # Charlie (35) deleted. Remaining: 25, 30, 40, 45
        assert stats["valid_count"] == [4]
        assert stats["avg"] == [pytest.approx(35.0)]

    def test_calculate_with_tombstone(self, chain_with_deletion):
        """calculate() should not crash with tombstones."""
        filtered = chain_with_deletion.select("Person")
        # calculate returns a new graph; verify it doesn't panic
        result = filtered.calculate("age * 2", store_as="double_age")
        assert result is not None

    def test_count_with_tombstone(self, chain_with_deletion):
        """node_count should correctly skip tombstones."""
        filtered = chain_with_deletion.select("Person")
        assert filtered.len() == 4


# ============================================================================
# 9. Save/Load After DELETE
# ============================================================================


class TestSaveLoadAfterDelete:
    """Serialization with tombstones present."""

    def test_save_load_with_tombstone(self, chain_with_deletion):
        """Save/load should preserve graph state with tombstones."""
        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            chain_with_deletion.save(path)
            loaded = kglite.load(path)
            filtered = loaded.select("Person")
            assert filtered.len() == 4

            # Charlie should not exist
            charlie = loaded.node("Person", 3)
            assert charlie is None
        finally:
            os.unlink(path)

    def test_save_load_preserves_surviving_data(self, chain_with_deletion):
        """Surviving node data should be intact after save/load."""
        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            chain_with_deletion.save(path)
            loaded = kglite.load(path)
            alice = loaded.node("Person", 1)
            assert alice is not None
            assert alice["title"] == "Alice"
            assert alice["age"] == 25
        finally:
            os.unlink(path)

    def test_save_load_preserves_edges(self, chain_with_deletion):
        """Surviving edges should be intact after save/load."""
        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            chain_with_deletion.save(path)
            loaded = kglite.load(path)
            # D->E should still be connected
            path_result = loaded.shortest_path(
                source_type="Person",
                source_id=4,
                target_type="Person",
                target_id=5,
            )
            assert path_result is not None
        finally:
            os.unlink(path)

    def test_save_load_after_vacuum(self, chain_with_deletion):
        """Save/load after vacuum should produce a clean graph."""
        chain_with_deletion.vacuum()
        with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
            path = f.name
        try:
            chain_with_deletion.save(path)
            loaded = kglite.load(path)
            info = loaded.graph_info()
            assert info["node_tombstones"] == 0
        finally:
            os.unlink(path)


# ============================================================================
# 10. Index Operations After DELETE
# ============================================================================


class TestIndexAfterDelete:
    """Property index operations with tombstones present."""

    def test_create_index_with_tombstone(self, chain_with_deletion):
        """Creating an index after deletion should skip deleted nodes."""
        chain_with_deletion.create_index("Person", "city")
        info = chain_with_deletion.graph_info()
        assert info["property_index_count"] == 1

    def test_index_lookup_after_delete(self, chain_graph):
        """Index lookup after deletion should not return deleted nodes."""
        chain_graph.create_index("Person", "city")
        chain_graph.cypher("MATCH (p:Person {name: 'Charlie'}) DETACH DELETE p")
        chain_graph.reindex()  # Rebuild after delete

        # Oslo people: was Alice, Charlie, Eve — now Alice, Eve
        oslo = chain_graph.select("Person").where({"city": "Oslo"})
        assert oslo.len() == 2

    def test_build_id_indices_after_delete(self, chain_with_deletion):
        """build_id_indices should work with tombstones."""
        chain_with_deletion.build_id_indices(["Person"])
        # Should be able to look up surviving nodes
        alice = chain_with_deletion.node("Person", 1)
        assert alice is not None
        assert alice["title"] == "Alice"


# ============================================================================
# 11. node for Deleted Nodes
# ============================================================================


class TestGetNodeByIdAfterDelete:
    """node for deleted nodes should return None, not crash."""

    def test_get_deleted_node_returns_none(self, chain_with_deletion):
        """Looking up a deleted node by ID should return None."""
        charlie = chain_with_deletion.node("Person", 3)
        assert charlie is None

    def test_get_surviving_node_works(self, chain_with_deletion):
        """Looking up a surviving node should still work."""
        alice = chain_with_deletion.node("Person", 1)
        assert alice is not None
        assert alice["title"] == "Alice"

    def test_get_node_after_multiple_deletes(self, chain_graph):
        """node should handle multiple deletions correctly."""
        chain_graph.cypher("MATCH (p:Person {name: 'Bob'}) DETACH DELETE p")
        chain_graph.cypher("MATCH (p:Person {name: 'Dave'}) DETACH DELETE p")

        assert chain_graph.node("Person", 2) is None  # Bob deleted
        assert chain_graph.node("Person", 4) is None  # Dave deleted
        assert chain_graph.node("Person", 1) is not None  # Alice alive
        assert chain_graph.node("Person", 5) is not None  # Eve alive


# ============================================================================
# 12. to_df After DELETE
# ============================================================================


class TestToDfAfterDelete:
    """DataFrame export with tombstones present."""

    def test_to_df_with_tombstone(self, chain_with_deletion):
        """to_df should exclude deleted nodes."""
        filtered = chain_with_deletion.select("Person")
        df = filtered.to_df()
        assert len(df) == 4  # 5 - 1 deleted
        assert "Charlie" not in df["title"].values

    def test_to_df_column_integrity(self, chain_with_deletion):
        """DataFrame columns should be intact after deletion."""
        filtered = chain_with_deletion.select("Person")
        df = filtered.to_df()
        assert "age" in df.columns
        assert "city" in df.columns
        # Verify data correctness
        alice_row = df[df["title"] == "Alice"]
        assert len(alice_row) == 1
        assert alice_row.iloc[0]["age"] == 25


# ============================================================================
# 13. Combined Stress Test — Full Workflow After Deletion
# ============================================================================


class TestFullWorkflowAfterDelete:
    """End-to-end workflow: delete → operate with tombstones → vacuum → operate clean."""

    def test_full_lifecycle_with_tombstones(self):
        """Exercise every major operation with tombstones, then after vacuum."""
        graph = KnowledgeGraph()
        people = pd.DataFrame(
            {
                "person_id": list(range(1, 11)),
                "name": [f"P_{i}" for i in range(1, 11)],
                "age": [20 + i * 5 for i in range(1, 11)],
                "city": ["Oslo"] * 5 + ["Bergen"] * 5,
            }
        )
        graph.add_nodes(people, "Person", "person_id", "name")

        edges = pd.DataFrame(
            {
                "from_id": list(range(1, 10)),
                "to_id": list(range(2, 11)),
            }
        )
        graph.add_connections(edges, "KNOWS", "Person", "from_id", "Person", "to_id")

        # Create index before deletion
        graph.create_index("Person", "city")

        # Delete 3 nodes — create tombstones
        for name in ["P_3", "P_5", "P_7"]:
            graph.cypher(f"MATCH (p:Person {{name: '{name}'}}) DETACH DELETE p")

        info = graph.graph_info()
        assert info["node_tombstones"] == 3

        # --- With tombstones present ---

        # Type filter
        filtered = graph.select("Person")
        assert filtered.len() == 7

        # Property filter
        oslo = filtered.where({"city": "Oslo"})
        assert oslo.len() > 0

        # Cypher MATCH
        result = graph.cypher("MATCH (n:Person) RETURN count(*) AS cnt")
        assert result[0]["cnt"] == 7

        # Traversal
        p1 = graph.select("Person").where({"title": "P_1"})
        friends = p1.traverse(connection_type="KNOWS", direction="outgoing")
        assert friends.len() >= 1

        # to_df
        df = graph.select("Person").to_df()
        assert len(df) == 7

        # Graph algorithms
        components = graph.connected_components()
        assert len(components) >= 1

        # Add new node with tombstones present
        graph.cypher("CREATE (n:Person {name: 'New1', age: 99, city: 'Oslo'})")
        result = graph.cypher("MATCH (n:Person) RETURN count(*) AS cnt")
        assert result[0]["cnt"] == 8

        # --- Vacuum ---
        vacuum_result = graph.vacuum()
        assert vacuum_result["tombstones_removed"] >= 2  # At least 2 of 3 tombstones

        # --- After vacuum ---
        info = graph.graph_info()
        assert info["node_tombstones"] == 0

        # Everything still works
        result = graph.cypher("MATCH (n:Person) RETURN count(*) AS cnt")
        assert result[0]["cnt"] == 8

        components = graph.connected_components()
        assert len(components) >= 1

        # Index should still exist
        assert info["property_index_count"] == 1
