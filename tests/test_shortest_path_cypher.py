"""Tests for shortestPath() in Cypher queries."""

import pytest

from kglite import KnowledgeGraph


@pytest.fixture
def chain_graph():
    """Linear chain: Alice -> Bob -> Charlie -> Dave -> Eve."""
    graph = KnowledgeGraph()

    for name in ["Alice", "Bob", "Charlie", "Dave", "Eve"]:
        graph.cypher(f"CREATE (:Person {{name: '{name}'}})")

    graph.cypher("MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[:KNOWS]->(b)")
    graph.cypher("MATCH (a:Person {name: 'Bob'}), (b:Person {name: 'Charlie'}) CREATE (a)-[:KNOWS]->(b)")
    graph.cypher("MATCH (a:Person {name: 'Charlie'}), (b:Person {name: 'Dave'}) CREATE (a)-[:KNOWS]->(b)")
    graph.cypher("MATCH (a:Person {name: 'Dave'}), (b:Person {name: 'Eve'}) CREATE (a)-[:KNOWS]->(b)")

    return graph


@pytest.fixture
def diamond_graph():
    """Diamond graph with shortcut: A -> B -> D and A -> C -> D (two paths of length 2)."""
    graph = KnowledgeGraph()

    for name in ["A", "B", "C", "D"]:
        graph.cypher(f"CREATE (:Node {{name: '{name}'}})")

    graph.cypher("MATCH (a:Node {name: 'A'}), (b:Node {name: 'B'}) CREATE (a)-[:EDGE]->(b)")
    graph.cypher("MATCH (a:Node {name: 'A'}), (c:Node {name: 'C'}) CREATE (a)-[:EDGE]->(c)")
    graph.cypher("MATCH (b:Node {name: 'B'}), (d:Node {name: 'D'}) CREATE (b)-[:EDGE]->(d)")
    graph.cypher("MATCH (c:Node {name: 'C'}), (d:Node {name: 'D'}) CREATE (c)-[:EDGE]->(d)")

    return graph


@pytest.fixture
def shortcut_graph():
    """Graph where direct edge is shorter than chain: A->B->C and A->C (shortcut)."""
    graph = KnowledgeGraph()

    for name in ["A", "B", "C"]:
        graph.cypher(f"CREATE (:Node {{name: '{name}'}})")

    graph.cypher("MATCH (a:Node {name: 'A'}), (b:Node {name: 'B'}) CREATE (a)-[:EDGE]->(b)")
    graph.cypher("MATCH (b:Node {name: 'B'}), (c:Node {name: 'C'}) CREATE (b)-[:EDGE]->(c)")
    graph.cypher("MATCH (a:Node {name: 'A'}), (c:Node {name: 'C'}) CREATE (a)-[:SHORTCUT]->(c)")

    return graph


class TestShortestPathBasic:
    """Basic shortestPath functionality."""

    def test_simple_chain(self, chain_graph):
        """Shortest path along a chain."""
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Person {name: 'Eve'})) "
            "RETURN length(p)"
        )
        assert len(result) == 1
        assert result[0]["length(p)"] == 4

    def test_adjacent_nodes(self, chain_graph):
        """Shortest path between directly connected nodes."""
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Person {name: 'Bob'})) "
            "RETURN length(p)"
        )
        assert len(result) == 1
        assert result[0]["length(p)"] == 1

    def test_shortcut_found(self, shortcut_graph):
        """Direct shortcut should be found over longer chain."""
        result = shortcut_graph.cypher(
            "MATCH p = shortestPath((a:Node {name: 'A'})-[*..10]->(b:Node {name: 'C'})) RETURN length(p)"
        )
        assert len(result) == 1
        assert result[0]["length(p)"] == 1  # direct edge A->C

    def test_diamond_length(self, diamond_graph):
        """Both paths through diamond are length 2."""
        result = diamond_graph.cypher(
            "MATCH p = shortestPath((a:Node {name: 'A'})-[:EDGE*..10]->(b:Node {name: 'D'})) RETURN length(p)"
        )
        assert len(result) == 1
        assert result[0]["length(p)"] == 2

    def test_no_path_returns_empty(self, chain_graph):
        """No path between disconnected nodes returns empty result."""
        chain_graph.cypher("CREATE (:Person {name: 'Isolated'})")
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Person {name: 'Isolated'})) "
            "RETURN length(p)"
        )
        assert len(result) == 0

    def test_same_type_filter(self, chain_graph):
        """Type filter restricts endpoints correctly."""
        chain_graph.cypher("CREATE (:Animal {name: 'Rex'})")
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Animal {name: 'Rex'})) "
            "RETURN length(p)"
        )
        assert len(result) == 0


class TestShortestPathFunctions:
    """Test length(), nodes(), relationships() on paths."""

    def test_length_function(self, chain_graph):
        """length(p) returns hop count."""
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Person {name: 'Charlie'})) "
            "RETURN length(p)"
        )
        assert result[0]["length(p)"] == 2

    def test_nodes_function(self, chain_graph):
        """nodes(p) returns list of node dicts."""
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Person {name: 'Charlie'})) "
            "RETURN nodes(p)"
        )
        nodes = result[0]["nodes(p)"]
        assert isinstance(nodes, list)
        titles = [n["properties"]["title"] for n in nodes]
        assert titles == ["Alice", "Bob", "Charlie"]

    def test_relationships_function(self, chain_graph):
        """relationships(p) returns list of Rel dicts (post-A.1).

        Pre-A.1 returned list of type strings; Phase A.1 / C2 now
        returns full Rel dicts. Extract `.type` for legacy shape.
        """
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Person {name: 'Charlie'})) "
            "RETURN relationships(p)"
        )
        rels = result[0]["relationships(p)"]
        assert isinstance(rels, list)
        assert [r["type"] for r in rels] == ["KNOWS", "KNOWS"]

    def test_all_path_functions_together(self, chain_graph):
        """All path functions work in the same RETURN."""
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Person {name: 'Dave'})) "
            "RETURN length(p), nodes(p), relationships(p)"
        )
        row = result[0]
        assert row["length(p)"] == 3
        nodes = row["nodes(p)"]
        assert isinstance(nodes, list)
        titles = [n["properties"]["title"] for n in nodes]
        assert "Alice" in titles and "Dave" in titles
        rels = row["relationships(p)"]
        assert isinstance(rels, list)
        # Phase A.1 / C2 — Rel dicts; check .type field.
        assert all(r["type"] == "KNOWS" for r in rels)

    def test_source_target_variables(self, chain_graph):
        """Source and target node variables are accessible."""
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Person {name: 'Dave'})) "
            "RETURN a.name, b.name, length(p)"
        )
        row = result[0]
        assert row["a.name"] == "Alice"
        assert row["b.name"] == "Dave"
        assert row["length(p)"] == 3


class TestShortestPathEdgeCases:
    """Edge cases for shortestPath."""

    def test_reverse_direction_no_path(self, chain_graph):
        """Chain is directed; reverse direction should find no path."""
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Eve'})-[:KNOWS*..10]->(b:Person {name: 'Alice'})) "
            "RETURN length(p)"
        )
        assert len(result) == 0

    def test_single_node_graph(self):
        """shortestPath with single node — no path to itself."""
        graph = KnowledgeGraph()
        graph.cypher("CREATE (:Person {name: 'Alone'})")
        result = graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alone'})-[:KNOWS*..10]->(b:Person {name: 'Alone'})) "
            "RETURN length(p)"
        )
        # Same source and target — executor skips self-loops
        assert len(result) == 0

    def test_multiple_types_unfiltered(self, shortcut_graph):
        """Without edge type filter, any edge type is traversed."""
        result = shortcut_graph.cypher(
            "MATCH p = shortestPath((a:Node {name: 'A'})-[*..10]->(b:Node {name: 'C'})) RETURN length(p)"
        )
        assert result[0]["length(p)"] == 1  # uses SHORTCUT

    def test_columns_correct(self, chain_graph):
        """Row dicts have correct keys."""
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Person {name: 'Bob'})) "
            "RETURN length(p), nodes(p), relationships(p)"
        )
        assert len(result) == 1
        assert set(result[0].keys()) == {"length(p)", "nodes(p)", "relationships(p)"}


class TestShortestPathWithClause:
    """Regression tests for path_bindings surviving WITH clauses."""

    def test_length_after_with(self, chain_graph):
        """length(p) works after path passes through WITH."""
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Person {name: 'Charlie'})) "
            "WITH p "
            "RETURN length(p)"
        )
        assert len(result) == 1
        assert result[0]["length(p)"] == 2

    def test_nodes_after_with(self, chain_graph):
        """nodes(p) works after path passes through WITH."""
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Person {name: 'Charlie'})) "
            "WITH p "
            "RETURN nodes(p)"
        )
        nodes = result[0]["nodes(p)"]
        titles = [n["properties"]["title"] for n in nodes]
        assert titles == ["Alice", "Bob", "Charlie"]

    def test_path_with_aggregation(self, chain_graph):
        """length(p) works after WITH that includes aggregation."""
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Person {name: 'Eve'})) "
            "WITH p, 1 AS dummy "
            "RETURN length(p), dummy"
        )
        assert len(result) == 1
        assert result[0]["length(p)"] == 4
        assert result[0]["dummy"] == 1

    def test_empty_graph(self):
        """shortestPath on empty graph returns no rows."""
        graph = KnowledgeGraph()
        # This should fail to parse or return empty — depends on whether patterns can match
        # With no nodes of type Person, both source and target candidates are empty
        result = graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'X'})-[:KNOWS*..10]->(b:Person {name: 'Y'})) RETURN length(p)"
        )
        assert len(result) == 0


class TestShortestPathUndirected:
    """Tests for undirected shortestPath using -[]- syntax."""

    def test_undirected_finds_reverse_path(self, chain_graph):
        """Undirected -[]- syntax should find path even against edge direction."""
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Eve'})-[:KNOWS*..10]-(b:Person {name: 'Alice'})) RETURN length(p)"
        )
        assert len(result) == 1
        assert result[0]["length(p)"] == 4

    def test_directed_still_respects_direction(self, chain_graph):
        """Directed -[]-> syntax should still require edge direction."""
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Eve'})-[:KNOWS*..10]->(b:Person {name: 'Alice'})) "
            "RETURN length(p)"
        )
        assert len(result) == 0  # No path in reverse direction

    def test_undirected_nodes_function(self, chain_graph):
        """nodes(p) works correctly with undirected shortestPath."""
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Eve'})-[:KNOWS*..10]-(b:Person {name: 'Alice'})) "
            "RETURN nodes(p), length(p)"
        )
        assert len(result) == 1
        nodes = result[0]["nodes(p)"]
        assert len(nodes) == 5
        titles = [n["properties"]["title"] for n in nodes]
        assert titles[0] == "Eve"
        assert titles[-1] == "Alice"

    def test_undirected_forward_also_works(self, chain_graph):
        """Undirected syntax also works in the forward direction."""
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]-(b:Person {name: 'Eve'})) RETURN length(p)"
        )
        assert len(result) == 1
        assert result[0]["length(p)"] == 4

    def test_undirected_bidirectional_pair_dedup(self):
        """B4: a→b and b→a both exist; undirected neighbor collection must
        not double-count, and shortestPath must still report length 1."""
        g = KnowledgeGraph()
        for nm in ["A", "B"]:
            g.cypher(f"CREATE (:N {{name: '{nm}'}})")
        g.cypher("MATCH (a:N {name:'A'}), (b:N {name:'B'}) CREATE (a)-[:R]->(b)")
        g.cypher("MATCH (a:N {name:'A'}), (b:N {name:'B'}) CREATE (b)-[:R]->(a)")
        rows = g.cypher("MATCH p = shortestPath((a:N {name:'A'})-[*]-(b:N {name:'B'})) RETURN length(p) AS L")
        assert len(rows) == 1
        assert rows[0]["L"] == 1

    def test_undirected_parallel_edges_dedup(self):
        """B4: two parallel a→b edges of the same type must surface b once,
        not twice, in the undirected neighbor list."""
        g = KnowledgeGraph()
        for nm in ["A", "B", "C"]:
            g.cypher(f"CREATE (:N {{name: '{nm}'}})")
        # Two parallel edges A→B, then a chain to C.
        g.cypher("MATCH (a:N {name:'A'}), (b:N {name:'B'}) CREATE (a)-[:R {tag:'x'}]->(b)")
        g.cypher("MATCH (a:N {name:'A'}), (b:N {name:'B'}) CREATE (a)-[:R {tag:'y'}]->(b)")
        g.cypher("MATCH (b:N {name:'B'}), (c:N {name:'C'}) CREATE (b)-[:R]->(c)")
        rows = g.cypher("MATCH p = shortestPath((a:N {name:'A'})-[*]-(c:N {name:'C'})) RETURN length(p) AS L")
        assert rows[0]["L"] == 2


class TestShortestPathConsistency:
    """Tests for consistency between Cypher shortestPath and fluent API."""

    def test_undirected_cypher_matches_fluent_api(self):
        """Undirected Cypher shortestPath should match fluent API result."""
        graph = KnowledgeGraph()
        # Use add_nodes so nodes have proper unique IDs for the fluent API
        import pandas as pd

        nodes = pd.DataFrame({"name": ["Alice", "Bob", "Charlie", "Dave", "Eve"]})
        graph.add_nodes(data=nodes, node_type="Person", unique_id_field="name")
        graph.add_connections(
            source_type="Person",
            target_type="Person",
            data=pd.DataFrame(
                {
                    "source": ["Alice", "Bob", "Charlie", "Dave"],
                    "target": ["Bob", "Charlie", "Dave", "Eve"],
                }
            ),
            source_id_field="source",
            target_id_field="target",
            connection_type="KNOWS",
        )
        # Fluent API (always undirected)
        fluent_result = graph.shortest_path(
            source_type="Person",
            source_id="Alice",
            target_type="Person",
            target_id="Eve",
        )
        # Cypher undirected
        cypher_result = graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]-(b:Person {name: 'Eve'})) RETURN length(p)"
        )
        assert fluent_result is not None
        assert len(cypher_result) == 1
        assert fluent_result["length"] == cypher_result[0]["length(p)"]

    def test_reverse_undirected_cypher_matches_fluent_api(self):
        """Reverse undirected Cypher should match reverse fluent API."""
        graph = KnowledgeGraph()
        import pandas as pd

        nodes = pd.DataFrame({"name": ["Alice", "Bob", "Charlie", "Dave", "Eve"]})
        graph.add_nodes(data=nodes, node_type="Person", unique_id_field="name")
        graph.add_connections(
            source_type="Person",
            target_type="Person",
            data=pd.DataFrame(
                {
                    "source": ["Alice", "Bob", "Charlie", "Dave"],
                    "target": ["Bob", "Charlie", "Dave", "Eve"],
                }
            ),
            source_id_field="source",
            target_id_field="target",
            connection_type="KNOWS",
        )
        # Fluent API (always undirected — finds path against edge direction)
        fluent_result = graph.shortest_path(
            source_type="Person",
            source_id="Eve",
            target_type="Person",
            target_id="Alice",
        )
        # Cypher undirected
        cypher_result = graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Eve'})-[:KNOWS*..10]-(b:Person {name: 'Alice'})) RETURN length(p)"
        )
        assert fluent_result is not None
        assert len(cypher_result) == 1
        assert fluent_result["length"] == cypher_result[0]["length(p)"]

    def test_connection_type_filter_in_cypher(self, shortcut_graph):
        """Cypher should respect connection type filter in shortestPath."""
        # With [:EDGE*] filter — must use only EDGE connections (A->B->C = 2 hops)
        result_filtered = shortcut_graph.cypher(
            "MATCH p = shortestPath((a:Node {name: 'A'})-[:EDGE*..10]->(b:Node {name: 'C'})) RETURN length(p)"
        )
        # Without type filter — can use SHORTCUT (A->C = 1 hop)
        result_unfiltered = shortcut_graph.cypher(
            "MATCH p = shortestPath((a:Node {name: 'A'})-[*..10]->(b:Node {name: 'C'})) RETURN length(p)"
        )
        assert len(result_filtered) == 1
        assert result_filtered[0]["length(p)"] == 2  # only EDGE edges
        assert len(result_unfiltered) == 1
        assert result_unfiltered[0]["length(p)"] == 1  # uses SHORTCUT

    def test_directed_cypher_respects_direction(self, chain_graph):
        """Directed Cypher should not find path against edge direction."""
        # Directed: Eve -[:KNOWS*]-> Alice — no path (edges go Alice->...->Eve)
        result_directed = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Eve'})-[:KNOWS*..10]->(b:Person {name: 'Alice'})) "
            "RETURN length(p)"
        )
        assert len(result_directed) == 0

        # Undirected: Eve -[:KNOWS*]- Alice — should find path
        result_undirected = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Eve'})-[:KNOWS*..10]-(b:Person {name: 'Alice'})) RETURN length(p)"
        )
        assert len(result_undirected) == 1
        assert result_undirected[0]["length(p)"] == 4


class TestNormalMatchNotBroken:
    """Ensure normal MATCH patterns still work after shortestPath changes."""

    def test_simple_node_match(self):
        """Normal node MATCH still works."""
        graph = KnowledgeGraph()
        graph.cypher("CREATE (:Person {name: 'Alice'})")
        result = graph.cypher("MATCH (n:Person) RETURN n.name")
        assert result[0]["n.name"] == "Alice"

    def test_edge_match(self):
        """Normal edge MATCH still works."""
        graph = KnowledgeGraph()
        graph.cypher("CREATE (:Person {name: 'Alice'})")
        graph.cypher("CREATE (:Person {name: 'Bob'})")
        graph.cypher("MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[:KNOWS]->(b)")
        result = graph.cypher("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name")
        assert result[0]["a.name"] == "Alice"
        assert result[0]["b.name"] == "Bob"

    def test_multi_pattern_match(self):
        """Comma-separated patterns still work."""
        graph = KnowledgeGraph()
        graph.cypher("CREATE (:Person {name: 'Alice'})")
        graph.cypher("CREATE (:Person {name: 'Bob'})")
        result = graph.cypher("MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) RETURN a.name, b.name")
        assert result[0]["a.name"] == "Alice"
        assert result[0]["b.name"] == "Bob"

    def test_where_clause_match(self):
        """MATCH with WHERE still works."""
        graph = KnowledgeGraph()
        graph.cypher("CREATE (:Person {name: 'Alice', age: 30})")
        graph.cypher("CREATE (:Person {name: 'Bob', age: 25})")
        result = graph.cypher("MATCH (n:Person) WHERE n.age > 28 RETURN n.name")
        assert len(result) == 1
        assert result[0]["n.name"] == "Alice"


# ── 0.8.20: weighted shortest path ─────────────────────────────


class TestWeightedShortestPath:
    @pytest.fixture
    def weighted_graph(self):
        import pandas as pd

        g = KnowledgeGraph()
        g.add_nodes(
            pd.DataFrame({"id": ["A", "B", "C", "D"], "name": ["A", "B", "C", "D"]}),
            "P",
            "id",
            "name",
        )
        # A→B direct weight 1.0; A→C→B detour 0.4 + 0.4 = 0.8 (cheaper)
        # A→D detour weight 5.0
        g.add_connections(
            pd.DataFrame(
                {
                    "f": ["A", "A", "C", "A"],
                    "t": ["B", "C", "B", "D"],
                    "w": [1.0, 0.4, 0.4, 5.0],
                }
            ),
            "L",
            "P",
            "f",
            "P",
            "t",
            columns=["w"],
        )
        return g

    def test_unweighted_returns_hops(self, weighted_graph):
        result = weighted_graph.shortest_path("P", "A", "P", "B")
        assert result["length"] == 1  # direct hop
        assert "weight" not in result

    def test_weighted_picks_cheaper_detour(self, weighted_graph):
        result = weighted_graph.shortest_path("P", "A", "P", "B", weight_property="w")
        assert result["length"] == 2  # A → C → B
        assert abs(result["weight"] - 0.8) < 1e-9
        ids = [n["id"] for n in result["path"]]
        assert ids == ["A", "C", "B"]

    def test_weighted_length_returns_float(self, weighted_graph):
        d = weighted_graph.shortest_path_length("P", "A", "P", "B", weight_property="w")
        assert isinstance(d, float)
        assert abs(d - 0.8) < 1e-9

    def test_unweighted_length_returns_int(self, weighted_graph):
        d = weighted_graph.shortest_path_length("P", "A", "P", "B")
        assert isinstance(d, int)
        assert d == 1

    def test_missing_property_falls_back_to_one(self):
        import pandas as pd

        g = KnowledgeGraph()
        g.add_nodes(pd.DataFrame({"id": ["A", "B"], "name": ["A", "B"]}), "P", "id", "name")
        # No 'w' property at all — every edge defaults to weight 1.0
        g.add_connections(pd.DataFrame({"f": ["A"], "t": ["B"]}), "L", "P", "f", "P", "t")
        result = g.shortest_path("P", "A", "P", "B", weight_property="w")
        assert result["weight"] == 1.0
        assert result["length"] == 1

    def test_no_path(self):
        import pandas as pd

        g = KnowledgeGraph()
        g.add_nodes(pd.DataFrame({"id": ["A", "B"], "name": ["A", "B"]}), "P", "id", "name")
        # No edges
        result = g.shortest_path("P", "A", "P", "B", weight_property="w")
        assert result is None
        assert g.shortest_path_length("P", "A", "P", "B", weight_property="w") is None
