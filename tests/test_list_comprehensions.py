"""Tests for list comprehension in Cypher queries."""

import pytest

from kglite import KnowledgeGraph


class TestListComprehensions:
    """Test list comprehension [x IN list WHERE pred | expr] syntax."""

    def test_simple_list_comprehension_map(self):
        """Basic list comprehension with map expression."""
        graph = KnowledgeGraph()
        result = graph.cypher("UNWIND [1] AS dummy RETURN [x IN [1, 2, 3, 4, 5] | x * 2] AS doubled")

        assert len(result) == 1
        doubled = result[0]["doubled"]
        assert isinstance(doubled, list)
        assert doubled == [2, 4, 6, 8, 10]

    def test_list_comprehension_filter_only(self):
        """List comprehension with WHERE filter, no map."""
        graph = KnowledgeGraph()
        result = graph.cypher("UNWIND [1] AS dummy RETURN [x IN [1, 2, 3, 4, 5] WHERE x > 3] AS filtered")

        assert len(result) == 1
        filtered = result[0]["filtered"]
        assert isinstance(filtered, list)
        assert filtered == [4, 5]

    def test_list_comprehension_filter_and_map(self):
        """List comprehension with both filter and map."""
        graph = KnowledgeGraph()
        result = graph.cypher("UNWIND [1] AS dummy RETURN [x IN [1, 2, 3, 4, 5] WHERE x > 3 | x * 2] AS result")

        assert len(result) == 1
        result_val = result[0]["result"]
        assert isinstance(result_val, list)
        assert result_val == [8, 10]

    def test_list_comprehension_identity(self):
        """List comprehension without filter or map (identity)."""
        graph = KnowledgeGraph()
        result = graph.cypher("UNWIND [1] AS dummy RETURN [x IN [1, 2, 3]] AS identity")

        assert len(result) == 1
        identity = result[0]["identity"]
        assert isinstance(identity, list)
        assert identity == [1, 2, 3]

    def test_list_comprehension_empty_result(self):
        """List comprehension where filter excludes all items."""
        graph = KnowledgeGraph()
        result = graph.cypher("UNWIND [1] AS dummy RETURN [x IN [1, 2, 3] WHERE x > 10] AS empty")

        assert len(result) == 1
        empty = result[0]["empty"]
        assert isinstance(empty, list)
        assert empty == []

    def test_list_comprehension_with_string_literals(self):
        """List comprehension with string literals."""
        graph = KnowledgeGraph()
        # Note: Need to escape quotes properly
        result = graph.cypher("UNWIND [1] AS dummy RETURN [x IN ['a', 'b', 'c'] WHERE x <> 'b'] AS result")

        assert len(result) == 1
        result_val = result[0]["result"]
        assert isinstance(result_val, list)
        assert "a" in result_val and "c" in result_val
        assert "b" not in result_val

    def test_list_comprehension_arithmetic_filter(self):
        """List comprehension with arithmetic in filter."""
        graph = KnowledgeGraph()
        result = graph.cypher("UNWIND [1] AS dummy RETURN [x IN [1, 2, 3, 4, 5] WHERE x * x > 10] AS result")

        assert len(result) == 1
        result_val = result[0]["result"]
        assert isinstance(result_val, list)
        assert result_val == [4, 5]

    def test_list_comprehension_nested_expression(self):
        """List comprehension with complex map expression."""
        graph = KnowledgeGraph()
        result = graph.cypher("UNWIND [1] AS dummy RETURN [x IN [1, 2, 3] | x * 2 + 1] AS result")

        assert len(result) == 1
        result_val = result[0]["result"]
        assert isinstance(result_val, list)
        assert result_val == [3, 5, 7]


class TestListComprehensionPathFunctions:
    """Test list comprehensions with nodes(p) and relationships(p)."""

    @pytest.fixture
    def chain_graph(self):
        """Linear chain: Alice -> Bob -> Charlie."""
        graph = KnowledgeGraph()
        for name in ["Alice", "Bob", "Charlie"]:
            graph.cypher(f"CREATE (:Person {{name: '{name}', age: {20 + len(name)}}})")
        graph.cypher("MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[:KNOWS]->(b)")
        graph.cypher("MATCH (a:Person {name: 'Bob'}), (b:Person {name: 'Charlie'}) CREATE (a)-[:KNOWS]->(b)")
        return graph

    def test_nodes_name_extraction(self, chain_graph):
        """[n IN nodes(p) | n.name] extracts name from each node."""
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Person {name: 'Charlie'})) "
            "RETURN [n IN nodes(p) | n.name] AS names"
        )
        assert len(result) == 1
        names = result[0]["names"]
        assert isinstance(names, list)
        assert names == ["Alice", "Bob", "Charlie"]

    def test_nodes_type_extraction(self, chain_graph):
        """[n IN nodes(p) | n.type] extracts type from each node."""
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Person {name: 'Charlie'})) "
            "RETURN [n IN nodes(p) | n.type] AS types"
        )
        assert len(result) == 1
        types = result[0]["types"]
        assert isinstance(types, list)
        assert types == ["Person", "Person", "Person"]

    def test_nodes_property_access(self, chain_graph):
        """[n IN nodes(p) | n.age] extracts custom property from each node."""
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Person {name: 'Charlie'})) "
            "RETURN [n IN nodes(p) | n.age] AS ages"
        )
        assert len(result) == 1
        ages = result[0]["ages"]
        assert isinstance(ages, list)
        assert ages == [25, 23, 27]  # 20 + len(name)

    def test_nodes_with_filter(self, chain_graph):
        """[n IN nodes(p) WHERE pred | n.name] filters path nodes."""
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Person {name: 'Charlie'})) "
            "RETURN [n IN nodes(p) WHERE n.name <> 'Bob' | n.name] AS names"
        )
        assert len(result) == 1
        names = result[0]["names"]
        assert isinstance(names, list)
        assert names == ["Alice", "Charlie"]

    def test_nodes_identity(self, chain_graph):
        """[n IN nodes(p)] without map returns node dicts."""
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Person {name: 'Charlie'})) "
            "RETURN [n IN nodes(p)] AS node_list"
        )
        assert len(result) == 1
        node_list = result[0]["node_list"]
        assert isinstance(node_list, list)
        assert len(node_list) == 3

    def test_relationships_identity(self, chain_graph):
        """[r IN relationships(p)] returns the relationships themselves.

        Phase A.1 / C2 — `r` is now a Rel dict; pre-A.1 it was a type
        string. Check `.type` for the legacy shape.
        """
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Person {name: 'Charlie'})) "
            "RETURN [r IN relationships(p)] AS rels"
        )
        assert len(result) == 1
        rels = result[0]["rels"]
        assert isinstance(rels, list)
        assert [r["type"] for r in rels] == ["KNOWS", "KNOWS"]

    def test_nodes_after_with(self, chain_graph):
        """nodes(p) in list comprehension works after WITH clause."""
        result = chain_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..10]->(b:Person {name: 'Charlie'})) "
            "WITH p "
            "RETURN [n IN nodes(p) | n.name] AS names"
        )
        assert len(result) == 1
        names = result[0]["names"]
        assert names == ["Alice", "Bob", "Charlie"]


class TestListComprehensionIntegration:
    """Test list comprehensions integrated with MATCH queries."""

    def test_list_comprehension_with_collect(self):
        """Use list comprehension on collect() result."""
        graph = KnowledgeGraph()
        graph.cypher("CREATE (:Person {name: 'alice'})")
        graph.cypher("CREATE (:Person {name: 'bob'})")
        graph.cypher("CREATE (:Person {name: 'charlie'})")

        result = graph.cypher("""
            MATCH (n:Person)
            WITH collect(n.name) AS names
            UNWIND [1] AS dummy RETURN [x IN names | toUpper(x)] AS upper_names
        """)

        assert len(result) == 1
        upper_names = result[0]["upper_names"]
        assert isinstance(upper_names, list)
        # Should contain uppercase versions
        assert "ALICE" in upper_names

    def test_list_comprehension_in_return_with_aggregation(self):
        """List comprehension combined with aggregation."""
        graph = KnowledgeGraph()
        graph.cypher("CREATE (:Product {name: 'A', price: 10})")
        graph.cypher("CREATE (:Product {name: 'B', price: 20})")
        graph.cypher("CREATE (:Product {name: 'C', price: 30})")

        result = graph.cypher("""
            MATCH (p:Product)
            WITH collect(p.price) AS prices
            UNWIND [1] AS dummy RETURN [x IN prices WHERE x >= 20] AS expensive_prices
        """)

        assert len(result) == 1
        expensive = result[0]["expensive_prices"]
        assert isinstance(expensive, list)
        assert 20 in expensive and 30 in expensive
        assert 10 not in expensive

    def test_multiple_list_comprehensions(self):
        """Multiple list comprehensions in same RETURN."""
        graph = KnowledgeGraph()
        result = graph.cypher("""
            UNWIND [1] AS dummy
            RETURN
                [x IN [1, 2, 3] | x * 2] AS doubled,
                [x IN [1, 2, 3] WHERE x > 1] AS filtered
        """)

        assert len(result) == 1
        row = result[0]
        assert row["doubled"] == [2, 4, 6]
        assert row["filtered"] == [2, 3]
