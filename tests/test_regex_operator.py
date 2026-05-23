"""Tests for the =~ regex match operator in Cypher queries."""

import pytest

from kglite import KnowledgeGraph


@pytest.fixture
def name_graph():
    """Graph with various names for regex testing."""
    g = KnowledgeGraph()
    for name in ["Alice", "Bob", "Charlie", "alice", "ALICE", "Alex", "Brian"]:
        g.cypher(f"CREATE (:Person {{name: '{name}'}})")
    return g


class TestRegexBasic:
    """Basic regex =~ operator tests."""

    def test_simple_match(self, name_graph):
        """Simple substring match."""
        result = name_graph.cypher("""
            MATCH (p:Person)
            WHERE p.name =~ '.*lic.*'
            RETURN p.name
            ORDER BY p.name
        """)
        names = [r["p.name"] for r in result]
        assert names == ["Alice", "alice"]

    def test_anchored_match(self, name_graph):
        """Anchored pattern with ^ and $."""
        result = name_graph.cypher("""
            MATCH (p:Person)
            WHERE p.name =~ '^A.*'
            RETURN p.name
            ORDER BY p.name
        """)
        names = [r["p.name"] for r in result]
        assert names == ["ALICE", "Alex", "Alice"]

    def test_exact_match(self, name_graph):
        """Exact match with anchors."""
        result = name_graph.cypher("""
            MATCH (p:Person)
            WHERE p.name =~ '^Bob$'
            RETURN p.name
        """)
        assert len(result) == 1
        assert result[0]["p.name"] == "Bob"

    def test_no_match(self, name_graph):
        """Pattern that matches nothing."""
        result = name_graph.cypher("""
            MATCH (p:Person)
            WHERE p.name =~ '^Zorro$'
            RETURN p.name
        """)
        assert len(result) == 0


class TestRegexAdvanced:
    """Advanced regex features."""

    def test_case_insensitive(self, name_graph):
        """Case-insensitive match with (?i)."""
        result = name_graph.cypher("""
            MATCH (p:Person)
            WHERE p.name =~ '(?i)^alice$'
            RETURN p.name
            ORDER BY p.name
        """)
        names = [r["p.name"] for r in result]
        assert names == ["ALICE", "Alice", "alice"]

    def test_character_class(self, name_graph):
        """Character class [A-C] match."""
        result = name_graph.cypher("""
            MATCH (p:Person)
            WHERE p.name =~ '^[A-C].*'
            RETURN p.name
            ORDER BY p.name
        """)
        names = [r["p.name"] for r in result]
        assert names == ["ALICE", "Alex", "Alice", "Bob", "Brian", "Charlie"]

    def test_alternation(self, name_graph):
        """Alternation with |."""
        result = name_graph.cypher("""
            MATCH (p:Person)
            WHERE p.name =~ '^(Bob|Charlie)$'
            RETURN p.name
            ORDER BY p.name
        """)
        names = [r["p.name"] for r in result]
        assert names == ["Bob", "Charlie"]

    def test_invalid_regex_raises_error(self, name_graph):
        """Invalid regex pattern raises typed kglite.CypherExecutionError (was RuntimeError pre-A.2)."""
        import pytest

        import kglite

        with pytest.raises(kglite.KgError, match="Invalid regular expression"):
            name_graph.cypher("""
                MATCH (p:Person)
                WHERE p.name =~ '[invalid('
                RETURN p.name
            """)

    def test_regex_with_not(self, name_graph):
        """NOT combined with regex."""
        result = name_graph.cypher("""
            MATCH (p:Person)
            WHERE NOT p.name =~ '^A.*'
            RETURN p.name
            ORDER BY p.name
        """)
        names = [r["p.name"] for r in result]
        assert names == ["Bob", "Brian", "Charlie", "alice"]
