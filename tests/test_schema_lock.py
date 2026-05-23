"""Tests for schema locking: lock_schema(), unlock_schema(), schema_locked."""

import pytest

import kglite

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_graph():
    """Build a small graph with Person and Paper types + AUTHORED edges."""
    g = kglite.KnowledgeGraph()
    import pandas as pd

    persons = pd.DataFrame({"pid": [1, 2], "name": ["Alice", "Bob"], "age": [30, 25]})
    g.add_nodes(persons, "Person", "pid", node_title_field="name")

    papers = pd.DataFrame({"doi": ["10.1", "10.2"], "title": ["Paper A", "Paper B"], "year": [2020, 2021]})
    g.add_nodes(papers, "Paper", "doi", node_title_field="title")

    edges = pd.DataFrame({"pid": [1, 2], "doi": ["10.1", "10.2"]})
    g.add_connections(edges, "AUTHORED", "Person", "pid", "Paper", "doi")

    return g


# ── API basics ───────────────────────────────────────────────────────────────


class TestSchemaLockAPI:
    def test_default_unlocked(self):
        g = kglite.KnowledgeGraph()
        assert g.schema_locked is False

    def test_lock_unlock_toggle(self):
        g = _make_graph()
        g.lock_schema()
        assert g.schema_locked is True
        g.unlock_schema()
        assert g.schema_locked is False

    def test_lock_returns_self(self):
        g = _make_graph()
        result = g.lock_schema()
        assert result is not None  # returns Self for chaining


# ── CREATE node validation ───────────────────────────────────────────────────


class TestCreateNodeValidation:
    def test_create_valid_node(self):
        g = _make_graph()
        g.lock_schema()
        g.cypher("CREATE (p:Person {name: 'Carol', age: 35})")

    def test_create_unknown_node_type(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.KgError, match="Unknown node type 'Persom'"):
            g.cypher("CREATE (p:Persom {name: 'x'})")

    def test_create_unknown_type_suggests(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.KgError, match="Did you mean 'Person'"):
            g.cypher("CREATE (p:Persom {name: 'x'})")

    def test_create_unknown_property(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.KgError, match="Unknown property 'agee' on Person"):
            g.cypher("CREATE (p:Person {name: 'x', agee: 30})")

    def test_create_unknown_property_suggests(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.KgError, match="Did you mean 'age'"):
            g.cypher("CREATE (p:Person {name: 'x', agee: 30})")

    def test_create_type_mismatch(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.KgError, match="expects integer, got string"):
            g.cypher("CREATE (p:Person {name: 'x', age: 'thirty'})")


# ── CREATE edge validation ───────────────────────────────────────────────────


class TestCreateEdgeValidation:
    def test_create_valid_edge(self):
        g = _make_graph()
        g.lock_schema()
        g.cypher("""
            MATCH (p:Person {name: 'Alice'}), (pa:Paper {title: 'Paper B'})
            CREATE (p)-[:AUTHORED]->(pa)
        """)

    def test_create_unknown_edge_type(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.KgError, match="Unknown edge type 'WRITES'"):
            g.cypher("""
                MATCH (p:Person {name: 'Alice'}), (pa:Paper {title: 'Paper B'})
                CREATE (p)-[:WRITES]->(pa)
            """)

    def test_create_invalid_edge_endpoints(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.KgError, match="AUTHORED edges connect"):
            g.cypher("""
                MATCH (a:Paper {title: 'Paper A'}), (b:Paper {title: 'Paper B'})
                CREATE (a)-[:AUTHORED]->(b)
            """)


# ── SET validation ───────────────────────────────────────────────────────────


class TestSetValidation:
    def test_set_valid_property(self):
        g = _make_graph()
        g.lock_schema()
        g.cypher("MATCH (p:Person {name: 'Alice'}) SET p.age = 31")

    def test_set_unknown_property(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.KgError, match="Unknown property 'salary' on Person"):
            g.cypher("MATCH (p:Person {name: 'Alice'}) SET p.salary = 100000")

    def test_set_type_mismatch(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.KgError, match="expects integer, got string"):
            g.cypher("MATCH (p:Person {name: 'Alice'}) SET p.age = 'old'")

    def test_set_title_always_allowed(self):
        g = _make_graph()
        g.lock_schema()
        g.cypher("MATCH (p:Person {name: 'Alice'}) SET p.title = 'Dr.'")

    def test_set_name_always_allowed(self):
        g = _make_graph()
        g.lock_schema()
        g.cypher("MATCH (p:Person {name: 'Alice'}) SET p.name = 'Alicia'")


# ── MERGE validation ────────────────────────────────────────────────────────


class TestMergeValidation:
    def test_merge_valid_existing(self):
        g = _make_graph()
        g.lock_schema()
        # Should match existing node — no creation needed
        g.cypher("MERGE (p:Person {name: 'Alice'})")

    def test_merge_valid_new(self):
        g = _make_graph()
        g.lock_schema()
        # Should create new node — valid type and properties
        g.cypher("MERGE (p:Person {name: 'Zara', age: 28})")

    def test_merge_unknown_type(self):
        g = _make_graph()
        g.lock_schema()
        with pytest.raises(kglite.KgError, match="Unknown node type 'Auther'"):
            g.cypher("MERGE (a:Auther {name: 'x'})")


# ── Unlock behavior ─────────────────────────────────────────────────────────


class TestUnlock:
    def test_unlock_allows_unknown_type(self):
        g = _make_graph()
        g.lock_schema()
        g.unlock_schema()
        # Should succeed now — schema unlocked
        g.cypher("CREATE (x:NewType {name: 'anything'})")

    def test_reads_always_allowed(self):
        g = _make_graph()
        g.lock_schema()
        # MATCH should always work regardless of lock
        result = g.cypher("MATCH (p:Person) RETURN p.name")
        assert len(result) == 2

    def test_delete_always_allowed(self):
        g = _make_graph()
        g.lock_schema()
        # DELETE should work even when schema locked
        g.cypher("MATCH (p:Person {name: 'Bob'}) DETACH DELETE p")


# ── Introspection ────────────────────────────────────────────────────────────


class TestIntrospection:
    def test_describe_shows_schema_locked(self):
        g = _make_graph()
        g.lock_schema()
        desc = g.describe()
        assert "schema-locked" in desc

    def test_describe_no_notice_when_unlocked(self):
        g = _make_graph()
        desc = g.describe()
        assert "schema-locked" not in desc
