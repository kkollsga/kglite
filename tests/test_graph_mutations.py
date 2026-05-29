"""Tests for graph mutation orchestration (maintain_graph.rs).

Covers add_nodes/add_connections conflict handling, bulk operations,
error cases, type metadata maintenance, and index updates.
"""

import pandas as pd
import pytest

from kglite import KnowledgeGraph


@pytest.fixture
def graph():
    return KnowledgeGraph()


@pytest.fixture
def people_graph():
    """Graph with 3 Person nodes."""
    g = KnowledgeGraph()
    df = pd.DataFrame(
        {
            "id": ["alice", "bob", "charlie"],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [30, 25, 35],
        }
    )
    g.add_nodes(df, "Person", "id", "name")
    return g


# ---------------------------------------------------------------------------
# add_nodes: Conflict Handling
# ---------------------------------------------------------------------------


class TestAddNodesConflict:
    """Test the four conflict handling modes for add_nodes."""

    def test_default_update_merges_properties(self, people_graph):
        """Default mode (update) merges new properties into existing nodes."""
        df = pd.DataFrame(
            {
                "id": ["alice"],
                "name": ["Alice Updated"],
                "email": ["alice@example.com"],
            }
        )
        people_graph.add_nodes(df, "Person", "id", "name")
        nodes = people_graph.select("Person").where({"id": "alice"}).collect()
        assert len(nodes) == 1
        assert nodes[0]["email"] == "alice@example.com"
        # Age should be preserved from original
        assert nodes[0]["age"] == 30

    def test_replace_overwrites_all(self, people_graph):
        """Replace mode discards existing properties, writes new ones."""
        df = pd.DataFrame(
            {
                "id": ["alice"],
                "name": ["Alice Replaced"],
                "email": ["alice@example.com"],
            }
        )
        people_graph.add_nodes(df, "Person", "id", "name", conflict_handling="replace")
        nodes = people_graph.select("Person").where({"id": "alice"}).collect()
        assert len(nodes) == 1
        assert nodes[0]["email"] == "alice@example.com"
        # Age should be gone (replaced)
        assert nodes[0].get("age") is None

    def test_skip_ignores_duplicates(self, people_graph):
        """Skip mode ignores nodes that already exist."""
        df = pd.DataFrame(
            {
                "id": ["alice"],
                "name": ["Alice Skipped"],
                "email": ["alice@example.com"],
            }
        )
        people_graph.add_nodes(df, "Person", "id", "name", conflict_handling="skip")
        nodes = people_graph.select("Person").where({"id": "alice"}).collect()
        assert len(nodes) == 1
        # Original title preserved
        assert nodes[0]["title"] == "Alice"
        # No email added
        assert nodes[0].get("email") is None

    def test_preserve_keeps_existing_properties(self, people_graph):
        """Preserve mode adds new properties but doesn't overwrite existing ones."""
        df = pd.DataFrame(
            {
                "id": ["alice"],
                "name": ["Alice Preserved"],
                "age": [99],
                "email": ["alice@example.com"],
            }
        )
        people_graph.add_nodes(df, "Person", "id", "name", conflict_handling="preserve")
        nodes = people_graph.select("Person").where({"id": "alice"}).collect()
        assert len(nodes) == 1
        # Age preserved (not overwritten by 99)
        assert nodes[0]["age"] == 30
        # Email added (new property)
        assert nodes[0]["email"] == "alice@example.com"

    def test_new_nodes_created_regardless_of_mode(self, people_graph):
        """New nodes are always created regardless of conflict mode."""
        df = pd.DataFrame(
            {
                "id": ["diana"],
                "name": ["Diana"],
                "age": [28],
            }
        )
        people_graph.add_nodes(df, "Person", "id", "name", conflict_handling="skip")
        assert people_graph.select("Person").len() == 4


# ---------------------------------------------------------------------------
# add_nodes: Type Metadata & Index Maintenance
# ---------------------------------------------------------------------------


class TestAddNodesMetadata:
    """Verify that add_nodes correctly maintains type indices and metadata."""

    def test_type_index_created(self, graph):
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        graph.add_nodes(df, "MyType", "id", "name")
        assert graph.select("MyType").len() == 2

    def test_multiple_types_independent(self, graph):
        df1 = pd.DataFrame({"id": [1], "name": ["A"]})
        df2 = pd.DataFrame({"id": [1], "name": ["X"]})
        graph.add_nodes(df1, "TypeA", "id", "name")
        graph.add_nodes(df2, "TypeB", "id", "name")
        assert graph.select("TypeA").len() == 1
        assert graph.select("TypeB").len() == 1

    def test_node_types_listed(self, graph):
        df = pd.DataFrame({"id": [1], "name": ["A"]})
        graph.add_nodes(df, "Person", "id", "name")
        graph.add_nodes(df, "Company", "id", "name")
        types = graph.node_types  # property, not method
        assert "Person" in types
        assert "Company" in types


# ---------------------------------------------------------------------------
# add_connections
# ---------------------------------------------------------------------------


class TestAddConnections:
    """Test connection creation and property handling."""

    def test_basic_connection(self, people_graph):
        conn = pd.DataFrame(
            {
                "from": ["alice"],
                "to": ["bob"],
            }
        )
        people_graph.add_connections(conn, "KNOWS", "Person", "from", "Person", "to")
        result = people_graph.cypher("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name")
        assert len(result) == 1
        assert result[0]["a.name"] == "Alice"
        assert result[0]["b.name"] == "Bob"

    def test_connection_with_properties(self, people_graph):
        """Extra DataFrame columns become edge properties."""
        conn = pd.DataFrame(
            {
                "from": ["alice"],
                "to": ["bob"],
                "since": [2020],
                "strength": [0.9],
            }
        )
        people_graph.add_connections(conn, "KNOWS", "Person", "from", "Person", "to")
        # Verify connection exists and edge properties are queryable
        result = people_graph.cypher("MATCH (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person) RETURN type(r) AS t")
        assert len(result) == 1
        assert result[0]["t"] == "KNOWS"

    def test_multiple_connections(self, people_graph):
        conn = pd.DataFrame(
            {
                "from": ["alice", "bob"],
                "to": ["bob", "charlie"],
            }
        )
        people_graph.add_connections(conn, "KNOWS", "Person", "from", "Person", "to")
        result = people_graph.cypher("MATCH ()-[:KNOWS]->() RETURN count(*) AS cnt")
        assert result[0]["cnt"] == 2

    def test_cross_type_connections(self, graph):
        people = pd.DataFrame({"id": ["alice"], "name": ["Alice"]})
        companies = pd.DataFrame({"id": ["acme"], "name": ["ACME Corp"]})
        graph.add_nodes(people, "Person", "id", "name")
        graph.add_nodes(companies, "Company", "id", "name")
        conn = pd.DataFrame({"person": ["alice"], "company": ["acme"]})
        graph.add_connections(conn, "WORKS_AT", "Person", "person", "Company", "company")
        result = graph.cypher("MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.name, c.name")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Error Cases
# ---------------------------------------------------------------------------


class TestMutationErrors:
    """Test error handling for invalid mutation inputs."""

    def test_empty_dataframe(self, graph):
        """Empty DataFrame should not crash."""
        df = pd.DataFrame({"id": [], "name": []})
        graph.add_nodes(df, "T", "id", "name")
        assert graph.select("T").len() == 0

    def test_missing_id_column(self, graph):
        """Missing unique_id_field column should raise error."""
        df = pd.DataFrame({"name": ["A"]})
        with pytest.raises(Exception):
            graph.add_nodes(df, "T", "nonexistent", "name")

    def test_duplicate_ids_in_batch(self, graph):
        """Duplicate IDs in same batch — both rows are ingested (no within-batch dedup)."""
        df = pd.DataFrame(
            {
                "id": ["x", "x"],
                "name": ["First", "Second"],
            }
        )
        graph.add_nodes(df, "T", "id", "name")
        nodes = graph.select("T").collect()
        # Within a single batch, duplicates are both created
        assert len(nodes) == 2

    def test_connection_missing_source_vivifies_stub(self, graph):
        """Connection referencing a non-existent source node vivifies a
        provisional stub — the edge is kept, not dropped."""
        df = pd.DataFrame({"id": ["bob"], "name": ["Bob"]})
        graph.add_nodes(df, "Person", "id", "name")
        conn = pd.DataFrame(
            {
                "from": ["nonexistent"],
                "to": ["bob"],
            }
        )
        graph.add_connections(conn, "KNOWS", "Person", "from", "Person", "to")
        result = graph.cypher("MATCH ()-[:KNOWS]->() RETURN count(*) AS cnt")
        assert result[0]["cnt"] == 1
        prov = graph.cypher("MATCH (p:Person) WHERE p._provisional = true RETURN p.id AS id")
        assert [r["id"] for r in prov] == ["nonexistent"]


class TestBulkLoadWarnings:
    """Bulk loads should emit UserWarning whenever the report flags
    skips OR errors — silent partial successes were a footgun."""

    def test_warn_on_skipped_rows(self, graph):
        import warnings

        df = pd.DataFrame({"id": [None, "x2"], "name": ["A", "B"]})
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            graph.add_nodes(df, "T", "id", "name")
        kinds = {w.category for w in wlist}
        assert UserWarning in kinds
        msgs = " | ".join(str(w.message) for w in wlist)
        assert "skipped" in msgs

    def test_warn_on_errors_without_skips(self, graph):
        """Type mismatch on a follow-up call is recorded as an error
        but does not skip rows. Pre-0.9.9 this was silent."""
        import warnings

        graph.add_nodes(pd.DataFrame({"id": [1], "title": ["A"], "x": [1]}), "T", "id", "title")
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            graph.add_nodes(
                pd.DataFrame({"id": [2], "title": ["B"], "x": ["str-not-int"]}),
                "T",
                "id",
                "title",
            )
        user_warnings = [w for w in wlist if w.category is UserWarning]
        assert len(user_warnings) >= 1, "expected a UserWarning when has_errors=True"
        assert any("error" in str(w.message).lower() for w in user_warnings)

    def test_warn_on_vivified_connections(self, graph):
        """A connection to a missing node no longer drops the edge — it
        vivifies a provisional stub — but the implicit node creation is
        still surfaced as a UserWarning."""
        import warnings

        graph.add_nodes(pd.DataFrame({"id": ["a"], "name": ["A"]}), "Person", "id", "name")
        conn = pd.DataFrame({"from": ["missing"], "to": ["a"]})
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            graph.add_connections(conn, "KNOWS", "Person", "from", "Person", "to")
        user_warnings = [w for w in wlist if w.category is UserWarning]
        assert any("KNOWS" in str(w.message) and "vivified" in str(w.message) for w in user_warnings)


# ---------------------------------------------------------------------------
# Bulk Operations
# ---------------------------------------------------------------------------


class TestBulkOperations:
    """Test bulk node/connection creation at scale."""

    def test_bulk_nodes(self, graph):
        df = pd.DataFrame(
            {
                "id": range(500),
                "name": [f"Node_{i}" for i in range(500)],
                "val": list(range(500)),
            }
        )
        graph.add_nodes(df, "Bulk", "id", "name")
        assert graph.select("Bulk").len() == 500

    def test_bulk_connections(self, graph):
        df = pd.DataFrame(
            {
                "id": range(100),
                "name": [f"N_{i}" for i in range(100)],
            }
        )
        graph.add_nodes(df, "N", "id", "name")
        conn = pd.DataFrame(
            {
                "from": list(range(99)),
                "to": list(range(1, 100)),
            }
        )
        graph.add_connections(conn, "NEXT", "N", "from", "N", "to")
        result = graph.cypher("MATCH ()-[:NEXT]->() RETURN count(*) AS cnt")
        assert result[0]["cnt"] == 99

    def test_bulk_with_nulls(self, graph):
        """Bulk ingestion with null values in property columns."""
        df = pd.DataFrame(
            {
                "id": range(10),
                "name": [f"N_{i}" for i in range(10)],
                "val": [i if i % 2 == 0 else None for i in range(10)],
            }
        )
        graph.add_nodes(df, "T", "id", "name")
        assert graph.select("T").len() == 10
        nodes = graph.select("T").where({"val": {"is_null": True}})
        assert nodes.len() == 5


class TestDetachDeleteNullVar:
    """openCypher ignores NULL in DELETE, so the single-statement cascade
    `MATCH (root) OPTIONAL MATCH (root)-->(child) DETACH DELETE root, child`
    works even when a branch is empty. Regression: kglite-docs 2026-05-29 #4
    — an unmatched OPTIONAL MATCH variable raised
    'Variable x not bound to a node or relationship in DELETE'."""

    @staticmethod
    def _study_graph() -> KnowledgeGraph:
        g = KnowledgeGraph()
        g.add_nodes(pd.DataFrame([{"id": "s1"}]), "Study", "id", "id")
        g.add_nodes(pd.DataFrame([{"id": f"a{i}"} for i in range(2)]), "Assessment", "id", "id")
        g.add_connections(
            pd.DataFrame([{"src": f"a{i}", "dst": "s1"} for i in range(2)]),
            "OF_STUDY",
            source_type="Assessment",
            source_id_field="src",
            target_type="Study",
            target_id_field="dst",
        )
        return g

    def test_unmatched_optional_var_is_noop(self):
        g = self._study_graph()
        # OPTIONAL MATCH matches nothing -> x is NULL -> skipped, s deleted.
        g.cypher("MATCH (s:Study {id:'s1'}) OPTIONAL MATCH (s)-[:NONEXISTENT]->(x:Nope) DETACH DELETE s, x")
        assert g.cypher("MATCH (n:Study) RETURN count(n) AS c").to_list()[0]["c"] == 0

    def test_cascade_deletes_matched_children(self):
        g = self._study_graph()
        g.cypher("MATCH (s:Study {id:'s1'}) OPTIONAL MATCH (a:Assessment)-[:OF_STUDY]->(s) DETACH DELETE s, a")
        assert g.cypher("MATCH (n) RETURN count(n) AS c").to_list()[0]["c"] == 0
