"""Regression tests for v0.4.11 bugfixes.

Bug 1: labels(n)[0] - parser had no index expression support
Bug 2: nodes(p)/relationships(p) returned strings instead of Python lists
Bug 3: Multi-hop Cypher with OPTIONAL MATCH + chained WITH hung indefinitely
"""

import pytest

from kglite import KnowledgeGraph


@pytest.fixture
def social_graph():
    """Small social graph: Alice->Bob->Charlie, Alice->Charlie."""
    g = KnowledgeGraph()
    for name in ["Alice", "Bob", "Charlie"]:
        g.cypher(f"CREATE (:Person {{name: '{name}'}})")
    g.cypher("MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[:KNOWS]->(b)")
    g.cypher("MATCH (a:Person {name: 'Bob'}), (b:Person {name: 'Charlie'}) CREATE (a)-[:KNOWS]->(b)")
    g.cypher("MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Charlie'}) CREATE (a)-[:FRIENDS]->(b)")
    return g


# ============================================================================
# Bug 1: labels(n)[0] and index expressions
# ============================================================================


class TestLabelsFunction:
    """labels(n) should return a list; labels(n)[0] should work."""

    def test_labels_returns_list(self, social_graph):
        rows = social_graph.cypher("MATCH (n:Person {name: 'Alice'}) RETURN labels(n)")
        labels = rows[0]["labels(n)"]
        assert isinstance(labels, list)
        assert labels == ["Person"]

    def test_labels_index_zero(self, social_graph):
        rows = social_graph.cypher("MATCH (n:Person {name: 'Alice'}) RETURN labels(n)[0] AS label")
        assert rows[0]["label"] == "Person"

    def test_labels_in_where(self, social_graph):
        rows = social_graph.cypher("MATCH (n:Person) WHERE labels(n)[0] = 'Person' RETURN n.name ORDER BY n.name")
        names = [r["n.name"] for r in rows]
        assert names == ["Alice", "Bob", "Charlie"]

    # --- Track-C swap-point lock-in (B5): pin the consumer invariants ---

    def test_in_labels_matches(self, social_graph):
        """'Label' IN labels(n) works through the JSON-string encoding."""
        rows = social_graph.cypher("MATCH (n) WHERE 'Person' IN labels(n) RETURN count(n) AS c")
        assert rows[0]["c"] == 3

    def test_in_labels_does_not_match(self, social_graph):
        rows = social_graph.cypher("MATCH (n) WHERE 'Nope' IN labels(n) RETURN count(n) AS c")
        assert rows[0]["c"] == 0

    def test_size_of_labels(self, social_graph):
        """Single-label model: size(labels(n)) == 1 for every node."""
        rows = social_graph.cypher("MATCH (n:Person) RETURN size(labels(n)) AS s LIMIT 1")
        assert rows[0]["s"] == 1

    def test_labels_escape_double_quote(self):
        """JSON-escape path handles labels containing `"` correctly."""
        g = KnowledgeGraph()
        g.cypher('CREATE (n:`Has"Quote` {name: "x"})')
        rows = g.cypher('MATCH (n) WHERE n.name = "x" RETURN labels(n) AS l')
        assert rows[0]["l"] == ['Has"Quote']

    def test_labels_escape_backslash(self):
        """JSON-escape path handles labels containing `\\` correctly."""
        g = KnowledgeGraph()
        g.cypher(r'CREATE (n:`Has\Backslash` {name: "x"})')
        rows = g.cypher('MATCH (n) WHERE n.name = "x" RETURN labels(n) AS l')
        assert rows[0]["l"] == [r"Has\Backslash"]


class TestIndexAccess:
    """Index access expr[i] on various list expressions."""

    def test_list_literal_index(self):
        g = KnowledgeGraph()
        rows = g.cypher("UNWIND [1] AS d RETURN [10, 20, 30][0] AS val")
        assert rows[0]["val"] == 10

    def test_list_literal_index_middle(self):
        g = KnowledgeGraph()
        rows = g.cypher("UNWIND [1] AS d RETURN [10, 20, 30][1] AS val")
        assert rows[0]["val"] == 20

    def test_negative_index(self):
        g = KnowledgeGraph()
        rows = g.cypher("UNWIND [1] AS d RETURN [10, 20, 30][-1] AS val")
        assert rows[0]["val"] == 30

    def test_out_of_bounds_returns_null(self):
        g = KnowledgeGraph()
        rows = g.cypher("UNWIND [1] AS d RETURN [10, 20, 30][99] AS val")
        assert rows[0]["val"] is None

    def test_collect_then_index(self, social_graph):
        rows = social_graph.cypher("MATCH (n:Person) WITH collect(n.name) AS names RETURN names[0] AS first")
        # collect order is not guaranteed, but should return a valid name
        assert rows[0]["first"] in ["Alice", "Bob", "Charlie"]

    def test_column_name_for_index(self):
        g = KnowledgeGraph()
        rows = g.cypher("UNWIND [1] AS d RETURN [1, 2][0]")
        assert "[1, 2][0]" in rows[0]

    def test_index_on_function_result(self, social_graph):
        """Index on function call result like labels(n)[0]."""
        rows = social_graph.cypher("MATCH (n:Person {name: 'Bob'}) RETURN labels(n)[0] AS lbl")
        assert rows[0]["lbl"] == "Person"


# ============================================================================
# Bug 2: nodes(p)/relationships(p) return native Python types
# ============================================================================


class TestPathFunctionReturnTypes:
    """nodes(p) should return list of dicts, relationships(p) list of strings."""

    def test_nodes_returns_list_of_dicts(self, social_graph):
        # Phase A.1 / C6 — nodes() returns full Node dicts with
        # id/labels/properties (was JSON-string-of-dicts pre-A.1).
        rows = social_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..5]->(b:Person {name: 'Charlie'})) "
            "RETURN nodes(p)"
        )
        nodes = rows[0]["nodes(p)"]
        assert isinstance(nodes, list)
        assert all(isinstance(n, dict) for n in nodes)
        titles = [n["properties"]["title"] for n in nodes]
        assert "Alice" in titles
        assert "Charlie" in titles

    def test_relationships_returns_list_of_strings(self, social_graph):
        # Phase A.1 / C6 — relationships() returns full Rel dicts
        # (was list of type strings pre-A.1). Extract `.type` to get
        # the prior shape.
        rows = social_graph.cypher(
            "MATCH p = shortestPath((a:Person {name: 'Alice'})-[:KNOWS*..5]->(b:Person {name: 'Bob'})) "
            "RETURN relationships(p)"
        )
        rels = rows[0]["relationships(p)"]
        assert isinstance(rels, list)
        assert all(isinstance(r, dict) for r in rels)
        assert [r["type"] for r in rels] == ["KNOWS"]

    def test_collect_returns_list(self, social_graph):
        rows = social_graph.cypher("MATCH (n:Person) RETURN collect(n.name) AS names")
        names = rows[0]["names"]
        assert isinstance(names, list)
        assert set(names) == {"Alice", "Bob", "Charlie"}

    def test_list_comprehension_returns_list(self):
        g = KnowledgeGraph()
        rows = g.cypher("UNWIND [1] AS d RETURN [x IN [1, 2, 3] | x * 10] AS tens")
        tens = rows[0]["tens"]
        assert isinstance(tens, list)
        assert tens == [10, 20, 30]


# ============================================================================
# Bug 3: WITH + OPTIONAL MATCH hang
# ============================================================================


class TestWithOptionalMatchHang:
    """Chained WITH + OPTIONAL MATCH should not hang."""

    def test_with_optional_match_basic(self, social_graph):
        """WITH piping into OPTIONAL MATCH should complete quickly."""
        rows = social_graph.cypher("""
            MATCH (p:Person {name: 'Alice'})
            WITH p
            OPTIONAL MATCH (p)-[:KNOWS]->(friend:Person)
            RETURN p.name, friend.name
            ORDER BY friend.name
        """)
        assert len(rows) >= 1
        assert rows[0]["p.name"] == "Alice"

    def test_with_aggregation_then_optional_match(self, social_graph):
        """WITH aggregation then OPTIONAL MATCH (the original hang scenario)."""
        rows = social_graph.cypher("""
            MATCH (p:Person)
            WITH p, count(*) AS cnt
            OPTIONAL MATCH (p)-[:KNOWS]->(friend:Person)
            RETURN p.name, cnt, friend.name
            ORDER BY p.name
        """)
        # Should have results for all 3 people
        names = sorted(set(r["p.name"] for r in rows))
        assert names == ["Alice", "Bob", "Charlie"]

    def test_multiple_with_chains(self, social_graph):
        """Multiple WITH clauses chained together."""
        rows = social_graph.cypher("""
            MATCH (p:Person {name: 'Alice'})
            WITH p
            WITH p, 'hello' AS greeting
            RETURN p.name, greeting
        """)
        assert rows[0]["p.name"] == "Alice"
        assert rows[0]["greeting"] == "hello"


# ============================================================================
# Performance: pre-binding push-down for OPTIONAL MATCH / subsequent MATCH
# ============================================================================


class TestPreBindingOptimization:
    """Verify pre-bound variables are correctly constrained in pattern execution."""

    def test_with_count_then_optional_match(self):
        """WITH aggregation -> OPTIONAL MATCH returns correct counts (the user's bug report)."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Company {name: 'Equinor'})")
        g.cypher("CREATE (:Company {name: 'Shell'})")
        g.cypher("CREATE (:Well {name: 'W1'})")
        g.cypher("CREATE (:Well {name: 'W2'})")
        g.cypher("CREATE (:Well {name: 'W3'})")
        g.cypher("CREATE (:Licence {name: 'L1'})")
        g.cypher("MATCH (c:Company {name: 'Equinor'}), (w:Well {name: 'W1'}) CREATE (w)-[:DRILLED_BY]->(c)")
        g.cypher("MATCH (c:Company {name: 'Equinor'}), (w:Well {name: 'W2'}) CREATE (w)-[:DRILLED_BY]->(c)")
        g.cypher("MATCH (c:Company {name: 'Shell'}), (w:Well {name: 'W3'}) CREATE (w)-[:DRILLED_BY]->(c)")
        g.cypher("MATCH (c:Company {name: 'Equinor'}), (l:Licence {name: 'L1'}) CREATE (l)-[:LICENSED_TO]->(c)")

        rows = g.cypher("""
            MATCH (c:Company)<-[:DRILLED_BY]-(w:Well)
            WITH c, count(w) AS wells
            OPTIONAL MATCH (c)<-[:LICENSED_TO]-(l:Licence)
            WITH c, wells, count(l) AS licences
            RETURN c.name, wells, licences ORDER BY wells DESC
        """)
        equinor = next(r for r in rows if r["c.name"] == "Equinor")
        shell = next(r for r in rows if r["c.name"] == "Shell")
        assert equinor["wells"] == 2
        assert equinor["licences"] == 1
        assert shell["wells"] == 1
        assert shell["licences"] == 0

    def test_subsequent_match_uses_bindings(self):
        """Regular MATCH after WITH should also use pre-bindings."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice'})")
        g.cypher("CREATE (:Person {name: 'Bob'})")
        g.cypher("CREATE (:City {name: 'Oslo'})")
        g.cypher("CREATE (:City {name: 'London'})")
        g.cypher("MATCH (a:Person {name: 'Alice'}), (c:City {name: 'Oslo'}) CREATE (a)-[:LIVES_IN]->(c)")
        g.cypher("MATCH (a:Person {name: 'Bob'}), (c:City {name: 'London'}) CREATE (a)-[:LIVES_IN]->(c)")

        rows = g.cypher("""
            MATCH (p:Person)
            WITH p
            MATCH (p)-[:LIVES_IN]->(c:City)
            RETURN p.name, c.name ORDER BY p.name
        """)
        assert len(rows) == 2
        assert rows[0]["p.name"] == "Alice"
        assert rows[0]["c.name"] == "Oslo"
        assert rows[1]["p.name"] == "Bob"
        assert rows[1]["c.name"] == "London"

    def test_sequential_match_rebinds_variable(self):
        """Two separate MATCH clauses re-binding a variable must use the prior anchor.

        Regression for the multi-MATCH planner bug found on Wikidata 2026-04-20:
        the inverted-index fast path in pattern_matching/matcher.rs was firing
        on bare re-bindings like `MATCH (f)-[:R]->(c)` even when `f` was already
        bound, returning every R-source in the graph (124M+ on Wikidata) instead
        of just the bound node.
        """
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice'})")
        g.cypher("CREATE (:City {name: 'Oslo'})")
        g.cypher("MATCH (a:Person {name: 'Alice'}), (c:City {name: 'Oslo'}) CREATE (a)-[:LIVES_IN]->(c)")

        # Two separate MATCH clauses, no WITH between — the failing case.
        rows = g.cypher("""
            MATCH (p:Person {name: 'Alice'})
            MATCH (p)-[:LIVES_IN]->(c:City)
            RETURN p.name, c.name
        """)
        assert len(rows) == 1
        assert rows[0]["p.name"] == "Alice"
        assert rows[0]["c.name"] == "Oslo"

    def test_sequential_match_rebind_perf_no_full_scan(self):
        """Perf-regression guard: re-bound MATCH must not scan the whole graph.

        Build a graph where every node has a :REL edge — a buggy planner would
        scan all of them and take 100x longer than the prebound-anchor path.
        """
        import time

        g = KnowledgeGraph()
        # 5_000 Source nodes, each with a :REL to a single Target.
        for i in range(5_000):
            g.cypher(f"CREATE (:Source {{sid: {i}}})")
        g.cypher("CREATE (:Target {name: 'T'})")
        g.cypher("MATCH (s:Source), (t:Target) WHERE s.sid < 5000 CREATE (s)-[:REL]->(t)")
        # Pick one Source to anchor on.
        anchor_id = g.cypher("MATCH (s:Source {sid: 42}) RETURN s.id AS sid")[0]["sid"]

        start = time.perf_counter()
        rows = g.cypher(f"MATCH (s {{id: {anchor_id}}}) MATCH (s)-[:REL]->(t:Target) RETURN t.name AS tname")
        elapsed = time.perf_counter() - start

        assert len(rows) == 1
        assert rows[0]["tname"] == "T"
        # On a 5K-edge graph, anchored lookup is sub-millisecond. A full scan
        # would be ~tens of ms. 200ms is a generous ceiling that catches the
        # regression while tolerating CI noise.
        assert elapsed < 0.2, f"Re-bound MATCH took {elapsed * 1000:.1f}ms — likely full-scanning"

    def test_optional_match_null_for_unmatched(self):
        """OPTIONAL MATCH produces nulls when no match, not cross-products."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice'})")
        g.cypher("CREATE (:Person {name: 'Bob'})")
        g.cypher("CREATE (:Pet {name: 'Rex'})")
        g.cypher("MATCH (a:Person {name: 'Alice'}), (p:Pet {name: 'Rex'}) CREATE (a)-[:OWNS]->(p)")

        rows = g.cypher("""
            MATCH (p:Person)
            WITH p
            OPTIONAL MATCH (p)-[:OWNS]->(pet:Pet)
            RETURN p.name, pet.name ORDER BY p.name
        """)
        assert len(rows) == 2
        alice = next(r for r in rows if r["p.name"] == "Alice")
        bob = next(r for r in rows if r["p.name"] == "Bob")
        assert alice["pet.name"] == "Rex"
        assert bob["pet.name"] is None
