"""Regression tests for the deferred-property-materialization optimization in
GROUP BY aggregation (return_clause.rs).

Background: queries like `RETURN x.title, count(*)` on Wikidata-scale graphs
were doing O(input rows) random-I/O property reads to build group keys, even
when the result collapsed to dozens of distinct groups. The optimization
hashes by NodeIndex during the per-row pass, then resolves property values
once per resulting group. This file pins both correctness (Cypher semantics
preserved) and perf (no O(rows) regression on the in-memory path).
"""

import time

import pytest

from kglite import KnowledgeGraph

# ============================================================================
# Correctness: same Cypher semantics, regardless of internal grouping path
# ============================================================================


class TestDeferredGroupByCorrectness:
    """The optimization must preserve exact Cypher GROUP BY semantics."""

    def test_duplicate_property_values_collapse_to_one_group(self):
        """Two distinct nodes with the same name must group together.

        Cypher GROUP BY x.name treats two Person(name='Alice') nodes as one
        group. The surrogate-NodeIndex grouping must re-bucket on resolved
        Value before emitting, otherwise this returns 3 rows instead of 2.
        """
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice'})")
        g.cypher("CREATE (:Person {name: 'Alice'})")  # second Alice — different node
        g.cypher("CREATE (:Person {name: 'Bob'})")

        rows = g.cypher("MATCH (p:Person) RETURN p.name AS name, count(*) AS n ORDER BY n DESC")
        assert len(rows) == 2
        by_name = {r["name"]: r["n"] for r in rows}
        assert by_name == {"Alice": 2, "Bob": 1}

    def test_relationship_count_cache_tracks_create_and_delete(self):
        """A warmed grouped-count cache must never survive edge mutations."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Source {sid: 1}), (:Source {sid: 2})")
        g.cypher("CREATE (:Group {gid: 1, name: 'A'}), (:Group {gid: 2, name: 'B'})")
        g.cypher("MATCH (s:Source {sid: 1}), (t:Group {gid: 1}) CREATE (s)-[:RELATES_TO]->(t)")

        query = "MATCH (s:Source)-[:RELATES_TO]->(t:Group) RETURN t.name AS name, count(s) AS n ORDER BY n DESC"
        assert g.cypher(query).to_dicts() == [{"name": "A", "n": 1}]
        # The second read is served by the warmed per-type histogram.
        assert g.cypher(query).to_dicts() == [{"name": "A", "n": 1}]

        g.cypher("MATCH (s:Source {sid: 2}), (t:Group {gid: 1}) CREATE (s)-[:RELATES_TO]->(t)")
        assert g.cypher(query).to_dicts() == [{"name": "A", "n": 2}]

        g.cypher("MATCH (:Source {sid: 2})-[r:RELATES_TO]->(:Group {gid: 1}) DELETE r")
        assert g.cypher(query).to_dicts() == [{"name": "A", "n": 1}]

    def test_optional_match_null_groups_under_null_key(self):
        """Variable that's null for some rows (OPTIONAL MATCH) must group as Null.

        The fallback path: when row.node_bindings doesn't contain the variable,
        the surrogate strategy falls back to per-row evaluation, which yields
        Value::Null. All such rows must collapse into a single null group.
        """
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice'})")
        g.cypher("CREATE (:Person {name: 'Bob'})")
        g.cypher("CREATE (:Pet {name: 'Rex'})")
        g.cypher("MATCH (a:Person {name: 'Alice'}), (p:Pet {name: 'Rex'}) CREATE (a)-[:OWNS]->(p)")

        rows = g.cypher("""
            MATCH (p:Person)
            OPTIONAL MATCH (p)-[:OWNS]->(pet:Pet)
            RETURN pet.name AS pname, count(*) AS n
        """)
        # Alice → Rex, Bob → null. Expect two groups: 'Rex' and null (each count=1).
        counts = {r["pname"]: r["n"] for r in rows}
        assert counts == {"Rex": 1, None: 1}

    def test_multiple_grouping_keys(self):
        """Compound GROUP BY (multiple non-aggregate items) still groups correctly."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Sale {customer: 'Alice', status: 'paid', amount: 10})")
        g.cypher("CREATE (:Sale {customer: 'Alice', status: 'paid', amount: 20})")
        g.cypher("CREATE (:Sale {customer: 'Alice', status: 'pending', amount: 5})")
        g.cypher("CREATE (:Sale {customer: 'Bob', status: 'paid', amount: 15})")

        rows = g.cypher("""
            MATCH (o:Sale)
            RETURN o.customer AS c, o.status AS s, sum(o.amount) AS total
            ORDER BY c, s
        """)
        assert len(rows) == 3
        triples = [(r["c"], r["s"], r["total"]) for r in rows]
        assert triples == [
            ("Alice", "paid", 30),
            ("Alice", "pending", 5),
            ("Bob", "paid", 15),
        ]

    def test_grouping_on_non_property_expression(self):
        """Grouping on a computed expression (not direct binding.prop) still works.

        These take the GroupExprStrategy::Eval path — the optimization should
        leave their behavior unchanged.
        """
        g = KnowledgeGraph()
        for age in [10, 15, 20, 25, 30]:
            g.cypher(f"CREATE (:Person {{age: {age}}})")

        rows = g.cypher("""
            MATCH (p:Person)
            RETURN p.age < 20 AS young, count(*) AS n
            ORDER BY young
        """)
        # Ages 10, 15 → young=true (2); 20, 25, 30 → young=false (3)
        pairs = [(r["young"], r["n"]) for r in rows]
        assert pairs == [(False, 3), (True, 2)]

    def test_grouping_preserves_node_binding_for_downstream_match(self):
        """Group-key node binding must survive into subsequent MATCH clauses.

        The post-aggregation code path preserves node_bindings on the result
        rows so chained MATCH/OPTIONAL MATCH after the aggregation can still
        anchor on the grouped variable.
        """
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice'})")
        g.cypher("CREATE (:Pet {name: 'Rex'})")
        g.cypher("CREATE (:Pet {name: 'Whiskers'})")
        g.cypher("MATCH (a:Person {name: 'Alice'}), (p:Pet {name: 'Rex'}) CREATE (a)-[:OWNS]->(p)")
        g.cypher("MATCH (a:Person {name: 'Alice'}), (p:Pet {name: 'Whiskers'}) CREATE (a)-[:OWNS]->(p)")

        rows = g.cypher("""
            MATCH (p:Person)-[:OWNS]->(pet:Pet)
            WITH p, count(pet) AS pet_count
            MATCH (p)-[:OWNS]->(pet:Pet)
            RETURN p.name AS owner, pet.name AS pet, pet_count
            ORDER BY pet
        """)
        assert len(rows) == 2
        assert all(r["owner"] == "Alice" and r["pet_count"] == 2 for r in rows)


# ============================================================================
# Performance: no O(rows) regression on in-memory path
# ============================================================================


class TestDeferredGroupByPerf:
    """Smoke perf test — fan-out aggregation must complete in bounded time."""

    def test_high_fanout_grouping_completes_quickly(self):
        """100K source nodes connected to 50 concept nodes via :IS_A.

        `RETURN c.title, count(*)` should complete in well under 1s on the
        in-memory backend. Pre-fix, this would do 100K title reads to build
        keys; post-fix, it does 50 (one per resulting group).
        """
        g = KnowledgeGraph()
        # 50 concept nodes
        g.cypher("UNWIND range(0, 49) AS i CREATE (:Concept {cid: i, label: 'c' + toString(i)})")
        # 100K source nodes
        g.cypher("UNWIND range(0, 99999) AS i CREATE (:Source {sid: i})")
        # Each source → one concept (sid % 50)
        g.cypher("""
            MATCH (s:Source), (c:Concept)
            WHERE s.sid % 50 = c.cid
            CREATE (s)-[:IS_A]->(c)
        """)

        start = time.perf_counter()
        rows = g.cypher("""
            MATCH (s:Source)-[:IS_A]->(c:Concept)
            RETURN c.label AS label, count(*) AS n
            ORDER BY n DESC LIMIT 10
        """)
        elapsed = time.perf_counter() - start

        assert len(rows) == 10
        # Each concept gets ~2000 sources (100K / 50)
        assert all(r["n"] == 2000 for r in rows)
        # Generous ceiling — in-memory should be < 100ms; CI noise tolerance.
        assert elapsed < 1.0, f"Aggregation took {elapsed * 1000:.0f}ms — likely O(rows) regression"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
