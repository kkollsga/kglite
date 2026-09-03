"""Tests for WHERE EXISTS { pattern } subpattern predicate in Cypher queries."""

import pytest

from kglite import KnowledgeGraph


@pytest.fixture
def social_graph():
    """Graph with people, some with relationships, some without."""
    graph = KnowledgeGraph()

    # Create people
    graph.cypher("CREATE (:Person {name: 'Alice', city: 'Oslo'})")
    graph.cypher("CREATE (:Person {name: 'Bob', city: 'Bergen'})")
    graph.cypher("CREATE (:Person {name: 'Charlie', city: 'Oslo'})")
    graph.cypher("CREATE (:Person {name: 'Diana', city: 'Stavanger'})")

    # Create products
    graph.cypher("CREATE (:Product {name: 'Widget', price: 10})")
    graph.cypher("CREATE (:Product {name: 'Gadget', price: 25})")

    # Alice knows Bob and Charlie
    graph.cypher("""
        MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
        CREATE (a)-[:KNOWS]->(b)
    """)
    graph.cypher("""
        MATCH (a:Person {name: 'Alice'}), (c:Person {name: 'Charlie'})
        CREATE (a)-[:KNOWS]->(c)
    """)

    # Bob knows Charlie
    graph.cypher("""
        MATCH (b:Person {name: 'Bob'}), (c:Person {name: 'Charlie'})
        CREATE (b)-[:KNOWS]->(c)
    """)

    # Alice purchased Widget
    graph.cypher("""
        MATCH (a:Person {name: 'Alice'}), (w:Product {name: 'Widget'})
        CREATE (a)-[:PURCHASED]->(w)
    """)

    # Bob purchased Gadget
    graph.cypher("""
        MATCH (b:Person {name: 'Bob'}), (g:Product {name: 'Gadget'})
        CREATE (b)-[:PURCHASED]->(g)
    """)

    return graph


class TestExistsCrossPatternRelUniqueness:
    """Relationship uniqueness (the openCypher trail rule) applies across
    the comma patterns of one EXISTS subquery, but NOT across the separate
    MATCH clauses of the multi-clause subquery form."""

    @pytest.fixture
    def one_edge_graph(self):
        graph = KnowledgeGraph()
        graph.cypher("CREATE (:P {id: 1})")
        graph.cypher("CREATE (:X {id: 1}), (:Y {id: 2})")
        graph.cypher("MATCH (x:X {id: 1}), (y:Y {id: 2}) CREATE (x)-[:R]->(y)")
        return graph

    def test_two_comma_patterns_cannot_share_the_only_edge(self, one_edge_graph):
        rows = one_edge_graph.cypher(
            "MATCH (p:P) WHERE EXISTS { (a)-[r1:R]->(b), (c)-[r2:R]->(d) } RETURN count(p) AS c"
        ).to_list()
        assert rows[0]["c"] == 0

    def test_single_pattern_still_matches(self, one_edge_graph):
        rows = one_edge_graph.cypher("MATCH (p:P) WHERE EXISTS { (a)-[r1:R]->(b) } RETURN count(p) AS c").to_list()
        assert rows[0]["c"] == 1

    def test_anonymous_comma_pattern_edges_also_enforced(self, one_edge_graph):
        rows = one_edge_graph.cypher(
            "MATCH (p:P) WHERE EXISTS { (:X)-[:R]->(), ()-[:R]->(:Y) } RETURN count(p) AS c"
        ).to_list()
        assert rows[0]["c"] == 0

    def test_two_edges_satisfy_two_comma_patterns(self, one_edge_graph):
        one_edge_graph.cypher("CREATE (:X {id: 3}), (:Y {id: 4})")
        one_edge_graph.cypher("MATCH (x:X {id: 3}), (y:Y {id: 4}) CREATE (x)-[:R]->(y)")
        rows = one_edge_graph.cypher(
            "MATCH (p:P) WHERE EXISTS { (a)-[r1:R]->(b), (c)-[r2:R]->(d) } RETURN count(p) AS c"
        ).to_list()
        assert rows[0]["c"] == 1

    def test_multi_match_subquery_clauses_may_reuse_an_edge(self, one_edge_graph):
        # Separate MATCH clauses inside EXISTS are separate clause scopes —
        # both may bind the single stored edge (as across top-level MATCHes).
        rows = one_edge_graph.cypher(
            "MATCH (p:P) WHERE EXISTS { MATCH (a)-[r1:R]->(b) MATCH (c)-[r2:R]->(d) } RETURN count(p) AS c"
        ).to_list()
        assert rows[0]["c"] == 1


class TestWhereExists:
    """Test WHERE EXISTS { pattern } subpattern predicate."""

    def test_exists_basic(self, social_graph):
        """Find people who know someone."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE EXISTS { (p)-[:KNOWS]->(:Person) }
            RETURN p.name
            ORDER BY p.name
        """)

        names = [row["p.name"] for row in result]
        assert names == ["Alice", "Bob"]

    def test_exists_with_label_filter(self, social_graph):
        """Find people who purchased something."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE EXISTS { (p)-[:PURCHASED]->(:Product) }
            RETURN p.name
            ORDER BY p.name
        """)

        names = [row["p.name"] for row in result]
        assert names == ["Alice", "Bob"]

    def test_exists_no_match(self, social_graph):
        """EXISTS returns false when no matching pattern."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE EXISTS { (p)-[:WORKS_AT]->(:Company) }
            RETURN p.name
        """)

        assert len(result) == 0

    def test_not_exists(self, social_graph):
        """NOT EXISTS — find people who don't know anyone."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE NOT EXISTS { (p)-[:KNOWS]->(:Person) }
            RETURN p.name
            ORDER BY p.name
        """)

        names = [row["p.name"] for row in result]
        assert names == ["Charlie", "Diana"]

    def test_not_exists_purchase(self, social_graph):
        """NOT EXISTS — find people who haven't purchased anything."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE NOT EXISTS { (p)-[:PURCHASED]->(:Product) }
            RETURN p.name
            ORDER BY p.name
        """)

        names = [row["p.name"] for row in result]
        assert names == ["Charlie", "Diana"]

    def test_exists_with_property_filter(self, social_graph):
        """EXISTS with property filter in inner pattern."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE EXISTS { (p)-[:KNOWS]->(:Person {city: 'Oslo'}) }
            RETURN p.name
            ORDER BY p.name
        """)

        # Alice knows Charlie (Oslo), Bob knows Charlie (Oslo)
        names = [row["p.name"] for row in result]
        assert names == ["Alice", "Bob"]

    def test_exists_with_specific_target(self, social_graph):
        """EXISTS with specific property on target."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE EXISTS { (p)-[:PURCHASED]->(:Product {name: 'Widget'}) }
            RETURN p.name
        """)

        assert len(result) == 1
        assert result[0]["p.name"] == "Alice"

    def test_exists_and_other_conditions(self, social_graph):
        """EXISTS combined with other WHERE conditions."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE p.city = 'Oslo' AND EXISTS { (p)-[:KNOWS]->(:Person) }
            RETURN p.name
        """)

        # Alice is in Oslo and knows people
        assert len(result) == 1
        assert result[0]["p.name"] == "Alice"

    def test_exists_or_condition(self, social_graph):
        """EXISTS combined with OR."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE EXISTS { (p)-[:KNOWS]->(:Person) } OR p.city = 'Stavanger'
            RETURN p.name
            ORDER BY p.name
        """)

        # Alice and Bob know people, Diana is in Stavanger
        names = [row["p.name"] for row in result]
        assert names == ["Alice", "Bob", "Diana"]

    def test_exists_incoming_relationship(self, social_graph):
        """EXISTS with incoming relationship direction."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE EXISTS { (p)<-[:KNOWS]-(:Person) }
            RETURN p.name
            ORDER BY p.name
        """)

        # Bob is known by Alice, Charlie is known by Alice and Bob
        names = [row["p.name"] for row in result]
        assert names == ["Bob", "Charlie"]


class TestWhereExistsEdgeCases:
    """Edge cases for WHERE EXISTS."""

    def test_exists_empty_graph(self):
        """EXISTS on empty graph returns no rows."""
        graph = KnowledgeGraph()
        graph.cypher("CREATE (:Person {name: 'Alice'})")

        result = graph.cypher("""
            MATCH (p:Person)
            WHERE EXISTS { (p)-[:KNOWS]->(:Person) }
            RETURN p.name
        """)

        assert len(result) == 0

    def test_exists_different_variables(self):
        """EXISTS with distinct source and target variables."""
        graph = KnowledgeGraph()
        graph.cypher("CREATE (:Person {name: 'Alice'})")
        graph.cypher("CREATE (:Person {name: 'Bob'})")
        graph.cypher("""
            MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
            CREATE (a)-[:KNOWS]->(b)
        """)

        # Alice knows Bob; query checks for outgoing KNOWS
        result = graph.cypher("""
            MATCH (p:Person)
            WHERE EXISTS { (p)-[:KNOWS]->(:Person) }
            RETURN p.name
        """)

        assert len(result) == 1
        assert result[0]["p.name"] == "Alice"

    def test_exists_multiple_relationship_types(self, social_graph):
        """EXISTS checking for multiple relationship types."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE EXISTS { (p)-[:KNOWS]->(:Person) } AND EXISTS { (p)-[:PURCHASED]->(:Product) }
            RETURN p.name
            ORDER BY p.name
        """)

        # Only Alice and Bob know people AND purchased something
        names = [row["p.name"] for row in result]
        assert names == ["Alice", "Bob"]

    def test_not_exists_with_multiple_conditions(self, social_graph):
        """NOT EXISTS with additional conditions."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE NOT EXISTS { (p)-[:KNOWS]->(:Person) } AND NOT EXISTS { (p)-[:PURCHASED]->(:Product) }
            RETURN p.name
            ORDER BY p.name
        """)

        # Charlie and Diana don't know anyone and haven't purchased anything
        names = [row["p.name"] for row in result]
        assert names == ["Charlie", "Diana"]


class TestExistsMatchSyntax:
    """Tests for EXISTS { MATCH pattern } syntax (optional MATCH keyword)."""

    def test_exists_match_keyword(self, social_graph):
        """EXISTS { MATCH (pattern) } works like EXISTS { (pattern) }."""
        result_with_match = social_graph.cypher("""
            MATCH (p:Person)
            WHERE EXISTS { MATCH (p)-[:KNOWS]->(:Person) }
            RETURN p.name
            ORDER BY p.name
        """)

        result_without_match = social_graph.cypher("""
            MATCH (p:Person)
            WHERE EXISTS { (p)-[:KNOWS]->(:Person) }
            RETURN p.name
            ORDER BY p.name
        """)

        assert [r["p.name"] for r in result_with_match] == [r["p.name"] for r in result_without_match]

    def test_not_exists_match_keyword(self, social_graph):
        """NOT EXISTS { MATCH (pattern) } works correctly."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE NOT EXISTS { MATCH (p)-[:KNOWS]->(:Person) }
            RETURN p.name
            ORDER BY p.name
        """)

        names = [row["p.name"] for row in result]
        assert names == ["Charlie", "Diana"]

    def test_exists_match_with_label(self, social_graph):
        """EXISTS { MATCH (pattern) } with edge type filter."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE EXISTS { MATCH (p)-[:PURCHASED]->(:Product) }
            RETURN p.name
            ORDER BY p.name
        """)

        names = [row["p.name"] for row in result]
        assert names == ["Alice", "Bob"]


class TestExistsParenSyntax:
    """Tests for EXISTS((...)) parenthesis syntax (alternative to brace syntax)."""

    def test_exists_paren_basic(self, social_graph):
        """EXISTS((...)) returns same results as EXISTS { ... }."""
        brace_result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE EXISTS { (p)-[:KNOWS]->(:Person) }
            RETURN p.name
            ORDER BY p.name
        """)

        paren_result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE EXISTS((p)-[:KNOWS]->(:Person))
            RETURN p.name
            ORDER BY p.name
        """)

        assert brace_result.to_list() == paren_result.to_list()

    def test_not_exists_paren(self, social_graph):
        """NOT EXISTS((...)) works correctly."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE NOT EXISTS((p)-[:KNOWS]->(:Person))
            RETURN p.name
            ORDER BY p.name
        """)

        names = [row["p.name"] for row in result]
        # Same result as brace syntax
        brace_result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE NOT EXISTS { (p)-[:KNOWS]->(:Person) }
            RETURN p.name
            ORDER BY p.name
        """)
        assert names == [row["p.name"] for row in brace_result]

    def test_exists_paren_with_label(self, social_graph):
        """EXISTS((...)) with specific edge type."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE EXISTS((p)-[:PURCHASED]->(:Product))
            RETURN p.name
            ORDER BY p.name
        """)

        assert len(result) > 0
        for row in result:
            assert isinstance(row["p.name"], str)


class TestInlinePatternPredicates:
    """Tests for inline pattern predicates in WHERE — desugared to EXISTS."""

    def test_inline_pattern_basic(self, social_graph):
        """WHERE (p)-[:KNOWS]->(:Person) works like EXISTS { ... }."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE (p)-[:KNOWS]->(:Person)
            RETURN p.name
            ORDER BY p.name
        """)

        names = [row["p.name"] for row in result]
        assert names == ["Alice", "Bob"]

    def test_inline_pattern_matches_exists(self, social_graph):
        """Inline pattern produces same results as EXISTS { ... }."""
        inline = social_graph.cypher("""
            MATCH (p:Person)
            WHERE (p)-[:KNOWS]->(:Person)
            RETURN p.name ORDER BY p.name
        """)
        exists = social_graph.cypher("""
            MATCH (p:Person)
            WHERE EXISTS { (p)-[:KNOWS]->(:Person) }
            RETURN p.name ORDER BY p.name
        """)
        assert inline.to_list() == exists.to_list()

    def test_not_inline_pattern(self, social_graph):
        """WHERE NOT (p)-[:KNOWS]->(:Person) — negated pattern predicate."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE NOT (p)-[:KNOWS]->(:Person)
            RETURN p.name
            ORDER BY p.name
        """)

        names = [row["p.name"] for row in result]
        assert names == ["Charlie", "Diana"]

    def test_inline_pattern_with_label(self, social_graph):
        """Inline pattern with specific relationship type."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE (p)-[:PURCHASED]->(:Product)
            RETURN p.name
            ORDER BY p.name
        """)

        names = [row["p.name"] for row in result]
        assert names == ["Alice", "Bob"]

    def test_inline_pattern_incoming(self, social_graph):
        """Inline pattern with incoming relationship."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE (p)<-[:KNOWS]-(:Person)
            RETURN p.name
            ORDER BY p.name
        """)

        names = [row["p.name"] for row in result]
        assert names == ["Bob", "Charlie"]

    def test_inline_pattern_with_property(self, social_graph):
        """Inline pattern with property filter on target node."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE (p)-[:KNOWS]->(:Person {city: 'Oslo'})
            RETURN p.name
            ORDER BY p.name
        """)

        names = [row["p.name"] for row in result]
        assert names == ["Alice", "Bob"]

    def test_inline_pattern_combined_with_and(self, social_graph):
        """Inline pattern combined with AND condition."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE p.city = 'Oslo' AND (p)-[:KNOWS]->(:Person)
            RETURN p.name
        """)

        assert len(result) == 1
        assert result[0]["p.name"] == "Alice"

    def test_inline_pattern_combined_with_or(self, social_graph):
        """Inline pattern combined with OR condition."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE (p)-[:KNOWS]->(:Person) OR p.city = 'Stavanger'
            RETURN p.name
            ORDER BY p.name
        """)

        names = [row["p.name"] for row in result]
        assert names == ["Alice", "Bob", "Diana"]

    def test_inline_pattern_no_match(self, social_graph):
        """Inline pattern that matches nothing."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE (p)-[:WORKS_AT]->(:Company)
            RETURN p.name
        """)

        assert len(result) == 0

    def test_not_inline_pattern_with_and(self, social_graph):
        """NOT pattern combined with AND."""
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE NOT (p)-[:KNOWS]->(:Person) AND NOT (p)-[:PURCHASED]->(:Product)
            RETURN p.name
            ORDER BY p.name
        """)

        names = [row["p.name"] for row in result]
        assert names == ["Charlie", "Diana"]


class TestExistsInlinePropertyRegression:
    """Regression for the silent-wrong-answer bug where inline property
    filters on the target node of an EXISTS subpattern (e.g.
    ``{id: 20}``) were dropped by the fast-path check, because
    get_property('id') missed the id_column. Reported from a Wikidata
    MCP session; fix lives in match_clause.rs::try_fast_exists_check
    by resolving title/id aliases before looking up values.

    All three queries below describe the same logical filter — T2 and
    T3 worked before the fix; T1 returned zero rows. After the fix,
    all three must return the same single row."""

    @pytest.fixture
    def id_graph(self):
        import pandas as pd

        graph = KnowledgeGraph()
        # add_nodes honours user-provided IDs (unlike bare CREATE which
        # auto-assigns sequentially), so {id: 17764457} matches reality.
        graph.add_nodes(
            pd.DataFrame({"nid": [17764457], "name": ["Gina Krog"]}),
            "Field",
            "nid",
            "name",
        )
        graph.add_nodes(
            pd.DataFrame({"nid": [20], "name": ["Norway"]}),
            "Country",
            "nid",
            "name",
        )
        graph.add_connections(
            pd.DataFrame({"from_id": [17764457], "to_id": [20]}),
            "P17",
            "Field",
            "from_id",
            "Country",
            "to_id",
        )
        return graph

    def test_t1_exists_inline_id_property(self, id_graph):
        # Previously returned [] — the {id: 20} filter inside EXISTS
        # was silently dropped.
        result = id_graph.cypher("""
            MATCH (f {id: 17764457})
            WHERE EXISTS { MATCH (f)-[:P17]->({id: 20}) }
            RETURN f.id
        """)
        ids = [row["f.id"] for row in result]
        assert ids == [17764457]

    def test_t2_exists_where_id_equals(self, id_graph):
        result = id_graph.cypher("""
            MATCH (f {id: 17764457})
            WHERE EXISTS { MATCH (f)-[:P17]->(c) WHERE c.id = 20 }
            RETURN f.id
        """)
        ids = [row["f.id"] for row in result]
        assert ids == [17764457]

    def test_t3_inline_id_property_in_match(self, id_graph):
        result = id_graph.cypher("""
            MATCH (f {id: 17764457})-[:P17]->({id: 20})
            RETURN f.id
        """)
        ids = [row["f.id"] for row in result]
        assert ids == [17764457]

    def test_exists_inline_title_property(self, id_graph):
        # Same bug shape but for the `title` alias column.
        result = id_graph.cypher("""
            MATCH (f {id: 17764457})
            WHERE EXISTS { MATCH (f)-[:P17]->({title: 'Norway'}) }
            RETURN f.id
        """)
        ids = [row["f.id"] for row in result]
        assert ids == [17764457]


class TestExistsMultiMatchSubquery:
    """EXISTS { MATCH ... MATCH ... [WHERE ...] } — multi-clause subquery
    form (issue #12). Previously errored with `Unexpected token in EXISTS
    pattern: Match`; now treated as multiple sequential patterns."""

    def test_two_match_clauses(self, social_graph):
        # People who know someone AND have purchased something.
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE EXISTS {
                MATCH (p)-[:KNOWS]->(:Person)
                MATCH (p)-[:PURCHASED]->(:Product)
            }
            RETURN p.name
            ORDER BY p.name
        """)
        names = [row["p.name"] for row in result]
        assert names == ["Alice", "Bob"]

    def test_two_match_clauses_with_where(self, social_graph):
        # People who know someone AND purchased a product priced > 20.
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE EXISTS {
                MATCH (p)-[:KNOWS]->(:Person)
                MATCH (p)-[:PURCHASED]->(prod:Product)
                WHERE prod.price > 20
            }
            RETURN p.name
        """)
        names = sorted(row["p.name"] for row in result)
        # Alice purchased Widget (price 10) — fails the WHERE; only Bob
        # (Gadget, price 25) qualifies.
        assert names == ["Bob"]

    def test_not_exists_multi_match(self, social_graph):
        # NOT EXISTS subquery — people who haven't done both.
        result = social_graph.cypher("""
            MATCH (p:Person)
            WHERE NOT EXISTS {
                MATCH (p)-[:KNOWS]->(:Person)
                MATCH (p)-[:PURCHASED]->(:Product)
            }
            RETURN p.name
            ORDER BY p.name
        """)
        names = [row["p.name"] for row in result]
        # Charlie and Diana neither know anyone outbound nor purchased
        # — they fail the multi-MATCH subquery, so NOT EXISTS keeps them.
        assert names == ["Charlie", "Diana"]


def test_exists_pattern_accepts_parameter(social_graph):
    """A parameter inside an EXISTS {} pattern's inline map must parse and
    evaluate identically to the literal form. Regression: kglite-docs
    2026-05-29 #1 — `EXISTS { MATCH (...{id:$id}) }` raised a syntax error
    while the literal worked, blocking the natural parameterised work-list
    query."""
    param = social_graph.cypher(
        "MATCH (p:Person) WHERE EXISTS { MATCH (p)-[:PURCHASED]->(:Product {name:$name}) } "
        "RETURN p.name AS name ORDER BY name",
        params={"name": "Widget"},
    ).to_list()
    literal = social_graph.cypher(
        "MATCH (p:Person) WHERE EXISTS { MATCH (p)-[:PURCHASED]->(:Product {name:'Widget'}) } "
        "RETURN p.name AS name ORDER BY name"
    ).to_list()
    assert param == literal
    assert [r["name"] for r in param] == ["Alice"]


# ---------------------------------------------------------------------------
# The witness cap
# ---------------------------------------------------------------------------
#
# An existence check needs ONE match, so the pattern predicate hands the
# executor `max_matches = 1`. The differential corpus cannot see this: the cap
# is not a planner pass, so both of its legs run with it. These are absolute
# goldens — the answer, not an agreement between two paths.


def _ring(size: int, extra_edges: list[tuple[int, int]] | None = None) -> KnowledgeGraph:
    """`size` `:N` nodes on a directed ring, plus any extra relationships."""
    import pandas as pd

    graph = KnowledgeGraph()
    graph.add_nodes(pd.DataFrame({"id": list(range(size))}), "N", "id")
    edges = [(i, (i + 1) % size) for i in range(size)] + list(extra_edges or [])
    graph.add_connections(
        pd.DataFrame({"s": [e[0] for e in edges], "t": [e[1] for e in edges]}),
        "R",
        "N",
        "s",
        "N",
        "t",
    )
    return graph


class TestExistsWitnessCap:
    def test_a_witness_past_the_candidate_cap_is_still_found(self):
        """The cap induces the executor's advisory 1000-source pre-cap, and a
        pattern whose only witness sits behind it comes back empty on the
        capped pass. The uncapped retry is what makes the answer right — with
        the retry disabled this query answers 0 instead of 1."""
        import pandas as pd

        graph = KnowledgeGraph()
        graph.add_nodes(pd.DataFrame({"id": list(range(1, 5001))}), "Person", "id")
        graph.add_nodes(pd.DataFrame({"id": [1]}), "Target", "id")
        graph.cypher("MATCH (p:Person {id: 5000}), (t:Target) CREATE (p)-[:R]->(t)")

        assert graph.cypher(
            "MATCH (t:Target) WHERE EXISTS { (p:Person)-[:R*1..2]->(t) } RETURN count(t) AS c"
        ).to_list() == [{"c": 1}]
        assert graph.cypher(
            "MATCH (t:Target) WHERE NOT EXISTS { (p:Person)-[:R*1..2]->(t) } RETURN count(t) AS c"
        ).to_list() == [{"c": 0}]

    def test_not_exists_is_true_only_after_the_full_search(self):
        graph = _ring(6)
        # No `:S` relationship exists at all, so every node qualifies.
        assert [
            row["i"]
            for row in graph.cypher("MATCH (n:N) WHERE NOT EXISTS { (n)-[:S*1..3]->(:N) } RETURN n.id AS i ORDER BY i")
        ] == [0, 1, 2, 3, 4, 5]
        # ... and with one witness each, none does.
        assert graph.cypher("MATCH (n:N) WHERE NOT EXISTS { (n)-[:R*1..3]->(:N) } RETURN count(n) AS c").to_list() == [
            {"c": 0}
        ]

    def test_min_two_hops_answers_trail_reachability_under_the_cap(self):
        """`min >= 2` stays on the per-path expansion — the cap only stops it
        at the first complete trail. On a triangle every node is trail-
        reachable to both peers at exactly two hops."""
        graph = _ring(3)
        assert [
            row["i"]
            for row in graph.cypher("MATCH (n:N) WHERE EXISTS { (n)-[:R*2..2]-(:N) } RETURN n.id AS i ORDER BY i")
        ] == [0, 1, 2]

    def test_a_bound_relationship_variable_still_pins_the_subquery(self):
        """`r` is bound by the outer MATCH, so the subquery may only match
        that one relationship — a cap of one would answer from whichever
        relationship the expansion reached first."""
        graph = _ring(4, extra_edges=[(0, 2)])
        rows = graph.cypher(
            "MATCH (a:N {id: 0})-[r:R]->(b:N) WHERE EXISTS { (a)-[r]->(x:N) } RETURN b.id AS i ORDER BY i"
        ).to_list()
        # Node 0 has two outgoing relationships, so a cap of one would answer
        # both rows from whichever the expansion reached first and drop the
        # other.
        assert rows == [{"i": 1}, {"i": 2}]

    def test_optional_match_with_where_exists_null_extends(self):
        """The OPTIONAL MATCH row survives with NULLs when its predicate finds
        no witness; the cap must not turn a found witness into a miss."""
        graph = _ring(3, extra_edges=[])
        rows = graph.cypher(
            "MATCH (n:N) OPTIONAL MATCH (n)-[:R]->(m:N) "
            "WHERE EXISTS { (m)-[:R*1..2]->(:N {id: 0}) } "
            "RETURN n.id AS i, m.id AS j ORDER BY i"
        ).to_list()
        # 0→1 (1 reaches 0 in two hops), 1→2 (2 reaches 0 in one), 2→0
        # (0 reaches 0 only over a three-hop closed trail, outside *1..2).
        assert rows == [{"i": 0, "j": 1}, {"i": 1, "j": 2}, {"i": 2, "j": None}]

    def test_several_predicates_are_capped_independently(self):
        graph = _ring(4)
        rows = graph.cypher(
            "MATCH (n:N) WHERE EXISTS { (n)-[:R*1..2]->(:N {id: 0}) } "
            "AND NOT EXISTS { (n)-[:R*1..1]->(:N {id: 0}) } "
            "RETURN n.id AS i ORDER BY i"
        ).to_list()
        # 2 reaches 0 in two hops but not one; 3 reaches it in one.
        assert rows == [{"i": 2}]

    def test_exists_in_projection_and_case_position(self):
        graph = _ring(3)
        rows = graph.cypher(
            "MATCH (n:N) RETURN n.id AS i, "
            "EXISTS { (n)-[:R*1..1]->(:N {id: 0}) } AS direct, "
            "CASE WHEN EXISTS { (n)-[:R*1..2]->(:N {id: 0}) } THEN 'y' ELSE 'n' END AS near "
            "ORDER BY i"
        ).to_list()
        assert rows == [
            {"i": 0, "direct": False, "near": "n"},
            {"i": 1, "direct": False, "near": "y"},
            {"i": 2, "direct": True, "near": "y"},
        ]


class TestExistsRelationshipAlternation:
    """`[:A|B]` parses in a MATCH but was rejected inside `EXISTS { }`: the
    subquery re-serializer had no `|` arm, so the token never reached the
    pattern parser that has understood alternation all along."""

    @pytest.fixture
    def one_type_graph(self):
        """Two nodes joined by `:A`. `:B` is declared in the query only —
        an alternation branch that matches nothing must not change the
        answer."""
        graph = KnowledgeGraph()
        graph.cypher("CREATE (:N {id: 1}), (:N {id: 2})")
        graph.cypher("MATCH (a:N {id: 1}), (b:N {id: 2}) CREATE (a)-[:A]->(b)")
        return graph

    def test_exists_alternation_matches_the_match_form(self, one_type_graph):
        expected = one_type_graph.cypher("MATCH (n:N)-[:A|B]->() RETURN n.id AS i ORDER BY i").to_list()
        assert expected == [{"i": 1}]
        for form in (
            "MATCH (n:N) WHERE EXISTS { (n)-[:A|B]->() } RETURN n.id AS i ORDER BY i",
            "MATCH (n:N) WHERE EXISTS { MATCH (n)-[:A|B]->() } RETURN n.id AS i ORDER BY i",
        ):
            assert one_type_graph.cypher(form).to_list() == expected, form

    def test_alternation_of_two_live_types(self):
        graph = KnowledgeGraph()
        graph.cypher("CREATE (:N {id: 1}), (:N {id: 2}), (:N {id: 3}), (:N {id: 4})")
        graph.cypher("MATCH (a:N {id: 1}), (b:N {id: 2}) CREATE (a)-[:A]->(b)")
        graph.cypher("MATCH (a:N {id: 2}), (b:N {id: 3}) CREATE (a)-[:B]->(b)")
        graph.cypher("MATCH (a:N {id: 3}), (b:N {id: 4}) CREATE (a)-[:C]->(b)")
        rows = graph.cypher("MATCH (n:N) WHERE EXISTS { (n)-[:A|B]->() } RETURN n.id AS i ORDER BY i").to_list()
        assert rows == [{"i": 1}, {"i": 2}]

    def test_alternation_inside_count_and_size(self):
        graph = KnowledgeGraph()
        graph.cypher("CREATE (:N {id: 1}), (:N {id: 2}), (:N {id: 3})")
        graph.cypher("MATCH (a:N {id: 1}), (b:N {id: 2}) CREATE (a)-[:A]->(b)")
        graph.cypher("MATCH (a:N {id: 1}), (b:N {id: 3}) CREATE (a)-[:B]->(b)")
        rows = graph.cypher(
            "MATCH (n:N {id: 1}) RETURN count { (n)-[:A|B]->() } AS c, size((n)-[:A|B]->()) AS s"
        ).to_list()
        assert rows == [{"c": 2, "s": 2}]

    def test_three_way_alternation_and_negation(self, one_type_graph):
        rows = one_type_graph.cypher(
            "MATCH (n:N) WHERE NOT EXISTS { (n)-[:A|B|C]->() } RETURN n.id AS i ORDER BY i"
        ).to_list()
        assert rows == [{"i": 2}]
