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
