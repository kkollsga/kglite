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


# ── Invalid / unsupported patterns on every execution path ───────────────
#
# An uncompilable pattern is wrong for every row and can never become right,
# so it must raise — on the fused execution paths exactly as on the unfused
# one. Until 0.16.6 the fused paths swallowed the compile error along with the
# "this predicate does not evaluate for this row" errors they drop by design,
# and answered an invalid query with a silent empty result.
#
# The differential corpus cannot pin this: its harness compares *rows* from the
# optimized and naive runs, so a query that must raise on both sides errors the
# test rather than expressing the contract. These are absolute goldens instead,
# each run through both paths.

INVALID_PATTERNS = [
    ("unclosed_class", "["),
    ("unclosed_group", "(a"),
    # Valid in Neo4j (java.util.regex), unsupported by the Rust regex crate.
    # These used to be the worst case: a pattern a user has every reason to
    # believe in, silently matching nothing.
    ("lookahead", "A(?=l)ice"),
    ("lookbehind", "(?<=A)lice"),
    # The doubled backslash is Cypher string-literal escaping: `\\1` in the
    # query text is the single backslash the regex engine sees.
    ("backreference", r"(a)\\1"),
]


@pytest.fixture
def team_graph():
    """People with edges, for the fused aggregate / WITH-aggregate shapes."""
    import pandas as pd

    g = KnowledgeGraph()
    people = pd.DataFrame([{"id": i, "title": n} for i, n in enumerate(["Alice", "Bob", "Cyd", "Dee"], 1)])
    g.add_nodes(people, "Person", "id", "title")
    edges = pd.DataFrame([(1, 2), (1, 3), (1, 4), (2, 3)], columns=["s", "t"])
    g.add_connections(edges, "R", "Person", "s", "Person", "t")
    return g


# Each entry names an execution path the WHERE predicate is evaluated on.
FUSED_SHAPES = [
    # Unfused reference — this one has always raised.
    ("plain_rows", "MATCH (p:Person) WHERE p.title =~ '{pat}' RETURN p.title"),
    # FusedNodeScanAggregate.
    ("fused_scan_aggregate", "MATCH (p:Person) WHERE p.title =~ '{pat}' RETURN count(*) AS c"),
    ("fused_scan_group", "MATCH (p:Person) WHERE p.title =~ '{pat}' RETURN p.title AS t, count(*) AS c"),
    # FusedNodeScanTopK.
    ("fused_scan_top_k", "MATCH (p:Person) WHERE p.title =~ '{pat}' RETURN p.title AS t ORDER BY t LIMIT 3"),
    # WITH … WHERE over a fused match+aggregate.
    (
        "fused_with_aggregate",
        "MATCH (n:Person)-[:R]->(m) WITH n, count(m) AS c WHERE n.title =~ '{pat}' RETURN n.title AS t, c",
    ),
    # HAVING over the same.
    (
        "fused_having",
        "MATCH (n:Person)-[:R]->(m) RETURN n.title AS t, count(m) AS c HAVING t =~ '{pat}'",
    ),
]


class TestInvalidPatternRaisesEverywhere:
    """An uncompilable pattern raises on every path, optimizer on or off."""

    @pytest.mark.parametrize("shape,query", FUSED_SHAPES, ids=[s[0] for s in FUSED_SHAPES])
    @pytest.mark.parametrize("pattern_id,pattern", INVALID_PATTERNS, ids=[p[0] for p in INVALID_PATTERNS])
    @pytest.mark.parametrize("disable_optimizer", [False, True], ids=["optimized", "naive"])
    def test_raises(self, team_graph, shape, query, pattern_id, pattern, disable_optimizer):
        import kglite

        with pytest.raises(kglite.KgError, match="Invalid regular expression"):
            team_graph.cypher(query.format(pat=pattern), disable_optimizer=disable_optimizer)

    def test_regex_function_pattern_also_raises_on_fused_path(self, team_graph):
        """`text_match_regex()` shares the swallow site, and the fix."""
        import kglite

        with pytest.raises(kglite.KgError, match="invalid pattern"):
            team_graph.cypher("MATCH (p:Person) WHERE text_match_regex(p.title, '[') RETURN count(*) AS c")


class TestUnevaluablePredicateStillDropsRows:
    """The swallow the fused paths keep: a predicate that does not evaluate.

    Only regex *compile* failures propagate. A predicate that merely cannot be
    evaluated for a row — an `OPTIONAL MATCH` binding that never matched, an
    aggregate reference resolved post-aggregation — still drops the row, which
    is what every one of these queries has always returned.
    """

    def test_optional_match_unbound_binding_in_where(self, team_graph):
        rows = team_graph.cypher(
            "MATCH (p:Person) OPTIONAL MATCH (p)-[:R]->(m) WITH p, m WHERE m.title = 'Bob' RETURN p.title AS t"
        ).to_list()
        assert [r["t"] for r in rows] == ["Alice"]

    def test_optional_match_unbound_binding_with_aggregate(self, team_graph):
        rows = team_graph.cypher(
            "MATCH (p:Person) OPTIONAL MATCH (p)-[:R]->(m) "
            "WITH p, count(m) AS c WHERE c = 0 RETURN p.title AS t ORDER BY t"
        ).to_list()
        assert [r["t"] for r in rows] == ["Cyd", "Dee"]

    def test_having_on_aggregate_expression(self, team_graph):
        rows = team_graph.cypher(
            "MATCH (n:Person)-[:R]->(m) RETURN n.title AS t, count(m) AS c HAVING count(m) > 1"
        ).to_list()
        assert [(r["t"], r["c"]) for r in rows] == [("Alice", 3)]

    def test_valid_regex_on_fused_paths_unaffected(self, team_graph):
        assert team_graph.cypher("MATCH (p:Person) WHERE p.title =~ '^A.*' RETURN count(*) AS c").scalar() == 1
        rows = team_graph.cypher(
            "MATCH (p:Person) WHERE p.title =~ '^[A-C].*' RETURN p.title AS t ORDER BY t LIMIT 3"
        ).to_list()
        assert [r["t"] for r in rows] == ["Alice", "Bob", "Cyd"]
