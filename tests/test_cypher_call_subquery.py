"""CALL { } subqueries — Phase 1 (parser) + Phase 2 (validation) +
Phase 3 (executor: uncorrelated).

Phase 1 ships the parser; Phase 2 adds v1 structural validation
(write / unit / UNION bodies rejected, importing-WITH restrictions,
mutation classification). Phase 3 makes the **uncorrelated** form
(``CALL { ... }`` importing nothing) executable: the body runs exactly
once and its result rows are cartesian-producted with the outer row
stream (§1.1 of ``dev-documentation/design/call-subqueries.md``). The
body sees no outer variables (§1.2 rule 1); only its RETURN columns flow
out (§1.2 rule 3); a RETURN alias colliding with an outer variable is a
compile/execution error (§1.2 rule 4). The **correlated** form (leading
importing ``WITH``) still raises a clean not-yet-executable error
(Phase 4).
"""

import pytest

import kglite
from kglite import KnowledgeGraph

NOT_EXECUTABLE = "not yet executable"


@pytest.fixture
def graph():
    g = KnowledgeGraph()
    g.cypher("CREATE (:Person {name: 'Alice'})")
    g.cypher("CREATE (:Person {name: 'Bob'})")
    g.cypher("""
        MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
        CREATE (a)-[:KNOWS]->(b)
    """)
    return g


@pytest.fixture
def tagged():
    """2 Person + 3 Tag, deterministic titles for cartesian assertions."""
    g = KnowledgeGraph()
    for n in ("Alice", "Bob"):
        g.cypher("CREATE (:P {title: $t})", params={"t": n})
    for t in ("t1", "t2", "t3"):
        g.cypher("CREATE (:Tag {title: $t})", params={"t": t})
    return g


# ──────────────────────────────────────────────────────────────────
# Phase 3 — uncorrelated execution
# ──────────────────────────────────────────────────────────────────


class TestUncorrelatedCallSubquery:
    def test_leading_call_aggregate(self, tagged):
        """CALL { ... RETURN count(n) AS c } as the first clause → S rows."""
        rows = tagged.cypher("CALL { MATCH (n:P) RETURN count(n) AS c } RETURN c").to_list()
        assert rows == [{"c": 2}]

    def test_outer_cartesian_inner(self, tagged):
        """2 outer (P) × 3 inner (Tag) = 6 rows; only RETURN cols escape."""
        rows = tagged.cypher(
            "MATCH (p:P) CALL { MATCH (x:Tag) RETURN x.title AS t } RETURN p.title AS pt, t ORDER BY pt, t"
        ).to_list()
        assert rows == [
            {"pt": "Alice", "t": "t1"},
            {"pt": "Alice", "t": "t2"},
            {"pt": "Alice", "t": "t3"},
            {"pt": "Bob", "t": "t1"},
            {"pt": "Bob", "t": "t2"},
            {"pt": "Bob", "t": "t3"},
        ]

    def test_body_with_where_order_limit(self, tagged):
        """WHERE / ORDER BY / LIMIT inside the body are honoured."""
        rows = tagged.cypher(
            "CALL { MATCH (x:Tag) WHERE x.title <> 't2' RETURN x.title AS t ORDER BY t DESC LIMIT 1 } RETURN t"
        ).to_list()
        assert rows == [{"t": "t3"}]

    def test_multiple_return_columns(self, tagged):
        rows = tagged.cypher("CALL { MATCH (x:Tag) RETURN x.title AS t, 1 AS one ORDER BY t } RETURN t, one").to_list()
        assert rows == [
            {"t": "t1", "one": 1},
            {"t": "t2", "one": 1},
            {"t": "t3", "one": 1},
        ]

    def test_empty_subquery_result_drops_all_rows(self, tagged):
        """Cartesian with an empty (non-aggregating) subquery → zero rows."""
        rows = tagged.cypher("MATCH (p:P) CALL { MATCH (z:Nope) RETURN z.title AS t } RETURN p.title, t").to_list()
        assert rows == []

    def test_aggregating_body_runs_once(self, tagged):
        """count() body returns one row per outer row, all with the same value."""
        rows = tagged.cypher(
            "MATCH (p:P) CALL { MATCH (x:Tag) RETURN count(x) AS c } RETURN p.title AS pt, c ORDER BY pt"
        ).to_list()
        assert rows == [{"pt": "Alice", "c": 3}, {"pt": "Bob", "c": 3}]

    def test_body_executes_exactly_once_via_uuid(self, tagged):
        """randomUUID() in the body must be the SAME across all output rows
        if the body ran once (determinism probe per the design doc)."""
        rows = tagged.cypher(
            "MATCH (p:P) CALL { RETURN randomUUID() AS u } RETURN p.title AS pt, u ORDER BY pt"
        ).to_list()
        assert len(rows) == 2  # one per outer P row (cartesian with 1 inner row)
        assert len({r["u"] for r in rows}) == 1  # body ran once → one UUID

    def test_nested_uncorrelated_call(self, tagged):
        """A nested uncorrelated CALL { } inside a body (§1.4) executes."""
        rows = tagged.cypher("CALL { CALL { MATCH (x:Tag) RETURN count(x) AS c } RETURN c AS cc } RETURN cc").to_list()
        assert rows == [{"cc": 3}]

    def test_subquery_returning_multiple_rows_after_match(self, tagged):
        """R outer × S inner product where the inner returns N>1 rows."""
        rows = tagged.cypher(
            "MATCH (p:P) WHERE p.title = 'Alice' "
            "CALL { MATCH (x:Tag) RETURN x.title AS t } "
            "RETURN p.title AS pt, t ORDER BY t"
        ).to_list()
        assert rows == [
            {"pt": "Alice", "t": "t1"},
            {"pt": "Alice", "t": "t2"},
            {"pt": "Alice", "t": "t3"},
        ]


class TestUncorrelatedScoping:
    def test_body_does_not_see_outer_variables(self, tagged):
        """§1.2 rule 1 — the body sees a fresh scope. An un-imported outer
        `p` referenced inside the body is a *fresh* unbound variable, so
        `MATCH (p:Tag)` inside matches all Tags, independent of the outer p."""
        rows = tagged.cypher(
            "MATCH (p:P) WHERE p.title = 'Alice' CALL { MATCH (p:Tag) RETURN count(p) AS c } RETURN c"
        ).to_list()
        # The inner `p` is a fresh Tag node, not the outer Person — so the
        # count is the number of Tags (3), not influenced by the outer p.
        assert rows == [{"c": 3}]

    def test_return_alias_collision_with_outer_variable_errors(self, tagged):
        """§1.2 rule 4 — a subquery RETURN alias clashing with an in-scope
        outer variable is an error (Neo4j errors on shadowing)."""
        with pytest.raises(kglite.CypherExecutionError, match="already exists in"):
            tagged.cypher("MATCH (p:P) CALL { MATCH (x:Tag) RETURN x.title AS p } RETURN p")


# ──────────────────────────────────────────────────────────────────
# Phase 4 deferral — correlated still errors cleanly
# ──────────────────────────────────────────────────────────────────


class TestCorrelatedNotExecutable:
    @pytest.mark.parametrize(
        "query",
        [
            "MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS c } RETURN p.name, c",
            "MATCH (p:Person), (q:Person) CALL { WITH p, q MATCH (p)-[:KNOWS]->(q) RETURN count(*) AS c } RETURN c",
        ],
    )
    def test_correlated_clean_error(self, graph, query):
        with pytest.raises(kglite.CypherExecutionError, match=NOT_EXECUTABLE):
            graph.cypher(query)


# ──────────────────────────────────────────────────────────────────
# Phase 1/2 — parse errors and structural rejections (unchanged)
# ──────────────────────────────────────────────────────────────────


class TestCallSubqueryParseErrors:
    def test_missing_closing_brace(self, graph):
        with pytest.raises(kglite.CypherSyntaxError):
            graph.cypher("CALL { MATCH (n:Person) RETURN n")

    @pytest.mark.parametrize(
        "query",
        [
            "MATCH (p:Person) CALL { WITH p AS x MATCH (x) RETURN count(x) AS c } RETURN c",
            "MATCH (p:Person) CALL { WITH p.name MATCH (n) RETURN n } RETURN n",
        ],
    )
    def test_importing_with_violation_is_syntax_error(self, graph, query):
        with pytest.raises(kglite.CypherSyntaxError, match="importing WITH"):
            graph.cypher(query)


class TestCallSubqueryWriteBodyRejected:
    """v1 rejects write clauses inside a CALL { } body (deferred)."""

    @pytest.mark.parametrize(
        "query",
        [
            "MATCH (p:Person) CALL { WITH p CREATE (:Tag {name: p.name}) RETURN p } RETURN p",
            "MATCH (p:Person) CALL { WITH p SET p.seen = true RETURN p } RETURN p",
            "MATCH (p:Person) CALL { WITH p DELETE p RETURN 1 AS x } RETURN x",
            "MATCH (p:Person) CALL { WITH p REMOVE p.name RETURN p } RETURN p",
            "CALL { MERGE (:Tag {name: 'x'}) RETURN 1 AS x } RETURN x",
        ],
    )
    def test_write_in_call_rejected(self, graph, query):
        with pytest.raises(kglite.CypherSyntaxError, match="write clauses.*CALL"):
            graph.cypher(query)

    def test_write_in_call_rejected_even_on_read_only_graph(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice'})")
        g.read_only(True)
        with pytest.raises(kglite.CypherSyntaxError, match="write clauses.*CALL"):
            g.cypher("CALL { CREATE (:Tag {name: 'x'}) RETURN 1 AS x } RETURN x")

    def test_nested_write_in_inner_call_rejected(self, graph):
        with pytest.raises(kglite.CypherSyntaxError, match="write clauses.*CALL"):
            graph.cypher("CALL { CALL { CREATE (:Tag) RETURN 1 AS y } RETURN y } RETURN y")


class TestCallSubqueryUnitBodyRejected:
    @pytest.mark.parametrize(
        "query",
        [
            "MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]->(f) } RETURN p",
            "CALL { MATCH (n:Person) WITH n } RETURN n",
        ],
    )
    def test_unit_subquery_rejected(self, graph, query):
        with pytest.raises(kglite.CypherSyntaxError, match="must end with RETURN"):
            graph.cypher(query)


class TestCallSubqueryUnionBodyRejected:
    def test_union_in_call_rejected(self, graph):
        with pytest.raises(kglite.CypherSyntaxError, match="UNION.*inside a CALL"):
            graph.cypher(
                "CALL { MATCH (n:Person) RETURN n.name AS nm UNION MATCH (m:Person) RETURN m.name AS nm } RETURN nm"
            )


class TestCallProcedureUnaffected:
    def test_call_procedure_still_executes(self, graph):
        result = graph.cypher("CALL pagerank() YIELD node, score RETURN node, score")
        rows = result.to_list()
        assert len(rows) == 2  # Alice + Bob
        assert all("score" in row for row in rows)
