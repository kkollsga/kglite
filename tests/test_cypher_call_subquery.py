"""CALL { } subqueries — Phase 1 (parser) + Phase 2 (validation).

Phase 1 ships the parser: well-formed CALL { } subqueries *parse*
successfully, but reaching the executor raises a clean, helpful
``CypherExecutionError`` (not a panic, not silent wrong results).
Malformed subqueries — missing closing brace, an illegal importing
``WITH`` — fail at parse time as ``CypherSyntaxError``. Existing
``CALL procedure() YIELD ...`` queries are unaffected.

Phase 2 adds v1 structural validation of the body and mutation
classification: write / unit / UNION subquery bodies are *rejected*
(deferred to a future release per the design doc, §1.4 / §6), and a
write buried inside CALL { } is classified as a mutation so it can
never slip through the read path. Execution + planner integration
land in later phases; this file guards the parse-then-validate-
then-clean-error contract for the new form.
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


class TestCallSubqueryParsesButNotExecutable:
    """Well-formed CALL { } parses, then raises a clean execution error.

    These shapes pass Phase 2 validation (read-only body, terminal
    RETURN, no UNION, nested CALL allowed) and reach the not-yet-
    executable executor stub.
    """

    @pytest.mark.parametrize(
        "query",
        [
            # Uncorrelated.
            "CALL { MATCH (n:Person) RETURN count(n) AS c } RETURN c",
            # Correlated, single import.
            "MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS c } RETURN p.name, c",
            # Correlated, multi-import.
            "MATCH (p:Person), (q:Person) CALL { WITH p, q MATCH (p)-[:KNOWS]->(q) RETURN count(*) AS c } RETURN c",
            # Nested CALL {} inside the body — allowed in v1 (§1.4).
            "CALL { CALL { MATCH (n) RETURN n LIMIT 1 } MATCH (m) RETURN m AS r } RETURN r",
            # Map literal in the RETURN of the body must not close the brace early.
            "CALL { MATCH (n) RETURN {a: n.name} AS m } RETURN m",
            # Terminal RETURN followed by ORDER BY / LIMIT is still a valid body.
            "CALL { MATCH (n:Person) RETURN n.name AS nm ORDER BY nm LIMIT 1 } RETURN nm",
        ],
    )
    def test_parses_then_clean_execution_error(self, graph, query):
        with pytest.raises(kglite.CypherExecutionError, match=NOT_EXECUTABLE):
            graph.cypher(query)


class TestCallSubqueryParseErrors:
    """Malformed subqueries fail at parse time, not execution."""

    def test_missing_closing_brace(self, graph):
        with pytest.raises(kglite.CypherSyntaxError):
            graph.cypher("CALL { MATCH (n:Person) RETURN n")

    @pytest.mark.parametrize(
        "query",
        [
            # Aliasing in the importing position.
            "MATCH (p:Person) CALL { WITH p AS x MATCH (x) RETURN count(x) AS c } RETURN c",
            # Projection in the importing position.
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
            # CREATE.
            "MATCH (p:Person) CALL { WITH p CREATE (:Tag {name: p.name}) RETURN p } RETURN p",
            # SET.
            "MATCH (p:Person) CALL { WITH p SET p.seen = true RETURN p } RETURN p",
            # DELETE.
            "MATCH (p:Person) CALL { WITH p DELETE p RETURN 1 AS x } RETURN x",
            # REMOVE.
            "MATCH (p:Person) CALL { WITH p REMOVE p.name RETURN p } RETURN p",
            # MERGE.
            "CALL { MERGE (:Tag {name: 'x'}) RETURN 1 AS x } RETURN x",
        ],
    )
    def test_write_in_call_rejected(self, graph, query):
        with pytest.raises(kglite.CypherSyntaxError, match="write clauses.*CALL"):
            graph.cypher(query)

    def test_write_in_call_rejected_even_on_read_only_graph(self):
        """A write-in-CALL must error (never slip through the read path),
        whether the graph is read-only or not. The mutation is rejected
        at validation before the read/write routing decision."""
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice'})")
        g.read_only(True)
        with pytest.raises(kglite.CypherSyntaxError, match="write clauses.*CALL"):
            g.cypher("CALL { CREATE (:Tag {name: 'x'}) RETURN 1 AS x } RETURN x")

    def test_nested_write_in_inner_call_rejected(self, graph):
        """A write buried in a *nested* CALL body is still rejected."""
        with pytest.raises(kglite.CypherSyntaxError, match="write clauses.*CALL"):
            graph.cypher("CALL { CALL { CREATE (:Tag) RETURN 1 AS y } RETURN y } RETURN y")


class TestCallSubqueryUnitBodyRejected:
    """v1 rejects unit subqueries (a body with no terminal RETURN)."""

    @pytest.mark.parametrize(
        "query",
        [
            # No RETURN at all.
            "MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]->(f) } RETURN p",
            # Body ends in WITH, not RETURN.
            "CALL { MATCH (n:Person) WITH n } RETURN n",
        ],
    )
    def test_unit_subquery_rejected(self, graph, query):
        with pytest.raises(kglite.CypherSyntaxError, match="must end with RETURN"):
            graph.cypher(query)


class TestCallSubqueryUnionBodyRejected:
    """v1 rejects UNION inside a CALL { } body (deferred)."""

    def test_union_in_call_rejected(self, graph):
        with pytest.raises(kglite.CypherSyntaxError, match="UNION.*inside a CALL"):
            graph.cypher(
                "CALL { MATCH (n:Person) RETURN n.name AS nm UNION MATCH (m:Person) RETURN m.name AS nm } RETURN nm"
            )


class TestCallProcedureUnaffected:
    """The existing CALL procedure() YIELD form still works."""

    def test_call_procedure_still_executes(self, graph):
        result = graph.cypher("CALL pagerank() YIELD node, score RETURN node, score")
        rows = result.to_list()
        assert len(rows) == 2  # Alice + Bob
        assert all("score" in row for row in rows)
