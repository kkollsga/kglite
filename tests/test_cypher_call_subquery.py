"""CALL { } subqueries — Phase 1 (parser + AST node) behaviour.

Phase 1 ships the parser only: well-formed CALL { } subqueries *parse*
successfully, but reaching the executor raises a clean, helpful
``CypherExecutionError`` (not a panic, not silent wrong results).
Malformed subqueries — missing closing brace, an illegal importing
``WITH`` — fail at parse time as ``CypherSyntaxError``. Existing
``CALL procedure() YIELD ...`` queries are unaffected.

Execution + planner integration land in later phases; this file
guards the parse-then-clean-error contract for the new form.
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
    """Well-formed CALL { } parses, then raises a clean execution error."""

    @pytest.mark.parametrize(
        "query",
        [
            # Uncorrelated.
            "CALL { MATCH (n:Person) RETURN count(n) AS c } RETURN c",
            # Correlated, single import.
            "MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS c } RETURN p.name, c",
            # Correlated, multi-import.
            "MATCH (p:Person), (q:Person) CALL { WITH p, q MATCH (p)-[:KNOWS]->(q) RETURN count(*) AS c } RETURN c",
            # Nested CALL {} inside the body.
            "CALL { CALL { MATCH (n) RETURN n LIMIT 1 } MATCH (m) RETURN m AS r } RETURN r",
            # Map literal in the RETURN of the body must not close the brace early.
            "CALL { MATCH (n) RETURN {a: n.name} AS m } RETURN m",
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


class TestCallProcedureUnaffected:
    """The existing CALL procedure() YIELD form still works."""

    def test_call_procedure_still_executes(self, graph):
        result = graph.cypher("CALL pagerank() YIELD node, score RETURN node, score")
        rows = result.to_list()
        assert len(rows) == 2  # Alice + Bob
        assert all("score" in row for row in rows)
