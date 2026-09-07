"""Absolute goldens for the WITH-boundary planner rewrites.

`hoist_with_where` (T2-1) and `fold_aliasing_with` (T2-2) both remove a
projection barrier the executor used to materialise. The differential
corpus compares the optimised plan against the unoptimised one, so it is
structurally blind to anything the two paths would get wrong together:
a column renamed on both sides, a tie broken the same wrong way on both
sides, an error that stops being raised on both sides. These are the
absolute answers.

Every case runs under both plan profiles — the default pipeline and
`disable_optimizer=True` — and asserts the same expected value for each.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite


@pytest.fixture
def boundary_graph() -> kglite.KnowledgeGraph:
    """Five `P` nodes over a deliberately tie-heavy sort key.

    `age` repeats (30, 30, 30) so ordering ties are reachable, and `P5`
    carries no `age` at all so NULL placement is reachable. `K` edges make
    a multi-row driving pattern available for the scope-error case.
    """
    graph = kglite.KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "title": ["a", "b", "c", "d", "e"],
                "city": ["X", "X", "Y", "Y", "Y"],
                "age": [30, 30, 30, 55, None],
            }
        ),
        "P",
        "id",
        "title",
    )
    graph.add_connections(
        pd.DataFrame({"src": [1, 1, 2, 3], "tgt": [2, 3, 3, 4]}),
        "K",
        "P",
        "src",
        "P",
        "tgt",
    )
    return graph


def both_profiles(graph: kglite.KnowledgeGraph, query: str, **kwargs) -> list[list[dict]]:
    """Rows from the optimised and the fully-unoptimised plan."""
    return [
        graph.cypher(query, **kwargs).to_list(),
        graph.cypher(query, disable_optimizer=True, **kwargs).to_list(),
    ]


# ── hoist_with_where (T2-1) ──────────────────────────────────────────


def test_hoisted_where_answers_the_same_rows(boundary_graph) -> None:
    """The trigger shape, against absolute values rather than each other."""
    for rows in both_profiles(boundary_graph, "MATCH (p:P) WITH p WHERE p.age > 30 RETURN count(*) AS n"):
        assert rows == [{"n": 1}]
    for rows in both_profiles(boundary_graph, "MATCH (p:P) WITH p WHERE p.city = 'X' RETURN p.title AS t"):
        assert rows == [{"t": "a"}, {"t": "b"}]


def test_the_with_scope_error_is_preserved(boundary_graph) -> None:
    """`WITH a WHERE b.x` is a Cypher scope error and must stay one.

    The hoist would make it evaluable — `b` is bound by the MATCH — so H5's
    second half exists purely to keep this refusal. Both profiles refuse.
    """
    query = "MATCH (a:P)-[:K]->(b:P) WITH a WHERE b.city = 'X' RETURN a.title AS t"
    for kwargs in ({}, {"disable_optimizer": True}):
        with pytest.raises(kglite.SchemaError, match="Undefined variable 'b'"):
            boundary_graph.cypher(query, **kwargs).to_list()


def test_the_having_spelling_is_the_same_field_and_the_same_bail(boundary_graph) -> None:
    """`HAVING` and a WITH-attached `WHERE` parse into one field.

    So the hoist sees them identically, and the aggregate bail (H2) is what
    keeps a HAVING from being applied before its own aggregation.
    """
    for spelling in ("WHERE", "HAVING"):
        query = f"MATCH (p:P) WITH p.city AS c, count(*) AS k {spelling} k > 2 RETURN c, k"
        for rows in both_profiles(boundary_graph, query):
            assert rows == [{"c": "Y", "k": 3}]


def test_an_unbound_parameter_still_raises_through_the_boundary(boundary_graph) -> None:
    """A missing parameter must raise, not silently match nothing.

    The hoist moves the predicate onto the fused-aggregate path that the
    0.17.0 release hardened for exactly this, and `WITH … WHERE` was one of
    the four routes that used to swallow it.
    """
    query = "MATCH (p:P) WITH p WHERE p.city = $missing RETURN count(*) AS n"
    for kwargs in ({}, {"disable_optimizer": True}):
        with pytest.raises(kglite.CypherExecutionError, match="Missing parameter"):
            boundary_graph.cypher(query, **kwargs).to_list()


def test_an_aggregate_inside_the_predicate_is_still_refused(boundary_graph) -> None:
    """`WITH p WHERE count(*) > 1` has no aggregation to be a HAVING over.

    The refusal is the contract; H2's predicate half keeps the rewrite from
    turning it into a plain WHERE with a different diagnosis.
    """
    query = "MATCH (p:P) WITH p WHERE count(*) > 1 RETURN p.title AS t"
    for kwargs in ({}, {"disable_optimizer": True}):
        with pytest.raises(kglite.CypherExecutionError, match="cannot be used outside of RETURN/WITH"):
            boundary_graph.cypher(query, **kwargs).to_list()
