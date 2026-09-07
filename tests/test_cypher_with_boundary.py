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


# ── fold_aliasing_with / hoist_terminal_return_over_with_top_k (T2-2) ──


def test_column_names_survive_the_substitution(boundary_graph) -> None:
    """`RETURN i` after `i := p.id` must still be column `i`.

    The corpus compares two plans against each other, so a column both
    plans renamed identically would stay green. These are the names.
    """
    for rows in both_profiles(boundary_graph, "MATCH (p:P) WITH p.title AS n RETURN n"):
        assert list(rows[0]) == ["n"]
        assert rows == [{"n": t} for t in "abcde"]
    for rows in both_profiles(boundary_graph, "MATCH (p:P) WITH p, p.age AS a RETURN p.title ORDER BY a DESC LIMIT 2"):
        assert rows == [{"p.title": "e"}, {"p.title": "d"}]


def test_the_reorder_keeps_tie_order_and_null_placement(boundary_graph) -> None:
    """Three rows share the sort key and one carries no key at all.

    `hoist_terminal_return_over_with_top_k` moves the RETURN ahead of the
    ORDER BY / LIMIT it followed. That is only an identity if the projection
    is order-preserving, so ties must still resolve by input order and the
    NULL must still land where its direction's default puts it (DESC →
    NULLS FIRST, ASC → NULLS LAST).
    """
    descending = "MATCH (p:P) WITH p, p.age AS a ORDER BY a DESC LIMIT 3 RETURN p.title"
    for rows in both_profiles(boundary_graph, descending):
        # e has no age (NULLS FIRST), d is 55, then the 30-tie in input order.
        assert rows == [{"p.title": "e"}, {"p.title": "d"}, {"p.title": "a"}]
    ascending = "MATCH (p:P) WITH p, p.age AS a ORDER BY a ASC LIMIT 3 RETURN p.title"
    for rows in both_profiles(boundary_graph, ascending):
        # NULLS LAST, so the whole 30-tie comes first, in input order.
        assert rows == [{"p.title": "a"}, {"p.title": "b"}, {"p.title": "c"}]
    skipped = "MATCH (p:P) WITH p, p.age AS a ORDER BY a DESC SKIP 1 LIMIT 2 RETURN p.title"
    for rows in both_profiles(boundary_graph, skipped):
        assert rows == [{"p.title": "d"}, {"p.title": "a"}]


def test_a_hidden_order_key_is_still_a_scope_error(boundary_graph) -> None:
    """F4: substitution must not re-expose what the WITH dropped.

    `WITH p.id AS i, …` drops `p`, so `ORDER BY p.city` is undefined. The
    fold could make it evaluable, and must not.
    """
    query = "MATCH (p:P) WITH p.id AS i, p.age AS a RETURN i ORDER BY p.city LIMIT 2"
    for kwargs in ({}, {"disable_optimizer": True}):
        with pytest.raises(kglite.SchemaError, match="Undefined variable 'p'"):
            boundary_graph.cypher(query, **kwargs).to_list()


def test_order_by_reads_a_with_alias_the_return_does_not_project(boundary_graph) -> None:
    """A sort key naming a WITH value the RETURN drops must still sort.

    ORDER BY runs after the projection, and the projection replaced the
    row's projected map — so this sorted on a null key for every row and
    silently returned input order, on *both* plan profiles. The LIMIT form
    was right only because a top-K fusion happened to claim it.
    """
    unlimited = "MATCH (p:P) WITH p, p.age AS a RETURN p.title AS t ORDER BY a DESC"
    for rows in both_profiles(boundary_graph, unlimited):
        assert [row["t"] for row in rows] == ["e", "d", "a", "b", "c"]
    limited = "MATCH (p:P) WITH p, p.age AS a RETURN p.title AS t ORDER BY a DESC LIMIT 3"
    for rows in both_profiles(boundary_graph, limited):
        assert [row["t"] for row in rows] == ["e", "d", "a"]


def test_a_projected_column_wins_over_the_carried_scope(boundary_graph) -> None:
    """`RETURN p.title AS a ORDER BY a` sorts by the projected `a`.

    The carried pre-projection value only fills a hole; a column the RETURN
    defines shadows it, which is the Cypher precedence.
    """
    query = "MATCH (p:P) WITH p, p.age AS a RETURN p.title AS a ORDER BY a DESC LIMIT 3"
    for rows in both_profiles(boundary_graph, query):
        assert [row["a"] for row in rows] == ["e", "d", "c"]


def test_return_star_is_not_fused_into_a_top_k(boundary_graph) -> None:
    """`RETURN *` expands from the runtime row, which no fused operator builds.

    The top-K fusions projected the literal `Star` instead, so the same
    query answered `[{'*': 1}, …]` with a LIMIT and the real rows without
    one.
    """
    with_limit = boundary_graph.cypher("MATCH (p:P) RETURN * ORDER BY p.age DESC LIMIT 2").to_list()
    without = boundary_graph.cypher("MATCH (p:P) RETURN * ORDER BY p.age DESC").to_list()
    assert list(with_limit[0]) == ["p"]
    assert with_limit == without[:2]
