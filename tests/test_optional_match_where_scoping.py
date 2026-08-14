"""Absolute goldens for ``OPTIONAL MATCH ... WHERE`` predicate scoping.

A ``WHERE`` written directly after ``OPTIONAL MATCH`` is **part of that
clause's pattern description**, not an independent post-filter. Neo4j's Cypher
manual states it plainly: *"The WHERE clause is part of the pattern
description, and its predicates will be considered while looking for matches,
not after"*, and *"this matters especially in the case of multiple (OPTIONAL)
MATCH clauses, where it is crucial to put WHERE together with the MATCH it
belongs to"*. openCypher's grammar agrees — ``Match = ['OPTIONAL'] 'MATCH'
Pattern [Where]``, one production, not two clauses.

KGLite parsed it as a separate ``Clause::Where`` and ran it over the already
null-padded rows, so a candidate that failed the predicate did not merely fail
to match — it **deleted the null-extended row that its absence should have
produced**::

    MATCH (p:P) OPTIONAL MATCH (p)-[:KNOWS]->(x:Q) WHERE x.w > 5
    RETURN p.name, x.name        -- 1 row, Neo4j returns 2

The inline-property spelling of the same filter
(``OPTIONAL MATCH (p)-[:KNOWS]->(x:Q {w: 9})``) was always correct, so the two
spellings of one filter disagreed. Both the optimised and unoptimised paths
shared the defect, which is why the differential corpus was blind to it: every
case here is an **absolute** expected value, and every one names the
null-extended row explicitly — a row-count assertion alone would have been
satisfied by the wrong answer in half of them.

The contrast case at the bottom is load-bearing: ``WITH ... WHERE`` after an
OPTIONAL MATCH *is* an independent post-filter (it belongs to the WITH), and
must keep deleting rows.
"""

from __future__ import annotations

import pytest

import kglite


def rows(g: kglite.KnowledgeGraph, query: str, **kwargs) -> list:
    return g.cypher(query, **kwargs).to_list()


@pytest.fixture
def g() -> kglite.KnowledgeGraph:
    """Two P nodes, each with one KNOWS target; only ``b``'s passes ``w > 5``.

    ``a``'s target exists but fails the predicate — the case that separates
    "no match, null-extend" from "matched then filtered away".
    """
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:P {id: 1, name: 'a', threshold: 5})")
    g.cypher("CREATE (:P {id: 2, name: 'b', threshold: 5})")
    g.cypher("MATCH (p:P {name: 'a'}) CREATE (p)-[:KNOWS]->(:Q {id: 11, name: 'q1', w: 1})")
    g.cypher("MATCH (p:P {name: 'b'}) CREATE (p)-[:KNOWS]->(:Q {id: 12, name: 'q2', w: 9})")
    return g


# ── the filed repro ──────────────────────────────────────────────────────


def test_predicate_on_the_optional_variable_null_extends_instead_of_deleting(g):
    """The filed repro: 2 rows, one of them NULL-extended."""
    assert rows(
        g,
        "MATCH (p:P) OPTIONAL MATCH (p)-[:KNOWS]->(x:Q) WHERE x.w > 5 RETURN p.name AS p, x.name AS x ORDER BY p",
    ) == [{"p": "a", "x": None}, {"p": "b", "x": "q2"}]


def test_inline_property_spelling_agrees_with_the_where_spelling(g):
    """The control that was already correct — the two spellings must match."""
    inline = rows(
        g,
        "MATCH (p:P) OPTIONAL MATCH (p)-[:KNOWS]->(x:Q {w: 9}) RETURN p.name AS p, x.name AS x ORDER BY p",
    )
    where = rows(
        g,
        "MATCH (p:P) OPTIONAL MATCH (p)-[:KNOWS]->(x:Q) WHERE x.w = 9 RETURN p.name AS p, x.name AS x ORDER BY p",
    )
    assert inline == [{"p": "a", "x": None}, {"p": "b", "x": "q2"}]
    assert where == inline


# ── predicate variable scopes ────────────────────────────────────────────


def test_predicate_mixing_outer_and_optional_variables(g):
    """``x.w > p.threshold`` — a correlated predicate is still a match filter."""
    assert rows(
        g,
        "MATCH (p:P) OPTIONAL MATCH (p)-[:KNOWS]->(x:Q) WHERE x.w > p.threshold "
        "RETURN p.name AS p, x.name AS x ORDER BY p",
    ) == [{"p": "a", "x": None}, {"p": "b", "x": "q2"}]


def test_predicate_on_outer_variables_only_still_scopes_to_the_match(g):
    """Neo4j semantics: an outer-only predicate governs *matching*, not the row.

    ``WHERE p.name = 'b'`` inside the OPTIONAL MATCH is a join condition, so
    ``a`` keeps its row with ``x`` NULL. It does NOT filter ``a`` out — that is
    what ``MATCH ... WHERE`` or ``WITH ... WHERE`` would do.
    """
    assert rows(
        g,
        "MATCH (p:P) OPTIONAL MATCH (p)-[:KNOWS]->(x:Q) WHERE p.name = 'b' RETURN p.name AS p, x.name AS x ORDER BY p",
    ) == [{"p": "a", "x": None}, {"p": "b", "x": "q2"}]


def test_predicate_matching_nothing_null_extends_every_row(g):
    assert rows(
        g,
        "MATCH (p:P) OPTIONAL MATCH (p)-[:KNOWS]->(x:Q) WHERE x.w > 100 RETURN p.name AS p, x.name AS x ORDER BY p",
    ) == [{"p": "a", "x": None}, {"p": "b", "x": None}]


def test_null_evaluating_predicate_is_no_match_not_a_kept_row(g):
    """Three-valued logic: a predicate on an absent property is not TRUE."""
    assert rows(
        g,
        "MATCH (p:P) OPTIONAL MATCH (p)-[:KNOWS]->(x:Q) WHERE x.missing > 1 RETURN p.name AS p, x.name AS x ORDER BY p",
    ) == [{"p": "a", "x": None}, {"p": "b", "x": None}]


def test_parameterised_predicate_scopes_the_same_way(g):
    assert rows(
        g,
        "MATCH (p:P) OPTIONAL MATCH (p)-[:KNOWS]->(x:Q) WHERE x.w > $floor RETURN p.name AS p, x.name AS x ORDER BY p",
        params={"floor": 5},
    ) == [{"p": "a", "x": None}, {"p": "b", "x": "q2"}]


# ── first clause of the query ────────────────────────────────────────────


def test_leading_optional_match_with_where_that_matches(g):
    assert rows(
        g,
        "OPTIONAL MATCH (x:Q) WHERE x.w > 5 RETURN x.name AS x",
    ) == [{"x": "q2"}]


def test_leading_optional_match_with_where_that_matches_nothing(g):
    """No candidate passes → one all-NULL row, not zero rows."""
    assert rows(
        g,
        "OPTIONAL MATCH (x:Q) WHERE x.w > 100 RETURN x.name AS x",
    ) == [{"x": None}]


# ── several OPTIONAL MATCHes, each owning its own WHERE ──────────────────


def test_each_optional_match_owns_the_where_that_follows_it(g):
    g.cypher("MATCH (p:P {name: 'a'}) CREATE (p)-[:LIKES]->(:R {id: 21, name: 'r1', v: 7})")
    g.cypher("MATCH (p:P {name: 'b'}) CREATE (p)-[:LIKES]->(:R {id: 22, name: 'r2', v: 2})")
    assert rows(
        g,
        "MATCH (p:P) "
        "OPTIONAL MATCH (p)-[:KNOWS]->(x:Q) WHERE x.w > 5 "
        "OPTIONAL MATCH (p)-[:LIKES]->(y:R) WHERE y.v > 5 "
        "RETURN p.name AS p, x.name AS x, y.name AS y ORDER BY p",
    ) == [
        {"p": "a", "x": None, "y": "r1"},
        {"p": "b", "x": "q2", "y": None},
    ]


def test_later_optional_match_referencing_a_null_binding(g):
    """``x`` is NULL for ``a``; the chained pattern off ``x`` cannot match."""
    g.cypher("MATCH (q:Q {name: 'q2'}) CREATE (q)-[:OWNS]->(:S {id: 31, name: 's2', k: 3})")
    assert rows(
        g,
        "MATCH (p:P) "
        "OPTIONAL MATCH (p)-[:KNOWS]->(x:Q) WHERE x.w > 5 "
        "OPTIONAL MATCH (x)-[:OWNS]->(s:S) WHERE s.k > 1 "
        "RETURN p.name AS p, x.name AS x, s.name AS s ORDER BY p",
    ) == [
        {"p": "a", "x": None, "s": None},
        {"p": "b", "x": "q2", "s": "s2"},
    ]


# ── aggregation (the fusion passes) ──────────────────────────────────────


def test_scoped_where_with_count_keeps_the_zero_group(g):
    """``a`` must survive with ``count(x) = 0``, not vanish from the result."""
    assert rows(
        g,
        "MATCH (p:P) OPTIONAL MATCH (p)-[:KNOWS]->(x:Q) WHERE x.w > 5 RETURN p.name AS p, count(x) AS n ORDER BY p",
    ) == [{"p": "a", "n": 0}, {"p": "b", "n": 1}]


def test_scoped_where_with_count_through_with(g):
    assert rows(
        g,
        "MATCH (p:P) OPTIONAL MATCH (p)-[:KNOWS]->(x:Q) WHERE x.w > 5 "
        "WITH p, count(x) AS n RETURN p.name AS p, n ORDER BY p",
    ) == [{"p": "a", "n": 0}, {"p": "b", "n": 1}]


def test_scoped_where_with_collect(g):
    assert rows(
        g,
        "MATCH (p:P) OPTIONAL MATCH (p)-[:KNOWS]->(x:Q) WHERE x.w > 5 "
        "RETURN p.name AS p, collect(x.name) AS xs ORDER BY p",
    ) == [{"p": "a", "xs": []}, {"p": "b", "xs": ["q2"]}]


# ── contrast: forms that are NOT scoped and must keep filtering ──────────


def test_with_where_after_optional_match_still_filters_rows(g):
    """``WITH ... WHERE`` belongs to the WITH — it deletes the NULL row."""
    assert rows(
        g,
        "MATCH (p:P) OPTIONAL MATCH (p)-[:KNOWS]->(x:Q) WITH p, x WHERE x.w > 5 "
        "RETURN p.name AS p, x.name AS x ORDER BY p",
    ) == [{"p": "b", "x": "q2"}]


def test_plain_match_where_is_unchanged(g):
    assert rows(
        g,
        "MATCH (p:P)-[:KNOWS]->(x:Q) WHERE x.w > 5 RETURN p.name AS p, x.name AS x ORDER BY p",
    ) == [{"p": "b", "x": "q2"}]


def test_where_after_a_plain_match_preceding_an_optional_match(g):
    """The WHERE binds to the plain MATCH it follows, not to the OPTIONAL one."""
    assert rows(
        g,
        "MATCH (p:P) WHERE p.name = 'b' OPTIONAL MATCH (p)-[:KNOWS]->(x:Q) RETURN p.name AS p, x.name AS x ORDER BY p",
    ) == [{"p": "b", "x": "q2"}]
