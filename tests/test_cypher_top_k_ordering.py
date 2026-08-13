"""Golden orderings for multi-key ORDER BY + LIMIT (fused top-K).

The differential corpus compares *sets* of rows (`_normalize` sorts them), so
it cannot see a fused top-K that picks the right rows and emits them in the
wrong order — and it cannot see a bug the fused and unfused paths share,
because they now rank rows through one comparator
(`executor/ordering.rs::compare_sort_keys`). Both gaps need absolute expected
values, which is what this file holds.

Every case asserts three things:

1. the *fused* plan is what runs (a golden that passes because the pass bailed
   proves nothing);
2. the exact emitted order, written out by hand;
3. equality with the same query run through `disable_optimizer=True`.

Regression origin (0.15.14): both fused top-K executors dropped rows whose
sort key was NULL, so `ORDER BY x DESC LIMIT k` — which places NULLs *first*
by Neo4j 5+ default — returned the wrong rows, and `LIMIT k` could return
fewer than k rows that existed. The `nulls_*` cases pin that.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite


@pytest.fixture
def rank_graph() -> kglite.KnowledgeGraph:
    """Ties on every key, NULLs in both key positions, both orders of arrival.

    id | name | grp    | score | tier
     1 | A    | 'x'    | 10    | 2
     2 | B    | 'x'    | 10    | 1     <- ties A on grp+score, split by tier
     3 | C    | 'x'    | None  | 1     <- NULL leading-key-adjacent
     4 | D    | 'y'    | 5     | 3
     5 | E    | None   | 5     | 1     <- NULL in the leading key
     6 | F    | 'y'    | 5     | None  <- NULL in the trailing key
     7 | G    | 'x'    | 10    | 1     <- exact tie with B, split by insertion
    """
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        [
            {"id": 1, "name": "A", "grp": "x", "score": 10.0, "tier": 2},
            {"id": 2, "name": "B", "grp": "x", "score": 10.0, "tier": 1},
            {"id": 3, "name": "C", "grp": "x", "score": None, "tier": 1},
            {"id": 4, "name": "D", "grp": "y", "score": 5.0, "tier": 3},
            {"id": 5, "name": "E", "grp": None, "score": 5.0, "tier": 1},
            {"id": 6, "name": "F", "grp": "y", "score": 5.0, "tier": None},
            {"id": 7, "name": "G", "grp": "x", "score": 10.0, "tier": 1},
        ]
    )
    g.add_nodes(df, "N", "id", "name")
    return g


def _plan(graph: kglite.KnowledgeGraph, query: str) -> list[str]:
    return [row["operation"] for row in graph.cypher(f"EXPLAIN {query}").to_list()]


def _names(graph: kglite.KnowledgeGraph, query: str, *, fused: str | None) -> list[str]:
    """Run `query`, assert the intended plan, assert optimizer/naive agreement."""
    if fused is not None:
        plan = _plan(graph, query)
        assert any(op.startswith(fused) for op in plan), (
            f"expected a {fused} plan for `{query}`, got {plan} — a golden that "
            "passes on the unfused path does not test the fused one"
        )
    optimized = [row["n"] for row in graph.cypher(query).to_list()]
    naive = [row["n"] for row in graph.cypher(query, disable_optimizer=True).to_list()]
    assert optimized == naive, f"fused top-K diverged from the full sort on `{query}`"
    return optimized


# ── multi-key selection + order ──────────────────────────────────────


def test_two_key_order_by_limit(rank_graph):
    q = "MATCH (x:N) RETURN x.name AS n ORDER BY x.grp, x.score LIMIT 4"
    # grp ASC → NULLS LAST, so E (NULL grp) sorts last and is out of the top 4.
    # 'x': C (NULL score, ASC → last within its group) after A/B/G (10.0)...
    # A/B/G tie on (grp='x', score=10) → stable input order A, B, G.
    assert _names(rank_graph, q, fused="FusedNodeScanTopK") == ["A", "B", "G", "C"]


def test_three_key_order_by_limit(rank_graph):
    q = "MATCH (x:N) RETURN x.name AS n ORDER BY x.grp, x.score, x.tier LIMIT 4"
    # The third key splits the A/B/G tie: tier 2, 1, 1 → B and G (tier 1,
    # input order) then A.
    assert _names(rank_graph, q, fused="FusedNodeScanTopK") == ["B", "G", "A", "C"]


def test_mixed_directions_are_per_key(rank_graph):
    q = "MATCH (x:N) RETURN x.name AS n ORDER BY x.grp DESC, x.score ASC LIMIT 5"
    # grp DESC → NULLS FIRST: E first. Then 'y' (D, F both score 5, input
    # order), then 'x' ascending by score: A/B/G at 10 lose to... no: ASC puts
    # 5 before 10, and 'x' has no 5s, so C's NULL score (ASC → LAST) trails
    # the 10s.
    assert _names(rank_graph, q, fused="FusedNodeScanTopK") == ["E", "D", "F", "A", "B"]


def test_ties_on_every_key_keep_input_order(rank_graph):
    q = "MATCH (x:N) WHERE x.score = 10.0 RETURN x.name AS n ORDER BY x.grp, x.score LIMIT 2"
    assert _names(rank_graph, q, fused="FusedNodeScanTopK") == ["A", "B"]


# ── NULL placement (the 0.15.14 regression) ──────────────────────────


def test_nulls_lead_a_descending_key(rank_graph):
    q = "MATCH (x:N) RETURN x.name AS n ORDER BY x.score DESC, x.name ASC LIMIT 3"
    # DESC defaults to NULLS FIRST → C (NULL score) wins outright. The fused
    # paths used to drop it and answer A, B, G.
    assert _names(rank_graph, q, fused="FusedNodeScanTopK") == ["C", "A", "B"]


def test_nulls_trail_an_ascending_key(rank_graph):
    q = "MATCH (x:N) RETURN x.name AS n ORDER BY x.score ASC, x.name ASC LIMIT 7"
    assert _names(rank_graph, q, fused="FusedNodeScanTopK") == [
        "D",
        "E",
        "F",
        "A",
        "B",
        "G",
        "C",
    ]


def test_limit_is_filled_when_keys_are_null(rank_graph):
    """A NULL sort key must not shrink the result below LIMIT."""
    q = "MATCH (x:N) WHERE x.score IS NULL RETURN x.name AS n ORDER BY x.score DESC LIMIT 3"
    assert _names(rank_graph, q, fused="FusedNodeScanTopK") == ["C"]
    q = "MATCH (x:N) RETURN x.name AS n ORDER BY x.tier ASC, x.name ASC LIMIT 6"
    # tier NULL (F) goes last under ASC; six rows still come back.
    assert len(_names(rank_graph, q, fused="FusedNodeScanTopK")) == 6


def test_explicit_nulls_modifier_overrides_the_direction_default(rank_graph):
    q = "MATCH (x:N) RETURN x.name AS n ORDER BY x.score DESC NULLS LAST, x.name ASC LIMIT 4"
    assert _names(rank_graph, q, fused="FusedNodeScanTopK") == ["A", "B", "G", "D"]
    q = "MATCH (x:N) RETURN x.name AS n ORDER BY x.grp ASC NULLS FIRST, x.name ASC LIMIT 3"
    assert _names(rank_graph, q, fused="FusedNodeScanTopK") == ["E", "A", "B"]


def test_nulls_in_the_second_key(rank_graph):
    q = "MATCH (x:N) RETURN x.name AS n ORDER BY x.grp DESC, x.tier ASC, x.name ASC LIMIT 4"
    # grp DESC → NULLS FIRST: E. Then 'y': D (tier 3) vs F (tier NULL, ASC →
    # last) → D, F. Then 'x' by tier ASC: B/G (1), A (2), C (1) → B, C, G...
    assert _names(rank_graph, q, fused="FusedNodeScanTopK") == ["E", "D", "F", "B"]


# ── alias resolution ─────────────────────────────────────────────────


def test_order_by_projected_aliases(rank_graph):
    q = "MATCH (x:N) RETURN x.name AS n, x.grp AS g, x.score AS s ORDER BY g, s DESC LIMIT 4"
    # g ASC (NULLS LAST → E out), then s DESC within 'x', which defaults to
    # NULLS FIRST → C leads its group, then the 10.0 ties in input order.
    assert _names(rank_graph, q, fused="FusedNodeScanTopK") == ["C", "A", "B", "G"]


def test_order_by_an_expression_over_an_alias_does_not_fuse(rank_graph):
    """`s` is unbound before projection — fusing this returned zero rows."""
    q = "MATCH (x:N) RETURN x.name AS n, x.score AS s ORDER BY s + 1 DESC, x.name LIMIT 3"
    plan = _plan(rank_graph, q)
    assert not any("TopK" in op for op in plan), f"a sort key reading a RETURN alias must not fuse; plan was {plan}"
    assert _names(rank_graph, q, fused=None) == ["C", "A", "B"]


def test_order_by_alias_over_a_with_binding_still_fuses(rank_graph):
    q = "MATCH (x:N) WITH x.name AS n, x.grp AS g, x.score AS s RETURN n, g, s ORDER BY g, s LIMIT 3"
    assert _names(rank_graph, q, fused="FusedOrderByTopK") == ["A", "B", "G"]


# ── the generic (non-node-scan) fused path ───────────────────────────


@pytest.fixture
def edge_graph() -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    nodes = pd.DataFrame(
        [
            {"id": 1, "name": "A", "w": 3},
            {"id": 2, "name": "B", "w": 1},
            {"id": 3, "name": "C", "w": 1},
            {"id": 4, "name": "D", "w": 2},
        ]
    )
    g.add_nodes(nodes, "N", "id", "name")
    edges = pd.DataFrame(
        [
            {"src": 1, "dst": 2, "cost": 5.0},
            {"src": 1, "dst": 3, "cost": None},
            {"src": 2, "dst": 3, "cost": 5.0},
            {"src": 4, "dst": 1, "cost": 1.0},
        ]
    )
    g.add_connections(edges, "LINK", "N", "src", "N", "dst")
    return g


def test_two_key_order_by_limit_over_a_pattern(edge_graph):
    q = (
        "MATCH (a:N)-[r:LINK]->(b:N) RETURN a.name AS n, b.name AS bn, r.cost AS c "
        "ORDER BY r.cost ASC, b.name DESC LIMIT 3"
    )
    # cost ASC → NULLS LAST: 1.0 (D→A), then the two 5.0 rows split by b.name
    # DESC (C before B), then the NULL-cost row.
    assert _names(edge_graph, q, fused="FusedOrderByTopK") == ["D", "B", "A"]


def test_two_key_order_by_desc_over_a_pattern_keeps_null_rows(edge_graph):
    q = "MATCH (a:N)-[r:LINK]->(b:N) RETURN a.name AS n, r.cost AS c ORDER BY r.cost DESC, a.name ASC LIMIT 2"
    # DESC → NULLS FIRST: the NULL-cost row (A→C) leads.
    assert _names(edge_graph, q, fused="FusedOrderByTopK") == ["A", "A"]


def test_sort_key_column_is_projected_from_the_computed_key(edge_graph):
    """A RETURN item that *is* a sort key is projected from the key tuple —
    it must keep the value's type, not a coerced float."""
    rows = edge_graph.cypher("MATCH (x:N) RETURN x.name AS n, x.w AS w ORDER BY x.w DESC, x.name LIMIT 2").to_list()
    assert rows == [{"n": "A", "w": 3}, {"n": "D", "w": 2}]
    assert all(isinstance(row["w"], int) for row in rows)


def test_a_sort_key_beyond_float_precision_is_projected_exactly():
    """The fused top-K used to rank on an `f64` score and project the RETURN
    column *from that score*, so an integer past 2^53 came back rounded."""
    g = kglite.KnowledgeGraph()
    big = 9007199254740993  # 2**53 + 1: not representable as f64
    g.add_nodes(
        pd.DataFrame([{"id": 1, "name": "A", "v": big}, {"id": 2, "name": "B", "v": 1}]),
        "N",
        "id",
        "name",
    )
    q = "MATCH (x:N) RETURN x.name AS n, x.v AS v ORDER BY x.v DESC LIMIT 1"
    assert g.cypher(q).to_list() == [{"n": "A", "v": big}]
    assert g.cypher(q).to_list() == g.cypher(q, disable_optimizer=True).to_list()
