"""Repro: WHERE-on-rel-var drops cohort fast path on Wikidata-style cohorts.

Models the user-reported timeout on Wikidata:

    MATCH (p)-[:P27]->({id: 20})            -- citizens of Norway (Q20)
    WITH p
    MATCH (p)-[r]-(other)                   -- all incident edges
    WHERE NOT (type(r) = 'P50' AND startNode(r) = other)
    RETURN p.title AS name, p.description AS desc,
           count(r) AS connections
    ORDER BY connections DESC LIMIT 10

The semantics are equivalent to:

    total - (incoming P50 edges)

A second variant computes the same answer via the "narrow-then-enrich" form
documented in the cohort fusion fall-back memo.

The test asserts both forms return the same answer and the property-valued
grouping stays on the safe eager aggregate path. The fused accumulator groups
by node identity, which is not equivalent when multiple nodes share the same
title and description.
"""

from __future__ import annotations

import time

import pytest

from kglite import KnowledgeGraph


def _build_mini_wikidata() -> tuple[KnowledgeGraph, int]:
    """Build a tiny Wikidata-style graph and return (graph, country_id).

    `id` is auto-assigned by kglite; we capture the country's id and use
    it in the test queries so the anchor `{id: <id>}` matches.
    """
    g = KnowledgeGraph()
    g.cypher("CREATE (:Country {title: 'Norway'})")
    g.cypher("CREATE (:Place {title: 'Bergen'})")
    # Cohort: 50 humans with a deterministic degree spread. Higher-i
    # humans have fewer outgoing edges and fewer incoming P50, so the
    # `total - p50_in` scoreboard has a stable order.
    for i in range(50):
        g.cypher(f"CREATE (:Human {{title: 'H{i:02d}', description: 'd{i}'}})")
        g.cypher(f"MATCH (h:Human {{title: 'H{i:02d}'}}), (c:Country {{title: 'Norway'}}) CREATE (h)-[:P27]->(c)")
        out_edges = 60 - i
        for _ in range(out_edges):
            g.cypher(f"MATCH (h:Human {{title: 'H{i:02d}'}}), (p:Place {{title: 'Bergen'}}) CREATE (h)-[:P19]->(p)")
        in_edges = max(0, 30 - i)
        for k in range(in_edges):
            g.cypher(f"CREATE (:Book {{title: 'b{i:02d}_{k}'}})")
            g.cypher(
                f"MATCH (b:Book {{title: 'b{i:02d}_{k}'}}), (h:Human {{title: 'H{i:02d}'}}) CREATE (b)-[:P50]->(h)"
            )
    norway_id = list(g.cypher("MATCH (c:Country {title: 'Norway'}) RETURN c.id AS id"))[0]["id"]
    return g, int(norway_id)


@pytest.fixture(scope="module")
def mini_wikidata():
    g, norway_id = _build_mini_wikidata()
    return g, norway_id


def _where_query(country_id: int) -> str:
    return f"""
MATCH (p)-[:P27]->({{id: {country_id}}})
WITH p
MATCH (p)-[r]-(other)
WHERE NOT (type(r) = 'P50' AND startNode(r) = other)
RETURN p.title AS name, p.description AS desc, count(r) AS connections
ORDER BY connections DESC LIMIT 10
"""


def _optional_query(country_id: int) -> str:
    return f"""
MATCH (p)-[:P27]->({{id: {country_id}}})
WITH p
MATCH (p)-[r]-()
WITH p, count(r) AS total
OPTIONAL MATCH ()-[rp:P50]->(p)
RETURN p.title AS name, total, count(rp) AS p50_in,
       total - count(rp) AS connections
ORDER BY connections DESC LIMIT 10
"""


def _names(rows):
    return [(r["name"], r["connections"]) for r in rows]


def test_where_form_matches_optional_form(mini_wikidata):
    """Both query shapes must produce identical top-10 cohort results."""
    g, country_id = mini_wikidata
    a = g.cypher(_where_query(country_id))
    b = g.cypher(_optional_query(country_id))
    a_rows, b_rows = list(a), list(b)
    assert len(a_rows) == 10
    assert len(b_rows) == 10
    assert _names(a_rows) == _names(b_rows)


def test_where_form_avoids_node_keyed_fusion(mini_wikidata):
    """Property-valued grouping must not land on a NodeIndex-keyed fusion."""
    g, country_id = mini_wikidata
    plan = g.cypher("EXPLAIN " + _where_query(country_id))
    ops = [row["operation"] for row in plan]
    assert "FusedMatchWithAggregate" not in ops, ops
    assert "With" in ops, ops


def test_where_form_runtime(mini_wikidata):
    """Sanity timing: the safe path remains well under 1s on this tiny graph."""
    g, country_id = mini_wikidata
    t0 = time.perf_counter()
    rows = list(g.cypher(_where_query(country_id)))
    elapsed = time.perf_counter() - t0
    assert len(rows) == 10
    # Loose ceiling — the regression we care about is "minutes / timeout".
    assert elapsed < 2.0, f"Query took {elapsed:.2f}s on a 50-human graph"
