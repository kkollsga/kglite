"""Aggregates nested inside wrapper expressions — golden expected-value tests.

Fixed 2026-08-15. Pre-fix, the materialized aggregate evaluator recognized
aggregates only at the top level of a projection item: any aggregate nested in
a map literal, list literal, negation, CASE result, comparison, map
projection, or list comprehension source failed with the self-contradicting
"Aggregate function '…' cannot be used outside of RETURN/WITH" — while
sitting in RETURN. `[count(*)]` additionally wasn't classified as an
aggregate projection at all (missing ListLiteral arm in
`ast::is_aggregate_expression`).

This is the class the differential corpus alone cannot pin to values (it only
checks the two optimizer legs agree), so the expected values live here. The
Neo4j Browser connect sequence (`dbMetaDuck.ts` metaTypesQuery /
metaCountQuery) is included verbatim — it is the real-world consumer that
surfaced the bug.
"""

from __future__ import annotations

import pytest

import kglite


@pytest.fixture
def graph():
    g = kglite.KnowledgeGraph()
    g.cypher(
        "CREATE (:Person {name: 'ann', age: 28}) CREATE (:Person {name: 'bob', age: 35}) CREATE (:City {name: 'Oslo'})"
    )
    g.cypher("MATCH (a:Person {name: 'ann'}), (c:City) CREATE (a)-[:LIVES_IN]->(c)")
    return g


def test_count_in_map_literal(graph):
    rows = graph.cypher("MATCH (n) RETURN {name:'nodes', data: count(*)} AS r")
    assert rows[0]["r"] == {"name": "nodes", "data": 3}


def test_collect_in_map_literal(graph):
    rows = graph.cypher("MATCH (p:Person) RETURN {names: collect(p.name)} AS r")
    assert sorted(rows[0]["r"]["names"]) == ["ann", "bob"]


def test_count_in_list_literal(graph):
    rows = graph.cypher("MATCH (p:Person) RETURN [count(*), min(p.age)] AS r")
    assert rows[0]["r"] == [2, 28]


def test_negated_aggregate(graph):
    rows = graph.cypher("MATCH (n) RETURN -count(*) AS r")
    assert rows[0]["r"] == -3


def test_aggregate_in_case_result(graph):
    rows = graph.cypher("MATCH (n) RETURN CASE WHEN true THEN count(*) ELSE 0 END AS r")
    assert rows[0]["r"] == 3


def test_aggregate_in_comparison(graph):
    rows = graph.cypher("MATCH (n) RETURN count(*) > 2 AS r")
    assert rows[0]["r"] is True
    rows = graph.cypher("MATCH (n) RETURN count(*) > 99 AS r")
    assert rows[0]["r"] is False


def test_aggregate_in_map_projection(graph):
    rows = graph.cypher("MATCH (p:Person {name: 'ann'}) RETURN p {.name, total: count(*)} AS r")
    assert rows[0]["r"] == {"name": "ann", "total": 1}


def test_collect_feeding_list_comprehension(graph):
    rows = graph.cypher("MATCH (p:Person) RETURN [v IN collect(p.age) | v * 2] AS r")
    assert sorted(rows[0]["r"]) == [56, 70]


def test_grouped_nested_aggregate(graph):
    rows = graph.cypher("MATCH (n) RETURN labels(n)[0] AS l, {c: count(*)} AS r ORDER BY l")
    assert rows.to_dicts() == [
        {"l": "City", "r": {"c": 1}},
        {"l": "Person", "r": {"c": 2}},
    ]


def test_nested_aggregate_in_with(graph):
    rows = graph.cypher("MATCH (n) WITH {c: count(*)} AS m RETURN m.c AS r")
    assert rows[0]["r"] == 3


def test_nested_aggregate_over_empty_rowset(graph):
    """Neo4j: aggregating zero rows still yields one row — {c: 0}, not []."""
    rows = graph.cypher("MATCH (p:Person) WHERE p.age > 999 RETURN {c: count(*)} AS r")
    assert rows.to_dicts() == [{"r": {"c": 0}}]


def test_size_of_collect_still_works(graph):
    """The pre-existing FunctionCall-wrapping-aggregate path is unchanged."""
    rows = graph.cypher("MATCH (p:Person) RETURN size(collect(p.name)) AS r")
    assert rows[0]["r"] == 2


def test_collect_slice_returns_list(graph):
    """`collect(x)[..k]` is a list, matching the scalar slice path — pre-fix
    the aggregate-path ListSlice arm serialized to a JSON string."""
    rows = graph.cypher("MATCH (p:Person) RETURN collect(p.name)[..1] AS r")
    assert isinstance(rows[0]["r"], list)
    rows = graph.cypher("MATCH (p:Person) RETURN collect(p.name)[..0] AS r")
    assert rows[0]["r"] == []


# ── The Neo4j Browser connect sequence, verbatim from dbMetaDuck.ts ──


def test_browser_meta_types_query(graph):
    rows = graph.cypher(
        "CALL db.labels() YIELD label\n"
        "RETURN {name:'labels', data:COLLECT(label)[..1000]} AS result\n"
        "UNION ALL\n"
        "CALL db.relationshipTypes() YIELD relationshipType\n"
        "RETURN {name:'relationshipTypes', data:COLLECT(relationshipType)[..1000]} AS result\n"
        "UNION ALL\n"
        "CALL db.propertyKeys() YIELD propertyKey\n"
        "RETURN {name:'propertyKeys', data:COLLECT(propertyKey)[..1000]} AS result"
    ).to_dicts()
    by_name = {row["result"]["name"]: row["result"]["data"] for row in rows}
    assert sorted(by_name["labels"]) == ["City", "Person"]
    assert by_name["relationshipTypes"] == ["LIVES_IN"]
    assert "age" in by_name["propertyKeys"]
    # The sidebar needs real arrays, not JSON strings.
    assert all(isinstance(v, list) for v in by_name.values())


def test_browser_meta_count_query(graph):
    rows = graph.cypher(
        "MATCH () RETURN { name:'nodes', data:count(*) } AS result\n"
        "UNION ALL\n"
        "MATCH ()-[]->() RETURN { name:'relationships', data: count(*)} AS result"
    ).to_dicts()
    assert rows == [
        {"result": {"name": "nodes", "data": 3}},
        {"result": {"name": "relationships", "data": 1}},
    ]


# ── Zero-length paths (fixed 2026-08-15; measured via G.V()'s Data Explorer) ─


def test_zero_length_path_assignment(graph):
    """`MATCH p = (n:Label) RETURN p` binds a one-node, zero-hop path —
    pre-fix p projected as NULL on every row (G.V()'s Data Explorer sends
    exactly this shape and showed "No results" against real matches)."""
    rows = graph.cypher("MATCH p = (s0 :Person)  RETURN p").to_dicts()
    assert len(rows) == 2
    assert all(r["p"] is not None for r in rows)
    assert all(len(r["p"]["nodes"]) == 1 and r["p"]["relationships"] == [] for r in rows)
    fns = graph.cypher("MATCH p = (n:Person {name: 'ann'}) RETURN length(p) AS l, size(nodes(p)) AS n").to_dicts()
    assert fns == [{"l": 0, "n": 1}]
