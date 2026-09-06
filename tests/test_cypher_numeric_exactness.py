"""Exact mixed numeric predicates and integer-cast boundaries."""

import math

import pytest

import kglite

BOUNDARY_ROWS = [
    {"id": 0, "v": 2**53},
    {"id": 1, "v": 2**53 + 1},
    {"id": 2, "v": float(2**53)},
    {"id": 3, "v": -(2**53)},
    {"id": 4, "v": -(2**53) - 1},
    {"id": 5, "v": float(-(2**53))},
]


def boundary_graph(index_kind):
    graph = kglite.KnowledgeGraph()
    graph.cypher("UNWIND $rows AS r CREATE(:N {id:r.id,v:r.v})", params={"rows": BOUNDARY_ROWS})
    if index_kind == "equality":
        graph.create_index("N", "v")
    elif index_kind == "range":
        graph.create_range_index("N", "v")
    return graph


def ids(graph, predicate, value, disabled):
    rows = graph.cypher(
        f"MATCH(n:N) WHERE {predicate} RETURN n.id AS id ORDER BY id",
        params={"v": value},
        disable_optimizer=disabled,
    ).to_list()
    return [row["id"] for row in rows]


@pytest.mark.parametrize("index_kind", ["scan", "equality", "range"])
@pytest.mark.parametrize("disabled", [False, True])
def test_mixed_numeric_predicates_are_exact_across_scan_index_and_optimizer(index_kind, disabled):
    graph = boundary_graph(index_kind)
    cases = [
        ("n.v = $v", float(2**53), [0, 2]),
        ("n.v <> $v", float(2**53), [1, 3, 4, 5]),
        ("n.v < $v", float(2**53), [3, 4, 5]),
        ("n.v <= $v", float(2**53), [0, 2, 3, 4, 5]),
        ("n.v > $v", float(2**53), [1]),
        ("n.v >= $v", float(2**53), [0, 1, 2]),
        ("n.v = $v", 2**53 + 1, [1]),
        ("n.v < $v", 2**53 + 1, [0, 2, 3, 4, 5]),
        ("n.v = $v", float(-(2**53)), [3, 5]),
        ("n.v = $v", -(2**53) - 1, [4]),
        ("n.v < $v", float(-(2**53)), [4]),
        ("n.v > $v", -(2**53) - 1, [0, 1, 2, 3, 5]),
    ]
    for predicate, value, expected in cases:
        assert ids(graph, predicate, value, disabled) == expected


@pytest.mark.parametrize("disabled", [False, True])
def test_pattern_property_equality_is_exact(disabled):
    graph = boundary_graph("equality")
    rows = graph.cypher(
        "MATCH(n:N {v:$v}) RETURN n.id AS id ORDER BY id",
        params={"v": float(2**53)},
        disable_optimizer=disabled,
    ).to_list()
    assert rows == [{"id": 0}, {"id": 2}]


@pytest.mark.parametrize("size", [8, 9])
@pytest.mark.parametrize("disabled", [False, True])
def test_short_and_indexed_in_membership_are_exact(size, disabled):
    graph = boundary_graph("equality")
    values = [float(2**53), *[f"filler-{i}" for i in range(size - 1)]]
    rows = graph.cypher(
        "MATCH(n:N) WHERE n.v IN $values RETURN n.id AS id ORDER BY id",
        params={"values": values},
        disable_optimizer=disabled,
    ).to_list()
    assert rows == [{"id": 0}, {"id": 2}]


@pytest.mark.parametrize("disabled", [False, True])
def test_to_integer_returns_null_for_nonfinite_and_out_of_range_float(disabled):
    graph = kglite.KnowledgeGraph()
    below_lower = math.nextafter(float(-(2**63)), -math.inf)
    values = [math.nan, math.inf, -math.inf, float(2**63), below_lower]
    for value in values:
        assert graph.cypher(
            "RETURN toInteger($value) AS value",
            params={"value": value},
            disable_optimizer=disabled,
        ).to_list() == [{"value": None}]


@pytest.mark.parametrize("disabled", [False, True])
def test_to_integer_accepts_the_finite_half_open_i64_float_interval(disabled):
    graph = kglite.KnowledgeGraph()
    cases = [
        (float(-(2**63)), -(2**63)),
        (math.nextafter(float(2**63), -math.inf), 9_223_372_036_854_774_784),
        (3.7, 3),
        (-3.7, -3),
    ]
    for value, expected in cases:
        assert graph.cypher(
            "RETURN toInteger($value) AS value",
            params={"value": value},
            disable_optimizer=disabled,
        ).to_list() == [{"value": expected}]
