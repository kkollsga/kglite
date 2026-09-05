"""Absolute SUM oracles: optimizer parity cannot see shared f64 rounding."""

import math

import pytest

import kglite

MAX = 2**63 - 1
MIN = -(2**63)


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("expression", ["sum(x)", "sum(DISTINCT x)", "sum(x)+0"])
@pytest.mark.parametrize(
    "values,expected",
    [
        ([2**53 + 1], 2**53 + 1),
        ([2**53 + 1, -(2**53)], 1),
        ([MIN + 1], MIN + 1),
        ([MAX, 1, -MAX], 1),
        ([], 0),
        ([None, "skip", 3], 3),
    ],
)
def test_sum_integer_exact(values, expected, expression, disabled):
    graph = kglite.KnowledgeGraph()
    actual = graph.cypher(
        f"UNWIND $values AS x RETURN {expression} AS s",
        params={"values": values},
        disable_optimizer=disabled,
    ).scalar()
    assert type(actual) is int
    assert actual == expected


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("values", [[MAX, 1], [MIN, -1]])
def test_sum_integer_overflow_is_reported(values, disabled):
    graph = kglite.KnowledgeGraph()
    with pytest.raises(kglite.CypherExecutionError, match="Integer overflow in sum"):
        graph.cypher("UNWIND $values AS x RETURN sum(x)", params={"values": values}, disable_optimizer=disabled)


@pytest.mark.parametrize("disabled", [False, True])
def test_sum_graph_routes_and_group_distinct(disabled):
    graph = kglite.KnowledgeGraph()
    graph.cypher(
        "UNWIND $rows AS r CREATE(:N{id:r.id,g:r.g,v:r.v})",
        params={
            "rows": [
                {"id": 0, "g": "A", "v": 2**53 + 1},
                {"id": 1, "g": "A", "v": -(2**53)},
                {"id": 2, "g": "B", "v": MIN + 1},
                {"id": 3, "g": "A", "v": 2**53 + 1},
            ]
        },
    )
    actual = graph.cypher(
        "MATCH(n:N) RETURN n.g AS g,sum(DISTINCT n.v) AS s,collect(n.id) AS ids ORDER BY g",
        disable_optimizer=disabled,
    ).to_list()
    assert actual == [{"g": "A", "s": 1, "ids": [0, 1, 3]}, {"g": "B", "s": MIN + 1, "ids": [2]}]
    assert all(type(row["s"]) is int for row in actual)
    assert (
        graph.cypher("MATCH(n:N) WHERE n.g='A' RETURN sum(n.v) AS s", disable_optimizer=disabled).scalar() == 2**53 + 2
    )


@pytest.mark.parametrize(
    "values,expected", [([1, 2.0], 3.0), ([MAX, 1, 0.0], float(2**63)), ([0.0, MAX, 1], float(2**63))]
)
def test_sum_mixed_float_policy_unchanged(values, expected):
    graph = kglite.KnowledgeGraph()
    for disabled in [False, True]:
        result = graph.cypher("UNWIND $v AS x RETURN sum(x)", params={"v": values}, disable_optimizer=disabled).scalar()
        assert type(result) is float
        assert result == expected


@pytest.mark.parametrize("expression", ["sum(r.v)", "sum(r.v)+0"])
def test_sum_materialized_grouped_integer_exact(expression):
    graph = kglite.KnowledgeGraph()
    rows = [{"g": "A", "v": value} for value in [MAX, 1, -MAX]]
    rows.insert(1, {"g": "B", "v": MIN + 1})
    actual = graph.cypher(
        f"UNWIND $rows AS r RETURN r.g AS g,{expression} AS s ORDER BY g",
        params={"rows": rows},
        streaming=False,
        disable_optimizer=True,
    ).to_list()
    assert actual == [{"g": "A", "s": 1}, {"g": "B", "s": MIN + 1}]
    assert all(type(row["s"]) is int for row in actual)


@pytest.mark.parametrize("expression", ["sum(r.v)", "sum(r.v)+0"])
def test_sum_materialized_grouped_integer_overflow_is_reported(expression):
    graph = kglite.KnowledgeGraph()
    with pytest.raises(kglite.CypherExecutionError, match="Integer overflow in sum"):
        graph.cypher(
            f"UNWIND $rows AS r RETURN r.g AS g,{expression} AS s ORDER BY g",
            params={"rows": [{"g": "A", "v": value} for value in [MAX, 1]]},
            streaming=False,
            disable_optimizer=True,
        ).to_list()


@pytest.mark.parametrize("projection", ["sum(r.v) AS s", "r.g AS g,sum(r.v)*1.0 AS s"])
def test_sum_materialized_preserves_negative_zero(projection):
    graph = kglite.KnowledgeGraph()
    actual = graph.cypher(
        f"UNWIND $rows AS r RETURN {projection}",
        params={"rows": [{"g": "A", "v": -0.0}]},
        streaming=False,
        disable_optimizer=True,
    ).to_list()
    assert len(actual) == 1
    assert type(actual[0]["s"]) is float
    assert actual[0]["s"] == 0.0
    assert math.copysign(1.0, actual[0]["s"]) == -1.0
