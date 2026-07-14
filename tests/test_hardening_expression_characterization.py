"""Golden contracts for the expression dispatcher before decomposition.

Each row names one ``Expression`` AST variant. These are expected-value
oracles, not optimized-vs-naive comparisons: both execution paths share the
same dispatcher, so a differential test alone cannot catch a uniformly
misrouted arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

import kglite


@dataclass(frozen=True)
class ExpressionCase:
    variant: str
    query: str
    expected: list[dict[str, Any]]
    params: dict[str, Any] = field(default_factory=dict)


EXPRESSION_CASES = [
    ExpressionCase("PropertyAccess", "MATCH (n:Thing {id: 1}) RETURN n.name AS value", [{"value": "Ada"}]),
    ExpressionCase("Variable", "WITH 7 AS x RETURN x AS value", [{"value": 7}]),
    ExpressionCase("Literal", "RETURN 42 AS value", [{"value": 42}]),
    ExpressionCase("FunctionCall", "RETURN toUpper('ab') AS value", [{"value": "AB"}]),
    ExpressionCase("Add", "RETURN 7 + 5 AS value", [{"value": 12}]),
    ExpressionCase("Subtract", "RETURN 7 - 5 AS value", [{"value": 2}]),
    ExpressionCase("Multiply", "RETURN 7 * 5 AS value", [{"value": 35}]),
    ExpressionCase("Divide", "RETURN 7 / 2 AS value", [{"value": 3}]),
    ExpressionCase("Modulo", "RETURN 7 % 5 AS value", [{"value": 2}]),
    ExpressionCase("Concat", "RETURN 'ab' || 'cd' AS value", [{"value": "abcd"}]),
    ExpressionCase("Negate", "RETURN -7 AS value", [{"value": -7}]),
    ExpressionCase("Star", "MATCH (n:Thing) RETURN count(*) AS value", [{"value": 2}]),
    ExpressionCase("ListLiteral", "RETURN [1, null, 'x'] AS value", [{"value": [1, None, "x"]}]),
    ExpressionCase(
        "Case",
        "RETURN CASE 2 WHEN 1 THEN 'a' WHEN 2 THEN 'b' ELSE 'c' END AS value",
        [{"value": "b"}],
    ),
    ExpressionCase("Parameter", "RETURN $p AS value", [{"value": 9}], {"p": 9}),
    ExpressionCase(
        "ListComprehension",
        "RETURN [x IN [1, 2, 3] WHERE x > 1 | x * 10] AS value",
        [{"value": [20, 30]}],
    ),
    ExpressionCase("IndexAccess", "RETURN [10, 20, 30][-1] AS value", [{"value": 30}]),
    ExpressionCase("ListSlice", "RETURN [10, 20, 30, 40][1..3] AS value", [{"value": [20, 30]}]),
    ExpressionCase(
        "MapProjection",
        "MATCH (n:Thing {id: 1}) RETURN n {.name, doubled: n.age * 2} AS value",
        [{"value": {"name": "Ada", "doubled": 14}}],
    ),
    ExpressionCase("IsNull", "RETURN null IS NULL AS value", [{"value": True}]),
    ExpressionCase("IsNotNull", "RETURN 1 IS NOT NULL AS value", [{"value": True}]),
    ExpressionCase("MapLiteral", "RETURN {a: 1, b: [2]} AS value", [{"value": {"a": 1, "b": [2]}}]),
    ExpressionCase("QuantifiedList", "RETURN all(x IN [1, 2, 3] WHERE x > 0) AS value", [{"value": True}]),
    ExpressionCase("Reduce", "RETURN reduce(s = 0, x IN [1, 2, 3] | s + x) AS value", [{"value": 6}]),
    ExpressionCase("PredicateExpr", "RETURN 2 > 1 AS value", [{"value": True}]),
    ExpressionCase("ExprPropertyAccess", "RETURN date('2024-03-15').year AS value", [{"value": 2024}]),
    ExpressionCase(
        "WindowFunction",
        "MATCH (n:Thing) RETURN row_number() OVER (ORDER BY n.id) AS value ORDER BY value",
        [{"value": 1}, {"value": 2}],
    ),
    ExpressionCase("CountSubquery", "RETURN COUNT { (:Thing) } AS value", [{"value": 2}]),
]


@pytest.fixture
def expression_graph() -> kglite.KnowledgeGraph:
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:Thing {id: 1, name: 'Ada', age: 7}), (:Thing {id: 2, name: 'Bob', age: 9})")
    return graph


@pytest.mark.parametrize("case", EXPRESSION_CASES, ids=lambda case: case.variant)
def test_every_expression_variant_has_a_golden_result(
    expression_graph: kglite.KnowledgeGraph, case: ExpressionCase
) -> None:
    assert expression_graph.cypher(case.query, params=case.params).to_list() == case.expected


@pytest.mark.parametrize(
    ("query", "exact_message"),
    [
        ("RETURN $missing AS value", "Cypher execution error: Missing parameter: $missing"),
        (
            "RETURN [0]['x'] AS value",
            "Cypher execution error: String index requires a map, node, or relationship; got List([Int64(0)])",
        ),
        (
            "RETURN [0][1.5] AS value",
            "Cypher execution error: List index must be an integer, got Float64(1.5)",
        ),
        (
            "RETURN [1, 2]['x'..] AS value",
            'Cypher execution error: Slice start must be integer, got String("x")',
        ),
    ],
)
def test_expression_errors_are_exact(expression_graph: kglite.KnowledgeGraph, query: str, exact_message: str) -> None:
    with pytest.raises(kglite.CypherExecutionError) as caught:
        expression_graph.cypher(query).to_list()
    assert str(caught.value) == exact_message


def test_binary_operands_fail_left_to_right(expression_graph: kglite.KnowledgeGraph) -> None:
    with pytest.raises(kglite.CypherExecutionError) as left_first:
        expression_graph.cypher("RETURN $missing + [0]['x'] AS value").to_list()
    assert str(left_first.value) == "Cypher execution error: Missing parameter: $missing"

    with pytest.raises(kglite.CypherExecutionError) as right_second:
        expression_graph.cypher("RETURN [0]['x'] + $missing AS value").to_list()
    assert str(right_second.value).startswith("Cypher execution error: String index requires")


def test_count_subquery_currently_bypasses_the_shared_budget(
    expression_graph: kglite.KnowledgeGraph,
) -> None:
    """Characterize the defect fixed in Phase 14.

    The nested scan produces two matches despite max_rows=1. Phase 14 changes
    this assertion to expect a max_rows error.
    """

    assert expression_graph.cypher("RETURN COUNT { (n:Thing) } AS value", max_rows=1).to_list() == [{"value": 2}]


def test_count_subquery_currently_swallows_where_errors(
    expression_graph: kglite.KnowledgeGraph,
) -> None:
    """Characterize the defect fixed in Phase 14."""

    assert expression_graph.cypher("RETURN COUNT { (n:Thing) WHERE n.age > $missing } AS value").to_list() == [
        {"value": 0}
    ]
