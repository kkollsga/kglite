"""Independent regression cases for three-valued and list expressions."""

from __future__ import annotations

import pytest

import kglite


@pytest.fixture
def graph():
    return kglite.KnowledgeGraph()


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("null IN [1, null]", None),
        ("1 IN [1, null]", True),
        ("2 IN [1, null]", None),
        ("NOT (null IN [1])", None),
        ("true AND null", None),
        ("false AND null", False),
        ("true OR null", True),
        ("false OR null", None),
        ("true XOR null", None),
        ("NOT null", None),
        ("true OR false AND false", True),
    ],
)
def test_boolean_expressions_preserve_unknown_and_precedence(graph, expression, expected):
    assert graph.cypher(f"RETURN {expression} AS value").to_list() == [{"value": expected}]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("any(x IN [null, false] WHERE x)", None),
        ("any(x IN [null, true] WHERE x)", True),
        ("all(x IN [true, null] WHERE x)", None),
        ("all(x IN [false, null] WHERE x)", False),
        ("none(x IN [false, null] WHERE x)", None),
        ("none(x IN [true, null] WHERE x)", False),
        ("single(x IN [true, null] WHERE x)", None),
        ("single(x IN [true, true, null] WHERE x)", False),
        ("single(x IN [true, false] WHERE x)", True),
    ],
)
def test_list_quantifiers_preserve_unknown(graph, expression, expected):
    assert graph.cypher(f"RETURN {expression} AS value").to_list() == [{"value": expected}]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("[1, 2] + [3, 4]", [1, 2, 3, 4]),
        ("0 + [1, 2]", [0, 1, 2]),
        ("[1, 2] + 3", [1, 2, 3]),
        ("[1] + null", [1, None]),
    ],
)
def test_plus_composes_lists_and_elements(graph, expression, expected):
    assert graph.cypher(f"RETURN {expression} AS value").to_list() == [{"value": expected}]


@pytest.mark.parametrize(
    ("expression", "message"),
    (("[0]['x']", "String index requires"), ("[0][1.2]", "index.*integer"), ("[0][true]", "index.*integer")),
)
def test_list_index_rejects_non_integer_types(graph, expression, message):
    with pytest.raises(kglite.CypherExecutionError, match=message):
        graph.cypher(f"RETURN {expression} AS value").to_list()
