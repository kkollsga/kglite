"""Empty global aggregates keep the values of constant wrapper children."""

import pytest

import kglite


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize(
    "expression,expected",
    [
        ("sum(x)+1", 1),
        ("1+sum(x)", 1),
        ("count(*)-$delta", -2),
        ("sum(x)*3", 0),
        ("sum(x)/2", 0.0),
        ("sum(x)+size([1,2])", 2),
        ("{total:sum(x)+1}", {"total": 1}),
        ("sum(x)+null", None),
    ],
)
def test_empty_aggregate_constant_children(expression, expected, streaming, disabled):
    graph = kglite.KnowledgeGraph()
    rows = graph.cypher(
        f"UNWIND [] AS x RETURN {expression} AS value",
        params={"delta": 2},
        streaming=streaming,
        disable_optimizer=disabled,
    ).to_list()
    assert rows == [{"value": expected}]
