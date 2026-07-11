"""Cross-storage golden semantics for shared scalar execution paths."""

import pytest

import kglite

pytestmark = pytest.mark.parity


@pytest.fixture(params=["memory", "mapped", "disk"])
def scalar_graph(request, tmp_path):
    mode = request.param
    if mode == "memory":
        return kglite.KnowledgeGraph()
    if mode == "mapped":
        return kglite.KnowledgeGraph(storage="mapped")
    return kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "scalar-disk"))


def test_range_temporal_duration_and_regex_golden(scalar_graph):
    query = (
        "WITH duration({months: 2, days: 3}) * 2 AS d "
        "RETURN range(-2, 2) AS r, "
        "add_years(date('2024-02-29'), 1) AS shifted, "
        "d.months AS months, d.days AS days, "
        "'Alpha42' =~ '^Alpha[0-9]+$' AS regex_op, "
        "text_match_regex('Alpha42', '^Alpha[0-9]+$') AS regex_fn"
    )
    expected = {
        "r": [-2, -1, 0, 1, 2],
        "shifted": "2025-02-28",
        "months": 4,
        "days": 6,
        "regex_op": True,
        "regex_fn": True,
    }
    assert scalar_graph.cypher(query).to_list() == [expected]
    assert scalar_graph.cypher(query, disable_optimizer=True).to_list() == [expected]
