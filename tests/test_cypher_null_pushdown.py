"""Golden results for NULL handling in predicates pushed into MATCH patterns.

A pushed node matcher is the only enforcement of its conjunct once another
conjunct keeps the residual WHERE alive, and a fused scan drops a fully
subsumed WHERE altogether. So every rule here must answer exactly what the
unoptimised WHERE answers, in every storage mode.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite

STORAGE_MODES = ("memory", "mapped", "disk")


def _new_graph(mode: str, tmp_path) -> kglite.KnowledgeGraph:
    if mode == "memory":
        return kglite.KnowledgeGraph()
    if mode == "mapped":
        return kglite.KnowledgeGraph(storage="mapped")
    return kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "g"))


@pytest.fixture(params=STORAGE_MODES, ids=STORAGE_MODES)
def people(request, tmp_path):
    g = _new_graph(request.param, tmp_path)
    df = pd.DataFrame(
        {
            "pid": [1, 2, 3, 4],
            "name": ["a", "b", "c", "d"],
            "age": [20, 30, 40, 50],
        }
    )
    g.add_nodes(df, "P", "pid", "name")
    return g


# ── A NULL comparison value is never true ────────────────────────────────────
#
# `n.age > null` is null for every row. The unpushable regex conjunct keeps a
# residual WHERE, so the pushed matcher alone decides the comparison.

NULL_COMPARISONS = [
    ("MATCH (n:P) WHERE n.age > $x AND toString(n.id) =~ '.*' RETURN count(*) AS c", {"x": None}),
    ("MATCH (n:P) WHERE n.age >= null AND toString(n.id) =~ '.*' RETURN count(*) AS c", None),
    ("MATCH (n:P) WHERE $x < n.age AND toString(n.id) =~ '.*' RETURN count(*) AS c", {"x": None}),
    ("MATCH (n:P) WHERE n.age <= $x AND toString(n.id) =~ '.*' RETURN count(*) AS c", {"x": None}),
    ("MATCH (n:P) WHERE n.age > $x RETURN count(*) AS c", {"x": None}),
    ("MATCH (n:P) WHERE n.age > 10 AND n.age < $x RETURN count(*) AS c", {"x": None}),
]


@pytest.mark.parametrize("query,params", NULL_COMPARISONS)
def test_null_comparison_value_matches_nothing(people, query, params):
    assert people.cypher(query, params=params).to_list() == [{"c": 0}]


def test_null_comparison_grouped_aggregate_is_empty(people):
    rows = people.cypher("MATCH (n:P) WHERE n.age > null RETURN n.name AS name, count(n) AS k").to_list()
    assert rows == []
