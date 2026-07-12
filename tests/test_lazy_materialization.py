"""Checked lazy-result materialization across storage backends."""

from __future__ import annotations

import pandas as pd
import pytest

import kglite


@pytest.fixture(params=("memory", "mapped", "disk"))
def graph(request, tmp_path):
    mode = request.param
    if mode == "memory":
        graph = kglite.KnowledgeGraph()
    elif mode == "mapped":
        graph = kglite.KnowledgeGraph(storage="mapped")
    else:
        graph = kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "disk-graph"))
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": [1, 2],
                "title": ["Alice", "Bob"],
                "age": [30, 25],
            }
        ),
        "Person",
        "id",
        "title",
        columns=["age"],
    )
    return graph


def lazy(graph):
    return graph.cypher("MATCH (n:Person) RETURN n.title AS name, n.age AS age", streaming=True)


def test_lazy_matches_eager_after_intervening_query(graph):
    result = lazy(graph)
    graph.cypher("MATCH (n:Person) RETURN count(n) AS count").scalar()
    eager = graph.cypher("MATCH (n:Person) RETURN n.title AS name, n.age AS age", streaming=False).to_dicts()
    assert result.to_dicts() == eager


def test_lazy_resultview_access_forms(graph):
    assert lazy(graph)[0] == {"name": "Alice", "age": 30}
    assert lazy(graph).scalar() == "Alice"
    assert lazy(graph).column("age") == [30, 25]
    assert lazy(graph).head(1).to_dicts() == [{"name": "Alice", "age": 30}]
    assert lazy(graph).tail(1).to_dicts() == [{"name": "Bob", "age": 25}]
    assert lazy(graph)[1:].to_dicts() == [{"name": "Bob", "age": 25}]
    assert lazy(graph).to_df().to_dict("records") == [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
    ]
    assert "Alice" in repr(lazy(graph))
