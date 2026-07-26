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


@pytest.fixture(params=(64, 8192), ids=("under-eager-threshold", "over-eager-threshold"))
def sized_graph(request):
    """A memory graph whose Person count straddles EAGER_MATERIALISE_MAX_ROWS.

    Small lazy-eligible results are materialised at construction and drop the
    graph reference; large ones stay deferred. The two paths must be
    indistinguishable through the public API — this fixture is what makes the
    boundary testable from Python at all, since the difference is otherwise
    only observable as a whole-graph copy.
    """
    count = request.param
    graph = kglite.KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": list(range(count)),
                "title": [f"P{i}" for i in range(count)],
                "age": [i % 90 for i in range(count)],
            }
        ),
        "Person",
        "id",
        "title",
        columns=["age"],
    )
    return graph, count


def test_streaming_matches_eager_across_the_threshold(sized_graph):
    """Deferred and up-front materialisation agree on both sides of the cutoff."""
    graph, count = sized_graph
    query = "MATCH (n:Person) RETURN n.title AS name, n.age AS age"
    streamed = graph.cypher(query, streaming=True)
    eager = graph.cypher(query, streaming=False)
    assert len(streamed) == count
    assert streamed.to_dicts() == eager.to_dicts()


def test_streaming_result_survives_an_intervening_write(sized_graph):
    """A held result keeps its pre-write rows whether or not it was deferred.

    Below the threshold the rows are already materialised; above it the view
    still holds the graph it was built from and the write copies-on-write.
    Either way the caller sees the data as of query time — the guarantee that
    lets the eager path drop the graph reference without changing semantics.
    """
    graph, count = sized_graph
    query = "MATCH (n:Person) RETURN n.title AS name, n.age AS age"
    held = graph.cypher(query, streaming=True)
    graph.cypher("MATCH (n:Person) SET n.age = 999")
    rows = held.to_dicts()
    assert len(rows) == count
    assert all(row["age"] != 999 for row in rows)
    assert graph.cypher("MATCH (n:Person) RETURN DISTINCT n.age AS age").to_dicts() == [{"age": 999}]
