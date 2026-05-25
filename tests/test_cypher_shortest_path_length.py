"""shortest_path_length(a, b) Cypher function — 2026-05-25 Batch 4.

Wraps `graph_algorithms::shortest_path_cost` so every binding can
ask "how many hops between A and B" via the universal cypher_query
interface, without materializing the full path.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite


@pytest.fixture
def chain_graph():
    """Linear chain A — B — C — D — E."""
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["A", "B", "C", "D", "E"],
        }
    )
    g.add_nodes(df, "Node", "id", "name")
    g.add_connections(
        pd.DataFrame(
            {
                "src": [1, 2, 3, 4],
                "dst": [2, 3, 4, 5],
            }
        ),
        "NEXT",
        "Node",
        "src",
        "Node",
        "dst",
    )
    return g


@pytest.fixture
def disconnected_graph():
    """Two components: A—B and C—D."""
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame({"id": [1, 2, 3, 4], "name": ["A", "B", "C", "D"]})
    g.add_nodes(df, "Node", "id", "name")
    g.add_connections(
        pd.DataFrame({"src": [1, 3], "dst": [2, 4]}),
        "NEXT",
        "Node",
        "src",
        "Node",
        "dst",
    )
    return g


def test_one_hop(chain_graph):
    rows = chain_graph.cypher("MATCH (a:Node {id: 1}), (b:Node {id: 2}) RETURN shortest_path_length(a, b) AS hops")
    assert rows[0]["hops"] == 1


def test_multi_hop(chain_graph):
    rows = chain_graph.cypher("MATCH (a:Node {id: 1}), (b:Node {id: 5}) RETURN shortest_path_length(a, b) AS hops")
    assert rows[0]["hops"] == 4


def test_self_loop_is_zero(chain_graph):
    rows = chain_graph.cypher("MATCH (a:Node {id: 3}) RETURN shortest_path_length(a, a) AS hops")
    assert rows[0]["hops"] == 0


def test_disconnected_returns_null(disconnected_graph):
    rows = disconnected_graph.cypher(
        "MATCH (a:Node {id: 1}), (b:Node {id: 3}) RETURN shortest_path_length(a, b) AS hops"
    )
    assert rows[0]["hops"] is None


def test_wrong_arg_count_errors():
    g = kglite.KnowledgeGraph()
    with pytest.raises(Exception, match="requires 2"):
        g.cypher("RETURN shortest_path_length(1) AS h")


def test_non_variable_args_error(chain_graph):
    # Passing literals instead of node variables
    with pytest.raises(Exception, match="bound node variables"):
        chain_graph.cypher("RETURN shortest_path_length(1, 2) AS h")


def test_returns_int_type(chain_graph):
    rows = chain_graph.cypher("MATCH (a:Node {id: 1}), (b:Node {id: 4}) RETURN shortest_path_length(a, b) AS hops")
    # Should be an integer, not a float
    assert isinstance(rows[0]["hops"], int)


def test_used_in_where_clause(chain_graph):
    """Real use case: filter pairs by hop distance."""
    rows = chain_graph.cypher(
        "MATCH (a:Node {id: 1}), (b:Node) WHERE shortest_path_length(a, b) <= 2 RETURN b.name AS name ORDER BY b.id"
    )
    # From A: A(0), B(1), C(2) qualify. D=3, E=4 don't.
    assert [r["name"] for r in rows] == ["A", "B", "C"]
