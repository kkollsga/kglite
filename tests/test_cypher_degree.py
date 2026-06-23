"""Regression tests for the degree() / inDegree() / outDegree() Cypher functions.

These count a node's edges: degree = both directions (a self-loop counts
twice, the standard graph-theory convention), inDegree = incoming,
outDegree = outgoing. The motivating query is hub-finding
(`WHERE degree(n) > N`) — previously impossible (no degree function, and
`size((n)--())` is a parser error).
"""

import pandas as pd
import pytest

from kglite import KnowledgeGraph


@pytest.fixture
def deg_graph():
    """A small directed graph with a self-loop and an isolated node.

    Edges (R): a->b, a->c, b->c, c->d, a->a (self-loop).
    Degrees (in/out/total):
      a: out {b,c,a}=3, in {a}=1, total 4
      b: out {c}=1,     in {a}=1, total 2
      c: out {d}=1,     in {a,b}=2, total 3
      d: out {}=0,      in {c}=1, total 1
      e: isolated -> 0/0/0
    """
    g = KnowledgeGraph()
    g.cypher(
        "CREATE (a:N {id:'a'}), (b:N {id:'b'}), (c:N {id:'c'}), "
        "       (d:N {id:'d'}), (e:N {id:'e'}) "
        "CREATE (a)-[:R]->(b), (a)-[:R]->(c), (b)-[:R]->(c), "
        "       (c)-[:R]->(d), (a)-[:R]->(a)"
    )
    return g


def _by_id(rows):
    return {r["id"]: r for r in rows}


def test_degree_in_out_counts(deg_graph):
    rows = deg_graph.cypher(
        "MATCH (n:N) RETURN n.id AS id, degree(n) AS deg, inDegree(n) AS ind, outDegree(n) AS outd ORDER BY id"
    ).to_dicts()
    got = _by_id(rows)
    assert got["a"] == {"id": "a", "deg": 4, "ind": 1, "outd": 3}  # self-loop counts twice
    assert got["b"] == {"id": "b", "deg": 2, "ind": 1, "outd": 1}
    assert got["c"] == {"id": "c", "deg": 3, "ind": 2, "outd": 1}
    assert got["d"] == {"id": "d", "deg": 1, "ind": 1, "outd": 0}


def test_isolated_node_is_zero(deg_graph):
    rows = deg_graph.cypher(
        "MATCH (n:N {id:'e'}) RETURN degree(n) AS d, inDegree(n) AS i, outDegree(n) AS o"
    ).to_dicts()
    assert rows[0] == {"d": 0, "i": 0, "o": 0}


def test_hub_filter_in_where(deg_graph):
    """The motivating use case: find hubs by degree threshold."""
    rows = deg_graph.cypher("MATCH (n:N) WHERE degree(n) >= 3 RETURN n.id AS id ORDER BY id").to_dicts()
    assert [r["id"] for r in rows] == ["a", "c"]


def test_degree_distribution(deg_graph):
    """degree() makes degree distribution a one-liner."""
    rows = deg_graph.cypher(
        "MATCH (n:N) WITH degree(n) AS d RETURN d AS degree, count(*) AS freq ORDER BY degree"
    ).to_dicts()
    assert rows == [
        {"degree": 0, "freq": 1},
        {"degree": 1, "freq": 1},
        {"degree": 2, "freq": 1},
        {"degree": 3, "freq": 1},
        {"degree": 4, "freq": 1},
    ]


def test_case_insensitive(deg_graph):
    rows = deg_graph.cypher(
        "MATCH (n:N {id:'c'}) RETURN DEGREE(n) AS d, INDEGREE(n) AS i, OUTDEGREE(n) AS o"
    ).to_dicts()
    assert rows[0] == {"d": 3, "i": 2, "o": 1}


def test_resolves_through_with_rename(deg_graph):
    """degree() resolves a node carried through WITH n AS x — consistent
    with id()/labels(), which already work on materialised node values."""
    rows = deg_graph.cypher("MATCH (n:N {id:'a'}) WITH n AS x RETURN degree(x) AS d, outDegree(x) AS o").to_dicts()
    assert rows[0] == {"d": 4, "o": 3}


def test_resolves_through_unwind_collect(deg_graph):
    rows = deg_graph.cypher(
        "MATCH (n:N) WITH collect(n) AS ns UNWIND ns AS x RETURN x.id AS id, degree(x) AS d ORDER BY id"
    ).to_dicts()
    assert _by_id(rows)["c"]["d"] == 3


def test_numeric_id_nodes():
    """Resolution works for nodes with numeric (UniqueId) ids too."""
    g = KnowledgeGraph()
    g.add_nodes(pd.DataFrame({"uid": [1, 2, 3], "name": ["a", "b", "c"]}), "P", "uid", "name")
    g.add_connections(pd.DataFrame({"s": [1, 1], "t": [2, 3]}), "R", "P", "s", "P", "t")
    direct = g.cypher("MATCH (n:P {uid:1}) RETURN outDegree(n) AS o").to_dicts()
    renamed = g.cypher("MATCH (n:P {uid:1}) WITH n AS x RETURN outDegree(x) AS o").to_dicts()
    assert direct[0]["o"] == 2
    assert renamed[0]["o"] == 2  # Value::Node fallback resolves the UniqueId


def test_null_on_non_node(deg_graph):
    """A non-resolvable argument yields Null, not an error."""
    rows = deg_graph.cypher("RETURN degree(42) AS d, degree('x') AS d2").to_dicts()
    assert rows[0] == {"d": None, "d2": None}
