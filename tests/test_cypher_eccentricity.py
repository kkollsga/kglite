"""Regression tests for the eccentricity / diameter CALL procedures.

`eccentricity` yields (node, eccentricity) — a node's greatest shortest-path
distance to any node in its connected component. `diameter` yields the max
eccentricity as a single aggregate row. Both are all-pairs O(V*(V+E)) BFS,
capped at 20k scoped nodes (guarded), with optional {node_type, relationship}
scoping. Distances ignore unreachable nodes, so they're well-defined on
disconnected graphs (unlike NetworkX, which errors).
"""

import pytest

from kglite import KnowledgeGraph


def _path5():
    """Path a-b-c-d-e. Eccentricities: a/e=4, b/d=3, c=2; diameter=4."""
    g = KnowledgeGraph()
    g.cypher(
        "CREATE (a:N {id:'a'}), (b:N {id:'b'}), (c:N {id:'c'}), "
        "       (d:N {id:'d'}), (e:N {id:'e'}) "
        "CREATE (a)-[:R]->(b), (b)-[:R]->(c), (c)-[:R]->(d), (d)-[:R]->(e)"
    )
    return g


def test_eccentricity_path():
    g = _path5()
    rows = g.cypher(
        "CALL eccentricity() YIELD node, eccentricity RETURN node.id AS id, eccentricity ORDER BY id"
    ).to_dicts()
    assert rows == [
        {"id": "a", "eccentricity": 4},
        {"id": "b", "eccentricity": 3},
        {"id": "c", "eccentricity": 2},
        {"id": "d", "eccentricity": 3},
        {"id": "e", "eccentricity": 4},
    ]


def test_diameter_path():
    g = _path5()
    assert g.cypher("CALL diameter() YIELD diameter RETURN diameter").to_dicts() == [{"diameter": 4}]


def test_diameter_equals_max_eccentricity():
    g = _path5()
    dia = g.cypher("CALL diameter() YIELD diameter RETURN diameter").to_dicts()[0]["diameter"]
    eccs = [
        r["eccentricity"] for r in g.cypher("CALL eccentricity() YIELD eccentricity RETURN eccentricity").to_dicts()
    ]
    assert dia == max(eccs)


def test_disconnected_graph_is_well_defined():
    """Two separate edges: each node's eccentricity is 1 (over its own
    component); diameter is the max component diameter = 1."""
    g = KnowledgeGraph()
    g.cypher(
        "CREATE (a:N {id:'a'}), (b:N {id:'b'}), (c:N {id:'c'}), (d:N {id:'d'}) CREATE (a)-[:R]->(b), (c)-[:R]->(d)"
    )
    assert g.cypher("CALL diameter() YIELD diameter RETURN diameter").to_dicts() == [{"diameter": 1}]
    eccs = g.cypher("CALL eccentricity() YIELD eccentricity RETURN eccentricity").to_dicts()
    assert all(r["eccentricity"] == 1 for r in eccs)


def test_isolated_node_zero():
    g = KnowledgeGraph()
    g.cypher("CREATE (x:N {id:'x'})")
    assert g.cypher("CALL eccentricity() YIELD eccentricity RETURN eccentricity").to_dicts() == [{"eccentricity": 0}]


def test_triangle_all_one():
    g = KnowledgeGraph()
    g.cypher("CREATE (a:N {id:'a'}), (b:N {id:'b'}), (c:N {id:'c'}) CREATE (a)-[:R]->(b), (b)-[:R]->(c), (c)-[:R]->(a)")
    assert g.cypher("CALL diameter() YIELD diameter RETURN diameter").to_dicts() == [{"diameter": 1}]


def test_scoping():
    g = _path5()
    rows = g.cypher("CALL diameter({node_type:'N', relationship:'R'}) YIELD diameter RETURN diameter").to_dicts()
    assert rows == [{"diameter": 4}]


def test_node_cap_guard():
    """Above the 20k-node cap the procedure errors with guidance, rather
    than churning through an O(V*(V+E)) all-pairs BFS."""
    g = KnowledgeGraph()
    g.cypher("UNWIND range(1, 20001) AS i CREATE (:N {id: i})")
    with pytest.raises(Exception) as exc:
        g.cypher("CALL diameter() YIELD diameter RETURN diameter")
    msg = str(exc.value)
    assert "all-pairs" in msg and "cap" in msg


def test_listed_in_list_procedures():
    g = KnowledgeGraph()
    names = {r["name"] for r in g.cypher("CALL list_procedures() YIELD name RETURN name").to_dicts()}
    assert {"eccentricity", "diameter"} <= names
