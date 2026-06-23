"""Regression tests for the triangle_count / transitivity CALL procedure.

`triangle_count` returns the global number of triangles (3-cliques) plus the
transitivity (global clustering coefficient = 3*triangles / connected_triples)
as a single aggregate row. `transitivity` is an alias yielding the same
columns. Optional {node_type, relationship} scoping, like clustering_coefficient.
"""

from kglite import KnowledgeGraph


def _two_triangles():
    """Two triangles sharing edge b-c, plus a pendant e off d.

    Triangles: (a,b,c) and (b,c,d) -> 2.
    Undirected degrees: a=2, b=3, c=3, d=3, e=1.
    connected triples = C(2,2)+C(3,2)*3+C(1,2) = 1+3+3+3+0 = 10.
    link_sum (= 3*triangles) = 6 -> transitivity = 6/10 = 0.6.
    """
    g = KnowledgeGraph()
    g.cypher(
        "CREATE (a:N {id:'a'}), (b:N {id:'b'}), (c:N {id:'c'}), "
        "       (d:N {id:'d'}), (e:N {id:'e'}) "
        "CREATE (a)-[:R]->(b), (a)-[:R]->(c), (b)-[:R]->(c), "
        "       (b)-[:R]->(d), (c)-[:R]->(d), (d)-[:R]->(e)"
    )
    return g


def test_triangle_count_and_transitivity():
    g = _two_triangles()
    rows = g.cypher("CALL triangle_count() YIELD triangles, transitivity RETURN triangles, transitivity").to_dicts()
    assert rows == [{"triangles": 2, "transitivity": 0.6}]


def test_matches_cypher_pattern_join():
    """Cross-check the native count against the (slow) Cypher pattern-join."""
    g = _two_triangles()
    native = g.cypher("CALL triangle_count() YIELD triangles RETURN triangles").to_dicts()[0]
    by_pattern = g.cypher(
        "MATCH (a:N)-[]-(b:N)-[]-(c:N)-[]-(a:N) WHERE a.id < b.id AND b.id < c.id RETURN count(*) AS tri"
    ).to_dicts()[0]
    assert native["triangles"] == by_pattern["tri"] == 2


def test_transitivity_alias():
    g = _two_triangles()
    rows = g.cypher("CALL transitivity() YIELD transitivity RETURN transitivity").to_dicts()
    assert rows == [{"transitivity": 0.6}]


def test_scoping_by_type_and_relationship():
    g = _two_triangles()
    rows = g.cypher(
        "CALL triangle_count({node_type:'N', relationship:'R'}) "
        "YIELD triangles, transitivity RETURN triangles, transitivity"
    ).to_dicts()
    assert rows == [{"triangles": 2, "transitivity": 0.6}]


def test_triangle_free_graph():
    g = KnowledgeGraph()
    g.cypher("CREATE (x:N {id:'x'})-[:R]->(y:N {id:'y'})")
    rows = g.cypher("CALL triangle_count() YIELD triangles, transitivity RETURN triangles, transitivity").to_dicts()
    assert rows == [{"triangles": 0, "transitivity": 0.0}]


def test_single_complete_triangle():
    g = KnowledgeGraph()
    g.cypher("CREATE (a:N {id:'a'}), (b:N {id:'b'}), (c:N {id:'c'}) CREATE (a)-[:R]->(b), (b)-[:R]->(c), (c)-[:R]->(a)")
    # One triangle; every node has degree 2 -> 3 connected triples ->
    # transitivity = 3*1 / 3 = 1.0 (a perfect cluster).
    rows = g.cypher("CALL triangle_count() YIELD triangles, transitivity RETURN triangles, transitivity").to_dicts()
    assert rows == [{"triangles": 1, "transitivity": 1.0}]


def test_listed_in_list_procedures():
    g = KnowledgeGraph()
    names = {r["name"] for r in g.cypher("CALL list_procedures() YIELD name RETURN name").to_dicts()}
    assert "triangle_count" in names
