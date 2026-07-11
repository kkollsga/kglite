"""Relationship identity and trail semantics for Cypher path bindings.

These cases are independently authored from the public language contract.  They
lock the two invariants path consumers rely on: each hop retains the exact
relationship matched, and one relationship cannot occur twice in one path.
"""

from kglite import KnowledgeGraph


def _parallel_path_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.cypher("CREATE (:N {id: 1}), (:N {id: 2}), (:N {id: 3})")
    for tag in ("r1", "r2"):
        graph.cypher(f"MATCH (a:N {{id: 1}}), (b:N {{id: 2}}) CREATE (a)-[:R {{tag: '{tag}'}}]->(b)")
    for tag in ("s1", "s2"):
        graph.cypher(f"MATCH (b:N {{id: 2}}), (c:N {{id: 3}}) CREATE (b)-[:S {{tag: '{tag}'}}]->(c)")
    return graph


def test_fixed_path_retains_each_parallel_relationship():
    graph = _parallel_path_graph()
    rows = graph.cypher(
        "MATCH p=(a:N {id: 1})-[:R]->(b:N)-[:S]->(c:N {id: 3}) RETURN [r IN relationships(p) | r.tag] AS tags"
    ).to_list()

    assert sorted(tuple(row["tags"]) for row in rows) == [
        ("r1", "s1"),
        ("r1", "s2"),
        ("r2", "s1"),
        ("r2", "s2"),
    ]


def test_fixed_incoming_path_retains_relationship_identity_and_orientation():
    graph = _parallel_path_graph()
    rows = graph.cypher(
        "MATCH p=(b:N {id: 2})<-[r:R]-(a:N {id: 1}) "
        "RETURN id(r) AS bound_id, id(head(relationships(p))) AS path_id, "
        "relationships(p) AS relationships "
        "ORDER BY bound_id"
    ).to_list()

    assert len(rows) == 2
    assert all(row["path_id"] == row["bound_id"] for row in rows)
    assert all((row["relationships"][0]["start"], row["relationships"][0]["end"]) == (0, 1) for row in rows)


def test_fixed_path_cannot_reuse_one_relationship():
    graph = KnowledgeGraph()
    graph.cypher("CREATE (a:N {id: 1})")
    graph.cypher("MATCH (a:N {id: 1}) CREATE (a)-[:LOOP]->(a)")

    query = "MATCH p=(a:N {id: 1})-[:LOOP]->(b)-[:LOOP]->(c) RETURN count(*) AS paths"
    rows = graph.cypher(query).to_list()
    unfused = graph.cypher(query, disabled_passes=["fuse_match_return_aggregate"]).to_list()
    assert rows == [{"paths": 0}]
    assert unfused == rows


def test_variable_path_enumerates_parallel_relationships_exactly():
    graph = _parallel_path_graph()
    rows = graph.cypher(
        "MATCH p=(a:N {id: 1})-[:R*1..1]->(b:N {id: 2}) "
        "RETURN id(head(relationships(p))) AS rel_id, "
        "head(relationships(p)).tag AS tag ORDER BY tag"
    ).to_list()

    assert [row["tag"] for row in rows] == ["r1", "r2"]
    assert len({row["rel_id"] for row in rows}) == 2


def test_variable_path_cannot_reuse_relationship_but_may_repeat_nodes():
    graph = KnowledgeGraph()
    graph.cypher("CREATE (:N {id: 1}), (:N {id: 2})")
    graph.cypher("MATCH (a:N {id: 1}) CREATE (a)-[:LOOP]->(a)")
    graph.cypher("MATCH (a:N {id: 1}), (b:N {id: 2}) CREATE (a)-[:GO]->(b), (b)-[:BACK]->(a)")

    reused = graph.cypher("MATCH p=(a:N {id: 1})-[:LOOP*2..2]->(b) RETURN count(*) AS paths").to_list()
    repeated_node = graph.cypher(
        "MATCH p=(a:N {id: 1})-[:GO|BACK*2..2]->(b:N {id: 1}) RETURN [n IN nodes(p) | n.id] AS ids"
    ).to_list()

    assert reused == [{"paths": 0}]
    assert repeated_node == [{"ids": [1, 2, 1]}]


def test_undirected_variable_path_cannot_reverse_over_same_relationship():
    graph = KnowledgeGraph()
    graph.cypher("CREATE (:N {id: 1}), (:N {id: 2})")
    graph.cypher("MATCH (a:N {id: 1}), (b:N {id: 2}) CREATE (a)-[:R]->(b)")

    rows = graph.cypher("MATCH p=(a:N {id: 1})-[:R*2..2]-(b:N {id: 1}) RETURN count(*) AS paths").to_list()
    assert rows == [{"paths": 0}]


def test_all_shortest_paths_distinguishes_parallel_relationships():
    graph = _parallel_path_graph()
    rows = graph.cypher(
        "MATCH p=allShortestPaths((a:N {id: 1})-[:R*]->(b:N {id: 2})) "
        "RETURN head(relationships(p)).tag AS tag ORDER BY tag"
    ).to_list()

    assert rows == [{"tag": "r1"}, {"tag": "r2"}]


def test_shortest_path_preserves_multi_type_filter():
    graph = KnowledgeGraph()
    graph.cypher("CREATE (:N {id: 1}), (:N {id: 2}), (:N {id: 3})")
    graph.cypher(
        "MATCH (a:N {id: 1}), (b:N {id: 2}), (c:N {id: 3}) CREATE (a)-[:A {tag: 'a'}]->(b), (b)-[:B {tag: 'b'}]->(c)"
    )

    rows = graph.cypher(
        "MATCH p=shortestPath((a:N {id: 1})-[:A|B*]->(c:N {id: 3})) RETURN [r IN relationships(p) | r.tag] AS tags"
    ).to_list()

    assert rows == [{"tags": ["a", "b"]}]
