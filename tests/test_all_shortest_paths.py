"""allShortestPaths(...) — enumerate every minimal-length path.

Complements shortestPath() (one path). 0.12 Tier 2.
"""

import kglite


def _diamond():
    # Two equal shortest routes 1->2->4 and 1->3->4 (length 2), plus a
    # longer detour 1->5->6->4 (length 3) that must be excluded.
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (a:N {id:1})-[:R]->(b:N {id:2})-[:R]->(d:N {id:4})")
    g.cypher("MATCH (a:N {id:1}),(d:N {id:4}) CREATE (a)-[:R]->(c:N {id:3})-[:R]->(d)")
    g.cypher("MATCH (a:N {id:1}),(d:N {id:4}) CREATE (a)-[:R]->(e:N {id:5})")
    g.cypher("MATCH (e:N {id:5}),(d:N {id:4}) CREATE (e)-[:R]->(f:N {id:6})-[:R]->(d)")
    return g


def test_all_shortest_paths_enumerates_both_routes():
    g = _diamond()
    rows = g.cypher(
        "MATCH p = allShortestPaths((a:N {id:1})-[:R*..5]->(d:N {id:4})) "
        "RETURN [n IN nodes(p) | n.id] AS ids, length(p) AS len"
    ).to_list()
    routes = sorted(r["ids"] for r in rows)
    assert routes == [[1, 2, 4], [1, 3, 4]]
    assert all(r["len"] == 2 for r in rows)  # the length-3 detour is excluded


def test_shortest_path_returns_single():
    g = _diamond()
    rows = g.cypher("MATCH p = shortestPath((a:N {id:1})-[:R*..5]->(d:N {id:4})) RETURN length(p) AS len").to_list()
    assert len(rows) == 1
    assert rows[0]["len"] == 2


def test_all_shortest_paths_undirected():
    g = _diamond()
    # Undirected traversal still finds the two minimal routes.
    rows = g.cypher(
        "MATCH p = allShortestPaths((a:N {id:1})-[:R*..5]-(d:N {id:4})) RETURN [n IN nodes(p) | n.id] AS ids"
    ).to_list()
    assert sorted(r["ids"] for r in rows) == [[1, 2, 4], [1, 3, 4]]


def test_all_shortest_paths_no_path():
    g = _diamond()
    g.cypher("CREATE (:N {id:99})")  # isolated
    rows = g.cypher(
        "MATCH p = allShortestPaths((a:N {id:1})-[:R*..5]->(z:N {id:99})) RETURN length(p) AS len"
    ).to_list()
    assert rows == []
