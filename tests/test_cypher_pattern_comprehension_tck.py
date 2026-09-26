"""Pattern comprehensions against the openCypher TCK.

Scenarios from `tck/features/expressions/pattern/Pattern2.feature`, and the
null-correlation rule from `tck/features/clauses/match/Match3.feature`
([27], [28]) and `Match7.feature` ([10]): a pattern anchored on a null node
finds no matches. The TCK has no scenario for a null-correlated pattern
comprehension, `COUNT { }` or `EXISTS { }`; each follows that rule, so the
comprehension is `[]`, the count `0` and the existence `false`.

Where a scenario reads `name`, these read `v`: KGLite resolves `name` to a
node's title, which every node has.
"""

from collections import Counter

import kglite


def graph(create: str) -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    for statement in create.split(";"):
        g.cypher(statement)
    return g


def path_shape(path: dict) -> tuple:
    """`<(:A)-[:T]->(:B)>` as (('A',), 'T', ('B',))."""
    nodes = [tuple(node["labels"]) for node in path["nodes"]]
    out = [nodes[0]]
    for rel, node in zip(path["relationships"], nodes[1:]):
        out += [rel["type"], node]
    return tuple(out)


def column(g, query, name):
    return [row[name] for row in g.cypher(query).to_list()]


def test_pattern2_1_return_a_pattern_comprehension():
    g = graph("CREATE (a:A), (b:B) CREATE (a)-[:T]->(b), (b)-[:T]->(:C)")
    lists = column(g, "MATCH (n) RETURN [p = (n)-->() | p] AS list", "list")
    assert Counter(tuple(path_shape(p) for p in ps) for ps in lists) == Counter(
        [((("A",), "T", ("B",)),), ((("B",), "T", ("C",)),), ()]
    )


def test_pattern2_2_with_label_predicate():
    g = graph("CREATE (a:A), (b:B), (c:C), (d:D) CREATE (a)-[:T]->(b), (a)-[:T]->(c), (a)-[:T]->(d)")
    lists = column(g, "MATCH (n:A) RETURN [p = (n)-->(:B) | p] AS list", "list")
    assert [[path_shape(p) for p in ps] for ps in lists] == [[(("A",), "T", ("B",))]]


def test_pattern2_3_with_bound_nodes():
    g = graph("CREATE (a:A), (b:B) CREATE (a)-[:T]->(b)")
    lists = column(g, "MATCH (a:A), (b:B) RETURN [p = (a)-->(b) | p] AS list", "list")
    assert [[path_shape(p) for p in ps] for ps in lists] == [[(("A",), "T", ("B",))]]


def test_pattern2_4_and_5_introduce_new_variables():
    g = graph("CREATE (a:N), (b:N {v: 'val'}), (c:N) CREATE (a)-[:T]->(b), (b)-[:T]->(c)")
    lists = column(g, "MATCH (n) RETURN [(n)-[:T]->(b) | b.v] AS list", "list")
    assert Counter(tuple(x) for x in lists) == Counter([("val",), (None,), ()])
    g = graph("CREATE (a:N), (b:N), (c:N) CREATE (a)-[:T {v: 'val'}]->(b), (b)-[:T]->(c)")
    lists = column(g, "MATCH (n) RETURN [(n)-[r:T]->() | r.v] AS list", "list")
    assert Counter(tuple(x) for x in lists) == Counter([("val",), (None,), ()])


def test_pattern2_6_aggregate_on_a_pattern_comprehension():
    g = graph("CREATE (a:A), (:A), (:A) CREATE (a)-[:HAS]->(:X)")
    assert column(g, "MATCH (n:A) RETURN count([p = (n)-[:HAS]->() | p]) AS c", "c") == [3]


def test_pattern2_7_inside_a_list_comprehension():
    g = graph(
        "CREATE (n1:X {n: 1}), (m1:Y), (i1:Y), (i2:Y) "
        "CREATE (n1)-[:T]->(m1), (m1)-[:T]->(i1), (m1)-[:T]->(i2) "
        "CREATE (n2:X {n: 2}), (m2:Z), (i3:L), (i4:Y) "
        "CREATE (n2)-[:T]->(m2), (m2)-[:T]->(i3), (m2)-[:T]->(i4)"
    )
    rows = g.cypher("MATCH p = (n:X)-->() RETURN n.n AS n, [x IN nodes(p) | size([(x)-->(:Y) | 1])] AS list").to_list()
    assert sorted((r["n"], r["list"]) for r in rows) == [(1, [1, 2]), (2, [0, 1])]


def test_pattern2_8_and_9_in_with():
    g = graph("CREATE (a:A), (b:B) CREATE (a)-[:T]->(b), (b)-[:T]->(:C)")
    rows = g.cypher("MATCH (n)-->(b) WITH [p = (n)-->() | p] AS ps, count(b) AS c RETURN ps, c").to_list()
    assert Counter((tuple(path_shape(p) for p in r["ps"]), r["c"]) for r in rows) == Counter(
        [(((("A",), "T", ("B",)),), 1), (((("B",), "T", ("C",)),), 1)]
    )
    g = graph("CREATE (:A)-[:T]->(:B)")
    rows = g.cypher("MATCH (a:A), (b:B) WITH [p = (a)-[*]->(b) | p] AS paths, count(a) AS c RETURN paths, c").to_list()
    assert [([path_shape(p) for p in r["paths"]], r["c"]) for r in rows] == [([(("A",), "T", ("B",))], 1)]


def test_null_correlated_variable_matches_nothing():
    """Match3 [27]/[28], Match7 [10]: a pattern from a null node has no matches."""
    g = graph("CREATE (:A)-[:T]->(:B)")
    assert g.cypher("OPTIONAL MATCH (a:Missing) WITH a MATCH (a)-->(b) RETURN b").to_list() == []
    assert g.cypher("OPTIONAL MATCH (a:Missing) WITH a OPTIONAL MATCH (a)-->(b) RETURN b").to_list() == [{"b": None}]
    assert g.cypher(
        "OPTIONAL MATCH (a:Missing) RETURN [(a)-->(b) | b] AS pc, [p = (a)<--() | p] AS pcp, "
        "COUNT { (a)-->() } AS c, EXISTS { (a)-->() } AS e"
    ).to_list() == [{"pc": [], "pcp": [], "c": 0, "e": False}]
    assert g.cypher("OPTIONAL MATCH (a:Missing) WITH a WHERE EXISTS { (a)-->() } RETURN count(*) AS n").to_list() == [
        {"n": 0}
    ]
