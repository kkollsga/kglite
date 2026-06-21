"""FOREACH (var IN list | <update clauses>) — mutation control-flow.

Runs the body's update clauses once per list element with the loop
variable bound. A side-effect loop: the surrounding rows are unchanged.
0.12 Tier 2.
"""

import pytest

import kglite


def test_foreach_standalone_create():
    g = kglite.KnowledgeGraph()
    g.cypher("FOREACH (x IN [1, 2, 3] | CREATE (:N {id: x}))")
    ids = sorted(r["id"] for r in g.cypher("MATCH (n:N) RETURN n.id AS id").to_list())
    assert ids == [1, 2, 3]


def test_foreach_param_list_of_dicts():
    g = kglite.KnowledgeGraph()
    g.cypher(
        "FOREACH (r IN $rows | CREATE (:P {id: r.id, name: r.name}))",
        params={"rows": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]},
    )
    names = [r["n"] for r in g.cypher("MATCH (p:P) RETURN p.name AS n ORDER BY p.id").to_list()]
    assert names == ["A", "B"]


def test_foreach_set_per_matched_row():
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Q {id: 1}), (:Q {id: 2})")
    g.cypher("MATCH (q:Q) FOREACH (_ IN [1] | SET q.touched = true)")
    assert g.cypher("MATCH (q:Q) WHERE q.touched = true RETURN count(q) AS c")[0]["c"] == 2


def test_foreach_over_node_property_list():
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Acc {id: 1, items: [10, 20, 30]})")
    g.cypher("MATCH (a:Acc) FOREACH (i IN a.items | CREATE (:Item {v: i}))")
    vs = sorted(r["v"] for r in g.cypher("MATCH (i:Item) RETURN i.v AS v").to_list())
    assert vs == [10, 20, 30]


def test_foreach_nested():
    g = kglite.KnowledgeGraph()
    g.cypher("FOREACH (x IN [1, 2] | FOREACH (y IN [10, 20] | CREATE (:Pair {x: x, y: y})))")
    assert g.cypher("MATCH (p:Pair) RETURN count(p) AS c")[0]["c"] == 4


def test_foreach_over_null_is_noop():
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Acc {id: 1})")  # no `items` property → null
    g.cypher("MATCH (a:Acc) FOREACH (i IN a.items | CREATE (:Item {v: i}))")
    assert g.cypher("MATCH (i:Item) RETURN count(i) AS c")[0]["c"] == 0


def test_foreach_empty_list_is_noop():
    g = kglite.KnowledgeGraph()
    g.cypher("FOREACH (x IN [] | CREATE (:N {id: x}))")
    assert g.cypher("MATCH (n:N) RETURN count(n) AS c")[0]["c"] == 0


def test_foreach_delete_in_body():
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Tmp {id: 1}), (:Tmp {id: 2}), (:Keep {id: 3})")
    g.cypher("MATCH (t:Tmp) FOREACH (_ IN [1] | DELETE t)")
    assert g.cypher("MATCH (t:Tmp) RETURN count(t) AS c")[0]["c"] == 0
    assert g.cypher("MATCH (k:Keep) RETURN count(k) AS c")[0]["c"] == 1


def test_foreach_non_list_errors():
    g = kglite.KnowledgeGraph()
    with pytest.raises(Exception):
        g.cypher("FOREACH (x IN 42 | CREATE (:N {id: x}))")


def test_foreach_body_rejects_read_clause():
    g = kglite.KnowledgeGraph()
    # A FOREACH body may only contain update clauses; MATCH is rejected at parse.
    with pytest.raises(Exception):
        g.cypher("FOREACH (x IN [1] | MATCH (n) RETURN n)")
