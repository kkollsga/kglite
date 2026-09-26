"""``create_relationships(properties=...)`` copies node properties from every
node on the traversal path between the source and target levels.

Red proof: the multi-parent rewrite of ``create_connections`` walked each
target back to its source but kept only the two endpoints, so a type named
for an intermediate level (the documented ``{'B': [...]}`` case) copied
nothing."""

import pytest

import kglite


def _graph():
    g = kglite.KnowledgeGraph()
    g.cypher(
        """
        CREATE (a:A {id: 1, title: 'A1', label: 'a'}),
               (b:B {id: 10, title: 'B10', score: 1, weight: 0.5}),
               (c:C {id: 100, title: 'C100', w: 9}),
               (a)-[:AB]->(b), (b)-[:BC]->(c)
        """
    )
    return g


def _edge_props(g, rel="T"):
    rows = g.cypher(f"MATCH (s)-[r:{rel}]->(t) RETURN s.title AS s, t.title AS t, properties(r) AS p ORDER BY s, t")
    return [(r["s"], r["t"], {k: v for k, v in r["p"].items() if k != "type"}) for r in rows.to_list()]


def _chain(g):
    return g.select("A").traverse("AB").traverse("BC")


def test_intermediate_level_properties_are_copied():
    g = _chain(_graph()).create_relationships("T", properties={"B": ["score"]})
    assert _edge_props(g) == [("A1", "C100", {"score": 1})]


def test_empty_list_copies_every_intermediate_property():
    g = _chain(_graph()).create_relationships("T", properties={"B": []})
    props = _edge_props(g)[0][2]
    assert (props["score"], props["weight"], props["title"]) == (1, 0.5, "B10")


def test_source_intermediate_and_target_combine():
    g = _chain(_graph()).create_relationships("T", properties={"A": ["label"], "B": ["score"], "C": ["w"]})
    assert _edge_props(g) == [("A1", "C100", {"label": "a", "score": 1, "w": 9})]


def test_levels_outside_the_source_target_span_are_not_copied():
    g = _chain(_graph()).create_relationships("T", properties={"A": ["label"], "B": ["score"]}, source_type="B")
    assert _edge_props(g) == [("B10", "C100", {"score": 1})]


def _two_paths():
    g = kglite.KnowledgeGraph()
    g.cypher(
        """
        CREATE (a:A {id: 1, title: 'A1'}),
               (b1:B {id: 10, title: 'B10', score: 1}),
               (b2:B {id: 11, title: 'B11', score: 2}),
               (c:C {id: 100, title: 'C100'}),
               (a)-[:AB]->(b1), (a)-[:AB]->(b2), (b1)-[:BC]->(c), (b2)-[:BC]->(c)
        """
    )
    return g


@pytest.mark.parametrize(("mode", "score"), [("update", 2), ("skip", 1), ("preserve", 1), ("sum", 3)])
def test_paths_sharing_endpoints_fold_under_conflict_handling(mode, score):
    """Each path is one row, taken in node order (B10 before B11); rows
    joining the same pair fold like any other conflict — ``update`` keeps the
    later path's value, ``skip``/``preserve`` the first, ``sum`` adds them."""
    g = _chain(_two_paths()).create_relationships("T", conflict_handling=mode, properties={"B": ["score"]})
    assert _edge_props(g) == [("A1", "C100", {"score": score})]


def test_four_level_chain_copies_every_intermediate_level():
    g = kglite.KnowledgeGraph()
    g.cypher(
        """
        CREATE (a:A {id: 1, title: 'A1'}), (b:B {id: 10, title: 'B10', score: 1}),
               (c:C {id: 100, title: 'C100', depth: 3}), (d:D {id: 1000, title: 'D1000'}),
               (a)-[:AB]->(b), (b)-[:BC]->(c), (c)-[:CD]->(d)
        """
    )
    g = g.select("A").traverse("AB").traverse("BC").traverse("CD")
    g = g.create_relationships("T", properties={"B": ["score"], "C": ["depth"]})
    assert _edge_props(g) == [("A1", "D1000", {"score": 1, "depth": 3})]
