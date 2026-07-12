"""Outline projection — `CALL outline` (engine: tree structure) + `kglite.outline`
(binding: nested-markdown render).

The disciplined "graph as a skimmable document" projection: the engine yields
the spanning-tree structure; presentation lives in the binding layer.
"""

import kglite


def _tree():
    g = kglite.KnowledgeGraph()
    g.cypher(
        "CREATE (a:T {id: 'a', title: 'Build API'}), (b:T {id: 'b', title: 'Schema'}),"
        " (c:T {id: 'c', title: 'Handlers'}), (d:T {id: 'd', title: 'Tests'})"
    )
    for s, t in [("a", "b"), ("a", "c"), ("c", "d")]:
        g.cypher(f"MATCH (s:T {{id:'{s}'}}), (t:T {{id:'{t}'}}) CREATE (s)-[:DEP]->(t)")
    return g


def test_call_outline_yields_tree_structure():
    g = _tree()
    rows = g.cypher(
        "CALL outline({root: 'a', edge: 'DEP'}) YIELD node, depth, parent_id "
        "RETURN node.id AS id, depth, parent_id ORDER BY depth, id"
    ).to_dicts()
    assert rows == [
        {"id": "a", "depth": 0, "parent_id": None},
        {"id": "b", "depth": 1, "parent_id": "a"},
        {"id": "c", "depth": 1, "parent_id": "a"},
        {"id": "d", "depth": 2, "parent_id": "c"},
    ]


def test_call_outline_max_depth_bounds_descent():
    g = _tree()
    ids = {
        r["id"]
        for r in g.cypher(
            "CALL outline({root: 'a', edge: 'DEP', max_depth: 1}) YIELD node RETURN node.id AS id"
        ).to_dicts()
    }
    assert ids == {"a", "b", "c"}  # d (depth 2) excluded


def test_call_outline_dedups_dag():
    """A node reachable by two paths appears once (BFS first-discovery)."""
    g = _tree()
    g.cypher("MATCH (b:T {id:'b'}), (d:T {id:'d'}) CREATE (b)-[:DEP]->(d)")  # d now under b and c
    n = g.cypher(
        "CALL outline({root: 'a', edge: 'DEP'}) YIELD node WHERE node.id = 'd' RETURN count(node) AS c"
    ).to_dicts()[0]["c"]
    assert n == 1


def test_outline_renders_nested_markdown():
    assert kglite.outline(_tree(), "a", "DEP") == ("- Build API\n  - Schema\n  - Handlers\n    - Tests")


def test_outline_max_depth():
    assert kglite.outline(_tree(), "a", "DEP", max_depth=1) == ("- Build API\n  - Schema\n  - Handlers")


def test_outline_embeds_body_prose():
    g = _tree()
    g.cypher("MATCH (n:T {id:'a'}) SET n.notes = 'The public REST surface.'")
    out = kglite.outline(g, "a", "DEP", body="notes")
    assert out.splitlines()[:2] == ["- Build API", "  The public REST surface."]


def test_outline_empty_for_unknown_root():
    # CALL errors on a missing root; the binding surfaces it.
    import pytest

    with pytest.raises(Exception):
        kglite.outline(_tree(), "nope", "DEP")


def test_outline_in_list_procedures():
    g = _tree()
    names = {r["name"] for r in g.cypher("CALL list_procedures() YIELD name RETURN name").to_dicts()}
    assert "outline" in names


def test_outline_handles_depth_beyond_python_recursion_limit():
    import pandas as pd

    g = kglite.KnowledgeGraph()
    depth = 1_050
    g.add_nodes(
        pd.DataFrame(
            {
                "id": [f"n{i}" for i in range(depth + 1)],
                "title": [str(i) for i in range(depth + 1)],
            }
        ),
        "T",
        "id",
        "title",
    )
    g.add_connections(
        pd.DataFrame(
            {
                "source": [f"n{i}" for i in range(depth)],
                "target": [f"n{i + 1}" for i in range(depth)],
            }
        ),
        "DEP",
        "T",
        "source",
        "T",
        "target",
    )

    rendered = kglite.outline(g, "n0", "DEP")
    assert rendered.splitlines()[-1].strip() == f"- {depth}"
