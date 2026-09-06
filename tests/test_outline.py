"""Outline projection — `CALL outline` (engine: tree structure) + `kglite.outline`
(binding: nested-markdown render).

The disciplined "graph as a skimmable document" projection: the engine yields
the spanning-tree structure; presentation lives in the binding layer.
"""

import pytest

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


def _duplicate_id_children():
    g = kglite.KnowledgeGraph()
    g.cypher(
        "CREATE (r:R {id: 0, title: 'Root'}),"
        " (a:A {id: 1, title: 'Left'}),"
        " (b:B {id: 1, title: 'Right'}),"
        " (r)-[:DEP]->(a), (r)-[:DEP]->(b)"
    )
    return g


def _python_equal_id_children():
    g = kglite.KnowledgeGraph()
    g.cypher(
        "CREATE (r:R {id: 0, title: 'Root'}),"
        " (i:N {id: 1, title: 'Integer'}),"
        " (f:N {id: 1.0, title: 'Float'}),"
        " (b:N {id: true, title: 'Boolean'}),"
        " (r)-[:DEP]->(i), (r)-[:DEP]->(f), (r)-[:DEP]->(b)"
    )
    return g


def _unique_and_int_id_children():
    g = kglite.KnowledgeGraph()
    g.cypher(
        "CREATE (r:R {id: 'root', title: 'Root'}),"
        " (a:N {title: 'Auto'}),"
        " (i:N {id: 1, title: 'Explicit'}),"
        " (r)-[:DEP]->(a), (r)-[:DEP]->(i)"
    )
    return g


def test_call_outline_yields_unambiguous_node_and_parent_types():
    rows = (
        _duplicate_id_children()
        .cypher(
            "CALL outline({root: 0, root_type: 'R', edge: 'DEP'}) "
            "YIELD node, depth, parent_id, node_type, node_id_type, parent_type, parent_id_type, "
            "node_token, parent_token "
            "RETURN node.id AS id, depth, parent_id, node_type, node_id_type, parent_type, "
            "parent_id_type, node_token, parent_token "
            "ORDER BY depth, node_type"
        )
        .to_dicts()
    )
    assert rows == [
        {
            "id": 0,
            "depth": 0,
            "parent_id": None,
            "node_type": "R",
            "node_id_type": "Int64",
            "parent_type": None,
            "parent_id_type": None,
            "node_token": 0,
            "parent_token": None,
        },
        {
            "id": 1,
            "depth": 1,
            "parent_id": 0,
            "node_type": "A",
            "node_id_type": "Int64",
            "parent_type": "R",
            "parent_id_type": "Int64",
            "node_token": 1,
            "parent_token": 0,
        },
        {
            "id": 1,
            "depth": 1,
            "parent_id": 0,
            "node_type": "B",
            "node_id_type": "Int64",
            "parent_type": "R",
            "parent_id_type": "Int64",
            "node_token": 2,
            "parent_token": 0,
        },
    ]


def test_outline_renders_duplicate_ids_from_different_types_once_each():
    assert kglite.outline(_duplicate_id_children(), 0, "DEP", root_type="R") == ("- Root\n  - Left\n  - Right")


def test_outline_distinguishes_python_equal_ids_by_value_type():
    g = _python_equal_id_children()
    rows = g.cypher(
        "CALL outline({root: 0, root_type: 'R', edge: 'DEP'}) "
        "YIELD node, depth, node_id_type, parent_id_type "
        "WHERE depth = 1 "
        "RETURN node.id AS id, node.title AS title, node_id_type, parent_id_type "
        "ORDER BY node_id_type"
    ).to_dicts()
    assert rows == [
        {"id": True, "title": "Boolean", "node_id_type": "Boolean", "parent_id_type": "Int64"},
        {"id": 1.0, "title": "Float", "node_id_type": "Float64", "parent_id_type": "Int64"},
        {"id": 1, "title": "Integer", "node_id_type": "Int64", "parent_id_type": "Int64"},
    ]
    assert kglite.outline(g, 0, "DEP", root_type="R") == ("- Root\n  - Integer\n  - Float\n  - Boolean")


def test_outline_distinguishes_unique_and_explicit_integer_ids():
    g = _unique_and_int_id_children()
    rows = g.cypher(
        "CALL outline({root: 'root', root_type: 'R', edge: 'DEP'}) "
        "YIELD node, depth, node_id_type "
        "WHERE depth = 1 "
        "RETURN node.id AS id, node.title AS title, node_id_type "
        "ORDER BY node_id_type"
    ).to_dicts()
    assert rows == [
        {"id": 1, "title": "Explicit", "node_id_type": "Int64"},
        {"id": 1, "title": "Auto", "node_id_type": "UniqueId"},
    ]
    assert kglite.outline(g, "root", "DEP", root_type="R") == ("- Root\n  - Explicit\n  - Auto")


def _bound_renderer_visits(monkeypatch, limit=10):
    """Turn an identity loop into a bounded assertion instead of a hang."""
    real_sorted = sorted
    calls = 0

    def bounded_sorted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > limit:
            raise AssertionError(f"outline renderer exceeded {limit} node visits")
        return real_sorted(*args, **kwargs)

    monkeypatch.setitem(kglite.outline.__globals__, "sorted", bounded_sorted)


def test_outline_same_id_parent_child_respects_depth_bound_and_terminates(monkeypatch):
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (r:R {id: 1, title: 'Root'}), (c:C {id: 1, title: 'Child'}), (r)-[:DEP]->(c)")
    _bound_renderer_visits(monkeypatch)
    assert kglite.outline(g, 1, "DEP", root_type="R", max_depth=1) == "- Root\n  - Child"


def test_outline_ambiguous_root_requires_type_and_typed_root_selects_exact_node():
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:A {id: 1, title: 'Left'}), (:B {id: 1, title: 'Right'})")

    import pytest

    with pytest.raises(Exception, match="ambiguous.*root_type"):
        kglite.outline(g, 1, "DEP")
    assert kglite.outline(g, 1, "DEP", root_type="B") == "- Right"


def test_outline_auto_id_round_trips_and_publicly_equal_same_type_ids_are_ambiguous():
    auto = kglite.KnowledgeGraph()
    auto.cypher("CREATE (:T {title: 'Auto'})")
    root = auto.cypher("MATCH (n:T) RETURN n.id AS id").to_dicts()[0]["id"]
    assert root == 0
    assert kglite.outline(auto, root, "DEP", root_type="T") == "- Auto"

    ambiguous = kglite.KnowledgeGraph()
    ambiguous.cypher("CREATE (:T {title: 'Auto'}), (:T {id: 0, title: 'Explicit'})")
    import pytest

    with pytest.raises(Exception, match="ambiguous"):
        kglite.outline(ambiguous, 0, "DEP", root_type="T")


def test_outline_unknown_typed_root_is_an_explicit_error():
    import pytest

    with pytest.raises(Exception, match="no node"):
        kglite.outline(_tree(), "nope", "DEP", root_type="T")


def test_outline_renders_admitted_unhashable_ids():
    for root_id, child_id in [([1, 2], [3, 4]), ({"a": 1}, {"a": 2})]:
        g = kglite.KnowledgeGraph()
        g.cypher(
            "CREATE (r:R {id: $root, title: 'Root'}), (c:C {id: $child, title: 'Child'}), (r)-[:DEP]->(c)",
            params={"root": root_id, "child": child_id},
        )
        assert kglite.outline(g, root_id, "DEP", root_type="R") == "- Root\n  - Child"


def test_outline_null_ids_do_not_alias_the_root_parent_sentinel():
    root_null = kglite.KnowledgeGraph()
    root_null.cypher("CREATE (r:R {id: null, title: 'Root'}), (c:C {id: 'c', title: 'Child'}), (r)-[:DEP]->(c)")
    assert kglite.outline(root_null, None, "DEP", root_type="R") == "- Root\n  - Child"

    child_null = kglite.KnowledgeGraph()
    child_null.cypher(
        "CREATE (r:R {id: 'r', title: 'Root'}), (c:C {id: null, title: 'Child'}), "
        "(g:G {id: 'g', title: 'Grand'}), (r)-[:DEP]->(c), (c)-[:DEP]->(g)"
    )
    assert kglite.outline(child_null, "r", "DEP", root_type="R") == "- Root\n  - Child\n    - Grand"


def test_outline_cycle_renders_each_typed_node_once(monkeypatch):
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (a:A {id: 1, title: 'A'}), (b:B {id: 1, title: 'B'}), (a)-[:DEP]->(b), (b)-[:DEP]->(a)")
    _bound_renderer_visits(monkeypatch)
    assert kglite.outline(g, 1, "DEP", root_type="A") == "- A\n  - B"


def test_outline_repeated_ids_obey_zero_and_one_depth_bounds():
    g = _duplicate_id_children()
    assert kglite.outline(g, 0, "DEP", root_type="R", max_depth=0) == "- Root"
    assert kglite.outline(g, 0, "DEP", root_type="R", max_depth=1) == ("- Root\n  - Left\n  - Right")


def test_call_outline_rejects_negative_max_depth():
    import pytest

    with pytest.raises(Exception, match="max_depth.*non-negative"):
        _tree().cypher("CALL outline({root: 'a', root_type: 'T', edge: 'DEP', max_depth: -1}) YIELD node RETURN node")


def test_outline_registry_and_bare_call_expose_identity_columns():
    g = _tree()
    listed = g.cypher(
        "CALL list_procedures() YIELD name, yield_columns WHERE name = 'outline' RETURN yield_columns"
    ).to_dicts()
    assert listed == [
        {
            "yield_columns": (
                "node, depth, parent_id, node_type, node_id_type, parent_type, parent_id_type, node_token, parent_token"
            )
        }
    ]

    table = g.cypher(
        "CALL outline({root: 'a', root_type: 'T', edge: 'DEP', max_depth: 0})",
        to_df=True,
    )
    assert list(table.columns) == [
        "node",
        "depth",
        "parent_id",
        "node_type",
        "node_id_type",
        "parent_type",
        "parent_id_type",
        "node_token",
        "parent_token",
    ]


@pytest.mark.parametrize("storage", [None, "mapped", "disk"])
def test_outline_warm_id_index_tracks_duplicate_delete_and_slot_reuse(storage, tmp_path):
    kwargs = {} if storage is None else {"storage": storage}
    if storage == "disk":
        kwargs["path"] = str(tmp_path / "outline-disk")
    graph = kglite.KnowledgeGraph(**kwargs)
    graph.cypher("CREATE (:T {id: 'root', title: 'First'}), (:T {id: 'other'})")
    assert kglite.outline(graph, "root", "DEP", root_type="T") == "- First"

    graph.cypher("CREATE (:T {id: 'root', title: 'Duplicate'})")
    with pytest.raises(Exception, match="ambiguous"):
        kglite.outline(graph, "root", "DEP", root_type="T")

    graph.cypher("MATCH (n:T {title: 'Duplicate'}) DETACH DELETE n")
    assert kglite.outline(graph, "root", "DEP", root_type="T") == "- First"
    graph.cypher("MATCH (n:T {title: 'First'}) DETACH DELETE n")
    graph.cypher("CREATE (:T {id: 'root', title: 'Reused'})")
    assert kglite.outline(graph, "root", "DEP", root_type="T") == "- Reused"


@pytest.mark.parametrize("root_id", [True, 1.0, [1, 2], {"a": [1]}, None])
def test_outline_exact_duplicate_ids_fall_back_to_ambiguity_scan(root_id):
    graph = kglite.KnowledgeGraph()
    graph.cypher(
        "CREATE (:T {id: $id, title: 'Left'}), (:T {id: $id, title: 'Right'})",
        params={"id": root_id},
    )
    with pytest.raises(Exception, match="ambiguous"):
        kglite.outline(graph, root_id, "DEP", root_type="T")
