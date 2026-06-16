"""#1 — subgraph scoping for centrality / community procedures (operator
report 2026-06-16).

The centrality (pagerank, degree, betweenness, closeness) and community
(louvain, leiden, label_propagation) procedures accept an optional
`{node_type: '...', where: 'n.<prop> ...'}` scope so the algorithm runs on a
property-filtered subgraph — e.g. non-test, non-external functions — instead of
the polluted whole graph.
"""

from pathlib import Path

import pytest


def _code_graph(tmp_path: Path):
    """A tiny code graph: 3 library functions calling each other, plus a test
    helper (`assert_identical`) called from two test functions — the exact
    shape the report flagged (a test helper crowding the PageRank top-N)."""
    from kglite import code_tree

    root = tmp_path / "proj"
    root.mkdir()
    (root / "core.py").write_text(
        "def hub():\n    return leaf() + leaf2()\ndef leaf():\n    return 1\ndef leaf2():\n    return 2\n"
    )
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_x.py").write_text(
        "from core import hub\n"
        "def assert_identical():\n    return hub()\n"
        "def test_a():\n    return assert_identical()\n"
        "def test_b():\n    return assert_identical()\n"
    )
    return code_tree.build(str(root))


def _names(g, query):
    return [r["n"] for r in g.cypher(query)]


def test_pagerank_where_excludes_test_nodes(tmp_path: Path) -> None:
    g = _code_graph(tmp_path)

    unscoped = _names(
        g,
        "CALL pagerank({node_type:'Function', relationship:'CALLS'}) "
        "YIELD node, score RETURN node.name AS n ORDER BY score DESC",
    )
    # The report's symptom: the test helper pollutes the ranking.
    assert "assert_identical" in unscoped
    assert any(x.startswith("test_") for x in unscoped)

    scoped = _names(
        g,
        "CALL pagerank({node_type:'Function', relationship:'CALLS', "
        "where:'n.is_test = false'}) YIELD node, score RETURN node.name AS n ORDER BY score DESC",
    )
    assert scoped, "scoped pagerank should still return library functions"
    assert "assert_identical" not in scoped
    assert not any(x.startswith("test_") for x in scoped)
    assert set(scoped) == {"hub", "leaf", "leaf2"}


def test_pagerank_node_type_only_scope(tmp_path: Path) -> None:
    """node_type alone scopes to a label without a property predicate."""
    g = _code_graph(tmp_path)
    scoped = _names(
        g,
        "CALL pagerank({node_type:'Function'}) YIELD node, score RETURN node.name AS n",
    )
    # Only Function nodes appear (no File / Module).
    all_fn = {r["n"] for r in g.cypher("MATCH (f:Function) RETURN f.name AS n")}
    assert set(scoped) == all_fn


def test_where_multi_predicate(tmp_path: Path) -> None:
    g = _code_graph(tmp_path)
    scoped = _names(
        g,
        "CALL degree({node_type:'Function', "
        "where:'n.is_test = false AND n.is_benchmark = false'}) "
        "YIELD node, score RETURN node.name AS n ORDER BY score DESC",
    )
    assert set(scoped) == {"hub", "leaf", "leaf2"}


def test_louvain_where_scopes_community_set(tmp_path: Path) -> None:
    g = _code_graph(tmp_path)
    full = g.cypher("CALL louvain({node_type:'Function'}) YIELD node RETURN count(DISTINCT node) AS c")[0]["c"]
    scoped = g.cypher(
        "CALL louvain({node_type:'Function', where:'n.is_test = false'}) YIELD node RETURN count(DISTINCT node) AS c"
    )[0]["c"]
    assert scoped == 3, scoped
    assert scoped < full, (scoped, full)


def test_invalid_where_predicate_errors(tmp_path: Path) -> None:
    g = _code_graph(tmp_path)
    with pytest.raises(Exception):
        list(
            g.cypher(
                "CALL pagerank({node_type:'Function', where:'this is not valid ('}) YIELD node, score RETURN node.name"
            )
        )
