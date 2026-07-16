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

import kglite

# Hand-built code-schema graph mirroring the shape the operator report
# flagged: three library functions (`hub` calling `leaf`/`leaf2`) plus a
# test helper (`assert_identical`) called from two test functions. Node
# props are exactly what a code-graph build emits for centrality/community
# scoping: `name`, `is_test`, `is_benchmark`, and CALLS edges.
#
# (fn_id, name, file_path, line, end_line, is_test)
_FUNCTIONS = [
    ("core.hub", "hub", "core.py", 1, 2, False),
    ("core.leaf", "leaf", "core.py", 3, 4, False),
    ("core.leaf2", "leaf2", "core.py", 5, 6, False),
    ("tests.test_x.assert_identical", "assert_identical", "tests/test_x.py", 2, 3, True),
    ("tests.test_x.test_a", "test_a", "tests/test_x.py", 4, 5, True),
    ("tests.test_x.test_b", "test_b", "tests/test_x.py", 6, 7, True),
]
_CALLS = [
    ("core.hub", "core.leaf"),
    ("core.hub", "core.leaf2"),
    ("tests.test_x.assert_identical", "core.hub"),
    ("tests.test_x.test_a", "tests.test_x.assert_identical"),
    ("tests.test_x.test_b", "tests.test_x.assert_identical"),
]


def _code_graph(tmp_path: Path):
    """A tiny code graph: 3 library functions calling each other, plus a test
    helper (`assert_identical`) called from two test functions — the exact
    shape the report flagged (a test helper crowding the PageRank top-N)."""
    g = kglite.KnowledgeGraph()
    for fn_id, name, file_path, line, end_line, is_test in _FUNCTIONS:
        g.cypher(
            "CREATE (f:Function {id: $id, name: $name, file_path: $fp, "
            "line_number: $line, end_line: $end, is_test: $is_test, is_benchmark: false})",
            params={
                "id": fn_id,
                "name": name,
                "fp": file_path,
                "line": line,
                "end": end_line,
                "is_test": is_test,
            },
        )
    for src, dst in _CALLS:
        g.cypher(
            "MATCH (a:Function {id: $src}), (b:Function {id: $dst}) CREATE (a)-[:CALLS]->(b)",
            params={"src": src, "dst": dst},
        )
    return g


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


def test_relationship_and_connection_types_are_interchangeable(tmp_path: Path) -> None:
    """A2 (operator 2026-06-17): the edge-scope key was inconsistent — centrality
    reads `connection_types`, connected_components reads `relationship`. Both are
    now aliased, so either term works on any procedure."""
    g = _code_graph(tmp_path)
    # `relationship` on pagerank (which natively reads `connection_types`) is now
    # honored — same scoped result as `connection_types`.
    via_conn = _names(
        g,
        "CALL pagerank({node_type:'Function', connection_types:'CALLS', where:'n.is_test = false'}) "
        "YIELD node, score RETURN node.name AS n ORDER BY score DESC",
    )
    via_rel = _names(
        g,
        "CALL pagerank({node_type:'Function', relationship:'CALLS', where:'n.is_test = false'}) "
        "YIELD node, score RETURN node.name AS n ORDER BY score DESC",
    )
    assert via_rel == via_conn, (via_rel, via_conn)
    assert "assert_identical" not in via_rel


def test_unknown_algo_config_key_errors_with_hint(tmp_path: Path) -> None:
    """A2b: an unknown config key is a clear boot error, not a silent no-op."""
    g = _code_graph(tmp_path)
    with pytest.raises(Exception, match="unknown config key 'bogus_key'"):
        list(g.cypher("CALL pagerank({node_type:'Function', bogus_key:'x'}) YIELD node RETURN node.name"))
    # A near-miss gets a did-you-mean suggestion.
    with pytest.raises(Exception, match="Did you mean 'connection_types'"):
        list(g.cypher("CALL pagerank({node_type:'Function', connection_typ:'CALLS'}) YIELD node RETURN node.name"))
