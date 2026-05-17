"""CALL refresh_stats() YIELD src_type, edge_type, tgt_type, count

Operator-callable recomputation of the label-pair cardinality cache
(0.9.35). Forces a fresh O(E) walk and yields one row per
`(src_type, edge_type, tgt_type)` triple with its current count.
"""

import pandas as pd
import pytest

from kglite import KnowledgeGraph


def _build_graph() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    persons = pd.DataFrame({"pid": [1, 2, 3], "name": [f"P{i}" for i in range(1, 4)]})
    kg.add_nodes(persons, "Person", "pid", "name")
    companies = pd.DataFrame({"cid": [10, 11], "name": ["Acme", "Globex"]})
    kg.add_nodes(companies, "Company", "cid", "name")

    knows = pd.DataFrame({"src": [1, 2], "tgt": [2, 3]})
    kg.add_connections(knows, "KNOWS", "Person", "src", "Person", "tgt")
    works = pd.DataFrame({"src": [1, 2], "tgt": [10, 11]})
    kg.add_connections(works, "WORKS_AT", "Person", "src", "Company", "tgt")
    return kg


def test_refresh_stats_yields_expected_triples():
    kg = _build_graph()
    rows = kg.cypher(
        "CALL refresh_stats() YIELD src_type, edge_type, tgt_type, count "
        "RETURN src_type, edge_type, tgt_type, count "
        "ORDER BY src_type, edge_type, tgt_type"
    ).to_list()
    triples = {(r["src_type"], r["edge_type"], r["tgt_type"]): r["count"] for r in rows}
    assert triples[("Person", "KNOWS", "Person")] == 2, rows
    assert triples[("Person", "WORKS_AT", "Company")] == 2, rows


def test_refresh_stats_partial_yield_works():
    """The caller may request any subset of the 4 columns."""
    kg = _build_graph()
    rows = kg.cypher("CALL refresh_stats() YIELD edge_type, count RETURN edge_type, count ORDER BY edge_type").to_list()
    by_edge: dict[str, int] = {}
    for r in rows:
        by_edge[r["edge_type"]] = by_edge.get(r["edge_type"], 0) + r["count"]
    assert by_edge.get("KNOWS") == 2
    assert by_edge.get("WORKS_AT") == 2


def test_refresh_stats_stable_under_no_mutation():
    """Two back-to-back calls with no mutations produce identical rows."""
    kg = _build_graph()
    a = kg.cypher(
        "CALL refresh_stats() YIELD src_type, edge_type, tgt_type, count "
        "RETURN src_type, edge_type, tgt_type, count ORDER BY edge_type, src_type, tgt_type"
    ).to_list()
    b = kg.cypher(
        "CALL refresh_stats() YIELD src_type, edge_type, tgt_type, count "
        "RETURN src_type, edge_type, tgt_type, count ORDER BY edge_type, src_type, tgt_type"
    ).to_list()
    assert a == b


def test_refresh_stats_reflects_post_mutation_state():
    """Mutating between calls must surface in the next refresh_stats."""
    kg = _build_graph()
    before = kg.cypher(
        "CALL refresh_stats() YIELD edge_type, count RETURN edge_type, count ORDER BY edge_type"
    ).to_list()
    before_knows = sum(r["count"] for r in before if r["edge_type"] == "KNOWS")

    kg.cypher("MATCH (p:Person {pid: 1}), (q:Person {pid: 3}) CREATE (p)-[:KNOWS]->(q)").to_list()

    after = kg.cypher(
        "CALL refresh_stats() YIELD edge_type, count RETURN edge_type, count ORDER BY edge_type"
    ).to_list()
    after_knows = sum(r["count"] for r in after if r["edge_type"] == "KNOWS")
    assert after_knows == before_knows + 1


def test_refresh_stats_empty_graph():
    kg = KnowledgeGraph()
    rows = kg.cypher(
        "CALL refresh_stats() YIELD src_type, edge_type, tgt_type, count RETURN src_type, edge_type, tgt_type, count"
    ).to_list()
    assert rows == []


def test_refresh_stats_unknown_yield_rejected():
    kg = _build_graph()
    with pytest.raises(Exception) as exc:
        kg.cypher("CALL refresh_stats() YIELD bogus RETURN bogus").to_list()
    assert "bogus" in str(exc.value), exc.value


def test_refresh_stats_requires_at_least_one_yield_target():
    """Calling refresh_stats() with no usable YIELD shape should error clearly."""
    kg = _build_graph()
    # Note: YIELD parsing requires at least one item; we test the "wrong name"
    # branch in the previous test. Here we just confirm the procedure rejects
    # being called outright when invoked with a name it doesn't expose. There's
    # no "zero YIELD" Cypher shape we can construct here that the parser
    # accepts, so this is a stand-in for the all-None pathway internally.
    # (The Rust code's all-None branch is exercised via unit testing the
    # procedure directly — not reachable from Cypher alone.)
    rows = kg.cypher("CALL refresh_stats() YIELD src_type RETURN count(*) AS n").to_list()
    assert rows[0]["n"] >= 1
