"""Label-pair edge-count cardinality cache.

The Cypher planner uses `(src_type, edge_type, tgt_type) → count` triples
to cost MATCH patterns more accurately than the previous "all R edges"
proxy. The cache is lazy: first read computes O(E); edge mutations
invalidate it. This file covers the cache itself; planner-output
guarantees live in tests/test_cypher_differential.py.
"""

import pandas as pd

from kglite import KnowledgeGraph


def _build_skewed_graph() -> KnowledgeGraph:
    """3 node types + 2 edge types, with asymmetric label-pair counts:

      KNOWS:  Person→Person × 4
      WORKS_AT: Person→Company × 2, Person→Charity × 1

    The two KNOWS triples (Person, KNOWS, Person) = 4 and WORKS_AT
    spreads across two target types — exactly the shape the planner's
    new selectivity-aware branch is meant to distinguish.
    """
    kg = KnowledgeGraph()
    persons = pd.DataFrame({"pid": [1, 2, 3, 4, 5], "name": [f"P{i}" for i in range(1, 6)]})
    kg.add_nodes(persons, "Person", "pid", "name")
    companies = pd.DataFrame({"cid": [10, 11], "name": ["Acme", "Globex"]})
    kg.add_nodes(companies, "Company", "cid", "name")
    charities = pd.DataFrame({"chid": [20], "name": ["RedCross"]})
    kg.add_nodes(charities, "Charity", "chid", "name")

    knows = pd.DataFrame({"src": [1, 1, 2, 3], "tgt": [2, 3, 3, 4]})
    kg.add_connections(knows, "KNOWS", "Person", "src", "Person", "tgt")

    works = pd.DataFrame({"src": [1, 2, 3], "kind": ["Company", "Company", "Charity"], "tgt": [10, 11, 20]})
    # Split by target type — add_connections is per-target-type
    kg.add_connections(
        works[works["kind"] == "Company"][["src", "tgt"]],
        "WORKS_AT",
        "Person",
        "src",
        "Company",
        "tgt",
    )
    kg.add_connections(
        works[works["kind"] == "Charity"][["src", "tgt"]],
        "WORKS_AT",
        "Person",
        "src",
        "Charity",
        "tgt",
    )
    return kg


def test_label_pair_counts_basic_shape():
    kg = _build_skewed_graph()
    rows = kg.label_pair_counts()
    pairs = {(s, e, t): c for (s, e, t, c) in rows}
    assert pairs[("Person", "KNOWS", "Person")] == 4, pairs
    assert pairs[("Person", "WORKS_AT", "Company")] == 2, pairs
    assert pairs[("Person", "WORKS_AT", "Charity")] == 1, pairs


def test_label_pair_counts_invalidated_on_cypher_create():
    kg = _build_skewed_graph()
    before = {(s, e, t): c for (s, e, t, c) in kg.label_pair_counts()}
    kg.cypher("MATCH (p:Person {pid: 4}), (q:Person {pid: 5}) CREATE (p)-[:KNOWS]->(q)").to_list()
    after = {(s, e, t): c for (s, e, t, c) in kg.label_pair_counts()}
    assert after[("Person", "KNOWS", "Person")] == before[("Person", "KNOWS", "Person")] + 1


def test_label_pair_counts_invalidated_on_cypher_delete():
    kg = _build_skewed_graph()
    before = {(s, e, t): c for (s, e, t, c) in kg.label_pair_counts()}
    kg.cypher("MATCH (p:Person {pid: 1})-[r:KNOWS]->(q:Person {pid: 2}) DELETE r").to_list()
    after = {(s, e, t): c for (s, e, t, c) in kg.label_pair_counts()}
    assert after[("Person", "KNOWS", "Person")] == before[("Person", "KNOWS", "Person")] - 1


def test_label_pair_counts_invalidated_on_python_add_connections():
    """Pre-0.9.35, the Python bulk-mutation path didn't invalidate the
    edge-cardinality caches — Step 1 fixes that gap as a side effect."""
    kg = _build_skewed_graph()
    before = {(s, e, t): c for (s, e, t, c) in kg.label_pair_counts()}
    extra = pd.DataFrame({"src": [4], "tgt": [5]})
    kg.add_connections(extra, "KNOWS", "Person", "src", "Person", "tgt")
    after = {(s, e, t): c for (s, e, t, c) in kg.label_pair_counts()}
    assert after[("Person", "KNOWS", "Person")] == before[("Person", "KNOWS", "Person")] + 1


def test_label_pair_counts_stable_under_repeat_reads():
    """Two reads back-to-back, no mutations in between, must agree byte-for-byte.
    Guards against the lazy cache silently re-computing on each call."""
    kg = _build_skewed_graph()
    a = sorted(kg.label_pair_counts())
    b = sorted(kg.label_pair_counts())
    assert a == b


def test_label_pair_counts_empty_graph():
    """An empty graph yields zero triples — not an error."""
    kg = KnowledgeGraph()
    assert kg.label_pair_counts() == []
