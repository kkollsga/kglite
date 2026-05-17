"""Path-decomposition Cypher functions: `length(p)`, `nodes(p)`, `relationships(p)`.

KGLite represents Cypher lists as `Value::String` of JSON (same shape as
`collect()`); the Python layer materialises them back to lists+dicts.
This file covers the shape contract that downstream agents rely on:

- `length(p)` → Int64 (number of hops).
- `nodes(p)` → list of node dicts, each carrying **every** property the
  node exposes (0.9.35 enrichment; pre-0.9.35 dicts only had
  `id`/`title`/`type`).
- `relationships(p)` (and the `rels(p)` alias) → list of edge-type strings.
"""

import pandas as pd

from kglite import KnowledgeGraph


def _build_graph() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    persons = pd.DataFrame({"pid": [1, 2, 3], "name": ["Alice", "Bob", "Carol"], "age": [28, 35, 42]})
    kg.add_nodes(persons, "Person", "pid", "name")
    knows = pd.DataFrame({"src": [1, 2], "tgt": [2, 3]})
    kg.add_connections(knows, "KNOWS", "Person", "src", "Person", "tgt")
    return kg


def test_length_returns_int():
    kg = _build_graph()
    rows = kg.cypher(
        "MATCH p = (a:Person {pid:1})-[:KNOWS*1..2]->(b:Person) RETURN length(p) AS L ORDER BY L"
    ).to_list()
    assert [r["L"] for r in rows] == [1, 2], rows


def test_nodes_returns_list_of_dicts():
    """nodes(p) materialises into a Python list (KGLite serialises lists
    as JSON strings which the Python layer auto-decodes)."""
    kg = _build_graph()
    rows = kg.cypher("MATCH p = (a:Person {pid:1})-[:KNOWS]->(b:Person) RETURN nodes(p) AS N").to_list()
    n = rows[0]["N"]
    assert isinstance(n, list) and len(n) == 2, n
    assert isinstance(n[0], dict) and isinstance(n[1], dict), n


def test_nodes_dict_carries_all_properties():
    """0.9.35 enrichment: dicts include every node property, not just
    id/title/type. Lets agents do `UNWIND nodes(p) AS n RETURN n.age`
    without re-MATCHing each node."""
    kg = _build_graph()
    rows = kg.cypher("MATCH p = (a:Person {pid:1})-[:KNOWS]->(b:Person) RETURN nodes(p) AS N").to_list()
    first = rows[0]["N"][0]
    # Canonical fields always present.
    assert first["id"] == 1
    assert first["title"] == "Alice"
    assert first["type"] == "Person"
    # 0.9.35: real property values are now there.
    assert first["age"] == 28, first


def test_unwind_nodes_then_access_property():
    """The agent-relevant workflow: `UNWIND nodes(p) AS n RETURN n.age`."""
    kg = _build_graph()
    rows = kg.cypher(
        "MATCH p = (a:Person {pid:1})-[:KNOWS*1..2]->(b:Person) UNWIND nodes(p) AS n RETURN n.age AS age ORDER BY age"
    ).to_list()
    ages = [r["age"] for r in rows]
    # Variable-length match yields paths of length 1 (Alice→Bob) and 2
    # (Alice→Bob→Carol); UNWIND flattens both → [28,35] + [28,35,42] = 5 rows.
    assert sorted(ages) == [28, 28, 35, 35, 42], rows


def test_relationships_returns_list_of_strings():
    kg = _build_graph()
    rows = kg.cypher(
        "MATCH p = (a:Person {pid:1})-[:KNOWS*1..2]->(b:Person) RETURN relationships(p) AS R ORDER BY length(p)"
    ).to_list()
    assert rows[0]["R"] == ["KNOWS"]
    assert rows[1]["R"] == ["KNOWS", "KNOWS"]


def test_rels_alias_works():
    """`rels(p)` is the short alias for `relationships(p)` (existing surface)."""
    kg = _build_graph()
    rows = kg.cypher("MATCH p = (a:Person {pid:1})-[:KNOWS]->(b:Person) RETURN rels(p) AS R").to_list()
    assert rows[0]["R"] == ["KNOWS"]


def test_size_of_nodes_consistent_with_length():
    """size(nodes(p)) = length(p) + 1 (path has N+1 nodes for N hops)."""
    kg = _build_graph()
    rows = kg.cypher(
        "MATCH p = (a:Person {pid:1})-[:KNOWS*1..2]->(b:Person) RETURN length(p) AS L, size(nodes(p)) AS NS ORDER BY L"
    ).to_list()
    for r in rows:
        assert r["NS"] == r["L"] + 1, r


def test_special_chars_in_property_values_escaped():
    """A property value containing a double-quote must round-trip through
    the JSON serialization."""
    kg = KnowledgeGraph()
    kg.add_nodes(
        pd.DataFrame({"pid": [1, 2], "name": ['has "quote"', "ok"]}),
        "Person",
        "pid",
        "name",
    )
    kg.add_connections(
        pd.DataFrame({"src": [1], "tgt": [2]}),
        "KNOWS",
        "Person",
        "src",
        "Person",
        "tgt",
    )
    rows = kg.cypher("MATCH p = (a:Person {pid:1})-[:KNOWS]->(b:Person) RETURN nodes(p) AS N").to_list()
    assert rows[0]["N"][0]["title"] == 'has "quote"', rows
