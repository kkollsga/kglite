"""Parity oracle for the `MappedGraph` property index.

0.8.16 adds a lazy per-(node_type, property) and cross-type
`MappedPropertyIndex` so `MATCH (n:Type {prop: val})` and
`MATCH (n) WHERE n.prop STARTS WITH 'x'` hit a binary-search path
instead of scanning every node. The matcher's contract is that both
storage modes return identical rowsets; this test pins that.
"""

import pandas as pd
import pytest

from kglite import KnowledgeGraph


def _seed(graph):
    people = pd.DataFrame(
        {
            "nid": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Carol", "Alex", "Bertram"],
            "city": ["Oslo", "Bergen", "Oslo", "Oslo", "Bergen"],
            "age": [28, 35, 41, 22, 55],
        }
    )
    graph.add_nodes(people, "Person", "nid", "name")
    companies = pd.DataFrame({"nid": [100, 101], "name": ["Acme", "Alphabet"], "country": ["NO", "US"]})
    graph.add_nodes(companies, "Company", "nid", "name")


def _make_graph(mode, tmp_path):
    if mode == "default":
        return KnowledgeGraph()
    if mode == "mapped":
        return KnowledgeGraph(storage="mapped")
    if mode == "disk":
        return KnowledgeGraph(storage="disk", path=str(tmp_path))
    raise ValueError(mode)


STORAGE_MODES = ["default", "mapped", "disk"]


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_typed_property_equality(mode, tmp_path):
    g = _make_graph(mode, tmp_path)
    _seed(g)
    df = g.cypher("MATCH (p:Person {city: 'Oslo'}) RETURN p.name AS name ORDER BY name").to_df()
    assert list(df["name"]) == ["Alex", "Alice", "Carol"]


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_typed_property_starts_with(mode, tmp_path):
    g = _make_graph(mode, tmp_path)
    _seed(g)
    df = g.cypher("MATCH (p:Person) WHERE p.name STARTS WITH 'Al' RETURN p.name AS name ORDER BY name").to_df()
    assert list(df["name"]) == ["Alex", "Alice"]


@pytest.mark.parametrize("mode", STORAGE_MODES)
@pytest.mark.parametrize(
    ("predicate", "params", "expected"),
    [
        ("p.name CONTAINS 'li'", {}, ["Alice"]),
        ("p.name ENDS WITH $suffix", {"suffix": "m"}, ["Bertram"]),
    ],
)
def test_typed_property_text_filter(mode, tmp_path, predicate, params, expected):
    g = _make_graph(mode, tmp_path)
    _seed(g)
    df = g.cypher(
        f"MATCH (p:Person) WHERE {predicate} RETURN p.name AS name ORDER BY name",
        params=params,
    ).to_df()
    assert list(df["name"]) == expected


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_cross_type_property_equality(mode, tmp_path):
    g = _make_graph(mode, tmp_path)
    _seed(g)
    # `name` exists on both Person and Company; `{name: 'Alphabet'}`
    # should pick up the Company node across all modes.
    df = g.cypher("MATCH (n {name: 'Alphabet'}) RETURN n.name AS name").to_df()
    assert list(df["name"]) == ["Alphabet"]


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_cross_type_property_starts_with(mode, tmp_path):
    g = _make_graph(mode, tmp_path)
    _seed(g)
    df = g.cypher("MATCH (n) WHERE n.name STARTS WITH 'A' RETURN n.name AS name ORDER BY name").to_df()
    # Alice, Alex, Acme, Alphabet — sorted lexicographically
    assert list(df["name"]) == ["Acme", "Alex", "Alice", "Alphabet"]


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_property_lookup_empty(mode, tmp_path):
    g = _make_graph(mode, tmp_path)
    _seed(g)
    df = g.cypher("MATCH (p:Person {city: 'Trondheim'}) RETURN p.name AS name").to_df()
    assert len(df) == 0


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_invalidation_on_add_node(mode, tmp_path):
    g = _make_graph(mode, tmp_path)
    _seed(g)
    # First query builds the index (for mapped) / uses the scan (for memory/disk).
    df = g.cypher("MATCH (p:Person {city: 'Oslo'}) RETURN p.name AS name").to_df()
    assert len(df) == 3

    # Add a new Oslo resident; the mapped index must be invalidated and
    # rebuilt on the next query.
    more = pd.DataFrame({"nid": [6], "name": ["Dani"], "city": ["Oslo"], "age": [40]})
    g.add_nodes(more, "Person", "nid", "name")

    df = g.cypher("MATCH (p:Person {city: 'Oslo'}) RETURN p.name AS name ORDER BY name").to_df()
    assert list(df["name"]) == ["Alex", "Alice", "Carol", "Dani"]
