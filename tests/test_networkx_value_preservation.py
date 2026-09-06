"""Adapter-created frames must not round exact integers before ingestion."""

from fractions import Fraction

import pytest

import kglite

nx = pytest.importorskip("networkx")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")


@pytest.mark.parametrize("value", [9007199254740993, -(2**63), 2**63 - 1])
@pytest.mark.parametrize("missing", [True, False])
def test_networkx_nullable_integer_node_and_edge_properties(value, missing):
    graph = nx.DiGraph()
    graph.add_node(1, value=value)
    graph.add_node(2, **({} if missing else {"value": None}))
    graph.add_node(3)
    graph.add_edge(1, 2, value=value)
    graph.add_edge(2, 3, **({} if missing else {"value": None}))

    imported = kglite.from_networkx(graph)
    nodes = imported.cypher("MATCH (n) RETURN n.id AS id,n.value AS value ORDER BY id").to_list()
    edges = imported.cypher("MATCH (n)-[r]->() RETURN n.id AS id,r.value AS value ORDER BY id").to_list()
    assert nodes == [{"id": 1, "value": value}, {"id": 2, "value": None}, {"id": 3, "value": None}]
    assert edges == [{"id": 1, "value": value}, {"id": 2, "value": None}]
    assert type(nodes[0]["value"]) is int
    assert type(edges[0]["value"]) is int


@pytest.mark.parametrize(
    "value",
    [
        9007199254740993,
        np.int64(9007199254740993),
        np.uint64(9007199254740993),
        np.int64(-(2**63)),
        np.uint64(2**63 - 1),
    ],
    ids=["python-int", "numpy-int64", "numpy-uint64", "numpy-int64-min", "numpy-uint64-i64-max"],
)
@pytest.mark.parametrize("missing", [np.nan, pd.NA], ids=["numpy-nan", "pandas-na"])
def test_networkx_nullable_integer_missing_sentinels(value, missing):
    graph = nx.DiGraph()
    graph.add_node(1, value=value)
    graph.add_node(2, value=missing)
    graph.add_edge(1, 2, value=value)
    graph.add_edge(2, 1, value=missing)

    imported = kglite.from_networkx(graph)
    nodes = imported.cypher("MATCH (n) RETURN n.id AS id,n.value AS value ORDER BY id").to_list()
    edges = imported.cypher("MATCH (n)-[r]->() RETURN n.id AS id,r.value AS value ORDER BY id").to_list()
    assert nodes == [{"id": 1, "value": int(value)}, {"id": 2, "value": None}]
    assert edges == [{"id": 1, "value": int(value)}, {"id": 2, "value": None}]
    assert type(nodes[0]["value"]) is int
    assert type(edges[0]["value"]) is int


@pytest.mark.parametrize("missing", [np.nan, pd.NA], ids=["numpy-nan", "pandas-na"])
def test_networkx_nullable_integer_property_families_share_a_column(missing):
    values = [9007199254740993, np.int64(-(2**63)), np.uint64(2**63 - 1), missing]
    graph = nx.DiGraph()
    for node_id, value in enumerate(values, 1):
        graph.add_node(node_id, value=value)
        graph.add_edge(node_id, node_id % len(values) + 1, value=value)

    imported = kglite.from_networkx(graph)
    expected = [9007199254740993, -(2**63), 2**63 - 1, None]
    assert imported.cypher("MATCH (n) RETURN n.value AS v ORDER BY n.id").column("v") == expected
    assert imported.cypher("MATCH (n)-[r]->() RETURN r.value AS v ORDER BY n.id").column("v") == expected


@pytest.mark.parametrize("values", [[1, 2], [1.25, 2.5], [True, None], ["left", None]])
def test_networkx_ordinary_property_types(values):
    graph = nx.DiGraph()
    for index, value in enumerate(values):
        graph.add_node(index, value=value)
    actual = kglite.from_networkx(graph).cypher("MATCH (n) RETURN n.value AS v ORDER BY n.id").column("v")
    assert actual == values
    assert [type(value) for value in actual] == [type(value) for value in values]


def test_networkx_mixed_numeric_column_keeps_explicit_text_coercion():
    graph = nx.DiGraph()
    graph.add_node(1, value=9007199254740993)
    graph.add_node(2, value=1.25)
    with pytest.warns(UserWarning, match="stored as text"):
        imported = kglite.from_networkx(graph)
    assert imported.cypher("MATCH (n) RETURN n.value AS v ORDER BY n.id").column("v") == ["9007199254740993", "1.25"]


def test_networkx_huge_fraction_keeps_object_column_policy():
    value = Fraction(10**400)
    graph = nx.DiGraph()
    graph.add_node(1, value=value)
    graph.add_node(2, value=None)
    graph.add_edge(1, 2, value=value)
    graph.add_edge(2, 1, value=None)

    with pytest.warns(UserWarning, match="stored as text") as caught:
        imported = kglite.from_networkx(graph)
    assert len(caught) == 2
    expected = [str(value), None]
    assert imported.cypher("MATCH (n) RETURN n.value AS v ORDER BY n.id").column("v") == expected
    assert imported.cypher("MATCH (n)-[r]->() RETURN r.value AS v ORDER BY n.id").column("v") == expected
