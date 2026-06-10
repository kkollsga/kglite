"""NetworkX interop: KnowledgeGraph.to_networkx() <-> kglite.from_networkx()."""

import sys
import time

import pandas as pd
import pytest

import kglite

nx = pytest.importorskip("networkx")


def _build_typed_graph():
    """A small two-type graph with mixed-type properties and parallel
    typed edges between the same node pair."""
    g = kglite.KnowledgeGraph()
    people = pd.DataFrame(
        [
            {"id": 1, "name": "Alice", "age": 30, "score": 9.5, "active": True, "note": None},
            {"id": 2, "name": "Bob", "age": 25, "score": 7.0, "active": False, "note": "x"},
        ]
    )
    g.add_nodes(people, "Person", "id", "name")
    cities = pd.DataFrame([{"id": 100, "name": "Oslo", "pop": 700000}])
    g.add_nodes(cities, "City", "id", "name")

    knows = pd.DataFrame([{"src": 1, "tgt": 2, "since": 2010}])
    g.add_connections(knows, "KNOWS", "Person", "src", "Person", "tgt", columns=["since"])
    # Two different edge types between the same pair (1 -> 2).
    likes = pd.DataFrame([{"src": 1, "tgt": 2}])
    g.add_connections(likes, "LIKES", "Person", "src", "Person", "tgt")
    lives = pd.DataFrame([{"src": 1, "tgt": 100}])
    g.add_connections(lives, "LIVES_IN", "Person", "src", "City", "tgt")
    return g


def test_to_networkx_basic_shape():
    g = _build_typed_graph()
    nxg = g.to_networkx()
    assert isinstance(nxg, nx.MultiDiGraph)
    assert nxg.number_of_nodes() == 3
    assert nxg.number_of_edges() == 3  # KNOWS, LIKES, LIVES_IN


def test_to_networkx_node_attrs():
    g = _build_typed_graph()
    nxg = g.to_networkx()
    assert set(nxg.nodes) == {1, 2, 100}
    alice = nxg.nodes[1]
    assert alice["node_type"] == "Person"
    assert alice["title"] == "Alice"
    assert alice["age"] == 30
    assert alice["score"] == 9.5
    assert alice["active"] is True
    city = nxg.nodes[100]
    assert city["node_type"] == "City"
    assert city["pop"] == 700000


def test_to_networkx_parallel_typed_edges():
    g = _build_typed_graph()
    nxg = g.to_networkx()
    # Two parallel edges between 1 and 2, keyed by connection_type.
    keys = set(nxg[1][2].keys())
    assert keys == {"KNOWS", "LIKES"}
    assert nxg[1][2]["KNOWS"]["connection_type"] == "KNOWS"
    assert nxg[1][2]["KNOWS"]["since"] == 2010


def test_round_trip_fidelity():
    g = _build_typed_graph()
    nxg = g.to_networkx()
    g2 = kglite.from_networkx(nxg)

    rt = g2.to_networkx()
    assert set(rt.nodes) == {1, 2, 100}
    assert rt.nodes[1]["node_type"] == "Person"
    assert rt.nodes[1]["title"] == "Alice"
    assert rt.nodes[1]["age"] == 30
    assert rt.nodes[1]["score"] == 9.5
    assert rt.nodes[1]["active"] is True
    assert rt.nodes[100]["node_type"] == "City"
    assert rt.nodes[100]["pop"] == 700000

    # Edge types + parallel typed edges survive.
    assert set(rt[1][2].keys()) == {"KNOWS", "LIKES"}
    assert rt[1][2]["KNOWS"]["since"] == 2010
    assert "LIVES_IN" in rt[1][100]


def test_from_networkx_plain_graph_defaults():
    nxg = nx.DiGraph()
    nxg.add_node("a")
    nxg.add_node("b")
    nxg.add_edge("a", "b")
    g = kglite.from_networkx(nxg)
    out = g.to_networkx()
    assert set(out.nodes) == {"a", "b"}
    assert out.nodes["a"]["node_type"] == "Node"
    assert out.nodes["a"]["title"] == "a"  # node key used as title
    assert out["a"]["b"]["RELATED"]["connection_type"] == "RELATED"


def test_from_networkx_custom_defaults():
    nxg = nx.DiGraph()
    nxg.add_edge("x", "y")
    g = kglite.from_networkx(nxg, default_node_type="Widget", default_edge_type="USES")
    out = g.to_networkx()
    assert out.nodes["x"]["node_type"] == "Widget"
    assert "USES" in out["x"]["y"]


def test_undirected_becomes_single_directed():
    nxg = nx.Graph()  # undirected
    nxg.add_edge("a", "b")
    g = kglite.from_networkx(nxg)
    out = g.to_networkx()
    # One directed edge total (undirected -> single directed).
    assert out.number_of_edges() == 1
    assert out.number_of_nodes() == 2


def test_empty_graph():
    nxg = nx.MultiDiGraph()
    g = kglite.from_networkx(nxg)
    out = g.to_networkx()
    assert out.number_of_nodes() == 0
    assert out.number_of_edges() == 0


def test_to_networkx_empty_kglite():
    g = kglite.KnowledgeGraph()
    out = g.to_networkx()
    assert isinstance(out, nx.MultiDiGraph)
    assert out.number_of_nodes() == 0


def test_missing_networkx_error(monkeypatch):
    """to_networkx() raises a clear ImportError when networkx is absent."""
    g = _build_typed_graph()
    # Hide networkx from the import machinery.
    monkeypatch.setitem(sys.modules, "networkx", None)
    with pytest.raises(ImportError, match="pip install networkx"):
        g.to_networkx()


def test_missing_networkx_error_from_networkx(monkeypatch):
    monkeypatch.setitem(sys.modules, "networkx", None)
    with pytest.raises(ImportError, match="pip install networkx"):
        kglite.from_networkx(object())


def test_10k_node_timing_sanity():
    """Round-trip on a 10k-node graph stays well under a generous bound,
    catching accidental O(n^2) behaviour."""
    n = 10_000
    nxg = nx.gnm_random_graph(n, n * 3, directed=True, seed=42)
    t0 = time.perf_counter()
    g = kglite.from_networkx(nxg)
    out = g.to_networkx()
    elapsed = time.perf_counter() - t0
    assert out.number_of_nodes() == n
    assert elapsed < 5.0, f"round-trip took {elapsed:.2f}s (expected < 5s)"
