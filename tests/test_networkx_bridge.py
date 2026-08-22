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


def test_to_networkx_preserves_same_type_parallel_edges():
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (a:N {id:1}), (b:N {id:2}) CREATE (a)-[:R {rank:1}]->(b), (a)-[:R {rank:2}]->(b)")
    nxg = g.to_networkx()
    assert nxg.number_of_edges(1, 2) == 2
    edges = list(nxg.get_edge_data(1, 2).values())
    assert {edge["connection_type"] for edge in edges} == {"R"}
    assert {edge["rank"] for edge in edges} == {1, 2}


@pytest.mark.parametrize("mode", ["memory", "mapped", "disk"])
def test_to_networkx_preserves_columnar_node_properties(mode, tmp_path):
    kwargs = {}
    if mode == "mapped":
        kwargs["storage"] = "mapped"
    elif mode == "disk":
        kwargs.update(storage="disk", path=str(tmp_path / "disk_graph"))
    g = kglite.KnowledgeGraph(**kwargs)
    g.add_nodes(pd.DataFrame({"id": [1], "name": ["A"], "score": [9.5]}), "N", "id", "name")
    assert g.to_networkx().nodes[1]["score"] == 9.5

    saved = str(tmp_path / f"{mode}.kgl")
    g.save(saved)
    assert kglite.load(saved).to_networkx().nodes[1]["score"] == 9.5


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


# ── id-collision refusal ───────────────────────────────────────────────
#
# Node ids are unique per type, not across types. A bare-id nx node key
# therefore MERGES two types that share an id (the second add_node
# overwrites the first's attrs and both nodes' edges rewire onto the
# survivor) — a silently wrong graph. The export refuses instead.


def _colliding_id_graph():
    """Two types whose ids overlap on 5, with one edge each."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame([{"id": 5, "name": "Alice"}, {"id": 6, "name": "Bob"}]), "Person", "id", "name")
    g.add_nodes(pd.DataFrame([{"id": 5, "name": "Oslo"}]), "City", "id", "name")
    g.add_connections(pd.DataFrame([{"src": 6, "tgt": 5}]), "KNOWS", "Person", "src", "Person", "tgt")
    g.add_connections(pd.DataFrame([{"src": 6, "tgt": 5}]), "LIVES_IN", "Person", "src", "City", "tgt")
    return g


def test_to_networkx_raises_on_cross_type_id_collision():
    g = _colliding_id_graph()
    with pytest.raises(kglite.ArgumentError, match="sharing id"):
        g.to_networkx()


def test_to_networkx_collision_error_names_the_workaround():
    """The export is whole-graph, so the recipe cannot be 'narrow the
    selection' — it names disjoint ids and the `node_key='type_id'`
    escape hatch, both of which work on the graph that collided."""
    g = _colliding_id_graph()
    with pytest.raises(kglite.ArgumentError) as excinfo:
        g.to_networkx()
    message = str(excinfo.value)
    assert "to_networkx()" in message
    assert "disjoint ids" in message
    assert "node_key='type_id'" in message
    assert "whole graph" in message


def test_to_networkx_ignores_selection_so_selecting_one_type_still_raises():
    """Pins the honest limitation the error message asserts: v1 exports the
    whole graph, so filtering to one type does NOT dodge the collision."""
    g = _colliding_id_graph()
    with pytest.raises(kglite.ArgumentError, match="sharing id"):
        g.select("Person").to_networkx()


def test_to_networkx_single_type_multi_node_unchanged():
    """The refusal is collision-scoped: one type with many nodes still
    exports every node with its attributes."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            [
                {"id": 1, "name": "Alice", "age": 30},
                {"id": 2, "name": "Bob", "age": 25},
                {"id": 3, "name": "Cleo", "age": 41},
            ]
        ),
        "Person",
        "id",
        "name",
    )
    g.add_connections(
        pd.DataFrame([{"src": 1, "tgt": 2}, {"src": 2, "tgt": 3}]),
        "KNOWS",
        "Person",
        "src",
        "Person",
        "tgt",
    )
    nxg = g.to_networkx()
    assert nxg.number_of_nodes() == 3
    assert nxg.number_of_edges() == 2
    assert set(nxg.nodes) == {1, 2, 3}
    assert nxg.nodes[3]["node_type"] == "Person"
    assert nxg.nodes[3]["title"] == "Cleo"
    assert nxg.nodes[3]["age"] == 41


# ── node_key="type_id": the collision-free escape hatch ────────────────
#
# Bare-id keys are the default and still refuse a cross-type collision.
# `node_key="type_id"` keys every node by its `(node_type, id)` pair,
# which IS graph-unique, so a colliding graph exports losslessly.


def test_to_networkx_type_id_exports_colliding_graph():
    g = _colliding_id_graph()
    nxg = g.to_networkx(node_key="type_id")
    assert nxg.number_of_nodes() == 3
    assert set(nxg.nodes) == {("Person", 5), ("Person", 6), ("City", 5)}
    assert nxg.number_of_edges() == 2


def test_to_networkx_type_id_attrs_and_edge_endpoints():
    g = _colliding_id_graph()
    nxg = g.to_networkx(node_key="type_id")
    assert nxg.nodes[("Person", 5)]["node_type"] == "Person"
    assert nxg.nodes[("Person", 5)]["title"] == "Alice"
    assert nxg.nodes[("City", 5)]["node_type"] == "City"
    assert nxg.nodes[("City", 5)]["title"] == "Oslo"
    # Endpoints are tuples too — the two same-id nodes stay distinct.
    assert nxg.has_edge(("Person", 6), ("Person", 5), "KNOWS")
    assert nxg.has_edge(("Person", 6), ("City", 5), "LIVES_IN")
    assert not nxg.has_edge(("Person", 6), ("Person", 5), "LIVES_IN")


def test_to_networkx_type_id_parallel_edge_keys_unchanged():
    """Edge keying is orthogonal to node keying: first same-type edge keeps
    the bare connection type, the parallel one gets the composite key."""
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (a:N {id:1}), (b:N {id:2}) CREATE (a)-[:R {rank:1}]->(b), (a)-[:R {rank:2}]->(b)")
    nxg = g.to_networkx(node_key="type_id")
    src, tgt = ("N", 1), ("N", 2)
    assert nxg.number_of_edges(src, tgt) == 2
    keys = list(nxg[src][tgt].keys())
    assert "R" in keys
    composite = [key for key in keys if key != "R"]
    assert len(composite) == 1 and isinstance(composite[0], tuple) and composite[0][0] == "R"
    assert {edge["rank"] for edge in nxg[src][tgt].values()} == {1, 2}


def test_to_networkx_default_node_key_is_bare_id():
    """The default is byte-for-byte today's behaviour: bare-id node keys."""
    g = _build_typed_graph()
    assert set(g.to_networkx().nodes) == {1, 2, 100}
    assert set(g.to_networkx(node_key="id").nodes) == {1, 2, 100}


def test_to_networkx_unknown_node_key_rejected_by_name():
    g = _build_typed_graph()
    with pytest.raises(kglite.ArgumentError) as excinfo:
        g.to_networkx(node_key="uuid")
    message = str(excinfo.value)
    assert "uuid" in message
    assert "'id'" in message
    assert "'type_id'" in message


def test_to_networkx_identity_attrs_win_over_same_named_properties():
    """A property literally named `node_type` (or `title`) must not shadow
    the real identity attribute — the importer reads those back."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame([{"id": 1, "name": "Alice", "node_type": "IMPOSTOR", "title": "FAKE"}]),
        "Person",
        "id",
        "name",
    )
    for key in ("id", "type_id"):
        nxg = g.to_networkx(node_key=key)
        node = nxg.nodes[1 if key == "id" else ("Person", 1)]
        assert node["node_type"] == "Person"
        assert node["title"] == "Alice"
