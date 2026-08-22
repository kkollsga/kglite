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
    """Two types whose ids overlap on 5, with one edge each. Both node types
    and one edge type carry a property, so a round trip has something to lose
    beyond the identity columns."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame([{"id": 5, "name": "Alice", "age": 30}, {"id": 6, "name": "Bob", "age": 25}]),
        "Person",
        "id",
        "name",
    )
    g.add_nodes(pd.DataFrame([{"id": 5, "name": "Oslo", "pop": 700000}]), "City", "id", "name")
    g.add_connections(
        pd.DataFrame([{"src": 6, "tgt": 5, "since": 2010}]),
        "KNOWS",
        "Person",
        "src",
        "Person",
        "tgt",
        columns=["since"],
    )
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


# ── from_networkx: tuple-key round trip + no more warn-and-drop-to-empty ──
#
# `to_networkx(node_key="type_id")` emits (node_type, id) tuple keys. The
# importer auto-detects that shape — a foreign tuple-labelled graph cannot
# masquerade as one, because the detection requires each key's first element
# to equal the node's own `node_type` attribute, which only the export writes.
# Anything else the id column cannot store now RAISES up front instead of
# warn-and-dropping every row into a silently empty graph.


def test_type_id_export_round_trips_through_from_networkx():
    """THE round trip: a colliding-id graph survives export and re-import."""
    g = _colliding_id_graph()
    g2 = kglite.from_networkx(g.to_networkx(node_key="type_id"))

    rt = g2.to_networkx(node_key="type_id")
    assert rt.number_of_nodes() == 3
    assert rt.number_of_edges() == 2
    assert set(rt.nodes) == {("Person", 5), ("Person", 6), ("City", 5)}
    assert rt.nodes[("Person", 5)]["title"] == "Alice"
    assert rt.nodes[("City", 5)]["title"] == "Oslo"
    assert rt.has_edge(("Person", 6), ("Person", 5), "KNOWS")
    assert rt.has_edge(("Person", 6), ("City", 5), "LIVES_IN")
    assert not rt.has_edge(("Person", 6), ("Person", 5), "LIVES_IN")


def test_type_id_round_trip_preserves_node_properties():
    g = _colliding_id_graph()
    g2 = kglite.from_networkx(g.to_networkx(node_key="type_id"))
    rt = g2.to_networkx(node_key="type_id")
    assert rt.nodes[("Person", 5)]["age"] == 30
    assert rt.nodes[("Person", 6)]["age"] == 25
    assert rt.nodes[("City", 5)]["pop"] == 700000
    # The tuple key itself must not leak into the imported node.
    assert "node_type" in rt.nodes[("City", 5)]
    assert rt.nodes[("City", 5)]["node_type"] == "City"


def test_type_id_round_trip_preserves_edge_properties():
    g = _colliding_id_graph()
    g2 = kglite.from_networkx(g.to_networkx(node_key="type_id"))
    rt = g2.to_networkx(node_key="type_id")
    assert rt[("Person", 6)][("Person", 5)]["KNOWS"]["since"] == 2010
    assert rt[("Person", 6)][("City", 5)]["LIVES_IN"]["connection_type"] == "LIVES_IN"


def test_type_id_round_trip_preserves_same_type_parallel_edges():
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (a:N {id:1}), (b:N {id:2}) CREATE (a)-[:R {rank:1}]->(b), (a)-[:R {rank:2}]->(b)")
    g2 = kglite.from_networkx(g.to_networkx(node_key="type_id"))
    rt = g2.to_networkx(node_key="type_id")
    assert rt.number_of_nodes() == 2
    assert rt.number_of_edges(("N", 1), ("N", 2)) == 2
    assert {edge["rank"] for edge in rt[("N", 1)][("N", 2)].values()} == {1, 2}


def test_from_networkx_mixed_key_shapes_raise():
    """One tuple-keyed node and one plain-keyed node is not a shape the
    importer can honour for both — a partial import is the disease."""
    nxg = nx.MultiDiGraph()
    nxg.add_node(("Person", 5), node_type="Person", title="Alice")
    nxg.add_node(7, node_type="Person", title="Bob")
    with pytest.raises(kglite.ArgumentError) as excinfo:
        kglite.from_networkx(nxg)
    message = str(excinfo.value)
    assert "from_networkx()" in message
    assert "mixes node-key shapes" in message
    assert "7" in message


def test_from_networkx_tuple_key_disagreeing_with_node_type_attr_raises():
    """A 2-tuple key whose first element is not the node's own node_type is
    not an export key; alongside a real one it is a shape conflict, and the
    message must say which half disagreed."""
    nxg = nx.MultiDiGraph()
    nxg.add_node(("Person", 5), node_type="Person", title="Alice")
    nxg.add_node(("Person", 9), node_type="City", title="Oslo")
    with pytest.raises(kglite.ArgumentError) as excinfo:
        kglite.from_networkx(nxg)
    message = str(excinfo.value)
    assert "node_type" in message
    assert "City" in message


def test_from_networkx_grid_graph_raises_unrepresentable_id():
    """nx.grid_2d_graph nodes are (x, y) int tuples with no attributes, so
    they can never be mistaken for a type_id export. They are also not
    storable as ids — that used to yield a silently empty graph."""
    nxg = nx.grid_2d_graph(3, 3)
    with pytest.raises(kglite.ArgumentError) as excinfo:
        kglite.from_networkx(nxg)
    message = str(excinfo.value)
    assert "from_networkx()" in message
    assert "9 of 9" in message  # every node, named as a count
    assert "tuple" in message
    assert "integer or a string" in message
    assert "node_type" in message  # why these tuples are not export keys
    assert "relabel" in message.lower()  # the fix


def test_from_networkx_three_tuple_key_raises():
    """A 3-tuple is not the exported shape even when it starts with the
    node_type, so it takes the plain-id path and is refused there."""
    nxg = nx.MultiDiGraph()
    nxg.add_node(("Person", 5, "extra"), node_type="Person")
    with pytest.raises(kglite.ArgumentError, match="integer or a string"):
        kglite.from_networkx(nxg)


def test_from_networkx_tuple_key_without_node_type_attr_raises():
    nxg = nx.MultiDiGraph()
    nxg.add_node(("Person", 5), title="Alice")
    with pytest.raises(kglite.ArgumentError, match="integer or a string"):
        kglite.from_networkx(nxg)


def test_from_networkx_fractional_float_key_raises_naming_the_count():
    """A fractional float is the quiet half of the shrinkage class: node 1
    imports, node 1.5 drops, and the caller is handed a smaller graph."""
    nxg = nx.MultiDiGraph()
    nxg.add_node(1)
    nxg.add_node(1.5)
    with pytest.raises(kglite.ArgumentError) as excinfo:
        kglite.from_networkx(nxg)
    message = str(excinfo.value)
    assert "1 of 2" in message
    assert "1.5" in message
    assert "float" in message


def test_from_networkx_non_scalar_key_raises():
    """Any nx-legal label the id column cannot store is refused, not just
    tuples — here a frozenset."""
    nxg = nx.MultiDiGraph()
    nxg.add_node(frozenset({"a"}))
    with pytest.raises(kglite.ArgumentError, match="frozenset"):
        kglite.from_networkx(nxg)


@pytest.mark.parametrize("keys", [(1, 2), ("a", "b"), (1.0, 2.0)])
def test_from_networkx_representable_keys_still_import(keys):
    """The refusal is scoped to what the id column genuinely cannot store:
    ints, strings and whole floats keep importing exactly as before."""
    src, tgt = keys
    nxg = nx.MultiDiGraph()
    nxg.add_edge(src, tgt)
    assert kglite.from_networkx(nxg).to_networkx().number_of_nodes() == 2


def test_from_networkx_type_id_key_with_unstorable_id_half_raises():
    """The detected path is not a bypass: a tuple key whose *id* half cannot
    be stored is refused there too, instead of dropping the row."""
    nxg = nx.MultiDiGraph()
    nxg.add_node(("Person", 5), node_type="Person")
    nxg.add_node(("Person", (1, 2)), node_type="Person")
    with pytest.raises(kglite.ArgumentError) as excinfo:
        kglite.from_networkx(nxg)
    message = str(excinfo.value)
    assert "1 of 2" in message
    assert "id half" in message
    assert "('Person', (1, 2))" in message


def test_from_networkx_mixed_int_and_str_keys_raise():
    """The keys are individually storable; the *combination* is not. Pandas
    types the column as object, add_nodes writes it as text, so key 1 becomes
    id "1" while the edge endpoint stays int 1, misses, and vivifies a stub —
    a 2-node graph imported as 3."""
    nxg = nx.MultiDiGraph()
    nxg.add_edge(1, "b")
    with pytest.raises(kglite.ArgumentError) as excinfo:
        kglite.from_networkx(nxg)
    message = str(excinfo.value)
    assert "from_networkx()" in message
    assert "mixes node-id types" in message
    assert "1 integer key" in message  # both shapes counted
    assert "1 string key" in message
    assert "'b'" in message
    assert "relabel" in message.lower()  # the fix


def test_from_networkx_mixed_id_families_count_every_node():
    """The counts are of the whole graph, not of the first disagreement."""
    nxg = nx.MultiDiGraph()
    nxg.add_nodes_from([1, 2, 3, "a", "b"])
    with pytest.raises(kglite.ArgumentError) as excinfo:
        kglite.from_networkx(nxg)
    message = str(excinfo.value)
    assert "3 integer keys" in message
    assert "2 string keys" in message


def test_from_networkx_three_id_families_are_all_listed():
    """All three families in one type are named, not just the first two."""
    nxg = nx.MultiDiGraph()
    nxg.add_nodes_from([2, "a", True])  # True == 1 in nx, so avoid key 1
    with pytest.raises(kglite.ArgumentError) as excinfo:
        kglite.from_networkx(nxg)
    message = str(excinfo.value)
    assert "1 integer key (e.g. 2), 1 string key (e.g. 'a') and 1 boolean key (e.g. True)" in message


def test_from_networkx_type_id_mixed_id_halves_raise():
    """The tuple path is not a bypass: two nodes of one type whose id halves
    disagree land in the same id column and hit the same coercion — 2 nodes
    imported as 3."""
    nxg = nx.MultiDiGraph()
    nxg.add_node(("A", 1), node_type="A")
    nxg.add_node(("A", "x"), node_type="A")
    nxg.add_edge(("A", 1), ("A", "x"))
    with pytest.raises(kglite.ArgumentError) as excinfo:
        kglite.from_networkx(nxg)
    message = str(excinfo.value)
    assert "mixes node-id types" in message
    assert "id half" in message
    assert "('A', 1)" in message
    assert "('A', 'x')" in message


def test_from_networkx_id_families_may_differ_between_node_types():
    """The refusal is per node type, because the coercion is: each type is
    loaded as its own DataFrame, so int-keyed People and string-keyed Cities
    never share a column and import exactly right. Refusing them would break
    a graph that works today."""
    nxg = nx.MultiDiGraph()
    nxg.add_node(1, node_type="Person")
    nxg.add_node("oslo", node_type="City")
    nxg.add_edge(1, "oslo")
    g = kglite.from_networkx(nxg)
    assert g.len() == 2
    assert g.select("Person").ids() == [1]
    assert g.select("City").ids() == ["oslo"]


def test_from_networkx_type_id_families_may_differ_between_node_types():
    """Same, on the tuple path: ('A', 1) and ('B', 'x') are different types,
    hence different columns, hence a clean import."""
    nxg = nx.MultiDiGraph()
    nxg.add_node(("A", 1), node_type="A")
    nxg.add_node(("B", "x"), node_type="B")
    nxg.add_edge(("A", 1), ("B", "x"))
    g = kglite.from_networkx(nxg)
    assert g.len() == 2
    assert g.select("A").ids() == [1]
    assert g.select("B").ids() == ["x"]


def test_from_networkx_bool_key_with_int_key_raises():
    """`bool` is an `int` subclass in Python but not in a pandas column:
    [True, 2] types as object, both endpoints stringify, and a 2-node graph
    imports as 4. So booleans are their own id family here."""
    nxg = nx.MultiDiGraph()
    nxg.add_edge(True, 2)
    with pytest.raises(kglite.ArgumentError) as excinfo:
        kglite.from_networkx(nxg)
    message = str(excinfo.value)
    assert "1 boolean key" in message
    assert "1 integer key" in message
    assert "relabel" in message.lower()


def test_from_networkx_all_bool_keys_still_import():
    """A single family is always allowed, booleans included: [True, False]
    types as a bool column and stores ids 1 and 0."""
    nxg = nx.MultiDiGraph()
    nxg.add_edge(True, False)
    assert kglite.from_networkx(nxg).len() == 2


def test_from_networkx_bytes_keys_are_the_string_family():
    """bytes stringify into a text column, so they sit with strings (which
    imports cleanly) and not with integers (which does not)."""
    with_str = nx.MultiDiGraph()
    with_str.add_edge(b"a", "b")
    assert kglite.from_networkx(with_str).len() == 2

    with_int = nx.MultiDiGraph()
    with_int.add_edge(b"a", 2)
    with pytest.raises(kglite.ArgumentError, match="mixes node-id types"):
        kglite.from_networkx(with_int)


def test_from_networkx_int_and_whole_float_keys_are_one_family():
    """A whole float is already normalised to an integer id by the
    representability rule, and a mixed int/float column types as float64 —
    so this pair is one family and must keep importing."""
    nxg = nx.MultiDiGraph()
    nxg.add_edge(1, 2.0)
    g = kglite.from_networkx(nxg)
    assert g.len() == 2
    assert sorted(g.select("Node").ids()) == [1, 2]
