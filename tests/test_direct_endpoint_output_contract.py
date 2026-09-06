"""Direct graph projections preserve admitted endpoint-title snapshots."""

import pytest

import kglite


@pytest.fixture
def stored_endpoints():
    graph = kglite.KnowledgeGraph()
    graph.cypher(
        "CREATE (a:Item {id:1,title:'Alpha',payload:[0],nested:{pair:[0]}}),"
        "(b:Item {id:2,title:'Beta',payload:[0],nested:{pair:[0]}}),"
        "(a)-[:LINK {payload:[0]}]->(b)"
    )
    graph.cypher(
        "MATCH (a:Item)-[r:LINK]->(b:Item) "
        "SET a.payload=[startNode(r)], a.nested={pair:[startNode(r),endNode(r)]},"
        "b.payload=[endNode(r)], b.nested={pair:[endNode(r),startNode(r)]},"
        "r.payload=[startNode(r),endNode(r)]"
    )
    return graph


def assert_item(row, node_id):
    title, peer = ("Alpha", "Beta") if node_id == 1 else ("Beta", "Alpha")
    assert row == {
        "type": "Item",
        "id": node_id,
        "title": title,
        "payload": [title],
        "nested": {"pair": [title, peer]},
    }
    assert type(row["id"]) is int


@pytest.mark.parametrize("restored", [False, True], ids=["live", "bytes_roundtrip"])
@pytest.mark.parametrize("consumer", ["collect", "sample", "frame", "grouped", "node"])
def test_direct_full_node_consumers_resolve_stored_endpoints(stored_endpoints, restored, consumer):
    graph = kglite.from_bytes(stored_endpoints.to_bytes()) if restored else stored_endpoints
    if consumer == "node":
        rows = [graph.node("Item", node_id) for node_id in [1, 2]]
    elif consumer == "sample":
        rows = graph.sample("Item", n=2).to_list()
    elif consumer == "collect":
        rows = graph.select("Item").collect().to_list()
    elif consumer == "grouped":
        rows = graph.select("Item").collect_grouped("title")
    else:
        rows = graph.select("Item").to_df().to_dict("records")
    assert len(rows) == 2
    rows = sorted(rows, key=lambda row: row["id"])
    for row, node_id in zip(rows, [1, 2], strict=True):
        assert_item(row, node_id)
    # The stored property is a title snapshot; the relationship endpoint keeps
    # query-time identity independently.
    assert graph.cypher("MATCH (a:Item)-[r:LINK]->() RETURN a.payload[0] = startNode(r) AS same").scalar() is False
    assert graph.cypher("MATCH ()-[r:LINK]->() RETURN startNode(r) = startNode(r) AS same").scalar() is True


def test_direct_property_and_sample_outputs_resolve_after_retrieval(stored_endpoints):
    graph = stored_endpoints
    assert graph.select("Item").get_properties(["payload", "nested"]) == [
        (["Alpha"], {"pair": ["Alpha", "Beta"]}),
        (["Beta"], {"pair": ["Beta", "Alpha"]}),
    ]
    unique = graph.select("Item").unique_values("payload")
    assert list(unique) == ["Root"]
    assert sorted(unique["Root"]) == [["Alpha"], ["Beta"]]
    stats = graph.properties("Item")
    assert sorted(stats["payload"]["values"]) == [["Alpha"], ["Beta"]]
    assert stats["payload"]["unique"] == 2


def test_connection_property_payloads_resolve_without_changing_keys_or_ids(stored_endpoints):
    connections = stored_endpoints.select("Item").connections(include_node_properties=True)
    outgoing = connections["Alpha"]["outgoing"]["LINK"]["Beta"]
    incoming = connections["Beta"]["incoming"]["LINK"]["Alpha"]
    assert outgoing["node_id"] == 2
    assert incoming["node_id"] == 1
    assert outgoing["connection_properties"] == {"payload": ["Alpha", "Beta"]}
    assert incoming["connection_properties"] == {"payload": ["Alpha", "Beta"]}
    assert outgoing["node_properties"]["payload"] == ["Beta"]
    assert incoming["node_properties"]["payload"] == ["Alpha"]
    assert outgoing["node_properties"]["nested"] == {"pair": ["Beta", "Alpha"]}


def test_direct_projection_without_endpoints_retains_exact_types():
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:Item {id:1,title:'Plain',payload:[1,true,null],nested:{x:'1'}})")
    expected = {"type": "Item", "id": 1, "title": "Plain", "payload": [1, True, None], "nested": {"x": "1"}}
    for row in [graph.node("Item", 1), graph.select("Item").collect().to_list()[0]]:
        assert row == expected
        assert [type(value) for value in row["payload"]] == [int, bool, type(None)]


def test_find_context_and_ambiguity_materialise_whole_nodeinfo(stored_endpoints):
    graph = stored_endpoints
    found = graph.find("Alpha", node_type="Item")
    assert len(found) == 1
    assert_item(found[0], 1)
    context = graph.context("Alpha", node_type="Item")
    assert_item(context["node"], 1)
    assert len(context["LINK"]) == 1
    assert_item(context["LINK"][0], 2)
    graph.cypher("MATCH (a:Item {id:1}) CREATE (:Item {id:3,title:'Alpha',payload:a.payload,nested:a.nested})")
    for result in [graph.context("Alpha", node_type="Item"), graph.source("Alpha", node_type="Item")]:
        assert result["ambiguous"] is True
        matches = sorted(result["matches"], key=lambda row: row["id"])
        assert [row["id"] for row in matches] == [1, 3]
        assert [row["payload"] for row in matches] == [["Alpha"], ["Alpha"]]
        assert [row["nested"] for row in matches] == [{"pair": ["Alpha", "Beta"]}] * 2


def test_materialised_direct_result_keeps_admission_titles(stored_endpoints):
    graph = stored_endpoints
    earlier = graph.select("Item").collect()
    graph.cypher("MATCH (a:Item {id:1}) SET a.title='Changed'")
    old = sorted(earlier.to_list(), key=lambda row: row["id"])
    assert_item(old[0], 1)
    assert_item(old[1], 2)
    assert graph.node("Item", 1)["payload"] == ["Alpha"]
    assert graph.node("Item", 2)["nested"] == {"pair": ["Beta", "Alpha"]}


def test_projection_deduplicates_equal_stored_title_snapshots():
    graph = kglite.KnowledgeGraph()
    graph.cypher(
        "CREATE (a:Item {id:1,title:'Same',payload:[0]}),(b:Item {id:2,title:'Same',payload:[0]}),(a)-[:LINK]->(b)"
    )
    graph.cypher("MATCH (a:Item)-[r:LINK]->(b:Item) SET a.payload=[startNode(r)],b.payload=[endNode(r)]")
    assert graph.select("Item").unique_values("payload") == {"Root": [["Same"]]}
    stats = graph.properties("Item")["payload"]
    assert stats["unique"] == 1
    assert stats["values"] == [["Same"]]


@pytest.mark.parametrize("mode", ["memory", "mapped", "disk"])
def test_stored_snapshot_does_not_follow_current_backend_titles(tmp_path, mode):
    graph = kglite.KnowledgeGraph(storage=mode, path=str(tmp_path / "disk") if mode == "disk" else None)
    graph.cypher(
        "CREATE (a:Item {id:1,title:'Alpha',payload:[0]}),(b:Item {id:2,title:'Beta',payload:[0]}),(a)-[:LINK]->(b)"
    )
    if mode == "disk":
        graph.save(str(tmp_path / "disk"))
    graph.cypher("MATCH (a:Item)-[r:LINK]->() SET a.payload=[endNode(r)]")
    earlier = graph.select("Item").collect()
    graph.cypher("MATCH(b:Item {id:2}) SET b.title='Updated'")
    assert graph.cypher("MATCH(a:Item {id:1}) RETURN a.payload AS value").scalar() == ["Beta"]
    rows = graph.select("Item").collect().to_list()
    assert next(row for row in rows if row["id"] == 1)["payload"] == ["Beta"]
    assert next(row for row in earlier.to_list() if row["id"] == 1)["payload"] == ["Beta"]
