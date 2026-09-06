import pytest

import kglite


def current(g):
    return g.cypher("CALL db.cdc.current()").to_list()[0]["id"]


def events(g, cursor):
    return g.cypher("CALL db.cdc.query({from:$c})", params={"c": cursor}).to_list()


def test_cdc_only_session_refuses_writes_without_ghost_source_events():
    g = kglite.KnowledgeGraph()
    g.cypher("CALL db.cdc.enable()")
    start = current(g)
    s = g.session()
    with pytest.raises(kglite.ArgumentError, match="change-data capture"):
        s.execute("CREATE (:N {id:1})")
    assert events(g, start) == events(s, start) == []
    for view in (g, s):
        assert view.cypher("MATCH(n) RETURN count(*) AS n").scalar() == 0


def test_retained_durable_session_with_cdc_stays_read_only_after_close(tmp_path):
    path = str(tmp_path / "home.kgl")
    g = kglite.open(path, durable="full")
    g.cypher("CALL db.cdc.enable()")
    g.cypher("CREATE (:N {id:1})")
    start = current(g)
    s = g.session()
    g.close()
    with pytest.raises(kglite.ArgumentError, match="change-data capture"):
        s.execute("CREATE (:N {id:2})")
    assert events(g, start) == events(s, start) == []
    with kglite.open(path, durable="full") as fresh:
        assert fresh.cypher("MATCH(n) RETURN n.id AS id").to_list() == [{"id": 1}]


def test_retained_durable_session_without_cdc_writes_only_detached_data(tmp_path):
    path = str(tmp_path / "home.kgl")
    g = kglite.open(path, durable="full")
    g.cypher("CREATE (:N {id:1})")
    s = g.session()
    g.close()
    with kglite.open(path, durable="full") as fresh:
        s.execute("CREATE (:N {id:2})")
        fresh.cypher("CREATE (:N {id:3})")
    assert s.cypher("MATCH(n) RETURN n.id AS id ORDER BY id").to_list() == [{"id": 1}, {"id": 2}]
    with kglite.open(path, durable="full") as fresh:
        assert fresh.cypher("MATCH(n) RETURN n.id AS id ORDER BY id").to_list() == [{"id": 1}, {"id": 3}]


def test_close_keeps_existing_cdc_cursor_and_publishes_one_detached_write(tmp_path):
    g = kglite.open(str(tmp_path / "home.kgl"), durable="full")
    g.cypher("CALL db.cdc.enable()")
    g.cypher("CREATE (:N {id:1})")
    start = current(g)
    g.close()
    assert current(g) == start
    g.cypher("CREATE (:N {id:2})")
    assert [event["nodeId"] for event in events(g, start)] == [2]


@pytest.mark.parametrize("route", ("cursor", "fluent", "transaction"))
def test_session_cursor_cannot_publish_into_source_cdc_stream(route):
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:N {id:0})")
    g.cypher("CALL db.cdc.enable()")
    start = current(g)
    s = g.session()
    cursor = s.cursor()
    target = cursor.select("N") if route == "fluent" else cursor
    if route == "transaction":
        tx = target.begin()
        tx.cypher("CREATE (:N {id:1})")
        with pytest.raises(kglite.ArgumentError, match="change-data capture"):
            tx.commit()
    else:
        with pytest.raises(kglite.ArgumentError, match="change-data capture"):
            target.cypher("CREATE (:N {id:1})")
    for view in (g, s, cursor, target):
        assert view.cypher("MATCH(n) RETURN n.id AS id").to_list() == [{"id": 0}]
        assert events(view, start) == []
    independent = cursor.copy()
    independent.cypher("CREATE (:N {id:2})")
    assert g.cypher("MATCH(n) RETURN count(*) AS n").scalar() == 1
    assert events(g, start) == []


@pytest.mark.parametrize("route", ("select", "set_operation"))
def test_cdc_owning_graph_derivatives_cannot_publish_source_events(route):
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:N {id:0})")
    g.cypher("CALL db.cdc.enable()")
    start = current(g)
    derived = g.select("N")
    if route == "set_operation":
        derived = derived.union(g.select("N"))
    with pytest.raises(kglite.ArgumentError, match="change-data capture"):
        derived.cypher("CREATE (:N {id:1})")
    assert g.cypher("MATCH(n) RETURN n.id AS id").to_list() == [{"id": 0}]
    assert derived.cypher("MATCH(n) RETURN n.id AS id").to_list() == [{"id": 0}]
    assert events(g, start) == events(derived, start) == []


@pytest.mark.parametrize("action", ("add_label", "remove_label", "add_nodes"))
def test_cdc_cursor_direct_mutation_refuses_before_changing_retained_data(action):
    import pandas as pd

    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:N:Marked {id:0})")
    g.cypher("CALL db.cdc.enable()")
    start = current(g)
    h = g.session().cursor()
    query = "MATCH(n) RETURN n.id AS id, labels(n) AS labels ORDER BY id"
    before = h.cypher(query).to_list()
    with pytest.raises(kglite.ArgumentError, match="change-data capture"):
        if action == "add_label":
            h.add_label("N", [0], "Added")
        elif action == "remove_label":
            h.remove_label("N", [0], "Marked")
        else:
            h.add_nodes(pd.DataFrame({"id": [1]}), "N", "id")
    assert h.cypher(query).to_list() == before
    assert g.cypher(query).to_list() == before
    assert events(g, start) == events(h, start) == []
