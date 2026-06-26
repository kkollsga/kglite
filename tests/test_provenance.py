"""Freshness provenance — opt-in `auto_timestamp` stamping of `updated_at`.

`define_schema({"nodes": {"T": {"auto_timestamp": True}}})` makes the engine
stamp a reserved `updated_at` (a Timestamp) on every write to that type —
Cypher CREATE/MERGE/SET and `add_nodes` — never user-supplied. Off by default,
so writes stay deterministic. Queryable via `n.updated_at` for stale checks.
(Hiding from data views + edges + git_sha land in later phases.)
"""

import time

import pandas as pd

import kglite


def _opted_graph():
    g = kglite.KnowledgeGraph()
    g.define_schema({"nodes": {"Task": {"auto_timestamp": True}, "Other": {}}})
    return g


def _updated_at(g, t, i):
    return g.cypher(f"MATCH (n:{t} {{id:{i}}}) RETURN n.updated_at AS u").to_dicts()[0]["u"]


def test_cypher_create_stamps_opted_in_type():
    g = _opted_graph()
    g.cypher("CREATE (:Task {id: 1})")
    assert _updated_at(g, "Task", 1) is not None


def test_cypher_create_does_not_stamp_off_type():
    g = _opted_graph()
    g.cypher("CREATE (:Other {id: 1})")
    assert _updated_at(g, "Other", 1) is None


def test_no_schema_means_no_stamp_determinism():
    g = kglite.KnowledgeGraph()  # nothing opted in
    g.cypher("CREATE (:Task {id: 1})")
    assert _updated_at(g, "Task", 1) is None


def test_merge_create_stamps():
    g = _opted_graph()
    g.cypher("MERGE (:Task {id: 7})")
    assert _updated_at(g, "Task", 7) is not None


def test_add_nodes_stamps_create_and_update():
    g = _opted_graph()
    g.add_nodes(pd.DataFrame([{"id": 1}, {"id": 2}]), node_type="Task", unique_id_field="id")
    first = _updated_at(g, "Task", 1)
    assert first is not None
    time.sleep(0.01)
    g.add_nodes(
        pd.DataFrame([{"id": 1, "x": 9}]),
        node_type="Task",
        unique_id_field="id",
        conflict_handling="update",
    )
    assert _updated_at(g, "Task", 1) > first  # update advanced it


def test_add_nodes_off_type_not_stamped():
    g = _opted_graph()
    g.add_nodes(pd.DataFrame([{"id": 1}]), node_type="Other", unique_id_field="id")
    assert _updated_at(g, "Other", 1) is None


def test_set_bumps_once_and_advances():
    g = _opted_graph()
    g.cypher("CREATE (:Task {id: 1, a: 1, b: 2})")
    created = _updated_at(g, "Task", 1)
    time.sleep(0.01)
    g.cypher("MATCH (t:Task {id: 1}) SET t.a = 10, t.b = 20")  # two props, one node
    assert _updated_at(g, "Task", 1) > created


def test_set_off_type_not_stamped():
    g = _opted_graph()
    g.cypher("CREATE (:Other {id: 1})")
    g.cypher("MATCH (o:Other {id: 1}) SET o.a = 5")
    assert _updated_at(g, "Other", 1) is None


def test_updated_at_is_queryable_for_stale_checks():
    g = _opted_graph()
    g.cypher("CREATE (:Task {id: 1})")
    rows = g.cypher("MATCH (t:Task) WHERE t.updated_at > '2020-01-01T00:00:00' RETURN count(t) AS c").to_dicts()
    assert rows[0]["c"] == 1


def test_user_supplied_updated_at_is_overwritten_by_engine():
    g = _opted_graph()
    # Engine owns the reserved key — a user value is replaced by the stamp.
    g.cypher("CREATE (:Task {id: 1, updated_at: 'nonsense'})")
    assert _updated_at(g, "Task", 1) != "nonsense"


def test_reserved_key_hidden_from_data_views_but_directly_queryable():
    g = _opted_graph()
    g.cypher("CREATE (:Task {id: 1, name: 'x'})")
    # Hidden from every property enumeration...
    assert "updated_at" not in g.cypher("MATCH (n:Task) RETURN keys(n) AS k").to_dicts()[0]["k"]
    assert "updated_at" not in g.cypher("MATCH (n:Task) RETURN properties(n) AS p").to_dicts()[0]["p"]
    assert "updated_at" not in g.cypher("MATCH (n:Task) RETURN n {.*} AS m").to_dicts()[0]["m"]
    assert "updated_at" not in g.cypher("MATCH (n:Task) RETURN n").to_dicts()[0]["n"]["properties"]
    assert "updated_at" not in g.describe()
    # ...but explicitly accessible for stale checks.
    assert _updated_at(g, "Task", 1) is not None
    assert "updated_at" in g.cypher("MATCH (n:Task) RETURN n {.updated_at} AS m").to_dicts()[0]["m"]


# --- edges / connections (phase 4) -------------------------------------------


def _edge_graph():
    g = kglite.KnowledgeGraph()
    g.define_schema(
        {
            "connections": {
                "LINKS": {"source": "N", "target": "N", "auto_timestamp": True},
                "PLAIN": {"source": "N", "target": "N"},
            }
        }
    )
    g.cypher("CREATE (a:N {id: 1}), (b:N {id: 2})")
    return g


def _edge_updated_at(g, rel):
    return g.cypher(f"MATCH ()-[r:{rel}]->() RETURN r.updated_at AS u").to_dicts()[0]["u"]


def test_cypher_edge_create_stamps_opted_in_only():
    g = _edge_graph()
    g.cypher("MATCH (a:N {id: 1}), (b:N {id: 2}) CREATE (a)-[:LINKS]->(b)")
    g.cypher("MATCH (a:N {id: 1}), (b:N {id: 2}) CREATE (a)-[:PLAIN]->(b)")
    assert _edge_updated_at(g, "LINKS") is not None
    assert _edge_updated_at(g, "PLAIN") is None


def test_add_connections_edge_stamps():
    g = _edge_graph()
    g.add_connections(pd.DataFrame([{"s": 1, "t": 2}]), "LINKS", "N", "s", "N", "t")
    assert _edge_updated_at(g, "LINKS") is not None


def test_edge_set_bumps_and_advances():
    g = _edge_graph()
    g.cypher("MATCH (a:N {id: 1}), (b:N {id: 2}) CREATE (a)-[:LINKS {w: 1}]->(b)")
    created = _edge_updated_at(g, "LINKS")
    time.sleep(0.01)
    g.cypher("MATCH ()-[r:LINKS]->() SET r.w = 5")
    assert _edge_updated_at(g, "LINKS") > created


def test_edge_reserved_key_hidden_but_queryable():
    g = _edge_graph()
    g.cypher("MATCH (a:N {id: 1}), (b:N {id: 2}) CREATE (a)-[:LINKS {w: 1}]->(b)")
    assert "updated_at" not in g.cypher("MATCH ()-[r:LINKS]->() RETURN keys(r) AS k").to_dicts()[0]["k"]
    assert "updated_at" not in g.cypher("MATCH ()-[r:LINKS]->() RETURN properties(r) AS p").to_dicts()[0]["p"]
    assert "updated_at" not in g.cypher("MATCH ()-[r:LINKS]->() RETURN r").to_dicts()[0]["r"]["properties"]
    assert "updated_at" not in g.describe()
    assert _edge_updated_at(g, "LINKS") is not None  # direct access works


def test_connection_auto_timestamp_roundtrips():
    g = _edge_graph()
    sd = g.schema_definition()
    assert sd["connections"]["LINKS"]["auto_timestamp"] is True
    assert "auto_timestamp" not in sd["connections"]["PLAIN"]


def test_survives_save_load(tmp_path):
    g = _opted_graph()
    g.cypher("CREATE (:Task {id: 1})")
    stamped = _updated_at(g, "Task", 1)
    p = str(tmp_path / "g.kgl")
    g.save(p)
    assert _updated_at(kglite.load(p), "Task", 1) == stamped
