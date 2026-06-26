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


def test_survives_save_load(tmp_path):
    g = _opted_graph()
    g.cypher("CREATE (:Task {id: 1})")
    stamped = _updated_at(g, "Task", 1)
    p = str(tmp_path / "g.kgl")
    g.save(p)
    assert _updated_at(kglite.load(p), "Task", 1) == stamped
