"""`db.cdc.*` — the change stream, reached the way every binding reaches it.

Cypher-first: the wheel exposes no CDC method, so everything here goes through
``kg.cypher(...)``. What these tests are actually proving is that the *binding*
publishes at its commit boundaries — the engine tests in ``graph::cdc`` cover
the log itself, and they can drive ``drain_at_commit`` by hand. Python cannot,
so a missing drain in ``flush_wal`` shows up only here, as a stream that stays
empty while the graph changes.

The three properties worth naming:

- **No phantom events.** A rolled-back transaction, and a statement that
  failed, must leave nothing in the stream.
- **Exactly once under copy-on-write.** A held ``ResultView`` forces the next
  write to fork; the fork shares the log through its ``Arc``, so the commit
  must publish once — not twice, not zero times.
- **A cursor addresses one log.** Retention eviction, a reload, and a
  re-``enable`` each make an old cursor unusable, and each says so distinctly.
"""

from __future__ import annotations

import pytest

import kglite


def enabled_graph(capacity: int | None = None, storage: str | None = None, path: str | None = None):
    """A graph with capture running."""
    kwargs: dict = {}
    if storage:
        kwargs["storage"] = storage
    if path:
        kwargs["path"] = path
    g = kglite.KnowledgeGraph(**kwargs)
    arg = f"{{capacity: {capacity}}}" if capacity is not None else ""
    g.cypher(f"CALL db.cdc.enable({arg})")
    return g


def events(graph, cursor: str | None = None) -> list[dict]:
    if cursor is None:
        return graph.cypher("CALL db.cdc.query()").to_dicts()
    return graph.cypher("CALL db.cdc.query({from: $c})", params={"c": cursor}).to_dicts()


def cursor(graph, which: str = "current") -> str:
    return graph.cypher(f"CALL db.cdc.{which}()").to_dicts()[0]["id"]


def summary(rows) -> list[tuple]:
    """(operation, elementType, identity) per row — the assertion shape."""
    return [
        (
            r["operation"],
            r["elementType"],
            r["nodeId"] if r["elementType"] == "node" else (r["relationshipType"], r["srcId"], r["tgtId"]),
        )
        for r in rows
    ]


@pytest.fixture
def graph():
    return enabled_graph()


# ── lifecycle ──────────────────────────────────────────────────────────────


def test_enable_reports_status_and_a_starting_cursor():
    g = kglite.KnowledgeGraph()
    rows = g.cypher("CALL db.cdc.enable()").to_dicts()
    assert len(rows) == 1
    assert rows[0]["enabled"] is True
    assert rows[0]["epoch"] > 0
    assert rows[0]["capacity"] == 65536
    assert rows[0]["cursor"].startswith("cdc:")


def test_reading_before_enable_says_capture_is_off():
    g = kglite.KnowledgeGraph()
    for proc in ("current", "earliest", "query"):
        with pytest.raises(kglite.CypherExecutionError, match="not enabled on this graph"):
            g.cypher(f"CALL db.cdc.{proc}()")


def test_disable_reports_whether_it_was_running_and_is_idempotent(graph):
    first = graph.cypher("CALL db.cdc.disable()").to_dicts()[0]
    assert first == {"enabled": False, "wasEnabled": True}
    again = graph.cypher("CALL db.cdc.disable()").to_dicts()[0]
    assert again == {"enabled": False, "wasEnabled": False}


def test_writes_while_disabled_are_not_recovered_by_re_enabling(graph):
    graph.cypher("CALL db.cdc.disable()")
    graph.cypher("CREATE (:P {id: 1})")
    reenabled = graph.cypher("CALL db.cdc.enable()").to_dicts()[0]
    assert reenabled["epoch"] > 1, "a fresh enable mints a new epoch"
    assert events(graph) == [], "capture is opt-in: nothing before enable is in the log"


def test_capacity_is_configurable_and_resizes_in_place():
    g = kglite.KnowledgeGraph()
    first = g.cypher("CALL db.cdc.enable({capacity: 16})").to_dicts()[0]
    assert first["capacity"] == 16
    again = g.cypher("CALL db.cdc.enable({capacity: 4})").to_dicts()[0]
    assert again["capacity"] == 4
    assert again["epoch"] == first["epoch"], "a resize keeps live cursors valid"


def test_capacity_and_parameter_names_are_validated():
    g = kglite.KnowledgeGraph()
    with pytest.raises(kglite.CypherExecutionError, match="positive integer"):
        g.cypher("CALL db.cdc.enable({capacity: 0})")
    with pytest.raises(kglite.CypherExecutionError, match="unknown parameter 'cap'"):
        g.cypher("CALL db.cdc.enable({cap: 8})")
    with pytest.raises(kglite.CypherExecutionError, match="unknown parameter"):
        g.cypher("CALL db.cdc.enable()")  # enable first so query's own check is reached
        g.cypher("CALL db.cdc.query({fromm: 'x'})")


def test_the_family_is_listed_in_show_procedures(graph):
    names = {r["name"] for r in graph.cypher("SHOW PROCEDURES YIELD name").to_dicts()}
    assert {
        "db.cdc.enable",
        "db.cdc.disable",
        "db.cdc.current",
        "db.cdc.earliest",
        "db.cdc.query",
    } <= names
    listed = {r["name"] for r in graph.cypher("CALL list_procedures() YIELD name").to_dicts()}
    assert listed >= {"db.cdc.enable", "db.cdc.query"}, "the two registries must not drift"


# ── the mutation vocabulary ────────────────────────────────────────────────


def test_full_write_vocabulary_publishes_once_each(graph):
    graph.cypher("CREATE (a:P {id: 1, x: 1}), (b:P {id: 2})")
    graph.cypher("MERGE (c:P {id: 3})")
    graph.cypher("MATCH (a:P {id: 1}), (b:P {id: 2}) MERGE (a)-[:R]->(b)")
    graph.cypher("MATCH (a:P {id: 1})-[r:R]->() DELETE r")
    graph.cypher("MATCH (n:P {id: 2}) DETACH DELETE n")

    assert summary(events(graph)) == [
        ("create", "node", 1),
        ("create", "node", 2),
        ("create", "node", 3),
        ("create", "relationship", ("R", 1, 2)),
        ("delete", "relationship", ("R", 1, 2)),
        ("delete", "node", 2),
    ]


def test_both_set_map_spellings_publish_an_update_with_the_after_image(graph):
    graph.cypher("CREATE (:P {id: 1, x: 1})")
    graph.cypher("MATCH (n:P {id: 1}) SET n += {y: 2}")
    graph.cypher("MATCH (n:P {id: 1}) SET n = {z: 3}")
    graph.cypher("MATCH (n:P {id: 1}) SET n.w = 4")

    rows = events(graph)[1:]
    assert [r["operation"] for r in rows] == ["update", "update", "update"]
    # `+=` merges into the existing property set; `=` replaces it wholesale.
    assert rows[0]["state"]["properties"] == {"x": 1, "y": 2}
    assert rows[1]["state"]["properties"] == {"z": 3}
    assert rows[2]["state"]["properties"] == {"z": 3, "w": 4}


def test_a_delete_carries_identity_but_no_state(graph):
    graph.cypher("CREATE (:P {id: 1, x: 1})")
    graph.cypher("MATCH (n:P {id: 1}) DELETE n")
    deleted = events(graph)[-1]
    assert deleted["operation"] == "delete"
    assert deleted["nodeType"] == "P"
    assert deleted["nodeId"] == 1
    # v1 keeps no before-image, so there is genuinely nothing to report —
    # null rather than an empty map, which would read as "no properties".
    assert deleted["state"] is None


def test_edge_rows_carry_endpoints_and_null_node_columns(graph):
    graph.cypher("CREATE (:P {id: 1})-[:R {w: 2}]->(:Q {id: 9})")
    edge = [r for r in events(graph) if r["elementType"] == "relationship"][0]
    assert edge["relationshipType"] == "R"
    assert (edge["srcType"], edge["srcId"]) == ("P", 1)
    assert (edge["tgtType"], edge["tgtId"]) == ("Q", 9)
    assert edge["nodeType"] is None and edge["nodeId"] is None
    assert edge["state"]["properties"] == {"w": 2}


def test_yield_projects_and_aliases_like_any_procedure(graph):
    graph.cypher("CREATE (:P {id: 1})")
    rows = graph.cypher("CALL db.cdc.query() YIELD operation, nodeId").to_dicts()
    assert rows == [{"operation": "create", "nodeId": 1}]
    aliased = graph.cypher("CALL db.cdc.query() YIELD seq AS s RETURN s").to_dicts()
    assert aliased == [{"s": 1}]


# ── cursors ────────────────────────────────────────────────────────────────


def test_a_cursor_advances_past_what_it_has_already_seen(graph):
    graph.cypher("CREATE (:P {id: 1})")
    seen = cursor(graph)
    assert events(graph, seen) == [], "current() is exclusive — nothing after it yet"

    graph.cypher("CREATE (:P {id: 2})")
    fresh = events(graph, seen)
    assert summary(fresh) == [("create", "node", 2)]

    # The row's own id is the cursor for the next poll.
    assert events(graph, fresh[-1]["id"]) == []


def test_earliest_reads_the_whole_retained_log(graph):
    graph.cypher("CREATE (:P {id: 1})")
    graph.cypher("CREATE (:P {id: 2})")
    assert events(graph, cursor(graph, "earliest")) == events(graph)
    assert len(events(graph)) == 2


def test_a_cursor_older_than_retention_is_refused_with_the_gap_named():
    g = enabled_graph(capacity=3)
    stale = cursor(g, "earliest")
    for i in range(6):
        g.cypher(f"CREATE (:P {{id: {i}}})")
    assert [r["seq"] for r in events(g)] == [4, 5, 6], "the ring evicts oldest-first"

    with pytest.raises(kglite.CypherExecutionError) as exc:
        events(g, stale)
    message = str(exc.value)
    assert "too old" in message
    assert "oldest change still retained is 4" in message
    assert "db.cdc.earliest()" in message, "the error must name the resync move"


def test_a_malformed_cursor_is_rejected_on_sight(graph):
    for bad in ("garbage", "cdc:zz", "cdc:0000000000000001", "cdc:1:2"):
        with pytest.raises(kglite.CypherExecutionError, match="is not a change-stream cursor"):
            events(graph, bad)
    with pytest.raises(kglite.CypherExecutionError, match="must be a cursor string"):
        graph.cypher("CALL db.cdc.query({from: 7})")


def test_a_cursor_from_another_epoch_names_both_epochs(graph):
    stale = cursor(graph)
    graph.cypher("CALL db.cdc.disable()")
    graph.cypher("CALL db.cdc.enable()")
    with pytest.raises(kglite.CypherExecutionError) as exc:
        events(graph, stale)
    message = str(exc.value)
    assert "belongs to change-stream epoch" in message
    assert "db.cdc.earliest()" in message


# ── no phantom events ──────────────────────────────────────────────────────


def test_a_rolled_back_transaction_publishes_nothing(graph):
    tx = graph.begin()
    tx.cypher("CREATE (:P {id: 99})")
    tx.rollback()
    assert events(graph) == []


def test_a_transaction_that_raises_publishes_nothing(graph):
    with pytest.raises(RuntimeError):
        with graph.begin() as tx:
            tx.cypher("CREATE (:P {id: 99})")
            raise RuntimeError("boom")
    assert events(graph) == []


def test_a_committed_transaction_publishes_its_whole_batch_at_commit(graph):
    tx = graph.begin()
    tx.cypher("CREATE (:P {id: 1})")
    tx.cypher("CREATE (:P {id: 2})")
    assert events(graph) == [], "an open transaction has not committed anything yet"
    tx.commit()
    assert summary(events(graph)) == [("create", "node", 1), ("create", "node", 2)]


def test_a_failed_statement_publishes_nothing(graph):
    graph.cypher("CREATE (:P {id: 1})")
    before = events(graph)
    with pytest.raises(kglite.KgError):
        graph.cypher("MATCH (n:P) SET n.id = 7")  # id is immutable
    assert events(graph) == before


# ── copy-on-write ──────────────────────────────────────────────────────────


def test_a_write_behind_a_held_view_publishes_exactly_once(graph):
    graph.cypher("CREATE (:P {id: 1})")
    # Holding the view keeps a second Arc alive, so the next mutation forks
    # copy-on-write. The fork shares this log; the commit must land in it once.
    view = graph.cypher("MATCH (n:P) RETURN n.id")
    assert len(view) == 1

    graph.cypher("CREATE (:P {id: 2})")
    assert summary(events(graph)) == [("create", "node", 1), ("create", "node", 2)]

    graph.cypher("CREATE (:P {id: 3})")
    assert summary(events(graph)) == [
        ("create", "node", 1),
        ("create", "node", 2),
        ("create", "node", 3),
    ]
    del view


def test_the_log_survives_the_fork_a_held_view_forces(graph):
    view = graph.cypher("MATCH (n) RETURN n")
    epoch_before = graph.cypher("CALL db.cdc.enable()").to_dicts()[0]["epoch"]
    graph.cypher("CREATE (:P {id: 1})")
    del view
    graph.cypher("CREATE (:P {id: 2})")
    rows = events(graph)
    assert summary(rows) == [("create", "node", 1), ("create", "node", 2)]
    assert graph.cypher("CALL db.cdc.enable()").to_dicts()[0]["epoch"] == epoch_before


# ── persistence boundary ───────────────────────────────────────────────────


def test_a_loaded_graph_starts_with_capture_off_and_a_new_epoch(tmp_path):
    g = enabled_graph()
    g.cypher("CREATE (:P {id: 1})")
    stale = cursor(g)
    path = str(tmp_path / "cdc.kgl")
    g.save(path)

    loaded = kglite.load(path)
    with pytest.raises(kglite.CypherExecutionError, match="not enabled on this graph"):
        loaded.cypher("CALL db.cdc.current()")

    # Enabling again works, and refuses the pre-save cursor rather than
    # silently reinterpreting its sequence against a different log.
    loaded.cypher("CALL db.cdc.enable()")
    with pytest.raises(kglite.CypherExecutionError, match="belongs to change-stream epoch"):
        events(loaded, stale)

    loaded.cypher("CREATE (:P {id: 2})")
    assert summary(events(loaded)) == [("create", "node", 2)]


def test_saving_an_enabled_graph_leaves_its_own_stream_intact(tmp_path):
    g = enabled_graph()
    g.cypher("CREATE (:P {id: 1})")
    g.save(str(tmp_path / "cdc.kgl"))
    g.cypher("CREATE (:P {id: 2})")
    assert summary(events(g)) == [("create", "node", 1), ("create", "node", 2)]


# ── storage modes ──────────────────────────────────────────────────────────


def test_mapped_mode_serves_the_same_stream_as_memory():
    per_mode = {}
    for mode in ("memory", "mapped"):
        g = enabled_graph(storage=mode) if mode != "memory" else enabled_graph()
        g.cypher("CREATE (a:P {id: 1, x: 1})-[:R]->(b:Q {id: 2})")
        g.cypher("MATCH (n:P {id: 1}) SET n.x = 2")
        g.cypher("MATCH (n:Q {id: 2}) DETACH DELETE n")
        per_mode[mode] = summary(events(g))
    assert per_mode["mapped"] == per_mode["memory"], per_mode


def test_disk_mode_refuses_enable_and_explains_why(tmp_path):
    g = kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "disk"))
    with pytest.raises(kglite.KgError) as exc:
        g.cypher("CALL db.cdc.enable()")
    message = str(exc.value)
    assert "not supported for storage='disk'" in message
    assert "immutable generation" in message
    assert "in-memory or mapped" in message, "the refusal must name the supported modes"
