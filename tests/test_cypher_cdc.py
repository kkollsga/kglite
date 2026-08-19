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
    assert rows[0]["enrichment"] == "off"
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
        "db.cdc.status",
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
    assert rows[0]["state"]["after"]["properties"] == {"x": 1, "y": 2}
    assert rows[1]["state"]["after"]["properties"] == {"z": 3}
    assert rows[2]["state"]["after"]["properties"] == {"z": 3, "w": 4}


def test_a_delete_carries_identity_and_a_pair_with_both_halves_empty(graph):
    graph.cypher("CREATE (:P {id: 1, x: 1})")
    graph.cypher("MATCH (n:P {id: 1}) DELETE n")
    deleted = events(graph)[-1]
    assert deleted["operation"] == "delete"
    assert deleted["nodeType"] == "P"
    assert deleted["nodeId"] == 1
    # The pair is always there — a consumer reads state["after"] without
    # null-checking the container. Each half is null rather than an empty
    # map, which would read as "an entity with no properties": the delete
    # left no entity, and capture keeps no pre-image yet.
    assert deleted["state"] == {"before": None, "after": None}


def test_edge_rows_carry_endpoints_and_null_node_columns(graph):
    graph.cypher("CREATE (:P {id: 1})-[:R {w: 2}]->(:Q {id: 9})")
    edge = [r for r in events(graph) if r["elementType"] == "relationship"][0]
    assert edge["relationshipType"] == "R"
    assert (edge["srcType"], edge["srcId"]) == ("P", 1)
    assert (edge["tgtType"], edge["tgtId"]) == ("Q", 9)
    assert edge["nodeType"] is None and edge["nodeId"] is None
    assert edge["state"]["after"] == {"properties": {"w": 2}}
    assert edge["state"]["before"] is None


# ── the state pair ─────────────────────────────────────────────────────────


def test_state_is_the_before_after_pair_for_every_row(graph):
    """Every row's `state` is `{before, after}` — the shape, not just the keys.

    Non-vacuity: the values are asserted through the nesting, so the flat v1
    shape (`state["properties"]`) fails on the missing key rather than
    passing on a coincidence. `test_the_pair_is_traversable_from_cypher`
    covers the other direction, where flat would read as null instead.
    """
    graph.cypher("CREATE (:P {id: 1, name: 'one'})")
    graph.cypher("MATCH (n:P {id: 1}) SET n.name = 'uno'")
    graph.cypher("MATCH (a:P {id: 1}) CREATE (a)-[:R {w: 1}]->(:Q {id: 2})")
    graph.cypher("MATCH (n:P {id: 1}) DETACH DELETE n")

    rows = events(graph)
    assert rows, "an empty stream would pass every assertion below vacuously"
    for row in rows:
        assert set(row["state"]) == {"before", "after"}, row
        assert row["state"]["before"] is None, "before is empty until capture keeps a pre-image"

    def pick(**match):
        return [r for r in rows if all(r[key] == value for key, value in match.items())]

    created = pick(operation="create", elementType="node", nodeId=1)[0]
    assert set(created["state"]["after"]) == {"title", "labels", "properties"}
    assert created["state"]["after"]["properties"] == {"name": "one"}
    assert created["state"]["after"]["labels"] == []

    updated = pick(operation="update", nodeId=1)[0]
    assert updated["state"]["after"]["properties"] == {"name": "uno"}

    edge = pick(operation="create", elementType="relationship")[0]
    assert edge["state"]["after"] == {"properties": {"w": 1}}, "an edge image is properties only"

    deleted = pick(operation="delete")
    assert deleted, "the DETACH DELETE must publish"
    assert all(r["state"]["after"] is None for r in deleted)


def test_the_pair_is_traversable_from_cypher(graph):
    """The nesting is real map structure, reachable with `.` in a projection.

    This is the assertion that would have caught a `state` left flat: against
    the v1 shape `state.after.properties.name` resolves to null, silently,
    with every other column still correct.
    """
    graph.cypher("CREATE (:P {id: 1, name: 'one'})")
    rows = graph.cypher(
        "CALL db.cdc.query() YIELD state RETURN state.after.properties.name AS name, state.before AS before"
    ).to_dicts()
    assert rows == [{"name": "one", "before": None}]


# ── enrichment and status ──────────────────────────────────────────────────


def test_enrichment_defaults_to_off_and_accepts_full():
    g = kglite.KnowledgeGraph()
    assert g.cypher("CALL db.cdc.enable()").to_dicts()[0]["enrichment"] == "off"
    assert g.cypher("CALL db.cdc.enable({enrichment: 'full'})").to_dicts()[0]["enrichment"] == "full"


def test_changing_the_enrichment_keeps_the_epoch(graph):
    """A mode change is a reconfiguration, so live cursors must survive it."""
    graph.cypher("CREATE (:P {id: 1})")
    held = cursor(graph)
    before = graph.cypher("CALL db.cdc.status()").to_dicts()[0]

    graph.cypher("CALL db.cdc.enable({enrichment: 'full'})")
    after = graph.cypher("CALL db.cdc.status()").to_dicts()[0]
    assert after["epoch"] == before["epoch"]
    assert after["enrichment"] == "full"

    graph.cypher("CREATE (:P {id: 2})")
    assert [r["nodeId"] for r in events(graph, held)] == [2], "the held cursor still resolves"


def test_an_omitted_enrichment_resets_it_like_an_omitted_capacity():
    """`enable` is declarative: what you pass is what the log ends up as."""
    g = kglite.KnowledgeGraph()
    g.cypher("CALL db.cdc.enable({enrichment: 'full', capacity: 8})")
    assert g.cypher("CALL db.cdc.enable({capacity: 8})").to_dicts()[0]["enrichment"] == "off"


def test_a_diff_enrichment_is_refused_with_the_reason():
    g = kglite.KnowledgeGraph()
    with pytest.raises(kglite.CypherExecutionError, match="does not accept 'diff'"):
        g.cypher("CALL db.cdc.enable({enrichment: 'diff'})")
    with pytest.raises(kglite.CypherExecutionError, match="not a change-capture enrichment mode"):
        g.cypher("CALL db.cdc.enable({enrichment: 'partial'})")
    with pytest.raises(kglite.CypherExecutionError, match="unknown parameter 'enrichmnt'"):
        g.cypher("CALL db.cdc.enable({enrichmnt: 'full'})")
    assert g.cypher("CALL db.cdc.status()").to_dicts()[0]["enabled"] is False


def test_status_answers_whether_capture_is_off_or_on():
    g = kglite.KnowledgeGraph()
    off = g.cypher("CALL db.cdc.status()").to_dicts()[0]
    assert off == {
        "enabled": False,
        "epoch": None,
        "capacity": None,
        "enrichment": None,
        "buffered": None,
        "earliest": None,
        "current": None,
    }

    g.cypher("CALL db.cdc.enable({capacity: 4})")
    g.cypher("CREATE (:P {id: 1})")
    on = g.cypher("CALL db.cdc.status()").to_dicts()[0]
    assert on["enabled"] is True
    assert on["epoch"] > 0
    assert (on["capacity"], on["enrichment"], on["buffered"]) == (4, "off", 1)
    assert (on["earliest"], on["current"]) == (1, 1)


def test_yield_projects_and_aliases_like_any_procedure(graph):
    graph.cypher("CREATE (:P {id: 1})")
    rows = graph.cypher("CALL db.cdc.query() YIELD operation, nodeId").to_dicts()
    assert rows == [{"operation": "create", "nodeId": 1}]
    aliased = graph.cypher("CALL db.cdc.query() YIELD seq AS s RETURN s").to_dicts()
    assert aliased == [{"s": 1}]


# ── before-images (enrichment: full) ───────────────────────────────────────


@pytest.fixture
def full_graph():
    """A graph capturing both halves of every change."""
    g = kglite.KnowledgeGraph()
    g.cypher("CALL db.cdc.enable({enrichment: 'full'})")
    return g


def test_full_capture_reports_both_halves_through_cypher(full_graph):
    full_graph.cypher("CREATE (:P {id: 1, name: 'one', qty: 10})")
    full_graph.cypher("MATCH (n:P {id: 1}) SET n.qty = 99")
    full_graph.cypher("MATCH (n:P {id: 1}) DELETE n")

    created, updated, deleted = events(full_graph)

    assert created["state"]["before"] is None, "a create had no prior state"
    assert created["state"]["after"]["properties"] == {"name": "one", "qty": 10}

    assert updated["state"]["before"]["properties"] == {"name": "one", "qty": 10}
    assert updated["state"]["after"]["properties"] == {"name": "one", "qty": 99}

    assert deleted["state"]["after"] is None, "a delete left no entity"
    assert deleted["state"]["before"]["properties"] == {"name": "one", "qty": 99}
    assert deleted["state"]["before"]["title"] == "one"


def test_the_before_half_is_traversable_from_cypher(full_graph):
    """Both halves are real map structure, reachable with `.` in a projection."""
    full_graph.cypher("CREATE (:P {id: 1, qty: 10})")
    full_graph.cypher("MATCH (n:P {id: 1}) SET n.qty = 99")
    rows = full_graph.cypher(
        "CALL db.cdc.query() YIELD operation, state "
        "WHERE operation = 'update' "
        "RETURN state.before.properties.qty AS was, state.after.properties.qty AS now"
    ).to_dicts()
    assert rows == [{"was": 10, "now": 99}]


def test_before_is_the_state_at_the_start_of_the_commit(full_graph):
    """Three writes to one entity collapse to one event whose `before` is what
    the commit found — not what the last write replaced."""
    full_graph.cypher("CREATE (:P {id: 1, qty: 0})")
    cur = cursor(full_graph)
    with full_graph.begin() as tx:
        tx.cypher("MATCH (n:P {id: 1}) SET n.qty = 1")
        tx.cypher("MATCH (n:P {id: 1}) SET n.qty = 2")
        tx.cypher("MATCH (n:P {id: 1}) SET n.qty = 3")

    rows = events(full_graph, cur)
    assert len(rows) == 1, rows
    assert rows[0]["state"]["before"]["properties"]["qty"] == 0
    assert rows[0]["state"]["after"]["properties"]["qty"] == 3


def test_a_label_change_reports_the_label_set_it_replaced(full_graph):
    full_graph.cypher("CREATE (:P {id: 1})")
    full_graph.cypher("MATCH (n:P {id: 1}) SET n:Featured")
    cur = cursor(full_graph)
    full_graph.cypher("MATCH (n:P {id: 1}) SET n:Archived")

    row = events(full_graph, cur)[0]
    assert row["state"]["before"]["labels"] == ["Featured"]
    assert sorted(row["state"]["after"]["labels"]) == ["Archived", "Featured"]


def test_a_deleted_node_keeps_its_labels_in_the_before_image(full_graph):
    full_graph.cypher("CREATE (:P {id: 1, x: 1})")
    full_graph.cypher("MATCH (n:P {id: 1}) SET n:Featured")
    cur = cursor(full_graph)
    full_graph.cypher("MATCH (n:P {id: 1}) DELETE n")

    row = events(full_graph, cur)[0]
    assert row["operation"] == "delete"
    assert row["state"]["before"]["labels"] == ["Featured"]
    assert row["state"]["before"]["properties"] == {"x": 1}


def test_relationship_before_images(full_graph):
    full_graph.cypher("CREATE (:P {id: 1})-[:R {w: 1}]->(:Q {id: 2})")
    cur = cursor(full_graph)
    full_graph.cypher("MATCH ()-[r:R]->() SET r.w = 2")
    full_graph.cypher("MATCH ()-[r:R]->() DELETE r")

    updated, deleted = events(full_graph, cur)
    assert updated["state"]["before"] == {"properties": {"w": 1}}
    assert updated["state"]["after"] == {"properties": {"w": 2}}
    assert deleted["state"]["before"] == {"properties": {"w": 2}}
    assert deleted["state"]["after"] is None


def test_off_mode_still_reports_no_before(graph):
    """The default mode is unchanged by any of this — and pays for nothing."""
    graph.cypher("CREATE (:P {id: 1, x: 1})")
    graph.cypher("MATCH (n:P {id: 1}) SET n.x = 2")
    assert all(row["state"]["before"] is None for row in events(graph))
    assert graph.cypher("CALL db.cdc.status()").to_dicts()[0]["enrichment"] == "off"


def test_switching_to_full_starts_capturing_from_the_next_write(graph):
    """Enrichment applies to writes, not to the log: events already in the ring
    keep the shape they were captured with."""
    graph.cypher("CREATE (:P {id: 1, x: 1})")
    graph.cypher("CALL db.cdc.enable({enrichment: 'full'})")
    graph.cypher("MATCH (n:P {id: 1}) SET n.x = 2")

    created, updated = events(graph)
    assert created["state"]["before"] is None
    assert updated["state"]["before"]["properties"] == {"x": 1}


def test_a_rolled_back_transaction_captures_no_before_image(full_graph):
    full_graph.cypher("CREATE (:P {id: 1, x: 1})")
    cur = cursor(full_graph)
    with pytest.raises(RuntimeError):
        with full_graph.begin() as tx:
            tx.cypher("MATCH (n:P {id: 1}) SET n.x = 2")
            raise RuntimeError("abort")
    assert events(full_graph, cur) == []
    # And the next real write images the restored state.
    full_graph.cypher("MATCH (n:P {id: 1}) SET n.x = 3")
    assert events(full_graph, cur)[0]["state"]["before"]["properties"] == {"x": 1}


# ── selectors ──────────────────────────────────────────────────────────────


def _seeded_selector_graph():
    g = kglite.KnowledgeGraph()
    g.cypher("CALL db.cdc.enable()")
    g.cypher("CREATE (:P {id: 1, name: 'one'})")
    g.cypher("CREATE (:Q {id: 2})")
    g.cypher("MATCH (p:P {id: 1}) SET p:Featured")
    g.cypher("MATCH (a:P {id: 1}), (b:Q {id: 2}) CREATE (a)-[:R {w: 1}]->(b)")
    g.cypher("MATCH (q:Q {id: 2}) DETACH DELETE q")
    return g


def _selected(graph, selectors: str, extra: str = ""):
    return graph.cypher(
        f"CALL db.cdc.query({{selectors: {selectors}{extra}}}) YIELD operation, elementType, nodeType"
    ).to_dicts()


def test_selectors_parse_as_a_nested_list_of_maps_through_real_cypher():
    """The argument is a list of maps inside a map — pinned because nothing
    else in the procedure surface nests that deep, and a parser that could not
    would make the whole feature unreachable."""
    g = _seeded_selector_graph()
    rows = _selected(g, "[{operation: 'delete'}, {nodeType: 'P'}]")
    assert rows, "the nested literal must parse and match"
    assert {r["operation"] for r in rows} >= {"delete"}


def test_a_malformed_selector_nesting_errors_cleanly():
    g = _seeded_selector_graph()
    # A map where a list belongs.
    with pytest.raises(kglite.CypherExecutionError, match="must be a list of maps"):
        g.cypher("CALL db.cdc.query({selectors: {nodeType: 'P'}})")
    # A scalar where a map belongs.
    with pytest.raises(kglite.CypherExecutionError, match="must be a map of constraints"):
        g.cypher("CALL db.cdc.query({selectors: ['P']})")
    # A typo one level in must not silently widen the filter.
    with pytest.raises(kglite.CypherExecutionError, match="unknown selector key 'nodeTyp'"):
        g.cypher("CALL db.cdc.query({selectors: [{nodeTyp: 'P'}]})")
    # An empty map constrains nothing.
    with pytest.raises(kglite.CypherExecutionError, match="constrains nothing"):
        g.cypher("CALL db.cdc.query({selectors: [{}]})")


def test_each_dimension_filters_and_misses():
    g = _seeded_selector_graph()
    assert all(r["elementType"] == "relationship" for r in _selected(g, "[{elementType: 'relationship'}]"))
    assert all(r["operation"] == "delete" for r in _selected(g, "[{operation: 'delete'}]"))
    assert all(r["nodeType"] == "P" for r in _selected(g, "[{nodeType: 'P'}]"))
    assert _selected(g, "[{nodeType: 'Absent'}]") == []
    assert _selected(g, "[{relationshipType: 'NOPE'}]") == []
    assert _selected(g, "[{srcType: 'P', tgtType: 'Q'}]"), "endpoint types select the edge"
    assert _selected(g, "[{srcType: 'Q', tgtType: 'P'}]") == [], "endpoints are directional"
    assert _selected(g, "[{nodeId: 1}]"), "id equality selects"
    assert _selected(g, "[{nodeId: 404}]") == []


def test_labels_requires_all_of_them():
    g = kglite.KnowledgeGraph()
    g.cypher("CALL db.cdc.enable()")
    g.cypher("CREATE (:P {id: 1})")
    g.cypher("MATCH (n:P {id: 1}) SET n:Archived:Cold")
    g.cypher("CREATE (:P {id: 2})")
    g.cypher("MATCH (n:P {id: 2}) SET n:Archived")

    one = graph_ids(g, "[{labels: ['Archived', 'Cold']}]")
    assert one == [1], "every listed label must be present"
    assert sorted(set(graph_ids(g, "[{labels: ['Archived']}]"))) == [1, 2]
    assert graph_ids(g, "[{labels: ['P']}]") == [], "the primary type is nodeType's job"


def graph_ids(graph, selectors: str):
    return [r["nodeId"] for r in graph.cypher(f"CALL db.cdc.query({{selectors: {selectors}}}) YIELD nodeId").to_dicts()]


def test_changes_to_needs_full_enrichment_and_says_so(graph):
    graph.cypher("CREATE (:P {id: 1, qty: 1})")
    with pytest.raises(kglite.CypherExecutionError, match="changesTo"):
        graph.cypher("CALL db.cdc.query({selectors: [{changesTo: ['qty']}]})")


def test_changes_to_selects_the_properties_that_moved(full_graph):
    full_graph.cypher("CREATE (:P {id: 1, name: 'one', qty: 1})")
    cur = cursor(full_graph)
    full_graph.cypher("MATCH (n:P {id: 1}) SET n.qty = 2")

    moved = full_graph.cypher(
        "CALL db.cdc.query({from: $c, selectors: [{changesTo: ['qty']}]}) YIELD nodeId",
        params={"c": cur},
    ).to_dicts()
    assert len(moved) == 1
    still = full_graph.cypher(
        "CALL db.cdc.query({from: $c, selectors: [{changesTo: ['name']}]}) YIELD nodeId",
        params={"c": cur},
    ).to_dicts()
    assert still == [], "name did not move"


def test_max_rows_caps_matches_not_scanned_rows(graph):
    for i in range(10):
        graph.cypher(f"CREATE (:Filler {{id: {i}}})")
    graph.cypher("CREATE (:Wanted {id: 99})")
    rows = graph.cypher("CALL db.cdc.query({selectors: [{nodeType: 'Wanted'}], maxRows: 1}) YIELD nodeId").to_dicts()
    assert rows == [{"nodeId": 99}], "the cap counts matches, not the window they came from"


def test_a_selective_consumer_loop_neither_repeats_nor_skips(graph):
    """The documented polling shape: take `current()` first, then read up to it.

    The point is the rounds that match *nothing* — a filtered poll returning
    zero rows must still advance the consumer, which is why the cursor is taken
    before the query rather than from the last row returned.
    """
    seen: list[int] = []
    cur = cursor(graph)
    for round_no in range(1, 5):
        graph.cypher(f"CREATE (:Noise {{id: {round_no}}})")
        if round_no % 2 == 0:
            graph.cypher(f"CREATE (:Wanted {{id: {round_no}}})")
        nxt = cursor(graph)
        batch = graph.cypher(
            "CALL db.cdc.query({from: $c, selectors: [{nodeType: 'Wanted'}]}) YIELD nodeId",
            params={"c": cur},
        ).to_dicts()
        seen.extend(r["nodeId"] for r in batch)
        cur = nxt
    assert seen == [2, 4], "each wanted row exactly once, across empty polls"


def test_selectors_work_in_mapped_mode():
    g = kglite.KnowledgeGraph(storage="mapped")
    g.cypher("CALL db.cdc.enable()")
    g.cypher("CREATE (:P {id: 1, name: 'one'})")
    g.cypher("CREATE (:Q {id: 2})")
    g.cypher("MATCH (n:P {id: 1}) SET n.name = 'uno'")

    rows = g.cypher(
        "CALL db.cdc.query({selectors: [{nodeType: 'P', operation: 'update'}]}) YIELD nodeId, operation"
    ).to_dicts()
    assert rows == [{"nodeId": 1, "operation": "update"}]


def test_a_filtered_row_keeps_its_unfiltered_cursor_id(graph):
    graph.cypher("CREATE (:P {id: 1})")
    graph.cypher("CREATE (:Q {id: 2})")
    everything = {r["seq"]: r["id"] for r in graph.cypher("CALL db.cdc.query()").to_dicts()}
    filtered = graph.cypher("CALL db.cdc.query({selectors: [{nodeType: 'Q'}]}) YIELD id, seq").to_dicts()
    assert filtered, "non-vacuous"
    for row in filtered:
        assert everything[row["seq"]] == row["id"], "a cursor addresses the log, not the view"


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
