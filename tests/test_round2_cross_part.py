"""Interactions between the three round-2 parts, which no single part's suite sees.

Relationship constraints (R), CDC v2 before-images (C) and the `PropMap`
property container (N) each landed with their own tests, and each of those
suites holds one part fixed while it exercises the other. The cases below are
the ones that only break when two of them are combined:

- **A refused edge write must not leave a before-image behind.** The gate runs
  before the `GraphWrite` call, so a refusal should cost nothing — but "no
  event" is the weak half of that claim. The sharp half is that the *next*
  accepted write still reports the original value as its `before`: a
  first-touch pre-image captured by the refused attempt would survive in the
  op buffer and make the following commit's `before` a lie.
- **An accepted edge write on a constrained graph still carries both images.**
  `test_an_accepted_relationship_write_still_publishes` covers this under
  `enrichment: 'off'`, where `before` is always `None` and therefore cannot be
  wrong. Under `'full'` it can.
- **`PropMap` is built on rayon workers.** Projection fans out at
  `PROJECTION_MIN_ROWS` (4096) rows regardless of the `parallel=` hint, so
  `RETURN n` over a large match already builds every node's property container
  off-thread. Every existing parallel test projects scalars, and every
  whole-node golden runs under the threshold, so that path is otherwise
  unexercised.
- **A map-valued property is a `PropMap` inside a `PropMap`.** CDC clones both
  halves out of the graph; nesting is where a flat sorted container is most
  likely to differ from the `BTreeMap` it replaced.
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite

# ── helpers ────────────────────────────────────────────────────────────────


def _rel_graph() -> kglite.KnowledgeGraph:
    """Three `Person`s, one `KNOWS {since: 2020, weight: 3}`."""
    graph = kglite.KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame({"person_id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]}),
        "Person",
        "person_id",
        "name",
    )
    graph.add_connections(
        pd.DataFrame({"s": [1], "t": [2], "since": [2020], "weight": [3]}),
        "KNOWS",
        "Person",
        "s",
        "Person",
        "t",
    )
    return graph


def _cursor(graph) -> str:
    return graph.cypher("CALL db.cdc.current() YIELD id RETURN id").to_dicts()[0]["id"]


def _events(graph, cursor: str) -> list[dict]:
    return graph.cypher("CALL db.cdc.query({from: $c})", params={"c": cursor}).to_dicts()


def _stored_edge(graph) -> list[dict]:
    return graph.cypher("MATCH ()-[r:KNOWS]->() RETURN r.since AS since, r.weight AS weight").to_dicts()


# ── R x C: a refused edge write captures nothing, and poisons no later image ──


@pytest.mark.parametrize(
    "refused",
    [
        "MATCH ()-[r:KNOWS]->() SET r.weight = 'heavy'",
        "MATCH ()-[r:KNOWS]->() SET r = {since: 2021, weight: 'heavy'}",
        "MATCH ()-[r:KNOWS]->() SET r += {weight: 'heavy'}",
        "MATCH (a:Person {name: 'Alice'}), (c:Person {name: 'Carol'}) "
        "CREATE (a)-[:KNOWS {since: 2022, weight: 'heavy'}]->(c)",
    ],
)
def test_a_refused_edge_write_leaves_no_before_image_behind(refused):
    """The no-phantom claim, taken past "no event" to "no stale pre-image".

    A pre-image is captured at an entity's *first touch* in a commit. If the
    refused write captured one before the gate rejected it, the buffer would
    still hold it, and the next accepted write to the same edge would report
    that stale image as its `before` instead of re-reading the live value.
    """
    graph = _rel_graph()
    graph.cypher("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.weight IS :: INTEGER")
    graph.cypher("CALL db.cdc.enable({enrichment: 'full'})")
    cursor = _cursor(graph)

    with pytest.raises(kglite.ConstraintViolationError):
        graph.cypher(refused)

    assert _events(graph, cursor) == [], "a refused write published an event"
    assert _stored_edge(graph) == [{"since": 2020, "weight": 3}], "a refused write changed the edge"

    # The sharp half: the next accepted write must re-read the live value.
    graph.cypher("MATCH ()-[r:KNOWS]->() SET r.weight = 7")
    events = _events(graph, cursor)
    assert [event["operation"] for event in events] == ["update"], events
    assert events[0]["state"]["before"] == {"properties": {"since": 2020, "weight": 3}}
    assert events[0]["state"]["after"] == {"properties": {"since": 2020, "weight": 7}}


def test_an_accepted_edge_write_on_a_constrained_graph_carries_both_images():
    """`enrichment: 'off'` cannot detect a wrong `before`, because it has none."""
    graph = _rel_graph()
    graph.cypher("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.weight IS :: INTEGER")
    graph.cypher("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.since IS NOT NULL")
    graph.cypher("CALL db.cdc.enable({enrichment: 'full'})")
    cursor = _cursor(graph)

    graph.cypher("MATCH ()-[r:KNOWS]->() SET r.weight = 42")
    graph.cypher("MATCH ()-[r:KNOWS]->() DELETE r")

    updated, deleted = _events(graph, cursor)
    assert updated["state"]["before"] == {"properties": {"since": 2020, "weight": 3}}
    assert updated["state"]["after"] == {"properties": {"since": 2020, "weight": 42}}
    assert deleted["state"]["before"] == {"properties": {"since": 2020, "weight": 42}}
    assert deleted["state"]["after"] is None


def test_a_refused_bulk_frame_captures_nothing_under_full_enrichment():
    """The bulk gate's no-phantom arm is tested in Rust under `Off` only.

    `Off` never captures a pre-image, so it cannot see a gate that refuses a
    frame *after* the capture pass has already walked it.
    """
    graph = _rel_graph()
    graph.cypher("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.weight IS :: INTEGER")
    graph.cypher("CALL db.cdc.enable({enrichment: 'full'})")
    cursor = _cursor(graph)

    bad = pd.DataFrame({"s": [1, 1], "t": [2, 3], "since": [2021, 2022], "weight": [4, "heavy"]})
    with pytest.raises(kglite.ConstraintViolationError):
        graph.add_connections(bad, "KNOWS", "Person", "s", "Person", "t")

    assert _events(graph, cursor) == [], "a refused frame published an event"
    assert _stored_edge(graph) == [{"since": 2020, "weight": 3}], "a refused frame changed the edge"

    # A clean frame over the same rows still reports the pre-refusal state.
    good = pd.DataFrame({"s": [1], "t": [2], "since": [2021], "weight": [4]})
    graph.add_connections(good, "KNOWS", "Person", "s", "Person", "t")
    events = _events(graph, cursor)
    assert [event["operation"] for event in events] == ["update"], events
    assert events[0]["state"]["before"] == {"properties": {"since": 2020, "weight": 3}}


# ── N x R: the type gate reached through a map literal ──────────────────────


def test_map_assignment_gates_the_offending_key_whatever_its_position():
    """`SET r = {...}` desugars by iterating the map literal, which is now a
    `PropMap` — a *sorted* container, so the literal's written order is not the
    order the gate sees. The violating key must be the one named, whether it
    sorts first or last among the keys written.
    """
    for literal, offender in (
        ("{weight: 'heavy', since: 2021}", "weight"),
        ("{since: 2021, weight: 'heavy'}", "weight"),
        ("{alpha: 1, weight: 'heavy', zulu: 9}", "weight"),
    ):
        graph = _rel_graph()
        graph.cypher("CREATE CONSTRAINT FOR ()-[r:KNOWS]-() REQUIRE r.weight IS :: INTEGER")
        with pytest.raises(kglite.ConstraintViolationError) as excinfo:
            graph.cypher(f"MATCH ()-[r:KNOWS]->() SET r = {literal}")
        assert offender in str(excinfo.value), (literal, str(excinfo.value))
        assert _stored_edge(graph) == [{"since": 2020, "weight": 3}], literal


# ── N x C: a map-valued property is a PropMap inside a PropMap ──────────────


def test_nested_container_properties_survive_both_cdc_images():
    graph = kglite.KnowledgeGraph()
    graph.cypher("CALL db.cdc.enable({enrichment: 'full'})")
    graph.cypher("CREATE (:P {id: 1, meta: {b: 'two', a: 1}, tags: ['x', 'y']})")
    cursor = _cursor(graph)
    graph.cypher("MATCH (n:P {id: 1}) SET n.meta = {a: 2, b: 'two', c: 3.5}")

    (updated,) = _events(graph, cursor)
    assert updated["state"]["before"]["properties"]["meta"] == {"a": 1, "b": "two"}
    assert updated["state"]["before"]["properties"]["tags"] == ["x", "y"]
    assert updated["state"]["after"]["properties"]["meta"] == {"a": 2, "b": "two", "c": 3.5}
    assert updated["state"]["after"]["properties"]["tags"] == ["x", "y"]


# ── N x parallel runtime: PropMap built on rayon workers ────────────────────

#: `parallel::PROJECTION_MIN_ROWS`, the row count at which `return_clause.rs`
#: switches `RETURN n` onto `par_iter_mut()`. It keys off the result-set size
#: alone, *not* the `parallel=` hint, so it is reached by an ordinary query.
#: Mirrored here so the sizes below can assert they still straddle it — if the
#: engine raises it past `ROWS`, the differential quietly becomes
#: serial-against-serial and stops testing anything.
PROJECTION_FAN_OUT_ROWS = 4096

ROWS = 5_000
CHUNK = 2_000

assert ROWS > PROJECTION_FAN_OUT_ROWS, "the fixture must fan out"
assert CHUNK < PROJECTION_FAN_OUT_ROWS, "the reference chunks must not fan out"


def _item_frame(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nid": list(range(rows)),
            "name": [f"I_{i}" for i in range(rows)],
            "val": [float(i % 97) for i in range(rows)],
            "cat": [f"c_{i % 7}" for i in range(rows)],
        }
    )


@pytest.fixture(scope="module")
def item_graph():
    graph = kglite.KnowledgeGraph()
    graph.add_nodes(_item_frame(ROWS), "Item", "nid", "name")
    return graph


def test_whole_node_projection_agrees_across_the_fan_out_threshold(item_graph):
    """`RETURN n` over 5000 rows builds every `PropMap` on a rayon worker.

    The reference reads the same nodes in sub-threshold chunks, which the
    executor projects serially — so a container that loses, reorders or
    corrupts a key only when built off-thread shows up as a mismatch here and
    nowhere else.
    """
    parallel = [row["n"] for row in item_graph.cypher("MATCH (n:Item) RETURN n ORDER BY n.nid").to_dicts()]
    assert len(parallel) == ROWS

    serial: list[dict] = []
    for start in range(0, ROWS, CHUNK):
        rows = item_graph.cypher(
            "MATCH (n:Item) WHERE n.nid >= $lo AND n.nid < $hi RETURN n ORDER BY n.nid",
            params={"lo": start, "hi": start + CHUNK},
        ).to_dicts()
        assert len(rows) <= CHUNK
        serial.extend(row["n"] for row in rows)

    assert serial == parallel


def test_whole_node_projection_is_absolutely_correct_above_the_threshold(item_graph):
    """An absolute golden, so the differential above cannot pass by both paths
    being wrong in the same way."""
    rows = [row["n"] for row in item_graph.cypher("MATCH (n:Item) RETURN n ORDER BY n.nid").to_dicts()]

    for index in (0, 1, 4095, 4096, 4097, ROWS - 1):
        record = rows[index]
        assert record["labels"] == ["Item"]
        assert record["properties"] == {
            "cat": f"c_{index % 7}",
            "id": index,
            "name": f"I_{index}",
            "nid": index,
            "title": f"I_{index}",
            "type": "Item",
            "val": float(index % 97),
        }, index

    # Key order is part of the record contract, and a flat sorted container is
    # exactly where it could drift.
    assert list(rows[0]["properties"]) == sorted(rows[0]["properties"])


def test_parallel_hint_does_not_change_a_whole_node_projection(item_graph):
    query = "MATCH (n:Item) WHERE toUpper(n.cat) = 'C_3' RETURN n ORDER BY n.nid"
    assert item_graph.cypher(query, parallel=True).to_list() == item_graph.cypher(query, parallel=False).to_list()
