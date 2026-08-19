"""Cross-program interactions: the parallel runtime against CDC and against
property-type constraints.

Two independent programs landed on this branch — EE-parity (property-type
constraints + change data capture) and the opt-in parallel Cypher runtime —
and each was gated on its own. Nothing in either design says they touch, but
"nothing says they touch" is an argument, not a test, and both programs put
new state on `DirGraph`:

- CDC hangs an `Arc<Mutex<..>>` change log off the graph and publishes at
  commit boundaries. A parallel *read* must never reach it — if it did, the
  mutex would be shared across rayon workers and the log could gain phantom
  events.
- Property-type constraints are enforced on the write path, which never fans
  out. A `parallel=True` flag on a mutation must therefore be inert, not
  merely harmless-looking.

Fixtures are sized above the runtime gate so `parallel=True` genuinely fans
out; below it these would all compare two sequential runs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kglite import KnowledgeGraph

# Above `PARALLEL_MIN_ROWS_COMPILED` (20_000), so the candidate scan and the
# fused scan aggregate both fan out.
ROWS = 20_137

SCAN = "MATCH (n:Item) WHERE n.value > 100 RETURN n.name AS nm"
AGG = "MATCH (n:Item) RETURN n.cat AS c, count(*) AS n"


def _frame(rows: int, start: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nid": list(range(start, start + rows)),
            "name": [f"Item_{i}" for i in range(start, start + rows)],
            "value": [float(i % 1000) for i in range(start, start + rows)],
            "cat": [f"cat_{i % 7}" for i in range(start, start + rows)],
        }
    )


def _events(graph) -> list[dict]:
    return graph.cypher("CALL db.cdc.query()").to_dicts()


@pytest.fixture
def cdc_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.add_nodes(_frame(ROWS), "Item", "nid", "name")
    graph.cypher("CALL db.cdc.enable()")
    return graph


# ── CDC × parallel ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("query", [SCAN, AGG])
def test_parallel_read_on_a_cdc_enabled_graph_matches_serial(cdc_graph, query):
    """The answer must not depend on whether capture is running."""
    serial = cdc_graph.cypher(query, parallel=False).to_list()
    parallel = cdc_graph.cypher(query, parallel=True).to_list()
    assert serial == parallel
    assert serial, "fixture produced no rows — the comparison would be vacuous"


@pytest.mark.parametrize("query", [SCAN, AGG])
def test_a_parallel_read_publishes_no_cdc_events(cdc_graph, query):
    """The load-bearing one. A read fanned across rayon workers must not reach
    the change log at all — not to append, and not to publish an empty commit.
    """
    before = _events(cdc_graph)
    cdc_graph.cypher(query, parallel=True)
    assert _events(cdc_graph) == before, "a parallel read touched the change stream"


def test_cdc_procedures_work_with_the_parallel_flag_set(cdc_graph):
    """`db.cdc.*` are procedures, reached through the same executor the flag
    rides on. Passing it must neither break them nor make them fan out."""
    assert cdc_graph.cypher("CALL db.cdc.query()", parallel=True).to_dicts() == []
    current = cdc_graph.cypher("CALL db.cdc.current()", parallel=True).to_dicts()
    assert current and "id" in current[0]


def test_writes_still_capture_with_parallel_reads_interleaved(cdc_graph):
    """A parallel read between two writes must not lose, duplicate or reorder
    what capture sees."""
    cdc_graph.cypher("CREATE (:Marker {id: 'a'})")
    cdc_graph.cypher(SCAN, parallel=True)
    cdc_graph.cypher("CREATE (:Marker {id: 'b'})")

    captured = [(row["operation"], row["elementType"]) for row in _events(cdc_graph)]
    assert captured == [("create", "node"), ("create", "node")], captured


def test_a_mutation_with_the_parallel_flag_still_captures_exactly_once(cdc_graph):
    """`parallel=True` on a write is inert — the write path never fans out —
    and must not disturb the commit boundary that publishes."""
    cdc_graph.cypher("CREATE (:Marker {id: 'c'})", parallel=True)
    assert len(_events(cdc_graph)) == 1


# ── property-type constraints × parallel ────────────────────────────────────


@pytest.fixture
def typed_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.add_nodes(_frame(ROWS), "Item", "nid", "name")
    graph.cypher("CREATE CONSTRAINT item_value FOR (n:Item) REQUIRE n.value IS :: FLOAT")
    return graph


@pytest.mark.parametrize("query", [SCAN, AGG])
def test_parallel_read_under_a_type_constraint_matches_serial(typed_graph, query):
    serial = typed_graph.cypher(query, parallel=False).to_list()
    parallel = typed_graph.cypher(query, parallel=True).to_list()
    assert serial == parallel
    assert serial


def test_a_type_constraint_still_rejects_with_the_parallel_flag_set(typed_graph):
    """Enforcement is on the write path, so the flag must not weaken it —
    a violation raises whether or not the caller asked for parallelism."""
    with pytest.raises(Exception) as excinfo:
        typed_graph.cypher("CREATE (:Item {nid: 999999, value: 'not-a-float'})", parallel=True)
    assert "PROPERTY TYPE constraint" in str(excinfo.value), str(excinfo.value)

    # And a conforming write still succeeds with the flag set.
    typed_graph.cypher("CREATE (:Item {nid: 999998, value: 1.5})", parallel=True)


def test_constraint_survives_a_parallel_read_in_between(typed_graph):
    """A fanned-out read must not disturb the schema state the constraint
    lives in."""
    typed_graph.cypher(SCAN, parallel=True)
    shown = typed_graph.cypher("SHOW CONSTRAINTS").to_dicts()
    assert any(row.get("name") == "item_value" for row in shown), shown


# ── all three at once ───────────────────────────────────────────────────────


def test_cdc_and_type_constraint_and_parallel_together():
    """Both programs' state on one graph, with the flag on."""
    graph = KnowledgeGraph()
    graph.add_nodes(_frame(ROWS), "Item", "nid", "name")
    graph.cypher("CREATE CONSTRAINT item_value FOR (n:Item) REQUIRE n.value IS :: FLOAT")
    graph.cypher("CALL db.cdc.enable()")

    serial = graph.cypher(AGG, parallel=False).to_list()
    assert graph.cypher(AGG, parallel=True).to_list() == serial
    assert _events(graph) == [], "a parallel read published a change event"

    with pytest.raises(Exception):
        graph.cypher("CREATE (:Item {nid: 999999, value: 'nope'})", parallel=True)
    assert _events(graph) == [], "a rejected write published a change event"

    graph.cypher("CREATE (:Item {nid: 999997, value: 2.5})", parallel=True)
    assert len(_events(graph)) == 1


def test_mapped_mode_cdc_and_parallel_together():
    """Mapped is the other backend the parallel gate admits, and CDC supports
    it too — so it is the one mode where both programs' new state is live at
    once on a fanned-out read."""
    graph = KnowledgeGraph(storage="mapped")
    graph.add_nodes(_frame(ROWS), "Item", "nid", "name")
    graph.cypher("CALL db.cdc.enable()")

    serial = graph.cypher(AGG, parallel=False).to_list()
    assert graph.cypher(AGG, parallel=True).to_list() == serial
    assert _events(graph) == [], "a parallel read on a mapped graph published an event"

    graph.cypher("CREATE (:Marker {id: 'm'})")
    assert len(_events(graph)) == 1


def test_held_view_forks_the_graph_around_a_parallel_read(cdc_graph):
    """CDC's "exactly once under copy-on-write" property, with a fanned-out
    read in the middle.

    A held `ResultView` forces the next write to fork the graph; the fork
    shares the change log through its `Arc`, so the commit must publish once.
    A parallel read taken while the view is held must not disturb either the
    view or that count.
    """
    held = cdc_graph.cypher("MATCH (n:Item) RETURN n.name AS nm LIMIT 5")
    cdc_graph.cypher(AGG, parallel=True)
    cdc_graph.cypher("CREATE (:Marker {id: 'x'})")

    assert len(_events(cdc_graph)) == 1, "the forked write did not publish exactly once"
    assert len(held.to_list()) == 5, "the held view did not survive the parallel read"
