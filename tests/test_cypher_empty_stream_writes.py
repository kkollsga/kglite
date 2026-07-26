"""A write clause fed zero rows must write nothing.

Cypher's pipeline conceptually starts with one implicit empty row, so a
*leading* `CREATE`/`MERGE`/`FOREACH` runs once. Once any clause has run,
though, an empty row set means genuinely zero rows and every downstream
clause — read or write — must produce zero rows and no side effects.

The engine used to conflate the two states: `execute_create`,
`execute_merge`, and `execute_foreach` each re-derived "am I leading?" from
`rows.is_empty()`, which is also what a MATCH that found nothing leaves
behind. So `MATCH (p:Project {key:'NOPE'}) CREATE (t:Task)` fabricated a row
and created an unbound node. `SET`/`DELETE`/`REMOVE` never had the bug
because they simply iterate the incoming rows.

Why this matters beyond row counts: with no referential-integrity
constraints, "MATCH the parent, then CREATE the child" is the only mechanism
an application has to enforce a foreign key. The two-variable form was worse
than a spurious row — it fabricated the *unmatched endpoint* as a real node,
leaving an edge pointing at a label-less, property-less node.

The fix moved the implicit-start-row seed up into `run_clause_pipeline`,
which is the only place that can tell "not started" from "zero rows".
"""

from __future__ import annotations

import pytest

import kglite


def _graph_with_project():
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:Project {key: 'WEB'})").to_list()
    return graph


def _node_count(graph, pattern="(n)"):
    return graph.cypher(f"MATCH {pattern} RETURN count(*) AS c").to_list()[0]["c"]


def _edge_count(graph):
    return graph.cypher("MATCH ()-[r]->() RETURN count(*) AS c").to_list()[0]["c"]


# Every shape here ends in a MATCH-produced empty stream feeding a CREATE.
# Parametrized rather than written out so a future clause form is one line.
EMPTY_STREAM_CREATE_QUERIES = {
    "inline_map_match": ("MATCH (p:Project {key: 'NOPE'}) CREATE (t:Task {title: 'orphan'}) RETURN t.id AS id"),
    "where_match": ("MATCH (p:Project) WHERE p.key = 'NOPE' CREATE (t:Task {title: 'orphan'}) RETURN t.id AS id"),
    "match_with_create": ("MATCH (p:Project {key: 'NOPE'}) WITH p CREATE (t:Task {title: 'orphan'}) RETURN t.id AS id"),
    "relationship_pattern_match": (
        "MATCH (p:Project)-[:HAS]->(q:Project) CREATE (t:Task {title: 'orphan'}) RETURN t.id AS id"
    ),
    "with_where_filters_to_zero": (
        "MATCH (p:Project) WITH p WHERE p.key = 'NOPE' CREATE (t:Task {title: 'orphan'}) RETURN t.id AS id"
    ),
    "unwind_empty_list": "UNWIND [] AS x CREATE (t:Task {title: 'orphan'}) RETURN t.id AS id",
}


@pytest.mark.parametrize("query", EMPTY_STREAM_CREATE_QUERIES.values(), ids=EMPTY_STREAM_CREATE_QUERIES)
def test_create_after_empty_stream_returns_no_rows_and_creates_nothing(query):
    graph = _graph_with_project()
    before = _node_count(graph)

    rows = graph.cypher(query).to_list()

    assert rows == [], "a CREATE fed zero rows must return zero rows"
    assert graph.last_mutation_stats["nodes_created"] == 0
    assert _node_count(graph) == before
    assert _node_count(graph, "(t:Task)") == 0


def test_create_after_empty_stream_does_not_run_twice_for_two_create_clauses():
    """Two chained CREATEs after an empty MATCH create neither node.

    The first CREATE must leave the stream empty rather than re-seeding it,
    otherwise the second one inherits a fabricated row.
    """
    graph = _graph_with_project()
    rows = graph.cypher(
        "MATCH (p:Project {key: 'NOPE'}) CREATE (t:Task {title: 'a'}) CREATE (u:Task {title: 'b'}) RETURN count(*) AS c"
    ).to_list()

    assert rows == [{"c": 0}]
    assert graph.last_mutation_stats["nodes_created"] == 0
    assert _node_count(graph, "(t:Task)") == 0


def test_two_variable_match_does_not_fabricate_the_unmatched_endpoint():
    """The phantom-endpoint case: neither the missing node nor the edge.

    `(u:User {...})` matches nothing, so the whole MATCH is empty. Creating
    the relationship would have to invent `u`, producing an edge that points
    at a node carrying no label and no email — unreachable by any label scan
    but reachable by traversal.
    """
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:Task {title: 'a'})").to_list()

    rows = graph.cypher(
        "MATCH (t:Task {title: 'a'}), (u:User {email: 'ghost@x.com'}) "
        "CREATE (t)-[:ASSIGNED_TO]->(u) RETURN count(*) AS n"
    ).to_list()

    # count(*) over zero rows is one row holding 0 — the aggregate is the row,
    # not the match. What must be zero is the writes.
    assert rows == [{"n": 0}]
    stats = graph.last_mutation_stats
    assert stats["nodes_created"] == 0
    assert stats["relationships_created"] == 0
    assert _node_count(graph) == 1
    assert _edge_count(graph) == 0
    assert _node_count(graph, "(u:User)") == 0


def test_merge_after_empty_stream_creates_nothing():
    graph = _graph_with_project()
    rows = graph.cypher("MATCH (p:Project {key: 'NOPE'}) MERGE (t:Task {title: 'm'}) RETURN t.title AS t").to_list()

    assert rows == []
    assert graph.last_mutation_stats["nodes_created"] == 0
    assert _node_count(graph, "(t:Task)") == 0


def test_merge_after_empty_unwind_creates_nothing():
    graph = kglite.KnowledgeGraph()
    rows = graph.cypher("UNWIND [] AS x MERGE (t:Task {title: 'm'}) RETURN t.title AS t").to_list()

    assert rows == []
    assert _node_count(graph, "(t:Task)") == 0


def test_foreach_after_empty_stream_never_runs_its_body():
    """FOREACH is a side-effect loop, so the evidence is the graph, not rows."""
    graph = _graph_with_project()
    graph.cypher("MATCH (p:Project {key: 'NOPE'}) FOREACH (i IN [1, 2] | CREATE (:Task {title: 'f'}))").to_list()

    assert graph.last_mutation_stats["nodes_created"] == 0
    assert _node_count(graph, "(t:Task)") == 0


def test_foreach_after_empty_unwind_never_runs_its_body():
    graph = kglite.KnowledgeGraph()
    graph.cypher("UNWIND [] AS x FOREACH (i IN [1] | CREATE (:Task {title: 'f'}))").to_list()

    assert _node_count(graph, "(t:Task)") == 0


# ---------------------------------------------------------------------------
# Controls — the behaviour the fix must NOT change.
# ---------------------------------------------------------------------------


def test_bare_create_still_creates_exactly_one_node():
    """No preceding clause means the implicit start row applies."""
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (t:Task {title: 'x'})").to_list()

    assert graph.last_mutation_stats["nodes_created"] == 1
    assert _node_count(graph, "(t:Task)") == 1


def test_leading_merge_still_creates():
    graph = kglite.KnowledgeGraph()
    rows = graph.cypher("MERGE (t:Task {title: 'm'}) RETURN t.title AS t").to_list()

    assert rows == [{"t": "m"}]
    assert _node_count(graph, "(t:Task)") == 1


def test_standalone_foreach_still_runs_once():
    graph = kglite.KnowledgeGraph()
    graph.cypher("FOREACH (i IN [1, 2] | CREATE (:Task {title: 'f'}))").to_list()

    assert _node_count(graph, "(t:Task)") == 2


def test_foreach_over_empty_list_creates_nothing_but_still_runs():
    """The inner loop, not the outer stream, is what's empty here."""
    graph = kglite.KnowledgeGraph()
    graph.cypher("FOREACH (i IN [] | CREATE (:Task {title: 'f'}))").to_list()

    assert _node_count(graph, "(t:Task)") == 0


def test_leading_unwind_and_with_still_seed_a_row():
    graph = kglite.KnowledgeGraph()
    rows = graph.cypher("UNWIND [1, 2, 3] AS x CREATE (t:Task {n: x}) RETURN count(*) AS c").to_list()
    assert rows == [{"c": 3}]
    assert _node_count(graph, "(t:Task)") == 3

    other = kglite.KnowledgeGraph()
    assert other.cypher("WITH 5 AS v CREATE (t:Task {n: v}) RETURN t.n AS n").to_list() == [{"n": 5}]


def test_optional_match_still_yields_one_null_padded_row():
    """The deliberate opposite of this fix — OPTIONAL MATCH must NOT go empty.

    Guarding this here because the fix touches the same pipeline branch that
    decides whether a clause sees a row.
    """
    graph = _graph_with_project()
    rows = graph.cypher("MATCH (p:Project) OPTIONAL MATCH (u:User) RETURN p.key AS k, u AS u").to_list()

    assert len(rows) == 1
    assert rows[0]["k"] == "WEB"
    assert rows[0]["u"] is None


def test_optional_match_row_can_still_drive_a_create():
    """A null-padded OPTIONAL MATCH row is a real row, so CREATE runs."""
    graph = _graph_with_project()
    graph.cypher(
        "MATCH (p:Project) OPTIONAL MATCH (u:User) CREATE (t:Task {title: 'ok'}) RETURN t.title AS t"
    ).to_list()

    assert graph.last_mutation_stats["nodes_created"] == 1
    assert _node_count(graph, "(t:Task)") == 1


def test_set_and_delete_after_empty_stream_stay_no_ops():
    """These were already correct; pinned so the shared path can't regress them."""
    graph = _graph_with_project()
    assert graph.cypher("MATCH (p:Project {key: 'NOPE'}) SET p.x = 1 RETURN p.x AS x").to_list() == []
    assert graph.last_mutation_stats["properties_set"] == 0

    graph.cypher("MATCH (p:Project {key: 'NOPE'}) DELETE p").to_list()
    assert graph.last_mutation_stats["nodes_deleted"] == 0
    assert _node_count(graph, "(p:Project)") == 1
