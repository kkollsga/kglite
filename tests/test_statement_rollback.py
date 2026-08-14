"""Statement atomicity: a mutation that fails partway must leave nothing behind.

Rollback used to be a whole-graph clone; it is now an inverse-op undo journal
(``crates/kglite/src/graph/storage/undo.rs``). The in-engine tests in
``dir_graph/rollback_tests.rs`` pin the parts only Rust can see — petgraph slot
identity, inverted-index bucket order. This file pins the contract a *user*
observes, and adds a randomized generator so the guarantee is not limited to
the shapes someone thought to write down.

Two levers force a failure after the first write, both deterministic:

* ``write_scope`` — a role whitelist that rejects a ``CREATE``/``SET`` of a
  non-whitelisted node type, which can fire between two writes of one
  statement.
* an overflowing ``duration({months: 2147483648})`` expression, which fires
  while a later pattern's properties are evaluated.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
import pytest

import kglite

BOOM = "duration({months: 2147483648})"


# ── observable state ─────────────────────────────────────────────────────


def snapshot(g: kglite.KnowledgeGraph) -> dict:
    """Everything about the graph a user can observe through Cypher.

    Ordering is normalized away here on purpose: this is the *user-visible*
    contract. Bucket order and slot identity are asserted in the Rust suite,
    which can read them directly.
    """
    nodes = sorted(
        (
            row["labels"][0],
            repr(row["id"]),
            repr(row["props"]),
            tuple(sorted(row["labels"])),
        )
        for row in g.cypher("MATCH (n) RETURN labels(n) AS labels, n.id AS id, properties(n) AS props").to_list()
    )
    edges = sorted(
        (row["t"], repr(row["a"]), repr(row["b"]), repr(row["props"]))
        for row in g.cypher(
            "MATCH (a)-[r]->(b) RETURN type(r) AS t, a.id AS a, b.id AS b, properties(r) AS props"
        ).to_list()
    )
    return {"nodes": nodes, "edges": edges}


def assert_rolls_back(g: kglite.KnowledgeGraph, query: str, **kwargs) -> None:
    """Assert `query` raises and leaves the graph exactly as it was."""
    before = snapshot(g)
    with pytest.raises(Exception) as excinfo:
        g.cypher(query, **kwargs)
    after = snapshot(g)
    assert before == after, f"{query} left changes behind (error was: {excinfo.value})"


@pytest.fixture
def g() -> kglite.KnowledgeGraph:
    graph = kglite.KnowledgeGraph()
    graph.cypher(
        "CREATE (a:Item {id: 1, name: 'a', qty: 10}), "
        "(b:Item {id: 2, name: 'b', qty: 20}), "
        "(c:Item {id: 3, name: 'c', qty: 30})"
    )
    graph.cypher("CREATE (:Tag:Hot {id: 1, name: 'urgent'})")
    graph.cypher("CREATE (:Tag:Cold {id: 2, name: 'later'})")
    graph.cypher("MATCH (a:Item {id: 1}), (b:Item {id: 2}) CREATE (a)-[:LINKS {weight: 5}]->(b)")
    graph.cypher("MATCH (b:Item {id: 2}), (c:Item {id: 3}) CREATE (b)-[:LINKS {weight: 7}]->(c)")
    graph.cypher("MATCH (a:Item {id: 1}), (t:Tag {id: 1}) CREATE (a)-[:TAGGED]->(t)")
    return graph


# ── one test per mutation shape ──────────────────────────────────────────


def test_create_rolls_back(g):
    assert_rolls_back(g, f"CREATE (:Item {{id: 100}}), (:Item {{id: 101, bad: {BOOM}}})")


def test_create_with_edge_rolls_back(g):
    assert_rolls_back(
        g,
        "CREATE (x:Item {id: 200})-[:LINKS {weight: 1}]->(y:Item {id: 201}), (z:Blocked {id: 202})",
        write_scope=["Item"],
    )


def test_create_across_pattern_parts_rolls_back(g):
    """Parts 1-2 create nodes and part 3 links them; part 4 is rejected.

    The whole statement — both nodes and the edge wired between them by a
    *later* part than the one that introduced their variables — must reverse.
    """
    assert_rolls_back(
        g,
        "CREATE (x:Item {id: 210, name: 'x'}), (y:Item {id: 211, name: 'y'}), "
        "(x)-[:LINKS {weight: 3}]->(y), (z:Blocked {id: 212})",
        write_scope=["Item"],
    )


def test_create_with_anonymous_endpoints_rolls_back(g):
    assert_rolls_back(
        g,
        "CREATE (:Item {id: 220})-[:LINKS]->(:Item {id: 221}), (:Blocked {id: 222})",
        write_scope=["Item"],
    )


def test_match_plus_anonymous_endpoint_create_rolls_back(g):
    assert_rolls_back(
        g,
        f"MATCH (a:Item {{id: 1}}) CREATE (a)-[:LINKS]->(:Item {{id: 230, name: 'anon', bad: {BOOM}}})",
    )


def test_create_with_secondary_labels_rolls_back(g):
    assert_rolls_back(
        g,
        "CREATE (:Tag:Hot:Fresh {id: 300}), (:Blocked {id: 301})",
        write_scope=["Tag"],
    )


def test_set_rolls_back(g):
    assert_rolls_back(
        g,
        f"MATCH (n:Item) SET n.qty = n.qty + 1, n.name = 'touched', n.bad = {BOOM}",
    )


def test_set_label_rolls_back(g):
    assert_rolls_back(g, f"MATCH (t:Tag {{id: 2}}) SET t:Hot, t.bad = {BOOM}")


def test_remove_property_and_label_roll_back(g):
    assert_rolls_back(
        g,
        "MATCH (t:Tag {id: 1}) REMOVE t.name, t:Hot CREATE (:Blocked {id: 400})",
        write_scope=["Tag"],
    )


def test_detach_delete_rolls_back(g):
    assert_rolls_back(
        g,
        "MATCH (n:Item {id: 2}) DETACH DELETE n CREATE (:Blocked {id: 500})",
        write_scope=["Item"],
    )


def test_detach_delete_everything_rolls_back(g):
    assert_rolls_back(
        g,
        "MATCH (n) DETACH DELETE n CREATE (:Blocked {id: 501})",
        write_scope=["Item", "Tag"],
    )


def test_delete_relationship_rolls_back(g):
    assert_rolls_back(
        g,
        "MATCH ()-[r:LINKS]->() DELETE r CREATE (:Blocked {id: 600})",
        write_scope=["Item"],
    )


def test_merge_create_arm_rolls_back(g):
    assert_rolls_back(
        g,
        "MERGE (n:Item {id: 700}) ON CREATE SET n.name = 'new' CREATE (:Blocked {id: 701})",
        write_scope=["Item"],
    )


def test_merge_match_arm_rolls_back(g):
    assert_rolls_back(
        g,
        "MERGE (n:Item {id: 1}) ON MATCH SET n.name = 'seen' CREATE (:Blocked {id: 702})",
        write_scope=["Item"],
    )


def test_foreach_rolls_back(g):
    assert_rolls_back(
        g,
        "FOREACH (i IN [1, 2, 3] | CREATE (:Item {id: 800 + i})) CREATE (:Blocked {id: 804})",
        write_scope=["Item"],
    )


def test_relationship_property_write_rolls_back(g):
    assert_rolls_back(
        g,
        f"MATCH ()-[r:LINKS]->() SET r.weight = 99, r.bad = {BOOM}",
    )


def test_delete_then_recreate_rolls_back(g):
    assert_rolls_back(
        g,
        "MATCH (n:Item {id: 2}) DETACH DELETE n "
        "CREATE (:Item {id: 2, name: 'replacement'}) "
        "CREATE (:Blocked {id: 900})",
        write_scope=["Item"],
    )


# ── the graph stays usable afterwards ────────────────────────────────────


def test_graph_is_fully_usable_after_a_rollback(g):
    before = snapshot(g)
    with pytest.raises(Exception):
        g.cypher(
            "MATCH (n) DETACH DELETE n CREATE (:Blocked {id: 1})",
            write_scope=["Item", "Tag"],
        )
    assert snapshot(g) == before
    # Writes, reads and traversals all still work on restored slots.
    g.cypher("CREATE (:Item {id: 1000, name: 'after'})")
    assert g.cypher("MATCH (n:Item) RETURN count(n) AS c").to_list()[0]["c"] == 4
    assert g.cypher("MATCH (a:Item {id: 1})-[:LINKS]->(b) RETURN b.id AS bid").to_list()[0]["bid"] == 2
    g.cypher("MATCH (n:Item {id: 1000}) DETACH DELETE n")
    assert snapshot(g) == before


def test_repeated_rollbacks_do_not_accumulate(g):
    before = snapshot(g)
    for _ in range(25):
        with pytest.raises(Exception):
            g.cypher(
                "MATCH (n:Item) DETACH DELETE n CREATE (:Blocked {id: 1})",
                write_scope=["Item"],
            )
    assert snapshot(g) == before


def test_rollback_does_not_leak_the_failed_types_into_the_schema(g):
    types_before = set(g.cypher("MATCH (n) RETURN DISTINCT labels(n)[0] AS t").to_list()[0])
    with pytest.raises(Exception):
        g.cypher(f"CREATE (:BrandNew {{id: 1}}), (:BrandNew {{id: 2, bad: {BOOM}}})")
    described = g.describe()
    assert "BrandNew" not in described, "a rolled-back statement must not leave its node type in the schema"
    assert types_before


# ── randomized mutate-fail-compare ───────────────────────────────────────
#
# The hand-written cases above cover the shapes we thought of. This generates
# statement bodies instead: a random prefix of valid writes followed by a
# guaranteed failure, so the journal must reverse an arbitrary interleaving.
# Found counterexamples persist in `.hypothesis/` as a regression corpus.

# Each entry is a clause template that mutates the seeded fixture and is valid
# on its own. `{i}` is a per-statement unique integer so repeated draws of the
# same clause cannot collide on a primary key.
WRITE_CLAUSES = [
    "CREATE (:Item {{id: {i}, name: 'gen{i}'}})",
    "CREATE (:Tag:Hot {{id: {i}}})",
    "CREATE (x{i}:Item {{id: {i}}})-[:LINKS {{weight: {i}}}]->(y{i}:Item {{id: -{i}}})",
    "MATCH (n{i}:Item {{id: 1}}) SET n{i}.qty = {i}",
    "MATCH (n{i}:Item) SET n{i}.batch = {i}",
    "MATCH (n{i}:Item {{id: 3}}) REMOVE n{i}.name",
    "MATCH (t{i}:Tag {{id: 2}}) SET t{i}:Warm",
    "MATCH (t{i}:Tag {{id: 1}}) REMOVE t{i}:Hot",
    "MATCH (d{i}:Item {{id: 2}}) DETACH DELETE d{i}",
    "MATCH (a{i})-[r{i}:LINKS]->(b{i}) SET r{i}.weight = {i}",
    "MATCH (a{i})-[r{i}:TAGGED]->(b{i}) DELETE r{i}",
    "FOREACH (k IN [1, 2] | CREATE (:Item {{id: {i} * 10 + k}}))",
]


@given(
    plan=st.lists(st.sampled_from(WRITE_CLAUSES), min_size=1, max_size=5),
    poison_with_scope=st.booleans(),
)
@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_random_statement_that_fails_rolls_back_completely(plan, poison_with_scope):
    graph = kglite.KnowledgeGraph()
    graph.cypher(
        "CREATE (a:Item {id: 1, name: 'a', qty: 10}), "
        "(b:Item {id: 2, name: 'b', qty: 20}), "
        "(c:Item {id: 3, name: 'c', qty: 30})"
    )
    graph.cypher("CREATE (:Tag:Hot {id: 1, name: 'urgent'})")
    graph.cypher("CREATE (:Tag:Cold {id: 2, name: 'later'})")
    graph.cypher("MATCH (a:Item {id: 1}), (b:Item {id: 2}) CREATE (a)-[:LINKS {weight: 5}]->(b)")
    graph.cypher("MATCH (a:Item {id: 1}), (t:Tag {id: 1}) CREATE (a)-[:TAGGED]->(t)")

    body = " ".join(clause.format(i=1000 + n) for n, clause in enumerate(plan))
    if poison_with_scope:
        # A node type outside the whitelist: rejected between writes.
        query = f"{body} CREATE (:Blocked {{id: 7}})"
        kwargs = {"write_scope": ["Item", "Tag"]}
    else:
        query = f"{body} CREATE (:Item {{id: 8, bad: {BOOM}}})"
        kwargs = {}

    before = snapshot(graph)
    try:
        graph.cypher(query, **kwargs)
    except Exception:
        assert snapshot(graph) == before, f"incomplete rollback for: {query}"
        return
    # A generated prefix can legitimately make the poison clause unreachable
    # (e.g. everything it would have matched was deleted first). Nothing to
    # assert then — the statement committed, which is correct.
    pytest.skip("generated statement did not fail")
