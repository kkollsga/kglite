"""Absolute goldens for CREATE element bookkeeping.

Two halves of one defect in ``execute_create``'s element -> NodeIndex record:

* **Cross-part bindings.** The variable map was rebuilt for every
  comma-separated pattern part and seeded only from the *incoming* row, so a
  later part could not see a variable an earlier part introduced.
  ``CREATE (a:T {id:5}), (b:T {id:7}), (b)-[:E]->(a)`` silently produced four
  nodes — two anonymous, untyped ones standing in for ``a`` and ``b`` — and
  wired the ``:E`` between the fabricated pair, leaving the real nodes
  unconnected.
* **Anonymous endpoints.** The map was keyed by variable name only, so a node
  pattern with no variable had nowhere to be recorded, and the edge pass
  rejected the whole statement with *"CREATE edge requires named source and
  target nodes"* — including ``CREATE (:A)-[:R]->(:B)``, the most idiomatic
  Neo4j CREATE form there is.

Both are invisible to the optimiser differential corpus (the unoptimised path
produces the same wrong answer), and a "does a :T node exist?" assertion passes
under either bug. So every case here asserts the **node census and the edge
endpoints by identity** — the count alone would have been satisfied by the
fabricated nodes, and the endpoints alone by the count being wrong.
"""

from __future__ import annotations

import pytest

import kglite


def census(g: kglite.KnowledgeGraph) -> tuple[list, list]:
    """(nodes, edges) as sorted plain tuples — the whole observable graph."""
    nodes = sorted(
        (row["name"], row["labels"][0])
        for row in g.cypher("MATCH (n) RETURN n.title AS name, labels(n) AS labels").to_list()
    )
    edges = sorted(
        (row["s"], row["t"], row["d"])
        for row in g.cypher("MATCH (a)-[r]->(b) RETURN a.title AS s, type(r) AS t, b.title AS d").to_list()
    )
    return nodes, edges


@pytest.fixture
def g() -> kglite.KnowledgeGraph:
    return kglite.KnowledgeGraph()


# ── cross-part bindings ──────────────────────────────────────────────────


def test_third_part_references_variables_bound_by_the_first_two(g):
    g.cypher("CREATE (a:T {id: 5, name: 'a'}), (b:T {id: 7, name: 'b'}), (b)-[:E]->(a)")
    assert census(g) == ([("a", "T"), ("b", "T")], [("b", "E", "a")])


def test_four_part_chain_reuses_bindings_across_every_part(g):
    g.cypher(
        "CREATE (a:T {id: 1, name: 'a'}), (b:T {id: 2, name: 'b'}), "
        "(c:T {id: 3, name: 'c'}), (a)-[:E]->(b), (b)-[:E]->(c)"
    )
    assert census(g) == (
        [("a", "T"), ("b", "T"), ("c", "T")],
        [("a", "E", "b"), ("b", "E", "c")],
    )


def test_match_bound_variable_links_to_a_node_created_in_an_earlier_part(g):
    g.cypher("CREATE (a:T {id: 1, name: 'a'})")
    g.cypher("MATCH (a:T) CREATE (b:T {id: 2, name: 'b'}), (a)-[:E]->(b)")
    assert census(g) == ([("a", "T"), ("b", "T")], [("a", "E", "b")])


# ── anonymous endpoints ──────────────────────────────────────────────────


def test_fully_inline_anonymous_create(g):
    g.cypher("CREATE (:A1 {name: 'x'})-[:R]->(:A2 {name: 'y'})")
    assert census(g) == ([("x", "A1"), ("y", "A2")], [("x", "R", "y")])


def test_match_then_anonymous_target(g):
    """The T8 fixture shape — OPTIONAL MATCH…WHERE fixtures need it to author."""
    g.cypher("CREATE (a:P {id: 1, name: 'p'})")
    g.cypher("MATCH (a:P) CREATE (a)-[:KNOWS]->(:Q {name: 'q', w: 1})")
    assert census(g) == ([("p", "P"), ("q", "Q")], [("p", "KNOWS", "q")])
    assert g.cypher("MATCH (:P)-[:KNOWS]->(q:Q) RETURN q.w AS w").to_list() == [{"w": 1}]


def test_bare_parenthesis_endpoint(g):
    g.cypher("CREATE (h:H {id: 1, name: 'h'})")
    g.cypher("MATCH (h:H) CREATE (h)-[:R]->()")
    nodes, edges = census(g)
    assert len(nodes) == 2
    assert ("h", "H") in nodes
    # The untyped endpoint takes the default `Node` label; its identity relative
    # to the edge is what is contractual.
    assert [(s, t) for s, t, _ in edges] == [("h", "R")]
    assert edges[0][2] != "h"


def test_anonymous_endpoints_on_both_sides_of_a_two_hop_inline_pattern(g):
    g.cypher("CREATE (:A {name: 'a'})-[:R]->(m:B {name: 'm'})-[:R]->(:C {name: 'c'})")
    assert census(g) == (
        [("a", "A"), ("c", "C"), ("m", "B")],
        [("a", "R", "m"), ("m", "R", "c")],
    )


# ── controls: shapes that were already correct and must stay so ──────────


def test_control_single_inline_named_pattern(g):
    g.cypher("CREATE (a:T {id: 1, name: 'a'})-[:E]->(b:T {id: 2, name: 'b'})")
    assert census(g) == ([("a", "T"), ("b", "T")], [("a", "E", "b")])


def test_control_edge_between_two_matched_nodes(g):
    g.cypher("CREATE (a:T {id: 1, name: 'a'}), (b:T {id: 2, name: 'b'})")
    g.cypher("MATCH (a:T {id: 1}), (b:T {id: 2}) CREATE (b)-[:E]->(a)")
    assert census(g) == ([("a", "T"), ("b", "T")], [("b", "E", "a")])


def test_control_a_later_statement_rebinds_the_same_variable_name(g):
    """Variable scope ends with the statement — the second CREATE makes a node."""
    g.cypher("CREATE (a:T {id: 1, name: 'a'})")
    g.cypher("CREATE (a:T {id: 2, name: 'b'})")
    assert census(g)[0] == [("a", "T"), ("b", "T")]


def test_an_already_bound_variable_is_referenced_not_recreated(g):
    """Documented behaviour, pinned so a change to it is deliberate.

    Neo4j raises "Variable `a` already declared" when a bound variable
    reappears carrying a label or properties. This engine silently *references*
    the bound node and drops the second occurrence's label/properties — and
    does so identically whether the binding came from a preceding MATCH or from
    an earlier part of the same CREATE. Making the two agree is the point; the
    reference-vs-error question is a separate semantics decision.
    """
    g.cypher("CREATE (a:T {id: 1, name: 'a'})")
    g.cypher("MATCH (a:T) CREATE (a:T {id: 99, name: 'z'})")
    assert census(g)[0] == [("a", "T")]

    g2 = kglite.KnowledgeGraph()
    g2.cypher("CREATE (a:T {id: 1, name: 'a'}), (a:T {id: 2, name: 'b'})")
    assert census(g2)[0] == [("a", "T")]


def test_a_named_but_previously_unknown_endpoint_creates_and_binds_that_node(g):
    """A CREATE endpoint is a node to *create*, whether or not it is named.

    ``zzz`` is not bound by anything before this statement, so the node pass
    creates it and binds the name — the same treatment an anonymous endpoint
    now gets, and the same as Neo4j. There is therefore no reachable
    "unbound endpoint" input for CREATE; that arm of the endpoint resolver is
    a structural assertion, not a user-facing error.
    """
    g.cypher("CREATE (b:T {id: 1, name: 'b'})-[:E]->(zzz {name: 'z'})")
    assert census(g) == ([("b", "T"), ("z", "Node")], [("b", "E", "z")])
    # And the name is bound for the rest of the statement.
    rows = g.cypher("CREATE (p:T {id: 2, name: 'p'})-[:E]->(q {name: 'q'}), (q)-[:E]->(p) RETURN q.name AS q").to_list()
    assert rows == [{"q": "q"}]
    assert ("q", "E", "p") in census(g)[1]


# ── the created bindings are usable downstream ───────────────────────────


def test_created_anonymous_endpoint_is_countable_in_the_same_statement(g):
    rows = g.cypher(
        "CREATE (a:T {id: 1, name: 'a'})-[r:E]->(:U {name: 'u'}) RETURN a.title AS a, type(r) AS t"
    ).to_list()
    assert rows == [{"a": "a", "t": "E"}]


def test_multi_part_create_survives_save_and_reload(g, tmp_path):
    g.cypher("CREATE (a:T {id: 5, name: 'a'}), (b:T {id: 7, name: 'b'}), (b)-[:E]->(a)")
    path = tmp_path / "create_bindings.kgl"
    g.save(str(path))
    reloaded = kglite.load(str(path))
    assert census(reloaded) == ([("a", "T"), ("b", "T")], [("b", "E", "a")])
