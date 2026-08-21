"""Role-scoped writes (P5) — the write_scope perimeter.

`write_scope=[...]` on `KnowledgeGraph.cypher` / `Session.execute` /
`Transaction.cypher` restricts a Cypher statement's mutations to a node-type
whitelist (integrity, not secrecy). A coding role may write its own types but
not, say, research-owned `Algorithm` nodes.

**The boundary, exactly:**

* **Node writes** — `CREATE`, `MERGE`'s create arm, `SET n.p`, `SET n += {…}`,
  `SET n:Label`, `REMOVE n.p`, `REMOVE n:Label`, `DELETE n`, `DETACH DELETE n`,
  and index/constraint DDL for a node type — are judged by the node's **stored
  type**, never by a pattern label, so label smuggling cannot widen the scope.
* **Relationship writes** — `CREATE (a)-[:R]->(b)`, `DELETE r`, `SET r.p`,
  `REMOVE r.p` — are allowed iff **at least one endpoint's stored type is in
  scope**. Linking a runtime node to a matched, out-of-scope node does not
  mutate that node, and that pattern is load-bearing (see
  `test_edge_to_matched_out_of_scope_endpoint_allowed`); an edge between two
  out-of-scope nodes is a write the role owns nothing in, and is refused.
* **`DETACH DELETE` collateral** — the incident edges a detach removes are
  authorized by the node-delete check, not re-checked per far endpoint.
* **Outside the perimeter** (documented, deliberate): relationship-constraint
  DDL, `db.cdc.enable`/`db.cdc.disable`, and the bulk loaders
  (`add_nodes`/`add_connections`) — `write_scope` is a per-Cypher-execution
  concept and does not reach the Python loader API.
* An **empty** whitelist (`write_scope=[]`) denies every mutation.
"""

import pytest

import kglite

SCOPE = ["Plan", "Task"]


@pytest.fixture
def kg():
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Plan {id: 1})")
    g.cypher("CREATE (:Algorithm {id: 1})")
    return g


def test_in_scope_create_and_set_ok(kg):
    kg.cypher("CREATE (:Task {id: 1})", write_scope=SCOPE)
    kg.cypher("MATCH (n:Plan) SET n.status = 'done'", write_scope=SCOPE)
    assert kg.cypher("MATCH (n:Task) RETURN count(n) AS c").to_dicts() == [{"c": 1}]
    assert kg.cypher("MATCH (n:Plan) RETURN n.status AS s").to_dicts() == [{"s": "done"}]


def test_out_of_scope_create_rejected(kg):
    with pytest.raises(Exception, match="write scope"):
        kg.cypher("CREATE (:Algorithm {id: 2})", write_scope=SCOPE)
    # the rejected CREATE must not have landed
    assert kg.cypher("MATCH (n:Algorithm) RETURN count(n) AS c").to_dicts() == [{"c": 1}]


def test_out_of_scope_set_rejected(kg):
    with pytest.raises(Exception, match="write scope"):
        kg.cypher("MATCH (n:Algorithm) SET n.note = 'x'", write_scope=SCOPE)


def test_edge_to_matched_out_of_scope_endpoint_allowed(kg):
    # Linking a runtime node to a *matched* (existing) out-of-scope node does
    # not mutate that node — it must be allowed (the central agent-contract
    # pattern: Task -[:IMPLEMENTS_SPEC]-> AlgorithmSpec). Regression for the
    # 0.12.0 over-aggressive endpoint guard (SimulatoRS report 2026-06-25).
    kg.cypher("CREATE (:Task {id: 2})", write_scope=SCOPE)
    kg.cypher(
        "MATCH (t:Task {id: 2}), (a:Algorithm {id: 1}) CREATE (t)-[:USES]->(a)",
        write_scope=SCOPE,
    )
    kg.cypher(
        "MATCH (t:Task {id: 2}), (a:Algorithm {id: 1}) MERGE (t)-[:ALSO_USES]->(a)",
        write_scope=SCOPE,
    )
    assert kg.cypher("MATCH (:Task)-[r]->(:Algorithm) RETURN count(r) AS c").to_dicts() == [{"c": 2}]


def test_edge_creating_new_out_of_scope_endpoint_rejected(kg):
    # But *creating* a new out-of-scope node as an edge endpoint is still
    # blocked (the node CREATE goes through the guarded path).
    kg.cypher("CREATE (:Task {id: 3})", write_scope=SCOPE)
    with pytest.raises(Exception, match="write scope"):
        kg.cypher(
            "MATCH (t:Task {id: 3}) CREATE (t)-[:USES]->(:Algorithm {id: 99})",
            write_scope=SCOPE,
        )


def test_merge_node_create_is_scoped(kg):
    # MERGE that would create an out-of-scope node is rejected (better than the
    # 0.12.0 note said — MERGE *is* scoped).
    with pytest.raises(Exception, match="write scope"):
        kg.cypher("MERGE (:Algorithm {id: 42})", write_scope=SCOPE)


def test_transaction_cypher_enforces_scope(kg):
    tx = kg.begin()
    tx.cypher("CREATE (:Task {id: 7})", write_scope=SCOPE)
    with pytest.raises(Exception, match="write scope"):
        tx.cypher("CREATE (:Algorithm {id: 7})", write_scope=SCOPE)


def test_no_scope_is_unrestricted(kg):
    # Default (no write_scope) keeps the permissive behaviour.
    kg.cypher("CREATE (:Algorithm {id: 5})")
    assert kg.cypher("MATCH (n:Algorithm) RETURN count(n) AS c").to_dicts() == [{"c": 2}]


def test_scope_does_not_leak_across_calls(kg):
    with pytest.raises(Exception, match="write scope"):
        kg.cypher("CREATE (:Algorithm {id: 7})", write_scope=SCOPE)
    # A later unscoped call is unaffected by the prior scoped (and failed) one.
    kg.cypher("CREATE (:Algorithm {id: 8})")
    assert kg.cypher("MATCH (n:Algorithm) RETURN count(n) AS c").to_dicts() == [{"c": 2}]


def test_session_execute_enforces_scope(kg):
    s = kg.session()
    s.execute("CREATE (:Task {id: 9})", write_scope=SCOPE)
    with pytest.raises(Exception, match="write scope"):
        s.execute("CREATE (:Algorithm {id: 9})", write_scope=SCOPE)


def test_index_ddl_respects_the_write_scope(kg):
    """An index is schema state for one node type, so `CREATE INDEX` /
    `DROP INDEX` on a type outside the whitelist is a scope violation like any
    other write to it."""
    kg.cypher("CREATE INDEX FOR (n:Plan) ON (n.status)", write_scope=SCOPE)
    assert kg.has_index("Plan", "status")

    with pytest.raises(Exception, match="write scope"):
        kg.cypher("CREATE INDEX FOR (n:Algorithm) ON (n.id)", write_scope=SCOPE)
    assert not kg.has_index("Algorithm", "id")

    kg.create_index("Algorithm", "id")
    with pytest.raises(Exception, match="write scope"):
        kg.cypher("DROP INDEX Algorithm.id", write_scope=SCOPE)
    assert kg.has_index("Algorithm", "id")


def test_show_indexes_is_unaffected_by_the_write_scope(kg):
    """`SHOW INDEXES` is a read; a write whitelist restricts mutations, not
    visibility (integrity, not secrecy)."""
    kg.create_index("Algorithm", "id")
    rows = kg.cypher("SHOW INDEXES", write_scope=SCOPE).to_dicts()
    assert [row["name"] for row in rows] == ["Algorithm.id"]


# ── node writes beyond CREATE/SET: DELETE / REMOVE / SET label ───────────


def test_out_of_scope_delete_rejected(kg):
    with pytest.raises(Exception, match="write scope"):
        kg.cypher("MATCH (n:Algorithm) DELETE n", write_scope=SCOPE)
    assert kg.cypher("MATCH (n:Algorithm) RETURN count(n) AS c").to_dicts() == [{"c": 1}]


def test_out_of_scope_detach_delete_rejected(kg):
    kg.cypher("MATCH (p:Plan {id: 1}), (a:Algorithm {id: 1}) CREATE (p)-[:USES]->(a)")
    with pytest.raises(Exception, match="write scope"):
        kg.cypher("MATCH (n:Algorithm) DETACH DELETE n", write_scope=SCOPE)
    assert kg.cypher("MATCH (n:Algorithm) RETURN count(n) AS c").to_dicts() == [{"c": 1}]
    assert kg.cypher("MATCH ()-[r]->() RETURN count(r) AS c").to_dicts() == [{"c": 1}]


def test_detach_delete_of_an_in_scope_node_takes_its_edges_with_it(kg):
    """Incident-edge collateral is authorized by the node-delete check.

    The far endpoint's type is deliberately NOT re-checked: a detach of a node
    the role owns removes that node's relationships, whatever they point at.
    """
    kg.cypher("MATCH (p:Plan {id: 1}), (a:Algorithm {id: 1}) CREATE (p)-[:USES]->(a)")
    kg.cypher("MATCH (n:Plan {id: 1}) DETACH DELETE n", write_scope=SCOPE)
    assert kg.cypher("MATCH (n:Plan) RETURN count(n) AS c").to_dicts() == [{"c": 0}]
    assert kg.cypher("MATCH ()-[r]->() RETURN count(r) AS c").to_dicts() == [{"c": 0}]
    assert kg.cypher("MATCH (n:Algorithm) RETURN count(n) AS c").to_dicts() == [{"c": 1}]


def test_a_refused_delete_leaves_every_row_untouched(kg):
    """Refusal happens at collection, before the first node is unlinked — the
    in-scope rows of the same statement must not be deleted either."""
    kg.cypher("CREATE (:Task {id: 1})", write_scope=SCOPE)
    with pytest.raises(Exception, match="write scope"):
        kg.cypher("MATCH (n) DETACH DELETE n", write_scope=SCOPE)
    assert kg.cypher("MATCH (n) RETURN count(n) AS c").to_dicts() == [{"c": 3}]


def test_out_of_scope_remove_property_rejected(kg):
    kg.cypher("MATCH (n:Algorithm) SET n.note = 'keep'")
    with pytest.raises(Exception, match="write scope"):
        kg.cypher("MATCH (n:Algorithm) REMOVE n.note", write_scope=SCOPE)
    assert kg.cypher("MATCH (n:Algorithm) RETURN n.note AS note").to_dicts() == [{"note": "keep"}]


def test_out_of_scope_remove_label_rejected(kg):
    kg.cypher("MATCH (n:Algorithm) SET n:Hot")
    with pytest.raises(Exception, match="write scope"):
        kg.cypher("MATCH (n:Algorithm) REMOVE n:Hot", write_scope=SCOPE)
    assert kg.cypher("MATCH (n:Hot) RETURN count(n) AS c").to_dicts() == [{"c": 1}]


def test_out_of_scope_set_label_rejected(kg):
    with pytest.raises(Exception, match="write scope"):
        kg.cypher("MATCH (n:Algorithm) SET n:Hot", write_scope=SCOPE)
    assert kg.cypher("MATCH (n:Hot) RETURN count(n) AS c").to_dicts() == [{"c": 0}]


# ── relationship writes: one endpoint in scope is enough ─────────────────


@pytest.fixture
def linked(kg):
    """`(:Algorithm)-[:CALLS]->(:Algorithm)` (neither endpoint in scope) and
    `(:Plan)-[:USES]->(:Algorithm)` (one endpoint in scope), built unscoped."""
    kg.cypher("CREATE (:Algorithm {id: 2})")
    kg.cypher("MATCH (a:Algorithm {id: 1}), (b:Algorithm {id: 2}) CREATE (a)-[:CALLS {since: 1}]->(b)")
    kg.cypher("MATCH (p:Plan {id: 1}), (a:Algorithm {id: 1}) CREATE (p)-[:USES {since: 1}]->(a)")
    return kg


def _rel_count(kg, rel_type):
    return kg.cypher(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS c").to_dicts()[0]["c"]


def _since(kg, rel_type):
    return kg.cypher(f"MATCH ()-[r:{rel_type}]->() RETURN r.since AS s").to_dicts()[0]["s"]


def test_delete_relationship_between_two_out_of_scope_endpoints_rejected(linked):
    with pytest.raises(Exception, match="write scope"):
        linked.cypher("MATCH ()-[r:CALLS]->() DELETE r", write_scope=SCOPE)
    assert _rel_count(linked, "CALLS") == 1


def test_delete_relationship_with_one_endpoint_in_scope_allowed(linked):
    linked.cypher("MATCH ()-[r:USES]->() DELETE r", write_scope=SCOPE)
    assert _rel_count(linked, "USES") == 0


def test_set_relationship_property_between_two_out_of_scope_endpoints_rejected(linked):
    with pytest.raises(Exception, match="write scope"):
        linked.cypher("MATCH ()-[r:CALLS]->() SET r.since = 2", write_scope=SCOPE)
    assert _since(linked, "CALLS") == 1


def test_set_relationship_property_with_one_endpoint_in_scope_allowed(linked):
    linked.cypher("MATCH ()-[r:USES]->() SET r.since = 2", write_scope=SCOPE)
    assert _since(linked, "USES") == 2


def test_remove_relationship_property_between_two_out_of_scope_endpoints_rejected(linked):
    with pytest.raises(Exception, match="write scope"):
        linked.cypher("MATCH ()-[r:CALLS]->() REMOVE r.since", write_scope=SCOPE)
    assert _since(linked, "CALLS") == 1


def test_remove_relationship_property_with_one_endpoint_in_scope_allowed(linked):
    linked.cypher("MATCH ()-[r:USES]->() REMOVE r.since", write_scope=SCOPE)
    assert _since(linked, "USES") is None


def test_edge_forgery_between_two_out_of_scope_nodes_rejected(linked):
    """The eval's `CREATE (v)-[:FORGED]->(c)` repro: neither endpoint is a type
    the role may write, so the relationship is not the role's to create."""
    with pytest.raises(Exception, match="write scope"):
        linked.cypher(
            "MATCH (a:Algorithm {id: 1}), (b:Algorithm {id: 2}) CREATE (a)-[:FORGED]->(b)",
            write_scope=SCOPE,
        )
    assert _rel_count(linked, "FORGED") == 0


def test_merge_edge_between_two_out_of_scope_nodes_rejected(linked):
    with pytest.raises(Exception, match="write scope"):
        linked.cypher(
            "MATCH (a:Algorithm {id: 1}), (b:Algorithm {id: 2}) MERGE (a)-[:FORGED]->(b)",
            write_scope=SCOPE,
        )
    assert _rel_count(linked, "FORGED") == 0


def test_the_relationship_refusal_names_both_endpoint_types(linked):
    with pytest.raises(Exception) as excinfo:
        linked.cypher(
            "MATCH (a:Algorithm {id: 1}), (b:Algorithm {id: 2}) CREATE (a)-[:FORGED]->(b)",
            write_scope=SCOPE,
        )
    message = str(excinfo.value)
    assert "write scope violation: relationship 'FORGED' connects 'Algorithm' to 'Algorithm'" in message
    assert "neither endpoint type is in the allowed write set (Plan, Task)" in message


# ── smuggling vectors: FOREACH and MERGE ─────────────────────────────────


def test_foreach_delete_of_an_out_of_scope_node_rejected(kg):
    with pytest.raises(Exception, match="write scope"):
        kg.cypher(
            "MATCH (n:Algorithm) WITH collect(n) AS ns FOREACH (x IN ns | DETACH DELETE x)",
            write_scope=SCOPE,
        )
    assert kg.cypher("MATCH (n:Algorithm) RETURN count(n) AS c").to_dicts() == [{"c": 1}]


def test_foreach_edge_create_between_out_of_scope_nodes_rejected(linked):
    with pytest.raises(Exception, match="write scope"):
        linked.cypher(
            "MATCH (a:Algorithm {id: 1}), (b:Algorithm {id: 2}) FOREACH (i IN [1] | CREATE (a)-[:FORGED]->(b))",
            write_scope=SCOPE,
        )
    assert _rel_count(linked, "FORGED") == 0


def test_foreach_remove_and_set_label_on_an_out_of_scope_node_rejected(kg):
    kg.cypher("MATCH (n:Algorithm) SET n.note = 'keep'")
    for body in ("REMOVE n.note", "REMOVE n:Hot", "SET n:Hot"):
        with pytest.raises(Exception, match="write scope"):
            kg.cypher(f"MATCH (n:Algorithm) FOREACH (i IN [1] | {body})", write_scope=SCOPE)
    assert kg.cypher("MATCH (n:Algorithm) RETURN n.note AS note").to_dicts() == [{"note": "keep"}]
    assert kg.cypher("MATCH (n:Hot) RETURN count(n) AS c").to_dicts() == [{"c": 0}]


# ── an empty whitelist denies everything ─────────────────────────────────


def test_empty_scope_denies_every_mutation(linked):
    for query in (
        "CREATE (:Task {id: 99})",
        "MATCH (n:Plan) SET n.status = 'x'",
        "MATCH (n:Plan) SET n:Hot",
        "MATCH (n:Plan) REMOVE n.status",
        "MATCH (n:Plan) DETACH DELETE n",
        "MATCH ()-[r:USES]->() DELETE r",
        "MATCH ()-[r:USES]->() SET r.since = 9",
        "MATCH ()-[r:USES]->() REMOVE r.since",
        "MATCH (p:Plan {id: 1}), (a:Algorithm {id: 1}) CREATE (p)-[:MORE]->(a)",
    ):
        with pytest.raises(Exception, match="write scope"):
            linked.cypher(query, write_scope=[])
    assert linked.cypher("MATCH (n) RETURN count(n) AS c").to_dicts() == [{"c": 3}]
    assert linked.cypher("MATCH ()-[r]->() RETURN count(r) AS c").to_dicts() == [{"c": 2}]
