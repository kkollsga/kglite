"""Role-scoped writes (P5).

`write_scope=[...]` on `KnowledgeGraph.cypher` / `Session.execute` restricts
Cypher CREATE/SET to a node-type whitelist (integrity, not secrecy). A coding
role may write its own types but not, say, research-owned `Algorithm` nodes.
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
