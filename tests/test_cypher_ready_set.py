"""`CALL ready_set(...)` — dependency-frontier topological procedure (P6).

Over a DAG on a chosen edge type, return the nodes whose dependencies (their
outgoing-E neighbours) all satisfy a `done` predicate. General graph op (build
ordering, scheduling, dataflow), opt-in like pagerank/louvain.
"""

import pytest

import kglite


@pytest.fixture
def dag():
    # (x)-[:DEPENDS_ON]->(y) means x depends on y.
    #   A -> B -> C ,  A -> C ,  D -> B
    # C is done. So B (deps: C done) is ready; A, D are not (B not done).
    g = kglite.KnowledgeGraph()
    for n, s in [("A", "todo"), ("B", "todo"), ("C", "done"), ("D", "todo")]:
        g.cypher(f"CREATE (:Task {{id:'{n}', status:'{s}'}})")
    for a, b in [("B", "C"), ("A", "B"), ("A", "C"), ("D", "B")]:
        g.cypher(f"MATCH (x:Task {{id:'{a}'}}),(y:Task {{id:'{b}'}}) CREATE (x)-[:DEPENDS_ON]->(y)")
    return g


_Q = (
    "CALL ready_set({relationship:'DEPENDS_ON', done:'n.status = \"done\"'}) "
    "YIELD node, dependency_count RETURN node.id AS id, dependency_count AS deps ORDER BY id"
)


def test_initial_ready_frontier(dag):
    assert dag.cypher(_Q).to_dicts() == [{"id": "B", "deps": 1}]


def test_frontier_advances_as_work_completes(dag):
    dag.cypher("MATCH (n:Task {id:'B'}) SET n.status='done'")
    ready = dag.cypher(_Q).to_dicts()
    # B done → A (deps B,C both done) and D (dep B done) become ready.
    assert [r["id"] for r in ready] == ["A", "D"]


def test_root_with_no_deps_is_ready_when_not_done():
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Task {id:'root', status:'todo'})")
    ready = g.cypher(
        "CALL ready_set({relationship:'DEPENDS_ON', done:'n.status = \"done\"'}) YIELD node RETURN node.id AS id"
    ).to_dicts()
    assert ready == [{"id": "root"}]


def test_done_nodes_excluded(dag):
    # C is done — it must never appear in the ready set.
    ids = [r["id"] for r in dag.cypher(_Q).to_dicts()]
    assert "C" not in ids


def test_node_type_scoping(dag):
    # Add an unrelated type; scoping to Task must not emit it.
    dag.cypher("CREATE (:Note {id:'n1', status:'todo'})")
    ready = dag.cypher(
        "CALL ready_set({relationship:'DEPENDS_ON', done:'n.status = \"done\"', node_type:'Task'}) "
        "YIELD node RETURN node.id AS id ORDER BY id"
    ).to_dicts()
    assert all(r["id"] != "n1" for r in ready)


def test_missing_done_predicate_errors(dag):
    with pytest.raises(Exception, match="done"):
        dag.cypher("CALL ready_set({relationship:'DEPENDS_ON'}) YIELD node RETURN node")


def test_unknown_config_key_rejected(dag):
    with pytest.raises(Exception, match="unknown config key"):
        dag.cypher(
            "CALL ready_set({relationship:'DEPENDS_ON', done:'n.status=\"done\"', bogus:1}) YIELD node RETURN node"
        )


def test_unknown_relationship_type_is_refused_not_answered(dag):
    """A typo'd dependency edge used to report the *whole graph* as ready.

    `ready_set` answers "are all my dependencies done?" with a universal
    quantifier over the node's outgoing edges of that type. A type matching no
    edges makes every dependency list empty and every quantifier vacuously
    true, so the typo widens the frontier to everything — the most permissive
    possible answer to a question asked to decide what may safely start.
    """
    with pytest.raises(Exception) as excinfo:
        dag.cypher(
            "CALL ready_set({relationship:'DEPENDS_O', done:'n.status = \"done\"'}) YIELD node RETURN node.id AS id"
        )
    message = str(excinfo.value)
    assert "unknown relationship type 'DEPENDS_O'" in message
    assert "Did you mean 'DEPENDS_ON'?" in message, f"no suggestion in: {message}"
    assert "Valid types: DEPENDS_ON" in message, f"no valid set in: {message}"
    # The alias is the same procedure and refuses identically.
    with pytest.raises(Exception, match="unknown relationship type"):
        dag.cypher(
            "CALL dependency_frontier({relationship:'NOPE', done:'n.status = \"done\"'}) "
            "YIELD node RETURN node.id AS id"
        )


def test_unknown_relationship_type_only_warns_on_procedures_that_fail_closed(dag, capfd):
    """Everything else degrades toward an empty answer, which is visibly wrong.

    pagerank on an edgeless subgraph returns uniform scores for every node —
    a result the caller can see is degenerate — so an unknown type there is a
    warning, and the rows are unchanged.
    """
    before = dag.cypher("CALL pagerank() YIELD node RETURN count(*) AS c").to_dicts()
    capfd.readouterr()
    after = dag.cypher("CALL pagerank({relationship:'NOPE'}) YIELD node RETURN count(*) AS c").to_dicts()
    assert after == before, "a warning must not change the result"
    assert "unknown relationship type 'NOPE'" in capfd.readouterr().err


def test_unknown_node_type_warns_with_a_suggestion(dag, capfd):
    capfd.readouterr()
    rows = dag.cypher(
        "CALL ready_set({relationship:'DEPENDS_ON', node_type:'Tsk', "
        "done:'n.status = \"done\"'}) YIELD node RETURN node.id AS id"
    ).to_dicts()
    # Unknown node types already fail *closed* — no candidates, no rows.
    assert rows == []
    warning = capfd.readouterr().err
    assert "unknown node type 'Tsk'" in warning
    assert "Did you mean 'Task'?" in warning
