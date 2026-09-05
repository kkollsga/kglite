"""Compaction must rebuild physical UNIQUE occupants without changing declarations."""

import pytest

import kglite
from kglite import KnowledgeGraph

KINDS = {
    "unique": "p.email IS UNIQUE",
    "composite": "(p.tenant, p.email) IS UNIQUE",
    "node_key": "(p.tenant, p.email) IS NODE KEY",
}
EXPECTED = [{"id": 1, "email": "b"}, {"id": 2, "email": "c"}]


def fixture(kind, mode="memory", path=None):
    graph = (
        KnowledgeGraph()
        if mode == "memory"
        else KnowledgeGraph(storage=mode, **({"path": str(path)} if mode == "disk" else {}))
    )
    graph.set_auto_vacuum(None)
    for identifier, email in enumerate("abc"):
        graph.cypher("CREATE (:Person {id:$id, tenant:'t', email:$email})", params={"id": identifier, "email": email})
    graph.cypher(f"CREATE CONSTRAINT person_key FOR (p:Person) REQUIRE {KINDS[kind]}")
    return graph


def rows(graph):
    return graph.cypher("MATCH (p:Person) RETURN p.id AS id, p.email AS email ORDER BY id").to_list()


def assert_claim(graph, operation):
    assert rows(graph) == EXPECTED
    query = "MATCH (p:Person {id:$id}) SET p.email = 'b'"
    if operation == "own_value":
        graph.cypher(query, params={"id": 1})
    else:
        with pytest.raises(Exception, match="constraint"):
            graph.cypher(query, params={"id": 2})
    assert rows(graph) == EXPECTED


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode", ["memory", "mapped", "disk"])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("operation", ["own_value", "duplicate"])
def test_vacuum_rebuilds_unique_occupants(kind, mode, compact, operation, tmp_path):
    graph = fixture(kind, mode, tmp_path / "disk")
    graph.cypher("MATCH (p:Person {id:0}) DETACH DELETE p")
    declarations = graph.cypher("SHOW CONSTRAINTS").to_list()
    if compact:
        graph.vacuum()
    assert graph.cypher("SHOW CONSTRAINTS").to_list() == declarations
    assert_claim(graph, operation)


@pytest.mark.parametrize("kind", KINDS)
def test_vacuum_empty_constrained_type_keeps_enforcement(kind):
    graph = fixture(kind)
    declarations = graph.cypher("SHOW CONSTRAINTS").to_list()
    graph.cypher("MATCH (p:Person) DETACH DELETE p")
    graph.vacuum()
    graph.reindex()
    assert graph.cypher("SHOW CONSTRAINTS").to_list() == declarations
    graph.cypher("CREATE (:Person {id:3, tenant:'t', email:'b'})")
    with pytest.raises(Exception, match="constraint"):
        graph.cypher("CREATE (:Person {id:4, tenant:'t', email:'b'})")
    assert rows(graph) == [{"id": 3, "email": "b"}]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("operation", ["own_value", "duplicate"])
def test_loaded_vacuum_enforces_before_resave(kind, deferred, operation, tmp_path, monkeypatch):
    graph = fixture(kind)
    graph.save(str(tmp_path / "before.kgl"))
    monkeypatch.setenv("KGLITE_DEFER_INDEX_REBUILD", "1" if deferred else "0")
    loaded = kglite.load(str(tmp_path / "before.kgl"))
    loaded.set_auto_vacuum(None)
    loaded.cypher("MATCH (p:Person {id:0}) DETACH DELETE p")
    loaded.vacuum()
    assert_claim(loaded, operation)
    declarations = loaded.cypher("SHOW CONSTRAINTS").to_list()
    loaded.save(str(tmp_path / "after.kgl"))
    reloaded = kglite.load(str(tmp_path / "after.kgl"))
    assert reloaded.cypher("SHOW CONSTRAINTS").to_list() == declarations
    assert_claim(reloaded, operation)


@pytest.mark.parametrize("kind", KINDS)
def test_vacuum_rollback_value_reuse_and_held_snapshot(kind):
    graph = fixture(kind)
    snapshot = graph.freeze()
    graph.cypher("MATCH (p:Person {id:0}) DETACH DELETE p")
    graph.vacuum()
    with pytest.raises(Exception, match="constraint"):
        graph.cypher("MATCH (p:Person) SET p.email = 'z'")
    assert rows(graph) == EXPECTED
    graph.cypher("CREATE (:Person {id:3, tenant:'t', email:'z'})")
    graph.cypher("CREATE (:Person {id:4, tenant:'t', email:'a'})")
    with pytest.raises(Exception, match="constraint"):
        graph.cypher("CREATE (:Person {id:5, tenant:'t', email:'b'})")
    assert rows(graph) == EXPECTED + [{"id": 3, "email": "z"}, {"id": 4, "email": "a"}]
    assert rows(snapshot) == [{"id": 0, "email": "a"}] + EXPECTED


def test_vacuum_incomplete_composite_unique_and_noop():
    graph = fixture("composite")
    graph.cypher("MATCH (p:Person {id:0}) DETACH DELETE p")
    graph.cypher("CREATE (:Person {id:3, email:'b'}), (:Person {id:4, email:'b'})")
    graph.vacuum()
    graph.vacuum()
    graph.cypher("MATCH (p:Person {id:1}) SET p.email = 'b'")
    with pytest.raises(Exception, match="constraint"):
        graph.cypher("MATCH (p:Person {id:3}) SET p.tenant = 't'")
    assert rows(graph) == EXPECTED + [{"id": 3, "email": "b"}, {"id": 4, "email": "b"}]
