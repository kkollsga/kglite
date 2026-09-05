"""Absolute WAL recovery goldens; children exit before checkpoint/teardown."""

import os
import subprocess
import sys

import pytest

import kglite


def crash(path, storage, body):
    script = f"""import kglite, os
path = {str(path)!r}
g = kglite.open(path, durable='normal', storage={storage!r})
g.cypher('CREATE (:Item {{id: 1}}), (:Item {{id: 2}})')
{body}
os._exit(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def edge_rows(graph):
    return graph.cypher(
        "MATCH (a:Item)-[r:LINK]->(b:Item) RETURN a.id AS a,b.id AS b,r.n AS n ORDER BY a,b,n"
    ).to_list()


@pytest.mark.parametrize("storage", ["memory", "mapped"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_parallel_equal_maps_selected_update_delete_and_cdc_toggle(tmp_path, storage, checkpoint):
    path = tmp_path / "parallel.kgl"
    body = """g.cypher('MATCH (a:Item {id: 1}),(b:Item {id: 2}) CREATE '
         '(a)-[:LINK {n: 1}]->(b),(a)-[:LINK {n: 1}]->(b),(a)-[:LINK {n: 2}]->(b)')
"""
    if checkpoint:
        body += "g.save()\n"
    body += """g.cypher('CALL db.cdc.enable()')
g.cypher('MATCH ()-[r:LINK]->() WHERE r.n=2 SET r.n=3')
g.cypher('CALL db.cdc.disable()')
"""
    # Updating and deleting a different parallel member can accidentally mask
    # legacy single-edge replay's wrong target, so the checkpoint arm pins the
    # update alone; the no-checkpoint arm also exercises deletion.
    expected_values = [1, 1, 3]
    if not checkpoint:
        body += "g.cypher('MATCH ()-[r:LINK]->() WHERE r.n=1 WITH r LIMIT 1 DELETE r')\n"
        expected_values = [1, 3]
    body += (
        "assert g.cypher('MATCH ()-[r:LINK]->() RETURN r.n AS n ORDER BY n').to_list() == "
        + repr([{"n": value} for value in expected_values])
        + "\n"
    )
    crash(path, storage, body)
    graph = kglite.open(str(path), durable="normal")
    assert edge_rows(graph) == [{"a": 1, "b": 2, "n": value} for value in expected_values]


@pytest.mark.parametrize("storage", ["memory", "mapped"])
def test_recreated_endpoint_and_self_loop_recover_exactly(tmp_path, storage):
    path = tmp_path / "recreated.kgl"
    crash(
        path,
        storage,
        """g.cypher('MATCH (a:Item {id: 1}),(b:Item {id: 2}) SET a:Old CREATE (a)-[:LINK {n: 1}]->(b)')
g.save()
g.cypher('MATCH (a:Item {id: 1}) DETACH DELETE a')
g.cypher('CREATE (:Item {id: 1})')
g.cypher('MATCH (a:Item {id: 1}),(b:Item {id: 2}) CREATE '
         '(a)-[:LINK {n: 7}]->(b),(a)-[:LINK {n: 7}]->(b),(a)-[:LINK {n: 8}]->(a)')
""",
    )
    graph = kglite.open(str(path), durable="normal")
    assert edge_rows(graph) == [
        {"a": 1, "b": 1, "n": 8},
        {"a": 1, "b": 2, "n": 7},
        {"a": 1, "b": 2, "n": 7},
    ]
    assert graph.cypher("MATCH (n:Item {id: 1}) RETURN labels(n) AS labels").to_list() == [{"labels": ["Item"]}]


def test_duplicate_checkpoint_adoption_fails_without_sidecar_changes(tmp_path):
    path = tmp_path / "duplicates.kgl"
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:Item {id: 1}),(:Item {id: 1})")
    graph.save(str(path))
    del graph
    before = path.read_bytes()
    sidecar = tmp_path / "duplicates.kgl-wal"
    with pytest.raises(Exception, match="duplicate logical node identity"):
        kglite.open(str(path), durable="normal")
    assert path.read_bytes() == before
    assert not sidecar.exists()
    plain = kglite.load(str(path))
    assert plain.cypher("MATCH (n:Item) RETURN count(*) AS c").to_list() == [{"c": 2}]
