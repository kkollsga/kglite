"""Bolt column-order and typed graph-value interface contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kglite import KnowledgeGraph
from tests.conftest import _bolt_binary_available, _spawn_bolt_server, _teardown_bolt_server

neo4j = pytest.importorskip("neo4j")

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tests" / "api-baselines" / "bolt-interface.json"
pytestmark = [pytest.mark.bolt, pytest.mark.skipif(not _bolt_binary_available(), reason="Bolt binary not built")]


def _cell(value):
    if isinstance(value, neo4j.graph.Node):
        return {
            "kind": "node",
            "id": int(value.element_id),
            "labels": sorted(value.labels),
            "properties": dict(sorted(dict(value).items())),
        }
    if isinstance(value, neo4j.graph.Relationship):
        return {
            "kind": "relationship",
            "id": int(value.element_id),
            "start": int(value.start_node.element_id),
            "end": int(value.end_node.element_id),
            "type": value.type,
            "properties": dict(sorted(dict(value).items())),
        }
    if isinstance(value, neo4j.graph.Path):
        return {
            "kind": "path",
            "nodes": [_cell(node) for node in value.nodes],
            "relationships": [_cell(rel) for rel in value.relationships],
        }
    return value


def _run(session, query: str) -> dict:
    result = session.run(query)
    keys = list(result.keys())
    rows = [[_cell(value) for value in record.values()] for record in result]
    return {"keys": keys, "rows": rows}


def capture_bolt_contract(path: Path) -> dict:
    graph = KnowledgeGraph()
    graph.cypher("CREATE (:N {id: 1, name: 'a'}), (:N {id: 2, name: 'b'})")
    graph.cypher(
        "MATCH (a:N {id: 1}), (b:N {id: 2}) CREATE (a)-[:R {tag: 'first'}]->(b), (a)-[:R {tag: 'second'}]->(b)"
    )
    graph.save(str(path))

    proc, url = _spawn_bolt_server(path, readonly=True)
    try:
        with neo4j.GraphDatabase.driver(url, auth=("neo4j", "password")) as driver:
            with driver.session() as session:
                return {
                    "alias_order": _run(session, "RETURN 1 AS z, 'x' AS a, null AS m"),
                    "empty_result": _run(session, "MATCH (n:Missing) RETURN n.id AS id, n.name AS name"),
                    "procedure_columns": _run(
                        session,
                        "CALL db.labels() YIELD label RETURN label AS kind ORDER BY kind",
                    ),
                    "typed_path": _run(
                        session,
                        "MATCH p=(a:N {id: 1})-[r:R]->(b:N {id: 2}) "
                        "RETURN a AS source, r AS edge, p AS path ORDER BY r.tag",
                    ),
                    "union_columns": _run(session, "RETURN 1 AS value UNION ALL RETURN 2 AS value"),
                }
    finally:
        _teardown_bolt_server(proc)


def test_bolt_interface_matches_reviewed_baseline(tmp_path):
    expected = json.loads(BASELINE.read_text())
    actual = capture_bolt_contract(tmp_path / "bolt-interface.kgl")
    assert actual == expected, "Bolt columns or typed Node/Relationship/Path shapes drifted"
