"""Compatibility fixtures captured with the published KGLite 0.13.3 wheel."""

from pathlib import Path

import kglite

FIXTURES = Path(__file__).parent / "fixtures"
LEGACY_KGL = FIXTURES / "legacy_0_13_3.kgl"
LEGACY_KGLE = FIXTURES / "legacy_0_13_3.kgle"


def _matching_people() -> kglite.KnowledgeGraph:
    graph = kglite.KnowledgeGraph()
    graph.cypher(
        "UNWIND [{id:'a', title:'Ada'}, {id:'b', title:'Bjarne'}, "
        "{id:'c', title:'Carol'}] AS row "
        "CREATE (:Person {id: row.id, title: row.title})"
    )
    return graph


def test_released_v4_graph_loads_with_embeddings_and_resaves_as_v5(tmp_path):
    assert LEGACY_KGL.read_bytes()[:4] == b"RGF\x04"
    graph = kglite.load(str(LEGACY_KGL))

    rows = graph.cypher("MATCH (n:Person) RETURN n.id, n.score ORDER BY n.id").to_list()
    assert rows == [
        {"n.id": "a", "n.score": 1},
        {"n.id": "b", "n.score": 2},
        {"n.id": "c", "n.score": 3},
    ]
    info = graph.embedding_info("Person", "title")
    assert info["count"] == 3
    assert info["dimension"] == 2
    assert info["metric"] == "euclidean"

    current = tmp_path / "current.kgl"
    graph.save(str(current))
    assert current.read_bytes()[:5] == b"RGF\x05\x02"
    assert kglite.load(str(current)).embedding_info("Person", "title")["count"] == 3


def test_released_kgle_v2_imports_and_current_export_is_tagged_v3(tmp_path):
    assert LEGACY_KGLE.read_bytes()[:8] == b"KGLE\x02\x00\x00\x00"
    graph = _matching_people()
    stats = graph.import_embeddings(str(LEGACY_KGLE))
    assert stats == {"stores": 1, "imported": 3, "skipped": 0, "dropped_stores": 0}
    assert graph.embedding_info("Person", "title")["metric"] == "euclidean"

    current = tmp_path / "current.kgle"
    graph.export_embeddings(str(current))
    assert current.read_bytes()[:9] == b"KGLE\x03\x00\x00\x00\x02"

    reloaded = _matching_people()
    assert reloaded.import_embeddings(str(current))["imported"] == 3
    assert reloaded.embedding_info("Person", "title")["metric"] == "euclidean"
