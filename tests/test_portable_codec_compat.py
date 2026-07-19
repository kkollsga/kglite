"""The 0.14 persistence boundary rejects pre-Postcard portable artifacts."""

from pathlib import Path

import pytest

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


def test_released_v4_graph_requires_a_pre_014_converter():
    assert LEGACY_KGL.read_bytes()[:4] == b"RGF\x04"
    with pytest.raises(kglite.FileFormatError, match="pre-0.14.*0.13.4"):
        kglite.load(str(LEGACY_KGL))


def test_released_kgle_v2_is_rejected_and_current_export_is_tagged_v3(tmp_path):
    assert LEGACY_KGLE.read_bytes()[:8] == b"KGLE\x02\x00\x00\x00"
    graph = _matching_people()
    with pytest.raises(OSError, match="pre-0.14.*0.13.4"):
        graph.import_embeddings(str(LEGACY_KGLE))

    graph.set_embeddings(
        "Person",
        "title",
        {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.5, 0.5]},
        metric="euclidean",
    )

    current = tmp_path / "current.kgle"
    graph.export_embeddings(str(current))
    assert current.read_bytes()[:9] == b"KGLE\x03\x00\x00\x00\x02"

    reloaded = _matching_people()
    assert reloaded.import_embeddings(str(current))["imported"] == 3
    assert reloaded.embedding_info("Person", "title")["metric"] == "euclidean"
