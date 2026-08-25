"""Ontology declaration store: define/get/clear, validation, persistence."""

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph

SCHOOL = {
    "classes": {
        "Person": {"abstract": True, "description": "Any human"},
        "Student": {"is_a": "Person"},
        "Teacher": {"is_a": "Person"},
    },
    "relationships": {
        "ENROLLED_IN": {
            "domain": "Student",
            "range": "Class",
            "inverse_name": "HAS_STUDENT",
            "cardinality": {"min": 1},
            "required": True,
            "enforcement": "warn",
        },
    },
}


@pytest.fixture
def g() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.add_nodes(pd.DataFrame({"id": [1, 2], "title": ["a", "b"]}), "Student", "id", node_title_field="title")
    graph.add_nodes(pd.DataFrame({"id": [3], "title": ["c"]}), "Class", "id", node_title_field="title")
    return graph


def test_define_get_clear_roundtrip(g):
    warnings = g.define_ontology(SCHOOL)
    # Teacher is concrete with no live nodes -> one warning.
    assert warnings == [w for w in warnings if "Teacher" in w] and len(warnings) == 1
    doc = g.ontology()
    assert doc["classes"]["Person"]["abstract"] is True
    assert doc["classes"]["Student"]["is_a"] == "Person"
    assert doc["relationships"]["ENROLLED_IN"]["enforcement"] == "warn"
    g.clear_ontology()
    assert g.ontology() is None


def test_define_ontology_rejects_malformed(g):
    with pytest.raises(ValueError, match="unknown key"):
        g.define_ontology({"clases": {}})
    with pytest.raises(ValueError, match="cycle"):
        g.define_ontology({"classes": {"A": {"is_a": "B"}, "B": {"is_a": "A"}}})
    with pytest.raises(ValueError, match="abstract"):
        # Student is a live primary type; abstract may not shadow it.
        g.define_ontology({"classes": {"Student": {"abstract": True}}})
    with pytest.raises(ValueError, match="advisory"):
        g.define_ontology({"relationships": {"R": {"enforcement": "fatal"}}})


def test_ontology_persists_through_save_load(g, tmp_path):
    g.define_ontology(SCHOOL)
    path = tmp_path / "g.kgl"
    g.save(str(path))
    loaded = kglite.load(str(path))
    doc = loaded.ontology()
    assert doc["classes"]["Person"]["abstract"] is True
    assert doc["relationships"]["ENROLLED_IN"]["required"] is True


def test_ontology_free_graph_reports_none(g):
    assert g.ontology() is None
