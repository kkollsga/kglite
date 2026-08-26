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


# ─── declaration-driven validators + ontology_audit ────────────────────────


@pytest.fixture
def school() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.cypher("CREATE (:Student {id: 1, name: 'Ann'}), (:Student {id: 2, name: 'Bo'})")
    graph.cypher("CREATE (:Teacher {id: 10, name: 'Tea'})")
    graph.cypher("CREATE (:Class {id: 100, name: 'Math'})")
    # Ann enrolled (with inverse), Bo NOT enrolled (required violation).
    graph.cypher("MATCH (s:Student {id: 1}), (c:Class {id: 100}) CREATE (s)-[:ENROLLED_IN]->(c)")
    graph.cypher("MATCH (s:Student {id: 1}), (c:Class {id: 100}) CREATE (c)-[:HAS_STUDENT]->(s)")
    # A Teacher also ENROLLED_IN (domain violation vs domain=Student),
    # with no inverse edge (inverse violation).
    graph.cypher("MATCH (t:Teacher {id: 10}), (c:Class {id: 100}) CREATE (t)-[:ENROLLED_IN]->(c)")
    graph.define_ontology(
        {
            "classes": {
                "Person": {"abstract": True},
                "Student": {"is_a": "Person"},
                "Teacher": {"is_a": "Person"},
            },
            "relationships": {
                "ENROLLED_IN": {
                    "domain": "Student",
                    "range": "Class",
                    "inverse_name": "HAS_STUDENT",
                    "required": True,
                    "enforcement": "warn",
                },
            },
        }
    )
    return graph


def test_no_arg_type_domain_violation_reads_declarations(school):
    rows = school.cypher(
        "CALL type_domain_violation() YIELD source, target, rule RETURN source.name AS s, rule"
    ).to_list()
    assert [(r["s"], r["rule"]) for r in rows] == [("Tea", "ENROLLED_IN.domain")]


def test_no_arg_missing_required_edge(school):
    rows = school.cypher("CALL missing_required_edge() YIELD node, rule RETURN node.name AS n, rule").to_list()
    assert [(r["n"], r["rule"]) for r in rows] == [("Bo", "ENROLLED_IN.required")]


def test_no_arg_inverse_violation(school):
    rows = school.cypher("CALL inverse_violation() YIELD a, b RETURN a.name AS a ORDER BY a").to_list()
    assert [r["a"] for r in rows] == ["Tea"]


def test_abstract_domain_widens_to_descendants(school):
    # MENTORS domain=Person (abstract): Student- and Teacher-sourced edges
    # are fine; a Class-sourced edge violates.
    school.cypher("MATCH (t:Teacher {id: 10}), (s:Student {id: 1}) CREATE (t)-[:MENTORS]->(s)")
    school.cypher("MATCH (c:Class {id: 100}), (s:Student {id: 1}) CREATE (c)-[:MENTORS]->(s)")
    doc = school.ontology()
    doc["relationships"]["MENTORS"] = {"domain": "Person", "range": "Person"}
    school.define_ontology(doc)
    rows = school.cypher(
        "CALL type_domain_violation() YIELD source, target, rule WHERE rule = 'MENTORS.domain' RETURN source.name AS s"
    ).to_list()
    assert [r["s"] for r in rows] == ["Math"]


def test_explicit_params_still_work_and_stamp_rule(school):
    rows = school.cypher(
        "CALL type_domain_violation({edge: 'ENROLLED_IN', expected_source: 'Student'}) "
        "YIELD source, target, rule RETURN source.name AS s, rule"
    ).to_list()
    assert [(r["s"], r["rule"]) for r in rows] == [("Tea", "ENROLLED_IN")]


def test_no_arg_without_ontology_keeps_param_error(g):
    with pytest.raises(Exception, match="type_domain_violation"):
        g.cypher("CALL type_domain_violation() YIELD source, target RETURN source")


def test_ontology_audit_scorecard(school):
    rows = school.cypher(
        "CALL ontology_audit() YIELD rule, severity, violations, total, pct "
        "RETURN rule, severity, violations, total, pct ORDER BY rule"
    ).to_list()
    by_rule = {r["rule"]: r for r in rows}
    assert set(by_rule) == {
        "ENROLLED_IN.domain",
        "ENROLLED_IN.range",
        "ENROLLED_IN.required",
        "ENROLLED_IN.inverse",
    }
    dom = by_rule["ENROLLED_IN.domain"]
    assert (dom["severity"], dom["violations"], dom["total"]) == ("warn", 1, 2)
    assert dom["pct"] == 50.0
    req = by_rule["ENROLLED_IN.required"]
    assert (req["violations"], req["total"]) == (1, 2)


def test_ontology_audit_requires_ontology(g):
    with pytest.raises(Exception, match="define_ontology"):
        g.cypher("CALL ontology_audit() YIELD rule RETURN rule")
