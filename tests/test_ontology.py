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


# ─── SHOW ONTOLOGY + describe() integration ────────────────────────────────


def test_show_ontology_rows(school):
    rows = school.cypher("SHOW ONTOLOGY").to_list()
    by = {(r["kind"], r["name"]): r for r in rows}
    assert by[("class", "Person")]["abstract"] is True
    assert by[("class", "Student")]["is_a"] == "Person"
    rel = by[("relationship", "ENROLLED_IN")]
    assert (rel["domain"], rel["range"], rel["enforcement"]) == ("Student", "Class", "warn")


def test_show_ontology_empty_without_declarations(g):
    assert g.cypher("SHOW ONTOLOGY").to_list() == []


def test_describe_renders_ontology_section(school):
    text = school.describe()
    assert "<ontology " in text
    assert 'name="Person" abstract="true"' in text
    assert 'is_a="Person"' in text
    assert "annotations, not axioms" in text


def test_describe_without_ontology_unchanged(g):
    assert "<ontology" not in g.describe()


def test_describe_topic_ontology(g):
    text = g.describe(cypher=["ontology"])
    assert "ontology_audit" in text
    assert "never changes what a query matches" in text


# ─── materialization (mode D) ──────────────────────────────────────────────


@pytest.fixture
def mat(school) -> KnowledgeGraph:
    report = school.materialize_ontology()
    assert {(r["label"], r["state"]) for r in report} == {("Person", "closed")}
    return school


def test_materialize_makes_supertype_matchable(mat):
    rows = mat.cypher("MATCH (p:Person) RETURN p.name AS n ORDER BY n").to_list()
    assert [r["n"] for r in rows] == ["Ann", "Bo", "Tea"]
    assert mat.cypher("MATCH (n:Student {id: 1}) RETURN labels(n) AS l").scalar() == [
        "Student",
        "Person",
    ]
    # Idempotent re-apply stamps nothing new.
    again = mat.materialize_ontology()
    assert all(r["stamped"] == 0 for r in again)


def test_materialized_labels_persist(mat, tmp_path):
    path = tmp_path / "mat.kgl"
    mat.save(str(path))
    loaded = kglite.load(str(path))
    assert loaded.cypher("MATCH (p:Person) RETURN count(p) AS c").scalar() == 3
    assert loaded.ontology_diff() == [{"label": "Person", "state": "closed", "extra": 0, "missing": 0}]


def test_remove_managed_label_refused(mat):
    with pytest.raises(Exception, match="managed by the materialized ontology"):
        mat.cypher("MATCH (n:Student {id: 1}) REMOVE n:Person")
    # The exit works and empties the managed set.
    removed = mat.dematerialize_ontology()
    assert removed == 3
    assert mat.cypher("MATCH (p:Person) RETURN count(p) AS c").scalar() == 0
    assert mat.ontology_diff() == []
    # Declarations survive the exit.
    assert mat.ontology() is not None


def test_adopt_collision_policy(school):
    # A manual :Person on a Class node predates materialization.
    school.cypher("MATCH (c:Class) SET c:Person")
    with pytest.raises(ValueError, match="adopt"):
        school.materialize_ontology()
    report = school.materialize_ontology(adopt=True)
    assert report == [{"label": "Person", "stamped": 3, "state": "open"}]
    diff = school.ontology_diff()
    assert diff == [{"label": "Person", "state": "open", "extra": 1, "missing": 0}]


def test_manual_set_downgrades_to_open(mat):
    # SET of a managed label on a node outside the closure: allowed, but the
    # label opens (correctness preserved, optimizations off).
    mat.cypher("MATCH (c:Class) SET c:Person")
    diff = mat.ontology_diff()
    assert diff[0]["state"] == "open"
    assert diff[0]["extra"] == 1


def test_clear_ontology_dematerializes_first(mat):
    mat.clear_ontology()
    assert mat.cypher("MATCH (p:Person) RETURN count(p) AS c").scalar() == 0
    assert mat.ontology() is None


# ─── write-funnel closure maintenance ──────────────────────────────────────


def test_create_stamps_closure(mat):
    mat.cypher("CREATE (:Student {id: 5, name: 'New'})")
    assert mat.cypher("MATCH (n:Student {id: 5}) RETURN labels(n) AS l").scalar() == [
        "Person",
        "Student",
    ] or mat.cypher("MATCH (n:Student {id: 5}) RETURN labels(n) AS l").scalar() == [
        "Student",
        "Person",
    ]
    # Explicit redundant label normalizes to the same set.
    mat.cypher("CREATE (:Student:Person {id: 6, name: 'Also'})")
    assert mat.cypher("MATCH (p:Person) RETURN count(p) AS c").scalar() == 5
    assert mat.ontology_diff() == [{"label": "Person", "state": "closed", "extra": 0, "missing": 0}]


def test_add_nodes_stamps_closure(mat):
    mat.add_nodes(
        pd.DataFrame({"id": [7, 8], "name": ["a", "b"]}),
        "Teacher",
        "id",
        node_title_field="name",
    )
    rows = mat.cypher("MATCH (p:Person) RETURN count(p) AS c").scalar()
    assert rows == 5
    assert mat.ontology_diff() == [{"label": "Person", "state": "closed", "extra": 0, "missing": 0}]


def test_abstract_create_refused(mat):
    with pytest.raises(Exception, match="abstract ontology class"):
        mat.cypher("CREATE (:Person {id: 99, name: 'Ghost'})")
    with pytest.raises(Exception, match="abstract ontology class"):
        mat.add_nodes(pd.DataFrame({"id": [99], "name": ["Ghost"]}), "Person", "id", node_title_field="name")
    # The message names the concrete subtypes.
    try:
        mat.cypher("CREATE (:Person {id: 99, name: 'Ghost'})")
    except Exception as e:
        assert "Student" in str(e) and "Teacher" in str(e)


def test_merge_create_stamps_closure(mat):
    mat.cypher("MERGE (s:Student {id: 42}) ON CREATE SET s.name = 'Merged'")
    labels = set(mat.cypher("MATCH (n:Student {id: 42}) RETURN labels(n) AS l").scalar())
    assert labels == {"Student", "Person"}


# ─── closure-aware index probe (Closed-gated) ──────────────────────────────


def test_closure_probe_uses_descendant_indexes(mat):
    mat.create_index("Student", "name")
    mat.create_index("Teacher", "name")
    rows = mat.cypher("MATCH (p:Person {name: 'Ann'}) RETURN p.id AS id").to_list()
    assert [r["id"] for r in rows] == [1]
    # Same answer with the ontology passes off / general path (differential
    # sanity at the surface level).
    rows2 = mat.cypher("MATCH (p:Person) WHERE p.name = 'Ann' RETURN p.id AS id").to_list()
    assert rows2 == rows


def test_closure_probe_correct_when_label_open(mat):
    # A manual carrier outside the closure opens the label; the probe must
    # NOT run (it would miss the Class node), and the scan must find it.
    mat.create_index("Student", "name")
    mat.create_index("Teacher", "name")
    mat.cypher("MATCH (c:Class) SET c:Person")
    rows = mat.cypher("MATCH (p:Person {name: 'Math'}) RETURN p.id AS id").to_list()
    assert [r["id"] for r in rows] == [100]


def test_closure_probe_declines_on_partial_indexes(mat):
    # Only Student is indexed: the probe must fall back wholesale (a partial
    # union would silently drop Teacher rows).
    mat.create_index("Student", "name")
    rows = mat.cypher("MATCH (p:Person {name: 'Tea'}) RETURN p.id AS id").to_list()
    assert [r["id"] for r in rows] == [10]


# ---- 0.16.12: declared-but-dead required_properties / property_types ----
# (operator report 2026-08-26: both keys parsed+persisted but never checked)


@pytest.fixture
def props_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.cypher("CREATE (:Student {id: 1, title: 'A'}), (:Class {id: 1, title: 'C'})")
    # One edge with `since` (correctly typed), one without it.
    graph.cypher("MATCH (s:Student {id: 1}), (c:Class {id: 1}) CREATE (s)-[:ENROLLED_IN {since: 2024}]->(c)")
    graph.cypher("CREATE (:Student {id: 2, title: 'B'})")
    graph.cypher("MATCH (s:Student {id: 2}), (c:Class {id: 1}) CREATE (s)-[:ENROLLED_IN]->(c)")
    return graph


def _audit(graph):
    rows = graph.cypher("CALL ontology_audit() YIELD rule, severity, violations, total RETURN *").to_list()
    return {r["rule"]: r for r in rows}


def test_required_properties_audit_row(props_graph):
    props_graph.define_ontology(
        {
            "classes": {"Student": {}, "Class": {}},
            "relationships": {
                "ENROLLED_IN": {
                    "domain": "Student",
                    "range": "Class",
                    "required_properties": ["since"],
                    "enforcement": "error",
                }
            },
        }
    )
    audit = _audit(props_graph)
    row = audit["ENROLLED_IN.required_properties"]
    assert (row["violations"], row["total"], row["severity"]) == (1, 2, "error")


def test_property_types_audit_row(props_graph):
    # `since` is an integer on the edge that has it; declare it as a string
    # -> 1 violation (only present values are checked; absence is
    # required_properties' business).
    props_graph.define_ontology(
        {
            "classes": {"Student": {}, "Class": {}},
            "relationships": {
                "ENROLLED_IN": {
                    "domain": "Student",
                    "range": "Class",
                    "property_types": {"since": "string"},
                }
            },
        }
    )
    audit = _audit(props_graph)
    row = audit["ENROLLED_IN.property_types"]
    assert (row["violations"], row["total"]) == (1, 2)
    # Correctly declared type -> clean row.
    props_graph.define_ontology(
        {
            "classes": {"Student": {}, "Class": {}},
            "relationships": {
                "ENROLLED_IN": {
                    "domain": "Student",
                    "range": "Class",
                    "property_types": {"since": "integer"},
                }
            },
        }
    )
    assert _audit(props_graph)["ENROLLED_IN.property_types"]["violations"] == 0


def test_property_types_unknown_type_name_refused(props_graph):
    # value_matches_type is permissive on unknown names, so a typo would
    # silently never fail -- the declaration must refuse it up front.
    with pytest.raises(Exception, match="property_types"):
        props_graph.define_ontology(
            {
                "classes": {"Student": {}, "Class": {}},
                "relationships": {"ENROLLED_IN": {"property_types": {"since": "strig"}}},
            }
        )
