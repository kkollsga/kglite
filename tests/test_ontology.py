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
                    "inverse_enforced": True,
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
    # The note is one prose sentence: source indentation must not leak into
    # it as mid-sentence whitespace runs.
    note = text.split('note="', 1)[1].split('"', 1)[0]
    assert "  " not in note, note


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


def test_dematerialize_withdraws_the_labels_and_re_materialize_restores_them(mat):
    # Standalone cover for the exit itself: `test_remove_managed_label_refused`
    # reaches it only after asserting a refusal, and `clear_ontology` reaches it
    # on the way to dropping the declarations.
    assert mat.cypher("MATCH (p:Person) RETURN count(p) AS c").scalar() == 3

    assert mat.dematerialize_ontology() == 3
    assert mat.cypher("MATCH (p:Person) RETURN count(p) AS c").scalar() == 0
    assert mat.cypher("MATCH (n:Student {id: 1}) RETURN labels(n) AS l").scalar() == ["Student"]
    assert mat.ontology_diff() == []
    # A second exit is a no-op, not a double count.
    assert mat.dematerialize_ontology() == 0

    # Declarations survive the exit, so the label can be rebuilt from them.
    assert mat.materialize_ontology() == [{"label": "Person", "stamped": 3, "state": "closed"}]
    assert mat.cypher("MATCH (p:Person) RETURN count(p) AS c").scalar() == 3


def test_dematerialize_removes_foreign_members_of_an_open_label(mat):
    # A user SET outside the closure opens the label; the exit withdraws the
    # label wholesale, foreign members included.
    mat.cypher("MATCH (c:Class) SET c:Person")
    assert mat.ontology_diff()[0]["state"] == "open"

    assert mat.dematerialize_ontology() == 4
    assert mat.cypher("MATCH (p:Person) RETURN count(p) AS c").scalar() == 0
    assert mat.cypher("MATCH (c:Class) RETURN labels(c) AS l").scalar() == ["Class"]


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
    mat.create_index("Student", "title")
    mat.create_index("Teacher", "title")
    assert _probes(mat, "MATCH (p:Person {title: 'Ann'}) RETURN p.id"), "the probe must engage"
    rows = mat.cypher("MATCH (p:Person {title: 'Ann'}) RETURN p.id AS id").to_list()
    assert [r["id"] for r in rows] == [1]
    # Same answer with the ontology passes off / general path (differential
    # sanity at the surface level).
    rows2 = mat.cypher("MATCH (p:Person) WHERE p.title = 'Ann' RETURN p.id AS id").to_list()
    assert rows2 == rows


def test_closure_probe_finds_a_value_held_by_one_member(mat):
    # The shape the probe never used to reach: a unique value lives in at most
    # ONE member's index, so a union that declined on any member's value-miss
    # declined structurally, for every closure with two live members.
    mat.create_index("Student", "title")
    mat.create_index("Teacher", "title")
    for value, expected in [("Ann", [1]), ("Bo", [2]), ("Tea", [10]), ("Nobody", [])]:
        query = f"MATCH (p:Person {{title: '{value}'}}) RETURN p.id AS id"
        assert _probes(mat, query), value
        assert [r["id"] for r in mat.cypher(query).to_list()] == expected, value


def test_closure_probe_sees_nodes_written_after_the_index(mat):
    # Index completeness is the premise of reading a value-miss as "empty":
    # a write that the index did not absorb would make the probe answer
    # nothing where a scan answers a row.
    mat.create_index("Student", "title")
    mat.create_index("Teacher", "title")
    mat.add_nodes(
        pd.DataFrame({"id": [7], "name": ["Zed"]}),
        "Teacher",
        "id",
        node_title_field="name",
    )
    mat.cypher("CREATE (:Student {id: 8, name: 'Yin'})")
    mat.cypher("MATCH (s:Student {id: 2}) SET s.title = 'Renamed'")
    query = "MATCH (p:Person {title: '%s'}) RETURN p.id AS id"
    assert _probes(mat, query % "Zed")
    assert [r["id"] for r in mat.cypher(query % "Zed").to_list()] == [7]
    assert [r["id"] for r in mat.cypher(query % "Yin").to_list()] == [8]
    assert [r["id"] for r in mat.cypher(query % "Renamed").to_list()] == [2]
    assert mat.cypher(query % "Bo").to_list() == []


def test_closure_probe_is_alias_aware(mat):
    # `title` and a type's registered title-alias spelling name one field and
    # one set of index contents, so an index built under either serves a query
    # written with the other.
    mat.add_nodes(
        pd.DataFrame({"id": [20], "fullname": ["Ada"]}),
        "Teacher",
        "id",
        node_title_field="fullname",
    )
    mat.create_index("Student", "title")
    mat.create_index("Teacher", "fullname")
    query = "MATCH (p:Person {title: 'Ada'}) RETURN p.id AS id"
    assert _probes(mat, query)
    assert [r["id"] for r in mat.cypher(query).to_list()] == [20]


def test_closure_probe_declines_on_a_soft_alias_index(mat):
    # `name` resolves through the structural fallback (a node with no stored
    # `name` answers with its title) while `create_index` reads the stored
    # property alone, so the index is a subset of what a scan matches and
    # cannot cover a member. Correct answer, by scan.
    mat.create_index("Student", "name")
    mat.create_index("Teacher", "name")
    query = "MATCH (p:Person {name: 'Ann'}) RETURN p.id AS id"
    assert not _probes(mat, query)
    assert [r["id"] for r in mat.cypher(query).to_list()] == [1]


def test_an_index_on_a_soft_alias_name_does_not_change_the_answer(g):
    # The defect the exclusion above closes, at the surface: node 2's `name`
    # comes from its title and no index ever held it.
    g.cypher("CREATE (:T {id: 1, name: 'Ann'})")
    g.cypher("CREATE (:T {id: 2, title: 'Ann'})")
    before = [r["id"] for r in g.cypher("MATCH (n:T {name: 'Ann'}) RETURN n.id AS id ORDER BY id").to_list()]
    assert before == [1, 2]
    g.create_index("T", "name")
    after = [r["id"] for r in g.cypher("MATCH (n:T {name: 'Ann'}) RETURN n.id AS id ORDER BY id").to_list()]
    assert after == before


def test_indexed_absent_value_short_circuits_without_a_scan(mat):
    # Single-type sibling of the closure fix: a covered value with no bucket
    # is proven empty, not an unbuilt index.
    mat.create_index("Student", "title")
    assert mat.cypher("MATCH (s:Student {title: 'Absent'}) RETURN s.id AS id").to_list() == []
    assert [r["id"] for r in mat.cypher("MATCH (s:Student {title: 'Bo'}) RETURN s.id AS id").to_list()] == [2]


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


# ─── EXPLAIN observability for the closure probe ───────────────────────────


def _ops(graph, query):
    return [row["operation"] for row in graph.cypher(query).to_list()]


def _probes(graph, query):
    """Whether the plan for `query` claims a closure probe.

    The marker and the runtime gate read one predicate, so this is also the
    only observation of *engagement* the Python surface has.
    """
    return any(op.startswith("ClosureProbe") for op in _ops(graph, f"EXPLAIN {query}"))


def test_explain_marks_closure_probe_when_every_member_is_indexed(mat):
    mat.create_index("Student", "title")
    mat.create_index("Teacher", "title")
    ops = _ops(mat, "EXPLAIN MATCH (p:Person {title: 'Ann'}) RETURN p.id")
    assert "ClosureProbe :Person (Student, Teacher)" in ops
    # The marker sits directly after the clause row it belongs to.
    assert ops.index("ClosureProbe :Person (Student, Teacher)") == ops.index("Match :Person") + 1


def test_explain_marks_closure_probe_for_id_lookups(mat):
    # No property index anywhere: the canonical `id` is covered by every
    # member's id map, which is why the id arm of the probe already works.
    ops = _ops(mat, "EXPLAIN MATCH (p:Person {id: 1}) RETURN p.name")
    assert "ClosureProbe :Person (Student, Teacher)" in ops


def test_explain_omits_closure_probe_on_partial_index_coverage(mat):
    mat.create_index("Student", "title")
    ops = _ops(mat, "EXPLAIN MATCH (p:Person {title: 'Ann'}) RETURN p.id")
    assert not any(op.startswith("ClosureProbe") for op in ops)
    # …and the scan still answers for the uncovered member.
    assert [r["id"] for r in mat.cypher("MATCH (p:Person {title: 'Tea'}) RETURN p.id AS id").to_list()] == [10]


def test_explain_omits_closure_probe_when_label_is_not_materialized(school):
    school.create_index("Student", "title")
    school.create_index("Teacher", "title")
    ops = _ops(school, "EXPLAIN MATCH (p:Person {title: 'Ann'}) RETURN p.id")
    assert not any(op.startswith("ClosureProbe") for op in ops)


def test_explain_omits_closure_probe_when_label_is_open(mat):
    # A carrier outside the closure opens the label; the probe cannot be the
    # complete answer, so the plan must not advertise it.
    mat.create_index("Student", "title")
    mat.create_index("Teacher", "title")
    mat.cypher("MATCH (c:Class) SET c:Person")
    ops = _ops(mat, "EXPLAIN MATCH (p:Person {title: 'Ann'}) RETURN p.id")
    assert not any(op.startswith("ClosureProbe") for op in ops)
    # The opened bucket's foreign carrier is still found — by scan.
    assert [r["id"] for r in mat.cypher("MATCH (p:Person {title: 'Math'}) RETURN p.id AS id").to_list()] == [100]


def test_explain_counts_materialized_supertype_members(mat):
    # :Person has no primary bucket at all — every carrier holds it as a
    # secondary label. Counting `type_indices` alone reported 0.
    rows = mat.cypher("EXPLAIN MATCH (p:Person) RETURN p.name").to_list()
    assert rows[0]["operation"] == "Match :Person"
    assert rows[0]["estimated_rows"] == 3
    assert mat.cypher("MATCH (p:Person) RETURN count(p) AS c").scalar() == 3


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
    rows = graph.cypher(
        "CALL ontology_audit() YIELD rule, severity, violations, exempted, total, pct RETURN *"
    ).to_list()
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


# ---- 0.16.12: inverse_name is naming-only unless inverse_enforced ----


def test_naming_only_inverse_not_audited(props_graph):
    # The guide defines inverse_name as a reading-direction alias — "no
    # second edge exists or is implied" — so a correctly modelled one-edge
    # graph must not score inverse violations.
    props_graph.define_ontology(
        {
            "classes": {"Student": {}, "Class": {}},
            "relationships": {
                "ENROLLED_IN": {
                    "domain": "Student",
                    "range": "Class",
                    "inverse_name": "HAS_STUDENT",
                }
            },
        }
    )
    assert "ENROLLED_IN.inverse" not in _audit(props_graph)
    rows = props_graph.cypher("CALL inverse_violation() YIELD a, b RETURN a").to_list()
    assert rows == []


def test_inverse_enforced_opts_into_the_physical_check(props_graph):
    props_graph.define_ontology(
        {
            "classes": {"Student": {}, "Class": {}},
            "relationships": {
                "ENROLLED_IN": {
                    "domain": "Student",
                    "range": "Class",
                    "inverse_name": "HAS_STUDENT",
                    "inverse_enforced": True,
                }
            },
        }
    )
    row = _audit(props_graph)["ENROLLED_IN.inverse"]
    assert (row["violations"], row["total"]) == (2, 2)


def test_inverse_enforced_requires_inverse_name(props_graph):
    with pytest.raises(Exception, match="inverse_enforced"):
        props_graph.define_ontology(
            {
                "classes": {"Student": {}},
                "relationships": {"ENROLLED_IN": {"inverse_enforced": True}},
            }
        )


# ---- 0.16.12: per-check enforcement severities ----


def test_enforcement_map_sets_per_check_severity(props_graph):
    props_graph.define_ontology(
        {
            "classes": {"Student": {}, "Class": {}},
            "relationships": {
                "ENROLLED_IN": {
                    "domain": "Student",
                    "range": "Class",
                    "required_properties": ["since"],
                    "enforcement": {"required_properties": "error", "domain": "warn"},
                }
            },
        }
    )
    audit = _audit(props_graph)
    assert audit["ENROLLED_IN.required_properties"]["severity"] == "error"
    assert audit["ENROLLED_IN.domain"]["severity"] == "warn"
    # Unlisted checks keep the advisory base.
    assert audit["ENROLLED_IN.range"]["severity"] == "advisory"


def test_describe_renders_per_check_enforcement_overrides(props_graph):
    # describe() and SHOW ONTOLOGY share one rendering: a declaration whose
    # overrides raise a check above its advisory base must never read as the
    # bare base in either surface.
    props_graph.define_ontology(
        {
            "classes": {"Student": {}, "Class": {}},
            "relationships": {
                "ENROLLED_IN": {
                    "domain": "Student",
                    "range": "Class",
                    "required_properties": ["since"],
                    "enforcement": {"required_properties": "error", "domain": "warn"},
                }
            },
        }
    )
    summary = "advisory; domain=warn, required_properties=error"
    assert f'enforcement="{summary}"' in props_graph.describe()
    rel = [r for r in props_graph.cypher("SHOW ONTOLOGY").to_list() if r["kind"] == "relationship"]
    assert [r["enforcement"] for r in rel] == [summary]


def test_enforcement_map_unknown_check_refused(props_graph):
    with pytest.raises(Exception, match="enforcement"):
        props_graph.define_ontology(
            {
                "classes": {"Student": {}},
                "relationships": {"ENROLLED_IN": {"enforcement": {"inverze": "error"}}},
            }
        )


def test_new_declaration_fields_persist(props_graph, tmp_path):
    decl = {
        "classes": {"Student": {}, "Class": {}},
        "relationships": {
            "ENROLLED_IN": {
                "domain": "Student",
                "range": "Class",
                "inverse_name": "HAS_STUDENT",
                "inverse_enforced": True,
                "required_properties": ["since"],
                "property_types": {"since": "integer"},
                "enforcement": {"required_properties": "error"},
            }
        },
    }
    props_graph.define_ontology(decl)
    path = str(tmp_path / "g.kgl")
    props_graph.save(path)
    loaded = kglite.load(path)
    rel = loaded.ontology()["relationships"]["ENROLLED_IN"]
    assert rel["inverse_enforced"] is True
    assert rel["required_properties"] == ["since"]
    assert rel["property_types"] == {"since": "integer"}
    assert rel["enforcement_overrides"] == {"required_properties": "error"}
    assert _audit(loaded)["ENROLLED_IN.required_properties"]["severity"] == "error"


# ---- per-source-class exemption on required_properties / property_types ----
# (operator report 2026-08-26: 581 permanent, legitimate violations from ONE
# source type pinned HAS_OPERATOR.required_properties at `warn` forever, so
# the rule could never protect the other source types.)


@pytest.fixture
def mixed_props_graph() -> KnowledgeGraph:
    """Two source types on one relationship: one conforms, one never can."""
    graph = KnowledgeGraph()
    graph.cypher("CREATE (:Class {id: 1, title: 'C'})")
    # Student: one edge carries `since`, one does not -> a real violation.
    graph.cypher("CREATE (:Student {id: 1, title: 'A'}), (:Student {id: 2, title: 'B'})")
    graph.cypher("MATCH (s:Student {id: 1}), (c:Class {id: 1}) CREATE (s)-[:ENROLLED_IN {since: 2024}]->(c)")
    graph.cypher("MATCH (s:Student {id: 2}), (c:Class {id: 1}) CREATE (s)-[:ENROLLED_IN]->(c)")
    # Auditor: neither edge carries `since` — the class to exempt.
    graph.cypher("CREATE (:Auditor {id: 3, title: 'X'}), (:Auditor {id: 4, title: 'Y'})")
    graph.cypher("MATCH (a:Auditor {id: 3}), (c:Class {id: 1}) CREATE (a)-[:ENROLLED_IN]->(c)")
    graph.cypher("MATCH (a:Auditor {id: 4}), (c:Class {id: 1}) CREATE (a)-[:ENROLLED_IN]->(c)")
    return graph


def _since_decl(classes, exempt=None):
    rel = {"required_properties": ["since"], "enforcement": "error"}
    if exempt is not None:
        rel["exempt"] = exempt
    return {"classes": classes, "relationships": {"ENROLLED_IN": rel}}


FLAT_CLASSES = {"Student": {}, "Auditor": {}, "Class": {}}


def test_exempt_counts_separately_instead_of_against_severity(mixed_props_graph):
    # Without the exemption the rule cannot distinguish the source types.
    mixed_props_graph.define_ontology(_since_decl(FLAT_CLASSES))
    row = _audit(mixed_props_graph)["ENROLLED_IN.required_properties"]
    assert (row["violations"], row["exempted"], row["total"]) == (3, 0, 4)

    mixed_props_graph.define_ontology(_since_decl(FLAT_CLASSES, {"required_properties": ["Auditor"]}))
    row = _audit(mixed_props_graph)["ENROLLED_IN.required_properties"]
    # Same rows flagged; two of them now attributed to the exemption, and
    # `pct` follows the severity-bearing count.
    assert (row["violations"], row["exempted"], row["total"]) == (1, 2, 4)
    assert row["violations"] + row["exempted"] == 3
    assert row["pct"] == 25.0
    assert row["severity"] == "error"


def test_exempt_widens_over_declared_descendants(mixed_props_graph):
    # An exemption naming an abstract ancestor covers its concrete
    # descendants — the same widening domain/range acceptance uses.
    classes = {
        "Observer": {"abstract": True},
        "Auditor": {"is_a": "Observer"},
        "Student": {},
        "Class": {},
    }
    mixed_props_graph.define_ontology(_since_decl(classes, {"required_properties": ["Observer"]}))
    row = _audit(mixed_props_graph)["ENROLLED_IN.required_properties"]
    assert (row["violations"], row["exempted"]) == (1, 2)
    # A sibling that does not descend from the exempted class is untouched:
    # exempting Student excuses only Student's own row.
    mixed_props_graph.define_ontology(_since_decl(classes, {"required_properties": ["Student"]}))
    row = _audit(mixed_props_graph)["ENROLLED_IN.required_properties"]
    assert (row["violations"], row["exempted"]) == (2, 1)


def test_exempt_applies_to_property_types_too(mixed_props_graph):
    # `since` is an integer on the only edge that carries it; declare it a
    # string, then exempt that edge's source class.
    def decl(exempt=None):
        rel = {"property_types": {"since": "string"}}
        if exempt is not None:
            rel["exempt"] = exempt
        return {"classes": FLAT_CLASSES, "relationships": {"ENROLLED_IN": rel}}

    mixed_props_graph.define_ontology(decl())
    row = _audit(mixed_props_graph)["ENROLLED_IN.property_types"]
    assert (row["violations"], row["exempted"]) == (1, 0)
    mixed_props_graph.define_ontology(decl({"property_types": ["Student"]}))
    row = _audit(mixed_props_graph)["ENROLLED_IN.property_types"]
    assert (row["violations"], row["exempted"]) == (0, 1)


def test_exempt_refuses_the_flat_list_form(mixed_props_graph):
    with pytest.raises(Exception, match=r"\{check: \[class, \.\.\.\]\}"):
        mixed_props_graph.define_ontology(
            {"classes": FLAT_CLASSES, "relationships": {"ENROLLED_IN": {"exempt": ["Auditor"]}}}
        )


@pytest.mark.parametrize(
    ("check", "marker"),
    [
        ("domain", "already tests the domain-side class"),
        ("range", "already tests the domain-side class"),
        ("cardinality", "already tests the domain-side class"),
        ("transitive", "no single domain-side class"),
        ("symmetric", "no single domain-side class"),
    ],
)
def test_exempt_refuses_unexemptable_checks_with_the_reason(mixed_props_graph, check, marker):
    with pytest.raises(Exception, match=marker):
        mixed_props_graph.define_ontology(
            {"classes": FLAT_CLASSES, "relationships": {"ENROLLED_IN": {"exempt": {check: ["Auditor"]}}}}
        )


def test_exempt_refuses_an_unknown_check_name(mixed_props_graph):
    with pytest.raises(Exception, match="Did you mean 'required_properties'"):
        mixed_props_graph.define_ontology(
            {
                "classes": FLAT_CLASSES,
                "relationships": {"ENROLLED_IN": {"exempt": {"required_propertys": ["Auditor"]}}},
            }
        )


def test_exempt_refuses_an_undeclared_class(mixed_props_graph):
    # A typo here would silently exempt nothing and leave the rule pinned —
    # the exact failure the feature exists to fix.
    with pytest.raises(Exception, match="not a declared class"):
        mixed_props_graph.define_ontology(_since_decl(FLAT_CLASSES, {"required_properties": ["Audtor"]}))


def test_exempt_persists_through_save_load(mixed_props_graph, tmp_path):
    decl = {
        "classes": FLAT_CLASSES,
        "relationships": {
            "ENROLLED_IN": {
                "required_properties": ["since"],
                "property_types": {"since": "integer"},
                "enforcement": "error",
                "exempt": {"required_properties": ["Auditor"], "property_types": ["Student"]},
            }
        },
    }
    mixed_props_graph.define_ontology(decl)
    path = str(tmp_path / "g.kgl")
    mixed_props_graph.save(path)
    loaded = kglite.load(path)
    rel = loaded.ontology()["relationships"]["ENROLLED_IN"]
    assert rel["exempt"] == {"required_properties": ["Auditor"], "property_types": ["Student"]}
    assert _audit(loaded)["ENROLLED_IN.required_properties"]["exempted"] == 2


def test_exempt_renders_in_describe_and_show_ontology(mixed_props_graph):
    mixed_props_graph.define_ontology(_since_decl(FLAT_CLASSES, {"required_properties": ["Auditor"]}))
    summary = "required_properties: [Auditor]"
    assert f'exempt="{summary}"' in mixed_props_graph.describe()
    rel = [r for r in mixed_props_graph.cypher("SHOW ONTOLOGY").to_list() if r["kind"] == "relationship"]
    assert [r["exempt"] for r in rel] == [summary]
    # Absent when nothing is exempted, in both surfaces.
    mixed_props_graph.define_ontology(_since_decl(FLAT_CLASSES))
    assert "exempt=" not in mixed_props_graph.describe()
    rows = mixed_props_graph.cypher("SHOW ONTOLOGY").to_list()
    assert all(r["exempt"] is None for r in rows)


# ---- ontology_audit({by: 'domain_class'}) breakdown ------------------------
# (operator report 2026-08-26: the first question after every non-zero audit
# row is "which source types are violating?", answered until now by a
# hand-written Cypher query per rule.)


def _audit_rows(graph, by=None):
    """Every audit row, breakdown column included, grouped by rule."""
    param = f"({{by: '{by}'}})" if by else "()"
    rows = graph.cypher(
        f"CALL ontology_audit{param} YIELD rule, severity, violations, exempted, total, pct, domain_class RETURN *"
    ).to_list()
    out = {}
    for row in rows:
        out.setdefault(row["rule"], []).append(row)
    return out


def test_audit_breakdown_fans_a_rule_per_violating_class(mixed_props_graph):
    mixed_props_graph.define_ontology(_since_decl(FLAT_CLASSES))
    aggregate = _audit_rows(mixed_props_graph)["ENROLLED_IN.required_properties"]
    assert len(aggregate) == 1
    assert (aggregate[0]["violations"], aggregate[0]["domain_class"]) == (3, None)

    fanned = _audit_rows(mixed_props_graph, by="domain_class")["ENROLLED_IN.required_properties"]
    assert {r["domain_class"]: r["violations"] for r in fanned} == {"Student": 1, "Auditor": 2}
    # The fan-out is a partition of the aggregate count, and the per-rule
    # columns ride along unchanged on every row.
    assert sum(r["violations"] for r in fanned) == aggregate[0]["violations"]
    assert {(r["severity"], r["total"], r["exempted"]) for r in fanned} == {("error", 4, 0)}
    assert [r["pct"] for r in sorted(fanned, key=lambda r: r["domain_class"])] == [50.0, 25.0]


def test_audit_breakdown_leaves_exempted_classes_out(mixed_props_graph):
    mixed_props_graph.define_ontology(_since_decl(FLAT_CLASSES, {"required_properties": ["Auditor"]}))
    fanned = _audit_rows(mixed_props_graph, by="domain_class")["ENROLLED_IN.required_properties"]
    # Auditor's two flagged edges are excused, so the class has no
    # severity-bearing violations and no row of its own; the per-rule
    # `exempted` tail rides along on the row that remains, unsplit.
    assert [(r["domain_class"], r["violations"], r["exempted"]) for r in fanned] == [("Student", 1, 2)]

    # A rule whose every violation is exempted still reports itself — a rule
    # vanishing from the scorecard would read as "not declared".
    mixed_props_graph.define_ontology(_since_decl(FLAT_CLASSES, {"required_properties": ["Auditor", "Student"]}))
    fanned = _audit_rows(mixed_props_graph, by="domain_class")["ENROLLED_IN.required_properties"]
    assert [(r["domain_class"], r["violations"], r["exempted"]) for r in fanned] == [(None, 0, 3)]


def test_audit_breakdown_attributes_pair_rules_to_the_source_side(school):
    """inverse/symmetric/transitive rows have no domain-side *endpoint*; they
    are attributed to the first bound node — the source of the edge or chain
    whose partner is missing."""
    fanned = _audit_rows(school, by="domain_class")
    assert [(r["domain_class"], r["violations"]) for r in fanned["ENROLLED_IN.inverse"]] == [("Teacher", 1)]
    assert [(r["domain_class"], r["violations"]) for r in fanned["ENROLLED_IN.required"]] == [("Student", 1)]
    # A clean rule keeps its single aggregate row with a Null class.
    assert [(r["domain_class"], r["violations"]) for r in fanned["ENROLLED_IN.range"]] == [(None, 0)]


def test_audit_bare_call_gains_a_null_domain_class_column(school):
    rows = school.cypher("CALL ontology_audit()").to_list()
    assert len(rows) == 4
    assert all(r["domain_class"] is None for r in rows)
    assert set(rows[0]) == {"rule", "severity", "violations", "exempted", "total", "pct", "domain_class"}


def test_audit_by_param_is_validated(school):
    with pytest.raises(Exception, match="invalid 'by' value 'nonsense'"):
        school.cypher("CALL ontology_audit({by: 'nonsense'}) YIELD rule RETURN rule")
    with pytest.raises(Exception, match="unknown parameter 'group_by'"):
        school.cypher("CALL ontology_audit({group_by: 'domain_class'}) YIELD rule RETURN rule")
    # The column is still a normal YIELD target, aliasable like any other.
    rows = school.cypher(
        "CALL ontology_audit({by: 'domain_class'}) YIELD rule, domain_class AS cls RETURN rule, cls"
    ).to_list()
    assert ("ENROLLED_IN.domain", "Teacher") in {(r["rule"], r["cls"]) for r in rows}


# ---- edge_property_violation drill-down -----------------------------------


def _property_decl(exempt=None):
    rel = {
        "required_properties": ["since"],
        "property_types": {"since": "string"},
        "enforcement": "error",
    }
    if exempt is not None:
        rel["exempt"] = exempt
    return {"classes": FLAT_CLASSES, "relationships": {"ENROLLED_IN": rel}}


def test_edge_property_violation_lists_both_checks(mixed_props_graph):
    mixed_props_graph.define_ontology(_property_decl({"required_properties": ["Auditor"]}))
    rows = mixed_props_graph.cypher(
        "CALL edge_property_violation() YIELD relationship, check, source, target, property, exempt "
        "RETURN relationship, check, source.title AS src, target.title AS tgt, property, exempt "
        "ORDER BY check, src"
    ).to_list()
    assert [(r["check"], r["src"], r["property"], r["exempt"]) for r in rows] == [
        # `since` is an integer on the one edge that carries it.
        ("property_types", "A", "since", False),
        # Absent on the other three; the two Auditor-sourced ones are excused.
        ("required_properties", "B", "since", False),
        ("required_properties", "X", "since", True),
        ("required_properties", "Y", "since", True),
    ]
    assert {(r["relationship"], r["tgt"]) for r in rows} == {("ENROLLED_IN", "C")}

    # Every flagged row is attributable: the listing reconciles with the
    # scorecard it drills into, exempted rows included.
    audit = _audit(mixed_props_graph)
    for check in ("required_properties", "property_types"):
        line = audit[f"ENROLLED_IN.{check}"]
        listed = [r for r in rows if r["check"] == check]
        assert len(listed) == line["violations"] + line["exempted"]
        assert sum(1 for r in listed if r["exempt"]) == line["exempted"]


def test_edge_property_violation_yields_column_subsets(mixed_props_graph):
    mixed_props_graph.define_ontology(_property_decl())
    checks = mixed_props_graph.cypher(
        "CALL edge_property_violation() YIELD check RETURN check ORDER BY check"
    ).to_list()
    assert [r["check"] for r in checks] == ["property_types"] + ["required_properties"] * 3
    bare = mixed_props_graph.cypher("CALL edge_property_violation()").to_list()
    assert len(bare) == 4
    assert set(bare[0]) == {"relationship", "check", "source", "target", "property", "exempt"}
    with pytest.raises(Exception, match="does not yield"):
        mixed_props_graph.cypher("CALL edge_property_violation() YIELD bogus RETURN bogus")


def test_edge_property_violation_refuses_params_and_a_bare_graph(mixed_props_graph, g):
    mixed_props_graph.define_ontology(_property_decl())
    with pytest.raises(Exception, match="takes no parameters"):
        mixed_props_graph.cypher("CALL edge_property_violation({edge: 'ENROLLED_IN'}) YIELD source RETURN source")
    with pytest.raises(Exception, match="define_ontology"):
        g.cypher("CALL edge_property_violation() YIELD source RETURN source")
