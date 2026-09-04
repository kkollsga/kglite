"""Class property contracts share audit, drill-down and blueprint enforcement."""

import json

import pandas as pd
import pytest

import kglite


def _audit(g, by=None):
    args = "" if by is None else "{by: '" + by + "'}"
    return g.cypher(
        f"CALL ontology_audit({args}) YIELD entity_kind, rule, severity, violations, total, pct, exempted, "
        "property, domain_class RETURN *"
    ).to_list()


@pytest.mark.parametrize("storage", ["memory", "mapped", "disk"])
def test_inherited_contracts_audit_drilldown_and_persistence(storage, tmp_path):
    opts = {} if storage == "memory" else {"storage": storage}
    if storage == "disk":
        opts["path"] = str(tmp_path / "disk")
    g = kglite.KnowledgeGraph(**opts)
    g.cypher("CREATE (:Study {id: 1, title: 'one', tags: []}), (:Study {id: 2, title: 'two', tags: 'wrong'})")
    g.cypher("CREATE (:Trial {id: 3, title: 'three', tags: ['ok'], design: 'RCT'})")
    g.cypher("CREATE (:Other:Study {id: 4, title: 'secondary label only'})")
    g.cypher("MATCH (s:Study {id: 1}), (t:Trial) CREATE (s)-[:Study]->(t)")
    ontology = {
        "classes": {
            "Study": {
                "required_properties": ["design", "tags", "title"],
                "property_types": {"tags": "list", "title": "string"},
                "enforcement": {"required_properties": "error"},
            },
            "Trial": {"is_a": "Study", "required_properties": ["registration"]},
            "Empty": {"required_properties": ["unused"]},
        },
        "relationships": {"Study": {"required_properties": ["evidence"]}},
    }
    g.define_ontology(ontology)
    rows = _audit(g)
    audit = {(r["entity_kind"], r["rule"]): r for r in rows}
    assert len(audit) == 5
    expected = {
        ("node", "Study.required_properties"): (2, 3, 66.7, "error"),
        ("node", "Study.property_types"): (1, 3, 33.3, "advisory"),
        ("node", "Trial.required_properties"): (1, 1, 100.0, "advisory"),
        ("node", "Empty.required_properties"): (0, 0, 0.0, "advisory"),
        ("edge", "Study.required_properties"): (1, 1, 100.0, "advisory"),
    }
    for key, values in expected.items():
        assert tuple(audit[key][c] for c in ["violations", "total", "pct", "severity"]) == values
        assert audit[key]["exempted"] == 0
    findings = g.cypher(
        "CALL node_property_violation() YIELD class, check, node AS n, property, properties "
        "RETURN class, check, n.id AS id, property, properties ORDER BY class, check, id"
    ).to_list()
    assert [(r["class"], r["check"], r["id"], r["properties"]) for r in findings] == [
        ("Study", "property_types", 2, ["tags"]),
        ("Study", "required_properties", 1, ["design"]),
        ("Study", "required_properties", 2, ["design"]),
        ("Trial", "required_properties", 3, ["registration"]),
    ]
    assert all(r["property"] == r["properties"][0] for r in findings)
    census = {(r["entity_kind"], r["rule"], r["property"]): r["violations"] for r in _audit(g, "property")}
    assert census[("node", "Study.required_properties", "title")] == 0
    assert census[("node", "Study.required_properties", "design")] == 2
    partition = [r for r in _audit(g, "domain_class") if r["entity_kind"] == "node" and r["violations"]]
    assert [(r["rule"], r["domain_class"]) for r in partition] == [
        ("Study.required_properties", "Study"),
        ("Study.property_types", "Study"),
        ("Trial.required_properties", "Trial"),
    ]
    path = tmp_path / "saved.kgl"
    g.save(str(path))
    loaded = kglite.load(str(path))
    assert _audit(loaded) == rows
    assert loaded.ontology()["classes"]["Study"]["required_properties"] == ["design", "tags", "title"]
    loaded.define_ontology({"classes": {"Study": {}}})
    assert _audit(loaded) == []
    assert loaded.cypher("CALL node_property_violation()").to_list() == []


def test_aliases_resolve_against_each_actual_subclass_and_multi_property_census():
    g = kglite.KnowledgeGraph()
    for kind, id_field, title_field in [("A", "code", "name"), ("B", "key", "label")]:
        g.add_nodes(pd.DataFrame({id_field: [1], title_field: [kind]}), kind, id_field, title_field)
    g.define_ontology(
        {
            "classes": {
                "Base": {
                    "required_properties": ["title", "id", "missing1", "missing2"],
                    "property_types": {"title": "string"},
                },
                "A": {"is_a": "Base", "required_properties": ["code", "name"]},
                "B": {"is_a": "Base", "required_properties": ["key", "label"]},
            }
        }
    )
    rows = _audit(g)
    assert [(r["rule"], r["violations"]) for r in rows if r["violations"]] == [("Base.required_properties", 2)]
    findings = g.cypher("CALL node_property_violation() YIELD properties RETURN properties").to_list()
    assert findings == [{"properties": ["missing1", "missing2"]}] * 2
    census = {r["property"]: r["violations"] for r in _audit(g, "property") if r["rule"] == "Base.required_properties"}
    assert census == {"title": 0, "id": 0, "missing1": 2, "missing2": 2}


@pytest.mark.parametrize(
    "decl",
    [
        {"required_properties": "title"},
        {"required_properties": [1]},
        {"property_types": {"tags": "list[string]"}},
        {"property_types": {"tags": 4}},
        {"enforcement": {"domain": "error"}},
        {"enforcement": "fatal"},
        {"exempt": {}},
    ],
)
def test_invalid_class_contracts_are_rejected_without_replacing_ontology(decl):
    g = kglite.KnowledgeGraph()
    g.define_ontology({"classes": {"Old": {}}})
    with pytest.raises(ValueError):
        g.define_ontology({"classes": {"New": decl}})
    assert set(g.ontology()["classes"]) == {"Old"}


def test_node_drilldown_rejects_parameters_unknown_yields_and_missing_ontology():
    g = kglite.KnowledgeGraph()
    with pytest.raises(Exception, match="no ontology declared"):
        g.cypher("CALL node_property_violation()")
    g.define_ontology({"classes": {"Study": {}}})
    with pytest.raises(Exception, match="no parameters"):
        g.cypher("CALL node_property_violation({class: 'Study'})")
    with pytest.raises(Exception, match="(?i)(yield|column)"):
        g.cypher("CALL node_property_violation() YIELD source")


@pytest.mark.parametrize("severity", ["error", "warn", "advisory"])
@pytest.mark.parametrize("storage", ["memory", "disk"])
def test_blueprint_enforces_node_contracts_before_publication(tmp_path, severity, recwarn, storage):
    (tmp_path / "nodes.csv").write_text("id,title\n1,A\n", encoding="utf-8")
    (tmp_path / "ontology.json").write_text(
        json.dumps({"classes": {"Study": {"required_properties": ["design"], "enforcement": severity}}}),
        encoding="utf-8",
    )
    bp = {
        "settings": {"root": str(tmp_path), "output": "out.kgl"},
        "ontology": "ontology.json",
        "nodes": {"Study": {"csv": "nodes.csv", "pk": "id", "title": "title"}},
    }
    (tmp_path / "blueprint.json").write_text(json.dumps(bp), encoding="utf-8")
    options = {} if storage == "memory" else {"storage": "disk", "path": str(tmp_path / "disk-build")}
    if severity == "error":
        with pytest.raises(ValueError, match="node Study.required_properties: 1/1"):
            kglite.from_blueprint(str(tmp_path / "blueprint.json"), **options)
        assert not (tmp_path / "out.kgl").exists()
    else:
        g = kglite.from_blueprint(str(tmp_path / "blueprint.json"), save=False, verbose=True, **options)
        assert _audit(g)[0]["violations"] == 1
        warned = any("node Study.required_properties" in str(w.message) for w in recwarn)
        assert warned == (severity == "warn")


@pytest.mark.parametrize("kind", ["node", "edge"])
def test_repeated_required_name_does_not_double_count_one_property(kind):
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (a:Study {id: 1})-[:LINK]->(b:Target {id: 2})")
    key, name = ("classes", "Study") if kind == "node" else ("relationships", "LINK")
    g.define_ontology({key: {name: {"required_properties": ["missing", "missing"]}}})
    rows = _audit(g, "property")
    assert [(r["property"], r["violations"], r["total"]) for r in rows] == [("missing", 1, 1)]
    proc = "node_property_violation" if kind == "node" else "edge_property_violation"
    assert g.cypher(f"CALL {proc}() YIELD properties RETURN properties").to_list() == [{"properties": ["missing"]}]


@pytest.mark.parametrize("storage", ["memory", "disk"])
def test_legacy_class_without_contract_fields_loads_with_empty_defaults(storage, tmp_path):
    opts = {} if storage == "memory" else {"storage": "disk", "path": str(tmp_path / "disk")}
    g = kglite.KnowledgeGraph(**opts)
    g.cypher("CREATE (:Study {id: 1})")
    g.define_ontology({"classes": {"Study": {}}})
    assert g.ontology()["classes"]["Study"] == {}
    path = tmp_path / "legacy.kgl"
    g.save(str(path))
    loaded = kglite.load(str(path))
    assert loaded.ontology()["classes"]["Study"] == {}
    assert _audit(loaded) == []


def test_contract_introspection_and_null_type_rules():
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Study {id: 1, tags: null}), (:Study {id: 2, tags: []})")
    g.define_ontology(
        {
            "classes": {
                "Study": {"required_properties": ["tags"], "property_types": {"tags": "list"}, "enforcement": "warn"}
            }
        }
    )
    rows = g.cypher("SHOW ONTOLOGY").to_list()
    assert rows[0]["required_properties"] == ["tags"]
    assert rows[0]["property_types"] == {"tags": "list"}
    assert rows[0]["enforcement"] == "warn"
    desc = g.describe()
    assert 'required_properties="tags"' in desc
    assert 'property_types="tags: list"' in desc
    assert [(r["rule"], r["violations"]) for r in _audit(g)] == [
        ("Study.required_properties", 1),
        ("Study.property_types", 0),
    ]
    g.clear_ontology()
    with pytest.raises(Exception, match="no ontology declared"):
        g.cypher("CALL node_property_violation()")


def test_property_contract_description_escapes_xml_attributes():
    from xml.etree import ElementTree

    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Study {id: 1})")
    field = 'a"b&c<d'
    g.define_ontology(
        {
            "classes": {
                "Study": {"required_properties": [field], "property_types": {field: "string"}, "description": field}
            }
        }
    )
    text = g.describe()
    ontology = text[text.index("<ontology ") : text.index("</ontology>") + len("</ontology>")]
    cls = ElementTree.fromstring(ontology).find("class")
    assert cls.attrib["required_properties"] == field
    assert cls.attrib["property_types"] == field + ": string"
    assert cls.attrib["desc"] == field
