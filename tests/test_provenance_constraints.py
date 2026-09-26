"""Constraints on the reserved provenance keys are refused at declaration.

`updated_at`, `git_sha` and `modified_by` are stamped by the engine on writes
to `auto_timestamp` types. The bulk loaders stamp after the constraint gate,
Cypher CREATE before it, and a SET's stamp is never gated, so a constraint on
one of them held on some paths and not others: a forbidden value was stored,
a NOT NULL the stamp satisfies was refused, and UNIQUE admitted duplicates.
Every declaration surface now refuses them, whether or not the type has opted
in yet. A file saved by an earlier version that holds one still loads, and the
constraint is dropped on load with a warning on stderr.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

import kglite

RESERVED = ["updated_at", "git_sha", "modified_by"]
FIXTURE = Path(__file__).parent / "fixtures" / "provenance_constraints_pre_refusal.kgl"


def _edge_graph(opted: bool = True) -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.define_schema(
        {
            "nodes": {"N": {"auto_timestamp": opted}},
            "connections": {"LINKS": {"source": "N", "target": "N", "auto_timestamp": opted}},
        }
    )
    g.cypher("CREATE (:N {id: 1}), (:N {id: 2})")
    return g


def _constraints(g: kglite.KnowledgeGraph) -> list[tuple[str, tuple[str, ...]]]:
    rows = g.cypher("SHOW CONSTRAINTS").to_list()
    return sorted((row["labelsOrTypes"][0], tuple(row["properties"])) for row in rows)


def _assert_refused(message: str, key: str) -> None:
    assert f"'{key}'" in message, message
    assert "engine owns" in message, message


# ── declaration matrix ───────────────────────────────────────────────


@pytest.mark.parametrize("opted", [True, False], ids=["opted-in", "not-opted-in"])
@pytest.mark.parametrize("requirement", ["IS NOT NULL", "IS :: STRING", "IS :: INTEGER"])
@pytest.mark.parametrize("key", RESERVED)
def test_relationship_constraint_on_a_reserved_key_is_refused(key, requirement, opted):
    g = _edge_graph(opted)
    with pytest.raises(kglite.CypherExecutionError) as exc:
        g.cypher(f"CREATE CONSTRAINT FOR ()-[r:LINKS]-() REQUIRE r.{key} {requirement}")
    _assert_refused(str(exc.value), key)
    assert _constraints(g) == []


@pytest.mark.parametrize("opted", [True, False], ids=["opted-in", "not-opted-in"])
@pytest.mark.parametrize("requirement", ["IS NOT NULL", "IS UNIQUE", "IS NODE KEY", "IS :: STRING", "IS :: INTEGER"])
@pytest.mark.parametrize("key", RESERVED)
def test_node_constraint_on_a_reserved_key_is_refused(key, requirement, opted):
    g = _edge_graph(opted)
    with pytest.raises(kglite.CypherExecutionError) as exc:
        g.cypher(f"CREATE CONSTRAINT FOR (n:N) REQUIRE n.{key} {requirement}")
    _assert_refused(str(exc.value), key)
    assert _constraints(g) == []


def test_a_composite_naming_one_reserved_key_is_refused():
    g = _edge_graph()
    with pytest.raises(kglite.CypherExecutionError) as exc:
        g.cypher("CREATE CONSTRAINT FOR (n:N) REQUIRE (n.name, n.git_sha) IS UNIQUE")
    _assert_refused(str(exc.value), "git_sha")
    assert _constraints(g) == []


def _node_declarations(key):
    return {
        "required": {"required": [key]},
        "types": {"types": {key: "string"}},
        "primary_key": {"primary_key": key},
        "unique": {"unique": [["name", key]]},
    }


def _connection_declarations(key):
    return {
        "required_properties": {"required_properties": [key]},
        "property_types": {"property_types": {key: "string"}},
    }


@pytest.mark.parametrize("opted", [True, False], ids=["opted-in", "not-opted-in"])
@pytest.mark.parametrize("field", ["required", "types", "primary_key", "unique"])
@pytest.mark.parametrize("key", RESERVED)
def test_define_schema_node_declaration_on_a_reserved_key_is_refused(key, field, opted):
    g = kglite.KnowledgeGraph()
    declaration = {"auto_timestamp": opted, **_node_declarations(key)[field]}
    with pytest.raises(ValueError) as exc:
        g.define_schema({"nodes": {"Task": declaration}})
    _assert_refused(str(exc.value), key)
    assert _constraints(g) == []


@pytest.mark.parametrize("opted", [True, False], ids=["opted-in", "not-opted-in"])
@pytest.mark.parametrize("field", ["required_properties", "property_types"])
@pytest.mark.parametrize("key", RESERVED)
def test_define_schema_connection_declaration_on_a_reserved_key_is_refused(key, field, opted):
    g = kglite.KnowledgeGraph()
    declaration = {
        "source": "N",
        "target": "N",
        "auto_timestamp": opted,
        **_connection_declarations(key)[field],
    }
    with pytest.raises(ValueError) as exc:
        g.define_schema({"connections": {"LINKS": declaration}})
    _assert_refused(str(exc.value), key)


def test_optional_may_still_name_a_reserved_key():
    # `optional` documents a field and constrains nothing.
    g = kglite.KnowledgeGraph()
    g.define_schema({"nodes": {"Task": {"auto_timestamp": True, "optional": ["updated_at"]}}})
    g.cypher("CREATE (:Task {id: 1})")
    assert g.cypher("MATCH (t:Task) RETURN t.updated_at IS NOT NULL AS stamped").to_list() == [{"stamped": True}]


# ── the reported shapes, pinned at declaration ───────────────────────


def test_case_a_bulk_load_can_no_longer_store_a_value_the_constraint_forbids():
    """Before: the declaration installed, `add_connections(git_sha="abc")` was
    accepted and stored a STRING, while the Cypher CREATE twin was refused."""
    g = _edge_graph()
    with pytest.raises(kglite.CypherExecutionError):
        g.cypher("CREATE CONSTRAINT FOR ()-[r:LINKS]-() REQUIRE r.git_sha IS :: INTEGER")
    g.add_connections(pd.DataFrame([{"s": 1, "t": 2}]), "LINKS", "N", "s", "N", "t", git_sha="abc")
    g.cypher("MATCH (a:N {id: 2}), (b:N {id: 1}) CREATE (a)-[:LINKS]->(b)", git_sha="abc")
    rows = g.cypher("MATCH ()-[r:LINKS]->() RETURN r.git_sha AS s").to_list()
    assert rows == [{"s": "abc"}, {"s": "abc"}]


def test_case_d_required_updated_at_is_refused_instead_of_refusing_every_write():
    """Before: `required: ["updated_at"]` installed, then both CREATE and
    `add_nodes` were refused although the engine stamps the key."""
    g = kglite.KnowledgeGraph()
    with pytest.raises(ValueError):
        g.define_schema({"nodes": {"Task": {"auto_timestamp": True, "required": ["updated_at"]}}})
    g.define_schema({"nodes": {"Task": {"auto_timestamp": True}}})
    g.cypher("CREATE (:Task {id: 1})")
    g.add_nodes(pd.DataFrame({"id": [2]}), "Task", "id")
    rows = g.cypher("MATCH (t:Task) RETURN t.id AS id, t.updated_at IS NOT NULL AS stamped ORDER BY id").to_list()
    assert rows == [{"id": 1, "stamped": True}, {"id": 2, "stamped": True}]


def test_case_e_unique_git_sha_is_refused_instead_of_admitting_duplicates():
    """Before: the UNIQUE installed and two CREATEs stamped with the same
    `git_sha` were both accepted."""
    g = kglite.KnowledgeGraph()
    g.define_schema({"nodes": {"Task": {"auto_timestamp": True}}})
    with pytest.raises(kglite.CypherExecutionError):
        g.cypher("CREATE CONSTRAINT FOR (n:Task) REQUIRE n.git_sha IS UNIQUE")
    assert _constraints(g) == []


# ── a user-written updated_at is overwritten by the stamp on SET too ──


def test_set_of_updated_at_on_a_node_is_overwritten_by_the_stamp():
    g = kglite.KnowledgeGraph()
    g.define_schema({"nodes": {"Task": {"auto_timestamp": True}}})
    g.cypher("CREATE (:Task {id: 1})")
    g.cypher("MATCH (t:Task) SET t.updated_at = 'x'")
    value = g.cypher("MATCH (t:Task) RETURN t.updated_at AS u").to_list()[0]["u"]
    assert value != "x" and value is not None


def test_set_of_updated_at_on_a_relationship_is_overwritten_by_the_stamp():
    g = _edge_graph()
    g.cypher("MATCH (a:N {id: 1}), (b:N {id: 2}) CREATE (a)-[:LINKS]->(b)")
    g.cypher("MATCH ()-[r:LINKS]->() SET r.updated_at = 'x'")
    value = g.cypher("MATCH ()-[r:LINKS]->() RETURN r.updated_at AS u").to_list()[0]["u"]
    assert value != "x" and value is not None


def test_set_of_updated_at_on_a_type_that_has_not_opted_in_is_stored():
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Task {id: 1})")
    g.cypher("MATCH (t:Task) SET t.updated_at = 'x'")
    assert g.cypher("MATCH (t:Task) RETURN t.updated_at AS u").to_list() == [{"u": "x"}]


# ── read-compat: a file saved before the refusal ─────────────────────

SURVIVING = [("LINKS", ("weight",)), ("Task", ("name",))]


def test_a_saved_reserved_key_constraint_is_dropped_on_load(tmp_path, capfd):
    """The fixture was written by the published 0.18.1 wheel (see
    `fixtures/build_provenance_constraint_fixture.py`) with a reserved-key
    constraint in every declaration store, plus two ordinary ones."""
    path = tmp_path / "graph.kgl"
    shutil.copy(FIXTURE, path)
    g = kglite.load(str(path))
    err = capfd.readouterr().err
    assert "dropped a constraint on load" in err, err
    for key in RESERVED:
        assert key in err, err

    assert _constraints(g) == SURVIVING
    names = {row["name"] for row in g.cypher("SHOW CONSTRAINTS").to_list()}
    assert "task_name" in names and "task_sha" not in names
    assert g.cypher("MATCH (t:Task) RETURN count(t) AS c").to_list() == [{"c": 2}]

    # The ordinary constraints still enforce; writes the dropped ones refused
    # or would have judged are accepted and stamped.
    with pytest.raises(kglite.ConstraintViolationError):
        g.cypher("CREATE (:Task {id: 9})")
    g.cypher("CREATE (:Task {id: 3, name: 'c'})", git_sha="abc")
    g.cypher("CREATE (:Task {id: 4, name: 'd'})", git_sha="abc")
    g.add_connections(pd.DataFrame([{"s": 3, "t": 4, "weight": 2}]), "LINKS", "Task", "s", "Task", "t", git_sha="abc")
    rows = g.cypher("MATCH (t:Task) WHERE t.git_sha = 'abc' RETURN count(t) AS c").to_list()
    assert rows == [{"c": 2}]

    # A re-save writes the graph without them, so it reloads silently.
    resaved = tmp_path / "resaved.kgl"
    g.save(str(resaved))
    capfd.readouterr()
    reloaded = kglite.load(str(resaved))
    assert "dropped a constraint" not in capfd.readouterr().err
    assert _constraints(reloaded) == SURVIVING


def test_a_disk_graph_holding_a_reserved_key_constraint_loads_without_it(tmp_path, capfd):
    """A disk graph's metadata goes through the same load step. The state is
    synthesized into the saved generation's `metadata.json`, because no
    current declaration path can write it."""
    path = tmp_path / "dg"
    g = kglite.KnowledgeGraph(storage="disk", path=str(path))
    g.define_schema({"nodes": {"Task": {"auto_timestamp": True}}})
    g.cypher("CREATE (:Task {id: 1, name: 'a'})")
    g.cypher("CREATE CONSTRAINT task_name FOR (n:Task) REQUIRE n.name IS NOT NULL")
    g.save(str(path))
    del g
    current = (path / "CURRENT").read_text(encoding="utf-8").strip()
    meta_path = path / "generations" / current / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["ddl_not_null_constraints"].append(["Task", "updated_at"])
    meta["schema_definition"]["node_schemas"]["Task"]["required_fields"].append("updated_at")
    meta.setdefault("rel_ddl_property_type_constraints", {})["LINKS"] = {"git_sha": "Integer"}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    capfd.readouterr()

    loaded = kglite.load(str(path))
    err = capfd.readouterr().err
    assert "NOT NULL on Task.updated_at" in err, err
    assert "LINKS.git_sha" in err, err
    assert _constraints(loaded) == [("Task", ("name",))]
    loaded.cypher("CREATE (:Task {id: 2, name: 'b'})")
    assert loaded.cypher("MATCH (t:Task) RETURN count(t) AS c").to_list() == [{"c": 2}]
