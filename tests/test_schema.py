"""Tests for schema definition and validation."""

import warnings

import pandas as pd
import pytest

from kglite import KnowledgeGraph


class TestSchemaDefinition:
    def test_define_schema_basic(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"], "value": [10, 20]})
        graph.add_nodes(df, "Node", "id", "name")

        graph.define_schema(
            {
                "nodes": {
                    "Node": {
                        "required": ["id", "title"],
                        "types": {"id": "integer", "title": "string"},
                    }
                }
            }
        )
        assert graph.has_schema()

    def test_clear_schema(self):
        graph = KnowledgeGraph()
        graph.define_schema({"nodes": {"Node": {"required": ["id"]}}})
        assert graph.has_schema()
        graph.clear_schema()
        assert not graph.has_schema()

    def test_schema_definition(self):
        graph = KnowledgeGraph()
        schema_def = {
            "nodes": {
                "Node": {
                    "required": ["id", "title"],
                    "types": {"id": "integer"},
                }
            }
        }
        graph.define_schema(schema_def)
        retrieved = graph.schema_definition()
        assert retrieved is not None


class TestSchemaValidation:
    def test_valid_graph(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"], "value": [10, 20]})
        graph.add_nodes(df, "Node", "id", "name")

        graph.define_schema(
            {
                "nodes": {
                    "Node": {
                        "required": ["id", "title"],
                        "types": {"id": "integer", "title": "string"},
                    }
                }
            }
        )
        errors = graph.validate_schema()
        assert len(errors) == 0

    def test_missing_required_field(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"]})
        graph.add_nodes(df, "Node", "id", "name")

        graph.define_schema(
            {
                "nodes": {
                    "Node": {
                        "required": ["id", "title", "missing_field"],
                    }
                }
            }
        )
        errors = graph.validate_schema()
        assert len(errors) > 0

    def test_type_mismatch(self):
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "name": ["A"], "count": ["not_a_number"]})
        graph.add_nodes(df, "Node", "id", "name")

        graph.define_schema(
            {
                "nodes": {
                    "Node": {
                        "types": {"count": "integer"},
                    }
                }
            }
        )
        errors = graph.validate_schema()
        assert len(errors) > 0

    def test_connection_schema(self):
        graph = KnowledgeGraph()
        users = pd.DataFrame({"id": [1], "name": ["A"]})
        products = pd.DataFrame({"id": [101], "name": ["P"]})
        graph.add_nodes(users, "User", "id", "name")
        graph.add_nodes(products, "Product", "id", "name")
        conn_df = pd.DataFrame({"user_id": [1], "product_id": [101]})
        graph.add_connections(conn_df, "PURCHASED", "User", "user_id", "Product", "product_id")

        graph.define_schema(
            {
                "connections": {
                    "PURCHASED": {"source": "User", "target": "Product"},
                }
            }
        )
        errors = graph.validate_schema()
        assert len(errors) == 0


class TestPrimaryKeyDeclaration:
    """Phase 1 — declaring a PRIMARY KEY (declaration + round-trip only; the
    write-path enforcement lands in a later phase)."""

    def test_declare_primary_key_roundtrips(self):
        graph = KnowledgeGraph()
        graph.define_schema({"nodes": {"Person": {"primary_key": "id", "required": ["id"]}}})
        sd = graph.schema_definition()
        assert sd["nodes"]["Person"]["primary_key"] == "id"

    def test_non_id_primary_key_is_accepted_and_enforced(self):
        """A primary key on an arbitrary property is declarable and enforced —
        unique *and* present (NODE KEY), backed by a unique secondary index.
        Supersedes the earlier restriction that the key had to be ``id``."""
        graph = KnowledgeGraph()
        graph.define_schema({"nodes": {"T": {"primary_key": "name"}}})
        graph.cypher("CREATE (:T {id: 1, name: 'first'})")

        # Duplicating the key is rejected.
        try:
            graph.cypher("CREATE (:T {id: 2, name: 'first'})")
            raise AssertionError("duplicate non-id primary key should be rejected")
        except Exception as e:
            assert "NODE KEY constraint" in str(e), str(e)

        # Omitting it is rejected too — a primary key implies NOT NULL.
        try:
            graph.cypher("CREATE (:T {id: 3})")
            raise AssertionError("missing non-id primary key should be rejected")
        except Exception as e:
            assert "NODE KEY constraint" in str(e), str(e)

        assert graph.cypher("MATCH (n:T) RETURN count(n) AS c").to_dicts()[0]["c"] == 1

    def test_no_primary_key_means_none(self):
        graph = KnowledgeGraph()
        graph.define_schema({"nodes": {"Doc": {"required": ["id"]}}})
        sd = graph.schema_definition()
        assert sd["nodes"]["Doc"].get("primary_key") is None

    def test_primary_key_survives_save_load(self, tmp_path):
        graph = KnowledgeGraph()
        graph.define_schema({"nodes": {"Person": {"primary_key": "id"}}})
        graph.cypher("CREATE (:Person {id: 1, name: 'A'})")
        p = str(tmp_path / "g.kgl")
        graph.save(p)

        import kglite

        reloaded = kglite.load(p)
        sd = reloaded.schema_definition()
        assert sd["nodes"]["Person"]["primary_key"] == "id"


class TestManagedReloadGuard:
    """Per-type `layer` + `add_nodes(managed_reload=True)`: a managed reload
    (research rebuilding from source) never writes a runtime-owned (agent) type."""

    def test_managed_reload_skips_runtime_type(self):
        g = KnowledgeGraph()
        g.define_schema({"nodes": {"Spec": {"layer": "managed"}, "Task": {"layer": "runtime"}}})
        g.cypher("CREATE (:Task {id: 1, status: 'in_progress'})")
        rep = g.add_nodes(
            pd.DataFrame({"id": [1], "status": ["RESET"]}),
            "Task",
            "id",
            "id",
            conflict_handling="update",
            managed_reload=True,
        )
        assert rep.get("skipped_runtime_layer") is True
        # The agent's live field is untouched.
        assert g.cypher("MATCH (t:Task {id: 1}) RETURN t.status AS s").to_dicts()[0]["s"] == "in_progress"

    def test_managed_type_writes_in_managed_reload(self):
        g = KnowledgeGraph()
        g.define_schema({"nodes": {"Spec": {"layer": "managed"}}})
        g.add_nodes(
            pd.DataFrame({"id": [10], "title": ["A"]}),
            "Spec",
            "id",
            "title",
            managed_reload=True,
        )
        assert g.cypher("MATCH (s:Spec) RETURN count(s) AS c").to_dicts()[0]["c"] == 1

    def test_guard_is_opt_in(self):
        """Without managed_reload, a runtime type is written normally."""
        g = KnowledgeGraph()
        g.define_schema({"nodes": {"Task": {"layer": "runtime"}}})
        g.cypher("CREATE (:Task {id: 1, status: 'old'})")
        g.add_nodes(
            pd.DataFrame({"id": [1], "status": ["new"]}),
            "Task",
            "id",
            "id",
            conflict_handling="update",
        )
        assert g.cypher("MATCH (t:Task {id: 1}) RETURN t.status AS s").to_dicts()[0]["s"] == "new"

    def test_layer_roundtrips_and_validates(self):
        g = KnowledgeGraph()
        g.define_schema({"nodes": {"Task": {"layer": "runtime"}}})
        assert g.schema_definition()["nodes"]["Task"]["layer"] == "runtime"
        try:
            KnowledgeGraph().define_schema({"nodes": {"X": {"layer": "bogus"}}})
            raise AssertionError("bogus layer should be rejected")
        except ValueError as e:
            assert "'managed' or 'runtime'" in str(e)

    def test_define_schema_merges_leaving_undeclared_types_alone(self):
        """define_schema merges per node type: a subset call keeps the types it
        does not name. Declaring per module is the natural pattern, and under the
        old replace default it silently un-enforced every type a call omitted."""
        g = KnowledgeGraph()
        g.define_schema({"nodes": {"A": {"primary_key": "id"}, "B": {"layer": "runtime"}}})
        assert set(g.schema_definition()["nodes"]) == {"A", "B"}
        g.define_schema({"nodes": {"A": {"primary_key": "id"}}})  # subset
        assert set(g.schema_definition()["nodes"]) == {"A", "B"}  # B retained

    def test_define_schema_replaces_a_named_type_wholesale(self):
        """Merging is per *type*, not per field: a type the call names takes the
        new declaration entire, so re-declaring it is still how you narrow it."""
        g = KnowledgeGraph()
        g.define_schema({"nodes": {"A": {"required": ["x", "y"], "layer": "runtime"}}})
        g.define_schema({"nodes": {"A": {"required": ["x"]}}})
        assert g.schema_definition()["nodes"]["A"]["required"] == ["x"]
        assert "layer" not in g.schema_definition()["nodes"]["A"]

    def test_define_schema_replace_true_drops_omitted_types_and_warns(self):
        """replace=True restores whole-schema replacement — and names every
        constraint it stops enforcing, so the loss is never silent."""
        g = KnowledgeGraph()
        g.define_schema({"nodes": {"A": {"primary_key": "email"}, "B": {"required": ["t"]}}})
        with pytest.warns(UserWarning, match=r"A\.email \(PRIMARY KEY\)"):
            g.define_schema({"nodes": {"B": {"required": ["t"]}}}, replace=True)
        assert set(g.schema_definition()["nodes"]) == {"B"}

    def test_define_schema_replace_true_is_quiet_when_nothing_is_enforced(self):
        """The warning tracks lost *enforcement*, not merely lost declarations,
        so a replace that drops nothing enforced stays quiet."""
        g = KnowledgeGraph()
        g.define_schema({"nodes": {"A": {"optional": ["x"]}}})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            g.define_schema({"nodes": {"B": {"optional": ["y"]}}}, replace=True)
        assert set(g.schema_definition()["nodes"]) == {"B"}

    def test_auto_timestamp_tag_roundtrips(self, tmp_path):
        g = KnowledgeGraph()
        g.define_schema({"nodes": {"Task": {"auto_timestamp": True}, "Spec": {"layer": "managed"}}})
        sd = g.schema_definition()
        assert sd["nodes"]["Task"]["auto_timestamp"] is True
        # Not opted in → absent (off by default).
        assert "auto_timestamp" not in sd["nodes"]["Spec"]
        # Survives save → load (additive serde field).
        p = str(tmp_path / "g.kgl")
        g.cypher("CREATE (:Task {id: 1})")
        g.save(p)
        import kglite

        assert kglite.load(p).schema_definition()["nodes"]["Task"]["auto_timestamp"] is True


class TestSchemaDialectIsShared:
    """`define_schema` parses through the core chokepoint, not a Python-local walk.

    The grammar lives in `kglite::api::schema_from_value`; the C ABI's
    `kglite_define_schema` parses the same document through the same function.
    That is what keeps the C surface — where a published signature can never
    change within an ABI major — from acquiring a second, permanent dialect.
    These tests pin the Python end of the contract: the accepted shapes, and
    the exception *class* each refusal keeps.
    """

    def test_every_declaration_key_round_trips(self):
        graph = KnowledgeGraph()
        graph.define_schema(
            {
                "nodes": {
                    "User": {
                        "required": ["id", "email"],
                        "optional": ["nickname"],
                        "types": {"email": "string"},
                        "primary_key": "email",
                        "unique": [["handle"]],
                        "layer": "managed",
                        "auto_timestamp": True,
                    }
                },
                "connections": {
                    "KNOWS": {
                        "source": "User",
                        "target": "User",
                        "cardinality": "many-to-many",
                        "required_properties": ["since"],
                        "property_types": {"since": "integer"},
                    }
                },
            }
        )
        constraints = {
            row["name"]: row["type"]
            for row in graph.cypher("CALL db.constraints() YIELD name, type RETURN name, type").to_list()
        }
        assert constraints["User.email"] == "NODE_KEY"
        assert constraints["User.handle"] == "UNIQUENESS"
        assert constraints["User.id"] == "NODE_PROPERTY_EXISTENCE"

    @pytest.mark.parametrize(
        "unique,expected_constraints",
        [
            ("email", ["U.email"]),
            (["email", "handle"], ["U.email", "U.handle"]),
            ([["first", "last"]], ["U.(first, last)"]),
        ],
    )
    def test_the_three_unique_shorthands(self, unique, expected_constraints):
        # A flat list is one single-property constraint *per entry*, not one
        # composite over all of them.
        graph = KnowledgeGraph()
        graph.define_schema({"nodes": {"U": {"unique": unique}}})
        names = sorted(row["name"] for row in graph.cypher("CALL db.constraints() YIELD name RETURN name").to_list())
        assert names == sorted(expected_constraints)

    def test_a_tuple_is_accepted_wherever_a_list_is(self):
        # `py_value_to_value` maps a tuple to the same list Value a list maps
        # to, exactly as PyO3's `Vec<String>` extraction used to accept both.
        graph = KnowledgeGraph()
        graph.define_schema({"nodes": {"N": {"required": ("id", "title")}}})
        assert graph.has_schema()

    def test_either_section_may_stand_alone(self):
        graph = KnowledgeGraph()
        graph.define_schema({"connections": {"R": {"source": "A", "target": "B"}}})
        assert graph.has_schema()
        # A non-dict section is ignored rather than rejected — long-standing
        # behaviour of the walk this delegates to.
        KnowledgeGraph().define_schema({"nodes": ["Person"]})

    @pytest.mark.parametrize(
        "schema,exc,needle",
        [
            ({"nodes": {"P": ["required"]}}, TypeError, "must be a dictionary"),
            (
                {"nodes": {"P": {"required": 7}}},
                TypeError,
                "must be a list of property names",
            ),
            ({"nodes": {"P": {"types": ["id"]}}}, TypeError, "types must be a dictionary"),
            (
                {"nodes": {"P": {"auto_timestamp": 1}}},
                TypeError,
                "must be true or false",
            ),
            (
                {"nodes": {"P": {"primary_key": ""}}},
                ValueError,
                "must name a property.",
            ),
            (
                {"nodes": {"P": {"layer": "other"}}},
                ValueError,
                "'managed' or 'runtime'",
            ),
            (
                {"nodes": {"P": {"unique": 7}}},
                ValueError,
                "must be a property name, a list of property names",
            ),
            (
                {"nodes": {"P": {"unique": [[]]}}},
                ValueError,
                "contains an empty property tuple.",
            ),
            (
                {"connections": {"R": {"target": "B"}}},
                KeyError,
                "missing required 'source' field",
            ),
            (
                {"connections": {"R": {"source": "A"}}},
                KeyError,
                "missing required 'target' field",
            ),
        ],
    )
    def test_each_refusal_keeps_its_python_exception_class(self, schema, exc, needle):
        # The core parser carries the *class* of refusal so the wrapper can
        # raise Python's conventional exception rather than flattening
        # everything to one type.
        graph = KnowledgeGraph()
        with pytest.raises(exc) as excinfo:
            graph.define_schema(schema)
        assert needle in str(excinfo.value)
        assert not graph.has_schema(), "a rejected declaration must install nothing"

    def test_the_property_type_ddl_refusal_names_the_key_the_parser_accepts(self):
        """The refusal used to suggest `field_types` — the Rust field's name.

        The schema dialect's key is `types`; `field_types` is silently ignored,
        so a user who followed the old advice declared nothing and
        `validate_schema()` then reported no violations — the exact
        enforces-nothing-but-reports-success outcome the message exists to
        prevent.
        """
        graph = KnowledgeGraph()
        df = pd.DataFrame({"id": [1], "title": ["a"], "age": ["not an int"]})
        graph.add_nodes(df, "Person", "id", "title", columns=["age"])
        with pytest.raises(Exception) as excinfo:
            graph.cypher("CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS :: INTEGER")
        message = str(excinfo.value)
        assert "field_types" not in message, message
        assert "'types': {'age': 'integer'}" in message, message

        # And the key it names actually works.
        graph.define_schema({"nodes": {"Person": {"types": {"age": "integer"}}}})
        assert [e["error_type"] for e in graph.validate_schema()] == ["type_mismatch"]
