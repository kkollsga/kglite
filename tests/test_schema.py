"""Tests for schema definition and validation."""

import pandas as pd

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

    def test_non_id_primary_key_rejected(self):
        """MVP enforces only on the identity key; a PK on an arbitrary property
        is rejected at declaration (not silently a no-op)."""
        graph = KnowledgeGraph()
        try:
            graph.define_schema({"nodes": {"T": {"primary_key": "name"}}})
            raise AssertionError("non-id primary_key should be rejected")
        except ValueError as e:
            assert "must be 'id'" in str(e)

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
