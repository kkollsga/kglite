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
