"""Tests for field alias resolution (Phase 1: Rename Hell fix).

When users call add_nodes(df, 'Type', 'npdid', 'prospect_name'),
the original column names should still work as property accessors in Cypher queries,
where() calls, and the fluent API.
"""

import os
import re
import tempfile

import pandas as pd
import pytest

import kglite


@pytest.fixture
def graph_with_aliases():
    """Graph where original column names differ from canonical id/title."""
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        {
            "npdid": [1, 2, 3],
            "prospect_name": ["Alpha", "Beta", "Gamma"],
            "status": ["active", "inactive", "active"],
        }
    )
    g.add_nodes(df, "Prospect", "npdid", "prospect_name")
    return g


@pytest.fixture
def graph_default_fields():
    """Graph where id/title fields use default names (no aliasing)."""
    g = kglite.KnowledgeGraph()
    df = pd.DataFrame(
        {
            "id": [10, 20, 30],
            "title": ["X", "Y", "Z"],
            "category": ["A", "B", "A"],
        }
    )
    g.add_nodes(df, "Item", "id", "title")
    return g


class TestCypherAliasResolution:
    """Cypher queries should resolve original column names to id/title."""

    def test_cypher_alias_id_field(self, graph_with_aliases):
        """n.npdid should resolve to the id field."""
        result = graph_with_aliases.cypher("MATCH (n:Prospect) RETURN n.npdid ORDER BY n.npdid")
        values = [r["n.npdid"] for r in result]
        assert values == [1, 2, 3]

    def test_cypher_alias_title_field(self, graph_with_aliases):
        """n.prospect_name should resolve to the title field."""
        result = graph_with_aliases.cypher("MATCH (n:Prospect) RETURN n.prospect_name ORDER BY n.prospect_name")
        values = [r["n.prospect_name"] for r in result]
        assert values == ["Alpha", "Beta", "Gamma"]

    def test_cypher_canonical_still_works(self, graph_with_aliases):
        """n.id and n.title should still work alongside aliases."""
        result = graph_with_aliases.cypher("MATCH (n:Prospect) RETURN n.id, n.title ORDER BY n.id")
        assert result[0]["n.id"] == 1
        assert result[0]["n.title"] == "Alpha"

    def test_cypher_where_with_alias(self, graph_with_aliases):
        """WHERE clause should resolve aliases."""
        result = graph_with_aliases.cypher("MATCH (n:Prospect) WHERE n.npdid = 2 RETURN n.prospect_name")
        assert len(result) == 1
        assert result[0]["n.prospect_name"] == "Beta"

    def test_cypher_no_alias_no_interference(self, graph_default_fields):
        """When fields use default names, no aliasing occurs."""
        result = graph_default_fields.cypher("MATCH (n:Item) RETURN n.id ORDER BY n.id")
        values = [r["n.id"] for r in result]
        assert values == [10, 20, 30]

    def test_cypher_regular_property_unaffected(self, graph_with_aliases):
        """Regular properties (not aliased) should work normally."""
        result = graph_with_aliases.cypher(
            "MATCH (n:Prospect) WHERE n.status = 'active' RETURN n.npdid ORDER BY n.npdid"
        )
        values = [r["n.npdid"] for r in result]
        assert values == [1, 3]


class TestPropertiesKeysAliases:
    """properties(n) / keys(n) / n {.*} must surface the alias-recovered
    columns (the non-literal unique_id_field / node_title_field hoisted into
    node.id()/node.title()), matching the canonical `RETURN n` shape.

    Regression: properties(n) had its own property_keys() walk that omitted
    the hoisted alias columns while RETURN n (materialize_node_value)
    included them — properties(n) returned only {id, title, type, status}.
    """

    def test_properties_includes_alias_columns(self, graph_with_aliases):
        result = graph_with_aliases.cypher("MATCH (n:Prospect) WHERE n.npdid = 1 RETURN properties(n) AS p").to_list()
        props = result[0]["p"]
        assert props["npdid"] == 1
        assert props["prospect_name"] == "Alpha"
        # Canonical virtuals + the regular property still present.
        assert props["id"] == 1
        assert props["title"] == "Alpha"
        assert props["type"] == "Prospect"
        assert props["status"] == "active"

    def test_keys_includes_alias_columns(self, graph_with_aliases):
        result = graph_with_aliases.cypher("MATCH (n:Prospect) WHERE n.npdid = 1 RETURN keys(n) AS k").to_list()
        keys = set(result[0]["k"])
        assert {"id", "title", "type", "npdid", "prospect_name", "status"} <= keys

    def test_properties_equals_return_n_properties(self, graph_with_aliases):
        """properties(n) must equal the properties dict inside RETURN n's
        node (the materialize_node_value lockstep guarantee)."""
        node = graph_with_aliases.cypher("MATCH (n:Prospect) WHERE n.npdid = 2 RETURN n").to_list()[0]["n"]
        props = graph_with_aliases.cypher("MATCH (n:Prospect) WHERE n.npdid = 2 RETURN properties(n) AS p").to_list()[
            0
        ]["p"]
        assert props == node["properties"]

    def test_keys_equals_properties_keys(self, graph_with_aliases):
        keys = set(
            graph_with_aliases.cypher("MATCH (n:Prospect) WHERE n.npdid = 3 RETURN keys(n) AS k").to_list()[0]["k"]
        )
        props = graph_with_aliases.cypher("MATCH (n:Prospect) WHERE n.npdid = 3 RETURN properties(n) AS p").to_list()[
            0
        ]["p"]
        assert keys == set(props.keys())

    def test_map_projection_star_includes_alias_columns(self, graph_with_aliases):
        """n {.*} (MapProjection AllProperties) must also surface aliases."""
        result = graph_with_aliases.cypher("MATCH (n:Prospect) WHERE n.npdid = 1 RETURN n {.*} AS m").to_list()
        m = result[0]["m"]
        assert m["npdid"] == 1
        assert m["prospect_name"] == "Alpha"
        # Matches properties(n) for the same node.
        props = graph_with_aliases.cypher("MATCH (n:Prospect) WHERE n.npdid = 1 RETURN properties(n) AS p").to_list()[
            0
        ]["p"]
        assert m == props


class TestFilterAliasResolution:
    """Fluent API where() should resolve original column names."""

    def test_filter_by_alias_id(self, graph_with_aliases):
        """where({'npdid': 2}) should find the node."""
        g = graph_with_aliases
        result = g.where({"type": "Prospect"}).where({"npdid": 2}).collect()
        assert len(result) == 1
        assert result[0]["title"] == "Beta"

    def test_filter_by_alias_title(self, graph_with_aliases):
        """where({'prospect_name': 'Alpha'}) should find the node."""
        g = graph_with_aliases
        result = g.where({"type": "Prospect"}).where({"prospect_name": "Alpha"}).collect()
        assert len(result) == 1
        assert result[0]["id"] == 1

    def test_filter_by_canonical_still_works(self, graph_with_aliases):
        """where({'id': 1}) should still work."""
        g = graph_with_aliases
        result = g.where({"type": "Prospect"}).where({"id": 1}).collect()
        assert len(result) == 1
        assert result[0]["title"] == "Alpha"


class TestSaveLoadAliases:
    """Aliases should survive save/load round-trips."""

    def test_aliases_persist(self, graph_with_aliases):
        """Save and reload should preserve alias resolution."""
        with tempfile.NamedTemporaryFile(suffix=".kglite", delete=False) as f:
            path = f.name

        try:
            graph_with_aliases.save(path)
            g2 = kglite.load(path)

            # Alias should work after reload
            result = g2.cypher("MATCH (n:Prospect) WHERE n.npdid = 1 RETURN n.prospect_name")
            assert len(result) == 1
            assert result[0]["n.prospect_name"] == "Alpha"
        finally:
            os.unlink(path)


class TestMultipleNodeTypes:
    """Aliases are per-node-type — different types can have different aliases."""

    def test_different_aliases_per_type(self):
        g = kglite.KnowledgeGraph()
        df1 = pd.DataFrame(
            {
                "npdid": [1, 2],
                "prospect_name": ["A", "B"],
                "area": ["North", "South"],
            }
        )
        g.add_nodes(df1, "Prospect", "npdid", "prospect_name")

        df2 = pd.DataFrame(
            {
                "well_id": [10, 20],
                "well_name": ["W1", "W2"],
                "depth": [100, 200],
            }
        )
        g.add_nodes(df2, "Well", "well_id", "well_name")

        # Each type should resolve its own aliases
        result = g.cypher("MATCH (n:Prospect) WHERE n.npdid = 1 RETURN n.prospect_name")
        assert result[0]["n.prospect_name"] == "A"

        result = g.cypher("MATCH (n:Well) WHERE n.well_id = 10 RETURN n.well_name")
        assert result[0]["n.well_name"] == "W1"

    def test_alias_does_not_cross_types(self):
        """npdid alias on Prospect should not affect Well type."""
        g = kglite.KnowledgeGraph()
        df1 = pd.DataFrame({"npdid": [1], "name": ["A"]})
        g.add_nodes(df1, "Prospect", "npdid")

        df2 = pd.DataFrame({"id": [10], "title": ["W1"], "npdid_ref": [1]})
        g.add_nodes(df2, "Well", "id", "title")

        # n.npdid_ref on Well should NOT resolve to id (it's a regular property there)
        result = g.cypher("MATCH (n:Well) RETURN n.npdid_ref")
        assert result[0]["n.npdid_ref"] == 1


class TestDescribeAliases:
    """describe() should include alias info in XML."""

    def test_aliases_in_xml(self, graph_with_aliases):
        xml = graph_with_aliases.describe()
        assert 'id_alias="npdid"' in xml
        assert 'title_alias="prospect_name"' in xml

    def test_no_aliases_when_default(self, graph_default_fields):
        xml = graph_default_fields.describe()
        assert 'id_alias="' not in xml
        assert 'title_alias="' not in xml

    def test_schema_adapted_example_uses_alias_anchor(self, graph_with_aliases):
        """describe() emits a per-type example anchored on the type's real
        identifier property (the id alias), with a concrete sampled value —
        so a discovery client copies a query that matches THIS type's key
        shape. (mcp-servers inbox 2026-07-01, Codex/code_mode on-ramp.)"""
        xml = graph_with_aliases.describe()
        assert "<example query=" in xml
        # anchored on the alias (npdid), not the builtin `id`, with a value
        assert "MATCH (n:Prospect {npdid:" in xml

    def test_schema_adapted_example_uses_id_when_no_alias(self, graph_default_fields):
        xml = graph_default_fields.describe()
        assert "MATCH (n:Item {id:" in xml

    def test_schema_adapted_example_is_runnable(self, graph_with_aliases):
        """The generated example must be valid Cypher that matches a node —
        a wrong-property example would be worse than none."""
        xml = graph_with_aliases.describe()
        m = re.search(r'<example query="([^"]+)"', xml)
        assert m, "no example query in describe() output"
        query = m.group(1)
        rows = graph_with_aliases.cypher(query)
        assert len(rows) >= 1, f"generated example returned no rows: {query!r}"


class TestRepeatedAddNodesPreservesAlias:
    """A second add_nodes(..., node_title_field=None) on an existing
    type must not silently rebind the title alias to unique_id_field
    (which made `s.id` resolve to the title slot)."""

    def test_followup_without_title_field_does_not_clobber_id(self):
        g = kglite.KnowledgeGraph()
        g.add_nodes(
            pd.DataFrame([{"id": "x1", "title": "Hello", "x": 1}]),
            "S",
            "id",
            "title",
        )

        # Second pass adds timeseries data, no node_title_field.
        ts = pd.DataFrame(
            [
                {"id": "x1", "ts": "2024-01-01", "y": 10},
                {"id": "x1", "ts": "2024-01-02", "y": 20},
            ]
        )
        g.add_nodes(
            ts,
            "S",
            "id",
            timeseries={"time": "ts", "channels": ["y"], "resolution": "day"},
            conflict_handling="update",
        )

        rows = list(g.cypher("MATCH (s:S) RETURN s.id AS id, s.title AS title"))
        assert len(rows) == 1
        assert rows[0]["id"] == "x1", "id field was clobbered with title"
        assert rows[0]["title"] == "Hello"

    def test_followup_preserves_existing_non_default_title_alias(self):
        """If the first call registered title_alias='prospect_name', a
        follow-up without node_title_field must keep that alias intact."""
        g = kglite.KnowledgeGraph()
        g.add_nodes(
            pd.DataFrame({"npdid": [1], "prospect_name": ["Alpha"], "status": ["active"]}),
            "Prospect",
            "npdid",
            "prospect_name",
        )
        g.add_nodes(
            pd.DataFrame({"npdid": [1], "extra": ["foo"]}),
            "Prospect",
            "npdid",
            conflict_handling="update",
        )

        xml = g.describe()
        assert 'title_alias="prospect_name"' in xml, xml
        rows = list(g.cypher("MATCH (n:Prospect) RETURN n.npdid AS id, n.prospect_name AS name"))
        assert rows[0]["id"] == 1
        assert rows[0]["name"] == "Alpha"
