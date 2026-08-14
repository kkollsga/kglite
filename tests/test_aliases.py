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


@pytest.fixture
def completion_graph():
    """Every category the columnar projection-completion pass can recover, at once.

    `Site` declares a wide column set from one seeded row; two later rows carry a
    subset, so for them the type declares columns their row does not hold. The
    declared set deliberately includes:

    * the two **field aliases** (`siteid` -> id, `site_name` -> title), whose
      values live in the node's identity fields rather than in `properties`;
    * the three **soft structural aliases** (`name` -> title, `label` /
      `node_type` -> the type string), which a row that stores them shadows
      (KG-1) and a row that does not falls back to the structural value;
    * four **spatial virtuals** (`location`, `geometry`, `anchor`, `outline`),
      declared as ordinary string columns by the seed row and *also* configured
      as spatial names, so a row without the stored value gets the synthesized
      Point / WKT instead;
    * a **reserved provenance key** (`updated_at`, via `auto_timestamp`), which
      must stay out of `properties(n)` / `keys(n)` on every row;
    * `extra`, a declared column no measured row carries and nothing recovers —
      it must be absent, not null.
    """
    g = kglite.KnowledgeGraph()
    g.define_schema({"nodes": {"Site": {"auto_timestamp": True}}})
    seed = pd.DataFrame(
        {
            "siteid": [1000],
            "site_name": ["Seed"],
            "lat": [60.0],
            "lon": [5.0],
            "wkt": ["POINT(5 60)"],
            "alat": [61.0],
            "alon": [6.0],
            "outline_wkt": ["POLYGON((0 0,1 0,1 1,0 0))"],
            "name": ["seed-name"],
            "label": ["seed-label"],
            "node_type": ["seed-nt"],
            "extra": [42],
            "location": ["seed-loc"],
            "geometry": ["seed-geom"],
            "anchor": ["seed-anchor"],
            "outline": ["seed-outline"],
        }
    )
    g.add_nodes(seed, "Site", "siteid", "site_name")
    sparse = pd.DataFrame(
        {
            "siteid": [1, 2],
            "site_name": ["A", "B"],
            "lat": [59.0, 58.0],
            "lon": [10.0, 11.0],
            "alat": [70.0, 71.0],
            "alon": [20.0, 21.0],
            "wkt": ["POINT(10 59)", "POINT(11 58)"],
            "outline_wkt": ["POLYGON((1 1,2 1,2 2,1 1))", "POLYGON((3 3,4 3,4 4,3 3))"],
        }
    )
    g.add_nodes(sparse, "Site", "siteid", "site_name")
    g.set_spatial(
        "Site",
        location=("lat", "lon"),
        geometry="wkt",
        points={"anchor": ("alat", "alon")},
        shapes={"outline": "outline_wkt"},
    )
    return g


#: `properties(n)` for the sparse row `siteid = 1`, key by key. Nothing here is
#: incidental: drop the soft-alias recovery and `label`/`name`/`node_type` go
#: missing; drop the spatial recovery and `location`/`geometry`/`anchor`/
#: `outline` do; stop filtering provenance and `updated_at` appears; complete
#: from the row instead of the type and `extra` appears as null.
SPARSE_EXPECTED = {
    "id": 1,
    "title": "A",
    "type": "Site",
    "siteid": 1,
    "site_name": "A",
    "lat": 59.0,
    "lon": 10.0,
    "alat": 70.0,
    "alon": 20.0,
    "wkt": "POINT(10 59)",
    "outline_wkt": "POLYGON((1 1,2 1,2 2,1 1))",
    "name": "A",
    "label": "Site",
    "node_type": "Site",
    "location": {"latitude": 59.0, "longitude": 10.0},
    "geometry": "POINT(10 59)",
    "anchor": {"latitude": 70.0, "longitude": 20.0},
    "outline": "POLYGON((1 1,2 1,2 2,1 1))",
}

#: `properties(n)` for the dense seed row — the same type, every declared column
#: stored. Every soft alias and every spatial name is shadowed by its stored
#: value (KG-1), so the completion pass must add *nothing* here.
DENSE_EXPECTED = {
    "id": 1000,
    "title": "Seed",
    "type": "Site",
    "siteid": 1000,
    "site_name": "Seed",
    "lat": 60.0,
    "lon": 5.0,
    "alat": 61.0,
    "alon": 6.0,
    "wkt": "POINT(5 60)",
    "outline_wkt": "POLYGON((0 0,1 0,1 1,0 0))",
    "name": "seed-name",
    "label": "seed-label",
    "node_type": "seed-nt",
    "extra": 42,
    "location": "seed-loc",
    "geometry": "seed-geom",
    "anchor": "seed-anchor",
    "outline": "seed-outline",
}


class TestColumnarProjectionCompletion:
    """Exact output of the projection-completion pass, per category.

    The pass recovers what a row does not store but its *type* declares. Its
    cost used to be paid per materialized node over every declared key; the
    recoverable set is a per-type fact, so it is now derived from the type's
    schema instead. These cells pin the output that equivalence has to hold —
    they are byte-exact dict comparisons on purpose: a category silently
    dropped from the derivation is a key silently missing from `properties(n)`,
    which no subset assertion would catch.
    """

    def test_sparse_row_properties_are_exact(self, completion_graph):
        props = completion_graph.cypher("MATCH (n:Site {siteid: 1}) RETURN properties(n) AS p").to_list()[0]["p"]
        assert props == SPARSE_EXPECTED

    def test_dense_row_properties_are_exact(self, completion_graph):
        props = completion_graph.cypher("MATCH (n:Site {siteid: 1000}) RETURN properties(n) AS p").to_list()[0]["p"]
        assert props == DENSE_EXPECTED

    def test_declared_but_unrecoverable_column_stays_absent(self, completion_graph):
        """`extra` is declared by the type and carried by no sparse row.

        It must be *absent*, not present-and-null: a completion pass that
        inserted every declared key would change what `keys(n)` reports.
        """
        keys = completion_graph.cypher("MATCH (n:Site {siteid: 2}) RETURN keys(n) AS k").to_list()[0]["k"]
        assert "extra" not in keys
        assert "updated_at" not in keys, "reserved provenance keys stay out of the materialised value"

    @pytest.mark.parametrize("siteid,expected", [(1, SPARSE_EXPECTED), (1000, DENSE_EXPECTED)])
    def test_every_materialisation_route_agrees(self, completion_graph, siteid, expected):
        """`properties(n)`, `keys(n)`, `RETURN n` and `n {.*}` share one pass."""
        g = completion_graph
        props = g.cypher(f"MATCH (n:Site {{siteid: {siteid}}}) RETURN properties(n) AS p").to_list()[0]["p"]
        keys = g.cypher(f"MATCH (n:Site {{siteid: {siteid}}}) RETURN keys(n) AS k").to_list()[0]["k"]
        node = g.cypher(f"MATCH (n:Site {{siteid: {siteid}}}) RETURN n").to_list()[0]["n"]
        star = g.cypher(f"MATCH (n:Site {{siteid: {siteid}}}) RETURN n {{.*}} AS m").to_list()[0]["m"]
        assert props == expected
        # Sorted, and equal as a *sequence*: `keys(n)` no longer builds the
        # property map to read its keys off, so the emitted order is a contract
        # of its own rather than a by-product of `BTreeMap::into_keys`.
        assert keys == sorted(expected)
        assert node["properties"] == expected
        assert star == expected

    def test_completion_survives_save_load(self, completion_graph, tmp_path):
        """The pass reads the type's declared schema, which round-trips."""
        path = str(tmp_path / "completion.kgl")
        completion_graph.save(path)
        reloaded = kglite.load(path)
        props = reloaded.cypher("MATCH (n:Site {siteid: 1}) RETURN properties(n) AS p").to_list()[0]["p"]
        assert props == SPARSE_EXPECTED


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
