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


class TestWritePathHonoursAliases:
    """The Cypher write path resolves a type's declared identity-field
    spellings exactly as the read path does.

    Until 0.16.1 it did not. `CREATE (:Prospect {npdid: 99})` stored 99 as an
    ordinary property and minted a *separate* identity — and because `n.npdid`
    resolves the alias to the identity, the dot read then answered with the
    minted id while `properties(n)` still showed 99. Two routes, one node, two
    answers. The title half was worse: `CREATE (:Prospect {prospect_name: 'C'})`
    fabricated `Prospect_3` and served that engine-minted string back for
    `n.prospect_name`, over the caller's own value.

    Every assertion below is therefore a *two-route* one: `properties(n)` (the
    materialised map) and `n.<alias>` (the dot route) must agree, and agree with
    the value the write supplied.
    """

    @staticmethod
    def _both_routes(g, npdid):
        """`(properties(n), {alias: dot-read})` for one node."""
        row = g.cypher(
            "MATCH (n:Prospect) WHERE n.npdid = $i "
            "RETURN properties(n) AS p, n.npdid AS pid, n.prospect_name AS pname, "
            "n.id AS id, n.title AS title",
            params={"i": npdid},
        ).to_list()
        assert len(row) == 1, f"expected exactly one node with npdid={npdid}, got {len(row)}"
        return row[0]

    def test_create_promotes_both_declared_spellings(self, graph_with_aliases):
        graph_with_aliases.cypher("CREATE (:Prospect {npdid: 99, prospect_name: 'Delta', status: 'new'})")
        row = self._both_routes(graph_with_aliases, 99)

        # The dot route and the canonical field agree…
        assert row["pid"] == 99 and row["id"] == 99
        assert row["pname"] == "Delta" and row["title"] == "Delta"
        # …and so does the materialised map.
        assert row["p"]["npdid"] == 99
        assert row["p"]["prospect_name"] == "Delta"
        assert row["p"]["id"] == 99
        assert row["p"]["title"] == "Delta"
        assert row["p"]["status"] == "new"

    def test_create_does_not_fabricate_a_title_over_the_supplied_one(self, graph_with_aliases):
        """The `<Label>_<n>` fabrication is the last resort it always was — it
        must not override a value the caller supplied under the type's own
        title column."""
        graph_with_aliases.cypher("CREATE (:Prospect {npdid: 50, prospect_name: 'Epsilon'})")
        titles = graph_with_aliases.cypher("MATCH (n:Prospect {npdid: 50}) RETURN n.title AS t").to_list()
        assert titles == [{"t": "Epsilon"}]
        # A CREATE that supplies neither spelling still gets a fabricated title
        # rather than a null one.
        graph_with_aliases.cypher("CREATE (:Prospect {npdid: 51})")
        fabricated = graph_with_aliases.cypher("MATCH (n:Prospect {npdid: 51}) RETURN n.title AS t").scalar()
        assert isinstance(fabricated, str) and fabricated.startswith("Prospect_")

    def test_the_promoted_value_is_not_also_stored_as_a_property(self, graph_with_aliases):
        """Promoted, not duplicated — `add_nodes` keeps its id/title columns out
        of the property columns, and CREATE now agrees. A stored copy would
        shadow the identity in `properties(n)` (the materialiser prefers a key
        the map already carries) while the dot route kept resolving to the
        identity, which is the disagreement this whole path closes."""
        graph_with_aliases.cypher("CREATE (:Prospect {npdid: 60, prospect_name: 'Zeta'})")
        # One authority for each field: rewriting the identity moves BOTH routes.
        graph_with_aliases.cypher("MATCH (n:Prospect {npdid: 60}) SET n.prospect_name = 'Zeta-2'")
        row = self._both_routes(graph_with_aliases, 60)
        assert row["pname"] == "Zeta-2"
        assert row["title"] == "Zeta-2"
        assert row["p"]["prospect_name"] == "Zeta-2"

    def test_merge_creates_through_the_same_promotion(self, graph_with_aliases):
        graph_with_aliases.cypher("MERGE (n:Prospect {npdid: 101}) ON CREATE SET n.status = 'fresh'")
        row = self._both_routes(graph_with_aliases, 101)
        assert row["pid"] == 101 and row["id"] == 101
        assert row["p"]["npdid"] == 101
        assert row["p"]["status"] == "fresh"

    def test_merge_matches_the_node_its_own_create_arm_made(self, graph_with_aliases):
        """The proof that the promotion is real: a second MERGE on the same
        aliased key must find the node rather than create a twin."""
        graph_with_aliases.cypher("MERGE (n:Prospect {npdid: 102}) ON CREATE SET n.status = 'first'")
        graph_with_aliases.cypher("MERGE (n:Prospect {npdid: 102}) ON CREATE SET n.status = 'second'")
        rows = graph_with_aliases.cypher("MATCH (n:Prospect) WHERE n.npdid = 102 RETURN n.status AS s").to_list()
        assert rows == [{"s": "first"}], "the second MERGE must match, not create"

    def test_set_on_the_title_alias_is_visible_on_both_routes(self, graph_with_aliases):
        """It used to store an ordinary property that `n.prospect_name` — which
        resolves the alias to the title — could never return: the write was
        invisible on the route that asked for it."""
        graph_with_aliases.cypher("MATCH (n:Prospect {npdid: 1}) SET n.prospect_name = 'Alpha-2'")
        row = self._both_routes(graph_with_aliases, 1)
        assert row["pname"] == "Alpha-2"
        assert row["title"] == "Alpha-2"
        assert row["p"]["prospect_name"] == "Alpha-2"
        assert row["p"]["title"] == "Alpha-2"

    def test_set_on_the_id_alias_is_refused_as_immutable(self, graph_with_aliases):
        """Same answer as the literal `SET n.id`, for the same reason: the
        identity is the row's key, not a value `add_nodes` updates."""
        with pytest.raises(Exception, match="immutable"):
            graph_with_aliases.cypher("MATCH (n:Prospect {npdid: 1}) SET n.npdid = 77")
        assert graph_with_aliases.cypher("MATCH (n:Prospect {npdid: 1}) RETURN n.npdid AS i").to_list() == [{"i": 1}]

    def test_remove_on_the_id_alias_is_refused_as_immutable(self, graph_with_aliases):
        with pytest.raises(Exception, match="immutable"):
            graph_with_aliases.cypher("MATCH (n:Prospect {npdid: 1}) REMOVE n.npdid")
        assert graph_with_aliases.cypher("MATCH (n:Prospect {npdid: 1}) RETURN n.npdid AS i").to_list() == [{"i": 1}]

    def test_remove_on_the_title_alias_clears_the_title(self, graph_with_aliases):
        graph_with_aliases.cypher("MATCH (n:Prospect {npdid: 2}) REMOVE n.prospect_name")
        row = graph_with_aliases.cypher(
            "MATCH (n:Prospect) WHERE n.npdid = 2 RETURN n.prospect_name AS pname, n.title AS title"
        ).to_list()[0]
        assert row["title"] is None
        assert row["pname"] is None

    def test_a_type_without_aliases_is_untouched(self, graph_default_fields):
        """`Item` declares `id`/`title` literally, so nothing resolves and the
        `name`-writes-the-title behaviour every type has is unchanged."""
        graph_default_fields.cypher("CREATE (:Item {id: 40, name: 'W', category: 'C'})")
        row = graph_default_fields.cypher(
            "MATCH (n:Item {id: 40}) RETURN properties(n) AS p, n.name AS name, n.title AS title"
        ).to_list()[0]
        assert row["name"] == "W" and row["title"] == "W"
        assert row["p"]["name"] == "W"
        assert row["p"]["category"] == "C"


def test_a_declared_id_column_agrees_on_both_read_routes_after_create():
    """The accuracy-sweep case, in its own fixture: a type whose id column is
    `pid`. `properties(n)['pid']` said 101 and `n.pid` said the auto-minted id —
    the same node, two answers, one of them invented."""
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame({"pid": [1, 2], "label_text": ["a", "b"]}), "T", "pid")
    g.cypher("CREATE (:T {pid: 101, label_text: 'c'})")

    row = g.cypher("MATCH (n:T) WHERE n.pid = 101 RETURN properties(n) AS p, n.pid AS pid, n.id AS id").to_list()
    assert len(row) == 1
    assert row[0]["pid"] == 101
    assert row[0]["id"] == 101
    assert row[0]["p"]["pid"] == 101
    assert row[0]["p"]["id"] == 101


class TestEmbeddingsOnIdentityAliases:
    """`set_embeddings`/`add_embeddings` take a *source column*, and on an
    aliased type the column the user knows is the alias (`name`), not the
    canonical `title`. Rejecting it made the whole embedding surface
    unreachable for every graph built with a `title_field`."""

    @staticmethod
    def _people():
        return kglite.from_records(
            {
                "nodes": [
                    {
                        "type": "Person",
                        "id_field": "pid",
                        "title_field": "name",
                        "records": [
                            {"pid": 1, "name": "Alpha", "dept": "x"},
                            {"pid": 2, "name": "Beta", "dept": "y"},
                        ],
                    }
                ]
            }
        )

    def test_set_embeddings_accepts_the_title_alias(self):
        g = self._people()
        report = g.set_embeddings("Person", "name", {1: [1.0, 0.0], 2: [0.0, 1.0]})
        assert report["embeddings_stored"] == 2

    def test_set_embeddings_accepts_the_id_alias(self):
        g = self._people()
        report = g.set_embeddings("Person", "pid", {1: [1.0, 0.0]})
        assert report["embeddings_stored"] == 1

    def test_add_embeddings_accepts_the_title_alias(self):
        g = self._people()
        report = g.add_embeddings("Person", "name", {1: [1.0, 0.0]})
        assert report["embeddings_stored"] == 1

    def test_an_unknown_column_is_still_rejected(self):
        """The typo guard survives: aliases widen the accepted set, they do not
        remove it."""
        g = self._people()
        with pytest.raises(ValueError, match="not found on any 'Person' node"):
            g.set_embeddings("Person", "headline", {1: [1.0, 0.0]})

    def test_the_store_name_typo_is_still_rejected(self):
        g = self._people()
        with pytest.raises(ValueError, match="not found on any 'Person' node"):
            g.set_embeddings("Person", "dept_emb", {1: [1.0, 0.0]})

    def test_the_store_is_keyed_by_the_spelling_the_caller_used(self):
        """Store-key decision, pinned from the Python side: resolving `name` to
        the title for the *read* never renames the *store*, so a store written
        as `name` is read back as `name` on every surface."""
        g = self._people()
        g.set_embeddings("Person", "name", {1: [1.0, 0.0], 2: [0.0, 1.0]})

        listing = g.list_embeddings()
        assert [(e["text_column"], e["store_name"]) for e in listing] == [("name", "name_emb")]

        hits = g.vector_search("name", [1.0, 0.0], top_k=1)
        assert [h["id"] for h in hits] == [1]

        scored = g.cypher(
            "MATCH (p:Person) RETURN p.pid AS pid, text_score(p, 'name', $q) AS s ORDER BY s DESC",
            params={"q": [1.0, 0.0]},
        ).to_list()
        assert scored[0]["pid"] == 1
        assert scored[0]["s"] > scored[1]["s"]


class _RecordingEmbedder:
    """Deterministic stub embedder that records the texts it was handed."""

    def __init__(self, dim: int = 3) -> None:
        self.dimension = dim
        self.seen: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.seen.extend(texts)
        return [[float(len(t)), float(sum(map(ord, t)) % 97), 1.0] for t in texts]


class TestEmbedTextsOnIdentityAliases:
    """`embed_texts` is the other half of the same feature as
    `set_embeddings`, and it read the column with a raw property lookup that
    excludes `id`/`title` by contract — so it silently embedded *nothing* for
    an identity column instead of raising. The two halves must agree on what a
    column means."""

    @staticmethod
    def _people():
        return kglite.from_records(
            {
                "nodes": [
                    {
                        "type": "Person",
                        "id_field": "pid",
                        "title_field": "name",
                        "records": [
                            {"pid": 1, "name": "Alpha", "dept": "sales"},
                            {"pid": 2, "name": "Beta", "dept": "eng"},
                        ],
                    }
                ]
            }
        )

    def test_embed_texts_reads_the_title_alias(self):
        g = self._people()
        emb = _RecordingEmbedder()
        g.set_embedder(emb)
        report = g.embed_texts("Person", "name", show_progress=False)
        assert report["embedded"] == 2
        assert report["skipped"] == 0
        assert sorted(emb.seen) == ["Alpha", "Beta"]

    def test_embed_texts_reads_the_canonical_title(self):
        g = self._people()
        emb = _RecordingEmbedder()
        g.set_embedder(emb)
        report = g.embed_texts("Person", "title", show_progress=False)
        assert report["embedded"] == 2
        assert sorted(emb.seen) == ["Alpha", "Beta"]

    def test_embed_texts_reads_a_plain_property_unchanged(self):
        g = self._people()
        emb = _RecordingEmbedder()
        g.set_embedder(emb)
        report = g.embed_texts("Person", "dept", show_progress=False)
        assert report["embedded"] == 2
        assert sorted(emb.seen) == ["eng", "sales"]

    def test_embed_texts_rejects_what_set_embeddings_rejects(self):
        """Parity, the failing direction: a column the ingest guard refuses is
        an error here too, not a silent `{'embedded': 0}`."""
        g = self._people()
        g.set_embedder(_RecordingEmbedder())
        with pytest.raises(ValueError, match="not found on any 'Person' node"):
            g.embed_texts("Person", "headline", show_progress=False)

    @pytest.mark.parametrize(
        "column,embedded",
        [
            ("name", 2),  # the type's title column, by its original name
            ("title", 2),  # the canonical identity field
            ("dept", 2),  # an ordinary stored property
            ("label", 2),  # structural alias -> the node type string
            # An id alias resolves, and every row's value is an integer, so
            # every row is *reported* as skipped rather than silently dropped.
            ("pid", 0),
            ("id", 0),
        ],
    )
    def test_the_two_halves_accept_exactly_the_same_columns(self, column, embedded):
        """Parity: no column may be writable by one half of the feature and
        invisible to the other. Whatever `set_embeddings` accepts,
        `embed_texts` resolves too — and accounts for every node, either as
        embedded or as explicitly skipped."""
        g = self._people()
        assert g.set_embeddings("Person", column, {1: [1.0, 0.0]})["embeddings_stored"] == 1

        g2 = self._people()
        g2.set_embedder(_RecordingEmbedder())
        report = g2.embed_texts("Person", column, show_progress=False)
        assert report["embedded"] == embedded
        assert report["embedded"] + report["skipped"] == 2

    def test_embed_texts_reads_a_string_id_alias(self):
        """The id half of the predicate is really wired, not merely accepted:
        a type whose id column holds text embeds it."""
        g = kglite.from_records(
            {
                "nodes": [
                    {
                        "type": "Sku",
                        "id_field": "code",
                        "records": [{"code": "aa-1"}, {"code": "bb-2"}],
                    }
                ]
            }
        )
        emb = _RecordingEmbedder()
        g.set_embedder(emb)
        report = g.embed_texts("Sku", "code", show_progress=False)
        assert report["embedded"] == 2
        assert sorted(emb.seen) == ["aa-1", "bb-2"]
