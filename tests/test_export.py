"""Tests for export: GraphML, GEXF, D3 JSON, CSV, export_string."""

import json
import os
import tempfile

import pandas as pd
import pytest

from kglite import KnowledgeGraph


class TestExportToFile:
    def test_export_graphml(self, small_graph):
        with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
            path = f.name
        try:
            small_graph.export(path, format="graphml")
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_export_gexf(self, small_graph):
        with tempfile.NamedTemporaryFile(suffix=".gexf", delete=False) as f:
            path = f.name
        try:
            small_graph.export(path, format="gexf")
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_export_d3_json(self, small_graph):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            small_graph.export(path, format="d3")
            assert os.path.exists(path)
            with open(path) as fh:
                data = json.load(fh)
            assert "nodes" in data
            assert "links" in data
        finally:
            os.unlink(path)

    def test_export_json(self, small_graph):
        """Export as JSON (alias for d3)."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            small_graph.export(path, format="json")
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)


class TestExportString:
    def test_graphml_string(self, small_graph):
        result = small_graph.export_string(format="graphml")
        assert isinstance(result, str)
        assert len(result) > 0
        assert "graphml" in result.lower()

    def test_d3_string(self, small_graph):
        result = small_graph.export_string(format="d3")
        data = json.loads(result)
        assert "nodes" in data
        assert "links" in data

    def test_export_empty_graph(self):
        graph = KnowledgeGraph()
        result = graph.export_string(format="graphml")
        assert isinstance(result, str)


class TestExportStringFormats:
    """Ensure all string export formats produce valid output."""

    def test_gexf_string(self, small_graph):
        result = small_graph.export_string(format="gexf")
        assert isinstance(result, str)
        assert "gexf" in result.lower()

    def test_graphml_contains_nodes(self, small_graph):
        result = small_graph.export_string(format="graphml")
        assert "<node" in result
        assert "<edge" in result

    def test_d3_node_count(self, small_graph):
        result = small_graph.export_string(format="d3")
        data = json.loads(result)
        assert len(data["nodes"]) >= 2

    def test_special_chars_in_export(self):
        """Properties with special XML characters don't break export."""
        graph = KnowledgeGraph()
        df = pd.DataFrame(
            {
                "id": [1],
                "name": ["O'Brien & <Co>"],
                "desc": ['Has "quotes"'],
            }
        )
        graph.add_nodes(df, "T", "id", "name")
        # Should not crash
        graphml = graph.export_string(format="graphml")
        assert isinstance(graphml, str)
        d3 = graph.export_string(format="d3")
        data = json.loads(d3)
        assert len(data["nodes"]) == 1


class TestExportWithSelection:
    def test_export_selection_only(self, social_graph):
        selection = social_graph.select("Person").where({"city": "Oslo"})
        result = selection.export_string(format="d3", selection_only=True)
        data = json.loads(result)
        assert len(data["nodes"]) > 0

    def test_export_expanded_selection(self, small_graph):
        expanded = small_graph.select("Person").where({"title": "Alice"}).expand(hops=1)
        subgraph = expanded.to_subgraph()
        result = subgraph.export_string(format="d3")
        data = json.loads(result)
        assert len(data["nodes"]) >= 2


# ─────────────────────────────────────────────────────────────────────────
# Property-naming round-trip — surfaced by the test-suite fortification's
# Phase 5 Neo4j conformance check. The d3 export and the to_neo4j adapter
# both rename properties relative to what the user originally set. This
# matters because (a) the d3 format is the on-the-wire shape every export
# consumer sees, and (b) `to_neo4j` is the canonical Neo4j bridge that
# downstream Cypher queries run against — and they expect the original
# column names.
#
# The findings:
#   1. d3 export collapses `unique_id_field` → `id` and `title_field` →
#      `title`. Column names lost. KGLite supports BOTH the original
#      names and the virtual aliases at Cypher time (via the alias
#      machinery), so `n.person_id` and `n.id` both return the same value.
#   2. `keys(n)` returns the *virtual* names (id, title, type) and does
#      NOT include the user's `person_id` or `name` column names. This is
#      a real discoverability bug: the user can READ `n.person_id` but
#      it doesn't appear in the enumerable property list. Xfail-pinned
#      below with a fix sketch.
#   3. `to_neo4j._sanitize_props` further renames `id` → `_kglite_id` for
#      Neo4j collision safety. Documented behaviour; pinned so downstream
#      tooling that targets the on-wire format keeps working.
# ─────────────────────────────────────────────────────────────────────────


class TestPropertyNamingRoundTrip:
    """Pin the property-renaming behaviour for downstream consumers
    (d3 export, to_neo4j, anything reading `keys(n)`)."""

    @staticmethod
    def _fixture():
        """Mini fixture: 2 Person nodes with `person_id` as the unique-id
        column and `name` as the title column. Mirrors how every test
        and downstream tool builds a kglite graph."""
        g = KnowledgeGraph()
        g.add_nodes(
            pd.DataFrame(
                [
                    {"person_id": 1, "name": "Alice", "city": "Oslo"},
                    {"person_id": 2, "name": "Bob", "city": "Bergen"},
                ]
            ),
            "Person",
            "person_id",
            "name",
        )
        return g

    # ── d3 export shape ────────────────────────────────────────────────

    def test_d3_export_flattens_unique_id_and_title_columns(self):
        """d3 export emits `id` (from `person_id`) and `title` (from
        `name`). The original column names DO NOT appear in the output.

        Consequence for downstream consumers: any tool that reads d3
        and writes elsewhere (Neo4j, ArangoDB, plain JSON pipelines)
        sees `id` / `title`, NOT `person_id` / `name`. If round-trip
        of original column names matters, the consumer needs to
        reconstruct them — see test_to_neo4j_renames_id_for_collision
        below for the canonical example.
        """
        g = self._fixture()
        data = json.loads(g.export_string("d3"))
        keys = set(data["nodes"][0].keys())
        # Present:
        assert "id" in keys
        assert "title" in keys
        # Absent — user's original column names lost in the flattening:
        assert "person_id" not in keys, f"d3 export still has person_id; behaviour changed: {keys}"
        assert "name" not in keys, f"d3 export still has name; behaviour changed: {keys}"

    def test_cypher_can_read_original_column_names_despite_d3_flattening(self):
        """Counterpoint to the d3 flattening: the KGLite Cypher engine
        retains the aliasing, so `n.person_id` and `n.name` still
        resolve at query time. This is the divergence — internal model
        preserves the names; the on-wire d3 shape does not."""
        g = self._fixture()
        rows = g.cypher(
            "MATCH (n:Person {person_id: 1}) RETURN n.person_id AS pid, n.name AS nm, n.id AS id, n.title AS tt"
        ).to_list()
        assert rows[0] == {"pid": 1, "nm": "Alice", "id": 1, "tt": "Alice"}

    # ── keys() discoverability bug ─────────────────────────────────────

    @pytest.mark.xfail(
        reason="discoverability bug surfaced by Phase 5 conformance work: "
        "keys(n) returns the virtual names (id/title/type) and omits the "
        "user's `person_id` / `name` columns. n.person_id is READABLE but "
        "not ENUMERABLE — inconsistent. Fix lives in the scalar `keys()` "
        "implementation in cypher executor; should reflect the alias table "
        "for the node's type so callers see the names they set."
    )
    def test_keys_includes_user_set_column_names(self):
        """`keys(n)` should enumerate `person_id` and `name` since both
        are readable via `n.person_id` / `n.name`. Currently returns the
        virtual `['id', 'title', 'type']` set instead."""
        g = self._fixture()
        rows = g.cypher("MATCH (n:Person {person_id: 1}) RETURN keys(n) AS k").to_list()
        keys = set(rows[0]["k"])
        assert "person_id" in keys, f"keys(n) should include 'person_id'; got {sorted(keys)}"
        assert "name" in keys, f"keys(n) should include 'name'; got {sorted(keys)}"

    def test_keys_currently_returns_virtual_names_only(self):
        """Pin the current keys(n) output so the discoverability bug is
        visible. When the xfail test above starts passing, the user
        column names will appear here too — at which point this test
        becomes redundant and can be removed."""
        g = self._fixture()
        rows = g.cypher("MATCH (n:Person {person_id: 1}) RETURN keys(n) AS k").to_list()
        keys = set(rows[0]["k"])
        # Today: virtual names + non-aliased user columns (city).
        assert "id" in keys
        assert "title" in keys
        assert "type" in keys
        assert "city" in keys  # Non-aliased user column does appear.

    # ── to_neo4j naming ────────────────────────────────────────────────

    def test_to_neo4j_renames_id_for_collision_safety(self):
        """to_neo4j's _sanitize_props renames `id` → `_kglite_id` to
        avoid collision with Neo4j's internal `id()` function.

        Side-effect surfaced by the Phase 5 conformance run: any
        downstream Cypher query that selects `p.id` against the
        Neo4j-pushed graph gets NULL — the property is named
        `_kglite_id` after export. Tooling that round-trips queries
        across KGLite ↔ Neo4j must know about this rename.
        """
        from kglite.neo4j_export import _sanitize_props

        raw = {"id": 42, "title": "Alice", "city": "Oslo"}
        sanitized = _sanitize_props(raw, exclude={"type"}, id_key="id")
        assert "_kglite_id" in sanitized
        assert sanitized["_kglite_id"] == 42
        assert "id" not in sanitized
        # Other fields pass through unchanged.
        assert sanitized["title"] == "Alice"
        assert sanitized["city"] == "Oslo"

    def test_to_neo4j_drops_none_valued_props(self):
        """_sanitize_props drops None-valued props because Neo4j cannot
        store NULL property values. Documenting because this changes the
        observable schema: `keys(n)` on the Neo4j side will be smaller
        than on the KGLite side for any node with optional NULL fields."""
        from kglite.neo4j_export import _sanitize_props

        raw = {"id": 1, "name": "Alice", "email": None, "city": "Oslo"}
        sanitized = _sanitize_props(raw, id_key="id")
        assert "email" not in sanitized
        # Non-null props survive.
        assert sanitized["name"] == "Alice"
        assert sanitized["city"] == "Oslo"
