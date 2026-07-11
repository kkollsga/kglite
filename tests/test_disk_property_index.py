"""End-to-end tests for persistent disk-backed property indexes.

Verifies that ``create_index`` on a ``storage='disk'`` graph:
  1. Succeeds and reports ``persistent=True``.
  2. Publishes four ``property_index_*`` files next to the CSR in an
     immutable generation when the graph is saved.
  3. Is consulted by the Cypher planner on ``WHERE n.prop = 'X'`` queries.
  4. Survives a save/load roundtrip (lazy-loaded on first lookup).
"""

from pathlib import Path
import shutil
import tempfile

import pandas as pd
import pytest

from kglite import KnowledgeGraph, load


@pytest.fixture
def disk_dir():
    d = tempfile.mkdtemp(prefix="kglite_prop_idx_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _build_disk_graph(path: str) -> KnowledgeGraph:
    g = KnowledgeGraph(storage="disk", path=path)
    nodes = pd.DataFrame(
        {
            "nid": [f"Q{i}" for i in range(1, 6)],
            "label": ["Norway", "Sweden", "Denmark", "Finland", "Iceland"],
            "type": ["Country"] * 5,
        }
    )
    g.add_nodes(nodes, "Country", "nid", "label")
    return g


def _published_snapshot(path: str) -> Path:
    root = Path(path)
    generation = (root / "CURRENT").read_text().strip()
    return root / "generations" / generation


class TestPersistentIndexBuild:
    def test_create_index_reports_persistent_on_disk(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        info = g.create_index("Country", "label")
        assert info["persistent"] is True
        assert info["created"] is True
        assert info["unique_values"] == 5

    def test_create_index_writes_files_to_disk(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        g.create_index("Country", "label")
        g.save(disk_dir)
        # Mutation-time index files remain private until save atomically
        # publishes a complete immutable generation.
        csr_dir = _published_snapshot(disk_dir) / "seg_000"
        meta = next(csr_dir.glob("property_index_v2_*_meta.bin"))
        stem = meta.name.removesuffix("_meta.bin")
        assert (csr_dir / f"{stem}_keys.bin").exists()
        assert (csr_dir / f"{stem}_offsets.bin").exists()
        assert (csr_dir / f"{stem}_ids.bin").exists()


class TestCollisionFreeIdentity:
    def test_former_delimiter_collision_round_trips_independently(self, disk_dir):
        g = KnowledgeGraph(storage="disk", path=disk_dir)
        g.add_nodes(pd.DataFrame({"id": ["L"], "c": ["left"]}), "a_b", "id", "id")
        g.add_nodes(pd.DataFrame({"id": ["R"], "b_c": ["right"]}), "a", "id", "id")
        g.create_index("a_b", "c")
        g.create_index("a", "b_c")
        g.save(disk_dir)
        first_snapshot = _published_snapshot(disk_dir)
        meta_files = sorted((first_snapshot / "seg_000").glob("property_index_v2_*_meta.bin"))
        assert len(meta_files) == 2
        assert all(len(path.name) < 128 for path in meta_files)
        del g

        reloaded = load(disk_dir)
        left = reloaded.cypher("MATCH (n:a_b {c: 'left'}) RETURN n.id").to_list()
        right = reloaded.cypher("MATCH (n:a {b_c: 'right'}) RETURN n.id").to_list()
        assert left == [{"n.id": "L"}]
        assert right == [{"n.id": "R"}]
        reloaded.save(disk_dir)
        import json

        manifest = json.loads((_published_snapshot(disk_dir) / "seg_manifest.json").read_text())
        pairs = {(entry[0], entry[1]) for entry in manifest["segments"][0]["indexed_prop_ranges"]}
        assert pairs == {
            (_fnv1a_64("a_b"), _fnv1a_64("c")),
            (_fnv1a_64("a"), _fnv1a_64("b_c")),
        }

    def test_drop_one_index_publishes_removal_without_touching_other(self, disk_dir):
        g = KnowledgeGraph(storage="disk", path=disk_dir)
        g.add_nodes(pd.DataFrame({"id": ["L"], "c": ["left"]}), "a_b", "id", "id")
        g.add_nodes(pd.DataFrame({"id": ["R"], "b_c": ["right"]}), "a", "id", "id")
        g.create_index("a_b", "c")
        g.create_index("a", "b_c")
        g.save(disk_dir)
        first_snapshot = _published_snapshot(disk_dir)
        first_files = {path.name: path.read_bytes() for path in (first_snapshot / "seg_000").iterdir()}
        del g

        writer = load(disk_dir)
        assert writer.drop_index("a_b", "c") is True
        writer.save(disk_dir)
        second_snapshot = _published_snapshot(disk_dir)
        assert first_snapshot != second_snapshot
        assert {path.name: path.read_bytes() for path in (first_snapshot / "seg_000").iterdir()} == first_files
        assert len(list((second_snapshot / "seg_000").glob("property_index_v2_*_meta.bin"))) == 1
        del writer

        newest = load(disk_dir)
        assert newest.cypher("MATCH (n:a {b_c: 'right'}) RETURN n.id").to_list() == [{"n.id": "R"}]
        assert newest.drop_index("a_b", "c") is False


class TestPlannerRoutesToIndex:
    def test_equality_lookup_finds_node_via_index(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        g.create_index("Country", "label")
        result = g.cypher("MATCH (n:Country {label: 'Norway'}) RETURN n.nid").to_df()
        assert len(result) == 1
        assert result["n.nid"][0] == "Q1"

    def test_equality_lookup_no_match_returns_empty(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        g.create_index("Country", "label")
        result = g.cypher("MATCH (n:Country {label: 'Atlantis'}) RETURN n.nid").to_df()
        assert len(result) == 0

    def test_lookup_without_index_still_works_via_scan(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        # No index created — fallback scan should still find the node.
        result = g.cypher("MATCH (n:Country {label: 'Sweden'}) RETURN n.nid").to_df()
        assert len(result) == 1
        assert result["n.nid"][0] == "Q2"

    def test_starts_with_uses_prefix_index(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        g.create_index("Country", "label")
        # 4 labels start with {N,S,D,F,I}; only "F" matches Finland.
        result = g.cypher("MATCH (n:Country) WHERE n.label STARTS WITH 'F' RETURN n.nid").to_df()
        assert len(result) == 1
        assert result["n.nid"][0] == "Q4"  # Finland

    def test_starts_with_no_match(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        g.create_index("Country", "label")
        result = g.cypher("MATCH (n:Country) WHERE n.label STARTS WITH 'Z' RETURN n.nid").to_df()
        assert len(result) == 0

    def test_starts_with_works_without_index(self, disk_dir):
        # STARTS WITH falls back to post-filter scan when no index exists.
        g = _build_disk_graph(disk_dir)
        result = g.cypher("MATCH (n:Country) WHERE n.label STARTS WITH 'I' RETURN n.nid").to_df()
        assert len(result) == 1
        assert result["n.nid"][0] == "Q5"  # Iceland


class TestQueryDiagnostics:
    def test_diagnostics_reports_elapsed_and_timeout(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        g.create_index("Country", "label")
        r = g.cypher("MATCH (n:Country {label: 'Norway'}) RETURN n.nid")
        d = r.diagnostics
        assert d is not None
        assert d["elapsed_ms"] >= 0
        assert d["timed_out"] is False
        # Built-in default timeout = 180_000 ms (3 min); user did not override.
        assert d["timeout_ms"] == 180_000

    def test_diagnostics_respects_explicit_timeout(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        r = g.cypher("MATCH (n:Country) RETURN n.nid", timeout_ms=500)
        assert r.diagnostics["timeout_ms"] == 500

    def test_diagnostics_none_when_timeout_disabled(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        r = g.cypher("MATCH (n:Country) RETURN n.nid", timeout_ms=0)
        assert r.diagnostics["timeout_ms"] is None


class TestDescribeAnnotations:
    def test_indexed_regular_property_annotated_memory(self):
        # Memory backend: describe() emits a <properties> block with
        # column stats for every non-title/non-id column, so indexed
        # annotations are verifiable. (Disk inventory skips the block
        # for small types; the annotation plumbing is the same.)
        g = KnowledgeGraph()
        nodes = pd.DataFrame(
            {
                "nid": [f"Q{i}" for i in range(1, 4)],
                "label": ["Alpha", "Beta", "Gamma"],
                "continent": ["Europe"] * 3,
            }
        )
        g.add_nodes(nodes, "Country", "nid", "label")
        g.create_index("Country", "continent")
        d = g.describe()
        # String columns get both equality and prefix indexing.
        assert 'indexed="eq,prefix"' in d

    def test_indexing_hint_in_extensions(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        d = g.describe()
        assert "<indexing hint=" in d


class TestGlobalIndexAndSearch:
    """Cross-type global index + the ``search()`` helper."""

    def _build_multi_type_graph(self, path):
        g = KnowledgeGraph(storage="disk", path=path)
        g.add_nodes(
            pd.DataFrame({"nid": ["Q1", "Q2", "Q3"], "label": ["Norway", "Sweden", "Iceland"]}),
            "Country",
            "nid",
            "label",
        )
        g.add_nodes(
            pd.DataFrame({"nid": ["P1", "P2"], "label": ["Oslo", "Stockholm"]}),
            "City",
            "nid",
            "label",
        )
        return g

    def test_create_global_index_reports_count(self, disk_dir):
        g = self._build_multi_type_graph(disk_dir)
        info = g.create_global_index("label")
        assert info["property"] == "label"
        assert info["unique_values"] == 5
        assert info["created"] is True

    def test_search_finds_node_across_types(self, disk_dir):
        g = self._build_multi_type_graph(disk_dir)
        g.create_global_index("label")
        assert [h["title"] for h in g.search("Oslo")] == ["Oslo"]
        assert [h["title"] for h in g.search("Norway")] == ["Norway"]
        assert g.search("Atlantis") == []

    def test_search_falls_back_to_prefix(self, disk_dir):
        g = self._build_multi_type_graph(disk_dir)
        g.create_global_index("label")
        hits = g.search("S")  # matches Stockholm + Sweden
        titles = sorted(h["title"] for h in hits)
        assert titles == ["Stockholm", "Sweden"]

    def test_search_returns_type_per_hit(self, disk_dir):
        g = self._build_multi_type_graph(disk_dir)
        g.create_global_index("label")
        hits = g.search("Oslo")
        assert hits[0]["type"] == "City"
        assert hits[0]["id_value"] == "P1"

    def test_untyped_cypher_match_uses_global_index(self, disk_dir):
        g = self._build_multi_type_graph(disk_dir)
        g.create_global_index("label")
        # No :Country / :City label on the pattern — only resolvable
        # via the cross-type index.
        r = g.cypher("MATCH (n {label: 'Stockholm'}) RETURN n.nid").to_df()
        assert len(r) == 1
        assert r["n.nid"][0] == "P2"

    def test_search_returns_empty_without_index(self, disk_dir):
        # No create_global_index call — search still works but returns
        # empty (would otherwise require a 124M-node scan on Wikidata).
        g = self._build_multi_type_graph(disk_dir)
        assert g.search("Oslo") == []


class TestPersistenceAcrossReload:
    def test_index_survives_save_and_reload(self, disk_dir):
        g = _build_disk_graph(disk_dir)
        g.create_index("Country", "label")
        g.save(disk_dir)
        del g
        reloaded = load(disk_dir)
        # First lookup after reload triggers lazy mmap open of the index.
        result = reloaded.cypher("MATCH (n:Country {label: 'Denmark'}) RETURN n.nid").to_df()
        assert len(result) == 1
        assert result["n.nid"][0] == "Q3"


def _fnv1a_64(s: str) -> int:
    """Mirror of `InternedKey::from_str` (FNV-1a 64-bit) for manifest
    hash checks. Must stay in sync with `src/graph/storage/interner.rs`."""
    h = 0xCBF29CE484222325
    prime = 0x100000001B3
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * prime) & 0xFFFFFFFFFFFFFFFF
    return h


class TestSegmentManifestRecordsIndexes:
    """PR1 phase 5: every PropertyIndex in a segment should show up in
    the saved ``seg_manifest.json`` under ``indexed_prop_ranges`` so the
    planner (phase 7+) can consult it for pruning."""

    def test_manifest_lists_built_index(self, disk_dir):
        import json

        g = _build_disk_graph(disk_dir)
        g.create_index("Country", "label")
        g.save(disk_dir)
        del g

        manifest = json.loads((_published_snapshot(disk_dir) / "seg_manifest.json").read_text())
        assert len(manifest["segments"]) == 1
        ranges = manifest["segments"][0]["indexed_prop_ranges"]
        # Expect exactly one entry for (Country, label) as StringBloomPlaceholder.
        t_hash = _fnv1a_64("Country")
        p_hash = _fnv1a_64("label")
        assert (t_hash, p_hash) in {(e[0], e[1]) for e in ranges}
        placeholder_entry = next(e for e in ranges if (e[0], e[1]) == (t_hash, p_hash))
        # Phase 5 only emits the placeholder; numeric / bloom variants land later.
        assert placeholder_entry[2] == "StringBloomPlaceholder"

    def test_manifest_empty_without_indexes(self, disk_dir):
        import json

        g = _build_disk_graph(disk_dir)
        g.save(disk_dir)
        manifest = json.loads((_published_snapshot(disk_dir) / "seg_manifest.json").read_text())
        assert manifest["segments"][0]["indexed_prop_ranges"] == []

    def test_manifest_survives_reload_and_resave(self, disk_dir):
        """Indexes present on disk (but not yet looked up in the loaded
        session) must still show up in the manifest after a resave —
        the disk-scan fallback covers this."""
        import json

        g = _build_disk_graph(disk_dir)
        g.create_index("Country", "label")
        g.save(disk_dir)
        del g

        # Reload without querying the index; then resave. The cache is
        # empty, so the manifest entry comes from scan_segment_hashes.
        reloaded = load(disk_dir)
        reloaded.save(disk_dir)

        manifest = json.loads((_published_snapshot(disk_dir) / "seg_manifest.json").read_text())
        ranges = manifest["segments"][0]["indexed_prop_ranges"]
        t_hash = _fnv1a_64("Country")
        p_hash = _fnv1a_64("label")
        assert (t_hash, p_hash) in {(e[0], e[1]) for e in ranges}

    def test_manifest_lists_global_index_too(self, disk_dir):
        """Cross-type global indexes use a different file prefix
        (``global_index_*``) so they won't appear via
        ``scan_segment_hashes``. That's intentional for phase 5 — the
        manifest's ``indexed_prop_ranges`` tracks per-type indexes,
        which the phase 7 planner will consult for
        ``MATCH (n:Type {prop: v})``. This test just pins the behaviour."""
        import json

        g = KnowledgeGraph(storage="disk", path=disk_dir)
        g.add_nodes(
            pd.DataFrame({"nid": ["Q1"], "label": ["Norway"]}),
            "Country",
            "nid",
            "label",
        )
        g.create_global_index("label")
        g.save(disk_dir)

        manifest = json.loads((_published_snapshot(disk_dir) / "seg_manifest.json").read_text())
        # Only the global index was built — no per-type indexes.
        assert manifest["segments"][0]["indexed_prop_ranges"] == []
