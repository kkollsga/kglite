"""OKF (Open Knowledge Format) ingestion tests.

Tier 1 — golden synthetic fixtures (deterministic regression backbone). The
committed bundles under ``tests/fixtures/okf/golden/`` exercise every parse and
build path: labelled concepts, the edge-type ladder, dangling → provisional
stubs, an orphan, nested-frontmatter flattening, a no-frontmatter degrade, the
loose/obsidian wikilink dialect, and reserved-file handling.

(Tier 2 — real-corpus integration against Google's OKF bundles — lives in
``test_okf_corpus.py``.)
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import kglite
from kglite import okf

FIXTURES = Path(__file__).parent / "fixtures" / "okf" / "golden"
OKF_BUNDLE = FIXTURES / "okf"
OBSIDIAN_BUNDLE = FIXTURES / "obsidian"


def _labels(g) -> Counter:
    rows = g.cypher("MATCH (n) RETURN labels(n)[0] AS l").to_list()
    return Counter(r["l"] for r in rows)


def _edge_types(g) -> Counter:
    rows = g.cypher("MATCH ()-[r]->() RETURN type(r) AS t").to_list()
    return Counter(r["t"] for r in rows)


class TestOkfGoldenBundle:
    """Strict OKF dialect over the committed golden bundle."""

    def test_node_count_and_labels(self):
        g = okf.build(str(OKF_BUNDLE), respect_skip=False)
        # 8 concepts (plain.md has no frontmatter → skipped by default) +
        # 1 `tables/ghost` stub + 2 Tag (sales, orders) + 1 Source +
        # 6 Folder (tables, datasets, references, playbooks, meta, guide) = 18.
        assert g.cypher("MATCH (n) RETURN count(n) AS c").to_list()[0]["c"] == 18
        labels = _labels(g)
        assert labels["Folder"] == 6
        assert labels["BigQuery Table"] == 2
        assert labels["BigQuery Dataset"] == 1
        assert labels["Reference"] == 1
        assert labels["Playbook"] == 1
        # profile.md has no top-level `type` → label falls back to metadata.type.
        assert labels["user"] == 1
        assert labels["Guide"] == 1
        assert labels["Section"] == 1
        # only the ghost stub is a bare Concept now (plain.md was skipped).
        assert labels["Concept"] == 1
        # synthesized nodes
        assert labels["Tag"] == 2
        assert labels["Source"] == 1

    def test_concept_id_and_title(self):
        g = okf.build(str(OKF_BUNDLE), respect_skip=False)
        rows = g.cypher("MATCH (n {concept_id:'tables/orders'}) RETURN n.title AS title, n.file_path AS fp").to_list()
        assert rows == [{"title": "Orders", "fp": "tables/orders.md"}]
        # plain.md (no frontmatter) is skipped by default (require_frontmatter).
        plain = g.cypher("MATCH (n {concept_id:'plain'}) RETURN count(n) AS c").to_list()
        assert plain[0]["c"] == 0

    def test_label_and_title_fallback(self):
        g = okf.build(str(OKF_BUNDLE), respect_skip=False)
        # profile.md: no top-level `type`/`title` → label from metadata.type,
        # title from `name` (the Claude-memory shape).
        rows = g.cypher("MATCH (n {concept_id:'meta/profile'}) RETURN labels(n)[0] AS l, n.title AS t").to_list()
        assert rows == [{"l": "user", "t": "User Profile"}]

    def test_require_frontmatter_false_includes_plain(self):
        g = okf.build(str(OKF_BUNDLE), respect_skip=False, require_frontmatter=False)
        plain = g.cypher("MATCH (n {concept_id:'plain'}) RETURN labels(n)[0] AS l").to_list()
        assert plain == [{"l": "Concept"}]

    def test_frontmatter_mapping(self):
        g = okf.build(str(OKF_BUNDLE), respect_skip=False)
        rows = g.cypher("MATCH (n {concept_id:'tables/orders'}) RETURN n.tags AS tags, n.timestamp AS ts").to_list()
        # `tags` list → JSON string; ISO timestamp stays a string.
        assert rows[0]["tags"] == '["sales","orders"]'
        assert rows[0]["ts"] == "2026-05-28T14:30:00Z"
        # nested `metadata:` flattens to dotted keys.
        meta = g.cypher(
            "MATCH (n {concept_id:'meta/profile'}) RETURN n.`metadata.type` AS mt, n.`metadata.scope` AS ms"
        ).to_list()
        assert meta == [{"mt": "user", "ms": "project"}]

    def test_edge_type_ladder(self):
        g = okf.build(str(OKF_BUNDLE), respect_skip=False)
        et = _edge_types(g)
        assert et["JOINS_WITH"] == 1  # "# Joins" section
        assert et["PART_OF"] == 1  # explicit link title
        assert et["CITES"] == 2  # "# Citations": internal note + external Source
        assert et["LINKS_TO"] == 1  # untyped (the dangling ghost link)
        assert et["CONTAINS"] == 7  # folder → concept across the 6 dirs
        assert et["TAGGED"] == 3  # orders→{sales,orders}, customers→sales

        # spot-check endpoints of the typed edges
        joins = g.cypher("MATCH (a)-[:JOINS_WITH]->(b) RETURN a.concept_id AS a, b.concept_id AS b").to_list()
        assert joins == [{"a": "tables/orders", "b": "tables/customers"}]
        contains = g.cypher("MATCH (f:Folder)-[:CONTAINS]->(c) RETURN f.id AS f, c.concept_id AS c").to_list()
        pairs = {(r["f"], r["c"]) for r in contains}
        assert ("tables", "tables/orders") in pairs
        assert ("guide", "guide/intro") in pairs

    def test_tag_nodes_connect_concepts(self):
        g = okf.build(str(OKF_BUNDLE), respect_skip=False)
        # the shared `sales` tag links both tables through a Tag hub (the
        # densification that makes clustering meaningful).
        tagged = g.cypher("MATCH (a)-[:TAGGED]->(:Tag {id:'sales'}) RETURN a.concept_id AS a").to_list()
        assert {r["a"] for r in tagged} == {"tables/orders", "tables/customers"}

    def test_external_citation_becomes_source(self):
        g = okf.build(str(OKF_BUNDLE), respect_skip=False)
        # the external citation URL became a Source node with a CITES edge.
        src = g.cypher("MATCH (a {concept_id:'tables/orders'})-[:CITES]->(s:Source) RETURN s.id AS url").to_list()
        assert any("cloud.google.com" in r["url"] for r in src)

    def test_folder_nodes_and_index_enrichment(self):
        g = okf.build(str(OKF_BUNDLE), respect_skip=False)
        # the tables/ directory is a Folder containing its concepts...
        contained = g.cypher("MATCH (:Folder {id:'tables'})-[:CONTAINS]->(c) RETURN c.concept_id AS c").to_list()
        assert {r["c"] for r in contained} == {"tables/orders", "tables/customers"}
        # ...and its title comes from tables/index.md (reserved file recovered).
        title = g.cypher("MATCH (f:Folder {id:'tables'}) RETURN f.title AS t").to_list()
        assert title == [{"t": "All Tables"}]

    def test_dangling_link_becomes_provisional_stub(self):
        g = okf.build(str(OKF_BUNDLE), respect_skip=False)
        stubs = g.cypher("MATCH (n {_provisional:true}) RETURN n.concept_id AS id").to_list()
        assert stubs == [{"id": "tables/ghost"}]

    def test_orphan_detectable(self):
        g = okf.build(str(OKF_BUNDLE), respect_skip=False)
        # With Folder nodes every concept has a structural CONTAINS edge, so a
        # meaningful "orphan" is one with no *semantic* edge (exclude the
        # structural CONTAINS/TAGGED). The playbook is deliberately unlinked.
        deg = g.cypher(
            "MATCH (n {concept_id:'playbooks/incident'}) "
            "OPTIONAL MATCH (n)-[r]-(m) WHERE NOT type(r) IN ['CONTAINS', 'TAGGED'] "
            "RETURN count(r) AS d"
        ).to_list()
        assert deg[0]["d"] == 0

    def test_reserved_index_not_a_node(self):
        g = okf.build(str(OKF_BUNDLE), respect_skip=False)
        # index.md must not appear as a concept.
        assert g.cypher("MATCH (n {concept_id:'index'}) RETURN count(n) AS c").to_list()[0]["c"] == 0

    def test_build_is_deterministic(self):
        a = okf.build(str(OKF_BUNDLE), respect_skip=False)
        b = okf.build(str(OKF_BUNDLE), respect_skip=False)
        for q in (
            "MATCH (n) RETURN count(n) AS c",
            "MATCH ()-[r]->() RETURN count(r) AS c",
        ):
            assert a.cypher(q).to_list() == b.cypher(q).to_list()

    def test_save_load_roundtrip(self, tmp_path):
        g = okf.build(str(OKF_BUNDLE), respect_skip=False)
        before_n = g.cypher("MATCH (n) RETURN count(n) AS c").to_list()[0]["c"]
        before_e = g.cypher("MATCH ()-[r]->() RETURN count(r) AS c").to_list()[0]["c"]
        path = str(tmp_path / "okf.kgl")
        g.save(path)
        h = kglite.load(path)
        assert h.cypher("MATCH (n) RETURN count(n) AS c").to_list()[0]["c"] == before_n
        assert h.cypher("MATCH ()-[r]->() RETURN count(r) AS c").to_list()[0]["c"] == before_e
        # a property survives the round-trip
        assert (
            h.cypher("MATCH (n {concept_id:'tables/orders'}) RETURN n.tags AS t").to_list()[0]["t"]
            == '["sales","orders"]'
        )


class TestOkfObsidianDialect:
    """Loose / obsidian dialect: [[wikilinks]] + frontmatter without `type`."""

    def test_wikilinks_and_degrade(self):
        g = okf.build(str(OBSIDIAN_BUNDLE), respect_skip=False, dialect="obsidian")
        # alice + bob + carol-missing stub = 3 (MEMORY.md has no frontmatter →
        # skipped by default).
        assert g.cypher("MATCH (n) RETURN count(n) AS c").to_list()[0]["c"] == 3
        # alice has metadata.type → :person; bob (name only) + stub → :Concept.
        assert set(_labels(g)) == {"person", "Concept"}

    def test_wikilink_resolution_and_dangling(self):
        g = okf.build(str(OBSIDIAN_BUNDLE), respect_skip=False, dialect="obsidian")
        edges = g.cypher("MATCH (a)-[r]->(b) RETURN a.concept_id AS a, b.concept_id AS b ORDER BY b").to_list()
        assert {"a": "alice", "b": "bob"} in edges
        assert {"a": "alice", "b": "carol-missing"} in edges
        stubs = g.cypher("MATCH (n {_provisional:true}) RETURN n.concept_id AS id").to_list()
        assert stubs == [{"id": "carol-missing"}]

    def test_wikilinks_ignored_in_strict_dialect(self):
        # In the default (okf) dialect, [[wikilinks]] are not links → no edges.
        g = okf.build(str(OBSIDIAN_BUNDLE), respect_skip=False)
        assert g.cypher("MATCH ()-[r]->() RETURN count(r) AS c").to_list()[0]["c"] == 0


def test_empty_directory_builds_empty_graph(tmp_path):
    g = okf.build(str(tmp_path))
    assert g.cypher("MATCH (n) RETURN count(n) AS c").to_list()[0]["c"] == 0


def test_skip_dirs_prunes_subtrees(tmp_path):
    (tmp_path / "keep").mkdir()
    (tmp_path / "keep" / "a.md").write_text("---\ntype: Note\n---\nkeep")
    (tmp_path / "vendor" / "repos").mkdir(parents=True)
    (tmp_path / "vendor" / "repos" / "b.md").write_text("---\ntype: Note\n---\nclone")
    (tmp_path / "deep" / "cache").mkdir(parents=True)
    (tmp_path / "deep" / "cache" / "c.md").write_text("---\ntype: Note\n---\ndep")
    g = okf.build(str(tmp_path), skip_dirs=["cache", "vendor/repos"])
    ids = {r["id"] for r in g.cypher("MATCH (n) WHERE n.concept_id IS NOT NULL RETURN n.concept_id AS id").to_list()}
    assert ids == {"keep/a"}


def test_kg_skip_excludes_by_default(tmp_path):
    (tmp_path / "keep.md").write_text("---\ntype: Note\n---\nkeep me")
    (tmp_path / "scratch.md").write_text("---\ntype: Note\nkg_skip: true\n---\nignore me")
    # Default: kg_skip files are excluded from the sweep.
    g = okf.build(str(tmp_path))
    ids = {r["id"] for r in g.cypher("MATCH (n) WHERE n.concept_id IS NOT NULL RETURN n.concept_id AS id").to_list()}
    assert ids == {"keep"}
    # respect_skip=False ingests them anyway.
    g2 = okf.build(str(tmp_path), respect_skip=False)
    ids2 = {r["id"] for r in g2.cypher("MATCH (n) WHERE n.concept_id IS NOT NULL RETURN n.concept_id AS id").to_list()}
    assert ids2 == {"keep", "scratch"}
