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
from datetime import date, datetime
import json
from pathlib import Path
import shutil

import pytest

import kglite
from kglite import okf

FIXTURES = Path(__file__).parent / "fixtures" / "okf" / "golden"
OKF_BUNDLE = FIXTURES / "okf"
OBSIDIAN_BUNDLE = FIXTURES / "obsidian"
VAULT_BUNDLE = FIXTURES / "vault"
STRUCTURE_BUNDLE = FIXTURES / "vault-structure"


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
    """Obsidian vault dialect over a bundle with no folders: root notes only."""

    def test_wikilinks_and_degrade(self):
        g = okf.build(str(OBSIDIAN_BUNDLE), respect_skip=False, dialect="obsidian")
        # alice + bob + MEMORY + carol-missing stub = 4. A vault does not
        # require frontmatter, so MEMORY.md is an ordinary note.
        assert g.cypher("MATCH (n) RETURN count(n) AS c").to_list()[0]["c"] == 4
        # All three are root-level with no `type:`, so the label ladder falls
        # through to `Note`; `metadata.type` is not a vault rung. The dangling
        # stub keeps the `Concept` label every dialect gives a stub.
        assert _labels(g) == Counter({"Note": 3, "Concept": 1})
        assert (
            g.cypher("MATCH (n {concept_id:'alice'}) RETURN n.`metadata.type` AS mt").to_list()[0]["mt"] == "person"
        ), "metadata.type survives as an ordinary property"

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


class TestVaultGoldenBundle:
    """The Obsidian vault dialect over the committed ``golden/vault`` bundle.

    Thirteen notes across three top-level folders plus two root notes,
    carrying a ``type:`` override, an ``id:`` override, a stem-collision pair, a
    case-collision pair, a native list and an ISO date — plus the link
    semantics of VAULT.md §5: ``aliases:``, a ``#section`` anchor, wikilink-
    valued frontmatter keys (``depends_on:`` and the reserved ``parent:``),
    inline ``#tags``, an ``![[embed]]`` and one dangling link; VAULT.md §6:
    an ``img/`` folder whose two PNGs and one PDF are reached by all three
    rungs of the resolution ladder, plus one reference to a file that is not
    there and one written inside a heading line; and VAULT.md §2.3–2.4: ``projects.md`` is the folder note for
    ``projects/`` and ``notes/index.md`` is an ordinary note. VAULT.md §7-§8:
    the bundle carries a ``.kglite/vault.yaml`` declaring ``default_label``, a
    case-folding ``keywords`` hub, a ``heading_edges`` entry, a ``types``
    coercion, two indexes, a text index and an ``embed`` target, plus one
    ``.kglite/skills/`` and one ``.kglite/recipes/`` file — so every label and
    edge below is the *declared* vault's. The build *report* for the same
    bundle (the collision findings, the dangling warning, the declaration
    counters, the embed targets) is asserted in Rust, at
    ``okf::build::build_tests::golden_vault_bundle_report`` — the report has no
    Python surface yet.
    """

    def build(self):
        return okf.build(str(VAULT_BUNDLE), dialect="obsidian")

    def test_validate_reads_a_vault_without_being_told_to(self):
        # VAULT.md §9: `okf.validate` is the Python half of `kglite okf check`,
        # which has always read a directory as a vault. It defaulted to `okf`
        # instead, so the same call resolved no wikilink, minted no `Tag`, read
        # no `vault.yaml` — and reported a *healthier* vault than the command
        # did. `okf.build` still defaults to `okf`, which is what the third
        # assertion pins: the two entry points differ on purpose.
        default = okf.validate(str(VAULT_BUNDLE))
        vault = okf.validate(str(VAULT_BUNDLE), dialect="obsidian")
        bundle = okf.validate(str(VAULT_BUNDLE), dialect="okf")
        assert default.counts == vault.counts and default.warnings == vault.warnings
        assert default.counts != bundle.counts

    def test_label_ladder(self):
        # `type:` → `default_label` → top-level folder → `Note`. The vault
        # declares `default_label: Article`, which sits ahead of the folder
        # rung, so every note without a `type:` is an Article.
        assert _labels(self.build()) == Counter(
            {
                "Article": 11,
                "Folder": 3,  # `projects/` has a folder note, so it has no Folder node
                "Tag": 3,
                "Initiative": 2,
                # `faults` and `horizons`, folded from five spellings by the
                # declared case-insensitive hub
                "Keyword": 2,
                "Concept": 1,  # the `[[Missing]]` stub, labelled as every dialect labels one
                # img/diagram.png and img/faults.png
                "Image": 2,
                # img/handbook.pdf, plus the absent img/appendix.pdf stub —
                # the extension labels a file that is not there too
                "Attachment": 2,
                # `.kglite/skills/one.md` and `.kglite/recipes/one.md` — system
                # labels, but ordinary nodes to Cypher (VAULT.md §8)
                "KgliteSkill": 1,
                "KgliteRecipe": 1,
            }
        )

    def test_edge_types(self):
        assert _edge_types(self.build()) == Counter(
            {
                "CONTAINS": 8,
                "LINKS_TO": 7,
                # four notes under the `projects` folder note, plus the reserved
                # `parent:` key on seismic.md
                "CHILD_OF": 5,
                "TAGGED": 4,
                "DEPENDS_ON": 2,  # `depends_on:` names two wikilinks
                "EMBEDS": 1,  # `![[old]]`
                # `## Related topics`, retyped from the ladder's `RELATED` by
                # the vault's `heading_edges`
                "RELATED_TO": 1,
                "HAS_KEYWORD": 4,  # two notes x two folded keywords
                # links.md reaches both images, plus faults.png again from a
                # heading line; seismic.md re-reaches faults.png from another
                # folder by its bare filename
                "HAS_IMAGE": 4,
                # index.md → handbook.pdf, links.md → the absent appendix,
                # and links.md → handbook.pdf again through a plain
                # `[text](…)` link, which §6.1 reads as a reference too
                "HAS_ATTACHMENT": 3,
            }
        )

    def test_ids_are_stems_declared_ids_and_collision_fallbacks(self):
        g = self.build()
        ids = sorted(
            r["id"] for r in g.cypher("MATCH (n) WHERE n.concept_id IS NOT NULL RETURN n.concept_id AS id").to_list()
        )
        assert ids == [
            "Missing",  # the dangling `[[Missing]]` stub keeps its raw name
            "Roadmap",  # case-collision pair: ids are left alone
            "atlas",
            "index",  # `index.md` is an ordinary note in a vault
            "links",
            "mtg-2026-01",  # declared `id:` wins over the stem `meeting`
            "nested",  # a stem, three folders deep
            "notes/alpha",  # stem collision → path-relative fallback
            "old",
            "projects",  # the folder note for `projects/`
            "projects/alpha",
            "roadmap",
            "seismic",
            "welcome",
        ]

    def test_declared_id_is_not_also_a_property(self):
        g = self.build()
        rows = g.cypher("MATCH (n {concept_id:'mtg-2026-01'}) RETURN n.title AS t, n.file_path AS f").to_list()
        assert rows == [{"t": "Kickoff", "f": "notes/meeting.md"}]

    def test_title_falls_back_to_the_first_h1(self):
        g = self.build()
        rows = g.cypher("MATCH (n {concept_id:'welcome'}) RETURN n.title AS t").to_list()
        assert rows == [{"t": "Welcome"}]

    def test_body_is_stored_by_default(self):
        g = self.build()
        body = g.cypher("MATCH (n {concept_id:'welcome'}) RETURN n.body AS b").to_list()[0]["b"]
        assert body.startswith("# Welcome")
        # …and the explicit option still turns it off.
        off = okf.build(str(VAULT_BUNDLE), dialect="obsidian", with_body=False)
        assert off.cypher("MATCH (n {concept_id:'welcome'}) RETURN n.body AS b").to_list() == [{"b": None}]

    def test_require_frontmatter_defaults_off(self):
        # welcome.md has no frontmatter at all and is still a node.
        assert self.build().cypher("MATCH (n {concept_id:'welcome'}) RETURN count(n) AS c").to_list()[0]["c"] == 1
        on = okf.build(str(VAULT_BUNDLE), dialect="obsidian", require_frontmatter=True)
        assert on.cypher("MATCH (n {concept_id:'welcome'}) RETURN count(n) AS c").to_list()[0]["c"] == 0

    def test_lists_stay_native(self):
        g = self.build()
        rows = g.cypher("MATCH (n {concept_id:'atlas'}) RETURN n.keywords AS k, n.tags AS t").to_list()
        assert rows == [{"k": ["faults", "horizons"], "t": ["seismic"]}]
        # The okf dialect still JSON-encodes them.
        j = okf.build(str(VAULT_BUNDLE), require_frontmatter=False)
        assert j.cypher("MATCH (n {concept_id:'projects/atlas'}) RETURN n.keywords AS k").to_list() == [
            {"k": '["faults","horizons"]'}
        ]

    def test_iso_strings_become_temporal_values(self):
        g = self.build()
        rows = g.cypher(
            "MATCH (n {concept_id:'atlas'}) "
            "RETURN n.updated AS u, n.reviewed AS r, "
            "n.updated + duration({days: 1}) AS plus, n.updated < date('2026-02-01') AS lt"
        ).to_list()
        # A date comes back as a `datetime.date`: arithmetic and ordering
        # against date() both work, which a string could not do.
        assert rows[0]["u"] == date(2026, 1, 15)
        assert rows[0]["plus"] == date(2026, 1, 16)
        assert rows[0]["lt"] is True
        assert rows[0]["r"] == datetime(2026, 1, 15, 9, 30)

    def test_folders_come_from_the_path_not_the_stem_id(self):
        g = self.build()
        rows = g.cypher("MATCH (f:Folder)-[:CONTAINS]->(c) RETURN f.id AS f, c.concept_id AS c, c.id AS fid").to_list()
        pairs = {(r["f"], r["c"] if r["c"] is not None else r["fid"]) for r in rows}
        # `projects/` has a folder note, so no Folder CONTAINS its notes.
        assert pairs == {
            ("notes", "links"),
            ("notes", "index"),
            ("notes", "notes/alpha"),
            ("notes", "roadmap"),
            ("notes", "mtg-2026-01"),
            ("notes", "notes/deep"),
            ("notes/deep", "nested"),
            ("archive", "old"),
        }

    def test_links_resolve_through_the_ladder(self):
        g = self.build()
        edges = sorted(
            (r["a"], r["b"])
            for r in g.cypher("MATCH (a)-[:LINKS_TO]->(b) RETURN a.concept_id AS a, b.concept_id AS b").to_list()
        )
        assert edges == [
            ("links", "Roadmap"),  # an exact id
            ("links", "atlas"),  # `[[atlas#Overview]]` — the anchor never resolves
            # the second `[[atlas]]`, written inside the `## Gallery` heading:
            # a different `section` is a different edge (VAULT.md §5.4)
            ("links", "atlas"),
            ("links", "seismic"),  # `[[Seismic interpretation]]` — an alias
            ("nested", "atlas"),
            ("welcome", "atlas"),
            ("welcome", "old"),  # a path link, relative to the linking note
        ]

    def test_only_the_named_dangling_link_is_a_link_stub(self):
        g = self.build()
        stubs = g.cypher("MATCH (n {_provisional:true}) WHERE n.missing IS NULL RETURN n.concept_id AS id").to_list()
        # `![[diagram.png]]` resolves to a real `Image`; the one absent
        # attachment is a stub of its own kind (`missing: true`, §6.6), which
        # is what this filter excludes.
        assert stubs == [{"id": "Missing"}]

    def test_body_link_edges_carry_section_and_anchor(self):
        g = self.build()
        rows = g.cypher(
            "MATCH (a)-[r:LINKS_TO]->(b) WHERE a.concept_id = 'links' "
            "RETURN b.concept_id AS b, r.section AS section, r.anchor AS anchor ORDER BY b, section"
        ).to_list()
        assert rows == [
            {"b": "Roadmap", "section": None, "anchor": None},  # above the first heading
            {"b": "atlas", "section": "Deep dive", "anchor": "Overview"},
            # a link written *in* a heading carries that heading verbatim —
            # the same string the links below it carry
            {"b": "atlas", "section": "Gallery ![in a heading](../img/faults.png) beside [[atlas]]", "anchor": None},
            {"b": "seismic", "section": "Deep dive", "anchor": None},
        ]

    def test_embed_of_a_note_is_an_edge(self):
        g = self.build()
        rows = g.cypher("MATCH (a)-[:EMBEDS]->(b) RETURN a.concept_id AS a, b.concept_id AS b").to_list()
        assert rows == [{"a": "links", "b": "old"}]

    def test_wikilink_valued_frontmatter_keys_are_edges_not_properties(self):
        g = self.build()
        deps = sorted(
            r["b"]
            for r in g.cypher("MATCH (a {concept_id:'seismic'})-[:DEPENDS_ON]->(b) RETURN b.concept_id AS b").to_list()
        )
        assert deps == ["Missing", "atlas"]
        # `parent:` emits the folder note's edge type and direction, not
        # `PARENT` — alongside the one the folder layout gives the same note.
        parents = sorted(
            r["b"]
            for r in g.cypher("MATCH (a {concept_id:'seismic'})-[:CHILD_OF]->(b) RETURN b.concept_id AS b").to_list()
        )
        assert parents == ["atlas", "projects"]
        rows = g.cypher(
            "MATCH (n {concept_id:'seismic'}) RETURN n.depends_on AS d, n.parent AS p, n.reviewers AS r, n.aliases AS a"
        ).to_list()
        assert rows == [
            {
                "d": None,
                "p": None,
                # a list mixing wikilinks with plain strings is never split
                "r": ["[[atlas]]", "ada"],
                "a": ["Seismic interpretation", "seismics"],
            }
        ]

    def test_inline_tags_join_the_frontmatter_hub(self):
        g = self.build()
        tags = sorted(r["t"] for r in g.cypher("MATCH (t:Tag) RETURN t.id AS t").to_list())
        assert tags == ["field-work", "geoscience", "seismic"], "`#incode` / fenced tags do not count"
        tagged = sorted(
            (r["a"], r["t"])
            for r in g.cypher("MATCH (a)-[:TAGGED]->(t:Tag) RETURN a.concept_id AS a, t.id AS t").to_list()
        )
        assert tagged == [
            ("atlas", "seismic"),
            ("seismic", "field-work"),
            ("seismic", "geoscience"),
            ("seismic", "seismic"),
        ]
        # …and the `tags` property still reports only what the frontmatter said.
        assert g.cypher("MATCH (n {concept_id:'seismic'}) RETURN n.tags AS t").to_list() == [{"t": ["seismic"]}]

    def test_the_folder_note_took_the_folders_place(self):
        g = self.build()
        # No `Folder` node for `projects/` at all …
        assert g.cypher("MATCH (f:Folder) RETURN f.id AS f ORDER BY f").to_list() == [
            {"f": "archive"},
            {"f": "notes"},
            {"f": "notes/deep"},
        ]
        # … and its notes hang off the note instead, by the declared edge.
        children = sorted(
            r["c"]
            for r in g.cypher("MATCH (c)-[:CHILD_OF]->(p {concept_id:'projects'}) RETURN c.concept_id AS c").to_list()
        )
        assert children == ["Roadmap", "atlas", "projects/alpha", "seismic"]
        # The folder note itself is at the root, so no Folder contains it.
        assert (
            g.cypher("MATCH (:Folder)-[:CONTAINS]->(n {concept_id:'projects'}) RETURN count(*) AS c").to_list()[0]["c"]
            == 0
        )

    def test_index_md_is_an_ordinary_note_in_a_vault(self):
        g = self.build()
        rows = g.cypher("MATCH (n {concept_id:'index'}) RETURN labels(n)[0] AS l, n.title AS t").to_list()
        assert rows == [{"l": "Article", "t": "Notes index"}]
        # The `okf` dialect still diverts it to the folder's metadata.
        j = okf.build(str(VAULT_BUNDLE), require_frontmatter=False)
        assert j.cypher("MATCH (n {concept_id:'notes/index'}) RETURN count(n) AS c").to_list()[0]["c"] == 0

    def test_hub_nodes_carry_a_title(self):
        g = self.build()
        rows = g.cypher("MATCH (t:Tag) RETURN t.id AS id, t.title AS title ORDER BY id").to_list()
        # The built-in tag hub folds casing in a vault, and this fixture
        # writes every tag in one casing, so each title is its own id.
        assert rows == [
            {"id": "field-work", "title": "field-work"},
            {"id": "geoscience", "title": "geoscience"},
            {"id": "seismic", "title": "seismic"},
        ]

    def test_attachment_nodes_carry_their_stat_metadata(self):
        import datetime

        g = self.build()
        rows = g.cypher(
            "MATCH (n:Image) RETURN n.path AS path, n.title AS title, n.mime AS mime, "
            "n.size_bytes AS size, n.mtime AS mtime ORDER BY path"
        ).to_list()
        assert [r["path"] for r in rows] == ["img/diagram.png", "img/faults.png"]
        assert [r["title"] for r in rows] == ["diagram.png", "faults.png"]
        assert {r["mime"] for r in rows} == {"image/png"}
        # The committed PNGs are 69 bytes each; the point is that `stat` was
        # read and the bytes were not.
        assert [r["size"] for r in rows] == [69, 69]
        assert all(isinstance(r["mtime"], datetime.datetime) for r in rows)
        assert g.cypher("MATCH (n:Attachment {path:'img/handbook.pdf'}) RETURN n.mime AS m").to_list() == [
            {"m": "application/pdf"}
        ]

    def test_image_text_carries_alts_and_using_note_titles(self):
        # VAULT.md §6.3: captions stay text-searchable — an edge property is
        # not. `faults.png` is used by two notes, one alt text between them.
        g = self.build()
        assert g.cypher("MATCH (n:Image {path:'img/faults.png'}) RETURN n.text AS t").to_list() == [
            {"t": "Fault map\nLink semantics\nin a heading\nseismic"}
        ]

    def test_attachment_edges_carry_alt_section_and_ordinal(self):
        g = self.build()
        rows = g.cypher(
            "MATCH (a)-[r:HAS_IMAGE]->(b) RETURN a.concept_id AS src, b.path AS tgt, "
            "r.alt AS alt, r.section AS section, r.ordinal AS ordinal ORDER BY src, ordinal"
        ).to_list()
        assert rows == [
            {"src": "links", "tgt": "img/faults.png", "alt": "Fault map", "section": "Figures", "ordinal": 0},
            # the `![[diagram.png]]` spelling carries no alt at all
            {"src": "links", "tgt": "img/diagram.png", "alt": None, "section": "Figures", "ordinal": 1},
            # written inside the `## Gallery` heading: the reference counts,
            # and its section is that heading verbatim
            {
                "src": "links",
                "tgt": "img/faults.png",
                "alt": "in a heading",
                "section": "Gallery ![in a heading](../img/faults.png) beside [[atlas]]",
                "ordinal": 2,
            },
            # a second note numbers from zero again, above any heading
            {"src": "seismic", "tgt": "img/faults.png", "alt": "Fault map", "section": None, "ordinal": 0},
        ]

    def test_missing_attachment_is_a_provisional_stub(self):
        g = self.build()
        assert g.cypher(
            "MATCH (n {missing:true}) RETURN labels(n)[0] AS label, n.path AS path, n._provisional AS prov"
        ).to_list() == [
            {"label": "Attachment", "path": "img/appendix.pdf", "prov": True},
        ]

    def test_okf_and_loose_still_drop_image_references(self):
        # VAULT.md §6 is an `"obsidian"` rule: the same fixture read as
        # `"loose"` mints no attachment node and no `HAS_*` edge.
        g = okf.build(str(VAULT_BUNDLE), dialect="loose", require_frontmatter=False)
        labels = _labels(g)
        assert labels["Image"] == 0
        assert labels["Attachment"] == 0
        assert _edge_types(g)["HAS_IMAGE"] == 0
        assert _edge_types(g)["HAS_ATTACHMENT"] == 0

    # ── `.kglite/vault.yaml` and `.kglite/` (VAULT.md §7-§8) ──────────────

    def test_the_declared_hub_folds_casing_and_titles_by_frequency(self):
        g = self.build()
        rows = g.cypher("MATCH (k:Keyword) RETURN k.id AS id, k.title AS title ORDER BY id").to_list()
        # atlas.md writes `faults`/`horizons`, seismic.md `faults`/`Faults`/
        # `Horizons`: two nodes, titled by the commonest casing and, for the
        # one-all tie, alphabetically.
        assert rows == [
            {"id": "faults", "title": "faults"},
            {"id": "horizons", "title": "Horizons"},
        ]
        edges = sorted(
            (r["a"], r["k"])
            for r in g.cypher("MATCH (a)-[:HAS_KEYWORD]->(k:Keyword) RETURN a.concept_id AS a, k.id AS k").to_list()
        )
        assert edges == [
            ("atlas", "faults"),
            ("atlas", "horizons"),
            ("seismic", "faults"),
            ("seismic", "horizons"),
        ]
        # …and the key is still a property: a hub reads it, it does not drain it.
        assert g.cypher("MATCH (n {concept_id:'atlas'}) RETURN n.keywords AS k").to_list() == [
            {"k": ["faults", "horizons"]}
        ]

    def test_heading_edges_retype_the_related_topics_links(self):
        g = self.build()
        assert g.cypher("MATCH (a)-[:RELATED_TO]->(b) RETURN a.concept_id AS a, b.concept_id AS b").to_list() == [
            {"a": "links", "b": "roadmap"}
        ]
        # The ladder's own rung for that heading is gone, not doubled.
        assert g.cypher("MATCH ()-[r:RELATED]->() RETURN count(r) AS c").to_list()[0]["c"] == 0

    def test_declared_types_win_over_inference(self):
        g = self.build()
        # atlas.md writes `toc_depth: "2"` — a quoted string, which inference
        # would leave a string. The declaration makes it an integer, so it
        # orders and compares as one.
        rows = g.cypher(
            "MATCH (n:Initiative) RETURN n.concept_id AS id, n.toc_depth AS d, n.toc_depth > 1 AS gt ORDER BY id"
        ).to_list()
        assert rows == [{"id": "atlas", "d": 2, "gt": True}, {"id": "seismic", "d": None, "gt": None}]

    def test_declared_indexes_and_text_index_are_installed(self):
        g = self.build()
        rows = sorted((r["name"], r["type"]) for r in g.cypher("CALL db.indexes()").to_list())
        assert rows == [
            ("Initiative.body", "FULLTEXT"),
            ("Initiative.concept_id", "PROPERTY"),
            ("Initiative.toc_depth", "RANGE"),
        ]
        assert g.has_index("Initiative", "concept_id")
        assert g.has_text_index("Initiative", "body")
        # The BM25 index answers, rather than merely existing: only atlas.md's
        # body holds "umbrella".
        hits = [
            r["id"]
            for r in g.cypher(
                "MATCH (n:Initiative) RETURN n.concept_id AS id, text_bm25(n, 'body', 'umbrella') AS s"
            ).to_list()
            if r["s"] > 0
        ]
        assert hits == ["atlas"]

    def test_the_vault_carries_its_own_skill_and_recipe(self):
        g = self.build()
        assert [s["name"] for s in g.list_skills()] == ["vault_overview"]
        assert g.get_skill("vault_overview")["body"].startswith("Notes with no `type:`")
        assert [(r["recipe"], r["name"]) for r in g.list_recipes()] == [("vault", "by_keyword")]
        recipe = g.get_recipe("vault", "by_keyword")
        assert recipe["recipe_description"] == "Navigating the golden vault."
        # The JSON Schema stayed nested — a flattening reader would have stored
        # one property literally named `properties.keyword.type`.
        assert recipe["parameters"]["properties"]["keyword"] == {"type": "string"}
        assert recipe["cypher"].startswith("MATCH (n)-[:HAS_KEYWORD]->")

    def test_okf_and_loose_ignore_the_vault_config(self):
        # The same bundle read as `loose`: no `Article`, no `Keyword`, and no
        # skill or recipe node — `.kglite/` is a vault construct.
        g = okf.build(str(VAULT_BUNDLE), dialect="loose", require_frontmatter=False)
        labels = _labels(g)
        assert labels["Article"] == 0
        assert labels["Keyword"] == 0
        assert labels["KgliteSkill"] == 0
        assert g.cypher("CALL db.indexes()").to_list() == []

    def test_build_is_deterministic(self):
        a, b = self.build(), self.build()
        for q in ("MATCH (n) RETURN count(n) AS c", "MATCH ()-[r]->() RETURN count(r) AS c"):
            assert a.cypher(q).to_list() == b.cypher(q).to_list()


def test_empty_directory_builds_empty_graph(tmp_path):
    g = okf.build(str(tmp_path))
    assert g.cypher("MATCH (n) RETURN count(n) AS c").to_list()[0]["c"] == 0


def test_skip_dirs_prunes_subtrees(tmp_path):
    (tmp_path / "keep").mkdir()
    (tmp_path / "keep" / "a.md").write_text("---\ntype: Note\n---\nkeep", encoding="utf-8")
    (tmp_path / "vendor" / "repos").mkdir(parents=True)
    (tmp_path / "vendor" / "repos" / "b.md").write_text("---\ntype: Note\n---\nclone", encoding="utf-8")
    (tmp_path / "deep" / "cache").mkdir(parents=True)
    (tmp_path / "deep" / "cache" / "c.md").write_text("---\ntype: Note\n---\ndep", encoding="utf-8")
    g = okf.build(str(tmp_path), skip_dirs=["cache", "vendor/repos"])
    ids = {r["id"] for r in g.cypher("MATCH (n) WHERE n.concept_id IS NOT NULL RETURN n.concept_id AS id").to_list()}
    assert ids == {"keep/a"}


def test_kg_skip_excludes_by_default(tmp_path):
    (tmp_path / "keep.md").write_text("---\ntype: Note\n---\nkeep me", encoding="utf-8")
    (tmp_path / "scratch.md").write_text("---\ntype: Note\nkg_skip: true\n---\nignore me", encoding="utf-8")
    # Default: kg_skip files are excluded from the sweep.
    g = okf.build(str(tmp_path))
    ids = {r["id"] for r in g.cypher("MATCH (n) WHERE n.concept_id IS NOT NULL RETURN n.concept_id AS id").to_list()}
    assert ids == {"keep"}
    # respect_skip=False ingests them anyway.
    g2 = okf.build(str(tmp_path), respect_skip=False)
    ids2 = {r["id"] for r in g2.cypher("MATCH (n) WHERE n.concept_id IS NOT NULL RETURN n.concept_id AS id").to_list()}
    assert ids2 == {"keep", "scratch"}


class TestVaultStructureProfile:
    """``structure:`` — the nodes a note's own body derives (VAULT.md §7.1).

    A **second** golden vault, because the feature is vault-wide: declaring it
    in ``golden/vault`` would move every count that fixture pins. That one
    therefore stays structure-free and is the compatibility record; this one
    declares ``sections:`` + ``chunks:`` (``max_chars: 120``, small on purpose)
    plus ``inherit:`` and ``embed_text:``, over six notes holding a duplicate
    heading path, a section that packs into two chunks, a ``^block-id``
    paragraph, anchored wikilinks that retarget onto all of it, — in
    ``constructs.md`` — callouts (titled, untitled, nested), three fences (one
    captioned ``python``, one bare, one ``~~~``), a three-step procedure with a
    sub-step and a one-item list the ``min_items: 2`` gate excludes, and — in
    ``tables.md`` — a ``Parameters`` table read as nodes, a ``Worked on by``
    table read as edges, and a symbol heading ``key_from_heading:`` relabels.
    """

    def build(self):
        return okf.build(str(STRUCTURE_BUNDLE), dialect="obsidian")

    def test_labels(self):
        assert _labels(self.build()) == Counter(
            {
                "Article": 5,  # welcome, and the four under `structure/`
                # `tables.md` declares `type: Api` — the label
                # `key_from_heading.under_label` gates on.
                "Api": 1,
                "Folder": 1,  # `structure/`
                # Fifteen headings: two in chunky, three in duplicate (the
                # second `## Details` included), one each in links and welcome,
                # four in constructs, four in tables — less the one the symbol
                # rule relabels.
                "Section": 14,
                # Relabelled in place, never duplicated (VAULT.md §7.1).
                "ApiSymbol": 1,
                # chunky packs 2 + its own `^cite-1` chunk; duplicate 3;
                # welcome 1; tables 4 (its intro, its prose, and one per table
                # — a table is prose too). constructs 6 and links 2: each
                # holds one block over the vault's 120-char cap on its own —
                # the nested callout and a two-line paragraph — which the cap
                # splits at its line boundaries (VAULT.md §7.1).
                "Chunk": 19,
                # One node per body row of the `Parameters` table.
                "ApiParameter": 3,
                # The edge table's second row names a note nobody wrote.
                "Concept": 1,
                # constructs.md: two callouts under `## Notes`, one nested
                "Note": 3,
                # every fence qualifies — `langs:` is omitted
                "Example": 3,
                "Procedure": 1,  # the three-step list; the one-item one is out
                "ProcedureStep": 4,  # three steps and one sub-step
            }
        )

    def test_edge_types(self):
        assert _edge_types(self.build()) == Counter(
            {
                # One per heading, relabelled or not: a symbol keeps its
                # section edges (VAULT.md §7.1).
                "HAS_SECTION": 15,
                "PARENT_SECTION": 9,  # only the nested ones
                "NEXT_SECTION": 4,  # duplicate's two `## Details`, constructs' three `##`
                "HAS_CHUNK": 19,
                # chunky's two, two sections of two, and the two pieces the
                # cap forced inside a block chain like any consecutive chunks.
                "NEXT_CHUNK": 6,
                "HAS_NOTE": 3,  # two from the section, one from the callout it nests in
                "HAS_EXAMPLE": 3,
                # `HAS_<UPPER_SNAKE(container)>`, spelled from the label rather
                # than declared (VAULT.md §7.1)
                "HAS_PROCEDURE": 1,
                "HAS_STEP": 4,  # three from the container, one from step 2
                "NEXT_STEP": 2,  # consecutive steps at one level only
                "HAS_PARAMETER": 3,  # the section to each of its rows
                "WORKED_ON_BY": 2,  # one per row of the edge table
                "CONTAINS": 5,
                # Three prose links, plus the two the edge table's own cells
                # state as prose: a row rule reads a cell, never swallows it.
                "LINKS_TO": 5,
            }
        )

    def test_a_section_is_keyed_by_its_whole_heading_path(self):
        g = self.build()
        rows = g.cypher(
            "MATCH (n:Section) RETURN n.concept_id AS id, n.title AS title, n.level AS level, "
            "n.ordinal AS ordinal, n.path AS path ORDER BY id"
        ).to_list()
        assert [r["id"] for r in rows] == [
            "chunky#Chunky",
            "chunky#Chunky#Sub",
            "constructs#Constructs",
            "constructs#Constructs#Examples",
            "constructs#Constructs#Notes",
            "constructs#Constructs#Steps",
            "duplicate#Notes",
            "duplicate#Notes#Details",
            # Obsidian resolves a heading link to the first of that text and
            # has no syntax for a later one, so the second takes `~2`.
            "duplicate#Notes#Details~2",
            "links#Links",
            "tables#Tables",
            # The symbol heading between them is an `ApiSymbol`, not a
            # `Section`, and its sub-headings are Sections under it.
            "tables#Tables#rmsapi.Project.open(path) → Project#Parameters",
            "tables#Tables#rmsapi.Project.open(path) → Project#Worked on by",
            "welcome#Welcome",
        ]
        second = rows[8]
        assert (second["title"], second["level"], second["ordinal"]) == ("Details", 2, 1)
        assert second["path"] == ["Notes", "Details"]

    def test_a_sections_text_is_the_verbatim_slice_below_its_heading(self):
        g = self.build()
        text = g.cypher("MATCH (n {concept_id:'chunky#Chunky#Sub'}) RETURN n.text AS t").to_list()[0]["t"]
        assert text.startswith("\nParagraph one is written long enough")
        assert text.endswith("around it. ^cite-1"), "trailing blank lines trimmed, nothing else"

    def test_chunks_pack_to_the_declared_limit_and_a_block_id_keys_its_own(self):
        g = self.build()
        rows = g.cypher(
            "MATCH (n:Chunk {note_id:'chunky'}) RETURN n.concept_id AS id, n.ordinal AS o, "
            "n.section_id AS section ORDER BY o"
        ).to_list()
        assert [r["id"] for r in rows] == [
            "chunky#Chunky#Sub~chunk1",
            "chunky#Chunky#Sub~chunk2",
            # The one lever an author has over where a section divides, and the
            # only derived id that survives editing around it.
            "chunky#^cite-1",
        ]
        assert {r["section"] for r in rows} == {"chunky#Chunky#Sub"}
        assert len(g.cypher("MATCH (n:Chunk) WHERE n.chunk_hash IS NULL RETURN n").to_list()) == 0

    def test_inherit_and_embed_text_decorate_every_derived_node(self):
        g = self.build()
        rows = g.cypher(
            "MATCH (n {concept_id:'chunky#^cite-1'}) RETURN n.corpus AS corpus, "
            "n.embed_text AS embed, n.note_id AS note"
        ).to_list()
        assert rows == [
            {
                "corpus": "golden",  # `inherit: [corpus]`, from the note's frontmatter
                "embed": "Chunky | Chunky > Sub\n\nThis sentence is addressable on its own, "
                "whatever is written around it. ^cite-1",
                "note": "chunky",
            }
        ]
        # welcome.md carries no `corpus:`, so its nodes carry none either.
        assert g.cypher("MATCH (n {concept_id:'welcome#Welcome'}) RETURN n.corpus AS c").to_list() == [{"c": None}]

    def test_an_anchored_link_retargets_onto_the_derived_node(self):
        g = self.build()
        rows = g.cypher(
            "MATCH (:Article {concept_id:'links'})-[r:LINKS_TO]->(t) "
            "RETURN t.concept_id AS target, r.anchor AS anchor ORDER BY target"
        ).to_list()
        assert rows == [
            # A fragment naming no heading leaves the edge on the note…
            {"target": "chunky", "anchor": "Nowhere"},
            # …a bare heading reaches the first section of that title…
            {"target": "chunky#Chunky#Sub", "anchor": "Sub"},
            # …and a block id reaches its chunk. The anchor is kept either way.
            {"target": "chunky#^cite-1", "anchor": "^cite-1"},
        ]

    def test_an_index_page_with_no_blank_line_still_honours_the_chunk_caps(self, tmp_path):
        """The operator's shape (RMS_HelpDesk, 2026-09-19): a class index
        written as one 400-item list with no paragraph break used to become a
        single 29 kB chunk under ``max_chars: 6000`` — returned whole by a
        recipe, diluting BM25 and making its embedding meaningless.
        """
        vault = tmp_path / "vault"
        (vault / ".kglite").mkdir(parents=True)
        (vault / ".kglite" / "vault.yaml").write_text(
            "kglite_vault: 1\ndefault_label: Article\nstructure:\n"
            "  sections: {label: Section, edge: HAS_SECTION}\n"
            "  chunks: {label: Chunk, edge: HAS_CHUNK, next: NEXT_CHUNK, "
            "max_words: 650, max_chars: 6000}\n",
            encoding="utf-8",
        )
        items = [f"- [[symbol{i}]] — the {i}th entry of the class index" for i in range(400)]
        (vault / "class_index.md").write_text("# Class index\n\n" + "\n".join(items) + "\n", encoding="utf-8")

        report = okf.validate(str(vault), dialect="obsidian")
        assert report.counts["forced_splits"] > 0, "the cap placed no boundary inside the list"

        g = okf.build(str(vault), dialect="obsidian")
        rows = g.cypher("MATCH (n:Chunk) RETURN n.text AS t, n.ordinal AS o ORDER BY o").to_list()
        assert len(rows) > 1
        assert report.counts["forced_splits"] == len(rows) - 1
        assert [r["o"] for r in rows] == list(range(len(rows))), "ordinals stay contiguous"
        for row in rows:
            size = len(row["t"])
            assert size <= 6000, f"a chunk of {size} chars under a 6 000 cap"
            for line in row["t"].splitlines():
                assert line.startswith("- [[symbol"), f"a boundary inside an item: {line!r}"
        assert "\n".join(r["t"] for r in rows) == "\n".join(items), "the list, whole and in order"

    def test_the_three_warnings_the_fixture_is_built_to_produce(self):
        report = okf.validate(str(STRUCTURE_BUNDLE), dialect="obsidian")
        assert report.errors == []
        assert [w.split(":")[0] for w in report.warnings] == [
            "structure/duplicate.md",
            "structure/links.md",
            "dangling link",
        ]
        assert "duplicate heading path `Notes#Details`" in report.warnings[0]
        assert "names no heading or block id in `chunky`" in report.warnings[1]
        # The edge table's second row names a note nobody wrote: a stub and a
        # warning, exactly as a prose link to it would be (VAULT.md §5.6).
        assert report.warnings[2] == "dangling link: `nobody`"

    def test_a_derived_node_is_never_a_file(self, tmp_path):
        """VAULT.md §7.1/§10.1: a derived node carries no ``file_path``, and an
        export therefore writes exactly the notes — not a file per section."""
        g = self.build()
        assert (
            g.cypher(
                "MATCH (n) WHERE n.note_id IS NOT NULL AND n.file_path IS NOT NULL RETURN count(n) AS c"
            ).to_list()[0]["c"]
            == 0
        )
        out = tmp_path / "vault"
        okf.export(g, str(out), source_root=str(STRUCTURE_BUNDLE))
        written = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
        assert written == [
            ".kglite/export-manifest.json",
            "Api/tables.md",
            "Article/chunky.md",
            "Article/constructs.md",
            "Article/duplicate.md",
            "Article/links.md",
            "Article/welcome.md",
        ]

    def test_a_callout_is_a_node_with_its_kind_title_and_stripped_text(self):
        g = self.build()
        rows = g.cypher(
            "MATCH (n:Note) RETURN n.concept_id AS id, n.kind AS kind, n.title AS title, "
            "n.fold AS fold, n.text AS text, n.section_id AS section ORDER BY id"
        ).to_list()
        assert [r["id"] for r in rows] == [
            "constructs#Constructs#Notes~note1",
            "constructs#Constructs#Notes~note2",
            # A nested callout is keyed under the callout it sits in, and still
            # names the section it sits under.
            "constructs#Constructs#Notes~note2~note1",
        ]
        assert [r["kind"] for r in rows] == ["warning", "versionadded", "tip"]
        assert [r["title"] for r in rows] == ["Check the survey datum", None, "Nested"]
        assert [r["fold"] for r in rows] == ["+", None, None]
        assert rows[0]["text"] == "Depth values are metres below MSL."
        assert rows[2]["text"] == "A callout inside a callout hangs off that callout."
        assert {r["section"] for r in rows} == {"constructs#Constructs#Notes"}

    def test_a_fence_is_an_example_with_its_language_code_and_caption(self):
        g = self.build()
        rows = g.cypher(
            "MATCH (n:Example) RETURN n.concept_id AS id, n.lang AS lang, n.code AS code, "
            "n.caption AS caption ORDER BY n.ordinal"
        ).to_list()
        assert [r["lang"] for r in rows] == ["python", None, None]
        assert rows[0]["code"] == "depths = [1, 2]\n"
        assert rows[0]["caption"] == "Read the depths like this:"
        # …and only when the paragraph above ends with a colon.
        assert [r["caption"] for r in rows[1:]] == [None, None]

    def test_the_step_chain_runs_in_source_order_and_a_sub_step_hangs_off_its_step(self):
        g = self.build()
        chain = g.cypher(
            "MATCH (a:ProcedureStep)-[:NEXT_STEP]->(b:ProcedureStep) RETURN a.text AS a, b.text AS b ORDER BY a.ordinal"
        ).to_list()
        assert chain == [
            {"a": "Open the survey.", "b": "Pick the datum."},
            {"a": "Pick the datum.", "b": "Save the selection."},
        ]
        sub = g.cypher(
            "MATCH (s:ProcedureStep)-[:HAS_STEP]->(t:ProcedureStep) "
            "RETURN s.text AS step, t.text AS sub, t.level AS level"
        ).to_list()
        assert sub == [{"step": "Pick the datum.", "sub": "Metres below MSL.", "level": 1}]
        procedure = g.cypher(
            "MATCH (p:Procedure) RETURN p.concept_id AS id, p.title AS title, p.step_count AS steps"
        ).to_list()
        assert procedure == [{"id": "constructs#Constructs#Steps~list1", "title": "Steps", "steps": 3}]

    def test_edge_defaults_reach_every_edge_of_their_type(self):
        """VAULT.md §7.2: a constant stated once, never authored in a note."""
        g = self.build()
        assert g.cypher("MATCH ()-[r:HAS_STEP]->() RETURN DISTINCT r.derivation AS d").to_list() == [
            {"d": "source_order"}
        ]
        # A derived edge and a prose one take their own type's default.
        rows = g.cypher(
            "MATCH ()-[r:LINKS_TO]->() RETURN r.derivation AS d, r.anchor AS anchor ORDER BY anchor"
        ).to_list()
        assert [r["d"] for r in rows] == ["prose_reference"] * 5
        # A type with no entry carries nothing extra.
        assert g.cypher("MATCH ()-[r:HAS_SECTION]->() RETURN DISTINCT r.derivation AS d").to_list() == [{"d": None}]

    def test_a_table_row_is_a_node_keyed_by_its_key_column(self):
        """VAULT.md §7.1 ``tables:`` — one node per body row, columns as
        properties, ``types:`` deciding how a cell is built."""
        g = self.build()
        rows = g.cypher(
            "MATCH (n:ApiParameter) RETURN n.concept_id AS id, n.name AS name, "
            "n.type AS type, n.required AS required, n.section_id AS section, "
            "n.corpus AS corpus ORDER BY id"
        ).to_list()
        section = "tables#Tables#rmsapi.Project.open(path) → Project#Parameters"
        assert rows == [
            {
                "id": f"{section}~mode",
                "name": "mode",
                "type": "string",
                "required": False,
                "section": section,
                "corpus": "golden",  # `inherit:` reaches a row like any node
            },
            {
                "id": f"{section}~path",
                "name": "path",
                "type": "string",
                "required": True,
                "section": section,
                "corpus": "golden",
            },
            {
                "id": f"{section}~readonly",
                "name": "readonly",
                "type": "bool",
                # An empty cell writes no property at all (VAULT.md §7.1).
                "required": None,
                "section": section,
                "corpus": "golden",
            },
        ]
        assert (
            g.cypher(
                f"MATCH (:Section {{concept_id:'{section}'}})-[:HAS_PARAMETER]->(n) RETURN count(n) AS c"
            ).to_list()[0]["c"]
            == 3
        )

    def test_an_edge_table_row_is_an_edge_carrying_its_other_columns(self):
        """VAULT.md §7.1 ``edges: true`` — the row states an edge from the
        note, never a node, and an unwritten target is the usual stub."""
        g = self.build()
        rows = g.cypher(
            "MATCH (:Api {concept_id:'tables'})-[r:WORKED_ON_BY]->(t) "
            "RETURN t.concept_id AS target, t._provisional AS stub, r.role AS role, "
            "r.since AS since, r.row AS row, r.section AS section, r.label AS label "
            "ORDER BY row"
        ).to_list()
        assert rows == [
            {
                "target": "chunky",
                "stub": None,
                "role": "author",
                "since": "2024",
                "row": 1,
                "section": "Worked on by",
                # `[[chunky\|The chunky note]]`: the escaped pipe is the
                # separator, so the target is the note and the display text is
                # the edge's label (VAULT.md §5.1, §5.4).
                "label": "The chunky note",
            },
            {
                "target": "nobody",
                "stub": True,
                "role": "reviewer",
                "since": "2025",
                "row": 2,
                "section": "Worked on by",
                "label": None,
            },
        ]

    def test_a_symbol_heading_is_relabelled_and_split(self):
        """VAULT.md §7.1 ``key_from_heading:`` — the section is renamed in
        place; its id, its properties and its section edges are unchanged."""
        g = self.build()
        rows = g.cypher(
            "MATCH (n:ApiSymbol) RETURN n.concept_id AS id, n.title AS title, "
            "n.qualified_name AS name, n.signature AS signature, n.level AS level"
        ).to_list()
        assert rows == [
            {
                "id": "tables#Tables#rmsapi.Project.open(path) → Project",
                "title": "rmsapi.Project.open(path) → Project",
                "name": "rmsapi.Project.open",
                "signature": "(path) → Project",
                "level": 2,
            }
        ]
        # Only the notes carrying `under_label: Api` are read that way: the
        # Articles' sections are untouched.
        assert g.cypher("MATCH (n:Section) WHERE n.note_id = 'tables' RETURN count(n) AS c").to_list()[0]["c"] == 3

    def test_the_first_golden_vault_is_untouched_by_the_feature(self):
        """The compatibility promise: a vault that declares no ``structure:``
        builds exactly what it built before (VAULT.md §7.1)."""
        g = okf.build(str(VAULT_BUNDLE), dialect="obsidian")
        assert "Section" not in _labels(g)
        assert "HAS_SECTION" not in _edge_types(g)


class TestValidate:
    """``okf.validate`` — the build report as a value (VAULT.md §9)."""

    def test_the_golden_vault_reports_its_deliberate_faults(self):
        report = okf.validate(str(VAULT_BUNDLE), dialect="obsidian")
        assert report.errors == [
            "id collision: 2 notes resolve to id `alpha` "
            "(notes/alpha.md, projects/alpha.md); each falls back to its path-relative id"
        ]
        assert report.warnings == [
            "case-insensitive id collision: `Roadmap` (projects/Roadmap.md), `roadmap` (notes/roadmap.md)",
            "missing attachment: `img/appendix.pdf`",
            "dangling link: `Missing`",
        ]
        assert report.ok is False

    def test_counts_describe_the_same_build(self):
        report = okf.validate(str(VAULT_BUNDLE), dialect="obsidian")
        graph = okf.build(str(VAULT_BUNDLE), dialect="obsidian")
        counts = report.counts
        assert counts["files_scanned"] == 13
        assert counts["concepts"] == 13
        # Every node the build made is here, except the carried skill and
        # recipe nodes — those are counted by `skills_imported` /
        # `recipes_imported` instead of as notes.
        assert counts["nodes_by_label"] == {
            label: n for label, n in _labels(graph).items() if label not in ("KgliteSkill", "KgliteRecipe")
        }
        assert counts["edges_by_type"] == dict(_edge_types(graph))
        assert counts["dangling"] == 1
        assert counts["folder_notes"] == 1
        assert counts["missing_attachments"] == 1
        assert counts["ambiguous_attachments"] == 0
        assert counts["indexes_declared"] == 2
        assert counts["text_indexes_built"] == 1
        assert counts["skills_imported"] == 1
        assert counts["recipes_imported"] == 1
        # Reported, never computed: core links no embedder.
        assert counts["embed_targets"] == [("Article", "body")]

    def test_strict_promotes_warnings_only(self, tmp_path):
        (tmp_path / "a.md").write_text("[[Nowhere]]\n", encoding="utf-8")
        lenient = okf.validate(str(tmp_path), dialect="obsidian")
        strict = okf.validate(str(tmp_path), dialect="obsidian", strict=True)
        assert lenient.warnings == strict.warnings == ["dangling link: `Nowhere`"]
        assert lenient.errors == strict.errors == []
        assert lenient.ok is True
        assert strict.ok is False

    def test_str_renders_the_counts_header_then_the_findings(self, tmp_path):
        (tmp_path / "a.md").write_text("[[Nowhere]]\n", encoding="utf-8")
        text = str(okf.validate(str(tmp_path), dialect="obsidian"))
        assert text.startswith("files scanned: 1\nconcepts: 1\n")
        assert "errors: none\n" in text
        assert text.endswith("warnings (1):\n  - dangling link: `Nowhere`\n")

    def test_a_broken_vault_yaml_is_one_error_not_an_exception(self, tmp_path):
        (tmp_path / "a.md").write_text("prose\n", encoding="utf-8")
        (tmp_path / ".kglite").mkdir()
        (tmp_path / ".kglite" / "vault.yaml").write_text("kglite_vault: 1\nnonsense: 3\n", encoding="utf-8")
        # The build refuses it outright (VAULT.md §7).
        with pytest.raises(RuntimeError):
            okf.build(str(tmp_path), dialect="obsidian")
        report = okf.validate(str(tmp_path), dialect="obsidian")
        assert len(report.errors) == 1
        assert "vault.yaml" in report.errors[0] and "nonsense" in report.errors[0]
        assert report.ok is False

    def test_an_unreadable_root_still_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="does not exist"):
            okf.validate(str(tmp_path / "nope"), dialect="obsidian")

    def test_path_safety_errors_name_the_note_and_the_target(self, tmp_path):
        (tmp_path / "notes").mkdir()
        (tmp_path / "notes" / "a.md").write_text(
            "[plan](C:/secrets/plan.md) and [out](../../elsewhere/x.md)\n", encoding="utf-8"
        )
        report = okf.validate(str(tmp_path), dialect="obsidian")
        assert report.errors == [
            "notes/a.md: `C:/secrets/plan.md` is an absolute filesystem path",
            "notes/a.md: `../../elsewhere/x.md` escapes the vault root",
        ]

    def test_a_percent_encoded_reference_reaches_the_file_it_names(self, tmp_path):
        (tmp_path / "img").mkdir()
        (tmp_path / "img" / "a b.png").write_bytes(b"PNG")
        (tmp_path / "note.md").write_text("![chart](img/a%20b.png)\n", encoding="utf-8")
        report = okf.validate(str(tmp_path), dialect="obsidian")
        assert report.counts["missing_attachments"] == 0
        assert report.counts["nodes_by_label"]["Image"] == 1
        assert report.ok is True


class TestExport:
    """``okf.export`` — the vault written back out (VAULT.md §10)."""

    def _export(self, tmp_path, **kwargs):
        graph = okf.build(str(VAULT_BUNDLE), dialect="obsidian")
        out = tmp_path / "vault"
        return graph, okf.export(graph, str(out), **kwargs), out

    def test_the_golden_vault_writes_files_a_manifest_and_its_kglite_dir(self, tmp_path):
        _graph, report, out = self._export(tmp_path, source_root=str(VAULT_BUNDLE))
        assert report.ok is True
        assert report.files_refused == 0
        assert report.refusals == []
        assert report.files_written > 0
        assert (report.skills_written, report.recipes_written) == (1, 1)
        assert (report.attachments_copied, report.attachments_unresolved) == (3, 0)

        manifest = json.loads((out / ".kglite" / "export-manifest.json").read_text(encoding="utf-8"))
        assert manifest["kglite_vault"] == 1
        written = {
            str(p.relative_to(out)).replace("\\", "/")
            for p in out.rglob("*")
            if p.is_file() and p.name != "export-manifest.json"
        }
        assert set(manifest["files"]) == written

        assert (out / ".kglite" / "skills" / "vault_overview.md").is_file()
        assert (out / ".kglite" / "recipes" / "vault.by_keyword.md").is_file()
        assert (out / "img" / "faults.png").is_file()

    def test_str_renders_the_counts_then_the_refusals(self, tmp_path):
        _graph, report, _out = self._export(tmp_path)
        text = str(report)
        assert text.startswith(f"files written: {report.files_written}\n")
        assert "refusals: none" in text
        assert repr(report).startswith("<ExportReport written=")

    def test_the_export_reimports_to_the_same_note_labels(self, tmp_path):
        graph, _report, out = self._export(tmp_path, source_root=str(VAULT_BUNDLE))
        back = okf.build(str(out), dialect="obsidian")
        query = "MATCH (n) RETURN labels(n)[0] AS k, count(*) AS c ORDER BY k"
        before = {r["k"]: r["c"] for r in graph.cypher(query).to_list()}
        after = {r["k"]: r["c"] for r in back.cypher(query).to_list()}
        for label in ("Article", "Initiative"):
            assert after[label] == before[label], label
        # `Keyword` needed the `hubs:` declaration, which lives in
        # `.kglite/vault.yaml` — not in the graph, so not in the export (§10.9).
        assert "Keyword" not in after

    def test_a_second_export_reports_every_file_unchanged(self, tmp_path):
        graph, first, out = self._export(tmp_path, source_root=str(VAULT_BUNDLE))
        again = okf.export(graph, str(out), source_root=str(VAULT_BUNDLE))
        assert again.files_written == 0
        assert again.files_deleted == 0
        assert again.files_unchanged == first.files_written

    def test_a_hand_edited_file_is_refused_until_force(self, tmp_path):
        graph, _first, out = self._export(tmp_path)
        edited = out / "Article" / "welcome.md"
        original = edited.read_text(encoding="utf-8")
        edited.write_text("a human rewrote this\n", encoding="utf-8")

        refused = okf.export(graph, str(out))
        assert refused.ok is False
        assert refused.files_refused == 1
        assert refused.refusals == ["Article/welcome.md: edited since the last export (use force to replace)"]
        assert edited.read_text(encoding="utf-8") == "a human rewrote this\n"

        forced = okf.export(graph, str(out), force=True)
        assert forced.ok is True
        assert forced.files_written == 1
        assert edited.read_text(encoding="utf-8") == original

    def test_a_declared_edge_table_writes_its_properties_and_reimports_equal(self, tmp_path):
        """VAULT.md §7.3/§10.6. The structure vault declares `WORKED_ON_BY`, so
        the export writes the table back instead of a `worked_on_by:` list that
        would drop `role` and `since`. Re-importing it needs the vault's own
        `vault.yaml`, which no export writes (§10.9 loss 4) — so the test copies
        it, exactly as the spec asks an author to."""
        graph = okf.build(str(STRUCTURE_BUNDLE), dialect="obsidian")
        out = tmp_path / "vault"
        report = okf.export(graph, str(out), source_root=str(STRUCTURE_BUNDLE))
        assert report.warnings == []

        body = (out / "Api" / "tables.md").read_text(encoding="utf-8")
        assert "## Worked on by\n\n| person | role | since |\n| --- | --- | --- |\n" in body
        assert "| [[chunky\\|The chunky note]] | author | 2024 |" in body
        assert "| [[nobody]] | reviewer | 2025 |" in body
        assert "worked_on_by:" not in body

        shutil.copy2(STRUCTURE_BUNDLE / ".kglite" / "vault.yaml", out / ".kglite" / "vault.yaml")
        back = okf.build(str(out), dialect="obsidian")
        query = (
            "MATCH ()-[r:WORKED_ON_BY]->(t) "
            "RETURN r.row AS row, r.section AS section, r.label AS label, "
            "r.role AS role, r.since AS since ORDER BY r.row"
        )
        assert back.cypher(query).to_list() == graph.cypher(query).to_list()

    def test_the_caller_can_declare_an_edge_table_a_vault_does_not(self, tmp_path):
        """`edge_tables=` is how a graph that never was a vault declares one —
        and it wins per type over the file."""
        graph = okf.build(str(STRUCTURE_BUNDLE), dialect="obsidian")
        out = tmp_path / "vault"
        report = okf.export(
            graph,
            str(out),
            source_root=str(STRUCTURE_BUNDLE),
            edge_tables={"WORKED_ON_BY": "Contributors"},
        )
        body = (out / "Api" / "tables.md").read_text(encoding="utf-8")
        assert "## Contributors\n\n| target | role | since |\n" in body
        assert any("no `structure.tables:` rule" in w for w in report.warnings), report.warnings

    def test_a_missing_target_directory_is_created(self, tmp_path):
        graph = okf.build(str(VAULT_BUNDLE), dialect="obsidian")
        out = tmp_path / "deep" / "nested" / "vault"
        okf.export(graph, str(out))
        assert out.is_dir()

    def test_a_target_that_is_not_a_directory_raises(self, tmp_path):
        graph = okf.build(str(VAULT_BUNDLE), dialect="obsidian")
        blocker = tmp_path / "file.txt"
        blocker.write_text("not a directory\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="not a directory"):
            okf.export(graph, str(blocker))


class TestProvenanceAndRebuild:
    """``VAULT.md`` §12: a built graph remembers the directory it came from, and
    can be asked whether that directory has moved on."""

    @staticmethod
    def _vault(tmp_path) -> Path:
        vault = tmp_path / "vault"
        (vault / "notes").mkdir(parents=True)
        (vault / "notes" / "alpha.md").write_text("---\nid: alpha\n---\nAlpha.\n", encoding="utf-8")
        (vault / "notes" / "beta.md").write_text("---\nid: beta\n---\nSee [[alpha]].\n", encoding="utf-8")
        return vault

    def test_fingerprint_is_an_int_and_is_stable(self, tmp_path):
        vault = self._vault(tmp_path)
        first = okf.fingerprint(str(vault), dialect="obsidian")
        assert isinstance(first, int)
        assert first == okf.fingerprint(str(vault), dialect="obsidian")

        (vault / "notes" / "gamma.md").write_text("New.\n", encoding="utf-8")
        assert okf.fingerprint(str(vault), dialect="obsidian") != first

    def test_a_built_graph_carries_its_root_and_fingerprint(self, tmp_path):
        vault = self._vault(tmp_path)
        g = okf.build(str(vault), dialect="obsidian")
        assert g.source_root == str(vault.resolve())
        assert g.source_fingerprint == okf.fingerprint(str(vault), dialect="obsidian")

        # A graph that was not built from a directory has no answer.
        assert kglite.KnowledgeGraph().source_root is None
        assert kglite.KnowledgeGraph().source_fingerprint is None

    def test_provenance_survives_a_save_and_load(self, tmp_path):
        vault = self._vault(tmp_path)
        g = okf.build(str(vault), dialect="obsidian")
        path = str(tmp_path / "vault.kgl")
        g.save(path)
        loaded = kglite.load(path)
        assert loaded.source_root == g.source_root
        assert loaded.source_fingerprint == g.source_fingerprint

    def test_rebuild_is_none_until_the_vault_changes(self, tmp_path):
        vault = self._vault(tmp_path)
        g = okf.build(str(vault), dialect="obsidian")
        assert okf.rebuild_if_changed(g, dialect="obsidian") is None

        (vault / "notes" / "gamma.md").write_text("---\nid: gamma\n---\nNew.\n", encoding="utf-8")
        fresh = okf.rebuild_if_changed(g, dialect="obsidian")
        assert fresh is not None
        # No `type:` and no `default_label`, so the top-level folder labels them.
        assert _labels(fresh)["notes"] == 3
        # The graph passed in is never modified — the rebuild is a new object.
        assert _labels(g)["notes"] == 2
        assert okf.rebuild_if_changed(fresh, dialect="obsidian") is None

    def test_open_builds_once_then_loads_its_own_cache(self, tmp_path):
        vault = self._vault(tmp_path)
        cache = vault / ".kglite" / "graph.kgl"

        g = okf.open(str(vault), dialect="obsidian")
        assert _labels(g)["notes"] == 2
        assert cache.is_file(), "the first open left the vault its graph"

        # The load is not observable from the returned object, so the
        # observable is the one that matters: the cache the save wrote did not
        # invalidate itself, so `rebuild_if_changed` on the loaded graph is
        # still `None`.
        again = okf.open(str(vault), dialect="obsidian")
        assert _labels(again)["notes"] == 2
        assert okf.rebuild_if_changed(again) is None

        (vault / "notes" / "gamma.md").write_text("---\nid: gamma\n---\nNew.\n", encoding="utf-8")
        assert _labels(okf.open(str(vault), dialect="obsidian"))["notes"] == 3
        assert okf.rebuild_if_changed(okf.open(str(vault), dialect="obsidian")) is None

    def test_open_takes_a_cache_path_and_switches_off_on_false(self, tmp_path):
        vault = self._vault(tmp_path)
        elsewhere = tmp_path / "cache" / "vault.kgl"

        okf.open(str(vault), dialect="obsidian", cache=str(elsewhere))
        assert elsewhere.is_file()
        assert not (vault / ".kglite" / "graph.kgl").exists()

        g = okf.open(str(vault), dialect="obsidian", cache=False)
        assert _labels(g)["notes"] == 2
        assert not (vault / ".kglite" / "graph.kgl").exists()

    def test_open_is_never_failed_by_its_cache(self, tmp_path):
        """Every way the cache can be wrong is a miss, never an exception."""
        vault = self._vault(tmp_path)
        (vault / ".kglite").mkdir(exist_ok=True)
        (vault / ".kglite" / "graph.kgl").write_bytes(b"not a kgl file at all")
        assert _labels(okf.open(str(vault), dialect="obsidian"))["notes"] == 2

        # A cache that cannot be written is the same: the graph still comes
        # back. (A plain file where the cache's directory should be fails for
        # every user, root included.)
        (tmp_path / "in-the-way").write_text("a file", encoding="utf-8")
        g = okf.open(str(vault), dialect="obsidian", cache=str(tmp_path / "in-the-way" / "g.kgl"))
        assert _labels(g)["notes"] == 2

    def test_open_defaults_to_okf_like_build_does(self, tmp_path):
        """A path carries no stamp, so `open` keeps `build`'s default rather
        than guessing that a directory is a vault."""
        vault = self._vault(tmp_path)
        (vault / ".kglite").mkdir(exist_ok=True)
        (vault / ".kglite" / "vault.yaml").write_text("kglite_vault: 1\ndefault_label: Note\n", encoding="utf-8")
        assert "Note" not in _labels(okf.open(str(vault)))
        assert _labels(okf.open(str(vault), dialect="obsidian"))["Note"] == 2

    def test_rebuild_refuses_a_graph_with_no_provenance(self):
        with pytest.raises(RuntimeError, match="source_root"):
            okf.rebuild_if_changed(kglite.KnowledgeGraph(), dialect="obsidian")

    def test_a_rebuild_embeds_only_the_note_that_changed(self, tmp_path):
        """The declared ``embed:`` target runs in changed mode over the carried
        hashes, through whatever model the graph already has bound."""
        vault = self._vault(tmp_path)
        (vault / ".kglite").mkdir()
        (vault / ".kglite" / "vault.yaml").write_text(
            "kglite_vault: 1\ndefault_label: Note\nembed:\n  Note: body\n", encoding="utf-8"
        )

        seen: list[str] = []

        class Model:
            dimension = 2

            def embed(self, texts):
                seen.extend(texts)
                return [[1.0, 0.0] for _ in texts]

        g = okf.build(str(vault), dialect="obsidian")
        g.set_embedder(Model())
        g.embed_texts("Note", "body", show_progress=False)
        assert len(seen) == 2
        seen.clear()

        (vault / "notes" / "beta.md").write_text("---\nid: beta\n---\nRewritten.\n", encoding="utf-8")
        fresh = okf.rebuild_if_changed(g, dialect="obsidian")
        assert fresh is not None
        assert seen == ["Rewritten."], "only the rewritten note reached the model"
        assert fresh.embedding_dim("Note", "body") == 2
        # The rebuilt graph keeps the model, so the next round needs no
        # re-registration — `embed_texts` raises without one.
        seen.clear()
        (vault / "notes" / "alpha.md").write_text("---\nid: alpha\n---\nAlso rewritten.\n", encoding="utf-8")
        again = okf.rebuild_if_changed(fresh, dialect="obsidian")
        assert again is not None
        assert seen == ["Also rewritten."]

    def test_export_defaults_its_source_root_to_the_graphs_own(self, tmp_path):
        """A vault-built graph knows where its pictures are (§12), so an export
        that is told nothing still copies them."""
        vault = self._vault(tmp_path)
        (vault / "img").mkdir()
        (vault / "img" / "x.png").write_bytes(b"\x89PNG\r\n")
        (vault / "notes" / "alpha.md").write_text("---\nid: alpha\n---\n![map](../img/x.png)\n", encoding="utf-8")
        g = okf.build(str(vault), dialect="obsidian")

        out = tmp_path / "out"
        report = okf.export(g, str(out))
        assert (report.attachments_copied, report.attachments_unresolved) == (1, 0)
        assert (out / "img" / "x.png").read_bytes() == b"\x89PNG\r\n"

    def test_rebuild_reads_the_dialect_off_the_stamp(self, tmp_path):
        """The operator's repro (2026-09-19): a vault built as ``obsidian``,
        saved, reopened, and rebuilt by a caller who named no dialect was
        rebuilt as an OKF bundle — a near-empty graph that loads and looks
        valid. The stamp answers the question the caller left open."""
        vault = self._vault(tmp_path)
        (vault / ".kglite").mkdir()
        (vault / ".kglite" / "vault.yaml").write_text("kglite_vault: 1\ndefault_label: Note\n", encoding="utf-8")
        built = okf.build(str(vault), dialect="obsidian")
        assert built.source_dialect == "obsidian"
        path = str(tmp_path / "vault.kgl")
        built.save(path)
        g = kglite.load(path)
        assert g.source_dialect == "obsidian"

        assert okf.rebuild_if_changed(g) is None, "nothing moved, whoever asked"

    def test_a_rebuild_dialect_that_contradicts_the_stamp_is_refused(self, tmp_path):
        vault = self._vault(tmp_path)
        built = okf.build(str(vault), dialect="obsidian")
        with pytest.raises(RuntimeError, match="obsidian") as caught:
            okf.rebuild_if_changed(built, dialect="okf")
        assert "okf" in str(caught.value), "both dialects are named"


def test_typed_inline_link_names_the_edge(tmp_path):
    """VAULT.md §5.3 rung 0: `[[Target]]{type}` types that one link, above the
    heading it sits under, and a brace that names no type is a warning."""
    (tmp_path / ".kglite").mkdir()
    (tmp_path / ".kglite" / "vault.yaml").write_text("kglite_vault: 1\nstructure:\n  sections: {}\n", encoding="utf-8")
    (tmp_path / "guide.md").write_text(
        "## Related work\n\nRead [[atlas|the atlas]]{see-also} and [[atlas]]{3d}.\n",
        encoding="utf-8",
    )
    (tmp_path / "atlas.md").write_text("The atlas.\n", encoding="utf-8")

    g = okf.build(str(tmp_path), dialect="obsidian")
    rows = g.cypher(
        "MATCH (:Note {concept_id:'guide'})-[r]->(:Note {concept_id:'atlas'}) "
        "RETURN type(r) AS t, r.label AS label ORDER BY t"
    ).to_list()
    assert [(r["t"], r["label"]) for r in rows] == [
        ("RELATED", None),
        ("SEE_ALSO", "the atlas"),
    ], "the suffix beats the heading; `{3d}` starts with a digit, so that link keeps RELATED"

    report = okf.validate(str(tmp_path), dialect="obsidian")
    assert [w for w in report.warnings if "{3d}" in w] == [
        "guide.md: `[[atlas]]{3d}`: a link type holds no whitespace and must normalise to a "
        "name that does not start with a digit — the brace is left as prose"
    ]


def test_directive_states_a_section_property_and_a_typed_edge(tmp_path):
    """VAULT.md §5.8, the RMS help shape: a GUI path written beside the
    sentence it describes is a Section property, queryable by Cypher and
    absent from the section's text and from every chunk's text and
    ``embed_text``; a wikilink value is a typed edge leaving that Section.
    """
    (tmp_path / ".kglite").mkdir()
    (tmp_path / ".kglite" / "vault.yaml").write_text(
        "kglite_vault: 1\ndefault_label: Article\nstructure:\n"
        "  sections: {label: Section, edge: HAS_SECTION}\n"
        "  chunks: {label: Chunk, edge: HAS_CHUNK, next: NEXT_CHUNK, "
        "max_words: 650, max_chars: 6000}\n"
        '  embed_text: "{title} | {section_title}\\n\\n{text}"\n',
        encoding="utf-8",
    )
    (tmp_path / "annotations.md").write_text(
        "### Annotation Table\n\n"
        "To open the **Annotation Table** dialog box, click the button.\n\n"
        "<!-- kglite address: Data tree -> Wells | Task pane: Wells -> Annotations table -->\n"
        "<!-- kglite documented_in: [[Wells]] -->\n\n"
        "Then pick a well.\n",
        encoding="utf-8",
    )
    (tmp_path / "Wells.md").write_text("The wells guide.\n", encoding="utf-8")

    g = okf.build(str(tmp_path), dialect="obsidian")
    rows = g.cypher("MATCH (s:Section) RETURN s.concept_id AS id, s.address AS address, s.text AS text").to_list()
    assert rows == [
        {
            "id": "annotations#Annotation Table",
            "address": "Data tree -> Wells | Task pane: Wells -> Annotations table",
            "text": "\nTo open the **Annotation Table** dialog box, click the button.\n\n\nThen pick a well.",
        }
    ]
    assert g.cypher("MATCH (:Section)-[r:DOCUMENTED_IN]->(t) RETURN t.concept_id AS target").to_list() == [
        {"target": "Wells"}
    ]

    chunks = g.cypher("MATCH (c:Chunk) RETURN c.text AS t, c.embed_text AS e").to_list()
    assert chunks, "the section packs at least one chunk"
    for row in chunks:
        assert "kglite" not in row["t"], row["t"]
        assert "kglite" not in row["e"], row["e"]

    body = g.cypher("MATCH (n:Article {concept_id:'annotations'}) RETURN n.body AS b").to_list()[0]["b"]
    assert "<!-- kglite address:" in body, "the note's own prose is verbatim"


def test_tag_labels_model_a_tag_as_its_own_node(tmp_path):
    """VAULT.md §5.5: a tag under a declared prefix becomes a node of its own,
    joined from the innermost derived node that holds it, and every inline tag
    lands in a ``tags`` list on that same node — so a paragraph-scoped marker
    is selectable per chunk instead of per note.
    """
    (tmp_path / ".kglite").mkdir()
    (tmp_path / ".kglite" / "vault.yaml").write_text(
        "kglite_vault: 1\ndefault_label: Article\n"
        'tag_labels:\n  "intent/*": {label: Intent, edge: HAS_INTENT}\n'
        "structure:\n"
        "  sections: {label: Section, edge: HAS_SECTION}\n"
        "  chunks: {label: Chunk, edge: HAS_CHUNK, next: NEXT_CHUNK, max_words: 10, max_chars: 200}\n",
        encoding="utf-8",
    )
    (tmp_path / "wells.md").write_text(
        "## Importing wells\n\n"
        "Use the import dialog. #intent/import-wells\n\n"
        "The datum is not checked on import. #warning\n",
        encoding="utf-8",
    )

    g = okf.build(str(tmp_path), dialect="obsidian")
    assert g.cypher(
        "MATCH (c:Chunk)-[:HAS_INTENT]->(i:Intent {id:'import-wells'}) RETURN c.concept_id AS chunk, i.title AS title"
    ).to_list() == [{"chunk": "wells#Importing wells~chunk1", "title": "import-wells"}]

    assert g.cypher("MATCH (c:Chunk) WHERE 'warning' IN c.tags RETURN c.concept_id AS id").to_list() == [
        {"id": "wells#Importing wells~chunk2"}
    ]

    assert g.cypher("MATCH (t:Tag) RETURN t.id AS id ORDER BY id").to_list() == [{"id": "warning"}], (
        "the modelled tag left the Tag hub; the plain one did not"
    )


def test_heading_directive_promotes_a_bold_line_to_a_heading(tmp_path):
    """VAULT.md §5.8: ``<!-- kglite heading -->`` promotes the line below it to
    a heading at the enclosing level + 1, so a converted API page's bold
    signature lines become addressable sections without touching the file.
    """
    (tmp_path / ".kglite").mkdir()
    (tmp_path / ".kglite" / "vault.yaml").write_text(
        "kglite_vault: 1\ndefault_label: Article\nstructure:\n"
        "  sections: {label: Section, edge: HAS_SECTION, parent: PARENT_SECTION, next: NEXT_SECTION}\n",
        encoding="utf-8",
    )
    body = (
        "### rmsapi.Project\n\nProject access.\n\n"
        "<!-- kglite heading -->\n"
        "**open(filename)**\n\nOpens a project.\n\n"
        "<!-- kglite heading -->\n"
        "**close()**\n\nCloses it.\n"
    )
    (tmp_path / "project.md").write_text(body, encoding="utf-8")
    (tmp_path / "guide.md").write_text("Start at [[project#rmsapi.Project#open(filename)]].\n", encoding="utf-8")

    g = okf.build(str(tmp_path), dialect="obsidian")
    assert g.cypher("MATCH (s:Section) RETURN s.concept_id AS id, s.level AS level ORDER BY id").to_list() == [
        {"id": "project#rmsapi.Project", "level": 3},
        {"id": "project#rmsapi.Project#close()", "level": 4},
        {"id": "project#rmsapi.Project#open(filename)", "level": 4},
    ], "two markers under one heading are siblings, not a chain"

    assert g.cypher(
        "MATCH (:Article {concept_id:'guide'})-[:LINKS_TO]->(s:Section) RETURN s.concept_id AS id"
    ).to_list() == [{"id": "project#rmsapi.Project#open(filename)"}]

    assert g.cypher("MATCH (n:Article {concept_id:'project'}) RETURN n.body AS b").to_list()[0]["b"] == body, (
        "the file is unchanged; only the tree the reader builds from it is"
    )
