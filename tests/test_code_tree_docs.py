"""`code_tree.build(..., include_docs=True)` — ingest a repo's markdown as
`:Doc` nodes and link them to the code they describe.

Builds a tiny mixed repo (a Rust source file + a README + a docs/ guide) and
asserts the docs pass:
  - adds `:Doc` nodes only when `include_docs=True` (default off),
  - classifies each doc's `kind` from its filename,
  - links `(:Doc)-[:MENTIONS]->(:symbol)` conservatively (unique symbols link,
    stop-words / absent names do not),
  - links `(:Doc)-[:DOCUMENTS]->(:Doc|:File)` from markdown links.
"""

import pathlib
import textwrap

from kglite.code_tree import build


def _build_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    repo = tmp_path / "proj"
    (repo / "src").mkdir(parents=True)
    (repo / "docs").mkdir()
    (repo / "src" / "lib.rs").write_text(
        textwrap.dedent(
            """
            pub fn parse_wkt() {}
            pub struct KnowledgeGraph;
            pub fn run() {}
            """
        )
    )
    (repo / "docs" / "design.md").write_text("# Design\nInternal notes.\n")
    (repo / "README.md").write_text(
        textwrap.dedent(
            """\
            # Demo Project

            Call `parse_wkt` then build a `KnowledgeGraph`. Do not `run` this,
            and the `nonexistent` symbol is absent.

            See the [design notes](docs/design.md) and the
            [entry point](src/lib.rs).
            """
        )
    )
    return repo


def test_include_docs_off_by_default(tmp_path):
    repo = _build_repo(tmp_path)
    g = build(str(repo))
    docs = g.cypher("MATCH (d:Doc) RETURN count(d) AS c").to_list()[0]["c"]
    assert docs == 0
    fns = g.cypher("MATCH (f:Function) RETURN count(f) AS c").to_list()[0]["c"]
    assert fns >= 1, "code is still parsed"


def test_include_docs_adds_doc_nodes_and_kinds(tmp_path):
    repo = _build_repo(tmp_path)
    g = build(str(repo), include_docs=True)
    kinds = {r["k"]: r["c"] for r in g.cypher("MATCH (d:Doc) RETURN d.kind AS k, count(*) AS c").to_list()}
    assert kinds.get("readme") == 1
    assert kinds.get("guide") == 1, "docs/ file classified as guide"


def test_mentions_are_conservative(tmp_path):
    repo = _build_repo(tmp_path)
    g = build(str(repo), include_docs=True)
    names = {r["sym"] for r in g.cypher("MATCH (:Doc)-[:MENTIONS]->(c) RETURN c.name AS sym").to_list()}
    assert "parse_wkt" in names, "unique function links"
    assert "KnowledgeGraph" in names, "unique struct links"
    assert "run" not in names, "stop-word must not link"
    assert "nonexistent" not in names, "absent symbol must not link"


def test_documents_links_doc_and_file(tmp_path):
    repo = _build_repo(tmp_path)
    g = build(str(repo), include_docs=True)
    targets = {
        (r["lbl"], r["name"])
        for r in g.cypher(
            "MATCH (:Doc {kind: 'readme'})-[:DOCUMENTS]->(t) "
            "RETURN labels(t)[0] AS lbl, coalesce(t.title, t.filename, t.name) AS name"
        ).to_list()
    }
    labels = {lbl for lbl, _ in targets}
    assert "Doc" in labels, "README -> docs/design.md (Doc)"
    assert "File" in labels, "README -> src/lib.rs (File)"
