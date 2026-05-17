"""CSS parser smoke tests.

Coverage in 0.9.36:
  - One Selector node per `rule_set`. Selector-list `.foo, .bar` emits
    ONE node named `.foo, .bar`, not three.
  - CSS custom properties (`--my-color: red`) → ConstantInfo with
    kind="css_custom_property".
  - `@import "./theme.css"` / `@import url("./reset.css")` populates
    FileInfo.imports.
  - @media-wrapped rules emit Selector nodes for their inner rule_sets
    (the @media wrapper is parse noise for v1).

Restraint: @media / @keyframes / @font-face don't emit structural
nodes for the wrapper itself.
"""

import pathlib

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402


def _write(tmp_path, name: str, content: str) -> pathlib.Path:
    pkg = tmp_path / "styles"
    pkg.mkdir(exist_ok=True)
    (pkg / name).write_text(content)
    return pkg


def test_basic_selectors_emit_one_per_rule_set(tmp_path):
    pkg = _write(
        tmp_path,
        "main.css",
        """.foo { color: red; }
#bar { padding: 1em; }
:root { font-family: sans-serif; }
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (s:Selector) RETURN s.name AS n ORDER BY s.line_number").to_list()
    names = [r["n"] for r in rows]
    assert names == [".foo", "#bar", ":root"], names


def test_selector_list_emits_one_node(tmp_path):
    """`.foo, .bar, .baz { ... }` emits ONE Selector named with the full list."""
    pkg = _write(
        tmp_path,
        "lists.css",
        """.foo, .bar, .baz { color: red; }
.button, .btn { padding: 0; }
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (s:Selector) RETURN s.name AS n ORDER BY s.line_number").to_list()
    names = [r["n"] for r in rows]
    assert names == [".foo, .bar, .baz", ".button, .btn"], names


def test_css_custom_properties_as_constants(tmp_path):
    pkg = _write(
        tmp_path,
        "tokens.css",
        """:root {
    --primary-color: #1abc9c;
    --spacing-md: 16px;
}
""",
    )
    g = build(str(pkg))
    rows = g.cypher(
        "MATCH (c:Constant {kind: 'css_custom_property'}) RETURN c.name AS n, c.value_preview AS v ORDER BY n"
    ).to_list()
    names = [r["n"] for r in rows]
    assert names == ["--primary-color", "--spacing-md"], rows
    # value_preview contains the full declaration text.
    assert "#1abc9c" in rows[0]["v"]


def test_media_block_emits_inner_selectors(tmp_path):
    """A `@media (...) { .nav { ... } }` block emits a Selector node
    for `.nav` (the @media wrapper is intentionally parse noise)."""
    pkg = _write(
        tmp_path,
        "responsive.css",
        """.always {
    display: block;
}

@media (max-width: 700px) {
    .nav {
        display: none;
    }
    .hero {
        font-size: 1rem;
    }
}
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (s:Selector) RETURN s.name AS n ORDER BY n").to_list()
    names = [r["n"] for r in rows]
    assert names == [".always", ".hero", ".nav"], names


def test_at_import_quoted_string(tmp_path):
    """`@import "./theme.css"` → FileInfo.imports → File→Module
    IMPORTS edge resolution. We can't directly inspect FileInfo from
    Python, but the existing 0.9.34 IMPORTS resolver wires it through
    when the target module path matches a parsed file."""
    pkg = _write(
        tmp_path,
        "main.css",
        """@import "./theme.css";

.button { color: red; }
""",
    )
    # Parse-only sibling — the relative-path resolution path doesn't
    # generate IMPORTS edges (pre-existing limitation, same as the
    # HTML parser's script-src behaviour), but the .css file should
    # still parse and land as a File node.
    (pkg / "theme.css").write_text(".dark { color: white; }")
    g = build(str(pkg))
    files = g.cypher("MATCH (f:File) RETURN f.path AS p, f.language AS l ORDER BY p").to_list()
    langs = {r["p"]: r["l"] for r in files}
    assert langs.get("main.css") == "css"
    assert langs.get("theme.css") == "css"


def test_at_import_url_form(tmp_path):
    """`@import url("./reset.css")` — the url() form parses as a
    call_expression containing the string_value."""
    pkg = _write(
        tmp_path,
        "main.css",
        """@import url("./reset.css");

#nav { padding: 1em; }
""",
    )
    (pkg / "reset.css").write_text("body { margin: 0; }")
    g = build(str(pkg))
    files = g.cypher("MATCH (f:File) RETURN f.path AS p ORDER BY p").to_list()
    paths = [r["p"] for r in files]
    assert "main.css" in paths
    assert "reset.css" in paths


def test_no_function_or_class_nodes_emitted(tmp_path):
    """Regression guard: future scope creep into 'rules as functions'
    shouldn't sneak in. CSS files emit Selector + (optionally) Constant
    nodes — never Function or Class."""
    pkg = _write(
        tmp_path,
        "styles.css",
        """.foo { color: red; padding: 10px; }
#bar { display: flex; }
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (f:Function) RETURN count(f) AS n").to_list()
    assert rows[0]["n"] == 0
    rows = g.cypher("MATCH (c:Class) RETURN count(c) AS n").to_list()
    assert rows[0]["n"] == 0


def test_language_tag_on_files(tmp_path):
    pkg = _write(tmp_path, "x.css", "/* empty */")
    g = build(str(pkg))
    rows = g.cypher("MATCH (f:File) RETURN f.language AS l").to_list()
    assert any(r["l"] == "css" for r in rows), rows
