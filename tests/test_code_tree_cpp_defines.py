"""C/C++ ``#define`` object-like macro constants → :Constant nodes.

The ``#define`` → :Constant pipeline (``parse_preproc_def`` → ``ConstantInfo``
→ :Constant node with ``value_preview`` + ``line_number``) exists end to end,
but was dead for the common case: the shared ``looks_like_macro_decorator``
filter (which legitimately protects the *function-name* slot from export
macros like ``KUZU_API``) dropped every SCREAMING_SNAKE_CASE ``#define`` name,
and ``#define``s guarded by ``#if`` / ``#ifdef`` were never reached because the
top-level walk only visited direct translation-unit children.

These tests assert both defects are fixed while the export-macro / function
extraction behaviour guarded by ``test_code_tree_cpp_macros.py`` is unchanged.
"""

import textwrap

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402


def _write(tmp_path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        fp = tmp_path / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(textwrap.dedent(content))


def _constants(graph) -> dict[str, dict]:
    """Map of constant name → {value, line} for every :Constant node."""
    rows = graph.cypher("MATCH (c:Constant) RETURN c.name AS n, c.value_preview AS v, c.line_number AS ln").to_list()
    return {r["n"]: {"value": r["v"], "line": r["ln"]} for r in rows}


def test_allcaps_and_mixedcase_defines_captured(tmp_path):
    """Both an ALL-CAPS and a MixedCase object-like ``#define`` land as
    :Constant nodes with the right ``value_preview`` and line number."""
    _write(
        tmp_path,
        {
            "config.h": """
            #define ALLCAPS_CONST 7
            #define MixedCase 42
            """,
        },
    )
    g = build(str(tmp_path))
    consts = _constants(g)
    assert "ALLCAPS_CONST" in consts, consts
    assert consts["ALLCAPS_CONST"]["value"] == "7", consts["ALLCAPS_CONST"]
    assert consts["ALLCAPS_CONST"]["line"] == 2, consts["ALLCAPS_CONST"]
    assert "MixedCase" in consts, consts
    assert consts["MixedCase"]["value"] == "42", consts["MixedCase"]
    assert consts["MixedCase"]["line"] == 3, consts["MixedCase"]


def test_guarded_define_inside_if_block_captured(tmp_path):
    """A ``#define`` inside ``#if defined(__APPLE__)`` is captured (the loop
    now recurses into preprocessor conditionals)."""
    _write(
        tmp_path,
        {
            "prim.h": """
            #if defined(__APPLE__)
            #define GUARDED_SLOT 108
            #endif
            """,
        },
    )
    g = build(str(tmp_path))
    consts = _constants(g)
    assert "GUARDED_SLOT" in consts, consts
    assert consts["GUARDED_SLOT"]["value"] == "108", consts["GUARDED_SLOT"]


def test_export_macro_function_behaviour_unchanged(tmp_path):
    """The macro-decorator filter still protects the function-name slot: an
    export-macro-decorated function keeps its real name and return type, and no
    ``unknown`` / macro-named function is produced."""
    _write(
        tmp_path,
        {
            "lib.cpp": """
            #define SPDLOG_INLINE inline

            SPDLOG_INLINE void log_msg() {}
            """,
        },
    )
    g = build(str(tmp_path))
    rows = g.cypher(
        "MATCH (f:Function) WHERE f.qualified_name ENDS WITH '::log_msg' "
        "RETURN f.name AS n, f.return_type AS rt LIMIT 1"
    ).to_list()
    assert rows and rows[0]["n"] == "log_msg", rows
    assert rows[0]["rt"] not in ("SPDLOG_INLINE", None), rows[0]
    # No function is named after the export macro or degenerate 'unknown'.
    names = {r["n"] for r in g.cypher("MATCH (f:Function) RETURN f.name AS n").to_list()}
    assert "SPDLOG_INLINE" not in names, names
    assert "unknown" not in names, names


def test_mimalloc_prim_shape_query(tmp_path):
    """The motivating query: a header mimicking mimalloc's ``prim.h`` shape —
    an ALL-CAPS guarded ``#define`` alongside a top-level define and an
    export-macro-decorated function — and a cypher lookup of the Constant by
    name returns its value.

    NOTE: guard *context* is out of scope (plan C.2). Branch-guarded defines
    that share a macro name collapse to one :Constant node (they share a
    ``qualified_name``); here ``MI_TLS_MODEL`` is defined once, under a single
    ``#if`` guard, so the lookup is deterministic.
    """
    _write(
        tmp_path,
        {
            "src/main.cpp": "int main() { return 0; }\n",
            "src/prim.h": """
            #define MI_CACHE_LINE 64

            #if defined(__APPLE__)
            #define MI_TLS_MODEL 3
            #endif

            #define MI_API inline
            MI_API void mi_init() {}
            """,
        },
    )
    g = build(str(tmp_path))
    rows = g.cypher(
        "MATCH (c:Constant {name: 'MI_TLS_MODEL'}) RETURN c.value_preview AS v, c.line_number AS ln"
    ).to_list()
    assert rows, "MI_TLS_MODEL constant not found"
    assert rows[0]["v"] == "3", rows  # value from the __APPLE__ branch
    # The top-level unguarded define is also present.
    consts = _constants(g)
    assert consts.get("MI_CACHE_LINE", {}).get("value") == "64", consts
    # The export-macro function survived intact.
    fn = g.cypher("MATCH (f:Function) WHERE f.qualified_name ENDS WITH '::mi_init' RETURN f.name AS n").to_list()
    assert fn and fn[0]["n"] == "mi_init", fn
