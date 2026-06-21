"""`CALL dead_code(...)` — graph-native dead-code detection.

A Function is reported when nothing CALLS it, references it as a value
(REFERENCES_FN), HANDLES it (route), or IMPLEMENTED_BY (procedure), and it
isn't a DECORATES participant — minus the always-excluded entry points
(tests, dunder methods, `main`). Public functions are included by default
(every non-underscore Python name is nominally public); `exclude_public`
drops pub/exported visibility.
"""

import textwrap

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return repr(v)


def _dead(graph, **params) -> set[str]:
    arg = ""
    if params:
        inner = ", ".join(f"{k}: {_fmt(v)}" for k, v in params.items())
        arg = f"{{{inner}}}"
    rows = graph.cypher(f"CALL dead_code({arg}) YIELD node RETURN node.name AS name").to_list()
    return {r["name"] for r in rows}


def test_dead_code_basic(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "a.py").write_text(
        textwrap.dedent(
            """
            def used():
                return 1

            def caller():
                return used()

            def orphan():
                return 2

            def main():
                caller()

            def __getitem__(self):
                return 3
            """
        )
    )
    g = build(str(tmp_path))
    dead = _dead(g)
    # orphan is never referenced → dead. `used` is called → alive.
    assert "orphan" in dead
    assert "used" not in dead
    # main and dunder methods are implicit entry points → never reported.
    assert "main" not in dead
    assert "__getitem__" not in dead


def test_dead_code_excludes_decorated(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "a.py").write_text(
        textwrap.dedent(
            """
            def deco(fn):
                return fn

            @deco
            def decorated():
                return 1
            """
        )
    )
    g = build(str(tmp_path))
    dead = _dead(g)
    # `decorated` has an inbound DECORATES edge → framework-registered, not dead.
    assert "decorated" not in dead


def test_dead_code_excludes_fn_pointer_reference(tmp_path):
    # A function passed by reference (REFERENCES_FN) is used, not dead.
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "fixture"\nversion = "0.0.0"\nedition = "2021"\n')
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.rs").write_text(
        textwrap.dedent(
            """
            pub fn helper(n: u32) -> Option<u32> { Some(n + 1) }

            pub fn caller() -> Option<u32> {
                Some(0u32).and_then(helper)
            }
            """
        )
    )
    g = build(str(tmp_path))
    dead = _dead(g)
    # `helper` has no CALLS edge but is referenced as a value → not dead.
    assert "helper" not in dead


def test_dead_code_exclude_public_param(tmp_path):
    # Rust `pub` is a real public-API marker; exclude_public should drop it.
    (tmp_path / "Cargo.toml").write_text(
        textwrap.dedent(
            """
            [package]
            name = "fixture"
            version = "0.0.0"
            edition = "2021"
            """
        )
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.rs").write_text(
        textwrap.dedent(
            """
            pub fn public_unused() -> u32 { 1 }
            fn private_unused() -> u32 { 2 }
            """
        )
    )
    g = build(str(tmp_path))
    default_dead = _dead(g)
    assert "public_unused" in default_dead
    assert "private_unused" in default_dead

    pruned = _dead(g, exclude_public=True)
    assert "public_unused" not in pruned
    assert "private_unused" in pruned
