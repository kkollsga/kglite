"""Inheritance-aware CALLS resolution: a `self.method()` call whose method
is defined on an *ancestor* (not the caller's own class) resolves to the
inherited definition via EXTENDS/IMPLEMENTS — even when the same method
name exists on unrelated classes that the same-file / global fallbacks
would otherwise pick.
"""

import textwrap

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402


def _pairs(graph) -> set[tuple[str, str]]:
    rows = graph.cypher(
        "MATCH (a:Function)-[:CALLS]->(b:Function) "
        "RETURN a.qualified_name AS caller, b.qualified_name AS callee"
    ).to_list()
    return {(r["caller"], r["callee"]) for r in rows}


def test_self_call_resolves_to_inherited_base_method(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "a.py").write_text(
        textwrap.dedent(
            """
            class Base:
                def run(self):
                    return 1

            class Other:
                def run(self):
                    return 2

            class Derived(Base):
                def caller(self):
                    return self.run()
            """
        )
    )
    g = build(str(tmp_path))
    pairs = _pairs(g)
    # Derived.caller -> Base.run via inheritance.
    assert any(
        caller.endswith(".Derived.caller") and callee.endswith(".Base.run")
        for caller, callee in pairs
    ), f"self.run() should resolve to inherited Base.run: {pairs}"
    # Must NOT mis-resolve to the unrelated Other.run.
    assert not any(
        caller.endswith(".Derived.caller") and callee.endswith(".Other.run")
        for caller, callee in pairs
    ), f"self.run() must not resolve to unrelated Other.run: {pairs}"


def test_multi_level_inheritance(tmp_path):
    # Derived -> Mid -> Base; method defined two levels up.
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "a.py").write_text(
        textwrap.dedent(
            """
            class Base:
                def deep(self):
                    return 1

            class Unrelated:
                def deep(self):
                    return 9

            class Mid(Base):
                pass

            class Derived(Mid):
                def caller(self):
                    return self.deep()
            """
        )
    )
    g = build(str(tmp_path))
    pairs = _pairs(g)
    assert any(
        caller.endswith(".Derived.caller") and callee.endswith(".Base.deep")
        for caller, callee in pairs
    ), f"self.deep() should resolve through Mid to Base.deep: {pairs}"
