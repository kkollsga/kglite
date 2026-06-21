"""Python function-pointer references → REFERENCES_FN edges.

A function passed *by value* (as a callback argument) is a reference, not
a call. The Python parser records these so the builder emits REFERENCES_FN
edges — matching the Rust parser — which keeps callback-only functions out
of dead-code results and reflects real usage.
"""

import textwrap

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402


def _refs(graph) -> set[tuple[str, str]]:
    rows = graph.cypher("MATCH (a:Function)-[:REFERENCES_FN]->(b:Function) RETURN a.name AS a, b.name AS b").to_list()
    return {(r["a"], r["b"]) for r in rows}


def _build(tmp_path, body: str):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "a.py").write_text(textwrap.dedent(body))
    return build(str(tmp_path))


def test_positional_callback_arg(tmp_path):
    g = _build(
        tmp_path,
        """
        def handler(x):
            return x

        def uses(items):
            return list(map(handler, items))
        """,
    )
    assert ("uses", "handler") in _refs(g)


def test_keyword_callback_arg(tmp_path):
    g = _build(
        tmp_path,
        """
        def keyfn(x):
            return -x

        def uses(items):
            return sorted(items, key=keyfn)
        """,
    )
    assert ("uses", "keyfn") in _refs(g)


def test_plain_call_is_not_a_reference(tmp_path):
    # A directly-invoked function is a CALLS edge, not REFERENCES_FN.
    g = _build(
        tmp_path,
        """
        def helper():
            return 1

        def uses():
            return helper()
        """,
    )
    assert ("uses", "helper") not in _refs(g)


def test_variable_arg_does_not_emit_edge(tmp_path):
    # `items` is a variable, not a project function — no false REFERENCES_FN.
    g = _build(
        tmp_path,
        """
        def uses(items):
            return list(filter(None, items))
        """,
    )
    assert not any(b == "items" for _, b in _refs(g))
