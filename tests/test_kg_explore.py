"""KnowledgeGraph.explore() — one-call codebase exploration.

Lexically ranks Function/Class/Interface nodes against a free-text
query, takes the top entries, 2-hop traverses CALLS/USES_TYPE/HAS_METHOD/
DEFINES/REFERENCES_FN, and returns a markdown report.
"""

import pathlib
import textwrap

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402


def _make_pkg(tmp_path, files: dict[str, str]) -> pathlib.Path:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for rel, content in files.items():
        fp = pkg / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(textwrap.dedent(content))
    return pkg


def test_explore_finds_entry_points_by_name(tmp_path):
    pkg = _make_pkg(
        tmp_path,
        {
            "auth.py": """
            def authenticate(user, password):
                '''Verify user credentials.'''
                return _check_password(user, password)

            def _check_password(user, password):
                return password == 'secret'

            def unrelated_helper():
                return 42
            """,
        },
    )
    g = build(str(pkg))
    md = g.explore("authenticate", max_entities=5, max_depth=1, include_source=False)
    assert "## Entry points" in md, md
    assert "authenticate" in md, md
    # The query-name match should rank above 'unrelated_helper'.
    auth_idx = md.find("authenticate")
    unrelated_idx = md.find("unrelated_helper")
    assert auth_idx >= 0
    if unrelated_idx >= 0:
        assert auth_idx < unrelated_idx, "authenticate should rank first"


def test_explore_traverses_to_neighbors(tmp_path):
    """Related functions reachable via CALLS show up under Related."""
    pkg = _make_pkg(
        tmp_path,
        {
            "auth.py": """
            def authenticate(user, password):
                return _check_password(user, password)

            def _check_password(user, password):
                return password == 'secret'
            """,
        },
    )
    g = build(str(pkg))
    md = g.explore("authenticate", max_entities=3, max_depth=2, include_source=False)
    # Both the entry point and its CALLS neighbor should appear.
    assert "_check_password" in md, md


def test_explore_empty_graph_returns_no_match_message(tmp_path):
    """A graph with no matching entities yields a clear 'no match' message."""
    pkg = _make_pkg(
        tmp_path,
        {
            "lib.py": """
            def add(a, b):
                return a + b
            """,
        },
    )
    g = build(str(pkg))
    md = g.explore("authenticate", max_entities=10)
    assert "No matching" in md or "0" in md, md


def test_explore_empty_query_handled(tmp_path):
    """Empty query is a benign no-op, not an error."""
    pkg = _make_pkg(
        tmp_path,
        {"lib.py": "def f(): return 1"},
    )
    g = build(str(pkg))
    md = g.explore("", max_entities=5)
    assert "empty query" in md.lower(), md


def test_explore_include_source_emits_code(tmp_path):
    """With include_source=True (default), source slices are emitted."""
    pkg = _make_pkg(
        tmp_path,
        {
            "auth.py": """
            def authenticate(user, password):
                '''Verify credentials.'''
                return password == 'secret'
            """,
        },
    )
    g = build(str(pkg))
    md = g.explore("authenticate", max_entities=3, include_source=True, source_roots=[str(pkg)])
    assert "## Source" in md, md
    # The authenticate function body should appear in a fenced block.
    assert "def authenticate" in md, md


def test_explore_include_source_false_omits_source(tmp_path):
    pkg = _make_pkg(
        tmp_path,
        {"auth.py": "def authenticate(u, p): return True"},
    )
    g = build(str(pkg))
    md = g.explore("authenticate", include_source=False, source_roots=[str(pkg)])
    assert "## Source" not in md, md


def test_explore_ranks_signature_match(tmp_path):
    """A name not matching the query but signature substring matching still surfaces."""
    pkg = _make_pkg(
        tmp_path,
        {
            "lib.py": """
            from typing import Optional

            def lookup(key: str) -> Optional[str]:
                return None

            def add(a: int, b: int) -> int:
                return a + b
            """,
        },
    )
    g = build(str(pkg))
    md = g.explore("Optional", max_entities=3, include_source=False)
    # 'lookup' has 'Optional' in its return-type signature.
    assert "lookup" in md, md


def test_explore_on_non_code_graph_returns_no_match(tmp_path):
    """Calling explore() on a graph with no code-tree node types returns the
    'no match' message rather than erroring."""
    from kglite import KnowledgeGraph

    kg = KnowledgeGraph()
    md = kg.explore("anything")
    assert "No matching" in md or "0" in md, md
