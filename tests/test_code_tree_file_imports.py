"""File -[IMPORTS]-> File edges — direct file-level dependency.

Sibling to the existing File -[IMPORTS]-> Module edge. Resolves import strings
to project files via the `module_path → file_path` reverse index, walking the
import path from longest to shortest prefix.

Powers transitive impact-analysis queries — given a set of changed files,
find the files (and test files) that depend on them.
"""

import pathlib
import textwrap

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402


def _make_pkg(tmp_path, files: dict[str, str]) -> pathlib.Path:
    """Write a small Python package under tmp_path/pkg and return its root.

    Using ``pkg`` as the build root makes ``pkg`` the project's basename,
    so module qualified names are ``pkg.util`` etc. — matching the import
    strings the parser captures literally from source (e.g. ``from pkg.util
    import helper``). Without this alignment the import resolution can't
    find the target module.
    """
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for rel, content in files.items():
        fp = pkg / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(textwrap.dedent(content))
    return pkg


def test_python_file_imports_resolve_to_files(tmp_path):
    pkg = _make_pkg(
        tmp_path,
        {
            "util.py": """
            def helper():
                return 1
            """,
            "core.py": """
            from pkg.util import helper

            def run():
                return helper()
            """,
        },
    )
    g = build(str(pkg))
    rows = g.cypher(
        """
        MATCH (s:File)-[:IMPORTS]->(t:File)
        RETURN s.path AS src, t.path AS tgt
        """
    ).to_list()
    pairs = {(r["src"], r["tgt"]) for r in rows}
    assert ("core.py", "util.py") in pairs, f"missing core→util edge; got {pairs}"


def test_file_imports_carry_count(tmp_path):
    """Multiple imports from one file to the same target collapse into one edge whose count records the multiplicity."""
    pkg = _make_pkg(
        tmp_path,
        {
            "util.py": """
            def a():
                return 1

            def b():
                return 2
            """,
            "core.py": """
            from pkg.util import a
            from pkg.util import b

            def run():
                return a() + b()
            """,
        },
    )
    g = build(str(pkg))
    rows = g.cypher(
        """
        MATCH (s:File)-[r:IMPORTS]->(t:File)
        WHERE s.path = "core.py" AND t.path = "util.py"
        RETURN r.import_count AS c
        """
    ).to_list()
    assert rows, "expected one File→File IMPORTS edge"
    assert rows[0]["c"] == 2, f"expected import_count=2, got {rows[0]['c']}"


def test_no_self_imports(tmp_path):
    """A file's own module path should never produce a self-loop."""
    pkg = _make_pkg(
        tmp_path,
        {
            "solo.py": """
            def alone():
                return 1
            """,
        },
    )
    g = build(str(pkg))
    rows = g.cypher(
        """
        MATCH (s:File)-[:IMPORTS]->(t:File)
        WHERE s.path = t.path
        RETURN s.path AS p
        """
    ).to_list()
    assert rows == [], f"expected no self-IMPORTS, got {rows}"


def test_file_to_module_imports_still_emitted(tmp_path):
    """Regression — the existing File→Module IMPORTS edges must keep working."""
    pkg = _make_pkg(
        tmp_path,
        {
            "util.py": """
            def helper():
                return 1
            """,
            "core.py": """
            from pkg.util import helper

            def run():
                return helper()
            """,
        },
    )
    g = build(str(pkg))
    rows = g.cypher(
        """
        MATCH (s:File)-[:IMPORTS]->(t:Module)
        WHERE s.path = "core.py"
        RETURN t.qualified_name AS m
        """
    ).to_list()
    modules = {r["m"] for r in rows}
    assert "pkg.util" in modules, f"expected File→Module IMPORTS to pkg.util, got {modules}"
