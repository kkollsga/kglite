"""Dart parser smoke tests.

Coverage grows per implementation phase:
  - Phase 1: class + method extraction (HAS_METHOD), top-level functions,
    File.language == "dart".

Import-edge resolution (URI → module/file) lands with the part/part-of
work in a later phase; that's where import assertions are added.
"""

import pathlib

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402


def _write(tmp_path, name: str, content: str) -> pathlib.Path:
    pkg = tmp_path / "dart_pkg"
    pkg.mkdir(exist_ok=True)
    (pkg / name).write_text(content)
    return pkg


def test_dart_file_indexed(tmp_path):
    pkg = _write(
        tmp_path,
        "main.dart",
        """
void main() {
  print('hello');
}
""",
    )
    g = build(str(pkg))
    files = g.cypher("MATCH (f:File) RETURN f.path AS p, f.language AS lang").to_list()
    assert files, "expected the .dart file to be indexed"
    assert all(r["lang"] == "dart" for r in files), files


def test_dart_class_and_method(tmp_path):
    pkg = _write(
        tmp_path,
        "greeter.dart",
        """
class Greeter {
  String greet(String name) {
    return 'hi $name';
  }

  void _whisper() {
    print('psst');
  }
}
""",
    )
    g = build(str(pkg))
    classes = g.cypher("MATCH (c:Class {name: 'Greeter'}) RETURN c.qualified_name AS q").to_list()
    assert classes, "expected Greeter class to be extracted"

    methods = g.cypher(
        "MATCH (c:Class {name: 'Greeter'})-[:HAS_METHOD]->(f:Function) RETURN f.name AS n ORDER BY n"
    ).to_list()
    names = {r["n"] for r in methods}
    assert {"greet", "_whisper"} <= names, methods


def test_dart_method_visibility(tmp_path):
    pkg = _write(
        tmp_path,
        "vis.dart",
        """
class Vis {
  void publicOne() {}
  void _privateOne() {}
}
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (f:Function) RETURN f.name AS n, f.visibility AS v").to_list()
    vis = {r["n"]: r["v"] for r in rows}
    assert vis.get("publicOne") == "public", vis
    assert vis.get("_privateOne") == "private", vis


def test_dart_top_level_function(tmp_path):
    pkg = _write(
        tmp_path,
        "util.dart",
        """
int add(int a, int b) {
  return a + b;
}
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (f:Function {name: 'add'}) RETURN f.is_method AS m, f.return_type AS rt").to_list()
    assert rows, "expected top-level function add to be extracted"
    assert rows[0]["m"] is False, rows
    assert rows[0]["rt"] == "int", rows
