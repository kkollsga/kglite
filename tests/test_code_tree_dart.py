"""Dart parser smoke tests.

Coverage grows per implementation phase:
  - Phase 1: class + method extraction (HAS_METHOD), top-level functions,
    File.language == "dart".
  - Phase 2: inheritance (EXTENDS / IMPLEMENTS), mixins (:Mixin nodes),
    extensions (:Class kind="extension"), enums (:Enum + variants).
  - Phase 3: named/factory constructors, getters/setters, member fields,
    top-level const/final → Constant, typedef → Constant, async flag.
  - Phase 4: CALLS edges, cyclomatic branch counts, part/part-of module
    sharing, TODO/FIXME comment annotations.
  - Phase 5: Flutter pass — widget subclasses (flutter_widget) and their
    build methods (flutter_build); constructor flags queryable.
  - Bug-fix pass: relative-import resolution → IMPORTS edges;
    multi-byte comment-annotation panic regression.
"""

import json
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


def test_dart_mixin_is_mixin_node(tmp_path):
    pkg = _write(
        tmp_path,
        "walker.dart",
        """
mixin Walker {
  void walk() {
    print('walking');
  }
}
""",
    )
    g = build(str(pkg))
    mixins = g.cypher("MATCH (m:Mixin {name: 'Walker'}) RETURN m.qualified_name AS q").to_list()
    assert mixins, "expected Walker to be a :Mixin node"
    methods = g.cypher("MATCH (m:Mixin {name: 'Walker'})-[:HAS_METHOD]->(f:Function) RETURN f.name AS n").to_list()
    assert {r["n"] for r in methods} == {"walk"}, methods


def test_dart_class_extends(tmp_path):
    pkg = _write(
        tmp_path,
        "animals.dart",
        """
class Animal {}

class Dog extends Animal {
  void bark() {}
}
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (d:Class {name: 'Dog'})-[:EXTENDS]->(a:Class) RETURN a.name AS n").to_list()
    assert {r["n"] for r in rows} == {"Animal"}, rows


def test_dart_class_implements(tmp_path):
    pkg = _write(
        tmp_path,
        "shapes.dart",
        """
abstract class Shape {}

class Circle implements Shape {}
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (c:Class {name: 'Circle'})-[:IMPLEMENTS]->(s) RETURN s.name AS n").to_list()
    assert "Shape" in {r["n"] for r in rows}, rows


def test_dart_class_with_mixin(tmp_path):
    pkg = _write(
        tmp_path,
        "robot.dart",
        """
mixin Walker {
  void walk() {}
}

class Robot with Walker {}
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (r:Class {name: 'Robot'})-[:IMPLEMENTS]->(m:Mixin) RETURN m.name AS n").to_list()
    assert {r["n"] for r in rows} == {"Walker"}, rows


def test_dart_extension_is_class_kind_extension(tmp_path):
    pkg = _write(
        tmp_path,
        "ext.dart",
        """
extension StringX on String {
  bool get isBlank => trim().isEmpty;
}
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (c:Class {name: 'StringX'}) RETURN c.kind AS k").to_list()
    assert rows, "expected StringX extension to be a :Class node"
    assert rows[0]["k"] == "extension", rows


def test_dart_enum_with_variants(tmp_path):
    pkg = _write(
        tmp_path,
        "color.dart",
        """
enum Color { red, green, blue }
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (e:Enum {name: 'Color'}) RETURN e.variants AS v").to_list()
    assert rows, "expected Color enum to be extracted"
    # `variants` is stored as a comma-joined string property.
    variants = {v.strip() for v in (rows[0]["v"] or "").split(",")}
    assert variants == {"red", "green", "blue"}, rows[0]["v"]


def test_dart_named_and_factory_constructors(tmp_path):
    pkg = _write(
        tmp_path,
        "point.dart",
        """
class Point {
  final int x;
  final int y;
  Point(this.x, this.y);
  Point.origin() : x = 0, y = 0;
  factory Point.fromJson(Map<String, dynamic> j) => Point(j['x'], j['y']);
}
""",
    )
    g = build(str(pkg))
    rows = g.cypher(
        "MATCH (c:Class {name: 'Point'})-[:HAS_METHOD]->(f:Function) RETURN f.name AS n, f.qualified_name AS q"
    ).to_list()
    by_name = {r["n"]: r["q"] for r in rows}
    assert {"Point", "Point.origin", "Point.fromJson"} <= set(by_name), by_name
    # Every constructor resolves to a distinct, addressable qualified_name.
    qnames = [by_name["Point"], by_name["Point.origin"], by_name["Point.fromJson"]]
    assert len(set(qnames)) == 3, qnames


def test_dart_getter_and_setter(tmp_path):
    pkg = _write(
        tmp_path,
        "temp.dart",
        """
class Temp {
  double _c = 0;
  double get celsius => _c;
  set celsius(double v) => _c = v;
}
""",
    )
    g = build(str(pkg))
    methods = g.cypher("MATCH (c:Class {name: 'Temp'})-[:HAS_METHOD]->(f:Function) RETURN f.name AS n").to_list()
    names = {r["n"] for r in methods}
    # Getter and setter share a bare name; the `=` suffix keeps the
    # setter's qualified name distinct.
    assert "celsius" in names, names
    assert "celsius=" in names, names


def test_dart_typedef_is_type_alias_constant(tmp_path):
    pkg = _write(
        tmp_path,
        "types.dart",
        """
typedef IntList = List<int>;
typedef Callback = void Function(int);
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (c:Constant) WHERE c.kind = 'type_alias' RETURN c.name AS n").to_list()
    names = {r["n"] for r in rows}
    assert {"IntList", "Callback"} <= names, names


def test_dart_top_level_const(tmp_path):
    pkg = _write(
        tmp_path,
        "consts.dart",
        """
const int maxRetries = 3;
final String appName = 'KGLite';
var mutable = 1;
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (c:Constant) WHERE c.kind = 'constant' RETURN c.name AS n").to_list()
    names = {r["n"] for r in rows}
    assert {"maxRetries", "appName"} <= names, names
    # Plain mutable `var` is not a constant.
    assert "mutable" not in names, names


def test_dart_member_fields(tmp_path):
    pkg = _write(
        tmp_path,
        "account.dart",
        """
class Account {
  int balance = 0;
  String owner = 'anon';
}
""",
    )
    g = build(str(pkg))
    # Member fields are embedded as the `fields` JSON string property on
    # the Class node (there is no separate Attribute node type).
    rows = g.cypher("MATCH (c:Class {name: 'Account'}) RETURN c.fields AS f").to_list()
    assert rows, "expected Account class node"
    fields = json.loads(rows[0]["f"] or "[]")
    names = {fld["name"] for fld in fields}
    assert {"balance", "owner"} <= names, fields


def test_dart_async_flag(tmp_path):
    pkg = _write(
        tmp_path,
        "fetch.dart",
        """
Future<int> fetchValue() async {
  return 42;
}

int plain() {
  return 1;
}
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (f:Function) RETURN f.name AS n, f.is_async AS a").to_list()
    by_name = {r["n"]: r["a"] for r in rows}
    assert by_name.get("fetchValue") is True, by_name
    assert by_name.get("plain") is False, by_name


def test_dart_calls_resolved(tmp_path):
    pkg = _write(
        tmp_path,
        "calls.dart",
        """
int helper() => 1;

int caller() {
  return helper() + helper();
}
""",
    )
    g = build(str(pkg))
    rows = g.cypher(
        "MATCH (a:Function {name: 'caller'})-[:CALLS]->(b:Function {name: 'helper'}) RETURN b.name AS n"
    ).to_list()
    assert rows, "expected caller → helper CALLS edge"


def test_dart_branch_count(tmp_path):
    pkg = _write(
        tmp_path,
        "classify.dart",
        """
int classify(int n) {
  if (n > 0) {
    return 1;
  } else if (n < 0) {
    return -1;
  }
  return 0;
}
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (f:Function {name: 'classify'}) RETURN f.branch_count AS bc").to_list()
    assert rows, "expected classify function"
    assert (rows[0]["bc"] or 0) >= 2, rows


def test_dart_part_files_share_module(tmp_path):
    pkg = tmp_path / "dart_pkg"
    pkg.mkdir(exist_ok=True)
    (pkg / "lib.dart").write_text("library mylib;\npart 'lib_part.dart';\n\nclass Core {}\n")
    (pkg / "lib_part.dart").write_text("part of 'lib.dart';\n\nclass Helper {}\n")
    g = build(str(pkg))
    rows = g.cypher("MATCH (f:File) RETURN f.filename AS n, f.module AS m").to_list()
    by_file = {r["n"]: r["m"] for r in rows}
    # The `part of` file adopts the parent library's module path.
    assert by_file.get("lib.dart") == by_file.get("lib_part.dart"), by_file
    assert by_file.get("lib_part.dart"), by_file


def test_dart_comment_annotations(tmp_path):
    pkg = _write(
        tmp_path,
        "todo.dart",
        """
// TODO: refactor this mess
void messy() {}
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (f:File {filename: 'todo.dart'}) RETURN f.annotations AS a").to_list()
    assert rows, "expected todo.dart File node"
    annotations = rows[0]["a"] or ""
    assert "refactor" in str(annotations), annotations


def test_dart_flutter_stateless_widget(tmp_path):
    pkg = _write(
        tmp_path,
        "home.dart",
        """
import 'package:flutter/material.dart';

class HomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container();
  }
}

class PlainModel {
  void build() {}
}
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (c:Class) RETURN c.name AS n, c.flutter_widget AS w").to_list()
    widget = {r["n"]: r["w"] for r in rows}
    assert widget.get("HomePage") == "stateless", widget
    assert widget.get("PlainModel") in (None, ""), widget

    builds = g.cypher(
        "MATCH (c:Class {name: 'HomePage'})-[:HAS_METHOD]->(f:Function {name: 'build'}) RETURN f.flutter_build AS fb"
    ).to_list()
    assert builds and builds[0]["fb"] is True, builds
    # The plain class's build() is not a Flutter build method.
    plain = g.cypher(
        "MATCH (c:Class {name: 'PlainModel'})-[:HAS_METHOD]->(f:Function {name: 'build'}) RETURN f.flutter_build AS fb"
    ).to_list()
    assert plain and plain[0]["fb"] is False, plain


def test_dart_flutter_stateful_and_state(tmp_path):
    pkg = _write(
        tmp_path,
        "counter.dart",
        """
import 'package:flutter/material.dart';

class Counter extends StatefulWidget {
  @override
  State<Counter> createState() => _CounterState();
}

class _CounterState extends State<Counter> {
  @override
  Widget build(BuildContext context) => Container();
}
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (c:Class) RETURN c.name AS n, c.flutter_widget AS w").to_list()
    widget = {r["n"]: r["w"] for r in rows}
    assert widget.get("Counter") == "stateful", widget
    assert widget.get("_CounterState") == "state", widget


def test_dart_constructor_flag_queryable(tmp_path):
    pkg = _write(
        tmp_path,
        "box.dart",
        """
class Box {
  Box();
  factory Box.empty() => Box();
}
""",
    )
    g = build(str(pkg))
    ctors = g.cypher("MATCH (f:Function) WHERE f.is_constructor = true RETURN f.name AS n").to_list()
    names = {r["n"] for r in ctors}
    assert {"Box", "Box.empty"} <= names, names
    factory = g.cypher("MATCH (f:Function) WHERE f.is_factory = true RETURN f.name AS n").to_list()
    assert {r["n"] for r in factory} == {"Box.empty"}, factory


def test_dart_multibyte_comment_no_panic(tmp_path):
    # Regression: a TODO comment whose body exceeds 200 bytes with
    # multi-byte box-drawing chars straddling the truncation boundary
    # used to panic extract_comment_annotations (`&body[..200]` slicing
    # inside a `─`). The build must complete cleanly.
    rule = "─" * 90  # 270 bytes of U+2500 (3 bytes each)
    pkg = _write(tmp_path, "boxed.dart", f"// TODO {rule}\nvoid noop() {{}}\n")
    g = build(str(pkg))
    rows = g.cypher("MATCH (f:File {filename: 'boxed.dart'}) RETURN f.path AS p").to_list()
    assert rows, "build must succeed on a multi-byte comment body"


def test_dart_relative_import_resolves(tmp_path):
    pkg = tmp_path / "dart_pkg"
    pkg.mkdir(exist_ok=True)
    (pkg / "helper.dart").write_text("int help() => 1;\n")
    (pkg / "app.dart").write_text("import 'helper.dart';\n\nvoid main() {\n  help();\n}\n")
    g = build(str(pkg))
    # A relative Dart import normalises to the imported file's module and
    # produces an IMPORTS edge.
    rows = g.cypher(
        "MATCH (a:File {filename: 'app.dart'})-[:IMPORTS]->(t) "
        "RETURN labels(t)[0] AS lbl, t.filename AS fn, t.qualified_name AS qn"
    ).to_list()
    assert rows, "expected app.dart → helper IMPORTS edge"
    targets = {(r["fn"] or r["qn"]) for r in rows}
    assert any("helper" in str(t) for t in targets), rows
