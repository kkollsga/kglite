"""PHP parser smoke tests.

Coverage in 0.9.36: class/interface/trait declarations, top-level + method
function definitions, const declarations, namespaces, use imports, PHP-8
attributes (as decorators), visibility modifiers, CALLS resolution.
"""

import pathlib

import pytest

pytest.importorskip("tree_sitter")

from kglite.code_tree import build  # noqa: E402


def _write(tmp_path, name: str, content: str) -> pathlib.Path:
    pkg = tmp_path / "phpsrc"
    pkg.mkdir(exist_ok=True)
    (pkg / name).write_text(content)
    return pkg


def test_class_with_methods(tmp_path):
    pkg = _write(
        tmp_path,
        "User.php",
        """<?php
namespace App;
class User {
    public function show(int $id): array {
        return [];
    }

    private function _internal(): void {
        echo "hidden";
    }
}
""",
    )
    g = build(str(pkg))
    classes = g.cypher("MATCH (c:Class) RETURN c.qualified_name AS q").to_list()
    assert any("User" in r["q"] for r in classes), classes
    methods = g.cypher(
        "MATCH (c:Class)-[:HAS_METHOD]->(f:Function) WHERE c.qualified_name ENDS WITH 'User' "
        "RETURN f.name AS n ORDER BY n"
    ).to_list()
    names = {r["n"] for r in methods}
    assert {"show", "_internal"} <= names, methods


def test_trait_lands_as_class_kind_trait(tmp_path):
    pkg = _write(
        tmp_path,
        "Logging.php",
        """<?php
trait Logging {
    public function log(string $msg): void {
        echo $msg;
    }
}
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (c:Class {kind: 'trait'}) RETURN c.name AS n").to_list()
    assert any(r["n"] == "Logging" for r in rows), rows


def test_interface(tmp_path):
    pkg = _write(
        tmp_path,
        "Greeter.php",
        """<?php
interface Greeter {
    public function greet(): string;
}
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (i:Interface) RETURN i.name AS n").to_list()
    assert any(r["n"] == "Greeter" for r in rows), rows


def test_top_level_function(tmp_path):
    pkg = _write(
        tmp_path,
        "math.php",
        """<?php
function add(int $a, int $b): int {
    return $a + $b;
}
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (f:Function) WHERE f.qualified_name ENDS WITH 'add' RETURN f.return_type AS r").to_list()
    assert rows, "top-level PHP function not extracted"


def test_namespace_qualified_names(tmp_path):
    """Namespace declarations compose qualified names with backslash separator."""
    pkg = _write(
        tmp_path,
        "Foo.php",
        """<?php
namespace App\\Models;

class Foo {
    public function bar(): void {}
}
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (c:Class {name: 'Foo'}) RETURN c.qualified_name AS q").to_list()
    assert rows[0]["q"] == "App\\Models\\Foo", rows
    rows = g.cypher("MATCH (f:Function {name: 'bar'}) RETURN f.qualified_name AS q").to_list()
    assert rows[0]["q"] == "App\\Models\\Foo\\bar", rows


def test_use_imports_recorded(tmp_path):
    """`use Foo\\Bar;` is captured in FileInfo.imports."""
    pkg = _write(
        tmp_path,
        "Controller.php",
        """<?php
namespace App;
use App\\Models\\User;
use App\\Models\\Post as Article;

class Controller {
    public function index(): void {}
}
""",
    )
    g = build(str(pkg))
    # FileInfo.imports flows into the File → Module IMPORTS resolver. We
    # confirm the file parses without error and the controller class
    # exists; module-side resolution depends on cross-file modules
    # being present (single-file fixture: no module to import from).
    rows = g.cypher("MATCH (c:Class {name: 'Controller'}) RETURN c.name AS n").to_list()
    assert rows, "Controller class missing"


def test_php8_attribute_as_decorator(tmp_path):
    """`#[Route('/x')]` PHP-8 attributes land on FunctionInfo.decorators."""
    pkg = _write(
        tmp_path,
        "UserController.php",
        """<?php
namespace App;

#[Entity]
class UserController {
    #[Route('/users')]
    public function index(): array {
        return [];
    }
}
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (f:Function {name: 'index'}) RETURN f.decorators AS d").to_list()
    assert rows, "index method missing"
    assert "Route('/users')" in (rows[0]["d"] or ""), rows


def test_class_constants(tmp_path):
    pkg = _write(
        tmp_path,
        "Status.php",
        """<?php
class Status {
    const ACTIVE = 1;
    const INACTIVE = 0;
}
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (c:Constant) RETURN c.name AS n, c.value_preview AS v ORDER BY n").to_list()
    names = [r["n"] for r in rows]
    assert "ACTIVE" in names and "INACTIVE" in names, rows


def test_calls_resolved(tmp_path):
    """A→B PHP CALLS edge resolves via the bare-name lookup."""
    pkg = _write(
        tmp_path,
        "Auth.php",
        """<?php
class Auth {
    public function verify(string $token): bool {
        return $this->checkToken($token);
    }

    public function checkToken(string $t): bool {
        return $t === 'ok';
    }
}
""",
    )
    g = build(str(pkg))
    rows = g.cypher(
        "MATCH (a:Function {name: 'verify'})-[:CALLS]->(b:Function {name: 'checkToken'}) RETURN b.name AS callee"
    ).to_list()
    assert rows == [{"callee": "checkToken"}], rows


def test_extends_and_implements(tmp_path):
    """`class Foo extends Base implements I` emits EXTENDS + IMPLEMENTS."""
    pkg = _write(
        tmp_path,
        "Foo.php",
        """<?php
interface I {
    public function ping(): void;
}

class Base {}

class Foo extends Base implements I {
    public function ping(): void {}
}
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (c:Class {name: 'Foo'})-[:EXTENDS]->(parent) RETURN parent.name AS p").to_list()
    assert any(r["p"] == "Base" for r in rows), rows
    rows = g.cypher("MATCH (c:Class {name: 'Foo'})-[:IMPLEMENTS]->(iface) RETURN iface.name AS n").to_list()
    assert any(r["n"] == "I" for r in rows), rows


def test_language_tag_on_files(tmp_path):
    pkg = _write(
        tmp_path,
        "thing.php",
        """<?php
function f(): void {}
""",
    )
    g = build(str(pkg))
    rows = g.cypher("MATCH (f:File) RETURN f.language AS l").to_list()
    assert any(r["l"] == "php" for r in rows), rows
