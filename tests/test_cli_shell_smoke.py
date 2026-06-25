"""End-to-end smoke tests for the `kglite` interactive shell binary.

Drives the REPL the way a user would: pipe newline-separated input on stdin
and assert on stdout. Skipped when the binary isn't built. Build it with::

    cargo build --release -p kglite-cli

The release binary lands at target/release/kglite.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

# Prefer the release binary (what CI/users ship); fall back to debug so a local
# `cargo build -p kglite-cli` is enough to exercise these.
_ROOT = Path(__file__).resolve().parent.parent
_RELEASE = _ROOT / "target" / "release" / "kglite"
_DEBUG = _ROOT / "target" / "debug" / "kglite"
BINARY = _RELEASE if _RELEASE.exists() else _DEBUG

pytestmark = pytest.mark.skipif(
    not BINARY.exists(),
    reason=f"kglite shell binary not built (looked at {_RELEASE} and {_DEBUG}). "
    "Build with: cargo build --release -p kglite-cli",
)


def _run(script: str) -> str:
    """Feed `script` to the shell on stdin, return combined stdout+stderr."""
    proc = subprocess.run(
        [str(BINARY)],
        input=script,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc.stdout + proc.stderr


def test_create_and_query_roundtrip():
    out = _run(
        'CREATE (:Person {name: "Alice", age: 30})\n'
        'CREATE (:Person {name: "Bob", age: 25})\n'
        "MATCH (p:Person) RETURN p.name AS name ORDER BY name\n"
        ".quit\n"
    )
    assert "Alice" in out
    assert "Bob" in out
    assert "(2 rows)" in out


def test_db_introspection_in_shell():
    """The new db.* procedures are reachable from the shell."""
    out = _run(
        "CREATE (:Person {name: 'A'})\n"
        "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey ORDER BY propertyKey\n"
        ".quit\n"
    )
    assert "propertyKey" in out
    assert "name" in out


def test_help_and_unknown_dotcommand():
    out = _run(".help\n.nope\n.quit\n")
    assert ".quit" in out  # help lists it
    assert "Unknown command '.nope'" in out


def test_cypher_error_is_reported_not_fatal():
    """A bad query prints an error but the session continues."""
    out = _run("MATCH bogus syntax\nRETURN 1 AS one\n.quit\n")
    assert "error:" in out
    assert "(1 row)" in out  # the next statement still ran


def test_mode_csv_and_json():
    # CREATE first (table mode), then switch mode and run only the query, so the
    # formatted output is a single result (a write under json mode renders []).
    create = 'CREATE (:Person {name: "Alice", age: 30})\n'
    query = "MATCH (p:Person) RETURN p.name AS name, p.age AS age\n"

    csv_out = _run(create + ".mode csv\n" + query + ".quit\n")
    assert "name,age" in csv_out
    assert "Alice,30" in csv_out  # string unquoted, int bare

    json_out = _run(create + ".mode json\n" + query + ".quit\n")
    import json

    start = json_out.index("[")
    end = json_out.rindex("]") + 1
    parsed = json.loads(json_out[start:end])
    assert parsed[0]["name"] == "Alice"
    assert parsed[0]["age"] == 30  # number, not "30"


def test_schema_dotcommand():
    out = _run("CREATE (:Person {name: 'A', city: 'Oslo'})\n.schema\n.quit\n")
    assert "Person" in out


def test_dump_roundtrips_via_from_blueprint(tmp_path):
    """`.dump` writes a portable copy that from_blueprint() rebuilds."""
    import kglite

    dump_dir = tmp_path / "backup"
    _run(
        'CREATE (:Person {name: "Alice", age: 30})\n'
        'CREATE (:Person {name: "Bob", age: 25})\n'
        f".dump {dump_dir}\n.quit\n"
    )
    assert (dump_dir / "blueprint.json").exists()
    g = kglite.from_blueprint(str(dump_dir / "blueprint.json"))
    rows = g.cypher("MATCH (p:Person) RETURN count(p) AS n")
    assert rows[0]["n"] == 2


def test_save_roundtrips_via_load(tmp_path):
    """`.save` writes a .kgl that kglite.load() reopens."""
    import kglite

    kgl = tmp_path / "demo.kgl"
    _run(f'CREATE (:Person {{name: "Alice"}})\nCREATE (:Person {{name: "Bob"}})\n.save {kgl}\n.quit\n')
    assert kgl.exists()
    g = kglite.load(str(kgl))
    rows = g.cypher("MATCH (p:Person) RETURN count(p) AS n")
    assert rows[0]["n"] == 2


def test_import_csv_loads_nodes(tmp_path):
    """`.import file.csv Type` loads rows as nodes with type inference; `id`
    becomes the node identity."""
    csv = tmp_path / "people.csv"
    csv.write_text("id,name,age\n1,Alice,30\n2,Bob,25\n")
    out = _run(
        f".import {csv} Person\n"
        "MATCH (p:Person) RETURN count(p) AS c\n"
        "MATCH (p:Person {id: 2}) RETURN p.name AS n, p.age AS a\n"
        ".quit\n"
    )
    assert "imported 2 Person node(s)" in out
    assert "(1 row)" in out
    assert "Bob" in out
    assert "25" in out  # age inferred as a number, matchable


def test_import_rejects_bad_node_type(tmp_path):
    csv = tmp_path / "x.csv"
    csv.write_text("id\n1\n")
    out = _run(f".import {csv} 9bad\n.quit\n")
    assert "not a valid node type" in out


def test_read_runs_a_cypher_file(tmp_path):
    script = tmp_path / "seed.cypher"
    script.write_text("CREATE (:Person {name: 'Alice'});\nCREATE (:Person {name: 'Bob'});\n")
    out = _run(f".read {script}\nMATCH (p:Person) RETURN count(p) AS n\n.quit\n")
    assert "(1 row)" in out
    assert "2" in out  # the count after seeding
