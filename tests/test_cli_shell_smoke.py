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
