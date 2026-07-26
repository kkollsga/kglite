"""`LOAD CSV` over Bolt — the filesystem capability, from the client's side.

A Bolt client is a *remote* caller. Serving `LOAD CSV` to it without a gate
would publish an arbitrary-file-read primitive: `LOAD CSV FROM
'file:///etc/passwd' AS row RETURN row` runs on the server, with the server's
filesystem rights, and returns the contents to whoever opened the socket.

So the capability is opt-in per server:

* no flag                        → every `LOAD CSV` is refused
* `--allow-csv-import <DIR>`     → files inside DIR only, after symlink
                                   resolution, so `..` and symlinks cannot
                                   escape

These tests assert that from the outside, over the wire, with the official
Python driver — the in-process behaviour is covered by `tests/test_load_csv.py`
and the resolution logic by the engine's own unit tests.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

neo4j = pytest.importorskip("neo4j")

from tests.conftest import (  # noqa: E402
    _BOLT_SKIP_REASON,
    _bolt_binary_available,
    _build_bolt_fixture_graph,
    _spawn_bolt_server,
    _teardown_bolt_server,
)

pytestmark = pytest.mark.bolt


def _write_csv(path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "name"])
        writer.writerow(["1", "Alice"])
        writer.writerow(["2", "Bob"])
    return path


@pytest.fixture
def import_dir(tmp_path: Path) -> Path:
    d = tmp_path / "import"
    d.mkdir()
    _write_csv(d / "people.csv")
    return d


def _server(tmp_path: Path, extra_args: list[str] | None = None):
    if not _bolt_binary_available():
        pytest.skip(_BOLT_SKIP_REASON)
    fixture = tmp_path / "capability.kgl"
    _build_bolt_fixture_graph(fixture)
    return _spawn_bolt_server(fixture, readonly=False, extra_args=extra_args)


def _run(url: str, query: str) -> list[dict]:
    with neo4j.GraphDatabase.driver(url, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            return [dict(record) for record in session.run(query)]


def test_load_csv_is_refused_without_the_flag(tmp_path: Path, import_dir: Path) -> None:
    proc, url = _server(tmp_path)
    try:
        with pytest.raises(neo4j.exceptions.Neo4jError) as excinfo:
            _run(url, f"LOAD CSV WITH HEADERS FROM 'file://{import_dir / 'people.csv'}' AS row RETURN row.name AS name")
        message = str(excinfo.value)
        assert "not enabled for this connection" in message
        assert "--allow-csv-import" in message
    finally:
        _teardown_bolt_server(proc)


def test_arbitrary_server_file_is_not_readable_by_default(tmp_path: Path) -> None:
    """The reason the gate exists. `/etc/hosts` is present on every CI runner
    and every dev machine, and readable by the server process."""
    proc, url = _server(tmp_path)
    try:
        with pytest.raises(neo4j.exceptions.Neo4jError) as excinfo:
            _run(url, "LOAD CSV FROM 'file:///etc/hosts' AS row RETURN row[0] AS line")
        assert "not enabled for this connection" in str(excinfo.value)
    finally:
        _teardown_bolt_server(proc)


def test_load_csv_works_inside_the_import_directory(tmp_path: Path, import_dir: Path) -> None:
    proc, url = _server(tmp_path, ["--allow-csv-import", str(import_dir)])
    try:
        rows = _run(
            url,
            f"LOAD CSV WITH HEADERS FROM 'file://{import_dir / 'people.csv'}' AS row RETURN row.name AS name",
        )
        assert [r["name"] for r in rows] == ["Alice", "Bob"]
    finally:
        _teardown_bolt_server(proc)


def test_relative_paths_resolve_against_the_import_directory(tmp_path: Path, import_dir: Path) -> None:
    """A relative path resolves against the import root, not the server's
    working directory — more useful, and it removes cwd as an escape route."""
    proc, url = _server(tmp_path, ["--allow-csv-import", str(import_dir)])
    try:
        rows = _run(url, "LOAD CSV WITH HEADERS FROM 'people.csv' AS row RETURN row.name AS name")
        assert [r["name"] for r in rows] == ["Alice", "Bob"]
    finally:
        _teardown_bolt_server(proc)


def test_files_outside_the_import_directory_are_refused(tmp_path: Path, import_dir: Path) -> None:
    outside = _write_csv(tmp_path / "outside.csv")
    proc, url = _server(tmp_path, ["--allow-csv-import", str(import_dir)])
    try:
        with pytest.raises(neo4j.exceptions.Neo4jError) as excinfo:
            _run(url, f"LOAD CSV WITH HEADERS FROM 'file://{outside}' AS row RETURN row.name AS name")
        assert "resolves outside it" in str(excinfo.value)
    finally:
        _teardown_bolt_server(proc)


def test_dot_dot_traversal_out_of_the_import_directory_is_refused(tmp_path: Path, import_dir: Path) -> None:
    outside = _write_csv(tmp_path / "secret.csv")
    proc, url = _server(tmp_path, ["--allow-csv-import", str(import_dir)])
    try:
        traversal = f"{import_dir}/../{outside.name}"
        with pytest.raises(neo4j.exceptions.Neo4jError) as excinfo:
            _run(url, f"LOAD CSV WITH HEADERS FROM 'file://{traversal}' AS row RETURN row.name AS name")
        assert "resolves outside it" in str(excinfo.value)
    finally:
        _teardown_bolt_server(proc)


def test_symlink_out_of_the_import_directory_is_refused(tmp_path: Path, import_dir: Path) -> None:
    """Containment is checked after `canonicalize`, so a symlink planted inside
    the import directory does not become a tunnel out of it."""
    outside = _write_csv(tmp_path / "linked.csv")
    link = import_dir / "sneaky.csv"
    link.symlink_to(outside)
    proc, url = _server(tmp_path, ["--allow-csv-import", str(import_dir)])
    try:
        with pytest.raises(neo4j.exceptions.Neo4jError) as excinfo:
            _run(url, f"LOAD CSV WITH HEADERS FROM 'file://{link}' AS row RETURN row.name AS name")
        assert "resolves outside it" in str(excinfo.value)
    finally:
        _teardown_bolt_server(proc)


def test_http_source_is_refused_even_when_imports_are_enabled(tmp_path: Path, import_dir: Path) -> None:
    """The network-free rejection is about the engine having no HTTP client, so
    enabling local imports must not open a network path."""
    proc, url = _server(tmp_path, ["--allow-csv-import", str(import_dir)])
    try:
        with pytest.raises(neo4j.exceptions.Neo4jError) as excinfo:
            _run(url, "LOAD CSV FROM 'https://example.com/data.csv' AS row RETURN row")
        assert "network-free" in str(excinfo.value)
    finally:
        _teardown_bolt_server(proc)
