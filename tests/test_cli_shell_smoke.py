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


def _run_args(*args: str) -> str:
    """Run the binary as a non-interactive subcommand, return stdout."""
    proc = subprocess.run([str(BINARY), *args], capture_output=True, text=True, timeout=30)
    return proc.stdout


def _run_args_proc(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the binary as a non-interactive subcommand, return the full process."""
    return subprocess.run([str(BINARY), *args], capture_output=True, text=True, timeout=30)


def test_export_text_subcommand(tmp_path):
    import kglite

    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (a:N {id: 1, s: 'todo'})-[:R]->(b:N {id: 2})")
    p = str(tmp_path / "g.kgl")
    g.save(p)
    out = _run_args("export-text", p)
    assert "# N (2 node(s))" in out
    assert "1 | N_0 | s=todo" in out
    assert "(1)-[R]->(2)" in out


def test_diff_subcommand(tmp_path):
    import kglite

    a = str(tmp_path / "a.kgl")
    b = str(tmp_path / "b.kgl")
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:N {id: 1, s: 'todo'}), (:N {id: 2})")
    g.save(a)
    g2 = kglite.KnowledgeGraph()
    g2.cypher("CREATE (:N {id: 1, s: 'done'}), (:N {id: 3})")
    g2.save(b)
    out = _run_args("diff", a, b)
    assert "-1 | N_0 | s=todo" in out  # node 1 changed
    assert "+1 | N_0 | s=done" in out
    assert "-2 | N_1" in out  # node 2 removed
    assert "+3 | N_1" in out  # node 3 added


def test_query_subcommand_json(tmp_path):
    import json

    import kglite

    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Person {name: 'Alice', age: 30})")
    p = tmp_path / "g.kgl"
    g.save(str(p))

    out = _run_args(
        "query",
        str(p),
        "MATCH (p:Person) RETURN p.name AS name, p.age AS age",
        "--format",
        "json",
    )
    rows = json.loads(out)
    assert rows == [{"name": "Alice", "age": 30}]


def test_write_subcommand_saves_graph(tmp_path):
    import kglite

    p = tmp_path / "g.kgl"
    proc = _run_args_proc("write", str(p), "CREATE (:Task {id: 't1'})", "--save")
    assert proc.returncode == 0, proc.stderr
    assert p.exists()
    g = kglite.load(str(p))
    rows = g.cypher("MATCH (t:Task) RETURN t.id AS id").to_dicts()
    assert rows == [{"id": "t1"}]


def test_write_subcommand_scope_rejects_out_of_scope(tmp_path):
    p = tmp_path / "g.kgl"
    proc = _run_args_proc(
        "write",
        str(p),
        "CREATE (:Algorithm {id: 'a1'})",
        "--save",
        "--write-scope",
        "Plan,Task",
    )
    assert proc.returncode != 0
    assert "write scope" in proc.stderr
    assert not p.exists()


def test_write_subcommand_stamps_provenance(tmp_path):
    import kglite

    p = tmp_path / "g.kgl"
    g = kglite.KnowledgeGraph()
    g.define_schema({"nodes": {"Task": {"auto_timestamp": True}}})
    g.save(str(p))

    proc = _run_args_proc(
        "write",
        str(p),
        "CREATE (:Task {id: 't1'})",
        "--save",
        "--git-sha",
        "abc123",
        "--modified-by",
        "cli-agent",
    )
    assert proc.returncode == 0, proc.stderr
    rows = (
        kglite.load(str(p)).cypher("MATCH (t:Task {id: 't1'}) RETURN t.git_sha AS sha, t.modified_by AS by").to_dicts()
    )
    assert rows == [{"sha": "abc123", "by": "cli-agent"}]


def test_concurrent_write_save_serializes(tmp_path):
    import kglite

    p = tmp_path / "shared.kgl"
    seed = kglite.KnowledgeGraph()
    seed.save(str(p))

    commands = [
        [str(BINARY), "write", str(p), "CREATE (:Task {id: 'a'})", "--save"],
        [str(BINARY), "write", str(p), "CREATE (:Task {id: 'b'})", "--save"],
    ]
    procs = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for cmd in commands]
    results = [proc.communicate(timeout=30) + (proc.returncode,) for proc in procs]
    for stdout, stderr, code in results:
        assert code == 0, f"stdout={stdout}\nstderr={stderr}"

    rows = kglite.load(str(p)).cypher("MATCH (t:Task) RETURN t.id AS id ORDER BY id").to_dicts()
    assert rows == [{"id": "a"}, {"id": "b"}]
    assert not (tmp_path / "shared.kgl.lock").exists()


def test_ready_set_subcommand(tmp_path):
    import json

    import kglite

    p = tmp_path / "dag.kgl"
    g = kglite.KnowledgeGraph()
    for n, s in [("A", "todo"), ("B", "todo"), ("C", "done")]:
        g.cypher(f"CREATE (:Task {{id:'{n}', status:'{s}'}})")
    for a, b in [("B", "C"), ("A", "B")]:
        g.cypher(f"MATCH (x:Task {{id:'{a}'}}),(y:Task {{id:'{b}'}}) CREATE (x)-[:DEPENDS_ON]->(y)")
    g.save(str(p))

    out = _run_args(
        "ready-set",
        str(p),
        "--done",
        'n.status = "done"',
        "--node-type",
        "Task",
        "--format",
        "json",
    )
    rows = json.loads(out)
    assert rows == [{"dependency_count": 1, "id": "B", "title": "Task_1"}]


def test_describe_subcommand(tmp_path):
    import kglite

    p = tmp_path / "g.kgl"
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Task {id: 't1', status: 'todo'})")
    g.save(str(p))

    out = _run_args("describe", str(p), "--types", "Task")
    assert '<type name="Task"' in out
    cypher = _run_args("describe", str(p), "--cypher")
    assert "<cypher" in cypher


def test_session_keeps_graph_loaded_between_requests(tmp_path):
    import json

    import kglite

    p = tmp_path / "session.kgl"
    seed = kglite.KnowledgeGraph()
    seed.save(str(p))
    requests = "\n".join(
        [
            json.dumps({"id": "w1", "op": "write", "query": "CREATE (:Task {id: 'one'})"}),
            json.dumps(
                {
                    "id": "q1",
                    "op": "query",
                    "query": "MATCH (t:Task) RETURN count(t) AS n",
                    "format": "json",
                }
            ),
            json.dumps({"id": "d1", "op": "describe", "types": ["Task"]}),
            json.dumps({"id": "s1", "op": "save"}),
            json.dumps({"id": "x1", "op": "exit"}),
            "",
        ]
    )
    proc = subprocess.run(
        [str(BINARY), "session", str(p), "--format", "json"],
        input=requests,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    responses = [json.loads(line) for line in proc.stdout.splitlines()]
    assert all(r["ok"] for r in responses), responses
    assert [r["id"] for r in responses] == ["w1", "q1", "d1", "s1", "x1"]
    assert responses[1]["rows"] == [{"n": 1}]
    assert "output" not in responses[1]
    assert '<type name="Task"' in responses[2]["description"]
    rows = kglite.load(str(p)).cypher("MATCH (t:Task) RETURN t.id AS id").to_dicts()
    assert rows == [{"id": "one"}]


def test_create_and_query_roundtrip():
    out = _run(
        'CREATE (:Person {name: "Alice", age: 30});\n'
        'CREATE (:Person {name: "Bob", age: 25});\n'
        "MATCH (p:Person) RETURN p.name AS name ORDER BY name;\n"
        ".quit\n"
    )
    assert "Alice" in out
    assert "Bob" in out
    assert "(2 rows)" in out


def test_db_introspection_in_shell():
    """The new db.* procedures are reachable from the shell."""
    out = _run(
        "CREATE (:Person {name: 'A'});\n"
        "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey ORDER BY propertyKey;\n"
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
    out = _run("MATCH bogus syntax;\nRETURN 1 AS one;\n.quit\n")
    assert "error:" in out
    assert "(1 row)" in out  # the next statement still ran


def test_multiline_statement_runs_on_semicolon():
    """A statement split across lines runs only once `;` terminates it."""
    out = _run("MATCH (n)\nRETURN count(n)\nAS c;\n.quit\n")
    assert "(1 row)" in out  # combined into one statement, ran once


def test_mode_csv_and_json():
    # CREATE first (table mode), then switch mode and run only the query, so the
    # formatted output is a single result (a write under json mode renders []).
    create = 'CREATE (:Person {name: "Alice", age: 30});\n'
    query = "MATCH (p:Person) RETURN p.name AS name, p.age AS age;\n"

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
    out = _run("CREATE (:Person {name: 'A', city: 'Oslo'});\n.schema\n.quit\n")
    assert "Person" in out


def test_dump_roundtrips_via_from_blueprint(tmp_path):
    """`.dump` writes a portable copy that from_blueprint() rebuilds."""
    import kglite

    dump_dir = tmp_path / "backup"
    _run(
        'CREATE (:Person {name: "Alice", age: 30});\n'
        'CREATE (:Person {name: "Bob", age: 25});\n'
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
    _run(f'CREATE (:Person {{name: "Alice"}});\nCREATE (:Person {{name: "Bob"}});\n.save {kgl}\n.quit\n')
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
        "MATCH (p:Person) RETURN count(p) AS c;\n"
        "MATCH (p:Person {id: 2}) RETURN p.name AS n, p.age AS a;\n"
        ".quit\n"
    )
    assert "imported 2 Person node(s)" in out
    assert "(1 row)" in out
    assert "Bob" in out
    assert "25" in out  # age inferred as a number, matchable


def test_timing_reports_walltime():
    out = _run(".timing on\nRETURN 1 AS x;\n.quit\n")
    assert "timing on" in out
    assert "ms)" in out  # a "(... ms)" line after the result


def test_import_rejects_bad_node_type(tmp_path):
    csv = tmp_path / "x.csv"
    csv.write_text("id\n1\n")
    out = _run(f".import {csv} 9bad\n.quit\n")
    assert "not a valid node type" in out


def test_read_runs_a_cypher_file(tmp_path):
    script = tmp_path / "seed.cypher"
    script.write_text("CREATE (:Person {name: 'Alice'});\nCREATE (:Person {name: 'Bob'});\n")
    out = _run(f".read {script}\nMATCH (p:Person) RETURN count(p) AS n;\n.quit\n")
    assert "(1 row)" in out
    assert "2" in out  # the count after seeding
