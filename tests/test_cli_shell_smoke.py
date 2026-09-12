"""End-to-end smoke tests for the `kglite` interactive shell binary.

Drives the REPL the way a user would: pipe newline-separated input on stdin
and assert on stdout. Skipped when the binary isn't built. Build it with::

    cargo build --release -p kglite-cli

The release binary lands at target/release/kglite.
"""

from __future__ import annotations

import subprocess

import pytest

# Resolve the newest built profile (release or debug) so a local
# `cargo build -p kglite-cli` is enough to exercise these, and a stale
# release binary never shadows a fresh debug build. Skips (with the rebuild
# command) when nothing fresh is built; CI always builds fresh.
from tests.conftest import binary_skip_reason, workspace_binary

BINARY = workspace_binary("kglite")
SKIP_REASON = binary_skip_reason("kglite shell binary", BINARY, "cargo build -p kglite-cli")

pytestmark = pytest.mark.skipif(SKIP_REASON is not None, reason=SKIP_REASON or "")


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


def _run_proc(script: str) -> subprocess.CompletedProcess[str]:
    """Feed `script` to the shell on stdin, return the full process (exit code
    and streams kept apart — the piped-shell contract is about both)."""
    return subprocess.run(
        [str(BINARY)],
        input=script,
        capture_output=True,
        text=True,
        timeout=30,
    )


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


def test_query_subcommand_timeout_ms_bounds_a_runaway_query(tmp_path):
    """`--timeout-ms` is the CLI's only deadline. There is deliberately no
    default (see `docs/operators/cli.md`): Ctrl-C is the interactive cancel and
    a batch query over a very large graph may legitimately run for hours, so a
    silent three-minute kill would be a regression. But a caller who *wants* a
    bound had no way to ask for one."""
    import kglite

    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Person {name: 'Alice'})")
    p = tmp_path / "g.kgl"
    g.save(str(p))

    proc = _run_args_proc(
        "query",
        str(p),
        "UNWIND range(1, 400000) AS a UNWIND range(1, 400000) AS b RETURN count(a + b) AS n",
        "--timeout-ms",
        "50",
    )
    assert proc.returncode != 0
    assert "timed out" in (proc.stdout + proc.stderr)


def test_query_subcommand_has_no_deadline_without_the_flag(tmp_path):
    """The declared absence, asserted rather than assumed: the same query the
    test above kills at 50 ms is still running well past the 180 s another
    surface would have stopped it at — so this asserts the *flag's* absence
    means no bound, via a `--timeout-ms 0` control and the help text, without
    waiting three minutes to prove it."""
    import kglite

    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Person {name: 'Alice'})")
    p = tmp_path / "g.kgl"
    g.save(str(p))

    for argv in ([], ["--timeout-ms", "0"]):
        proc = _run_args_proc("query", str(p), "MATCH (p:Person) RETURN p.name AS name", *argv)
        assert proc.returncode == 0, proc.stderr
        assert "Alice" in proc.stdout

    help_text = _run_args("query", "--help")
    assert "--timeout-ms" in help_text
    assert "Omitted means no deadline" in help_text, help_text


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
    # The sibling lock file persists so every contender locks the same inode;
    # only the OS advisory lock is released. The human-readable holder record
    # lives in the `.lock-owner` sidecar; its text is diagnostic, not liveness.
    # Both writers have exited cleanly, so the last one to hold the lease
    # stamped its own record with the moment it gave it back — a reader of the
    # sidecar sees a released lease rather than a stale "held since".
    assert (tmp_path / "shared.kgl.lock").exists()
    owner_record = (tmp_path / "shared.kgl.lock-owner").read_text(encoding="utf-8")
    assert owner_record.startswith("pid="), owner_record
    assert "released=" in owner_record, owner_record


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


def test_session_describe_accepts_detail_objects(tmp_path):
    import json

    import kglite

    p = tmp_path / "detail-objects.kgl"
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Task {id: 'a'})")
    g.cypher("CREATE (:Task {id: 'b'})")
    g.cypher("""
        MATCH (a:Task {id: 'a'}), (b:Task {id: 'b'})
        CREATE (a)-[:DEPENDS_ON]->(b)
    """)
    g.save(str(p))
    requests = "\n".join(
        [
            json.dumps(
                {
                    "id": "overview",
                    "op": "describe",
                    "connections": {"detail": "overview"},
                }
            ),
            json.dumps(
                {
                    "id": "topic",
                    "op": "describe",
                    "connections": {"types": ["DEPENDS_ON"]},
                }
            ),
            json.dumps({"id": "cypher", "op": "describe", "cypher": {"detail": "overview"}}),
            json.dumps({"id": "exit", "op": "exit"}),
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
    assert [r["id"] for r in responses] == ["overview", "topic", "cypher", "exit"]
    assert '<conn type="DEPENDS_ON" count="1"' in responses[0]["description"]
    assert '<pair from="Task" to="Task" count="1"/>' in responses[1]["description"]
    assert "<cypher>" in responses[2]["description"]


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


def test_mode_csv_preserves_scalar_and_nested_precision():
    out = _run(
        ".mode csv\n"
        "RETURN 1.23456789 AS f, "
        "[1.23456789, datetime('2024-01-15T10:30:00.123456789')] AS nested, "
        "datetime('2024-01-15T10:30:00.123456789') AS stamp;\n"
        ".quit\n"
    )
    assert "f,nested,stamp" in out
    assert "1.23456789" in out
    assert '[1.23456789, ""2024-01-15T10:30:00.123456789""]' in out
    assert out.count("2024-01-15T10:30:00.123456789") == 2


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
    csv.write_text("id,name,age\n1,Alice,30\n2,Bob,25\n", encoding="utf-8")
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
    csv.write_text("id\n1\n", encoding="utf-8")
    out = _run(f".import {csv} 9bad\n.quit\n")
    assert "not a valid node type" in out


def test_read_runs_a_cypher_file(tmp_path):
    script = tmp_path / "seed.cypher"
    script.write_text("CREATE (:Person {name: 'Alice'});\nCREATE (:Person {name: 'Bob'});\n", encoding="utf-8")
    out = _run(f".read {script}\nMATCH (p:Person) RETURN count(p) AS n;\n.quit\n")
    assert "(1 row)" in out
    assert "2" in out  # the count after seeding


def test_cdc_stream_is_readable_from_the_shell():
    """The shell's commit boundary is the statement, and the change stream has
    to see it.

    A binding publishes CDC events by draining the capture buffer where it
    knows a commit happened; the engine cannot know that for it. Without that
    call in the shell's execute path, every one of these statements succeeds
    and `db.cdc.query()` still reports zero rows — a stream that is silently
    empty rather than wrong. This is the test that can tell the difference.
    """
    out = _run(
        "CALL db.cdc.enable();\n"
        "CREATE (:Person {id: 1, name: 'Alice'});\n"
        "MATCH (p:Person {id: 1}) SET p.name = 'Alicia';\n"
        "MATCH (p:Person {id: 1}) DELETE p;\n"
        "CALL db.cdc.query() YIELD seq, operation, nodeType, nodeId;\n"
        ".quit\n"
    )
    assert "65536" in out, f"enable did not report its capacity: {out}"
    assert "(3 rows)" in out, f"expected create/update/delete in the stream: {out}"
    for operation in ('"create"', '"update"', '"delete"'):
        assert operation in out, f"{operation} missing from the stream: {out}"


def test_cdc_is_off_until_enabled():
    """Capture is opt-in — reading before `enable` explains that rather than
    reporting an empty stream."""
    out = _run("CREATE (:Person {id: 1});\nCALL db.cdc.query();\n.quit\n")
    assert "not enabled on this graph" in out


# --- piped (non-interactive) shell ---------------------------------------
#
# stdin is not a terminal here, which is the whole point: rustyline's non-TTY
# fallback accumulated continuation lines into a buffer it discarded at EOF, so
# a script whose last statement lacked `;` ran nothing and exited 0. The shell
# reads stdin itself in that case (`repl::run_piped`), sharing the prompt's
# termination rule via `helper::is_terminated`.
#
# The terminal path cannot be exercised from pytest (no portable pty here);
# it is covered by the Rust unit tests over the shared rule in `helper.rs`.


def test_piped_final_statement_without_semicolon_runs():
    """The eval repro: a final statement with no `;`, then `.quit`.

    Before the fix this printed nothing at all and exited 0 — the query and
    the `.quit` both vanished into rustyline's discarded buffer.
    """
    proc = _run_proc("CREATE (:Person {name: 'Alice'});\nMATCH (p:Person) RETURN p.name AS name\n.quit\n")
    assert proc.returncode == 0, proc.stderr
    assert "Alice" in proc.stdout, proc.stdout
    assert "(1 row)" in proc.stdout, proc.stdout


def test_piped_statement_without_semicolon_at_eof_runs():
    """No `.quit` either — end of input terminates a balanced statement."""
    proc = _run_proc("RETURN 1 AS one\n")
    assert proc.returncode == 0, proc.stderr
    assert "(1 row)" in proc.stdout, proc.stdout


def test_piped_trailing_semicolon_behaviour_is_unchanged():
    proc = _run_proc("RETURN 1 AS one;\n.quit\n")
    assert proc.returncode == 0, proc.stderr
    assert "(1 row)" in proc.stdout, proc.stdout


def test_piped_multiline_statement_without_semicolon_runs_once():
    """Continuation lines still accumulate; EOF closes the statement."""
    proc = _run_proc("MATCH (n)\nRETURN count(n)\nAS c\n")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.count("(1 row)") == 1, proc.stdout


def test_piped_unterminated_tail_warns_and_exits_nonzero():
    """An unclosed quote is a truncated script, not a statement: nothing runs
    for it, the tail is named on stderr, and the exit code is non-zero."""
    proc = _run_proc("RETURN 1 AS one;\nRETURN 'oops\n")
    assert proc.returncode != 0
    assert "unterminated" in proc.stderr, proc.stderr
    assert "RETURN 'oops" in proc.stderr, proc.stderr
    # The complete statement before it still ran.
    assert "(1 row)" in proc.stdout, proc.stdout


def test_piped_unclosed_bracket_is_not_executed():
    proc = _run_proc("MATCH (n RETURN n\n")
    assert proc.returncode != 0
    assert "unterminated" in proc.stderr, proc.stderr


def test_piped_empty_stdin_is_a_clean_exit():
    proc = _run_proc("")
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == "", proc.stderr


def test_piped_blank_lines_are_ignored():
    proc = _run_proc("\n\n   \nRETURN 1 AS one;\n\n")
    assert proc.returncode == 0, proc.stderr
    assert "(1 row)" in proc.stdout, proc.stdout


def test_piped_dot_command_after_unterminated_statement_still_runs():
    """`.schema` is a shell command, never Cypher: it flushes the balanced
    statement waiting for its `;` instead of being swallowed by it."""
    proc = _run_proc("CREATE (:Person {name: 'A'});\nMATCH (p:Person) RETURN p.name AS name\n.schema\n.quit\n")
    assert proc.returncode == 0, proc.stderr
    assert "A" in proc.stdout
    assert "Person" in proc.stdout, proc.stdout


def test_piped_table_output_is_not_width_truncated():
    """Piped table output is data, not a screen: no `…`, values in full.

    `.schema`'s property list is the eval's repro — it lost its tail to the
    60-char cell cap even though nothing was reading a terminal.
    """
    long_value = "x" * 100
    proc = _run_proc(
        f"CREATE (:Person {{name: '{long_value}', city: 'Oslo', country: 'NO', team: 'Core'}});\n"
        "MATCH (p:Person) RETURN p.name AS name;\n"
        ".schema\n"
        ".quit\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert long_value in proc.stdout, proc.stdout
    assert "\u2026" not in proc.stdout, proc.stdout
    # The schema row keeps every property name, including the last one.
    assert "country" in proc.stdout, proc.stdout


def _session(graph_path, *requests: dict) -> list[dict]:
    """Drive `kglite session` with JSONL requests, return parsed responses."""
    import json

    payload = "".join(json.dumps(r) + "\n" for r in requests)
    proc = subprocess.run(
        [str(BINARY), "session", str(graph_path), "--format", "json"],
        input=payload,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return [json.loads(line) for line in proc.stdout.splitlines()]


def test_session_help_op_lists_the_protocol(tmp_path):
    """`{"op":"help"}` — the op set is discoverable from inside the protocol."""
    import kglite

    p = tmp_path / "help.kgl"
    kglite.KnowledgeGraph().save(str(p))
    responses = _session(p, {"id": "h1", "op": "help"}, {"op": "exit"})
    help_response = responses[0]
    assert help_response["ok"] is True
    assert help_response["op"] == "help"
    assert help_response["id"] == "h1"
    listed = {entry["op"] for entry in help_response["ops"]}
    assert listed == {
        "query",
        "write",
        "response_expand",
        "describe",
        "save",
        "help",
        "exit",
    }
    assert all(entry["description"] for entry in help_response["ops"])
    assert "id" in help_response["protocol"]


def test_session_unknown_op_names_the_valid_ops(tmp_path):
    import kglite

    p = tmp_path / "unknown-op.kgl"
    kglite.KnowledgeGraph().save(str(p))
    responses = _session(p, {"id": "bad", "op": "delete"}, {"op": "exit"})
    error = responses[0]
    assert error["ok"] is False
    assert error["id"] == "bad"
    for op in ("query", "write", "describe", "save", "help", "exit", "quit"):
        assert op in error["error"], error["error"]


def test_session_agent_help_and_validation_precede_mutation(tmp_path, monkeypatch):
    import kglite

    cache = tmp_path / "cache"
    monkeypatch.setenv("KGLITE_AGENT_CACHE_DIR", str(cache))
    p = tmp_path / "session-agent-validation.kgl"
    kglite.KnowledgeGraph().save(str(p))
    responses = _session(
        p,
        {"id": "h", "op": "help"},
        {
            "id": "bad",
            "op": "write",
            "query": "CREATE (:Guard {id: 1})",
            "format": "agent",
            "response": {"max_bytes": 4095},
        },
        {"id": "count", "op": "query", "query": "MATCH (n:Guard) RETURN count(n) AS n"},
        {"op": "exit"},
    )
    assert "response_expand" in {entry["op"] for entry in responses[0]["ops"]}
    assert responses[1]["isError"] is True
    assert "4096" in str(responses[1]["structuredContent"])
    assert responses[2]["rows"] == [{"n": 0}]


def test_session_agent_budget_counts_escaped_echo_id_and_reports_mandatory_overage(tmp_path, monkeypatch):
    import json

    import kglite

    monkeypatch.setenv("KGLITE_AGENT_CACHE_DIR", str(tmp_path / "cache"))
    p = tmp_path / "echo-budget.kgl"
    kglite.KnowledgeGraph().save(str(p))
    fitting_id = {"escaped": '\\"' * 100}
    oversized_id = {"escaped": '\\"' * 5000}
    responses = _session(
        p,
        {
            "id": fitting_id,
            "op": "query",
            "query": f"UNWIND range(0,999) AS i RETURN i, '{'x' * 80}'",
            "format": "agent",
            "response": {"max_bytes": 4096},
        },
        {
            "id": oversized_id,
            "op": "query",
            "query": "RETURN 1",
            "format": "agent",
            "response": {"max_bytes": 4096},
        },
        {"op": "exit"},
    )
    assert responses[0]["id"] == fitting_id
    assert len(json.dumps(responses[0], separators=(",", ":")).encode()) <= 4096
    assert responses[1]["id"] == oversized_id
    assert responses[1]["retention"]["budget_exceeded"] is True
    assert responses[1]["retention"]["max_bytes"] == 4096
    final_size = len(json.dumps(responses[1], separators=(",", ":")).encode())
    assert final_size > 4096
    assert responses[1]["retention"]["actual_bytes"] == final_size


def test_session_legacy_unknown_and_nonstring_formats_keep_default_fallback(tmp_path):
    import kglite

    p = tmp_path / "legacy-format.kgl"
    kglite.KnowledgeGraph().save(str(p))
    responses = _session(
        p,
        {"id": "unknown", "op": "query", "query": "RETURN 1 AS n", "format": "future"},
        {"id": "object", "op": "query", "query": "RETURN 2 AS n", "format": {}},
        {"op": "exit"},
    )
    assert responses[0]["rows"] == [{"n": 1}]
    assert responses[1]["rows"] == [{"n": 2}]


def test_session_agent_expands_retained_snapshot_without_replay(tmp_path, monkeypatch):
    import json

    import kglite

    cache = tmp_path / "cache"
    monkeypatch.setenv("KGLITE_AGENT_CACHE_DIR", str(cache))
    p = tmp_path / "session-agent.kgl"
    graph = kglite.KnowledgeGraph()
    body = "x" * 80
    graph.cypher(f"UNWIND range(0, 999) AS i CREATE (:Snapshot {{id:i, body:'{body}'}})")
    graph.save(str(p))

    first = _session(
        p,
        {
            "id": "preview",
            "op": "query",
            "query": "MATCH (n:Snapshot) RETURN n.id, {nested:{body:n.body}} ORDER BY n.id",
            "format": "agent",
            "response": {"max_bytes": 4096},
        },
        {"op": "exit"},
    )[0]
    assert first["id"] == "preview"
    assert first["op"] == "query"
    assert first["isError"] is False
    assert len(json.dumps(first, separators=(",", ":")).encode()) <= 4096
    budget = json.loads(first["content"][0]["text"])["response_budget"]
    handle = budget["result_id"]

    changed = kglite.load(str(p))
    changed.cypher("MATCH (n:Snapshot {id:999}) SET n.body = 'changed'")
    changed.save(str(p))
    second = _session(
        p,
        {
            "id": "expand",
            "op": "response_expand",
            "handle": handle,
            "path": "/rows/999/1/nested/body",
            "response": {"mode": "full"},
        },
        {"op": "exit"},
    )[0]
    assert second["id"] == "expand"
    assert second["op"] == "response_expand"
    assert second["isError"] is False
    assert json.loads(second["content"][0]["text"]) == body


def test_session_agent_write_expands_once_only_result_after_live_state_changes(tmp_path, monkeypatch):
    import json

    import kglite

    monkeypatch.setenv("KGLITE_AGENT_CACHE_DIR", str(tmp_path / "cache"))
    p = tmp_path / "session-agent-write.kgl"
    kglite.KnowledgeGraph().save(str(p))
    body = "original-" + "x" * 80
    first = _session(
        p,
        {
            "id": "write",
            "op": "write",
            "query": (
                f"UNWIND range(0,47) AS i CREATE (n:OnceOnly {{id:i, body:'{body}'}}) RETURN i, n.body ORDER BY i"
            ),
            "format": "agent",
            "response": {"max_bytes": 4096},
        },
        {"op": "save"},
        {"op": "exit"},
    )[0]
    assert first["isError"] is False
    budget = json.loads(first["content"][0]["text"])["response_budget"]

    second = _session(
        p,
        {
            "op": "write",
            "query": "MATCH (n:OnceOnly {id:47}) SET n.body = 'changed'",
        },
        {"op": "save"},
        {
            "id": "old",
            "op": "response_expand",
            "handle": budget["result_id"],
            "path": "/rows/47/0",
            "response": {"mode": "full"},
        },
        {
            "id": "state",
            "op": "query",
            "query": (
                "MATCH (n:OnceOnly) RETURN count(n) AS n, "
                "sum(CASE WHEN n.body = 'changed' THEN 1 ELSE 0 END) AS changed"
            ),
        },
        {"op": "exit"},
    )
    assert json.loads(second[2]["content"][0]["text"]) == 47
    assert second[3]["rows"] == [{"n": 48, "changed": 1}]


def test_session_agent_unknown_handle_is_structured_and_session_continues(tmp_path, monkeypatch):
    import kglite

    monkeypatch.setenv("KGLITE_AGENT_CACHE_DIR", str(tmp_path / "cache"))
    p = tmp_path / "missing-handle.kgl"
    kglite.KnowledgeGraph().save(str(p))
    responses = _session(
        p,
        {"id": "x", "op": "response_expand", "handle": "does-not-exist"},
        {"id": "q", "op": "query", "query": "RETURN 1 AS n"},
        {"op": "exit"},
    )
    assert responses[0]["isError"] is True
    assert responses[0]["op"] == "response_expand"
    assert responses[0]["id"] == "x"
    assert responses[1]["rows"] == [{"n": 1}]
