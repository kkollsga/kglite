"""Bolt v5 wire-protocol smoke tests against `target/release/kglite-bolt-server`.

These 8 tests are the **failing-by-design contract** for Phase C of
`bolt_implementation.md`. Each is `pytest.mark.xfail(strict=True)` and
tagged with the Phase C sub-phase that retires it. Strict mode does two
things:

1. **Every test must fail in Phase B.** The bolt-server binary boots and
   binds a port, but every `BoltBackend` method body is
   `unimplemented!("phase C.X ...")` — so the first real Bolt message
   panics the connection task and the driver sees a broken pipe. The
   driver's `verify_connectivity` raises, the test catches via `xfail`,
   pytest reports XFAIL, the suite is green.

2. **Each test must turn green on exactly its retiring sub-phase.** If
   Phase C.2 (RUN/PULL scalars) accidentally fixes test #4 (Node return)
   before Phase C.4 lands, strict mode flips XFAIL → XPASS → fail, and
   CI alerts the maintainer to the surprise regression-in-reverse.

The suite is gated three ways:

- `pytest.importorskip("neo4j")` — silent skip if the driver isn't
  installed (the `[neo4j]` extra is opt-in for the conformance runner).
- `pytestmark = pytest.mark.skipif(not BINARY.exists(), ...)` — silent
  skip if the release binary hasn't been built (the CI binary-build step
  is what makes this active).
- `pytest.mark.bolt` — excluded from the default test run via
  `pyproject.toml` `addopts`; opt in via `pytest -m bolt`.
"""

import pytest

neo4j = pytest.importorskip("neo4j")

# Fixtures `bolt_server` + `bolt_server_readonly` and the spawn helpers
# live in `tests/conftest.py` since the robustness pass — they're shared
# across all `tests/test_bolt_server_*.py` files.

pytestmark = [pytest.mark.bolt]


# ── The 8 tests — each xfail(strict=True), tagged to its retiring sub-phase ──


def test_bolt_handshake_and_verify_connectivity(bolt_server):
    """Phase C.1 ✓: HELLO/LOGON/GOODBYE + a `verify_connectivity()` ping.

    The backend's `create_session` / `get_server_info` / `close_session`
    + the auto-no-op `set_session_auth` (boltr skips it when no
    `AuthValidator` is wired) are enough to satisfy the neo4j Python
    driver's handshake. Driver opens a connection, sends HELLO + LOGON,
    receives SUCCESS for both, sends GOODBYE, closes. No queries run.
    """
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        driver.verify_connectivity()


def test_bolt_run_returns_scalar_rows(bolt_server):
    """Phase C.2 ✓: RUN a trivial scalar query, PULL all, check the rows."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            result = session.run("MATCH (n:Person) RETURN n.title AS name ORDER BY name")
            names = [record["name"] for record in result]
    assert names == ["Alice", "Bob", "Carol", "Dave"]


def test_bolt_run_supports_parameters(bolt_server):
    """Phase C.3 ✓: RUN with `$param` map decoded from PackStream."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            result = session.run(
                "MATCH (n:Person {city: $c}) RETURN n.title AS name ORDER BY name",
                c="Oslo",
            )
            names = [record["name"] for record in result]
    assert names == ["Alice", "Carol"]


def test_bolt_return_node_yields_node_struct(bolt_server):
    """Phase C.4 ✓: `RETURN n` maps Value::Node → BoltNode PackStream struct (0x4E)."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            result = session.run("MATCH (n:Person {title: 'Alice'}) RETURN n")
            record = result.single()
            assert record is not None
            node = record["n"]
            assert isinstance(node, neo4j.graph.Node)
            assert "Person" in node.labels
            assert node["title"] == "Alice"
            assert node["city"] == "Oslo"


def test_bolt_return_relationship_yields_rel_struct(bolt_server):
    """Phase C.4 ✓: `RETURN r` maps Value::Relationship → BoltRelationship (0x52)."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            result = session.run("MATCH (:Person {title: 'Alice'})-[r:KNOWS]->(:Person {title: 'Bob'}) RETURN r")
            record = result.single()
            assert record is not None
            rel = record["r"]
            assert isinstance(rel, neo4j.graph.Relationship)
            assert rel.type == "KNOWS"


def test_bolt_transaction_commit_and_rollback(bolt_server):
    """Phase C.5 ✓: explicit `tx.run()` + `tx.commit()` / `tx.rollback()`."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            # Commit a mutation, verify it's visible.
            tx = session.begin_transaction()
            tx.run("CREATE (:Person {id: 99, title: 'Eve', city: 'Stavanger'})")
            tx.commit()
            after_commit = session.run("MATCH (n:Person {title: 'Eve'}) RETURN count(n) AS c").single()["c"]
            assert after_commit == 1

            # Rollback another mutation, verify it's discarded.
            tx2 = session.begin_transaction()
            tx2.run("CREATE (:Person {id: 100, title: 'Frank', city: 'Tromsø'})")
            tx2.rollback()
            after_rb = session.run("MATCH (n:Person {title: 'Frank'}) RETURN count(n) AS c").single()["c"]
            assert after_rb == 0


def test_bolt_rejects_writes_when_readonly(bolt_server_readonly):
    """Phase C.5 ✓: `--readonly` flag rejects mutations with a Bolt FAILURE.

    Uses its own readonly fixture (the default `bolt_server` fixture is
    read-write). The CREATE attempts — both auto-commit and explicit-tx —
    should fail because the server is `--readonly`.
    """
    with neo4j.GraphDatabase.driver(bolt_server_readonly, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            # Auto-commit CREATE: rejected at the execute() boundary.
            with pytest.raises(neo4j.exceptions.ClientError):
                session.run("CREATE (:Person {id: 999, title: 'Bad'})").consume()
            # Explicit BEGIN: rejected at begin_transaction.
            with pytest.raises(neo4j.exceptions.ClientError):
                session.begin_transaction()


def test_bolt_returns_failure_on_parse_error(bolt_server):
    """Phase C.6 ✓: a syntactically invalid Cypher returns Bolt FAILURE
    with a `Neo.ClientError.Statement.SyntaxError` code (the canonical
    Neo4j status code for this case), driven by the KgErrorCode
    enum that Phase A.2 shipped + the kg_to_bolt mapper that Phase C.6
    added."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            with pytest.raises(neo4j.exceptions.ClientError) as exc_info:
                session.run("MATCH NOT VALID CYPHER").consume()
            assert "Syntax" in str(exc_info.value.code)
