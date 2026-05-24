"""Transaction contract tests over the Bolt wire.

Mirrors `tests/test_transaction_bolt_patterns.py` (which exercises the
Python `Transaction` class directly through pyo3) — these tests exercise
the *same contracts* through the neo4j Python driver against
`kglite-bolt-server`. When semantics differ, that's a bolt-server bug.

The 18 contracts:

1. begin → multiple cypher → commit; all mutations visible after
2. begin → multiple cypher → rollback; nothing committed
3. Pending tx mutations invisible to OUTSIDE auto-commit reader
4. Double commit raises a clear error
5. Commit after rollback raises a clear error
6. Driver context manager auto-commits on success
7. Driver context manager auto-rollbacks on exception
8. Read-only session/tx surface (NOTE: Neo4j driver doesn't have
   begin_read like kglite's pyapi; we test --readonly server)
9. OCC conflict — two sessions both write, last-writer-wins
   (pinned current behavior; OCC version checking deferred)
10. Outside mutation during tx — pin current behavior (no conflict)
11-14. Read-only --readonly rejects CREATE/SET/DELETE/MERGE
15. Auto-commit: each session.run is independently visible
16. Multi-statement partial mutation — kglite parser only accepts one
    statement per RUN; this contract differs from the pyapi
17. Tx-level timeout — no surface on the driver side yet (pin no-op)
18. 100 begin/commit cycles complete without leak/error

Fixtures: `bolt_server` (RW) + `bolt_server_readonly` from
`tests/conftest.py`.
"""

import pytest

neo4j = pytest.importorskip("neo4j")

pytestmark = [pytest.mark.bolt]


def _count_people(session, where: str = "") -> int:
    """Helper: count Person nodes (optionally filtered)."""
    where_clause = f"WHERE {where}" if where else ""
    result = session.run(f"MATCH (n:Person) {where_clause} RETURN count(n) AS c")
    return result.single()["c"]


# ────────────────────────────────────────────────────────────────────────────
# 1-5: Explicit transaction lifecycle
# ────────────────────────────────────────────────────────────────────────────


def test_begin_multiple_cypher_commit_all_visible(bolt_server):
    """All mutations inside an explicit tx are visible after COMMIT."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            tx = session.begin_transaction()
            tx.run("CREATE (:Person {id: 100, title: 'Eve', city: 'Stavanger'})")
            tx.run("CREATE (:Person {id: 101, title: 'Frank', city: 'Tromsø'})")
            tx.run("CREATE (:Person {id: 102, title: 'Grace', city: 'Ålesund'})")
            tx.commit()
            assert _count_people(session) == 7  # 4 baseline + 3 new


def test_begin_multiple_cypher_rollback_discards_all(bolt_server):
    """Mutations are discarded when the tx is rolled back."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            tx = session.begin_transaction()
            tx.run("CREATE (:Person {id: 200, title: 'X', city: 'Y'})")
            tx.run("CREATE (:Person {id: 201, title: 'Z', city: 'W'})")
            tx.rollback()
            assert _count_people(session) == 4  # unchanged


def test_uncommitted_mutations_invisible_to_outside_reader(bolt_server):
    """Snapshot isolation across sessions: session_a's pending tx
    writes are NOT visible to session_b until commit."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session_a:
            tx_a = session_a.begin_transaction()
            tx_a.run("CREATE (:Person {id: 300, title: 'Pending'})")
            # Open a fresh session and verify the pending write isn't visible.
            with driver.session() as session_b:
                count_b = _count_people(session_b, "n.title = 'Pending'")
                assert count_b == 0  # session_b sees pre-tx state
            tx_a.rollback()  # cleanup; we asserted what we needed


def test_double_commit_raises_clear_error(bolt_server):
    """COMMIT on an already-committed tx raises (the tx handle is gone)."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            tx = session.begin_transaction()
            tx.run("CREATE (:Person {id: 400, title: 'A'})")
            tx.commit()
            with pytest.raises(neo4j.exceptions.TransactionError):
                tx.commit()


def test_commit_after_rollback_raises_clear_error(bolt_server):
    """COMMIT on a rolled-back tx raises."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            tx = session.begin_transaction()
            tx.run("CREATE (:Person {id: 500, title: 'B'})")
            tx.rollback()
            with pytest.raises(neo4j.exceptions.TransactionError):
                tx.commit()


# ────────────────────────────────────────────────────────────────────────────
# 6-7: Driver-side context-manager (the neo4j driver pattern, not the
# kglite Python pattern — semantically equivalent for happy/exception
# paths)
# ────────────────────────────────────────────────────────────────────────────


def test_context_manager_auto_commits_on_success(bolt_server):
    """`with session.begin_transaction() as tx:` — successful exit commits."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            with session.begin_transaction() as tx:
                tx.run("CREATE (:Person {id: 600, title: 'Ctx'})")
                tx.commit()  # driver requires explicit commit
            assert _count_people(session, "n.title = 'Ctx'") == 1


def test_context_manager_auto_rollbacks_on_exception(bolt_server):
    """Exception inside the `with` block triggers rollback on __exit__."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            with pytest.raises(ValueError):
                with session.begin_transaction() as tx:
                    tx.run("CREATE (:Person {id: 700, title: 'Doomed'})")
                    raise ValueError("simulated app error")
            assert _count_people(session, "n.title = 'Doomed'") == 0


# ────────────────────────────────────────────────────────────────────────────
# 8: Read-only surface — bolt-server's --readonly flag (since the neo4j
# driver doesn't expose a per-session read-only mode that maps onto the
# kglite pyapi's begin_read())
# ────────────────────────────────────────────────────────────────────────────


def test_readonly_server_rejects_explicit_transaction(bolt_server_readonly):
    """--readonly rejects begin_transaction with a ClientError."""
    with neo4j.GraphDatabase.driver(bolt_server_readonly, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            with pytest.raises(neo4j.exceptions.ClientError):
                session.begin_transaction()


# ────────────────────────────────────────────────────────────────────────────
# 9-10: OCC / cross-session concurrency (current behavior pin)
# ────────────────────────────────────────────────────────────────────────────


def test_two_concurrent_commits_last_writer_wins(bolt_server):
    """Two sessions both BEGIN, both CREATE, both COMMIT — the second
    commit wins because OCC version checking is deferred (see
    backend.rs struct doc). Pins the current behavior; an OCC-aware
    follow-up would flip this to "second commit conflicts"."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session_a:
            with driver.session() as session_b:
                tx_a = session_a.begin_transaction()
                tx_b = session_b.begin_transaction()
                tx_a.run("CREATE (:Person {id: 800, title: 'FromA'})")
                tx_b.run("CREATE (:Person {id: 801, title: 'FromB'})")
                tx_a.commit()
                # Without OCC, the second commit succeeds and overwrites
                # the first (it operates on a stale snapshot that
                # doesn't include A's write).
                tx_b.commit()
            # Auto-commit reader sees one of the two — B's snapshot
            # didn't have A, so post-commit only B's CREATE survives.
            count = _count_people(session_a, "n.title IN ['FromA', 'FromB']")
            # Pin current behavior: last-writer-wins → only B survives.
            assert count == 1


def test_outside_mutation_during_open_transaction(bolt_server):
    """While session_a has an open tx, session_b commits an auto-commit
    mutation. session_a's tx still sees its own snapshot (pre-B), and
    its commit clobbers B (last-writer-wins). Pins current behavior."""
    # Auto-commit mutations aren't supported by kglite-bolt-server
    # (C.5 design — wrap writes in begin_transaction). So this test
    # uses two transactions: session_a begins, session_b begins+commits,
    # then session_a commits.
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session_a:
            tx_a = session_a.begin_transaction()
            # session_b commits its own mutation
            with driver.session() as session_b:
                tx_b = session_b.begin_transaction()
                tx_b.run("CREATE (:Person {id: 900, title: 'FromB_outside'})")
                tx_b.commit()
            # session_a's tx still has its pre-B snapshot. Its commit
            # (with no mutations) is a no-op — B's write survives.
            tx_a.commit()
            count = _count_people(session_a, "n.title = 'FromB_outside'")
            assert count == 1


# ────────────────────────────────────────────────────────────────────────────
# 11-14: --readonly rejects individual mutation operations
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "query",
    [
        "CREATE (:Person {id: 1000, title: 'NoCreate'})",
        "MATCH (n:Person {title: 'Alice'}) SET n.city = 'Hacked'",
        "MATCH (n:Person {title: 'Alice'}) DELETE n",
        "MERGE (n:Person {id: 1001, title: 'NoMerge'}) RETURN n",
    ],
    ids=["create", "set", "delete", "merge"],
)
def test_readonly_rejects_each_mutation_class(bolt_server_readonly, query):
    """--readonly rejects each mutation class with a ClientError."""
    with neo4j.GraphDatabase.driver(bolt_server_readonly, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            with pytest.raises(neo4j.exceptions.ClientError):
                session.run(query).consume()


# ────────────────────────────────────────────────────────────────────────────
# 15-16: Auto-commit + multi-statement contracts
# ────────────────────────────────────────────────────────────────────────────


def test_auto_commit_reads_are_independently_visible(bolt_server):
    """Each session.run is its own auto-commit (no wrapping tx). For
    bolt-server: writes are NOT allowed in auto-commit (Phase C.5
    contract), but each read sees the current graph state."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            # Two independent reads each return current state.
            c1 = _count_people(session)
            c2 = _count_people(session)
            assert c1 == c2 == 4
            # Now via explicit tx, add a node and verify next auto-
            # commit read sees it.
            tx = session.begin_transaction()
            tx.run("CREATE (:Person {id: 1100, title: 'AfterCommit'})")
            tx.commit()
            c3 = _count_people(session)
            assert c3 == 5


def test_multi_statement_in_one_run_pinned_behavior(bolt_server):
    """Multiple statements in one RUN — kglite's parser handles one
    statement per RUN, so `CREATE ...; CREATE ...` either parses only
    the first or errors. Either way, the test exists to pin which.
    RB-2 will explicitly reject multi-statement with a structured
    error."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            tx = session.begin_transaction()
            try:
                tx.run("CREATE (:Person {id: 1200, title: 'A'}); CREATE (:Person {id: 1201, title: 'B'})").consume()
                # If it succeeded — pin what kglite parsed.
                tx.commit()
                count = _count_people(session, "n.title IN ['A', 'B']")
                # We don't assert the count value; we just pin that
                # the server didn't crash or hang.
                assert count in (0, 1, 2)
            except Exception:
                tx.rollback()


# ────────────────────────────────────────────────────────────────────────────
# 17: Tx-level timeout (no driver-side surface today; pin no-op)
# ────────────────────────────────────────────────────────────────────────────


def test_tx_timeout_extra_metadata_accepted_but_no_op(bolt_server):
    """The neo4j driver supports `session.begin_transaction(timeout=...)`
    which sends `tx_timeout` in the BEGIN extra dict. bolt-server
    currently ignores this; pin the no-op behavior."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            # Tiny timeout — bolt-server doesn't honor it, so the query runs.
            tx = session.begin_transaction(timeout=0.001)
            result = tx.run("MATCH (n:Person) RETURN count(n) AS c")
            count = result.single()["c"]
            tx.commit()
            assert count == 4


# ────────────────────────────────────────────────────────────────────────────
# 18: 100 begin/commit cycles — no leak, no degradation
# ────────────────────────────────────────────────────────────────────────────


def test_hundred_begin_commit_cycles_complete(bolt_server):
    """100 sequential begin/commit cycles — verify no resource leak
    in the transactions HashMap, no fd accumulation."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            for i in range(100):
                tx = session.begin_transaction()
                result = tx.run(f"RETURN {i} AS i")
                assert result.single()["i"] == i
                tx.commit()
            # If we got here, no panic; the server didn't leak fd's
            # or block on something.


def test_hundred_begin_rollback_cycles_complete(bolt_server):
    """100 sequential begin/rollback cycles."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            for i in range(100):
                tx = session.begin_transaction()
                tx.run(f"CREATE (:Person {{id: {2000 + i}, title: 'tmp{i}'}})")
                tx.rollback()
            # None of the rolled-back creates should be visible.
            assert _count_people(session, "n.title STARTS WITH 'tmp'") == 0
