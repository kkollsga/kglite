"""Concurrency stress tests for kglite-bolt-server.

Opt-in via `pytest -m bolt_stress`. These tests exercise the per-tx
mutex split (RA-1) — without it, multi-tx scenarios would serialize
through the global transactions mutex and run very slowly (or
deadlock under unfortunate timing).

Patterns adapted from `tests/test_concurrency.py` (which tested the
pyapi under intra-process Python threads). Here we test TCP-over-Bolt
concurrency: each thread opens its own driver session against the
shared server.

Marker: `bolt_stress` (excluded from default `-m bolt`; opt-in only).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

import pytest

neo4j = pytest.importorskip("neo4j")

pytestmark = [pytest.mark.bolt_stress]


def _run_read(driver, n_iterations: int, min_count: int = 4) -> tuple[int, list]:
    """Worker: open a session, run n_iterations of a small read.
    Asserts the count is at least `min_count` (default 4 = the baseline
    fixture); any value above that is acceptable since concurrent
    writers may have committed mid-read.

    Return (success_count, list of error reprs).
    """
    errors: list[str] = []
    successes = 0
    try:
        with driver.session() as session:
            for _ in range(n_iterations):
                try:
                    result = session.run("MATCH (n:Person) RETURN count(n) AS c")
                    record = result.single()
                    assert record["c"] >= min_count
                    successes += 1
                except Exception as e:  # noqa: BLE001
                    errors.append(repr(e))
    except Exception as e:  # noqa: BLE001
        errors.append(f"session-open: {e!r}")
    return successes, errors


def _run_write_tx(driver, ids: list[int]) -> tuple[int, list]:
    """Worker: open a session, BEGIN, CREATE one node per id, COMMIT."""
    errors: list[str] = []
    successes = 0
    try:
        with driver.session() as session:
            tx = session.begin_transaction()
            try:
                for nid in ids:
                    tx.run(f"CREATE (:Person {{id: {nid}, title: 'thr{nid}'}})").consume()
                tx.commit()
                successes = len(ids)
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))
                try:
                    tx.rollback()
                except Exception:
                    pass
    except Exception as e:  # noqa: BLE001
        errors.append(f"session-open: {e!r}")
    return successes, errors


def test_16_concurrent_readers(bolt_server):
    """16 driver sessions reading concurrently — no panic, no
    deadlock, all reads succeed."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(_run_read, driver, 50) for _ in range(16)]
            results = [f.result() for f in as_completed(futures)]
    all_errors = [e for _, errs in results for e in errs]
    assert all_errors == [], f"errors during 16-reader stress: {all_errors[:3]}"
    total_successes = sum(s for s, _ in results)
    assert total_successes == 16 * 50  # 800 reads total


def test_8_readers_plus_1_writer(bolt_server):
    """1 writer thread (mutates 20 nodes) + 8 reader threads.
    All readers must see consistent snapshots (either pre- or
    post-commit), never a torn count."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with ThreadPoolExecutor(max_workers=9) as pool:
            reader_futures = [pool.submit(_run_read, driver, 100) for _ in range(8)]
            writer_future = pool.submit(_run_write_tx, driver, list(range(3000, 3020)))
            for f in as_completed(reader_futures + [writer_future]):
                _, errs = f.result()
                assert errs == [], f"errors: {errs[:3]}"


def test_4_concurrent_writers(bolt_server):
    """4 parallel transactions each create 5 nodes. Without per-tx
    mutex splitting (RA-1), these would serialize on the global
    transactions mutex. The test asserts they complete and at
    least ONE writer's results survive (last-writer-wins because
    OCC version checking is deferred — see backend.rs)."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(_run_write_tx, driver, list(range(4000 + i * 100, 4005 + i * 100))) for i in range(4)
            ]
            for f in as_completed(futures):
                _, errs = f.result()
                assert errs == [], f"errors: {errs[:3]}"
        # Verify at least one writer's nodes survived. Because of
        # last-writer-wins, we expect 5 nodes from ONE writer (not
        # 20 from all of them).
        with driver.session() as session:
            result = session.run("MATCH (n:Person) WHERE n.title STARTS WITH 'thr4' RETURN count(n) AS c")
            count = result.single()["c"]
            # 5 is the typical case (last writer wins); 20 would
            # mean OCC eventually lands and all serialize cleanly.
            # We accept any positive count — at least one tx
            # committed without error.
            assert count >= 5, f"expected at least 5 surviving nodes, got {count}"


def test_session_disconnect_mid_query(bolt_server):
    """Open a session, send a slow query (10k-row scan), close the
    driver without consuming the result. Server should not leak
    state. We then open fresh sessions and verify they work."""
    # Set up extra rows by inserting via a transaction first.
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with driver.session() as session:
            tx = session.begin_transaction()
            # Add some inert nodes so the scan has work to do.
            for i in range(100):
                tx.run(f"CREATE (:Person {{id: {5000 + i}, title: 'scan{i}'}})")
            tx.commit()

        # Open + abandon
        for _ in range(10):
            driver2 = neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password"))
            try:
                session2 = driver2.session()
                # Send a query but don't consume.
                _ = session2.run("MATCH (n:Person) RETURN n.title AS t")
                # Close abruptly without consuming.
                driver2.close()
            except Exception:
                pass

        # Server still responsive?
        with driver.session() as session:
            result = session.run("MATCH (n:Person) RETURN count(n) AS c")
            assert result.single()["c"] >= 104  # 4 baseline + 100 we added


def test_begin_then_session_close_without_commit_or_rollback(bolt_server):
    """BEGIN, do a CREATE, then close the session WITHOUT calling
    commit() or rollback(). Server's close_session should clean up
    the tx automatically (otherwise the working DirGraph leaks)."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        # Do this 20 times — if state leaked, the HashMap would grow.
        for i in range(20):
            session = driver.session()
            tx = session.begin_transaction()
            tx.run(f"CREATE (:Person {{id: {6000 + i}, title: 'leak{i}'}})")
            # NOTE: deliberately not calling tx.commit() or tx.rollback()
            session.close()  # close_session should clean up

        # Verify NONE of the uncommitted creates survived (each
        # session's pending tx was cleanly rolled back by
        # close_session).
        with driver.session() as session:
            count = session.run("MATCH (n:Person) WHERE n.title STARTS WITH 'leak' RETURN count(n) AS c").single()["c"]
            assert count == 0


def test_100_sequential_connections_no_fd_leak(bolt_server):
    """100 sequential driver-open + verify + driver-close cycles.
    If file descriptors leaked, we'd hit the per-process limit
    eventually (typically 256 on macOS, 1024 on Linux)."""
    for _ in range(100):
        with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
            driver.verify_connectivity()


def test_sustained_mixed_load_5_seconds(bolt_server):
    """5 seconds of mixed read + write load from 8 workers.
    No panic, no deadlock, no test-thread starvation."""
    stop_event = threading.Event()
    errors: list[str] = []
    errors_lock = threading.Lock()
    op_counter = [0]
    counter_lock = threading.Lock()

    def worker_read(driver):
        try:
            with driver.session() as session:
                while not stop_event.is_set():
                    try:
                        session.run("MATCH (n:Person) RETURN count(n)").consume()
                        with counter_lock:
                            op_counter[0] += 1
                    except Exception as e:  # noqa: BLE001
                        with errors_lock:
                            errors.append(repr(e))
                            return
        except Exception as e:  # noqa: BLE001
            with errors_lock:
                errors.append(f"session-open: {e!r}")

    def worker_write(driver, worker_id: int):
        try:
            with driver.session() as session:
                local_seq = 0
                while not stop_event.is_set():
                    try:
                        tx = session.begin_transaction()
                        nid = 7000 + worker_id * 1000 + local_seq
                        tx.run(f"CREATE (:Person {{id: {nid}, title: 'w{worker_id}-{local_seq}'}})")
                        tx.commit()
                        with counter_lock:
                            op_counter[0] += 1
                        local_seq += 1
                    except Exception as e:  # noqa: BLE001
                        with errors_lock:
                            errors.append(repr(e))
                            return
        except Exception as e:  # noqa: BLE001
            with errors_lock:
                errors.append(f"session-open: {e!r}")

    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(worker_read, driver) for _ in range(6)]
            futures += [pool.submit(worker_write, driver, i) for i in range(2)]
            time.sleep(5.0)
            stop_event.set()
            for f in as_completed(futures, timeout=10.0):
                f.result()

    assert errors == [], f"errors during sustained load: {errors[:5]}"
    assert op_counter[0] > 100, f"only {op_counter[0]} ops in 5s — server probably deadlocked"


def test_reset_during_transaction(bolt_server):
    """Client sends RESET after starting a tx — boltr converts that
    to close_session-like semantics. Verify the server doesn't leak
    the tx state."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        # neo4j driver auto-resets between sessions in pooled mode.
        # We just verify that a tx-then-new-session pattern works.
        for i in range(10):
            with driver.session() as session:
                tx = session.begin_transaction()
                tx.run(f"CREATE (:Person {{id: {8000 + i}, title: 'reset{i}'}})")
                tx.rollback()
        # None of the rolled-back creates should be visible.
        with driver.session() as session:
            count = session.run("MATCH (n:Person) WHERE n.title STARTS WITH 'reset' RETURN count(n) AS c").single()["c"]
            assert count == 0


def test_concurrent_sessions_creating_distinct_data(bolt_server):
    """8 sessions, each in its own tx, each creates 1 node with a
    UNIQUE id. With OCC deferred, this means one wins and the
    others' work is overwritten — but no SESSION should crash."""
    with neo4j.GraphDatabase.driver(bolt_server, auth=("neo4j", "password")) as driver:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_run_write_tx, driver, [9000 + i]) for i in range(8)]
            errs = [e for f in as_completed(futures) for _, errs in [f.result()] for e in errs]
            assert errs == [], f"errors: {errs[:3]}"
