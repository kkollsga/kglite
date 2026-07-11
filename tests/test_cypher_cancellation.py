"""Interruptible Cypher: deadline + Ctrl-C (SIGINT) cancellation.

The query deadline and cooperative cancellation share the engine's
checkpoints — the pattern matcher's (``interrupt_reason`` /
``check_scan_deadline``) for ``MATCH`` scans, and the graph algorithms'
per-iteration ``Interrupt::exceeded()`` for ``CALL`` procedures. Both a
deadline and a SIGINT-flipped cancel flag abort at the same points.

The SIGINT test runs ``CALL betweenness()`` (O(V·E), reliably multi-second on
the seeded graph) rather than a scan, so a single well-before-completion
signal lands deterministically — no timing-dependent skip.
"""

import os
import signal
import threading
import time

import pytest

import kglite

# Unanchored full scan, non-indexable predicate matching ~nothing: bounded
# memory, hits the scan checkpoint repeatedly. Used for the deadline test.
SCAN_QUERY = "MATCH (a:N) WHERE a.id % 999983 = 1 AND a.id * 2 > 9000000000000000000 RETURN count(a) AS c"

# Betweenness over a modest connected graph is O(V·E) — reliably seconds, so a
# SIGINT fired a fraction of a second in always lands mid-run.
BETWEENNESS_QUERY = "CALL betweenness() YIELD node, score RETURN count(*) AS c"


def _scan_graph(n=3_000_000):
    g = kglite.KnowledgeGraph()
    g.cypher(f"UNWIND range(0, {n}) AS i CREATE (:N {{id: i}})")
    return g


def _algo_graph(n=12_000):
    g = kglite.KnowledgeGraph()
    g.cypher(f"UNWIND range(1, {n}) AS i CREATE (:N {{id: i}})")
    # Cheap anchored edge build (id-index O(1) lookup, O(N) per pass).
    for d in (1, 7, 53, 211):
        g.cypher(f"MATCH (a:N) WITH a, a.id + {d} AS nb MATCH (b:N {{id: nb}}) CREATE (a)-[:R]->(b)")
    return g


def test_scan_deadline_still_raises():
    """A tiny timeout_ms aborts a scan (the pattern-matcher deadline works)."""
    g = _scan_graph()
    t0 = time.time()
    with pytest.raises(Exception):
        g.cypher(SCAN_QUERY, timeout_ms=1)
    assert time.time() - t0 < 1.0  # bailed early, didn't run the whole scan


def test_algorithm_deadline_still_raises():
    """A tiny timeout_ms aborts a CALL algorithm (its iteration checkpoint)."""
    g = _algo_graph()
    with pytest.raises(Exception):
        g.cypher(BETWEENNESS_QUERY, timeout_ms=1)


@pytest.mark.skipif(not hasattr(signal, "SIGINT"), reason="POSIX SIGINT only")
def test_session_mutation_cancel_is_atomic():
    """A `Session.execute` mutation interrupted by Ctrl-C is atomic: the
    transactional working copy is discarded on abort, so the graph is either
    fully mutated (it finished) or unchanged (it was cancelled) — never partial.

    (Live `KnowledgeGraph` / `Transaction` mutations are deliberately NOT
    cancellable — they mutate in place / unreliably roll back — so this
    invariant is only guaranteed for the `Session` path.)
    """
    g = _scan_graph(4_000_000)
    s = g.session()
    flag_all = "MATCH (a:N) SET a.flag = 1"  # matches every node — slow over 4M

    def fire():
        time.sleep(0.08)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=fire, daemon=True).start()
    try:
        s.execute(flag_all, timeout_ms=0)
    except KeyboardInterrupt:
        pass
    else:
        pytest.fail("SIGINT did not interrupt the long-running mutation")

    flagged = s.cypher("MATCH (a:N) WHERE a.flag = 1 RETURN count(a) AS c").to_list()[0]["c"]
    total = s.cypher("MATCH (a:N) RETURN count(a) AS c").to_list()[0]["c"]
    assert flagged == 0, f"cancelled mutation left partial state: {flagged}/{total} flagged"


@pytest.mark.skipif(not hasattr(signal, "SIGINT"), reason="POSIX SIGINT only")
def test_ctrl_c_interrupts_algorithm():
    """A SIGINT during a long CALL algorithm raises KeyboardInterrupt, and the
    previous (Python) SIGINT handler is restored afterwards."""
    g = _algo_graph()
    prev_handler = signal.getsignal(signal.SIGINT)

    def fire():
        time.sleep(0.2)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=fire, daemon=True).start()
    t0 = time.time()
    with pytest.raises(KeyboardInterrupt):
        g.cypher(BETWEENNESS_QUERY, timeout_ms=0)  # no deadline -> only Ctrl-C stops it
    elapsed = time.time() - t0
    assert elapsed < 5.0, f"interrupt was not prompt ({elapsed:.1f}s)"

    # Handler restored, and the graph is still usable after the interrupt.
    assert signal.getsignal(signal.SIGINT) == prev_handler
    assert g.cypher("MATCH (n:N) WHERE n.id = 1 RETURN n.id AS id").to_list() == [{"id": 1}]
