"""Interruptible Cypher: deadline + Ctrl-C (SIGINT) cancellation.

The query deadline and cooperative cancellation share the engine's
pattern-matcher checkpoints (``interrupt_reason`` / ``check_scan_deadline``).
A long unanchored scan with a non-indexable predicate reaches those
checkpoints repeatedly without materializing a large result, so it can be
stopped promptly by either a deadline or a SIGINT-flipped cancel flag.
"""

import os
import signal
import threading
import time

import pytest

import kglite

# Unanchored full scan, non-indexable arithmetic predicate matching ~nothing:
# bounded memory, loops through every node hitting the cancel checkpoint.
SLOW_QUERY = "MATCH (a:N) WHERE a.id % 999983 = 1 AND a.id * 2 > 9000000000000000000 RETURN count(a) AS c"
N_NODES = 3_000_000


def _big_graph():
    g = kglite.KnowledgeGraph()
    g.cypher(f"UNWIND range(0, {N_NODES}) AS i CREATE (:N {{id: i}})")
    return g


def test_deadline_still_raises():
    """A tiny timeout_ms aborts the scan (the deadline checkpoint works)."""
    g = _big_graph()
    t0 = time.time()
    with pytest.raises(Exception):
        g.cypher(SLOW_QUERY, timeout_ms=1)
    # Should bail almost immediately, not run the whole scan.
    assert time.time() - t0 < 1.0


@pytest.mark.skipif(not hasattr(signal, "SIGINT"), reason="POSIX SIGINT only")
def test_ctrl_c_raises_keyboard_interrupt():
    """A SIGINT during a long query surfaces as KeyboardInterrupt, and the
    previous (Python) SIGINT handler is restored afterwards."""
    g = _big_graph()
    prev_handler = signal.getsignal(signal.SIGINT)

    fired = threading.Event()

    def fire():
        time.sleep(0.08)
        if not fired.is_set():
            os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=fire, daemon=True).start()
    try:
        g.cypher(SLOW_QUERY, timeout_ms=0)  # no deadline -> only Ctrl-C stops it
        fired.set()
        pytest.skip("scan finished before the SIGINT landed (host too fast)")
    except KeyboardInterrupt:
        fired.set()

    # Handler restored to what it was before the query.
    assert signal.getsignal(signal.SIGINT) == prev_handler

    # The graph is still usable after an interrupted read (no poisoning).
    assert g.cypher("MATCH (n:N) WHERE n.id = 0 RETURN n.id AS id").to_list() == [{"id": 0}]
