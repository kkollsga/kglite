"""`kglite.trim_memory()` — the opt-in allocator trim.

The behavioural test measures the *process footprint*, not RSS. mimalloc's
decommit on macOS is `MADV_FREE_REUSABLE`, which returns the pages immediately
but leaves them counted in `resident_size` until the kernel reclaims them — so
`ps`, `psutil.rss` and `resource.getrusage` all read flat across a trim that
released hundreds of megabytes. `ri_phys_footprint` (the number Activity
Monitor shows) is what moves. On Linux the decommit is `MADV_DONTNEED` and
ordinary RSS is the right observable.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import os
import struct
import sys
import threading

import pytest

import kglite

# 200k nodes with a padded string property: a few hundred MB of Rust-side
# allocation, ~1 s to build. Large enough that the trim is unambiguous against
# whatever else the test session leaves lying around.
TRANSIENT_NODES = 200_000
INGEST = (
    "UNWIND range(1, %d) AS i "
    "CREATE (:Transient {id: i, name: 'node-' + toString(i), "
    "pad: 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'})" % TRANSIENT_NODES
)


def _footprint_bytes() -> int:
    """Memory this process is charged for, in bytes, or raise if unavailable."""
    if sys.platform == "darwin":
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        buf = ctypes.create_string_buffer(512)
        # proc_pid_rusage(pid, RUSAGE_INFO_V0, &buf). rusage_info_v0 is a
        # 16-byte uuid followed by uint64s; ri_phys_footprint is the 8th.
        if libc.proc_pid_rusage(ctypes.c_int(os.getpid()), ctypes.c_int(0), ctypes.byref(buf)) != 0:
            raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
        return struct.unpack_from("=8Q", buf.raw, 16)[7]
    if sys.platform.startswith("linux"):
        with open("/proc/self/statm", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    raise RuntimeError(f"no cheap process-footprint source on {sys.platform}")


requires_footprint = pytest.mark.skipif(
    sys.platform != "darwin" and not sys.platform.startswith("linux"),
    reason="process-footprint measurement is implemented for macOS and Linux only",
)


def test_trim_memory_is_callable_before_any_graph_exists():
    # The allocator is process-global, so the lever must not need a graph.
    assert kglite.trim_memory() is None


def test_trim_memory_is_reentrant_and_thread_safe():
    # Not a proof that the GIL is released — that is a latency property and
    # timing it would be flaky. It does prove the call is safe off the main
    # thread (where mimalloc takes a different path: no abandoned-segment
    # reclaim) and that concurrent callers neither deadlock nor abort.
    errors: list[BaseException] = []

    def hammer() -> None:
        try:
            for _ in range(5):
                assert kglite.trim_memory() is None
        except BaseException as exc:  # pragma: no cover - only on a real failure
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads), "trim_memory() deadlocked"
    assert not errors, errors


@requires_footprint
def test_trim_memory_returns_a_dropped_peak_to_the_os():
    # Settle first: earlier tests in the session have their own retention.
    gc.collect()
    kglite.trim_memory()
    baseline = _footprint_bytes()

    graph = kglite.KnowledgeGraph()
    graph.cypher(INGEST)
    peak = _footprint_bytes()
    transient = peak - baseline
    assert transient > 32 * 1024 * 1024, f"ingest did not build a measurable peak ({transient} bytes)"

    del graph
    gc.collect()
    dropped = _footprint_bytes()

    kglite.trim_memory()
    trimmed = _footprint_bytes()

    released = dropped - trimmed
    # A no-op trim releases nothing and fails here. The bar is deliberately far
    # below what is observed (a 400k-node peak measured 406 MB -> 15 MB, ~96%)
    # so it survives allocator-version and platform variation.
    assert released >= transient // 5, (
        f"trim_memory() released {released} of a {transient}-byte transient "
        f"(baseline={baseline}, peak={peak}, after drop={dropped}, after trim={trimmed})"
    )
