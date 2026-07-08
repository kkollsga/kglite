"""Canary: kglite must coexist with pyarrow through normal interpreter teardown.

WHAT THIS PROTECTS
------------------
Importing both ``pyarrow`` and ``kglite`` into one CPython process and letting
it run to normal at-exit teardown used to SIGSEGV (exit 139). Root cause: two
statically-linked mimalloc-v3 instances in a single process — kglite's Rust
``#[global_allocator]`` and the copy CPython 3.14 vendors into libarrow —
colliding during thread-heap teardown (``_mi_theap_collect_retired``). Fixed in
commit a2f52032 by pinning kglite's bundled mimalloc to the v2 series, which
coexists cleanly with the v3 copy. See
``dev-docs/plans/dataset-fetch-standardization.md`` (Phase 6 sections) for the
full bisect archaeology.

This file is a permanent regression canary so that class of dual-allocator
teardown crash can never come back silently.

WHY SUBPROCESS
--------------
The crash is a segfault at *interpreter teardown* — exit code 139 (128 + SIGSEGV
11), observable only after ``main()`` returns. We can't catch it in-process; the
signal is the assertion. So each case spawns a fresh interpreter, imports the
two libraries, does the minimal work, and lets it exit. ``returncode == 0`` means
teardown completed cleanly; ``139`` means the allocators clashed.

WHY find_spec (AND NEVER ``import pyarrow`` IN THIS PROCESS)
-----------------------------------------------------------
Importing pyarrow into the pytest runner itself would arm the exact teardown
crash in the test process. A regression would then segfault the *whole suite* at
exit (a single red exit 139 with no attribution) instead of failing this one
test with a readable message. We therefore only ever *detect* pyarrow's presence
via ``importlib.util.find_spec`` and confine every real import to a subprocess.
When pyarrow isn't installed the whole module skips.

WHY NEUTRAL cwd (tmp_path)
--------------------------
Running ``python -c "import kglite ..."`` from the repo root puts cwd on
``sys.path[0]``, so the source-tree ``kglite/`` package shadows the installed
wheel — the snippet would silently test the wrong (uncompiled / dev) build. Each
subprocess runs with ``cwd=tmp_path`` (a neutral dir) so the installed wheel
wins. This shadowing footgun is documented in the plan doc's harness note.
"""

import importlib.util
import subprocess
import sys

import pytest

# Module-level skip when pyarrow is absent. Detection is import-free on purpose
# (see module docstring): find_spec never loads the module, so it can't arm the
# teardown crash in the pytest process.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None,
    reason="pyarrow not installed — coexistence canary requires it to exercise the dual-allocator teardown path",
)

# The bug reference every case cites: pyarrow-24 dual-mimalloc-v3 teardown
# SIGSEGV, surfaced 2026-07-08, fixed by pinning kglite's bundled mimalloc to
# the v2 series (commit a2f52032).
_BUG = "pyarrow-24 dual-mimalloc-v3 teardown SIGSEGV (2026-07-08, fixed by mimalloc v2 pin)"


def _run_snippet(snippet: str, tmp_path) -> subprocess.CompletedProcess:
    """Run a one-liner in a fresh interpreter from a neutral cwd.

    A clean teardown returns 0; the dual-allocator clash is killed by SIGSEGV —
    which ``subprocess`` reports as ``-11`` (POSIX negated signal) and a shell
    shows as ``139`` (128 + 11). Either way it is nonzero.
    """
    return subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(tmp_path),  # neutral cwd so the installed wheel wins, not source tree
        capture_output=True,
        text=True,
        timeout=120,
    )


def _assert_clean(result: subprocess.CompletedProcess, what: str) -> None:
    assert result.returncode == 0, (
        f"{what} exited {result.returncode} "
        f"(-11/139 = SIGSEGV at teardown — regression of {_BUG}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_forward_import_order(tmp_path):
    """pyarrow-then-kglite import + construct a graph must tear down cleanly.

    Regression guard for the pyarrow-24 dual-mimalloc-v3 teardown SIGSEGV
    (2026-07-08, fixed by mimalloc v2 pin). Exit 139 here means the two
    statically-linked mimalloc copies clashed at interpreter teardown.
    """
    result = _run_snippet(
        "import pyarrow; import kglite; kglite.KnowledgeGraph()",
        tmp_path,
    )
    _assert_clean(result, "forward import (pyarrow, then kglite)")


def test_reverse_import_order(tmp_path):
    """kglite-then-pyarrow import + construct a graph must tear down cleanly.

    Regression guard for the pyarrow-24 dual-mimalloc-v3 teardown SIGSEGV
    (2026-07-08, fixed by mimalloc v2 pin). Import order should not matter —
    both copies exist in the process either way — so we assert both directions.
    """
    result = _run_snippet(
        "import kglite; import pyarrow; kglite.KnowledgeGraph()",
        tmp_path,
    )
    _assert_clean(result, "reverse import (kglite, then pyarrow)")


def test_forward_import_with_graph_op(tmp_path):
    """pyarrow + kglite + a trivial graph write must tear down cleanly.

    Regression guard for the pyarrow-24 dual-mimalloc-v3 teardown SIGSEGV
    (2026-07-08, fixed by mimalloc v2 pin). Exercises the allocator on a real
    (if tiny) write — a Cypher CREATE — so allocations touched by the graph
    engine, not just import-time state, are live at teardown.
    """
    result = _run_snippet(
        "import pyarrow; import kglite; g = kglite.KnowledgeGraph(); g.cypher('CREATE (n:Node {id: 1})')",
        tmp_path,
    )
    _assert_clean(result, "forward import + Cypher CREATE")
