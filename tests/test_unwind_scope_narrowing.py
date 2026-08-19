"""`UNWIND` over a collected list must not be quadratic in memory.

# The defect (fixed by the `narrow_unwind_source` planner pass)

`UNWIND` turns one row into `n`, and each expanded row was a clone of the
source row — which still held the collected list under its own name. So
`WITH collect(n) AS ns UNWIND ns AS m` produced `n` rows each retaining an
`n`-element copy of *identical* data. Measured before the fix, on the
`node_projection_graph` shape:

    500 rows ->   212 MB
  1 000 rows ->   837 MB
  2 000 rows ->  3 334 MB      (4x memory per 2x rows -- textbook quadratic)
 10 000 rows ->  SIGKILL

After the fix the same shape is 5.3 / 6.4 / 8.7 MB.

# Why this test shells out

Peak RSS is measured in a **fresh subprocess per size**. Measuring both sizes
in one process would let the allocator's retained arenas from the first run
absorb the second, understating the delta -- i.e. failing in the *reassuring*
direction, which is exactly the class of instrument bug CLAUDE.md warns about.
A fresh process has nothing to hide behind.

The sizes are 500 and 2 000: 4x the rows, comfortably clear of the SIGKILL
cliff, and fast enough for the suite's 120 s hang ceiling (~3 s total).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from kglite import KnowledgeGraph

STORAGE_MODES = ("memory", "mapped", "disk")

# 4x the rows must not cost more than this many times the memory. Quadratic
# growth would be ~16x; the pre-fix measurement was 15.7x. Linear is ~1.6x.
# 6x sits far from both, so neither a pass nor a failure is a coin flip.
MAX_GROWTH_FACTOR = 6.0

# Absolute ceiling for the larger size. Pre-fix this shape cost 3 334 MB, so a
# regression overshoots by more than an order of magnitude. Generous enough to
# absorb interpreter and pandas overhead on any runner.
MAX_PEAK_DELTA_MB = 250.0

_PROBE = textwrap.dedent(
    """
    import resource, sys
    import pandas as pd
    import kglite

    def peak_mb():
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux kilobytes.
        return v / (1024 * 1024) if sys.platform == "darwin" else v / 1024

    n = int(sys.argv[1])
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {
                "pid": list(range(n)),
                "name": [f"P{i}" for i in range(n)],
                "age": [20 + (i % 60) for i in range(n)],
                "city": [f"city_{i % 100}" for i in range(n)],
            }
        ),
        "Person", "pid", "name",
    )
    before = peak_mb()
    rows = list(g.cypher("MATCH (n:Person) WITH collect(n) AS ns UNWIND ns AS m RETURN m"))
    assert len(rows) == n, f"expected {n} rows, got {len(rows)}"
    print(peak_mb() - before)
    """
)


def _peak_delta_mb(n: int) -> float:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, str(n)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == 0, (
        f"probe at n={n} failed (rc={proc.returncode}); a SIGKILL here is the "
        f"defect itself resurfacing.\nstderr:\n{proc.stderr}"
    )
    return float(proc.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(sys.platform == "win32", reason="peak-RSS probe uses resource.getrusage (POSIX-only)")
def test_collect_unwind_memory_is_not_quadratic():
    """4x the rows must not cost ~16x the memory."""
    small = _peak_delta_mb(500)
    large = _peak_delta_mb(2_000)

    assert large <= MAX_PEAK_DELTA_MB, (
        f"UNWIND over a collected list used {large:.1f} MB at 2 000 rows "
        f"(ceiling {MAX_PEAK_DELTA_MB} MB). Before the narrow_unwind_source "
        f"pass this shape cost 3 334 MB — the quadratic has returned."
    )
    # Guard the ratio against a small-side measurement so close to zero that
    # any large side passes: require a floor before trusting the ratio.
    if small >= 1.0:
        growth = large / small
        assert growth <= MAX_GROWTH_FACTOR, (
            f"4x the rows cost {growth:.1f}x the memory "
            f"({small:.1f} MB -> {large:.1f} MB). Linear is ~1.6x, quadratic "
            f"~16x; the ceiling is {MAX_GROWTH_FACTOR}x."
        )


# ───────────────────────────────────────────────────────────────────────────
# Correctness pin — the optimisation must be invisible
# ───────────────────────────────────────────────────────────────────────────


def _graph(mode: str, tmp_path, n: int = 25) -> KnowledgeGraph:
    if mode == "memory":
        kg = KnowledgeGraph()
    elif mode == "mapped":
        kg = KnowledgeGraph(storage="mapped")
    elif mode == "disk":
        kg = KnowledgeGraph(storage="disk", path=str(tmp_path / "unwind-disk"))
    else:
        raise ValueError(mode)
    kg.add_nodes(
        pd.DataFrame(
            {
                "pid": list(range(n)),
                "name": [f"P{i}" for i in range(n)],
                "age": [20 + i for i in range(n)],
            }
        ),
        "Person",
        "pid",
        "name",
    )
    return kg


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_collect_unwind_round_trips_exact_node_set_and_order(mode, tmp_path):
    """collect() then UNWIND returns the same nodes, in the same order."""
    kg = _graph(mode, tmp_path)
    direct = [r["n"]["properties"]["pid"] for r in kg.cypher("MATCH (n:Person) RETURN n")]
    round_tripped = [
        r["m"]["properties"]["pid"] for r in kg.cypher("MATCH (n:Person) WITH collect(n) AS ns UNWIND ns AS m RETURN m")
    ]
    assert round_tripped == direct, f"mode={mode}: collect+UNWIND changed the node set or order"

    # The whole record survives, not just the id — the source binding is taken
    # by move, so a shallow-copy bug would surface as a truncated record.
    full_direct = list(kg.cypher("MATCH (n:Person) RETURN n"))
    full_round = list(kg.cypher("MATCH (n:Person) WITH collect(n) AS ns UNWIND ns AS m RETURN m"))
    assert [r["n"] for r in full_direct] == [r["m"] for r in full_round], (
        f"mode={mode}: a node's materialised record changed through collect+UNWIND"
    )


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_unwind_source_stays_visible_when_referenced(mode, tmp_path):
    """The pass must NOT narrow when the source list is still read downstream.

    Each of these would silently lose data if `narrow_unwind_source` dropped a
    live binding. They are the guards that keep the optimisation honest.
    """
    kg = _graph(mode, tmp_path, n=5)

    # 1. The list is read by a later RETURN item.
    rows = list(
        kg.cypher("MATCH (n:Person) WITH collect(n.pid) AS ns UNWIND ns AS m RETURN m, size(ns) AS c ORDER BY m")
    )
    assert [r["c"] for r in rows] == [5] * 5, (
        f"mode={mode}: size(ns) lost its list after UNWIND — a live binding was dropped"
    )

    # 2. `RETURN *` names no variable in the AST; the executor expands it from
    #    the runtime row, so the pass must treat it as reading everything.
    star = list(kg.cypher("MATCH (n:Person) WITH collect(n.pid) AS ns UNWIND ns AS m RETURN * ORDER BY m"))
    assert star, f"mode={mode}: RETURN * produced no rows"
    assert "ns" in star[0], f"mode={mode}: RETURN * lost the `ns` column — got {sorted(star[0])}"
    assert star[0]["ns"] == [0, 1, 2, 3, 4]

    # 3. A second UNWIND over the same list needs it intact.
    pairs = list(kg.cypher("MATCH (n:Person) WITH collect(n.pid) AS ns UNWIND ns AS a UNWIND ns AS b RETURN a, b"))
    assert len(pairs) == 25, (
        f"mode={mode}: double UNWIND produced {len(pairs)} rows, expected 25 "
        "(the first UNWIND consumed a list the second still needed)"
    )


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_unwind_narrowing_matches_the_unoptimised_path(mode, tmp_path):
    """Disabling the pass must not change any result.

    The differential corpus covers this globally; this is the local pin, so a
    failure names the pass directly.
    """
    kg = _graph(mode, tmp_path, n=8)
    queries = [
        "MATCH (n:Person) WITH collect(n.pid) AS ns UNWIND ns AS m RETURN m ORDER BY m",
        "MATCH (n:Person) WITH collect(n.pid) AS ns UNWIND ns AS m RETURN m, size(ns) AS c ORDER BY m",
        "MATCH (n:Person) WITH collect(n.pid) AS ns UNWIND ns AS m WITH m WHERE m > 2 RETURN m ORDER BY m",
        "MATCH (n:Person) WITH collect(n.pid) AS ns UNWIND ns AS m RETURN collect(m) AS back",
        "UNWIND [1, 2, 3] AS m RETURN m",
    ]
    for q in queries:
        on = list(kg.cypher(q))
        off = list(kg.cypher(q, disabled_passes=["narrow_unwind_source"]))
        assert on == off, f"mode={mode}: pass changed results for `{q}`\non={on}\noff={off}"
