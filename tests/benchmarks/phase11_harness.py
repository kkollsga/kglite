"""Phase 11 N-trial benchmark harness.

Standalone runner — not collected by pytest. Self-contained: builders
and query shapes are inlined below, and each timed cell runs N times,
collecting mean/stddev/p50/p95/p99 per (test, scale, mode) into JSON.

Designed so the same harness runs on any KGLite checkout (0.8.0 main
and v0.7.17 under a worktree), producing comparable JSON artefacts.

Usage
-----
    python tests/benchmarks/phase11_harness.py \
        --json dev-documentation/archive/phase_artifacts/phase11_main.json \
        --n 20

    # v0.7.17 replay: memory + mapped only (disk format incompatible)
    python tests/benchmarks/phase11_harness.py \
        --json dev-documentation/archive/phase_artifacts/phase11_v0_7_17.json \
        --n 20 --modes memory,mapped

    # Quick dry-run
    python tests/benchmarks/phase11_harness.py \
        --json /tmp/phase11_dry.json --n 2

Output shape
------------
    {
      "harness_version": 1,
      "kglite_version": "...",
      "git_sha": "...",
      "n_trials": 20,
      "cells": [
        {
          "name": "construction", "scale": 10000, "mode": "memory",
          "trials": 5, "unit": "s",
          "mean": ..., "stddev": ..., "p50": ..., "p95": ..., "p99": ...,
          "min": ..., "max": ...,
          "build_rss_bytes": ..., "build_dir_bytes": ...
        },
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

# Make tests/benchmarks/ importable.
_BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH_DIR))

import kglite  # noqa: E402

# ─── Constants ──────────────────────────────────────────────────────────────

CONSTRUCTION_SCALES = (100, 1_000, 10_000, 50_000)
QUERY_SCALE = 10_000  # Phase 6 baseline's reference scale
ALL_MODES = ("memory", "mapped", "disk")
BUILD_TRIALS = 5  # fewer trials for build because each is costly at scale


# ─── Utilities ──────────────────────────────────────────────────────────────


def _rss_bytes() -> int | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except ImportError:
        return None


def _dir_size(path: Path | str) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def _percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    if not s:
        return 0.0
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _stats(values: list[float], unit: str) -> dict:
    return {
        "trials": len(values),
        "unit": unit,
        "mean": statistics.mean(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=10", "HEAD"],
            cwd=_BENCH_DIR.parent.parent,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _kglite_version() -> str:
    return getattr(kglite, "__version__", "unknown")


# ─── Build helpers ──────────────────────────────────────────────────────────


def _make_shape(scale: int, edge_factor: int = 3, seed: int = 42):
    """Deterministic graph shape — inlined so the harness works on any
    checkout regardless of which other benchmark files are present."""
    import random as _r

    rng = _r.Random(seed)
    names = [f"N{i}" for i in range(scale)]
    values = [rng.uniform(0, 100) for _ in range(scale)]
    groups = [rng.randint(1, 5) for _ in range(scale)]
    n_edges = edge_factor * scale
    edge_set: set[tuple[int, int]] = set()
    while len(edge_set) < n_edges:
        s = rng.randint(0, scale - 1)
        t = rng.randint(0, scale - 1)
        if s != t:
            edge_set.add((s, t))
    edges = list(edge_set)
    return names, values, groups, edges


def _populate(kg, scale: int, names, values, groups, edges) -> None:
    import pandas as pd

    kg.add_nodes(
        pd.DataFrame({"id": list(range(scale)), "name": names, "value": values, "group": groups}),
        "Node",
        "id",
        "name",
    )
    kg.add_connections(
        pd.DataFrame({"from_id": [s for s, _ in edges], "to_id": [t for _, t in edges]}),
        "EDGE",
        "Node",
        "from_id",
        "Node",
        "to_id",
    )


def _build(scale: int, mode: str, tmpdir: Path):
    """Build a KG for the given scale+mode. Returns (kg, node_names, path)."""
    names, values, groups, edges = _make_shape(scale)
    path: Path | None = None
    if mode == "memory":
        kg = kglite.KnowledgeGraph()
    elif mode == "mapped":
        kg = kglite.KnowledgeGraph(storage="mapped")
    elif mode == "disk":
        path = tmpdir / f"disk_{scale}"
        path.mkdir(exist_ok=True)
        kg = kglite.KnowledgeGraph(storage="disk", path=str(path))
    else:
        raise ValueError(f"unknown mode: {mode}")
    _populate(kg, scale, names, values, groups, edges)
    return kg, names, path


# ─── Cell definitions ───────────────────────────────────────────────────────


def _cell_describe(kg) -> float:
    t0 = time.perf_counter()
    kg.describe()
    return time.perf_counter() - t0


def _cell_schema(kg) -> float:
    t0 = time.perf_counter()
    kg.schema()
    return time.perf_counter() - t0


def _cell_find_20x(kg, targets) -> float:
    t0 = time.perf_counter()
    for name in targets:
        kg.find(name, node_type="Node")
    return time.perf_counter() - t0


def _cell_multi_predicate(kg) -> float:
    q = "MATCH (n:Node) WHERE n.group = 1 AND n.value > 50 RETURN count(n) AS c"
    t0 = time.perf_counter()
    kg.cypher(q)
    return time.perf_counter() - t0


def _cell_pagerank(kg) -> float:
    t0 = time.perf_counter()
    kg.pagerank()
    return time.perf_counter() - t0


def _cell_two_hop_10x(kg, anchors) -> float:
    """Match 10 two-hop queries from random anchor IDs (matches Phase 6 shape)."""
    t0 = time.perf_counter()
    for anchor in anchors:
        kg.cypher(
            "MATCH (a:Node {id: $a})-[:EDGE]->(b)-[:EDGE]->(c) RETURN count(DISTINCT c) AS c",
            params={"a": anchor},
        )
    return time.perf_counter() - t0


def _cell_simple_filter(kg) -> float:
    t0 = time.perf_counter()
    kg.cypher("MATCH (n:Node) WHERE n.value > 75 RETURN count(n) AS c")
    return time.perf_counter() - t0


def _cell_pattern_match(kg) -> float:
    t0 = time.perf_counter()
    kg.cypher("MATCH (a:Node)-[:EDGE]->(b:Node) RETURN count(*) AS c")
    return time.perf_counter() - t0


def _cell_aggregation(kg) -> float:
    t0 = time.perf_counter()
    kg.cypher("MATCH (n:Node) RETURN n.group AS g, count(n) AS c, avg(n.value) AS v")
    return time.perf_counter() - t0


def _cell_order_by_limit(kg) -> float:
    t0 = time.perf_counter()
    kg.cypher("MATCH (n:Node) RETURN n.id AS id, n.value AS v ORDER BY v DESC LIMIT 50")
    return time.perf_counter() - t0


# ─── Runner ─────────────────────────────────────────────────────────────────


def run_construction(modes, n_trials, tmpdir) -> list[dict]:
    cells = []
    # Cap construction trials so the full sweep fits in budget.
    trials = min(n_trials, BUILD_TRIALS)
    for scale in CONSTRUCTION_SCALES:
        for mode in modes:
            samples = []
            last_rss = None
            last_dir = None
            names, values, groups, edges = _make_shape(scale)
            for i in range(trials):
                gc.collect()
                path: Path | None = None
                if mode == "disk":
                    path = tmpdir / f"build_{scale}_{mode}_{i}"
                    path.mkdir(exist_ok=True)
                t0 = time.perf_counter()
                if mode == "memory":
                    kg = kglite.KnowledgeGraph()
                elif mode == "mapped":
                    kg = kglite.KnowledgeGraph(storage="mapped")
                else:
                    kg = kglite.KnowledgeGraph(storage="disk", path=str(path))
                _populate(kg, scale, names, values, groups, edges)
                elapsed = time.perf_counter() - t0
                samples.append(elapsed)
                # Capture footprint from the last trial.
                last_rss = _rss_bytes()
                last_dir = _dir_size(path) if path else None
                del kg

            cell = {"name": "construction", "scale": scale, "mode": mode}
            cell.update(_stats(samples, "s"))
            if last_rss is not None:
                cell["build_rss_bytes"] = last_rss
            if last_dir is not None:
                cell["build_dir_bytes"] = last_dir
            cells.append(cell)
            print(
                f"  construction[{mode}] n={scale}: "
                f"p50={cell['p50'] * 1000:.2f}ms  "
                f"stddev={cell['stddev'] * 1000:.2f}ms  "
                f"trials={cell['trials']}"
            )
    return cells


def run_queries(modes, n_trials, tmpdir) -> list[dict]:
    cells = []
    import random as _r

    for mode in modes:
        gc.collect()
        kg, names, path = _build(QUERY_SCALE, mode, tmpdir)
        rng = _r.Random(11)
        find_targets = [rng.choice(names) for _ in range(20)]
        anchor_ids = [rng.randrange(QUERY_SCALE) for _ in range(10)]

        cell_fns = [
            ("describe", lambda kg=kg: _cell_describe(kg)),
            ("schema", lambda kg=kg: _cell_schema(kg)),
            ("find_20x", lambda kg=kg: _cell_find_20x(kg, find_targets)),
            ("multi_predicate", lambda kg=kg: _cell_multi_predicate(kg)),
            ("pagerank", lambda kg=kg: _cell_pagerank(kg)),
            ("two_hop_10x", lambda kg=kg: _cell_two_hop_10x(kg, anchor_ids)),
            ("simple_filter", lambda kg=kg: _cell_simple_filter(kg)),
            ("pattern_match", lambda kg=kg: _cell_pattern_match(kg)),
            ("aggregation", lambda kg=kg: _cell_aggregation(kg)),
            ("order_by_limit", lambda kg=kg: _cell_order_by_limit(kg)),
        ]

        for test_name, fn in cell_fns:
            try:
                fn()  # warmup
            except Exception as exc:
                print(f"  ! {test_name}[{mode}] skipped: {exc!r}")
                continue
            samples = [fn() for _ in range(n_trials)]
            cell = {"name": test_name, "scale": QUERY_SCALE, "mode": mode}
            cell.update(_stats(samples, "s"))
            cells.append(cell)
            print(
                f"  {test_name}[{mode}]: "
                f"p50={cell['p50'] * 1000:.3f}ms  "
                f"p95={cell['p95'] * 1000:.3f}ms  "
                f"stddev={cell['stddev'] * 1000:.3f}ms"
            )

        # Footprint for QUERY_SCALE
        rss = _rss_bytes()
        dsize = _dir_size(path) if path else None
        cells.append(
            {
                "name": "footprint",
                "scale": QUERY_SCALE,
                "mode": mode,
                "build_rss_bytes": rss,
                "build_dir_bytes": dsize,
            }
        )

        del kg
    return cells


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Output JSON path")
    parser.add_argument("--n", type=int, default=20, help="Trials per cell (query cells)")
    parser.add_argument(
        "--modes",
        default=",".join(ALL_MODES),
        help="Comma-separated storage modes (default: memory,mapped,disk)",
    )
    parser.add_argument(
        "--skip-construction",
        action="store_true",
        help="Skip the construction sweep (query-only)",
    )
    args = parser.parse_args()

    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    for m in modes:
        if m not in ALL_MODES:
            raise SystemExit(f"unknown mode {m!r}; expected one of {ALL_MODES}")

    import tempfile

    out_path = Path(args.json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Phase 11 harness — n={args.n}, modes={modes}, json={out_path}")
    print(f"  kglite version: {_kglite_version()}")
    print(f"  git sha: {_git_sha()}")
    wall_t0 = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="phase11_") as td:
        tmpdir = Path(td)
        cells: list[dict] = []
        if not args.skip_construction:
            print("— construction sweep —")
            cells.extend(run_construction(modes, args.n, tmpdir))
        print("— query sweep —")
        cells.extend(run_queries(modes, args.n, tmpdir))

    wall = time.perf_counter() - wall_t0
    out = {
        "harness_version": 1,
        "kglite_version": _kglite_version(),
        "git_sha": _git_sha(),
        "n_trials": args.n,
        "modes": list(modes),
        "wall_clock_seconds": wall,
        "cells": cells,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwrote {out_path} (wall_clock={wall:.1f}s, cells={len(cells)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
