"""Test load times on graph_500.0 (rebuilt) and the main Wikidata graph.

Each graph runs in its own subprocess to keep RSS attribution clean.
Measures cold (sudo purge between runs) + warm load times.
"""

import argparse
import json  # noqa: F401  — used by est_size block
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

GRAPHS = [
    ("graph_500.0 (~6M nodes)", "/Volumes/EksternalHome/Data/Wikidata/graph_500.0"),
    ("main wikidata (124M nodes)", "/Volumes/EksternalHome/Data/Wikidata/graph"),
]

TIMING_RE = re.compile(r"\[TIMING\]\s+stage=(\S+)\s+dur_ms=([\d.]+)")


def run_load(path: str, run_query: bool = True) -> tuple[dict[str, float], float, int]:
    """One load via subprocess; returns (per-stage ms, wall ms, peak RSS bytes).

    `run_query=False` skips the count(r) sanity scan — needed on huge graphs
    where the unanchored edge scan is too expensive."""
    query_block = (
        "rows = list(g.cypher('MATCH ()-[r]->() RETURN count(r) AS c'));"
        "edge_count_query = rows[0]['c'] if rows else None;"
        if run_query
        else "edge_count_query = None;"
    )
    code = (
        "import os, sys, time, resource;"
        "t0 = time.perf_counter();"
        f"import kglite; g = kglite.load({path!r});"
        "wall_ms = (time.perf_counter() - t0) * 1000.0;"
        "rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;"
        "rss_b = rss_kb if sys.platform == 'darwin' else rss_kb * 1024;"
        "info = g.graph_info();"
        + query_block
        + "print(f'__BENCH_WALL__={wall_ms:.1f}', file=sys.stderr);"
        "print(f'__BENCH_RSS__={rss_b}', file=sys.stderr);"
        "print(f'__BENCH_NODES__={info[\"node_count\"]}', file=sys.stderr);"
        "print(f'__BENCH_EDGES_META__={info[\"edge_count\"]}', file=sys.stderr);"
        "print(f'__BENCH_EDGES_QUERY__={edge_count_query}', file=sys.stderr);"
    )
    env = os.environ.copy()
    env["KGLITE_LOAD_TIMING"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"load failed (exit {proc.returncode})")

    stages: dict[str, float] = {}
    info: dict = {}
    for line in proc.stderr.splitlines():
        m = TIMING_RE.search(line)
        if m:
            stages[m.group(1)] = float(m.group(2))
            continue
        if line.startswith("__BENCH_"):
            k, v = line.split("=", 1)
            info[k.strip("_")] = v
    wall_ms = float(info.get("BENCH_WALL", 0))
    rss_b = int(info.get("BENCH_RSS", 0))
    return stages, wall_ms, rss_b, info


def fmt_ms(ms):
    if ms is None:
        return "      —"
    if ms >= 1000:
        return f"{ms / 1000:7.2f} s"
    return f"{ms:7.0f} ms"


def fmt_bytes(b):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(b) < 1024:
            return f"{b:6.1f} {unit}"
        b /= 1024
    return f"{b:6.1f} TB"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cold", action="store_true", help="Drop page cache before each load (sudo purge)")
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--graphs", type=str, default=None,
                   help="Comma-separated paths; default = graph_500 + main wikidata")
    args = p.parse_args()

    if args.graphs:
        graphs = []
        for path in args.graphs.split(","):
            path = path.strip()
            if path:
                graphs.append((Path(path).name, path))
    else:
        graphs = [(label, path) for label, path in GRAPHS if Path(path).exists()]

    if not graphs:
        print("error: no graphs available", file=sys.stderr)
        return 2

    for label, path in graphs:
        if not Path(path).exists():
            print(f"\n=== {label}: SKIPPED (not found at {path}) ===")
            continue

        print(f"\n{'=' * 70}")
        print(f"  {label}")
        print(f"  path: {path}")
        print("=" * 70)

        # File inventory — what format are the indexes in?
        bin_path = Path(path) / "id_indices.bin"
        zst_path = Path(path) / "id_indices.bin.zst"
        format_label = "new (.bin)" if bin_path.exists() else "legacy (.bin.zst)"
        print(f"  index format:    {format_label}")
        if bin_path.exists():
            print(f"    id_indices.bin:    {fmt_bytes(bin_path.stat().st_size)}")
        if zst_path.exists():
            print(f"    id_indices.bin.zst: {fmt_bytes(zst_path.stat().st_size)}")

        warm_walls = []
        warm_stages: list[dict[str, float]] = []
        warm_rss = []
        info = None
        # Skip the count(r) scan on graphs with >50M nodes — it would OOM
        # the subprocess on the main Wikidata graph.
        big_graph = False
        try:
            with (Path(path) / "disk_graph_meta.json").open() as fh:
                if json.load(fh).get("node_count", 0) > 50_000_000:
                    big_graph = True
        except Exception:
            pass

        for i in range(args.trials):
            if args.cold:
                print(f"\n  [cold trial {i + 1}] purging page cache…")
                subprocess.run(["sudo", "purge"], check=True)
                time.sleep(1.0)
                stages, wall, rss, info = run_load(path, run_query=not big_graph)
                print(f"  cold {i + 1}: {fmt_ms(wall)}  RSS {fmt_bytes(rss)}")
            stages, wall, rss, info = run_load(path, run_query=not big_graph)
            print(f"  warm {i + 1}: {fmt_ms(wall)}  RSS {fmt_bytes(rss)}")
            warm_walls.append(wall)
            warm_stages.append(stages)
            warm_rss.append(rss)

        # Per-stage median
        all_stages = set()
        for s in warm_stages:
            all_stages.update(s.keys())
        ordered = [
            "metadata_json", "interner_load", "disk_graph_load",
            "type_indices_load", "column_stores_load", "id_indices_load",
            "type_connectivity_load",
        ]
        ordered.extend(sorted(s for s in all_stages if s not in ordered and not s.startswith("dg.")))
        print(f"\n  warm wall-clock median: {fmt_ms(median(warm_walls))}")
        print(f"  warm RSS median:        {fmt_bytes(int(median(warm_rss)))}")
        if info:
            print(f"  nodes: {info.get('BENCH_NODES'):>15}  "
                  f"edges (meta): {info.get('BENCH_EDGES_META'):>12}  "
                  f"edges (query): {info.get('BENCH_EDGES_QUERY'):>12}")
        print(f"\n  warm per-stage (median across {args.trials} trials):")
        for stage in ordered:
            vals = [s.get(stage) for s in warm_stages if stage in s]
            if not vals:
                continue
            print(f"    {stage:<28} {fmt_ms(median(vals))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
