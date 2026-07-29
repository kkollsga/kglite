"""Per-stage load timing benchmark for disk-mode KGLite graphs.

Runs `kglite.load(path)` in a fresh subprocess with `KGLITE_LOAD_TIMING=1`
and parses the `[TIMING] stage=<name> dur_ms=<ms>` lines emitted to
stderr by `src/graph/io/load_timing.rs`. Measures cold-cache load time
(after `sudo purge`) and warm-cache load time (back-to-back run).

The cold/warm gap per stage isolates *page-fault I/O cost* from
*CPU-bound work* (zstd decompression + HashMap rebuild). Stages that
shrink dramatically warm-cache are I/O-dominated; stages that don't
change are CPU-bound and won't benefit from a lazier-load redesign.

Usage:
    # Default — Wikidata graph at /Volumes/EksternalHome/Data/Wikidata/graph
    python bench/bench_load_timing.py

    # Custom graph path, fewer trials
    python bench/bench_load_timing.py --path /path/to/graph --trials 2

    # Skip cold-cache measurements (no sudo prompt)
    python bench/bench_load_timing.py --no-cold

    # Skip warm runs (just a single cold pass)
    python bench/bench_load_timing.py --no-warm

`sudo purge` is invoked between cold trials. The script will prompt for
your sudo password the first time; subsequent invocations within the
sudo grace period (5 min) reuse the cached credential.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Optional

DEFAULT_PATH = "/Volumes/EksternalHome/Data/Wikidata/graph"

# Order matches the stage_timer/log_stage calls in src/graph/io/file.rs
# load_disk_dir. Any new stage added there will appear under "(other)".
KNOWN_STAGES = [
    "metadata_json",
    "interner_load",
    "disk_graph_load",
    "type_indices_load",
    "column_stores_load",
    "id_indices_load",
    "type_connectivity_load",
    "load_disk_dir_total",
]

TIMING_RE = re.compile(r"\[TIMING\]\s+stage=(\S+)\s+dur_ms=([\d.]+)")


def run_load(path: str) -> tuple[dict[str, float], float, int]:
    """Run kglite.load(path) in a subprocess. Returns (per-stage ms, total wall ms, peak RSS bytes)."""
    code = (
        "import os, sys, time, resource;"
        "t0 = time.perf_counter();"
        f"import kglite; g = kglite.load({path!r});"
        "wall_ms = (time.perf_counter() - t0) * 1000.0;"
        "rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;"
        # macOS reports ru_maxrss in bytes, Linux in KB.
        "rss_b = rss_kb if sys.platform == 'darwin' else rss_kb * 1024;"
        "print(f'__BENCH_WALL_MS__={wall_ms:.1f}', file=sys.stderr);"
        "print(f'__BENCH_RSS_B__={rss_b}', file=sys.stderr);"
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
        raise RuntimeError(f"kglite.load failed (exit {proc.returncode})")

    stages: dict[str, float] = {}
    wall_ms = 0.0
    rss_b = 0
    for line in proc.stderr.splitlines():
        m = TIMING_RE.search(line)
        if m:
            stages[m.group(1)] = float(m.group(2))
            continue
        if line.startswith("__BENCH_WALL_MS__="):
            wall_ms = float(line.split("=", 1)[1])
        elif line.startswith("__BENCH_RSS_B__="):
            rss_b = int(line.split("=", 1)[1])
    return stages, wall_ms, rss_b


def drop_page_cache() -> None:
    """Evict the OS page cache. macOS `sudo purge`, Linux drop_caches."""
    if sys.platform == "darwin":
        subprocess.run(["sudo", "purge"], check=True)
    else:
        # Linux: requires root.
        subprocess.run(
            ["sudo", "sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"],
            check=True,
        )


def fmt_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:6.2f} s"
    return f"{ms:6.0f} ms"


def fmt_bytes(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(b) < 1024:
            return f"{b:6.1f} {unit}"
        b /= 1024  # type: ignore[assignment]
    return f"{b:6.1f} TB"


def aggregate(runs: list[dict[str, float]]) -> dict[str, float]:
    """Median per stage across runs."""
    if not runs:
        return {}
    keys = set()
    for r in runs:
        keys.update(r.keys())
    return {k: median(r.get(k, 0.0) for r in runs) for k in keys}


def print_stage_table(
    label_cold: Optional[str],
    cold_runs: list[dict[str, float]],
    cold_walls: list[float],
    cold_rss: list[int],
    label_warm: Optional[str],
    warm_runs: list[dict[str, float]],
    warm_walls: list[float],
    warm_rss: list[int],
) -> None:
    cold = aggregate(cold_runs) if cold_runs else {}
    warm = aggregate(warm_runs) if warm_runs else {}

    # Order stages: known order first, then any extras alphabetically.
    seen = set(cold) | set(warm)
    ordered = [s for s in KNOWN_STAGES if s in seen]
    extras = sorted(s for s in seen if s not in KNOWN_STAGES)
    ordered.extend(extras)

    print()
    header = f"{'stage':<28}"
    if cold:
        header += f" {'cold (median)':>14}"
    if warm:
        header += f" {'warm (median)':>14}"
    if cold and warm:
        header += f" {'gap (I/O)':>12}"
    print(header)
    print("-" * len(header))

    total_cold = cold.get("load_disk_dir_total", sum(v for k, v in cold.items() if k != "load_disk_dir_total"))
    total_warm = warm.get("load_disk_dir_total", sum(v for k, v in warm.items() if k != "load_disk_dir_total"))

    for stage in ordered:
        if stage == "load_disk_dir_total":
            continue
        row = f"{stage:<28}"
        c = cold.get(stage)
        w = warm.get(stage)
        if cold:
            row += f" {fmt_ms(c) if c is not None else '       —':>14}"
        if warm:
            row += f" {fmt_ms(w) if w is not None else '       —':>14}"
        if cold and warm:
            if c is not None and w is not None:
                gap = c - w
                pct = (gap / c * 100.0) if c > 0 else 0.0
                row += f" {fmt_ms(gap):>12} ({pct:4.0f}%)"
            else:
                row += f" {'—':>12}"
        print(row)

    print("-" * len(header))
    row = f"{'TOTAL (sum of stages)':<28}"
    if cold:
        row += f" {fmt_ms(total_cold):>14}"
    if warm:
        row += f" {fmt_ms(total_warm):>14}"
    if cold and warm:
        row += f" {fmt_ms(total_cold - total_warm):>12}"
    print(row)

    print()
    if cold_walls:
        print(f"cold wall-clock (median over {len(cold_walls)}): {fmt_ms(median(cold_walls))}   peak RSS: {fmt_bytes(int(median(cold_rss)))}")
    if warm_walls:
        print(f"warm wall-clock (median over {len(warm_walls)}): {fmt_ms(median(warm_walls))}   peak RSS: {fmt_bytes(int(median(warm_rss)))}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--path", default=DEFAULT_PATH, help=f"Disk-graph directory (default: {DEFAULT_PATH})")
    p.add_argument("--trials", type=int, default=2, help="Trials per mode (default: 2)")
    p.add_argument("--no-cold", action="store_true", help="Skip cold-cache runs (no sudo)")
    p.add_argument("--no-warm", action="store_true", help="Skip warm-cache runs")
    p.add_argument("--json-out", type=Path, help="Optional: dump per-run timings as JSON")
    args = p.parse_args()

    graph_path = Path(args.path)
    if not graph_path.is_dir() or not (graph_path / "disk_graph_meta.json").exists():
        print(f"error: {graph_path} is not a disk-graph directory (missing disk_graph_meta.json)", file=sys.stderr)
        return 2

    print(f"graph: {graph_path}")

    # Quick disk-size summary so the human running this knows what they're testing.
    sizes: dict[str, int] = {}
    for sub in (".", "seg_000"):
        d = graph_path / sub
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.is_file():
                sizes[f.name] = sizes.get(f.name, 0) + f.stat().st_size
    big_files = sorted(sizes.items(), key=lambda kv: -kv[1])[:8]
    print("\nlargest files:")
    for name, sz in big_files:
        print(f"  {fmt_bytes(sz):>10}  {name}")

    cold_runs: list[dict[str, float]] = []
    cold_walls: list[float] = []
    cold_rss: list[int] = []
    warm_runs: list[dict[str, float]] = []
    warm_walls: list[float] = []
    warm_rss: list[int] = []

    for i in range(args.trials):
        if not args.no_cold:
            print(f"\n[cold trial {i + 1}/{args.trials}] purging page cache…")
            drop_page_cache()
            time.sleep(1.0)  # let purge settle
            print(f"[cold trial {i + 1}/{args.trials}] loading…")
            stages, wall, rss = run_load(str(graph_path))
            print(f"[cold trial {i + 1}/{args.trials}] done in {fmt_ms(wall)}")
            cold_runs.append(stages)
            cold_walls.append(wall)
            cold_rss.append(rss)

        if not args.no_warm:
            # Warm = run again immediately, page cache is hot from the cold run.
            # If --no-cold was given, the first warm trial may itself be cold-ish;
            # we still report it but the user can discard the first row.
            print(f"[warm trial {i + 1}/{args.trials}] loading…")
            stages, wall, rss = run_load(str(graph_path))
            print(f"[warm trial {i + 1}/{args.trials}] done in {fmt_ms(wall)}")
            warm_runs.append(stages)
            warm_walls.append(wall)
            warm_rss.append(rss)

    print_stage_table(
        "cold", cold_runs, cold_walls, cold_rss,
        "warm", warm_runs, warm_walls, warm_rss,
    )

    if args.json_out:
        payload = {
            "path": str(graph_path),
            "trials": args.trials,
            "cold": [{"stages": s, "wall_ms": w, "rss_b": r} for s, w, r in zip(cold_runs, cold_walls, cold_rss)],
            "warm": [{"stages": s, "wall_ms": w, "rss_b": r} for s, w, r in zip(warm_runs, warm_walls, warm_rss)],
        }
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
