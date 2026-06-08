"""Bench the in-memory baseline of save_subset on a real disk graph.

Phase 7 of the save_subset rollout. Runs in a subprocess so peak RSS
attribution is clean (mirrors bench_500_and_wiki.py).

Usage:
    python benchmarks/bench_save_subset.py \\
        --source /path/to/wiki_disk_graph \\
        --out /tmp/articles_authors.kgl \\
        [--node-type Article] \\
        [--hops 1]

What it measures:
- wall time of save_subset
- peak RSS during save_subset
- output file/directory size
- node + edge counts in the reloaded subset

Why subprocess: the parent process holds the source graph mmap'd, which
inflates RSS attribution. A subprocess call gets a clean slate.

Decision gate: if peak RSS exceeds ~1.5x what fits in the user's RAM
budget, the streaming swap (consume Pass A + RankIndex + kept_edges.tmp
primitives in src/graph/mutation/subgraph_streaming.rs) is needed. If
the in-memory baseline runs comfortably, we ship as-is.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def dir_size_bytes(path: str) -> int:
    """Recursive size of a path (file or directory)."""
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            total += Path(root, f).stat().st_size
    return total


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def run_save_subset_subprocess(source: str, out_path: str, node_type: str, hops: int) -> dict:
    """Run save_subset in a subprocess; return wall ms, peak RSS, counts."""
    code = f"""
import json, os, resource, sys, time
import kglite

src_path = {source!r}
out_path = {out_path!r}
node_type = {node_type!r}
hops = {hops}

# Load source.
t0 = time.perf_counter()
src = kglite.load(src_path)
load_ms = (time.perf_counter() - t0) * 1000.0
src_info = src.graph_info()

# Snapshot baseline RSS after load.
rss_kb_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_b_after_load = rss_kb_after_load if sys.platform == "darwin" else rss_kb_after_load * 1024

# Run save_subset.
t0 = time.perf_counter()
sel = src.select(node_type)
if hops > 0:
    sel = sel.expand(hops=hops)
sel.save_subset(out_path)
save_ms = (time.perf_counter() - t0) * 1000.0

rss_kb_after_save = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_b_after_save = rss_kb_after_save if sys.platform == "darwin" else rss_kb_after_save * 1024

# Reload subset and verify.
t0 = time.perf_counter()
sub = kglite.load(out_path)
reload_ms = (time.perf_counter() - t0) * 1000.0
sub_info = sub.graph_info()

result = {{
    "load_ms": load_ms,
    "save_ms": save_ms,
    "reload_ms": reload_ms,
    "rss_bytes_after_load": rss_b_after_load,
    "rss_bytes_after_save": rss_b_after_save,
    "src_node_count": src_info["node_count"],
    "src_edge_count": src_info["edge_count"],
    "sub_node_count": sub_info["node_count"],
    "sub_edge_count": sub_info["edge_count"],
}}
print("__BENCH_RESULT__=" + json.dumps(result), file=sys.stderr)
"""

    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print("Subprocess failed:", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        sys.exit(1)

    for line in proc.stderr.splitlines():
        if line.startswith("__BENCH_RESULT__="):
            return json.loads(line.removeprefix("__BENCH_RESULT__="))
    print("No bench result found in subprocess output:", file=sys.stderr)
    print(proc.stderr, file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        required=True,
        help="Path to source graph (any storage mode; disk recommended for Wikidata)",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output path for the subgraph file (single .kgl or directory)",
    )
    ap.add_argument(
        "--node-type",
        default="Article",
        help="Seed node type (default: Article)",
    )
    ap.add_argument(
        "--hops",
        type=int,
        default=1,
        help="Number of expand() hops (default: 1)",
    )
    ap.add_argument(
        "--keep-out",
        action="store_true",
        help="Don't remove the output after the bench",
    )
    args = ap.parse_args()

    source = os.path.abspath(args.source)
    out_path = os.path.abspath(args.out)

    if not Path(source).exists():
        print(f"Source path does not exist: {source}", file=sys.stderr)
        sys.exit(1)

    # Clean any prior output so size measurements are clean.
    if Path(out_path).exists():
        if Path(out_path).is_dir():
            shutil.rmtree(out_path)
        else:
            os.remove(out_path)

    src_size = dir_size_bytes(source)

    print(f"Source:    {source}  ({fmt_bytes(src_size)})")
    print(f"Output:    {out_path}")
    print(f"Filter:    select('{args.node_type}').expand(hops={args.hops})")
    print()
    print("Running...", flush=True)

    t0 = time.perf_counter()
    result = run_save_subset_subprocess(source, out_path, args.node_type, args.hops)
    total_wall = time.perf_counter() - t0

    out_size = dir_size_bytes(out_path) if Path(out_path).exists() else 0

    print()
    print(f"--- Results (total subprocess wall: {total_wall:.1f}s) ---")
    print(f"Source nodes / edges:    {result['src_node_count']:>15,} / {result['src_edge_count']:>15,}")
    print(f"Subset nodes / edges:    {result['sub_node_count']:>15,} / {result['sub_edge_count']:>15,}")
    print(f"Source size on disk:     {fmt_bytes(src_size):>15}")
    print(f"Output size on disk:     {fmt_bytes(out_size):>15}")
    print()
    print(f"Source load:             {result['load_ms']:>10.1f} ms")
    print(f"save_subset:             {result['save_ms']:>10.1f} ms")
    print(f"Reload subset:           {result['reload_ms']:>10.1f} ms")
    print()
    print(f"Peak RSS after load:     {fmt_bytes(result['rss_bytes_after_load']):>15}")
    print(f"Peak RSS after save:     {fmt_bytes(result['rss_bytes_after_save']):>15}")
    delta = result["rss_bytes_after_save"] - result["rss_bytes_after_load"]
    print(f"  Δ during save_subset:  {fmt_bytes(delta):>15}")
    print()
    print("Decision gate:")
    print("- If 'Δ during save_subset' fits comfortably in your RAM budget:")
    print("    in-memory baseline is sufficient; no streaming swap needed.")
    print("- If it does not fit (or OOMs):")
    print("    swap to streaming via Pass A + RankIndex + kept_edges.tmp")
    print("    primitives in src/graph/mutation/subgraph_streaming.rs.")

    if not args.keep_out and Path(out_path).exists():
        if Path(out_path).is_dir():
            shutil.rmtree(out_path)
        else:
            os.remove(out_path)
        print(f"\nCleaned up: {out_path}")


if __name__ == "__main__":
    main()
