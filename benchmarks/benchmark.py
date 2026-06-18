#!/usr/bin/env python3
"""KGLite benchmark — one command: run every installed graph backend on one
synthetic graph and (re)write the public comparison table, BENCHMARKS.md.

    python benchmarks/benchmark.py                  # medium graph, all installed libs
    python benchmarks/benchmark.py --scale large    # bigger graph
    python benchmarks/benchmark.py --libs kglite-cypher,kuzu,networkx
    python benchmarks/benchmark.py --report-only    # just rewrite BENCHMARKS.md from saved results

Install the engines you want to compare against (each is optional — a missing
one just drops out of the table):

    pip install kglite kuzu networkx rustworkx igraph duckdb

The dataset is a seed-deterministic org/social knowledge graph
(Person/Company/Project/Skill/City + 7 edge types) — the *same* schema the
bundled Rust `graphgen` streams at million-node scale (see
`benchmarks/competitive/largescale/` for the larger-than-RAM runs). Every
engine loads identical data and runs identical, seed-derived queries, so the
comparison reflects equal work.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", default="medium", help="dataset scale: small | medium (default) | large")
    ap.add_argument("--libs", default=None, help="comma-separated backends (default: all installed)")
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="skip the run; just regenerate BENCHMARKS.md from the saved results",
    )
    ap.add_argument("--out", default=str(ROOT / "BENCHMARKS.md"), help="output table path")
    args = ap.parse_args()

    if not args.report_only:
        cmd = [sys.executable, "-m", "benchmarks.competitive.graphsuite.run", "--scale", args.scale]
        if args.libs:
            cmd += ["--libs", args.libs]
        print(f"$ {' '.join(cmd)}\n", flush=True)
        proc = subprocess.run(cmd, cwd=str(ROOT))
        if proc.returncode != 0:
            print("benchmark run failed", file=sys.stderr)
            return proc.returncode

    # Render the public, topic-summed table.
    from benchmarks.competitive.graphsuite.marketing import render

    md = render()
    out = pathlib.Path(args.out)
    out.write_text(md)
    print(f"\nwrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
