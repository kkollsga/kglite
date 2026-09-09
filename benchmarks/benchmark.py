#!/usr/bin/env python3
"""KGLite benchmark — run the configured adapter workloads on one synthetic
graph and rewrite the public comparison table only from a qualified capture.

    python benchmarks/benchmark.py                  # medium graph, local adapter set
    python benchmarks/benchmark.py --scale large    # bigger graph
    python benchmarks/benchmark.py --libs kglite-cypher,ladybug,networkx
    python benchmarks/benchmark.py --report-only    # just rewrite BENCHMARKS.md from saved results

Install every engine named by the invocation. A missing or failed adapter
rejects a publication capture rather than reusing an older row:

    pip install kglite ladybug networkx rustworkx igraph duckdb neo4j

The dataset is a seed-deterministic org/social knowledge graph
(Person/Company/Project/Skill/City + 7 edge types) — the *same* schema the
bundled Rust `graphgen` streams at million-node scale (see
`benchmarks/competitive/largescale/` for the larger-than-RAM runs). Every
adapter receives the same staged input. Adapters use their idiomatic surfaces;
the report identifies workloads they did not exercise.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_PUBLIC_LIBS = "kglite-cypher,ladybug,networkx,rustworkx,igraph,duckdb"

# Resolve the benchmark package from this checkout even when the invoking
# interpreter has another editable KGLite checkout installed.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", default="medium", help="dataset scale: small | medium (default) | large")
    ap.add_argument(
        "--libs",
        default=DEFAULT_PUBLIC_LIBS,
        help=f"comma-separated backends (default: {DEFAULT_PUBLIC_LIBS})",
    )
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="skip the run; just regenerate BENCHMARKS.md from the saved results",
    )
    ap.add_argument("--out", default=str(ROOT / "BENCHMARKS.md"), help="output table path")
    args = ap.parse_args()

    if not args.report_only:
        import shutil
        import tempfile

        import kglite

        # Stage one canonical input for every adapter in this invocation.
        staged = tempfile.mkdtemp(prefix="kglite_bench_")
        try:
            stats = kglite.graphgen(args.scale, seed=1234, out=staged)
            print(
                f"[dataset] kglite.graphgen({args.scale!r}) -> {stats['nodes']:,} nodes / {stats['edges']:,} edges\n",
                flush=True,
            )
            cmd = [
                sys.executable,
                "-m",
                "benchmarks.competitive.graphsuite.run",
                "--scale",
                args.scale,
                "--staged",
                staged,
                "--publication-capture",
            ]
            cmd += ["--libs", args.libs]
            print(f"$ {' '.join(cmd)}\n", flush=True)
            proc = subprocess.run(cmd, cwd=str(ROOT))
            if proc.returncode != 0:
                print("benchmark run failed", file=sys.stderr)
                return proc.returncode
        finally:
            shutil.rmtree(staged, ignore_errors=True)

    # Render the public, topic-summed table.
    from benchmarks.competitive.graphsuite.marketing import render

    md = render()
    out = pathlib.Path(args.out)
    out.write_text(md, encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
