#!/usr/bin/env python3
"""Fresh-process peak RSS and first full Cypher query for geographic DBSCAN.

Uses the matched full-query harness and release identity checks. Every measured
case starts in a separate process; RSS is the actual process high-water mark,
not a matrix-size estimate or an allocator-only attribution. Existing bench/out
owns compressed captures. This script never builds or changes dependencies.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import lzma
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().with_name("dbscan_phase10.py")
spec = importlib.util.spec_from_file_location("dbscan_phase10", SOURCE)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def peak_bytes() -> int:
    import resource

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def child(case: str, nodes: int) -> dict:
    import os

    release = harness.release_provenance()
    graph, query, expected, fixture = harness.build_case(case, nodes)
    gc.collect()
    before = peak_bytes()
    start = time.perf_counter_ns()
    rows = graph.cypher(query).to_list()
    elapsed = time.perf_counter_ns() - start
    query_peak = peak_bytes()
    harness.check_rows(rows, expected)
    return {
        "case": case,
        "nodes": nodes,
        "pid": os.getpid(),
        "release": release,
        "query": query,
        "fixture_sha256": fixture,
        "exact_ordered_rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "every_value_type_order_checked": True,
        "query_ns": elapsed,
        "before_query_process_peak_rss_bytes": before,
        "after_query_process_peak_rss_bytes": query_peak,
        "after_oracle_process_peak_rss_bytes": peak_bytes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, choices=[32, 2048, 4096], default=4096)
    parser.add_argument("--child", choices=["geo_dense", "geo_sparse"])
    parser.add_argument("--label")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if sys.platform not in ("darwin", "linux"):
        parser.error("peak RSS normalization currently supports macOS and Linux")
    if args.child:
        print(json.dumps(child(args.child, args.nodes)))
        return
    if not args.label or args.out is None:
        parser.error("parent capture requires --label and --out")
    out = args.out.resolve()
    if out.exists() or out.suffixes[-2:] != [".json", ".xz"] or not out.is_relative_to(ROOT / "dev-docs/bench/out"):
        parser.error("--out must be a new .json.xz under dev-docs/bench/out")
    release = harness.release_provenance()
    cells = []
    for case in ["geo_dense", "geo_sparse"]:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", case, "--nodes", str(args.nodes)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode:
            raise RuntimeError(f"{case} child failed {proc.returncode}: {proc.stderr}\n{proc.stdout}")
        row = json.loads(proc.stdout)
        if row["release"] != release or not row["every_value_type_order_checked"]:
            raise AssertionError("child release or oracle differs")
        cells.append(row)
        print(f"Checked fresh {case} n={args.nodes}", flush=True)
    result = {
        "label": args.label,
        "head": harness.git("rev-parse", "HEAD"),
        "status": harness.git("status", "--porcelain"),
        "diff_sha256": hashlib.sha256(harness.git("diff", "HEAD").encode()).hexdigest(),
        "source_sha256": {
            str(SOURCE.relative_to(ROOT)): harness.sha(SOURCE),
            str(Path(__file__).relative_to(ROOT)): harness.sha(Path(__file__)),
        },
        "production_sha256": harness.sha(ROOT / "crates/kglite/src/graph/algorithms/clustering.rs"),
        "release": release,
        "cells": cells,
        "scope": (
            "fresh child per cell; first complete consumed Cypher query; "
            "setup/checks/disposal untimed; absolute process high-water RSS"
        ),
        "limitations": (
            "RSS includes interpreter/imports/fixture; before/after peak difference "
            "is not live allocation or exact allocator attribution"
        ),
    }
    raw = json.dumps(result, indent=2).encode()
    compressed = lzma.compress(raw)
    assert lzma.decompress(compressed) == raw
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as f:
        f.write(compressed)
    print(f"Saved {len(cells)} fresh process cells")


if __name__ == "__main__":
    main()
