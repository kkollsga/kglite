#!/usr/bin/env python3
"""First-save/load identity controls; release only, exact named baseline defects."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import kglite  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(values):
    return {
        "min_ns": min(values),
        "mean_ns": statistics.mean(values),
        "median_ns": statistics.median(values),
        "max_ns": max(values),
        "samples_ns": values,
    }


def capture(size, mode, kind, before, rounds):
    expected = [
        {"id": i if kind == "int" else i + 0.5 if kind == "float" else str(i), "title": f"row-{i}", "rank": i}
        for i in range(size)
    ]
    decoded = expected
    if before and mode == "disk" and kind in ("int", "float"):
        decoded = [
            {**r, "id": (i // 2 if i % 2 == 0 else 0) if kind == "int" else None} for i, r in enumerate(expected)
        ]
        assert decoded != expected
    clocks = {"save": [], "load": [], "first_read": []}
    for iteration in range(rounds + 2):
        with tempfile.TemporaryDirectory(prefix="kglite-identity-bench-") as tmp:
            path = str(Path(tmp) / "graph")
            options = {"storage": mode}
            if mode == "disk":
                options["path"] = path
            graph = kglite.KnowledgeGraph(**options)
            graph.cypher(
                "UNWIND $rows AS r CREATE (:Doc {id:r.id,title:r.title,rank:r.rank})", params={"rows": expected}
            )
            query = "MATCH(n:Doc) RETURN n.id AS id,n.title AS title,n.rank AS rank ORDER BY rank"
            assert graph.cypher(query).to_list() == expected
            start = time.perf_counter_ns()
            graph.save(path, fsync=False)
            saved = time.perf_counter_ns()
            loaded = kglite.load(path)
            opened = time.perf_counter_ns()
            actual = loaded.cypher(query).to_list()
            read = time.perf_counter_ns()
            assert actual == decoded, (mode, kind, size, actual[:4], decoded[:4])
            if iteration >= 2:
                for key, value in zip(clocks, (saved - start, opened - saved, read - opened)):
                    clocks[key].append(value)
            del graph, loaded
    return {
        "size": size,
        "storage": mode,
        "id_kind": kind,
        "expected_old_corruption": before and mode == "disk" and kind != "string",
        "oracle": "exact ordered ids/titles/ranks",
        "timings": {key: metrics(value) for key, value in clocks.items()},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", action="store_true")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    out = args.out.resolve()
    assert not out.exists() and out.is_relative_to(ROOT / "dev-docs/bench/out")
    extension = Path(importlib.import_module("kglite.kglite").__file__)
    release = ROOT / "target/release" / ("libkglite_py.dylib" if sys.platform == "darwin" else "libkglite_py.so")
    assert sha(extension) == sha(release), "install the current release extension first"
    sources = [
        ROOT / "Cargo.toml",
        *list((ROOT / "crates/kglite/src").rglob("*.rs")),
        *list((ROOT / "crates/kglite-py/src").rglob("*.rs")),
    ]
    assert release.stat().st_mtime_ns >= max(f.stat().st_mtime_ns for f in sources), "stale release artifact"
    cells = [
        capture(n, mode, kind, args.before, 1 if args.preflight else 10)
        for n in ([1000] if args.preflight else [1000, 20000])
        for mode, kind in [("disk", "int"), ("disk", "float"), ("disk", "string"), ("memory", "int"), ("mapped", "int")]
    ]
    record = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "driver_sha256": sha(__file__),
        "release_sha256": sha(extension),
        "before": args.before,
        "preflight": args.preflight,
        "statistic": "mean of first-event save/load/read; fsync disabled",
        "cells": cells,
    }
    out.write_text(json.dumps(record, indent=2) + "\n")
    print("Saved", len(cells), "exactly checked cells")


if __name__ == "__main__":
    main()
