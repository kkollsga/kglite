#!/usr/bin/env python3
"""Phase 6 current-release aggregate dispatch benchmark; artifacts in bench/out.

The original graph.cypher stage is retained. A separate consumed cell includes
conversion. Every result is checked outside clocks; CPU is process-wide query
cost, not attribution to a particular function or allocator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import lzma
from pathlib import Path
import resource
import runpy
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
kglite = importlib.import_module("kglite")

SOURCE = "crates/kglite/src/graph/languages/cypher/executor/aggregation/materialized.rs"
CONSTANTS = runpy.run_path(str(ROOT / "tests/benchmarks/test_bench_parallel_runtime.py"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    return float(values[lo]) + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def grouped(rows, key, val, percentile=False):
    groups = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row[val])
    if percentile:
        return [{"y": k, "m": quantile(v, 0.5), "p90": quantile(v, 0.9)} for k, v in groups.items()]
    alias = {"y": "y", "c": "c", "nm": "nm"}[key]
    return [{alias: k, "ages": v} for k, v in groups.items()]


def stats(values):
    lo, median = min(values), statistics.median(values)
    return {
        "samples_ns": values,
        "min_ns": lo,
        "median_ns": median,
        "mean_ns": statistics.mean(values),
        "statistic": "median" if lo < 0.7 * median else "min",
    }


def measure(graph, label, query, expected, args, consumed=False):
    result = []
    rounds = max(args.rounds, 100) if label.startswith("control_1000_") else args.rounds
    warmup = max(args.warmup, 20) if label.startswith("control_1000_") else args.warmup
    for parallel in [False, True]:
        wall, cpu = [], []
        for round_index in range(warmup + rounds):
            begin_cpu, begin = time.process_time_ns(), time.perf_counter_ns()
            view = graph.cypher(query, parallel=parallel)
            if consumed:
                rows = view.to_list()
            end = time.perf_counter_ns()
            end_cpu = time.process_time_ns()
            if not consumed:
                rows = view.to_list()
            assert rows == expected, (label, parallel, "ordered result mismatch")
            del rows, view
            if round_index >= warmup:
                wall.append(end - begin)
                cpu.append(end_cpu - begin_cpu)
        result.append(
            {
                "label": label,
                "query": query,
                "parallel": parallel,
                "stage": "cypher+to_list" if consumed else "cypher",
                "wall": stats(wall),
                "cpu": stats(cpu),
                "oracle_sha256": digest(expected),
                "oracle_rows": len(expected),
                "oracle_passed": True,
            }
        )
        print(label, parallel, round(statistics.median(wall) / 1e6, 3), "ms", flush=True)
    return result


def main_cells(args):
    graph = kglite.graphgen(persons=args.persons, seed=1234)
    inputs = graph.cypher(
        "MATCH (p:Person) RETURN p.joined_year AS y,p.city AS c,p.name AS nm,p.age AS age,p.score AS score"
    ).to_list()
    assert len(inputs) == args.persons
    cells = []
    for label, key, constant in [
        ("low", "y", "AGG_LOW_CARD"),
        ("mid", "c", "AGG_MID_CARD"),
        ("high", "nm", "AGG_HIGH_CARD"),
        ("percentile", "y", "AGG_PERCENTILE"),
    ]:
        expected = grouped(inputs, key, "age", label == "percentile")
        cells.extend(measure(graph, label, CONSTANTS[constant], expected, args))
    expected = grouped(inputs, "y", "age")
    cells.extend(measure(graph, "low_consumed", CONSTANTS["AGG_LOW_CARD"], expected, args, consumed=True))
    count = sum(p["score"] > 0.5 for p in inputs)
    cells.extend(measure(graph, "count", CONSTANTS["SCAN_AGG_COUNT"], [{"n": count}], args))
    cells.extend(
        measure(graph, "filter", CONSTANTS["SCAN_FILTER_WHERE"], [{"n": sum("a1" in p["nm"] for p in inputs)}], args)
    )
    groups = {}
    for p in inputs:
        if p["score"] > 0.5:
            groups.setdefault(p["y"], []).append(p["age"])
    expected = [{"y": key, "n": len(vals), "a": sum(vals) / len(vals)} for key, vals in groups.items()]
    cells.extend(measure(graph, "numeric", CONSTANTS["SCAN_AGG_GROUPED"], expected, args))
    reader = "MATCH (p:Person) WHERE p.score > 0.9 RETURN count(*) AS n"
    reader_expected = [{"n": sum(p["score"] > 0.9 for p in inputs)}]
    heavy = "MATCH (p:Person) WHERE p.score > 0.5 RETURN p.joined_year AS y, count(*) AS n"
    heavy_expected = [{"y": key, "n": len(vals)} for key, vals in groups.items()]
    for admitted in [False, True]:
        wall, cpu = [], []
        for iteration in range(args.warmup + args.rounds):
            outputs, errors = [], []

            def run(query, parallel, expected):
                try:
                    view = graph.cypher(query, parallel=parallel)
                    outputs.append((view, expected))
                except BaseException as error:
                    errors.append(error)

            threads = [threading.Thread(target=run, args=(reader, False, reader_expected)) for _ in range(8)]
            if admitted:
                threads.insert(0, threading.Thread(target=run, args=(heavy, True, heavy_expected)))
            cb, start = time.process_time_ns(), time.perf_counter_ns()
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            end, ce = time.perf_counter_ns(), time.process_time_ns()
            assert not errors, errors
            assert len(outputs) == len(threads)
            assert all(view.to_list() == expected for view, expected in outputs)
            if iteration >= args.warmup:
                wall.append(end - start)
                cpu.append(ce - cb)
        cells.append(
            {
                "label": "eight_readers_heavy" if admitted else "eight_readers",
                "parallel": admitted,
                "stage": "thread launch+cypher+join; consumption excluded",
                "wall": stats(wall),
                "cpu": stats(cpu),
                "oracle_passed": True,
                "oracle_sha256": digest([reader_expected, heavy_expected]),
            }
        )
        print(cells[-1]["label"], round(statistics.median(wall) / 1e6, 3), "ms", flush=True)
    return cells


def control_cells(args):
    cells = []
    for n in args.sizes:
        graph = kglite.KnowledgeGraph()
        graph.cypher(
            "UNWIND range(0,$n-1) AS i CREATE (:Cell {id:i,g:i%31,two:i%2,"
            "skew:CASE WHEN i< $n-31 THEN 0 ELSE i END,v:i,text:$text,items:range(0,15)})",
            params={"n": n, "text": "東京 graph " * 64},
        )
        values = list(range(n))
        for label, key, expression in [
            ("many", "g", "n.v"),
            ("one", "one", "n.v"),
            ("two", "two", "n.v"),
            ("skew", "skew", "n.v"),
            ("string", "g", "n.text"),
            ("list", "g", "n.items"),
            ("expensive", "g", "toUpper(n.text)"),
            ("distinct", "g", "DISTINCT n.v"),
            ("mixed", "g", "n.v"),
            ("whole_node", "g", "n"),
        ]:
            if n not in (min(args.sizes), max(args.sizes)) and label != "many":
                continue
            group_expr = "1" if key == "one" else "n." + key
            query = f"MATCH(n:Cell) RETURN {group_expr} AS g,collect({expression}) AS vals"
            if label == "mixed":
                query += ", percentile_cont(n.v,0.9) AS p90"
            groups = {}
            for i in values:
                group = {"g": i % 31, "one": 1, "two": i % 2, "skew": 0 if i < n - 31 else i}[key]
                groups.setdefault(group, []).append(i)
            expected = []
            for group, members in groups.items():
                vals = members
                if label in ("string", "expensive"):
                    vals = [("東京 graph " * 64).upper() if label == "expensive" else "東京 graph " * 64] * len(members)
                elif label == "list":
                    vals = [list(range(16))] * len(members)
                elif label == "whole_node":
                    vals = [
                        {
                            "id": i,
                            "labels": ["Cell"],
                            "properties": {
                                "id": i,
                                "title": f"Cell_{i}",
                                "type": "Cell",
                                "g": i % 31,
                                "two": i % 2,
                                "skew": 0 if i < n - 31 else i,
                                "v": i,
                                "text": "東京 graph " * 64,
                                "items": list(range(16)),
                            },
                        }
                        for i in members
                    ]
                row = {"g": group, "vals": vals}
                if label == "mixed":
                    row["p90"] = quantile(members, 0.9)
                expected.append(row)
            cells.extend(measure(graph, f"control_{n}_{label}", query, expected, args))
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--persons", type=int, default=800_000)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 20000, 100000])
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    extension = Path(importlib.import_module("kglite.kglite").__file__)
    release = ROOT / "target/release" / ("libkglite_py.dylib" if sys.platform == "darwin" else "libkglite_py.so")
    if not args.preflight:
        assert sha(extension) == sha(release), "release extension required"
        assert release.stat().st_mtime_ns > (ROOT / SOURCE).stat().st_mtime_ns
    assert not args.out.exists()
    record = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "harness_sha256": sha(__file__),
        "source_sha256": sha(ROOT / SOURCE),
        "extension_sha256": sha(extension),
        "profile": "debug-preflight-only" if args.preflight else "release",
        "args": vars(args) | {"out": str(args.out)},
        "cells": main_cells(args) + control_cells(args),
    }
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record["process_peak_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
    args.out.write_bytes(lzma.compress((json.dumps(record, indent=2) + "\n").encode()))
    print("Wrote", args.out, len(record["cells"]), "cells", flush=True)


if __name__ == "__main__":
    main()
