#!/usr/bin/env python3
"""Phase 7 current-tree full-node conversion benchmark.

No automatic build. Full-node output is eager; optional lazy-property cells are
separately labeled. This driver makes no allocation, retained-RSS or leak claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import json
import lzma
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
SHAPES = ("scan_person", "scan_item", "numeric", "short", "string_rich", "unicode", "long")
ROUTES = ("full_node", "two_properties", "scalar")
LAZY_STATES = ("cold", "warm", "first", "alternating", "head", "tail")
SOURCE_PATHS = (
    "crates/kglite-py/src/datatypes/py_out.rs",
    "crates/kglite-py/src/graph/pyapi/result_view.rs",
    "crates/kglite/src/graph/languages/cypher/executor/helpers.rs",
    "crates/kglite/src/graph/languages/cypher/result.rs",
)
sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def release_provenance() -> dict:
    extension = Path(importlib.import_module("kglite.kglite").__file__).resolve()
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    artifact = ROOT / "target/release" / f"libkglite_py{suffix}"
    if not artifact.is_file() or sha(extension) != sha(artifact):
        raise RuntimeError("installed extension must match release cdylib; worker must build/install first")
    sources = [ROOT / "Cargo.toml"]
    for crate in ("kglite", "kglite-py"):
        sources.append(ROOT / "crates" / crate / "Cargo.toml")
        sources.extend((ROOT / "crates" / crate / "src").rglob("*.rs"))
    if artifact.stat().st_mtime_ns < max(path.stat().st_mtime_ns for path in sources):
        raise RuntimeError("release artifact predates engine/wrapper source")
    return {
        "extension": str(extension),
        "release_artifact": str(artifact.resolve()),
        "sha256": sha(extension),
        "version": importlib.import_module("kglite").__version__,
    }


def record(shape: str, i: int, long_chars: int) -> dict:
    if shape == "scan_person":
        return {"id": i, "title": str(i), "age": i % 60, "city": f"city{i % 100}"}
    if shape == "scan_item":
        return {"id": i, "title": f"node{i}", "qty": i, "body": "text data 1234567890"}
    row = {"id": i, "title": i, "age": i % 97, "score": (i % 251) * 0.125}
    if shape == "numeric":
        return row
    row["title"] = f"P{i:06d}"
    if shape == "short":
        row["payload"] = f"s{i % 1000:03d}"
    elif shape == "string_rich":
        row["title"] = f"Document {i:06d} about graph query execution"
        row["payload"] = f"record-{i:06d}: " + "indexed graph properties and local search. " * 3
        row["category"] = f"category-{i % 31:02d}"
        row["description"] = f"Stored item {i:06d}; " + "deterministic text for result conversion. " * 2
        row["uri"] = f"https://example.invalid/items/{i:06d}/details"
    elif shape == "unicode":
        row["title"] = f"节点-{i:06d}-Ø"
        row["payload"] = ("", "a\0b", "東京 café Ω 😀 e\u0301 " * 8)[i % 3]
    else:
        prefix = f"record-{i:06d}:"
        row["payload"] = prefix + "x" * (long_chars - len(prefix))
    return row


def projected(shape: str, route: str, i: int, long_chars: int) -> dict:
    data = record(shape, i, long_chars)
    node_type = "Item" if shape == "scan_item" else "Person"
    if shape == "scan_item" and i == 0:
        data["qty"] = -1
    if route == "full_node":
        return {"n": {"id": i, "labels": [node_type], "properties": {**data, "type": node_type}}}
    if shape.startswith("scan_"):
        if route == "scalar":
            return {"id(n)": i}
        props = ("age", "city") if shape == "scan_person" else ("qty", "body")
        return {f"n.{prop}": data[prop] for prop in props}
    value = data["age"] if shape == "numeric" else data["payload"]
    return {"id": i, "value": value} if route == "two_properties" else {"value": value}


def exact(actual, expected) -> None:
    if type(actual) is not type(expected):
        raise AssertionError(f"native Python type differs: {type(actual)} != {type(expected)}")
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError("result keys differ")
        for key in expected:
            exact(actual[key], expected[key])
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError("nested list length differs")
        for a, e in zip(actual, expected):
            exact(a, e)
    elif actual != expected:
        raise AssertionError("result value differs")


def check_rows(rows: list, shape: str, route: str, n: int, long_chars: int) -> None:
    if type(rows) is not list or len(rows) != n:
        raise AssertionError("complete result length/type differs")
    for i, row in enumerate(rows):
        exact(row, projected(shape, route, i, long_chars))


def fixture(shape: str, n: int, long_chars: int):
    import pandas as pd

    from kglite import KnowledgeGraph

    rows = [record(shape, i, long_chars) for i in range(n)]
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=True).encode() + b"\n")
    graph = KnowledgeGraph()
    node_type = "Item" if shape == "scan_item" else "Person"
    frame = pd.DataFrame(rows)
    if shape.startswith("scan_"):
        graph.add_nodes(frame, node_type, "id", "title")
    else:
        graph.add_nodes(frame, node_type, "id", "title", columns=[k for k in rows[0] if k not in ("id", "title")])
    edges = 0
    if shape == "scan_item":
        graph.add_connections(
            pd.DataFrame({"source": [i % n for i in range(2 * n)], "target": [(i * 7 + 13) % n for i in range(2 * n)]}),
            "LINK",
            "Item",
            "source",
            "Item",
            "target",
        )
        frozen = graph.freeze()
        graph.cypher("MATCH (n:Item {id:0}) SET n.qty=-1")
        del frozen
        edges = 2 * n
    if graph.shape != (n, edges) or graph.node_type_counts() != {node_type: n}:
        raise AssertionError("fixture population differs")
    return graph, digest.hexdigest()


def query_for(shape: str, route: str) -> str:
    if shape.startswith("scan_"):
        node_type = "Item" if shape == "scan_item" else "Person"
        props = "n.qty,n.body" if shape == "scan_item" else "n.age,n.city"
        projection = {"full_node": "n", "two_properties": props, "scalar": "id(n)"}[route]
        return f"MATCH (n:{node_type}) RETURN {projection}"
    value = "age" if shape == "numeric" else "payload"
    projection = {
        "full_node": "n",
        "two_properties": f"n.id AS id, n.{value} AS value",
        "scalar": f"n.{value} AS value",
    }[route]
    return f"MATCH (n:Person) RETURN {projection}"


def moments(samples: list[int], statistic: str) -> dict:
    return {
        "sample_ns": samples,
        "count": len(samples),
        "min_ns": min(samples),
        "median_ns": statistics.median(samples),
        "mean_ns": statistics.mean(samples),
        "max_ns": max(samples),
        "primary_statistic": statistic,
    }


def measure_eager(graph, shape: str, route: str, n: int, args) -> dict:
    query = query_for(shape, route)
    # Ordinary RETURN n is eager even with the public streaming default enabled.
    # Projection controls explicitly use streaming=False to isolate conversion.
    streaming = route == "full_node"
    clocks = {name: [] for name in ("creation", "first_consumption", "repeat_consumption", "complete_event")}
    rounds = max(100, args.rounds) if n == 10000 and route != "full_node" and not args.preflight else args.rounds
    warmup = max(20, args.warmup) if n == 10000 and route != "full_node" and not args.preflight else args.warmup
    for i in range(warmup + rounds):
        start = time.perf_counter_ns()
        view = graph.cypher(query, streaming=streaming)
        created = time.perf_counter_ns()
        rows = view.to_list()
        consumed = time.perf_counter_ns()
        check_rows(rows, shape, route, n, args.long_chars)
        del rows
        repeat_start = time.perf_counter_ns()
        rows = view.to_list()
        repeat_end = time.perf_counter_ns()
        check_rows(rows, shape, route, n, args.long_chars)
        del rows, view
        complete_start = time.perf_counter_ns()
        view = graph.cypher(query, streaming=streaming)
        rows = view.to_list()
        complete_end = time.perf_counter_ns()
        check_rows(rows, shape, route, n, args.long_chars)
        del rows, view
        if i >= warmup:
            for name, elapsed in zip(
                clocks, (created - start, consumed - created, repeat_end - repeat_start, complete_end - complete_start)
            ):
                clocks[name].append(elapsed)
    return {
        "family": "full_node_eager" if route == "full_node" else "eager_projection_control",
        "route": route,
        "query": query,
        "streaming": streaming,
        "every_complete_result_checked": True,
        "timings": {
            name: moments(
                values, "mean of fresh events" if name != "repeat_consumption" else "median; min also retained"
            )
            for name, values in clocks.items()
        },
    }


def prepare_cache(view, state: str, shape: str, n: int, long_chars: int) -> None:
    if state == "cold":
        return
    if state == "warm":
        rows = view.to_list()
        check_rows(rows, shape, "two_properties", n, long_chars)
        return
    indices = {"first": range(1), "alternating": range(0, n, 2), "head": range(16), "tail": range(n - 16, n)}[state]
    # Head/tail name the cache region: individual indexing warms the original
    # ResultView, without depending on whether a sliced view shares its cache.
    for i in indices:
        exact(view[i], projected(shape, "two_properties", i, long_chars))


def measure_lazy(graph, shape: str, state: str, n: int, args) -> dict:
    query = query_for(shape, "two_properties")
    first, repeated = [], []
    for i in range(args.warmup + args.rounds):
        view = graph.cypher(query, streaming=True)
        if len(view) != n or n * 2 <= 32:
            raise AssertionError("optional probe must exceed eager cell threshold")
        prepare_cache(view, state, shape, n, args.long_chars)
        start = time.perf_counter_ns()
        rows = view.to_list()
        end = time.perf_counter_ns()
        check_rows(rows, shape, "two_properties", n, args.long_chars)
        del rows
        again = time.perf_counter_ns()
        rows = view.to_list()
        again_end = time.perf_counter_ns()
        check_rows(rows, shape, "two_properties", n, args.long_chars)
        del rows, view
        if i >= args.warmup:
            first.append(end - start)
            repeated.append(again_end - again)
    return {
        "family": "optional_lazy_property_probe",
        "cache_precondition": state,
        "query": query,
        "streaming": True,
        "every_complete_result_checked": True,
        "first_bulk_after_precondition": moments(first, "mean of freshly prepared states"),
        "repeated_bulk": moments(repeated, "median; min also retained"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[10000])
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=["numeric", "string_rich"])
    parser.add_argument("--routes", nargs="+", choices=ROUTES, default=list(ROUTES))
    parser.add_argument("--lazy-states", nargs="+", choices=LAZY_STATES, default=[])
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--long-chars", type=int, default=1024)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (
        set(args.sizes) <= {10000, 100000}
        and 1 <= args.rounds <= 200
        and 0 <= args.warmup <= 20
        and 128 <= args.long_chars <= 4096
    ):
        parser.error("sizes must be 10k/100k, rounds 1..200, warmup 0..20, long-chars 128..4096")
    if any(len(values) != len(set(values)) for values in (args.sizes, args.shapes, args.routes, args.lazy_states)):
        parser.error("selectors must not contain duplicates")
    out = args.out.resolve()
    if out.exists() or out.suffixes[-2:] != [".json", ".xz"] or not out.is_relative_to(ROOT / "dev-docs/bench/out"):
        parser.error("--out must be a new .json.xz under dev-docs/bench/out")
    metadata = {
        "schema": 1,
        "label": args.label,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "release": {"profile": "debug-preflight-only"} if args.preflight else release_provenance(),
        "head": git("rev-parse", "HEAD"),
        "status": git("status", "--porcelain"),
        "diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
        "driver_sha256": sha(Path(__file__)),
        "sources": {path: sha(ROOT / path) for path in SOURCE_PATHS},
        "python": sys.version,
        "gc_enabled": gc.isenabled(),
        "gc_thresholds": gc.get_threshold(),
        "platform": platform.platform(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "load_start": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "clock_scope": (
            "creation, first/repeated to_list, independent creation+to_list; "
            "checks, cache prewarming and object disposal outside clocks"
        ),
        "oracle": (
            "all rows in input order; exact native Python types, full keys and values; "
            "expected from fixture formula, not candidate output"
        ),
        "limitations": "No allocation, retained-RSS, leak, or concurrency measurement",
    }
    cells = []
    for n in args.sizes:
        for shape in args.shapes:
            graph, fixture_digest = fixture(shape, n, args.long_chars)
            for route in args.routes:
                result = measure_eager(graph, shape, route, n, args)
                cells.append({"shape": shape, "nodes": n, "fixture_sha256": fixture_digest, **result})
                print(f"Checked {shape} {route} n={n}", flush=True)
            for state in args.lazy_states:
                result = measure_lazy(graph, shape, state, n, args)
                cells.append({"shape": shape, "nodes": n, "fixture_sha256": fixture_digest, **result})
                print(f"Checked optional lazy {shape} {state} n={n}", flush=True)
            del graph
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metadata.update(
        {
            "cells": cells,
            "load_end": os.getloadavg() if hasattr(os, "getloadavg") else None,
            "process_peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
        }
    )
    raw = json.dumps(metadata, indent=2).encode()
    encoded = lzma.compress(raw)
    if lzma.decompress(encoded) != raw:
        raise AssertionError("compressed capture roundtrip failed")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as handle:
        handle.write(encoded)
    print(f"Saved {len(cells)} complete-oracle cells to {out}")


if __name__ == "__main__":
    main()
