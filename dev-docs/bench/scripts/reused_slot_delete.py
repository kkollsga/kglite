#!/usr/bin/env python3
"""Release-only Phase 4 reused-slot deletion benchmark. Never builds automatically.

Run only with a current release extension during the worker's measurement lease.
Default is a small 1k-node capture. See performance-phase-4-driver-handoff.md.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "tests/benchmarks/test_bench_delete_scaling.py"
sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release_provenance() -> dict:
    extension = Path(importlib.import_module("kglite.kglite").__file__).resolve()
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    artifact = ROOT / "target/release" / f"libkglite_py{suffix}"
    if not artifact.is_file() or sha(extension) != sha(artifact):
        raise RuntimeError(
            "installed extension does not match the release cdylib; worker must build/install release first"
        )
    sources = [ROOT / "Cargo.toml", ROOT / "crates/kglite/Cargo.toml", ROOT / "crates/kglite-py/Cargo.toml"]
    for crate in ("kglite", "kglite-py"):
        sources.extend((ROOT / "crates" / crate / "src").rglob("*.rs"))
    if artifact.stat().st_mtime_ns < max(path.stat().st_mtime_ns for path in sources):
        raise RuntimeError("release cdylib predates engine/wrapper source; refuse stale measurements")
    return {
        "extension": str(extension),
        "release_artifact": str(artifact.resolve()),
        "sha256": sha(extension),
        "version": importlib.import_module("kglite").__version__,
    }


def load_helper():
    spec = importlib.util.spec_from_file_location("phase4_delete_fixture", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._issue_comment_graph, module.PROBE_BASE


def snapshot(graph) -> list[dict]:
    # No ORDER BY: preserve the actual type-bucket scan order, alongside slots.
    return graph.cypher("MATCH (c:Comment) RETURN c.id AS id, elementId(c) AS slot, c.title AS title").to_list()


def create(graph, identifier: int) -> None:
    graph.cypher('CREATE (:Comment {id:$id,body:"probe"})', params={"id": identifier})


def delete(graph, identifier: int) -> None:
    graph.cypher("MATCH (c:Comment {id:$id}) DELETE c", params={"id": identifier})


def moments(values: list[int]) -> dict:
    ordered = sorted(values)
    return {
        "samples": len(values),
        "sum_ns": sum(values),
        "min_ns": min(values),
        "median_ns": statistics.median(values),
        "mean_ns": statistics.mean(values),
        "p95_ns": ordered[math.ceil(len(values) * 0.95) - 1],
        "max_ns": max(values),
    }


def assert_state(graph, expected: list[dict], edges: int) -> dict:
    actual = snapshot(graph)
    if actual != expected:
        raise AssertionError("survivor IDs, physical slots, titles or scan order changed")
    info = graph.graph_info()
    if info["edge_count"] != edges or info["auto_vacuums_run"] != 0:
        raise AssertionError("unexpected surviving edge count or auto-vacuum")
    return {
        "comments": len(actual),
        "edges": edges,
        "node_capacity": info["node_capacity"],
        "ordered_survivors_sha256": hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest(),
        "all_ids_slots_titles_order_checked": True,
    }


def fresh(make_graph, size: int):
    graph = make_graph(size)
    graph.set_auto_vacuum(None)
    original = snapshot(graph)
    if [row["id"] for row in original] != list(range(size)):
        raise AssertionError("seed does not have the expected initial type scan order")
    return graph, original


def recurring(make_graph, base: int, size: int, rounds: int, warmup: int, reuse: bool) -> dict:
    graph, original = fresh(make_graph, size)
    expected = original
    edges = size
    if reuse:
        hole = original[0]["slot"]
        graph.cypher("MATCH (c:Comment {id:0}) DETACH DELETE c")
        expected = original[1:]
        edges -= 1
        # Untimed exact-slot preflight: the one free node slot must be reused.
        create(graph, base - 1)
        reused = graph.cypher(
            "MATCH (c:Comment {id:$id}) RETURN elementId(c) AS slot", params={"id": base - 1}
        ).scalar()
        if reused != hole:
            raise AssertionError("preflight did not reuse the removed low node slot")
        delete(graph, base - 1)
    assert_state(graph, expected, edges)
    capacity = graph.graph_info()["node_capacity"]
    creates, deletes, complete = [], [], []
    for i in range(warmup + rounds):
        # IDs and the query shape match the original full-perf-churn probe.
        identifier = base + i
        start = time.perf_counter_ns()
        create(graph, identifier)
        split = time.perf_counter_ns()
        delete(graph, identifier)
        end = time.perf_counter_ns()
        if i >= warmup:
            creates.append(split - start)
            deletes.append(end - split)
            complete.append(end - start)
    verified = assert_state(graph, expected, edges)
    expected_capacity = capacity
    if verified["node_capacity"] != expected_capacity:
        raise AssertionError("unexpected final node bound after removing every probe")
    return {
        "case": "reused_low_slot_tail" if reuse else "sorted_append_delete_tail",
        "size": size,
        "create": moments(creates),
        "delete": moments(deletes),
        "full_cycle": moments(complete),
        "oracle": verified,
    }


def append_control(make_graph, base: int, size: int, rounds: int, warmup: int) -> dict:
    graph, original = fresh(make_graph, size)
    initial_capacity = graph.graph_info()["node_capacity"]
    timings = []
    for i in range(rounds + warmup):
        start = time.perf_counter_ns()
        create(graph, base + i)
        elapsed = time.perf_counter_ns() - start
        if i >= warmup:
            timings.append(elapsed)
    after = snapshot(graph)
    appended = after[size:]
    if after[:size] != original or [row["id"] for row in appended] != list(range(base, base + rounds + warmup)):
        raise AssertionError("ordinary appends changed original order or appended identities")
    if any(row["title"] != "probe" for row in appended) or [row["slot"] for row in appended] != [
        str(initial_capacity + i) for i in range(rounds + warmup)
    ]:
        raise AssertionError("append titles or physical slot identities are wrong")
    verified = assert_state(graph, after, size)
    return {"case": "ordinary_append_only", "size": size, "create": moments(timings), "oracle": verified}


def positional(make_graph, base: int, size: int, shape: str) -> tuple[int, dict]:
    graph, original = fresh(make_graph, size)
    initial_capacity = graph.graph_info()["node_capacity"]
    hole = original[0]["slot"]
    graph.cypher("MATCH (c:Comment {id:0}) DETACH DELETE c")
    # The first probe reuses a low slot; 64 later appends move it outside a
    # short suffix. This exercises the existing full retain fallback candidate.
    for i in range(65):
        create(graph, base + i)
    before = snapshot(graph)
    expected_ids = list(range(1, size)) + list(range(base, base + 65))
    if [row["id"] for row in before] != expected_ids or before[size - 1]["slot"] != hole:
        raise AssertionError("positional fixture did not establish ordered low-slot reuse")
    if before[: size - 1] != original[1:]:
        raise AssertionError("positional fixture moved original survivors")
    appended = before[size - 1 :]
    if [row["slot"] for row in appended] != [hole] + [str(initial_capacity + i) for i in range(64)] or any(
        row["title"] != "probe" for row in appended
    ):
        raise AssertionError("positional append slots or values differ")
    victims = {"early": [1], "middle": [size // 2], "outside_suffix": [base], "multi": [size // 2, base, base + 64]}[
        shape
    ]
    start = time.perf_counter_ns()
    graph.cypher("MATCH (c:Comment) WHERE c.id IN $ids DETACH DELETE c", params={"ids": victims})
    elapsed = time.perf_counter_ns() - start
    victim_set = set(victims)
    expected = [row for row in before if row["id"] not in victim_set]
    edges = size - 1 - sum(identifier < size for identifier in victims)
    return elapsed, assert_state(graph, expected, edges)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000])
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--position-rounds", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.sizes
        or len(set(args.sizes)) != len(args.sizes)
        or not set(args.sizes) <= {1000, 10_000, 100_000}
        or not 1 <= args.rounds <= 200
        or not 0 <= args.warmup <= 20
        or not 1 <= args.position_rounds <= 5
    ):
        parser.error("use distinct sizes from 1k/10k/100k and bounded rounds")
    out = args.out.resolve()
    if out.exists() or not any(out.is_relative_to(ROOT / root) for root in ("dev-docs/bench/out", "dev-docs/temp")):
        parser.error("--out must be a new file under bench/out or temp")
    provenance = release_provenance()
    make_graph, base = load_helper()
    cells = []
    for size in args.sizes:
        for reuse in (True, False):
            cells.append(recurring(make_graph, base, size, args.rounds, args.warmup, reuse))
        cells.append(append_control(make_graph, base, size, args.rounds, args.warmup))
        for shape in ("early", "middle", "outside_suffix", "multi"):
            results = [positional(make_graph, base, size, shape) for _ in range(args.position_rounds)]
            cells.append(
                {
                    "case": shape,
                    "size": size,
                    "delete": moments([r[0] for r in results]),
                    "oracles": [r[1] for r in results],
                }
            )
    metadata = {
        "schema": 1,
        "release": provenance,
        "driver_sha256": sha(Path(__file__)),
        "helper_sha256": sha(HELPER),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "timing_scope": "query result destruction included; setup and exact survivor/slot/order checks excluded",
        "claims": "recent single-slot reuse only; positional controls include their MATCH scan and detach work",
        "cells": cells,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(f"Saved {len(cells)} oracle-checked cells to {out}")


if __name__ == "__main__":
    main()
