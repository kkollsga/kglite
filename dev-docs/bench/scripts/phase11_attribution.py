#!/usr/bin/env python3
"""Bounded Phase 11 snapshot and label measurements with exact fixture oracles.

Candidate release only. Original helper fixtures are loaded at execution time.
No builds, extension installation, layer-cap changes or graph persistence.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import itertools
import json
import lzma
import math
from pathlib import Path
import statistics
import subprocess
import time

from reused_slot_delete import ROOT, release_provenance, sha

HOT = ROOT / "tests/benchmarks/test_bench_hotpaths.py"
LABELS = ROOT / "tests/benchmarks/test_bench_labels.py"
WRITE = "MATCH (n:Item {id: 1234}) SET n.rank_val = $v"
READ = "MATCH (n:Item {id: 1234}) RETURN n.rank_val AS v"
CAP = 32  # Observation grouping only; never writes an engine setting.


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def stats(values: list[int]) -> dict:
    ordered = sorted(values)
    return {
        "n": len(values),
        "sum_ns": sum(values),
        "mean_ns": statistics.mean(values),
        "median_ns": statistics.median(values),
        "min_ns": min(values),
        "p95_ns": ordered[math.ceil(len(values) * 0.95) - 1],
        "p99_ns": ordered[math.ceil(len(values) * 0.99) - 1],
        "max_ns": max(values),
    }


def event(fn) -> tuple[int, int]:
    cpu = time.process_time_ns()
    wall = time.perf_counter_ns()
    fn()
    return time.perf_counter_ns() - wall, time.process_time_ns() - cpu


def time_read(fn, rounds=32) -> dict:
    samples = [event(fn) for _ in range(rounds)]
    return {"wall": stats([x[0] for x in samples]), "cpu": stats([x[1] for x in samples])}


def hot_fixture(hot, size: int, index: str):
    previous = hot.N
    hot.N = size
    try:
        graph = hot.hot_graph.__wrapped__()
    finally:
        hot.N = previous
    graph.build_id_indices(["Item"])
    if index != "none":
        method = graph.create_index if index == "eq" else graph.create_range_index
        method("Item", "rank_val")
    return graph


def hot_oracle(graph, size: int, final: int, edge_count: int) -> str:
    actual = graph.cypher("MATCH (n:Item) RETURN n.id AS id, n.rank_val AS v ORDER BY n.id").to_list()
    expected = [{"id": i, "v": final if i == 1234 else (i * 7919) % size} for i in range(size)]
    if actual != expected or graph.cypher("MATCH ()-[r:LINKS]->() RETURN count(r)").scalar() != edge_count:
        raise AssertionError("snapshot fixture final values/topology differ")
    return hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()


def snapshot_cell(hot, args, index: str, lifetime: str) -> dict:
    graph = hot_fixture(hot, args.nodes, index)
    initial = (1234 * 7919) % args.nodes
    if graph.cypher(READ).scalar() != initial:
        raise AssertionError("seed target missing; do not use a fixture below ID 1234")
    before_info = graph.graph_info()
    edge_count = graph.cypher("MATCH ()-[r:LINKS]->() RETURN count(r)").scalar()
    if edge_count <= 0:
        raise AssertionError("original edge fixture is empty")
    read_before = time_read(lambda: graph.cypher(READ).scalar())
    counter = itertools.count()
    holder = None
    hold_acquire = (0, 0)
    if lifetime == "retained":
        start = time.perf_counter_ns()
        cpu = time.process_time_ns()
        holder = graph.select("Item")
        hold_acquire = (time.perf_counter_ns() - start, time.process_time_ns() - cpu)

    def write():
        graph.cypher(WRITE, params={"v": 900_000 + next(counter) % 2})

    def dropped_then_write():
        view = graph.select("Item")
        write()
        del view
        write()

    # The fresh arm invokes the exact original helper. The drop arm includes
    # both writes, holder acquisition and destruction in the same event.
    operation = (
        (lambda: hot._fork_then_write(graph, counter))
        if lifetime == "fresh"
        else (dropped_then_write if lifetime == "drop_then_write" else write)
    )
    samples = []
    for i in range(args.warmup + args.rounds):
        gc_before = sum(g["collections"] for g in gc.get_stats())
        wall, cpu = event(operation)
        samples.append(
            {
                "absolute_round": i,
                "wall_ns": wall,
                "cpu_ns": cpu,
                "gc_collections": sum(g["collections"] for g in gc.get_stats()) - gc_before,
            }
        )
    writes = (args.warmup + args.rounds) * (2 if lifetime == "drop_then_write" else 1)
    final = 900_000 + (writes - 1) % 2
    held_verified = None
    if holder is not None:
        held_verified = holder.cypher(READ).scalar() == initial
        if not held_verified:
            raise AssertionError("retained snapshot changed with writer")
    cpu = time.process_time_ns()
    start = time.perf_counter_ns()
    del holder
    hold_drop = (time.perf_counter_ns() - start, time.process_time_ns() - cpu)
    oracle = hot_oracle(graph, args.nodes, final, edge_count)
    read_after = time_read(lambda: graph.cypher(READ).scalar())
    index_read = "MATCH (n:Item) WHERE n.rank_val = $v RETURN count(n) AS c"
    if graph.cypher(index_read, params={"v": final}).scalar() != 1:
        raise AssertionError("indexed lookup lost the changed value")
    lookup = time_read(lambda: graph.cypher(index_read, params={"v": final}).scalar())
    measured = samples[args.warmup :]
    # Only full aligned cycles wholly inside the measured interval. Report
    # every other sample too, so warmup offsets and final partial cycles remain visible.
    first = math.ceil(args.warmup / CAP) * CAP
    cycles = [
        {
            "start": i,
            "wall_ns": sum(s["wall_ns"] for s in samples[i : i + CAP]),
            "cpu_ns": sum(s["cpu_ns"] for s in samples[i : i + CAP]),
        }
        for i in range(first, len(samples) - CAP + 1, CAP)
    ]
    return {
        "index": index,
        "lifetime": lifetime,
        "writes_per_event": 2 if lifetime == "drop_then_write" else 1,
        "wall": stats([s["wall_ns"] for s in measured]),
        "cpu": stats([s["cpu_ns"] for s in measured]),
        "full_aligned_cycles": cycles,
        "samples_including_warmup": samples,
        "holder_acquire_wall_cpu_ns": hold_acquire,
        "holder_drop_wall_cpu_ns": hold_drop,
        "lifecycle_wall_ns": sum(s["wall_ns"] for s in samples) + hold_acquire[0] + hold_drop[0],
        "held_snapshot_verified": held_verified,
        "final_values_sha256": oracle,
        "read_before": read_before,
        "read_after": read_after,
        "indexed_read_after": lookup,
        "graph_info_before": before_info,
        "graph_info_after": graph.graph_info(),
    }


def fresh_holder_oracle(hot, args, index: str) -> None:
    # Separate untimed companion leaves the original timed helper untouched.
    graph = hot_fixture(hot, args.nodes, index)
    old = (1234 * 7919) % args.nodes
    for i in range(CAP * 2):
        holder = graph.select("Item")
        new = 900_000 + i % 2
        graph.cypher(WRITE, params={"v": new})
        if holder.cypher(READ).scalar() != old or graph.cypher(READ).scalar() != new:
            raise AssertionError("fresh holder changed or writer lost value")
        del holder
        old = new


def label_oracle(graph, query: str, expected, projection: bool, parallel: bool) -> None:
    result = graph.cypher(query, parallel=True) if parallel else graph.cypher(query)
    if projection:
        rows = result.to_list()
        actual = {row["id"]: set(row["labels"]) for row in rows}
        if (
            len(rows) != len(expected)
            or actual != expected
            or any(
                type(row["id"]) is not int
                or type(row["labels"]) is not list
                or len(row["labels"]) != len(expected[row["id"]])
                or any(type(label) is not str for label in row["labels"])
                for row in rows
            )
        ):
            raise AssertionError("label projection membership differs")
    else:
        count = result.scalar()
        if type(count) is not int or count != expected:
            raise AssertionError(f"wrong label/control count or type: {query}")


def label_cells(labels, args) -> list[dict]:
    cells = []
    for coverage in args.coverage:
        graph = labels._flat_graph(args.nodes)
        ids = {
            "zero": [],
            "sparse": list(range(0, args.nodes, 100)),
            "half": list(range(args.nodes // 2)),
            "all": list(range(args.nodes)),
        }[coverage]
        if ids and graph.add_label("P", ids, "VIP")["labelled"] != len(ids):
            raise AssertionError("VIP fixture stamping failed")
        extra = 0
        if args.mixed:
            graph.cypher("UNWIND range(0,31) AS i CREATE (:Q {id:i})")
            if graph.add_label("Q", list(range(32)), "VIP")["labelled"] != 32:
                raise AssertionError("mixed type VIP fixture failed")
            graph.add_label("Q", [0], "Unrelated")
            extra = 32
        membership = set(ids)
        projection = {i: {"P", "VIP"} if i in membership else {"P"} for i in range(args.nodes)}
        queries = [
            ("target", "MATCH (n:P) WHERE n:VIP RETURN count(n) AS c", len(ids), False),
            ("primary_count", "MATCH (n:P) RETURN count(n) AS c", args.nodes, False),
            ("primary_predicate", "MATCH (n:P) WHERE n:P RETURN count(n) AS c", args.nodes, False),
            ("secondary_count", "MATCH (n:VIP) RETURN count(n) AS c", len(ids) + extra, False),
            ("numeric", "MATCH (n:P) WHERE n.id % 2 = 0 RETURN count(n) AS c", (args.nodes + 1) // 2, False),
            ("projection", "MATCH (n:P) RETURN n.id AS id, labels(n) AS labels", projection, True),
        ]
        for name, query, expected, is_projection in queries:

            def call():
                result = graph.cypher(query, parallel=True) if args.parallel else graph.cypher(query)
                if is_projection:
                    result.to_list()  # Projection control must materialize, not time a lazy descriptor.

            cold = event(call)
            label_oracle(graph, query, expected, is_projection, args.parallel)
            plan = graph.cypher("EXPLAIN " + query).to_list()
            if name in {"target", "primary_predicate"} and not any(
                "FusedNodeScanAggregate" in row["operation"] for row in plan
            ):
                raise AssertionError("label timing must use the intended fused scan consumer")
            for _ in range(args.warmup):
                call()
            values = [event(call) for _ in range(args.rounds)]
            label_oracle(graph, query, expected, is_projection, args.parallel)
            cells.append(
                {
                    "coverage": coverage,
                    "mixed": args.mixed,
                    "query": name,
                    "text": query,
                    "first_query_wall_cpu_ns": cold,
                    "warm_wall": stats([x[0] for x in values]),
                    "warm_cpu": stats([x[1] for x in values]),
                    "plan": plan,
                    "oracle_passed": True,
                    "expected_count": len(expected) if is_projection else expected,
                }
            )
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["snapshots", "labels"])
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--indexes", nargs="+", choices=["none", "eq", "range"], default=["eq", "range"])
    parser.add_argument(
        "--lifetimes", nargs="+", choices=["fresh", "none", "retained", "drop_then_write"], default=["fresh"]
    )
    parser.add_argument("--coverage", nargs="+", choices=["zero", "sparse", "half", "all"], default=["all"])
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.nodes = args.nodes if args.nodes is not None else (50_000 if args.kind == "snapshots" else 20_000)
    args.rounds = args.rounds if args.rounds is not None else (128 if args.kind == "snapshots" else 100)
    if not 0 <= args.warmup <= 64 or not 1 <= args.nodes <= 100_000 or not 1 <= args.rounds <= 8192:
        parser.error("unbounded node/round parameters")
    if args.kind == "snapshots" and (args.nodes <= 1234 or args.rounds < CAP * 4):
        parser.error("snapshot fixture needs ID1234 and at least three complete measured cycles")
    if args.kind == "labels" and args.rounds > 200:
        parser.error("label loops are bounded at 200 rounds")
    out = args.out.resolve()
    if out.exists() or not any(out.is_relative_to(ROOT / root) for root in ("dev-docs/bench/out", "dev-docs/temp")):
        parser.error("output must be a new file in bench/out or temp")
    provenance = release_provenance()
    if args.kind == "snapshots":
        hot = module(HOT, "phase11_hot")
        for index in args.indexes:
            fresh_holder_oracle(hot, args, index)
        cells = [snapshot_cell(hot, args, index, lifetime) for index in args.indexes for lifetime in args.lifetimes]
    else:
        cells = label_cells(module(LABELS, "phase11_labels"), args)
    result = {
        "schema": 1,
        "release": provenance,
        "driver_sha256": sha(Path(__file__)),
        "fixture_sha256": sha(HOT if args.kind == "snapshots" else LABELS),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "scope": "candidate release attribution, not published-wheel/source attribution; no cap or ownership changes",
        "cells": cells,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(out, "xt") if out.suffix == ".xz" else out.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(f"Saved {len(cells)} oracle-checked {args.kind} cells to {out}")


if __name__ == "__main__":
    main()
