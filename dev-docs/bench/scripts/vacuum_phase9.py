#!/usr/bin/env python3
"""Phase 9 original vacuum lifecycle and corrected maintenance controls.

Preserves the original lifecycle fixture/query/triggers. No builds or unsafe
UNIQUE remapping. Attribution, allocator capacity and memory are not measured.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import lzma
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
KINDS = {
    "unique": "p.email IS UNIQUE",
    "composite": "(p.tenant, p.email) IS UNIQUE",
    "node_key": "(p.tenant, p.email) IS NODE KEY",
}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def stats(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "samples": values,
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p99": ordered[max(0, (99 * len(values) + 99) // 100 - 1)],
    }


def rejected(graph, statement: str, params: dict, message: str) -> None:
    try:
        graph.cypher(statement, params=params)
    except Exception as error:
        if message not in str(error).lower():
            raise AssertionError("unexpected failure instead of required constraint refusal") from error
    else:
        raise AssertionError("required constraint refusal disappeared")


def exact_values(actual, expected) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            exact_values(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(exact_values(a, e) for a, e in zip(actual, expected))
    return actual == expected


def lifecycle_values(graph, size: int, off: bool) -> dict:
    issues = size // 10
    start = size * 4 // 5
    last_fire_removed = 154 * (size // 200)
    combined = hashlib.sha256()
    for label, begin, end in (("Issue", 0, issues), ("Comment", start, size)):
        rows = graph.cypher(f"MATCH (n:{label}) RETURN n ORDER BY n.id").to_list()
        if len(rows) != end - begin:
            raise AssertionError("full survivor population differs")
        for i, row in zip(range(begin, end), rows):
            if label == "Issue":
                slot = i
                props = {"id": i, "title": f"issue-{i}", "type": "Issue", "state": "open" if i % 3 else "closed"}
            else:
                slot = issues + i - (0 if off else last_fire_removed)
                props = {"id": i, "title": f"comment body {i}", "type": "Comment", "body": f"comment body {i}"}
            expected = {"n": {"id": slot, "labels": [label], "properties": props}}
            if not exact_values(row, expected):
                raise AssertionError("node value/physical remapping/order differs")
            combined.update(json.dumps(row, sort_keys=True).encode() + b"\n")
        del rows
    rows = graph.cypher(
        "MATCH (c:Comment)-[r:ON]->(i:Issue) RETURN c.id AS comment, i.id AS issue, r ORDER BY comment"
    ).to_list()
    if len(rows) != size - start:
        raise AssertionError("full relationship population differs")
    for i, row in zip(range(start, size), rows):
        offset = 0 if off else last_fire_removed
        expected = {
            "comment": i,
            "issue": i % issues,
            "r": {"id": i - offset, "start": issues + i - offset, "end": i % issues, "type": "ON", "properties": {}},
        }
        if not exact_values(row, expected):
            raise AssertionError("relationship value/endpoints/physical order differs")
        combined.update(json.dumps(row, sort_keys=True).encode() + b"\n")
    return {"all_survivor_nodes_edges_checked": True, "ordered_full_values_sha256": combined.hexdigest()}


def lifecycle(size: int, off: bool, bench) -> dict:
    graph = bench._issue_comment_graph(size)
    if off:
        graph.set_auto_vacuum(None)
    declarations = graph.cypher("SHOW CONSTRAINTS").to_list()
    initial = graph.graph_info()
    batches = bench._lifecycle_batches(size)
    timings, observations, fires = [], [], []
    previous_fires = initial["auto_vacuums_run"]
    previous_tombstones = initial["node_tombstones"]
    keys = (
        "node_count",
        "node_capacity",
        "node_tombstones",
        "edge_count",
        "edge_capacity",
        "edge_tombstones",
        "columnar_live_rows",
        "columnar_total_rows",
        "auto_vacuums_run",
    )
    for number, ids in enumerate(batches, 1):
        # Same callable and clock boundaries as _run_lifecycle; all graph_info
        # and oracle work is outside the delete latency measurement.
        started = time.perf_counter()
        graph.cypher(bench.LIFECYCLE_DELETE, params={"ids": ids})
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        info = graph.graph_info()
        delta = info["auto_vacuums_run"] - previous_fires
        if delta not in (0, 1):
            raise AssertionError("unexpected vacuum count delta")
        if bool(delta) != (info["node_tombstones"] < previous_tombstones):
            raise AssertionError("vacuum counter and original tombstone-drop trigger disagree")
        if delta:
            fires.append(number)
            if info["node_tombstones"] or info["edge_tombstones"]:
                raise AssertionError("trigger did not reclaim all physical tombstones")
            if (
                info["columnar_total_rows"] != info["columnar_live_rows"]
                or info["columnar_live_rows"] != info["node_count"]
            ):
                raise AssertionError("trigger did not reclaim column rows")
        previous_fires = info["auto_vacuums_run"]
        previous_tombstones = info["node_tombstones"]
        observations.append({key: info[key] for key in keys})
    expected_fires = [] if off else [61, 103, 133, 154]
    if fires != expected_fires or tuple(bench.EXPECTED_FIRE_BATCHES) != (61, 103, 133, 154):
        raise AssertionError("original default trigger schedule changed")
    final = observations[-1]
    expected = bench.LIFECYCLE_EXPECTED[size]
    arm = "off" if off else "on"
    for key in ("node_count", "edge_count"):
        if final[key] != expected[key]:
            raise AssertionError("final count differs")
    if (
        final["node_capacity"] != expected[f"{arm}_capacity"]
        or final["node_tombstones"] != expected[f"{arm}_tombstones"]
    ):
        raise AssertionError("final capacity/tombstone bound differs")
    if final["node_capacity"] - final["node_count"] != final["node_tombstones"]:
        raise AssertionError("node bookkeeping inconsistent")
    for number, info in enumerate(observations, 1):
        if info["node_capacity"] - info["node_count"] != info["node_tombstones"]:
            raise AssertionError("intermediate node bookkeeping inconsistent")
        if info["edge_capacity"] - info["edge_count"] != info["edge_tombstones"]:
            raise AssertionError("intermediate edge bookkeeping inconsistent")
        if off and (
            info["node_tombstones"] != number * size // 200 or info["node_capacity"] != expected["off_capacity"]
        ):
            raise AssertionError("off arm unexpectedly reclaimed slots")
    if graph.cypher("SHOW CONSTRAINTS").to_list() != declarations:
        raise AssertionError("lifecycle changed constraint declarations")
    parity_rows = graph.cypher(bench.LIFECYCLE_PARITY_QUERY).to_list()
    # Primary-id constraints use their dedicated error, independently of the
    # general UNIQUE-occupancy reconstruction covered by maintenance cells.
    rejected(graph, "CREATE (:Comment {id:$id})", {"id": size * 4 // 5}, "duplicate primary key")
    oracle = lifecycle_values(graph, size, off)
    trigger = [timings[number - 1] for number in fires]
    ordinary = [elapsed for number, elapsed in enumerate(timings, 1) if number not in fires]
    return {
        "family": "original_lifecycle",
        "size": size,
        "arm": "off" if off else "default_on",
        "query": bench.LIFECYCLE_DELETE,
        "batches": len(batches),
        "victims_sha256": digest(batches),
        "initial": {key: initial[key] for key in keys},
        "observations": observations,
        "declarations": declarations,
        "fires": fires,
        "vacuum_total_delete_s": sum(timings),
        "all_batches_s": stats(timings),
        "ordinary_batches_s": stats(ordinary),
        "trigger_batches_s": stats(trigger) if trigger else None,
        "parity_rows": parity_rows,
        "oracle": oracle,
    }


def maintenance_values(graph, size: int, operation: str) -> None:
    ids = [i for i in range(size) if operation != "vacuum" or i % 5]
    rows = graph.cypher("MATCH (p:Person) RETURN p ORDER BY p.id").to_list()
    if len(rows) != len(ids):
        raise AssertionError("maintenance survivor count differs")
    for slot, (i, row) in enumerate(zip(ids, rows)):
        props = {
            "id": i,
            "title": f"n{i}",
            "type": "Person",
            "nid": i,
            "name": f"n{i}",
            "email": f"e{i}",
            "tenant": "t",
        }
        if not exact_values(row, {"p": {"id": slot, "labels": ["Person"], "properties": props}}):
            raise AssertionError("maintenance full survivor properties/slots/order differ")


def maintenance(size: int, kind: str, operation: str, events: int, bench, kg) -> dict:
    frame = bench.maintenance_frame.__wrapped__(SimpleNamespace(param=size))
    samples, writes = [], []
    for _ in range(events):
        graph = kg.KnowledgeGraph()
        graph.set_auto_vacuum(None)
        graph.add_nodes(frame, "Person", "nid", "name")
        if kind != "none":
            graph.cypher(f"CREATE CONSTRAINT person_key FOR (p:Person) REQUIRE {KINDS[kind]}")
        if operation == "vacuum":
            graph.cypher("MATCH (p:Person) WHERE p.id % 5 = 0 DETACH DELETE p")
        declarations = graph.cypher("SHOW CONSTRAINTS").to_list()
        started = time.perf_counter_ns()
        getattr(graph, operation)()
        elapsed = time.perf_counter_ns() - started
        maintenance_values(graph, size, operation)
        if graph.cypher("SHOW CONSTRAINTS").to_list() != declarations:
            raise AssertionError("maintenance changed declarations")
        first, second = (1, 2) if operation == "vacuum" else (0, 1)
        write_started = time.perf_counter_ns()
        graph.cypher("MATCH (p:Person {id:$id}) SET p.email=$email", params={"id": first, "email": f"e{first}"})
        writes.append(time.perf_counter_ns() - write_started)
        if kind != "none":
            rejected(
                graph,
                "MATCH (p:Person {id:$id}) SET p.email=$email",
                {"id": second, "email": f"e{first}"},
                "constraint",
            )
        maintenance_values(graph, size, operation)
        samples.append(elapsed)
        del graph
    return {
        "family": "corrected_maintenance",
        "size": size,
        "kind": kind,
        "operation": operation,
        "timing_ns": stats(samples),
        "post_maintenance_own_value_write_ns": stats(writes),
        "statistic": "mean of fresh complete events",
        "full_values_declarations_enforcement_checked": True,
        "fixture_sha256": digest(frame.to_dict("records")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", choices=("lifecycle", "maintenance"), default=["lifecycle"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000])
    parser.add_argument("--arms", nargs="+", choices=("default_on", "off"), default=["default_on", "off"])
    parser.add_argument("--maintenance-sizes", nargs="+", type=int, default=[1000])
    parser.add_argument("--kinds", nargs="+", choices=("none", *KINDS), default=["none", "unique", "composite"])
    parser.add_argument("--operations", nargs="+", choices=("vacuum", "reindex"), default=["vacuum", "reindex"])
    parser.add_argument("--events", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (
        set(args.sizes) <= {1000, 100000, 1000000}
        and set(args.maintenance_sizes) <= {1000, 10000, 50000}
        and 1 <= args.events <= 20
        and 1 <= args.repeat <= 2
    ):
        parser.error("use bounded original sizes, maintenance sizes 1k/10k/50k, events 1..20 and repeat 1..2")
    if any(
        len(values) != len(set(values))
        for values in (args.families, args.sizes, args.arms, args.maintenance_sizes, args.kinds, args.operations)
    ):
        parser.error("selectors must be distinct")
    out = args.out.resolve()
    if out.exists() or out.suffixes[-2:] != [".json", ".xz"] or not out.is_relative_to(ROOT / "dev-docs/bench/out"):
        parser.error("--out must be a new .json.xz under dev-docs/bench/out")
    from reused_slot_delete import release_provenance, sha

    import kglite
    from tests.benchmarks import test_bench_delete_scaling as lifecycle_bench
    from tests.benchmarks import test_bench_reindex_constraints as maintenance_bench

    metadata = {
        "schema": 1,
        "label": args.label,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "release": release_provenance(),
        "head": git("rev-parse", "HEAD"),
        "status": git("status", "--porcelain"),
        "diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
        "sources": {
            str(path.relative_to(ROOT)): sha(path)
            for path in (
                Path(__file__),
                ROOT / "dev-docs/bench/scripts/reused_slot_delete.py",
                Path(lifecycle_bench.__file__),
                Path(maintenance_bench.__file__),
                ROOT / "crates/kglite/src/graph/dir_graph/mod.rs",
                ROOT / "crates/kglite/src/graph/dir_graph/columnar_rebuild.rs",
                ROOT / "crates/kglite/src/graph/dir_graph/constraints.rs",
            )
        },
        "python": sys.version,
        "platform": platform.platform(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "load_start": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "scope": (
            "original 160 delete-query latency sum and actual triggers; fresh maintenance event "
            "and separate same-value write latency; setup/info/oracles/disposal excluded"
        ),
        "limitations": (
            "This harness does not split internal phases or measure allocation/capacity/RSS; "
            "no unchecked UNIQUE translation"
        ),
    }
    cells = []
    for repeat in range(args.repeat):
        if "lifecycle" in args.families:
            for size in args.sizes:
                for arm in args.arms:
                    cells.append({"repeat": repeat, **lifecycle(size, arm == "off", lifecycle_bench)})
                    print(f"Checked lifecycle {size} {arm}", flush=True)
        if "maintenance" in args.families:
            for size in args.maintenance_sizes:
                for kind in args.kinds:
                    for operation in args.operations:
                        cells.append(
                            {
                                "repeat": repeat,
                                **maintenance(size, kind, operation, args.events, maintenance_bench, kglite),
                            }
                        )
                        print(f"Checked maintenance {size} {kind} {operation}", flush=True)
    metadata.update({"cells": cells, "load_end": os.getloadavg() if hasattr(os, "getloadavg") else None})
    raw = json.dumps(metadata, indent=2).encode()
    encoded = lzma.compress(raw)
    if lzma.decompress(encoded) != raw:
        raise AssertionError("compressed capture roundtrip failed")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as handle:
        handle.write(encoded)
    print(f"Saved {len(cells)} full-oracle cells to {out}")


if __name__ == "__main__":
    main()
