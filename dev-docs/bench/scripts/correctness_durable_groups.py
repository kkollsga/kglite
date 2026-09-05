#!/usr/bin/env python3
"""Release-only full write cost and exact checkpoint+WAL replay controls.

Every measured write changes the first inserted edge's stamp. This is a valid
legacy baseline even for a parallel group: its single-edge resolver updates
that same first member. The preflight rejects any different old behavior.
No checkpoint occurs after measured writes. Children exit without teardown;
parent recovery must reproduce all n/stamp values exactly. Files live in a
TemporaryDirectory; compressed records belong to existing bench result tiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "Cargo.toml").exists())
QUERY = "MATCH (a:Item {id:0})-[r:LINK]->() WHERE r.n=0 SET r.stamp=$stamp RETURN r.n AS n,r.stamp AS stamp"
ORACLE = "MATCH (a:Item)-[r:LINK]->(b:Item) RETURN r.n AS n,r.stamp AS stamp,b.id AS target ORDER BY n"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def child(args):
    import os

    import kglite

    graph = kglite.open(str(args.path), durable="normal")
    graph.cypher("CREATE(:Item{id:0}),(:Item{id:1})")
    if args.fanout:
        graph.cypher("UNWIND range(2,$last) AS i CREATE(:Item{id:i})", params={"last": args.fanout + 1})
        graph.cypher("MATCH(a:Item{id:0}),(b:Item) WHERE b.id>0 CREATE(a)-[:LINK{n:b.id-1,stamp:0}]->(b)")
    else:
        graph.cypher(
            "MATCH(a:Item{id:0}),(b:Item{id:1}) UNWIND range(0,$last) AS i CREATE(a)-[:LINK{n:i,stamp:0}]->(b)",
            params={"last": args.group - 1},
        )
    count = args.fanout + 1 if args.fanout else args.group
    fixture = [{"n": i, "stamp": 0, "target": i + 1 if args.fanout else 1} for i in range(count)]
    actual_fixture = graph.cypher(ORACLE).to_list()
    assert digest(actual_fixture) == digest(fixture), "exact fixture values/types"
    graph.save()
    total = args.warmup + args.rounds
    samples = []
    for stamp in range(1, total + 1):
        start = time.perf_counter_ns()
        actual = graph.cypher(QUERY, params={"stamp": stamp}).to_list()
        elapsed = time.perf_counter_ns() - start
        assert digest(actual) == digest([{"n": 0, "stamp": stamp}]), actual
        if stamp > args.warmup:
            samples.append(elapsed)
    rows = graph.cypher(ORACLE).to_list()
    result = {
        "samples_ns": samples,
        "min_ns": min(samples),
        "median_ns": statistics.median(samples),
        "mean_ns": statistics.mean(samples),
        "rows": rows,
        "fixture_sha256": digest(fixture),
        "live_result_sha256": digest(rows),
        "wal_bytes": Path(str(args.path) + "-wal").stat().st_size,
    }
    args.child_output.write_text(json.dumps(result), encoding="utf-8")
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, nargs="+", default=[1, 10, 1000, 10000])
    parser.add_argument("--fanouts", type=int, nargs="*", default=[1000, 10000])
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--group", type=int)
    parser.add_argument("--fanout", type=int)
    parser.add_argument("--path", type=Path)
    parser.add_argument("--child-output", type=Path)
    args = parser.parse_args()
    if args.child:
        child(args)
    if args.output is None or args.rounds < 1 or args.warmup < 0 or min(args.groups + args.fanouts, default=0) < 1:
        parser.error("positive sizes/rounds, nonnegative warmup and output required")
    if args.output.exists():
        parser.error("output already exists; captures are exclusive")
    if len(set(args.groups)) != len(args.groups) or len(set(args.fanouts)) != len(args.fanouts):
        parser.error("duplicate workload sizes would reuse a fixture")
    allowed = [ROOT / "dev-docs/bench/results", ROOT / "dev-docs/bench/out"]
    if not any(args.output.resolve().is_relative_to(path) for path in allowed):
        parser.error("output must use existing bench result tiers")
    sys.path[:0] = [str(ROOT), str(ROOT / "dev-docs/bench/scripts")]
    from reused_slot_delete import release_provenance, sha

    import kglite

    provenance = release_provenance()
    cells = []
    with tempfile.TemporaryDirectory(prefix="kglite-group-cost-") as directory:
        base = Path(directory)
        for kind, size in [("parallel", n) for n in args.groups] + [("fanout_control", n) for n in args.fanouts]:
            path = base / f"{kind}-{size}.kgl"
            capture = base / f"{kind}-{size}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                "--group",
                str(size if kind == "parallel" else 1),
                "--fanout",
                str(size if kind == "fanout_control" else 0),
                "--rounds",
                str(args.rounds),
                "--warmup",
                str(args.warmup),
                "--path",
                str(path),
                "--child-output",
                str(capture),
            ]
            process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
            if process.returncode:
                raise RuntimeError(f"{kind}/{size}: child exit {process.returncode}: {process.stderr}")
            data = json.loads(capture.read_text(encoding="utf-8"))
            count = size if kind == "parallel" else size + 1
            expected = [
                {
                    "n": i,
                    "stamp": args.rounds + args.warmup if i == 0 else 0,
                    "target": 1 if kind == "parallel" else i + 1,
                }
                for i in range(count)
            ]
            assert digest(data.pop("rows")) == digest(expected), f"{kind}/{size}: live absolute oracle"
            recovered = kglite.open(str(path), durable="normal")
            actual = recovered.cypher(ORACLE).to_list()
            assert digest(actual) == digest(expected), f"{kind}/{size}: exact replay oracle failed: {actual[:4]}"
            del recovered
            cells.append(
                {
                    "kind": kind,
                    "size": size,
                    "child_exit": process.returncode,
                    "replay_oracle_passed": True,
                    "expected_count": count,
                    "expected_sha256": digest(expected),
                    "replay_result_sha256": digest(actual),
                    **data,
                }
            )
    record = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
        "diff": subprocess.check_output(["git", "diff", "--stat"], text=True, cwd=ROOT),
        "release": provenance,
        "harness_sha256": sha(Path(__file__)),
        "provenance_helper_sha256": sha(ROOT / "dev-docs/bench/scripts/reused_slot_delete.py"),
        "arguments": {"groups": args.groups, "fanouts": args.fanouts, "rounds": args.rounds, "warmup": args.warmup},
        "statistic": "mean of complete distinct commit events; min/median retained",
        "timing_scope": (
            "cypher mutation, WAL normalization/encoding/write and consumed one-row result; replay outside clock"
        ),
        "cells": cells,
    }
    payload = (json.dumps(record, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as destination:
        destination.write(lzma.compress(payload) if args.output.suffix == ".xz" else payload)
    print(json.dumps({"output": str(args.output), "cells": len(cells), "oracles_passed": True}))


if __name__ == "__main__":
    main()
