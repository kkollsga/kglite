#!/usr/bin/env python3
"""Release-only blueprint repair cost controls.

Every sample builds a new graph from ordinary valid CSV in an owned temporary
folder. Fixture preparation, full graph/CSV oracles, teardown and cleanup are
outside the clock. Captures require current release provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import lzma
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

from reused_slot_delete import ROOT, release_provenance, sha

CELLS = ("declared_int", "declared_float_control", "derive", "aggregate")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fixture(size: int, groups: int):
    rows = [[i, f"g{i % groups}", i % 101] for i in range(size)]
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["id", "g", "v"])
    writer.writerows(rows)
    return stream.getvalue().encode(), rows


def specification(root: Path, cell: str) -> dict:
    properties = {"g": "string", "v": "float" if cell == "declared_float_control" else "int"}
    operations = []
    if cell == "derive":
        operations = [{"op": "derive", "from": "T", "set": {"twice": "v * 2"}}]
    elif cell == "aggregate":
        operations = [
            {
                "op": "aggregate",
                "from": "T",
                "group_by": ["g"],
                "into": "Summary",
                "agg": {"total": "sum(v)", "n": "count(*)"},
            }
        ]
    return {
        "settings": {"root": str(root)},
        "nodes": {"T": {"csv": "t.csv", "pk": "id", "properties": properties}},
        "compute": operations,
    }


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify(graph, root: Path, cell: str, rows: list) -> dict:
    value_type = float if cell == "declared_float_control" else int
    expected = [{"id": i, "g": group, "v": value_type(value)} for i, group, value in rows]
    actual = graph.cypher("MATCH(n:T) RETURN n.id AS id,n.g AS g,n.v AS v ORDER BY id").to_list()
    if actual != expected or any(type(row["v"]) is not value_type for row in actual):
        raise AssertionError(f"{cell}: complete loaded source value/type oracle failed")
    emitted_hash = None
    computed_hash = None
    if cell == "derive":
        expected_derived = [{"id": i, "twice": value * 2} for i, _, value in rows]
        derived = graph.cypher("MATCH(n:T) RETURN n.id AS id,n.twice AS twice ORDER BY id").to_list()
        if derived != expected_derived or any(type(row["twice"]) is not int for row in derived):
            raise AssertionError("complete derived graph value/type oracle failed")
        emitted = read_csv(root / "computed/T_derived.csv")
        expected_csv = [
            {"id": str(i), "g": group, "v": str(value), "twice": str(value * 2)} for i, group, value in rows
        ]
        if emitted != expected_csv:
            raise AssertionError("complete derived CSV oracle failed")
        emitted_hash, computed_hash = digest(emitted), digest(derived)
    elif cell == "aggregate":
        groups = {}
        for _, group, value in rows:
            total, count = groups.get(group, (0, 0))
            groups[group] = (total + value, count + 1)
        expected_summary = [
            {"g": group, "total": total, "n": count} for group, (total, count) in sorted(groups.items())
        ]
        summary = graph.cypher("MATCH(n:Summary) RETURN n.g AS g,n.total AS total,n.n AS n ORDER BY g").to_list()
        if summary != expected_summary or any(
            type(row["total"]) is not int or type(row["n"]) is not int for row in summary
        ):
            raise AssertionError("complete aggregate graph value/type oracle failed")
        emitted = read_csv(root / "computed/aggregate_Summary.csv")
        # Generated identities deliberately change in the repair; compare every
        # group's raw properties and counts while requiring IDs to stay unique.
        if len({row["summary_id"] for row in emitted}) != len(groups):
            raise AssertionError("aggregate CSV identity cardinality failed")
        values = sorted(
            ({"g": row["g"], "total": int(row["total"]), "n": int(row["n"])} for row in emitted),
            key=lambda row: row["g"],
        )
        if values != expected_summary:
            raise AssertionError("complete aggregate CSV value oracle failed")
        emitted_hash, computed_hash = digest(values), digest(summary)
    return {
        "loaded_rows": len(actual),
        "loaded_sha256": digest(actual),
        "computed_sha256": computed_hash,
        "csv_values_sha256": emitted_hash,
    }


def measure(cell: str, size: int, groups: int, rounds: int, warmup: int, scratch: Path):
    from kglite.blueprint import from_blueprint

    content, rows = fixture(size, groups)
    samples = []
    oracle = None
    for iteration in range(warmup + rounds):
        with tempfile.TemporaryDirectory(prefix="blueprint-", dir=scratch) as folder:
            root = Path(folder)
            (root / "t.csv").write_bytes(content)
            spec = specification(root, cell)
            path = root / "blueprint.json"
            path.write_text(json.dumps(spec), encoding="utf-8")
            start = time.perf_counter_ns()
            graph = from_blueprint(path, save=False, verbose=False)
            elapsed = time.perf_counter_ns() - start
            checked = verify(graph, root, cell, rows)
            if oracle is not None and checked != oracle:
                raise AssertionError("per-event oracle changed across equivalent fixtures")
            oracle = checked
            del graph
        if iteration >= warmup:
            samples.append(elapsed)
    return {
        "cell": cell,
        "size": size,
        "groups": groups,
        "fixture_csv_sha256": hashlib.sha256(content).hexdigest(),
        "fixture_csv_bytes": len(content),
        "specification": specification(Path("<owned-round-directory>"), cell),
        "samples_ns": samples,
        "median_ns": statistics.median(samples),
        "mean_ns": statistics.mean(samples),
        "min_ns": min(samples),
        "oracle": oracle,
        "all_oracles_passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[10000])
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--cells", nargs="+", choices=CELLS, default=list(CELLS))
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.sizes or min(args.sizes) < args.groups or args.groups < 1 or args.rounds < 20 or args.warmup < 0:
        parser.error("sizes must cover positive groups; rounds >=20; warmup >=0")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "dev-docs/bench/results") or output.suffix != ".xz":
        parser.error("output must be an exclusive .json.xz capture under dev-docs/bench/results")
    if output.exists():
        parser.error("output already exists")
    provenance = release_provenance()
    scratch = ROOT / "dev-docs/temp"
    scratch.mkdir(parents=True, exist_ok=True)
    results = [
        measure(cell, size, args.groups, args.rounds, args.warmup, scratch)
        for size in args.sizes
        for cell in args.cells
    ]
    result = {
        "release": provenance,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "diff_stat": subprocess.check_output(["git", "diff", "--stat"], cwd=ROOT, text=True),
        "harness_sha256": sha(Path(__file__)),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha(path)
            for directory in (ROOT / "crates/kglite/src/graph/blueprint", ROOT / "kglite/blueprint")
            for path in sorted(directory.rglob("*"))
            if path.suffix in {".rs", ".py"}
        },
        "provenance_helper_sha256": sha(ROOT / "dev-docs/bench/scripts/reused_slot_delete.py"),
        "arguments": {**vars(args), "output": str(output)},
        "statistic": "median of independent first build events; mean/min retained as secondary statistics",
        "timing_scope": (
            "from_blueprint(save=False, verbose=False); "
            "fixture setup, full consumed graph/CSV oracles and teardown excluded"
        ),
        "limitations": "full build lifecycle; no allocation attribution, isolated compute or cold filesystem claim",
        "cells": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(lzma.compress((json.dumps(result, indent=2) + "\n").encode()))
    print(json.dumps({"output": str(output), "cells": len(results), "all_oracles_passed": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
