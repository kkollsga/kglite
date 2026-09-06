#!/usr/bin/env python3
"""Release-only calendar-link correctness-cost events; no timing executed at preparation."""

from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import hashlib
import io
import json
import lzma
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "Cargo.toml").exists())
sys.path.insert(0, str(ROOT / "dev-docs/bench/scripts"))
from correctness_blueprint import digest, read_csv  # noqa: E402
from reused_slot_delete import release_provenance, sha  # noqa: E402

CELLS = ("calendar_source_control", "calendar_links")
DATES = [date(2026, 1, 1) + timedelta(days=i) for i in range(10)]


def fixture(size):
    rows = [{"id": i + 1, "day": DATES[i % 10].isoformat(), "v": i % 101} for i in range(size)]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["id", "day", "v"])
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode(), rows


def specification(root, cell):
    operation = {"op": "calendar", "type": "Date", "start": "2026-01-01", "end": "2026-01-10", "next_edge": "NEXT_DAY"}
    if cell == "calendar_links":
        operation["links"] = [{"from": "T", "date_col": "day", "edge": "ON_DATE"}]
    return {
        "settings": {"root": str(root)},
        "nodes": {"T": {"csv": "t.csv", "pk": "id", "properties": {"day": "string", "v": "int"}}},
        "compute": [operation],
    }


def verify(graph, root, cell, rows, content):
    source = graph.cypher("MATCH(n:T) RETURN n.id AS id,n.day AS day,n.v AS v ORDER BY id").to_list()
    if source != rows or any(
        type(r["id"]) is not int or type(r["v"]) is not int or type(r["day"]) is not str for r in source
    ):
        raise AssertionError("source values/types differ")
    expected_dates = [
        {"id": d.isoformat(), "year": 2026, "month": 1, "day": d.day, "quarter": 1, "weekday": d.strftime("%A")}
        for d in DATES
    ]
    actual_dates = graph.cypher(
        "MATCH(n:Date) RETURN toString(n.id) AS id,n.year AS year,n.month AS month, "
        "n.day AS day,n.quarter AS quarter,n.weekday AS weekday ORDER BY id"
    ).to_list()
    if actual_dates != expected_dates or any(
        type(row[key]) is not int for row in actual_dates for key in ["year", "month", "day", "quarter"]
    ):
        raise AssertionError("generated date values/types differ")
    expected_next = [{"a": a.isoformat(), "b": b.isoformat()} for a, b in zip(DATES, DATES[1:])]
    actual_next = graph.cypher(
        "MATCH(a:Date)-[:NEXT_DAY]->(b:Date) RETURN toString(a.id) AS a,toString(b.id) AS b ORDER BY a"
    ).to_list()
    expected_links = [{"id": r["id"], "date": r["day"]} for r in rows] if cell == "calendar_links" else []
    actual_links = graph.cypher(
        "MATCH(a:T)-[:ON_DATE]->(b:Date) RETURN a.id AS id,toString(b.id) AS date ORDER BY id"
    ).to_list()
    counts = graph.cypher("MATCH(n) RETURN count(n) AS n").to_list()
    edges = graph.cypher("MATCH()-[r]->() RETURN count(r) AS n").to_list()
    if (
        actual_next != expected_next
        or actual_links != expected_links
        or counts != [{"n": len(rows) + 10}]
        or edges != [{"n": len(expected_links) + 9}]
    ):
        raise AssertionError("exact node/edge endpoints/cardinality differ")
    expected_csv = {
        "calendar_Date.csv": [
            {"iso": r["id"], **{k: str(v) for k, v in r.items() if k != "id"}} for r in expected_dates
        ],
        "calendar_Date_NEXT_DAY.csv": [{"iso": r["a"], "next_iso": r["b"]} for r in expected_next],
    }
    if cell == "calendar_links":
        expected_csv["calendar_link_T_ON_DATE.csv"] = [{"id": str(r["id"]), "iso": r["date"]} for r in expected_links]
    actual_csv = {p.name: read_csv(p) for p in sorted((root / "computed").glob("*.csv"))}
    if actual_csv != expected_csv or (root / "t.csv").read_bytes() != content:
        raise AssertionError("emitted CSV oracle or original input preservation failed")
    return {
        "loaded_rows": len(source),
        "loaded_sha256": digest(source),
        "dates_sha256": digest(actual_dates),
        "edges_sha256": digest([actual_next, actual_links]),
        "csv_values_sha256": digest(actual_csv),
    }


def measure(cell, size, rounds, warmup):
    from kglite.blueprint import from_blueprint

    content, rows = fixture(size)
    samples, oracle = [], None
    for iteration in range(warmup + rounds):
        with tempfile.TemporaryDirectory(prefix="calendar-cost-", dir=ROOT / "dev-docs/temp") as folder:
            root = Path(folder)
            (root / "t.csv").write_bytes(content)
            path = root / "blueprint.json"
            path.write_text(json.dumps(specification(root, cell)), encoding="utf-8")
            start = time.perf_counter_ns()
            graph = from_blueprint(path, save=False, verbose=False)
            elapsed = time.perf_counter_ns() - start
            checked = verify(graph, root, cell, rows, content)
            if oracle is not None and checked != oracle:
                raise AssertionError("oracle changed across equivalent events")
            oracle = checked
            del graph
        if iteration >= warmup:
            samples.append(elapsed)
    return {
        "cell": cell,
        "size": size,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 10000])
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.sizes) < 1 or args.rounds < 20 or args.warmup < 0:
        parser.error("positive sizes, rounds>=20 and warmup>=0 required")
    output = args.output.resolve()
    if (
        not output.is_relative_to(ROOT / "dev-docs/bench/results")
        or not output.name.endswith(".json.xz")
        or output.exists()
    ):
        parser.error("output must be a new exclusive .json.xz under dev-docs/bench/results")
    provenance = release_provenance()
    (ROOT / "dev-docs/temp").mkdir(parents=True, exist_ok=True)
    result = {
        "release": provenance,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "diff_stat": subprocess.check_output(["git", "diff", "--stat"], cwd=ROOT, text=True),
        "harness_sha256": sha(Path(__file__)),
        "provenance_helper_sha256": sha(ROOT / "dev-docs/bench/scripts/reused_slot_delete.py"),
        "oracle_helper_sha256": sha(ROOT / "dev-docs/bench/scripts/correctness_blueprint.py"),
        "source_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for folder in [ROOT / "crates/kglite/src/graph/blueprint", ROOT / "kglite/blueprint"]
            for p in sorted(folder.rglob("*"))
            if p.suffix in {".rs", ".py"}
        },
        "arguments": {**vars(args), "output": str(output)},
        "statistic": "median of independent first build events; min/mean secondary",
        "timing_scope": (
            "from_blueprint(save=False,verbose=False); fixture setup, complete graph/CSV oracles, "
            "graph drop and temporary-folder cleanup excluded"
        ),
        "limitations": (
            "whole build cost; source control retains the same calendar without links; "
            "no isolated CSV-reader or allocation attribution"
        ),
        "cells": [measure(cell, size, args.rounds, args.warmup) for size in args.sizes for cell in CELLS],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(lzma.compress((json.dumps(result, indent=2) + "\n").encode()))
    print(json.dumps({"output": str(output), "cells": len(result["cells"]), "all_oracles_passed": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
