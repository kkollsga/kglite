#!/usr/bin/env python3
"""Read-only baseline for the Cypher CALL scope program.

The timed cells cover the executor paths that Phases 2 and 3 will change:

* a leading ordinary ``CALL procedure``;
* a legacy correlated ``CALL { WITH ... }`` subquery; and
* a legacy uncorrelated ``CALL { ... }`` subquery.

Each target has a no-CALL query with the same result. Those control cells stay
outside the CALL executor changes and make capture-wide machine drift visible.
Fixture construction and correctness checks happen before timing; only
``graph.cypher(...).to_list()`` is measured, so setup cannot leak into a cell.

Correctness-only check (a debug extension is valid)::

    uv run --no-sync python \
        dev-docs/bench/scripts/cypher_call_scope_phase1.py --self-check

Performance capture (release extension required)::

    uv run --no-sync maturin develop --release
    uv run --no-sync python \
        dev-docs/bench/scripts/cypher_call_scope_phase1.py measure \
        --run baseline-1 --output cypher-call-scope-baseline-1.json

Run ``measure`` twice. Compare minima unless a cell's minimum is at least 30%
below its median, in which case the repository performance protocol requires
the median. Generated captures are accepted only as basenames and are written
under ``dev-docs/bench/out/``, the bounded 14-day benchmark artifact tier.
Rebuild the debug extension before returning to correctness work.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
from typing import Literal

import pandas as pd

import kglite

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "dev-docs" / "bench" / "out"

BENCH_NODES = 4_096
BUCKETS = 32
MARKERS = 64
EXPECTED_ID_SUM = BENCH_NODES * (BENCH_NODES - 1) // 2

Role = Literal["target", "control"]


@dataclass(frozen=True)
class Cell:
    name: str
    role: Role
    anchor: str
    query: str
    expected: list[dict[str, object]]


CELLS = (
    Cell(
        name="leading_procedure_property_stats",
        role="target",
        anchor="ordinary_procedure",
        query=(
            "CALL db.property_stats({node_type: 'Bench', property: 'bucket'}) "
            "YIELD value_count, distinct_count "
            "RETURN value_count, distinct_count"
        ),
        expected=[{"value_count": BENCH_NODES, "distinct_count": BUCKETS}],
    ),
    Cell(
        name="control_property_aggregate",
        role="control",
        anchor="ordinary_procedure",
        query=("MATCH (n:Bench) RETURN count(n.bucket) AS value_count, count(DISTINCT n.bucket) AS distinct_count"),
        expected=[{"value_count": BENCH_NODES, "distinct_count": BUCKETS}],
    ),
    Cell(
        name="legacy_correlated_subquery",
        role="target",
        anchor="correlated_subquery",
        query=("MATCH (n:Bench) CALL { WITH n RETURN n.id + 1 AS next_id } RETURN sum(next_id) AS checksum"),
        expected=[{"checksum": EXPECTED_ID_SUM + BENCH_NODES}],
    ),
    Cell(
        name="control_correlated_projection",
        role="control",
        anchor="correlated_subquery",
        query="MATCH (n:Bench) RETURN sum(n.id + 1) AS checksum",
        expected=[{"checksum": EXPECTED_ID_SUM + BENCH_NODES}],
    ),
    Cell(
        name="legacy_uncorrelated_subquery",
        role="target",
        anchor="uncorrelated_subquery",
        query=(
            "MATCH (n:Bench) "
            "CALL { MATCH (m:Marker) RETURN count(m) AS marker_count } "
            "RETURN sum(n.id + marker_count) AS checksum"
        ),
        expected=[{"checksum": EXPECTED_ID_SUM + BENCH_NODES * MARKERS}],
    ),
    Cell(
        name="control_uncorrelated_projection",
        role="control",
        anchor="uncorrelated_subquery",
        query=f"MATCH (n:Bench) RETURN sum(n.id + {MARKERS}) AS checksum",
        expected=[{"checksum": EXPECTED_ID_SUM + BENCH_NODES * MARKERS}],
    ),
)

_MUTATING_CLAUSE = re.compile(
    r"\b(?:CREATE|MERGE|SET|DELETE|REMOVE|DROP|INSERT|FOREACH|LOAD\s+CSV)\b",
    re.IGNORECASE,
)


def build_fixture():
    graph = kglite.KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(BENCH_NODES),
                "name": [f"bench-{i}" for i in range(BENCH_NODES)],
                "bucket": [i % BUCKETS for i in range(BENCH_NODES)],
            }
        ),
        "Bench",
        "id",
        "name",
    )
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": range(MARKERS),
                "name": [f"marker-{i}" for i in range(MARKERS)],
            }
        ),
        "Marker",
        "id",
        "name",
    )
    return graph


def _graph_signature(graph) -> dict[str, int]:
    info = graph.graph_info()
    return {key: int(info[key]) for key in ("node_count", "edge_count", "type_count")}


def _validate_cell_definitions() -> None:
    names = [cell.name for cell in CELLS]
    if len(names) != len(set(names)):
        raise AssertionError("benchmark cell names must be unique")

    for anchor in ("ordinary_procedure", "correlated_subquery", "uncorrelated_subquery"):
        roles = {cell.role for cell in CELLS if cell.anchor == anchor}
        if roles != {"target", "control"}:
            raise AssertionError(f"{anchor} must have target and control cells, got {sorted(roles)}")

    for cell in CELLS:
        mutation = _MUTATING_CLAUSE.search(cell.query)
        if mutation:
            raise AssertionError(f"{cell.name} contains mutating clause {mutation.group(0)!r}")

    procedure_cells = [cell for cell in CELLS if cell.anchor == "ordinary_procedure" and cell.role == "target"]
    if len(procedure_cells) != 1 or not procedure_cells[0].query.startswith("CALL db.property_stats("):
        raise AssertionError("ordinary procedure target must remain the read-only db.property_stats call")

    correlated = [cell for cell in CELLS if cell.anchor == "correlated_subquery" and cell.role == "target"]
    if len(correlated) != 1 or "CALL { WITH n " not in correlated[0].query:
        raise AssertionError("correlated target must retain its legacy importing WITH")

    uncorrelated = [cell for cell in CELLS if cell.anchor == "uncorrelated_subquery" and cell.role == "target"]
    if len(uncorrelated) != 1 or "CALL { MATCH " not in uncorrelated[0].query:
        raise AssertionError("uncorrelated target must retain its legacy no-import body")

    for cell in (cell for cell in CELLS if cell.role == "control"):
        if "CALL" in cell.query.upper():
            raise AssertionError(f"{cell.name} control must remain outside CALL execution")


def _execute(graph, cell: Cell) -> list[dict[str, object]]:
    return graph.cypher(cell.query).to_list()


def _assert_result(cell: Cell, actual: list[dict[str, object]]) -> None:
    if actual != cell.expected:
        raise AssertionError(f"{cell.name}: expected {cell.expected!r}, got {actual!r}")


def self_check(graph=None) -> None:
    _validate_cell_definitions()
    if graph is None:
        graph = build_fixture()
    before = _graph_signature(graph)
    expected_signature = {"node_count": BENCH_NODES + MARKERS, "edge_count": 0, "type_count": 2}
    if before != expected_signature:
        raise AssertionError(f"fixture signature: expected {expected_signature!r}, got {before!r}")

    for cell in CELLS:
        _assert_result(cell, _execute(graph, cell))

    after = _graph_signature(graph)
    if after != before:
        raise AssertionError(f"read-only cells mutated graph signature: before={before!r}, after={after!r}")


def _measure_cell(graph, cell: Cell, *, warmup: int, rounds: int) -> dict[str, object]:
    for _ in range(warmup):
        _assert_result(cell, _execute(graph, cell))

    samples_ms: list[float] = []
    for _ in range(rounds):
        started = time.perf_counter_ns()
        actual = _execute(graph, cell)
        elapsed_ns = time.perf_counter_ns() - started
        _assert_result(cell, actual)
        samples_ms.append(elapsed_ns / 1_000_000.0)

    minimum = min(samples_ms)
    median = statistics.median(samples_ms)
    return {
        "name": cell.name,
        "role": cell.role,
        "anchor": cell.anchor,
        "rounds": rounds,
        "min_ms": minimum,
        "median_ms": median,
        "mean_ms": statistics.fmean(samples_ms),
        "max_ms": max(samples_ms),
        "heavy_tailed": minimum <= median * 0.7,
        "query": cell.query,
        "expected": cell.expected,
        "samples_ms": samples_ms,
    }


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_artifact(profile: str) -> Path:
    if sys.platform == "darwin":
        filename = "libkglite_py.dylib"
    elif sys.platform == "win32":
        filename = "kglite_py.dll"
    else:
        filename = "libkglite_py.so"
    return ROOT / "target" / profile / filename


def _release_provenance() -> dict[str, str]:
    build_command = "uv run --no-sync maturin develop --release"
    extension = Path(importlib.import_module("kglite.kglite").__file__).resolve()
    try:
        extension.relative_to(ROOT)
    except ValueError as exc:
        raise RuntimeError(
            f"loaded extension is outside this workspace: {extension}. Run `{build_command}` from {ROOT} and rerun."
        ) from exc

    artifacts = {
        profile: _profile_artifact(profile) for profile in ("release", "debug") if _profile_artifact(profile).is_file()
    }
    release = _profile_artifact("release")
    if "release" not in artifacts:
        raise RuntimeError(f"release cdylib is missing at {release}. Run `{build_command}` and rerun.")

    newest_profile, newest_artifact = max(artifacts.items(), key=lambda item: item[1].stat().st_mtime_ns)
    if newest_profile != "release":
        raise RuntimeError(
            f"newest workspace cdylib is {newest_artifact} ({newest_profile}), not release. "
            f"Run `{build_command}` and rerun."
        )

    extension_sha = _sha256(extension)
    release_sha = _sha256(release)
    if extension_sha != release_sha:
        debug = artifacts.get("debug")
        loaded_profile = "debug" if debug is not None and extension_sha == _sha256(debug) else "unknown/non-release"
        raise RuntimeError(
            f"loaded extension {extension} is {loaded_profile} and does not match {release}. "
            f"Run `{build_command}` and rerun."
        )

    sources = [ROOT / "Cargo.toml", ROOT / "crates/kglite/Cargo.toml", ROOT / "crates/kglite-py/Cargo.toml"]
    for crate in ("kglite", "kglite-py"):
        sources.extend((ROOT / "crates" / crate / "src").rglob("*.rs"))
    newest_source = max(sources, key=lambda path: path.stat().st_mtime_ns)
    if release.stat().st_mtime_ns < newest_source.stat().st_mtime_ns:
        raise RuntimeError(
            f"release cdylib {release} predates {newest_source.relative_to(ROOT)}. Run `{build_command}` and rerun."
        )

    return {
        "profile": "release",
        "extension": str(extension),
        "release_artifact": str(release.resolve()),
        "sha256": release_sha,
    }


def measure(*, warmup: int, rounds: int, run: str) -> dict[str, object]:
    if warmup < 20:
        raise ValueError("performance protocol requires at least 20 warmup iterations")
    if rounds < 100:
        raise ValueError("performance protocol requires at least 100 measured rounds")

    release = _release_provenance()
    setup_started = time.perf_counter_ns()
    graph = build_fixture()
    setup_ms = (time.perf_counter_ns() - setup_started) / 1_000_000.0
    self_check(graph)

    cells = [_measure_cell(graph, cell, warmup=warmup, rounds=rounds) for cell in CELLS]
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "run": run,
        "revision": _git_revision(),
        "release": release,
        "kglite_version": kglite.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "fixture": {"bench_nodes": BENCH_NODES, "markers": MARKERS, "setup_ms": setup_ms},
        "warmup_iterations": warmup,
        "cells": cells,
    }


def _write_capture(name: str, capture: dict[str, object]) -> Path:
    if Path(name).name != name or not name.endswith(".json"):
        raise ValueError("--output must be a .json basename; captures live in dev-docs/bench/out")
    output = OUT_DIR / name
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing capture: {output}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(capture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("measure",), help="capture release-mode timings")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="validate fixture, queries, answers, and read-only state",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--run", default="unspecified", help="capture label, such as baseline-1")
    parser.add_argument("--output", help="optional JSON basename under dev-docs/bench/out")
    args = parser.parse_args(argv)
    if args.self_check == (args.command == "measure"):
        parser.error("choose exactly one of --self-check or measure")
    if args.output and args.command != "measure":
        parser.error("--output requires measure")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_check:
        self_check()
        print(f"self-check passed: {len(CELLS)} read-only cells")
        return 0

    capture = measure(warmup=args.warmup, rounds=args.rounds, run=args.run)
    print(json.dumps(capture, indent=2, sort_keys=True))
    if args.output:
        print(f"capture written: {_write_capture(args.output, capture)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
