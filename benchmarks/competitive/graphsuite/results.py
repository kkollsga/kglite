"""Read/append/save the accumulating results datafile.

The datafile (`results.json`) is append-only across sessions: every
invocation adds one `run` per library, tagged with library name,
version and run date, plus per-group combined timings. New libraries and
new runs slot in without touching old data.

Schema
------
{
  "schema_version": 2,
  "groups": [[group_id, description], ...],     # registry snapshot
  "runs": [
    {
      "library": "kglite-memory-cypher",
      "version": "0.10.15",
      "run_date": "2026-06-13T09:40:00",
      "dataset": {"scale": "medium", "signature": "...", "n_nodes": ..., "n_edges": ...},
      "machine": {"platform": "...", "python": "..."},
      "provenance": {
        "harness_version": 3, "capture_id": "...", "origin": "manual",
        "source_commit": "...", "source_dirty": false, "base_repeats": 5,
        "requested_libraries": ["kglite-cypher", "ladybug"],
        "publication_qualified": true
      },
      "groups": {
        "build": {"min_s": .., "median_s": .., "reps": .., "sanity": .., "status": "ok"},
        "node_scan": {... "status": "ok"},
        "shortest_path": {"status": "skip", "reason": ".."},
        ...
      }
    }, ...
  ]
}
"""

from __future__ import annotations

import json
from pathlib import Path
import platform
import subprocess
from typing import Any
import uuid

from .base import GROUPS

SCHEMA_VERSION = 2
HARNESS_VERSION = 3
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"
REPO_ROOT = Path(__file__).resolve().parents[3]
_STRICT_KGLITE_MODES = {
    "kglite-cypher",
    "kglite-mapped",
    "kglite-disk",
    "kglite-bolt",
    "kglite-bolt-docker",
}


def _machine() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


def capture_context(*, origin: str, base_repeats: int, requested_libraries: list[str]) -> dict[str, Any]:
    """Capture once per invocation so every backend records one environment."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            ".",
            ":(exclude)BENCHMARKS.md",
            ":(exclude)benchmarks/competitive/graphsuite/results.json",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "harness_version": HARNESS_VERSION,
        "capture_id": uuid.uuid4().hex,
        "origin": origin,
        "source_commit": commit,
        "source_dirty": bool(status.strip()),
        "base_repeats": base_repeats,
        "requested_libraries": requested_libraries,
    }


def publication_issues(runs: list[dict[str, Any]], requested_libraries: list[str]) -> list[str]:
    """Return reasons why one invocation is unsafe to publish.

    Historical and exploratory runs remain valid raw evidence. Publication is
    narrower: every requested adapter must have produced one error-free row in
    the same clean capture, on the same dataset and machine.
    """
    issues: list[str] = []
    requested = list(dict.fromkeys(requested_libraries))
    actual = [run.get("library") for run in runs]
    missing = [library for library in requested if library not in actual]
    unexpected = [library for library in actual if library not in requested]
    duplicates = sorted({library for library in actual if actual.count(library) > 1})
    if missing:
        issues.append("unavailable or missing requested adapters: " + ", ".join(missing))
    if unexpected:
        issues.append("unexpected adapter rows: " + ", ".join(unexpected))
    if duplicates:
        issues.append("duplicate adapter rows: " + ", ".join(duplicates))
    if not runs:
        return issues or ["capture contains no adapter rows"]

    provenances = [run.get("provenance", {}) for run in runs]
    capture_ids = {p.get("capture_id") for p in provenances}
    commits = {p.get("source_commit") for p in provenances}
    datasets = {run.get("dataset", {}).get("signature") for run in runs}
    machines = {json.dumps(run.get("machine"), sort_keys=True) for run in runs}
    if None in capture_ids or len(capture_ids) != 1:
        issues.append("adapter rows do not share one capture id")
    if None in commits or len(commits) != 1:
        issues.append("adapter rows do not share one source commit")
    if len(datasets) != 1:
        issues.append("adapter rows do not share one dataset")
    if len(machines) != 1:
        issues.append("adapter rows do not share one machine")
    if any(p.get("source_dirty") is not False for p in provenances):
        issues.append("source worktree was dirty")
    if any(p.get("requested_libraries") != requested for p in provenances):
        issues.append("requested-adapter manifest is inconsistent")
    if any(p.get("harness_version") != HARNESS_VERSION for p in provenances):
        issues.append("capture was not produced by the current harness")

    errored = sorted(
        f"{run['library']}:{gid}"
        for run in runs
        for gid, result in run.get("groups", {}).items()
        if result.get("status") == "err"
    )
    if errored:
        issues.append("adapter errors: " + ", ".join(errored))

    strict_runs = [run for run in runs if run.get("library") in _STRICT_KGLITE_MODES]
    group_ids = {gid for run in strict_runs for gid in run.get("groups", {})}
    divergent: list[str] = []
    for gid in sorted(group_ids):
        digests = {
            result.get("digest")
            for run in strict_runs
            if (result := run.get("groups", {}).get(gid, {})).get("status") == "ok"
            and result.get("digest") is not None
        }
        if len(digests) > 1:
            divergent.append(gid)
    if divergent:
        issues.append("kglite storage/protocol result divergence: " + ", ".join(divergent))
    return issues


def load(path: Path = RESULTS_PATH) -> dict[str, Any]:
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {
        "schema_version": SCHEMA_VERSION,
        "harness": {"name": "graphsuite", "version": HARNESS_VERSION},
        "groups": [[g[0], g[1]] for g in GROUPS],
        "runs": [],
    }


def save(data: dict[str, Any], path: Path = RESULTS_PATH) -> None:
    # keep the group registry snapshot fresh
    data["schema_version"] = SCHEMA_VERSION
    data["harness"] = {"name": "graphsuite", "version": HARNESS_VERSION}
    data["groups"] = [[g[0], g[1]] for g in GROUPS]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def make_run(
    library: str,
    version: str,
    run_date: str,
    ds_scale: str,
    ds_signature: str,
    n_nodes: int,
    n_edges: int,
    groups: dict[str, dict[str, Any]],
    provenance: dict[str, Any],
    dataset_seed: int,
) -> dict[str, Any]:
    return {
        "library": library,
        "version": version,
        "run_date": run_date,
        "dataset": {
            "scale": ds_scale,
            "seed": dataset_seed,
            "signature": ds_signature,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
        },
        "machine": _machine(),
        "provenance": provenance,
        "groups": groups,
    }


def append_runs(new_runs: list[dict[str, Any]], path: Path = RESULTS_PATH) -> dict[str, Any]:
    data = load(path)
    data["runs"].extend(new_runs)
    save(data, path)
    return data
