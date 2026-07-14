"""Read/append/save the accumulating results datafile.

The datafile (`results.json`) is append-only across sessions: every
invocation adds one `run` per library, tagged with library name,
version and run date, plus per-group combined timings. New libraries and
new runs slot in without touching old data.

Schema
------
{
  "schema_version": 1,
  "groups": [[group_id, description], ...],     # registry snapshot
  "runs": [
    {
      "library": "kglite-memory-cypher",
      "version": "0.10.15",
      "run_date": "2026-06-13T09:40:00",
      "dataset": {"scale": "medium", "signature": "...", "n_nodes": ..., "n_edges": ...},
      "machine": {"platform": "...", "python": "..."},
      "provenance": {
        "harness_version": 2, "origin": "manual", "source_commit": "...",
        "source_dirty": false, "base_repeats": 5
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

from .base import GROUPS

SCHEMA_VERSION = 2
HARNESS_VERSION = 2
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"
REPO_ROOT = Path(__file__).resolve().parents[3]


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


def capture_context(*, origin: str, base_repeats: int) -> dict[str, Any]:
    """Capture once per invocation so every backend records one environment."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "harness_version": HARNESS_VERSION,
        "origin": origin,
        "source_commit": commit,
        "source_dirty": bool(status.strip()),
        "base_repeats": base_repeats,
    }


def load(path: Path = RESULTS_PATH) -> dict[str, Any]:
    if path.exists():
        with open(path) as fh:
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
    with open(path, "w") as fh:
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
