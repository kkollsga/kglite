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
from typing import Any

from .base import GROUPS

SCHEMA_VERSION = 1
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


def _machine() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }


def load(path: Path = RESULTS_PATH) -> dict[str, Any]:
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    return {
        "schema_version": SCHEMA_VERSION,
        "groups": [[g[0], g[1]] for g in GROUPS],
        "runs": [],
    }


def save(data: dict[str, Any], path: Path = RESULTS_PATH) -> None:
    # keep the group registry snapshot fresh
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
) -> dict[str, Any]:
    return {
        "library": library,
        "version": version,
        "run_date": run_date,
        "dataset": {
            "scale": ds_scale,
            "signature": ds_signature,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
        },
        "machine": _machine(),
        "groups": groups,
    }


def append_runs(new_runs: list[dict[str, Any]], path: Path = RESULTS_PATH) -> dict[str, Any]:
    data = load(path)
    data["runs"].extend(new_runs)
    save(data, path)
    return data
