#!/usr/bin/env python3
"""Promote a verified released-wheel Linux benchmark artifact to baselines."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

try:
    from scripts.benchmark_provenance import benchmark_names, sha256
except ModuleNotFoundError:  # Direct `python scripts/promote_linux_benchmark.py` execution.
    from benchmark_provenance import benchmark_names, sha256


def promote(
    reference_path: Path,
    provenance_path: Path,
    expected_path: Path,
    versioned_output: Path,
    current_output: Path,
    *,
    reference_version: str = "0.13.2",
) -> None:
    reference: dict[str, Any] = json.loads(reference_path.read_text())
    provenance: dict[str, Any] = json.loads(provenance_path.read_text())
    expected: dict[str, Any] = json.loads(expected_path.read_text())

    if provenance.get("schema_version") != 1:
        raise ValueError("unsupported benchmark provenance schema")
    reference_meta = provenance.get("reference", {})
    if reference_meta.get("version") != reference_version:
        raise ValueError("provenance is not from the released reference version")
    if reference_meta.get("result_sha256") != sha256(reference_path):
        raise ValueError("reference benchmark digest does not match provenance")

    machine = reference.get("machine_info", {})
    if machine.get("system") != "Linux":
        raise ValueError("only Linux reference artifacts can update the Linux baseline")
    if not str(machine.get("python_version", "")).startswith("3.12."):
        raise ValueError("Linux reference artifact must use Python 3.12")

    names = benchmark_names(reference)
    expected_names = benchmark_names(expected)
    if set(names) != set(expected_names):
        raise ValueError("reference workload set differs from the tracked core baseline")
    if sorted(provenance.get("benchmark_names", [])) != sorted(names):
        raise ValueError("provenance workload set differs from the reference result")

    promoted = deepcopy(reference)
    for benchmark in promoted["benchmarks"]:
        benchmark["stats"].pop("data", None)
    promoted["kglite_baseline"] = {
        "source_distribution": f"kglite=={reference_version} (PyPI wheel)",
        "wheel_sha256": reference_meta["wheel_sha256"],
        "harness_sha256": provenance["harness"]["sha256"],
        "github": provenance["github"],
    }
    rendered = json.dumps(promoted, indent=2) + "\n"
    versioned_output.write_text(rendered)
    current_output.write_text(rendered)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("provenance", type=Path)
    parser.add_argument("--expected-names", type=Path, required=True)
    parser.add_argument("--versioned-output", type=Path, required=True)
    parser.add_argument("--current-output", type=Path, required=True)
    parser.add_argument("--reference-version", default="0.13.2")
    args = parser.parse_args()
    promote(
        args.reference,
        args.provenance,
        args.expected_names,
        args.versioned_output,
        args.current_output,
        reference_version=args.reference_version,
    )
    print(f"promoted released-wheel Linux baseline to {args.versioned_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
