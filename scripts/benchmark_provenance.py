#!/usr/bin/env python3
"""Validate isolated benchmark runs and record reproducible provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def absolute_executable(path: Path) -> Path:
    """Make an interpreter path absolute without resolving its venv symlink."""
    return Path(os.path.abspath(path))


def benchmark_names(result: dict[str, Any]) -> list[str]:
    names = [entry["name"] for entry in result.get("benchmarks", [])]
    if not names or len(names) != len(set(names)):
        raise ValueError("benchmark result must contain unique benchmark names")
    return names


def validate_benchmark_results(
    reference: dict[str, Any], candidate: dict[str, Any], expected: dict[str, Any]
) -> list[str]:
    expected_names = set(benchmark_names(expected))
    reference_names = set(benchmark_names(reference))
    candidate_names = set(benchmark_names(candidate))
    if reference_names != expected_names:
        raise ValueError("released-wheel benchmark set differs from the tracked harness")
    if candidate_names != expected_names:
        raise ValueError("candidate benchmark set differs from the tracked harness")

    for label, result in (("reference", reference), ("candidate", candidate)):
        machine = result.get("machine_info", {})
        if machine.get("system") != "Linux":
            raise ValueError(f"{label} benchmark must be captured on Linux")
        if not str(machine.get("python_version", "")).startswith("3.12."):
            raise ValueError(f"{label} benchmark must use Python 3.12")
    return sorted(expected_names)


def probe_environment(python: Path, cwd: Path) -> dict[str, Any]:
    code = (
        "import json, pathlib, sys, kglite; "
        "print(json.dumps({'version': kglite.__version__, "
        "'module_file': str(pathlib.Path(kglite.__file__).resolve()), "
        "'prefix': str(pathlib.Path(sys.prefix).resolve())}))"
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    proc = subprocess.run(
        [str(python), "-c", code],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(proc.stdout)
    module_file = Path(info["module_file"])
    prefix = Path(info["prefix"])
    if not module_file.is_relative_to(prefix):
        raise ValueError(f"kglite import escaped isolated environment: {module_file}")
    freeze = subprocess.run(
        [str(python), "-m", "pip", "freeze", "--all"],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    info["pip_freeze"] = freeze.stdout.splitlines()
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-python", type=Path, required=True)
    parser.add_argument("--candidate-python", type=Path, required=True)
    parser.add_argument("--reference-wheel", type=Path, required=True)
    parser.add_argument("--candidate-wheel", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--expected-names", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-version", default="0.13.2")
    parser.add_argument("--candidate-version", required=True)
    args = parser.parse_args()

    reference = json.loads(args.reference_json.read_text())
    candidate = json.loads(args.candidate_json.read_text())
    expected = json.loads(args.expected_names.read_text())
    names = validate_benchmark_results(reference, candidate, expected)

    reference_env = probe_environment(absolute_executable(args.reference_python), args.harness.parent)
    candidate_env = probe_environment(absolute_executable(args.candidate_python), args.harness.parent)
    if reference_env["version"] != args.reference_version:
        raise ValueError(f"reference environment has {reference_env['version']}, expected {args.reference_version}")
    if candidate_env["version"] != args.candidate_version:
        raise ValueError(f"candidate environment has {candidate_env['version']}, expected {args.candidate_version}")

    provenance = {
        "schema_version": 1,
        "github": {
            "sha": os.environ.get("GITHUB_SHA", "local"),
            "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "local"),
        },
        "harness": {"path": args.harness.name, "sha256": sha256(args.harness)},
        "benchmark_names": names,
        "reference": {
            **reference_env,
            "wheel_sha256": sha256(args.reference_wheel),
            "result_sha256": sha256(args.reference_json),
        },
        "candidate": {
            **candidate_env,
            "wheel_sha256": sha256(args.candidate_wheel),
            "result_sha256": sha256(args.candidate_json),
        },
    }
    args.output.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"benchmark provenance: OK ({len(names)} workloads)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
