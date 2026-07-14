"""Contracts for released-wheel Linux benchmark capture and promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.benchmark_provenance import absolute_executable, sha256, validate_benchmark_results
from scripts.promote_linux_benchmark import promote


def _result(*names: str, system: str = "Linux", python: str = "3.12.9") -> dict:
    return {
        "machine_info": {"system": system, "python_version": python},
        "benchmarks": [{"name": name, "stats": {"min": 1.0, "data": [1.0, 1.1]}} for name in names],
    }


def test_capture_requires_same_complete_linux_python312_workload_set() -> None:
    expected = _result("one", "two")
    assert validate_benchmark_results(expected, expected, expected) == ["one", "two"]
    with pytest.raises(ValueError, match="released-wheel benchmark set"):
        validate_benchmark_results(_result("one"), expected, expected)
    with pytest.raises(ValueError, match="must be captured on Linux"):
        validate_benchmark_results(_result("one", "two", system="Darwin"), expected, expected)


def test_absolute_executable_preserves_virtualenv_symlink(tmp_path: Path) -> None:
    target = tmp_path / "system-python"
    target.touch()
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(target)
    assert absolute_executable(venv_python) == venv_python
    assert absolute_executable(venv_python) != venv_python.resolve()


def test_promotion_accepts_only_verified_reference_and_strips_samples(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.json"
    expected_path = tmp_path / "expected.json"
    provenance_path = tmp_path / "provenance.json"
    versioned = tmp_path / "0_13_2.linux.json"
    current = tmp_path / "current.linux.json"
    reference_path.write_text(json.dumps(_result("one", "two")))
    expected_path.write_text(json.dumps(_result("one", "two")))
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "github": {"sha": "abc", "run_id": "1", "run_attempt": "1"},
                "harness": {"sha256": "harness"},
                "benchmark_names": ["one", "two"],
                "reference": {
                    "version": "0.13.2",
                    "wheel_sha256": "wheel",
                    "result_sha256": sha256(reference_path),
                },
            }
        )
    )

    promote(reference_path, provenance_path, expected_path, versioned, current)
    promoted = json.loads(versioned.read_text())
    assert versioned.read_text() == current.read_text()
    assert promoted["kglite_baseline"]["source_distribution"] == "kglite==0.13.2 (PyPI wheel)"
    assert all("data" not in item["stats"] for item in promoted["benchmarks"])

    provenance = json.loads(provenance_path.read_text())
    provenance["reference"]["result_sha256"] = "tampered"
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="digest"):
        promote(reference_path, provenance_path, expected_path, versioned, current)
