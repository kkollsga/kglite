"""Tests for the standalone performance-regression gate."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "compare_bench.py"


def _write_result(path: Path, names: list[str]) -> None:
    path.write_text(
        json.dumps(
            {"benchmarks": [{"name": name, "stats": {"min": 1.0, "mean": 1.0, "median": 1.0}} for name in names]}
        )
    )


def _compare(
    tmp_path: Path, baseline: list[str], current: list[str], *, require_exact_set: bool = False
) -> subprocess.CompletedProcess[str]:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    _write_result(baseline_path, baseline)
    _write_result(current_path, current)
    command = [sys.executable, str(SCRIPT), str(baseline_path), str(current_path), "--quiet"]
    if require_exact_set:
        command.append("--require-exact-set")
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )


def test_missing_tracked_benchmark_fails(tmp_path: Path) -> None:
    result = _compare(tmp_path, ["kept", "removed"], ["kept"])
    assert result.returncode == 1
    assert "tracked benchmark(s) were not collected" in result.stdout
    assert "removed" in result.stdout


def test_new_benchmark_waits_for_baseline_refresh(tmp_path: Path) -> None:
    result = _compare(tmp_path, ["kept"], ["kept", "new"])
    assert result.returncode == 0
    assert "new benchmark(s)" in result.stdout


def test_exact_set_rejects_benchmark_without_baseline_row(tmp_path: Path) -> None:
    result = _compare(tmp_path, ["kept"], ["kept", "new"], require_exact_set=True)
    assert result.returncode == 1
    assert "collected benchmark(s) have no baseline row" in result.stdout
    assert "new" in result.stdout
