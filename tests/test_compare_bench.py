"""Tests for the standalone performance-regression gate."""

from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "compare_bench.py"


def _write_result(path: Path, names: list[str] | dict[str, float]) -> None:
    values = {name: 1.0 for name in names} if isinstance(names, list) else names
    path.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {"name": name, "stats": {"min": value, "mean": value, "median": value}}
                    for name, value in values.items()
                ]
            }
        ),
        encoding="utf-8",
    )


def _compare(
    tmp_path: Path,
    baseline: list[str] | dict[str, float],
    current: list[str] | dict[str, float],
    *,
    require_exact_set: bool = False,
    quiet: bool = True,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    _write_result(baseline_path, baseline)
    _write_result(current_path, current)
    command = [sys.executable, str(SCRIPT), str(baseline_path), str(current_path)]
    if quiet:
        command.append("--quiet")
    if require_exact_set:
        command.append("--require-exact-set")
    command.extend(extra_args or [])
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )


def _current_at(deltas: dict[str, float]) -> dict[str, float]:
    """Current-run values producing the requested percentage deltas against a
    baseline of 1.0 per cell."""
    return {name: 1.0 + pct / 100 for name, pct in deltas.items()}


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


def test_watch_band_names_a_cell_just_under_the_threshold(tmp_path: Path) -> None:
    """A passing cell one point below the gate is reported, exit code unchanged."""
    result = _compare(tmp_path, ["creeping", "steady"], _current_at({"creeping": 19.0, "steady": 0.0}))
    assert result.returncode == 0
    assert "APPROACHING (+12.0%..+20.0%)" in result.stdout
    assert "creeping" in result.stdout.split("APPROACHING")[1]
    assert "steady" not in result.stdout.split("APPROACHING")[1]
    assert "1 cell(s) approaching the +20.0% threshold" in result.stdout
    assert "FAIL" not in result.stdout


def test_watch_band_ignores_a_cell_below_the_band_floor(tmp_path: Path) -> None:
    """One point under the band floor (+12% on a 20% gate) stays silent."""
    result = _compare(tmp_path, ["quiet_cell"], _current_at({"quiet_cell": 11.0}))
    assert result.returncode == 0
    assert "APPROACHING" not in result.stdout


def test_watch_band_orders_worst_first(tmp_path: Path) -> None:
    result = _compare(tmp_path, ["mild", "severe"], _current_at({"mild": 13.0, "severe": 18.0}))
    assert result.returncode == 0
    block = result.stdout.split("APPROACHING")[1]
    assert block.index("severe") < block.index("mild")
    assert "2 cell(s) approaching the +20.0% threshold" in result.stdout


def test_regression_goes_to_the_fail_list_not_the_watch_band(tmp_path: Path) -> None:
    result = _compare(tmp_path, ["broken"], _current_at({"broken": 25.0}))
    assert result.returncode == 1
    assert "APPROACHING" not in result.stdout
    assert "1 benchmark(s) regressed > +20.0%" in result.stdout


def test_watch_band_derives_from_the_passed_threshold(tmp_path: Path) -> None:
    """CI's two legs gate at 20% and 30%; the band follows whichever was passed."""
    result = _compare(tmp_path, ["straddler"], _current_at({"straddler": 25.0}), extra_args=["--threshold", "30"])
    assert result.returncode == 0
    assert "APPROACHING (+22.0%..+30.0%)" in result.stdout
    assert "1 cell(s) approaching the +30.0% threshold" in result.stdout


def test_empty_band_leaves_the_quiet_pass_output_unchanged(tmp_path: Path) -> None:
    result = _compare(tmp_path, ["a", "b"], _current_at({"a": 0.0, "b": -5.0}))
    assert result.returncode == 0
    assert result.stdout == "\nOK: no regressions > +20.0% on min across 2 benchmark(s).\n"


def test_history_row_records_verdict_worst_cell_and_band(tmp_path: Path) -> None:
    history = tmp_path / "gate-history.csv"
    result = _compare(
        tmp_path,
        ["creeping", "steady"],
        _current_at({"creeping": 19.0, "steady": 0.0}),
        extra_args=["--record-history", str(history)],
    )
    assert result.returncode == 0
    rows = list(csv.reader(history.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == [
        "date",
        "sha",
        "verdict",
        "metric",
        "threshold_pct",
        "worst_cell",
        "worst_delta_pct",
        "approaching",
    ]
    assert len(rows) == 2
    row = dict(zip(rows[0], rows[1]))
    assert row["date"] == datetime.date.today().isoformat()
    assert row["verdict"] == "OK"
    assert row["metric"] == "min"
    assert row["threshold_pct"] == "20.0"
    assert row["worst_cell"] == "creeping"
    assert row["worst_delta_pct"] == "+19.0"
    assert row["approaching"] == "creeping:+19.0%"


def test_history_appends_without_clobbering_and_records_failures(tmp_path: Path) -> None:
    history = tmp_path / "gate-history.csv"
    first = _compare(
        tmp_path,
        ["cell"],
        _current_at({"cell": 1.0}),
        extra_args=["--record-history", str(history)],
    )
    second = _compare(
        tmp_path,
        ["cell"],
        _current_at({"cell": 40.0}),
        extra_args=["--record-history", str(history)],
    )
    assert (first.returncode, second.returncode) == (0, 1)
    rows = list(csv.reader(history.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 3, rows  # header + one row per run
    assert rows[1][2] == "OK"
    assert rows[2][2] == "FAIL"
    assert rows[2][5:] == ["cell", "+40.0", ""]


def test_history_is_skipped_when_the_local_folder_is_absent(tmp_path: Path) -> None:
    """CI checkouts and fresh clones have no dev-docs/; the gate must not care."""
    history = tmp_path / "not-a-dir" / "gate-history.csv"
    result = _compare(
        tmp_path,
        ["cell"],
        _current_at({"cell": 0.0}),
        extra_args=["--record-history", str(history)],
    )
    assert result.returncode == 0
    assert "recurrence record skipped" in result.stdout
    assert not history.exists()
