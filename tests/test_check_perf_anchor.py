"""Contract tests for `scripts/check_perf_anchor.py` — the cumulative-drift gate.

The script had no tests at all when the dispersion guard was added, so these
cover both what it already promised (a regression over the threshold fails, and
names the cell) and the guard itself: `min` reports the best round, so a capture
whose fast tail collapses reads as a same-sized regression on every cell at once
(the 0.16.17 anchor, +16.3% worst across 34/34 cells including the controls,
proved environmental by a wheel A/B). Comparing the two captures' `median/min`
shape separates that from real drift.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_perf_anchor.py"

#: Enough cells to clear the script's default `--min-overlap` of 10.
CELLS = [f"test_bench_cell_{i:02d}" for i in range(12)]


def _capture(path: Path, timings: dict[str, float], dispersion: float | None = 1.05) -> None:
    """Write one baseline JSON in the on-disk pytest-benchmark schema.

    ``dispersion`` is the ``median/min`` ratio given to every cell; ``None``
    omits ``median`` entirely, which is the legacy shape on disk
    (`hardening_0_13_2.json` carries only ``min``).
    """
    benchmarks = []
    for name, value in timings.items():
        stats: dict[str, float] = {"min": value, "max": value * 3, "rounds": 100}
        if dispersion is not None:
            stats["median"] = value * dispersion
            stats["mean"] = value * dispersion
        benchmarks.append({"name": name, "fullname": f"tests/benchmarks/x.py::{name}", "stats": stats})
    path.write_text(
        json.dumps(
            {"machine_info": {"system": "Darwin", "machine": "arm64"}, "datetime": path.stem, "benchmarks": benchmarks}
        ),
        encoding="utf-8",
    )
    from scripts.benchmark_qualification import qualify

    qualify(path, "accepted", "Synthetic reference for regression gate tests")


def run(baselines: Path, *args: str) -> tuple[int, str]:
    """Invoke the script against a synthetic baselines directory."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--baselines-dir", str(baselines), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _pair(tmp_path: Path, *, current: dict[str, float], anchor_dispersion=1.05, current_dispersion=1.05) -> Path:
    """An anchor at a flat 1 ms per cell plus a current capture over it."""
    root = tmp_path / "baselines"
    root.mkdir()
    _capture(root / "0_16_14.json", dict.fromkeys(CELLS, 1e-3), anchor_dispersion)
    _capture(root / "0_16_17.json", current, current_dispersion)
    return root


def test_a_regression_over_the_threshold_fails_and_names_the_cell(tmp_path: Path) -> None:
    timings = dict.fromkeys(CELLS, 1e-3)
    timings["test_bench_cell_04"] = 2e-3
    root = _pair(tmp_path, current=timings)

    code, out = run(root)
    assert code == 1, out
    assert "FAIL" in out
    assert "test_bench_cell_04: +100.0%" in out, out


def test_a_flat_capture_passes_with_no_dispersion_warning(tmp_path: Path) -> None:
    """The control: same timings, same distribution shape, nothing to say."""
    root = _pair(tmp_path, current=dict.fromkeys(CELLS, 1e-3))

    code, out = run(root)
    assert code == 0, out
    assert "WARNING" not in out, out
    assert "dispersion" in out
    assert "anchor 1.050, current 1.050" in out, out


def test_a_dispersion_shift_warns_without_changing_the_exit_code(tmp_path: Path) -> None:
    """The 0.16.17 shape: identical `min` values, a collapsed fast tail."""
    root = _pair(tmp_path, current=dict.fromkeys(CELLS, 1e-3), anchor_dispersion=1.05, current_dispersion=1.00)

    code, out = run(root)
    assert code == 0, out
    assert "WARNING" in out, out
    assert "-4.8%" in out, out


def test_a_capture_without_median_reports_n_a_and_no_warning(tmp_path: Path) -> None:
    """Legacy baselines carry only `min`; they get no number, not a made-up one."""
    root = _pair(tmp_path, current=dict.fromkeys(CELLS, 1e-3), anchor_dispersion=None)

    code, out = run(root)
    assert code == 0, out
    assert "anchor n/a" in out, out
    assert "WARNING" not in out, out
