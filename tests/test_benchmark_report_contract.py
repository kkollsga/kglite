"""Metadata and rendering contracts for the public benchmark snapshot."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from benchmarks.competitive.graphsuite import marketing, results

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "benchmarks" / "competitive" / "graphsuite" / "results.json"
REPORT = REPO_ROOT / "BENCHMARKS.md"


def test_historical_capture_records_known_provenance_and_unknowns() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert data["schema_version"] == results.SCHEMA_VERSION
    assert data["harness"] == {"name": "graphsuite", "version": 1}
    historical = data["historical_capture"]
    assert historical["origin"] == "manual"
    assert historical["dataset_seed"] == 1234
    assert historical["source_commit"] is None
    assert "did not capture" in historical["source_commit_note"]
    assert historical["results_first_committed_in"] == "2b61b350a3e7db99ff79cd43462ad8fe16d2cdca"


def test_public_report_is_generated_from_committed_metadata() -> None:
    assert REPORT.read_text(encoding="utf-8") == marketing.render()


def test_report_only_command_is_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "BENCHMARKS.md"
    subprocess.run(
        [
            sys.executable,
            REPO_ROOT / "benchmarks" / "benchmark.py",
            "--report-only",
            "--out",
            output,
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    assert output.read_bytes() == REPORT.read_bytes()


def test_future_run_shape_carries_exact_provenance(monkeypatch) -> None:
    monkeypatch.setattr(results, "_machine", lambda: {"platform": "test", "python": "3.12.0"})
    run = results.make_run(
        library="example",
        version="1.2.3",
        run_date="2026-07-14T12:00:00+02:00",
        ds_scale="small",
        ds_signature="small-s7-n1-e0",
        n_nodes=1,
        n_edges=0,
        groups={},
        dataset_seed=7,
        provenance={
            "harness_version": results.HARNESS_VERSION,
            "origin": "ci",
            "source_commit": "a" * 40,
            "source_dirty": False,
            "base_repeats": 9,
        },
    )
    assert run["run_date"].endswith("+02:00")
    assert run["dataset"]["seed"] == 7
    assert run["provenance"] == {
        "harness_version": 2,
        "origin": "ci",
        "source_commit": "a" * 40,
        "source_dirty": False,
        "base_repeats": 9,
    }
