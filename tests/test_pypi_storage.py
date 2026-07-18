from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_pypi_storage.py"


def _run(tmp_path: Path, releases: object, *args: str) -> subprocess.CompletedProcess[str]:
    payload = tmp_path / "project.json"
    payload.write_text(json.dumps({"releases": releases}), encoding="utf-8")
    return subprocess.run(
        [sys.executable, SCRIPT, "--json", payload, *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_storage_check_passes_below_projected_threshold(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        {"0.1.0": [{"size": 100}, {"size": 200, "yanked": True}], "0.1.1": []},
        "--project-limit-bytes",
        "1000",
        "--action-threshold",
        "0.8",
        "--reserve-bytes",
        "100",
    )
    assert result.returncode == 0, result.stderr
    assert "400" in result.stdout
    assert "2 files" in result.stdout


def test_storage_check_blocks_at_projected_threshold(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        {"0.1.0": [{"size": 700}]},
        "--project-limit-bytes",
        "1000",
        "--action-threshold",
        "0.8",
        "--reserve-bytes",
        "100",
    )
    assert result.returncode == 1
    assert "request a project-limit increase" in result.stderr
    assert "do not delete releases automatically" in result.stderr


def test_storage_check_rejects_malformed_file_size(tmp_path: Path) -> None:
    result = _run(tmp_path, {"0.1.0": [{"size": -1}]})
    assert result.returncode == 2
    assert "invalid file size" in result.stderr


def test_wheel_release_runs_capacity_check_before_builds() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "build_wheels.yml").read_text(encoding="utf-8")
    version_job = workflow.split("  ci-gate:", maxsplit=1)[0]
    assert "python scripts/check_pypi_storage.py" in version_job
    assert "--project-limit-bytes 10000000000" in version_job
    assert "--action-threshold 0.80" in version_job
    assert "--reserve-bytes 250000000" in version_job
