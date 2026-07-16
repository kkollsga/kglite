"""Smoke the Rust CLI as hosted by the installed ``kglite`` wheel."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import kglite


def test_wheel_exposes_cli_entry_point() -> None:
    assert hasattr(kglite, "_run_cli")


def test_python_module_launcher_forwards_help_to_rust() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "kglite.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "skill" in result.stdout
    assert "skill" in result.stdout


def test_python_module_launcher_runs_skill_dry_run(tmp_path: Path) -> None:
    env = {**os.environ, "HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kglite.cli",
            "skill",
            "install",
            "--dry-run",
            "--host",
            "codex",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert str(tmp_path / ".codex" / "skills" / "kglite-code-review") in result.stdout
    assert not (tmp_path / ".codex").exists()
