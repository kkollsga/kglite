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
    assert "query" in result.stdout
    assert "session" in result.stdout
    assert "skill" not in result.stdout


def test_wheel_cli_installs_no_skill(tmp_path: Path) -> None:
    """`kglite skill install` moved to codingest — the wheel must not honor it.

    `HOME` points at `tmp_path`, so a surviving installer leaves evidence on
    disk and the assertion fails on behaviour, not only on the exit code.
    """
    env = {**os.environ, "HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kglite.cli",
            "skill",
            "install",
            "--host",
            "codex",
        ],
        input="",
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=env,
    )
    assert result.returncode != 0, result.stdout
    assert "kglite-code-review" not in result.stdout
    assert not (tmp_path / ".codex").exists()
    assert not (tmp_path / ".claude").exists()
