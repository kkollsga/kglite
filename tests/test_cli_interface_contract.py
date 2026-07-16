"""Clap help, error, and version contracts for the shipped `kglite` CLI."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from tests.test_cli_shell_smoke import BINARY

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tests" / "api-baselines" / "cli-interface.json"
COMMANDS = {
    "query": ("query",),
    "write": ("write",),
    "ready-set": ("ready-set",),
    "describe": ("describe",),
    "session": ("session",),
    "export-text": ("export-text",),
    "diff": ("diff",),
    "skill": ("skill",),
    "skill-install": ("skill", "install"),
    "skill-uninstall": ("skill", "uninstall"),
}

pytestmark = pytest.mark.skipif(not BINARY.exists(), reason="kglite CLI binary not built")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(BINARY), *args], capture_output=True, text=True, timeout=30)


def _text(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().splitlines()) + "\n"


def capture_cli_contract() -> dict:
    help_text = {"root": _text(_run("--help").stdout)}
    for name, command in COMMANDS.items():
        help_text[name] = _text(_run(*command, "--help").stdout)

    errors = {}
    for name, args in {
        "unknown_subcommand": ("unknown-command",),
        "missing_query_args": ("query",),
        "graph_subcommand_conflict": ("graph.kgl", "query", "graph.kgl", "RETURN 1"),
    }.items():
        proc = _run(*args)
        errors[name] = {"code": proc.returncode, "stderr": _text(proc.stderr)}
    return {"help": help_text, "errors": errors}


def test_cli_help_and_error_contract_matches_baseline():
    assert capture_cli_contract() == json.loads(BASELINE.read_text())


def test_cli_version_tracks_workspace_version():
    metadata = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    packages = json.loads(metadata.stdout)["packages"]
    expected = next(package["version"] for package in packages if package["name"] == "kglite-cli")
    proc = _run("--version")
    assert proc.returncode == 0
    assert proc.stdout.strip() == f"kglite {expected}"
