"""Clap help, error, and version contracts for the shipped `kglite` CLI."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from tests.test_cli_shell_smoke import BINARY, SKIP_REASON

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
    "export-sqlite": ("export-sqlite",),
    "migrate": ("migrate",),
    "schema-version": ("schema-version",),
}

#: Current documentation that must route skill installation to codingest.
SKILL_DOCS = (
    ROOT / "crates" / "kglite-cli" / "README.md",
    ROOT / "docs" / "operators" / "cli.md",
)

requires_binary = pytest.mark.skipif(SKIP_REASON is not None, reason=SKIP_REASON or "")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(BINARY), *args], capture_output=True, text=True, timeout=30)


def _shell_input(*argv: str) -> subprocess.CompletedProcess[str]:
    """Run `argv` with stdin closed — an argv that falls through to the REPL
    would otherwise block on the interactive prompt."""
    return subprocess.run(list(argv), input="", capture_output=True, text=True, timeout=30)


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


@requires_binary
def test_cli_help_and_error_contract_matches_baseline():
    assert capture_cli_contract() == json.loads(BASELINE.read_text(encoding="utf-8"))


@requires_binary
def test_cli_surface_omits_the_retired_skill_command():
    """Skill installation moved to codingest; the CLI must not offer it.

    Asserted four ways, so a partial reintroduction — help entry without
    dispatch, dispatch without help entry, or a refreshed baseline that
    quietly records either — still fails. `kglite` has no unknown-subcommand
    error (a bare stray word becomes the `[GRAPH]` positional), so dispatch
    absence is proven by the missing-graph note as well as by the exit code.
    """
    root_help = _run("--help")
    assert root_help.returncode == 0
    assert "skill" not in root_help.stdout

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert not [key for key in baseline["help"] if "skill" in key]
    assert "skill" not in baseline["help"]["root"]

    bare = _shell_input(str(BINARY), "skill")
    assert bare.returncode == 0, bare.stderr
    assert "does not exist" in bare.stderr, bare.stderr + bare.stdout

    install = _shell_input(str(BINARY), "skill", "install", "--host", "codex")
    assert install.returncode != 0, install.stdout


def test_cli_docs_route_skill_installation_to_codingest():
    """Current docs must hand skill installation to codingest.

    Prose *about* the retired `kglite skill install` is allowed (the migration
    note names it); a runnable command line teaching it is not, so the check
    is on command lines, not on the word appearing anywhere.
    """
    for path in SKILL_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "codingest skill install" in text, path
        commands = [line.strip().lstrip("$ ").strip() for line in text.splitlines()]
        taught = [line for line in commands if line.startswith("kglite skill")]
        assert not taught, f"{path} still teaches: {taught}"


@requires_binary
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
