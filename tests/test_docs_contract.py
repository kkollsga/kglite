"""Contracts that keep active documentation tied to repository facts."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RENDER = REPO_ROOT / "scripts" / "render_docs_facts.py"
GENERATED = REPO_ROOT / "docs" / "_generated" / "project-facts.md"


def _active_markdown() -> list[Path]:
    docs = [path for path in (REPO_ROOT / "docs").rglob("*.md") if "history" not in path.parts]
    return [REPO_ROOT / "README.md", REPO_ROOT / "CONTRIBUTING.md", *docs]


def test_generated_project_facts_are_current() -> None:
    subprocess.run([sys.executable, RENDER, "--check"], cwd=REPO_ROOT, check=True)


def test_generator_is_idempotent(tmp_path: Path) -> None:
    output = tmp_path / "facts.md"
    command = [sys.executable, RENDER, "--output", output]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    first = output.read_bytes()
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    assert output.read_bytes() == first


def test_check_mode_rejects_a_stale_fixture(tmp_path: Path) -> None:
    output = tmp_path / "facts.md"
    output.write_text("stale\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, RENDER, "--check", "--output", output],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "python scripts/render_docs_facts.py" in result.stderr


def test_generated_source_paths_exist() -> None:
    paths = re.findall(r"`((?:crates|tests|scripts|docs)/[^`]+)`", GENERATED.read_text())
    assert paths
    missing = [path for path in paths if not (REPO_ROOT / path).exists()]
    assert not missing, f"generated documentation names missing source paths: {missing}"


def test_active_docs_only_name_declared_extras() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    optional = pyproject.split("[project.optional-dependencies]", maxsplit=1)[1]
    optional = optional.split("\n[", maxsplit=1)[0]
    declared = set(re.findall(r"(?m)^(\w+)\s*=", optional))
    references: set[str] = set()
    for path in _active_markdown():
        references.update(re.findall(r"kglite\[([a-z0-9_-]+)\]", path.read_text()))
    assert references <= declared, f"docs name removed extras: {sorted(references - declared)}"


def test_documented_make_commands_are_real_targets() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"(?m)^([A-Za-z0-9_.-]+):", makefile))
    documented: set[str] = set()
    for path in _active_markdown():
        documented.update(re.findall(r"(?m)^\s*(?:\$\s*)?make\s+([A-Za-z0-9_.-]+)", path.read_text()))
    assert documented <= targets, f"docs name missing Make targets: {sorted(documented - targets)}"


@pytest.mark.parametrize(
    ("path", "retired_claim"),
    [
        (REPO_ROOT / ".github" / "workflows" / "ci.yml", ".[mcp"),
        (REPO_ROOT / "README.md", "kglite[mcp]"),
        (REPO_ROOT / "docs" / "index.md", "kglite[mcp]"),
    ],
)
def test_retired_install_claims_do_not_return(path: Path, retired_claim: str) -> None:
    assert retired_claim not in path.read_text(encoding="utf-8")
