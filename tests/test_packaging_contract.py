"""Source contracts for Python extras and their CI consumers.

These tests deliberately inspect the project files without importing KGLite.
They catch metadata drift before a wheel is built; CI separately installs a
real wheel in a clean environment to prove the declared dependencies work.
"""

from __future__ import annotations

from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def _optional_dependency_blocks() -> dict[str, str]:
    text = PYPROJECT.read_text(encoding="utf-8")
    section = text.split("[project.optional-dependencies]", maxsplit=1)[1]
    section = section.split("\n[", maxsplit=1)[0]
    blocks: dict[str, str] = {}
    for match in re.finditer(r"(?m)^(\w+)\s*=\s*\[(.*?)\]", section, re.DOTALL):
        blocks[match.group(1)] = match.group(2)
    return blocks


def _requirement_names(block: str) -> set[str]:
    return {match.group(1).lower().replace("_", "-") for match in re.finditer(r"[\"']([A-Za-z0-9_-]+)", block)}


def test_networkx_extra_declares_every_runtime_dependency() -> None:
    requirements = _requirement_names(_optional_dependency_blocks()["networkx"])
    assert requirements == {"networkx", "pandas"}


def test_ci_only_installs_declared_project_extras() -> None:
    declared = set(_optional_dependency_blocks())
    referenced: set[str] = set()
    for workflow in WORKFLOWS.glob("*.yml"):
        text = workflow.read_text(encoding="utf-8")
        for match in re.finditer(r"\.\[([^]]+)\]", text):
            referenced.update(part.strip() for part in match.group(1).split(","))

    assert referenced <= declared, f"workflow references undefined project extras: {sorted(referenced - declared)}"


def test_ci_exercises_networkx_bridge_dependencies() -> None:
    ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    assert ".[neo4j,networkx]" in ci
