"""Source contracts for Python extras and their CI consumers.

These tests deliberately inspect the project files without importing KGLite.
They catch metadata drift before a wheel is built; CI separately installs a
real wheel in a clean environment to prove the declared dependencies work.
"""

from __future__ import annotations

import ast
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


def test_classifiers_match_native_cpython_artifacts() -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    block = re.search(r"(?ms)^classifiers\s*=\s*(\[.*?\])", text)
    assert block is not None
    classifiers = set(ast.literal_eval(block.group(1)))
    assert "Programming Language :: Python :: Implementation :: CPython" in classifiers
    assert "Programming Language :: Python :: Implementation :: PyPy" not in classifiers
    assert "Operating System :: OS Independent" not in classifiers
    assert {
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    } <= classifiers


def test_wheel_policy_and_support_page_match_workflow() -> None:
    workflow = (WORKFLOWS / "build_wheels.yml").read_text(encoding="utf-8")
    support = (REPO_ROOT / "docs" / "python" / "platform-support.md").read_text(encoding="utf-8")
    targets = set(re.findall(r"(?m)^\s+(?:-\s+)?target:\s+([\w-]+)\s*$", workflow))
    assert targets
    assert all(f"`{target}`" in support for target in targets)
    assert "continue-on-error: true" in workflow
    assert "uploads platform wheels" in support
    assert "source distribution" in support
    assert "dist/*.tar.gz" not in workflow


def test_cp310_abi3_policy_does_not_claim_pypy_compatibility() -> None:
    cargo = (REPO_ROOT / "crates" / "kglite-py" / "Cargo.toml").read_text(encoding="utf-8")
    support = (REPO_ROOT / "docs" / "python" / "platform-support.md").read_text(encoding="utf-8")
    assert 'abi3 = ["pyo3/abi3-py310"]' in cargo
    assert "PyPy is not a supported published-artifact target" in support
    assert "requires a `pp310`-compatible artifact" in support


def test_bundled_mcp_entry_point_stays_declared() -> None:
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    assert 'kglite-mcp-server = "kglite.mcp_server:main"' in pyproject


def test_ci_compiles_packaged_optional_features_outside_workspace() -> None:
    ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    script = (REPO_ROOT / "scripts" / "check_packaged_features.sh").read_text(encoding="utf-8")
    assert "bash scripts/check_packaged_features.sh" in ci
    assert "cargo package -p kglite" in script
    assert "--features parallel-bz2" in script
