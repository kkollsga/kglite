"""Contracts that keep Python coverage reports scoped and reproducible."""

from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
CI_TEXT = (ROOT / ".github" / "workflows" / "ci.yml").read_text()


def test_python_coverage_scope_is_production_code_with_branches() -> None:
    run = PYPROJECT["tool"]["coverage"]["run"]
    assert run["source"] == ["kglite"]
    assert run["branch"] is True
    assert run["relative_files"] is True
    assert run["omit"] == ["kglite/**/tests/**"]


def test_coverage_tool_versions_are_exactly_pinned() -> None:
    requirements = (ROOT / "requirements" / "coverage.txt").read_text().splitlines()
    pins = {line for line in requirements if line and not line.startswith("#")}
    assert pins == {"coverage[toml]==7.10.7", "pytest-cov==7.1.0"}


def test_ci_uses_the_pinned_branch_coverage_contract() -> None:
    assert "pip install maturin pytest pandas hypothesis -r requirements/coverage.txt" in CI_TEXT
    assert (
        "pytest tests/ -v --tb=short --cov=kglite --cov-branch "
        "--cov-config=pyproject.toml --cov-report=xml:coverage.xml"
    ) in CI_TEXT
    assert "flags: python" in CI_TEXT
    assert "name: python" in CI_TEXT
    assert "disable_search: true" in CI_TEXT
