"""Contracts that keep Python coverage reports scoped and reproducible."""

from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 support; installed via coverage[toml].
    import tomli as tomllib

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
    """Reproducibility contract: exactly these two tools, each `==`-pinned to
    a full version. The version numbers themselves are deliberately NOT
    duplicated here — a hard-coded copy turned every Dependabot coverage bump
    into a guaranteed CI failure (the requirements file is the single owner
    of the pin; this test owns the *shape*)."""
    requirements = (ROOT / "requirements" / "coverage.txt").read_text().splitlines()
    pins = sorted(line for line in requirements if line and not line.startswith("#"))
    assert [pin.split("==")[0] for pin in pins] == ["coverage[toml]", "pytest-cov"]
    for pin in pins:
        assert re.fullmatch(r"[a-zA-Z0-9_.\[\]-]+==\d+(\.\d+)+", pin), f"not an exact pin: {pin!r}"


def test_ci_uses_the_pinned_branch_coverage_contract() -> None:
    assert "pip install maturin pytest pytest-timeout pandas hypothesis -r requirements/coverage.txt" in CI_TEXT
    assert (
        "pytest tests/ -v --tb=short --cov=kglite --cov-branch "
        "--cov-config=pyproject.toml --cov-report=xml:coverage.xml"
    ) in CI_TEXT
    assert "flags: python" in CI_TEXT
    assert "name: python" in CI_TEXT
    assert "disable_search: true" in CI_TEXT


def test_rust_coverage_is_component_scoped_and_report_only() -> None:
    assert "tool: cargo-llvm-cov@0.8.7" in CI_TEXT
    command = (
        "cargo llvm-cov --package kglite --lib --tests --ignore-filename-regex 'src/bin/' "
        "--lcov --output-path rust-core-coverage.lcov"
    )
    assert command in CI_TEXT
    assert command in (ROOT / "Makefile").read_text()
    assert "flags: rust-core" in CI_TEXT
    assert "name: rust-core" in CI_TEXT
    assert "--fail-under" not in CI_TEXT
