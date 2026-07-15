"""Contracts that keep Python coverage reports scoped and reproducible."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 support; installed via coverage[toml].
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
CORE_MANIFEST = tomllib.loads((ROOT / "crates" / "kglite" / "Cargo.toml").read_text())
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
    assert pins == {"coverage[toml]==7.15.1", "pytest-cov==7.1.0"}


def test_ci_uses_the_pinned_branch_coverage_contract() -> None:
    assert "pip install maturin pytest pandas hypothesis -r requirements/coverage.txt" in CI_TEXT
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


def test_default_feature_coverage_excludes_sec_only_live_test() -> None:
    live_test = next(target for target in CORE_MANIFEST["test"] if target["name"] == "datasets_sec_fetch_live")
    assert live_test["required-features"] == ["sec"]
