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
    crate_readmes = sorted((REPO_ROOT / "crates").glob("*/README.md"))
    return [REPO_ROOT / "README.md", REPO_ROOT / "CONTRIBUTING.md", *crate_readmes, *docs]


def _prose_without_code(text: str) -> str:
    text = re.sub(r"(?ms)^```.*?^```\s*$", "", text)
    return re.sub(r"`[^`\n]+`", "", text)


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


def test_dataframe_walkthroughs_name_the_pandas_extra() -> None:
    for relative in (
        "README.md",
        "docs/python/getting-started.md",
        "docs/python/guides/data-loading.md",
    ):
        assert "kglite[pandas]" in (REPO_ROOT / relative).read_text(encoding="utf-8")


def test_readme_leads_with_install_query_and_reference_paths() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("## Start here")
    ecosystem = readme.index("## Ecosystem")
    onboarding = readme[start:ecosystem]
    assert start < 2_000, "README onboarding drifted below the opening screen"
    for required in (
        "pip install kglite",
        "kglite.from_records",
        "Getting Started",
        "Python API",
        "Cypher",
        "MCP and agents",
        "Operators",
        "all documentation",
    ):
        assert required in onboarding


def test_docs_home_leads_with_high_level_routes() -> None:
    index = (REPO_ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    start = index.index("## Start here")
    for required in ("Python", "Cypher", "MCP", "Rust", "Operators", "Reference"):
        assert required in index[start : start + 2_500]


def test_python_guide_navigation_includes_every_guide() -> None:
    guide_dir = REPO_ROOT / "docs" / "python" / "guides"
    guide_index = (REPO_ROOT / "docs" / "python" / "index.md").read_text(encoding="utf-8")
    missing = [
        path.stem
        for path in sorted(guide_dir.glob("*.md"))
        if path.name != "index.md" and f"guides/{path.stem}" not in guide_index
    ]
    assert not missing, f"Python guides missing from the ReadTheDocs navigation: {missing}"


def test_retired_documentation_contracts_do_not_return() -> None:
    active = "\n".join(path.read_text(encoding="utf-8") for path in _active_markdown())
    retired = (
        "returning partial results",
        "That's seven tools",
        "The 12 bundled tools",
        "Change the primary type via `SET n.type",
        "use `SET n.type = 'NewType'` to retype",
        "use ``SET n.type = 'NewType'`` to retype",
        "including the six structural validators",
        "kglite_string_free",
        '35 `extern "C"`',
        '30 `extern "C"`',
        "KGLITE_OK",
        "Status: Phase",
    )
    assert not [claim for claim in retired if claim in active]

    topics = (REPO_ROOT / "crates" / "kglite" / "src" / "graph" / "introspection" / "topics.rs").read_text(
        encoding="utf-8"
    )
    assert 'feature=\\"FOREACH\\"' not in topics


def test_documented_make_commands_are_real_targets() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"(?m)^([A-Za-z0-9_.-]+):", makefile))
    documented: set[str] = set()
    for path in _active_markdown():
        documented.update(re.findall(r"(?m)^\s*(?:\$\s*)?make\s+([A-Za-z0-9_.-]+)", path.read_text()))
    assert documented <= targets, f"docs name missing Make targets: {sorted(documented - targets)}"


def test_active_markdown_local_links_resolve() -> None:
    missing: list[str] = []
    for path in _active_markdown():
        prose = _prose_without_code(path.read_text(encoding="utf-8"))
        for target in re.findall(r"!?\[[^]]*\]\(([^)]+)\)", prose):
            target = target.split("#", maxsplit=1)[0]
            if not target or "://" in target or target.startswith(("mailto:", "{")):
                continue
            if not (path.parent / target).resolve().exists():
                missing.append(f"{path.relative_to(REPO_ROOT)} -> {target}")
    assert not missing, "active docs contain broken local links:\n" + "\n".join(missing)


def test_retired_architecture_claims_do_not_return() -> None:
    architecture = (REPO_ROOT / "docs" / "concepts" / "architecture.md").read_text()
    decisions = (REPO_ROOT / "docs" / "concepts" / "design-decisions.md").read_text()
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text()
    readme = (REPO_ROOT / "README.md").read_text()
    retired = {
        "architecture": ["There is no R-tree", "RGF\\x02", "Gzip-compressed"],
        "design decisions": ["Single-process only", "Memory-bound", "Why no R-tree"],
        "contributing": ["src/                          # Rust core", "no enforced formatter"],
        "readme": ["KGLite is `v0.11.x`", "pandas >= 1.5"],
    }
    for label, text in {
        "architecture": architecture,
        "design decisions": decisions,
        "contributing": contributing,
        "readme": readme,
    }.items():
        assert not [claim for claim in retired[label] if claim in text]


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


def test_c_abi_distribution_docs_match_source_only_releases() -> None:
    c_abi = (REPO_ROOT / "docs" / "rust" / "c-abi.md").read_text(encoding="utf-8")
    binding = (REPO_ROOT / "docs" / "rust" / "implementing-a-binding.md").read_text(encoding="utf-8")
    publishing = (REPO_ROOT / ".github" / "workflows" / "publish_crates.yml").read_text(encoding="utf-8")
    truth = "Precompiled C ABI libraries are not currently attached to releases"
    assert truth in c_abi
    assert truth in binding
    assert "release workflow separately" not in publishing
