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
    paths = re.findall(r"`((?:crates|tests|scripts|docs)/[^`]+)`", GENERATED.read_text(encoding="utf-8"))
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
        references.update(re.findall(r"kglite\[([a-z0-9_-]+)\]", path.read_text(encoding="utf-8")))
    assert references <= declared, f"docs name removed extras: {sorted(references - declared)}"


def test_dataframe_walkthroughs_name_the_pandas_extra() -> None:
    # The README installs plain `pip install kglite` (user call, 2026-08-27:
    # naming the extra there is pedantic); the guides still teach it.
    for relative in (
        "docs/python/getting-started.md",
        "docs/python/guides/data-loading.md",
    ):
        assert "kglite[pandas]" in (REPO_ROOT / relative).read_text(encoding="utf-8")


def test_readme_leads_with_install_query_and_reference_paths() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    # One merged onboarding section (2026-08-27 restructure): a single install
    # block, a single first snippet, and the two doorways at its end. The old
    # "## Start here" teaser that preceded it is gone.
    assert "## Start here" not in readme
    start = readme.index("## Quick Start")
    what_makes = readme.index("## What makes it different")
    onboarding = readme[start:what_makes]
    assert start < 2_000, "README onboarding drifted below the opening screen"
    install_block = "```bash\npip install kglite"
    assert readme.count(install_block) == 1, "onboarding install block duplicated"
    # The opening screen carries exactly two doorways (user call, 2026-08-27:
    # less early information): Getting Started and the AI-agents guide. The
    # reference stack and track indexes live in the body sections instead.
    for required in (
        "pip install kglite",
        "graph.cypher(",
        "Getting Started",
        "guides/ai-agents",
    ):
        assert required in onboarding


def test_docs_home_leads_with_high_level_routes() -> None:
    index = (REPO_ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    start = index.index("## Start here")
    for required in (
        "Python",
        "Cypher",
        "MCP",
        "Rust",
        "Operators",
        "Reference",
    ):
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


def test_legacy_manifest_cypher_docs_do_not_claim_schema_validation() -> None:
    active_docs = "\n".join(
        (REPO_ROOT / relative).read_text(encoding="utf-8")
        for relative in (
            "docs/python/guides/mcp-servers.md",
            "docs/python/examples/manifest_cypher_tool.md",
        )
    )
    describe = (REPO_ROOT / "crates/kglite/src/graph/introspection/describe.rs").read_text(encoding="utf-8")
    retired = (
        "Validation runs at server startup",
        "$param refs are validated at server startup",
        "JSON-Schema-validated `year` argument",
        "cypher references $params",
        "invalid parameters schema",
        "client enforces schema validation",
    )
    assert not [claim for claim in retired if claim in active_docs or claim in describe]
    assert "published as the tool's MCP input schema" in active_docs
    assert "does not validate that schema" in describe


def test_recipe_query_docs_cover_automation_and_domain_boundaries() -> None:
    guide = (REPO_ROOT / "docs/python/guides/mcp-servers.md").read_text(encoding="utf-8")
    example = (REPO_ROOT / "examples/local_code_review_mcp.yaml").read_text(encoding="utf-8")
    skill = (REPO_ROOT / "examples/local_code_review_mcp.skills/initial_code_review.md").read_text(encoding="utf-8")

    for contract in (
        "list_recipe_queries",
        "run_recipe_query",
        "parameter-free queries receive `{}`",
        "rows: []",
        "include_cypher=true",
        "result_limit_exceeded",
        "details.observed_count",
        "literal stored `LIMIT 200`",
        "stale_graph",
        "query_failed.details.cause",
        "multi_revision_graph_required",
        "unknown_revision",
        "ORDER BY",
        "FORMAT CSV",
    ):
        assert contract in guide, contract

    assert example.count("        resolve_function:") == 1
    assert example.count("        direct_callers:") == 1
    assert example.count("        affected_tests:") == 1
    assert example.count("required: [qualified_name]") == 3
    assert example.count("additionalProperties: false") == 3
    assert example.count("ORDER BY qualified_name, file_path, line_number") == 3
    assert "LIMIT 200" not in example

    resolve = skill.index('query="resolve_function"')
    callers = skill.index("call `direct_callers` and `affected_tests`")
    assert resolve < callers
    assert "Do not interpret empty caller or test rows" in skill
    assert "use raw" in skill
    assert "`cypher_query`" in skill


def test_pre_014_persistence_is_never_documented_as_readable() -> None:
    active = "\n".join(path.read_text(encoding="utf-8") for path in _active_markdown())
    retired = (
        "readers accept supported v4",
        "supported v4 files remain readable",
        "v4 remains readable",
        "reader accepts v5/v4",
        "supported v4 inputs load",
        "supported v4 legacy decoder",
        "format v1 — still import",
        "every `.kgl` you saved keeps loading",
        "`.kgl` files load across the boundary in both directions",
    )
    assert not [claim for claim in retired if claim in active]


def test_documented_make_commands_are_real_targets() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"(?m)^([A-Za-z0-9_.-]+):", makefile))
    documented: set[str] = set()
    for path in _active_markdown():
        documented.update(re.findall(r"(?m)^\s*(?:\$\s*)?make\s+([A-Za-z0-9_.-]+)", path.read_text(encoding="utf-8")))
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
    architecture = (REPO_ROOT / "docs" / "concepts" / "architecture.md").read_text(encoding="utf-8")
    decisions = (REPO_ROOT / "docs" / "concepts" / "design-decisions.md").read_text(encoding="utf-8")
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
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
    publishing = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    truth = "Precompiled C ABI libraries are not currently attached to releases"
    assert truth in c_abi
    assert truth in binding
    assert "release workflow separately" not in publishing


def test_readme_sells_its_distinctive_capabilities() -> None:
    """The README's allocation contract, from the 2026-08-27 editorial review.

    The drift this pins is append-per-release without a whole-document edit:
    duplicated boilerplate crowds out the capabilities no competing embedded
    engine has, and a feature that ships on Monday is invisible on Wednesday.
    Thresholds are the shipped document's actual counts, not aspirations — each
    one fails if the fact it guards is edited away.
    """
    readme = REPO_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    lines = text.splitlines()
    lowered = text.lower()

    # The 0.13 -> 0.14 migration is retired from the README entirely
    # (user call, 2026-08-27: too old to be relevant); the guide itself
    # stays in docs/ for stragglers.
    assert "0.13 → 0.14" not in text and "0.13-to-0.14" not in text

    # As-of temporal filtering and the declared ontology are the two zero- and
    # one-mention capabilities the review found buried. Keep them sold.
    assert lowered.count("valid_at") >= 2
    assert lowered.count("ontology") >= 3
    assert "define_ontology" in lowered or "ontology_audit" in lowered

    # House voice: no em-dashes, so an append cannot reintroduce them.
    assert "\u2014" not in text

    # Runnable value above the fold, and a length budget so the next append has
    # to displace something rather than accrete.
    quick_start = next(i for i, line in enumerate(lines, 1) if line.startswith("## Quick Start"))
    assert quick_start <= 37, f"Quick Start sank to line {quick_start}"
    assert len(lines) <= 600, f"README grew to {len(lines)} lines"


def test_readme_links_every_python_guide() -> None:
    """Every shipped guide is reachable from the README.

    The README's Documentation section is the guide index, so a guide added
    without a link there is a page no README reader can find. Guides are
    enumerated from disk, not listed here, so a new one fails this test until
    it is linked.
    """
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    guides = sorted(
        path.stem for path in (REPO_ROOT / "docs" / "python" / "guides").glob("*.md") if path.stem != "index"
    )
    assert guides, "no Python guides found on disk"
    missing = [stem for stem in guides if f"python/guides/{stem}.html" not in readme]
    assert not missing, f"guides with no README link: {missing}"
