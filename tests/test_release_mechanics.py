"""Guards on the release-mechanics tooling itself.

Every check here exists because the corresponding gate was, or could
silently become, vacuous. These are pure-Python and import no native
extension, so they cost the suite nothing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


bump_version = _load("bump_version")
check_release_hygiene = _load("check_release_hygiene")
wait_for_release_ci = _load("wait_for_release_ci")


# ── the version bump reaches every manifest ────────────────────────────


def test_every_internal_pin_matches_the_workspace_version():
    """The gate `make gate` runs. If this fails, `cargo metadata` is one
    minor bump away from refusing to resolve the workspace."""
    assert bump_version.check() == []


def test_the_four_publishing_members_actually_pin_the_engine():
    """Guards the *discovery*, not just the comparison: if the pins were
    renamed or dropped, `check()` would trivially pass on an empty set.
    """
    pinned = {
        manifest.parent.name
        for manifest in bump_version.member_manifests()
        if any(key == "kglite" for _, key, _ in bump_version.internal_version_pins(manifest))
    }
    assert pinned == {"kglite-bolt-server", "kglite-c", "kglite-cli", "kglite-mcp-server"}


def test_pins_carry_the_full_x_y_z_not_the_series():
    version = bump_version.read_workspace_version()
    assert version.count(".") == 2
    for manifest in bump_version.member_manifests():
        for _, _, requirement in bump_version.internal_version_pins(manifest):
            assert requirement == version


# ── CHANGELOG structure ────────────────────────────────────────────────


def test_changelog_unreleased_section_is_clean():
    assert check_release_hygiene.check_changelog() == []


def test_no_release_constant_todo_markers_survive():
    assert check_release_hygiene.check_refresh_todos() == []


def test_changelog_lint_rejects_a_duplicate_heading(tmp_path, monkeypatch):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        "# Changelog\n\n## [Unreleased]\n\n### Changed\n\n- a\n\n### Changed\n\n- b\n\n## [0.1.0] - 2020-01-01\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(check_release_hygiene, "CHANGELOG", changelog)
    problems = check_release_hygiene.check_changelog()
    assert len(problems) == 1
    assert "duplicate `### Changed`" in problems[0]


def test_changelog_lint_rejects_a_non_keep_a_changelog_heading(tmp_path, monkeypatch):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("# Changelog\n\n## [Unreleased]\n\n### Perfomance\n\n- typo\n", encoding="utf-8")
    monkeypatch.setattr(check_release_hygiene, "CHANGELOG", changelog)
    problems = check_release_hygiene.check_changelog()
    assert len(problems) == 1
    assert "not a Keep a Changelog group" in problems[0]


def test_changelog_lint_rejects_a_deleted_unreleased_section(tmp_path, monkeypatch):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("# Changelog\n\n## [0.1.0] - 2020-01-01\n\n### Added\n\n- a\n", encoding="utf-8")
    monkeypatch.setattr(check_release_hygiene, "CHANGELOG", changelog)
    problems = check_release_hygiene.check_changelog()
    assert len(problems) == 1
    assert "must be `## [Unreleased]`" in problems[0]


def test_todo_lint_rejects_a_surviving_marker(tmp_path, monkeypatch):
    written = tmp_path / "tests" / "test_phase5_parity.py"
    written.parent.mkdir(parents=True)
    written.write_text("# TODO: describe what grew since the prior baseline.\n", encoding="utf-8")
    monkeypatch.setattr(check_release_hygiene, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(check_release_hygiene, "REFRESH_WRITTEN_FILES", (Path("tests/test_phase5_parity.py"),))
    problems = check_release_hygiene.check_refresh_todos()
    assert len(problems) == 1
    assert "unresolved release-constants marker" in problems[0]


# ── the CI poller does not declare victory prematurely ─────────────────

WORKFLOWS = ("CI", "Publish to crates.io")


def _run(name, status, conclusion, number=1):
    return {
        "id": hash((name, number)) & 0xFFFF,
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "run_number": number,
    }


def test_empty_run_list_is_never_a_pass():
    """The recorded bug: a zero-incomplete loop exits instantly green on
    an empty array. `runs=0` persisted for an hour during 0.15.0."""
    all_terminal, missing, pending, failed = wait_for_release_ci.evaluate([], WORKFLOWS)
    assert all_terminal is False
    assert missing == list(WORKFLOWS)
    assert failed == []


def test_a_partial_workflow_set_is_never_a_pass():
    runs = [_run("CI", "completed", "success")]
    all_terminal, missing, pending, _ = wait_for_release_ci.evaluate(runs, WORKFLOWS)
    assert all_terminal is False
    assert missing == ["Publish to crates.io"]


def test_completed_but_failed_is_reported_as_failed_not_completed():
    runs = [_run("CI", "completed", "failure"), _run("Publish to crates.io", "completed", "success")]
    all_terminal, _, _, failed = wait_for_release_ci.evaluate(runs, WORKFLOWS)
    assert all_terminal is True
    assert failed == ["CI: failure"]


def test_all_success_is_the_only_pass():
    runs = [_run(name, "completed", "success") for name in WORKFLOWS]
    all_terminal, missing, pending, failed = wait_for_release_ci.evaluate(runs, WORKFLOWS)
    assert (all_terminal, missing, pending, failed) == (True, [], [], [])


def test_a_rerun_supersedes_the_earlier_attempt():
    runs = [_run("CI", "completed", "failure", 1), _run("CI", "completed", "success", 2)]
    _, _, _, failed = wait_for_release_ci.evaluate(runs, ("CI",))
    assert failed == []


@pytest.mark.parametrize("workflow", wait_for_release_ci.RELEASE_WORKFLOWS)
def test_expected_workflow_names_exist_in_the_repository(workflow):
    """A renamed workflow would make the poller wait forever on a name
    nobody publishes — so pin the names to the actual `name:` fields."""
    names = set()
    for path in (REPO_ROOT / ".github" / "workflows").glob("*.yml"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("name:"):
                names.add(line.split(":", 1)[1].strip().strip("\"'"))
                break
    assert workflow in names


# ── workspace coherence (what resolution alone cannot see) ─────────────


def test_every_member_inherits_the_workspace_version():
    """A member with its own explicit version still *resolves*, so
    `cargo metadata` cannot detect this class at all."""
    declared = {m.parent.name: bump_version.declared_member_version(m) for m in bump_version.member_manifests()}
    assert set(declared.values()) == {"workspace"}, declared


def test_published_crate_list_matches_the_publish_workflow():
    """If a crate is added to (or dropped from) the crates.io publish set
    without updating PUBLISHED_CRATES, the lockstep assertion silently
    stops covering it."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    published = set(re.findall(r"cargo publish -p ([\w-]+)", workflow))
    assert published == set(bump_version.PUBLISHED_CRATES)
