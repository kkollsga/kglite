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
semver_check = _load("semver_check")


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


def test_the_java_readme_install_snippets_are_found():
    """Guards the discovery of the install snippets `check()` holds at the
    workspace version: both coordinates, each found."""
    found = bump_version.doc_snippet_versions()
    assert [rel.as_posix() for rel, _ in found] == ["kglite-java/README.md"] * 2
    assert all(version for _, version in found), found


def test_a_stale_install_snippet_fails_the_check_and_the_bump_moves_it(tmp_path, monkeypatch):
    readme = tmp_path / "kglite-java" / "README.md"
    readme.parent.mkdir(parents=True)
    readme.write_text(
        "<artifactId>kglite</artifactId>\n  <version>0.1.0</version>\n"
        'implementation("io.github.kkollsga:kglite:0.1.0")\n',
        encoding="utf-8",
    )
    workspace = bump_version.read_workspace_version()
    monkeypatch.setattr(bump_version, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bump_version, "ROOT_MANIFEST", tmp_path / "Cargo.toml")
    monkeypatch.setattr(bump_version, "member_manifests", lambda: [])
    (tmp_path / "Cargo.toml").write_text(f'[workspace.package]\nversion = "{workspace}"\n', encoding="utf-8")
    assert bump_version.doc_snippet_versions() == [(Path("kglite-java/README.md"), "0.1.0")] * 2
    problems = bump_version.check()
    assert len(problems) == 2 and all("install snippet names '0.1.0'" in p for p in problems), problems
    assert bump_version.bump(workspace) == [readme]
    assert bump_version.check() == []
    readme.write_text("no snippet here\n", encoding="utf-8")
    assert any("no install snippet found" in p for p in bump_version.check())


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


def _gh_run(name, number, event, branch, status, conclusion, sha="abc123"):
    return {
        "id": number,
        "name": name,
        "run_number": number,
        "event": event,
        "head_branch": branch,
        "head_sha": sha,
        "status": status,
        "conclusion": conclusion,
    }


def _canned_api(monkeypatch, runs):
    """Both queries answer with every run, as the head_sha query does."""
    monkeypatch.setattr(wait_for_release_ci, "gh_api", lambda path: {"workflow_runs": runs})


def test_a_cancelled_pull_request_run_on_the_same_sha_is_not_a_failure(monkeypatch):
    """CI #1582 (pull_request, cancelled) shares a SHA with CI #1581
    (push/main, success); only the push run may decide the release."""
    runs = [
        _gh_run("CI", 1581, "push", "main", "completed", "success"),
        _gh_run("CI", 1582, "pull_request", "fix/0.21.1-review", "completed", "cancelled"),
        _gh_run("Release", 40, "push", "main", "completed", "success"),
    ]
    _canned_api(monkeypatch, runs)
    found, _ = wait_for_release_ci.fetch_runs("o/r", "main", "abc123")
    all_terminal, missing, pending, failed = wait_for_release_ci.evaluate(found, ("CI", "Release"))
    assert (all_terminal, missing, pending, failed) == (True, [], [], [])


def test_a_cancelled_push_run_on_main_still_fails(monkeypatch):
    runs = [
        _gh_run("CI", 1581, "push", "main", "completed", "cancelled"),
        _gh_run("CI", 1582, "pull_request", "x", "completed", "success"),
        _gh_run("Release", 40, "push", "main", "completed", "success"),
    ]
    _canned_api(monkeypatch, runs)
    found, _ = wait_for_release_ci.fetch_runs("o/r", "main", "abc123")
    _, _, _, failed = wait_for_release_ci.evaluate(found, ("CI", "Release"))
    assert failed == ["CI: cancelled"]


def test_only_pull_request_runs_present_is_missing_not_a_pass(monkeypatch):
    _canned_api(monkeypatch, [_gh_run("CI", 1582, "pull_request", "x", "completed", "success")])
    found, _ = wait_for_release_ci.fetch_runs("o/r", "main", "abc123")
    all_terminal, missing, _, _ = wait_for_release_ci.evaluate(found, ("CI",))
    assert (all_terminal, missing) == (False, ["CI"])


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


release_preflight = _load("release_preflight")


def _member(tmp_path: Path, name: str, version_line: str) -> Path:
    manifest = tmp_path / name / "Cargo.toml"
    manifest.parent.mkdir()
    manifest.write_text(f'[package]\nname = "{name}"\n{version_line}\n\n[dependencies]\n', encoding="utf-8")
    return manifest


@pytest.mark.parametrize(
    ("version_line", "ok"),
    [
        ("version.workspace = true", True),
        ("version = { workspace = true }", True),
        # The workspace's own version, hard-coded: it resolves today and goes
        # stale at the next bump, so preflight must refuse it now.
        ('version = "0.19.0"', False),
        ('version = "0.18.1"', False),
        ('edition = "2021"', False),
    ],
)
def test_preflight_member_inheritance_accepts_only_workspace_inheritance(tmp_path, monkeypatch, version_line, ok):
    manifest = _member(tmp_path, "kglite-fixture", version_line)
    monkeypatch.setattr(release_preflight, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(release_preflight.bump_version, "member_manifests", lambda: [manifest])
    result = release_preflight.check_member_inheritance("0.19.0")
    assert result.ok is ok, result.detail


# ── preflight: binary-size baselines describe the right releases ───────


def _pypi_index_with(monkeypatch, *published: str) -> None:
    """Stub PyPI so `published` are the versions with a manylinux x86_64 wheel."""
    import json

    refresh = sys.modules.get("refresh_release_constants") or _load("refresh_release_constants")
    index = {
        "releases": {
            v: [{"filename": f"kglite-{v}-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"}]
            for v in published
        }
    }
    monkeypatch.setattr(refresh, "_http_get", lambda url: json.dumps(index).encode())


def _size_fixture(tmp_path, monkeypatch, darwin: str, linux: str):
    _pypi_index_with(monkeypatch, "0.1.8", "0.1.9")
    baselines = tmp_path / "baselines"
    baselines.mkdir(parents=True)
    for name in ("0_2_0.json", "0_2_0.linux.json"):
        (baselines / name).write_text("{}", encoding="utf-8")
    phase5 = tmp_path / "test_phase5_parity.py"
    phase5.write_text(
        "BINARY_SIZE_BASELINES = {\n"
        f'    "darwin": (10_000, "{darwin}"),  # host release build\n'
        f'    "linux": (20_000, "{linux}"),  # published manylinux2014 x86_64 wheel member\n'
        "}\n"
        "    Baseline history:\n"
        "      - 0.2.0:  10,000 bytes.\n",
        encoding="utf-8",
    )
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("## [Unreleased]\n\n## [0.2.0] - x\n\n## [0.1.9] - x\n\n## [0.1.8] - x\n", encoding="utf-8")
    monkeypatch.setattr(release_preflight, "BASELINES_DIR", baselines)
    monkeypatch.setattr(release_preflight, "PHASE5_TEST", phase5)
    monkeypatch.setattr(release_preflight, "CHANGELOG", changelog)


def test_preflight_accepts_size_baselines_for_the_cut_and_the_previous_release(tmp_path, monkeypatch):
    _size_fixture(tmp_path, monkeypatch, darwin="0.2.0", linux="0.1.9")
    result = release_preflight.check_captured_constants("0.2.0")
    assert result.ok, result.detail


@pytest.mark.parametrize(
    ("darwin", "linux", "stale"),
    [
        # The 0.19.1 failure: the Linux row stayed releases behind.
        ("0.2.0", "0.1.8", "linux"),
        ("0.1.9", "0.1.9", "darwin"),
    ],
)
def test_preflight_refuses_a_stale_size_baseline(tmp_path, monkeypatch, darwin, linux, stale):
    _size_fixture(tmp_path, monkeypatch, darwin=darwin, linux=linux)
    result = release_preflight.check_captured_constants("0.2.0")
    assert not result.ok
    assert stale in result.detail


def test_preflight_linux_row_follows_pypi_not_the_changelog(tmp_path, monkeypatch):
    """0.1.9 has a CHANGELOG section but never reached PyPI (the 0.19.2
    crates-held-back shape, inverted). The refresh measures the newest wheel
    PyPI has, 0.1.8, so preflight must expect exactly that row."""
    _size_fixture(tmp_path, monkeypatch, darwin="0.2.0", linux="0.1.8")
    _pypi_index_with(monkeypatch, "0.1.8")
    result = release_preflight.check_captured_constants("0.2.0")
    assert result.ok, result.detail

    _size_fixture(tmp_path / "b", monkeypatch, darwin="0.2.0", linux="0.1.9")
    _pypi_index_with(monkeypatch, "0.1.8")
    result = release_preflight.check_captured_constants("0.2.0")
    assert not result.ok
    assert "linux binary-size baseline describes 0.1.9, expected 0.1.8" in result.detail


def test_preflight_reports_an_unreachable_pypi_instead_of_passing(tmp_path, monkeypatch):
    _size_fixture(tmp_path, monkeypatch, darwin="0.2.0", linux="0.1.9")
    refresh = sys.modules["refresh_release_constants"]

    def offline(url):
        raise OSError("network down")

    monkeypatch.setattr(refresh, "_http_get", offline)
    result = release_preflight.check_captured_constants("0.2.0")
    assert not result.ok
    assert "cannot tell which Linux wheel" in result.detail


# ── `make semver-check` distinguishes "found something" from "did not run" ──

FINDING_OUTPUT = """\
     Checking kglite v0.21.0 -> v0.21.1 (assume minor change)
--- failure constructible_struct_adds_field: pub struct field added ---
     Summary semver requires new major version: 1 major and 0 minor checks failed
"""


def _run_main(monkeypatch, capsys, *, baseline="0.21.0", tool=(0, ""), lookup_error=None):
    def fetch():
        if lookup_error:
            raise semver_check.SemverCheckError(lookup_error)
        return baseline

    monkeypatch.setattr(semver_check, "fetch_baseline", fetch)
    monkeypatch.setattr(semver_check, "run_tool", lambda b: tool)
    code = semver_check.main()
    return code, capsys.readouterr()


def test_semver_findings_pass_through_with_exit_zero(monkeypatch, capsys):
    code, out = _run_main(monkeypatch, capsys, tool=(1, FINDING_OUTPUT))
    assert code == 0
    assert "constructible_struct_adds_field" in out.out
    assert "DID NOT RUN" not in out.err


def test_semver_clean_summary_exits_zero(monkeypatch, capsys):
    code, _ = _run_main(monkeypatch, capsys, tool=(0, "     Summary no semver update required\n"))
    assert code == 0


def test_semver_checks_every_feature_but_fastembed():
    # fastembed in the checked set made the published-baseline build fail on a
    # fresh fastembed 7.8.0 resolve, so the check reported DID NOT RUN.
    features = semver_check.checked_features()
    assert "fastembed" not in features and "test-seam" not in features
    assert {"iana-zones", "rdf", "okf", "xlsx"} <= set(features)


def test_semver_tool_error_without_summary_did_not_run(monkeypatch, capsys):
    code, out = _run_main(monkeypatch, capsys, tool=(101, "error: failed to build rustdoc\n"))
    assert code != 0
    assert "semver-check DID NOT RUN" in out.err


def test_semver_exit_zero_without_summary_did_not_run(monkeypatch, capsys):
    code, out = _run_main(monkeypatch, capsys, tool=(0, ""))
    assert code != 0
    assert "semver-check DID NOT RUN" in out.err


def test_semver_baseline_lookup_failure_did_not_run(monkeypatch, capsys):
    code, out = _run_main(monkeypatch, capsys, lookup_error="baseline version lookup failed (offline)")
    assert code != 0
    assert "semver-check DID NOT RUN" in out.err


@pytest.mark.parametrize(
    "body",
    [
        "",
        "not json",
        "{}",
        '{"crate": {"max_stable_version": ""}}',
        '{"crate": {"max_stable_version": null}}',
        '{"crate": {"max_stable_version": "v1"}}',
    ],
)
def test_semver_malformed_baseline_is_rejected(body):
    with pytest.raises(semver_check.SemverCheckError):
        semver_check.parse_baseline(body)


def test_semver_good_baseline_parses():
    assert semver_check.parse_baseline('{"crate": {"max_stable_version": "0.21.0"}}') == "0.21.0"
