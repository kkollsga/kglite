"""Regression tests for the ecosystem version-consistency checker.

The checker's whole value is that it fails when a version requirement
disagrees with another declaration site. A checker that silently reports
"clean" is worse than none at all — it converts an unnoticed problem into a
*believed-absent* one. So every category it claims to detect is exercised here
against a synthetic ecosystem, and every one is also shown NOT to fire on the
consistent variant of the same shape.

The fixtures are synthetic rather than the real sibling repos: the real ones
are actively edited, and a test that depends on their contents fails for
reasons that have nothing to do with the checker.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_version_consistency.py"
_spec = importlib.util.spec_from_file_location("check_version_consistency", SCRIPT)
assert _spec and _spec.loader
vc = importlib.util.module_from_spec(_spec)
sys.modules["check_version_consistency"] = vc
_spec.loader.exec_module(vc)


# --------------------------------------------------------------------------
# Requirement algebra
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "version", "admits"),
    [
        # Cargo caret semantics, including the 0.x special case that governs
        # every version in this ecosystem.
        ("0.15", "0.15.3", True),
        ("0.15", "0.16.0", False),
        ("^0.14.4", "0.15.0", False),
        ("^0.14.4", "0.14.9", True),
        ("1.2", "1.9.0", True),
        ("1.2", "2.0.0", False),
        ("5", "5.17.3", True),
        ("5", "6.0.0", False),
        # PEP 440 shapes seen in pyproject/CI.
        (">=0.15.0,<0.16", "0.15.0", True),
        (">=0.15.0,<0.16", "0.16.0", False),
        (">=0.15.0,<0.16", "0.14.5", False),
        (">=0.13", "0.15.0", True),
        ("==0.14.5", "0.14.5", True),
        ("==0.14.5", "0.15.0", False),
        ("<0.14", "0.13.9", True),
        ("<0.14", "0.15.0", False),
        (">=0.3.4, <0.4", "0.3.9", True),
        (">=0.3.4, <0.4", "0.4.0", False),
    ],
)
def test_requirement_admission(spec: str, version: str, admits: bool) -> None:
    interval = vc.parse_requirement(spec)
    assert interval is not None, f"failed to parse {spec!r}"
    assert interval.contains(vc.parse_version(version)) is admits


def test_unparseable_requirements_are_declined_not_guessed() -> None:
    """A shape we do not model must yield None, never a wrong interval."""
    for spec in ["*", "", "git+https://example.invalid/x", "../local/path"]:
        assert vc.parse_requirement(spec) is None


# --------------------------------------------------------------------------
# Synthetic ecosystem
# --------------------------------------------------------------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture()
def ecosystem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A minimal, fully consistent two-repo ecosystem at kglite 0.15.0."""
    monkeypatch.setattr(vc, "DOWNSTREAM_REPOS", ("downstream",))
    monkeypatch.setattr(
        vc,
        "ECOSYSTEM_PACKAGES",
        {
            "KGLite": ("kglite",),
            "downstream": ("downstream",),
        },
    )

    root = tmp_path / "eco"
    _write(root / "KGLite" / "Cargo.toml", '[workspace.package]\nversion = "0.15.0"\n')
    _write(
        root / "downstream" / "Cargo.toml",
        '[workspace.package]\nversion = "1.0.0"\n\n[workspace.dependencies]\nkglite = { version = "0.15.0" }\n',
    )
    _write(
        root / "downstream" / "pyproject.toml",
        '[project]\nversion = "1.0.0"\ndependencies = [\n  "kglite>=0.15.0,<0.16",\n]\n',
    )
    _write(
        root / "downstream" / ".github" / "workflows" / "ci.yml",
        "jobs:\n  test:\n    steps:\n      - run: pip install kglite==0.15.0\n",
    )
    return root


def run(root: Path, *args: str) -> tuple[int, str]:
    """Invoke the checker, returning ``(exit_code, stdout)``."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = vc.main(["--root", str(root), *args])
    return code, buf.getvalue()


def test_consistent_ecosystem_passes(ecosystem: Path) -> None:
    code, out = run(ecosystem)
    assert code == 0, out
    assert "0 contradiction(s)" in out


def test_default_root_uses_current_checkout_as_upstream(
    ecosystem: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A release worktree must not read the stale primary checkout version."""
    worktree = tmp_path / "release-worktree"
    _write(worktree / "Cargo.toml", '[workspace.package]\nversion = "0.15.1"\n')
    monkeypatch.setattr(vc, "REPO_ROOT", worktree)
    monkeypatch.setattr(vc, "ECOSYSTEM_ROOT", ecosystem)

    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        code = vc.main([])

    out = buf.getvalue()
    assert code == 0, out
    assert "upstream: KGLite 0.15.1" in out
    assert "cannot install the current kglite 0.15.1" not in out


# --------------------------------------------------------------------------
# 1. Contradictions — the highest-severity category, and the exit-code contract
# --------------------------------------------------------------------------


def test_contradiction_is_detected_and_exits_non_zero(ecosystem: Path) -> None:
    """The real codingest failure: CI pins a version its metadata excludes.

    Neither site is wrong on its own, and no tool inside the repo compares
    them, which is exactly why this shipped.
    """
    ci = ecosystem / "downstream" / ".github" / "workflows" / "ci.yml"
    ci.write_text(
        "jobs:\n  test:\n    steps:\n      - run: pip install kglite==0.14.5\n",
        encoding="utf-8",
    )

    code, out = run(ecosystem)
    assert code == 1, out
    assert "CONTRADICTIONS" in out
    assert "kglite==0.14.5" in out
    assert "ci.yml" in out


def test_removing_the_contradiction_restores_a_clean_exit(ecosystem: Path) -> None:
    """The other half of the same proof: the failure was caused by the defect."""
    ci = ecosystem / "downstream" / ".github" / "workflows" / "ci.yml"
    ci.write_text(
        "jobs:\n  test:\n    steps:\n      - run: pip install kglite==0.14.5\n",
        encoding="utf-8",
    )
    assert run(ecosystem)[0] == 1

    ci.write_text(
        "jobs:\n  test:\n    steps:\n      - run: pip install kglite==0.15.0\n",
        encoding="utf-8",
    )
    code, out = run(ecosystem)
    assert code == 0, out


def test_separate_resolution_units_do_not_contradict(ecosystem: Path) -> None:
    """A standalone example with its own lockfile resolves independently.

    mcp-methods keeps `examples/downstream_binary` pinned to the last published
    major on purpose. Flagging that as a contradiction with the workspace would
    make the gate unusable.
    """
    ex = ecosystem / "downstream" / "examples" / "old_consumer"
    _write(ex / "Cargo.toml", '[dependencies]\nkglite = { version = "0.14" }\n')
    _write(ex / "Cargo.lock", '[[package]]\nname = "kglite"\nversion = "0.14.5"\n')

    code, out = run(ecosystem)
    assert "CONTRADICTIONS" not in out, out
    assert code == 0, out


# --------------------------------------------------------------------------
# 2. Cross-repo staleness
# --------------------------------------------------------------------------


def test_downstream_excluding_current_upstream_is_reported(ecosystem: Path) -> None:
    code, out = run(ecosystem, "--upstream-version", "0.16.0")
    assert "CROSS-REPO STALENESS" in out
    assert "cannot install the current kglite" in out
    # Staleness alone is a report, not a gate failure, unless asked for.
    assert code == 0, out
    assert run(ecosystem, "--upstream-version", "0.16.0", "--fail-on", "stale")[0] == 1


def test_permissive_floor_is_not_stale(ecosystem: Path) -> None:
    """kglite-datasets' deliberate `>=0.13` must stay a measured null."""
    _write(
        ecosystem / "downstream" / "pyproject.toml", '[project]\nversion = "1.0.0"\ndependencies = ["kglite>=0.13"]\n'
    )
    (ecosystem / "downstream" / "Cargo.toml").write_text('[workspace.package]\nversion = "1.0.0"\n', encoding="utf-8")
    (ecosystem / "downstream" / ".github" / "workflows" / "ci.yml").write_text(
        "jobs:\n  t:\n    steps:\n      - run: pip install kglite>=0.13\n", encoding="utf-8"
    )

    code, out = run(ecosystem, "--upstream-version", "0.16.0")
    assert "CROSS-REPO STALENESS" not in out, out
    assert code == 0


def test_floor_test_pin_is_classified_not_flagged(ecosystem: Path) -> None:
    """An exact pin equal to the declared floor is a CI floor-test leg."""
    _write(
        ecosystem / "downstream" / "pyproject.toml", '[project]\nversion = "1.0.0"\ndependencies = ["kglite>=0.13"]\n'
    )
    (ecosystem / "downstream" / "Cargo.toml").write_text('[workspace.package]\nversion = "1.0.0"\n', encoding="utf-8")
    (ecosystem / "downstream" / ".github" / "workflows" / "ci.yml").write_text(
        "jobs:\n  t:\n    strategy:\n      matrix:\n        spec: ['kglite>=0.13', 'kglite==0.13.0']\n",
        encoding="utf-8",
    )

    code, out = run(ecosystem)
    assert "FLOOR-TEST" in out
    assert "CROSS-REPO STALENESS" not in out, out
    assert code == 0


# --------------------------------------------------------------------------
# 3. Understated floors — the invisible-locally class
# --------------------------------------------------------------------------


def test_manifest_floor_below_the_governing_lock_is_reported(ecosystem: Path) -> None:
    """The mcp-methods 0.4-vs-0.4.1 shape, reproduced exactly."""
    _write(
        ecosystem / "downstream" / "Cargo.toml",
        '[workspace.package]\nversion = "1.0.0"\n\n[workspace.dependencies]\nkglite = { version = "0.15" }\n',
    )
    _write(ecosystem / "downstream" / "Cargo.lock", '[[package]]\nname = "kglite"\nversion = "0.15.4"\n')

    code, out = run(ecosystem)
    assert "UNDERSTATED FLOORS" in out
    assert "floor 0.15.0" in out
    assert "resolved 0.15.4" in out
    # A warning, not a gate failure: floors legitimately float.
    assert code == 0, out


def test_known_feature_floor_beats_the_lock_heuristic(ecosystem: Path) -> None:
    """`fastembed = "5"` selects a feature that first exists in 5.9.0."""
    _write(
        ecosystem / "downstream" / "Cargo.toml",
        '[workspace.package]\nversion = "1.0.0"\n\n[workspace.dependencies]\nfastembed = { version = "5" }\n',
    )
    _write(ecosystem / "downstream" / "Cargo.lock", '[[package]]\nname = "fastembed"\nversion = "5.17.3"\n')

    _, out = run(ecosystem)
    assert "first exists in 5.9.0" in out
    assert out.count("fastembed") >= 1
    # Exactly one finding, not one per detection route.
    assert out.count("UNDERSTATED FLOORS — declared minimum below what actually builds (1)") == 1


# --------------------------------------------------------------------------
# 4. Declaration sites outside package metadata
# --------------------------------------------------------------------------


def test_non_metadata_sites_are_inventoried_even_when_consistent(ecosystem: Path) -> None:
    _, out = run(ecosystem, "--show-sites")
    assert "OUTSIDE PACKAGE METADATA" in out
    assert "ci-yaml" in out


def test_install_hint_inside_a_source_string_is_a_site(ecosystem: Path) -> None:
    """sonagram tells users `pip install kglite>=0.15` from a Rust error path."""
    _write(
        ecosystem / "downstream" / "src" / "lib.rs", 'fn e() { panic!("install it: `pip install kglite>=0.15`"); }\n'
    )
    _, out = run(ecosystem, "--show-sites")
    assert "source-string" in out
    assert "lib.rs" in out


def test_advisory_sites_cannot_cause_a_contradiction(ecosystem: Path) -> None:
    """A version literal in prose-bearing code must not fail the gate.

    The checker's own docstring cites `kglite==0.14.5` as an example; when
    scripts counted as binding, running it in its own repo reported the script
    as a contradiction against the manifests.
    """
    _write(
        ecosystem / "downstream" / "scripts" / "helper.py",
        '"""Historical note: we used to pin kglite==0.9.0 here."""\n',
    )
    code, out = run(ecosystem)
    assert code == 0, out
    assert "CONTRADICTIONS" not in out


# --------------------------------------------------------------------------
# 5. Documented versions
# --------------------------------------------------------------------------


def test_stale_documented_version_is_reported(ecosystem: Path) -> None:
    _write(ecosystem / "downstream" / "README.md", "# downstream\n\nA thin KGLite 0.14.3 frontend.\n")
    code, out = run(ecosystem)
    assert "STALE DOCUMENTED VERSIONS" in out
    assert "0.14.3" in out
    assert code == 0, out


def test_history_bearing_documents_are_not_flagged(ecosystem: Path) -> None:
    """A changelog, a migration guide and a dated ledger row say old versions
    forever; flagging them would be wrong, not merely noisy."""
    d = ecosystem / "downstream"
    _write(d / "CHANGELOG.md", "## [0.9]\n\n- Moved kglite 0.13.0 to 0.14.0.\n")
    _write(d / "docs" / "migrating-to-2.md", "Convert with kglite 0.13.4 first.\n")
    _write(d / "GATE.md", "| 2026-07-17 | abc123 | kglite 0.13.0 sync (intended) |\n")

    code, out = run(ecosystem)
    assert "STALE DOCUMENTED VERSIONS" not in out, out
    assert code == 0


def test_pin_back_instruction_is_not_stale(ecosystem: Path) -> None:
    """`pip install "kglite<0.14"` is an upper bound, not a claim about now."""
    _write(
        ecosystem / "downstream" / "README.md", '# downstream\n\nPin back anytime with `pip install "kglite<0.14"`.\n'
    )
    code, out = run(ecosystem)
    assert "STALE DOCUMENTED VERSIONS" not in out, out
    assert code == 0


def test_inline_suppression_marker_silences_a_line(ecosystem: Path) -> None:
    _write(
        ecosystem / "downstream" / "README.md",
        "# downstream\n\nA thin KGLite 0.14.3 frontend. <!-- version-check: ignore -->\n",
    )
    _, out = run(ecosystem)
    assert "STALE DOCUMENTED VERSIONS" not in out, out


# --------------------------------------------------------------------------
# 6. The notifier's affected-only rule
# --------------------------------------------------------------------------


def test_blocked_downstream_is_notified(ecosystem: Path) -> None:
    code, out = run(ecosystem, "--upstream-version", "0.16.0", "--notify", "--dry-run")
    assert code == 0
    assert "NOTIFY downstream" in out
    assert "BLOCKED" in out
    assert "your declared range excludes it" in out
    # Dry run writes nothing.
    assert not (ecosystem / "downstream" / "inbox").exists()


def test_unaffected_downstream_is_not_notified(ecosystem: Path) -> None:
    """The core doctrine: a 'nothing changes for you' note is inbox noise."""
    code, out = run(ecosystem, "--notify", "--dry-run")
    assert code == 0
    assert "NOTIFY" not in out
    assert "SKIP   downstream" in out
    assert "0 downstream(s) affected, 1 deliberately not notified." in out


def test_non_consumer_skip_states_the_real_reason(ecosystem: Path) -> None:
    (ecosystem / "downstream" / "Cargo.toml").write_text('[workspace.package]\nversion = "1.0.0"\n', encoding="utf-8")
    (ecosystem / "downstream" / "pyproject.toml").write_text(
        '[project]\nversion = "1.0.0"\ndependencies = []\n', encoding="utf-8"
    )
    (ecosystem / "downstream" / ".github" / "workflows" / "ci.yml").write_text(
        "jobs:\n  t:\n    steps:\n      - run: echo hi\n", encoding="utf-8"
    )

    _, out = run(ecosystem, "--notify", "--dry-run")
    assert "not a consumer" in out


def test_breaking_symbol_use_triggers_a_notification(ecosystem: Path) -> None:
    """A repo that references a broken symbol is affected even when its range
    admits the new version."""
    _write(
        ecosystem / "downstream" / "src" / "main.rs",
        "use kglite::api::durable::Wal;\nfn main() { let _ = Wal::open(); }\n",
    )

    code, out = run(ecosystem, "--notify", "--dry-run", "--breaking-symbol", "kglite::api::durable::Wal")
    assert code == 0
    assert "NOTIFY downstream" in out
    assert "touched by a breaking change" in out
    assert "main.rs" in out


def test_notify_writes_only_to_affected_inboxes(ecosystem: Path) -> None:
    code, out = run(ecosystem, "--upstream-version", "0.16.0", "--notify")
    assert code == 0
    unread = ecosystem / "downstream" / "inbox" / "unread"
    notes = list(unread.glob("*.md"))
    assert len(notes) == 1
    body = notes[0].read_text(encoding="utf-8")
    # Matches the established inbox schema.
    assert body.startswith("# ")
    assert "- **From:** kglite" in body
    assert "- **To:** downstream" in body
    assert "## Ask / action requested" in body
    assert "## References" in body
    # The recipient owns the Status footer; the sender never writes one.
    assert "## Status" not in body
    assert notes[0].name.startswith("2026-") or notes[0].name[:4].isdigit()
    assert "-from-kglite-" in notes[0].name


def test_notify_is_idempotent_for_a_given_release(ecosystem: Path) -> None:
    """Re-running a release's notification must not pile up duplicates."""
    run(ecosystem, "--upstream-version", "0.16.0", "--notify", "--date", "2026-07-27")
    run(ecosystem, "--upstream-version", "0.16.0", "--notify", "--date", "2026-07-27")
    notes = list((ecosystem / "downstream" / "inbox" / "unread").glob("*.md"))
    assert len(notes) == 1


# --------------------------------------------------------------------------
# 7. Degradation when siblings are absent
# --------------------------------------------------------------------------


def test_missing_siblings_skip_with_a_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vc, "DOWNSTREAM_REPOS", ("nowhere",))
    monkeypatch.setattr(vc, "ECOSYSTEM_PACKAGES", {"KGLite": ("kglite",), "nowhere": ("nowhere",)})
    root = tmp_path / "eco"
    _write(root / "KGLite" / "Cargo.toml", '[workspace.package]\nversion = "0.15.0"\n')

    code, out = run(root)
    assert code == 0, out
    assert "not present" in out
    assert "nowhere" in out


def test_require_siblings_turns_absence_into_a_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vc, "DOWNSTREAM_REPOS", ("nowhere",))
    monkeypatch.setattr(vc, "ECOSYSTEM_PACKAGES", {"KGLite": ("kglite",), "nowhere": ("nowhere",)})
    root = tmp_path / "eco"
    _write(root / "KGLite" / "Cargo.toml", '[workspace.package]\nversion = "0.15.0"\n')

    code, out = run(root, "--require-siblings")
    assert code == 1
    assert "cannot be trusted" in out
