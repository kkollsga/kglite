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
        # 0.0.z: Cargo holds the *patch* fixed, so 0.0.4 is a breaking bump.
        ("^0.0.3", "0.0.3", True),
        ("^0.0.3", "0.0.4", False),
        ("0.0.3", "0.1.0", False),
        # A wildcard/partial-exact fixes the components it names and varies
        # the rest — never the caret's left-most-non-zero rule.
        ("==0.0.*", "0.0.9", True),
        ("==0.0.*", "0.1.0", False),
        ("==1.2.*", "1.2.9", True),
        ("==1.2.*", "1.3.0", False),
        ("=1.2", "1.2.9", True),
        ("=1.2", "1.3.0", False),
    ],
)
def test_requirement_admission(spec: str, version: str, admits: bool) -> None:
    interval = vc.parse_requirement(spec)
    assert interval is not None, f"failed to parse {spec!r}"
    assert interval.contains(vc.parse_version(version)) is admits


@pytest.mark.parametrize(
    ("spec", "hi"),
    [
        ("^1.2.3", (2, 0, 0)),
        ("^0.14.4", (0, 15, 0)),
        ("^0.0.3", (0, 0, 4)),
        ("^0.14", (0, 15, 0)),
        ("^0.0", (0, 1, 0)),
        ("^0", (1, 0, 0)),
        ("^5", (6, 0, 0)),
    ],
)
def test_caret_upper_bounds(spec: str, hi: tuple[int, int, int]) -> None:
    """Caret bounds are Cargo's, exactly — an over-wide `hi` suppresses the
    staleness findings the checker exists to report."""
    interval = vc.parse_requirement(spec)
    assert interval is not None, f"failed to parse {spec!r}"
    assert interval.hi == hi


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


#: The upstream changelog every note's "what changed" section must be read from.
#: Three distinct releases, each with a lead sentence that appears nowhere else,
#: so a note quoting the wrong one is detectable by string alone. Bold leads wrap
#: across lines exactly as the real file's do — that wrapping is the reason a
#: naive first-line read produces a truncated quote.
CHANGELOG = """# Changelog

## [Unreleased]

## [0.17.0] - 2026-08-12

### Fixed

- **A join filtered on a type's title field is no longer planned as if the
  filter matched everything.** The planner estimated a non-indexed equality
  filter as `type_count / distinct_values` and the scan read the property map
  only.

- **`.statistics()` on a type's own title field returned nothing.** Same root
  cause, same fix.

## [0.16.0] - 2026-08-01

### Changed

- **A lost Bolt write conflict is now retriable.** The server returned a bare
  failure where the driver expected a transient error.

## [0.15.0] - 2026-07-27

### Added

- **Durability rungs land behind `save(durability=...)`.** Three rungs, one
  knob.
"""


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
    _write(root / "KGLite" / "CHANGELOG.md", CHANGELOG)
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


def test_stale_maven_xml_dependency_is_reported(ecosystem: Path) -> None:
    """A Maven `<dependency>` block splits the coordinate over three lines, so
    the `:`-separated coordinate regex cannot see it — yet it is the form a
    Java consumer copies into a pom.xml. Left unwatched, half of an install
    section is gated and the other half rots."""
    _write(
        ecosystem / "downstream" / "README.md",
        "# downstream\n\n```xml\n<dependency>\n"
        "  <groupId>io.github.kkollsga</groupId>\n"
        "  <artifactId>kglite</artifactId>\n"
        "  <version>0.14.3</version>\n"
        "</dependency>\n```\n",
    )
    code, out = run(ecosystem)
    assert "STALE DOCUMENTED VERSIONS" in out, out
    assert "0.14.3" in out
    assert code == 0, out


def test_matching_maven_xml_dependency_is_not_flagged(ecosystem: Path) -> None:
    """The same block at the current version is the state a release leaves
    behind, and must stay silent — otherwise the check is noise the moment it
    is correct."""
    _write(
        ecosystem / "downstream" / "README.md",
        "# downstream\n\n```xml\n<dependency>\n"
        "  <groupId>io.github.kkollsga</groupId>\n"
        "  <artifactId>kglite</artifactId>\n"
        "  <version>0.15.0</version>\n"
        "</dependency>\n```\n",
    )
    code, out = run(ecosystem)
    assert "STALE DOCUMENTED VERSIONS" not in out, out
    assert code == 0, out


def test_maven_xml_version_of_an_untracked_artifact_is_ignored(ecosystem: Path) -> None:
    """A pom snippet documenting some other dependency is not our problem, and
    a `<version>` must never be attributed to whatever artifactId came last in
    a different block."""
    _write(
        ecosystem / "downstream" / "README.md",
        "# downstream\n\n```xml\n<dependency>\n"
        "  <artifactId>junit-jupiter</artifactId>\n"
        "  <version>5.11.4</version>\n"
        "</dependency>\n```\n",
    )
    code, out = run(ecosystem)
    assert "5.11.4" not in out, out
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
# 7. "What changed in this release" comes from the named release's CHANGELOG
#
# The defect these cover shipped three times: 0.15.11, 0.15.12 and 0.15.13 all
# carried a byte-identical blurb describing 0.15.9's Rust-API break, because the
# section was rendered from a hand-maintained constant that nothing forced to
# move. A downstream reading it at face value plans a migration that does not
# exist in the release it names — the failure manufactures work.
# --------------------------------------------------------------------------


def test_what_changed_quotes_the_named_releases_changelog_entry(ecosystem: Path) -> None:
    code, out = run(ecosystem, "--upstream-version", "0.17.0", "--notify", "--dry-run")
    assert code == 0, out
    assert "What changed in this release" in out
    assert "A join filtered on a type's title field is no longer planned" in out
    assert "`.statistics()` on a type's own title field returned nothing." in out
    # The neighbouring releases' entries must not leak in.
    assert "retriable" not in out, out
    assert "Durability rungs" not in out, out


def test_a_different_release_gets_its_own_entry(ecosystem: Path) -> None:
    """The other direction of the same proof: the quote tracks the version."""
    code, out = run(ecosystem, "--upstream-version", "0.16.0", "--notify", "--dry-run")
    assert code == 0, out
    assert "A lost Bolt write conflict is now retriable." in out
    assert "A join filtered on a type's title field" not in out, out


def test_an_unresolvable_version_emits_no_what_changed_section(ecosystem: Path) -> None:
    """The report's explicit ask: an absent blurb beats a silently wrong one."""
    code, out = run(ecosystem, "--upstream-version", "0.18.0", "--notify", "--dry-run")
    assert code == 0, out
    assert "NOTIFY downstream" in out
    assert "What changed in this release" not in out, out
    assert "warning:" in out and "0.18.0" in out


def test_a_note_never_carries_a_previous_releases_breaking_prose(ecosystem: Path) -> None:
    """The staleness trap itself, as a contract.

    Nothing in a note announcing 0.17.0 may quote 0.15.9's API-break prose —
    the exact text three shipped notes carried. This fails on any design that
    keeps a hand-edited highlight list, whatever its current contents.
    """
    _, out = run(ecosystem, "--upstream-version", "0.17.0", "--notify", "--dry-run")
    for stale in (
        "NodeData property readers removed",
        "column_stores became accessors",
        "1,789x",
        "Saved-graph writes no longer sweep",
    ):
        assert stale not in out, f"note announcing 0.17.0 quoted 0.15.9 prose: {stale!r}"


def test_breaking_symbols_are_scoped_to_the_release_that_broke_them(ecosystem: Path) -> None:
    """Criterion 4 must not fire with a previous release's symbol set.

    A repo referencing `NodeView` is affected by 0.15.9 and by no later release
    that did not touch it; asserting otherwise is the same manufactured work as
    the stale blurb, wearing the evidence section's clothes.
    """
    assert "NodeView" in vc.breaking_symbols_for("0.15.9")
    assert vc.breaking_symbols_for("0.17.0") == []

    _write(
        ecosystem / "downstream" / "src" / "lib.rs",
        "fn f(node: &kglite::api::NodeView<'_>) -> bool { node.title().is_some() }\n",
    )
    _, out = run(ecosystem, "--upstream-version", "0.17.0", "--notify", "--dry-run")
    assert "touched by a breaking change" not in out, out


# --------------------------------------------------------------------------
# 8. Declaration vs citation — a version *pin* must move, a version *record*
#    must not. Following the scan's advice on a provenance line falsifies it.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prose",
    [
        # codingest/PARITY.md — a completed, verified migration record.
        "The kglite 0.14.1 → 0.14.5 engine move and the NodeView/`set_node_property`\n"
        "API migration left every digest byte-identical — verified, not assumed.",
        # codingest/docs/mcp.md:63 — when a behaviour was introduced.
        "Containment is `sandbox_root`, it is **opt-in**, and it was\n"
        "introduced in kglite 0.14.5: with it, a swap outside the boundary is\n"
        "refused and the active root does not move.",
        # codingest/docs/mcp.md:73 — the same fact, phrased as a floor in prose.
        # The cue ("onward") wraps onto the *next* line, which is why the
        # classifier reads the whole sentence rather than the matched line.
        "The boundary is real only from kglite 0.14.5\nonward, and only when you set `sandbox_root`.",
        # A dated correction block.
        "> **Correction (2026-07-31).** Earlier revisions of this page described\n"
        "> the kglite 0.14.5 behaviour incorrectly.",
        "We migrated at kglite 0.14.5 and the digests were unchanged.",
        # A range wins over the word "floor": nothing can move a two-ended
        # transition to the current version without rewriting which move
        # actually happened. codingest/PARITY.md:13 is exactly this shape.
        "The engine floor moved kglite 0.14.1 → 0.14.5, which the corpus verified.",
        # The possessive: `<version>'s <noun>` attributes a thing to one named
        # release, which is a statement about that release and true forever.
        # codingest reported this as the third instance of the defect
        # (2026-08-14): their `tests/benchmarks/README.md:128` reads "kglite
        # 0.15.13's planner fix made that exact query 5.7× faster", and two
        # consecutive release notes told them to renumber it — i.e. to falsify a
        # measurement record. Version pinned to the fixture's stale value so the
        # case is non-vacuous (a version *newer* than the ecosystem's is not a
        # staleness finding at all).
        "kglite 0.14.5's planner fix made that exact query 5.7× faster.",
        "kglite 0.14.5's release notes name the same three files.",
    ],
)
def test_historical_citations_are_not_flagged_as_drift(ecosystem: Path, prose: str) -> None:
    _write(ecosystem / "downstream" / "PARITY.md", f"# Parity\n\n{prose}\n")
    code, out = run(ecosystem)
    assert "STALE DOCUMENTED VERSIONS" not in out, out
    assert code == 0, out


@pytest.mark.parametrize(
    "prose",
    [
        "A thin KGLite 0.14.3 frontend.",
        "Requires kglite 0.14.3 or newer.",
        "The engine floor is kglite 0.14.3.",
        "Install it with `pip install kglite==0.14.3`.",  # version-check: ignore
        "Set the pin to kglite 0.14.3 before building.",
        # `X+` is a floor written compactly, so it stays a declaration even
        # inside a past-tense sentence that would otherwise read as a record.
        "It was gated on kglite 0.14.3+ here.",
        # The possessive rule must not swallow a pin: a declaration cue in the
        # same sentence still wins, so `<version>'s <noun>` cannot be used to
        # smuggle a stale requirement past the scan.
        "This requires kglite 0.14.3's schema loader.",
    ],
)
def test_declarations_are_still_flagged(ecosystem: Path, prose: str) -> None:
    """Non-vacuity in the other direction: the classifier must not exempt a pin.

    A missed stale pin is worse than a false positive, so anything that is not
    recognisably a record stays a finding.
    """
    _write(ecosystem / "downstream" / "README.md", f"# downstream\n\n{prose}\n")
    _, out = run(ecosystem)
    assert "STALE DOCUMENTED VERSIONS" in out, out
    assert "0.14.3" in out


def test_a_citation_is_classified_out_loud_not_dropped_silently(ecosystem: Path) -> None:
    """A silent exemption is how a scanner goes quietly blind; it gets a class."""
    _write(
        ecosystem / "downstream" / "PARITY.md",
        "# Parity\n\nThe kglite 0.14.1 → 0.14.5 engine move left every digest byte-identical.\n",
    )
    _, out = run(ecosystem)
    assert "HISTORICAL CITATIONS" in out, out
    assert "PARITY.md" in out


def test_a_citation_alone_does_not_earn_a_note(ecosystem: Path) -> None:
    """The delivered defect: three notes told codingest to rewrite its history."""
    _write(
        ecosystem / "downstream" / "PARITY.md",
        "# Parity\n\nThe kglite 0.14.1 → 0.14.5 engine move left every digest byte-identical.\n",
    )
    _, out = run(ecosystem, "--notify", "--dry-run")
    assert "NOTIFY" not in out, out
    assert "SKIP   downstream" in out


# --------------------------------------------------------------------------
# 9. "Surface you actually reference" must cite code, not prose
# --------------------------------------------------------------------------


def test_symbol_evidence_cites_the_code_site_not_a_comment(ecosystem: Path) -> None:
    """codingest's shape exactly: a `Cargo.toml` comment naming the migration,
    and the real use in a `.rs` file the scan never reached."""
    _write(
        ecosystem / "downstream" / "Cargo.toml",
        '[workspace.package]\nversion = "1.0.0"\n\n'
        "# The `NodeData` -> `NodeView` API break from 0.15.9, migrated here.\n"
        '[workspace.dependencies]\nkglite = { version = "0.15.0" }\n',
    )
    _write(
        ecosystem / "downstream" / "src" / "rev.rs",
        "/// Reads the node title.\n"
        "fn title_is(node: &kglite::api::NodeView<'_>, name: &str) -> bool {\n"
        "    node.title().as_ref() == name\n}\n",
    )

    code, out = run(ecosystem, "--notify", "--dry-run", "--breaking-symbol", "NodeView")
    assert code == 0, out
    assert "Breaking-change surface you actually reference" in out
    assert "`NodeView` — in `src/rev.rs:2`" in out, out
    assert "in `Cargo.toml`" not in out, out
    # There are no declaration sites in this note, so it must not ask for any
    # to be moved — the ask has to match the evidence it is based on.
    assert "Move the sites above" not in out, out


def test_a_symbol_only_ever_in_comments_is_not_a_reference(ecosystem: Path) -> None:
    _write(
        ecosystem / "downstream" / "Cargo.toml",
        '[workspace.package]\nversion = "1.0.0"\n\n'
        "# The `NodeData` -> `NodeView` API break from 0.15.9, migrated here.\n"
        '[workspace.dependencies]\nkglite = { version = "0.15.0" }\n',
    )
    _write(
        ecosystem / "downstream" / "src" / "rev.rs",
        "// We used to call NodeView here.\n/* NodeView again */\nfn f() {}\n",
    )
    _write(
        ecosystem / "downstream" / "notes.py",
        '"""Historical note: NodeView replaced NodeData."""\n\n\ndef f() -> None:\n    pass\n',
    )

    _, out = run(ecosystem, "--notify", "--dry-run", "--breaking-symbol", "NodeView")
    assert "touched by a breaking change" not in out, out
    assert "SKIP   downstream" in out, out


# --------------------------------------------------------------------------
# 10. Degradation when siblings are absent
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


# --------------------------------------------------------------------------
# 10. Self-gated downstreams (sonagram feedback, 2026-08-26)
# --------------------------------------------------------------------------


def _docs_only_drift(ecosystem: Path) -> None:
    """Metadata admits 0.15.1 everywhere; only prose states the old version."""
    (ecosystem / "downstream" / ".github" / "workflows" / "ci.yml").write_text(
        "jobs:\n  t:\n    steps:\n      - run: pip install downstream\n", encoding="utf-8"
    )
    _write(
        ecosystem / "downstream" / "README.md",
        "# downstream\n\nBuilt on kglite 0.15.0.\n",
    )


def test_docs_only_note_never_claims_the_recipients_ci_is_blind(ecosystem: Path) -> None:
    """The note's justification asserted 'no CI job will ever notice' — a fact
    about the recipient's CI the script has no way to know (sonagram gates
    exactly this in cargo test). The claim stays scoped to what we do know."""
    _docs_only_drift(ecosystem)
    _, out = run(ecosystem, "--upstream-version", "0.15.1", "--notify", "--dry-run")
    assert "NOTIFY downstream" in out, out
    assert "CI job" not in out, out


def test_self_gated_repo_skips_docs_only_notes(ecosystem: Path) -> None:
    """A downstream declaring it gates upstream-version prose itself gets a
    SKIP, not a fourth identical drift note."""
    _docs_only_drift(ecosystem)
    _write(
        ecosystem / "downstream" / ".upstream-version-gated",
        "tests/version_consistency.rs via cargo test --workspace\n",
    )
    _, out = run(ecosystem, "--upstream-version", "0.15.1", "--notify", "--dry-run")
    assert "NOTIFY" not in out, out
    assert "gates upstream-version prose itself" in out, out


def test_self_gate_does_not_suppress_a_blocked_range(ecosystem: Path) -> None:
    """The marker covers internal prose consistency only: that a *new upstream
    version exists* is knowledge the downstream's own gate cannot have."""
    _write(
        ecosystem / "downstream" / ".upstream-version-gated",
        "tests/version_consistency.rs\n",
    )
    _, out = run(ecosystem, "--upstream-version", "0.16.0", "--notify", "--dry-run")
    assert "NOTIFY downstream" in out, out
    assert "BLOCKED" in out, out
