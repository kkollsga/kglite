#!/usr/bin/env python3
"""Report whether the tree is ready for the release commit. Change nothing.

The release skill encodes several ordering dependencies in prose, and
each one fails in a way that does not name its own cause:

* **Bump before refreshing constants.** The golden ``.kgl`` digest embeds
  the version string, so refreshing first captures the outgoing version
  and the phase-4 parity test fails later with a digest mismatch that
  says nothing about ordering.
* **Rebuild the server binaries after the bump.** A stale prebuilt
  binary fails its suite with a contract error that never mentions the
  version.
* **Re-run ``ruff format`` after the refresh.** The refresh rewrites
  ``tests/test_phase4_parity.py`` / ``tests/test_phase5_parity.py`` and
  can leave them unformatted, so ``make gate`` goes red immediately
  after a clean refresh.

This script turns those into named checks. Run it after promoting the
CHANGELOG and before writing the ``release(x.y.z)`` commit.

**It is a checker, not a driver, and that is deliberate.** It never
bumps, never refreshes, never builds, never formats, never commits.
There is no ``--fix``. A tool that reports "these four things are not
ready" makes the maintainer faster; a tool that quietly performs the
steps it checks is how gates stop gating — which is the whole lesson of
the 0.15.0 release. Every remediation is printed as the exact command to
run, and running it stays the maintainer's decision.

Nothing here judges *whether* to release, what size the bump should be,
or how to read ``make semver-check``. Those are human calls.

Usage:
    python scripts/release_preflight.py            # exit 1 if anything is not ready
    python scripts/release_preflight.py --base origin/main
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import bump_version  # noqa: E402
import check_release_hygiene  # noqa: E402

ROOT_MANIFEST = REPO_ROOT / "Cargo.toml"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
PHASE5_TEST = REPO_ROOT / "tests" / "test_phase5_parity.py"
BASELINES_DIR = REPO_ROOT / "tests" / "benchmarks" / "baselines"
SERVER_BINARIES = ("kglite-mcp-server", "kglite-bolt-server")

RELEASE_SECTION_RE = re.compile(r"^## \[(?P<version>\d+\.\d+\.\d+)\]")


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    remedy: str = ""


def _version_slug(version: str) -> str:
    return version.replace(".", "_")


def check_changelog_version(version: str) -> Check:
    """The topmost released section must be the version being shipped."""
    for line in CHANGELOG.read_text(encoding="utf-8").splitlines():
        match = RELEASE_SECTION_RE.match(line)
        if match is None:
            continue
        top = match.group("version")
        if top == version:
            return Check("changelog version", True, f"top release section is [{version}]")
        return Check(
            "changelog version",
            False,
            f"workspace is at {version} but the top release section is [{top}]",
            "promote CHANGELOG [Unreleased] -> [%s] (release skill step 7)" % version,
        )
    return Check(
        "changelog version",
        False,
        "no `## [x.y.z]` release section found",
        "promote CHANGELOG [Unreleased] -> [%s] (release skill step 7)" % version,
    )


def check_internal_pins() -> Check:
    problems = bump_version.check()
    if problems:
        return Check(
            "internal version pins",
            False,
            problems[0] if len(problems) == 1 else f"{len(problems)} manifests out of sync",
            "make bump-version VERSION=%s" % bump_version.read_workspace_version(),
        )
    return Check("internal version pins", True, "all internal kglite requirements match the workspace")


def check_member_inheritance(version: str) -> Check:
    """No member may silently carry its own version.

    `make bump-version` fixes the forward path, but nothing stops the
    state drifting some other way — a merge, a hand-edit, a rebase that
    resolves a manifest conflict wrongly. A member with a stale explicit
    `version = "..."` still *resolves*, so dependency resolution cannot
    see this class at all; only reading the manifests can.
    """
    problems: list[str] = []
    for manifest in bump_version.member_manifests():
        declared = bump_version.declared_member_version(manifest)
        rel = manifest.relative_to(REPO_ROOT)
        if declared is None:
            problems.append(f"{rel}: [package] declares no version at all")
        elif declared != "workspace" and declared != version:
            problems.append(f"{rel}: [package] version = {declared!r}, workspace is {version!r}")
    if problems:
        return Check(
            "member inheritance",
            False,
            problems[0] if len(problems) == 1 else f"{len(problems)} members diverge",
            "set `version.workspace = true` in the offending manifest(s) — do not hand-set a version",
        )
    return Check("member inheritance", True, "every member inherits [workspace.package] version")


def check_workspace_resolves(resolved: dict[str, str] | None, failure: str) -> Check:
    """Every internal `kglite*` requirement must be satisfiable by the
    version in the tree — the exact condition that broke 0.15.0."""
    if resolved is None:
        reason = next(
            (line for line in failure.splitlines() if "failed to select a version" in line),
            failure.splitlines()[0] if failure else "cargo metadata failed",
        )
        return Check(
            "workspace resolves",
            False,
            reason.strip(),
            "make bump-version VERSION=%s   (do not hand-edit the manifests)" % bump_version.read_workspace_version(),
        )
    return Check("workspace resolves", True, f"cargo metadata resolved all {len(resolved)} members")


def check_publish_lockstep(version: str, resolved: dict[str, str] | None) -> Check:
    """All five published crates at one version — they publish as a set."""
    if resolved is None:
        return Check("publish lockstep", False, "workspace does not resolve; cannot check", "see `workspace resolves`")
    problems: list[str] = []
    for crate in bump_version.PUBLISHED_CRATES:
        actual = resolved.get(crate)
        if actual is None:
            problems.append(f"{crate} is not a workspace member")
        elif actual != version:
            problems.append(f"{crate} resolved to {actual}, expected {version}")
    if problems:
        return Check(
            "publish lockstep",
            False,
            "; ".join(problems),
            "make bump-version VERSION=%s" % version,
        )
    return Check("publish lockstep", True, f"all {len(bump_version.PUBLISHED_CRATES)} published crates at {version}")


def check_server_binaries() -> Check:
    """A prebuilt server binary older than the manifest is stale — its
    suite then fails on a contract mismatch that never mentions the
    version."""
    manifest_mtime = ROOT_MANIFEST.stat().st_mtime
    stale: list[str] = []
    missing: list[str] = []
    for name in SERVER_BINARIES:
        binary = REPO_ROOT / "target" / "release" / name
        if not binary.exists():
            missing.append(name)
        elif binary.stat().st_mtime < manifest_mtime:
            stale.append(name)
    if missing or stale:
        parts = []
        if missing:
            parts.append("missing: " + ", ".join(missing))
        if stale:
            parts.append("older than Cargo.toml: " + ", ".join(stale))
        return Check(
            "server binaries",
            False,
            "; ".join(parts),
            "cargo build -p kglite-mcp-server -p kglite-bolt-server --release",
        )
    return Check("server binaries", True, "both release binaries are newer than Cargo.toml")


def check_formatting() -> Check:
    """`cargo fmt --check` + `ruff format --check` — the refresh dirties
    formatting, so this is the check that catches a missed re-run."""
    failures: list[str] = []
    rust = subprocess.run(["cargo", "fmt", "--", "--check"], cwd=REPO_ROOT, capture_output=True, text=True)
    if rust.returncode != 0:
        failures.append("cargo fmt")
    ruff = REPO_ROOT / ".venv" / "bin" / "ruff"
    if not ruff.exists():
        return Check(
            "formatting",
            False,
            "no .venv/bin/ruff — cannot verify Python formatting",
            "uv venv .venv && uv pip install --python .venv/bin/python ruff",
        )
    python = subprocess.run([str(ruff), "format", "--check", "."], cwd=REPO_ROOT, capture_output=True, text=True)
    if python.returncode != 0:
        failures.append("ruff format")
    if failures:
        return Check(
            "formatting",
            False,
            ", ".join(failures) + " reports unformatted files",
            "cargo fmt && make fmt-py",
        )
    return Check("formatting", True, "cargo fmt and ruff format are clean")


def check_captured_constants(version: str) -> Check:
    """The captured constants that are cheap to verify from files alone:
    the per-version perf baseline and the phase-5 size-history entry.

    The `.kgl` golden digest is deliberately not recomputed here — that
    needs a built extension, and the phase-4 parity test is its gate.
    """
    problems: list[str] = []
    suffix = ".linux" if sys.platform.startswith("linux") else ""
    baseline = BASELINES_DIR / f"{_version_slug(version)}{suffix}.json"
    if not baseline.exists():
        problems.append(f"no perf baseline {baseline.name}")
    history = re.search(rf"^\s+- {re.escape(version)}:", PHASE5_TEST.read_text(encoding="utf-8"), re.MULTILINE)
    if history is None:
        problems.append(f"no {version} entry in the phase-5 binary-size history")
    if problems:
        return Check("captured constants", False, "; ".join(problems), "make refresh-release-constants")
    return Check("captured constants", True, f"perf baseline and size-history entry present for {version}")


def check_hygiene() -> Check:
    """Reuse the `make gate` lint so preflight and the gate cannot disagree."""
    problems = check_release_hygiene.check_changelog() + check_release_hygiene.check_refresh_todos()
    if problems:
        return Check(
            "release hygiene",
            False,
            problems[0] if len(problems) == 1 else f"{len(problems)} problems",
            "make check-release-hygiene   (full list)",
        )
    return Check("release hygiene", True, "CHANGELOG structure and release-constant markers clean")


def check_fast_forward(base: str) -> Check:
    """`base` must be an ancestor of HEAD, or the release push is not a
    fast-forward and step 9's `git push origin HEAD:main` will be
    rejected."""
    rev = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", base], cwd=REPO_ROOT, capture_output=True, text=True
    )
    if rev.returncode != 0:
        return Check(
            "fast-forward",
            False,
            f"{base} does not resolve — cannot verify the push would fast-forward",
            f"git fetch origin {base.split('/')[-1]}",
        )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    if ancestor.returncode != 0:
        return Check(
            "fast-forward",
            False,
            f"{base} is not an ancestor of HEAD — the release push would not fast-forward",
            f"git fetch origin && git rebase {base}",
        )
    return Check("fast-forward", True, f"{base} is an ancestor of HEAD")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--base", default="origin/main", help="Ref the release push must fast-forward (default origin/main)."
    )
    args = parser.parse_args()

    try:
        version = bump_version.read_workspace_version()
    except bump_version.BumpError as exc:
        print(f"release preflight: {exc}", file=sys.stderr)
        return 1

    print(f"release preflight for {version} — reporting only, nothing is modified\n")

    # One resolution pass feeds both workspace-coherence checks.
    resolved: dict[str, str] | None
    failure = ""
    try:
        resolved = bump_version.resolve_workspace_versions()
    except bump_version.BumpError as exc:
        resolved, failure = None, str(exc)

    checks = [
        check_internal_pins(),
        check_member_inheritance(version),
        check_workspace_resolves(resolved, failure),
        check_publish_lockstep(version, resolved),
        check_changelog_version(version),
        check_hygiene(),
        check_captured_constants(version),
        check_server_binaries(),
        check_formatting(),
        check_fast_forward(args.base),
    ]

    width = max(len(check.name) for check in checks)
    for check in checks:
        mark = "PASS" if check.ok else "NOT READY"
        print(f"  {mark:<9}  {check.name:<{width}}  {check.detail}")

    failed = [check for check in checks if not check.ok]
    if not failed:
        print(f"\nall {len(checks)} preconditions satisfied — the release commit can be written.")
        return 0

    print(f"\n{len(failed)} of {len(checks)} preconditions not satisfied. Run these yourself:")
    for check in failed:
        if check.remedy:
            print(f"  {check.name}: {check.remedy}")
    print("\n(This script performs no release step. Deciding what to run, and whether to")
    print(" release at all, stays yours.)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
