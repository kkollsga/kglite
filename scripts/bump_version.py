#!/usr/bin/env python3
"""Bump the workspace version *everywhere it is written down*.

The version is single-sourced for the crates' own `package.version`
(`[workspace.package] version` in the root `Cargo.toml`, inherited via
`version.workspace = true`) — but that is **not** the only place the
version appears. Four member manifests also declare an internal
dependency on the engine with an explicit `version` requirement, because
`cargo publish` refuses a `path`-only dependency:

    crates/kglite-bolt-server/Cargo.toml
    crates/kglite-c/Cargo.toml
    crates/kglite-cli/Cargo.toml
    crates/kglite-mcp-server/Cargo.toml

Editing only `[workspace.package]` therefore leaves the workspace
*unresolvable* the moment the bump crosses a minor: `cargo metadata`
fails with "failed to select a version for the requirement
`kglite = ^0.14`" and every downstream release step that shells out to
cargo dies on its first call. That happened during the 0.14.5 -> 0.15.0
release and broke `make refresh-release-constants` on its first step.

This script edits all five places at once and then *verifies* with
`cargo metadata` that every member actually resolved.

Why the internal requirement carries the full `X.Y.Z` and not the `X.Y`
series it used to:

1. **`^0.15` is a false claim.** The five published crates ship in
   lockstep and this project deliberately ships documented breaking
   engine changes in patch bumps. `kglite = "0.15"` tells cargo that
   `kglite-cli 0.15.3` builds against `kglite 0.15.0`; it frequently
   does not. `kglite = "0.15.3"` states the true floor and, being a
   caret requirement, still admits 0.15.4+ — it constrains nothing a
   lockstep release wants to do.
2. **It makes the bump path load-bearing every release.** With the `X.Y`
   form the requirement only had to change on a minor bump, so the
   omission stayed invisible across a dozen patch releases and then
   detonated on the first minor. With `X.Y.Z` the requirement changes
   every single release, which is exactly what `--check` gates.
3. **The publish order supports it.** `release.yml` publishes
   `kglite` first and waits for the crates.io index to serve it before
   publishing the four dependents, so `^X.Y.Z` is satisfiable by the
   time any dependent is verified.

Usage::

    python scripts/bump_version.py --check      # gate: everything in sync?
    python scripts/bump_version.py --set 0.15.1 # release: bump all five

`--check` is pure file reading (no cargo invocation) so it is cheap
enough for `make gate`. `--set` runs a resolving `cargo metadata` at the
end and fails loudly if any member did not resolve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_MANIFEST = REPO_ROOT / "Cargo.toml"

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
WORKSPACE_VERSION_RE = re.compile(
    r'(^\[workspace\.package\]\s*$.*?^\s*version\s*=\s*")([^"]+)(")',
    re.MULTILINE | re.DOTALL,
)
MEMBERS_RE = re.compile(r"^\[workspace\]\s*$.*?^\s*members\s*=\s*\[(?P<members>[^\]]*)\]", re.MULTILINE | re.DOTALL)
# A one-line inline-table dependency, e.g.
#   kglite = { version = "0.15", path = "../kglite", default-features = false }
INLINE_DEP_RE = re.compile(r"^(?P<key>[A-Za-z0-9_-]+)\s*=\s*\{(?P<body>[^}]*)\}\s*$")
BODY_VERSION_RE = re.compile(r'version\s*=\s*"(?P<version>[^"]*)"')
BODY_PATH_RE = re.compile(r'path\s*=\s*"(?P<path>[^"]*)"')

# The crates `release.yml` pushes to crates.io, in lockstep at one
# version. A divergence here is a broken publish set, not a warning.
PUBLISHED_CRATES = ("kglite", "kglite-bolt-server", "kglite-c", "kglite-cli", "kglite-mcp-server")


class BumpError(RuntimeError):
    """A condition that must stop the release, not print and continue."""


def read_workspace_version(text: str | None = None) -> str:
    text = ROOT_MANIFEST.read_text(encoding="utf-8") if text is None else text
    match = WORKSPACE_VERSION_RE.search(text)
    if match is None:
        raise BumpError(f"{ROOT_MANIFEST}: no [workspace.package] version found")
    return match.group(2)


def member_manifests() -> list[Path]:
    """Every `[workspace] members` manifest, in declaration order."""
    text = ROOT_MANIFEST.read_text(encoding="utf-8")
    match = MEMBERS_RE.search(text)
    if match is None:
        raise BumpError(f"{ROOT_MANIFEST}: no [workspace] members list found")
    members = re.findall(r'"([^"]+)"', match.group("members"))
    if not members:
        raise BumpError(f"{ROOT_MANIFEST}: [workspace] members list is empty")
    manifests = []
    for member in members:
        manifest = REPO_ROOT / member / "Cargo.toml"
        if not manifest.is_file():
            raise BumpError(f"workspace member {member!r} has no Cargo.toml")
        manifests.append(manifest)
    return manifests


def internal_version_pins(manifest: Path) -> list[tuple[int, str, str]]:
    """Dependencies on another workspace member that carry a `version`
    requirement — the pins a version bump has to move.

    Returns ``(line_number, dependency_key, version_requirement)``. A
    `path`-only dependency (no `version =`) is not a pin: it is never
    published, so nothing has to track the workspace version. A
    dependency whose path escapes the repository is external vendoring
    and is likewise left alone.
    """
    pins: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        dep = INLINE_DEP_RE.match(line)
        if dep is None:
            continue
        body = dep.group("body")
        path_match = BODY_PATH_RE.search(body)
        version_match = BODY_VERSION_RE.search(body)
        if path_match is None or version_match is None:
            continue
        target = (manifest.parent / path_match.group("path")).resolve()
        if not target.is_relative_to(REPO_ROOT):
            continue
        pins.append((lineno, dep.group("key"), version_match.group("version")))
    return pins


def check() -> list[str]:
    """Report every internal pin that disagrees with the workspace version."""
    workspace_version = read_workspace_version()
    if not SEMVER_RE.match(workspace_version):
        return [f"Cargo.toml: [workspace.package] version {workspace_version!r} is not X.Y.Z"]

    problems: list[str] = []
    for manifest in member_manifests():
        rel = manifest.relative_to(REPO_ROOT)
        for lineno, key, requirement in internal_version_pins(manifest):
            if requirement != workspace_version:
                problems.append(
                    f"{rel}:{lineno}: internal dependency {key!r} requires "
                    f"{requirement!r} but the workspace is at {workspace_version!r}"
                )
    return problems


def bump(new_version: str) -> list[Path]:
    """Rewrite the workspace version and every internal pin. Returns the
    files that changed."""
    if not SEMVER_RE.match(new_version):
        raise BumpError(f"{new_version!r} is not a X.Y.Z version")

    changed: list[Path] = []
    root_text = ROOT_MANIFEST.read_text(encoding="utf-8")
    old_version = read_workspace_version(root_text)
    if old_version != new_version:
        root_text = WORKSPACE_VERSION_RE.sub(rf"\g<1>{new_version}\g<3>", root_text, count=1)
        ROOT_MANIFEST.write_text(root_text, encoding="utf-8", newline="\n")
        changed.append(ROOT_MANIFEST)

    for manifest in member_manifests():
        pins = internal_version_pins(manifest)
        if not pins:
            continue
        lines = manifest.read_text(encoding="utf-8").splitlines(keepends=True)
        dirty = False
        for lineno, key, requirement in pins:
            if requirement == new_version:
                continue
            line = lines[lineno - 1]
            replaced, count = BODY_VERSION_RE.subn(f'version = "{new_version}"', line, count=1)
            if count != 1:
                raise BumpError(f"{manifest.relative_to(REPO_ROOT)}:{lineno}: could not rewrite {key!r} version")
            lines[lineno - 1] = replaced
            dirty = True
        if dirty:
            manifest.write_text("".join(lines), encoding="utf-8", newline="\n")
            changed.append(manifest)
    return changed


def declared_member_version(manifest: Path) -> str | None:
    """What the member's own ``[package]`` table says its version is.

    ``"workspace"`` when it inherits via ``version.workspace = true``, the
    literal string when it hard-codes one, ``None`` when it declares
    neither. A hard-coded value still *resolves* — cargo is perfectly
    happy with a member pinned to a stale version — so this is drift that
    dependency resolution cannot see.
    """
    in_package = False
    for line in manifest.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_package = stripped == "[package]"
            continue
        if not in_package:
            continue
        if re.match(r"^version\s*\.\s*workspace\s*=\s*true\s*$", stripped):
            return "workspace"
        literal = re.match(r'^version\s*=\s*"([^"]+)"', stripped)
        if literal:
            return literal.group(1)
    return None


def resolve_workspace_versions() -> dict[str, str]:
    """Resolve the workspace and return ``{member_name: version}``.

    Note the *absence* of ``--no-deps``: that flag skips dependency
    resolution altogether, so it happily reports success on a workspace
    whose internal requirement can no longer be satisfied. Verified
    directly — a tree with `[workspace.package] version = "0.16.0"` and
    `kglite = "^0.15.0"` passes `cargo metadata --no-deps` and fails
    plain `cargo metadata` with "failed to select a version for the
    requirement `kglite = ^0.15.0`". Only the resolving form is a check.
    """
    proc = subprocess.run(
        ["cargo", "metadata", "--format-version", "1"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "no diagnostic output"
        raise BumpError(f"cargo metadata could not resolve the workspace:\n{detail}")
    metadata = json.loads(proc.stdout)
    members = set(metadata["workspace_members"])
    return {pkg["name"]: pkg["version"] for pkg in metadata["packages"] if pkg["id"] in members}


def verify_resolves(expected: str) -> None:
    """The workspace must resolve and every member must be at `expected`."""
    packages = resolve_workspace_versions()
    wrong = sorted(f"{name} {version}" for name, version in packages.items() if version != expected)
    if wrong:
        raise BumpError(f"members did not resolve to {expected}: {', '.join(wrong)}")
    print(f"cargo metadata: all {len(packages)} workspace members resolved to {expected}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Verify every internal pin matches the workspace version.")
    group.add_argument("--set", metavar="X.Y.Z", help="Bump the workspace version and every internal pin.")
    args = parser.parse_args()

    try:
        if args.check:
            problems = check()
            if problems:
                print("internal kglite dependency requirements are out of sync:", file=sys.stderr)
                for problem in problems:
                    print(f"  {problem}", file=sys.stderr)
                print(
                    "\nfix: python scripts/bump_version.py --set "
                    f"{read_workspace_version()}   (or `make bump-version VERSION=X.Y.Z`)",
                    file=sys.stderr,
                )
                return 1
            print(f"internal dependency requirements in sync at {read_workspace_version()}")
            return 0

        changed = bump(args.set)
        for path in changed:
            print(f"bumped {path.relative_to(REPO_ROOT)}")
        if not changed:
            print(f"already at {args.set} — nothing to rewrite")
        verify_resolves(args.set)
        remaining = check()
        if remaining:
            raise BumpError("post-bump check still reports drift:\n  " + "\n  ".join(remaining))
        return 0
    except BumpError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
