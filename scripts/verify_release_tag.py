#!/usr/bin/env python3
"""Assert the release tag exists — locally, on the remote, and on the right commit.

The 0.15.3 release published `kglite` to crates.io and PyPI, and the local
clone had no `v0.15.3` tag for two days. Nothing was broken remotely: the tag
was on `origin` the whole time. Three things conspired, and each is still true
of every release:

1. **The tag is created remotely, by CI.** `softprops/action-gh-release` in
   `release.yml` creates `v<version>` if it does not exist. Nothing in
   the local working tree ever creates it.
2. **The release push never fetches.** The skill's ff-mechanic is
   `git push origin HEAD:main`, chosen so unrelated working-tree WIP never
   moves. Push does not fetch, and no later step does either — so the clone
   has no occasion to learn the tag exists.
3. **Version verification cannot see it.** Step 11 asks crates.io and PyPI
   what version is live. Both answered `0.15.3` correctly. "Did something
   publish" is not "is the release fully recorded" — the same shape as R9.

There is also a real failure this catches rather than merely tidies. Until
2026-08-30 the tag step lived **only** inside the PyPI publish job, so a failed
wheels job alongside a successful crates.io publish shipped the crates with no
tag and no GitHub Release — the 0.15.3 state above, and a version check reports
success for it. `release.yml`'s `tag-release` job now `needs:` all three publish
legs, which makes that state unreachable *going forward*; this check is what
notices if it becomes reachable again.

Read-only: `git fetch --tags` plus `git rev-parse` / `git ls-remote`. It never
creates, moves or deletes a tag — if the tag is missing, that is a finding for
a human, not something to paper over locally.

Usage:
    python scripts/verify_release_tag.py                 # version from Cargo.toml
    python scripts/verify_release_tag.py --version 0.15.3
    python scripts/verify_release_tag.py --no-fetch      # audit without network
    python scripts/verify_release_tag.py --self-test     # prove it can fail
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)


class TagProblem(Exception):
    """A verification failure with the remediation attached."""


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True)


def workspace_version(root: Path) -> str:
    """Read `[workspace.package] version` — the single source of truth."""
    manifest = root / "Cargo.toml"
    text = manifest.read_text(encoding="utf-8")
    section = text.split("[workspace.package]", 1)
    if len(section) != 2:
        raise TagProblem(f"{manifest} has no [workspace.package] section")
    match = VERSION_RE.search(section[1])
    if not match:
        raise TagProblem(f"{manifest} [workspace.package] declares no version")
    version = match.group(1)
    # An empty or malformed version read out of a manifest is how a green run
    # publishes nothing (CLAUDE.md, "verify the artifact SET"). Refuse it here
    # rather than build a tag name out of it.
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise TagProblem(f"workspace version is not a plain x.y.z release: {version!r}")
    return version


def verify(root: Path, version: str, *, fetch: bool = True, remote: str = "origin") -> list[str]:
    """Return human-readable confirmations, or raise TagProblem."""
    tag = f"v{version}"
    notes: list[str] = []

    if fetch:
        fetched = _run(["git", "fetch", "--tags", remote], root)
        if fetched.returncode != 0:
            raise TagProblem(
                f"git fetch --tags {remote} failed:\n{fetched.stderr.strip()}\n"
                f"Cannot decide anything about {tag} without it — a fetch failure "
                f"is not an absent tag."
            )

    local = _run(["git", "rev-parse", "--verify", f"refs/tags/{tag}"], root)
    if local.returncode != 0:
        raise TagProblem(
            f"{tag} does not exist locally, even after fetching from {remote}.\n"
            f"That means CI never created it. The tag comes from release.yml's\n"
            f"`tag-release` job, which runs only if every publish leg that was\n"
            f"supposed to ship succeeded — so check which leg did not, and note\n"
            f"that the registries can already hold this version."
        )
    local_sha = local.stdout.strip()
    notes.append(f"{tag} present locally at {local_sha[:12]}")

    ls = _run(["git", "ls-remote", "--tags", remote, tag], root)
    if ls.returncode != 0:
        raise TagProblem(f"git ls-remote {remote} {tag} failed:\n{ls.stderr.strip()}")
    # An annotated tag emits both `refs/tags/X` and the dereferenced
    # `refs/tags/X^{}`. Prefer the dereferenced line: it names the commit,
    # which is what `git rev-parse refs/tags/X` gives for a lightweight tag.
    remote_shas: dict[str, str] = {}
    for line in ls.stdout.splitlines():
        sha, _, ref = line.partition("\t")
        remote_shas["deref" if ref.endswith("^{}") else "direct"] = sha.strip()
    if not remote_shas:
        raise TagProblem(
            f"{tag} exists locally but not on {remote}. A local-only release tag "
            f"is invisible to everyone else; push it or find out why CI did not."
        )
    remote_sha = remote_shas.get("deref") or remote_shas["direct"]
    notes.append(f"{tag} present on {remote} at {remote_sha[:12]}")

    if remote_sha != local_sha:
        raise TagProblem(
            f"{tag} points at {local_sha[:12]} locally but {remote_sha[:12]} on "
            f"{remote}. A tag that moved after publish means the published "
            f"artifact and the tagged source are different code."
        )

    described = _run(["git", "log", "-1", "--format=%h %s", tag], root)
    if described.returncode == 0 and described.stdout.strip():
        notes.append(f"{tag} -> {described.stdout.strip()}")
    return notes


def _self_test() -> int:
    """Prove each branch fires on a real git repo built to break it.

    A verification that has never been observed failing is not a
    verification (doctrine R1).
    """
    failures: list[str] = []

    def git(repo: Path, *args: str) -> None:
        result = _run(["git", *args], repo)
        assert result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr}"

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        upstream = base / "upstream"
        upstream.mkdir()
        git(upstream, "init", "--quiet", "--bare")

        work = base / "work"
        work.mkdir()
        git(work, "init", "--quiet")
        git(work, "config", "user.email", "t@example.invalid")
        git(work, "config", "user.name", "T")
        (work / "Cargo.toml").write_text('[workspace.package]\nversion = "1.2.3"\n', encoding="utf-8")
        git(work, "add", "Cargo.toml")
        git(work, "commit", "--quiet", "-m", "init")
        git(work, "remote", "add", "origin", str(upstream))
        git(work, "push", "--quiet", "origin", "HEAD:refs/heads/main")

        # Version parsing is the input to everything else.
        if workspace_version(work) != "1.2.3":
            failures.append("workspace_version did not read the manifest")

        (work / "bad.toml").write_text("", encoding="utf-8")
        broken = base / "broken"
        broken.mkdir()
        (broken / "Cargo.toml").write_text('[workspace.package]\nversion = ""\n', encoding="utf-8")
        try:
            workspace_version(broken)
            failures.append("an empty version was accepted")
        except TagProblem:
            pass

        # No tag anywhere -> must fail. This is the 0.15.3 shape.
        try:
            verify(work, "1.2.3", fetch=False)
            failures.append("a missing tag was reported as present")
        except TagProblem as exc:
            if "does not exist locally" not in str(exc):
                failures.append(f"missing-tag message was unexpected: {exc}")

        # Local-only tag -> must fail (CI never made it; nobody else can see it).
        git(work, "tag", "v1.2.3")
        try:
            verify(work, "1.2.3", fetch=False)
            failures.append("a local-only tag was accepted")
        except TagProblem as exc:
            if "not on origin" not in str(exc):
                failures.append(f"local-only message was unexpected: {exc}")

        # Pushed and matching -> must pass.
        git(work, "push", "--quiet", "origin", "v1.2.3")
        try:
            notes = verify(work, "1.2.3", fetch=False)
            if not any("present on origin" in n for n in notes):
                failures.append("a correct tag produced no remote confirmation")
        except TagProblem as exc:
            failures.append(f"a correct tag was rejected: {exc}")

        # Divergent tag -> must fail. Otherwise the published artifact and the
        # tagged source could differ and nothing would say so.
        (work / "Cargo.toml").write_text('[workspace.package]\nversion = "1.2.3"\n# moved\n', encoding="utf-8")
        git(work, "commit", "--quiet", "-am", "second")
        git(work, "tag", "-f", "v1.2.3")
        try:
            verify(work, "1.2.3", fetch=False)
            failures.append("a tag that diverged from the remote was accepted")
        except TagProblem as exc:
            if "points at" not in str(exc):
                failures.append(f"divergence message was unexpected: {exc}")

    if failures:
        for line in failures:
            print(f"SELF-TEST FAIL  {line}", file=sys.stderr)
        return 1
    print("verify_release_tag --self-test: OK — every branch observed failing.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", help="release version; default reads Cargo.toml")
    parser.add_argument("--remote", default="origin")
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="audit local state without contacting the remote for new tags",
    )
    parser.add_argument("--self-test", action="store_true", help="prove this check can fail")
    args = parser.parse_args()

    if args.self_test:
        return _self_test()

    try:
        version = args.version or workspace_version(REPO_ROOT)
        for note in verify(REPO_ROOT, version, fetch=not args.no_fetch, remote=args.remote):
            print(f"release tag: {note}")
    except TagProblem as exc:
        print(f"release tag: FAILED\n{exc}", file=sys.stderr)
        return 1
    print("release tag: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
