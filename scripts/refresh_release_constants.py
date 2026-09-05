#!/usr/bin/env python3
"""Refresh the captured constants that drift across releases.

Four captured values are version-/toolchain-coupled and silently rot if
nobody updates them at release time:

  1. ``tests/test_phase4_parity.py::GOLDEN_V3_DIGEST`` — embeds the
     version string in the ``.kgl`` header. Every release shifts the
     digest even when the format itself is unchanged.

  2. ``tests/test_phase5_parity.py::test_binary_size_regression``
     baseline — the release-built ``libkglite`` size, +10% over baseline.

  3. ``tests/benchmarks/baselines/<version>.json`` — pytest-benchmark
     JSON for the tracked core benchmarks. ``current.json`` is a copy.

  4. ``tests/api-baselines/rust/*.txt`` — the feature-profiled ``kglite``
     public API surfaces (cargo-public-api on the manifest-pinned nightly).
     They drift whenever a profile's public API legitimately changes; the CI
     public-api gate fails until all exact baselines are refreshed and committed.

This script reads ``Cargo.toml`` for the version, then refreshes all
four. Idempotent: running it twice in a row produces no diff.

**No step is best-effort.** A step that cannot do its job — missing
release artifact, missing or wrong-version tooling, an anchor the
rewriter can no longer find, a failed benchmark run — exits non-zero
with the command that fixes it. It never prints a line and lets the
release continue: during the 0.15.0 release step 4 found
``cargo-public-api 0.52.0`` on ``PATH`` against a manifest pin of
``0.49.0``, reported a no-op, and the run carried on; only a human
reading the output stopped stale public-API baselines from shipping,
and those baselines gate downstream Rust embedders and the C ABI.

The one intentional skip is step 3's benchmark re-capture, which is
skipped when ``<version>.json`` already exists. That is idempotence, not
a swallowed failure: benchmark numbers are inherently noisy, so the
per-version file is written once and re-running must not churn it.
``--skip-benchmarks`` is the other intentional skip — an explicit
operator request, not a silent one.

Usage:
    python scripts/refresh_release_constants.py [--skip-benchmarks]

``--skip-benchmarks`` skips the perf-baseline capture (~15s wall-clock,
sometimes useful when iterating on the doc bits of a release commit).

When the diff is what you expected, stage and amend it into the
``release(x.y.z): ...`` commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

try:
    from scripts.benchmark_qualification import record_capture
except ModuleNotFoundError:  # Direct script execution.
    from benchmark_qualification import record_capture

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE4_TEST = REPO_ROOT / "tests" / "test_phase4_parity.py"
PHASE5_TEST = REPO_ROOT / "tests" / "test_phase5_parity.py"
BASELINES_DIR = REPO_ROOT / "tests" / "benchmarks" / "baselines"
API_PROFILE_MANIFEST = REPO_ROOT / "tests" / "api-baselines" / "rust-api-profiles.json"


class RefreshError(RuntimeError):
    """A step could not do its job.

    Raising this aborts the run with a non-zero exit and the remediation
    command. Use it for *every* condition that leaves a captured
    constant unrefreshed — a release must not proceed on the assumption
    that a printed line was read. A genuine no-op (the value is already
    current, or an operator asked for a skip) returns normally instead.
    """


def read_version() -> str:
    """Pull ``version = "X.Y.Z"`` from ``[workspace.package]`` in the root
    ``Cargo.toml``.

    Pre-G.4 the root ``Cargo.toml`` held the project version (line 3).
    Post-G.4 the workspace root went virtual; for a while the wheel version
    lived in ``crates/kglite-py/Cargo.toml``. Since the "single-source the
    crate version via ``[workspace.package]``" change every member crate sets
    ``version.workspace = true`` and inherits the one value declared under
    ``[workspace.package]`` in the root ``Cargo.toml`` — so that table is now
    the single source of truth the captured constants (.kgl header, binary
    size, benchmark baselines) all describe.
    """
    text = (REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8")
    # Scope the search to the [workspace.package] table so we don't pick up an
    # unrelated `version = ` from another top-level table.
    m = re.search(
        r'^\[workspace\.package\]\s*$.*?^\s*version\s*=\s*"([^"]+)"\s*$',
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not m:
        sys.exit("Cargo.toml: no [workspace.package] version found")
    return m.group(1)


def version_slug(version: str) -> str:
    """0.9.52 → '0_9_52' (the convention used by existing baselines)."""
    return version.replace(".", "_")


def find_release_dylib() -> Path | None:
    """Locate the wheel cdylib, falling back to the core Rust library."""
    for cand in (
        REPO_ROOT / "target" / "release" / "libkglite_py.dylib",
        REPO_ROOT / "target" / "release" / "libkglite_py.so",
        REPO_ROOT / "target" / "release" / "libkglite.dylib",
        REPO_ROOT / "target" / "release" / "libkglite.so",
    ):
        if cand.exists():
            return cand
    return None


# ── 1. .kgl v3 golden digest ───────────────────────────────────────────


def compute_kgl_digest() -> str:
    """Build the fixture graph and hash its .kgl bytes. Reuses the same
    helper the test imports so a digest mismatch can never be a fixture
    drift bug."""
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from test_phase4_parity import _save_memory_fixture_to_bytes  # type: ignore

    return hashlib.sha256(_save_memory_fixture_to_bytes()).hexdigest()


def refresh_kgl_golden(version: str, new_digest: str) -> tuple[bool, str]:
    """Update GOLDEN_V3_DIGEST + demote prior value into ACCEPTABLE_DIGESTS.
    Returns (changed, message)."""
    text = PHASE4_TEST.read_text(encoding="utf-8")

    cur_match = re.search(r'^(GOLDEN_V3_DIGEST = )"([0-9a-f]{64})"', text, re.MULTILINE)
    if cur_match is None:
        raise RefreshError(
            "tests/test_phase4_parity.py: GOLDEN_V3_DIGEST line not found. The test moved or "
            "was renamed; update this script's rewriter before releasing."
        )

    cur_digest = cur_match.group(2)
    if cur_digest == new_digest:
        return False, f"GOLDEN_V3_DIGEST already current ({new_digest[:12]}…)"

    if cur_digest in text.split("ACCEPTABLE_DIGESTS")[1] if "ACCEPTABLE_DIGESTS" in text else False:
        # Already in the allowlist; only need to update the primary.
        pass
    else:
        # Append into the allowlist before the closing brace. Find the
        # last entry inside ACCEPTABLE_DIGESTS and insert after it.
        marker = re.search(r"(    )\}\s*\n\)\s*\n", text)
        if marker is None:
            raise RefreshError(
                "tests/test_phase4_parity.py: ACCEPTABLE_DIGESTS closing brace not found, so the "
                "outgoing golden digest cannot be demoted. Fix the rewriter or demote it by hand."
            )
        indent = marker.group(1)
        insert = f'{indent}# Demoted from GOLDEN_V3_DIGEST when {version} took over.\n{indent}"{cur_digest}",\n'
        text = text[: marker.start()] + insert + text[marker.start() :]

    # Update the primary digest.
    text = re.sub(
        r'^GOLDEN_V3_DIGEST = "[0-9a-f]{64}"',
        f'GOLDEN_V3_DIGEST = "{new_digest}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )

    PHASE4_TEST.write_text(text, encoding="utf-8", newline="\n")
    return True, f"GOLDEN_V3_DIGEST -> {new_digest[:12]}… (prior {cur_digest[:12]}… demoted to ACCEPTABLE_DIGESTS)"


# ── 2. Binary-size baseline ────────────────────────────────────────────


def refresh_binary_size(version: str, current_size: int) -> tuple[bool, str]:
    """Update the current platform's size baseline + history note."""
    text = PHASE5_TEST.read_text(encoding="utf-8")

    platform_key = sys.platform if sys.platform in {"darwin", "linux"} else "linux"
    bl_match = re.search(
        rf'^(\s*"{platform_key}"\s*:\s*)([0-9_]+)(,\s*#\s*[^\n]+\n)',
        text,
        re.MULTILINE,
    )
    if bl_match is None:
        raise RefreshError(
            f"tests/test_phase5_parity.py: no {platform_key!r} baseline entry to update. The "
            "baseline table moved or was reformatted; fix the rewriter before releasing."
        )
    cur_baseline = int(bl_match.group(2).replace("_", ""))

    original_text = text
    unchanged = cur_baseline == current_size
    if unchanged:
        # An unchanged SIZE is not an unchanged RELEASE. The history is a
        # per-release ledger, and `release_preflight` requires a row for the
        # version being cut -- an absent row reads as an *unmeasured* release,
        # not an unchanged one. Returning early here made the two tools
        # disagree: this one reported "already current" while preflight
        # reported "no entry", which is exactly what happened cutting 0.15.1.
        # So fall through and write the row (and re-stamp the version comment
        # on the baseline line), just with the value left alone.
        pass

    # Re-stamp the baseline line so its comment names the version being cut,
    # even when the number itself did not move.
    formatted = f"{current_size:_}".replace("_", "_")  # "12_345_678" style
    # No "(unchanged)" marker here: the comment must be a pure function of
    # (size, version, platform) or re-stamping it is not idempotent, and the
    # second refresh of a release would report a change that did not happen.
    # The unchanged-ness is recorded in the history row's prose instead.
    new_line = f"{bl_match.group(1)}{formatted},  # {version} {platform_key} baseline\n"
    text = text[: bl_match.start()] + new_line + text[bl_match.end() :]

    # Best-effort: drop a marker into the docstring's "Baseline history:"
    # block so the growth narrative gains an entry. We don't try to
    # rewrite the whole prose — that's the maintainer's job; we just
    # leave a TODO so they don't forget.
    if unchanged:
        # Nothing grew, so there is nothing for a human to describe -- emitting
        # a TODO here would be a marker that can never be resolved, and the
        # hygiene lint would (correctly) block the release on it.
        todo_marker = (
            f"\n      - {version}:       {current_size:,} bytes — **unchanged** from the "
            "prior baseline; this release moved no code size.\n"
        )
    else:
        todo_marker = (
            f"\n      - {version}:       {current_size:,} bytes "
            f"(≈{current_size / (1024 * 1024):.1f} MB). "
            "TODO: describe what grew since the prior baseline.\n"
        )
    existing_history = re.search(rf"^      - {re.escape(version)}:.*$", text, re.MULTILINE)
    if existing_history is not None:
        # A row for this version already exists — leave it exactly as it is.
        #
        # Rewriting it was both wrong and unstable. Wrong: the maintainer
        # replaces the generated TODO with real prose describing what grew,
        # and regenerating would delete that. Unstable: `unchanged` compares
        # the size against the baseline *in the file*, which the first run
        # just updated, so a second run reclassifies the same release as
        # "unchanged" and rewrites the row it had written moments earlier.
        pass
    else:
        history_anchor = "    Raising the baseline is a deliberate act"
        if history_anchor in text:
            text = text.replace(history_anchor, todo_marker + "\n" + history_anchor, 1)

    # Update the in-message "+10% over X baseline" string to reference
    # the new version.
    text = re.sub(
        r"\(\+10% over [^)]* baseline \{baseline:,\}\)",
        f"(+10% over {version} {{platform_key}} baseline {{baseline:,}})",
        text,
    )

    if text == original_text:
        # Re-running after a completed refresh must be a no-op, or the caller
        # cannot tell "I just recorded this" from "this was already recorded".
        # The unchanged-size path still falls through to here, so idempotency
        # is decided by whether the FILE moved, never by whether the size did.
        return False, f"binary-size baseline and history already current ({current_size:,} bytes)"

    PHASE5_TEST.write_text(text, encoding="utf-8", newline="\n")
    if unchanged:
        return True, f"binary-size unchanged at {current_size:,} bytes — history row recorded"
    return True, f"binary-size baseline {cur_baseline:,} -> {current_size:,} bytes"


# ── 3. Perf baseline ───────────────────────────────────────────────────


def refresh_perf_baseline(version: str) -> tuple[bool, str]:
    """Capture pytest-benchmark JSON for the tracked core benchmarks
    and slim the per-iteration ``data`` field out of the result.

    Per-platform — Linux runners are ~2-3x slower than Apple Silicon for
    these benchmarks (same source, different hardware), so a single
    baseline can't gate both. The output filename gets a `.linux` infix
    on Linux; macOS uses the bare name (legacy / default). Both files
    coexist in `tests/benchmarks/baselines/`; CI picks
    `current.linux.json`, local macOS dev uses `current.json`.

    Idempotent: when ``<version>.json`` already exists for *this*
    platform, we skip the re-capture. Benchmark numbers are inherently
    noisy (thermal / system-load) so re-running would produce churn
    even when nothing relevant has changed. The version slug + platform
    are the trigger — bump Cargo.toml → file missing → fresh capture.
    """
    plat_suffix = ".linux" if sys.platform.startswith("linux") else ""
    target = BASELINES_DIR / f"{version_slug(version)}{plat_suffix}.json"
    current = BASELINES_DIR / f"current{plat_suffix}.json"

    if target.exists():
        record_capture(target, evidence="Existing release capture; qualification retained or pending review.")
        return False, f"perf capture {target.name} already present; qualification preserved"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_json = Path(tmp) / "bench.json"
        cmd = [
            # Use the active interpreter's pytest (not a bare `pytest` on PATH,
            # which may resolve to an env without the pytest-benchmark plugin —
            # then `--benchmark-*` args fail as "unrecognized arguments").
            sys.executable,
            "-m",
            "pytest",
            str(REPO_ROOT / "tests" / "benchmarks" / "test_bench_core.py"),
            "-m",
            "benchmark",
            "--benchmark-min-rounds=100",
            "--benchmark-warmup=on",
            "--benchmark-warmup-iterations=20",
            f"--benchmark-json={tmp_json}",
            "-q",
        ]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RefreshError(f"benchmark run failed (no perf baseline captured):\n{proc.stdout}\n{proc.stderr}")
        data = json.loads(tmp_json.read_text(encoding="utf-8"))

    # Strip per-iteration `data` — gates need aggregates only; carrying
    # the full series bloats commits to ~30 MB per release.
    for b in data["benchmarks"]:
        b["stats"].pop("data", None)

    target.write_text(json.dumps(data, indent=2), encoding="utf-8", newline="\n")
    shutil.copyfile(target, current)
    record_capture(target, alias=current, evidence="New release capture; awaiting control/measurement qualification.")
    return True, (
        f"raw perf capture written to {target.relative_to(REPO_ROOT)} and {current.name}; "
        "qualification pending, approved reference unchanged. After reviewing controls and repeat runs, "
        "use scripts/benchmark_qualification.py CAPTURE --status accepted --evidence REASON --promote."
    )


# ── orchestration ──────────────────────────────────────────────────────


# ── 4. public-api baseline ──────────────────────────────────────────────


def refresh_api_baseline() -> tuple[bool, str]:
    """Regenerate every manifest-declared cargo-public-api baseline.

    Both the toolchain pin and the tool-version pin are hard
    requirements. `rust_api_profiles.py` rejects a mismatched
    cargo-public-api version outright — a *different* version emits a
    different surface, so a baseline captured with the wrong one is not
    a baseline. Neither the missing-tool nor the wrong-version case may
    degrade to a skip: these baselines gate downstream Rust embedders
    and the C ABI.
    """
    manifest = json.loads(API_PROFILE_MANIFEST.read_text(encoding="utf-8"))
    if shutil.which("cargo-public-api") is None:
        raise RefreshError(
            "cargo-public-api is not installed, so the public-API baselines cannot be "
            "refreshed. Install the pinned toolchain + tool:\n"
            '  rustup toolchain install "$(python scripts/rust_api_profiles.py value nightly)"\n'
            "  cargo install cargo-public-api --locked --version "
            '"$(python scripts/rust_api_profiles.py value cargo_public_api_version)"'
        )
    paths = [REPO_ROOT / profile["baseline"] for profile in manifest["profiles"]]
    before = {path: path.read_text(encoding="utf-8") if path.exists() else None for path in paths}
    proc = subprocess.run(
        [sys.executable, "scripts/rust_api_profiles.py", "refresh", "--skip-if-unchanged"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr.strip() or proc.stdout.strip() or "unknown error").splitlines()[-1]
        raise RefreshError(
            f"cargo public-api profile refresh failed: {detail}\n"
            "The pinned toolchain and tool version are single-sourced in "
            "tests/api-baselines/rust-api-profiles.json:\n"
            '  rustup toolchain install "$(python scripts/rust_api_profiles.py value nightly)"\n'
            "  cargo install cargo-public-api --locked --version "
            '"$(python scripts/rust_api_profiles.py value cargo_public_api_version)"'
        )
    after = {path: path.read_text(encoding="utf-8") if path.exists() else None for path in paths}
    if before == after:
        return False, f"all {len(paths)} Rust API profile baselines already current"
    return True, f"refreshed {len(paths)} Rust API profile baselines"


def run() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-benchmarks", action="store_true", help="Skip the perf-baseline capture (~15s wall-clock).")
    args = p.parse_args()

    version = read_version()
    print(f"refreshing captured constants for {version}\n")

    # 1. .kgl golden
    print("1. .kgl v3 golden digest")
    digest = compute_kgl_digest()
    changed, msg = refresh_kgl_golden(version, digest)
    print(f"   {'CHANGED' if changed else 'no-op '}: {msg}\n")

    # 2. binary size
    print("2. binary-size baseline")
    dylib = find_release_dylib()
    if dylib is None:
        raise RefreshError(
            "no target/release/libkglite_py.{dylib,so} or libkglite.{dylib,so} — the binary-size "
            "baseline describes the release artifact and cannot be captured without one.\n"
            "  uv run --no-sync maturin develop --release   (or: cargo build --release)"
        )
    size = dylib.stat().st_size
    changed, msg = refresh_binary_size(version, size)
    print(f"   {'CHANGED' if changed else 'no-op '}: {msg}\n")

    # 3. perf baseline
    if args.skip_benchmarks:
        print("3. perf baseline — skipped (--skip-benchmarks).\n")
    else:
        print("3. perf baseline (running tracked core benchmarks)…")
        changed, msg = refresh_perf_baseline(version)
        print(f"   {'CHANGED' if changed else 'no-op '}: {msg}\n")

    # 4. public-api baseline
    print("4. feature-profiled public-api baselines (cargo-public-api on manifest-pinned nightly)")
    changed, msg = refresh_api_baseline()
    print(f"   {'CHANGED' if changed else 'no-op '}: {msg}\n")

    # 5. packaged-consumer fixture lockfile. The fixture consumes kglite by
    # version under --locked, so every workspace version bump (and any new
    # dependency) must re-resolve its lock or the packaged-feature CI jobs
    # fail on main (bit 0.14.4 and 0.14.5).
    print("5. packaged-consumer fixture lockfile")
    fixture_lock = REPO_ROOT / "tests/fixtures/rust-embed-consumer/Cargo.lock"
    before = fixture_lock.read_bytes()
    proc = subprocess.run(
        ["cargo", "update", "--manifest-path", "tests/fixtures/rust-embed-consumer/Cargo.toml", "-p", "kglite"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "no diagnostic output"
        raise RefreshError(
            f"packaged-consumer fixture lock did not re-resolve:\n{detail}\n"
            "The packaged-feature CI jobs consume kglite by version under --locked, so a stale "
            "fixture lock fails on main."
        )
    after = fixture_lock.read_bytes()
    print(f"   {'CHANGED' if before != after else 'no-op '}: tests/fixtures/rust-embed-consumer/Cargo.lock\n")

    # Pretty diff summary.
    # ── 6. Generated docs facts ────────────────────────────────────────
    # `docs/_generated/project-facts.md` embeds the version, so every bump
    # staled it. Nothing in this script regenerated it, and the only thing
    # that noticed was `make gate` -- *if* someone happened to run it after
    # the bump. Cutting 0.15.1 hit exactly that: a clean refresh, then a
    # red gate from a file the refresh should have owned.
    print("6. generated docs facts")
    facts = REPO_ROOT / "docs/_generated/project-facts.md"
    before_facts = facts.read_bytes() if facts.exists() else b""
    proc = subprocess.run(
        [sys.executable, "scripts/render_docs_facts.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "no diagnostic output"
        raise RefreshError(f"render_docs_facts.py failed:\n{detail}")
    after_facts = facts.read_bytes() if facts.exists() else b""
    if after_facts != before_facts:
        print(f"   CHANGED: {facts.relative_to(REPO_ROOT)}")
    else:
        print("   no-op : generated docs facts already current")
    print()

    diff = subprocess.run(
        [
            "git",
            "diff",
            "--stat",
            "--",
            "tests/test_phase4_parity.py",
            "tests/test_phase5_parity.py",
            "tests/benchmarks/baselines/",
            "tests/api-baselines/",
            "docs/_generated/project-facts.md",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if diff.stdout.strip():
        print("git diff --stat (relative to HEAD):")
        for line in diff.stdout.rstrip().splitlines():
            print(f"  {line}")
        print("\nIf the deltas are expected, stage the files and amend into the release commit:")
        print("  git add tests/test_phase4_parity.py tests/test_phase5_parity.py \\")
        print("          tests/benchmarks/baselines/ tests/api-baselines/ \\")
        print("          docs/_generated/project-facts.md")
        print("  git commit --amend --no-edit")
    else:
        print("All constants already current — no changes to stage.")


def main() -> int:
    try:
        run()
    except RefreshError as exc:
        print(f"\nrelease-constant refresh FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
