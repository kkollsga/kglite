#!/usr/bin/env python3
"""Refresh the captured constants that drift across releases.

Three captured values are version-coupled and silently rot if nobody
updates them at release time:

  1. ``tests/test_phase4_parity.py::GOLDEN_V3_DIGEST`` — embeds the
     version string in the ``.kgl`` header. Every release shifts the
     digest even when the format itself is unchanged.

  2. ``tests/test_phase5_parity.py::test_binary_size_regression``
     baseline — the release-built ``libkglite`` size, +10% over baseline.

  3. ``tests/benchmarks/baselines/<version>.json`` — pytest-benchmark
     JSON for the 11 tracked benchmarks. ``current.json`` is a copy.

This script reads ``Cargo.toml`` for the version, then refreshes all
three. Idempotent: running it twice in a row produces no diff.

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

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE4_TEST = REPO_ROOT / "tests" / "test_phase4_parity.py"
PHASE5_TEST = REPO_ROOT / "tests" / "test_phase5_parity.py"
BASELINES_DIR = REPO_ROOT / "tests" / "benchmarks" / "baselines"


def read_version() -> str:
    """Pull ``version = "X.Y.Z"`` from the wheel crate's ``Cargo.toml``.

    Pre-G.4 the root ``Cargo.toml`` held the project version (line 3).
    Post-G.4 the workspace root is virtual; the wheel version now lives
    in ``crates/kglite-py/Cargo.toml`` (read by maturin) and the core
    crate version in ``crates/kglite/Cargo.toml``. Both should match in
    a release commit. We read the wheel-crate value because the
    captured constants (.kgl header, binary size, benchmark baselines)
    all describe the wheel artifact.
    """
    text = (REPO_ROOT / "crates" / "kglite-py" / "Cargo.toml").read_text()
    m = re.search(r'^\s*version\s*=\s*"([^"]+)"\s*$', text, re.MULTILINE)
    if not m:
        sys.exit("crates/kglite-py/Cargo.toml: no version found")
    return m.group(1)


def version_slug(version: str) -> str:
    """0.9.52 → '0_9_52' (the convention used by existing baselines)."""
    return version.replace(".", "_")


def find_release_dylib() -> Path | None:
    """Locate the release-built kglite library; .dylib on macOS, .so on Linux."""
    for cand in (
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
    text = PHASE4_TEST.read_text()

    cur_match = re.search(r'^(GOLDEN_V3_DIGEST = )"([0-9a-f]{64})"', text, re.MULTILINE)
    if cur_match is None:
        return False, "tests/test_phase4_parity.py: GOLDEN_V3_DIGEST line not found — refusing to edit"

    cur_digest = cur_match.group(2)
    if cur_digest == new_digest:
        return False, f"GOLDEN_V3_DIGEST already current ({new_digest[:12]}…)"

    # Demote the current golden into ACCEPTABLE_DIGESTS if it isn't already there.
    if cur_digest not in text:
        # extremely defensive; this should always be true since we just matched it
        return False, "GOLDEN_V3_DIGEST value vanished mid-edit — refusing to edit"

    if cur_digest in text.split("ACCEPTABLE_DIGESTS")[1] if "ACCEPTABLE_DIGESTS" in text else False:
        # Already in the allowlist; only need to update the primary.
        pass
    else:
        # Append into the allowlist before the closing brace. Find the
        # last entry inside ACCEPTABLE_DIGESTS and insert after it.
        marker = re.search(r"(    )\}\s*\n\)\s*\n", text)
        if marker is None:
            return False, "ACCEPTABLE_DIGESTS closing brace not found — refusing to edit"
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

    PHASE4_TEST.write_text(text)
    return True, f"GOLDEN_V3_DIGEST → {new_digest[:12]}… (prior {cur_digest[:12]}… demoted to ACCEPTABLE_DIGESTS)"


# ── 2. Binary-size baseline ────────────────────────────────────────────


def refresh_binary_size(version: str, current_size: int) -> tuple[bool, str]:
    """Update the `baseline = NNNN` literal + the docstring."""
    text = PHASE5_TEST.read_text()

    bl_match = re.search(r"^(\s*)baseline\s*=\s*([0-9_]+)\s*#\s*([^\n]+)\n", text, re.MULTILINE)
    if bl_match is None:
        return False, "tests/test_phase5_parity.py: baseline line not found"
    indent = bl_match.group(1)
    cur_baseline = int(bl_match.group(2).replace("_", ""))

    if cur_baseline == current_size:
        return False, f"binary-size baseline already current ({current_size:,} bytes)"

    # Replace the baseline line.
    formatted = f"{current_size:_}".replace("_", "_")  # "12_345_678" style
    new_line = f"{indent}baseline = {formatted}  # {version} baseline\n"
    text = text[: bl_match.start()] + new_line + text[bl_match.end() :]

    # Best-effort: drop a marker into the docstring's "Baseline history:"
    # block so the growth narrative gains an entry. We don't try to
    # rewrite the whole prose — that's the maintainer's job; we just
    # leave a TODO so they don't forget.
    todo_marker = (
        f"\n      - {version}:       {current_size:,} bytes "
        f"(≈{current_size / (1024 * 1024):.1f} MB). "
        "TODO: describe what grew since the prior baseline.\n"
    )
    text = re.sub(
        r"(      - 0\.9\.52:\s+35,925,104 bytes \(≈34\.3 MB\)\.[^\n]*\n(?:[^-][^\n]*\n)+)",
        lambda m: m.group(1) + todo_marker if version != "0.9.52" else m.group(1),
        text,
        count=1,
    )

    # Update the in-message "+10% over X baseline" string to reference
    # the new version.
    text = re.sub(
        r"\(\+10% over [0-9.]+ baseline \{baseline:,\}\)",
        f"(+10% over {version} baseline {{baseline:,}})",
        text,
    )

    PHASE5_TEST.write_text(text)
    return True, f"binary-size baseline {cur_baseline:,} → {current_size:,} bytes"


# ── 3. Perf baseline ───────────────────────────────────────────────────


def refresh_perf_baseline(version: str) -> tuple[bool, str]:
    """Capture pytest-benchmark JSON for the 11 tracked core benchmarks
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
        return False, f"perf baseline {target.name} already present (delete it to force re-capture)"

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
            return False, f"benchmark run failed:\n{proc.stdout}\n{proc.stderr}"
        data = json.loads(tmp_json.read_text())

    # Strip per-iteration `data` — gates need aggregates only; carrying
    # the full series bloats commits to ~30 MB per release.
    for b in data["benchmarks"]:
        b["stats"].pop("data", None)

    target.write_text(json.dumps(data, indent=2))
    shutil.copyfile(target, current)
    return True, f"perf baseline written to {target.relative_to(REPO_ROOT)} (also copied to current.json)"


# ── orchestration ──────────────────────────────────────────────────────


def main() -> int:
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
        print("   SKIP   : no target/release/libkglite.{dylib,so} — run `cargo build --release` first.\n")
    else:
        size = dylib.stat().st_size
        changed, msg = refresh_binary_size(version, size)
        print(f"   {'CHANGED' if changed else 'no-op '}: {msg}\n")

    # 3. perf baseline
    if args.skip_benchmarks:
        print("3. perf baseline — skipped (--skip-benchmarks).\n")
    else:
        print("3. perf baseline (running 11 tracked benchmarks, ~15s)…")
        changed, msg = refresh_perf_baseline(version)
        print(f"   {'CHANGED' if changed else 'no-op '}: {msg}\n")

    # Pretty diff summary.
    diff = subprocess.run(
        [
            "git",
            "diff",
            "--stat",
            "--",
            "tests/test_phase4_parity.py",
            "tests/test_phase5_parity.py",
            "tests/benchmarks/baselines/",
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
        print("          tests/benchmarks/baselines/")
        print("  git commit --amend --no-edit")
    else:
        print("All constants already current — no changes to stage.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
