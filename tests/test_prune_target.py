"""Contracts for bounded cleanup of Cargo's symlinked target cache."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_TAG_SIGNATURE = "Signature: 8a477f597d28d172789f06886806bc55"

#: A `PRUNE_TARGET_GB` no real fixture can reach, so only the free-space branch
#: can fire the prune in the tests that set it.
UNREACHABLE_SIZE_GB = "999999"


def _make(worktree: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", "-f", str(REPO_ROOT / "Makefile"), *args],
        cwd=worktree,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_prune_target_cleans_symlink_referent(tmp_path: Path) -> None:
    """The prune must clean the cache, not merely unlink ``target``."""
    worktree = tmp_path / "worktree"
    cache = tmp_path / "cargo-cache"
    fake_bin = tmp_path / "bin"
    worktree.mkdir()
    (cache / "debug").mkdir(parents=True)
    fake_bin.mkdir()

    sentinel = cache / "debug" / "sentinel-artifact"
    sentinel.write_text("generated build artifact", encoding="utf-8")
    (worktree / "target").symlink_to(cache, target_is_directory=True)

    fake_cargo = fake_bin / "cargo"
    fake_cargo.write_text(
        """#!/bin/sh
set -eu
[ "$1" = clean ]
shift
if [ "${1:-}" = --target-dir ]; then
    target_dir=$2
    grep -qx 'Signature: 8a477f597d28d172789f06886806bc55' "$target_dir/CACHEDIR.TAG"
    rm -rf "$target_dir"
else
    rm -rf target
fi
""",
        encoding="utf-8",
    )
    fake_cargo.chmod(0o755)

    env = os.environ.copy()
    env.pop("CARGO_TARGET_DIR", None)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    result = subprocess.run(
        [
            "make",
            "-f",
            str(REPO_ROOT / "Makefile"),
            "prune-target",
            "PRUNE_TARGET_GB=0",
        ],
        cwd=worktree,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (worktree / "target").is_symlink()
    assert (worktree / "target").resolve() == cache.resolve()
    assert not sentinel.exists()
    assert (cache / "CACHEDIR.TAG").read_text(encoding="utf-8").splitlines()[0] == (CACHE_TAG_SIGNATURE)


def _cache_worktree(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    """A worktree whose ``target`` symlinks a cache holding one sentinel artifact."""
    worktree = tmp_path / "worktree"
    cache = tmp_path / "cargo-cache"
    fake_bin = tmp_path / "bin"
    worktree.mkdir()
    (cache / "debug").mkdir(parents=True)
    fake_bin.mkdir()
    (cache / "debug" / "sentinel-artifact").write_text("generated build artifact", encoding="utf-8")
    (worktree / "target").symlink_to(cache, target_is_directory=True)

    fake_cargo = fake_bin / "cargo"
    fake_cargo.write_text('#!/bin/sh\nset -eu\n[ "$1" = clean ]\nrm -rf "$3"\n', encoding="utf-8")
    fake_cargo.chmod(0o755)

    env = os.environ.copy()
    env.pop("CARGO_TARGET_DIR", None)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    return worktree, cache / "debug" / "sentinel-artifact", env


def test_low_free_space_fires_the_prune_even_below_the_size_gate(tmp_path: Path) -> None:
    """`du -sg` undercounts APFS clone-shared artifacts ~2x (48 GB metered, 95.1
    GiB actually freed), so the size branch alone let the volume run to zero
    twice. Free space is the meter that cannot be fooled by cloning."""
    worktree, sentinel, env = _cache_worktree(tmp_path)

    result = _make(worktree, env, "prune-target", f"PRUNE_TARGET_GB={UNREACHABLE_SIZE_GB}", "FREE_GB=1")

    assert result.returncode == 0, result.stdout + result.stderr
    assert not sentinel.exists(), result.stdout
    assert (worktree / "target").is_symlink()


def test_ample_free_space_under_the_size_gate_does_not_prune(tmp_path: Path) -> None:
    """Non-vacuity: the injected value decides, and a healthy volume is left alone."""
    worktree, sentinel, env = _cache_worktree(tmp_path)

    result = _make(worktree, env, "prune-target", f"PRUNE_TARGET_GB={UNREACHABLE_SIZE_GB}", "FREE_GB=500")

    assert result.returncode == 0, result.stdout + result.stderr
    assert sentinel.exists(), result.stdout


def test_check_free_space_refuses_a_build_below_the_fail_threshold(tmp_path: Path) -> None:
    worktree, _, env = _cache_worktree(tmp_path)

    result = _make(worktree, env, "check-free-space", "FREE_GB=1")

    assert result.returncode != 0, result.stdout + result.stderr
    assert "1 GB" in result.stdout + result.stderr


def test_check_free_space_passes_on_an_ample_volume(tmp_path: Path) -> None:
    worktree, _, env = _cache_worktree(tmp_path)

    result = _make(worktree, env, "check-free-space", "FREE_GB=500")

    assert result.returncode == 0, result.stdout + result.stderr
