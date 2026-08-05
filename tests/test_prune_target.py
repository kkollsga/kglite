"""Contracts for bounded cleanup of Cargo's symlinked target cache."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_TAG_SIGNATURE = "Signature: 8a477f597d28d172789f06886806bc55"


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
