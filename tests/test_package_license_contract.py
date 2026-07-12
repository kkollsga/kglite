"""Distribution artifacts retain KGLite's MIT notice and SPDX metadata."""

from __future__ import annotations

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
PUBLISHED_CRATES = ("kglite", "kglite-c", "kglite-cli", "kglite-mcp-server", "kglite-bolt-server")


def test_every_crate_package_includes_license():
    for crate in PUBLISHED_CRATES:
        output = subprocess.check_output(
            ["cargo", "package", "-p", crate, "--allow-dirty", "--no-verify", "--list"],
            cwd=ROOT,
            text=True,
        )
        assert "LICENSE" in output.splitlines(), f"{crate} package omits LICENSE"


def test_dependency_license_policy_gate():
    subprocess.run(["python", "scripts/check_dependency_licenses.py"], cwd=ROOT, check=True)
