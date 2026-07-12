#!/usr/bin/env python3
"""Enforce KGLite's reviewed dependency-license and package-notice policy."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import tomllib

ROOT = Path(__file__).resolve().parent.parent
POLICY = ROOT / "tests" / "api-baselines" / "dependency-licenses.json"
PUBLISHED_CRATES = ("kglite", "kglite-c", "kglite-cli", "kglite-mcp-server", "kglite-bolt-server")
FORBIDDEN = ("AGPL", "GPL", "SSPL", "BUSL", "Commons-Clause", "NonCommercial")


def main() -> int:
    policy = json.loads(POLICY.read_text())
    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--format-version", "1", "--locked", "--all-features", "--offline"],
            cwd=ROOT,
            text=True,
        )
    )
    allowed = set(policy["allowed_expressions"])
    review_required = set(policy["review_required_expressions"])
    reviewed = {(name, version, license) for name, version, license, _scope in policy["reviewed_packages"]}
    errors: list[str] = []

    for package in metadata["packages"]:
        license_expr = package.get("license")
        identity = (package["name"], package["version"], license_expr)
        if not license_expr:
            errors.append(f"{package['name']} {package['version']}: missing license metadata")
            continue
        if any(marker in license_expr for marker in FORBIDDEN):
            # An expression with an independently selectable MIT branch remains
            # usable under MIT; expressions that require copyleft do not.
            if "MIT OR" not in license_expr and "OR MIT" not in license_expr:
                errors.append(f"{package['name']} {package['version']}: forbidden {license_expr}")
        if license_expr not in allowed:
            errors.append(f"{package['name']} {package['version']}: unreviewed expression {license_expr}")
        if license_expr in review_required and identity not in reviewed:
            errors.append(f"{package['name']} {package['version']}: package requires explicit review ({license_expr})")

    present_reviewed = {
        (package["name"], package["version"], package.get("license"))
        for package in metadata["packages"]
        if package.get("license") in review_required
    }
    stale = reviewed - present_reviewed
    if stale:
        errors.append(f"stale reviewed package entries: {sorted(stale)!r}")

    root_license = (ROOT / "LICENSE").read_bytes()
    for crate in PUBLISHED_CRATES:
        crate_license = ROOT / "crates" / crate / "LICENSE"
        if not crate_license.is_file() or crate_license.read_bytes() != root_license:
            errors.append(f"crates/{crate}/LICENSE is missing or differs from root LICENSE")

    for pyproject in (ROOT / "pyproject.toml", ROOT / "crates" / "kglite-cli" / "pyproject.toml"):
        project = tomllib.loads(pyproject.read_text())["project"]
        if project.get("license") != "MIT":
            errors.append(f"{pyproject.relative_to(ROOT)}: project.license must be SPDX 'MIT'")
        if "LICENSE" not in project.get("license-files", []):
            errors.append(f"{pyproject.relative_to(ROOT)}: project.license-files must include LICENSE")

    if errors:
        print("dependency license policy failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"dependency license policy: OK ({len(metadata['packages'])} packages, all features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
