#!/usr/bin/env python3
"""Enforce KGLite's reviewed dependency-license and package-notice policy."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
POLICY = ROOT / "tests" / "api-baselines" / "dependency-licenses.json"
PUBLISHED_CRATES = ("kglite", "kglite-c", "kglite-cli", "kglite-mcp-server", "kglite-bolt-server")
FORBIDDEN = ("AGPL", "GPL", "SSPL", "BUSL", "Commons-Clause", "NonCommercial")


def project_literal(path: Path, key: str) -> str | None:
    """Read one simple literal from [project], rejecting ambiguous TOML."""
    in_project = False
    found: str | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("["):
            in_project = line == "[project]"
            continue
        if not in_project or line.startswith("#") or "=" not in line:
            continue
        candidate, value = line.split("=", 1)
        if candidate.strip() != key:
            continue
        if found is not None:
            return None
        found = value.strip()
    return found


def main() -> int:
    policy = json.loads(POLICY.read_text())
    # Prefer offline resolution (fast, deterministic), but fall back to a
    # networked call: after a Cargo.lock change the CI cargo cache misses and
    # offline metadata fails to resolve even long-standing packages.
    base_cmd = ["cargo", "metadata", "--format-version", "1", "--locked", "--all-features"]
    try:
        raw = subprocess.check_output([*base_cmd, "--offline"], cwd=ROOT, text=True)
    except subprocess.CalledProcessError:
        raw = subprocess.check_output(base_cmd, cwd=ROOT, text=True)
    metadata = json.loads(raw)
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
        if project_literal(pyproject, "license") != '"MIT"':
            errors.append(f"{pyproject.relative_to(ROOT)}: project.license must be SPDX 'MIT'")
        if project_literal(pyproject, "license-files") != '["LICENSE"]':
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
