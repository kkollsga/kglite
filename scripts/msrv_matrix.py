#!/usr/bin/env python3
"""Derive the MSRV verification matrix from the workspace manifests.

The `msrv` CI job builds every workspace member on the exact Rust version
that member declares as its `rust-version`. The matrix is *computed here*
rather than hard-coded in the workflow, so the declared contract and the
job that verifies it cannot drift apart -- which is the whole failure mode
a declared-but-unchecked MSRV creates.

Two modes:

    python scripts/msrv_matrix.py            # human-readable table
    python scripts/msrv_matrix.py --github   # `matrix=<json>` for GITHUB_OUTPUT

The script also *fails* if any member is missing a `rust-version`, so a new
crate cannot join the workspace without declaring its floor.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# `tomllib` is stdlib from 3.11, but ruff's `target-version = "py310"`
# classifies it as third-party, hence the separate import section. The CI
# job that runs this script pins Python 3.12.
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def workspace_members(root: pathlib.Path) -> list[str]:
    manifest = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    return list(manifest["workspace"]["members"])


def workspace_rust_version(root: pathlib.Path) -> str | None:
    manifest = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    return manifest.get("workspace", {}).get("package", {}).get("rust-version")


def member_rust_version(member_dir: pathlib.Path, inherited: str | None) -> tuple[str, str | None]:
    """Return `(package_name, effective_rust_version)` for one member."""
    manifest = tomllib.loads((member_dir / "Cargo.toml").read_text(encoding="utf-8"))
    package = manifest["package"]
    declared = package.get("rust-version")
    if isinstance(declared, dict):
        # `rust-version.workspace = true`
        declared = inherited if declared.get("workspace") else None
    return package["name"], declared


def collect(root: pathlib.Path) -> dict[str, list[str]]:
    """Group member package names by their effective `rust-version`."""
    inherited = workspace_rust_version(root)
    groups: dict[str, list[str]] = {}
    missing: list[str] = []
    for member in workspace_members(root):
        name, version = member_rust_version(root / member, inherited)
        if version is None:
            missing.append(name)
            continue
        groups.setdefault(version, []).append(name)
    if missing:
        joined = ", ".join(sorted(missing))
        raise SystemExit(
            f"error: workspace member(s) without a `rust-version`: {joined}\n"
            "Every published crate must declare its minimum supported Rust "
            "version, either directly or with `rust-version.workspace = true`."
        )
    return groups


def as_matrix(groups: dict[str, list[str]]) -> dict[str, list[dict[str, str]]]:
    include = [
        {
            "toolchain": version,
            # `-p a -p b` -- one cargo invocation per toolchain group.
            "packages": " ".join(f"-p {name}" for name in sorted(names)),
            # Used only for a readable job name in the Actions UI.
            "label": ", ".join(sorted(names)),
            # The group that owns the engine also checks the engine's
            # pure-Rust optional loaders (`okf`, `rdf`) on the same floor,
            # since a consumer may enable them. `fastembed` is excluded on
            # purpose -- it pulls a ~200 MB ONNX runtime, the same reason
            # the `kglite-c` job skips it.
            "core": "true" if "kglite" in names else "false",
        }
        for version, names in sorted(groups.items())
    ]
    return {"include": include}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--github",
        action="store_true",
        help="emit `matrix=<json>` suitable for appending to $GITHUB_OUTPUT",
    )
    args = parser.parse_args()

    groups = collect(REPO_ROOT)
    matrix = as_matrix(groups)

    if args.github:
        print(f"matrix={json.dumps(matrix, separators=(',', ':'))}")
        return 0

    for entry in matrix["include"]:
        print(f"{entry['toolchain']:>10}  {entry['label']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
