#!/usr/bin/env python3
"""Validate, capture, and check the feature-complete Rust API contract."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "tests" / "api-baselines" / "rust-api-profiles.json"
VALID_CLASSIFICATIONS = {"profile-root", "public-api", "implementation-only"}
# Refresh fast-path stamp. Lives under target/ (gitignored, wiped by
# `cargo clean`) — losing it only costs one redundant full refresh.
STAMP_PATH = ROOT / "target" / "rust-api-profiles.stamp"


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported rust API profile manifest schema")
    return manifest


def cargo_features(package: str) -> set[str]:
    proc = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(proc.stdout)
    crate = next((item for item in metadata["packages"] if item["name"] == package), None)
    if crate is None:
        raise ValueError(f"workspace package {package!r} does not exist")
    return set(crate["features"])


def validate_manifest(manifest: dict[str, Any]) -> None:
    package = manifest.get("package")
    if not isinstance(package, str) or not package:
        raise ValueError("manifest package must be a non-empty string")
    for key in ("nightly", "cargo_public_api_version"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError(f"manifest {key} must be a non-empty string")

    classifications = manifest.get("feature_classifications")
    if not isinstance(classifications, dict):
        raise ValueError("feature_classifications must be an object")
    unknown_classes = set(classifications.values()) - VALID_CLASSIFICATIONS
    if unknown_classes:
        raise ValueError(f"unknown feature classifications: {sorted(unknown_classes)}")

    declared = cargo_features(package)
    classified = set(classifications)
    if declared != classified:
        missing = sorted(declared - classified)
        stale = sorted(classified - declared)
        raise ValueError(f"Cargo feature classification drift: missing={missing}, stale={stale}")
    if classifications.get("default") != "profile-root":
        raise ValueError("the Cargo default feature must be classified as profile-root")

    profiles = manifest.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("profiles must be a non-empty list")
    names: set[str] = set()
    baselines: set[str] = set()
    individually_gated: set[str] = set()
    all_features_profiles = 0
    for profile in profiles:
        name = profile.get("name")
        baseline = profile.get("baseline")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"invalid or duplicate profile name: {name!r}")
        if not isinstance(baseline, str) or not baseline or baseline in baselines:
            raise ValueError(f"invalid or duplicate baseline path: {baseline!r}")
        if not baseline.startswith("tests/api-baselines/rust/"):
            raise ValueError(f"profile {name}: baseline must live under tests/api-baselines/rust/")
        names.add(name)
        baselines.add(baseline)

        all_features = profile.get("all_features", False)
        features = profile.get("features", [])
        if not isinstance(all_features, bool) or not isinstance(features, list):
            raise ValueError(f"profile {name}: invalid all_features/features shape")
        if all_features:
            all_features_profiles += 1
            if name != "all-features":
                raise ValueError("the all-features contract must be named all-features")
            if features:
                raise ValueError(f"profile {name}: all_features cannot be combined with features")
        else:
            if any(feature not in declared or feature == "default" for feature in features):
                raise ValueError(f"profile {name}: contains an unknown or invalid feature")
            if not features:
                if name != "default":
                    raise ValueError("the empty-feature contract must be named default")
            elif len(features) == 1 and name == features[0]:
                individually_gated.add(features[0])
            else:
                raise ValueError(f"profile {name}: expected one same-named public API feature")

    public_features = {feature for feature, classification in classifications.items() if classification == "public-api"}
    if public_features != individually_gated:
        missing = sorted(public_features - individually_gated)
        extra = sorted(individually_gated - public_features)
        raise ValueError(f"individual public API profile drift: missing={missing}, extra={extra}")
    if "default" not in names or all_features_profiles != 1:
        raise ValueError("profiles must contain default and exactly one all-features contract")

    # Companion crates: full default-feature public-API snapshots of sibling
    # workspace crates whose *library* surface downstream repos build against
    # (e.g. kglite-mcp-server's WorkspaceGraphHooks seam, consumed by codingest-mcp).
    # Exempt from the feature-classification machinery — one baseline each.
    for companion in manifest.get("companion_packages", []):
        pkg = companion.get("package")
        baseline = companion.get("baseline")
        if not isinstance(pkg, str) or not pkg or pkg == manifest["package"]:
            raise ValueError(f"companion package invalid: {pkg!r}")
        cargo_features(pkg)  # raises if the workspace package does not exist
        if (
            not isinstance(baseline, str)
            or not baseline.startswith("tests/api-baselines/rust/")
            or baseline in baselines
        ):
            raise ValueError(f"companion {pkg}: invalid or duplicate baseline {baseline!r}")
        baselines.add(baseline)


def compute_source_digest(manifest: dict[str, Any]) -> str:
    """Digest everything that determines the captured API text: the manifest
    (pins, profiles, classifications), each profiled package's `Cargo.toml` +
    `src/**/*.rs`, and the current baseline files themselves (so a hand-edited
    or deleted baseline can never be skipped over).

    Deliberately excludes `Cargo.lock`: a dependency bump can in principle
    rename an external type that appears in a signature, but that is rare and
    the CI public-api job remains the authority — a wrong local skip surfaces
    there as an exact-baseline diff.
    """
    digest = hashlib.sha256()
    digest.update(MANIFEST_PATH.read_bytes())

    packages = [manifest["package"]] + [c["package"] for c in manifest.get("companion_packages", [])]
    for package in packages:
        crate_dir = ROOT / "crates" / package
        manifest_toml = crate_dir / "Cargo.toml"
        if not manifest_toml.exists():
            raise RuntimeError(f"cannot digest {package}: {manifest_toml} not found")
        digest.update(manifest_toml.read_bytes())
        for source in sorted((crate_dir / "src").rglob("*.rs")):
            digest.update(str(source.relative_to(ROOT)).encode())
            digest.update(source.read_bytes())

    baselines = [p["baseline"] for p in manifest["profiles"]]
    baselines += [c["baseline"] for c in manifest.get("companion_packages", [])]
    for baseline in sorted(baselines):
        path = ROOT / baseline
        digest.update(baseline.encode())
        digest.update(path.read_bytes() if path.exists() else b"<absent>")
    return digest.hexdigest()


def public_api_command(manifest: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    cmd = ["cargo", "public-api", "-p", profile.get("package", manifest["package"]), "-ss", "--no-default-features"]
    if profile.get("all_features", False):
        cmd.append("--all-features")
    elif profile.get("features"):
        cmd.extend(["--features", ",".join(profile["features"])])
    return cmd


def capture_profile(manifest: dict[str, Any], profile: dict[str, Any]) -> str:
    proc = subprocess.run(
        public_api_command(manifest, profile),
        cwd=ROOT,
        env={**os.environ, "RUSTUP_TOOLCHAIN": manifest["nightly"]},
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        details = proc.stderr.strip() or proc.stdout.strip() or "no diagnostic output"
        raise RuntimeError(f"profile {profile['name']} capture failed:\n{details}")
    return proc.stdout


def run_profiles(manifest: dict[str, Any], *, check: bool) -> int:
    if shutil.which("cargo-public-api") is None:
        raise RuntimeError(
            "cargo-public-api is not installed; install the version printed by "
            "`python scripts/rust_api_profiles.py value cargo_public_api_version`"
        )
    version = subprocess.run(
        ["cargo", "public-api", "--version"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_version = f"cargo-public-api {manifest['cargo_public_api_version']}"
    if version != expected_version:
        raise RuntimeError(f"expected {expected_version!r}, found {version!r}")

    failed = False
    captures: dict[str, str] = {}
    for profile in manifest["profiles"]:
        name = profile["name"]
        print(f"{name}: capturing", flush=True)
        current = capture_profile(manifest, profile)
        captures[name] = current
        baseline = ROOT / profile["baseline"]
        if check:
            expected = baseline.read_text(encoding="utf-8") if baseline.exists() else ""
            if expected != current:
                failed = True
                diff = difflib.unified_diff(
                    expected.splitlines(keepends=True),
                    current.splitlines(keepends=True),
                    fromfile=profile["baseline"],
                    tofile=f"current:{name}",
                )
                sys.stdout.writelines(diff)
            else:
                print(f"{name}: exact API match")
        else:
            baseline.parent.mkdir(parents=True, exist_ok=True)
            baseline.write_text(current, encoding="utf-8")
            print(f"{name}: wrote {profile['baseline']}")

    for companion in manifest.get("companion_packages", []):
        pkg = companion["package"]
        profile = {"name": pkg, "package": pkg, "features": companion.get("features", [])}
        print(f"{pkg}: capturing (companion)", flush=True)
        current = capture_profile(manifest, profile)
        baseline = ROOT / companion["baseline"]
        if check:
            expected = baseline.read_text(encoding="utf-8") if baseline.exists() else ""
            if expected != current:
                failed = True
                diff = difflib.unified_diff(
                    expected.splitlines(keepends=True),
                    current.splitlines(keepends=True),
                    fromfile=companion["baseline"],
                    tofile=f"current:{pkg}",
                )
                sys.stdout.writelines(diff)
            else:
                print(f"{pkg}: exact API match")
        else:
            baseline.parent.mkdir(parents=True, exist_ok=True)
            baseline.write_text(current, encoding="utf-8")
            print(f"{pkg}: wrote {companion['baseline']}")

    default_surface = captures["default"]
    for feature, classification in manifest["feature_classifications"].items():
        if classification != "implementation-only":
            continue
        profile = {"name": feature, "features": [feature]}
        print(f"{feature}: verifying implementation-only surface", flush=True)
        current = capture_profile(manifest, profile)
        if current != default_surface:
            failed = True
            diff = difflib.unified_diff(
                default_surface.splitlines(keepends=True),
                current.splitlines(keepends=True),
                fromfile="current:default",
                tofile=f"current:{feature}",
            )
            sys.stdout.writelines(diff)
        else:
            print(f"{feature}: no public API delta")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="check manifest and Cargo feature coverage")
    refresh_parser = subparsers.add_parser("refresh", help="regenerate every committed profile baseline")
    refresh_parser.add_argument(
        "--skip-if-unchanged",
        action="store_true",
        help="Skip the (expensive) rustdoc captures when profiled sources, manifest, "
        "and baselines are byte-identical to the last successful refresh. "
        "CI's exact-baseline check remains the authority.",
    )
    subparsers.add_parser("check", help="compare every profile with its exact baseline")
    value_parser = subparsers.add_parser("value", help="print one top-level manifest value")
    value_parser.add_argument("key", choices=["nightly", "cargo_public_api_version"])
    args = parser.parse_args()

    try:
        manifest = load_manifest()
        validate_manifest(manifest)
        if args.command == "value":
            print(manifest[args.key])
            return 0
        if args.command == "validate":
            print(
                f"Rust API profiles valid: {len(manifest['profiles'])} profiles, "
                f"{len(manifest['feature_classifications'])} classified Cargo features"
            )
            return 0
        if args.command == "refresh" and args.skip_if_unchanged:
            source_digest = compute_source_digest(manifest)
            if STAMP_PATH.exists() and STAMP_PATH.read_text(encoding="utf-8").strip() == source_digest:
                print("rust API baselines: sources/manifest/baselines unchanged since last refresh — skipped")
                return 0
        result = run_profiles(manifest, check=args.command == "check")
        if args.command == "refresh" and result == 0:
            STAMP_PATH.parent.mkdir(parents=True, exist_ok=True)
            STAMP_PATH.write_text(compute_source_digest(manifest) + "\n", encoding="utf-8")
        return result
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"rust API profile error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
