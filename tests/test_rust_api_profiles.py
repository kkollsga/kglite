from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "rust_api_profiles.py"
SPEC = importlib.util.spec_from_file_location("rust_api_profiles", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
rust_api_profiles = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rust_api_profiles)


def test_manifest_classifies_every_cargo_feature_and_covers_public_profiles():
    manifest = rust_api_profiles.load_manifest()
    rust_api_profiles.validate_manifest(manifest)


def test_profiles_always_disable_implicit_defaults():
    manifest = rust_api_profiles.load_manifest()
    for profile in manifest["profiles"]:
        command = rust_api_profiles.public_api_command(manifest, profile)
        assert "--no-default-features" in command


def test_duplicate_profile_names_are_rejected(monkeypatch):
    manifest = rust_api_profiles.load_manifest()
    manifest["profiles"][1]["name"] = manifest["profiles"][0]["name"]
    monkeypatch.setattr(
        rust_api_profiles,
        "cargo_features",
        lambda _package: set(manifest["feature_classifications"]),
    )
    with pytest.raises(ValueError, match="duplicate profile name"):
        rust_api_profiles.validate_manifest(manifest)


def test_parallel_bz2_is_explicitly_implementation_only():
    manifest = rust_api_profiles.load_manifest()
    assert manifest["feature_classifications"]["parallel-bz2"] == "implementation-only"


def test_each_individual_public_surface_is_present_with_all_features():
    manifest = rust_api_profiles.load_manifest()
    profiles = {profile["name"]: profile for profile in manifest["profiles"]}
    all_surface = set((ROOT / profiles["all-features"]["baseline"]).read_text().splitlines())
    for feature, classification in manifest["feature_classifications"].items():
        if classification == "public-api":
            feature_surface = set((ROOT / profiles[feature]["baseline"]).read_text().splitlines())
            assert feature_surface <= all_surface
