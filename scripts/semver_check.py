#!/usr/bin/env python3
"""Run cargo-semver-checks for the kglite crate against the last published release.

Findings are evidence for the CHANGELOG and downstream notes, never a gate on
the version number (this project ships documented breaking changes in patch
bumps). The tool *not running* is a different matter: a baseline lookup that
came back empty, or a cargo-semver-checks error with no result summary, used
to exit 0 behind `|| true` and read as "no findings". This wrapper exits 0
only when the tool produced a `Summary` line.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
CRATE_URL = "https://crates.io/api/v1/crates/kglite"
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
SUMMARY_RE = re.compile(r"^\s*Summary\b", re.MULTILINE)
NOT_RUN = "semver-check DID NOT RUN"


class SemverCheckError(RuntimeError):
    """The check could not produce a result."""


def parse_baseline(body: str) -> str:
    """The `max_stable_version` of a crates.io crate document, validated."""
    try:
        version = json.loads(body)["crate"]["max_stable_version"]
    except (ValueError, KeyError, TypeError) as exc:
        raise SemverCheckError(f"baseline version lookup returned unusable data ({exc!r})") from exc
    if not isinstance(version, str) or not VERSION_RE.match(version):
        raise SemverCheckError(f"baseline version lookup returned {version!r}, not X.Y.Z")
    return version


def fetch_baseline() -> str:
    request = urllib.request.Request(CRATE_URL, headers={"User-Agent": "kglite-semver-check"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return parse_baseline(response.read().decode())
    except OSError as exc:
        raise SemverCheckError(f"baseline version lookup failed ({exc})") from exc


def ran(output: str) -> bool:
    """Whether cargo-semver-checks reached its result summary."""
    return SUMMARY_RE.search(output) is not None


#: Left out of the checked feature set. `fastembed`: cargo-semver-checks builds
#: the published baseline without our lockfile, so it gets the newest fastembed;
#: fastembed 7.8.0 broke that build for kglite <= 0.21.7 (its tokenizers regex
#: backend became opt-in), and the feature also downloads the ONNX runtime. Its
#: surface is the one `FastEmbedAdapter` type. `test-seam`: test builds only.
EXCLUDED_FEATURES = {"default", "fastembed", "test-seam"}


def checked_features() -> list[str]:
    meta = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    package = next(p for p in json.loads(meta.stdout)["packages"] if p["name"] == "kglite")
    return sorted(set(package["features"]) - EXCLUDED_FEATURES)


def run_tool(baseline: str) -> tuple[int, str]:
    features = ",".join(checked_features())
    proc = subprocess.run(
        [
            "cargo",
            "semver-checks",
            "check-release",
            "-p",
            "kglite",
            "--baseline-version",
            baseline,
            "--only-explicit-features",
            "--features",
            features,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    try:
        baseline = fetch_baseline()
        print(f"semver baseline: kglite {baseline}")
        code, output = run_tool(baseline)
    except SemverCheckError as exc:
        print(f"{NOT_RUN}: {exc}", file=sys.stderr)
        return 1
    print(output, end="" if output.endswith("\n") else "\n")
    if not ran(output):
        print(f"{NOT_RUN}: cargo-semver-checks exited {code} without a Summary line", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
