#!/usr/bin/env python3
"""Reject externally sourced suite artifacts from KGLite's Cypher contract."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTRACT_ROOT = ROOT / "tests" / "cypher_contract"
RUNNER = ROOT / "tests" / "test_cypher_clean_room_contract.py"
FORBIDDEN_SUFFIXES = {".feature", ".gherkin"}
FORBIDDEN_TEXT = (
    "Apache License",
    "Feature:",
    "Scenario:",
    "Scenario Outline:",
    "Examples:",
    "openCypher TCK",
    "Technology Compatibility Kit",
)
ALLOWED_RUNNER_IMPORTS = {"__future__", "json", "pathlib", "pytest", "kglite", "scripts"}


def validate() -> list[str]:
    errors: list[str] = []
    files = [path for path in CONTRACT_ROOT.rglob("*") if path.is_file()]
    for path in files:
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden external-suite artifact type: {path.relative_to(ROOT)}")
        text = path.read_text(errors="replace")
        for marker in FORBIDDEN_TEXT:
            if marker in text:
                errors.append(f"forbidden external-suite marker {marker!r}: {path.relative_to(ROOT)}")

    manifest_path = CONTRACT_ROOT / "cases.json"
    manifest = json.loads(manifest_path.read_text())
    authorship = manifest.get("authorship", {})
    if manifest.get("license") != "MIT":
        errors.append("contract manifest must be MIT licensed")
    if authorship.get("method") != "independent_behavioral_design":
        errors.append("contract manifest must declare independent behavioral design")
    if authorship.get("upstream_artifacts_used") is not False:
        errors.append("contract manifest must declare that no upstream artifacts were used")

    cases = manifest.get("cases", [])
    ids = [case.get("id") for case in cases]
    if len(ids) != len(set(ids)):
        errors.append("contract case IDs must be unique")
    for index, case in enumerate(cases):
        for field in ("id", "category", "requirement", "query", "expected"):
            if field not in case:
                errors.append(f"contract case {index} is missing {field!r}")

    tree = ast.parse(RUNNER.read_text(), filename=str(RUNNER))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = {alias.name.split(".", 1)[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported = {(node.module or "").split(".", 1)[0]}
        else:
            continue
        unexpected = imported - ALLOWED_RUNNER_IMPORTS
        if unexpected:
            errors.append(f"contract runner imports unexpected dependencies: {sorted(unexpected)}")
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("Cypher clean-room contract check failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Cypher clean-room contract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
