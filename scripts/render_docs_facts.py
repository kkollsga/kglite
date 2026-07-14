#!/usr/bin/env python3
"""Render documentation facts from their authoritative repository sources."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "_generated" / "project-facts.md"


def _section(text: str, name: str) -> str:
    match = re.search(rf"(?ms)^\[{re.escape(name)}\]\s*\n(.*?)(?=^\[|\Z)", text)
    if match is None:
        raise ValueError(f"missing [{name}] section")
    return match.group(1)


def _toml_array(section: str, key: str) -> list[str]:
    match = re.search(rf"(?ms)^{re.escape(key)}\s*=\s*(\[.*?\])", section)
    if match is None:
        raise ValueError(f"missing TOML array {key}")
    value = ast.literal_eval(match.group(1))
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} is not an array of strings")
    return value


def _workspace_facts() -> tuple[str, list[tuple[str, str]]]:
    result = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    workspace_ids = set(metadata["workspace_members"])
    packages = [package for package in metadata["packages"] if package["id"] in workspace_ids]
    versions = {package["version"] for package in packages}
    if len(versions) != 1:
        raise ValueError(f"workspace packages do not share one version: {sorted(versions)}")
    members = sorted(
        (
            package["name"],
            str(Path(package["manifest_path"]).relative_to(REPO_ROOT)),
        )
        for package in packages
    )
    return versions.pop(), members


def _python_facts() -> tuple[str, dict[str, list[str]], list[str]]:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = _section(text, "project")
    requires_match = re.search(r'^requires-python\s*=\s*"([^"]+)"', project, re.MULTILINE)
    if requires_match is None:
        raise ValueError("missing project.requires-python")
    classifiers = _toml_array(project, "classifiers")
    optional = _section(text, "project.optional-dependencies")
    extras = {
        match.group(1): ast.literal_eval(match.group(2))
        for match in re.finditer(r"(?m)^(\w+)\s*=\s*(\[[^\n]*\])", optional)
    }
    return requires_match.group(1), dict(sorted(extras.items())), classifiers


def _workflow_facts() -> tuple[list[str], list[str]]:
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    versions_match = re.search(r"python-version:\s*\[([^]]+)\]", ci)
    if versions_match is None:
        raise ValueError("CI Python matrix not found")
    python_versions = re.findall(r"\d+\.\d+", versions_match.group(1))

    wheels = (REPO_ROOT / ".github" / "workflows" / "build_wheels.yml").read_text(encoding="utf-8")
    wheel_targets = sorted(set(re.findall(r"(?m)^\s+(?:-\s+)?target:\s+([\w-]+)\s*$", wheels)))
    return python_versions, wheel_targets


def _engine_facts() -> tuple[list[str], int, int, str]:
    mode_path = REPO_ROOT / "crates" / "kglite" / "src" / "graph" / "storage" / "mode.rs"
    mode_source = mode_path.read_text(encoding="utf-8")
    as_str = re.search(
        r"pub fn as_str\(self\).*?match self \{(.*?)\n\s*\}",
        mode_source,
        re.DOTALL,
    )
    if as_str is None:
        raise ValueError("StorageMode::as_str implementation not found")
    modes = re.findall(r'Self::\w+\s*=>\s*"([^"]+)"', as_str.group(1))

    file_path = REPO_ROOT / "crates" / "kglite" / "src" / "graph" / "io" / "file.rs"
    file_source = file_path.read_text(encoding="utf-8")
    magic_versions = [int(value) for value in re.findall(r"const V(\d+)_MAGIC:", file_source)]
    core = re.search(r"CURRENT_CORE_DATA_VERSION:\s*u32\s*=\s*(\d+)", file_source)
    if not magic_versions or core is None:
        raise ValueError("persistence version constants not found")

    spatial = "crates/kglite/src/graph/languages/cypher/executor/spatial_join.rs"
    spatial_source = (REPO_ROOT / spatial).read_text(encoding="utf-8")
    if "RTree::<" not in spatial_source:
        raise ValueError("spatial join no longer constructs an RTree")
    return modes, max(magic_versions), int(core.group(1)), spatial


def _benchmark_facts() -> dict[str, Any]:
    path = REPO_ROOT / "tests" / "benchmarks" / "baselines" / "current.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    machine = data["machine_info"]
    commit = data["commit_info"]
    cpu = machine.get("cpu") or {}
    return {
        "captured": data["datetime"],
        "harness": data["version"],
        "commit": commit["id"],
        "dirty": commit["dirty"],
        "platform": f"{machine['system']} {machine['release']} {machine['machine']}",
        "cpu": cpu.get("brand_raw") or machine.get("processor") or "unknown",
        "python": f"{machine['python_implementation']} {machine['python_version']}",
        "count": len(data["benchmarks"]),
    }


def render() -> str:
    version, members = _workspace_facts()
    requires_python, extras, classifiers = _python_facts()
    ci_pythons, wheel_targets = _workflow_facts()
    modes, container_version, core_version, spatial_source = _engine_facts()
    benchmark = _benchmark_facts()

    lines = [
        "<!-- Generated by scripts/render_docs_facts.py; do not edit by hand. -->",
        "",
        "# Generated project facts",
        "",
        "Regenerate with `python scripts/render_docs_facts.py`. CI checks this file for drift.",
        "",
        "## Workspace",
        "",
        f"- Shared package version: `{version}`",
        "- Workspace crates:",
    ]
    lines.extend(f"  - `{name}` — `{manifest}`" for name, manifest in members)
    lines.extend(
        [
            "",
            "## Python distribution",
            "",
            f"- Declared Python floor: `{requires_python}`",
            f"- CI runtime matrix: {', '.join(f'`{item}`' for item in ci_pythons)}",
            "- Optional extras:",
        ]
    )
    lines.extend(
        f"  - `{name}`: {', '.join(f'`{requirement}`' for requirement in requirements)}"
        for name, requirements in extras.items()
    )
    lines.append("- Published classifiers:")
    lines.extend(f"  - `{classifier}`" for classifier in classifiers)
    lines.extend(
        [
            "- Wheel build targets:",
            *[f"  - `{target}`" for target in wheel_targets],
            "",
            "## Engine contracts",
            "",
            f"- User storage modes: {', '.join(f'`{mode}`' for mode in modes)}",
            f"- Snapshot container/core versions: RGF v{container_version} / core v{core_version}",
            f"- Spatial candidate index: per-query `rstar::RTree` in `{spatial_source}`",
            "- Disk publication pointer: `CURRENT` selects an immutable generation.",
            "",
            "## Current tracked benchmark capture",
            "",
            f"- Captured: `{benchmark['captured']}`",
            f"- Source commit: `{benchmark['commit']}` (dirty: `{str(benchmark['dirty']).lower()}`)",
            f"- Platform: `{benchmark['platform']}`",
            f"- CPU: `{benchmark['cpu']}`",
            f"- Python: `{benchmark['python']}`",
            f"- pytest-benchmark schema/plugin version: `{benchmark['harness']}`",
            f"- Recorded benchmarks: `{benchmark['count']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if output is stale")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    expected = render()
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if current != expected:
            shown = (
                str(args.output.relative_to(REPO_ROOT)) if args.output.is_relative_to(REPO_ROOT) else str(args.output)
            )
            print(
                f"{shown} is stale; run: python scripts/render_docs_facts.py",
                file=sys.stderr,
            )
            return 1
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
