#!/usr/bin/env python3
"""Smoke-test an installed native wheel and its bundled MCP launcher."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
import subprocess
import sys

import kglite


def main() -> int:
    distribution = metadata.distribution("kglite")
    entry_points = {entry.name: entry.value for entry in distribution.entry_points if entry.group == "console_scripts"}
    expected = {
        "kglite": "kglite.cli:main",
        "kglite-mcp-server": "kglite.mcp_server:main",
    }
    for name, value in expected.items():
        actual = entry_points.get(name)
        if actual != value:
            raise RuntimeError(f"installed {name} entry point is {actual!r}, expected {value!r}")

    package_path = Path(kglite.__file__).resolve()
    if "site-packages" not in package_path.parts:
        raise RuntimeError(f"kglite imported outside the smoke venv: {package_path}")

    suffix = ".exe" if sys.platform == "win32" else ""
    # Keep the venv path rather than resolving its interpreter symlink back to
    # the base Python installation, where console launchers do not live.
    bin_dir = Path(sys.executable).parent
    cli_launcher = bin_dir / f"kglite{suffix}"
    mcp_launcher = bin_dir / f"kglite-mcp-server{suffix}"
    for launcher in (cli_launcher, mcp_launcher):
        if not launcher.is_file():
            raise RuntimeError(f"installed launcher is missing: {launcher}")

    cli_result = subprocess.run(
        [str(cli_launcher), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    cli_output = cli_result.stdout + cli_result.stderr
    if cli_result.returncode != 0 or "query" not in cli_output or "skill" not in cli_output:
        raise RuntimeError(f"installed CLI launcher failed ({cli_result.returncode}):\n{cli_output}")

    mcp_result = subprocess.run(
        [str(mcp_launcher), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    mcp_output = mcp_result.stdout + mcp_result.stderr
    if mcp_result.returncode != 0 or "--selftest" not in mcp_output:
        raise RuntimeError(f"installed MCP launcher failed ({mcp_result.returncode}):\n{mcp_output}")

    print(f"installed wheel OK: {package_path}; launchers={cli_launcher},{mcp_launcher}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
