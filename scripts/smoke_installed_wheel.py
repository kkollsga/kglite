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
    expected = "kglite.mcp_server:main"
    actual = entry_points.get("kglite-mcp-server")
    if actual != expected:
        raise RuntimeError(f"installed MCP entry point is {actual!r}, expected {expected!r}")

    package_path = Path(kglite.__file__).resolve()
    if "site-packages" not in package_path.parts:
        raise RuntimeError(f"kglite imported outside the smoke venv: {package_path}")

    launcher_name = "kglite-mcp-server.exe" if sys.platform == "win32" else "kglite-mcp-server"
    # Keep the venv path rather than resolving its interpreter symlink back to
    # the base Python installation, where console launchers do not live.
    launcher = Path(sys.executable).parent / launcher_name
    if not launcher.is_file():
        raise RuntimeError(f"installed MCP launcher is missing: {launcher}")

    result = subprocess.run(
        [str(launcher), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0 or "--selftest" not in output:
        raise RuntimeError(f"installed MCP launcher failed ({result.returncode}):\n{output}")

    print(f"installed wheel OK: {package_path}; launcher={launcher}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
