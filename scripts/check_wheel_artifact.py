#!/usr/bin/env python3
"""Validate the contents and console entry point of built KGLite wheels."""

from __future__ import annotations

import configparser
from glob import glob
from pathlib import Path
import sys
import zipfile


def _wheel_paths(arguments: list[str]) -> list[Path]:
    paths: list[Path] = []
    for argument in arguments:
        matches = [Path(match) for match in glob(argument)]
        if matches:
            paths.extend(matches)
        else:
            path = Path(argument)
            if path.exists():
                paths.append(path)
    return sorted(set(paths))


def inspect_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        extensions = [name for name in names if name.startswith("kglite/") and name.endswith((".so", ".pyd"))]
        if not extensions:
            raise ValueError("missing the native kglite extension")
        if "kglite/cli.py" not in names:
            raise ValueError("missing bundled kglite/cli.py")
        if "kglite/mcp_server.py" not in names:
            raise ValueError("missing bundled kglite/mcp_server.py")

        entry_point_files = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        if len(entry_point_files) != 1:
            raise ValueError(f"expected one .dist-info/entry_points.txt, found {len(entry_point_files)}")
        parser = configparser.ConfigParser()
        parser.read_string(wheel.read(entry_point_files[0]).decode("utf-8"))
        cli_actual = parser.get("console_scripts", "kglite", fallback="")
        if cli_actual != "kglite.cli:main":
            raise ValueError("missing kglite = kglite.cli:main console entry point")
        mcp_actual = parser.get("console_scripts", "kglite-mcp-server", fallback="")
        if mcp_actual != "kglite.mcp_server:main":
            raise ValueError("missing kglite-mcp-server = kglite.mcp_server:main console entry point")

    print(f"{path}: extension={','.join(extensions)}; cli+mcp-entry-points=present")


def main() -> int:
    paths = _wheel_paths(sys.argv[1:])
    if not paths:
        print("error: no wheel artifacts matched", file=sys.stderr)
        return 2

    failed = False
    for path in paths:
        try:
            inspect_wheel(path)
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            failed = True
            print(f"error: {path}: {error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
