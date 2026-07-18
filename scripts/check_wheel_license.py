#!/usr/bin/env python3
"""Verify that built wheels carry KGLite's declared MIT license."""

from __future__ import annotations

import argparse
from email import policy
from email.parser import BytesParser
from glob import glob
from pathlib import Path
import sys
import zipfile


def wheel_paths(arguments: list[str]) -> list[Path]:
    """Expand wheel arguments consistently on shells that do not expand globs."""
    paths: list[Path] = []
    for argument in arguments:
        matches = [Path(match) for match in glob(argument)]
        if matches:
            paths.extend(matches)
        else:
            path = Path(argument)
            if path.exists():
                paths.append(path)
    return sorted({path for path in paths if path.suffix == ".whl"})


def inspect_wheel_license(path: Path, *, expected_name: str, license_path: Path) -> None:
    """Require exact MIT metadata and an exact copy of the source LICENSE."""
    expected_license = license_path.read_bytes()
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        metadata_files = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            raise ValueError(f"expected one .dist-info/METADATA, found {len(metadata_files)}")

        metadata = BytesParser(policy=policy.default).parsebytes(wheel.read(metadata_files[0]))
        if metadata.get_all("Name", []) != [expected_name]:
            raise ValueError(f"expected Name: {expected_name}")
        if metadata.get_all("License-Expression", []) != ["MIT"]:
            raise ValueError("expected License-Expression: MIT")
        if metadata.get_all("License-File", []) != ["LICENSE"]:
            raise ValueError("expected License-File: LICENSE")

        license_files = [name for name in names if name.endswith(".dist-info/licenses/LICENSE")]
        if len(license_files) != 1:
            raise ValueError(f"expected one .dist-info/licenses/LICENSE, found {len(license_files)}")
        if wheel.read(license_files[0]) != expected_license:
            raise ValueError(f"embedded LICENSE differs from {license_path}")

    print(f"{path}: {expected_name}; MIT metadata+LICENSE=verified")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-name", required=True, help="exact distribution Name from METADATA")
    parser.add_argument("--license", type=Path, default=Path("LICENSE"), help="checked-in LICENSE to compare")
    parser.add_argument("wheels", nargs="+", help="wheel paths or quoted glob patterns")
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    paths = wheel_paths(args.wheels)
    if not paths:
        print("error: no wheel artifacts matched", file=sys.stderr)
        return 2

    failed = False
    for path in paths:
        try:
            inspect_wheel_license(path, expected_name=args.expected_name, license_path=args.license)
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            failed = True
            print(f"error: {path}: {error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
