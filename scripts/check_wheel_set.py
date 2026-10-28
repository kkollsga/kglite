#!/usr/bin/env python3
"""Fail when the directory about to be published lacks a required kglite artifact.

PyPI publishing runs with `skip-existing`, so a partial set uploads without
complaint: a missing platform wheel is not an error to the uploader. This lists
what a release must contain and fails before the upload when any entry is
absent. The aarch64 Linux wheels are best-effort and not required here.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

#: `(label, regex over the file name)`; each must match at least one file.
REQUIRED: tuple[tuple[str, str], ...] = (
    ("abi3 Linux x86_64 glibc", r"-cp312-abi3-manylinux\w*x86_64"),
    ("abi3 Linux x86_64 musl", r"-cp312-abi3-musllinux\w*x86_64"),
    ("abi3 macOS arm64", r"-cp312-abi3-macosx_\d+_\d+_arm64"),
    ("abi3 macOS x86_64", r"-cp312-abi3-macosx_\d+_\d+_x86_64"),
    ("abi3 Windows x86_64", r"-cp312-abi3-win_amd64"),
    ("free-threaded (abi3t) Linux x86_64 glibc", r"-cp315-abi3\.abi3t-manylinux\w*x86_64"),
    ("free-threaded (abi3t) macOS arm64", r"-cp315-abi3\.abi3t-macosx_\d+_\d+_arm64"),
    ("free-threaded (abi3t) Windows x86_64", r"-cp315-abi3\.abi3t-win_amd64"),
    ("sdist", r"\.tar\.gz$"),
)


def missing(names: list[str], required: tuple[tuple[str, str], ...] = REQUIRED) -> list[str]:
    return [label for label, pattern in required if not any(re.search(pattern, name) for name in names)]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_wheel_set.py <dist-dir>", file=sys.stderr)
        return 2
    directory = Path(argv[1])
    names = sorted(path.name for path in directory.iterdir() if path.is_file())
    absent = missing(names)
    if absent:
        print(f"error: {directory} is missing: {', '.join(absent)}", file=sys.stderr)
        print("found:", *names, sep="\n  ", file=sys.stderr)
        return 1
    print(f"wheel set complete: {len(names)} file(s), {len(REQUIRED)} required entries present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
