#!/usr/bin/env python3
"""Fail before publish when a PyPI project approaches its storage budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
from urllib.request import Request, urlopen

DEFAULT_PROJECT_LIMIT_BYTES = 10_000_000_000
DEFAULT_ACTION_THRESHOLD = 0.80
DEFAULT_RELEASE_RESERVE_BYTES = 250_000_000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="kglite")
    parser.add_argument("--json", type=Path, help="read a saved project JSON response instead of PyPI")
    parser.add_argument("--project-limit-bytes", type=int, default=DEFAULT_PROJECT_LIMIT_BYTES)
    parser.add_argument("--action-threshold", type=float, default=DEFAULT_ACTION_THRESHOLD)
    parser.add_argument("--reserve-bytes", type=int, default=DEFAULT_RELEASE_RESERVE_BYTES)
    return parser


def _load_payload(project: str, json_path: Path | None) -> dict[str, Any]:
    if json_path is not None:
        with json_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        request = Request(
            f"https://pypi.org/pypi/{project}/json",
            headers={"User-Agent": "kglite-release-capacity-check/1"},
        )
        with urlopen(request, timeout=30) as response:  # noqa: S310 — fixed PyPI origin
            payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("PyPI project response must be a JSON object")
    return payload


def project_storage_bytes(payload: dict[str, Any]) -> tuple[int, int]:
    releases = payload.get("releases")
    if not isinstance(releases, dict):
        raise ValueError("PyPI project response has no releases object")

    total = 0
    file_count = 0
    for version, files in releases.items():
        if not isinstance(files, list):
            raise ValueError(f"release {version!r} must contain a file list")
        for file in files:
            if not isinstance(file, dict):
                raise ValueError(f"release {version!r} contains a non-object file record")
            size = file.get("size")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise ValueError(f"release {version!r} contains an invalid file size: {size!r}")
            total += size
            file_count += 1
    return total, file_count


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.project_limit_bytes <= 0:
        print("error: --project-limit-bytes must be positive", file=sys.stderr)
        return 2
    if not 0 < args.action_threshold <= 1:
        print("error: --action-threshold must be in (0, 1]", file=sys.stderr)
        return 2
    if args.reserve_bytes < 0:
        print("error: --reserve-bytes cannot be negative", file=sys.stderr)
        return 2

    try:
        payload = _load_payload(args.project, args.json)
        used, file_count = project_storage_bytes(payload)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: cannot determine PyPI storage: {error}", file=sys.stderr)
        return 2

    action_at = int(args.project_limit_bytes * args.action_threshold)
    projected = used + args.reserve_bytes
    print(
        f"{args.project}: {used / 1_000_000_000:.3f} GB ({used} bytes) across {file_count} files; "
        f"projected after reserve: {projected / 1_000_000_000:.3f} GB ({projected} bytes); "
        f"action threshold: {action_at / 1_000_000_000:.3f} GB"
    )
    if projected >= action_at:
        print(
            "error: PyPI storage action threshold reached; request a project-limit "
            "increase before publishing (do not delete releases automatically)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
