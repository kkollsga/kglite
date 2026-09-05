#!/usr/bin/env python3
"""Require the exact, non-yanked kglite version in crates.io's sparse index.

Cargo can exit successfully after its post-upload index poll times out. This
read-only check gates dependent publication without relying on textual search.
The workflow also bounds the whole step, including DNS and slow response reads.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

INDEX_URL = "https://index.crates.io/kg/li/kglite"
MAX_ATTEMPTS = 5
REQUEST_TIMEOUT = 10
RETRY_DELAY = 15
MAX_INDEX_BYTES = 8 * 1024 * 1024


def fetch_index() -> bytes:
    request = Request(
        INDEX_URL,
        headers={"User-Agent": "kglite-release-readiness", "Cache-Control": "no-cache"},
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        if response.status != 200:
            raise ValueError(f"index returned HTTP {response.status}")
        body = response.read(MAX_INDEX_BYTES + 1)
    if len(body) > MAX_INDEX_BYTES:
        raise ValueError("index response exceeds 8 MiB")
    return body


def index_has_version(body: bytes, version: str) -> bool:
    found = False
    for line in body.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("index entry is not an object")
        if row.get("name") == "kglite" and row.get("vers") == version and row.get("yanked") is False:
            found = True
    return found


def wait_for_version(version: str) -> bool:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            if index_has_version(fetch_index(), version):
                print(f"kglite {version} is available in the sparse index (attempt {attempt})")
                return True
            reason = "exact non-yanked version is absent"
        except (URLError, OSError, ValueError) as error:
            reason = str(error)
        print(f"Index not ready (attempt {attempt}/{MAX_ATTEMPTS}): {reason}", file=sys.stderr)
        if attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_DELAY)
    print(f"Cannot confirm kglite {version}; stopping dependent publication.", file=sys.stderr)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Exact kglite version required by the dependent crates")
    args = parser.parse_args(argv)
    return 0 if wait_for_version(args.version) else 1


if __name__ == "__main__":
    sys.exit(main())
