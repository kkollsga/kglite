#!/usr/bin/env python3
"""Validate temporary RustSec exceptions and optionally run cargo-audit."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "tests" / "api-baselines" / "rustsec-ignored-advisories.json"
ADVISORY_ID = re.compile(r"^RUSTSEC-\d{4}-\d{4}$")
REQUIRED_FIELDS = {"id", "reason", "reviewed", "expires"}
MAX_REVIEW_DAYS = 90


def validate_policy(path: Path) -> list[str]:
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read RustSec policy {path}: {error}") from error

    ignored = policy.get("ignored")
    if not isinstance(ignored, list):
        raise ValueError("RustSec policy field 'ignored' must be a list")

    today = date.today()
    advisory_ids: list[str] = []
    errors: list[str] = []
    for index, entry in enumerate(ignored):
        label = f"ignored[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{label} must be an object")
            continue
        if set(entry) != REQUIRED_FIELDS:
            errors.append(f"{label} fields must be exactly {sorted(REQUIRED_FIELDS)}")
            continue

        advisory_id = entry["id"]
        reason = entry["reason"]
        if not isinstance(advisory_id, str) or not ADVISORY_ID.fullmatch(advisory_id):
            errors.append(f"{label}.id must match RUSTSEC-YYYY-NNNN")
            continue
        advisory_ids.append(advisory_id)
        if not isinstance(reason, str) or len(reason.strip()) < 20:
            errors.append(f"{label}.reason must contain a concrete justification")

        try:
            reviewed = date.fromisoformat(entry["reviewed"])
            expires = date.fromisoformat(entry["expires"])
        except (TypeError, ValueError):
            errors.append(f"{label}.reviewed and .expires must be ISO dates")
            continue
        if reviewed > today:
            errors.append(f"{label}.reviewed cannot be in the future")
        if expires <= today:
            errors.append(f"{label} expired on {expires.isoformat()}")
        review_days = (expires - reviewed).days
        if review_days <= 0 or review_days > MAX_REVIEW_DAYS:
            errors.append(f"{label} review window must be 1–{MAX_REVIEW_DAYS} days")

    duplicates = sorted({item for item in advisory_ids if advisory_ids.count(item) > 1})
    if duplicates:
        errors.append(f"duplicate advisory exceptions: {duplicates}")
    if errors:
        raise ValueError("\n".join(errors))
    return advisory_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--policy-only", action="store_true")
    args = parser.parse_args()

    try:
        ignored = validate_policy(args.policy)
    except ValueError as error:
        print(f"RustSec exception policy failed:\n{error}")
        return 1
    print(f"RustSec exception policy: OK ({len(ignored)} temporary exceptions)")
    if args.policy_only:
        return 0

    # Treat yanked, unmaintained, and unsound notices as findings too. The
    # required gate blocks unreviewed findings; the scheduled workflow remains
    # report-first so dependency-update failures still produce useful output.
    command = ["cargo", "audit", "--deny", "warnings"]
    for advisory_id in ignored:
        command.extend(("--ignore", advisory_id))
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
