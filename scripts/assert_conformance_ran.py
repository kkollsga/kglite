#!/usr/bin/env python3
"""Fail unless every Bolt driver-conformance test actually executed and passed.

`pytest` exits 0 for an all-skipped run, and both suites in
`tests/test_bolt_driver_conformance.py` skip themselves when their toolchain is
missing — correct locally, but it means the `bolt-driver-conformance` CI job can
report green having tested nothing. That is exactly what hid there until the
job's first real execution.

This turns any skip in that job into a failure: the job exists to exercise the
official JavaScript and Java drivers, so "no toolchain" or "Maven was offline"
is a broken job, not a pass.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

# The two suites the job is for. Named explicitly so that a renamed or deleted
# test fails loudly here instead of shrinking the job's coverage in silence.
EXPECTED_TESTS = {
    "test_javascript_driver_conformance",
    "test_java_driver_conformance",
}

NON_PASS_TAGS = ("skipped", "failure", "error")


def outcomes(report: Path) -> dict[str, str]:
    """Map test name -> outcome from a pytest JUnit-XML report."""
    root = ET.parse(report).getroot()
    found: dict[str, str] = {}
    for case in root.iter("testcase"):
        name = case.get("name") or "<unnamed>"
        state = "passed"
        for child in case:
            if child.tag in NON_PASS_TAGS:
                state = child.tag
                break
        found[name] = state
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="pytest --junitxml output")
    args = parser.parse_args()

    if not args.report.is_file():
        print(
            f"::error::{args.report} does not exist — the conformance run "
            f"produced no JUnit report, so it cannot be shown to have tested anything",
            file=sys.stderr,
        )
        return 1

    found = outcomes(args.report)
    for name in sorted(found):
        print(f"  {found[name]:<8} {name}")

    missing = sorted(EXPECTED_TESTS - set(found))
    not_passed = {n: s for n, s in sorted(found.items()) if s != "passed"}

    if missing:
        print(
            f"::error::conformance tests never ran: {missing}. The job must "
            f"exercise both official drivers; a missing test means the suite "
            f"was renamed, deselected, or never collected.",
            file=sys.stderr,
        )
        return 1
    if not_passed:
        print(
            f"::error::conformance tests did not pass: {not_passed}. A skip "
            f"here is a job failure — install the toolchain or fix the "
            f"network, do not let the job go green without testing the drivers.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {len(found)} driver conformance test(s) executed and passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
