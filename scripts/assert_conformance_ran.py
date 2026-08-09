#!/usr/bin/env python3
"""Fail unless every Bolt driver-conformance test actually executed and passed.

`pytest` exits 0 for an all-skipped run, and both suites in
`tests/test_bolt_driver_conformance.py` skip themselves when their toolchain is
missing — correct locally, but it means the `bolt-driver-conformance` CI job can
report green having tested nothing. That is exactly what hid there until the
job's first real execution.

This turns any skip in that job into a failure: the job exists to exercise the
official JavaScript and Java drivers, so "no toolchain" is a broken job, not a
pass.

**One distinction, and it is not an escape hatch.** A registry or network
outage during *dependency resolution* — recognized by name, and only after
`scripts/conformance_resolution.py` exhausted its bounded retries — reports as
``INFRASTRUCTURE`` with exit 2 instead of ``PRODUCT`` with exit 1. Both are
failures; both fail the job. The split exists so a human triaging a red run
knows in one glance whether to re-run or to investigate the server. Any other
skip, including one whose reason merely mentions the network, stays exit 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from conformance_resolution import is_infrastructure_skip

# The two suites the job is for. Named explicitly so that a renamed or deleted
# test fails loudly here instead of shrinking the job's coverage in silence.
EXPECTED_TESTS = {
    "test_javascript_driver_conformance",
    "test_java_driver_conformance",
}

NON_PASS_TAGS = ("skipped", "failure", "error")

#: Exit codes. Distinct on purpose, and non-zero on purpose: `1` is "this
#: repository is broken", `2` is "the outside world was broken", and there is
#: no code that means "we did not test and that is fine".
EXIT_OK = 0
EXIT_PRODUCT = 1
EXIT_INFRASTRUCTURE = 2

#: The outcome label given to a skip that carries the outage marker. Kept out
#: of `NON_PASS_TAGS` because it is derived from the skip *reason*, not from
#: the XML tag.
INFRASTRUCTURE = "infra"


def outcomes(report: Path) -> dict[str, str]:
    """Map test name -> outcome from a pytest JUnit-XML report.

    A skipped case whose ``message`` attribute *opens* with the outage marker
    becomes ``infra``. Prefix, not substring, and only on the skip reason: a
    failure whose output quotes the marker is still ``failure``.
    """
    root = ET.parse(report).getroot()
    found: dict[str, str] = {}
    for case in root.iter("testcase"):
        name = case.get("name") or "<unnamed>"
        state = "passed"
        for child in case:
            if child.tag in NON_PASS_TAGS:
                state = child.tag
                if child.tag == "skipped" and is_infrastructure_skip(child.get("message")):
                    state = INFRASTRUCTURE
                break
        found[name] = state
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="pytest --junitxml output")
    args = parser.parse_args()

    if not args.report.is_file():
        print(
            f"::error::PRODUCT: {args.report} does not exist — the conformance "
            f"run produced no JUnit report, so it cannot be shown to have tested anything",
            file=sys.stderr,
        )
        return EXIT_PRODUCT

    found = outcomes(args.report)
    for name in sorted(found):
        print(f"  {found[name]:<8} {name}")

    missing = sorted(EXPECTED_TESTS - set(found))
    not_passed = {n: s for n, s in sorted(found.items()) if s != "passed"}
    # Product-red wins whenever both are present: an outage on one driver does
    # not soften a real conformance failure on the other.
    product = {n: s for n, s in not_passed.items() if s != INFRASTRUCTURE}
    infrastructure = {n: s for n, s in not_passed.items() if s == INFRASTRUCTURE}

    if missing:
        print(
            f"::error::PRODUCT: conformance tests never ran: {missing}. The job "
            f"must exercise both official drivers; a missing test means the "
            f"suite was renamed, deselected, or never collected.",
            file=sys.stderr,
        )
        return EXIT_PRODUCT
    if product:
        print(
            f"::error::PRODUCT: conformance tests did not pass: {product}. A skip "
            f"here is a job failure — install the toolchain or fix the "
            f"network, do not let the job go green without testing the drivers.",
            file=sys.stderr,
        )
        return EXIT_PRODUCT
    if infrastructure:
        print(
            f"::error::INFRASTRUCTURE: dependency resolution outage, retries "
            f"exhausted: {sorted(infrastructure)}. The drivers were never "
            f"exercised, so this job is red — but nothing here indicts kglite. "
            f"Re-run once the registry recovers; investigate the server only if "
            f"it repeats.",
            file=sys.stderr,
        )
        return EXIT_INFRASTRUCTURE

    print(f"OK: {len(found)} driver conformance test(s) executed and passed.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
