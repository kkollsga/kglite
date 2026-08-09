#!/usr/bin/env python3
"""Classify Bolt driver-conformance failures: infrastructure vs. product.

The conformance job (`.github/workflows/ci.yml`, `bolt-driver-conformance`)
runs two suites written against the *official* JavaScript and Java drivers.
Both first have to fetch that driver — `npm install` from the npm registry,
Maven from Central — and a registry hiccup there says nothing whatsoever about
whether kglite's Bolt server conforms. Before this module both outcomes landed
in the same bucket: a `pytest.skip`, which
`scripts/assert_conformance_ran.py` (rightly) turns into an undifferentiated
exit 1, indistinguishable from "the Java driver disagrees with our server".

Two changes, and one constraint that governs both:

* **Bounded retry, on dependency resolution only.** A transient registry
  failure that clears within the budget proceeds to a completely normal run.
  Test *execution* is never retried — a flaky-looking conformance failure is a
  finding, not something to paper over.
* **A distinct status for an exhausted, recognized outage.** The skip reason
  carries :data:`INFRASTRUCTURE_SKIP_PREFIX`, and the assert script exits 2
  (`INFRASTRUCTURE`) rather than 1 (`PRODUCT`) for it.

**The constraint: neither status is green.** Exit 2 fails the job exactly as
exit 1 does; the difference is only what a human reads in the annotation. This
follows the repo's expected-failure-contract rule — scope what is tolerated,
never swallow it — so no path exists where the suites did not run and CI is
happy. An unrecognized skip reason stays exit 1, unchanged.

**Why two signature tiers.** `mvn test` output contains the *driver's* errors
too, and a Bolt server that died mid-suite makes the Java driver print
"Connection refused" — a genuine product failure wearing a network failure's
words. So only the structural tier (:data:`MAVEN_RESOLUTION_SIGNATURES`,
phrases Maven itself emits solely when *resolution* failed) is consulted on
execution output. The raw network tier (:data:`MAVEN_NETWORK_SIGNATURES`) is
consulted only inside a resolution-only invocation, where nothing but the
resolver can speak.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import time

#: Marker that opens the skip reason for a recognized, retry-exhausted outage.
#: `scripts/assert_conformance_ran.py` matches it against the *start* of the
#: JUnit `<skipped message="...">` attribute — a prefix match on the skip
#: reason, never a substring search of arbitrary output, so a failure report
#: that happens to quote this text cannot promote itself out of product-red.
INFRASTRUCTURE_SKIP_PREFIX = "INFRASTRUCTURE-OUTAGE: "

#: Phrases Maven emits only when *dependency or plugin resolution* failed.
#: Safe to match anywhere in a Maven run, including one that also executed
#: tests, because no driver or test assertion produces them.
MAVEN_RESOLUTION_SIGNATURES = (
    "Could not resolve dependencies",
    "Could not transfer artifact",
    "Failure to transfer",
    "or one of its dependencies could not be resolved",
    "Non-resolvable parent POM",
    "Cannot access central",
)

#: Transport-level failures, matched **only** on the output of a
#: resolution-only Maven invocation. Deliberately excluded from execution
#: output: the Neo4j Java driver prints several of these verbatim when the
#: server under test misbehaves, which is precisely the product failure this
#: classification must not absorb.
MAVEN_NETWORK_SIGNATURES = (
    "Connection timed out",
    "Connection refused",
    "Connection reset",
    "Read timed out",
    "UnknownHostException",
    "Temporary failure in name resolution",
    "Name or service not known",
    "Remote host terminated the handshake",
    "status code: 502",
    "status code: 503",
    "status code: 504",
)

#: npm's own outage vocabulary. `npm install` is resolution and nothing else,
#: so there is no second tier here. Bare error codes are matched rather than
#: the surrounding prose because npm 10 replaced the `npm ERR!` prefix with
#: `npm error`; both prefixes are listed for the one phrase that needs one.
#: `E404` is deliberately absent — a package that does not exist is a broken
#: manifest, which must stay red as a product failure.
NPM_OUTAGE_SIGNATURES = (
    "ECONNRESET",
    "ECONNREFUSED",
    "ETIMEDOUT",
    "ERR_SOCKET_TIMEOUT",
    "ENOTFOUND",
    "EAI_AGAIN",
    "ENETUNREACH",
    "ENETDOWN",
    "npm ERR! network",
    "npm error network",
    "code E502",
    "code E503",
    "code E504",
)

#: Total resolution attempts, and the base of the exponential backoff between
#: them (2 s, then 4 s). Both are arguments to :func:`run_conformance_suite`;
#: tests inject a recording `sleep` so the schedule is asserted without a real
#: wait, which the 120 s per-test ceiling would not tolerate.
DEFAULT_RESOLUTION_ATTEMPTS = 3
DEFAULT_BACKOFF_BASE_SECONDS = 2.0

#: `(returncode, combined_output)` — what a resolution or execution step
#: reports back. The argument is the 1-based attempt number, so a caller can
#: log or vary per attempt.
StepRunner = Callable[[int], "tuple[int, str]"]


def classify(report: str, signatures: Sequence[str]) -> str | None:
    """Return the first signature in `report`, or None if none is recognized.

    Unrecognized is the safe answer, and the common one: anything this does not
    name stays a hard failure.
    """
    for signature in signatures:
        if signature in report:
            return signature
    return None


@dataclass(frozen=True)
class SuiteOutcome:
    """What happened, in the four states CI has to tell apart.

    ``status`` is one of:

    ``passed``
        Execution ran and exited 0.
    ``infrastructure``
        A recognized registry/network outage that survived every retry, or an
        execution run whose failure Maven attributes to resolution. Red, but
        distinctly so.
    ``unresolved``
        Dependency resolution failed for a reason not on any list. Hard red —
        an unrecognized skip must never be classified as infrastructure.
    ``failed``
        Execution ran and failed. Product red, with zero retries.
    """

    status: str
    report: str
    signature: str | None = None
    resolution_attempts: int = 0
    execution_attempts: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "passed"


def _resolve(
    resolve: StepRunner,
    signatures: Sequence[str],
    attempts: int,
    backoff_base: float,
    sleep: Callable[[float], None],
) -> SuiteOutcome | int:
    """Run `resolve` until it succeeds, and return the attempt count if it did.

    Retries fire **only** for a recognized signature. An unrecognized failure
    stops immediately: retrying it would burn CI minutes on something that is
    not going to change, and would blur the line this whole module draws.
    """
    for attempt in range(1, attempts + 1):
        returncode, report = resolve(attempt)
        if returncode == 0:
            return attempt
        signature = classify(report, signatures)
        if signature is None:
            return SuiteOutcome(
                status="unresolved",
                report=report,
                resolution_attempts=attempt,
            )
        if attempt == attempts:
            return SuiteOutcome(
                status="infrastructure",
                report=report,
                signature=signature,
                resolution_attempts=attempt,
            )
        sleep(backoff_base * (2 ** (attempt - 1)))
    raise AssertionError("unreachable: the final attempt always returns")


def run_conformance_suite(
    resolve: StepRunner | None,
    execute: StepRunner,
    *,
    resolution_signatures: Sequence[str],
    execution_signatures: Sequence[str] = (),
    attempts: int = DEFAULT_RESOLUTION_ATTEMPTS,
    backoff_base: float = DEFAULT_BACKOFF_BASE_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> SuiteOutcome:
    """Resolve dependencies (with bounded retry), then run the suite once.

    `resolve` may be None when dependencies are already present — a warm cache
    is not an outage and must not consume an attempt. `execute` is called
    exactly once whatever it returns: retrying a conformance run is how a real
    driver disagreement becomes a flake.
    """
    resolution_attempts = 0
    if resolve is not None:
        resolved = _resolve(resolve, resolution_signatures, attempts, backoff_base, sleep)
        if isinstance(resolved, SuiteOutcome):
            return resolved
        resolution_attempts = resolved

    returncode, report = execute(1)
    if returncode == 0:
        return SuiteOutcome(
            status="passed",
            report=report,
            resolution_attempts=resolution_attempts,
            execution_attempts=1,
        )
    signature = classify(report, execution_signatures)
    return SuiteOutcome(
        status="infrastructure" if signature else "failed",
        report=report,
        signature=signature,
        resolution_attempts=resolution_attempts,
        execution_attempts=1,
    )


def infrastructure_skip_reason(label: str, outcome: SuiteOutcome) -> str:
    """The skip reason for an outage, prefixed so the assert script sees it.

    Says what was tried and what was recognized, because the reader of a red
    job needs to decide "re-run" versus "investigate" from the annotation
    alone.
    """
    if outcome.resolution_attempts > 1:
        tried = f"after {outcome.resolution_attempts} bounded resolution attempts"
    else:
        tried = "while resolving dependencies"
    return (
        f"{INFRASTRUCTURE_SKIP_PREFIX}{label} dependency resolution failed {tried} "
        f"on a recognized registry/network signature ({outcome.signature!r}). "
        "This is an infrastructure outage, not a driver conformance failure — the job is "
        "still red, and scripts/assert_conformance_ran.py exits 2 (INFRASTRUCTURE) rather "
        f"than 1 (PRODUCT) so it can be told apart at a glance.\n{outcome.report}"
    )


def is_infrastructure_skip(message: str | None) -> bool:
    """True when a JUnit skip message opens with the outage marker."""
    return message is not None and message.startswith(INFRASTRUCTURE_SKIP_PREFIX)
