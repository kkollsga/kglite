"""The conformance job's outage classification, and the gate it must not open.

`scripts/conformance_resolution.py` lets a registry outage during dependency
resolution report as something other than a driver conformance failure. That is
a useful distinction and a dangerous one: every mechanism that makes a red job
less alarming is one step from a mechanism that makes an untested job green.

So these tests are written in both directions. The classification tests prove
the four end states are reachable and distinct; the assert-script tests prove
that *none* of them is exit 0 unless both suites actually executed and passed —
including a live pytest run whose suite skips itself entirely, the exact
vacuous-green shape `scripts/assert_conformance_ran.py` exists to catch. Real
pytest produces the XML in the two end-to-end cases so the hand-built fixtures
below cannot quietly describe a JUnit dialect pytest no longer emits.

No test here touches the network, sleeps, or needs a toolchain: resolution and
execution are fakes, and backoff is a recording callable.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from assert_conformance_ran import (  # noqa: E402
    EXIT_INFRASTRUCTURE,
    EXIT_OK,
    EXIT_PRODUCT,
)
from conformance_resolution import (  # noqa: E402
    DEFAULT_RESOLUTION_ATTEMPTS,
    INFRASTRUCTURE_SKIP_PREFIX,
    MAVEN_NETWORK_SIGNATURES,
    MAVEN_RESOLUTION_SIGNATURES,
    NPM_OUTAGE_SIGNATURES,
    SuiteOutcome,
    classify,
    infrastructure_skip_reason,
    is_infrastructure_skip,
    run_conformance_suite,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSERT_SCRIPT = REPO_ROOT / "scripts" / "assert_conformance_ran.py"

# Trimmed from real tool output. Each is the wording the classifier is meant to
# recognize, in the shape the tool prints it.
MAVEN_TRANSFER_OUTAGE = (
    "[ERROR] Failed to execute goal on project bolt-conformance: Could not resolve dependencies "
    "for project dev.kglite:bolt-conformance:jar:1.0: Could not transfer artifact "
    "org.neo4j.driver:neo4j-java-driver:jar:5.28.5 from/to central "
    "(https://repo.maven.apache.org/maven2): Connection timed out"
)
MAVEN_GATEWAY_OUTAGE = (
    "[ERROR] Failed to execute goal: transfer failed for "
    "https://repo.maven.apache.org/maven2/org/neo4j/driver/neo4j-java-driver/5.28.5/"
    "neo4j-java-driver-5.28.5.jar, status code: 503"
)
# A genuine conformance failure. It contains the word "Connection refused"
# because that is what the Neo4j Java driver prints when the server under test
# is gone — the case that makes a naive network-word match dangerous.
JAVA_DRIVER_FAILURE = (
    "[ERROR] Tests run: 22, Failures: 1, Errors: 0, Skipped: 0\n"
    "[ERROR] BoltConformanceTest.temporalRoundTrip:214 expected: <2024-01-01> but was: <2024-01-02>\n"
    "[ERROR] org.neo4j.driver.exceptions.ServiceUnavailableException: Unable to connect to "
    "localhost:7687, ensure the database is running. Connection refused"
)
NPM_OUTAGE = (
    "npm error code ECONNRESET\n"
    "npm error network request to https://registry.npmjs.org/neo4j-driver failed, "
    "reason: read ECONNRESET\n"
    "npm error network This is a problem related to network connectivity."
)
# A broken manifest, not an outage: the package simply does not exist.
NPM_MISSING_PACKAGE = "npm error code E404\nnpm error 404 Not Found - GET https://registry.npmjs.org/neo4j-drivver"


class FakeStep:
    """A resolution or execution step with a scripted sequence of results."""

    def __init__(self, *results: tuple[int, str]) -> None:
        self._results = list(results)
        self.calls = 0

    def __call__(self, attempt: int) -> tuple[int, str]:
        self.calls += 1
        assert attempt == self.calls, f"attempt number {attempt} out of step with call {self.calls}"
        assert self._results, "step called more times than it was scripted for"
        return self._results.pop(0)


class RecordingSleep:
    """Backoff without waiting — the schedule is asserted, never served."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def _run_suite(resolve, execute, **kwargs) -> tuple[SuiteOutcome, RecordingSleep]:
    sleep = RecordingSleep()
    kwargs.setdefault("resolution_signatures", (*MAVEN_RESOLUTION_SIGNATURES, *MAVEN_NETWORK_SIGNATURES))
    kwargs.setdefault("execution_signatures", MAVEN_RESOLUTION_SIGNATURES)
    return run_conformance_suite(resolve, execute, sleep=sleep, **kwargs), sleep


# --- the four end states ----------------------------------------------------


def test_transient_outage_recovers_and_the_suite_runs_normally() -> None:
    """(a) One failed fetch, one good one — nothing about the run is special."""
    resolve = FakeStep((1, MAVEN_TRANSFER_OUTAGE), (0, ""))
    execute = FakeStep((0, "Tests run: 22, Failures: 0"))

    outcome, sleep = _run_suite(resolve, execute)

    assert outcome.status == "passed"
    assert outcome.ok
    assert (resolve.calls, execute.calls) == (2, 1)
    assert outcome.resolution_attempts == 2
    assert sleep.delays == [2.0], "the first retry must back off by the base delay"


def test_exhausted_recognized_outage_is_infrastructure_and_never_executes() -> None:
    """(b) Retries are bounded, exponential, and end in a *distinct* status."""
    resolve = FakeStep(*[(1, MAVEN_GATEWAY_OUTAGE)] * DEFAULT_RESOLUTION_ATTEMPTS)
    execute = FakeStep()

    outcome, sleep = _run_suite(resolve, execute)

    assert outcome.status == "infrastructure"
    assert outcome.signature == "status code: 503"
    assert resolve.calls == DEFAULT_RESOLUTION_ATTEMPTS
    assert execute.calls == 0, "the suite must not run when its dependencies never arrived"
    assert sleep.delays == [2.0, 4.0], "backoff must be exponential and bounded by the attempt budget"

    reason = infrastructure_skip_reason("Java", outcome)
    assert is_infrastructure_skip(reason)
    assert "status code: 503" in reason


def test_conformance_failure_is_product_red_with_zero_retries() -> None:
    """(c) A driver disagreement is a finding. It is never retried, and it is
    never softened — even though its output says "Connection refused"."""
    resolve = FakeStep((0, ""))
    execute = FakeStep((1, JAVA_DRIVER_FAILURE))

    outcome, sleep = _run_suite(resolve, execute)

    assert outcome.status == "failed"
    assert outcome.signature is None
    assert execute.calls == 1, "test execution must never be retried"
    assert outcome.execution_attempts == 1
    assert sleep.delays == []


def test_unrecognized_resolution_failure_is_not_classified_as_infrastructure() -> None:
    """(d) Anything unnamed stays hard red, and burns no retries doing it."""
    resolve = FakeStep((1, "[ERROR] The JAVA_HOME environment variable is not defined correctly"))
    execute = FakeStep()

    outcome, sleep = _run_suite(resolve, execute)

    assert outcome.status == "unresolved"
    assert outcome.signature is None
    assert resolve.calls == 1, "an unrecognized failure is not going to change; do not retry it"
    assert execute.calls == 0
    assert sleep.delays == []


def test_a_warm_cache_spends_no_attempt() -> None:
    resolve = None
    execute = FakeStep((0, "PASS " * 22))

    outcome, _ = _run_suite(resolve, execute)

    assert outcome.status == "passed"
    assert outcome.resolution_attempts == 0


def test_maven_execution_output_is_classified_only_by_structural_signatures() -> None:
    """The driver's own network wording must not reach the network tier.

    Same report, two signature sets: with the tier the Java path actually
    passes it is a product failure; it would be misread as an outage if the
    transport tier were consulted on execution output.
    """
    assert classify(JAVA_DRIVER_FAILURE, MAVEN_RESOLUTION_SIGNATURES) is None
    assert classify(JAVA_DRIVER_FAILURE, MAVEN_NETWORK_SIGNATURES) == "Connection refused"


def test_maven_resolution_failure_during_execution_is_an_outage() -> None:
    """`mvn test` resolves too; a resolution failure there is still an outage."""
    outcome, _ = _run_suite(FakeStep((0, "")), FakeStep((1, MAVEN_TRANSFER_OUTAGE)))
    assert outcome.status == "infrastructure"
    assert outcome.signature == "Could not resolve dependencies"


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        (NPM_OUTAGE, "ECONNRESET"),
        (NPM_MISSING_PACKAGE, None),
        ("npm error code EACCES: permission denied, mkdir '/usr/lib/node_modules'", None),
    ],
)
def test_npm_signatures_recognize_outages_and_nothing_else(report: str, expected: str | None) -> None:
    assert classify(report, NPM_OUTAGE_SIGNATURES) == expected


def test_the_retried_maven_step_resolves_and_runs_no_tests(monkeypatch, tmp_path: Path) -> None:
    """The Java suite's wiring, checked where no JDK or Maven exists.

    Everything else about the Java path is CI-only. What must hold locally is
    the property that makes retrying safe at all: the step inside the retry
    loop downloads dependencies and executes nothing, so a retry can never
    re-run — or mask — a conformance result.
    """
    from tests import test_bolt_driver_conformance as suite

    class _Done:
        returncode = 0
        stdout = ""
        stderr = ""

    recorded: list[list[str]] = []

    def fake_run(command, **_kwargs):
        recorded.append(list(command))
        return _Done()

    monkeypatch.setattr(subprocess, "run", fake_run)
    # The real step the retry loop is handed, not a re-spelling of it: an
    # earlier version of this test called `_maven` with the goal written out
    # here, so changing the suite's goal to `test` left it green.
    returncode, _ = suite._java_resolution_step("mvn", {})(1)

    assert returncode == 0
    command = recorded[0]
    assert command[0] == "mvn"
    assert "-B" in command, "a retried step must not wait on an interactive prompt"
    assert any(arg.startswith("-Dmaven.repo.local=") for arg in command), (
        "downloads must land in the repo-local cache the CI cache key covers"
    )
    assert "test" not in command, "the retried step must not execute the conformance suite"
    assert not any(arg.startswith("-Dkglite.bolt.uri=") for arg in command), (
        "the resolution step must not be pointed at a server — it does not talk to one"
    )


# --- the assert script: which reds, and no greens ---------------------------


def _junit(tmp_path: Path, *cases: str) -> Path:
    report = tmp_path / "conformance.xml"
    body = "".join(cases)
    report.write_text(
        f'<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" tests="{len(cases)}">'
        f"{body}</testsuite></testsuites>",
        encoding="utf-8",
    )
    return report


def _case(name: str, tag: str | None = None, message: str = "") -> str:
    if tag is None:
        return f'<testcase classname="c" name="{name}"/>'
    escaped = message.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
    return f'<testcase classname="c" name="{name}"><{tag} message="{escaped}">detail</{tag}></testcase>'


def _assert_script(report: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ASSERT_SCRIPT), str(report)],
        capture_output=True,
        text=True,
        timeout=60,
    )


JS = "test_javascript_driver_conformance"
JAVA = "test_java_driver_conformance"
OUTAGE_REASON = f"{INFRASTRUCTURE_SKIP_PREFIX}Java dependency resolution failed after 3 attempts"


def test_both_suites_passing_is_the_only_green(tmp_path: Path) -> None:
    result = _assert_script(_junit(tmp_path, _case(JS), _case(JAVA)))
    assert result.returncode == EXIT_OK, result.stderr


def test_exhausted_outage_exits_infrastructure(tmp_path: Path) -> None:
    report = _junit(tmp_path, _case(JS), _case(JAVA, "skipped", OUTAGE_REASON))
    result = _assert_script(report)
    assert result.returncode == EXIT_INFRASTRUCTURE, result.stderr
    assert "INFRASTRUCTURE" in result.stderr
    assert "PRODUCT" not in result.stderr


def test_product_failure_outranks_a_concurrent_outage(tmp_path: Path) -> None:
    """An outage on one driver must not soften a real failure on the other."""
    report = _junit(tmp_path, _case(JS, "failure", "22 != 21"), _case(JAVA, "skipped", OUTAGE_REASON))
    result = _assert_script(report)
    assert result.returncode == EXIT_PRODUCT, result.stderr
    assert "PRODUCT" in result.stderr


def test_an_unrecognized_skip_is_still_hard_red(tmp_path: Path) -> None:
    report = _junit(tmp_path, _case(JS), _case(JAVA, "skipped", "mvn not found on PATH"))
    result = _assert_script(report)
    assert result.returncode == EXIT_PRODUCT, result.stderr


def test_a_failure_quoting_the_marker_cannot_promote_itself(tmp_path: Path) -> None:
    """Classification reads the skip reason's prefix, not the run's output —
    otherwise a suite could talk its way out of product-red by echoing text."""
    report = _junit(tmp_path, _case(JS), _case(JAVA, "failure", f"driver printed {INFRASTRUCTURE_SKIP_PREFIX}oops"))
    result = _assert_script(report)
    assert result.returncode == EXIT_PRODUCT, result.stderr


def test_a_missing_suite_is_hard_red_even_when_the_rest_passed(tmp_path: Path) -> None:
    result = _assert_script(_junit(tmp_path, _case(JS)))
    assert result.returncode == EXIT_PRODUCT, result.stderr
    assert JAVA in result.stderr


def test_exit_codes_are_distinct_and_none_of_them_means_untested_but_fine() -> None:
    assert EXIT_OK == 0
    assert EXIT_PRODUCT != EXIT_INFRASTRUCTURE
    assert EXIT_PRODUCT != EXIT_OK and EXIT_INFRASTRUCTURE != EXIT_OK


# --- end to end, against real pytest ---------------------------------------


def _pytest_junit(tmp_path: Path, source: str) -> tuple[int, Path]:
    """Run a synthetic suite under real pytest and return (exit code, report).

    `-o addopts=` and a tmp rootdir keep the repository's pytest configuration
    out of the child run.
    """
    suite = tmp_path / "test_synthetic_conformance.py"
    suite.write_text(source, encoding="utf-8")
    report = tmp_path / "conformance.xml"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(suite),
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "-q",
            f"--junitxml={report}",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    return proc.returncode, report


def test_a_real_pytest_outage_skip_reaches_the_script_as_infrastructure(tmp_path: Path) -> None:
    """The marker survives pytest's JUnit writer — asserted against pytest
    itself, so the hand-built fixtures above cannot describe a dialect pytest
    stopped emitting."""
    source = (
        "import pytest\n\n"
        f"def {JS}():\n    pass\n\n"
        f"def {JAVA}():\n"
        f"    pytest.skip({OUTAGE_REASON!r} + chr(10) + 'multi\\nline maven output')\n"
    )
    pytest_exit, report = _pytest_junit(tmp_path, source)
    assert pytest_exit == 0, "pytest exits 0 for a skip — which is exactly why the script exists"

    result = _assert_script(report)
    assert result.returncode == EXIT_INFRASTRUCTURE, result.stderr


def test_mutation_a_suite_that_skips_everything_and_exits_zero_goes_red(tmp_path: Path) -> None:
    """The anti-vacuity assertion, exercised rather than asserted about.

    This is the mutation the classification work had to survive: a runner that
    skips every conformance test still exits 0 under pytest. If the script ever
    returns 0 here, the whole job can pass having tested nothing.
    """
    source = (
        f"import pytest\n\ndef {JS}():\n    pytest.skip('disabled')\n\ndef {JAVA}():\n    pytest.skip('disabled')\n"
    )
    pytest_exit, report = _pytest_junit(tmp_path, source)
    assert pytest_exit == 0, "pytest must report an all-skipped run as success for this test to mean anything"

    result = _assert_script(report)
    assert result.returncode == EXIT_PRODUCT, result.stderr
    assert result.returncode != EXIT_OK
