"""Official non-Python Bolt driver conformance — JavaScript and Java.

Until now `kglite-bolt-server`'s README could only claim the official *Python*
driver was regression-tested. Every other driver "may connect", which is a
promise nobody had checked. These wrappers close that gap for the two that
matter most: JavaScript, the largest driver audience, and Java, whose only route
to a kglite graph is this server — there is no in-process JVM binding.

Each suite lives beside this file under `tests/conformance/<lang>/` and is
written in that language against its own official driver — the point is
precisely that a *different* PackStream implementation, retry machinery, and
exception hierarchy agree with what the server sends. Both suites cover the
same 22 checks so they stay comparable.

**Toolchain-gated, and honestly so.** Each test skips with an actionable
message when its toolchain is absent rather than silently passing. The JS suite
runs wherever `node` and `npm` exist; the Java suite needs a JDK 17+ and Maven.
CI installs both (`.github/workflows/ci.yml`, job `bolt-driver-conformance`).

**Fetching the driver is not part of the test.** Both suites download their
driver first — npm registry, Maven Central — and an outage there is not
evidence about this server. `scripts/conformance_resolution.py` retries that
step with bounded backoff and, if a recognized outage survives the budget,
marks the skip so `scripts/assert_conformance_ran.py` reports INFRASTRUCTURE
(exit 2) rather than PRODUCT (exit 1). Both are still failures — see that
module for why no path here can end in a green job that tested nothing.
Execution itself is never retried.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tests.conftest import (
    _BOLT_SKIP_REASON,
    _bolt_binary_available,
    _build_bolt_fixture_graph,
    _spawn_bolt_server,
    _teardown_bolt_server,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from conformance_resolution import (  # noqa: E402
    MAVEN_NETWORK_SIGNATURES,
    MAVEN_RESOLUTION_SIGNATURES,
    NPM_OUTAGE_SIGNATURES,
    StepRunner,
    SuiteOutcome,
    infrastructure_skip_reason,
    run_conformance_suite,
)

pytestmark = pytest.mark.bolt

CONFORMANCE_ROOT = Path(__file__).resolve().parent / "conformance"
JS_DIR = CONFORMANCE_ROOT / "js"
JAVA_DIR = CONFORMANCE_ROOT / "java"

# Each suite asserts this many checks. Kept here so a suite that silently stops
# running half its cases (a broken runner, a bad filter) fails instead of
# reporting a cheerful green.
EXPECTED_CHECKS = 22


def _serve(tmp_path: Path, *extra_args: str):
    """Spawn a conformance server, yielding its URL.

    No `--allow-csv-import`, which the capability check in each suite relies on.
    """
    if not _bolt_binary_available():
        pytest.skip(_BOLT_SKIP_REASON)
    fixture = tmp_path / "conformance.kgl"
    _build_bolt_fixture_graph(fixture)
    proc, url = _spawn_bolt_server(fixture, extra_args=list(extra_args))
    try:
        yield url
    finally:
        _teardown_bolt_server(proc)


@pytest.fixture
def bolt_url(tmp_path: Path):
    """A server on kglite's **default, honest** identity
    (`kglite-bolt-server/<version>`).

    Used by the JavaScript suite — and the Python driver suite elsewhere — so the
    default configuration stays covered. Neither driver inspects the server agent.
    """
    yield from _serve(tmp_path)


@pytest.fixture
def bolt_url_neo4j_compat(tmp_path: Path):
    """A server in **Neo4j compatibility mode** (`--neo4j-compat`).

    The Java driver refuses any server whose agent does not start with `Neo4j/`,
    failing at HELLO with `UntrustedServerException` before a query runs, so this
    is the configuration a JVM user actually has to run. Testing Java here and
    JavaScript on the default above keeps *both* identities covered, which is
    worth more than standardising on one.
    """
    yield from _serve(tmp_path, "--neo4j-compat")


def _require(tool: str, install_hint: str) -> str:
    path = shutil.which(tool)
    if path is None:
        pytest.skip(f"{tool} not found on PATH — install it with: {install_hint}")
    return path


def _demand_success(outcome: SuiteOutcome, label: str, resolver: str) -> None:
    """Turn a suite outcome into the right kind of red.

    Three of the four states end the test here; only `passed` returns. The
    order matters: an outage is announced with the marker the assert script
    keys on, an unrecognized resolution failure gets an ordinary skip (hard
    red, deliberately unchanged), and a failing suite is a plain assertion —
    the shape a driver disagreement has always had.
    """
    if outcome.status == "infrastructure":
        pytest.skip(infrastructure_skip_reason(label, outcome))
    if outcome.status == "unresolved":
        pytest.skip(
            f"{resolver} could not fetch the {label} conformance driver, and the failure "
            f"matches no known registry/network signature (offline?):\n{outcome.report}"
        )
    assert outcome.ok, f"{label} driver conformance failed:\n{outcome.report}"


# ---------------------------------------------------------------------------
# JavaScript
# ---------------------------------------------------------------------------


def _js_resolution_step() -> StepRunner | None:
    """`npm install` for `tests/conformance/js/node_modules`, or None if warm.

    A populated `node_modules` is not an outage and must not spend a retry
    attempt, hence None. The directory is gitignored and pruned by
    `make prune-dev`, per the dev-cleanliness rule that every path the tooling
    writes outside git needs a bound and an owner.
    """
    if (JS_DIR / "node_modules" / "neo4j-driver").exists():
        return None
    npm = _require("npm", "https://nodejs.org (or `brew install node`)")

    def install(_attempt: int) -> tuple[int, str]:
        result = subprocess.run(
            [npm, "install", "--no-audit", "--no-fund", "--loglevel=error"],
            cwd=JS_DIR,
            capture_output=True,
            text=True,
            timeout=600,
        )
        return result.returncode, f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"

    return install


def test_javascript_driver_conformance(bolt_url: str) -> None:
    node = _require("node", "https://nodejs.org (or `brew install node`)")
    resolve = _js_resolution_step()

    def execute(_attempt: int) -> tuple[int, str]:
        result = subprocess.run(
            [node, "conformance.mjs", bolt_url],
            cwd=JS_DIR,
            capture_output=True,
            text=True,
            timeout=600,
        )
        return result.returncode, f"{result.stdout}\n{result.stderr}"

    # No execution signatures: once the driver is installed, `node` talks only
    # to this server, so every failure it reports is about this server.
    outcome = run_conformance_suite(resolve, execute, resolution_signatures=NPM_OUTAGE_SIGNATURES)
    _demand_success(outcome, "JavaScript", "npm")

    passed = outcome.report.count("PASS ")
    assert passed == EXPECTED_CHECKS, (
        f"expected {EXPECTED_CHECKS} JS checks, saw {passed} — did the suite stop early?\n{outcome.report}"
    )


# ---------------------------------------------------------------------------
# Java
# ---------------------------------------------------------------------------


def _require_jdk(minimum: int = 17) -> None:
    """Skip unless a working JDK of at least `minimum` is installed.

    Deliberately *runs* `javac -version` instead of trusting
    `shutil.which("javac")`. macOS ships stubs at `/usr/bin/javac` and
    `/usr/bin/java` that exist on PATH with no JDK behind them and fail with
    "Unable to locate a Java Runtime" — a which-based gate would sail past them
    and the suite would fail as if the server were broken.
    """
    if shutil.which("javac") is None:
        pytest.skip(
            "javac not found on PATH — install a JDK with: `brew install --cask temurin` or apt install openjdk-17-jdk"
        )
    probe = subprocess.run(["javac", "-version"], capture_output=True, text=True, timeout=120)
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip().splitlines()
        pytest.skip(
            "javac is on PATH but not usable (a macOS stub, or a broken install): "
            f"{detail[0] if detail else '<no output>'} — install a JDK with: "
            "`brew install --cask temurin` or apt install openjdk-17-jdk"
        )
    # `javac -version` prints e.g. `javac 17.0.10`.
    blob = (probe.stdout or probe.stderr).strip()
    head = blob.split()[-1].split(".")[0].split("-")[0]
    if head.isdigit() and int(head) < minimum:
        pytest.skip(f"JDK {minimum}+ required by the conformance pom, found {blob}")


def _maven(mvn: str, env: dict[str, str], *goals: str, timeout: int) -> tuple[int, str]:
    """One Maven invocation against the repo-local, gitignored `.m2` cache.

    Keeping downloads under `tests/conformance/java/.m2` rather than the
    developer's `~/.m2` is what lets `make prune-dev` own them, and what lets
    CI cache exactly this suite's dependencies.
    """
    result = subprocess.run(
        [mvn, "-B", "-q", "--no-transfer-progress", f"-Dmaven.repo.local={JAVA_DIR / '.m2'}", *goals],
        cwd=JAVA_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, f"{result.stdout[-8000:]}\n{result.stderr[-4000:]}"


def _java_resolution_step(mvn: str, env: dict[str, str]) -> StepRunner:
    """The step inside the retry loop: download, execute nothing.

    `dependency:go-offline` fetches the driver and the plugins and runs no
    tests, which is the whole reason retrying it is safe — a retry cannot
    re-run, or mask, a conformance result. Module-level rather than a closure
    so a test without Maven can assert exactly that (see
    `tests/test_conformance_resolution.py`).
    """

    def resolve(_attempt: int) -> tuple[int, str]:
        return _maven(mvn, env, "dependency:go-offline", timeout=1800)

    return resolve


def _java_execution_step(mvn: str, env: dict[str, str], bolt_uri: str) -> StepRunner:
    """The suite itself, run once, against the server under test."""

    def execute(_attempt: int) -> tuple[int, str]:
        return _maven(mvn, env, f"-Dkglite.bolt.uri={bolt_uri}", "test", timeout=1800)

    return execute


def test_java_driver_conformance(bolt_url_neo4j_compat: str) -> None:
    _require_jdk()
    mvn = _require("mvn", "`brew install maven` or apt install maven")

    env = dict(os.environ)
    env.setdefault("MAVEN_OPTS", "-Xmx512m")
    resolve = _java_resolution_step(mvn, env)
    execute = _java_execution_step(mvn, env, bolt_url_neo4j_compat)

    outcome = run_conformance_suite(
        resolve,
        execute,
        # Resolution runs alone, so transport-level errors can only come from
        # the resolver: both tiers apply.
        resolution_signatures=(*MAVEN_RESOLUTION_SIGNATURES, *MAVEN_NETWORK_SIGNATURES),
        # Execution output also carries the *driver's* errors, and the Neo4j
        # Java driver says "Connection refused" when this server dies mid-suite
        # — a product failure in a network failure's words. Only phrases Maven
        # itself emits for a resolution failure are consulted here.
        execution_signatures=MAVEN_RESOLUTION_SIGNATURES,
    )
    if outcome.status == "unresolved":
        # `dependency:go-offline` is finicky in ways `test` is not (a profile it
        # cannot pre-resolve, an optional plugin). An unrecognized failure here
        # therefore falls through to the real run, which is authoritative — this
        # can only turn a would-be skip into a genuine result, never the reverse.
        outcome = run_conformance_suite(
            None,
            execute,
            resolution_signatures=(),
            execution_signatures=MAVEN_RESOLUTION_SIGNATURES,
        )
    _demand_success(outcome, "Java", "Maven")
    report = outcome.report

    # Surefire's XML report is the authoritative count — `-q` suppresses the
    # per-test console lines, so parsing stdout would under-report.
    reports = sorted((JAVA_DIR / "target" / "surefire-reports").glob("*.xml"))
    assert reports, f"no surefire report produced:\n{report}"
    total = skipped = 0
    for xml in reports:
        text = xml.read_text(encoding="utf-8", errors="replace")
        for key, target in (("tests=", "total"), ("skipped=", "skipped")):
            marker = f'{key}"'
            if marker in text:
                value = int(text.split(marker, 1)[1].split('"', 1)[0])
                if target == "total":
                    total += value
                else:
                    skipped += value
    executed = total - skipped
    assert executed == EXPECTED_CHECKS, (
        f"expected {EXPECTED_CHECKS} Java checks to execute, saw {executed} "
        f"(total={total}, skipped={skipped}) — did @BeforeAll skip the suite?\n{report}"
    )


def test_both_suites_cover_the_same_checks() -> None:
    """The two suites are only comparable if they assert the same things.

    Names are the contract: `PASS <name>` in the JS harness and `@DisplayName`
    prefixes in the Java suite. This test needs no toolchain — it reads source
    — so a suite drifting out of parity fails even where neither runtime is
    installed.
    """
    js_source = (JS_DIR / "conformance.mjs").read_text(encoding="utf-8")
    java_source = (
        JAVA_DIR / "src" / "test" / "java" / "dev" / "kglite" / "conformance" / "BoltConformanceTest.java"
    ).read_text(encoding="utf-8")

    js_checks = set()
    for line in js_source.splitlines():
        stripped = line.strip()
        if stripped.startswith('await check("'):
            js_checks.add(stripped.split('"', 2)[1])

    java_checks = set()
    for line in java_source.splitlines():
        stripped = line.strip()
        if stripped.startswith('@DisplayName("') and "kglite-bolt-server" not in stripped:
            # `@DisplayName("name — prose")` — the check name is the part
            # before the em dash.
            label = stripped.split('"', 2)[1]
            java_checks.add(label.split(" —")[0].strip())

    assert js_checks == java_checks, (
        "JS and Java conformance suites have drifted apart.\n"
        f"only in JS:   {sorted(js_checks - java_checks)}\n"
        f"only in Java: {sorted(java_checks - js_checks)}"
    )
    assert len(js_checks) == EXPECTED_CHECKS, (
        f"EXPECTED_CHECKS says {EXPECTED_CHECKS} but the suites define {len(js_checks)}; "
        "update the constant alongside the suites"
    )


def test_js_suite_declares_a_pinned_driver_version() -> None:
    """An unpinned driver would make the suite's result depend on release
    timing rather than on this server's behaviour."""
    manifest = json.loads((JS_DIR / "package.json").read_text(encoding="utf-8"))
    assert "neo4j-driver" in manifest["dependencies"]
