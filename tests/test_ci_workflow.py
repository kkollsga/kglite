"""Hermetic self-tests for required CI verification gates.

The workflows are parsed with PyYAML rather than pattern-matched as text.
Text matching had made three guards in this file structurally incapable of
failing: an artifact-name regex that could only capture names it was meant to
reject, a ``-rs`` flag satisfied by a comment inside the same job, and a
check invocation that is a strict prefix of the ``--self-test`` line asserted
directly above it.

Parsing fixes two of those three classes:

* **Comment subsumption.** "This command is present" was satisfied by the
  prose explaining the command. Comments never reach an assertion now —
  neither YAML comments (the parser drops them) nor ``#`` lines inside a
  ``run:`` script (:func:`_step_commands` drops them).
* **Scope.** ``steps``/``uses``/``with``/``if``/``env``/``needs`` are read as
  structured values instead of being guessed at with a regex over a text
  block, so an assertion about one step can no longer be satisfied by an
  unrelated step that happens to share the job.

Parsing does **not** fix substring subsumption. After parsing, a step's
``run:`` is still a string, so ``assert "cmd" in run`` still matches any
longer line that merely contains ``cmd`` — which is exactly how deleting the
real ``check_source_quality.py`` invocation stayed green. Every command
assertion here therefore compares whole stripped logical lines for equality
(:func:`_assert_runs`, :func:`_step_running`) or exact argument tokens
(:func:`_tokens`), never ``in`` against a raw script.

PyYAML maps the YAML 1.1 key ``on`` to the boolean ``True``; the trigger table
is read as ``workflow[True]`` rather than avoiding the parser over it.

Every assertion in this module is only as good as the helper that derived what
it inspects: a helper that silently returns an empty list leaves each test
built on it green having examined nothing — the exact shape of the artifact
guard that died. The final section therefore self-tests the derivation helpers
against synthetic workflow text, in both directions (the shape is extracted
when present, and *not* extracted when absent or only mentioned in a comment),
following ``scripts/check_source_quality.py::_self_test``.

``import yaml`` below is deliberately unguarded. ``pytest.importorskip`` would
turn a missing dependency into a skip, and a skip counts as a pass both to
pytest and to the ``ci-success`` aggregate job — reintroducing the precise
failure mode this module exists to prevent. A missing PyYAML must be a hard
collection error. It is declared in the ``python-tests`` install step of
``.github/workflows/ci.yml``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path, PurePosixPath
import re
import shlex

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
CI_PATH = WORKFLOWS / "ci.yml"
WHEELS_PATH = WORKFLOWS / "build_wheels.yml"
CLI_WHEELS_PATH = WORKFLOWS / "build_cli_wheels.yml"
CRATES_PATH = WORKFLOWS / "publish_crates.yml"

#: Jobs whose *existence* is a guarantee in its own right.
#:
#: This is deliberately not the aggregate-gate list — that one is derived (see
#: :func:`test_ci_success_needs_every_job_defined_in_ci_yml`), because a
#: hand-maintained copy of it reproduces the exact failure the `ci-success` job
#: was created to abolish. What a derivation *cannot* notice is a job deleted
#: from `ci.yml` outright: `ci-success` still needs every job that remains, so
#: the gate is truthfully complete while a specific guarantee has silently
#: disappeared. These names are the ones whose loss would be that.
REQUIRED_JOBS = {
    "docs",
    "rust-core-coverage",
    "source-quality",
    "rustsec-audit",
    "storage-parity",
    "disk-concurrency",
    "loom-session",
    "miri-loaders",
    "address-sanitizer",
    "dependency-maintenance",
    "scheduled-concurrency-stress",
    "bolt-driver-conformance",
    "perf-regression",
}


def _load_workflow(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path} did not parse as a YAML mapping"
    return data


CI = _load_workflow(CI_PATH)
WHEELS = _load_workflow(WHEELS_PATH)
CLI_WHEELS = _load_workflow(CLI_WHEELS_PATH)
CRATES = _load_workflow(CRATES_PATH)


class _Job(dict):
    """A parsed workflow job that remembers what to call itself in a failure."""

    def __init__(self, label: str, data: dict) -> None:
        super().__init__(data)
        self.label = label


def _job(workflow: dict, workflow_name: str, name: str) -> _Job:
    jobs = workflow.get("jobs") or {}
    data = jobs.get(name)
    assert isinstance(data, dict), f"missing required job in {workflow_name}: {name}"
    return _Job(f"{workflow_name} `{name}`", data)


def _ci_job(name: str) -> _Job:
    return _job(CI, "ci.yml", name)


def _wheels_job(name: str) -> _Job:
    return _job(WHEELS, "build_wheels.yml", name)


def _cli_wheels_job(name: str) -> _Job:
    return _job(CLI_WHEELS, "build_cli_wheels.yml", name)


def _crates_job(name: str) -> _Job:
    return _job(CRATES, "publish_crates.yml", name)


def _steps(job: dict) -> list[dict]:
    steps = job.get("steps") or []
    assert isinstance(steps, list), "job `steps` is not a list"
    return [step for step in steps if isinstance(step, dict)]


def _steps_using(job: dict, action_prefix: str) -> list[dict]:
    """Every step whose ``uses:`` starts with ``action_prefix``.

    Action versions are matched by prefix on purpose (``actions/setup-node@``,
    not ``@v7``): what these tests guard is that an action runs at all, and
    pinning the major here turned routine action bumps into failures with
    nothing wrong behind them.
    """
    return [step for step in _steps(job) if str(step.get("uses", "")).startswith(action_prefix)]


def _logical_lines(script: str) -> Iterator[str]:
    """Yield whole shell commands from a ``run:`` script.

    Comment lines are dropped, backslash continuations are joined, and runs of
    whitespace collapse to a single space — so a command wrapped across four
    lines compares as the one command it is, and the prose above a command can
    never stand in for the command itself.
    """
    buffered = ""
    for raw in script.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            buffered += stripped[:-1] + " "
            continue
        yield " ".join((buffered + stripped).split())
        buffered = ""
    if buffered.strip():
        yield " ".join(buffered.split())


def _step_commands(step: dict) -> list[str]:
    run = step.get("run")
    if not isinstance(run, str):
        return []
    return list(_logical_lines(run))


def _command_lines(job: dict) -> list[str]:
    return [line for step in _steps(job) for line in _step_commands(step)]


def _assert_runs(job: _Job, command: str) -> None:
    """Assert the job runs ``command`` as a command of its own.

    Whole-line equality, never containment: a substring check is satisfied by
    any longer line that merely contains the command, which is how
    ``python scripts/check_source_quality.py`` was "verified" by the
    ``--self-test`` line above it while the real check could be deleted.
    """
    lines = _command_lines(job)
    assert command in lines, (
        f"{job.label} does not run `{command}` as a command of its own.\nCommands it does run:\n  " + "\n  ".join(lines)
    )


def _step_running(job: _Job, command: str) -> dict:
    """The single step whose script contains ``command`` as a whole line."""
    matches = [step for step in _steps(job) if command in _step_commands(step)]
    assert len(matches) == 1, f"{job.label} runs `{command}` in {len(matches)} steps, expected exactly 1"
    return matches[0]


def _tokens(line: str) -> list[str]:
    """Shell-split a command line into exact argument tokens.

    Token equality is what makes flag and path assertions honest: ``-rs`` as a
    token cannot be satisfied by ``-rs`` inside a longer word, and
    ``tests/test_stress.py`` cannot be satisfied by
    ``tests/test_session_stress.py``.
    """
    try:
        return shlex.split(line, comments=False)
    except ValueError:
        return line.split()


def _command_tokens(job: dict) -> list[str]:
    return [token for line in _command_lines(job) for token in _tokens(line)]


_ENV_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")


def _pytest_invocations(job: dict) -> list[list[str]]:
    """Argument tokens for every pytest invocation among the job's commands.

    Covers `pytest …`, `.venv/bin/pytest …`, `python -m pytest …`, and any of
    those behind leading `VAR=value` environment assignments.
    """
    invocations: list[list[str]] = []
    for line in _command_lines(job):
        tokens = _tokens(line)
        while tokens and _ENV_ASSIGNMENT.match(tokens[0]):
            tokens = tokens[1:]
        if not tokens:
            continue
        if PurePosixPath(tokens[0]).name in {"pytest", "pytest.exe"}:
            invocations.append(tokens[1:])
        elif tokens[1:3] == ["-m", "pytest"]:
            invocations.append(tokens[3:])
    return invocations


def _markers(args: list[str]) -> list[str]:
    """The marker expressions a pytest invocation selects with ``-m``."""
    return [args[index + 1] for index, arg in enumerate(args[:-1]) if arg == "-m"]


def test_ci_success_needs_every_job_defined_in_ci_yml() -> None:
    """`ci-success` must depend on every job the workflow defines.

    Derived from `ci.yml`, never listed here. `ci-success` exists precisely
    because a hand-maintained allowlist of jobs let 0.11.6 ship while the
    free-threading job was red — and its own `needs:` is hand-maintained, so
    guarding it with a second hand-maintained list in this file reproduced the
    same shape one layer up: a job added to `ci.yml` and forgotten in `needs`
    was gated by nothing, and nothing noticed.

    `skipped` stays a pass in the aggregate's shell on purpose — three jobs are
    `if: github.event_name == 'schedule'` and skip on every push and PR. That
    is a recorded trade-off, not something this test may tighten.
    """
    jobs = set(CI["jobs"])
    assert len(jobs) > 1, "ci.yml defines no jobs — the derivation is broken"

    ci_success = _ci_job("ci-success")
    needs = set(ci_success["needs"])

    missing = jobs - needs - {"ci-success"}
    assert not missing, (
        f"ci.yml job(s) {sorted(missing)} are not in ci-success.needs, so nothing gates them. "
        "Every job in the file must be aggregated — add them to `needs:`."
    )

    unknown = needs - jobs
    assert not unknown, f"ci-success needs job(s) that ci.yml does not define: {sorted(unknown)}"
    assert "ci-success" not in needs, "ci-success cannot depend on itself"


def test_required_verification_jobs_still_exist() -> None:
    """Deleting a job outright leaves the derivation above green.

    `ci-success` needing every *remaining* job stays true when a job is removed
    from `ci.yml`, so the aggregate gate reports honestly while the guarantee
    is gone. These names are checked for existence for that reason only — their
    aggregation is the derivation's job, not this one's.
    """
    for name in sorted(REQUIRED_JOBS):
        _ci_job(name)


def test_ci_success_actually_fails_when_a_dependency_does() -> None:
    """The aggregate gate's own body must be able to go red.

    It is the single check both publish workflows wait on, and it runs under
    `if: always()` — so if its script stopped reading `needs.*.result`, or
    stopped exiting non-zero, every downstream release gate would pass on a
    workflow full of failures without anything else changing.
    """
    ci_success = _ci_job("ci-success")
    assert ci_success["if"] == "always()", (
        "ci-success must run under `always()`; otherwise a failed dependency skips it "
        "and there is no definitive check for the publish workflows to wait on"
    )
    _assert_runs(ci_success, "results=\"${{ join(needs.*.result, ',') }}\"")
    _assert_runs(ci_success, "*,failure,* | *,cancelled,*)")
    _assert_runs(ci_success, "exit 1")


def test_storage_and_disk_jobs_run_bounded_regression_targets() -> None:
    parity = _ci_job("storage-parity")
    parity_runs = _pytest_invocations(parity)
    assert len(parity_runs) == 1, "storage-parity should run exactly one pytest invocation"
    parity_args = parity_runs[0]
    assert _markers(parity_args) == ["parity"]
    for target in (
        "tests/test_storage_parity.py",
        "tests/test_phase1_parity.py",
        "tests/test_phase2_parity.py",
        "tests/test_phase3_parity.py",
        "tests/test_phase4_parity.py",
        "tests/test_phase5_parity.py::test_graph_copy_cow_correctness_memory",
        "tests/test_phase5_parity.py::test_graph_copy_cow_correctness_mapped",
    ):
        assert target in parity_args, f"storage-parity does not run {target}"

    disk = _ci_job("disk-concurrency")
    disk_args = [arg for invocation in _pytest_invocations(disk) for arg in invocation]
    assert disk_args, "disk-concurrency runs no pytest invocation"
    for node_id in (
        "tests/test_concurrency.py::TestConcurrentReads::test_concurrent_disk_reads_keep_materialized_nodes_alive",
        "tests/test_disk_mutation_roundtrip.py::test_disk_writer_lease_is_enforced_across_processes",
        "tests/test_session.py::test_disk_session_reuses_writer_lineage_and_composes",
    ):
        assert node_id in disk_args, f"disk-concurrency does not run {node_id}"


def test_python_job_builds_the_measured_release_extension() -> None:
    python_tests = _ci_job("python-tests")
    _assert_runs(python_tests, "maturin build --release --out /tmp/kglite-binary-size-wheel")
    _assert_runs(python_tests, "cargo build --release -p kglite-mcp-server -p kglite-bolt-server")


def test_free_threading_uses_the_pyo3_supported_python() -> None:
    free_threading = _ci_job("free-threading")
    versions = [
        step.get("with", {}).get("python-version") for step in _steps_using(free_threading, "actions/setup-python@")
    ]
    assert versions == ["3.14t"], f"free-threading sets up {versions}, expected only the PyO3-supported 3.14t"


def test_linux_perf_gate_uses_isolated_released_wheel_reference() -> None:
    perf = _ci_job("perf-regression")
    tokens = _command_tokens(perf)

    # The reference is a published wheel installed by version — never a source
    # build of another revision. `--only-binary=:all:` is what enforces that.
    _assert_runs(
        perf,
        'python -m pip download --only-binary=:all: --no-deps "kglite==$REFERENCE_VERSION" '
        '--dest "$RUNNER_TEMP/reference-wheel"',
    )
    assert perf["env"]["REFERENCE_VERSION"] == "0.13.2"

    # Reference, candidate, and the single retry recapture — each on the frozen
    # core harness, each selecting the benchmark marker, each writing its own
    # JSON so the evidence artifact can carry all of them.
    benchmark_runs = _pytest_invocations(perf)
    assert len(benchmark_runs) == 3, f"perf-regression runs {len(benchmark_runs)} pytest invocations, expected 3"
    for args in benchmark_runs:
        assert "test_bench_core.py" in args
        assert _markers(args) == ["benchmark"]
        assert [arg for arg in args if arg.startswith("--benchmark-json=")], f"capture without --benchmark-json: {args}"

    _assert_runs(perf, "sleep 30")
    _assert_runs(
        perf,
        'python scripts/compare_bench.py .bench-reference-0.13.2.json "$1" '
        "--metric min --threshold 20 --require-exact-set "
        '&& python scripts/compare_bench.py tests/benchmarks/baselines/current.linux.json "$1" '
        "--metric min --threshold 20 --require-exact-set",
    )
    assert tokens.count("--require-exact-set") == 2

    # Retry-once contract: a first-capture regression verdict triggers exactly
    # one recapture; only a repeated failure is red.
    _assert_runs(perf, "if compare .bench-candidate.json; then")
    _assert_runs(perf, "compare .bench-candidate-retry.json")

    assert "scripts/benchmark_provenance.py" in tokens

    # Evidence upload. `include-hidden-files` is the load-bearing setting (the
    # captures are dotfiles and would silently not upload without it) and
    # `if: always()` is what makes the evidence survive a failed verdict — both
    # asserted on the upload step itself, not merely somewhere in the job.
    uploads = _steps_using(perf, "actions/upload-artifact@")
    assert len(uploads) == 1, "perf-regression should upload exactly one evidence artifact"
    upload = uploads[0]
    assert upload["with"]["include-hidden-files"] is True
    assert upload["if"] == "always()"


def test_loom_and_unsafe_jobs_use_the_intended_commands() -> None:
    loom = _ci_job("loom-session")
    _assert_runs(loom, 'RUSTFLAGS="--cfg loom" cargo test -p kglite --test loom_session')

    miri = _ci_job("miri-loaders")
    for test_name in (
        "packed_primitives_decode_from_misaligned_little_endian_bytes",
        "parse_line_borrowed_uri_boundaries_are_valid",
        "parse_line_preserves_utf8_literal_boundaries",
    ):
        _assert_runs(miri, f"cargo miri test -p kglite --lib {test_name}")

    asan = _ci_job("address-sanitizer")
    asan_step = _step_running(
        asan,
        "cargo test -p kglite --lib --target x86_64-unknown-linux-gnu "
        "overlapping_query_guards_keep_materializations_alive",
    )
    assert asan_step["env"]["RUSTFLAGS"] == "-Zsanitizer=address"


def test_heavy_thread_sanitizer_is_scheduled_only() -> None:
    scheduled = _ci_job("scheduled-thread-sanitizer")
    assert scheduled["if"] == "github.event_name == 'schedule'"
    tsan_steps = [step for step in _steps(scheduled) if step.get("env", {}).get("RUSTFLAGS") == "-Zsanitizer=thread"]
    assert tsan_steps, "scheduled-thread-sanitizer runs no step under -Zsanitizer=thread"
    # PyYAML resolves the YAML 1.1 key `on` to the boolean True.
    assert "schedule" in CI[True], "ci.yml has no schedule trigger, so the scheduled-only jobs never run"


def test_live_github_smoke_requires_explicit_opt_in() -> None:
    smoke = (REPO_ROOT / "tests" / "test_mcp_server_smoke.py").read_text(encoding="utf-8")
    assert 'os.environ.get("KGLITE_GITHUB_INTEGRATION") == "1"' in smoke
    assert "and GITHUB_TOKEN is not None" in smoke
    assert smoke.count("not _github_live_enabled()") == 2


def test_docs_job_checks_generated_facts_and_warnings() -> None:
    docs = _ci_job("docs")
    _assert_runs(docs, "python scripts/render_docs_facts.py --check")
    _assert_runs(docs, "sphinx-build -W --keep-going -b html docs docs/_build/html")
    assert "myst.xref_missing" not in (REPO_ROOT / "docs" / "conf.py").read_text(encoding="utf-8")


def test_source_quality_runs_once_in_its_own_required_job() -> None:
    source_quality = _ci_job("source-quality")
    # The self-test line and the real check differ only by a suffix, so these
    # must be whole-line matches: under containment, deleting the real check
    # left this test green on the `--self-test` line above it.
    for command in (
        "python scripts/check_source_quality.py --self-test",
        "python scripts/check_source_quality.py",
        "python scripts/check_lint_allowances.py --self-test",
        "python scripts/check_lint_allowances.py",
        "python scripts/check_rustsec_advisories.py --policy-only",
    ):
        _assert_runs(source_quality, command)

    python_commands = _command_lines(_ci_job("python-tests"))
    for script in ("check_source_quality.py", "check_lint_allowances.py"):
        duplicated = [line for line in python_commands if script in line]
        assert not duplicated, f"python-tests re-runs {script}: {duplicated}"


def test_rustsec_audit_is_required_and_pinned() -> None:
    audit = _ci_job("rustsec-audit")
    assert "if" not in audit, "the required RustSec audit must run on every event, not behind a condition"
    tools = [step.get("with", {}).get("tool") for step in _steps_using(audit, "taiki-e/install-action@")]
    assert tools == ["cargo-audit@0.22.2"], f"rustsec-audit installs {tools}, expected the pinned cargo-audit"
    _assert_runs(audit, "python scripts/check_rustsec_advisories.py")
    assert "--policy-only" not in _command_tokens(audit), "the required audit must not run in policy-only mode"
    assert "rustsec-audit" in _ci_job("ci-success")["needs"]


def test_every_ci_job_has_a_wall_clock_timeout() -> None:
    jobs = CI["jobs"]
    assert jobs
    for name, job in jobs.items():
        assert "timeout-minutes" in job, f"CI job has no timeout: {name}"


def test_scheduled_dependency_maintenance_is_report_first() -> None:
    dependabot = yaml.safe_load((REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))
    cargo = [update for update in dependabot["updates"] if update.get("package-ecosystem") == "cargo"]
    assert len(cargo) == 1, "expected exactly one cargo dependabot policy"
    cargo_policy = cargo[0]
    assert "ignore" not in cargo_policy, "the cargo policy silences updates with an `ignore` list"
    groups = cargo_policy["groups"]
    assert len(groups) == 1
    (group,) = groups.values()
    assert "ignore" not in group
    assert group["update-types"] == ["minor", "patch"], "grouped cargo updates must exclude majors"

    maintenance = _ci_job("dependency-maintenance")
    assert maintenance["if"] == "github.event_name == 'schedule'"
    tools = [step.get("with", {}).get("tool") for step in _steps_using(maintenance, "taiki-e/install-action@")]
    assert tools == ["cargo-audit@0.22.2"]
    # Report-first: the two report steps tolerate failure, the policy check does
    # not — asserted per step rather than by counting the flag across the job.
    assert _step_running(maintenance, "cargo update --workspace --dry-run").get("continue-on-error") is True
    assert _step_running(maintenance, "python scripts/check_rustsec_advisories.py").get("continue-on-error") is True
    assert (
        _step_running(maintenance, "python scripts/check_rustsec_advisories.py --policy-only").get("continue-on-error")
        is None
    )
    tolerated = [step for step in _steps(maintenance) if step.get("continue-on-error") is True]
    assert len(tolerated) == 2, "only the two report steps may tolerate failure"


def test_scheduled_stress_is_bounded_and_excludes_large_runner_case() -> None:
    stress = _ci_job("scheduled-concurrency-stress")
    invocations = _pytest_invocations(stress)
    assert len(invocations) == 2, f"scheduled stress runs {len(invocations)} pytest invocations, expected 2"
    for target, marker in (
        ("tests/test_session_stress.py", "stress"),
        ("tests/test_bolt_server_concurrency.py", "bolt_stress"),
    ):
        matching = [args for args in invocations if target in args]
        assert len(matching) == 1, f"scheduled stress does not run {target} exactly once"
        assert _markers(matching[0]) == [marker], f"{target} is not bounded to the {marker} marker"
    assert "tests/test_stress.py" not in _command_tokens(stress), "the large-runner suite must stay out of CI"
    large = (REPO_ROOT / "tests" / "test_stress.py").read_text(encoding="utf-8")
    assert "manual/large-runner" in large


def test_bolt_driver_conformance_installs_both_toolchains() -> None:
    """The suites skip when their toolchain is missing, which is right locally
    and useless in CI — a runner without a JDK would report green while never
    executing the Java driver at all. So CI must install both, and `-rs` must
    stay on the conformance invocation so any skip is visible in the log rather
    than silent."""
    job = _ci_job("bolt-driver-conformance")
    java = _steps_using(job, "actions/setup-java@")
    assert len(java) == 1, "bolt-driver-conformance does not set up a JDK"
    assert java[0]["with"]["distribution"] == "temurin"
    assert _steps_using(job, "actions/setup-node@"), "bolt-driver-conformance does not set up Node"
    _assert_runs(job, "cargo build --release -p kglite-bolt-server")

    conformance = [args for args in _pytest_invocations(job) if "tests/test_bolt_driver_conformance.py" in args]
    assert len(conformance) == 1, "expected exactly one conformance pytest invocation"
    args = conformance[0]
    assert _markers(args) == ["bolt"]
    # Asserted as a token of the invocation itself. As a substring of the job
    # text this passed on the `-rs` in the comment above the command, so the
    # flag could be dropped and a silent all-skipped run would read as green.
    assert "-rs" in args, "the conformance run does not pass -rs, so skips would be invisible"


def test_bolt_conformance_outage_classification_cannot_be_swallowed() -> None:
    """`assert_conformance_ran.py` distinguishes an infrastructure outage (exit
    2) from a product failure (exit 1) — both red, and neither may be tolerated
    into green.

    The distinction is only safe while the step that reports it can still fail
    the job. `continue-on-error` at either the step or the *job* level would
    make it decorative: the annotation would still print and the run would go
    green having tested no driver at all. That is the shape this whole job
    exists to prevent, so it is asserted rather than assumed.
    """
    job = _ci_job("bolt-driver-conformance")
    command = "python scripts/assert_conformance_ran.py conformance.xml"
    _assert_runs(job, command)
    assert _step_running(job, command).get("continue-on-error") is None, (
        "the conformance assertion tolerates failure — an outage or a skipped suite would read as green"
    )
    assert job.get("continue-on-error") is None, (
        "bolt-driver-conformance tolerates failure at the job level, which makes every gate inside it decorative"
    )
    tolerated = [step for step in _steps(job) if step.get("continue-on-error") is True]
    assert tolerated == [], "no step in the conformance job may tolerate failure"


# --- the publish workflows --------------------------------------------------
# This file guarded ci.yml only, so the publish workflows — the ones that
# actually decide what reaches PyPI and crates.io — had no local gate at all.
# That is the same shape as the gap that let an action-major bump fail the
# whole Python matrix: `make gate` runs no Python tests, so a workflow edit is
# unverified until CI.
#
# Everything guarded below is a property whose failure is *silent*: the run
# still goes green, and what is missing is an artifact, a wheel, or a publish
# that simply never happened. Loud failures (a build that errors, a `cargo
# publish` that rejects a duplicate) need no guard here — they announce
# themselves.

_WHEELS_PARAM = pytest.param(WHEELS, "build_wheels.yml", id="build_wheels")
_CLI_WHEELS_PARAM = pytest.param(CLI_WHEELS, "build_cli_wheels.yml", id="build_cli_wheels")
_CRATES_PARAM = pytest.param(CRATES, "publish_crates.yml", id="publish_crates")

PUBLISHING_WORKFLOWS = [_WHEELS_PARAM, _CLI_WHEELS_PARAM, _CRATES_PARAM]

#: The two artifact-producing wheel workflows. `publish_crates.yml` uploads
#: nothing — it publishes source crates straight from the checkout — so the
#: artifact guards below do not apply to it and must not be parametrized over
#: it, or they would pass on an empty scan. Listed rather than sliced out of
#: the set above: a slice that silently shrinks drops a whole workflow's
#: guards without failing anything.
WHEEL_WORKFLOWS = [_WHEELS_PARAM, _CLI_WHEELS_PARAM]

#: Artifact producers each `publish` job deliberately does NOT require to
#: succeed. Each entry is a conscious "this platform may silently not ship"
#: decision, not an oversight — cross-compiled aarch64 wheels are fragile and
#: were judged not worth blocking a release over. `build_cli_wheels.yml` goes
#: further and marks its arm job `continue-on-error` at *job* level, which
#: makes the job structurally unable to fail.
BEST_EFFORT_PRODUCERS = {
    "build_wheels.yml": {"build-linux-arm"},
    "build_cli_wheels.yml": {"build-linux-arm"},
}


_MATRIX_REF = re.compile(r"\$\{\{\s*matrix\.([A-Za-z0-9_-]+)\s*\}\}")


def _rendered_job_names(job: dict, fallback: str) -> list[str]:
    """Every check name a job produces, with ``${{ matrix.* }}`` substituted.

    GitHub names each matrix leg by rendering the job's ``name:`` against that
    leg's values; the unrendered template appears only when the job never ran.
    A guard matching the template alone could not tell the must-pass legs from
    the best-effort ones — which is the whole distinction the crates.io gate's
    ``check-regexp`` exists to make.

    A reference to a key the leg does not define is left as-is rather than
    blanked, so a typo shows up as an unmatched name instead of quietly
    rendering to something that matches.
    """
    template = job.get("name") or fallback
    includes = ((job.get("strategy") or {}).get("matrix") or {}).get("include") or [{}]
    return [
        _MATRIX_REF.sub(lambda match: str(values.get(match.group(1), match.group(0))), template) for values in includes
    ]


def _upload_steps(workflow: dict) -> list[tuple[str, dict]]:
    """``(job name, step)`` for every ``upload-artifact`` step in ``workflow``."""
    return [
        (job_name, step)
        for job_name, job in (workflow.get("jobs") or {}).items()
        for step in _steps_using(job, "actions/upload-artifact@")
    ]


def _uploaded_artifacts(workflow: dict) -> list[tuple[str, str | None]]:
    """``(job name, declared artifact name)`` for every upload step in ``workflow``.

    The name is returned exactly as declared, including the two shapes the
    regex this replaced could not see: a ``${{ }}`` template (which may contain
    spaces), and ``None`` when the step omits ``name:`` altogether — that
    upload lands under the default name ``artifact``, which no ``wheels-*``
    pattern matches. Both are *found* here and judged by the caller; a
    derivation that silently skipped them is how the old guard came to inspect
    nothing.
    """
    return [(job_name, (step.get("with") or {}).get("name")) for job_name, step in _upload_steps(workflow)]


def _artifact_producers(workflow: dict) -> set[str]:
    producers = {job_name for job_name, _ in _uploaded_artifacts(workflow)}
    assert producers, "no artifact-producing jobs found — the scan is broken"
    return producers


def _gated_successes(job: dict) -> set[str]:
    """Jobs whose ``success`` result the ``if:`` condition explicitly requires."""
    return set(re.findall(r"needs\.([a-zA-Z0-9_-]+)\.result\s*==\s*'success'", str(job.get("if", ""))))


@pytest.mark.parametrize("workflow, workflow_name", WHEEL_WORKFLOWS)
def test_publish_gates_on_every_artifact_producer(workflow: dict, workflow_name: str) -> None:
    """`publish` must require every producer to have SUCCEEDED.

    Listing a job in `needs` is not enough, and that distinction is the point.
    Both publish jobs are guarded by `if: always() && ...`, and `always()`
    neutralises the implicit needs-succeeded gate — so what actually decides
    is the chain of `needs.<job>.result == 'success'` terms in that condition.
    A producer present in `needs` but absent from the `if:` can fail while
    publish proceeds: the wheels ship and its artifact silently does not.

    Not hypothetical. `build_wheels.yml`'s sdist job was added to `needs` only,
    and its commit message claimed publish gated on it. It did not. Caught in
    review, which is why this reads the `if:` rather than `needs`.
    """
    publish = _job(workflow, workflow_name, "publish")

    declared = set(publish["needs"])
    assert declared, f"{publish.label} declares no `needs`"

    assert "always()" in publish["if"], (
        f"{publish.label} no longer uses `always()`, so `needs` gates it directly — "
        "this test reads the `if:` chain and would now be checking the wrong thing"
    )
    gated = _gated_successes(publish)
    producers = _artifact_producers(workflow)

    missing_needs = producers - declared
    assert not missing_needs, f"{publish.label} does not `needs` producer(s): {sorted(missing_needs)}"

    ungated = producers - gated - BEST_EFFORT_PRODUCERS[workflow_name]
    assert not ungated, (
        f"{publish.label} does not require producer(s) {sorted(ungated)} to succeed — add "
        "`needs.<job>.result == 'success'` to its `if:`, or record the job in "
        "BEST_EFFORT_PRODUCERS with the reason it may silently not ship"
    )


@pytest.mark.parametrize("workflow, workflow_name", WHEEL_WORKFLOWS)
def test_publish_collects_every_uploaded_artifact_name(workflow: dict, workflow_name: str) -> None:
    """Every uploaded artifact must match the pattern `publish` downloads.

    `download-artifact` silently returns nothing for a non-matching name, so
    an artifact named outside the pattern is dropped without an error. The
    sdist is deliberately named `wheels-sdist` for this reason, and every
    `build_cli_wheels.yml` upload carries the `cli-wheels-` prefix.

    The names are read from the parsed `with.name` of each upload step. The
    regex this replaced scanned for `^\\s+name:\\s*(wheels…)$`, which could not
    see the three real wheel producers (their names contain spaces, inside
    `${{ }}`) and — because the capture group itself began with `wheels` —
    could not express the stray name it existed to reject.
    """
    publish = _job(workflow, workflow_name, "publish")
    downloads = _steps_using(publish, "actions/download-artifact@")
    assert len(downloads) == 1, f"{publish.label} should download artifacts in exactly one step"
    pattern = downloads[0]["with"]["pattern"]
    assert pattern.endswith("*"), f"{publish.label} downloads a literal name, not a pattern: {pattern!r}"
    prefix = pattern[:-1]

    uploaded = _uploaded_artifacts(workflow)
    assert uploaded, "no uploaded artifact names found — the scan is broken"
    unnamed = [job_name for job_name, name in uploaded if name is None]
    assert not unnamed, (
        f"{unnamed} upload an artifact with no `name` — it defaults to `artifact`, "
        f"which {publish.label}'s pattern {pattern!r} never matches"
    )
    assert {job_name for job_name, _ in uploaded} == _artifact_producers(workflow)

    stray = [(job_name, name) for job_name, name in uploaded if name is not None and not name.startswith(prefix)]
    assert not stray, f"artifact name(s) {stray} do not match {publish.label}'s pattern {pattern!r}"


@pytest.mark.parametrize("workflow, workflow_name", WHEEL_WORKFLOWS)
def test_no_upload_can_contribute_an_empty_artifact(workflow: dict, workflow_name: str) -> None:
    """Every upload must set `if-no-files-found: error`.

    The action's default is `warn`. A build job that exits 0 without producing
    a wheel therefore uploads an empty artifact and stays green; publish then
    ships whatever it collected, and a platform is missing from the release
    with nothing in any log marked as a failure.
    """
    uploads = _upload_steps(workflow)
    assert uploads, f"no upload steps found in {workflow_name} — the scan is broken"
    lax = [
        (job_name, (step.get("with") or {}).get("name"))
        for job_name, step in uploads
        if (step.get("with") or {}).get("if-no-files-found") != "error"
    ]
    assert not lax, (
        f"{workflow_name} upload(s) {lax} do not set `if-no-files-found: error`, so an "
        "empty artifact uploads silently and the release ships without that platform"
    )


def test_sdist_is_built_and_proven_usable() -> None:
    """The sdist job must verify the artifact resolves, not just that it built.

    A source fallback that cannot build is worse than none: it turns a clear
    "no matching distribution" into a compile error inside a stranger's
    install log. Producing a tarball proves nothing about that.
    """
    sdist = _wheels_job("build-sdist")
    _assert_runs(sdist, "maturin sdist --out dist")
    _assert_runs(sdist, 'tar -xzf "$sdist" -C "$work"')
    _assert_runs(sdist, "cargo metadata --format-version 1 > /dev/null")
    _assert_runs(sdist, 'test "$(tar -tzf "$sdist" | grep -c \'/LICENSE$\')" -ge 1')


_SINGLE_QUOTED = re.compile(r"'[^']*'")


def _runs_prose_as_a_command(line: str) -> bool:
    """True if ``line`` carries a backtick the shell would execute.

    Single-quoted spans are dropped first (``echo '```'`` is literal text) and
    ``\\```  is an escaped backtick. Anything left is legacy command
    substitution — which inside a double-quoted ``echo`` runs the very example
    the message was quoting.
    """
    return "`" in _SINGLE_QUOTED.sub("", line).replace("\\`", "")


def test_no_workflow_script_executes_the_prose_it_prints() -> None:
    """No `run:` command may contain an unescaped backtick.

    `publish_crates.yml`'s index-propagation fallback printed
    ``echo "Proceeding anyway — `cargo publish` for the next crate…"``. Inside
    double quotes those backticks are command substitution, so the explanation
    ran a bare `cargo publish` against whatever package cargo picks by default
    — and the step still exited 0, because the status reported is `echo`'s.

    Backticks are never needed here: every real substitution in these files
    uses `$( )`. Scanned across every workflow, since this is a shell
    property, not a property of one file.
    """
    offenders = [
        (path.name, job_name, line)
        for path in sorted(WORKFLOWS.glob("*.yml"))
        for job_name, job in (_load_workflow(path).get("jobs") or {}).items()
        for line in _command_lines(job)
        if _runs_prose_as_a_command(line)
    ]
    assert not offenders, (
        "unescaped backticks run as command substitution; escape them (\\`) or use single quotes:\n  "
        + "\n  ".join(f"{name} `{job}`: {line}" for name, job, line in offenders)
    )


# --- the version probe every publish decision hangs off ---------------------

#: The anchored semver test a version read out of `Cargo.toml` must survive.
SEMVER_GUARD = "grep -Eq '^[0-9]+[.][0-9]+[.][0-9]+'"

_CARGO_VERSION_READ = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=\$\(grep .*Cargo\.toml *\| *cut ")
_QUOTED_SHELL_VAR = re.compile(r'"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?"')
_GITHUB_OUTPUT_WRITE = re.compile(r'echo "([A-Za-z_][A-Za-z0-9_]*)=[^"]*" >> \$GITHUB_OUTPUT')
_STEP_OUTPUT_REF = re.compile(r"\$\{\{\s*steps\.[A-Za-z0-9_-]+\.outputs\.([A-Za-z0-9_]+)\s*\}\}")


def _cargo_version_reads(job: dict) -> set[str]:
    """Variables assigned from the ``grep … Cargo.toml | cut`` version pipeline.

    That pipeline reports ``cut``'s exit status, which is always 0, so a grep
    matching nothing assigns the empty string and the job carries on green.
    Every variable found here goes on to steer a publish decision, so every
    one of them has to be validated before it is used.
    """
    return {match.group(1) for line in _command_lines(job) if (match := _CARGO_VERSION_READ.match(line))}


def _semver_validated_variables(job: dict) -> set[str]:
    """Variables the job pipes through the anchored semver ``grep -Eq``.

    Read from whole logical lines, so the prose explaining the guard cannot
    stand in for the guard (``_logical_lines`` drops ``#`` comments) and the
    variable must appear on the same line as the test that validates it.
    """
    return {name for line in _command_lines(job) if SEMVER_GUARD in line for name in _QUOTED_SHELL_VAR.findall(line)}


def _github_outputs_written(job: dict) -> set[str]:
    """Output names the job's scripts append to ``$GITHUB_OUTPUT``."""
    return {match.group(1) for line in _command_lines(job) for match in _GITHUB_OUTPUT_WRITE.finditer(line)}


@pytest.mark.parametrize("workflow, workflow_name", PUBLISHING_WORKFLOWS)
def test_version_probe_validates_the_version_it_read(workflow: dict, workflow_name: str) -> None:
    """A malformed version read must abort, not publish nothing quietly.

    `VERSION=$(grep -m1 '^version' Cargo.toml | cut -d '"' -f 2)` reports
    `cut`'s status — always 0. Rename the key, reformat the table, and the
    grep matches nothing, `VERSION` is empty, the registry probe queries a
    malformed URL, its non-404 answer reads as "already published", and the
    whole workflow goes green having published nothing at all.
    """
    check = _job(workflow, workflow_name, "version-check")

    read = _cargo_version_reads(check)
    assert read, f"{check.label} does not read a version out of Cargo.toml — the scan is broken"

    unvalidated = read - _semver_validated_variables(check)
    assert not unvalidated, (
        f"{check.label} reads {sorted(unvalidated)} from Cargo.toml without asserting it looks "
        f"like a version ({SEMVER_GUARD}…) — an empty read then drives every publish decision"
    )

    guard_steps = [step for step in _steps(check) if any(SEMVER_GUARD in line for line in _step_commands(step))]
    assert len(guard_steps) == 1, f"{check.label} has {len(guard_steps)} semver guards, expected exactly 1"
    assert "exit 1" in _step_commands(guard_steps[0]), (
        f"{check.label}'s semver guard does not `exit 1` — it reports the problem and continues"
    )


@pytest.mark.parametrize("workflow, workflow_name", PUBLISHING_WORKFLOWS)
def test_version_check_outputs_are_actually_written(workflow: dict, workflow_name: str) -> None:
    """Every declared `version-check` output must be written by a step.

    A job output wired to a step output no step writes resolves to the empty
    string — GitHub raises nothing. Downstream, `... == 'true'` is then false
    forever: the build jobs skip, the publish steps skip, and the run is green
    having shipped nothing. That is the same silent shape as the empty version
    above, one layer further along.
    """
    check = _job(workflow, workflow_name, "version-check")
    written = _github_outputs_written(check)
    assert written, f"{check.label} writes nothing to $GITHUB_OUTPUT — the scan is broken"

    declared = check.get("outputs") or {}
    assert declared, f"{check.label} declares no outputs"

    for name, expression in declared.items():
        referenced = _STEP_OUTPUT_REF.findall(str(expression))
        assert len(referenced) == 1, (
            f"{check.label} output {name} is not a single step-output reference: {expression!r}"
        )
        assert referenced[0] in written, (
            f"{check.label} declares output `{name}` from `{referenced[0]}`, which no step writes to "
            "$GITHUB_OUTPUT — it resolves to the empty string and every decision it drives silently "
            "becomes a no-op"
        )


# --- the CI gate that stands between a commit and a published artifact ------

_CONCLUSION_IS_SUCCESS = re.compile(r'"\$conclusion"\s*=\s*"success"')
_EXITS_NONZERO = re.compile(r"(?:^|;\s*)exit [1-9]")


def _ci_poll_commands(job: _Job) -> list[str]:
    """Commands of the single step that polls ci.yml's push-triggered run.

    Scoped to that one step so the three things a working gate needs — the
    query, the success comparison, and the non-zero exit — must live together.
    Spread across the job they could be satisfied by unrelated steps.
    """
    polling = [
        step for step in _steps(job) if any("actions/workflows/ci.yml/runs" in line for line in _step_commands(step))
    ]
    assert len(polling) == 1, f"{job.label} polls ci.yml runs in {len(polling)} steps, expected exactly 1"
    return _step_commands(polling[0])


def _assert_waits_for_ci_success(job: _Job) -> None:
    lines = _ci_poll_commands(job)
    query = [line for line in lines if "actions/workflows/ci.yml/runs" in line]
    assert len(query) == 1, f"{job.label} queries the ci.yml runs endpoint {len(query)} times, expected 1"
    assert "event=push" in query[0], (
        f"{job.label} does not restrict the CI lookup to `event=push`; a same-SHA pull-request "
        "run would be accepted as the release gate"
    )
    assert any(_CONCLUSION_IS_SUCCESS.search(line) for line in lines), (
        f"{job.label} never compares the run's conclusion to `success` — it would accept any "
        "completed run, including a failed one"
    )
    assert any(_EXITS_NONZERO.search(line) for line in lines), (
        f"{job.label} never exits non-zero, so a red CI run cannot stop the publish"
    )


def test_pypi_publishes_cannot_outrun_the_ci_gate() -> None:
    """Both wheel workflows must block PyPI on ci.yml, in the two shapes they use.

    `build_wheels.yml` gates through a separate `ci-gate` job named in the
    publish `if:` chain; `build_cli_wheels.yml` waits inline as the publish
    job's first step. Both are load-bearing and neither had a test.
    """
    wheels_publish = _wheels_job("publish")
    assert "ci-gate" in _gated_successes(wheels_publish), (
        "build_wheels.yml `publish` does not require `needs.ci-gate.result == 'success'`; "
        "under `always()` the builds run in parallel with CI, so this term is the only thing "
        "keeping an un-CI'd commit off PyPI"
    )
    _assert_waits_for_ci_success(_wheels_job("ci-gate"))

    cli_publish = _cli_wheels_job("publish")
    assert "needs.version-check.outputs.should_publish == 'true'" in cli_publish["if"], (
        "build_cli_wheels.yml `publish` does not require a publish decision, so it would run on every push to main"
    )
    _assert_waits_for_ci_success(cli_publish)

    # The inline wait only gates what runs after it.
    steps = _steps(cli_publish)
    waits = [
        index
        for index, step in enumerate(steps)
        if any("actions/workflows/ci.yml/runs" in line for line in _step_commands(step))
    ]
    uploads = [
        index
        for index, step in enumerate(steps)
        if str(step.get("uses", "")).startswith("pypa/gh-action-pypi-publish@")
    ]
    assert len(waits) == 1 and len(uploads) == 1, f"expected one CI wait and one PyPI upload, got {waits} and {uploads}"
    assert waits[0] < uploads[0], "build_cli_wheels.yml waits for CI *after* uploading to PyPI — the wait gates nothing"


# --- publish_crates.yml: five irreversible publishes ------------------------

#: Each `cargo publish` command in `publish_crates.yml`, and the
#: `version-check` output that must gate it. Publishing a crate version is
#: irreversible; yanking does not free the version number.
CRATE_PUBLISH_GATES = {
    "cargo publish -p kglite --locked": "should_publish_kglite",
    "cargo publish -p kglite-bolt-server --locked": "should_publish_bolt",
    "cargo publish -p kglite-mcp-server --locked": "should_publish_mcp",
    "cargo publish -p kglite-c --locked": "should_publish_c",
    "cargo publish -p kglite-cli --locked": "should_publish_cli",
}


def test_crates_publish_waits_for_ci() -> None:
    """crates.io publishes must be downstream of a green ci.yml run.

    Unlike the PyPI publish jobs this one carries no `always()`, so `needs`
    gates it directly — which means the guard here is that `ci-gate` is in
    `needs` and that `always()` has not appeared since. If it ever did, every
    `needs` term would stop gating and the `if:` (an OR of five publish
    decisions, none of them about CI) would let publish run on a red CI.
    """
    publish = _crates_job("publish")
    assert "always()" not in publish["if"], (
        "publish_crates.yml `publish` gained `always()`, which neutralises its `needs` gate — "
        "add explicit `needs.<job>.result == 'success'` terms to the `if:` if that is intended"
    )
    assert "ci-gate" in set(publish["needs"]), "publish_crates.yml `publish` does not wait on `ci-gate`"

    gate = _crates_job("ci-gate")
    _assert_waits_for_ci_success(gate)


def test_crates_gate_waits_on_the_must_pass_wheel_legs() -> None:
    """The wheel-matrix wait must actually match the must-pass legs.

    It selects checks by regexp against *rendered* matrix names. A renamed job
    or a widened negative lookahead leaves the regexp matching the wrong set,
    and a gate that waits on nothing is the same as no gate — while looking
    identical in the file.
    """
    gate = _crates_job("ci-gate")
    waits = _steps_using(gate, "lewagon/wait-on-check-action@")
    assert len(waits) == 1, f"publish_crates.yml `ci-gate` has {len(waits)} check waits, expected 1"
    settings = waits[0]["with"]
    regexp = re.compile(settings["check-regexp"])

    must_pass = [
        name
        for job_name in ("build-native", "build-linux")
        for name in _rendered_job_names(_wheels_job(job_name), job_name)
    ]
    assert must_pass, "no must-pass wheel legs derived — the scan is broken"
    unmatched = [name for name in must_pass if not regexp.match(name)]
    assert not unmatched, (
        f"the crates.io gate's check-regexp {settings['check-regexp']!r} does not match must-pass "
        f"wheel leg(s) {unmatched} — it would wait on a smaller set than it appears to"
    )

    best_effort = _rendered_job_names(_wheels_job("build-linux-arm"), "build-linux-arm")
    assert best_effort
    blocking = [name for name in best_effort if regexp.match(name)]
    assert not blocking, (
        f"the crates.io gate waits on best-effort leg(s) {blocking}; those are `continue-on-error` "
        "and their failure must not wedge an irreversible crates.io publish"
    )

    allowed = {conclusion.strip() for conclusion in settings["allowed-conclusions"].split(",")}
    assert not allowed & {"failure", "cancelled", "timed_out"}, (
        f"the crates.io gate accepts {sorted(allowed)} as a pass — a failed wheel build would not block the publish"
    )


def test_every_crates_io_publish_is_gated_on_its_own_version_check() -> None:
    """Each of the five publishes must be gated by its own `should_publish_*`.

    Two silent shapes live here. An *ungated* `cargo publish` is loud (cargo
    rejects a duplicate version), but a publish gated on an output name that
    `version-check` does not declare is not: the expression resolves to the
    empty string, the step is skipped every single run, and the crate quietly
    stops being released. So the gate names are checked against what
    `version-check` actually declares, in both directions.
    """
    publish = _crates_job("publish")
    version_check = _crates_job("version-check")
    declared = set(version_check.get("outputs") or {})

    found = sorted(line for line in _command_lines(publish) if line.startswith("cargo publish"))
    assert found == sorted(CRATE_PUBLISH_GATES), (
        f"publish_crates.yml publishes {found}, but this test knows {sorted(CRATE_PUBLISH_GATES)} — "
        "a crate was added or removed and its gate is unverified"
    )

    for command, output in CRATE_PUBLISH_GATES.items():
        step = _step_running(publish, command)
        assert f"needs.version-check.outputs.{output} == 'true'" in str(step.get("if", "")), (
            f"`{command}` is not gated on needs.version-check.outputs.{output}"
        )
        assert output in declared, (
            f"`{command}` is gated on needs.version-check.outputs.{output}, which version-check does "
            "not declare — the expression is always empty, so this crate is never published"
        )

    orphaned = {name for name in declared if name.startswith("should_publish")} - set(CRATE_PUBLISH_GATES.values())
    assert not orphaned, (
        f"version-check computes {sorted(orphaned)} but no `cargo publish` step consumes it — that "
        "crate's version is probed against crates.io and then never published"
    )


# --- self-tests for the derivation helpers ----------------------------------
# Every guard above inspects only what the helpers at the top of this file
# derive. A helper that quietly stops matching — a moved key, a narrowed
# regex, an invocation form it no longer recognises — leaves each test built
# on it green while it examines nothing; that is precisely how the artifact
# guard died with 19 green tests around it. The tests below pin each helper
# against synthetic workflow text in both directions: the shape is extracted
# when present, and is *not* extracted when absent or merely mentioned in a
# comment. A rule that always fires is as useless as one that never does.

SYNTHETIC_WORKFLOW = """
name: synthetic
on:
  push:
jobs:
  build-wheels:
    steps:
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Test
        run: |
          set -euo pipefail
          # pytest tests/test_commented_out.py -m ghost
          pytest tests/test_real.py -m parity \\
              --benchmark-json=out.json
          KGLITE_LOG=debug .venv/bin/pytest tests/test_env_prefixed.py::test_one -m stress
          python -m pytest -m bolt -rs tests/test_module_form.py
          python -m mypy kglite
      - name: Upload wheels as artifact
        uses: actions/upload-artifact@v7
        with:
          name: wheels-${{ matrix.os }} ${{ matrix.target }}
          path: wheels/*.whl
      - name: Upload logs under no name at all
        uses: actions/upload-artifact@v7
        with:
          path: logs/
  build-sdist:
    steps:
      - uses: actions/upload-artifact@v7
        with:
          name: wheels-sdist
  lint:
    steps:
      - run: cargo fmt --check
"""

#: A workflow with jobs and steps but no uploads at all — the shape every
#: "the scan is broken" guard exists to tell apart from a clean scan.
UPLOADLESS_WORKFLOW = {"jobs": {"lint": {"steps": [{"run": "cargo fmt --check"}]}}}


def _synthetic() -> dict:
    return yaml.safe_load(SYNTHETIC_WORKFLOW)


def _synthetic_job(name: str = "build-wheels") -> _Job:
    return _job(_synthetic(), "synthetic.yml", name)


def test_helper_load_workflow_requires_a_parsed_mapping(tmp_path: Path) -> None:
    good = tmp_path / "good.yml"
    good.write_text("jobs:\n  lint:\n    steps: []\n", encoding="utf-8")
    assert _load_workflow(good)["jobs"] == {"lint": {"steps": []}}

    bad = tmp_path / "bad.yml"
    bad.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        _load_workflow(bad)


def test_helper_job_and_step_lookup_stay_scoped_to_the_named_job() -> None:
    workflow = _synthetic()
    assert _steps(_job(workflow, "synthetic.yml", "lint")) == [{"run": "cargo fmt --check"}]
    with pytest.raises(AssertionError):
        _job(workflow, "synthetic.yml", "no-such-job")
    with pytest.raises(AssertionError):
        _job({"jobs": {"lint": "not-a-mapping"}}, "synthetic.yml", "lint")

    build = _synthetic_job()
    assert len(_steps_using(build, "actions/upload-artifact@")) == 2
    assert _steps_using(build, "actions/setup-python@")[0]["with"]["python-version"] == "3.12"
    # Absent action, and an action present in a *different* job: neither may
    # leak into this job's step list.
    assert _steps_using(build, "actions/setup-java@") == []
    assert _steps_using(_synthetic_job("lint"), "actions/upload-artifact@") == []


def test_helper_logical_lines_drop_comments_and_join_continuations() -> None:
    script = "# cargo test --release\n\n  cargo test \\\n      --release\nsleep 30\n"
    assert list(_logical_lines(script)) == ["cargo test --release", "sleep 30"]
    # A continuation that runs off the end of the script is still a command.
    assert list(_logical_lines("echo a \\\n")) == ["echo a"]
    # Steps with no `run:` contribute nothing rather than raising.
    assert _step_commands({"uses": "actions/checkout@v5"}) == []
    assert _step_commands({"run": "cargo fmt --check"}) == ["cargo fmt --check"]
    assert _command_lines(_synthetic_job("lint")) == ["cargo fmt --check"]
    assert "python -m mypy kglite" in _command_lines(_synthetic_job())


def test_helper_command_assertions_reject_absent_prefix_and_commented_commands() -> None:
    lint = _synthetic_job("lint")
    _assert_runs(lint, "cargo fmt --check")
    with pytest.raises(AssertionError):
        _assert_runs(lint, "cargo clippy")
    # Whole-line equality: a strict prefix of a real command must not satisfy
    # the assertion, which is the subsumption that kept a deleted check green.
    with pytest.raises(AssertionError):
        _assert_runs(lint, "cargo fmt")

    build = _synthetic_job()
    # Present in the script, but only inside a `#` comment.
    with pytest.raises(AssertionError):
        _assert_runs(build, "pytest tests/test_commented_out.py -m ghost")

    assert _step_running(build, "python -m mypy kglite")["name"] == "Test"
    with pytest.raises(AssertionError):
        _step_running(build, "python -m mypy")
    with pytest.raises(AssertionError):
        _step_running(build, "cargo fmt --check")


def test_helper_tokens_are_exact_arguments() -> None:
    assert _tokens('python x.py --flag "a b"') == ["python", "x.py", "--flag", "a b"]
    # Unbalanced quotes fall back to a whitespace split instead of raising.
    assert _tokens("echo 'unterminated") == ["echo", "'unterminated"]

    tokens = _command_tokens(_synthetic_job())
    assert "-rs" in tokens
    assert "-r" not in tokens, "a flag must not be found inside a longer token"
    assert "tests/test_real.py" in tokens
    assert "tests/test_commented_out.py" not in tokens


def test_helper_pytest_invocations_cover_every_launcher_form() -> None:
    invocations = _pytest_invocations(_synthetic_job())
    assert len(invocations) == 3, f"expected the three real launcher forms, got {invocations}"
    assert [_markers(args) for args in invocations] == [["parity"], ["stress"], ["bolt"]]

    # `pytest …`, with the continuation joined so its flags belong to it.
    assert "tests/test_real.py" in invocations[0]
    assert "--benchmark-json=out.json" in invocations[0]
    # `VAR=value .venv/bin/pytest …`, node id preserved verbatim.
    assert "tests/test_env_prefixed.py::test_one" in invocations[1]
    # `python -m pytest …`, with `-m pytest` itself not read as a marker.
    assert "tests/test_module_form.py" in invocations[2]
    assert "pytest" not in _markers(invocations[2])

    # Non-pytest commands, comments, and jobs without any pytest run.
    flattened = [arg for args in invocations for arg in args]
    assert "kglite" not in flattened, "`python -m mypy kglite` was read as a pytest invocation"
    assert "tests/test_commented_out.py" not in flattened
    assert _pytest_invocations(_synthetic_job("lint")) == []

    assert _markers([]) == []
    assert _markers(["-m"]) == [], "a trailing -m with no expression must not be read as a marker"


def test_helper_artifact_derivation_sees_templated_and_unnamed_uploads() -> None:
    """The case whose absence made the previous artifact guard vacuous.

    A name inside `${{ }}` (with spaces) and an upload step with no `name:` at
    all are exactly what the old regex could not see; both must be *found*
    here, so the guard above can judge them.
    """
    uploaded = _uploaded_artifacts(_synthetic())
    assert uploaded == [
        ("build-wheels", "wheels-${{ matrix.os }} ${{ matrix.target }}"),
        ("build-wheels", None),
        ("build-sdist", "wheels-sdist"),
    ]
    assert _artifact_producers(_synthetic()) == {"build-wheels", "build-sdist"}

    # A workflow whose jobs upload nothing derives nothing — which is what
    # makes the guard below meaningful rather than decorative.
    assert _uploaded_artifacts(UPLOADLESS_WORKFLOW) == []
    assert _uploaded_artifacts({}) == []


def test_helper_empty_scan_guard_fires_instead_of_reporting_clean() -> None:
    """`assert producers, "the scan is broken"` must actually be reachable.

    Several guards above are vacuous the moment the derivation returns
    nothing, so the non-empty assertion is the only thing standing between a
    blinded helper and a wall of green tests. It has to fire.
    """
    for workflow in (UPLOADLESS_WORKFLOW, {"jobs": {}}, {}):
        with pytest.raises(AssertionError, match="the scan is broken"):
            _artifact_producers(workflow)


# --- self-tests for the publish-workflow helpers ----------------------------

SYNTHETIC_PUBLISH_WORKFLOW = r"""
name: synthetic-publish
on:
  push:
jobs:
  version-check:
    outputs:
      should_publish: ${{ steps.check.outputs.should_publish }}
      ghost: ${{ steps.check.outputs.never_written }}
      literal: not-a-step-output
    steps:
      - id: check
        run: |
          VERSION=$(grep -m 1 '^version' Cargo.toml | cut -d '"' -f 2)
          UNCHECKED=$(grep -m 1 '^other' Cargo.toml | cut -d '"' -f 2)
          # if ! printf '%s\n' "$UNCHECKED" | grep -Eq '^[0-9]+[.][0-9]+[.][0-9]+'; then
          if ! printf '%s\n' "$VERSION" | grep -Eq '^[0-9]+[.][0-9]+[.][0-9]+'; then
            echo "::error::bad version"
            exit 1
          fi
          echo "should_publish=true" >> $GITHUB_OUTPUT
  build:
    name: Build wheel - ${{ matrix.os }} - ${{ matrix.target }} (${{ matrix.absent }})
    strategy:
      matrix:
        include:
          - os: linux
            target: aarch64-unknown-linux-gnu
          - os: linux
            target: x86_64-unknown-linux-gnu
    steps:
      - uses: actions/upload-artifact@v7
        with:
          name: wheels-strict
          if-no-files-found: error
      - uses: actions/upload-artifact@v7
        with:
          name: wheels-lax
  publish:
    name: Publish
    needs: [version-check, build]
    if: always() && needs.build.result == 'success' && needs.other.result != 'success'
    steps:
      - name: Wait for CI to pass
        run: |
          # gh api "repos/$REPO/actions/workflows/ci.yml/runs?head_sha=$SHA" -q '.x'
          line=$(gh api "repos/$REPO/actions/workflows/ci.yml/runs?head_sha=$SHA&event=push" -q '.status')
          [ "$conclusion" = "success" ] && { echo ok; exit 0; }
          echo bad; exit 1
      - uses: pypa/gh-action-pypi-publish@release/v1
  lint:
    steps:
      - run: cargo fmt --check
"""


def _synthetic_publish() -> dict:
    return yaml.safe_load(SYNTHETIC_PUBLISH_WORKFLOW)


def _synthetic_publish_job(name: str) -> _Job:
    return _job(_synthetic_publish(), "synthetic-publish.yml", name)


def test_helper_upload_steps_and_rendered_names_are_derived_per_job() -> None:
    uploads = _upload_steps(_synthetic_publish())
    assert [(job_name, step["with"].get("if-no-files-found")) for job_name, step in uploads] == [
        ("build", "error"),
        ("build", None),
    ]
    assert _upload_steps(UPLOADLESS_WORKFLOW) == []

    # Matrix values are substituted per leg; a reference to a key no leg
    # defines is left visible rather than blanked.
    assert _rendered_job_names(_synthetic_publish_job("build"), "build") == [
        "Build wheel - linux - aarch64-unknown-linux-gnu (${{ matrix.absent }})",
        "Build wheel - linux - x86_64-unknown-linux-gnu (${{ matrix.absent }})",
    ]
    # A job with no matrix renders exactly one name, and a job with no `name:`
    # falls back to its key rather than to None.
    assert _rendered_job_names(_synthetic_publish_job("publish"), "publish") == ["Publish"]
    assert _rendered_job_names(_synthetic_publish_job("lint"), "lint") == ["lint"]


def test_helper_backtick_detection_tells_substitution_from_literal_text() -> None:
    # Runs as a command: bare, and inside double quotes.
    assert _runs_prose_as_a_command("VERSION=`cat VERSION`")
    assert _runs_prose_as_a_command('echo "Proceeding — `cargo publish` will surface it"')
    # Literal text: single-quoted (ci.yml fences a code block this way) and
    # backslash-escaped inside double quotes.
    assert not _runs_prose_as_a_command("echo '```'")
    assert not _runs_prose_as_a_command("echo 'run `cargo publish` yourself'")
    assert not _runs_prose_as_a_command('echo "Proceeding — \\`cargo publish\\` will surface it"')
    assert not _runs_prose_as_a_command("cargo publish -p kglite --locked")


def test_helper_gated_successes_reads_only_success_terms() -> None:
    assert _gated_successes(_synthetic_publish_job("publish")) == {"build"}
    # `!= 'success'` is not a gate, and a job with no `if:` gates nothing.
    assert "other" not in _gated_successes(_synthetic_publish_job("publish"))
    assert _gated_successes(_synthetic_publish_job("lint")) == set()


def test_helper_version_probe_derivations_see_the_read_and_the_guard() -> None:
    check = _synthetic_publish_job("version-check")

    # Both `grep … Cargo.toml | cut` reads are found — including the one the
    # workflow forgot to validate, which is the whole point of finding them.
    assert _cargo_version_reads(check) == {"VERSION", "UNCHECKED"}
    assert _cargo_version_reads(_synthetic_publish_job("lint")) == set()

    # Only the real guard counts; the commented-out one above it does not.
    assert _semver_validated_variables(check) == {"VERSION"}
    assert _cargo_version_reads(check) - _semver_validated_variables(check) == {"UNCHECKED"}
    assert _semver_validated_variables(_synthetic_publish_job("lint")) == set()

    assert _github_outputs_written(check) == {"should_publish"}
    assert _github_outputs_written(_synthetic_publish_job("lint")) == set()
    # `ghost` is declared from a step output nothing writes — the shape the
    # guard above exists to reject must actually be visible here.
    declared = check["outputs"]
    assert _STEP_OUTPUT_REF.findall(declared["ghost"]) == ["never_written"]
    assert _STEP_OUTPUT_REF.findall(declared["literal"]) == []


def test_helper_ci_wait_assertions_are_scoped_and_can_fail() -> None:
    publish = _synthetic_publish_job("publish")
    lines = _ci_poll_commands(publish)
    # The commented-out query does not count as the poll, so exactly one
    # command line carries it.
    assert len([line for line in lines if "actions/workflows/ci.yml/runs" in line]) == 1
    _assert_waits_for_ci_success(publish)

    # A job that never polls ci.yml must raise rather than pass vacuously.
    with pytest.raises(AssertionError, match="polls ci.yml runs in 0 steps"):
        _ci_poll_commands(_synthetic_publish_job("lint"))

    # Each of the three requirements must be independently able to fail.
    for script, expected in (
        (
            'line=$(gh api "repos/x/actions/workflows/ci.yml/runs?head_sha=$SHA")\n'
            '[ "$conclusion" = "success" ] || exit 1\n',
            "event=push",
        ),
        (
            'line=$(gh api "…/actions/workflows/ci.yml/runs?event=push")\n[ "$status" = "completed" ] || exit 1\n',
            "conclusion",
        ),
        (
            'line=$(gh api "…/actions/workflows/ci.yml/runs?event=push")\n[ "$conclusion" = "success" ] && exit 0\n',
            "exits non-zero",
        ),
    ):
        broken = _Job("synthetic.yml `broken`", {"steps": [{"run": script}]})
        with pytest.raises(AssertionError, match=expected):
            _assert_waits_for_ci_success(broken)


# --- job-level continue-on-error (doctrine RULES.md R1) ----------------------
#
# A *step*-level `continue-on-error` tolerates one fragile command. A
# *job*-level one makes the entire job structurally incapable of turning the
# build red — every gate inside it becomes decorative. `minimal-versions`
# carried one for two days while exiting 101, and two false claims were
# committed on its green before independent review caught it.
#
# So each one is an entry in `.github/workflows/doctrine-allowlist.txt`, with
# the reason and the compensating check written down. The allowlist is checked
# in *both* directions: an unlisted flag is a failure, and so is a listed job
# that no longer carries the flag — an exemption that outlives its reason is
# how a temporary tolerance becomes permanent blindness.

DOCTRINE_ALLOWLIST = WORKFLOWS / "doctrine-allowlist.txt"


def _parse_allowlist(text: str) -> set[str]:
    entries = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            entries.add(line)
    return entries


def _jobs_with_job_level_continue_on_error(paths: list[Path]) -> set[str]:
    found = set()
    scanned = 0
    for path in paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for name, job in (data.get("jobs") or {}).items():
            scanned += 1
            if isinstance(job, dict) and job.get("continue-on-error") not in (None, False):
                found.add(f"{path.name}:{name}")
    assert scanned, f"scanned no jobs across {[p.name for p in paths]} — the scan is broken"
    return found


def test_job_level_continue_on_error_is_allowlisted_and_the_allowlist_is_current() -> None:
    workflows = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
    assert workflows, "found no workflow files — an empty scan is not a pass"
    flagged = _jobs_with_job_level_continue_on_error(workflows)
    allowed = _parse_allowlist(DOCTRINE_ALLOWLIST.read_text(encoding="utf-8"))

    unlisted = sorted(flagged - allowed)
    assert not unlisted, (
        f"job-level continue-on-error without an allowlist entry: {unlisted}. "
        f"These jobs cannot turn the build red, so any gate inside them is "
        f"decorative. Narrow the flag to the fragile steps, or add the entry to "
        f"{DOCTRINE_ALLOWLIST.name} with its reason and compensating check."
    )
    stale = sorted(allowed - flagged)
    assert not stale, (
        f"{DOCTRINE_ALLOWLIST.name} exempts jobs that no longer carry a "
        f"job-level continue-on-error: {stale}. Delete the entries — an "
        f"exemption that outlives its reason is how a temporary tolerance "
        f"becomes permanent blindness."
    )
    # The allowlist must not be a bare list: each entry is a decision, and a
    # decision with no recorded reason cannot be reviewed later.
    assert "COMPENSATING CHECK" in DOCTRINE_ALLOWLIST.read_text(encoding="utf-8")


def test_helper_continue_on_error_scan_and_allowlist_can_fail(tmp_path: Path) -> None:
    # Each branch must be independently able to fail, or the guard above is
    # only accidentally green.
    clean = tmp_path / "clean.yml"
    clean.write_text("jobs:\n  build:\n    runs-on: ubuntu-latest\n", encoding="utf-8")
    assert _jobs_with_job_level_continue_on_error([clean]) == set()

    flagged = tmp_path / "flagged.yml"
    flagged.write_text(
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n    continue-on-error: true\n",
        encoding="utf-8",
    )
    assert _jobs_with_job_level_continue_on_error([flagged]) == {"flagged.yml:build"}

    # A step-level flag is the tolerated shape and must NOT be reported, or the
    # guard is unusable and gets deleted.
    step_level = tmp_path / "step.yml"
    step_level.write_text(
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - run: flaky\n        continue-on-error: true\n",
        encoding="utf-8",
    )
    assert _jobs_with_job_level_continue_on_error([step_level]) == set()

    # A workflow with no jobs at all is a broken scan, not a clean result.
    empty = tmp_path / "empty.yml"
    empty.write_text("name: nothing\non: [push]\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="scanned no jobs"):
        _jobs_with_job_level_continue_on_error([empty])

    # Comments and blank lines are not entries; real entries are.
    assert _parse_allowlist("# just a comment\n\n  a.yml:job  # trailing\n") == {"a.yml:job"}
    assert _parse_allowlist("# only comments\n") == set()
