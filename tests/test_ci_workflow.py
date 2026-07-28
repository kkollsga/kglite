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

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CI_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
WHEELS_PATH = REPO_ROOT / ".github" / "workflows" / "build_wheels.yml"

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
}


def _load_workflow(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path} did not parse as a YAML mapping"
    return data


CI = _load_workflow(CI_PATH)
WHEELS = _load_workflow(WHEELS_PATH)


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


def test_required_verification_jobs_are_aggregated() -> None:
    needs = set(_ci_job("ci-success")["needs"])
    for job in REQUIRED_JOBS:
        _ci_job(job)
        assert job in needs, f"ci-success.needs does not include {job}"


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


def test_perf_regression_is_part_of_the_aggregate_gate() -> None:
    assert "perf-regression" in _ci_job("ci-success")["needs"]


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


# --- build_wheels.yml -------------------------------------------------------
# This file guarded ci.yml only, so the publish workflow — the one that
# actually decides what reaches PyPI — had no local gate at all. That is the
# same shape as the gap that let an action-major bump fail the whole Python
# matrix: `make gate` runs no Python tests, so a workflow edit is unverified
# until CI. The sdist job below is the first thing to depend on it.

#: Artifact producers `publish` deliberately does NOT require to succeed.
#: Each entry is a conscious "this platform may silently not ship" decision,
#: not an oversight — cross-compiled aarch64 wheels are fragile and were
#: judged not worth blocking a release over.
BEST_EFFORT_PRODUCERS = {"build-linux-arm"}


def _artifact_producers() -> set[str]:
    producers = {name for name, job in WHEELS["jobs"].items() if _steps_using(job, "actions/upload-artifact@")}
    assert producers, "no artifact-producing jobs found — the scan is broken"
    return producers


def test_publish_gates_on_every_artifact_producer() -> None:
    """`publish` must require every producer to have SUCCEEDED.

    Listing a job in `needs` is not enough, and that distinction is the point.
    The publish job is guarded by `if: always() && ...`, and `always()`
    neutralises the implicit needs-succeeded gate — so what actually decides
    is the chain of `needs.<job>.result == 'success'` terms in that condition.
    A producer present in `needs` but absent from the `if:` can fail while
    publish proceeds: the wheels ship and its artifact silently does not.

    Not hypothetical. The sdist job was added to `needs` only, and its commit
    message claimed publish gated on it. It did not. Caught in review, which
    is why this reads the `if:` rather than `needs`.
    """
    publish = _wheels_job("publish")

    declared = set(publish["needs"])
    assert declared, "publish job declares no `needs`"

    gated = set(re.findall(r"needs\.([a-zA-Z0-9_-]+)\.result\s*==\s*'success'", publish["if"]))

    producers = _artifact_producers()

    missing_needs = producers - declared
    assert not missing_needs, f"publish does not `needs` producer(s): {sorted(missing_needs)}"

    ungated = producers - gated - BEST_EFFORT_PRODUCERS
    assert not ungated, (
        f"publish does not require producer(s) {sorted(ungated)} to succeed — add "
        "`needs.<job>.result == 'success'` to its `if:`, or record the job in "
        "BEST_EFFORT_PRODUCERS with the reason it may silently not ship"
    )


def test_publish_collects_every_uploaded_artifact_name() -> None:
    """Every uploaded artifact must match the pattern `publish` downloads.

    `download-artifact` silently returns nothing for a non-matching name, so
    an artifact named outside the pattern is dropped without an error. The
    sdist is deliberately named `wheels-sdist` for this reason.

    The names are read from the parsed `with.name` of each upload step. The
    regex this replaced scanned for `^\\s+name:\\s*(wheels…)$`, which could not
    see the three real wheel producers (their names contain spaces, inside
    `${{ }}`) and — because the capture group itself began with `wheels` —
    could not express the stray name it existed to reject.
    """
    publish = _wheels_job("publish")
    downloads = _steps_using(publish, "actions/download-artifact@")
    assert len(downloads) == 1, "publish should download artifacts in exactly one step"
    pattern = downloads[0]["with"]["pattern"]
    assert pattern.endswith("*"), f"publish downloads a literal name, not a pattern: {pattern!r}"
    prefix = pattern[:-1]

    uploaded: list[tuple[str, str]] = []
    for job_name, job in WHEELS["jobs"].items():
        for step in _steps_using(job, "actions/upload-artifact@"):
            name = (step.get("with") or {}).get("name")
            assert name is not None, (
                f"{job_name} uploads an artifact with no `name` — it defaults to `artifact`, "
                f"which publish's pattern {pattern!r} never matches"
            )
            uploaded.append((job_name, name))
    assert uploaded, "no uploaded artifact names found — the scan is broken"
    assert {job_name for job_name, _ in uploaded} == _artifact_producers()

    stray = [(job_name, name) for job_name, name in uploaded if not name.startswith(prefix)]
    assert not stray, f"artifact name(s) {stray} do not match publish's pattern {pattern!r}"


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
