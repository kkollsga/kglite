"""Hermetic self-tests for required CI verification gates.

These checks intentionally inspect the workflow as text instead of importing a
YAML parser: PyYAML treats the YAML 1.1 key ``on`` as a boolean, and CI's
configuration contract here is the job graph and exact verification commands.
"""

from __future__ import annotations

from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parent.parent
CI_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
CI_TEXT = CI_PATH.read_text(encoding="utf-8")

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


def _job_block(job: str) -> str:
    match = re.search(rf"(?ms)^  {re.escape(job)}:\n(.*?)(?=^  [a-zA-Z0-9_-]+:\n|\Z)", CI_TEXT)
    assert match is not None, f"missing required CI job: {job}"
    return match.group(1)


def test_required_verification_jobs_are_aggregated() -> None:
    success = _job_block("ci-success")
    for job in REQUIRED_JOBS:
        _job_block(job)
        assert f"- {job}" in success, f"ci-success.needs does not include {job}"


def test_storage_and_disk_jobs_run_bounded_regression_targets() -> None:
    parity = _job_block("storage-parity")
    assert "pytest -m parity" in parity
    for target in (
        "tests/test_storage_parity.py",
        "tests/test_phase1_parity.py",
        "tests/test_phase2_parity.py",
        "tests/test_phase3_parity.py",
        "tests/test_phase4_parity.py",
        "tests/test_phase5_parity.py::test_graph_copy_cow_correctness_memory",
        "tests/test_phase5_parity.py::test_graph_copy_cow_correctness_mapped",
    ):
        assert target in parity

    disk = _job_block("disk-concurrency")
    assert "test_concurrent_disk_reads_keep_materialized_nodes_alive" in disk
    assert "test_disk_writer_lease_is_enforced_across_processes" in disk
    assert "test_disk_session_reuses_writer_lineage_and_composes" in disk


def test_python_job_builds_the_measured_release_extension() -> None:
    python_tests = _job_block("python-tests")
    assert "maturin build --release --out /tmp/kglite-binary-size-wheel" in python_tests
    assert "cargo build --release -p kglite-mcp-server -p kglite-bolt-server" in python_tests


def test_free_threading_uses_the_pyo3_supported_python() -> None:
    free_threading = _job_block("free-threading")
    assert "python-version: '3.14t'" in free_threading
    assert "python-version: '3.13t'" not in free_threading


def test_linux_perf_gate_uses_isolated_released_wheel_reference() -> None:
    perf = _job_block("perf-regression")
    assert '"kglite==$REFERENCE_VERSION"' in perf
    assert "--only-binary=:all:" in perf
    assert "test_bench_core.py" in perf
    assert "sleep 30" in perf
    assert '.bench-reference-0.13.2.json "$1"' in perf
    assert "--require-exact-set" in perf
    # Unversioned for the same reason as the conformance job below: the real
    # requirement is the `include-hidden-files` setting asserted next (the
    # baseline JSON is a dotfile and silently would not upload without it),
    # not any particular release of upload-artifact.
    assert "actions/upload-artifact@" in perf
    assert "include-hidden-files: true" in perf
    assert "scripts/benchmark_provenance.py" in perf
    assert 'tests/benchmarks/baselines/current.linux.json "$1"' in perf
    assert perf.count("--require-exact-set") == 2
    # Retry-once contract: a first-capture regression verdict triggers exactly
    # one recapture; only a repeated failure is red, and both captures ship in
    # the evidence artifact (which must survive a failed verdict).
    assert "compare .bench-candidate.json" in perf
    assert "compare .bench-candidate-retry.json" in perf
    assert perf.count("--benchmark-json") >= 1
    assert ".bench-candidate-retry.json" in perf
    assert "if: always()" in perf


def test_perf_regression_is_part_of_the_aggregate_gate() -> None:
    assert "- perf-regression" in _job_block("ci-success")


def test_loom_and_unsafe_jobs_use_the_intended_commands() -> None:
    loom = _job_block("loom-session")
    assert 'RUSTFLAGS="--cfg loom" cargo test -p kglite --test loom_session' in loom

    miri = _job_block("miri-loaders")
    assert "cargo miri test -p kglite --lib" in miri
    assert "packed_primitives_decode_from_misaligned_little_endian_bytes" in miri
    assert "parse_line_borrowed_uri_boundaries_are_valid" in miri
    assert "parse_line_preserves_utf8_literal_boundaries" in miri

    asan = _job_block("address-sanitizer")
    assert "RUSTFLAGS: -Zsanitizer=address" in asan
    assert "overlapping_query_guards_keep_materializations_alive" in asan


def test_heavy_thread_sanitizer_is_scheduled_only() -> None:
    scheduled = _job_block("scheduled-thread-sanitizer")
    assert "if: github.event_name == 'schedule'" in scheduled
    assert "RUSTFLAGS: -Zsanitizer=thread" in scheduled
    assert "schedule:" in CI_TEXT


def test_live_github_smoke_requires_explicit_opt_in() -> None:
    smoke = (REPO_ROOT / "tests" / "test_mcp_server_smoke.py").read_text(encoding="utf-8")
    assert 'os.environ.get("KGLITE_GITHUB_INTEGRATION") == "1"' in smoke
    assert "and GITHUB_TOKEN is not None" in smoke
    assert smoke.count("not _github_live_enabled()") == 2


def test_docs_job_checks_generated_facts_and_warnings() -> None:
    docs = _job_block("docs")
    assert "python scripts/render_docs_facts.py --check" in docs
    assert "sphinx-build -W --keep-going -b html docs docs/_build/html" in docs
    assert "myst.xref_missing" not in (REPO_ROOT / "docs" / "conf.py").read_text(encoding="utf-8")


def test_source_quality_runs_once_in_its_own_required_job() -> None:
    source_quality = _job_block("source-quality")
    assert "python scripts/check_source_quality.py --self-test" in source_quality
    assert "python scripts/check_source_quality.py" in source_quality
    assert "python scripts/check_lint_allowances.py --self-test" in source_quality
    assert "python scripts/check_lint_allowances.py" in source_quality
    assert "python scripts/check_rustsec_advisories.py --policy-only" in source_quality
    python_tests = _job_block("python-tests")
    assert "check_source_quality.py" not in python_tests
    assert "check_lint_allowances.py" not in python_tests


def test_rustsec_audit_is_required_and_pinned() -> None:
    audit = _job_block("rustsec-audit")
    assert "if: github.event_name == 'schedule'" not in audit
    assert "cargo-audit@0.22.2" in audit
    assert "python scripts/check_rustsec_advisories.py" in audit
    assert "--policy-only" not in audit
    assert "- rustsec-audit" in _job_block("ci-success")


def test_every_ci_job_has_a_wall_clock_timeout() -> None:
    jobs = re.findall(r"(?m)^  ([a-zA-Z0-9_-]+):\n", CI_TEXT.split("jobs:\n", maxsplit=1)[1])
    assert jobs
    for job in jobs:
        assert "timeout-minutes:" in _job_block(job), f"CI job has no timeout: {job}"


def test_scheduled_dependency_maintenance_is_report_first() -> None:
    dependabot = (REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    assert 'package-ecosystem: "cargo"' in dependabot
    cargo_policy = dependabot.split('package-ecosystem: "cargo"', maxsplit=1)[1]
    assert 'update-types: ["minor", "patch"]' in cargo_policy
    assert "ignore:" not in cargo_policy
    assert 'update-types: ["major"' not in cargo_policy

    maintenance = _job_block("dependency-maintenance")
    assert "if: github.event_name == 'schedule'" in maintenance
    assert "cargo-audit@0.22.2" in maintenance
    assert "cargo update --workspace --dry-run" in maintenance
    assert maintenance.count("continue-on-error: true") == 2
    assert "python scripts/check_rustsec_advisories.py" in maintenance


def test_scheduled_stress_is_bounded_and_excludes_large_runner_case() -> None:
    stress = _job_block("scheduled-concurrency-stress")
    assert "tests/test_session_stress.py -m stress" in stress
    assert "tests/test_bolt_server_concurrency.py -m bolt_stress" in stress
    assert "tests/test_stress.py" not in stress
    large = (REPO_ROOT / "tests" / "test_stress.py").read_text(encoding="utf-8")
    assert "manual/large-runner" in large


def test_bolt_driver_conformance_installs_both_toolchains() -> None:
    """The suites skip when their toolchain is missing, which is right locally
    and useless in CI — a runner without a JDK would report green while never
    executing the Java driver at all. So CI must install both, and `-rs` must
    stay on so any skip is visible in the log rather than silent."""
    job = _job_block("bolt-driver-conformance")
    # Matched without the major version on purpose. What this test guards is
    # that both toolchains get installed; which release of setup-java/
    # setup-node does it is not part of that contract, and pinning it here
    # turned a routine "bump the action off a deprecated Node runtime" into a
    # test failure with nothing wrong behind it.
    assert "actions/setup-java@" in job
    assert "distribution: temurin" in job
    assert "actions/setup-node@" in job
    assert "cargo build --release -p kglite-bolt-server" in job
    assert "pytest tests/test_bolt_driver_conformance.py -m bolt" in job
    assert "-rs" in job


# --- build_wheels.yml -------------------------------------------------------
# This file guarded ci.yml only, so the publish workflow — the one that
# actually decides what reaches PyPI — had no local gate at all. That is the
# same shape as the gap that let an action-major bump fail the whole Python
# matrix: `make gate` runs no Python tests, so a workflow edit is unverified
# until CI. The sdist job below is the first thing to depend on it.

WHEELS_PATH = REPO_ROOT / ".github" / "workflows" / "build_wheels.yml"
WHEELS_TEXT = WHEELS_PATH.read_text(encoding="utf-8")


def _wheels_job(job: str) -> str:
    match = re.search(rf"(?ms)^  {re.escape(job)}:\n(.*?)(?=^  [a-zA-Z0-9_-]+:\n|\Z)", WHEELS_TEXT)
    assert match is not None, f"missing required build_wheels job: {job}"
    return match.group(1)


def _wheels_job_commands(job: str) -> str:
    """A job block with comment lines stripped.

    Asserting a command is present is worthless if the same words appear in
    the comment explaining that command — which is exactly what happened
    here: gutting `cargo metadata` from the sdist verification still passed,
    because the prose above it says "cargo metadata resolves after unpacking".
    Caught by mutating the workflow and watching the guard stay green.
    """
    return "\n".join(line for line in _wheels_job(job).splitlines() if not line.lstrip().startswith("#"))


#: Artifact producers `publish` deliberately does NOT require to succeed.
#: Each entry is a conscious "this platform may silently not ship" decision,
#: not an oversight — cross-compiled aarch64 wheels are fragile and were
#: judged not worth blocking a release over.
BEST_EFFORT_PRODUCERS = {"build-linux-arm"}


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

    needs = re.search(r"needs:\s*\[([^\]]*)\]", publish)
    assert needs is not None, "publish job declares no `needs`"
    declared = {n.strip() for n in needs.group(1).split(",")}

    gated = set(re.findall(r"needs\.([a-zA-Z0-9_-]+)\.result\s*==\s*'success'", publish))

    producers = {
        job for job in re.findall(r"(?m)^  ([a-zA-Z0-9_-]+):$", WHEELS_TEXT) if "upload-artifact" in _wheels_job(job)
    }
    assert producers, "no artifact-producing jobs found — the scan is broken"

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
    """
    publish = _wheels_job("publish")
    pattern = re.search(r"pattern:\s*(\S+)", publish)
    assert pattern is not None, "publish job downloads without a `pattern`"
    prefix = pattern.group(1).rstrip("*")

    names = re.findall(r"(?m)^\s+name:\s*(wheels[a-zA-Z0-9_${}.\-]*)$", WHEELS_TEXT)
    assert names, "no uploaded artifact names found — the scan is broken"
    stray = [n for n in names if not n.startswith(prefix)]
    assert not stray, f"artifact name(s) {stray} do not match publish's pattern {pattern.group(1)!r}"


def test_sdist_is_built_and_proven_usable() -> None:
    """The sdist job must verify the artifact resolves, not just that it built.

    A source fallback that cannot build is worse than none: it turns a clear
    "no matching distribution" into a compile error inside a stranger's
    install log. Producing a tarball proves nothing about that.
    """
    sdist = _wheels_job_commands("build-sdist")
    assert "maturin sdist" in sdist
    assert "cargo metadata" in sdist, "sdist job never resolves the unpacked artifact"
    assert "tar -xzf" in sdist, "sdist job never unpacks the artifact it built"
    assert "LICENSE" in sdist, "sdist job does not assert the licence ships"
