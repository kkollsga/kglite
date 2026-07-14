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
CI_TEXT = CI_PATH.read_text()

REQUIRED_JOBS = {
    "docs",
    "storage-parity",
    "disk-concurrency",
    "loom-session",
    "miri-loaders",
    "address-sanitizer",
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
    ):
        assert target in parity

    disk = _job_block("disk-concurrency")
    assert "test_concurrent_disk_reads_keep_materialized_nodes_alive" in disk
    assert "test_disk_writer_lease_is_enforced_across_processes" in disk
    assert "test_disk_session_reuses_writer_lineage_and_composes" in disk


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
    smoke = (REPO_ROOT / "tests" / "test_mcp_server_smoke.py").read_text()
    assert 'os.environ.get("KGLITE_GITHUB_INTEGRATION") == "1"' in smoke
    assert "and GITHUB_TOKEN is not None" in smoke
    assert smoke.count("not _github_live_enabled()") == 2


def test_docs_job_checks_generated_facts_and_warnings() -> None:
    docs = _job_block("docs")
    assert "python scripts/render_docs_facts.py --check" in docs
    assert "sphinx-build -W --keep-going -b html docs docs/_build/html" in docs
    assert "myst.xref_missing" not in (REPO_ROOT / "docs" / "conf.py").read_text()
