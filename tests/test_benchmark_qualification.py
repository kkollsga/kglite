"""Qualification must prevent unsuitable captures from producing a green verdict."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parents[1]
PLATFORM = "Darwin/arm64"


def capture(path, value=1.0):
    data = {
        "machine_info": {"system": "Darwin", "machine": "arm64"},
        "datetime": path.stem,
        "benchmarks": [
            {"name": f"cell_{i}", "stats": {"min": value, "median": value, "mean": value}} for i in range(12)
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def manifest(directory, statuses, reference=None):
    records = {}
    for name, status in statuses.items():
        path = directory / name
        records[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "platform": PLATFORM,
            "status": status,
            "evidence": "synthetic qualification contract",
        }
    data = {"schema_version": 1, "captures": records, "references": {PLATFORM: reference} if reference else {}}
    (directory / "qualifications.json").write_text(json.dumps(data), encoding="utf-8")


def run(script, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *map(str, args)], capture_output=True, text=True
    )


def test_rejected_newest_candidate_cannot_be_replaced(tmp_path):
    capture(tmp_path / "0_16_14.json")
    capture(tmp_path / "0_16_17.json")
    manifest(tmp_path, {"0_16_14.json": "accepted", "0_16_17.json": "rejected"})
    result = run("check_perf_anchor.py", "--baselines-dir", tmp_path)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "rejected" in result.stdout + result.stderr


def test_current_alias_selects_qualified_reference_and_exposes_regression(tmp_path):
    capture(tmp_path / "0_16_19.json")
    capture(tmp_path / "current.json", 2.0)
    candidate = capture(tmp_path / "candidate.json", 1.5)
    manifest(tmp_path, {"0_16_19.json": "accepted", "current.json": "rejected"}, "0_16_19.json")
    result = run("compare_bench.py", tmp_path / "current.json", candidate)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "0_16_19.json" in result.stdout
    assert "REGRESS" in result.stdout.upper()


def test_anchor_skips_rejected_target_without_changing_release_distance(tmp_path):
    for version in [14, 15, 16, 17, 18]:
        capture(tmp_path / f"0_16_{version}.json", 2.0 if version == 15 else 1.5 if version == 18 else 1.0)
    manifest(tmp_path, {f"0_16_{v}.json": "rejected" if v == 15 else "accepted" for v in [14, 15, 16, 17, 18]})
    result = run("check_perf_anchor.py", "--baselines-dir", tmp_path, "--releases-back", 3)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "vs 0_16_14.json" in result.stdout
    assert "0_16_15.json" in result.stdout and "rejected" in result.stdout


def test_no_qualified_anchor_is_not_success(tmp_path):
    capture(tmp_path / "0_16_14.json")
    capture(tmp_path / "0_16_17.json", 1.01)
    manifest(tmp_path, {"0_16_14.json": "pending", "0_16_17.json": "pending"})
    result = run("check_perf_anchor.py", "--baselines-dir", tmp_path)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "no qualified" in (result.stdout + result.stderr).lower()


def test_changed_registered_capture_cannot_keep_qualification(tmp_path):
    baseline = capture(tmp_path / "reference.json")
    candidate = capture(tmp_path / "candidate.json")
    manifest(tmp_path, {"reference.json": "accepted"})
    capture(baseline, 2.0)
    result = run("compare_bench.py", baseline, candidate)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "digest" in (result.stdout + result.stderr).lower()


def test_renaming_rejected_candidate_does_not_evade_digest(tmp_path):
    baseline = capture(tmp_path / "reference.json")
    bad = capture(tmp_path / "bad.json", 1.01)
    manifest(tmp_path, {"reference.json": "accepted", "bad.json": "rejected"})
    candidate = tmp_path / "renamed.json"
    candidate.write_bytes(bad.read_bytes())
    result = run("compare_bench.py", baseline, candidate)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "rejected" in (result.stdout + result.stderr).lower()


def test_platform_mismatch_and_invalid_measurements_cannot_pass(tmp_path):
    baseline = capture(tmp_path / "reference.json")
    candidate = capture(tmp_path / "candidate.json")
    data = json.loads(candidate.read_text(encoding="utf-8"))
    data["machine_info"]["system"] = "Linux"
    candidate.write_text(json.dumps(data), encoding="utf-8")
    result = run("compare_bench.py", baseline, candidate)
    assert result.returncode == 2
    assert "platform" in result.stderr
    capture(baseline, 0.0)
    capture(candidate, 0.0)
    result = run("compare_bench.py", baseline, candidate)
    assert result.returncode == 2
    assert "positive" in result.stderr


def test_explicit_older_candidate_does_not_use_a_future_anchor(tmp_path):
    for v, value in [(14, 1.0), (16, 1.5), (17, 2.0)]:
        capture(tmp_path / f"0_16_{v}.json", value)
    manifest(tmp_path, {f"0_16_{v}.json": "accepted" for v in [14, 16, 17]})
    result = run(
        "check_perf_anchor.py",
        "--baselines-dir",
        tmp_path,
        "--current",
        tmp_path / "0_16_16.json",
        "--releases-back",
        1,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "vs 0_16_14.json" in result.stdout


def test_insufficient_anchor_overlap_fails_without_a_verdict(tmp_path):
    capture(tmp_path / "0_16_14.json")
    capture(tmp_path / "0_16_17.json")
    manifest(tmp_path, {"0_16_14.json": "accepted", "0_16_17.json": "pending"})
    result = run("check_perf_anchor.py", "--baselines-dir", tmp_path, "--min-overlap", 13)
    assert result.returncode == 2
    assert "too few for a verdict" in result.stdout


def test_known_inflated_captures_remain_byte_identical_and_rejected():
    from scripts.benchmark_qualification import Registry

    directory = ROOT / "tests/benchmarks/baselines"
    registry = Registry(directory, required=True)
    for name, sha in {
        "0_16_22.json": "ed8f0fc8ffc3388a9fa9d1660b28bb0eb36348b3d9da94293ee48a35e2e62a8e",
        "0_16_23.json": "1b23da8d130234e10ed261987e411a69da11a1f2f6feae6443b8f06a8c00f0f7",
    }.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == sha
        assert registry.status(directory / name) == "rejected"


def test_capture_qualification_requires_its_original_platform(tmp_path):
    baseline = capture(tmp_path / "reference.json")
    candidate = capture(tmp_path / "candidate.json", 1.01)
    manifest(tmp_path, {"reference.json": "accepted"})
    path = tmp_path / "qualifications.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["captures"][baseline.name]["platform"] = "Linux/arm64"
    path.write_text(json.dumps(data), encoding="utf-8")
    result = run("compare_bench.py", baseline, candidate)
    assert result.returncode == 2
    assert "platform differs" in result.stderr


def test_linux_alias_preserves_current_ci_workload():
    path = ROOT / "tests/benchmarks/baselines/current.linux.json"
    result = run("compare_bench.py", path, path, "--require-exact-set", "--quiet")
    assert result.returncode == 0, result.stdout + result.stderr


def test_current_alias_is_not_its_own_anchor(tmp_path):
    capture(tmp_path / "0_16_23.json")
    latest = capture(tmp_path / "0_16_24.json", 1.5)
    (tmp_path / "current.json").write_bytes(latest.read_bytes())
    manifest(tmp_path, {name: "accepted" for name in ["0_16_23.json", "0_16_24.json", "current.json"]}, "0_16_24.json")
    result = run(
        "check_perf_anchor.py",
        "--baselines-dir",
        tmp_path,
        "--current",
        tmp_path / "current.json",
        "--releases-back",
        1,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "vs 0_16_23.json" in result.stdout


def test_missing_candidate_platform_cannot_bypass_qualification(tmp_path):
    baseline = capture(tmp_path / "reference.json")
    candidate = capture(tmp_path / "candidate.json", 1.01)
    manifest(tmp_path, {"reference.json": "accepted"})
    data = json.loads(candidate.read_text(encoding="utf-8"))
    del data["machine_info"]
    candidate.write_text(json.dumps(data), encoding="utf-8")
    result = run("compare_bench.py", baseline, candidate)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "platform" in result.stderr


def test_candidate_qualification_in_its_own_directory_is_enforced(tmp_path):
    baseline = capture(tmp_path / "reference.json")
    manifest(tmp_path, {"reference.json": "accepted"})
    directory = tmp_path / "captures"
    directory.mkdir()
    candidate = capture(directory / "candidate.json")
    manifest(directory, {"candidate.json": "rejected"})
    result = run("compare_bench.py", baseline, candidate)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "rejected" in result.stderr


def test_local_manifest_cannot_override_known_rejected_bytes(tmp_path):
    original = ROOT / "tests/benchmarks/baselines/0_16_23.json"
    candidate = tmp_path / "copied.json"
    candidate.write_bytes(original.read_bytes())
    reference = tmp_path / "reference.json"
    data = json.loads(original.read_text(encoding="utf-8"))
    data["datetime"] = "independent reference"
    reference.write_text(json.dumps(data), encoding="utf-8")
    manifest(tmp_path, {reference.name: "accepted", candidate.name: "pending"})
    result = run("compare_bench.py", reference, candidate)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "rejected" in result.stderr
