"""Regression tests for release-time captured-constant maintenance."""

from __future__ import annotations

from scripts import refresh_release_constants as refresh


def test_binary_size_refresh_updates_platform_entry_idempotently(tmp_path, monkeypatch):
    phase5 = tmp_path / "test_phase5_parity.py"
    phase5.write_text(
        "BINARY_SIZE_BASELINES = {\n"
        '    "darwin": 10_000,  # old macOS baseline\n'
        '    "linux": 20_000,  # old Linux baseline\n'
        "}\n\n"
        "    Baseline history:\n"
        "    Raising the baseline is a deliberate act\n"
        '    f"(+10% over 0.1.0 darwin baseline {baseline:,}). "\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(refresh, "PHASE5_TEST", phase5)
    monkeypatch.setattr(refresh.sys, "platform", "darwin")

    changed, _ = refresh.refresh_binary_size("1.2.3", 12_345)
    assert changed
    text = phase5.read_text(encoding="utf-8")
    assert '"darwin": 12_345,  # 1.2.3 darwin baseline' in text
    assert '"linux": 20_000' in text
    assert text.count("- 1.2.3:") == 1
    assert "+10% over 1.2.3 {platform_key} baseline" in text

    changed, _ = refresh.refresh_binary_size("1.2.3", 12_345)
    assert not changed
    assert phase5.read_text(encoding="utf-8").count("- 1.2.3:") == 1


def test_perf_capture_is_pending_until_explicit_qualification(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import subprocess

    from scripts.benchmark_qualification import qualify

    directory = tmp_path / "baselines"
    directory.mkdir()
    monkeypatch.setattr(refresh, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(refresh, "BASELINES_DIR", directory)
    monkeypatch.setattr(refresh.sys, "platform", "darwin")
    source = {
        "machine_info": {"system": "Darwin", "machine": "arm64"},
        "benchmarks": [{"name": "cell", "stats": {"min": 1.0, "data": [1.0]}}],
    }
    calls = []

    def benchmark(command, **kwargs):
        calls.append(command)
        output = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--benchmark-json=")))
        output.write_text(json.dumps(source), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(refresh.subprocess, "run", benchmark)
    changed, message = refresh.refresh_perf_baseline("0.17.0")
    assert changed and "pending" in message
    target = directory / "0_17_0.json"
    raw = target.read_bytes()
    assert (directory / "current.json").read_bytes() == raw
    assert "data" not in json.loads(raw)["benchmarks"][0]["stats"]
    manifest = json.loads((directory / "qualifications.json").read_text(encoding="utf-8"))
    assert manifest["captures"][target.name]["status"] == "pending"
    assert manifest["references"] == {}
    qualify(target, "accepted", "Two agreeing synthetic controls", promote=True)
    before = (directory / "qualifications.json").read_bytes()
    changed, _ = refresh.refresh_perf_baseline("0.17.0")
    assert not changed and len(calls) == 1
    assert target.read_bytes() == raw
    assert (directory / "qualifications.json").read_bytes() == before
