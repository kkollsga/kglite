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
        '    f"(+10% over 0.1.0 darwin baseline {baseline:,}). "\n'
    )
    monkeypatch.setattr(refresh, "PHASE5_TEST", phase5)
    monkeypatch.setattr(refresh.sys, "platform", "darwin")

    changed, _ = refresh.refresh_binary_size("1.2.3", 12_345)
    assert changed
    text = phase5.read_text()
    assert '"darwin": 12_345,  # 1.2.3 darwin baseline' in text
    assert '"linux": 20_000' in text
    assert text.count("- 1.2.3:") == 1
    assert "+10% over 1.2.3 {platform_key} baseline" in text

    changed, _ = refresh.refresh_binary_size("1.2.3", 12_345)
    assert not changed
    assert phase5.read_text().count("- 1.2.3:") == 1
