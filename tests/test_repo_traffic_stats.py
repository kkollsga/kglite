"""Contracts for the daily GitHub traffic collector.

The traffic API keeps only 14 days, so a snapshot lost to a transient GitHub
5xx is data lost permanently once that day ages out — which is exactly what a
`(HTTP 503)` did to the 2026-08-17 run. The retry below is therefore a
data-retention guarantee, not politeness, and a permanent failure (bad token,
wrong repo) must still fail on the first call rather than sleep through five.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "scripts" / "repo_traffic_stats.py"


def _load_collector() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("repo_traffic_stats", COLLECTOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="collector")
def _collector(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = _load_collector()
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    return module


def _responses(module: types.ModuleType, monkeypatch: pytest.MonkeyPatch, outcomes: list[tuple[int, str, str]]):
    """Queue `(returncode, stdout, stderr)` triples for successive `gh` calls."""
    calls: list[list[str]] = []

    def fake_run(cmd, capture_output, text):  # noqa: ANN001, ANN202 - mirrors subprocess.run's shape
        calls.append(cmd)
        code, out, err = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
        return subprocess.CompletedProcess(cmd, code, out, err)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    return calls


SERVER_ERROR = "gh: No server is currently available to service your request. (HTTP 503)"
NOT_FOUND = "gh: Not Found (HTTP 404)"


def test_transient_server_error_is_retried_until_it_succeeds(collector, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _responses(
        collector,
        monkeypatch,
        [(1, "", SERVER_ERROR), (1, "", SERVER_ERROR), (0, '{"count": 7}', "")],
    )
    assert collector.gh_api("repos/o/r/traffic/views") == {"count": 7}
    assert len(calls) == 3


def test_permanent_error_fails_on_the_first_call(collector, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _responses(collector, monkeypatch, [(1, "", NOT_FOUND)])
    with pytest.raises(SystemExit) as exc:
        collector.gh_api("repos/o/r/traffic/views")
    assert len(calls) == 1, "a 404 is permanent — retrying it only delays the failure"
    assert "404" in str(exc.value)


def test_persistent_transient_error_still_fails(collector, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _responses(collector, monkeypatch, [(1, "", SERVER_ERROR)])
    with pytest.raises(SystemExit) as exc:
        collector.gh_api("repos/o/r/traffic/views")
    assert len(calls) == collector.RETRY_ATTEMPTS
    assert "503" in str(exc.value)


def test_statusless_network_failure_is_retried(collector, monkeypatch: pytest.MonkeyPatch) -> None:
    """`gh` reports DNS/TLS failures with no HTTP status line at all."""
    calls = _responses(
        collector,
        monkeypatch,
        [(1, "", "dial tcp: lookup api.github.com: no such host"), (0, "[]", "")],
    )
    assert collector.gh_api("repos/o/r/traffic/popular/paths") == []
    assert len(calls) == 2
