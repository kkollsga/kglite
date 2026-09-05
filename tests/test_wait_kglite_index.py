"""Hermetic failures for the registry gate; no publication or network calls."""

import json
from urllib.error import HTTPError, URLError

import pytest
from scripts import wait_kglite_index as readiness

VERSION = "0.16.23"


def entry(version=VERSION, *, name="kglite", yanked=False):
    return json.dumps({"name": name, "vers": version, "yanked": yanked}).encode()


def install_responses(monkeypatch, responses):
    responses = iter(responses)
    calls, sleeps = [], []

    def fetch():
        calls.append("fetch")
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(readiness, "fetch_index", fetch)
    monkeypatch.setattr(readiness.time, "sleep", sleeps.append)
    return calls, sleeps


def test_ready_first_poll_does_not_sleep(monkeypatch):
    calls, sleeps = install_responses(monkeypatch, [entry("0.16.22") + b"\n" + entry()])
    assert readiness.main([VERSION]) == 0
    assert calls == ["fetch"]
    assert sleeps == []


def test_delayed_exact_version_retries_then_stops(monkeypatch):
    calls, sleeps = install_responses(monkeypatch, [entry("0.16.22"), entry("0.16.230"), entry()])
    assert readiness.main([VERSION]) == 0
    assert len(calls) == 3
    assert sleeps == [readiness.RETRY_DELAY] * 2


@pytest.mark.parametrize(
    "response",
    [
        b"",
        entry("0.16.230"),
        entry(name="kglite-cli"),
        entry(yanked=True),
        entry() + b"\nnot json",
        b"[]",
        HTTPError(readiness.INDEX_URL, 429, "rate limited", {}, None),
        URLError("unavailable"),
        TimeoutError("request timed out"),
    ],
)
def test_unconfirmed_index_exhausts_bound_and_fails(monkeypatch, response, capsys):
    calls, sleeps = install_responses(monkeypatch, [response] * readiness.MAX_ATTEMPTS)
    assert readiness.main([VERSION]) == 1
    assert len(calls) == readiness.MAX_ATTEMPTS
    assert sleeps == [readiness.RETRY_DELAY] * (readiness.MAX_ATTEMPTS - 1)
    assert "stopping dependent publication" in capsys.readouterr().err


def test_transient_request_failure_can_recover(monkeypatch):
    calls, sleeps = install_responses(monkeypatch, [TimeoutError("slow index"), entry()])
    assert readiness.main([VERSION]) == 0
    assert len(calls) == 2
    assert sleeps == [readiness.RETRY_DELAY]


@pytest.mark.parametrize("status,body,valid", [(200, entry(), True), (503, b"", False), (204, b"", False)])
def test_fetch_checks_status_and_uses_bounded_sparse_request(monkeypatch, status, body, valid):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            assert limit == readiness.MAX_INDEX_BYTES + 1
            return body

    response = Response()
    response.status = status

    def open_request(request, *, timeout):
        assert request.full_url == "https://index.crates.io/kg/li/kglite"
        assert request.get_header("Cache-control") == "no-cache"
        assert timeout == readiness.REQUEST_TIMEOUT
        return response

    monkeypatch.setattr(readiness, "urlopen", open_request)
    if valid:
        assert readiness.fetch_index() == body
    else:
        with pytest.raises(ValueError, match=f"HTTP {status}"):
            readiness.fetch_index()
