"""Pytest gate: every parity case has an owner, and every CI-owned case passes.

Use --parity-owner audit with --collect-only over tests/ to detect unowned new
cases. The storage-parity job uses its existing explicit test targets and this
plugin to reject missing, skipped or failed required cases at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REGISTRY = Path(__file__).resolve().parents[1] / "tests" / "parity_ownership.json"
STATIC_CASE = "tests/test_phase5_parity.py::test_dead_code_check"


def load_registry(path=REGISTRY):
    owners = json.loads(path.read_text(encoding="utf-8"))
    for nodeid, entry in owners.items():
        if entry["owner"] == "storage-parity":
            continue
        if nodeid != STATIC_CASE or entry["owner"] != "workspace-clippy" or not entry.get("reason", "").strip():
            raise ValueError(f"Undeclared parity exclusion: {nodeid}: {entry}")
    return owners


def validate_collection(nodeids, owners, owner):
    found = set(nodeids)
    if len(found) != len(nodeids):
        raise ValueError("Duplicate collected parity IDs")
    expected = {nodeid for nodeid, entry in owners.items() if owner == "audit" or entry["owner"] == owner}
    unknown = found - owners.keys()
    missing = expected - found
    if unknown or missing:
        raise ValueError(f"Parity ownership mismatch: unowned={sorted(unknown)}, missing={sorted(missing)}")
    return expected


def unsuccessful(required, outcomes):
    return {
        nodeid: outcomes.get(nodeid, "not executed") for nodeid in sorted(required) if outcomes.get(nodeid) != "passed"
    }


def pytest_addoption(parser):
    parser.addoption("--parity-owner", choices=("audit", "storage-parity"), help=__doc__)


def pytest_configure(config):
    owner = config.getoption("--parity-owner")
    if owner:
        if owner == "audit" and not config.option.collectonly:
            raise pytest.UsageError("Parity ownership audit must use --collect-only; it does not execute clippy")
        config.pluginmanager.register(ParityOwner(owner, load_registry()), "parity-owner-gate")


class ParityOwner:
    def __init__(self, owner, owners):
        self.owner = owner
        self.owners = owners
        self.required = set()
        self.outcomes = {}

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config, items):
        parity = [item for item in items if item.get_closest_marker("parity")]
        try:
            self.required = validate_collection([item.nodeid for item in parity], self.owners, self.owner)
        except ValueError as error:
            raise pytest.UsageError(str(error)) from error
        if self.owner != "audit":
            removed = [item for item in items if item.nodeid not in self.required]
            items[:] = [item for item in items if item.nodeid in self.required]
            config.hook.pytest_deselected(items=removed)

    def pytest_runtest_logreport(self, report):
        if report.nodeid not in self.required:
            return
        if report.failed or report.skipped:
            self.outcomes[report.nodeid] = report.outcome
        elif report.when == "call" and report.nodeid not in self.outcomes:
            self.outcomes[report.nodeid] = "passed"

    def pytest_sessionfinish(self, session, exitstatus):
        if session.config.option.collectonly or self.owner == "audit":
            return
        failures = unsuccessful(self.required, self.outcomes)
        if not self.required or failures:
            reporter = session.config.pluginmanager.get_plugin("terminalreporter")
            if reporter:
                reporter.write_sep(
                    "!", f"Required parity cases did not pass: {failures or 'no cases selected'}", red=True
                )
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
