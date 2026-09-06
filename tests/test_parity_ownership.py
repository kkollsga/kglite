"""Negative contracts for collected-ID ownership and actual execution."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from scripts.parity_ownership import ParityOwner, load_registry, unsuccessful, validate_collection


def test_new_and_omitted_cases_are_rejected():
    owners = {"a": {"owner": "storage-parity"}, "b": {"owner": "storage-parity"}}
    assert validate_collection(["a", "b"], owners, "storage-parity") == {"a", "b"}
    for nodeids in (["a"], ["a", "b", "new"], ["a", "b", "b"]):
        with pytest.raises(ValueError):
            validate_collection(nodeids, owners, "storage-parity")


def test_behavioral_case_cannot_be_hidden_as_static(tmp_path):
    registry = tmp_path / "owners.json"
    registry.write_text(
        json.dumps({"new": {"owner": "workspace-clippy", "reason": "already covered"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_registry(registry)


@pytest.mark.parametrize("bad", ["skipped", "failed", "not executed"])
def test_one_passing_sibling_does_not_excuse_missing_execution(bad):
    outcomes = {"a": "passed", "b": bad}
    assert unsuccessful({"a", "b"}, outcomes) == {"b": bad}
    plugin = ParityOwner("storage-parity", {})
    plugin.required = {"a", "b"}
    plugin.outcomes = outcomes
    session = SimpleNamespace(
        config=SimpleNamespace(
            option=SimpleNamespace(collectonly=False), pluginmanager=SimpleNamespace(get_plugin=lambda name: None)
        ),
        exitstatus=0,
    )
    plugin.pytest_sessionfinish(session, 0)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


def test_late_teardown_failure_is_not_overwritten():
    plugin = ParityOwner("storage-parity", {})
    plugin.required = {"a"}
    for when, outcome in [("setup", "passed"), ("call", "passed"), ("teardown", "failed")]:
        plugin.pytest_runtest_logreport(
            SimpleNamespace(
                nodeid="a", when=when, outcome=outcome, failed=outcome == "failed", skipped=outcome == "skipped"
            )
        )
    assert unsuccessful(plugin.required, plugin.outcomes) == {"a": "failed"}


def test_successful_owner_retains_success():
    plugin = ParityOwner("storage-parity", {})
    plugin.required = {"a"}
    plugin.outcomes = {"a": "passed"}
    session = SimpleNamespace(config=SimpleNamespace(option=SimpleNamespace(collectonly=False)), exitstatus=0)
    plugin.pytest_sessionfinish(session, 0)
    assert session.exitstatus == 0


def test_every_collected_parity_case_has_an_execution_owner():
    # Collection only: this never runs the heavy parity cases or cargo clippy.
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-m",
            "parity",
            "--collect-only",
            "-q",
            "-p",
            "scripts.parity_ownership",
            "--parity-owner",
            "audit",
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("skip_second", [False, True])
def test_real_pytest_owner_requires_every_case_to_execute(tmp_path, skip_second):
    root = Path(__file__).resolve().parents[1]
    fixture = tmp_path / "test_cases.py"
    fixture.write_text(
        "import pytest\npytestmark = pytest.mark.parity\n"
        "def test_first():\n    assert True\n"
        + ("@pytest.mark.skip(reason='missing required dependency')\n" if skip_second else "")
        + "def test_second():\n    assert True\n",
        encoding="utf-8",
    )
    owners = {f"test_cases.py::test_{name}": {"owner": "storage-parity"} for name in ("first", "second")}
    args = ["-q", "-c", str(root / "pyproject.toml"), "--rootdir", str(tmp_path), "-m", "parity", str(fixture)]
    code = (
        "import pytest\nfrom scripts.parity_ownership import ParityOwner\n"
        f"raise SystemExit(pytest.main({args!r}, plugins=[ParityOwner('storage-parity', {owners!r})]))"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=120)
    assert result.returncode == (1 if skip_second else 0), result.stdout + result.stderr
    if skip_second:
        assert "test_cases.py::test_second': 'skipped'" in result.stdout
        assert "1 passed, 1 skipped" in result.stdout
    else:
        assert "2 passed" in result.stdout
