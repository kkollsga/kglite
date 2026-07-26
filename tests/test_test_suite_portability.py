"""Guards against test-suite defects that are invisible on the host that runs it.

Three failure modes motivated this file, none of which looks like a failure:

- `workspace_binary` never appended `.exe`, so on Windows every binary-backed
  suite reported "not built" *after* a successful build and skipped. That is
  the vacuous-green mode `scripts/assert_conformance_ran.py` exists to prevent,
  ungated.
- A module-scope `import resource` (POSIX-only) in a collected test module does
  not fail four tests on Windows — pytest reports `Interrupted: N errors during
  collection` and runs *nothing*. Marker deselection does not help: collection
  imports before filtering.
- A test that signals itself with `os.kill(os.getpid(), signal.SIGINT)` guarded
  only by `hasattr(signal, "SIGINT")` does not fail on Windows — `os.kill`
  there honours only CTRL_C_EVENT/CTRL_BREAK_EVENT and calls TerminateProcess
  for anything else, so the pytest process dies with no report.

Every check here runs anywhere, which is the point: the platform that breaks is
not the platform that runs CI first.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys

from tests.conftest import workspace_binary

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_ROOT = REPO_ROOT / "tests"

# Signals CPython defines on Windows too, so `hasattr(signal, ...)` proves
# nothing about them. (`SIGKILL`, by contrast, genuinely is absent there, which
# makes it a valid guard — see tests/test_durability.py.)
SIGNALS_PRESENT_ON_WINDOWS = frozenset({"SIGINT", "SIGTERM", "SIGABRT", "SIGFPE", "SIGILL", "SIGSEGV", "SIGBREAK"})

# stdlib modules that simply do not exist on Windows. Importing one at module
# scope in a collected file takes down the entire run.
POSIX_ONLY_MODULES = frozenset(
    {"resource", "fcntl", "termios", "pwd", "grp", "posix", "syslog", "tty", "pty", "curses"}
)


def _collected_test_modules() -> list[Path]:
    """Files pytest imports at collection time (`test_*.py`), which is where a
    POSIX-only module-scope import is fatal rather than merely unusable."""
    return sorted(TESTS_ROOT.rglob("test_*.py"))


def test_workspace_binary_appends_the_windows_executable_suffix(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert workspace_binary("kglite").name == "kglite.exe"
    assert workspace_binary("kglite-bolt-server").name == "kglite-bolt-server.exe"


def test_workspace_binary_has_no_suffix_elsewhere(monkeypatch):
    for platform in ("darwin", "linux"):
        monkeypatch.setattr(sys, "platform", platform)
        assert workspace_binary("kglite").name == "kglite"


def test_no_vacuous_sigint_availability_guard():
    """`hasattr(signal, "SIGINT")` is never a platform guard.

    SIGINT exists on Windows, so the condition is always true — but `os.kill`
    there honours only CTRL_C_EVENT/CTRL_BREAK_EVENT and calls TerminateProcess
    for every other value. A test "guarded" this way does not fail on Windows;
    it terminates the pytest process with no report at all. Gate on
    `sys.platform == "win32"` instead.
    """
    offenders = []
    for path in _collected_test_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders += [
            f"{path.relative_to(REPO_ROOT)}:{node.lineno}: hasattr(signal, {node.args[1].value!r})"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "hasattr"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "signal"
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in SIGNALS_PRESENT_ON_WINDOWS
        ]
    assert not offenders, (
        "this signal exists on Windows, so the guard is always true — and "
        "os.kill() with it there calls TerminateProcess, killing the runner "
        "instead of failing:\n  " + "\n  ".join(offenders)
    )


def test_no_module_scope_posix_only_imports():
    """A POSIX-only import must be inside the function that needs it, and the
    test gated with `importlib.util.find_spec(...) is None`."""
    offenders = []
    for path in _collected_test_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in POSIX_ONLY_MODULES:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {name}")
    assert not offenders, (
        "POSIX-only module imported at module scope; on Windows this aborts "
        "collection for the whole suite:\n  " + "\n  ".join(offenders)
    )


def test_no_posix_only_dunder_import():
    """`__import__("resource")` inside a `skipif` condition is evaluated when
    the class or module body executes, i.e. before the marker can skip
    anything — the guard fails in exactly the way it exists to prevent."""
    offenders = []
    for path in _collected_test_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders += [
            f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "__import__"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and str(node.args[0].value).split(".")[0] in POSIX_ONLY_MODULES
        ]
    assert not offenders, (
        "__import__ of a POSIX-only module; use importlib.util.find_spec(...) is None "
        "in the skipif condition instead:\n  " + "\n  ".join(offenders)
    )
