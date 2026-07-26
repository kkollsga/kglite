"""Every text file this repo reads or writes is UTF-8; say so explicitly.

Python's text mode defaults to the locale encoding. On a UTF-8 host that is
invisible; on a Windows console it is cp1252 (or cp932/936/949), and the
consequences are not cosmetic:

- `kglite/claude_config.py` mutates the user's real Claude Desktop / Claude
  Code config. A locale-encoded read mis-decodes their data, `json.dump`
  re-escapes the result, and the atomic replace commits it over every
  unrelated MCP server entry.
- 25 tracked files in this repo are not cp1252-decodable *at all* — the UTF-8
  continuation bytes are undefined in that codepage, so a read raises rather
  than mojibaking. `README.md`, `CHANGELOG.md`, `tests/conftest.py` and three
  `crates/**/src/*.rs` are among them, which is enough to take down
  `scripts/check_source_quality.py` (so `make lint`), `test_docs_contract.py`,
  and `scripts/refresh_release_constants.py` *mid-edit*.

Ruff's PLW1514 is enabled and catches `open(...)` and `Path("x").read_text()`,
but not the `(ROOT / "name").read_text()` join form that dominates this repo,
so this check is the real net.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SCANNED_ROOTS = ("tests", "scripts", "benchmarks", "examples", "kglite")

TEXT_IO_METHODS = frozenset({"read_text", "write_text"})

#: `x.open(...)` is only the text-file builtin for some `x`. These receivers
#: name something else entirely — `kglite.open` opens a *graph*, `os.open`
#: returns a file descriptor — or take an encoding only in text mode
#: (`gzip`/`bz2`/`lzma` default to binary). Checking them produces noise, not
#: findings.
NON_TEXT_OPEN_RECEIVERS = frozenset({"os", "kglite", "gzip", "bz2", "lzma", "tarfile", "zipfile", "webbrowser"})

#: Positional slot at which each call already accepts `encoding`.
#: `Path.read_text(encoding, errors)`, `Path.write_text(data, encoding, ...)`,
#: `open(file, mode, buffering, encoding, ...)`, `Path.open(mode, buffering,
#: encoding, ...)` — the bound method drops the leading `file`, so both the
#: encoding slot and the mode slot shift by one.
ENCODING_POSITION = {"read_text": 0, "write_text": 1, "open": 3}
BOUND_OPEN_ENCODING_POSITION = 2


def _has_keyword(call: ast.Call, name: str) -> bool:
    return any(kw.arg == name for kw in call.keywords)


def _is_binary_mode(call: ast.Call, mode_position: int) -> bool:
    """`open(path, "rb")` / `path.open("rb")` / `mode="wb"` — passing an
    encoding alongside these is a `ValueError`, not an improvement.

    `mode` sits at a different positional slot for the builtin (`open(file,
    mode)`) than for the bound method (`Path.open(mode)`), which is exactly
    the distinction a mechanical sweep gets wrong.
    """
    candidates = [call.args[mode_position]] if len(call.args) > mode_position else []
    candidates += [kw.value for kw in call.keywords if kw.arg == "mode"]
    return any(isinstance(node, ast.Constant) and "b" in str(node.value) for node in candidates)


def _missing_encoding(call: ast.Call) -> str | None:
    """The call's name when it opens text without pinning an encoding."""
    func = call.func
    bound = isinstance(func, ast.Attribute)
    if bound:
        if isinstance(func.value, ast.Name) and func.value.id in NON_TEXT_OPEN_RECEIVERS:
            return None
        name = func.attr
    elif isinstance(func, ast.Name):
        name = func.id
    else:
        return None

    if name not in TEXT_IO_METHODS and name != "open":
        return None
    if _has_keyword(call, "encoding"):
        return None

    encoding_position = ENCODING_POSITION[name]
    mode_position = 1
    if name == "open" and bound:  # Path.open(mode, buffering, encoding, ...)
        encoding_position = BOUND_OPEN_ENCODING_POSITION
        mode_position = 0

    if len(call.args) > encoding_position:
        return None  # passed positionally
    if name == "open" and _is_binary_mode(call, mode_position):
        return None
    return name


def _python_files() -> list[Path]:
    files: list[Path] = []
    for root in SCANNED_ROOTS:
        files.extend((REPO_ROOT / root).rglob("*.py"))
    return sorted(files)


def test_text_io_pins_utf8():
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _missing_encoding(node)
                if name is not None:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {name}()")
    assert not offenders, (
        "text I/O without an explicit encoding falls back to the locale codepage, "
        'which corrupts or raises on non-UTF-8 hosts — pass encoding="utf-8":\n  ' + "\n  ".join(offenders)
    )
