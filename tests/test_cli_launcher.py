"""Unit tests for the `kglite` console-script launcher (kglite/cli.py).

These don't need the compiled binary bundled — they monkeypatch its location
to test the two paths: a present binary forwards argv + exit code, an absent
one prints a clear fallback.
"""

from __future__ import annotations

import os
import stat

import pytest

from kglite import cli


def test_missing_binary_prints_fallback(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_binary_path", lambda: tmp_path / "_bin" / "kglite")
    rc = cli.main([])
    assert rc == 1
    err = capsys.readouterr().err
    assert "cargo install kglite-cli" in err


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell stub")
def test_present_binary_forwards_argv_and_code(monkeypatch, tmp_path):
    # A tiny stub that echoes its args and exits with a chosen code.
    stub = tmp_path / "kglite"
    stub.write_text('#!/bin/sh\necho "got: $@" > "$KGLITE_STUB_OUT"\nexit 7\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    out_file = tmp_path / "out.txt"
    monkeypatch.setenv("KGLITE_STUB_OUT", str(out_file))
    monkeypatch.setattr(cli, "_binary_path", lambda: stub)

    rc = cli.main(["app.kgl", "--flag"])
    assert rc == 7  # the stub's exit code is passed through
    assert "got: app.kgl --flag" in out_file.read_text()


def test_binary_path_under_package():
    """The bundled binary is looked for under the installed package's _bin/."""
    p = cli._binary_path()
    assert p.parent.name == "_bin"
    assert p.parent.parent.name == "kglite"
    assert p.name in ("kglite", "kglite.exe")
