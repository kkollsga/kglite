"""Encoding safety for ``print(kg.cypher(...))``.

``ResultView.__repr__`` renders a Polars-style table out of box-drawing
characters. None of those exist in cp1252/cp932/cp936/cp949. On Windows a real
console is written through ``WriteConsoleW`` and copes, but the moment stdout is
a pipe or a file — CI logs, ``python script.py > out.txt``,
``subprocess(capture_output=True)``, notebook capture — CPython falls back to
the locale codepage and ``print`` raises ``UnicodeEncodeError``. A user's script
then works interactively and dies in their CI.

The renderer therefore probes ``sys.stdout.encoding`` and falls back to an
ASCII table. ``PYTHONIOENCODING=cp1252`` reproduces the failing stdout on any
platform, so these tests exercise the real mechanism rather than only the
override.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Renders a 2-column table plus a >20-row table so both the borders and the
# elided-rows marker have to be encodable.
CHILD = textwrap.dedent(
    """
    import sys
    import kglite

    g = kglite.KnowledgeGraph()
    g.cypher("UNWIND range(1, 25) AS i CREATE (:Row {name: 'name-' + toString(i), n: i})")
    print(g.cypher("MATCH (r:Row) RETURN r.name AS name, r.n AS n ORDER BY r.n LIMIT 3"))
    print(g.cypher("MATCH (r:Row) RETURN r.name AS name, r.n AS n ORDER BY r.n"))
    print("STDOUT-ENCODING:", sys.stdout.encoding)
    """
)

UNICODE_GLYPHS = "┌┬┐─│┆╞╪╡═└┴┘…"


def _run_child(env_overrides: dict[str, str]) -> subprocess.CompletedProcess[bytes]:
    """Run the printing child with stdout as a pipe (never a console)."""
    env = {**os.environ, **env_overrides}
    env.pop("KGLITE_ASCII_TABLE", None)
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", CHILD],
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
    )


def test_print_survives_a_cp1252_stdout():
    """The reported bug: this raised UnicodeEncodeError on redirected stdout."""
    done = _run_child({"PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"})
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    out = done.stdout.decode("cp1252")
    assert "STDOUT-ENCODING: cp1252" in out
    # An ASCII table, and nothing in it that cp1252 lacks.
    assert "+---" in out and "+===" in out
    assert "| ... " in out, "elided-rows marker must be ASCII too"
    assert not any(glyph in out for glyph in UNICODE_GLYPHS)


def test_unicode_table_is_kept_on_a_utf8_stdout():
    done = _run_child({"PYTHONIOENCODING": "utf-8"})
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    out = done.stdout.decode("utf-8")
    assert "┌" in out and "╞" in out and "└" in out
    assert "…" in out
    assert "+---" not in out


@pytest.mark.parametrize("flag", ["1", "true", "yes"])
def test_env_override_forces_ascii(flag):
    done = _run_child({"PYTHONIOENCODING": "utf-8", "KGLITE_ASCII_TABLE": flag})
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    out = done.stdout.decode("utf-8")
    assert "+---" in out
    assert not any(glyph in out for glyph in UNICODE_GLYPHS)


def test_env_override_forces_unicode_even_on_a_narrow_stdout():
    """`KGLITE_ASCII_TABLE=0` opts back in; the child writes bytes directly so
    the deliberately-unencodable table never goes through cp1252 `print`."""
    child = textwrap.dedent(
        """
        import sys, kglite
        g = kglite.KnowledgeGraph()
        g.cypher("CREATE (:Row {name: 'a'})")
        table = repr(g.cypher("MATCH (r:Row) RETURN r.name AS name"))
        sys.stdout.buffer.write(table.encode("utf-8"))
        """
    )
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "KGLITE_ASCII_TABLE": "0"}
    done = subprocess.run([sys.executable, "-c", child], capture_output=True, cwd=REPO_ROOT, env=env)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert "┌" in done.stdout.decode("utf-8")


def test_long_non_ascii_cell_does_not_panic(small_graph):
    """Cells are truncated at 30; the cut used to be a byte offset, so any
    multi-byte character straddling it panicked out of Rust."""
    for text in ["a" + "é" * 40, "日" * 40, "Kristján Þórðarson " * 3]:
        result = small_graph.cypher("RETURN $t AS t", params={"t": text})
        rendered = repr(result)
        assert " ... " in rendered


def test_rows_stay_aligned_with_non_ascii_data(small_graph):
    """Column widths are character counts, so accented data cannot skew them."""
    result = small_graph.cypher("UNWIND ['Kristján', 'Bob', '東京'] AS name RETURN name, size(name) AS n")
    lines = repr(result).splitlines()[1:]  # drop the short `shape:` header
    widths = {len(line) for line in lines}
    assert len(widths) == 1, f"ragged table: {widths}\n{repr(result)}"
