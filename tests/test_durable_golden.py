"""Golden durable-log fixtures: a ``.kgl`` + ``-wal`` pair written by the Python
durable path must reopen to the exact graph the scenario built.

The pairs are committed binaries under ``tests/fixtures/durable_golden/``,
produced by ``tests/fixtures/build_durable_golden_fixtures.py`` before the
durable path is rerouted through the core ``Session``. Asserting literal,
hand-derived values (``EXPECTED``) instead of comparing two engines means the
reroute cannot pass by agreeing with itself, and a fixture that stops loading is
a finding, never a prompt to regenerate.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest

import kglite
from tests.fixtures.build_durable_golden_fixtures import EXPECTED, FIXTURE_DIR, SCENARIOS, compare, observe

WAL_HEADER = b"KWAL\x0a"  # magic + WAL_FORMAT_VERSION 10
KGL_HEADER = b"RGF\x07\x02"
REPO_ROOT = Path(__file__).resolve().parents[1]
NAMES = sorted(SCENARIOS)


def _copy(name: str, tmp_path: Path) -> Path:
    """Copy a scenario: opening replays the log and checkpoints, consuming it."""
    target = tmp_path / name
    shutil.copytree(FIXTURE_DIR / name, target)
    return target / "app.kgl"


def _recover(path: Path, name: str) -> dict:
    graph = kglite.open(str(path), durable=SCENARIOS[name]["level"])
    try:
        return observe(name, graph)
    finally:
        del graph


def test_every_scenario_has_an_expectation_and_a_fixture():
    assert set(EXPECTED) == set(SCENARIOS) == {p.name for p in FIXTURE_DIR.iterdir() if p.is_dir()}


@pytest.mark.parametrize("name", NAMES)
def test_reopen_replays_to_the_exact_expected_graph(name, tmp_path):
    got = _recover(_copy(name, tmp_path), name)
    assert compare(name, got) == []


@pytest.mark.parametrize("name", NAMES)
def test_committed_files_carry_the_pinned_format_bytes(name):
    wal = (FIXTURE_DIR / name / "app.kgl-wal").read_bytes()
    assert wal[:5] == WAL_HEADER, "the WAL format version changed; old logs must stay readable"
    assert len(wal) > 32, "a log with no frames proves nothing"
    kgl = FIXTURE_DIR / name / "app.kgl"
    if kgl.exists():
        assert kgl.read_bytes()[:5] == KGL_HEADER


@pytest.mark.parametrize("name", NAMES)
def test_the_log_is_what_carries_the_graph(name, tmp_path):
    """Non-vacuity: without its log a scenario must not reproduce the expectation."""
    path = _copy(name, tmp_path)
    (path.parent / "app.kgl-wal").unlink()
    try:
        got = _recover(path, name)
    except Exception:
        return  # nothing to read back is also a difference
    assert compare(name, got) != []


@pytest.mark.parametrize("name", NAMES)
def test_a_corrupted_frame_is_never_replayed_silently(name, tmp_path):
    """Non-vacuity: flipping one byte inside the first frame either refuses to open or changes the graph."""
    path = _copy(name, tmp_path)
    wal = path.parent / "app.kgl-wal"
    data = bytearray(wal.read_bytes())
    data[len(WAL_HEADER) + 12] ^= 0xFF
    wal.write_bytes(bytes(data))
    try:
        got = _recover(path, name)
    except Exception:
        return
    assert compare(name, got) != []


@pytest.mark.parametrize("name", ["checkpoint_then_more_full", "second_crash_full"])
def test_a_replayed_log_extends_and_replays_again_after_another_crash(name, tmp_path):
    """A log that was replayed, extended, and killed again still recovers every frame."""
    path = _copy(name, tmp_path)
    script = textwrap.dedent(
        f"""
        import kglite, os
        g = kglite.open({str(path)!r}, durable={SCENARIOS[name]["level"]!r})
        g.cypher("CREATE (:Late {{id: 900, v: 'late'}})")
        os._exit(0)
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True, cwd=REPO_ROOT)
    graph = kglite.open(str(path), durable=SCENARIOS[name]["level"])
    got = observe(name, graph)
    late = graph.cypher("MATCH (n:Late) RETURN n.id AS id, n.v AS v").to_list()
    del graph
    assert compare(name, got) == []
    assert late == [{"id": 900, "v": "late"}]
