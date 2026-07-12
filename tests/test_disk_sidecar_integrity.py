"""Disk-graph sidecar integrity: corruption fails loudly, absence stays graceful.

Storage-hardening coverage for `load_disk_dir` (crates/kglite/src/graph/io/file.rs):

- The optional sidecars (``embeddings.bin.zst``, ``timeseries.bin.zst``,
  ``secondary_labels.bin.zst``) are legitimate to be *absent* (older graphs,
  feature unused). But a sidecar that EXISTS and fails to decode is corruption
  and must fail the load with an error naming the file — silently loading a
  complete-looking graph with embeddings/timeseries/labels quietly missing is
  data loss.
- ``columns.bin`` string bytes are untrusted disk input consumed by
  ``from_utf8_unchecked`` readers; the loader validates every string column
  once at load time (``MmapColumnStore::validate_utf8``) and refuses to load
  invalid UTF-8 or corrupt offsets.
- Durable (WAL) graphs force ``fsync=True`` on ``save()`` — the checkpoint
  truncates the fsync'd log, so an unflushed checkpoint could lose both.

Run: pytest tests/test_disk_sidecar_integrity.py
"""

from __future__ import annotations

import glob
import os
import warnings

import pandas as pd
import pytest

import kglite

SIDECARS = ["embeddings.bin.zst", "timeseries.bin.zst", "secondary_labels.bin.zst"]


def _build_disk_graph(tmp_path) -> str:
    """Disk graph carrying all three optional sidecars."""
    path = str(tmp_path / "dg")
    g = kglite.KnowledgeGraph(storage="disk", path=path)
    g.add_nodes(
        pd.DataFrame({"id": [1, 2, 3], "title": ["Zebra", "Unicorn", "Fjord"], "n": [1, 2, 3]}),
        "Doc",
        "id",
        "title",
    )
    g.set_embeddings("Doc", "title", {1: [0.1, 0.2], 2: [0.3, 0.4], 3: [0.5, 0.6]})
    g.add_timeseries(
        "Doc",
        data=pd.DataFrame({"fk": [1, 1, 2], "date": ["2020-01", "2020-02", "2020-01"], "v": [1.0, 2.0, 3.0]}),
        fk="fk",
        time_key=["date"],
        channels=["v"],
    )
    g.cypher("MATCH (n:Doc) WHERE n.id = 1 SET n:Extra")
    g.save(path)
    return path


def _sidecar(path: str, name: str) -> str:
    hits = glob.glob(os.path.join(path, "**", name), recursive=True)
    assert hits, f"fixture must have produced {name}"
    return hits[0]


@pytest.mark.parametrize("name", SIDECARS)
def test_corrupt_sidecar_fails_load_with_file_named(name, tmp_path):
    path = _build_disk_graph(tmp_path)
    f = _sidecar(path, name)
    data = open(f, "rb").read()
    # Truncate to half — undecodable zstd/payload, models a torn write
    # or bit rot. (Also exercised with garbage bytes below.)
    with open(f, "wb") as fh:
        fh.write(data[: len(data) // 2])
    with pytest.raises(Exception) as exc_info:
        kglite.load(path)
    msg = str(exc_info.value)
    assert name in msg, f"error must name the corrupt file, got: {msg}"
    assert "corrupt" in msg.lower()


@pytest.mark.parametrize("name", SIDECARS)
def test_garbage_sidecar_fails_load(name, tmp_path):
    path = _build_disk_graph(tmp_path)
    f = _sidecar(path, name)
    with open(f, "wb") as fh:
        fh.write(b"\xde\xad\xbe\xef not zstd at all")
    with pytest.raises(Exception) as exc_info:
        kglite.load(path)
    assert name in str(exc_info.value)


@pytest.mark.parametrize("name", SIDECARS)
def test_absent_sidecar_still_loads(name, tmp_path):
    # Absence is the legitimate older-graph / feature-unused state (and
    # the documented "delete the file to load without it" escape hatch).
    path = _build_disk_graph(tmp_path)
    os.remove(_sidecar(path, name))
    g = kglite.load(path)
    assert g.cypher("MATCH (n:Doc) RETURN count(n) AS c").scalar() == 3


def test_intact_graph_round_trips_all_sidecars(tmp_path):
    # Control: the fixture itself loads cleanly with all sidecar data.
    path = _build_disk_graph(tmp_path)
    g = kglite.load(path)
    assert g.cypher("MATCH (n:Doc) RETURN count(n) AS c").scalar() == 3
    assert g.cypher("MATCH (n:Extra) RETURN count(n) AS c").scalar() == 1
    assert g.timeseries(1) is not None


# ── columns.bin string validation (mmap fast path) ──────────────────────────


def _build_columns_bin_graph(tmp_path) -> str:
    """Disk graph whose columns live in the mmap'd ``columns.bin`` — the
    layout produced by the ntriples build pipeline (``add_nodes`` graphs
    reload through the packed ``columns.zst`` path instead, which has its
    own load-time validation in ``unpack_column``)."""
    nt_path = tmp_path / "tiny.nt"
    lines = []
    for i in range(1, 6):
        lines.append(
            f'<http://www.wikidata.org/entity/Q{i}> <http://www.w3.org/2000/01/rdf-schema#label> "ZebraTitle{i}"@en .'
        )
        lines.append(
            f"<http://www.wikidata.org/entity/Q{i}> "
            f"<http://www.wikidata.org/prop/direct/P31> "
            f"<http://www.wikidata.org/entity/Q5> ."
        )
    nt_path.write_text("\n".join(lines) + "\n")
    path = str(tmp_path / "dg_nt")
    g = kglite.KnowledgeGraph(storage="disk", path=path)
    g.load_ntriples(str(nt_path), languages=["en"], verbose=False)
    del g
    return path


def test_invalid_utf8_in_columns_bin_fails_load(tmp_path):
    path = _build_columns_bin_graph(tmp_path)
    # Sanity: intact graph loads.
    assert kglite.load(path).cypher("MATCH (n) RETURN count(n) AS c").scalar() == 5

    cols = glob.glob(os.path.join(path, "**", "columns.bin"), recursive=True)
    assert cols, "ntriples disk build must produce columns.bin"
    data = bytearray(open(cols[0], "rb").read())
    idx = data.find(b"ZebraTitle3")
    assert idx != -1, "title bytes must be locatable in columns.bin"
    data[idx + 2] = 0xFF  # 0xFF is never valid UTF-8
    with open(cols[0], "wb") as fh:
        fh.write(bytes(data))

    with pytest.raises(Exception) as exc_info:
        kglite.load(path)
    msg = str(exc_info.value)
    assert "columns.bin" in msg and "corrupt" in msg.lower(), msg


# ── durable graphs force fsync on save() ─────────────────────────────────────


def test_durable_save_forces_fsync_with_warning(tmp_path):
    p = str(tmp_path / "app.kgl")
    g = kglite.open(p, durable=True)
    g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g.save(fsync=False)
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("fsync=False" in m and "durable" in m for m in msgs), msgs
    # The save is a real checkpoint: .kgl written, WAL truncated to header.
    assert os.path.exists(p)
    assert os.path.getsize(p + "-wal") == 5  # magic + version, no frames
    # Data survives a reopen from the checkpoint alone.
    g2 = kglite.open(p, durable=True)
    assert g2.cypher("MATCH (p:Person) RETURN count(*) AS c").scalar() == 1


def test_durable_save_default_fsync_no_warning(tmp_path):
    p = str(tmp_path / "app.kgl")
    g = kglite.open(p, durable=True)
    g.cypher("CREATE (:Person {id: 1})")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g.save()
    assert not [w for w in caught if issubclass(w.category, UserWarning)], "default fsync=True must not warn"


def test_non_durable_save_fsync_false_no_warning(tmp_path):
    p = str(tmp_path / "g.kgl")
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame({"id": [1], "title": ["a"]}), "Doc", "id", "title")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g.save(p, fsync=False)
    assert not [w for w in caught if issubclass(w.category, UserWarning)], (
        "fsync=False stays honoured (and silent) for non-durable graphs"
    )
