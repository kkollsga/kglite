"""``estimate_load_memory`` and the ``max_load_mb`` load ceiling.

Two contracts, and the second depends on the first: a caller can ask what a
``.kgl`` would cost before paying it, and can refuse to pay above a stated
ceiling. The refusal is deliberately NOT ``FileFormatError`` — the file is
valid, and telling an operator to rebuild a graph that is not broken is the
failure this class exists to prevent.
"""

import pandas as pd
import pytest

import kglite

ESTIMATE_KEYS = {
    "index_rebuild_bytes",
    "section_heap_bytes",
    "transient_peak_bytes",
    "total_settled_bytes",
    "total_peak_bytes",
    "node_rows",
    "declared_indexes",
}


def _graph(rows: int = 400, *, indexed: bool = False) -> kglite.KnowledgeGraph:
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {
                "id": [f"n{i}" for i in range(rows)],
                "title": [f"Doc {i}" for i in range(rows)],
                "category": [f"cat-{i % 8}" for i in range(rows)],
                "body": ["x" * 64 for _ in range(rows)],
            }
        ),
        "Doc",
        "id",
        "title",
    )
    if indexed:
        g.cypher("CREATE INDEX FOR (n:Doc) ON (n.category)")
    return g


def _saved(tmp_path, name="graph.kgl", **kwargs) -> str:
    path = tmp_path / name
    _graph(**kwargs).save(str(path))
    return str(path)


def test_estimate_reports_every_term(tmp_path):
    est = kglite.estimate_load_memory(_saved(tmp_path))

    assert set(est) == ESTIMATE_KEYS
    assert all(isinstance(v, int) for v in est.values())
    assert est["node_rows"] == 400
    assert est["section_heap_bytes"] > 0
    assert est["transient_peak_bytes"] > 0
    # The sums are sums, not independent guesses.
    assert est["total_settled_bytes"] == est["section_heap_bytes"] + est["index_rebuild_bytes"]
    assert est["total_peak_bytes"] == est["total_settled_bytes"] + est["transient_peak_bytes"]


def test_the_index_term_appears_only_when_indexes_are_declared(tmp_path):
    plain = kglite.estimate_load_memory(_saved(tmp_path, "plain.kgl"))
    indexed = kglite.estimate_load_memory(_saved(tmp_path, "indexed.kgl", indexed=True))

    assert plain["index_rebuild_bytes"] == 0
    assert plain["declared_indexes"] == 0
    assert indexed["declared_indexes"] == 1
    assert indexed["index_rebuild_bytes"] > 0
    # Identical data: the declaration is the only difference, so it must be the
    # only term that moves.
    assert indexed["section_heap_bytes"] == plain["section_heap_bytes"]


def test_estimating_a_missing_or_unreadable_path_classifies(tmp_path):
    with pytest.raises(kglite.FileError):
        kglite.estimate_load_memory(str(tmp_path / "nope.kgl"))

    junk = tmp_path / "notes.csv"
    junk.write_text("id,name\n1,alice\n", encoding="utf-8")
    with pytest.raises(kglite.FileFormatError):
        kglite.estimate_load_memory(str(junk))

    # A disk-mode graph directory has no metadata head and never rebuilds its
    # indexes at load, so it is a bad argument rather than a number.
    directory = tmp_path / "disk_graph"
    directory.mkdir()
    with pytest.raises(kglite.ArgumentError):
        kglite.estimate_load_memory(str(directory))


def test_a_ceiling_under_the_estimate_refuses_with_its_own_class(tmp_path):
    path = _saved(tmp_path, indexed=True)
    est = kglite.estimate_load_memory(path)

    with pytest.raises(kglite.LoadMemoryLimitError) as caught:
        kglite.load(path, max_load_mb=0)

    # NOT a corrupt-file error: that would send an operator to rebuild a graph
    # that is perfectly fine. This is the whole point of the new class.
    assert not isinstance(caught.value, kglite.FileFormatError)
    assert isinstance(caught.value, kglite.KgError)
    assert caught.value.code == "LoadMemoryLimit"

    message = str(caught.value)
    assert "estimated to peak at" in message
    assert "KGLITE_MAX_LOAD_MB" in message
    assert "Nothing was decompressed" in message
    assert "defer_index_rebuild" in message
    assert f"{est['node_rows']} node rows" in message


def test_a_ceiling_above_the_estimate_loads(tmp_path):
    path = _saved(tmp_path)
    est = kglite.estimate_load_memory(path)
    assert est["total_peak_bytes"] < 64 * 1024 * 1024, "fixture outgrew the ceiling"

    # The control for every refusal above: the same file, a ceiling it fits
    # under, and no error.
    assert kglite.load(path, max_load_mb=64).cypher("MATCH (n:Doc) RETURN count(n) AS c").to_list()[0]["c"] == 400
    # And with no ceiling at all.
    assert kglite.load(path, max_load_mb=None) is not None


def test_the_kwarg_is_megabytes_not_bytes(tmp_path):
    """A unit mix-up is silent and catastrophic in both directions, so it is
    pinned: 1 must mean one megabyte, which this fixture fits under."""
    path = _saved(tmp_path, rows=50)
    est = kglite.estimate_load_memory(path)
    assert est["total_peak_bytes"] > 1024, "fixture is too small to tell the units apart"
    assert est["total_peak_bytes"] < 1024 * 1024

    # Read as bytes, `max_load_mb=1` would refuse this load.
    kglite.load(path, max_load_mb=1)


def test_the_ceiling_reaches_every_load_entry_point(tmp_path):
    path = _saved(tmp_path)
    with pytest.raises(kglite.LoadMemoryLimitError):
        kglite.open_session(path, max_load_mb=0)
    with pytest.raises(kglite.LoadMemoryLimitError):
        kglite.from_bytes(_graph().to_bytes(), max_load_mb=0)

    # Controls: the same three calls without a ceiling.
    kglite.open_session(path)
    kglite.from_bytes(_graph().to_bytes())


def test_deferring_the_rebuild_buys_the_headroom_the_message_offers(tmp_path):
    """The refusal recommends ``defer_index_rebuild``; that recommendation has
    to work, which means the ceiling must not charge for a rebuild the load is
    not going to do."""
    # 100k rows: enough that the modelled index term clears the kwarg's 1 MB
    # granularity, which is what makes the two projections separable at all.
    path = _saved(tmp_path, rows=100_000, indexed=True)
    est = kglite.estimate_load_memory(path)
    assert est["index_rebuild_bytes"] > 1024 * 1024

    deferred_peak = est["total_peak_bytes"] - est["index_rebuild_bytes"]
    # A ceiling between the two projections, in whole MB.
    ceiling_mb = (deferred_peak // (1024 * 1024)) + 1
    assert ceiling_mb * 1024 * 1024 < est["total_peak_bytes"]

    with pytest.raises(kglite.LoadMemoryLimitError):
        kglite.load(path, max_load_mb=ceiling_mb)
    assert kglite.load(path, max_load_mb=ceiling_mb, defer_index_rebuild=True) is not None
