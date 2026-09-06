"""Absolute ownership transitions and captured query-default contracts."""

from pathlib import Path

import pytest

import kglite

MODES = ("memory", "mapped", "disk")
MODES_DURABLE = (("memory", "off"), ("mapped", "off"), ("disk", "off"), ("memory", "full"), ("mapped", "full"))
Q = "UNWIND [1,2,3] AS i RETURN i"
IDS = "MATCH(n) RETURN n.id AS id ORDER BY id"


def graph(tmp_path, mode):
    return kglite.KnowledgeGraph(storage=mode, path=str(tmp_path / "disk") if mode == "disk" else None)


def ids(g):
    return [r["id"] for r in g.cypher(IDS).to_list()]


def child(g, name):
    if name == "frozen":
        return g.freeze()
    if name == "session":
        return g.session()
    if name == "cursor":
        return g.session().cursor()
    if name == "snapshot":
        return g.session().snapshot()
    return g.begin() if name == "rw" else g.begin_read()


@pytest.mark.parametrize("mode,durable", MODES_DURABLE)
def test_closed_owner_is_detached_without_old_wal_authority(tmp_path, mode, durable):
    path = str(tmp_path / "home.kgl")
    a = kglite.open(path, storage=mode, durable=durable)
    a.cypher("CREATE (:N {id:1})")
    a.close()
    b = kglite.open(path, storage=mode, durable=durable)
    wal = Path(path + "-wal")
    before = wal.read_bytes() if wal.exists() else None
    a.cypher("CREATE (:N {id:2})")
    assert ids(a) == [1, 2]
    assert (wal.read_bytes() if wal.exists() else None) == before
    with pytest.raises(ValueError, match="path"):
        a.save()
    b.cypher("CREATE (:N {id:3})")
    b.close()
    with kglite.open(path, storage=mode, durable=durable) as c:
        assert ids(c) == [1, 3]
    # A's data remains explicitly saveable somewhere else; the old home is untouched.
    a.save(str(tmp_path / "detached.kgl"))
    assert ids(kglite.load(str(tmp_path / "detached.kgl"))) == [1, 2]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("holder", ("frozen", "session", "ro"))
@pytest.mark.parametrize("exceptional", (False, True))
def test_end_ownership_keeps_old_readers_without_locking_new_writer(tmp_path, mode, holder, exceptional):
    path = str(tmp_path / "home.kgl")
    a = kglite.open(path, storage=mode, durable="off")
    a.cypher("CREATE (:N {id:1,late:'unread'})")
    view = child(a, holder)
    if exceptional:
        with pytest.raises(RuntimeError, match="application"):
            with a:
                raise RuntimeError("application")
    else:
        a.close()
    with kglite.open(path, storage=mode, durable="off") as b:
        assert ids(b) == ([] if exceptional else [1])
        b.cypher("CREATE (:N {id:3})")
    # First access to late happens after B published: preserve lazy columns/files too.
    assert view.cypher("MATCH(n:N) RETURN n.late AS late").to_list() == [{"late": "unread"}]
    assert ids(a) == [1]
    if holder == "session":
        view.execute("CREATE (:N {id:2})")
        assert ids(view) == [1, 2]
        assert ids(a) == [1]
    if holder == "ro":
        view.rollback()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("exceptional", (False, True))
def test_rw_transaction_cannot_publish_into_ended_owner(tmp_path, mode, exceptional):
    path = str(tmp_path / "home.kgl")
    g = kglite.open(path, storage=mode, durable="off")
    g.cypher("CREATE (:N {id:1})")
    tx = g.begin()
    tx.cypher("CREATE (:N {id:2})")
    if exceptional:
        with pytest.raises(RuntimeError):
            with g:
                raise RuntimeError("application")
    else:
        g.close()
    assert ids(tx) == [1, 2]
    with pytest.raises(kglite.ArgumentError, match="ownership"):
        tx.commit()
    assert ids(g) == [1]
    with kglite.open(path, storage=mode, durable="off") as reader:
        assert ids(reader) == ([] if exceptional else [1])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ("frozen", "session", "cursor", "snapshot", "rw", "ro"))
def test_child_query_defaults_are_creation_time_snapshots(tmp_path, mode, name):
    g = graph(tmp_path, mode)
    g.set_default_row_limit(1)
    g.set_default_timeout(0)
    h = child(g, name)
    g.set_default_row_limit(3)
    assert h.cypher(Q).to_list() == [{"i": 1}]
    assert h.cypher(Q, row_limit=None).to_list() == [{"i": 1}]
    assert h.cypher(Q, row_limit=2).to_list() == [{"i": 1}, {"i": 2}]
    assert h.cypher(Q, row_limit=0).to_list() == []
    assert len(child(g, "frozen").cypher(Q).to_list()) == 3
    if name in ("rw", "ro"):
        h.rollback()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ("frozen", "session", "cursor", "snapshot", "rw", "ro"))
def test_child_work_default_and_override(tmp_path, mode, name):
    g = graph(tmp_path, mode)
    g.set_default_max_work_units(1)
    h = child(g, name)
    g.set_default_max_work_units(10)
    for opts in ({}, {"max_work_units": None}, {"max_work_units": 0}):
        with pytest.raises(kglite.CypherExecutionError, match="budget"):
            h.cypher(Q, **opts)
    assert len(h.cypher(Q, max_work_units=10).to_list()) == 3
    if name in ("rw", "ro"):
        h.rollback()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("method", ("begin", "begin_read"))
def test_zero_transaction_timeout_has_no_lifetime_deadline(tmp_path, mode, method):
    g = graph(tmp_path, mode)
    tx = getattr(g, method)(timeout_ms=0)
    assert tx.cypher("RETURN 1 AS n", timeout_ms=0).to_list() == [{"n": 1}]
    tx.rollback()


@pytest.mark.parametrize("mode,durable", MODES_DURABLE)
def test_failed_close_keeps_data_owner_and_transaction_retry(tmp_path, mode, durable):
    import os

    if os.name != "posix" or os.geteuid() == 0:
        pytest.skip("requires ordinary POSIX permission enforcement")
    path = tmp_path / "home.kgl"
    g = kglite.open(str(path), storage=mode, durable=durable)
    g.cypher("CREATE (:N {id:1})")
    tx = g.begin()
    tx.cypher("CREATE (:N {id:2})")
    blocked_dir = path / "generations" if mode == "disk" else tmp_path
    previous_mode = blocked_dir.stat().st_mode & 0o777
    blocked_dir.chmod(0o500)
    try:
        with pytest.raises(kglite.FileIoError):
            g.close()
    finally:
        blocked_dir.chmod(previous_mode)
    assert ids(g) == [1]
    with pytest.raises(kglite.FileIoError, match="lock|writer"):
        kglite.open(str(path), storage=mode, durable=durable)
    if durable == "full":
        # Durable checkpoint preparation conservatively advances data OCC even
        # when its later file write fails; persistence authority is still live.
        with pytest.raises(kglite.TransactionConflictError, match="modified since begin"):
            tx.commit()
        tx = g.begin()
        tx.cypher("CREATE (:N {id:2})")
    tx.commit()
    assert ids(g) == [1, 2]
    g.close()
    with kglite.open(str(path), storage=mode, durable=durable) as reopened:
        assert ids(reopened) == [1, 2]
