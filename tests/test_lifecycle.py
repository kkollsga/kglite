"""Embedded-DB lifecycle: kglite.open() load-or-create, path-remembering
save(), and the context-manager auto-save-on-close (Stage 0 of the durability
work — ergonomics, not crash safety)."""

import os

import pytest

import kglite


def test_open_creates_then_loads(tmp_path):
    p = str(tmp_path / "app.kgl")
    g = kglite.open(p)  # does not exist yet -> create
    g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
    g.save()  # bare save uses the remembered path
    assert (tmp_path / "app.kgl").exists()
    del g  # release the writer lease before reopening the same path

    g2 = kglite.open(p)  # exists -> load
    names = [r["n"] for r in g2.cypher("MATCH (p:Person) RETURN p.name AS n")]
    assert names == ["Alice"]


def test_bare_save_round_trips(tmp_path):
    p = str(tmp_path / "app.kgl")
    g = kglite.open(p)
    g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
    g.save()
    del g  # each open() owns the path until released
    g = kglite.open(p)
    g.cypher("CREATE (:Person {id: 2, name: 'Bob'})")
    g.save()
    # Read-back uses load(): it takes no writer lease, so it works whether or
    # not `g` is still holding one.
    assert kglite.load(p).cypher("MATCH (p) RETURN count(*) AS c").scalar() == 2


def test_save_as_updates_remembered_path(tmp_path):
    p1 = str(tmp_path / "a.kgl")
    p2 = str(tmp_path / "b.kgl")
    g = kglite.open(p1)
    g.cypher("CREATE (:Person {id: 1})")
    g.save(p2)  # save-as
    assert (tmp_path / "b.kgl").exists()
    # The remembered path is now b.kgl: a further mutation + bare save lands there.
    g.cypher("CREATE (:Person {id: 2})")
    g.save()
    assert kglite.open(p2).cypher("MATCH (p) RETURN count(*) AS c").scalar() == 2


def test_bare_save_without_path_raises():
    g = kglite.KnowledgeGraph()  # in-memory, no origin
    with pytest.raises(ValueError, match="needs a path"):
        g.save()


def test_load_remembers_path(tmp_path):
    p = str(tmp_path / "app.kgl")
    kglite.open(p).save()  # create empty file
    g = kglite.load(p)
    g.cypher("CREATE (:Person {id: 1})")
    g.save()  # bare save works because load() remembered the path
    assert kglite.open(p).cypher("MATCH (p) RETURN count(*) AS c").scalar() == 1


def test_context_manager_autosaves_on_clean_exit(tmp_path):
    p = str(tmp_path / "app.kgl")
    with kglite.open(p) as g:
        g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
    # No explicit save — the clean exit persisted it.
    assert kglite.open(p).cypher("MATCH (p) RETURN count(*) AS c").scalar() == 1


def test_context_manager_binds_graph(tmp_path):
    p = str(tmp_path / "app.kgl")
    with kglite.open(p) as g:
        # __enter__ returns the graph itself.
        assert isinstance(g, kglite.KnowledgeGraph)
        g.cypher("CREATE (:Person {id: 1})")
        assert g.cypher("MATCH (p) RETURN count(*) AS c").scalar() == 1


def test_context_manager_skips_save_on_exception(tmp_path):
    """An exception still skips the checkpoint — but a *committed* mutation is
    durable regardless, because ``open()`` is crash-safe by default.

    A ``with`` block is not a transaction. Each ``cypher()`` mutation commits
    and is ``fsync``'d before it returns, so an exception later in the block
    cannot un-commit it; recovering it on reopen is the write-ahead log doing
    its job. What ``__exit__`` controls is whether a *checkpoint* is written,
    and on an exception it still declines to write one.

    Use ``g.begin()`` when you want discard-on-error, or ``durable=False`` for
    the old snapshot-only behaviour (asserted below).
    """
    p = str(tmp_path / "app.kgl")
    with kglite.open(p) as g:
        g.cypher("CREATE (:Person {id: 1})")
    checkpoint_size = os.path.getsize(p)

    with pytest.raises(RuntimeError):
        with kglite.open(p) as g:
            g.cypher("CREATE (:Person {id: 2})")
            raise RuntimeError("boom")

    # No checkpoint was written by the failed exit — the .kgl is untouched.
    assert os.path.getsize(p) == checkpoint_size
    # But the committed statement survives, recovered from the log.
    assert kglite.open(p).cypher("MATCH (p) RETURN count(*) AS c").scalar() == 2


def test_non_durable_context_manager_discards_on_exception(tmp_path):
    """``durable=False`` keeps the pre-durability contract exactly: nothing is
    persisted unless the block exits cleanly."""
    p = str(tmp_path / "app.kgl")
    with kglite.open(p, durable=False) as g:
        g.cypher("CREATE (:Person {id: 1})")

    with pytest.raises(RuntimeError):
        with kglite.open(p, durable=False) as g:
            g.cypher("CREATE (:Person {id: 2})")
            raise RuntimeError("boom")

    assert kglite.open(p, durable=False).cypher("MATCH (p) RETURN count(*) AS c").scalar() == 1


def test_context_manager_does_not_suppress_exception(tmp_path):
    p = str(tmp_path / "app.kgl")
    with pytest.raises(ValueError):
        with kglite.open(p):
            raise ValueError("propagate me")


def test_close_persists_and_keeps_graph_usable(tmp_path):
    p = str(tmp_path / "app.kgl")
    g = kglite.open(p)
    g.cypher("CREATE (:Person {id: 1})")
    g.close()
    assert kglite.open(p).cypher("MATCH (p) RETURN count(*) AS c").scalar() == 1
    # Graph is still usable after close().
    g.cypher("CREATE (:Person {id: 2})")
    assert g.cypher("MATCH (p) RETURN count(*) AS c").scalar() == 2


def test_close_noop_without_path():
    kglite.KnowledgeGraph().close()  # must not raise
