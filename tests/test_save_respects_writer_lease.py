"""A library ``save()`` does not land over a graph another process holds open
for writing.

The holder owns the file: its lease says nobody else writes it. A save from a
process that merely loaded or built a graph used to rename its own bytes over
the path, losing the holder's work. It now raises ``WriterLeaseHeldError``
naming the holder, as ``open()`` does, and leaves the path untouched. A holder
in this process is not foreign, so a graph saving to its own path is not
refused.
"""

import subprocess
import sys
import textwrap

import pytest

import kglite

HOLDER = textwrap.dedent(
    """
    import sys, kglite
    g = kglite.open({path!r})
    if {write!r}:
        g.cypher("CREATE (:Held {{id: 1}})")
    print("ready", flush=True)
    sys.stdin.readline()
    """
)


def _hold(path, write=False):
    child = subprocess.Popen(
        [sys.executable, "-I", "-c", HOLDER.format(path=path, write=write)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert child.stdout.readline().strip() == "ready"
    return child


def _release(child):
    child.stdin.write("\n")
    child.stdin.flush()
    child.wait(timeout=30)


def _scratch():
    graph = kglite.KnowledgeGraph()
    graph.cypher("CREATE (:Scratch {id: 7})")
    return graph


def test_save_over_a_path_another_process_holds_is_refused(tmp_path):
    path = str(tmp_path / "app.kgl")
    child = _hold(path, write=True)
    try:
        graph = _scratch()
        with pytest.raises(kglite.WriterLeaseHeldError) as caught:
            graph.save(path)
        assert caught.value.holder["pid"] == child.pid
        assert caught.value.holder["self"] is False
    finally:
        _release(child)
    # Nothing of the refused save reached the path: the holder's own state is
    # what a reopen finds once the lease is free.
    reopened = kglite.open(path)
    assert reopened.cypher("MATCH (n) RETURN labels(n)[0] AS l").to_list() == [{"l": "Held"}]
    del reopened


def test_save_succeeds_once_the_holder_has_exited(tmp_path):
    path = str(tmp_path / "app.kgl")
    _release(_hold(path))
    _scratch().save(path)
    loaded = kglite.load(path)
    assert loaded.cypher("MATCH (n:Scratch) RETURN n.id AS id").to_list() == [{"id": 7}]


def test_save_over_a_disk_graph_another_process_holds_is_refused(tmp_path):
    path = str(tmp_path / "disk_graph")
    seed = kglite.KnowledgeGraph(storage="disk", path=path)
    seed.cypher("CREATE (:Seed {id: 1})")
    seed.save()
    del seed
    child = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-c",
            textwrap.dedent(
                f"""
                import sys, kglite
                g = kglite.open({path!r})
                print("ready", flush=True)
                sys.stdin.readline()
                """
            ),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        loaded = kglite.load(path)
        with pytest.raises(kglite.WriterLeaseHeldError) as caught:
            loaded.save(path)
        assert caught.value.holder["pid"] == child.pid
    finally:
        _release(child)


def test_a_graph_saving_to_the_path_it_holds_is_not_refused(tmp_path):
    path = str(tmp_path / "app.kgl")
    graph = kglite.open(path)
    graph.cypher("CREATE (:Mine {id: 1})")
    graph.save()
    graph.cypher("CREATE (:Mine {id: 2})")
    graph.save(path)
    del graph
    assert kglite.load(path).cypher("MATCH (n:Mine) RETURN count(n) AS c").to_list() == [{"c": 2}]
