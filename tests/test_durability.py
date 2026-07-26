"""Durable (write-ahead-log) mode crash recovery — Stage 1 of the
embedded-Cypher-DB durability work.

The crash tests spawn a child process that mutates a durable graph and then
``os._exit(0)`` — a hard kill with no clean close and no ``save()`` — so the
only thing that can recover the data is WAL replay on reopen.
"""

import os
import subprocess
import sys
import textwrap

import pytest

import kglite

PYBIN = sys.executable


def _crash_child(tmp_path, body: str) -> None:
    """Run *body* in a child process that hard-exits (os._exit) at the end —
    no atexit, no Python finalizers, no clean close. Models a power-loss /
    kill -9 mid-session."""
    script = textwrap.dedent(
        f"""
        import kglite, os
        path = {str(tmp_path / "app.kgl")!r}
        {textwrap.indent(textwrap.dedent(body), "        ").strip()}
        os._exit(0)
        """
    )
    env = dict(os.environ)
    # Child must import the same built extension.
    subprocess.run([PYBIN, "-c", script], check=True, env=env)


def test_durable_create_survives_hard_crash(tmp_path):
    # Child: create a durable graph, mutate, hard-exit WITHOUT save().
    _crash_child(
        tmp_path,
        """
        g = kglite.open(path, durable=True)
        g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
        g.cypher("CREATE (:Person {id: 2, name: 'Bob'})")
        g.cypher("MATCH (a:Person {id:1}),(b:Person {id:2}) CREATE (a)-[:KNOWS]->(b)")
        """,
    )
    # No .kgl was ever written (never saved) — only the WAL sidecar.
    assert not (tmp_path / "app.kgl").exists()
    assert (tmp_path / "app.kgl-wal").exists()

    # Parent: reopen — WAL replay must recover everything.
    g = kglite.open(str(tmp_path / "app.kgl"), durable=True)
    assert g.cypher("MATCH (p:Person) RETURN count(*) AS c").scalar() == 2
    assert g.cypher("MATCH (:Person)-[r:KNOWS]->(:Person) RETURN count(r) AS c").scalar() == 1
    names = sorted(r["n"] for r in g.cypher("MATCH (p:Person) RETURN p.name AS n"))
    assert names == ["Alice", "Bob"]


def test_set_and_delete_survive_crash(tmp_path):
    g = kglite.open(str(tmp_path / "app.kgl"), durable=True)
    g.cypher("CREATE (:Person {id: 1, name: 'Alice', age: 30})")
    g.cypher("CREATE (:Person {id: 2, name: 'Bob'})")
    g.save()  # checkpoint

    _crash_child(
        tmp_path,
        """
        g = kglite.open(path, durable=True)
        g.cypher("MATCH (p:Person {id:1}) SET p.age = 41")
        g.cypher("MATCH (p:Person {id:2}) DETACH DELETE p")
        """,
    )

    g = kglite.open(str(tmp_path / "app.kgl"), durable=True)
    assert g.cypher("MATCH (p:Person {id:1}) RETURN p.age AS a").scalar() == 41
    names = sorted(r["n"] for r in g.cypher("MATCH (p:Person) RETURN p.name AS n"))
    assert names == ["Alice"]


def test_checkpoint_truncates_wal_then_recovers_post_checkpoint(tmp_path):
    g = kglite.open(str(tmp_path / "app.kgl"), durable=True)
    g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
    g.save()  # checkpoint: .kgl written, WAL truncated
    assert (tmp_path / "app.kgl").exists()

    # Post-checkpoint mutation in a child that crashes.
    _crash_child(
        tmp_path,
        """
        g = kglite.open(path, durable=True)
        g.cypher("CREATE (:Person {id: 2, name: 'Bob'})")
        """,
    )

    g = kglite.open(str(tmp_path / "app.kgl"), durable=True)
    names = sorted(r["n"] for r in g.cypher("MATCH (p:Person) RETURN p.name AS n"))
    assert names == ["Alice", "Bob"]


def test_clean_reopen_loop_accumulates(tmp_path):
    # Repeated open → mutate → (no save) → crash → reopen must accumulate.
    p = str(tmp_path / "app.kgl")
    for i in range(3):
        _crash_child(
            tmp_path,
            f"""
            g = kglite.open(path, durable=True)
            g.cypher("CREATE (:Item {{id: {i}}})")
            """,
        )
    g = kglite.open(p, durable=True)
    assert g.cypher("MATCH (n:Item) RETURN count(*) AS c").scalar() == 3


def test_secondary_labels_survive_crash(tmp_path):
    """Secondary labels live in ``DirGraph.secondary_label_index``, above the
    storage backend, so no ``GraphWrite`` call carries them and the WAL's
    write-capture seam cannot infer them. They are logged by their own
    ``SetNodeLabels`` op; before that op existed a durable node's properties
    survived a crash and its ``:Label``s silently did not."""
    _crash_child(
        tmp_path,
        """
        g = kglite.open(path, durable=True)
        g.cypher("CREATE (:Person:Manager:Employee {id: 1, name: 'Alice'})")
        g.cypher("MATCH (p:Person {id:1}) SET p:Contractor")
        g.cypher("MATCH (p:Person {id:1}) REMOVE p:Manager")
        g.cypher("CREATE (:Person:Employee {id: 2, name: 'Bob'})")
        """,
    )
    g = kglite.open(str(tmp_path / "app.kgl"), durable=True)

    # The exact list, not a set: labels() promises primary-first then
    # secondaries sorted by name, and replay must preserve that ordering.
    assert g.cypher("MATCH (p:Person {id:1}) RETURN labels(p) AS l").scalar() == [
        "Person",
        "Contractor",
        "Employee",
    ]
    assert g.cypher("MATCH (p:Person {id:2}) RETURN labels(p) AS l").scalar() == [
        "Person",
        "Employee",
    ]
    # The label *index* is recovered too, not just the reported list — this is
    # what makes MATCH (n:Label) find the node after recovery.
    assert g.cypher("MATCH (n:Employee) RETURN count(*) AS c").scalar() == 2
    # REMOVE p:Manager replayed: the label is gone from every node.
    assert g.cypher("MATCH (p:Person) WHERE 'Manager' IN labels(p) RETURN count(*) AS c").scalar() == 0


def test_labels_survive_checkpoint_then_crash(tmp_path):
    """The other half of the recovery path: labels folded into a ``.kgl``
    checkpoint, plus labels added after it and recovered from the WAL."""
    p = str(tmp_path / "app.kgl")
    g = kglite.open(p, durable=True)
    g.cypher("CREATE (:Person:Employee {id: 1, name: 'Alice'})")
    g.save()  # labels go into the .kgl secondary_labels section; WAL truncated

    _crash_child(
        tmp_path,
        """
        g = kglite.open(path, durable=True)
        g.cypher("MATCH (p:Person {id:1}) SET p:Manager")
        """,
    )

    g = kglite.open(p, durable=True)
    assert g.cypher("MATCH (p:Person {id:1}) RETURN labels(p) AS l").scalar() == [
        "Person",
        "Employee",  # from the checkpoint
        "Manager",  # from WAL replay on top of it
    ]


def test_durable_rejects_disk_mode(tmp_path):
    with pytest.raises(ValueError, match="in-memory"):
        kglite.open(str(tmp_path / "g"), storage="disk", durable=True)


def test_non_durable_open_writes_no_wal(tmp_path):
    g = kglite.open(str(tmp_path / "app.kgl"))
    g.cypher("CREATE (:Person {id: 1})")
    g.save()
    # Non-durable mode never creates a WAL sidecar.
    assert not (tmp_path / "app.kgl-wal").exists()
