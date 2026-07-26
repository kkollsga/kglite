"""Durable (write-ahead-log) mode crash recovery.

The crash tests spawn a child process that mutates a durable graph and then
dies without a clean close and without ``save()``, so the only thing that can
recover the data is WAL replay on reopen.

Two child-death models, both real:

- ``_crash_child`` — ``os._exit(0)``: skips atexit, Python finalizers, Rust
  ``Drop``, and the context-manager auto-save.
- ``_sigkill_child`` — the child sends itself ``SIGKILL``: uncatchable, so not
  even the interpreter's own teardown runs. The parent asserts the child died
  on signal 9, which is what makes it a crash test rather than a shutdown
  test.

Every crash test runs for each storage mode that supports durability
(:data:`DURABLE_STORAGE_MODES`). ``storage="disk"`` is deliberately excluded
and its refusal is asserted in ``test_durable_rejects_disk_mode``.
"""

import os
import signal
import subprocess
import sys
import textwrap

import pytest

import kglite

PYBIN = sys.executable

#: Storage modes for which ``durable=True`` is supported. Disk graphs commit by
#: publishing an immutable generation rather than by logging a write, so they
#: are refused rather than silently non-durable.
DURABLE_STORAGE_MODES = ("memory", "mapped")

#: ``signal.SIGKILL`` does not exist on Windows, so both the child script's
#: ``os.kill(os.getpid(), signal.SIGKILL)`` and the parent's
#: ``returncode == -signal.SIGKILL`` raise ``AttributeError`` at call time.
#: Skip rather than substitute a catchable signal: these tests are the
#: load-bearing evidence for the durability guarantee, and one that degrades
#: into a graceful-shutdown test is worse than one that does not run.
requires_sigkill = pytest.mark.skipif(
    not hasattr(signal, "SIGKILL"),
    reason="SIGKILL is POSIX-only; this is a crash test and must not degrade "
    "into a graceful-shutdown test on Windows",
)


def _open_kwargs(storage: str) -> str:
    """``kglite.open`` kwargs, as source text for a child script."""
    if storage == "memory":
        return "durable=True"
    return f"storage={storage!r}, durable=True"


def _open(path, storage: str = "memory"):
    """Open *path* durably in *storage* mode (parent-side counterpart)."""
    kwargs = {"durable": True}
    if storage != "memory":
        kwargs["storage"] = storage
    return kglite.open(str(path), **kwargs)


def _child_script(tmp_path, body: str, storage: str, ending: str) -> str:
    return textwrap.dedent(
        f"""
        import kglite, os, signal
        path = {str(tmp_path / "app.kgl")!r}

        def open_durable():
            return kglite.open(path, {_open_kwargs(storage)})

        {textwrap.indent(textwrap.dedent(body), "        ").strip()}
        {ending}
        """
    )


def _crash_child(tmp_path, body: str, storage: str = "memory") -> None:
    """Run *body* in a child that hard-exits (``os._exit``) at the end — no
    atexit, no Python finalizers, no clean close. Models a power loss
    mid-session."""
    script = _child_script(tmp_path, body, storage, "os._exit(0)")
    # Child must import the same built extension.
    subprocess.run([PYBIN, "-c", script], check=True, env=dict(os.environ))


def _sigkill_child(tmp_path, body: str, storage: str = "memory") -> None:
    """Run *body* in a child that then ``SIGKILL``s itself.

    Stronger than ``_crash_child``: ``SIGKILL`` cannot be caught, blocked, or
    handled, so no interpreter teardown, no buffered-stdio flush and no
    ``Drop`` runs. The assertion on ``returncode`` is load-bearing — a child
    that exited any other way would mean the test had stopped being a crash
    test.
    """
    script = _child_script(tmp_path, body, storage, "os.kill(os.getpid(), signal.SIGKILL)")
    done = subprocess.run([PYBIN, "-c", script], env=dict(os.environ), capture_output=True)
    assert done.returncode == -signal.SIGKILL, (
        f"child must die on SIGKILL, got returncode {done.returncode}; stderr={done.stderr.decode(errors='replace')}"
    )


# ── crash recovery, per storage mode ─────────────────────────────────


@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_durable_create_survives_hard_crash(tmp_path, storage):
    # Child: create a durable graph, mutate, hard-exit WITHOUT save().
    _crash_child(
        tmp_path,
        """
        g = open_durable()
        g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
        g.cypher("CREATE (:Person {id: 2, name: 'Bob'})")
        g.cypher("MATCH (a:Person {id:1}),(b:Person {id:2}) CREATE (a)-[:KNOWS]->(b)")
        """,
        storage,
    )
    # No .kgl was ever written (never saved) — only the WAL sidecar.
    assert not (tmp_path / "app.kgl").exists()
    assert (tmp_path / "app.kgl-wal").exists()

    # Parent: reopen — WAL replay must recover everything.
    g = _open(tmp_path / "app.kgl", storage)
    assert g.cypher("MATCH (p:Person) RETURN count(*) AS c").scalar() == 2
    assert g.cypher("MATCH (:Person)-[r:KNOWS]->(:Person) RETURN count(r) AS c").scalar() == 1
    names = sorted(r["n"] for r in g.cypher("MATCH (p:Person) RETURN p.name AS n"))
    assert names == ["Alice", "Bob"]


@requires_sigkill
@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_durable_create_survives_sigkill(tmp_path, storage):
    """The same recovery guarantee under a real, uncatchable ``SIGKILL``."""
    _sigkill_child(
        tmp_path,
        """
        g = open_durable()
        g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
        g.cypher("CREATE (:Person:Staff {id: 2, name: 'Bob'})")
        """,
        storage,
    )
    assert not (tmp_path / "app.kgl").exists()

    g = _open(tmp_path / "app.kgl", storage)
    names = sorted(r["n"] for r in g.cypher("MATCH (p:Person) RETURN p.name AS n"))
    assert names == ["Alice", "Bob"]
    assert g.cypher("MATCH (p:Person {id:2}) RETURN labels(p) AS l").scalar() == [
        "Person",
        "Staff",
    ]


@requires_sigkill
@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_rolled_back_statement_is_absent_after_crash(tmp_path, storage):
    """The other half of the contract: recovery must not invent state.

    A statement that fails partway is rolled back in the graph, and its writes
    must not survive into the log either — otherwise a crash would resurrect a
    mutation the user was told had failed. The overflowing ``duration(...)``
    fires *after* the first node of the statement was written, so this is a
    genuine partial write, not a pre-flight rejection.
    """
    _sigkill_child(
        tmp_path,
        """
        g = open_durable()
        g.cypher("CREATE (:Person {id: 1, name: 'Committed'})")
        try:
            g.cypher(
                "CREATE (:Person {id: 2, name: 'RolledBack'}) "
                "CREATE (:Person {id: 3, boom: duration({months: 2147483648})})"
            )
        except Exception:
            pass
        else:
            raise AssertionError("the poisoned statement was expected to fail")
        g.cypher("CREATE (:Person {id: 4, name: 'AfterFailure'})")
        """,
        storage,
    )
    g = _open(tmp_path / "app.kgl", storage)
    names = sorted(r["n"] for r in g.cypher("MATCH (p:Person) RETURN p.name AS n") if r["n"])
    assert names == ["AfterFailure", "Committed"], (
        "the rolled-back statement's writes must not be recovered, and a commit after the failure must still be"
    )


@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_set_and_delete_survive_crash(tmp_path, storage):
    g = _open(tmp_path / "app.kgl", storage)
    g.cypher("CREATE (:Person {id: 1, name: 'Alice', age: 30})")
    g.cypher("CREATE (:Person {id: 2, name: 'Bob'})")
    g.save()  # checkpoint

    _crash_child(
        tmp_path,
        """
        g = open_durable()
        g.cypher("MATCH (p:Person {id:1}) SET p.age = 41")
        g.cypher("MATCH (p:Person {id:2}) DETACH DELETE p")
        """,
        storage,
    )

    g = _open(tmp_path / "app.kgl", storage)
    assert g.cypher("MATCH (p:Person {id:1}) RETURN p.age AS a").scalar() == 41
    names = sorted(r["n"] for r in g.cypher("MATCH (p:Person) RETURN p.name AS n"))
    assert names == ["Alice"]


@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_checkpoint_truncates_wal_then_recovers_post_checkpoint(tmp_path, storage):
    g = _open(tmp_path / "app.kgl", storage)
    g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
    g.save()  # checkpoint: .kgl written, WAL truncated
    assert (tmp_path / "app.kgl").exists()

    # Post-checkpoint mutation in a child that crashes.
    _crash_child(
        tmp_path,
        """
        g = open_durable()
        g.cypher("CREATE (:Person {id: 2, name: 'Bob'})")
        """,
        storage,
    )

    g = _open(tmp_path / "app.kgl", storage)
    names = sorted(r["n"] for r in g.cypher("MATCH (p:Person) RETURN p.name AS n"))
    assert names == ["Alice", "Bob"]


@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_clean_reopen_loop_accumulates(tmp_path, storage):
    # Repeated open → mutate → (no save) → crash → reopen must accumulate.
    for i in range(3):
        _crash_child(
            tmp_path,
            f"""
            g = open_durable()
            g.cypher("CREATE (:Item {{id: {i}}})")
            """,
            storage,
        )
    g = _open(tmp_path / "app.kgl", storage)
    assert g.cypher("MATCH (n:Item) RETURN count(*) AS c").scalar() == 3


# ── secondary labels ─────────────────────────────────────────────────


@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_secondary_labels_survive_crash(tmp_path, storage):
    """Secondary labels live in ``DirGraph.secondary_label_index``, above the
    storage backend, so no ``GraphWrite`` call carries them and the WAL's
    write-capture seam cannot infer them. They are logged by their own
    ``SetNodeLabels`` op; before that op existed a durable node's properties
    survived a crash and its ``:Label``s silently did not."""
    _crash_child(
        tmp_path,
        """
        g = open_durable()
        g.cypher("CREATE (:Person:Manager:Employee {id: 1, name: 'Alice'})")
        g.cypher("MATCH (p:Person {id:1}) SET p:Contractor")
        g.cypher("MATCH (p:Person {id:1}) REMOVE p:Manager")
        g.cypher("CREATE (:Person:Employee {id: 2, name: 'Bob'})")
        """,
        storage,
    )
    g = _open(tmp_path / "app.kgl", storage)

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


@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_labels_survive_checkpoint_then_crash(tmp_path, storage):
    """The other half of the recovery path: labels folded into a ``.kgl``
    checkpoint, plus labels added after it and recovered from the WAL."""
    g = _open(tmp_path / "app.kgl", storage)
    g.cypher("CREATE (:Person:Employee {id: 1, name: 'Alice'})")
    g.save()  # labels go into the .kgl secondary_labels section; WAL truncated

    _crash_child(
        tmp_path,
        """
        g = open_durable()
        g.cypher("MATCH (p:Person {id:1}) SET p:Manager")
        """,
        storage,
    )

    g = _open(tmp_path / "app.kgl", storage)
    assert g.cypher("MATCH (p:Person {id:1}) RETURN labels(p) AS l").scalar() == [
        "Person",
        "Employee",  # from the checkpoint
        "Manager",  # from WAL replay on top of it
    ]


# ── mutation paths other than cypher() ───────────────────────────────
#
# The log is appended by an explicit flush at the end of each committing
# mutation; there is no choke point that can do it automatically. For a long
# time `cypher()` was the only caller, so every other way of writing to a
# durable graph buffered its ops and dropped them. One test per entry point,
# because "this one forgot to flush" is invisible until a crash.


@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_add_nodes_survives_crash(tmp_path, storage):
    """The bulk loader, not Cypher — the same `GraphWrite` seam, a different
    entry point."""
    _crash_child(
        tmp_path,
        """
        import pandas as pd
        g = open_durable()
        g.add_nodes(
            pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}),
            node_type="Person",
            unique_id_field="id",
            node_title_field="name",
        )
        """,
        storage,
    )
    g = _open(tmp_path / "app.kgl", storage)
    names = sorted(r["n"] for r in g.cypher("MATCH (p:Person) RETURN p.name AS n"))
    assert names == ["Alice", "Bob"]


@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_add_connections_survives_crash(tmp_path, storage):
    _crash_child(
        tmp_path,
        """
        import pandas as pd
        g = open_durable()
        g.add_nodes(
            pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}),
            node_type="Person",
            unique_id_field="id",
            node_title_field="name",
        )
        g.add_connections(
            pd.DataFrame({"src": [1], "tgt": [2]}),
            "KNOWS", "Person", "src", "Person", "tgt",
        )
        """,
        storage,
    )
    g = _open(tmp_path / "app.kgl", storage)
    assert g.cypher("MATCH (:Person)-[r:KNOWS]->(:Person) RETURN count(r) AS c").scalar() == 1


@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_batch_label_api_survives_crash(tmp_path, storage):
    """`add_label` / `remove_label` reach the same label choke point Cypher's
    `SET n:X` does, so they produce the same log entry."""
    _crash_child(
        tmp_path,
        """
        g = open_durable()
        g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
        g.cypher("CREATE (:Person {id: 2, name: 'Bob'})")
        g.add_label("Person", [1, 2], "Employee")
        g.add_label("Person", [1], "Manager")
        g.remove_label("Person", [2], "Employee")
        """,
        storage,
    )
    g = _open(tmp_path / "app.kgl", storage)
    assert g.cypher("MATCH (p:Person {id:1}) RETURN labels(p) AS l").scalar() == [
        "Person",
        "Employee",
        "Manager",
    ]
    assert g.cypher("MATCH (p:Person {id:2}) RETURN labels(p) AS l").scalar() == ["Person"]


@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_committed_transaction_survives_crash(tmp_path, storage):
    """A transaction commits as exactly one log entry — the whole transaction
    or none of it, which is the atomicity the caller asked for."""
    _crash_child(
        tmp_path,
        """
        g = open_durable()
        with g.begin() as tx:
            tx.cypher("CREATE (:Person {id: 1, name: 'InTxn'})")
            tx.cypher("CREATE (:Person {id: 2, name: 'AlsoInTxn'})")
        """,
        storage,
    )
    g = _open(tmp_path / "app.kgl", storage)
    names = sorted(r["n"] for r in g.cypher("MATCH (p:Person) RETURN p.name AS n"))
    assert names == ["AlsoInTxn", "InTxn"]


@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_rolled_back_transaction_leaves_nothing_after_crash(tmp_path, storage):
    """The converse: `Transaction.cypher` must not log, or a rollback would
    still be recoverable."""
    _crash_child(
        tmp_path,
        """
        g = open_durable()
        tx = g.begin()
        tx.cypher("CREATE (:Person {id: 1, name: 'Discarded'})")
        tx.rollback()
        g.cypher("CREATE (:Person {id: 2, name: 'Kept'})")
        """,
        storage,
    )
    g = _open(tmp_path / "app.kgl", storage)
    names = sorted(r["n"] for r in g.cypher("MATCH (p:Person) RETURN p.name AS n"))
    assert names == ["Kept"]


def test_durable_session_refuses_writes_but_allows_reads(tmp_path):
    """A `Session` holds only the graph, never the durability state, and its
    writes land on an independent working copy visible through the session
    alone — so they were unreachable by both the log and the parent's
    `save()`. Refusing beats losing them silently."""
    g = _open(tmp_path / "app.kgl")
    g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
    session = g.session()

    assert session.execute("MATCH (p:Person) RETURN count(*) AS c").to_list() == [{"c": 1}]

    with pytest.raises(Exception) as excinfo:
        session.execute("CREATE (:Person {id: 2})")
    message = str(excinfo.value)
    assert "durable=True" in message
    assert "g.cypher" in message, "must point at a logged alternative"
    assert "g.begin()" in message


def test_load_ntriples_does_not_panic_on_a_durable_graph(tmp_path):
    """The RDF loader's type-resolution pass matched on the backend and treated
    the write-capture wrapper as unreachable, so this panicked outright."""
    nt = tmp_path / "data.nt"
    nt.write_text(
        "<http://www.wikidata.org/entity/Q1> "
        '<http://www.w3.org/2000/01/rdf-schema#label> "Alice" .\n'
        "<http://www.wikidata.org/entity/Q1> "
        "<http://www.wikidata.org/prop/direct/P31> "
        "<http://www.wikidata.org/entity/Q5> .\n"
        "<http://www.wikidata.org/entity/Q5> "
        '<http://www.w3.org/2000/01/rdf-schema#label> "human" .\n'
    )
    g = _open(tmp_path / "app.kgl")
    stats = g.load_ntriples(str(nt))
    assert stats["entities"] == 2
    assert stats["edges"] == 1


# ── vacuum interaction ───────────────────────────────────────────────


@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_durability_survives_an_auto_vacuum(tmp_path, storage):
    """A vacuum rebuilds the graph with contiguous node indices. It used to
    replace the backend outright, which dropped the write-capture wrapper: the
    triggering statement's writes were discarded *and* the graph silently
    stopped logging for the rest of the session, so everything after it was
    lost on a crash with no error anywhere.

    The statement below deletes 400 nodes (crossing the auto-vacuum threshold)
    and creates one, so a buffered index-keyed op coexists with the remap —
    the exact shape that broke.
    """
    p = tmp_path / "app.kgl"
    g = _open(p, storage)
    for i in range(400):
        g.cypher(f"CREATE (:Doomed {{id: {i}}})")
    g.save()  # checkpoint, so recovery depends only on the WAL below
    g.set_auto_vacuum(0.3)

    _crash_child(
        tmp_path,
        """
        g = open_durable()
        g.set_auto_vacuum(0.3)
        g.cypher(
            "MATCH (p:Doomed) DETACH DELETE p WITH count(*) AS n "
            "CREATE (:Survivor {id: 999, name: 'NewOne'}) RETURN n"
        )
        # Logging must still be alive after the vacuum, so this is recoverable too.
        g.cypher("CREATE (:Person {id: 1, name: 'Later'})")
        """,
        storage,
    )

    g = _open(p, storage)
    assert g.cypher("MATCH (d:Doomed) RETURN count(*) AS c").scalar() == 0, (
        "the deletions that triggered the vacuum must be recovered"
    )
    assert g.cypher("MATCH (s:Survivor) RETURN s.name AS n").scalar() == "NewOne"
    assert g.cypher("MATCH (p:Person) RETURN p.name AS n").scalar() == "Later", (
        "the graph must still be logging after a vacuum"
    )


# ── mode gating ──────────────────────────────────────────────────────


def test_durable_rejects_disk_mode(tmp_path):
    """Disk graphs commit by publishing an immutable generation, so a logical
    WAL is not the right durability boundary for them. The refusal must say so
    and point at what to do instead — silently accepting the flag would promise
    crash safety the mode does not provide."""
    with pytest.raises(ValueError) as excinfo:
        kglite.open(str(tmp_path / "g"), storage="disk", durable=True)
    message = str(excinfo.value)
    assert "storage='disk'" in message
    assert "save()" in message, "must name the supported alternative"
    assert "mapped" in message, "must name the modes that do support durability"


def test_non_durable_open_writes_no_wal(tmp_path):
    g = kglite.open(str(tmp_path / "app.kgl"), durable=False)
    g.cypher("CREATE (:Person {id: 1})")
    g.save()
    # Non-durable mode never creates a WAL sidecar.
    assert not (tmp_path / "app.kgl-wal").exists()
