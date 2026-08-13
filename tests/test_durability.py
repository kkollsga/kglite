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

A parent that seeds a checkpoint must ``del`` its handle before spawning the
child. ``kglite.open`` holds a cross-process single-writer lease for the life
of the graph, so a parent still holding one would block the child outright —
and a parent that kept writing alongside the child would be the very
lost-update bug the lease exists to prevent (see ``test_single_writer.py``).
The ``del`` is the handover, not a formality.

Every crash test runs for each storage mode that supports durability
(:data:`DURABLE_STORAGE_MODES`). ``storage="disk"`` is deliberately excluded
and its refusal is asserted in ``test_durable_rejects_disk_mode``.

Tests below the "durability levels" heading run for each level that keeps a
log (:data:`LOGGING_LEVELS`). Note what a ``SIGKILL`` can and cannot show:
it kills the *process*, which is precisely ``durable="normal"``'s guarantee,
but it cannot evict the kernel page cache, so nothing here tests — or
claims to test — the OS-crash and power-loss cases that separate
``"normal"`` from ``"full"``.
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
    reason="SIGKILL is POSIX-only; this is a crash test and must not degrade into a graceful-shutdown test on Windows",
)


# `storage=` selects a backend for a graph being *created*; an existing path is
# loaded, and the load decides the backend. A `.kgl` checkpoint records no
# storage mode, so a reopened one always comes back as memory — passing the mode
# again is now an `ArgumentError` rather than being silently ignored.
#
# So the `[mapped]` parametrisation below means **"created mapped, recovered as
# memory"**, and cannot mean anything else while `.kgl` carries no mode. That is
# still a real durability test — the WAL frames under replay were written by a
# mapped graph — but it does not exercise recovery *into* the mapped backend,
# and before the kwarg was gated it silently claimed to.


def _open_body(storage: str, durable: object = True) -> str:
    """Source text for a child script's `open_durable()`.

    Emitted as a runtime check rather than a fixed kwarg string because a child
    may call `open_durable()` more than once — creating the graph on the first
    call and reopening it after a checkpoint on later ones.
    """
    mode = "None" if storage == "memory" else repr(storage)
    return textwrap.dedent(
        f"""
        def open_durable():
            kwargs = {{"durable": {durable!r}}}
            storage = {mode}
            if storage is not None and not os.path.exists(path):
                kwargs["storage"] = storage
            return kglite.open(path, **kwargs)
        """
    ).strip()


def _open(path, storage: str = "memory", durable: object = True):
    """Open *path* in *storage* mode at durability level *durable*
    (parent-side counterpart)."""
    kwargs = {"durable": durable}
    if storage != "memory" and not os.path.exists(str(path)):
        kwargs["storage"] = storage
    return kglite.open(str(path), **kwargs)


def _child_script(tmp_path, body: str, storage: str, ending: str, durable: object = True) -> str:
    return textwrap.dedent(
        f"""
        import kglite, os, signal
        path = {str(tmp_path / "app.kgl")!r}

        {textwrap.indent(_open_body(storage, durable), "        ").strip()}

        {textwrap.indent(textwrap.dedent(body), "        ").strip()}
        {ending}
        """
    )


def _crash_child(tmp_path, body: str, storage: str = "memory", durable: object = True) -> None:
    """Run *body* in a child that hard-exits (``os._exit``) at the end — no
    atexit, no Python finalizers, no clean close. Models a power loss
    mid-session."""
    script = _child_script(tmp_path, body, storage, "os._exit(0)", durable)
    # Child must import the same built extension.
    subprocess.run([PYBIN, "-c", script], check=True, env=dict(os.environ))


def _sigkill_child(tmp_path, body: str, storage: str = "memory", durable: object = True) -> None:
    """Run *body* in a child that then ``SIGKILL``s itself.

    Stronger than ``_crash_child``: ``SIGKILL`` cannot be caught, blocked, or
    handled, so no interpreter teardown, no buffered-stdio flush and no
    ``Drop`` runs. The assertion on ``returncode`` is load-bearing — a child
    that exited any other way would mean the test had stopped being a crash
    test.
    """
    script = _child_script(tmp_path, body, storage, "os.kill(os.getpid(), signal.SIGKILL)", durable)
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
    del g  # hand the write lease to the child; see the note at the top

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
    del g  # hand the write lease to the child; see the note at the top

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
    del g  # hand the write lease to the child; see the note at the top

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
        '<http://www.w3.org/2000/01/rdf-schema#label> "human" .\n',
        encoding="utf-8",
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
    del g  # hand the write lease to the child; see the note at the top

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


# ── recovery-on-open is unconditional ────────────────────────────────
#
# Opening a path is a decision about *that path's data*, not only about how
# future writes will be logged. A sidecar holding commits the checkpoint does
# not contain is unrecovered data at every level, so `off` — which would
# neither replay them nor keep them past the next `save()` — is refused rather
# than silently handing back a graph that is missing committed writes.


@pytest.mark.parametrize("level", [False, "off"])
def test_off_refuses_to_open_over_an_unreplayed_log(tmp_path, level):
    """The data-loss shape this refusal exists for: a crashed durable session
    leaves committed frames in the sidecar, and opening `off` used to ignore
    them — the first later `save()` then truncated the log and the commits were
    gone, with no error anywhere.

    The refusal must name the sidecar and both ways out, and the recovery route
    it advertises must actually work (asserted below the refusal)."""
    _crash_child(
        tmp_path,
        """
        g = open_durable()
        g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
        """,
        "memory",
    )
    assert (tmp_path / "app.kgl-wal").exists()
    assert not (tmp_path / "app.kgl").exists(), "the child never checkpointed"

    with pytest.raises(ValueError) as excinfo:
        kglite.open(str(tmp_path / "app.kgl"), durable=level)
    message = str(excinfo.value)
    assert "app.kgl-wal" in message, "must name the sidecar that holds the commits"
    assert "'full'" in message and "'normal'" in message, "must name the replaying levels"

    # The advertised route back: open at a logging level and the commits return.
    g = kglite.open(str(tmp_path / "app.kgl"), durable="full")
    assert g.cypher("MATCH (p:Person) RETURN p.name AS n").scalar() == "Alice"


@pytest.mark.parametrize("level", [False, "off"])
def test_off_accepts_a_log_the_checkpoint_already_contains(tmp_path, level):
    """The other direction, and what keeps the refusal non-vacuous: a sidecar
    whose frames are all at or below the checkpoint's `checkpoint_lsn` is the
    *harmless residue* of a crash between the `.kgl` write and the log
    truncation. Nothing is unrecovered, so `off` must still open — a refusal
    keyed on "the sidecar is non-empty" would strand those graphs."""
    path = tmp_path / "app.kgl"
    wal = tmp_path / "app.kgl-wal"

    g = _open(path, "memory", durable="full")
    g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
    residue = wal.read_bytes()  # the log as it stood one instant before save()
    g.save()  # folds 'Alice' in, stamps checkpoint_lsn, truncates the log
    del g  # release the single-writer lease

    wal.write_bytes(residue)  # crash between the .kgl write and the truncation

    g = kglite.open(str(path), durable=level)
    assert g.cypher("MATCH (p:Person) RETURN p.name AS n").scalar() == "Alice"


# ── durability levels ────────────────────────────────────────────────
#
# ``durable="normal"`` logs every commit but skips the per-commit barrier.
# Its guarantee is: **a mutation that has returned survives the process
# dying; an OS crash or power loss loses commits since the last save().**
#
# The first half is exactly what ``_sigkill_child`` models, so it is tested
# here as rigorously as ``"full"`` is. The second half is deliberately NOT
# tested: killing a process cannot evict the kernel page cache, so a test
# claiming to simulate power loss would be theatre. What is asserted instead
# is the honest boundary — ``"normal"`` writes a log, and recovery of that
# log is level-independent.

#: Levels that keep a write-ahead log. ``"off"`` is excluded: it has no log,
#: so none of the recovery contracts below apply to it.
LOGGING_LEVELS = ("full", "normal")


@requires_sigkill
@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
@pytest.mark.parametrize("level", LOGGING_LEVELS)
def test_logging_levels_survive_sigkill(tmp_path, storage, level):
    """**The claim the "normal" rung exists to make.** A committed mutation
    that has returned must survive an uncatchable ``SIGKILL`` — no interpreter
    teardown, no ``Drop``, no flush of any kind. ``"normal"`` must be
    indistinguishable from ``"full"`` here; that is the whole point, and the
    only thing separating them is a failure mode this test cannot produce.
    """
    _sigkill_child(
        tmp_path,
        """
        g = open_durable()
        g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
        g.cypher("CREATE (:Person:Staff {id: 2, name: 'Bob'})")
        g.cypher("MATCH (a:Person {id:1}),(b:Person {id:2}) CREATE (a)-[:KNOWS]->(b)")
        """,
        storage,
        durable=level,
    )
    # Never saved, so only the log can account for the data.
    assert not (tmp_path / "app.kgl").exists()
    assert (tmp_path / "app.kgl-wal").exists()

    g = _open(tmp_path / "app.kgl", storage, durable=level)
    names = sorted(r["n"] for r in g.cypher("MATCH (p:Person) RETURN p.name AS n"))
    assert names == ["Alice", "Bob"]
    assert g.cypher("MATCH (:Person)-[r:KNOWS]->(:Person) RETURN count(r) AS c").scalar() == 1
    assert g.cypher("MATCH (p:Person {id:2}) RETURN labels(p) AS l").scalar() == [
        "Person",
        "Staff",
    ]


@requires_sigkill
@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_normal_and_full_recover_identical_state(tmp_path, storage):
    """Same workload, same crash, two levels — the recovered graphs must be
    byte-for-byte equivalent in content. A divergence would mean the levels
    differ in *what they log*, not merely in when they barrier."""
    recovered = {}
    for level in LOGGING_LEVELS:
        run = tmp_path / level
        run.mkdir()
        _sigkill_child(
            run,
            """
            g = open_durable()
            for i in range(25):
                g.cypher("CREATE (:Item {id: $i, tag: $t})", params={"i": i, "t": f"t{i}"})
            g.cypher("MATCH (n:Item {id: 7}) SET n.tag = 'edited'")
            g.cypher("MATCH (n:Item {id: 9}) DELETE n")
            """,
            storage,
            durable=level,
        )
        g = _open(run / "app.kgl", storage, durable=level)
        recovered[level] = sorted((r["i"], r["t"]) for r in g.cypher("MATCH (n:Item) RETURN n.id AS i, n.tag AS t"))

    assert recovered["normal"] == recovered["full"]
    # And the workload actually exercised all three op shapes.
    assert len(recovered["full"]) == 24, "one node was deleted"
    assert (7, "edited") in recovered["full"]


@requires_sigkill
@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_normal_log_is_recoverable_by_a_full_reopen(tmp_path, storage):
    """The log format carries no level — it is a property of the *session*
    that wrote it, not of the file. A graph crashed under ``"normal"`` must
    recover completely when reopened under ``"full"`` (and the reverse), or
    the level would have silently become an on-disk format variant."""
    _sigkill_child(
        tmp_path,
        """
        g = open_durable()
        g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
        """,
        storage,
        durable="normal",
    )
    g = _open(tmp_path / "app.kgl", storage, durable="full")
    assert g.cypher("MATCH (p:Person) RETURN p.name AS n").scalar() == "Alice"


@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_normal_checkpoint_truncates_then_recovers_post_checkpoint(tmp_path, storage):
    """The checkpoint/log interaction at ``"normal"``, which is where the
    level's one genuine hazard lives.

    ``save()`` folds the log into a checkpoint and truncates it. Under
    ``"normal"`` the frames it folds in may still be in the page cache, so
    ``save()`` barriers the log *before* writing the checkpoint — otherwise a
    crash in the window between the checkpoint and the truncation could leave
    a stale prefix of the log, and replaying that prefix over a newer
    checkpoint would roll committed properties backwards.

    What this test pins is the observable end of that contract: data from
    before the checkpoint and after it must both survive a crash, and neither
    may revert the other.
    """
    _crash_child(
        tmp_path,
        """
        g = open_durable()
        g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
        g.cypher("MATCH (p:Person {id: 1}) SET p.name = 'Alice2'")
        g.save()                       # checkpoint: folds + truncates the log
        g.cypher("CREATE (:Person {id: 2, name: 'Bob'})")
        g.cypher("MATCH (p:Person {id: 1}) SET p.name = 'Alice3'")
        """,
        storage,
        durable="normal",
    )
    assert (tmp_path / "app.kgl").exists(), "the checkpoint was written"

    g = _open(tmp_path / "app.kgl", storage, durable="normal")
    names = sorted(r["n"] for r in g.cypher("MATCH (p:Person) RETURN p.name AS n"))
    # 'Alice3' — NOT 'Alice2'. A reverted value here is the stale-prefix
    # replay hazard the pre-checkpoint barrier exists to prevent.
    assert names == ["Alice3", "Bob"]


@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_stale_wal_prefix_is_gated_by_the_checkpoint_lsn(tmp_path, storage):
    """Replay is gated on the LSN the checkpoint says it already contains, so a
    **stale WAL prefix cannot roll a newer checkpoint backwards**.

    Why this test and not
    ``test_normal_checkpoint_truncates_then_recovers_post_checkpoint``: that one
    crashes the child *after* ``save()`` has already truncated, so the WAL it
    recovers holds only post-checkpoint frames — the one input that separates a
    gated replay from an ungated one is absent, and it passes identically with
    the gate deleted. It pins the pre-checkpoint barrier, not the gate.

    The barrier stops a stale prefix *arising* from a clean crash; nothing stops
    one arriving another way — an operator restoring a sidecar from backup, a
    half-copied graph directory, a filesystem that resurrects a truncated file.
    So the hazard is constructed directly: a real log captured before a
    checkpoint is written back over the sidecar afterwards, with a genuinely
    post-checkpoint frame appended behind it. Recovery must skip the first and
    apply the second — asserting both is what keeps this non-vacuous, since a
    gate that skipped *everything* would satisfy the first assertion alone.
    """
    path = tmp_path / "app.kgl"
    wal = tmp_path / "app.kgl-wal"

    g = _open(path, storage, durable="full")
    # The sidecar exists with a header and no frames the moment it is opened;
    # reading it here derives the header length rather than hard-coding it, so
    # the frame splice below survives a header-format change.
    header = wal.read_bytes()
    assert header, "the sidecar must exist before any commit for this splice"

    g.cypher("CREATE (:Person {id: 1, name: 'Alice1'})")
    stale = wal.read_bytes()
    assert len(stale) > len(header), "the captured prefix must hold a real frame"

    g.save()  # checkpoint 1 — folds 'Alice1' in, truncates the log
    g.cypher("MATCH (p:Person {id: 1}) SET p.name = 'Alice2'")
    g.save()  # checkpoint 2 — folds 'Alice2' in, truncates again

    # Committed after the newest checkpoint, so it lives only in the log.
    g.cypher("CREATE (:Person {id: 2, name: 'Bob'})")
    fresh_frames = wal.read_bytes()[len(header) :]
    assert fresh_frames, "the post-checkpoint frame must have been logged"

    del g  # release the single-writer lease; no save(), so checkpoint 2 stands

    # The hazard: the pre-checkpoint log, back in front of the frames that
    # legitimately postdate it.
    wal.write_bytes(stale + fresh_frames)

    g = _open(path, storage, durable="full")
    names = sorted(r["n"] for r in g.cypher("MATCH (p:Person) RETURN p.name AS n"))
    # 'Alice1' here would mean the stale frame was folded over the newer
    # checkpoint; a missing 'Bob' would mean the gate ate a live frame too.
    assert names == ["Alice2", "Bob"]


# ── level spellings and validation ───────────────────────────────────


@pytest.mark.parametrize(
    ("flag", "name"),
    [(True, "full"), (False, "off")],
)
def test_bool_spellings_match_their_named_levels(tmp_path, flag, name):
    """``True``/``False`` are accepted spellings of ``"full"``/``"off"``, not a
    second code path. Equivalence is observable through whether a log exists."""
    by_flag = tmp_path / "flag"
    by_name = tmp_path / "name"
    by_flag.mkdir()
    by_name.mkdir()
    for target, level in ((by_flag, flag), (by_name, name)):
        g = kglite.open(str(target / "app.kgl"), durable=level)
        g.cypher("CREATE (:Person {id: 1})")
    assert (by_flag / "app.kgl-wal").exists() == (by_name / "app.kgl-wal").exists()
    assert (by_flag / "app.kgl-wal").exists() == (name == "full")


def test_normal_writes_a_wal_sidecar(tmp_path):
    """``"normal"`` logs — it is the barrier it skips, not the log."""
    g = kglite.open(str(tmp_path / "app.kgl"), durable="normal")
    g.cypher("CREATE (:Person {id: 1})")
    assert (tmp_path / "app.kgl-wal").exists()


def test_unknown_level_is_refused_with_the_valid_set(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        kglite.open(str(tmp_path / "app.kgl"), durable="fsync")
    message = str(excinfo.value)
    assert "fsync" in message, "must echo what was asked for"
    for level in ("full", "normal", "off"):
        assert level in message, "must list the valid levels"


def test_non_string_non_bool_level_is_a_type_error(tmp_path):
    with pytest.raises(TypeError):
        kglite.open(str(tmp_path / "app.kgl"), durable=2)


@pytest.mark.parametrize("level", LOGGING_LEVELS)
def test_disk_mode_refuses_every_logging_level(tmp_path, level):
    """**The levels are not uniform across storage modes.** A disk graph keeps
    no log at *any* level — the blocker is the generation-publish commit
    boundary, not barrier strength — so ``"normal"`` must be refused exactly as
    ``"full"`` is. Accepting it would hand back a graph with none of the
    crash safety the level name promises."""
    with pytest.raises(ValueError) as excinfo:
        kglite.open(str(tmp_path / "g"), storage="disk", durable=level)
    message = str(excinfo.value)
    assert level in message, "must name the level that was asked for"
    assert "storage='disk'" in message
    assert "'off'" in message, "must name the level disk does support"
    assert "mapped" in message, "must name the modes that do support logging"


@pytest.mark.parametrize("level", [False, "off", None])
def test_disk_mode_accepts_the_non_logging_levels(tmp_path, level):
    """The counterpart: ``off`` is fine on disk, and the tri-state default
    still resolves to it rather than raising."""
    # `tmp_path` is unique per parametrised case, so a fixed name is safe and
    # keeps the level out of the filename (``"off"`` would carry quotes).
    g = kglite.open(str(tmp_path / "g"), storage="disk", durable=level)
    g.cypher("CREATE (:Person {id: 1})")
    assert not (tmp_path / "g-wal").exists()


# ── sync(): the on-demand barrier ────────────────────────────────────


def test_sync_raises_without_a_log(tmp_path):
    """A caller who believes they bought power-safety and silently got nothing
    is the failure direction that costs data, so this raises rather than
    no-ops. The message must point at the levels that do support it."""
    g = kglite.open(str(tmp_path / "app.kgl"), durable="off")
    g.cypher("CREATE (:Person {id: 1})")
    with pytest.raises(ValueError) as excinfo:
        g.sync()
    message = str(excinfo.value)
    assert "save()" in message, "must name the alternative for a log-less graph"
    assert "normal" in message, "must name the level that makes sync() useful"


@pytest.mark.parametrize("level", LOGGING_LEVELS)
def test_sync_is_a_no_op_for_the_data(tmp_path, level):
    """``sync()`` writes no checkpoint and truncates nothing — it only makes
    the existing log durable. Content and the log's existence are unchanged."""
    g = kglite.open(str(tmp_path / "app.kgl"), durable=level)
    g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
    g.sync()
    g.sync()  # idempotent
    assert g.cypher("MATCH (p:Person) RETURN count(*) AS c").scalar() == 1
    assert (tmp_path / "app.kgl-wal").exists()
    assert not (tmp_path / "app.kgl").exists(), "sync() is not a checkpoint"


@requires_sigkill
@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_sync_then_crash_recovers_everything(tmp_path, storage):
    """``sync()`` must not disturb the log: commits before *and* after it are
    still recovered. A barrier that truncated or reordered would show up here.
    """
    _sigkill_child(
        tmp_path,
        """
        g = open_durable()
        g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
        g.sync()
        g.cypher("CREATE (:Person {id: 2, name: 'Bob'})")
        """,
        storage,
        durable="normal",
    )
    g = _open(tmp_path / "app.kgl", storage, durable="normal")
    names = sorted(r["n"] for r in g.cypher("MATCH (p:Person) RETURN p.name AS n"))
    assert names == ["Alice", "Bob"]


@requires_sigkill
def test_sync_flushes_pending_ops_into_the_log(tmp_path):
    """``sync()`` folds anything still buffered in the capture layer into a
    frame before barriering, so it can never report success over ops that
    never reached the log. Regression guard for the shape of bug that an
    internal-only barrier with a single caller previously allowed."""
    _sigkill_child(
        tmp_path,
        """
        g = open_durable()
        g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
        g.sync()
        """,
        "memory",
        durable="normal",
    )
    g = _open(tmp_path / "app.kgl", "memory", durable="normal")
    assert g.cypher("MATCH (p:Person) RETURN p.name AS n").scalar() == "Alice"
