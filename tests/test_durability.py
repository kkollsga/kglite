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
import warnings

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
def test_create_set_delete_in_one_session_recover_exactly(tmp_path, storage):
    """Every write shape a statement can take, captured and replayed exactly.

    Since construction became columnar, a ``CREATE`` appends a row to the
    type's master store rather than building a row-shaped node, and a
    ``DELETE`` tombstones one — neither of which passes through the
    ``node_weight_mut`` seam the WAL recorder originally hooked. A create is
    still recorded by ``add_node``, a property write by
    ``set_node_property``, and a title write by ``set_node_title``; this runs
    all of them in one durable session and compares the recovered graph
    against what the crashing process actually had, so a capture that stopped
    seeing any one of them shows up as a difference rather than as a plausible
    graph.
    """
    live, recovered = _live_and_recovered(
        tmp_path,
        """
        g = open_durable()
        g.cypher("CREATE (:T {id: 1, tag: 'one'})")
        g.cypher("CREATE (:T {id: 2, tag: 'two'})")
        g.cypher("CREATE (:T {id: 3, tag: 'three'})")
        g.cypher("MATCH (n:T {id: 2}) SET n.tag = 'edited'")
        g.cypher("MATCH (n:T {id: 3}) DETACH DELETE n")
        g.cypher("CREATE (:T {id: 4, tag: 'four'})")
        g.cypher("MATCH (n:T {id: 4}) SET n.name = 'retitled'")
        """,
        storage,
    )
    assert recovered == live, f"recovered {recovered} != live {live}"
    assert "'edited'" in live and "(3," not in live, f"the fixture is not exercising SET/DELETE: {live}"


@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_typed_columns_do_not_change_what_the_wal_replays(tmp_path, storage):
    """A column's storage type must not be visible to WAL capture or replay.

    Phase 5(ii) types a newly created columnar column from declared metadata or
    the value in hand instead of always building a `TypedColumn::Mixed`. Both
    WAL layers read through `NodeView`, which resolves a property to a `Value`
    regardless of the column holding it — but a typed column is exactly where a
    silent coercion would hide (an `Int64` column accepting a float by
    truncation, a `Float64` column widening an integer on the way back out), and
    a coercion in capture or replay is unrecoverable data loss rather than a
    wrong answer you can re-query.

    Every value below is written into a column the store has to *create*, and
    each one exercises a different typed arm: int, float, string, bool, and the
    demote-to-Mixed case where the same property arrives with two types.
    """
    g = _open(tmp_path / "app.kgl", storage)
    g.cypher("CREATE (:Item {id: 1, seed: 1})")
    g.cypher("CREATE (:Item {id: 2, seed: 2})")
    g.save()  # checkpoint: the type is columnar from here
    del g

    _crash_child(
        tmp_path,
        """
        g = open_durable()
        g.cypher("MATCH (n:Item {id:1}) SET n.count = 7")
        g.cypher("MATCH (n:Item {id:1}) SET n.ratio = 0.5")
        g.cypher("MATCH (n:Item {id:1}) SET n.tag = 'seven'")
        g.cypher("MATCH (n:Item {id:1}) SET n.flag = true")
        """,
        storage,
    )

    g = _open(tmp_path / "app.kgl", storage)
    row = g.cypher("MATCH (n:Item {id:1}) RETURN n.count AS c, n.ratio AS r, n.tag AS t, n.flag AS f").to_list()[0]
    assert row["c"] == 7 and isinstance(row["c"], int)
    assert row["r"] == 0.5 and isinstance(row["r"], float)
    assert row["t"] == "seven"
    assert row["f"] is True
    # The row that never carried the new properties still reads absent, not a
    # backfilled zero from the appended column's null padding.
    assert g.cypher("MATCH (n:Item {id:2}) RETURN n.count AS c").scalar() is None


#: One property, a different value type per node. Mixed types under one
#: property are legal in a live graph (`Value` is a sum type; a columnar
#: column demotes to `Mixed`), so they have to survive recovery too — and a
#: bool/int/float trio needs `isinstance`, since `True == 1` and `2.0 == 2`
#: in Python. Dates are covered in the Rust replay tests instead: the Python
#: surface renders a `DateTime` as its ISO string, so a date coerced to text
#: would read back identical here and the check would be vacuous.
MIXED_BY_ID = {
    1: 1,
    2: "two",
    3: 3.5,
    4: True,
    5: [1, 2],
}


def _assert_mixed_intact(g, expected):
    for node_id, want in expected.items():
        got = g.cypher(f"MATCH (n:Item {{id:{node_id}}}) RETURN n.mixedish AS m").scalar()
        assert got == want and isinstance(got, type(want)), (
            f"node {node_id}: {got!r} ({type(got).__name__}) != {want!r} ({type(want).__name__})"
        )


@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_a_heterogeneous_property_keeps_its_types_across_replay(tmp_path, storage):
    """Recovery is value-faithful, not merely value-shaped.

    Replay folds a whole node type's upserts into one bulk `add_nodes` call,
    whose `DataFrame` columns are singly typed — so a property written as an
    int on one node and a string on another used to come back as two strings
    (and an int/float pair as two floats). Type loss across a crash cannot be
    re-queried away, which is what makes this a durability test rather than a
    correctness one.
    """
    g = _open(tmp_path / "app.kgl", storage)
    for node_id in MIXED_BY_ID:
        g.cypher(f"CREATE (:Item {{id: {node_id}, seed: {node_id}}})")
    del g

    def literal(value):
        # Cypher spells booleans lowercase; every other value here has a
        # `repr` Cypher reads the same way Python does.
        return str(value).lower() if isinstance(value, bool) else repr(value)

    sets = "\n".join(
        f'g.cypher("MATCH (n:Item {{id:{node_id}}}) SET n.mixedish = {literal(value)}")'
        for node_id, value in MIXED_BY_ID.items()
    )
    _crash_child(tmp_path, "g = open_durable()\n" + sets, storage)

    g = _open(tmp_path / "app.kgl", storage)
    _assert_mixed_intact(g, MIXED_BY_ID)


@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_node_ids_and_titles_keep_their_types_across_replay(tmp_path, storage):
    """Identity is a value too.

    ``id`` and ``title`` ride the same bulk load as the properties, but cannot
    be held back from it — rows are addressed by them — so a type carrying an
    integer id on one node and a string id on another is replayed one shape at
    a time. Getting this wrong does more than mistype a field: an edge whose
    endpoint id was stringified matches nothing, and the loader *vivifies* a
    stub for it, so recovery invents a node that never existed.
    """
    _crash_child(
        tmp_path,
        """
        g = open_durable()
        g.cypher("CREATE (:Item {id: 1, title: 'one'})")
        g.cypher("CREATE (:Item {id: 'x', title: 2})")
        g.cypher("MATCH (a:Item {id:1}), (b:Item {id:'x'}) CREATE (a)-[:LINKS]->(b)")
        """,
        storage,
    )

    g = _open(tmp_path / "app.kgl", storage)
    rows = g.cypher("MATCH (n:Item) RETURN n.id AS id, n.title AS t").to_list()
    by_id = {r["id"]: r["t"] for r in rows}
    assert len(rows) == 2, f"no invented stub node: {rows}"
    # `1` stringified would key this dict under `'1'`, and `2` under `'2'`.
    assert set(by_id) == {1, "x"}, f"ids kept their types: {rows}"
    assert by_id[1] == "one"
    assert by_id["x"] == 2 and isinstance(by_id["x"], int)
    assert g.cypher("MATCH (a:Item {id:1})-[r:LINKS]->(b:Item {id:'x'}) RETURN count(r) AS c").scalar() == 1


@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_a_heterogeneous_property_survives_a_folded_replay(tmp_path, storage):
    """The same, with the mixed property folded against later ops on the same
    node.

    The per-value writes land *after* the bulk upsert (which runs in `replace`
    mode and clears a row's properties first), so this pins the two halves in
    the right order: the last write to the mixed property still wins, and the
    ordinary properties written alongside it in the same frame survive.
    """
    g = _open(tmp_path / "app.kgl", storage)
    g.cypher("CREATE (:Item {id: 1, seed: 1})")
    g.cypher("CREATE (:Item {id: 2, seed: 2})")
    g.save()  # checkpoint first: the type is columnar for the replay below
    del g

    _crash_child(
        tmp_path,
        """
        g = open_durable()
        g.cypher("MATCH (n:Item {id:1}) SET n.mixedish = 1")
        g.cypher("MATCH (n:Item {id:2}) SET n.mixedish = 'two'")
        g.cypher("MATCH (n:Item {id:1}) SET n.mixedish = 4.5, n.note = 'later'")
        g.cypher("MATCH (n:Item {id:2}) SET n.note = 'kept'")
        """,
        storage,
    )

    g = _open(tmp_path / "app.kgl", storage)
    _assert_mixed_intact(g, {1: 4.5, 2: "two"})
    row = g.cypher("MATCH (n:Item) RETURN n.id AS id, n.note AS note, n.seed AS seed").to_list()
    assert sorted((r["id"], r["note"], r["seed"]) for r in row) == [
        (1, "later", 1),
        (2, "kept", 2),
    ]


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


# ── the save side of the same rule ───────────────────────────────────
#
# ``kglite.load()`` is deliberately NOT guarded: it is the primitive durable
# recovery is itself built on, and the documented way to read a graph another
# process is writing durably, where a sidecar ahead of the checkpoint is the
# steady state rather than a fault. So the hazard still reaches the disk from
# the other end — load, mutate, ``save()`` back over the path — and is refused
# there instead, which closes it without taking the read away.


def _crashed_writer_leaves_a_commit(tmp_path) -> None:
    """Seed ``app.kgl`` with ``age=1``, then leave a committed ``age=2`` in the
    sidecar that no checkpoint contains — a durable writer that died between a
    commit and its next ``save()``."""
    g = _open(tmp_path / "app.kgl", "memory", durable="full")
    g.cypher("CREATE (:Person {id: 1, name: 'Alice', age: 1})")
    g.save()
    del g  # release the single-writer lease before the child opens it

    _crash_child(
        tmp_path,
        """
        g = open_durable()
        g.cypher("MATCH (p:Person {id: 1}) SET p.age = 2")
        """,
    )
    assert (tmp_path / "app.kgl-wal").exists()


def test_save_over_an_unreplayed_log_is_refused(tmp_path):
    """The third route into the same corruption, measured end-to-end before
    this refusal existed: ``kglite.load()`` returns the checkpoint (silently
    missing ``age=2``), the caller writes ``age=3`` and saves — and because
    that save neither stamps ``checkpoint_lsn`` nor truncates the sidecar, the
    next durable open replayed the stale frame OVER it and ``age`` came back
    as ``2``. Saved data, rolled back to an older commit, with no error
    anywhere.

    The save is now refused, and the refusal must name both ways out. Both are
    then exercised below, because a refusal whose advertised remedy does not
    work is worse than none."""
    _crashed_writer_leaves_a_commit(tmp_path)
    path = tmp_path / "app.kgl"

    g = kglite.load(str(path))
    assert g.cypher("MATCH (p:Person) RETURN p.age AS a").scalar() == 1, (
        "load() reads the checkpoint alone — the committed frame is invisible to it"
    )
    g.cypher("MATCH (p:Person {id: 1}) SET p.age = 3")
    with pytest.raises(ValueError) as excinfo:
        g.save()
    message = str(excinfo.value)
    assert "app.kgl-wal" in message, "must name the sidecar that holds the commits"
    assert "'full'" in message and "'normal'" in message, "must name the replaying levels"
    assert "move the sidecar aside" in message, "must name the deliberate-discard exit"
    del g

    # Nothing was written and nothing was consumed: the checkpoint still holds
    # what it did, and the commit is still there for the advertised route.
    assert kglite.load(str(path)).cypher("MATCH (p:Person) RETURN p.age AS a").scalar() == 1
    g = kglite.open(str(path), durable="full")
    assert g.cypher("MATCH (p:Person) RETURN p.age AS a").scalar() == 2
    g.cypher("MATCH (p:Person {id: 1}) SET p.age = 3")
    g.save()  # the durable owner's checkpoint folds the log in and truncates it
    del g

    # …and once recovered, the same load → write → save round trip goes
    # through, and survives a durable reopen.
    g = kglite.load(str(path))
    g.cypher("MATCH (p:Person {id: 1}) SET p.age = 4")
    g.save()
    del g
    assert kglite.open(str(path), durable="full").cypher("MATCH (p:Person) RETURN p.age AS a").scalar() == 4


def test_save_as_onto_a_path_whose_log_runs_ahead_is_refused(tmp_path):
    """The target path's sidecar is what matters, not the origin's. Writing a
    graph loaded from somewhere else over a path with unreplayed frames
    orphans them in exactly the same way — the new checkpoint knows nothing
    about them, and the next durable open replays them over it."""
    _crashed_writer_leaves_a_commit(tmp_path)

    other = kglite.KnowledgeGraph()
    other.cypher("CREATE (:Person {id: 9, name: 'Elsewhere'})")
    with pytest.raises(ValueError) as excinfo:
        other.save(str(tmp_path / "app.kgl"))
    assert "app.kgl-wal" in str(excinfo.value)

    # A path with no sidecar at all is the common case and is untouched.
    other.save(str(tmp_path / "fresh.kgl"))
    assert (tmp_path / "fresh.kgl").exists()


def test_save_over_crash_residue_still_works(tmp_path):
    """What keeps the refusal non-vacuous in the other direction: frames the
    checkpoint already folded in are the harmless residue of a crash between
    the ``.kgl`` write and the log truncation. A refusal keyed on "the sidecar
    is non-empty" would strand every such graph behind an error."""
    path = tmp_path / "app.kgl"
    wal = tmp_path / "app.kgl-wal"

    g = _open(path, "memory", durable="full")
    g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
    residue = wal.read_bytes()  # the log one instant before the checkpoint
    g.save()
    del g
    wal.write_bytes(residue)  # crash between the .kgl write and the truncation

    g = kglite.load(str(path))
    g.cypher("CREATE (:Person {id: 2, name: 'Bob'})")
    g.save()
    assert kglite.load(str(path)).cypher("MATCH (p:Person) RETURN count(*) AS c").scalar() == 2


# ── the read side: staleness is warned about, never hidden ───────────
#
# ``load()`` and ``open_session()`` stay open over an unrecovered sidecar —
# reading a checkpoint while another process writes the path durably is what
# they are for, and there a log ahead of the checkpoint is the steady state.
# What they must not do is stay *silent* about it: the caller gets a graph that
# is missing committed writes and nothing in the return value says so. So the
# refusal ``kglite.open`` raises becomes a warning here, naming the sidecar,
# how far behind the checkpoint is, and the entry point that replays it.


def test_load_warns_when_the_sidecar_runs_ahead(tmp_path):
    """The silent-staleness shape: ``age=2`` is committed and durable, the
    checkpoint still says ``age=1``, and ``load()`` used to hand back the
    checkpoint with no signal of any kind — the caller's only clue was the
    ``save()`` refusal much later, if they ever saved at all."""
    _crashed_writer_leaves_a_commit(tmp_path)
    path = tmp_path / "app.kgl"

    with pytest.warns(UserWarning) as record:
        g = kglite.load(str(path))
    message = str(record[0].message)
    assert "app.kgl-wal" in message, "must name the sidecar that holds the commits"
    assert "1 commit" in message, f"must quantify the gap, got: {message}"
    assert "kglite.open" in message, "must name the entry point that replays them"

    # The data contract is unchanged: the checkpoint is still served, and the
    # commit is still in the log for the route the warning advertises.
    assert g.cypher("MATCH (p:Person) RETURN p.age AS a").scalar() == 1
    del g
    g = kglite.open(str(path), durable="full")
    assert g.cypher("MATCH (p:Person) RETURN p.age AS a").scalar() == 2


def test_open_session_warns_when_the_sidecar_runs_ahead(tmp_path):
    """``open_session()`` is ``load().session()`` in one call and reads the
    checkpoint the same way, so it carries the same blind spot — and, serving a
    thread pool, is the likelier place for stale reads to spread."""
    _crashed_writer_leaves_a_commit(tmp_path)

    with pytest.warns(UserWarning, match="app.kgl-wal"):
        session = kglite.open_session(str(tmp_path / "app.kgl"))
    assert session.cypher("MATCH (p:Person) RETURN p.age AS a").scalar() == 1


def test_load_is_silent_when_nothing_is_unrecovered(tmp_path):
    """What keeps the warning worth reading. A sidecar whose frames the
    checkpoint already folded in is crash residue, not missing data, and a
    warning keyed on "a sidecar exists" would fire on every durable graph's
    normal reopen until someone silenced the category — taking the real case
    with it."""
    path = tmp_path / "app.kgl"
    wal = tmp_path / "app.kgl-wal"

    g = _open(path, "memory", durable="full")
    g.cypher("CREATE (:Person {id: 1, name: 'Alice'})")
    residue = wal.read_bytes()  # the log one instant before the checkpoint
    g.save()
    del g
    wal.write_bytes(residue)  # crash between the .kgl write and the truncation

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        assert kglite.load(str(path)).cypher("MATCH (p:Person) RETURN p.name AS n").scalar() == "Alice"
        kglite.open_session(str(path))

    # …and neither does the ordinary case, with no sidecar at all.
    wal.unlink()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kglite.load(str(path))


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


# ── recovery fidelity: the live graph is the oracle ──────────────────
#
# Every test above asks "did the data survive?". These ask the stronger
# question the WAL actually promises: is the recovered graph the *same
# graph* the crashed process had? Recovery folds ops by ``(node_type,
# id)``, so anything that lets two live nodes share one id is silently
# merged on replay — a node count that drops between the live session and
# the recovered one.


def _live_and_recovered(tmp_path, body, storage="memory"):
    """Run *body* in a crashing child, returning ``(live, recovered)`` rows.

    The child prints its own final state before dying, so the comparison is
    against what the process actually had — not against what the test
    author believed it would have. ``flush=True`` is load-bearing:
    ``os._exit`` skips stdio flushing, so an unflushed print would silently
    hand back an empty "live" side and make the oracle vacuous.
    """
    probe = "MATCH (n:T) RETURN n.id AS id, n.tag AS tag"
    # Dedent *before* appending: `_child_script` dedents what it is given, and
    # an unindented tail line would make the common prefix empty and leave the
    # body's own indentation in the generated source.
    script = _child_script(
        tmp_path,
        textwrap.dedent(body).strip()
        + f"\nprint(sorted((d['id'], d['tag']) for d in g.cypher({probe!r}).to_dicts()), flush=True)",
        storage,
        "os._exit(0)",
    )
    done = subprocess.run([PYBIN, "-c", script], check=True, capture_output=True, env=dict(os.environ))
    live = done.stdout.decode().strip().splitlines()[-1]
    g = _open(tmp_path / "app.kgl", storage)
    recovered = sorted((d["id"], d["tag"]) for d in g.cypher(probe).to_dicts())
    return live, repr(recovered)


@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_recovery_matches_live_across_delete_then_create(tmp_path, storage):
    """``DELETE`` then ``CREATE`` must recover node-for-node.

    This shape collides on a plain in-memory graph (covered in Rust by
    ``test_auto_assigned_ids_are_never_reused_after_delete``) but happened
    *not* to collide through a durable session, whose per-commit fork
    reshapes the index space. It is pinned here anyway: the collision it
    guards against depends on allocator internals, not on anything a caller
    can see, so "durable mode is currently lucky" is not a property to leave
    untested. The genuinely red case is the checkpoint test below.
    """
    live, recovered = _live_and_recovered(
        tmp_path,
        """
        g = open_durable()
        for i in range(5):
            g.cypher("CREATE (:T {tag: 'a%d'})" % i)
        g.cypher("MATCH (n:T) WHERE n.tag IN ['a3','a4'] DELETE n")
        for i in range(3):
            g.cypher("CREATE (:T {tag: 'b%d'})" % i)
        """,
        storage,
    )
    assert recovered == live


@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_recovery_matches_live_across_checkpoint_then_create(tmp_path, storage):
    """Same oracle across a checkpoint, which is where the id allocator has
    to re-seed itself: the reopened graph must keep minting ids above the
    ones the ``.kgl`` already holds. Before the fix this history put *three*
    nodes on one id, and recovery kept one of them."""
    live, recovered = _live_and_recovered(
        tmp_path,
        """
        g = open_durable()
        for i in range(6):
            g.cypher("CREATE (:T {tag: 'a%d'})" % i)
        g.cypher("MATCH (n:T) WHERE n.tag IN ['a1','a2'] DELETE n")
        g.save()
        del g
        g = open_durable()
        for i in range(3):
            g.cypher("CREATE (:T {tag: 'c%d'})" % i)
        """,
        storage,
    )
    assert recovered == live


@pytest.mark.parametrize("storage", DURABLE_STORAGE_MODES)
def test_a_durable_graph_refuses_a_caller_supplied_duplicate_id(tmp_path, storage):
    """A durable graph refuses the write it cannot log, rather than logging one
    recovery would silently merge.

    ``id`` uniqueness is opt-in on a plain graph (CYPHER.md: "two ``CREATE (:T
    {id: 'k'})`` make two nodes"), but the WAL names every entity — nodes, and
    both endpoints of every edge — by its logical ``(node_type, id)``. A log in
    which one id denotes two nodes is not merely folded wrongly, it is
    *unwritable*: there is no discriminator to carry. Until 0.16.1 the second
    ``CREATE`` was accepted and the two nodes came back as one — a node lost
    across a reopen, with nothing at write time to warn anyone.

    So durable mode now refuses it, and the error names the two routes that do
    work. The refusal is scoped to a recording backend: a non-durable graph
    keeps the documented permissive behaviour, pinned below.
    """
    script = _child_script(
        tmp_path,
        """
        g = open_durable()
        g.cypher("CREATE (:T {id: 1, tag: 'first'})")
        try:
            g.cypher("CREATE (:T {id: 1, tag: 'second'})")
        except Exception as exc:
            print("REFUSED", exc, flush=True)
        else:
            print("ACCEPTED", flush=True)
        """,
        storage,
        "os._exit(0)",
    )
    done = subprocess.run([PYBIN, "-c", script], check=True, capture_output=True, env=dict(os.environ))
    out = done.stdout.decode()
    assert "REFUSED" in out, out
    # The message has to be actionable — both working routes by name.
    assert "MERGE" in out, out
    assert "primary key" in out and "define_schema" in out, out

    # And the refusal left the graph recoverable: the first node, intact.
    # Reaching this state at all is the proof that *replay* does not refuse its
    # own history — the child died without a checkpoint, so this open folds the
    # log back in, and it does so through the bulk upsert path (which keys by
    # ``(type, id)`` and therefore cannot produce a duplicate) before the graph
    # is wrapped for capture.
    g = _open(tmp_path / "app.kgl", storage)
    assert sorted((d["id"], d["tag"]) for d in g.cypher("MATCH (n:T) RETURN n.id AS id, n.tag AS tag").to_dicts()) == [
        (1, "first")
    ]
    # …and the gate covers a node that only exists because it was replayed.
    with pytest.raises(Exception, match="durable graph"):
        g.cypher("CREATE (:T {id: 1, tag: 'third'})")


def test_a_non_durable_graph_still_allows_duplicate_ids(tmp_path):
    """The counter-pin: the refusal is a property of *durable capture*, not of
    ``CREATE``. Without a log there is nothing that cannot be represented, and
    the documented opt-in-uniqueness behaviour is unchanged."""
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:T {id: 1, tag: 'first'})")
    g.cypher("CREATE (:T {id: 1, tag: 'second'})")
    assert g.cypher("MATCH (n:T) RETURN count(n) AS c").scalar() == 2


def test_a_durable_graph_takes_merge_as_the_route_the_refusal_names(tmp_path):
    """The guidance in the error must actually work: MERGE upserts the node
    rather than tripping the refusal."""
    script = _child_script(
        tmp_path,
        """
        g = open_durable()
        g.cypher("CREATE (:T {id: 1, tag: 'first'})")
        g.cypher("MERGE (n:T {id: 1}) ON MATCH SET n.tag = 'second'")
        """,
        "memory",
        "os._exit(0)",
    )
    subprocess.run([PYBIN, "-c", script], check=True, capture_output=True, env=dict(os.environ))
    g = _open(tmp_path / "app.kgl", "memory")
    assert sorted((d["id"], d["tag"]) for d in g.cypher("MATCH (n:T) RETURN n.id AS id, n.tag AS tag").to_dicts()) == [
        (1, "second")
    ]


# ── a refused append poisons the handle ──────────────────────────────
#
# The failure this section exists for was measured on a filling disk: the
# statement that could not be logged was still applied in memory, its ops were
# already drained, and it was reported to the caller as a ``FileIoError`` — but
# nothing marked the graph. The next ``save()`` serialized whatever was in
# memory, which included the failed statement, so a write the caller had been
# told did NOT happen was committed to disk minutes later.
#
# ``kglite._fail_wal_append`` injects the append error, because no portable
# filesystem trick fails a write on an already-open append handle.


def _rows(g) -> list[int]:
    return sorted(d["id"] for d in g.cypher("MATCH (r:Row) RETURN r.id AS id").to_dicts())


def test_a_failed_append_is_reported_and_never_committed(tmp_path):
    """The whole shape, end to end: rows 0–1 checkpointed, rows 2–3 committed
    to the log, row 4's append fails. Row 4 must be reported as failed, must
    not be saveable, and must not be there on reopen."""
    path = tmp_path / "app.kgl"
    g = _open(path, "memory", durable="full")
    for i in (0, 1):
        g.cypher(f"CREATE (:Row {{id: {i}}})")
    g.save()  # checkpoint: folds 0–1 in and truncates the log
    for i in (2, 3):
        g.cypher(f"CREATE (:Row {{id: {i}}})")

    kglite._fail_wal_append(g, True)
    with pytest.raises(kglite.FileIoError):
        g.cypher("CREATE (:Row {id: 4})")

    # (a) the handle is poisoned: no further logged write, and neither route
    # back to disk. Every one of these used to succeed, and `save()` was the
    # one that turned an acknowledged failure into committed data.
    kglite._fail_wal_append(g, False)  # the disk "recovered" — irrelevant now
    with pytest.raises(kglite.FileIoError, match="no longer describes the graph"):
        g.cypher("CREATE (:Row {id: 5})")
    with pytest.raises(kglite.FileIoError, match="reopen"):
        g.save()
    with pytest.raises(kglite.FileIoError, match="reopen"):
        g.sync()
    del g  # release the single-writer lease

    # (b) reopen: exactly the writes that were acknowledged, and nothing else.
    g = kglite.open(str(path), durable="full")
    assert _rows(g) == [0, 1, 2, 3], "the failed statement must not be in the recovered graph"


def test_a_failed_append_consumes_no_lsn(tmp_path):
    """The accounting half. ``save()`` stamps ``checkpoint_lsn = next_lsn - 1``,
    so an LSN consumed by a frame that never reached the log makes the
    checkpoint claim a commit that does not exist — and the replay gate, which
    skips every frame at or below the stamp, then skips the next real one."""
    path = tmp_path / "app.kgl"
    g = _open(path, "memory", durable="full")
    g.cypher("CREATE (:Row {id: 0})")
    before = kglite._wal_next_lsn(g)

    kglite._fail_wal_append(g, True)
    with pytest.raises(kglite.FileIoError):
        g.cypher("CREATE (:Row {id: 1})")

    assert kglite._wal_next_lsn(g) == before, "an append that failed must not consume its LSN"


def test_the_poison_does_not_leak_into_a_fresh_handle(tmp_path):
    """What keeps the latch usable: it lives on the durable session, so the
    reopen the message tells the caller to perform actually gives them a
    working graph — and a *different* durable graph in the same process is
    untouched."""
    path = tmp_path / "app.kgl"
    other = tmp_path / "other.kgl"

    fresh = _open(other, "memory", durable="full")
    g = _open(path, "memory", durable="full")
    g.cypher("CREATE (:Row {id: 0})")
    kglite._fail_wal_append(g, True)
    with pytest.raises(kglite.FileIoError):
        g.cypher("CREATE (:Row {id: 1})")
    del g

    # The other graph never saw the failure.
    fresh.cypher("CREATE (:Row {id: 7})")
    fresh.save()
    del fresh

    # And the advertised exit works: reopen, write, save.
    g = kglite.open(str(path), durable="full")
    assert _rows(g) == [0]
    g.cypher("CREATE (:Row {id: 1})")
    g.save()
    del g
    assert _rows(kglite.open(str(path), durable="full")) == [0, 1]


def test_the_hooks_refuse_a_graph_with_no_log(tmp_path):
    """The hooks are only meaningful on a logged graph; a test that forgot
    ``durable=`` must fail loudly rather than assert nothing."""
    g = kglite.KnowledgeGraph()
    with pytest.raises(ValueError):
        kglite._fail_wal_append(g, True)
    with pytest.raises(ValueError):
        kglite._wal_next_lsn(g)


# ---------------------------------------------------------------------------
# handles derived from a durable graph
# ---------------------------------------------------------------------------
#
# A ``KnowledgeGraph`` handle owns the write-ahead log; a handle derived from it
# cannot (the log is an OS ``File`` handle, not shareable state). Two families of
# method used to hand back such a handle silently:
#
# - configuration mutations (``set_instructions``, ``define_schema``,
#   ``clear_schema``, ``lock_schema``, ``set_schema_version``) returned a *copy*
#   of the graph purely so the call could be chained. They now return the same
#   object, so there is no derived handle to lose the log in the first place.
# - fluent methods (``select``/``where``/``traverse``/set operations/…) return a
#   genuinely derived view, and must. Writing through one forks the graph away
#   from the original *and* reaches no log, so it is refused rather than lost.
#
# The measured failure before the fix: ``g2 = g.set_instructions("hi")`` on a
# durable graph, then ``g2.cypher("CREATE …")`` — no frame appended, the node
# absent from ``g``, and absent again after a reopen. An acknowledged commit,
# gone.


#: Every configuration method that returns the graph for chaining.
CHAINING_CONFIG_CALLS = [
    pytest.param(lambda g: g.set_instructions("hi"), id="set_instructions"),
    pytest.param(lambda g: g.define_schema({"nodes": {"Row": {"required": ["id"]}}}), id="define_schema"),
    pytest.param(lambda g: g.clear_schema(), id="clear_schema"),
    pytest.param(lambda g: g.lock_schema(), id="lock_schema"),
    pytest.param(lambda g: g.unlock_schema(), id="unlock_schema"),
    pytest.param(lambda g: g.set_schema_version(3), id="set_schema_version"),
]


@pytest.mark.parametrize("call", CHAINING_CONFIG_CALLS)
def test_a_config_call_returns_the_same_handle(tmp_path, call):
    """The fix at its root: these are graph mutations, not views, so the
    returned handle is the graph itself and keeps the log with it."""
    path = tmp_path / "cfg.kgl"
    g = _open(path, "memory", durable="full")
    g.cypher("CREATE (:Row {id: 1})")
    assert call(g) is g


@pytest.mark.parametrize("call", CHAINING_CONFIG_CALLS)
def test_writes_survive_a_rebind_through_a_config_call(tmp_path, call):
    """The repro. ``g = g.set_instructions(...)`` is the documented chaining
    form; every later write through the rebound handle must still be logged.

    Before the fix the rebound handle carried ``durable=None`` with
    ``source_path`` intact, so the write below was applied to a forked graph,
    appended no frame, and was simply absent on reopen.
    """
    path = tmp_path / "rebind.kgl"
    g = _open(path, "memory", durable="full")
    g.cypher("CREATE (:Row {id: 1})")

    g = call(g)  # the rebind that used to drop the log
    g.cypher("CREATE (:Row {id: 2})")
    del g  # crash-equivalent for the log: no save(), no close()

    assert _rows(kglite.open(str(path), durable="full")) == [1, 2]


def test_a_config_call_does_not_fork_the_graph(tmp_path):
    """The other half of the same defect: the returned copy forked on its first
    write, so the *original* handle did not see it either — the two handles
    disagreed about the graph's contents with nothing to signal it."""
    path = tmp_path / "fork.kgl"
    g = _open(path, "memory", durable="full")
    g2 = g.set_instructions("hi")
    g2.cypher("CREATE (:Row {id: 1})")
    assert _rows(g) == [1]


def test_sync_still_works_through_a_rebound_config_call(tmp_path):
    """``sync()`` is the durability barrier a ``normal`` graph relies on; a
    rebound handle used to raise "this graph has no write-ahead log"."""
    path = tmp_path / "syncable.kgl"
    g = _open(path, "memory", durable="normal")
    g = g.set_instructions("hi")
    g.cypher("CREATE (:Row {id: 1})")
    g.sync()
    del g
    assert _rows(kglite.open(str(path), durable="normal")) == [1]


#: Fluent methods that hand back a derived view of the same storage.
DERIVING_CALLS = [
    pytest.param(lambda g: g.select("Row"), id="select"),
    pytest.param(lambda g: getattr(g.select("Row"), "where")({"id": 1}), id="where"),
    pytest.param(lambda g: g.select("Row").sort("id"), id="sort"),
    pytest.param(lambda g: g.select("Row").expand(hops=1), id="expand"),
    pytest.param(lambda g: g.select("Row").union(g.select("Row")), id="union"),
    pytest.param(lambda g: g.date("2020-01-01"), id="date"),
]


@pytest.mark.parametrize("call", DERIVING_CALLS)
def test_a_derived_view_refuses_a_logged_write(tmp_path, call):
    """A view shares the storage but cannot share the log. A write through it
    would fork the graph away from the original *and* reach no log, so it is
    refused — loudly, naming the handle that does own the log."""
    path = tmp_path / "view.kgl"
    g = _open(path, "memory", durable="full")
    g.cypher("CREATE (:Row {id: 1})")

    view = call(g)
    with pytest.raises(ValueError, match="derived from a durable graph"):
        view.cypher("CREATE (:Row {id: 2})")

    # ...and the refusal is total: neither route back to disk is open either.
    with pytest.raises(ValueError, match="derived from a durable graph"):
        view.save()
    with pytest.raises(ValueError, match="derived from a durable graph"):
        view.sync()

    # The owner is untouched and still writes normally.
    g.cypher("CREATE (:Row {id: 3})")
    del view, g
    assert _rows(kglite.open(str(path), durable="full")) == [1, 3]


def test_the_fence_is_inherited_by_a_view_of_a_view(tmp_path):
    """One level of derivation is not the interesting case — a fluent chain is
    many, and every link must stay fenced."""
    path = tmp_path / "chain.kgl"
    g = _open(path, "memory", durable="full")
    g.cypher("CREATE (:Row {id: 1})")
    deep = getattr(g.select("Row"), "where")({"id": 1}).sort("id").limit(10)
    with pytest.raises(ValueError, match="derived from a durable graph"):
        deep.cypher("CREATE (:Row {id: 2})")


#: Fluent mutations — the ones that write through the derived handle a chain
#: produced. Each one used to apply to a fork nobody kept and log nothing.
FLUENT_MUTATIONS = [
    pytest.param(lambda g: g.select("Row").unique_values("id", store_as="ids"), id="unique_values_store_as"),
    pytest.param(lambda g: g.select("Row").add_properties({"Row": ["id"]}), id="add_properties"),
    pytest.param(lambda g: g.select("Row").count(store_as="n"), id="count_store_as"),
    pytest.param(lambda g: g.select("Row").calculate("id * 2", store_as="double"), id="calculate_store_as"),
]


@pytest.mark.parametrize("call", FLUENT_MUTATIONS)
def test_a_fluent_mutation_at_the_end_of_a_chain_is_refused(tmp_path, call):
    """The pattern the docstrings advertise — ``g = g.select(...).add_properties(...)``
    — is exactly the one that used to leave ``g`` permanently unlogged. Every
    selection-based mutation is refused on a durable graph, because there is no
    way to reach one except through a derived handle; the message names
    ``cypher()``, which expresses the same writes and is logged.
    """
    path = tmp_path / "chainmut.kgl"
    g = _open(path, "memory", durable="full")
    g.cypher("CREATE (:Row {id: 1})")
    with pytest.raises(ValueError, match="derived from a durable graph"):
        call(g)
    # Nothing reached the graph, and the route the message names does work.
    g.cypher("MATCH (r:Row) SET r.ids = toString(r.id)")
    del g
    reopened = kglite.open(str(path), durable="full")
    assert reopened.cypher("MATCH (r:Row) RETURN r.ids AS v").to_dicts() == [{"v": "1"}]


def test_a_fluent_mutation_still_returns_the_same_handle(tmp_path):
    """``unique_values(store_as=...)`` handed back a *copy* of the receiver for
    chaining. Off a durable graph that copy is harmless, but it is still the
    anti-pattern the fix removes — and on a CDC graph the write it performed was
    never published either, for the same missing commit boundary."""
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Row {id: 1})")
    sel = g.select("Row")
    assert sel.unique_values("id", store_as="ids") is sel


def test_a_detached_copy_of_a_durable_graph_writes_freely(tmp_path):
    """``copy()`` / ``to_subgraph()`` build an independent graph rather than a
    view of this one, so the fence must not follow them — otherwise the message's
    own advice would be unusable."""
    path = tmp_path / "detach.kgl"
    g = _open(path, "memory", durable="full")
    g.cypher("CREATE (:Row {id: 1})")

    c = g.copy()
    c.cypher("CREATE (:Row {id: 2})")
    assert _rows(c) == [1, 2]
    c.save(str(tmp_path / "copy.kgl"))

    sub = g.select("Row").to_subgraph()
    sub.cypher("CREATE (:Row {id: 9})")
    assert _rows(sub) == [1, 9]

    # ...and none of it reached the durable graph or its log.
    assert _rows(g) == [1]


def test_a_non_durable_graph_keeps_its_fluent_write_path(tmp_path):
    """The fence is durability-scoped: an ordinary in-memory graph has no log to
    diverge from, and its fluent mutations must keep working exactly as before."""
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Row {id: 1})")
    view = g.select("Row")
    view.cypher("CREATE (:Row {id: 2})")
    assert _rows(view) == [1, 2]
    g.select("Row").unique_values("id", store_as="ids")
    g.select("Row").count(store_as="n")


def test_a_view_of_a_poisoned_graph_is_refused_too(tmp_path):
    """B2's divergence latch and this fence are the two halves of one rule —
    a graph whose log refused an append must not be writable through *any*
    handle, including one derived after the fact."""
    path = tmp_path / "poison.kgl"
    g = _open(path, "memory", durable="full")
    g.cypher("CREATE (:Row {id: 0})")
    kglite._fail_wal_append(g, True)
    with pytest.raises(kglite.FileIoError):
        g.cypher("CREATE (:Row {id: 1})")

    view = g.select("Row")
    with pytest.raises(ValueError, match="derived from a durable graph"):
        view.cypher("CREATE (:Row {id: 2})")
    with pytest.raises(ValueError, match="derived from a durable graph"):
        view.save()


def test_update_through_a_selection_is_refused(tmp_path):
    """``update()`` is the remaining selection-based write, and it reaches the
    graph through the same derived handle everything else in ``FLUENT_MUTATIONS``
    does."""
    path = tmp_path / "update.kgl"
    g = _open(path, "memory", durable="full")
    g.cypher("CREATE (:Row {id: 1})")
    with pytest.raises(ValueError, match="derived from a durable graph"):
        g.select("Row").update({"tag": "x"})
    assert g.cypher("MATCH (r:Row) RETURN r.tag AS t").to_dicts() == [{"t": None}]


# ── a failed write is a typed failure, not a bare OSError ────────────
#
# The taxonomy is the contract `docs/python/error-handling.md` states and
# `.code` makes machine-readable, and the write path was outside it: every
# I/O failure in `save()`, `sync()` and `to_bytes()` was raised as `PyIOError`
# — the *builtin* `OSError` — so a full disk was indistinguishable from any
# unrelated OS error the call stack could produce and carried no `.code`.
#
# A read-only directory is the portable stand-in for ENOSPC: it fails the same
# `File::create` on the save temp, on the same code path, without needing a
# full volume.


def _readonly_dir(tmp_path, name: str):
    d = tmp_path / name
    d.mkdir()
    os.chmod(d, 0o500)
    return d


def test_a_failed_save_raises_the_documented_typed_error(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    d = _readonly_dir(tmp_path, "readonly")
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Row {id: 1})")
    try:
        with pytest.raises(kglite.FileIoError) as caught:
            g.save(str(d / "app.kgl"))
    finally:
        os.chmod(d, 0o700)
    assert isinstance(caught.value, kglite.KgError)
    assert caught.value.code == "FileIo", caught.value.code


def test_a_failed_durable_save_raises_the_same_typed_error(tmp_path):
    """The durable path is a different function with its own error mapping —
    the checkpoint prologue, the write, and the log truncation each convert
    separately, so the graph that carries a log is asserted too."""
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    live = tmp_path / "live.kgl"
    g = _open(live, "memory", durable="full")
    g.cypher("CREATE (:Row {id: 1})")
    d = _readonly_dir(tmp_path, "readonly")
    try:
        with pytest.raises(kglite.FileIoError):
            g.save(str(d / "app.kgl"))
    finally:
        os.chmod(d, 0o700)
        g.close()
