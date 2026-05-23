"""Phase 2 crunch-point parity tests.

Guards the `GraphWrite` trait expansion (Phase 2 of the 0.8.0 storage
refactor). These tests exercise the **mutation** surface — conflict
handling, mid-batch failure, schema-locked rejection, tombstone
visibility, MERGE idempotency, collect-then-delete snapshot — and
assert byte-identical observable behaviour across memory / mapped /
disk modes.

Each test builds one graph per storage mode, runs the same mutation
sequence, and compares the observable state. Following the
test_phase1_parity.py pattern: iterate STORAGE_MODES inside a single
test function so the cross-mode assertion is explicit.

Run: pytest -m parity tests/test_phase2_parity.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from kglite import KnowledgeGraph

pytestmark = pytest.mark.parity

STORAGE_MODES = ("memory", "mapped", "disk")
# Modes for which all Phase 2 parity invariants hold today. `disk` has
# three known pre-existing divergences documented at the bottom of this
# file — see `test_known_disk_divergences_*`. Phase 5 reconciles them.
STRICT_PARITY_MODES = ("memory", "mapped")
# 0.9.26 intentionally turned off Cypher CREATE / MERGE on disk-backed
# graphs (DiskGraph::new_at_path's mutation surface is bulk-loader-only).
# Tests that exercise mutating Cypher must iterate this subset and let
# the dedicated disk-lockout test below guard the rejection.
MUTATING_CYPHER_MODES = ("memory", "mapped")


def _new_kg(mode: str, path: str | None = None) -> KnowledgeGraph:
    if mode == "memory":
        return KnowledgeGraph()
    if mode == "mapped":
        return KnowledgeGraph(storage="mapped")
    if mode == "disk":
        if path is None:
            raise ValueError("mode='disk' requires path")
        return KnowledgeGraph(storage="disk", path=path)
    raise ValueError(f"unknown mode: {mode}")


def _build_seed(kg: KnowledgeGraph):
    """3 Person nodes with integer IDs + age + email."""
    df = pd.DataFrame(
        [
            {"pid": 1, "name": "Alice", "age": 30, "email": "alice@old.com"},
            {"pid": 2, "name": "Bob", "age": 25, "email": "bob@old.com"},
            {"pid": 3, "name": "Carol", "age": 40, "email": "carol@old.com"},
        ]
    )
    kg.add_nodes(df, "Person", "pid", "name")


def _rows(result) -> list[dict]:
    try:
        rows = [dict(r) for r in result]
    except TypeError:
        rows = list(result)
    return sorted(rows, key=lambda r: repr(sorted(r.items())))


def _snapshot(kg: KnowledgeGraph, cypher: str) -> list[dict]:
    return _rows(kg.cypher(cypher))


def _per_mode(tmp_path, fn, modes: tuple[str, ...] = STORAGE_MODES):
    """Run `fn(kg)` on each backend in `modes`, return {mode: kg}.

    `fn` performs the mutations under test. The returned kg's are kept
    alive so follow-up queries can compare state. Pass `modes` to
    restrict to a subset (e.g. `MUTATING_CYPHER_MODES` for tests that
    use CREATE / MERGE, which are disk-locked since 0.9.26).
    """
    kgs: dict[str, KnowledgeGraph] = {}
    for mode in modes:
        path = str(tmp_path / f"kg_{mode}") if mode == "disk" else None
        kg = _new_kg(mode, path=path)
        _build_seed(kg)
        fn(kg)
        kgs[mode] = kg
    return kgs


# ─── 1. Conflict-handling matrix across backends ────────────────────────────


@pytest.mark.parametrize("conflict", ["update", "replace", "skip", "preserve"])
def test_conflict_matrix_nodes(conflict, tmp_path):
    """Re-inserting a Person with a conflict mode produces identical state."""

    def mutate(kg):
        df = pd.DataFrame([{"pid": 1, "name": "Alice Updated", "email": "alice@new.com"}])
        kg.add_nodes(df, "Person", "pid", "name", conflict_handling=conflict)

    kgs = _per_mode(tmp_path, mutate)

    q = "MATCH (n:Person) WHERE n.id = 1 RETURN n.age AS age, n.email AS email"
    ref = _snapshot(kgs["memory"], q)
    # 'update' and 'replace' diverge on disk (pre-existing, Phase 5).
    # 'skip' and 'preserve' hold across all three modes.
    compare_modes = STRICT_PARITY_MODES if conflict in ("update", "replace") else STORAGE_MODES[1:]
    for mode in compare_modes:
        if mode == "memory":
            continue
        got = _snapshot(kgs[mode], q)
        assert got == ref, f"conflict='{conflict}' diverged in {mode}:\n  memory: {ref}\n  {mode}: {got}"


def test_conflict_matrix_edges_replace(tmp_path):
    """add_connections conflict=replace yields identical edge props."""

    def mutate(kg):
        df1 = pd.DataFrame([{"src": 1, "tgt": 2, "weight": 0.1}])
        kg.add_connections(df1, "KNOWS", "Person", "src", "Person", "tgt")
        df2 = pd.DataFrame([{"src": 1, "tgt": 2, "weight": 0.9}])
        kg.add_connections(
            df2,
            "KNOWS",
            "Person",
            "src",
            "Person",
            "tgt",
            conflict_handling="replace",
        )

    kgs = _per_mode(tmp_path, mutate)

    q = "MATCH (a:Person)-[r:KNOWS]->(b:Person) WHERE a.id = 1 AND b.id = 2 RETURN r.weight AS w"
    ref = _snapshot(kgs["memory"], q)
    for mode in ("mapped", "disk"):
        got = _snapshot(kgs[mode], q)
        assert got == ref, f"edge replace diverged in {mode}: {got} vs memory {ref}"


# ─── 2. Mid-batch failure semantics ─────────────────────────────────────────


def test_mid_batch_failure_schema_locked(tmp_path):
    """Schema-locked add_nodes with one bad row: observable state identical."""

    def mutate(kg):
        kg.lock_schema()
        df = pd.DataFrame(
            [
                {"pid": 4, "name": "Diana", "age": 22},
                {"pid": 5, "name": "Evan", "age": 33, "unknown_prop": "bad"},
                {"pid": 6, "name": "Frank", "age": 44},
            ]
        )
        try:
            kg.add_nodes(df, "Person", "pid", "name")
        except Exception:
            pass  # expected; we assert on the resulting state

    kgs = _per_mode(tmp_path, mutate)

    q = "MATCH (n:Person) RETURN n.id AS id"
    ref = _snapshot(kgs["memory"], q)
    for mode in ("mapped", "disk"):
        got = _snapshot(kgs[mode], q)
        assert got == ref, f"mid-batch failure state diverged in {mode}:\n  memory: {ref}\n  {mode}: {got}"


# ─── 3. Schema-locked rejection parity ──────────────────────────────────────


def test_schema_locked_unknown_property_message(tmp_path):
    """Error message for CREATE with unknown property is identical across
    the modes that support mutating Cypher.

    Disk-backed graphs reject the CREATE itself in 0.9.26+ (before any
    schema check runs) — see test_disk_create_lockout_message. So the
    schema-violation message parity is asserted across memory + mapped
    only, and the disk lockout is guarded by its own dedicated test.
    """
    messages: dict[str, str] = {}
    for mode in MUTATING_CYPHER_MODES:
        kg = _new_kg(mode)
        _build_seed(kg)
        kg.lock_schema()
        try:
            kg.cypher("CREATE (p:Person {name: 'x', bogus_prop: 1})")
            pytest.fail(f"{mode}: CREATE with unknown property should have raised")
        except Exception as e:
            messages[mode] = str(e)

    ref = messages["memory"]
    for mode in MUTATING_CYPHER_MODES[1:]:
        assert messages[mode] == ref, (
            f"schema-lock message diverged in {mode}:\n  memory: {ref}\n  {mode}: {messages[mode]}"
        )


# ─── 4. Tombstone visibility + traversal ────────────────────────────────────


def test_tombstone_visibility_detach_delete(tmp_path):
    """DETACH DELETE bob → bob invisible, path through bob returns 0 rows."""

    def mutate(kg):
        df = pd.DataFrame(
            [
                {"src": 1, "tgt": 2},  # alice → bob
                {"src": 2, "tgt": 3},  # bob → carol
            ]
        )
        kg.add_connections(df, "KNOWS", "Person", "src", "Person", "tgt")
        kg.cypher("MATCH (n:Person) WHERE n.id = 2 DETACH DELETE n")

    kgs = _per_mode(tmp_path, mutate)

    queries = [
        ("bob_visible", "MATCH (n:Person) WHERE n.id = 2 RETURN n.id AS id"),
        (
            "path_through_bob",
            "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) "
            "WHERE a.id = 1 AND c.id = 3 RETURN a.id AS a, b.id AS b, c.id AS c",
        ),
        ("remaining_persons", "MATCH (n:Person) RETURN n.id AS id"),
    ]
    for name, q in queries:
        ref = _snapshot(kgs["memory"], q)
        for mode in ("mapped", "disk"):
            got = _snapshot(kgs[mode], q)
            assert got == ref, f"tombstone[{name}] diverged in {mode}:\n  memory: {ref}\n  {mode}: {got}"

    # Concrete expectations (memory is the reference)
    assert _snapshot(kgs["memory"], queries[0][1]) == []
    assert _snapshot(kgs["memory"], queries[1][1]) == []
    assert _snapshot(kgs["memory"], queries[2][1]) == [{"id": 1}, {"id": 3}]


# ─── 5. MERGE idempotency ───────────────────────────────────────────────────


def test_merge_idempotency_node(tmp_path):
    """Running the same MERGE twice: counts identical across the modes
    that support MERGE.

    Disk-mode MERGE is locked out in 0.9.26+ by design — guarded by
    `test_disk_merge_lockout_message`.
    """

    def mutate(kg):
        kg.cypher("MERGE (p:Person {id: 1})")
        kg.cypher("MERGE (p:Person {id: 1})")

    kgs = _per_mode(tmp_path, mutate, modes=MUTATING_CYPHER_MODES)
    q = "MATCH (n:Person) RETURN count(n) AS c"
    ref = _snapshot(kgs["memory"], q)
    for mode in MUTATING_CYPHER_MODES[1:]:
        got = _snapshot(kgs[mode], q)
        assert got == ref, f"MERGE node count diverged in {mode}: {got} vs memory {ref}"
    # Sanity: we started with 3 and MERGEd an existing id, so count should stay 3
    assert ref == [{"c": 3}]


def test_merge_idempotency_edge(tmp_path):
    """Same MERGE pattern for an edge, run twice, no duplicates.

    Disk-mode MERGE is locked out in 0.9.26+ — see
    `test_disk_merge_lockout_message`.
    """

    def mutate(kg):
        for _ in range(2):
            kg.cypher("MATCH (a:Person), (b:Person) WHERE a.id = 1 AND b.id = 2 MERGE (a)-[:KNOWS]->(b)")

    kgs = _per_mode(tmp_path, mutate, modes=MUTATING_CYPHER_MODES)
    q = "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.id = 1 AND b.id = 2 RETURN a.id AS a, b.id AS b"
    ref = _snapshot(kgs["memory"], q)
    for mode in MUTATING_CYPHER_MODES[1:]:
        got = _snapshot(kgs[mode], q)
        assert got == ref, f"MERGE edge diverged in {mode}: {got} vs memory {ref}"
    assert ref == [{"a": 1, "b": 2}]


# ─── 6. Collect-then-delete snapshot ────────────────────────────────────────


def test_collect_then_delete_snapshot(tmp_path):
    """Python-level snapshot of a MATCH result survives a subsequent DELETE,
    and the re-query agrees across backends.

    KGLite is single-threaded from Python (GIL), so this is the
    observable equivalent of the 'concurrent mutation + read' crunch
    point from the Phase 2 plan.
    """
    snapshots: dict[str, tuple] = {}

    for mode in STORAGE_MODES:
        path = str(tmp_path / f"kg_{mode}") if mode == "disk" else None
        kg = _new_kg(mode, path=path)
        _build_seed(kg)

        before = _snapshot(kg, "MATCH (n:Person) RETURN n.id AS id")
        assert before == [{"id": 1}, {"id": 2}, {"id": 3}]

        kg.cypher("MATCH (n:Person) WHERE n.id = 2 DETACH DELETE n")

        # Pre-delete snapshot (Python list) is unchanged.
        assert before == [{"id": 1}, {"id": 2}, {"id": 3}]

        after = _snapshot(kg, "MATCH (n:Person) RETURN n.id AS id")
        snapshots[mode] = (tuple(tuple(sorted(r.items())) for r in after),)

    ref = snapshots["memory"]
    for mode in ("mapped", "disk"):
        assert snapshots[mode] == ref, f"post-delete snapshot diverged in {mode}: {snapshots[mode]} vs memory {ref}"


# ─── Previously-known disk divergences (Phase 5 reconciled) ─────────────────
#
# These three tests pinned Phase-2-era disk-mode bugs: `add_nodes` conflict
# handling (update/replace) silently dropped property mutations, and
# `MERGE ... -[:KNOWS]->` failed to make the new edge visible to a
# subsequent MATCH. Phase 5 fixed both:
#   - Update/replace: `batch_operations.rs::flush_chunk` now mutates
#     `graph.column_stores` directly on disk graphs (the previous path
#     materialised NodeData into a per-query arena that `clear_arenas`
#     dropped before the next read).
#   - MERGE visibility: `DiskGraph::new_at_path` defaults `defer_csr=false`
#     so single-edge inserts route to overflow immediately; bulk loaders
#     (`add_connections` / ntriples) explicitly enable deferral.


def test_disk_conflict_update_applies(tmp_path):
    kg = _new_kg("disk", path=str(tmp_path / "kg_disk_upd"))
    _build_seed(kg)
    df = pd.DataFrame([{"pid": 1, "name": "Alice Updated", "email": "alice@new.com"}])
    kg.add_nodes(df, "Person", "pid", "name", conflict_handling="update")

    rows = _snapshot(kg, "MATCH (n:Person) WHERE n.id = 1 RETURN n.email AS email")
    assert rows == [{"email": "alice@new.com"}]


def test_disk_conflict_replace_clears_omitted_properties(tmp_path):
    kg = _new_kg("disk", path=str(tmp_path / "kg_disk_rep"))
    _build_seed(kg)
    df = pd.DataFrame([{"pid": 1, "name": "Alice Updated", "email": "alice@new.com"}])
    kg.add_nodes(df, "Person", "pid", "name", conflict_handling="replace")

    rows = _snapshot(kg, "MATCH (n:Person) WHERE n.id = 1 RETURN n.age AS age, n.email AS email")
    assert rows == [{"age": None, "email": "alice@new.com"}]


def test_disk_create_lockout_message(tmp_path):
    """CREATE on a disk-backed graph is intentionally rejected in
    0.9.26+ — bulk loaders only. The error message must point users
    at `add_nodes` and the build-then-`to_disk` escape hatch, so the
    diagnostic is actionable.

    Pinning the message guards against silent rewording — if the
    rejection text changes, downstream docs / agent guidance need to
    move with it.
    """
    kg = _new_kg("disk", path=str(tmp_path / "kg_disk_create_lockout"))
    _build_seed(kg)
    with pytest.raises(Exception) as ei:
        kg.cypher("CREATE (p:Person {pid: 99, name: 'New'})")
    msg = str(ei.value)
    assert "CREATE" in msg and "disk" in msg
    assert "add_nodes" in msg, msg
    assert "to_disk" in msg, msg


def test_disk_merge_lockout_message(tmp_path):
    """MERGE on a disk-backed graph is intentionally rejected in
    0.9.26+ (MERGE shares the CREATE write path). The diagnostic
    must explain the lockout and the workaround."""
    kg = _new_kg("disk", path=str(tmp_path / "kg_disk_merge_lockout"))
    _build_seed(kg)
    with pytest.raises(Exception) as ei:
        kg.cypher("MERGE (p:Person {pid: 1})")
    msg = str(ei.value)
    assert "MERGE" in msg and "disk" in msg
    assert "add_nodes" in msg, msg
    assert "to_disk" in msg, msg
