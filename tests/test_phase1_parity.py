"""Phase 1 crunch-point parity tests.

Guards the `GraphRead` trait expansion (Phase 1 of the 0.8.0 storage
refactor). These tests assert that iteration, neighbour enumeration,
and connection metadata are byte-identical (as sorted sets) across
memory / mapped / disk modes — so that any later migration to the
trait cannot silently reorder results.

Companions to `tests/test_storage_parity.py`, which covers the broader
Cypher oracle. Run: `pytest -m parity tests/test_phase1_parity.py`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import random
import tempfile

import pandas as pd
import pytest

from kglite import KnowledgeGraph

pytestmark = pytest.mark.parity

STORAGE_MODES = ("memory", "mapped", "disk")
N_NODES = 2_000


def _build_graph(mode: str, path: str | None = None) -> KnowledgeGraph:
    """Same heterogeneous fixture as test_storage_parity.py."""
    if mode == "memory":
        kg = KnowledgeGraph()
    elif mode == "mapped":
        kg = KnowledgeGraph(storage="mapped")
    elif mode == "disk":
        if path is None:
            raise ValueError("mode='disk' requires path")
        kg = KnowledgeGraph(storage="disk", path=path)
    else:
        raise ValueError(f"unknown mode: {mode}")

    rng = random.Random(42)
    n = N_NODES
    df_entities = pd.DataFrame(
        {
            "eid": list(range(n)),
            "title": [f"Entity_{i}" for i in range(n)],
            "category": [f"cat_{i % 20}" for i in range(n)],
            "score": [rng.uniform(0, 100) for _ in range(n)],
        }
    )
    kg.add_nodes(df_entities, "Entity", "eid", "title")

    df_topics = pd.DataFrame(
        {
            "tid": list(range(100)),
            "name": [f"Topic_{i}" for i in range(100)],
            "domain": [f"domain_{i % 5}" for i in range(100)],
        }
    )
    kg.add_nodes(df_topics, "Topic", "tid", "name")

    edge_count = n * 2
    df_edges = pd.DataFrame(
        {
            "src": [(i * 2654435761) % n for i in range(edge_count)],
            "dst": [((i + 1) * 40503) % n for i in range(edge_count)],
        }
    )
    kg.add_connections(df_edges, "RELATED", "Entity", "src", "Entity", "dst")

    df_about = pd.DataFrame({"eid": list(range(n)), "tid": [i % 100 for i in range(n)]})
    kg.add_connections(df_about, "ABOUT", "Entity", "eid", "Topic", "tid")

    return kg


@pytest.fixture(scope="module")
def graphs():
    with tempfile.TemporaryDirectory() as tmp:
        built = {
            "memory": _build_graph("memory"),
            "mapped": _build_graph("mapped"),
            "disk": _build_graph("disk", path=str(Path(tmp) / "kg_disk")),
        }
        yield built


def _rows(result) -> list[dict]:
    try:
        rows = [dict(r) for r in result]
    except TypeError:
        rows = list(result)
    return rows


def _digest(records: list[tuple]) -> str:
    """SHA-256 of the sorted tuple list."""
    h = hashlib.sha256()
    for t in sorted(records):
        h.update(repr(t).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# ─── Edge iteration stability ───────────────────────────────────────────────


def test_edge_iteration_stability_related(graphs):
    """Every RELATED edge, sorted by (src_eid, dst_eid) — digest must match."""
    q = "MATCH (a:Entity)-[:RELATED]->(b:Entity) RETURN a.eid AS s, b.eid AS t"
    per_mode = {}
    for mode in STORAGE_MODES:
        rows = _rows(graphs[mode].cypher(q))
        tuples = [(int(r["s"]), int(r["t"])) for r in rows]
        per_mode[mode] = (len(tuples), _digest(tuples))
    ref = per_mode["memory"]
    for mode in ("mapped", "disk"):
        assert per_mode[mode] == ref, f"RELATED edge digest mismatch in {mode}: {per_mode[mode]} vs memory {ref}"


def test_edge_iteration_stability_about(graphs):
    """ABOUT edges (Entity → Topic) — digest identical across modes."""
    q = "MATCH (e:Entity)-[:ABOUT]->(t:Topic) RETURN e.eid AS s, t.tid AS t"
    per_mode = {}
    for mode in STORAGE_MODES:
        rows = _rows(graphs[mode].cypher(q))
        tuples = [(int(r["s"]), int(r["t"])) for r in rows]
        per_mode[mode] = (len(tuples), _digest(tuples))
    ref = per_mode["memory"]
    for mode in ("mapped", "disk"):
        assert per_mode[mode] == ref, f"ABOUT edge digest mismatch in {mode}: {per_mode[mode]} vs {ref}"


# ─── Per-node neighbour enumeration ─────────────────────────────────────────


def test_neighbors_directed_per_node(graphs):
    """Outgoing neighbours for a sample of nodes, sorted — digest-equal."""
    probe_ids = [0, 1, 7, 100, 500, 999, 1500, 1999]
    q_template = "MATCH (a:Entity {{eid: {eid}}})-[:RELATED]->(b:Entity) RETURN b.eid AS peer"
    per_mode = {}
    for mode in STORAGE_MODES:
        collected = []
        for eid in probe_ids:
            rows = _rows(graphs[mode].cypher(q_template.format(eid=eid)))
            peers = sorted(int(r["peer"]) for r in rows)
            collected.append((eid, tuple(peers)))
        per_mode[mode] = _digest(collected)
    ref = per_mode["memory"]
    for mode in ("mapped", "disk"):
        assert per_mode[mode] == ref, f"neighbor-set digest mismatch in {mode}: {per_mode[mode]} vs memory {ref}"


def test_neighbors_undirected_per_node(graphs):
    """Undirected neighbours via MATCH (a)-[:RELATED]-(b) — digest-equal."""
    probe_ids = [0, 1, 7, 100, 500, 999, 1500, 1999]
    q_template = "MATCH (a:Entity {{eid: {eid}}})-[:RELATED]-(b:Entity) RETURN b.eid AS peer"
    per_mode = {}
    for mode in STORAGE_MODES:
        collected = []
        for eid in probe_ids:
            rows = _rows(graphs[mode].cypher(q_template.format(eid=eid)))
            peers = sorted(int(r["peer"]) for r in rows)
            collected.append((eid, tuple(peers)))
        per_mode[mode] = _digest(collected)
    ref = per_mode["memory"]
    for mode in ("mapped", "disk"):
        assert per_mode[mode] == ref, f"undirected neighbor digest mismatch in {mode}: {per_mode[mode]} vs memory {ref}"


# ─── Type index determinism (`nodes_of_type` proxy) ─────────────────────────


def test_nodes_of_type_determinism(graphs):
    """Full Entity list, sorted — identical digest across modes."""
    q = "MATCH (n:Entity) RETURN n.eid AS eid"
    per_mode = {}
    for mode in STORAGE_MODES:
        rows = _rows(graphs[mode].cypher(q))
        ids = tuple(sorted(int(r["eid"]) for r in rows))
        per_mode[mode] = (len(ids), _digest([ids]))
    ref = per_mode["memory"]
    for mode in ("mapped", "disk"):
        assert per_mode[mode] == ref, f"Entity enumeration mismatch in {mode}: {per_mode[mode]} vs memory {ref}"


def test_nodes_of_type_determinism_topic(graphs):
    """Same, for Topic — catches per-type index divergence."""
    q = "MATCH (n:Topic) RETURN n.tid AS tid"
    per_mode = {}
    for mode in STORAGE_MODES:
        rows = _rows(graphs[mode].cypher(q))
        ids = tuple(sorted(int(r["tid"]) for r in rows))
        per_mode[mode] = (len(ids), _digest([ids]))
    ref = per_mode["memory"]
    for mode in ("mapped", "disk"):
        assert per_mode[mode] == ref, f"Topic enumeration mismatch in {mode}: {per_mode[mode]} vs memory {ref}"


# ─── Connection metadata parity ─────────────────────────────────────────────


def test_connection_metadata_parity(graphs):
    """`schema()` connection types must have identical source/target/property sets."""
    schemas = {mode: graphs[mode].schema() for mode in STORAGE_MODES}
    ref = schemas["memory"]
    ref_conn = ref.get("connection_types") or ref.get("connections") or {}
    assert ref_conn, "memory-mode schema() missing connection_types"
    for mode in ("mapped", "disk"):
        got_conn = schemas[mode].get("connection_types") or schemas[mode].get("connections") or {}
        assert set(got_conn.keys()) == set(ref_conn.keys()), (
            f"{mode}: connection-type names differ: {set(got_conn.keys())} vs {set(ref_conn.keys())}"
        )
        for name in ref_conn:
            r_info = ref_conn[name]
            g_info = got_conn[name]
            for field in ("source_types", "target_types"):
                r_val = r_info.get(field)
                g_val = g_info.get(field)
                if r_val is None and g_val is None:
                    continue
                assert set(r_val or []) == set(g_val or []), (
                    f"{mode}: {name}.{field} differs: {set(g_val or [])} vs {set(r_val or [])}"
                )


# ─── Edge-count parity (catches count_edges_* trait divergence) ─────────────


def test_edge_count_per_type_parity(graphs):
    """Typed and untyped edge counts are identical across storage modes."""
    q_related = "MATCH ()-[r:RELATED]->() RETURN count(r) AS c"
    q_about = "MATCH ()-[r:ABOUT]->() RETURN count(r) AS c"
    q_all = "MATCH ()-[r]->() RETURN count(r) AS c"
    per_mode = {}
    for mode in STORAGE_MODES:
        r = _rows(graphs[mode].cypher(q_related))[0]["c"]
        a = _rows(graphs[mode].cypher(q_about))[0]["c"]
        total = _rows(graphs[mode].cypher(q_all))[0]["c"]
        per_mode[mode] = (int(r), int(a), int(total))
    ref = per_mode["memory"]
    for mode in ("mapped", "disk"):
        assert per_mode[mode] == ref, f"edge counts diverged in {mode}: {per_mode[mode]} vs memory {ref}"


def test_per_node_outdegree_parity(graphs):
    """Per-node outdegree for RELATED — catches neighbor-iteration divergence."""
    q = "MATCH (n:Entity) OPTIONAL MATCH (n)-[r:RELATED]->() RETURN n.eid AS eid, count(r) AS deg"
    per_mode = {}
    for mode in STORAGE_MODES:
        rows = _rows(graphs[mode].cypher(q))
        pairs = [(int(r["eid"]), int(r["deg"])) for r in rows]
        per_mode[mode] = _digest(pairs)
    ref = per_mode["memory"]
    for mode in ("mapped", "disk"):
        assert per_mode[mode] == ref, f"outdegree digest diverged in {mode}: {per_mode[mode]} vs memory {ref}"
