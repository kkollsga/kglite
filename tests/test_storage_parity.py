"""Cross-storage-mode parity oracle.

Builds the same synthetic graph in memory, mapped, and disk modes, then
asserts a battery of queries return identical results. This is the
safety net for the 0.8.0 storage-architecture refactor: any regression
that breaks mapped or disk mode silently (wrong count, missing rows,
diverging schema output) fails here.

Run: pytest -m parity tests/test_storage_parity.py

This is the abbreviated Phase 0 oracle — 10 queries + save/load
round-trip. Phase 1+ expands per-area as new trait methods are added.
"""

from __future__ import annotations

from pathlib import Path
import random
import tempfile

import pandas as pd
import pytest

from kglite import KnowledgeGraph

pytestmark = pytest.mark.parity

STORAGE_MODES = ("memory", "mapped", "disk")
N_NODES = 2_000  # Small enough to run fast, big enough to trigger column-store paths


# ─── Fixture builder ────────────────────────────────────────────────────────


def _build_graph(mode: str, path: str | None = None) -> KnowledgeGraph:
    """Build an identical heterogeneous graph in the requested storage mode."""
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
            "rank": [i % 1000 for i in range(n)],
            "description": [f"desc {i} cluster {i % 50}" for i in range(n)],
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

    # Deterministic-pseudorandom edges (avoids RNG divergence across modes)
    edge_count = n * 2
    df_edges = pd.DataFrame(
        {
            "src": [(i * 2654435761) % n for i in range(edge_count)],
            "dst": [((i + 1) * 40503) % n for i in range(edge_count)],
        }
    )
    kg.add_connections(df_edges, "RELATED", "Entity", "src", "Entity", "dst")

    # Entity → Topic edges
    df_about = pd.DataFrame({"eid": list(range(n)), "tid": [i % 100 for i in range(n)]})
    kg.add_connections(df_about, "ABOUT", "Entity", "eid", "Topic", "tid")

    return kg


@pytest.fixture(scope="module")
def graphs():
    """Build one graph per storage mode. Reused across all parity tests."""
    with tempfile.TemporaryDirectory() as tmp:
        built = {
            "memory": _build_graph("memory"),
            "mapped": _build_graph("mapped"),
            "disk": _build_graph("disk", path=str(Path(tmp) / "kg_disk")),
        }
        yield built


# ─── Query oracle ───────────────────────────────────────────────────────────


def _rows(result) -> list[dict]:
    """Normalise a cypher result to sorted list-of-dicts (stable comparison)."""
    try:
        rows = [dict(r) for r in result]
    except TypeError:
        rows = list(result)
    # Stable sort by full row repr — works even when keys differ per-row.
    return sorted(rows, key=lambda r: repr(sorted(r.items())))


ORACLE_QUERIES = [
    (
        "filter_eq_string",
        "MATCH (n:Entity) WHERE n.category = 'cat_3' RETURN count(n) AS c",
    ),
    (
        "filter_range_numeric",
        "MATCH (n:Entity) WHERE n.score >= 25.0 AND n.score < 75.0 RETURN count(n) AS c",
    ),
    (
        "filter_in_list",
        "MATCH (n:Entity) WHERE n.category IN ['cat_1', 'cat_3', 'cat_5'] RETURN count(n) AS c",
    ),
    (
        "filter_contains",
        "MATCH (n:Entity) WHERE n.description CONTAINS 'cluster 7' RETURN count(n) AS c",
    ),
    (
        "aggregation_group_by",
        "MATCH (n:Entity) RETURN n.category AS cat, count(n) AS cnt ORDER BY cat",
    ),
    (
        "two_hop_count",
        "MATCH (a:Entity)-[:RELATED]->(b:Entity)-[:ABOUT]->(t:Topic) "
        "WHERE t.domain = 'domain_2' RETURN count(DISTINCT a) AS c",
    ),
    (
        "exact_path_relationship_ids",
        "MATCH p=(a:Entity {eid: 0})-[:RELATED*1..2]->(b:Entity) "
        "RETURN [r IN relationships(p) | id(r)] AS ids ORDER BY ids",
    ),
    (
        "order_by_limit",
        "MATCH (n:Entity) RETURN n.eid AS id, n.score AS s ORDER BY s DESC LIMIT 5",
    ),
    (
        "optional_match",
        "MATCH (n:Entity) OPTIONAL MATCH (n)-[:ABOUT]->(t:Topic) "
        "WITH n, count(t) AS cnt RETURN cnt, count(n) AS entities ORDER BY cnt",
    ),
    (
        "distinct_on_target",
        "MATCH (e:Entity)-[:ABOUT]->(t:Topic) RETURN count(DISTINCT t) AS topics",
    ),
    (
        "property_exists",
        "MATCH (n:Entity) WHERE n.rank IS NOT NULL RETURN count(n) AS c",
    ),
]


@pytest.mark.parametrize("name,query", ORACLE_QUERIES)
def test_cypher_parity(graphs, name, query):
    """Each query must return identical rows across memory, mapped, disk."""
    results = {mode: _rows(graphs[mode].cypher(query)) for mode in STORAGE_MODES}
    ref = results["memory"]
    for mode in ("mapped", "disk"):
        assert results[mode] == ref, f"{name}: {mode} diverged from memory\nmemory: {ref}\n{mode}:   {results[mode]}"


def test_find_by_title_parity(graphs):
    """`find()` must return the same node set across modes for the same query."""
    targets = ["Entity_0", "Entity_500", "Entity_1999"]
    for name in targets:
        results = {mode: graphs[mode].find(name, node_type="Entity") for mode in STORAGE_MODES}
        ref_len = len(results["memory"])
        assert ref_len == 1, f"memory-mode find('{name}') returned {ref_len} hits"
        for mode in ("mapped", "disk"):
            assert len(results[mode]) == ref_len, f"find('{name}'): {mode} returned {len(results[mode])} vs {ref_len}"


def test_schema_parity(graphs):
    """schema() must report the same node types + counts across modes."""
    schemas = {mode: graphs[mode].schema() for mode in STORAGE_MODES}
    ref = schemas["memory"]
    for mode in ("mapped", "disk"):
        got = schemas[mode]
        # Node type names + counts must match; property details can have
        # benign ordering differences handled by dict equality below.
        ref_nt = {k: v["count"] for k, v in ref["node_types"].items()}
        got_nt = {k: v["count"] for k, v in got["node_types"].items()}
        assert got_nt == ref_nt, f"schema node_types differ in {mode}: {got_nt} vs {ref_nt}"


def test_describe_shape_parity(graphs):
    """describe() XML must have the same <graph> nodes/edges counts across modes."""
    import re

    outs = {mode: graphs[mode].describe() for mode in STORAGE_MODES}

    def extract_counts(xml: str) -> tuple[str, str] | None:
        nodes = re.search(r'\bnodes="(\d+)"', xml)
        edges = re.search(r'\bedges="(\d+)"', xml)
        if not (nodes and edges):
            return None
        return (nodes.group(1), edges.group(1))

    ref = outs["memory"]
    ref_header = extract_counts(ref)
    assert ref_header, "memory-mode describe() missing nodes/edges attributes"
    for mode in ("mapped", "disk"):
        got = extract_counts(outs[mode])
        assert got, f"{mode} describe() missing nodes/edges attributes"
        assert got == ref_header, f"{mode} describe() header differs: {got} vs {ref_header}"


def test_db_labels_parity(graphs):
    """CALL db.labels() must report the same node-type set across modes (A.3)."""
    rows = {mode: _rows(graphs[mode].cypher("CALL db.labels() YIELD label RETURN label")) for mode in STORAGE_MODES}
    ref = rows["memory"]
    assert ref == [{"label": "Entity"}, {"label": "Topic"}], f"unexpected memory baseline: {ref}"
    for mode in ("mapped", "disk"):
        assert rows[mode] == ref, f"db.labels() {mode} diverged: {rows[mode]} vs {ref}"


def test_db_relationship_types_parity(graphs):
    """CALL db.relationshipTypes() must report the same connection-type set across modes (A.3)."""
    rows = {
        mode: _rows(graphs[mode].cypher("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"))
        for mode in STORAGE_MODES
    }
    ref = rows["memory"]
    assert ref == [{"relationshipType": "ABOUT"}, {"relationshipType": "RELATED"}], f"unexpected memory baseline: {ref}"
    for mode in ("mapped", "disk"):
        assert rows[mode] == ref, f"db.relationshipTypes() {mode} diverged: {rows[mode]} vs {ref}"


def test_db_indexes_parity(graphs):
    """CALL db.indexes() must report the same indexes across modes (A.3).

    The shared fixture does not create indexes, so all three modes must
    return zero rows. This pins the "no indexes → empty result" parity
    contract and guards against accidental backend-specific implicit
    indexes (which would diverge between Memory/Mapped/Disk).
    """
    rows = {mode: _rows(graphs[mode].cypher("CALL db.indexes() YIELD name RETURN name")) for mode in STORAGE_MODES}
    for mode in STORAGE_MODES:
        assert rows[mode] == [], f"db.indexes() {mode}: expected 0 rows, got {rows[mode]}"


def test_save_load_round_trip(graphs, tmp_path):
    """Save memory, load back, assert identical query result.

    Round-trip between modes requires load support for directory-mode (disk),
    which isn't in scope here. This test covers the heap .kgl path only —
    sufficient to catch format drift during the refactor.
    """
    import kglite

    save_path = tmp_path / "rt.kgl"
    graphs["memory"].save(str(save_path))
    reloaded = kglite.load(str(save_path))

    query = "MATCH (n:Entity) WHERE n.category = 'cat_3' RETURN count(n) AS c"
    original = _rows(graphs["memory"].cypher(query))
    after = _rows(reloaded.cypher(query))
    assert original == after, f"save/load round-trip diverged: {original} vs {after}"


def test_strongly_connected_components_parity(tmp_path):
    """Directed SCC semantics must not degrade to weak components on disk."""
    nodes = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "title": ["A", "B", "C", "D", "E", "F"]})
    edges = pd.DataFrame(
        {
            "src": [1, 2, 3, 4, 4, 6],
            "dst": [2, 3, 2, 5, 6, 5],
        }
    )
    results: dict[str, list[list[str]]] = {}

    for mode in STORAGE_MODES:
        if mode == "memory":
            graph = KnowledgeGraph()
        elif mode == "mapped":
            graph = KnowledgeGraph(storage="mapped")
        else:
            graph = KnowledgeGraph(storage="disk", path=str(tmp_path / "scc-disk"))
        graph.add_nodes(nodes, "Node", "id", "title")
        graph.add_connections(edges, "LINK", "Node", "src", "Node", "dst")
        components = graph.connected_components(weak=False, titles_only=True)
        results[mode] = sorted((sorted(component) for component in components), key=lambda c: (len(c), c))

    # D fans out to E and F, while F also points to E. An iterative first
    # pass that marks sibling nodes too early incorrectly merges E and F on
    # the transpose pass, so this also guards the DFS finishing-order detail.
    expected = [["A"], ["D"], ["E"], ["F"], ["B", "C"]]
    assert results["memory"] == expected
    for mode in ("mapped", "disk"):
        assert results[mode] == expected, f"SCC {mode} diverged: {results[mode]}"
