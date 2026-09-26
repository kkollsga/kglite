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

import collections
import math
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
        "path_where_fixed",
        "MATCH p=(a:Entity)-[:RELATED]->(b:Entity) WHERE a.eid < 20 AND "
        "length(p) = 1 AND last(nodes(p)).rank >= 0 RETURN count(*) AS c",
    ),
    (
        "path_where_var_length",
        "MATCH p=(a:Entity {eid: 0})-[:RELATED*1..2]->(b:Entity) WHERE lengt"
        "h(p) = 2 RETURN [r IN relationships(p) | id(r)] AS ids ORDER BY ids",
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


def test_degrees_parity(graphs):
    """Bulk fluent degree materialization must agree across all backends."""
    results = {mode: graphs[mode].select("Entity").degrees() for mode in STORAGE_MODES}
    ref = results["memory"]
    assert len(ref) == N_NODES
    for mode in ("mapped", "disk"):
        assert results[mode] == ref, f"degrees: {mode} diverged from memory"


def test_bulk_selected_update_parity(tmp_path):
    """Selected-node bulk updates preserve counts and values on every backend."""
    results = {}
    for mode in STORAGE_MODES:
        path = str(tmp_path / f"update_{mode}") if mode == "disk" else None
        graph = _build_graph(mode, path=path)
        report = graph.select("Entity").where({"rank": 3}).update({"rank": 1003})
        updated = report["graph"]
        results[mode] = {
            "nodes_updated": report["nodes_updated"],
            "old": _rows(updated.cypher("MATCH (n:Entity) WHERE n.rank = 3 RETURN n.eid AS id ORDER BY id")),
            "new": _rows(updated.cypher("MATCH (n:Entity) WHERE n.rank = 1003 RETURN n.eid AS id ORDER BY id")),
        }

    expected = {
        "nodes_updated": 2,
        "old": [],
        "new": [{"id": 1003}, {"id": 3}],
    }
    assert results["memory"] == expected
    for mode in ("mapped", "disk"):
        assert results[mode] == expected, f"bulk update: {mode} diverged: {results[mode]}"


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


def test_property_type_constraint_parity(tmp_path):
    """A declared property type is backend-agnostic: it must install, refuse the
    same write, and report the same `SHOW CONSTRAINTS` row in every mode.

    Constraints live above the storage layer, but each mode reaches the write
    path differently (columnar master, mmap columns, CSR + overlay), so "the
    predicate is shared" is a claim that has to be tested rather than assumed.
    """
    import kglite

    installed: dict[str, list] = {}
    refusals: dict[str, str] = {}
    for mode in STORAGE_MODES:
        path = str(tmp_path / f"ptc_{mode}") if mode == "disk" else None
        kg = _build_graph(mode, path)

        kg.cypher("CREATE CONSTRAINT rank_typed FOR (e:Entity) REQUIRE e.rank IS :: INTEGER")
        installed[mode] = _rows(
            kg.cypher("CALL db.constraints() YIELD name, type, propertyType RETURN name, type, propertyType")
        )

        with pytest.raises(kglite.ConstraintViolationError) as exc:
            kg.cypher("MATCH (e:Entity) WHERE e.eid = 1 SET e.rank = 'high'")
        refusals[mode] = str(exc.value)

        # Null still passes, in every mode.
        kg.cypher("MATCH (e:Entity) WHERE e.eid = 1 SET e.rank = null")

    reference = installed["memory"]
    assert reference and reference[0]["type"] == "NODE_PROPERTY_TYPE", reference
    assert reference[0]["propertyType"] == "INTEGER", reference
    for mode in STORAGE_MODES:
        assert installed[mode] == reference, f"{mode} constraint listing diverged: {installed[mode]}"
        assert "INTEGER" in refusals[mode], f"{mode}: {refusals[mode]}"
        assert "STRING" in refusals[mode], f"{mode}: {refusals[mode]}"


def test_relationship_constraint_parity(tmp_path):
    """The relationship half of the same claim. Its existing-data scan reads
    edges through a backend-agnostic accessor whose disk arm is a different code
    path from its petgraph arms, so "it installs and enforces" has to be
    asserted per mode rather than inferred from the in-memory one.
    """
    import kglite

    installed: dict[str, list] = {}
    refusals: dict[str, str] = {}
    for mode in STORAGE_MODES:
        path = str(tmp_path / f"rel_{mode}") if mode == "disk" else None
        kg = _build_graph(mode, path)
        # `RELATED` edges carry no properties, so a presence constraint on one
        # is refused by the existing data — that refusal is itself parity.
        with pytest.raises(kglite.ConstraintCreationError):
            kg.cypher("CREATE CONSTRAINT FOR ()-[r:RELATED]-() REQUIRE r.weight IS NOT NULL")

        # A type constraint installs: absent values satisfy a declared type.
        kg.cypher("CREATE CONSTRAINT rel_weight FOR ()-[r:RELATED]-() REQUIRE r.weight IS :: INTEGER")
        installed[mode] = _rows(
            kg.cypher(
                "CALL db.constraints() YIELD name, type, entityType, propertyType "
                "RETURN name, type, entityType, propertyType"
            )
        )

        with pytest.raises(kglite.ConstraintViolationError) as exc:
            kg.cypher("MATCH ()-[r:RELATED]->() SET r.weight = 'heavy'")
        refusals[mode] = str(exc.value)

        # Null still passes a type declaration, in every mode.
        kg.cypher("MATCH ()-[r:RELATED]->() SET r.weight = null")

    reference = installed["memory"]
    assert reference and reference[0]["type"] == "RELATIONSHIP_PROPERTY_TYPE", reference
    assert reference[0]["entityType"] == "RELATIONSHIP", reference
    for mode in STORAGE_MODES:
        assert installed[mode] == reference, f"{mode} constraint listing diverged: {installed[mode]}"
        assert "INTEGER" in refusals[mode], f"{mode}: {refusals[mode]}"
        assert "relationship of type 'RELATED'" in refusals[mode], f"{mode}: {refusals[mode]}"


def test_cdc_mode_parity(tmp_path):
    """The change stream is backend-agnostic where it is served at all.

    Memory and mapped must publish the *same* events for the same writes — the
    capture seam sits above storage, but each mode reaches the write path
    differently, so "the same ops are buffered" is a claim to test. Disk is the
    documented refusal: its change boundary is the generation publish, not the
    per-commit capture this stream is derived from, so enabling would report a
    stream that silently missed writes.
    """
    import kglite

    published: dict[str, list] = {}
    for mode in ("memory", "mapped"):
        kg = _build_graph(mode)
        # Full enrichment on purpose: `before` is the half the two backends
        # capture *differently*. A mapped SET writes through the columnar
        # master store and notifies the capture seam afterwards, so its
        # before-image is read at a different site from memory's — and a site
        # that read it too late would report the new value here, matching
        # nothing.
        kg.cypher("CALL db.cdc.enable({enrichment: 'full'})")
        kg.cypher("CREATE (a:Widget {wid: 1, size: 1})-[:PAIRS {w: 2}]->(b:Widget {wid: 2})")
        kg.cypher("MATCH (n:Widget) WHERE n.wid = 1 SET n.size = 2")
        kg.cypher("MATCH (n:Widget) WHERE n.wid = 2 DETACH DELETE n")
        published[mode] = [
            (r["operation"], r["elementType"], r["nodeType"], r["nodeId"], r["relationshipType"], r["state"])
            for r in kg.cypher("CALL db.cdc.query()").to_dicts()
        ]

    assert published["memory"], "the stream must not be empty — that would pass vacuously"
    assert published["mapped"] == published["memory"], published
    assert any(row[5]["before"] is not None for row in published["memory"]), (
        "full enrichment must actually populate a before-image, or the parity is vacuous"
    )

    disk = _build_graph("disk", str(tmp_path / "cdc_disk"))
    with pytest.raises(kglite.KgError) as exc:
        disk.cypher("CALL db.cdc.enable()")
    assert "not supported for storage='disk'" in str(exc.value)


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


def test_relationship_alternation_parity(tmp_path):
    """`[:A|B]` from an untyped start node must find both branches in every mode.

    An untyped, unfiltered start node with a typed outgoing edge picks its
    start nodes from the connection-type inverted index — which `mapped` and
    `disk` have and `memory` does not. Looked up for the *first* branch alone,
    it dropped every start node whose only matching edge was on a later branch,
    so the two storage modes silently returned fewer rows than memory.

    The discriminator is that `WROTE`'s source set is a strict subset of
    `LIKES`'s: node 4 only writes and node 3 only likes, so neither branch's
    source list alone covers the pattern. The shared `ORACLE_QUERIES` fixture
    cannot express this — both of its edge types have every Entity as a source
    — which is why this builds its own graph. Absolute row counts are asserted
    alongside cross-mode agreement, since a bug all three modes shared would
    keep them in agreement.
    """
    nodes = pd.DataFrame({"id": [1, 2, 3, 4], "title": ["A", "B", "C", "D"]})
    docs = pd.DataFrame({"id": [10, 11], "title": ["Doc1", "Doc2"]})
    # LIKES sources: {1, 2, 3};  WROTE sources: {1, 2, 4}
    likes = pd.DataFrame({"src": [1, 2, 3], "dst": [2, 3, 1]})
    wrote = pd.DataFrame({"src": [1, 2, 4], "dst": [10, 11, 10]})

    for both_orders in (("LIKES|WROTE", "WROTE|LIKES"),):
        results: dict[str, dict[str, list[dict]]] = {}
        for mode in STORAGE_MODES:
            if mode == "memory":
                graph = KnowledgeGraph()
            elif mode == "mapped":
                graph = KnowledgeGraph(storage="mapped")
            else:
                graph = KnowledgeGraph(storage="disk", path=str(tmp_path / "alt-disk"))
            graph.add_nodes(nodes, "Person", "id", "title")
            graph.add_nodes(docs, "Doc", "id", "title")
            graph.add_connections(likes, "LIKES", "Person", "src", "Person", "dst")
            graph.add_connections(wrote, "WROTE", "Person", "src", "Doc", "dst")
            results[mode] = {
                rel: _rows(graph.cypher(f"MATCH (x)-[:{rel}]->(y) RETURN x.title AS a, y.title AS b"))
                for rel in both_orders
            }

        for mode in STORAGE_MODES:
            for rel in both_orders:
                rows = results[mode][rel]
                assert len(rows) == 6, f"{mode} / [:{rel}]: expected 6 rows, got {len(rows)}"
            assert results[mode][both_orders[0]] == results[mode][both_orders[1]], (
                f"{mode}: branch order changed the answer"
            )
        for mode in ("mapped", "disk"):
            assert results[mode] == results["memory"], f"{mode} diverged from memory on [:A|B]"


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


def test_index_freshness_parity(tmp_path):
    """An index must never make a mode answer differently from a scan.

    Disk is the only mode whose equality indexes are persistent mmap bundles,
    and nothing maintains one after it is built: a ``create_index``, and the
    ``title``/``nid`` globals every disk ``save()`` builds on its own, were read
    as authoritative forever. So after ``save()`` + ``load()`` every
    ``name``/``title`` lookup and ``search()`` missed the rows written since,
    and a user index missed every insert and ``SET`` after it — memory
    maintained incrementally and mapped invalidated, so the three modes
    disagreed on the same data (deep scan 2026-09-07, items 2+3).

    The op sequence walks every way a bundle can go out of date — bulk insert,
    Cypher ``CREATE``, ``SET`` of the indexed property, ``SET`` of the title,
    ``DELETE``, and a save/load in the middle — and asserts absolute expected
    values as well as cross-mode agreement, since a defect all three modes
    shared would keep them agreeing.

    ``search()`` is deliberately not probed here: it is documented as
    disk-only (memory and mapped have no cross-type global index and return an
    empty list), so it has no parity to assert. Its own stale-bundle case is
    ``tests/test_disk_property_index.py``.
    """
    people = pd.DataFrame(
        {
            "uid": [1, 2, 3],
            "name": ["nm_a", "nm_b", "nm_c"],
            "cat": ["c0", "c1", "c0"],
        }
    )
    more = pd.DataFrame({"uid": [4], "name": ["nm_d"], "cat": ["c1"]})

    def probe(graph):
        return {
            "cat_c1": sorted(r["u"] for r in _rows(graph.cypher("MATCH (n:P) WHERE n.cat = 'c1' RETURN n.uid AS u"))),
            "cat_c0": sorted(r["u"] for r in _rows(graph.cypher("MATCH (n:P) WHERE n.cat = 'c0' RETURN n.uid AS u"))),
            "map_cat": sorted(r["u"] for r in _rows(graph.cypher("MATCH (n:P {cat: 'c1'}) RETURN n.uid AS u"))),
            "by_name": sorted(
                r["u"] for r in _rows(graph.cypher("MATCH (n:P) WHERE n.name = 'nm_e' RETURN n.uid AS u"))
            ),
            "map_name": sorted(r["u"] for r in _rows(graph.cypher("MATCH (n:P {name: 'nm_e'}) RETURN n.uid AS u"))),
            "by_title": sorted(
                r["u"] for r in _rows(graph.cypher("MATCH (n:P) WHERE n.title = 'nm_d' RETURN n.uid AS u"))
            ),
        }

    results: dict[str, dict] = {}
    for mode in STORAGE_MODES:
        if mode == "memory":
            graph = KnowledgeGraph()
        elif mode == "mapped":
            graph = KnowledgeGraph(storage="mapped")
        else:
            graph = KnowledgeGraph(storage="disk", path=str(tmp_path / "freshness-disk"))

        graph.add_nodes(people, "P", "uid", "name")
        graph.create_index("P", "cat")

        snapshot = str(tmp_path / f"freshness-{mode}.kgl")
        if mode == "disk":
            snapshot = str(tmp_path / "freshness-disk-gen")
        graph.save(snapshot)
        graph = __import__("kglite").load(snapshot)

        # Every write the bundles were not built over.
        graph.add_nodes(more, "P", "uid", "name")
        graph.cypher("CREATE (n:P {uid: 5, name: 'nm_e', cat: 'c1'})")
        graph.cypher("MATCH (n:P) WHERE n.uid = 1 SET n.cat = 'c1'")
        graph.cypher("MATCH (n:P) WHERE n.uid = 3 SET n.name = 'nm_e'")
        graph.cypher("MATCH (n:P) WHERE n.uid = 2 DETACH DELETE n")

        results[mode] = probe(graph)
        # `reindex()` is the documented repair verb and must not change an answer.
        graph.reindex()
        assert probe(graph) == results[mode], f"{mode}: reindex() changed an answer"

    expected = {
        "cat_c1": [1, 4, 5],
        "cat_c0": [3],
        "map_cat": [1, 4, 5],
        "by_name": [3, 5],
        "map_name": [3, 5],
        "by_title": [4],
    }
    assert results["memory"] == expected
    for mode in ("mapped", "disk"):
        assert results[mode] == expected, f"{mode} diverged: {results[mode]}"


def test_soft_alias_index_parity(tmp_path):
    """An index on a structurally-resolved name must not change any answer.

    ``name``/``type``/``node_type``/``label`` resolve structurally when the
    node stores no such property (``n.label`` on a node with no stored label
    answers with its node type), while every index — the in-memory map and the
    persistent disk bundle alike — is built from stored values only. The
    in-memory arm has refused to read such an index since the ontology
    follow-ups; the disk arm did not, so ``create_index('C', 'label')`` turned
    ``WHERE n.label = 'C'`` from one row into zero, and
    ``create_global_index('label')`` did the same to the untyped spelling
    (deep scan 2026-09-07, T2-6b).

    Absolute expected values as well as cross-mode agreement: the defect was
    disk-only, but a fix that changed the scan would keep the modes agreeing
    while breaking all three.
    """
    countries = pd.DataFrame(
        {
            "cid": [1, 2, 3],
            # Node 3 stores no label, so `n.label` resolves to the type string.
            "label": ["Norway", "Sweden", None],
        }
    )

    def probe(graph):
        def ids(query):
            return sorted(r["c"] for r in _rows(graph.cypher(query)))

        return {
            "eq_type_string": ids("MATCH (n:C) WHERE n.label = 'C' RETURN n.cid AS c"),
            "eq_stored": ids("MATCH (n:C) WHERE n.label = 'Norway' RETURN n.cid AS c"),
            "map_type_string": ids("MATCH (n:C {label: 'C'}) RETURN n.cid AS c"),
            "map_stored": ids("MATCH (n:C {label: 'Norway'}) RETURN n.cid AS c"),
            "untyped_type_string": ids("MATCH (n {label: 'C'}) RETURN n.cid AS c"),
            "untyped_stored": ids("MATCH (n {label: 'Norway'}) RETURN n.cid AS c"),
            "starts_type_string": ids("MATCH (n:C) WHERE n.label STARTS WITH 'C' RETURN n.cid AS c"),
        }

    expected = {
        "eq_type_string": [3],
        "eq_stored": [1],
        "map_type_string": [3],
        "map_stored": [1],
        "untyped_type_string": [3],
        "untyped_stored": [1],
        "starts_type_string": [3],
    }

    for mode in STORAGE_MODES:
        if mode == "memory":
            graph = KnowledgeGraph()
        elif mode == "mapped":
            graph = KnowledgeGraph(storage="mapped")
        else:
            graph = KnowledgeGraph(storage="disk", path=str(tmp_path / "soft-alias-disk"))
        graph.add_nodes(countries, "C", "cid")

        assert probe(graph) == expected, f"{mode}: wrong before any index"

        info = graph.create_index("C", "label")
        assert info["serves_lookups"] is False, f"{mode}: {info}"
        assert "resolved structurally" in (info["not_serving"] or ""), f"{mode}: {info}"
        assert probe(graph) == expected, f"{mode}: create_index changed an answer"

        graph.create_global_index("label")
        assert probe(graph) == expected, f"{mode}: create_global_index changed an answer"

        # A save arms the auto-built persistent globals; the answers stand.
        snapshot = str(tmp_path / f"soft-alias-{mode}")
        if mode != "disk":
            snapshot += ".kgl"
        graph.save(snapshot)
        reloaded = __import__("kglite").load(snapshot)
        assert probe(reloaded) == expected, f"{mode}: reload changed an answer"


# ─── Relationship embedding stores ─────────────────────────────────────────
#
# Relationship vectors live in a store keyed by physical edge slot, resolved
# through each backend's edge-property substrate, and disk saves remap slots.
# The battery below is the relationship counterpart of the node oracle above,
# with **absolute** expected values computed by an independent Python oracle
# over the fixture's deterministic edge list: a defect all three modes share
# passes a pure cross-mode comparison, so agreement alone is not the contract.


class _ParityEmbedder:
    """Deterministic stub: one vector per text, so `text_score` is reproducible."""

    dimension = 3
    model_id = "parity/stub"

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    @staticmethod
    def vector(text: str) -> list[float]:
        return [float(sum(text.encode()) % 17 + 1), 1.0, 0.5]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vector(text) for text in texts]


_REL_QUERY = [1.0, 1.0, 1.0]
_REL_TOP_K = 15


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def _relationship_oracle() -> list[tuple[int, int, list[float]]]:
    """(src, dst, vector) for every RELATED edge, in the fixture's insertion order.

    The vector rule keeps vectors near-distinct (7·11·13·17 = 17017 combinations
    over 4000 edges): with hundreds of identical vectors HNSW recall on the
    fixture collapsed to 12/15 and varied between runs, which says nothing
    about storage parity. True parallel duplicates (same src, dst) still tie.
    """
    n, edge_count = N_NODES, N_NODES * 2
    return [
        (s, d, [float(s % 7) + 0.01 * float(s % 13), float(d % 11) + 0.01 * float(d % 17), 1.0])
        for s, d in (((i * 2654435761) % n, ((i + 1) * 40503) % n) for i in range(edge_count))
    ]


def _relationship_battery(kg: KnowledgeGraph) -> dict:
    """Every read the oracle compares, on one graph, as plain data."""
    listed = _rows(
        kg.cypher(
            "CALL db.relationship_embeddings.list({type:'RELATED', text_column:'text'}) "
            "YIELD entity,count,dimension,metric,model,index_state,delta "
            "RETURN entity,count,dimension,metric,model,index_state,delta"
        )
    )
    exact = kg.cypher(
        "CALL db.relationship_embeddings.query({type:'RELATED', text_column:'text', vector:$q, "
        "top_k:$k, exact:true}) YIELD relationship, score, search_method "
        "RETURN relationship.text AS text, score, search_method",
        params={"q": _REL_QUERY, "k": _REL_TOP_K},
    ).to_list()
    approximate = kg.cypher(
        "CALL db.relationship_embeddings.query({type:'RELATED', text_column:'text', vector:$q, "
        "top_k:$k}) YIELD relationship, score, search_method "
        "RETURN relationship.text AS text, score, search_method",
        params={"q": _REL_QUERY, "k": _REL_TOP_K},
    ).to_list()
    per_row = kg.cypher(
        "MATCH (a:Entity)-[r:RELATED]->(b:Entity) WHERE a.eid < 40 "
        "RETURN a.eid AS s, b.eid AS t, r.text AS text, "
        "vector_score(r,'text_emb',$q) AS score, embedding_norm(r,'text_emb') AS norm, "
        "text_score(r,'text','probe') AS text_score ORDER BY s, t, score, text",
        params={"q": _REL_QUERY},
    ).to_list()
    ids = kg.cypher("MATCH ()-[r:RELATED]->() RETURN count(r) AS c, min(id(r)) AS lo, max(id(r)) AS hi").to_list()
    indexes = _rows(kg.cypher("SHOW INDEXES"))
    return {
        "listed": listed,
        "exact_scores": [round(row["score"], 4) for row in exact],
        "exact_members": sorted((round(row["score"], 4), row["text"]) for row in exact),
        "exact_raw_order": [row["text"] for row in exact],
        "exact_route": {row["search_method"] for row in exact},
        "approx_scores": [round(row["score"], 4) for row in approximate],
        "approx_route": {row["search_method"] for row in approximate},
        "per_row": [
            {
                **row,
                "score": round(row["score"], 4),
                "norm": round(row["norm"], 4),
                "text_score": round(row["text_score"], 4),
            }
            for row in per_row
        ],
        "ids": ids,
        "indexes": indexes,
    }


def _seed_relationship_store(kg: KnowledgeGraph) -> None:
    kg.set_embedder(_ParityEmbedder())
    kg.cypher(
        "MATCH (a:Entity)-[r:RELATED]->(b:Entity) SET r.text = 'edge ' + toString(a.eid) + ' to ' + toString(b.eid)"
    )
    kg.cypher(
        "MATCH (a:Entity)-[r:RELATED]->(b:Entity) "
        "WITH collect({relationship: r, vector: ["
        "toFloat(a.eid % 7) + 0.01 * toFloat(a.eid % 13), "
        "toFloat(b.eid % 11) + 0.01 * toFloat(b.eid % 17), 1.0]}) AS entries "
        "CALL db.relationship_embeddings.set({type:'RELATED', text_column:'text', entries: entries}) "
        "YIELD stored RETURN stored"
    )
    kg.cypher(
        "CALL db.relationship_embeddings.build_index({type:'RELATED', text_column:'text'}) YIELD indexed RETURN indexed"
    )


def test_relationship_embedding_parity(tmp_path):
    """Relationship vectors: list, exact and HNSW top-k, per-row scoring,
    identity aggregates, index reporting, write → stale → refresh, rollback,
    and a `.kgl` round trip — identical across modes AND equal to the oracle."""
    import kglite

    graphs = {
        "memory": _build_graph("memory"),
        "mapped": _build_graph("mapped"),
        "disk": _build_graph("disk", path=str(tmp_path / "kg_disk")),
    }
    for kg in graphs.values():
        _seed_relationship_store(kg)

    # ── absolute goldens from the independent oracle ──
    oracle = _relationship_oracle()
    expected_count = len(oracle)
    expected_scores = sorted((_cosine(v, _REL_QUERY) for _, _, v in oracle), reverse=True)[:_REL_TOP_K]
    expected_scores = [round(score, 4) for score in expected_scores]
    probe = _ParityEmbedder.vector("probe")
    expected_rows = sorted(
        (
            {
                "s": s,
                "t": d,
                "text": f"edge {s} to {d}",
                "score": round(_cosine(v, _REL_QUERY), 4),
                "norm": round(math.sqrt(sum(x * x for x in v)), 4),
                "text_score": round(_cosine(v, probe), 4),
            }
            for s, d, v in oracle
            if s < 40
        ),
        key=lambda row: (row["s"], row["t"], row["score"], row["text"]),
    )

    before = {mode: _relationship_battery(kg) for mode, kg in graphs.items()}
    for mode, battery in before.items():
        assert battery["listed"] == [
            {
                "entity": "relationship",
                "count": expected_count,
                "dimension": 3,
                "metric": "cosine",
                "model": None,
                "index_state": "online",
                "delta": 0,
            }
        ], mode
        assert battery["exact_scores"] == expected_scores, mode
        assert battery["exact_route"] == {"exact"}, mode
        assert battery["approx_route"] == {"hnsw"}, mode
        # The approximate arm is held to route, ordering, the exact best at
        # rank 1 and a recall floor — member identity is not a contract of an
        # approximate index, and true parallel duplicates still tie.
        approx = battery["approx_scores"]
        assert approx == sorted(approx, reverse=True) and len(approx) == _REL_TOP_K, mode
        assert approx[0] == expected_scores[0], mode
        assert sum(score >= expected_scores[-1] - 1e-6 for score in approx) >= 0.9 * _REL_TOP_K, mode
        assert battery["per_row"] == expected_rows, mode
        assert battery["ids"][0]["c"] == expected_count, mode
        assert [(row["name"], row["type"], row["entityType"], row["state"]) for row in battery["indexes"]] == [
            ("relationship:RELATED.text", "VECTOR", "RELATIONSHIP", "ONLINE")
        ], mode
    comparable = lambda battery: {key: value for key, value in battery.items() if not key.startswith("approx")}  # noqa: E731
    for mode in ("mapped", "disk"):
        assert comparable(before[mode]) == comparable(before["memory"]), f"{mode} diverges from memory before save"

    # ── write → stale delta → refresh; a failing statement rolls a remove back ──
    for mode, kg in graphs.items():
        kg.cypher(
            "MATCH (a:Entity {eid: 1})-[r:RELATED]->() WITH collect(r)[0] AS r "
            "CALL db.relationship_embeddings.set({type:'RELATED', text_column:'text', "
            "entries:[{relationship: r, vector: [9.0, 9.0, 9.0]}]}) YIELD stored RETURN stored"
        )
        state = kg.cypher(
            "CALL db.relationship_embeddings.list({type:'RELATED', text_column:'text'}) "
            "YIELD index_state, delta RETURN index_state, delta"
        ).to_list()
        assert state == [{"index_state": "stale", "delta": 1}], mode
        assert kg.cypher(
            "CALL db.relationship_embeddings.refresh_index({type:'RELATED', text_column:'text'}) "
            "YIELD refreshed RETURN refreshed"
        ).to_list() == [{"refreshed": 1}], mode
        with pytest.raises(kglite.CypherExecutionError, match="division by zero"):
            kg.cypher(
                "MATCH (a:Entity {eid: 2})-[r:RELATED]->() WITH collect(r) AS rs "
                "CALL db.relationship_embeddings.remove({type:'RELATED', text_column:'text', relationships: rs}) "
                "YIELD removed WITH removed MATCH (n:Topic {tid: 0}) SET n.bad = 1/0 RETURN removed"
            )
        assert kg.cypher(
            "CALL db.relationship_embeddings.list({type:'RELATED', text_column:'text'}) "
            "YIELD count, index_state, delta RETURN count, index_state, delta"
        ).to_list() == [{"count": expected_count, "index_state": "online", "delta": 0}], mode

    # ── `.kgl` round trip from every mode: the index and the vectors reload ──
    after = {}
    for mode, kg in graphs.items():
        path = tmp_path / f"{mode}.kgl"
        kg.save(str(path))
        reloaded = kglite.load(str(path))
        reloaded.set_embedder(_ParityEmbedder())
        after[mode] = _relationship_battery(reloaded)
    reference = {mode: _relationship_battery(kg) for mode, kg in graphs.items()}
    # A memory or mapped graph writes a `.kgl`, which carries the HNSW index
    # (plan D4); a disk graph's `save()` writes a disk generation directory,
    # and generations persist neither node nor relationship indexes — the
    # node path reports `has_vector_index() == False` after the same round
    # trip. Vectors, scores and identity must still agree everywhere.
    expected_reload_index = {"memory": "online", "mapped": "online", "disk": "none"}
    volatile = ("index_state", "delta")
    for mode in STORAGE_MODES:
        listed = after[mode]["listed"]
        assert listed[0]["index_state"] == expected_reload_index[mode], f"{mode}: {listed}"
        assert {k: v for k, v in listed[0].items() if k not in volatile} == {
            k: v for k, v in reference[mode]["listed"][0].items() if k not in volatile
        }, mode
        assert after[mode]["exact_members"] == reference[mode]["exact_members"], mode
        assert after[mode]["per_row"] == reference[mode]["per_row"], mode
        assert after[mode]["ids"][0]["c"] == expected_count, mode
        expected_indexes = [] if mode == "disk" else [("relationship:RELATED.text", "VECTOR", "RELATIONSHIP", "ONLINE")]
        assert [
            (r["name"], r["type"], r["entityType"], r["state"]) for r in after[mode]["indexes"]
        ] == expected_indexes, mode
    reloaded_comparable = lambda battery: {  # noqa: E731
        k: v for k, v in battery.items() if k not in ("approx_scores", "approx_route", "listed", "indexes")
    }
    for mode in ("mapped", "disk"):
        assert reloaded_comparable(after[mode]) == reloaded_comparable(after["memory"]), (
            f"{mode} checkpoint diverges from the memory `.kgl` after reload"
        )

    # ── the Python writers: endpoint-addressed upsert, then a full re-embed ──
    # Every RELATED pair is a parallel pair (the fixture repeats each (src,
    # dst) once), so the upsert runs on a fresh single-edge type beside it.
    pair_counts = collections.Counter((s, d) for s, d, _ in oracle)
    parallel = next(pair for pair, count in pair_counts.items() if count > 1)
    written = {(eid, eid + 1): [float(eid % 5), float(eid % 3), 2.0] for eid in range(60)}
    for mode, kg in graphs.items():
        kg.cypher(
            "MATCH (a:Entity), (b:Entity) WHERE a.eid < 60 AND b.eid = a.eid + 1 "
            "CREATE (a)-[:NEXT {note: 'next ' + toString(a.eid)}]->(b)"
        )
        report = kg.set_relationship_embeddings("NEXT", "note", written, metric="euclidean")
        assert report == {"embeddings_stored": 60, "dimension": 3, "changed": 60, "store_created": True}, mode
        stored = {(row["source"], row["target"]): row["vector"] for row in kg.relationship_embeddings("NEXT", "note")}
        assert stored == written, mode
        assert kg.embedding_info("NEXT", "note", entity="relationship")["metric"] == "euclidean", mode
        # `set_` replaces (only the written rows remain), `add_` upserts.
        kg.set_relationship_embeddings("NEXT", "note", {(0, 1): [7.0, 7.0, 7.0]})
        assert [(r["source"], r["target"]) for r in kg.relationship_embeddings("NEXT", "note")] == [(0, 1)], mode
        kg.add_relationship_embeddings("NEXT", "note", written)
        stored = {(row["source"], row["target"]): row["vector"] for row in kg.relationship_embeddings("NEXT", "note")}
        assert stored == written, mode
        with pytest.raises(ValueError, match=r"is ambiguous: 2 'RELATED' relationships connect"):
            kg.set_relationship_embeddings("RELATED", "text", {parallel: [1.0, 1.0, 1.0]})
        outcome = kg.embed_relationship_texts("RELATED", "text", mode="all", show_progress=False)
        assert (outcome["embedded"], outcome["dimension"]) == (expected_count, 3), mode
        regenerated = sorted(
            (row["source"], row["target"], tuple(row["vector"]))
            for row in kg.relationship_embeddings("RELATED", "text")
        )
        assert regenerated == sorted((s, d, tuple(_ParityEmbedder.vector(f"edge {s} to {d}"))) for s, d, _ in oracle), (
            mode
        )
        info = kg.embedding_info("RELATED", "text", entity="relationship")
        assert (info["model"], info["hashed"]) == ("parity/stub", expected_count), mode


# ─── Relationship conflict merges ───────────────────────────────────────────
#
# A merge onto an existing relationship stages its write on disk; these pin
# that every read after the load — in the same session and after a save and
# reopen — sees the merged value, with absolute expected values rather than
# memory-as-oracle, since a merge bug shared by every mode is still a bug.


def _new_graph(mode: str, tmp_path, name: str) -> KnowledgeGraph:
    path = str(tmp_path / f"{name}_{mode}") if mode == "disk" else None
    return _build_graph_empty(mode, path)


def _build_graph_empty(mode: str, path: str | None) -> KnowledgeGraph:
    if mode == "memory":
        return KnowledgeGraph()
    if mode == "mapped":
        return KnowledgeGraph(storage="mapped")
    return KnowledgeGraph(storage="disk", path=path)


def _reopen(kg: KnowledgeGraph, mode: str, tmp_path, name: str) -> KnowledgeGraph:
    import kglite

    target = str(tmp_path / f"{name}_{mode}_saved" if mode == "disk" else tmp_path / f"{name}_{mode}.kgl")
    kg.save(target)
    return kglite.load(target)


def _merge_endpoints(kg: KnowledgeGraph) -> None:
    kg.add_nodes(pd.DataFrame({"aid": [1], "title": ["A1"]}), "A", "aid", "title")
    kg.add_nodes(pd.DataFrame({"cid": [100], "title": ["C100"]}), "C", "cid", "title")


def _edge_state(kg: KnowledgeGraph, rel: str) -> list[dict]:
    return _rows(kg.cypher(f"MATCH (:A)-[r:{rel}]->(:C) RETURN r.score AS score, properties(r) AS props"))


MERGE_EXPECTED = {
    "update": {"score": 7, "props": {"score": 7, "tag": "a", "x": 3, "type": "U"}},
    "sum": {"score": 12, "props": {"score": 12, "tag": "a", "x": 3, "type": "U"}},
    "replace": {"score": 7, "props": {"score": 7, "x": 3, "type": "U"}},
    "preserve": {"score": 5, "props": {"score": 5, "tag": "a", "x": 3, "type": "U"}},
    "skip": {"score": 5, "props": {"score": 5, "tag": "a", "type": "U"}},
}


def test_relationship_conflict_merge_parity(tmp_path):
    """`add_relationships` onto an existing relationship applies its conflict
    mode in every storage mode, visibly in-session and after save + reopen."""
    for conflict, expected in MERGE_EXPECTED.items():
        for mode in STORAGE_MODES:
            name = f"merge_{conflict}"
            kg = _new_graph(mode, tmp_path, name)
            _merge_endpoints(kg)
            kg.add_relationships(
                pd.DataFrame({"s": [1], "t": [100], "score": [5], "tag": ["a"]}), "U", "A", "s", "C", "t"
            )
            kg.add_relationships(
                pd.DataFrame({"s": [1], "t": [100], "score": [7], "x": [3]}),
                "U",
                "A",
                "s",
                "C",
                "t",
                conflict_handling=conflict,
            )
            assert _edge_state(kg, "U") == [expected], f"{conflict}/{mode} in-session"
            # A second read route: the property filter, not the projection.
            hits = kg.cypher(f"MATCH (:A)-[r:U]->(:C) WHERE r.score = {expected['score']} RETURN count(r) AS c")
            assert _rows(hits) == [{"c": 1}], f"{conflict}/{mode} filter"
            reopened = _reopen(kg, mode, tmp_path, name)
            assert _edge_state(reopened, "U") == [expected], f"{conflict}/{mode} after reopen"


@pytest.mark.filterwarnings("ignore:create_relationships")
def test_relationship_merge_folds_within_one_call_parity(tmp_path):
    """Rows that fold onto one relationship inside a single call — a loader
    frame repeating a pair, and `create_relationships` paths joining the same
    endpoints — fold identically in every mode."""
    for mode in STORAGE_MODES:
        kg = _new_graph(mode, tmp_path, "fold_loader")
        _merge_endpoints(kg)
        # The type must already exist: a type's first load owns every edge and
        # keeps repeated pairs as parallel relationships.
        kg.add_relationships(pd.DataFrame({"s": [1], "t": [100], "score": [0]}), "U", "A", "s", "C", "t")
        kg.add_relationships(
            pd.DataFrame({"s": [1, 1, 1], "t": [100, 100, 100], "score": [1, 2, 4]}),
            "U",
            "A",
            "s",
            "C",
            "t",
            conflict_handling="sum",
        )
        assert _rows(kg.cypher("MATCH (:A)-[r:U]->(:C) RETURN r.score AS s")) == [{"s": 7}], mode

    folds = {"update": 2, "sum": 3, "skip": 1, "preserve": 1, "replace": 2}
    for conflict, expected in folds.items():
        for mode in STORAGE_MODES:
            name = f"fold_{conflict}"
            kg = _new_graph(mode, tmp_path, name)
            kg.cypher(
                "CREATE (a:A {aid: 1, title: 'A1'}), (b1:B {bid: 10, title: 'B10', score: 1}), "
                "(b2:B {bid: 11, title: 'B11', score: 2}), (c:C {cid: 100, title: 'C100'}), "
                "(a)-[:AB]->(b1), (a)-[:AB]->(b2), (b1)-[:BC]->(c), (b2)-[:BC]->(c)"
            )
            folded = (
                kg.select("A")
                .traverse("AB")
                .traverse("BC")
                .create_relationships("T", conflict_handling=conflict, properties={"B": ["score"]})
            )
            got = _rows(folded.cypher("MATCH (:A)-[r:T]->(:C) RETURN r.score AS s"))
            assert got == [{"s": expected}], f"{conflict}/{mode}: {got}"
            reopened = _reopen(folded, mode, tmp_path, name)
            got = _rows(reopened.cypher("MATCH (:A)-[r:T]->(:C) RETURN r.score AS s"))
            assert got == [{"s": expected}], f"{conflict}/{mode} after reopen: {got}"


def test_replace_relationships_parity(tmp_path):
    """`replace_relationships` prunes then rewrites the source's edges of the
    type, identically in every mode, including a repeated pair in the frame."""
    for mode in STORAGE_MODES:
        kg = _new_graph(mode, tmp_path, "replace_rel")
        _merge_endpoints(kg)
        kg.add_relationships(pd.DataFrame({"s": [1], "t": [100], "score": [5]}), "U", "A", "s", "C", "t")
        kg.replace_relationships(
            pd.DataFrame({"s": [1, 1], "t": [100, 100], "score": [7, 9]}),
            "U",
            "A",
            "s",
            "C",
            "t",
            conflict_handling="sum",
        )
        assert _rows(kg.cypher("MATCH (:A)-[r:U]->(:C) RETURN r.score AS s")) == [{"s": 16}], mode
        reopened = _reopen(kg, mode, tmp_path, "replace_rel")
        assert _rows(reopened.cypher("MATCH (:A)-[r:U]->(:C) RETURN r.score AS s")) == [{"s": 16}], mode


def test_declared_temporal_relationship_merge_parity(tmp_path):
    """On a declared interval type the merge key is the endpoints plus the
    start: a repeated start merges into the stored relationship (closing its
    open period), a new start is a parallel relationship — in every mode."""
    periods = {"vf": "validFrom", "vt": "validTo"}
    for mode in STORAGE_MODES:
        kg = _new_graph(mode, tmp_path, "temporal_merge")
        _merge_endpoints(kg)

        def link(rows, conflict=None):
            frame = pd.DataFrame(rows, columns=["s", "t", "vf", "vt", "w"])
            return kg.add_relationships(
                frame, "IN", "A", "s", "C", "t", conflict_handling=conflict, column_types=periods
            )

        link([(1, 100, "2000-01-01", None, 1)])
        report = link([(1, 100, "2000-01-01", "2005-01-01", 2), (1, 100, "2010-01-01", None, 3)], "update")
        assert (report["connections_created"], report["connections_updated"]) == (1, 1), mode
        query = "MATCH (:A)-[r:IN]->(:C) RETURN toString(r.vf) AS vf, toString(r.vt) AS vt, r.w AS w ORDER BY vf"
        expected = [
            {"vf": "2000-01-01", "vt": "2005-01-01", "w": 2},
            {"vf": "2010-01-01", "vt": None, "w": 3},
        ]
        assert kg.cypher(query).to_list() == expected, mode
        # A merge-only call: no new relationship in the batch to flush it.
        report = link([(1, 100, "2010-01-01", "2012-01-01", 4)], "update")
        assert (report["connections_created"], report["connections_updated"]) == (0, 1), mode
        expected[1] = {"vf": "2010-01-01", "vt": "2012-01-01", "w": 4}
        assert kg.cypher(query).to_list() == expected, mode
        reopened = _reopen(kg, mode, tmp_path, "temporal_merge")
        assert reopened.cypher(query).to_list() == expected, mode


def test_add_properties_parity(tmp_path):
    """`add_properties` writes are visible to the next read in every mode —
    disk stages them, so the call itself must drain the stage."""
    for mode in STORAGE_MODES:
        kg = _new_graph(mode, tmp_path, "add_props")
        kg.cypher(
            "CREATE (a:A {aid: 1, title: 'A1', region: 'north'}), "
            "(b:B {bid: 10, title: 'B10', score: 4}), (a)-[:AB]->(b)"
        )
        enriched = kg.select("A").traverse("AB").add_properties({"A": ["region"]})
        query = "MATCH (b:B) RETURN b.region AS region, b.score AS score"
        expected = [{"region": "north", "score": 4}]
        assert _rows(enriched.cypher(query)) == expected, mode
        filtered = enriched.cypher("MATCH (b:B) WHERE b.region = 'north' RETURN count(b) AS c")
        assert _rows(filtered) == [{"c": 1}], mode
        assert _rows(_reopen(enriched, mode, tmp_path, "add_props").cypher(query)) == expected, mode
