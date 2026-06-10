"""Tests for KnowledgeGraph.extend() — native in-place graph merge.

Covers: disjoint merge, every conflict_handling mode on overlapping
nodes, property-schema extension, secondary-label union, edge dedup +
edge properties, self-extend, empty graphs both directions, the
in-memory-only scope restriction, the embedding-store warning,
report-dict count exactness, source-graph immutability, and a timing
sanity bound.
"""

from __future__ import annotations

import time
import warnings

import pandas as pd
import pytest

from kglite import KnowledgeGraph


def _people(g: KnowledgeGraph, rows, conflict=None):
    g.add_nodes(
        pd.DataFrame(rows),
        "Person",
        "pid",
        node_title_field="name",
        conflict_handling=conflict,
    )


def _all(g: KnowledgeGraph, q: str):
    return g.cypher(q).to_list()


# ─────────────────────────── disjoint merge ───────────────────────────


def test_disjoint_node_merge():
    g1 = KnowledgeGraph()
    g2 = KnowledgeGraph()
    _people(g1, [{"pid": 1, "name": "Alice"}, {"pid": 2, "name": "Bob"}])
    _people(g2, [{"pid": 3, "name": "Carol"}, {"pid": 4, "name": "Dave"}])

    rep = g1.extend(g2)
    assert rep["nodes_created"] == 2
    assert rep["nodes_updated"] == 0
    assert rep["nodes_skipped"] == 0
    assert rep["has_errors"] is False

    titles = sorted(r["t"] for r in _all(g1, "MATCH (n:Person) RETURN n.title AS t"))
    assert titles == ["Alice", "Bob", "Carol", "Dave"]


def test_disjoint_different_node_types():
    g1 = KnowledgeGraph()
    g2 = KnowledgeGraph()
    _people(g1, [{"pid": 1, "name": "Alice"}])
    g2.add_nodes(pd.DataFrame([{"cid": 10, "label": "ACME"}]), "Company", "cid")

    rep = g1.extend(g2)
    assert rep["nodes_created"] == 1
    assert rep["node_types_merged"] == 1
    assert _all(g1, "MATCH (n:Company) RETURN n.id AS id") == [{"id": 10}]


# ───────────────────── overlapping nodes per mode ──────────────────────


@pytest.fixture
def overlap_pair():
    g1 = KnowledgeGraph()
    g2 = KnowledgeGraph()
    # pid=1 overlaps; g1 has age, g2 overrides name + adds city.
    _people(g1, [{"pid": 1, "name": "Alice", "age": 30}])
    _people(g2, [{"pid": 1, "name": "Alicia", "city": "NYC"}])
    return g1, g2


def test_overlap_update(overlap_pair):
    g1, g2 = overlap_pair
    rep = g1.extend(g2, conflict_handling="update")
    assert rep["nodes_created"] == 0
    assert rep["nodes_updated"] == 1
    row = _all(g1, "MATCH (n:Person) RETURN n.title AS t, n.age AS age, n.city AS city")[0]
    assert row == {"t": "Alicia", "age": 30, "city": "NYC"}  # other wins on name


def test_overlap_preserve(overlap_pair):
    g1, g2 = overlap_pair
    g1.extend(g2, conflict_handling="preserve")
    row = _all(g1, "MATCH (n:Person) RETURN n.title AS t, n.age AS age, n.city AS city")[0]
    # existing values win; new-only property (city) still added
    assert row == {"t": "Alice", "age": 30, "city": "NYC"}


def test_overlap_skip(overlap_pair):
    g1, g2 = overlap_pair
    g1.extend(g2, conflict_handling="skip")
    row = _all(g1, "MATCH (n:Person) RETURN n.title AS t, n.age AS age, n.city AS city")[0]
    # untouched: no name change, no city added
    assert row == {"t": "Alice", "age": 30, "city": None}


def test_overlap_replace(overlap_pair):
    g1, g2 = overlap_pair
    g1.extend(g2, conflict_handling="replace")
    row = _all(g1, "MATCH (n:Person) RETURN n.title AS t, n.age AS age, n.city AS city")[0]
    # all props replaced by other's: age gone, city present, title=Alicia
    assert row == {"t": "Alicia", "age": None, "city": "NYC"}


def test_overlap_sum_nodes_act_as_update():
    # 'sum' adds numeric values for EDGE properties; for NODE properties it
    # acts as 'update' (matches add_nodes / ConflictHandling::Sum semantics).
    g1 = KnowledgeGraph()
    g2 = KnowledgeGraph()
    g1.add_nodes(pd.DataFrame([{"pid": 1, "score": 10}]), "P", "pid")
    g2.add_nodes(pd.DataFrame([{"pid": 1, "score": 5}]), "P", "pid")
    g1.extend(g2, conflict_handling="sum")
    assert _all(g1, "MATCH (n:P) RETURN n.score AS s") == [{"s": 5}]


def test_overlap_sum_edges_add():
    # On edges, 'sum' adds numeric properties.
    g1 = KnowledgeGraph()
    g2 = KnowledgeGraph()
    _people(g1, [{"pid": 1, "name": "A"}, {"pid": 2, "name": "B"}])
    _people(g2, [{"pid": 1, "name": "A"}, {"pid": 2, "name": "B"}])
    _knows(g1, [{"src": 1, "tgt": 2, "weight": 10}])
    _knows(g2, [{"src": 1, "tgt": 2, "weight": 5}])
    g1.extend(g2, conflict_handling="sum")
    assert _all(g1, "MATCH (a)-[r:KNOWS]->(b) RETURN r.weight AS w") == [{"w": 15}]


# ───────────────────── schema / property extension ─────────────────────


def test_property_schema_extension():
    g1 = KnowledgeGraph()
    g2 = KnowledgeGraph()
    _people(g1, [{"pid": 1, "name": "Alice", "age": 30}])
    # g2 introduces a brand-new property on a NEW node of the same type
    _people(g2, [{"pid": 2, "name": "Bob", "email": "bob@x.io"}])
    g1.extend(g2)
    row = _all(g1, "MATCH (n:Person {id: 2}) RETURN n.email AS e, n.title AS t")[0]
    assert row == {"e": "bob@x.io", "t": "Bob"}
    # existing node has no email
    assert _all(g1, "MATCH (n:Person {id: 1}) RETURN n.email AS e") == [{"e": None}]


# ─────────────────────── secondary-label union ─────────────────────────


def test_secondary_label_union():
    g1 = KnowledgeGraph()
    g2 = KnowledgeGraph()
    _people(g1, [{"pid": 1, "name": "Alice"}])
    _people(g2, [{"pid": 1, "name": "Alice"}, {"pid": 2, "name": "Bob"}])
    # Label both g2 people as :Employee
    g2.add_nodes(
        pd.DataFrame([{"pid": 1, "name": "Alice"}, {"pid": 2, "name": "Bob"}]),
        "Person",
        "pid",
        node_title_field="name",
        labels=["Employee"],
    )
    rep = g1.extend(g2)
    assert rep["labels_unioned"] >= 1
    # Both nodes now carry the Employee secondary label in g1
    emp = sorted(r["t"] for r in _all(g1, "MATCH (n:Employee) RETURN n.title AS t"))
    assert emp == ["Alice", "Bob"]


# ──────────────────────── edges: dedup + props ─────────────────────────


def _knows(g, pairs, conflict=None):
    df = pd.DataFrame(pairs)
    prop_cols = [c for c in df.columns if c not in ("src", "tgt")]
    g.add_connections(
        df,
        "KNOWS",
        "Person",
        "src",
        "Person",
        "tgt",
        columns=prop_cols or None,
        conflict_handling=conflict,
    )


def test_edge_disjoint_and_dedup():
    g1 = KnowledgeGraph()
    g2 = KnowledgeGraph()
    _people(g1, [{"pid": i, "name": f"P{i}"} for i in range(1, 5)])
    _people(g2, [{"pid": i, "name": f"P{i}"} for i in range(1, 5)])
    _knows(g1, [{"src": 1, "tgt": 2}])
    # g2: one duplicate (1->2) + one new (3->4)
    _knows(g2, [{"src": 1, "tgt": 2}, {"src": 3, "tgt": 4}])

    rep = g1.extend(g2)
    # The duplicate (1->2) is NOT re-created as a parallel edge: the graph
    # holds exactly two distinct edges afterwards.
    edges = _all(g1, "MATCH (a)-[:KNOWS]->(b) RETURN a.id AS s, b.id AS t")
    pairs = sorted((e["s"], e["t"]) for e in edges)
    assert pairs == [(1, 2), (3, 4)]
    # edges_created mirrors add_connections: it counts every edge touched
    # (one new 3->4 + one property-merge on the existing 1->2) = 2.
    assert rep["edges_created"] == 2


def test_edge_properties_merge_on_dedup():
    g1 = KnowledgeGraph()
    g2 = KnowledgeGraph()
    _people(g1, [{"pid": 1, "name": "A"}, {"pid": 2, "name": "B"}])
    _people(g2, [{"pid": 1, "name": "A"}, {"pid": 2, "name": "B"}])
    _knows(g1, [{"src": 1, "tgt": 2, "since": 2000}])
    _knows(g2, [{"src": 1, "tgt": 2, "weight": 5}])
    g1.extend(g2, conflict_handling="update")
    row = _all(
        g1,
        "MATCH (a)-[r:KNOWS]->(b) RETURN r.since AS since, r.weight AS weight",
    )[0]
    assert row == {"since": 2000, "weight": 5}


def test_edge_new_properties():
    g1 = KnowledgeGraph()
    g2 = KnowledgeGraph()
    _people(g1, [{"pid": 1, "name": "A"}, {"pid": 2, "name": "B"}])
    _people(g2, [{"pid": 1, "name": "A"}, {"pid": 2, "name": "B"}])
    _knows(g2, [{"src": 1, "tgt": 2, "since": 1999}])
    g1.extend(g2)
    assert _all(g1, "MATCH (a)-[r:KNOWS]->(b) RETURN r.since AS s") == [{"s": 1999}]


# ───────────────────────────── self-extend ─────────────────────────────


def test_self_extend_no_op_under_update():
    g = KnowledgeGraph()
    _people(g, [{"pid": 1, "name": "A"}, {"pid": 2, "name": "B"}])
    _knows(g, [{"src": 1, "tgt": 2}])
    before = sorted(r["t"] for r in _all(g, "MATCH (n:Person) RETURN n.title AS t"))
    rep = g.extend(g)  # self
    # No new nodes/edges materialise — every node and edge matches itself.
    assert rep["nodes_created"] == 0
    after = sorted(r["t"] for r in _all(g, "MATCH (n:Person) RETURN n.title AS t"))
    assert before == after
    # Crucially: node + edge COUNTS are unchanged (no duplicates created).
    assert _all(g, "MATCH (n:Person) RETURN count(n) AS c") == [{"c": 2}]
    assert _all(g, "MATCH ()-[r:KNOWS]->() RETURN count(r) AS c") == [{"c": 1}]


# ─────────────────────────── empty graphs ──────────────────────────────


def test_extend_with_empty_source():
    g1 = KnowledgeGraph()
    g2 = KnowledgeGraph()
    _people(g1, [{"pid": 1, "name": "A"}])
    rep = g1.extend(g2)
    assert rep["nodes_created"] == 0
    assert rep["edges_created"] == 0
    assert _all(g1, "MATCH (n:Person) RETURN count(n) AS c") == [{"c": 1}]


def test_extend_into_empty_target():
    g1 = KnowledgeGraph()
    g2 = KnowledgeGraph()
    _people(g2, [{"pid": 1, "name": "A"}, {"pid": 2, "name": "B"}])
    _knows(g2, [{"src": 1, "tgt": 2}])
    rep = g1.extend(g2)
    assert rep["nodes_created"] == 2
    assert rep["edges_created"] == 1
    assert _all(g1, "MATCH (a)-[:KNOWS]->(b) RETURN a.id AS s, b.id AS t") == [{"s": 1, "t": 2}]


# ─────────────────────── mode-restriction error ────────────────────────


def test_mapped_source_rejected():
    g1 = KnowledgeGraph()
    g2 = KnowledgeGraph(storage="mapped")
    _people(g1, [{"pid": 1, "name": "A"}])
    with pytest.raises(Exception) as exc:
        g1.extend(g2)
    assert "in-memory" in str(exc.value).lower()


def test_mapped_target_rejected():
    g1 = KnowledgeGraph(storage="mapped")
    g2 = KnowledgeGraph()
    _people(g2, [{"pid": 1, "name": "A"}])
    with pytest.raises(Exception) as exc:
        g1.extend(g2)
    assert "in-memory" in str(exc.value).lower()


# ───────────────────── embedding-store warning ─────────────────────────


def test_embedding_store_warning():
    g1 = KnowledgeGraph()
    g2 = KnowledgeGraph()
    _people(g1, [{"pid": 1, "name": "A"}])
    _people(g2, [{"pid": 2, "name": "B"}])
    # Attach an embedding to g2 directly (bypassing a model).
    g2.add_embeddings("Person", "vec", {2: [0.1, 0.2, 0.3]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g1.extend(g2)
    msgs = [str(w.message) for w in caught]
    assert any("embedding" in m.lower() for m in msgs), msgs
    # Node still merged despite skipped embeddings.
    assert _all(g1, "MATCH (n:Person {id: 2}) RETURN n.id AS id") == [{"id": 2}]


# ──────────────────── report-dict count exactness ──────────────────────


def test_report_dict_counts_exact():
    g1 = KnowledgeGraph()
    g2 = KnowledgeGraph()
    _people(g1, [{"pid": 1, "name": "A"}])
    _people(g2, [{"pid": 1, "name": "A"}, {"pid": 2, "name": "B"}, {"pid": 3, "name": "C"}])
    _knows(g2, [{"src": 1, "tgt": 2}, {"src": 2, "tgt": 3}])

    rep = g1.extend(g2)
    assert rep["nodes_created"] == 2  # pid 2,3
    assert rep["nodes_updated"] == 1  # pid 1
    assert rep["nodes_skipped"] == 0
    assert rep["edges_created"] == 2
    assert rep["node_types_merged"] == 1
    assert rep["connection_types_merged"] == 1
    assert "processing_time_ms" in rep
    assert rep["has_errors"] is False


# ──────────────────── source-graph immutability ────────────────────────


def test_source_graph_not_mutated():
    g1 = KnowledgeGraph()
    g2 = KnowledgeGraph()
    _people(g1, [{"pid": 1, "name": "Alice", "age": 30}])
    _people(g2, [{"pid": 1, "name": "Alicia"}, {"pid": 2, "name": "Bob"}])
    g2_before = sorted(r["t"] for r in _all(g2, "MATCH (n:Person) RETURN n.title AS t"))
    g2_count_before = _all(g2, "MATCH (n:Person) RETURN count(n) AS c")[0]["c"]

    g1.extend(g2, conflict_handling="replace")

    g2_after = sorted(r["t"] for r in _all(g2, "MATCH (n:Person) RETURN n.title AS t"))
    g2_count_after = _all(g2, "MATCH (n:Person) RETURN count(n) AS c")[0]["c"]
    assert g2_before == g2_after
    assert g2_count_before == g2_count_after == 2


# ──────────────────────────── timing sanity ────────────────────────────


@pytest.mark.parametrize("n", [50_000])
def test_timing_50k_into_50k_50pct_overlap(n):
    g1 = KnowledgeGraph()
    g2 = KnowledgeGraph()
    # g1: ids 0..n ; g2: ids n/2 .. 3n/2  → 50% overlap.
    half = n // 2
    df1 = pd.DataFrame({"pid": range(0, n), "name": [f"a{i}" for i in range(0, n)]})
    df2 = pd.DataFrame({"pid": range(half, half + n), "name": [f"b{i}" for i in range(half, half + n)]})
    g1.add_nodes(df1, "Person", "pid", node_title_field="name")
    g2.add_nodes(df2, "Person", "pid", node_title_field="name")
    # Edges in g2: a chain over its node range (n edges).
    e2 = pd.DataFrame({"src": range(half, half + n - 1), "tgt": range(half + 1, half + n)})
    g2.add_connections(e2, "NEXT", "Person", "src", "Person", "tgt")

    t0 = time.perf_counter()
    rep = g1.extend(g2)
    elapsed = time.perf_counter() - t0

    # 50% overlap → half created, half updated.
    assert rep["nodes_created"] == half
    assert rep["nodes_updated"] == n - half
    assert rep["edges_created"] == n - 1
    # Generous bound; the merge is single-pass with id-index lookups.
    assert elapsed < 5.0, f"extend took {elapsed:.2f}s (expected < 5s)"
