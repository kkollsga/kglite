"""Regression locks for the four issues kglite-docs re-verified on 0.10.9.

kglite-docs (2026-05-30) re-ran four previously-reported issues against
0.10.9 and confirmed all four now pass, asking us to confirm dedicated
regression coverage exists — N4 especially, since "silent edge loss on
reload" is a severe failure mode. N1–N3 had incidental coverage; this file
pins the docs-exact shapes. N4 had NO dedicated test (the only cypher-CREATE
save/load test created a node, not an edge) — that gap is closed here.

Shapes:
  N1 — shared-variable comma-join in MATCH
  N2 — reverse-arrow direction match (count parity vs forward)
  N3 — inline-map property in multi-MATCH then CREATE
  N4 — cypher-CREATEd edges surviving save -> load

Run: pytest tests/test_cypher_docs_followup_n1_n4.py
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph


def _docs_fixture(kg: KnowledgeGraph) -> KnowledgeGraph:
    """The shared node/edge setup from the kglite-docs report."""
    kg.add_nodes(pd.DataFrame([{"id": "s1"}]), "SourceDoc", unique_id_field="id", node_title_field="id")
    kg.add_nodes(
        pd.DataFrame([{"id": "c1"}, {"id": "c2"}]),
        "Chunk",
        unique_id_field="id",
        node_title_field="id",
    )
    kg.add_nodes(pd.DataFrame([{"id": "a1"}]), "Assessment", unique_id_field="id", node_title_field="id")
    kg.add_connections(
        pd.DataFrame([{"src": "s1", "dst": "c1"}]),
        "HAS_CHUNK",
        source_type="SourceDoc",
        source_id_field="src",
        target_type="Chunk",
        target_id_field="dst",
    )
    kg.add_connections(
        pd.DataFrame([{"src": "a1", "dst": "c1"}]),
        "ASSESSES",
        source_type="Assessment",
        source_id_field="src",
        target_type="Chunk",
        target_id_field="dst",
    )
    return kg


def test_n1_shared_variable_comma_join():
    """Two comma-separated patterns sharing `c` must JOIN on it."""
    g = _docs_fixture(KnowledgeGraph())
    rows = g.cypher(
        "MATCH (s:SourceDoc)-[:HAS_CHUNK]->(c:Chunk), (a:Assessment)-[:ASSESSES]->(c) RETURN c.id AS id"
    ).to_list()
    assert rows == [{"id": "c1"}], "shared var c must join the two patterns to c1 only"


def test_n2_reverse_arrow_count_parity():
    """`(c)<-[:ASSESSES]-(a)` must match the same rows as the forward form."""
    g = _docs_fixture(KnowledgeGraph())
    fwd = g.cypher("MATCH (a:Assessment)-[:ASSESSES]->(c:Chunk) RETURN count(a) AS n").to_list()
    rev = g.cypher("MATCH (c:Chunk)<-[:ASSESSES]-(a:Assessment) RETURN count(a) AS n").to_list()
    assert fwd == rev == [{"n": 1}]


def test_n3_inline_map_multi_match_then_create():
    """Inline-map filters in a comma-MATCH, then CREATE using both vars."""
    g = _docs_fixture(KnowledgeGraph())
    g.cypher("MATCH (s:SourceDoc {id: 's1'}), (c:Chunk {id: 'c1'}) CREATE (s)-[:HAS_CHUNK_2]->(c)")
    n = g.cypher("MATCH (:SourceDoc)-[:HAS_CHUNK_2]->(:Chunk) RETURN count(*) AS n").to_list()
    assert n == [{"n": 1}]


@pytest.mark.parametrize("storage", [None, "mapped"])
def test_n4_cypher_created_edge_survives_save_load(storage, tmp_path):
    """The important one: an edge created via cypher CREATE (not
    add_connections) must still be present after save -> kglite.load."""
    kg = KnowledgeGraph() if storage is None else KnowledgeGraph(storage=storage)
    kg.add_nodes(
        pd.DataFrame([{"id": 1}, {"id": 2}]),
        "P",
        unique_id_field="id",
        node_title_field="id",
    )
    kg.cypher("MATCH (a:P {id: 1}), (b:P {id: 2}) CREATE (a)-[:LINK]->(b)")

    pre = kg.cypher("MATCH ()-[:LINK]->() RETURN count(*) AS n").to_list()[0]["n"]
    assert pre == 1, "edge must exist in-session before save"

    with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
        path = f.name
    try:
        kg.save(path)
        loaded = kglite.load(path)
        post = loaded.cypher("MATCH ()-[:LINK]->() RETURN count(*) AS n").to_list()[0]["n"]
        assert post == 1, "cypher-CREATEd edge must survive save -> load"
    finally:
        os.unlink(path)
