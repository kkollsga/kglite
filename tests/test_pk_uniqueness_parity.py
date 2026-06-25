"""Cross-mode parity for declared PRIMARY KEY uniqueness enforcement.

A node type that declares `primary_key: 'id'` must enforce uniqueness on the
identity key IDENTICALLY across memory / mapped / disk: a duplicate CREATE is
rejected, MERGE still upserts, and undeclared types stay permissive. This is
the conformance net for the enforcement (the differential corpus and set-equality
oracles structurally can't assert "the write errored the same way everywhere").

Run: pytest tests/test_pk_uniqueness_parity.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from kglite import KnowledgeGraph

MODES = ("memory", "mapped", "disk")


def _fresh(mode: str, tmp_path) -> KnowledgeGraph:
    if mode == "memory":
        return KnowledgeGraph()
    if mode == "mapped":
        return KnowledgeGraph(storage="mapped")
    return KnowledgeGraph(storage="disk", path=str(tmp_path / "g"))


def _count(kg: KnowledgeGraph, label: str) -> int:
    return kg.cypher(f"MATCH (n:{label}) RETURN count(n) AS c").to_dicts()[0]["c"]


@pytest.mark.parametrize("mode", MODES)
def test_duplicate_create_rejected_everywhere(mode, tmp_path):
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"primary_key": "id"}}})
    kg.cypher("CREATE (:Person {id: 1, name: 'A'})")
    with pytest.raises(Exception, match="duplicate primary key"):
        kg.cypher("CREATE (:Person {id: 1, name: 'B'})")
    # The rejected write left exactly one node — no partial insert.
    assert _count(kg, "Person") == 1


@pytest.mark.parametrize("mode", MODES)
def test_merge_still_upserts_everywhere(mode, tmp_path):
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"primary_key": "id"}}})
    kg.cypher("CREATE (:Person {id: 1, name: 'A'})")
    # MERGE on the existing key matches-and-updates, never errors.
    kg.cypher("MERGE (p:Person {id: 1}) SET p.name = 'A2'")
    assert _count(kg, "Person") == 1
    assert kg.cypher("MATCH (p:Person {id: 1}) RETURN p.name AS n").to_dicts()[0]["n"] == "A2"


@pytest.mark.parametrize("mode", MODES)
def test_undeclared_type_stays_permissive(mode, tmp_path):
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"primary_key": "id"}}})
    # Doc has no declared PK → today's permissive behaviour (2 nodes).
    kg.cypher("CREATE (:Doc {id: 1})")
    kg.cypher("CREATE (:Doc {id: 1})")
    assert _count(kg, "Doc") == 2


@pytest.mark.parametrize("mode", MODES)
def test_within_statement_bulk_dup_rejected(mode, tmp_path):
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"T": {"primary_key": "id"}}})
    with pytest.raises(Exception, match="duplicate primary key"):
        kg.cypher("UNWIND [1, 2, 3, 2] AS i CREATE (:T {id: i})")


@pytest.mark.parametrize("mode", MODES)
def test_string_id_pk_enforced(mode, tmp_path):
    """The MAG worked example: an arbitrary string id as the primary key."""
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Mem": {"primary_key": "id"}}})
    kg.cypher("CREATE (:Mem {id: 'k1', body: 'first'})")
    with pytest.raises(Exception, match="duplicate primary key"):
        kg.cypher("CREATE (:Mem {id: 'k1', body: 'second'})")
    assert _count(kg, "Mem") == 1


@pytest.mark.parametrize("mode", MODES)
def test_add_nodes_within_batch_dup_rejected(mode, tmp_path):
    """A repeated id within one add_nodes input is a data error for a
    declared-PK type (it would otherwise become a hidden duplicate)."""
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"primary_key": "id"}}})
    df = pd.DataFrame({"id": [1, 2, 2, 3], "name": ["a", "b", "b2", "c"]})
    with pytest.raises(Exception, match="duplicate primary key"):
        kg.add_nodes(df, "Person", "id", "name")


@pytest.mark.parametrize("mode", MODES)
def test_add_nodes_vs_existing_still_upserts(mode, tmp_path):
    """add_nodes stays the upsert path vs the *existing* graph: re-adding an
    existing id updates in place (no new node, no error) — only within-batch
    repeats are rejected."""
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"primary_key": "id"}}})
    kg.add_nodes(pd.DataFrame({"id": [1, 2], "name": ["a", "b"]}), "Person", "id", "name")
    kg.add_nodes(pd.DataFrame({"id": [1], "name": ["a-updated"]}), "Person", "id", "name")
    assert _count(kg, "Person") == 2
    assert kg.cypher("MATCH (p:Person {id: 1}) RETURN p.name AS n").to_dicts()[0]["n"] == "a-updated"


@pytest.mark.parametrize("mode", MODES)
def test_add_nodes_unique_batch_ok_on_pk_type(mode, tmp_path):
    """A clean (already-unique) batch loads fine on a declared-PK type."""
    kg = _fresh(mode, tmp_path)
    kg.define_schema({"nodes": {"Person": {"primary_key": "id"}}})
    kg.add_nodes(
        pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]}),
        "Person",
        "id",
        "name",
    )
    assert _count(kg, "Person") == 3
