"""KG-1 regression: a user property named like a structural accessor.

Reported by kglite-docs (2026-05-30, verified on 0.10.9): a node property
named ``label`` (also ``type`` / ``node_type`` / ``name``) was silently
shadowed by KGLite's convenience accessor — ``n.label`` returned the node's
*type string* rather than the stored value. The data round-tripped; it was
just unreadable by that name.

Fix: **property-first** resolution. A stored property of that name wins; the
structural convenience is only a fallback used when no such property exists.
``id`` / ``title`` stay as the node's identity fields (no stored property can
shadow them).

Parametrised across storage modes because the two resolution paths differ:
memory / mapped go through ``resolve_node_property`` (helpers.rs); disk goes
through the column fast path (``resolve_property`` in expression.rs).

Run: pytest tests/test_cypher_reserved_property_names.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from kglite import KnowledgeGraph

STORAGE_MODES = ("memory", "mapped", "disk")

SOFT_ALIASES = ("label", "type", "node_type", "name")


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


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_soft_alias_properties_are_readable(mode, tmp_path):
    """A stored property named like a soft accessor reads back its value."""
    kg = _new_kg(mode, str(tmp_path / f"kg_{mode}"))
    kg.add_nodes(
        pd.DataFrame(
            [
                {
                    "id": "s1",
                    "label": "MyLabelProp",
                    "type": "Ty",
                    "node_type": "NT",
                    "name": "Nm",
                    "doc_type": "Contestacao",
                }
            ]
        ),
        "SourceDoc",
        unique_id_field="id",
        node_title_field="id",
    )

    got = kg.cypher(
        "MATCH (s:SourceDoc) "
        "RETURN s.label AS label, s.type AS type, s.node_type AS node_type, "
        "s.name AS name, s.doc_type AS doc_type, s.id AS id"
    ).to_list()[0]

    assert got["label"] == "MyLabelProp", "stored `label` property must win"
    assert got["type"] == "Ty"
    assert got["node_type"] == "NT"
    assert got["name"] == "Nm"
    assert got["doc_type"] == "Contestacao"  # control: a plain property
    assert got["id"] == "s1"  # identity field, unaffected


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_soft_alias_fallback_intact_when_unset(mode, tmp_path):
    """With no such property, the structural convenience still answers."""
    kg = _new_kg(mode, str(tmp_path / f"kg_{mode}"))
    kg.add_nodes(
        pd.DataFrame([{"id": "x1", "val": 7}]),
        "Thing",
        unique_id_field="id",
        node_title_field="id",
    )

    got = kg.cypher(
        "MATCH (n:Thing) RETURN n.label AS label, n.type AS type, n.node_type AS node_type, n.name AS name"
    ).to_list()[0]

    # type / node_type / label fall back to the node type string.
    assert got["label"] == "Thing"
    assert got["type"] == "Thing"
    assert got["node_type"] == "Thing"
    # name falls back to the node title.
    assert got["name"] == "x1"


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_soft_alias_in_map_projection(mode, tmp_path):
    """`RETURN n{.*}` emits each stored soft-alias property once, by value."""
    kg = _new_kg(mode, str(tmp_path / f"kg_{mode}"))
    kg.add_nodes(
        pd.DataFrame([{"id": "s1", "label": "L", "type": "T", "name": "N"}]),
        "SourceDoc",
        unique_id_field="id",
        node_title_field="id",
    )

    m = kg.cypher("MATCH (s:SourceDoc) RETURN s{.*} AS m").to_list()[0]["m"]

    assert m["label"] == "L"
    assert m["type"] == "T"
    assert m["name"] == "N"
    # No duplicate keys (a Python dict guarantees uniqueness, but assert the
    # values are the stored ones, not the structural type string).
    assert m["label"] != "SourceDoc"
    assert m["type"] != "SourceDoc"


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_soft_alias_in_where_filter(mode, tmp_path):
    """A WHERE predicate on a soft-alias property matches the stored value."""
    kg = _new_kg(mode, str(tmp_path / f"kg_{mode}"))
    kg.add_nodes(
        pd.DataFrame(
            [
                {"id": "a", "label": "keep"},
                {"id": "b", "label": "drop"},
            ]
        ),
        "Doc",
        unique_id_field="id",
        node_title_field="id",
    )

    rows = kg.cypher("MATCH (d:Doc) WHERE d.label = 'keep' RETURN d.id AS id").to_list()
    assert [r["id"] for r in rows] == ["a"]


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_count_by_type_groups_by_stored_property(mode, tmp_path):
    """`RETURN n.type, count(*)` groups by the stored `type` property, not the
    primary type — the count-fusion must be gated off when a shadow exists."""
    kg = _new_kg(mode, str(tmp_path / f"kg_{mode}"))
    kg.add_nodes(
        pd.DataFrame(
            [
                {"id": 1, "type": "T1"},
                {"id": 2, "type": "T1"},
                {"id": 3, "type": "T2"},
            ]
        ),
        "Thing",
        unique_id_field="id",
        node_title_field="id",
    )

    rows = sorted(
        kg.cypher("MATCH (n:Thing) RETURN n.type AS t, count(*) AS c").to_list(),
        key=lambda r: str(r["t"]),
    )
    # Must be grouped by the stored values T1/T2 — NOT a single 'Thing' bucket.
    assert rows == [{"t": "T1", "c": 2}, {"t": "T2", "c": 1}]


def test_count_by_type_still_fuses_without_shadow():
    """No shadow property → the fast count-by-type fusion still applies and
    returns the primary type. (Correctness check; perf verified separately.)"""
    kg = KnowledgeGraph()
    kg.add_nodes(
        pd.DataFrame([{"id": 1}, {"id": 2}]),
        "Alpha",
        unique_id_field="id",
        node_title_field="id",
    )
    kg.add_nodes(
        pd.DataFrame([{"id": 3}]),
        "Beta",
        unique_id_field="id",
        node_title_field="id",
    )

    rows = sorted(
        kg.cypher("MATCH (n) RETURN n.type AS t, count(*) AS c").to_list(),
        key=lambda r: str(r["t"]),
    )
    assert rows == [{"t": "Alpha", "c": 2}, {"t": "Beta", "c": 1}]
