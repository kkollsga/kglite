"""Golden assertions for node-`id` semantics in the Cypher path.

These are *correctness* assertions (expected result, not optimised-vs-naive
consistency) for how the `id` property maps to node identity — the layer the
differential corpus and parity oracles structurally cannot cover (a bug
present in every pass-config / storage-mode is invisible to them).

Covers the 0.10.10 fix: cypher `CREATE (n {id: X})` honours X as the node's
unique identity (consistent with `add_nodes(unique_id_field='id')`), so it
round-trips and is matchable by `{id: X}`. Previously CREATE discarded X and
auto-assigned a UniqueId.

Run: pytest tests/test_cypher_id_semantics.py
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

import kglite
from kglite import KnowledgeGraph

# cypher CREATE is supported on every storage mode (memory/mapped/disk).
# On disk, properties route through the per-type ColumnStore via
# DirGraph::insert_node_routed (the same mechanism add_nodes uses).
CREATE_MODES = ("memory", "mapped", "disk")


def _new_kg(mode: str, tmp_path=None) -> KnowledgeGraph:
    if mode == "memory":
        return KnowledgeGraph()
    if mode == "mapped":
        return KnowledgeGraph(storage="mapped")
    if mode == "disk":
        assert tmp_path is not None, "disk mode needs a path"
        return KnowledgeGraph(storage="disk", path=str(tmp_path / "kg"))
    raise ValueError(mode)


@pytest.mark.parametrize("mode", CREATE_MODES)
def test_create_honours_string_id(mode, tmp_path):
    kg = _new_kg(mode, tmp_path)
    kg.cypher("CREATE (:Doc {id: 's1', extra: 7})")
    assert kg.cypher("MATCH (n:Doc) RETURN n.id AS id").to_list() == [{"id": "s1"}]
    # matchable by the provided id
    assert kg.cypher("MATCH (n:Doc {id: 's1'}) RETURN n.extra AS e").to_list() == [{"e": 7}]
    assert kg.cypher("MATCH (n:Doc) WHERE n.id = 's1' RETURN n.extra AS e").to_list() == [{"e": 7}]


@pytest.mark.parametrize("mode", CREATE_MODES)
def test_create_honours_int_id(mode, tmp_path):
    kg = _new_kg(mode, tmp_path)
    kg.cypher("CREATE (:Doc {id: 42, v: 1})")
    assert kg.cypher("MATCH (n:Doc {id: 42}) RETURN n.id AS id, n.v AS v").to_list() == [{"id": 42, "v": 1}]


@pytest.mark.parametrize("mode", CREATE_MODES)
def test_create_merge_parity_with_save_reload(mode, tmp_path):
    """CREATE + MERGE produce identical results across modes; on disk they
    also survive save/reload with properties + edges intact (the disk-CREATE
    columnar write path)."""
    kg = _new_kg(mode, tmp_path)
    kg.cypher("CREATE (:Person {id: 1, name: 'Alice', age: 30})")
    kg.cypher("CREATE (:Person {id: 2, name: 'Bob', age: 25})")
    kg.cypher("MATCH (a:Person {id:1}),(b:Person {id:2}) CREATE (a)-[:KNOWS {since: 2020}]->(b)")
    kg.cypher("MERGE (:Company {id: 100, name: 'Acme'})")
    kg.cypher("MERGE (:Company {id: 100, name: 'Acme'})")  # match -> no duplicate

    def snapshot(g):
        return (
            g.cypher("MATCH (p:Person) RETURN count(p) AS c").scalar(),
            g.cypher("MATCH (c:Company) RETURN count(c) AS c").scalar(),
            g.cypher("MATCH (p:Person {id:1}) RETURN p.name AS n, p.age AS a").to_list(),
            g.cypher("MATCH (:Person)-[r:KNOWS]->(:Person) RETURN r.since AS s").scalar(),
            g.cypher("MATCH (c:Company {id:100}) RETURN c.name AS n").scalar(),
        )

    assert snapshot(kg) == (2, 1, [{"n": "Alice", "a": 30}], 2020, "Acme")

    if mode == "disk":
        path = str(tmp_path / "kg")
        kg.save(path)
        kg2 = kglite.load(path)
        # Properties (age), title (name), and edge props (since) survive the
        # round-trip — the disk-CREATE columnar write must persist them.
        assert snapshot(kg2) == (2, 1, [{"n": "Alice", "a": 30}], 2020, "Acme")


@pytest.mark.parametrize("mode", CREATE_MODES)
def test_merge_on_create_set(mode, tmp_path):
    kg = _new_kg(mode, tmp_path)
    kg.cypher("MERGE (c:Widget {id: 1}) ON CREATE SET c.tag = 'new'")
    kg.cypher("MERGE (c:Widget {id: 1}) ON CREATE SET c.tag = 'should-not-apply'")
    assert kg.cypher("MATCH (w:Widget {id:1}) RETURN w.tag AS t").scalar() == "new"
    assert kg.cypher("MATCH (w:Widget) RETURN count(w) AS c").scalar() == 1


def test_create_auto_assigns_when_no_id():
    kg = KnowledgeGraph()
    kg.cypher("CREATE (:Auto {x: 1})")
    # no provided id -> a deterministic auto-assigned UniqueId (0 for the first node)
    assert kg.cypher("MATCH (n:Auto) RETURN n.id AS id").to_list() == [{"id": 0}]


def test_create_and_add_nodes_identity_parity():
    """A node made via CREATE and one via add_nodes with the same id are
    indistinguishable: same n.id, same matchability, id is not a property."""
    g1 = KnowledgeGraph()
    g1.cypher("CREATE (:Doc {id: 's1', extra: 7})")
    g2 = KnowledgeGraph()
    g2.add_nodes(
        pd.DataFrame([{"id": "s1", "extra": 7}]),
        "Doc",
        unique_id_field="id",
        node_title_field="id",
    )
    for g in (g1, g2):
        assert g.cypher("MATCH (n:Doc) RETURN n.id AS id").to_list() == [{"id": "s1"}]
        assert g.cypher("MATCH (n:Doc {id: 's1'}) RETURN n.extra AS e").to_list() == [{"e": 7}]
    # Both expose the SAME key set — the CREATE-made node is indistinguishable
    # from the add_nodes-made one. (`keys(n)` includes the structural id/title/
    # type accessors in KGLite; the point here is parity, not their presence.)
    k1 = sorted(g1.cypher("MATCH (n:Doc) RETURN keys(n) AS k").to_list()[0]["k"])
    k2 = sorted(g2.cypher("MATCH (n:Doc) RETURN keys(n) AS k").to_list()[0]["k"])
    assert k1 == k2 and "extra" in k1


def test_create_id_survives_save_load():
    kg = KnowledgeGraph()
    kg.cypher("CREATE (:Doc {id: 'doc1', name: 'Original'})")
    with tempfile.NamedTemporaryFile(suffix=".kgl", delete=False) as f:
        path = f.name
    try:
        kg.save(path)
        loaded = kglite.load(path)
        assert loaded.cypher("MATCH (n:Doc {id: 'doc1'}) RETURN n.name AS n").to_list() == [{"n": "Original"}]
    finally:
        os.unlink(path)


def test_create_edge_by_matched_id_round_trips():
    """The kglite-docs N3 shape, but with cypher-CREATEd nodes — now works
    because CREATE honours the provided id, so the MATCH finds the nodes."""
    kg = KnowledgeGraph()
    kg.cypher("CREATE (:A {id: 'a1'}), (:B {id: 'b1'})")
    kg.cypher("MATCH (a:A {id: 'a1'}), (b:B {id: 'b1'}) CREATE (a)-[:R]->(b)")
    assert kg.cypher("MATCH (:A)-[:R]->(:B) RETURN count(*) AS n").to_list() == [{"n": 1}]


def test_unwind_id_match_above_transient_index_threshold():
    """Regression (0.11.2): `UNWIND $ids AS i MATCH (n {id:i})` must return
    every match even when the list exceeds the 64-row transient-eq-index
    activation threshold.

    The transient index (an executor optimisation for cross-MATCH equality
    joins) was being built over the `id` *virtual* — node identity, not a
    stored property — producing an empty/partial map. Every probe then missed
    and the bare point-MATCH silently dropped ALL rows once the unwound list
    crossed 64 elements. Found via cross-engine benchmark parity (kglite
    returned 0, every other engine 500). The fix skips id/title in the
    transient index; identity lookups already have their own seek path. This
    is invisible to the differential corpus (the bug is in the executor, not a
    pass — both optimiser-on/off paths returned 0), hence a golden assertion.
    """
    kg = KnowledgeGraph()
    n = 200  # well above the 64 transient-index threshold
    kg.cypher("UNWIND range(0, $n - 1) AS i CREATE (:Doc {id: i})", params={"n": n})
    ids = list(range(n))
    assert kg.cypher("UNWIND $ids AS i MATCH (m:Doc {id: i}) RETURN count(m) AS c", params={"ids": ids}).to_list() == [
        {"c": n}
    ]
    # the write path (SET) must see every match too
    assert kg.cypher(
        "UNWIND $ids AS i MATCH (m:Doc {id: i}) SET m.tag = 1 RETURN count(m) AS c",
        params={"ids": ids},
    ).to_list() == [{"c": n}]


def test_dict_and_list_of_dict_params_marshal_to_native_maps():
    """Regression (0.11.2): a Python `dict` param, and a list of dicts, must
    marshal into native `Value::Map`/`Value::List` — not `Value::Null`.

    The PyO3 param converter had no `dict` branch (fell through to `Null`) and
    flattened lists into a JSON *string*. So `UNWIND $rows AS r ... r.key` saw
    null rows and a batch-insert wrote nodes with null ids — unmatchable, so the
    follow-up SET/DELETE no-oped and the graph silently corrupted (memory/mapped
    showed phantom extra rows; disk dodged it via a different write path). Found
    via the cross-mode mutation-parity benchmark. The fix gives the converter
    native dict→Map and list/tuple→List branches.
    """
    kg = KnowledgeGraph()

    # 1. bare dict param — property access must resolve, not return null
    assert kg.cypher("WITH $m AS m RETURN m.a AS a, m.b AS b", params={"m": {"a": 1, "b": "x"}}).to_list() == [
        {"a": 1, "b": "x"}
    ]

    # 2. list-of-dicts UNWIND — each row is a real map
    rows = [{"id": 10, "nm": "a"}, {"id": 11, "nm": "b"}]
    assert kg.cypher("UNWIND $rows AS r RETURN r.id AS id, r.nm AS nm", params={"rows": rows}).to_list() == [
        {"id": 10, "nm": "a"},
        {"id": 11, "nm": "b"},
    ]

    # 3. the batch-insert shape end-to-end: CREATE from unwound dicts, then the
    #    nodes must be matchable by the id we wrote (the actual corruption path)
    kg.cypher("UNWIND $rows AS r CREATE (:Doc {id: r.id, name: r.nm})", params={"rows": rows})
    matched = kg.cypher("MATCH (d:Doc) WHERE d.id IN $ids RETURN count(d) AS c", params={"ids": [10, 11]}).scalar()
    assert matched == 2


# ─── Absent-key point lookups (P1, write-perf program) ─────────────────────


@pytest.mark.parametrize("mode", CREATE_MODES)
def test_absent_id_point_lookup_returns_empty(mode, tmp_path):
    """A point lookup on an id that does not exist returns nothing — in every
    spelling, and identically to the hit spelling that does exist.

    The anchor used to fall through to a full-type scan whenever the id index
    could not resolve the key, conflating "index not built" with "key absent".
    The scan could only ever return nothing, so this is a golden for the
    behaviour the fast path must preserve: same rows, whatever the spelling.
    """
    kg = _new_kg(mode, tmp_path)
    kg.cypher("UNWIND range(1, 50) AS i CREATE (:Doc {id: i, name: 'n' + toString(i)})")

    # inline map, literal
    assert kg.cypher("MATCH (n:Doc {id: 999}) RETURN n.name AS nm").to_list() == []
    assert kg.cypher("MATCH (n:Doc {id: 7}) RETURN n.name AS nm").to_list() == [{"nm": "n7"}]

    # inline map, parameter
    assert kg.cypher("MATCH (n:Doc {id: $i}) RETURN n.name AS nm", params={"i": 999}).to_list() == []
    assert kg.cypher("MATCH (n:Doc {id: $i}) RETURN n.name AS nm", params={"i": 7}).to_list() == [{"nm": "n7"}]

    # WHERE equality
    assert kg.cypher("MATCH (n:Doc) WHERE n.id = 999 RETURN n.name AS nm").to_list() == []
    assert kg.cypher("MATCH (n:Doc) WHERE n.id = 7 RETURN n.name AS nm").to_list() == [{"nm": "n7"}]

    # RETURN n (whole node) rather than a property
    assert kg.cypher("MATCH (n:Doc {id: 999}) RETURN n").to_list() == []
    assert len(kg.cypher("MATCH (n:Doc {id: 7}) RETURN n").to_list()) == 1

    # aggregate over the miss
    assert kg.cypher("MATCH (n:Doc {id: 999}) RETURN count(n) AS c").to_list() == [{"c": 0}]

    # untyped point lookup
    assert kg.cypher("MATCH (n {id: 999}) RETURN n.name AS nm").to_list() == []

    # a miss combined with another predicate is still empty; a hit still
    # honours the other predicate
    assert kg.cypher("MATCH (n:Doc {id: 999}) WHERE n.name = 'n7' RETURN n.name AS nm").to_list() == []
    assert kg.cypher("MATCH (n:Doc {id: 7}) WHERE n.name = 'n7' RETURN n.name AS nm").to_list() == [{"nm": "n7"}]
    assert kg.cypher("MATCH (n:Doc {id: 7}) WHERE n.name = 'zzz' RETURN n.name AS nm").to_list() == []

    # write path: SET on an absent key is a no-op, not an error
    assert kg.cypher("MATCH (n:Doc {id: 999}) SET n.tag = 1 RETURN count(n) AS c").to_list() == [{"c": 0}]
    assert kg.cypher("MATCH (n:Doc) WHERE n.tag IS NOT NULL RETURN count(n) AS c").to_list() == [{"c": 0}]


@pytest.mark.parametrize("mode", CREATE_MODES)
def test_unwind_mixed_hit_and_miss_ids_returns_exactly_the_hits(mode, tmp_path):
    """`UNWIND` over a list mixing present and absent ids returns one row per
    present id — the shape whose miss half cost 6.7 s over 16k absent ids."""
    kg = _new_kg(mode, tmp_path)
    kg.cypher("UNWIND range(1, 20) AS i CREATE (:Doc {id: i})")

    ids = [1, 500, 2, 501, 3, 502]
    assert kg.cypher("UNWIND $ids AS i MATCH (n:Doc {id: i}) RETURN n.id AS id", params={"ids": ids}).to_list() == [
        {"id": 1},
        {"id": 2},
        {"id": 3},
    ]
    assert kg.cypher(
        "UNWIND $ids AS i MATCH (n:Doc {id: i}) RETURN count(n) AS c", params={"ids": [900, 901, 902]}
    ).to_list() == [{"c": 0}]
    # IN-list anchor agrees with the point-lookup anchor
    assert kg.cypher("MATCH (n:Doc) WHERE n.id IN $ids RETURN count(n) AS c", params={"ids": ids}).to_list() == [
        {"c": 3}
    ]


def test_absent_string_id_and_alias_lookup_return_empty():
    """String ids and a user-declared id alias (`add_nodes(..., 'starId')`)
    take the same anchor: a miss is empty, a hit is unaffected."""
    kg = KnowledgeGraph()
    kg.cypher("CREATE (:Doc {id: 's1', v: 1}), (:Doc {id: 's2', v: 2})")
    assert kg.cypher("MATCH (n:Doc {id: 'nope'}) RETURN n.v AS v").to_list() == []
    assert kg.cypher("MATCH (n:Doc {id: 's2'}) RETURN n.v AS v").to_list() == [{"v": 2}]

    df = pd.DataFrame({"starId": [1, 2, 3], "title": ["a", "b", "c"]})
    kg.add_nodes(df, "Star", "starId", "title")
    assert kg.cypher("MATCH (s:Star {starId: 99}) RETURN s.title AS t").to_list() == []
    assert kg.cypher("MATCH (s:Star {starId: 2}) RETURN s.title AS t").to_list() == [{"t": "b"}]
    # and after a write invalidates/rebuilds the index
    kg.cypher("CREATE (:Star {id: 4, title: 'd'})")
    assert kg.cypher("MATCH (s:Star {starId: 99}) RETURN s.title AS t").to_list() == []
    assert kg.cypher("MATCH (s:Star {starId: 4}) RETURN s.title AS t").to_list() == [{"t": "d"}]


def test_absent_id_lookup_after_delete_and_recreate():
    """A deleted id must miss, and a recreated one must hit again — the index
    invalidation path the fast-empty now depends on."""
    kg = KnowledgeGraph()
    kg.cypher("UNWIND range(1, 10) AS i CREATE (:Doc {id: i})")
    assert kg.cypher("MATCH (n:Doc {id: 5}) RETURN count(n) AS c").to_list() == [{"c": 1}]

    kg.cypher("MATCH (n:Doc {id: 5}) DETACH DELETE n")
    assert kg.cypher("MATCH (n:Doc {id: 5}) RETURN count(n) AS c").to_list() == [{"c": 0}]
    assert kg.cypher("MATCH (n:Doc {id: 6}) RETURN count(n) AS c").to_list() == [{"c": 1}]

    kg.cypher("CREATE (:Doc {id: 5, name: 'back'})")
    assert kg.cypher("MATCH (n:Doc {id: 5}) RETURN n.name AS nm").to_list() == [{"nm": "back"}]


def test_absent_id_lookup_on_secondary_label():
    """A node carrying `:Extra` as a *secondary* label stays reachable by id:
    the queried label's id index covers primary members only, and the match
    unions the secondary-label carriers in."""
    kg = KnowledgeGraph()
    kg.cypher("CREATE (:Person:Director {id: 1, name: 'Ada'})")
    assert kg.cypher("MATCH (n:Director {id: 1}) RETURN n.name AS nm").to_list() == [{"nm": "Ada"}]
    assert kg.cypher("MATCH (n:Director {id: 2}) RETURN n.name AS nm").to_list() == []
    assert kg.cypher("MATCH (n:Person {id: 1}) RETURN n.name AS nm").to_list() == [{"nm": "Ada"}]
    assert kg.cypher("MATCH (n:Person {id: 2}) RETURN n.name AS nm").to_list() == []
