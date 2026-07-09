"""Guard: native list properties on EDGES (relationships).

The multi-rev merge core (B.2) tags every merged edge with a native
``revs: [str]`` list property and relies on ``WHERE 'v1' IN r.revs`` membership
to scope Cypher queries per revision. List props are first-class on *nodes*
(see ``test_list_properties.py``) and storage-supported on edges
(``Vec<(InternedKey, Value)>`` bincode), but the edge path had **no test at any
layer** before this file — the tested list-prop path was nodes-only.

These guards retire that risk BEFORE the merge depends on it:

(a) a list edge property round-trips through ``.kgl`` save/load,
(b) Cypher ``IN`` tests *membership* over an edge list prop (no false-positive
    substring match, and a genuine non-member returns nothing),
(c) both hold across all three storage modes (memory / mapped / disk),
(d) ``UNWIND`` over an edge list prop yields the individual elements.

Edges are created via Cypher ``CREATE`` with a list literal in the relationship
property map — the same shape the merge core writes at the Rust api layer.
"""

import pytest

import kglite

# Two edges: R{id:1} present in v1+v2, R{id:2} present in v2 only. This mirrors
# the merge core's `revs` tagging (an entity/edge carries the set of revs it
# appears in) so the membership assertions read like real rev-scoping.
_RD = "MATCH ()-[r:R]->() RETURN r.id AS id, r.revs AS revs ORDER BY id"
_EXPECTED_RD = [{"id": 1, "revs": ["v1", "v2"]}, {"id": 2, "revs": ["v2"]}]


def _make(mode: str, tmp_path) -> "kglite.KnowledgeGraph":
    if mode == "memory":
        kg = kglite.KnowledgeGraph()
    elif mode == "mapped":
        kg = kglite.KnowledgeGraph(storage="mapped")
    else:
        kg = kglite.KnowledgeGraph(storage="disk", path=str(tmp_path / "g"))
    for i in (1, 2, 3):
        kg.cypher(f"CREATE (:N {{id:{i}}})")
    kg.cypher("MATCH (a:N {id:1}),(b:N {id:2}) CREATE (a)-[:R {id:1, revs:['v1','v2']}]->(b)")
    kg.cypher("MATCH (a:N {id:2}),(b:N {id:3}) CREATE (a)-[:R {id:2, revs:['v2']}]->(b)")
    return kg


def test_edge_list_stored_as_native_list():
    kg = _make("memory", None)
    assert kg.cypher(_RD).to_dicts() == _EXPECTED_RD


def test_edge_in_is_membership_not_substring():
    kg = _make("memory", None)

    # 'v1' is a member of exactly one edge's revs.
    hit = kg.cypher("MATCH ()-[r:R]->() WHERE 'v1' IN r.revs RETURN r.id AS id ORDER BY id").to_dicts()
    assert hit == [{"id": 1}]

    # 'v2' is a member of both.
    both = kg.cypher("MATCH ()-[r:R]->() WHERE 'v2' IN r.revs RETURN r.id AS id ORDER BY id").to_dicts()
    assert both == [{"id": 1}, {"id": 2}]

    # A genuine non-member returns nothing (not a stringified-substring match).
    miss = kg.cypher("MATCH ()-[r:R]->() WHERE 'v9' IN r.revs RETURN r.id AS id").to_dicts()
    assert miss == []

    # 'v' is a substring of every element but a member of none — the whole
    # point of native lists is that this returns nothing.
    substr = kg.cypher("MATCH ()-[r:R]->() WHERE 'v' IN r.revs RETURN r.id AS id").to_dicts()
    assert substr == []


def test_unwind_over_edge_list():
    kg = _make("memory", None)
    rows = kg.cypher("MATCH ()-[r:R {id:1}]->() UNWIND r.revs AS x RETURN x ORDER BY x").to_dicts()
    assert [r["x"] for r in rows] == ["v1", "v2"]


@pytest.mark.parametrize("mode", ["memory", "mapped", "disk"])
def test_edge_list_cross_mode_live(mode, tmp_path):
    kg = _make(mode, tmp_path)
    assert kg.cypher(_RD).to_dicts() == _EXPECTED_RD
    hit = kg.cypher("MATCH ()-[r:R]->() WHERE 'v1' IN r.revs RETURN r.id AS id ORDER BY id").to_dicts()
    assert hit == [{"id": 1}]


@pytest.mark.parametrize("mode", ["memory", "mapped", "disk"])
def test_edge_list_save_reload(mode, tmp_path):
    kg = _make(mode, tmp_path)
    p = str(tmp_path / f"{mode}.kgl")
    kg.save(p)
    reloaded = kglite.load(p)
    assert reloaded.cypher(_RD).to_dicts() == _EXPECTED_RD
    # Membership still works post-reload — the round-trip preserved the list,
    # not a stringified scalar.
    hit = reloaded.cypher("MATCH ()-[r:R]->() WHERE 'v2' IN r.revs RETURN r.id AS id ORDER BY id").to_dicts()
    assert hit == [{"id": 1}, {"id": 2}]
