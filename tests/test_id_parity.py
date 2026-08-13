"""Cross-mode node-`id` parity (0.10.10).

The maintainer principle: the query interface must be IDENTICAL across storage
modes. For prefixed-id datasets (Wikidata `Q`-codes) the loader used to store
`id` as `String("Q42")` in memory/mapped but `UniqueId(42)` on disk, bridged by
a too-eager string→int coercion (`{id:'a1'}`→`UniqueId(1)`, a wrong-node bug).

Now the id is the **integer** in every mode (`n.id == 42`), the string form
lives in `n.nid == "Q42"`, and `{nid:'Q42'}` is a plain (indexed) string lookup.
These golden assertions lock that the SAME query returns the SAME result in
memory / mapped / disk — the conformance layer the differential corpus
(optimised-vs-naive) and parity oracles (set-equality) structurally can't give.

Run: pytest tests/test_id_parity.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kglite import KnowledgeGraph

_NT = str(Path(__file__).parent / "data" / "sample_wikidata.nt")
MODES = ("memory", "mapped", "disk")


def _load(mode: str, tmp_path) -> KnowledgeGraph:
    if mode == "memory":
        kg = KnowledgeGraph()
    elif mode == "mapped":
        kg = KnowledgeGraph(storage="mapped")
    else:
        kg = KnowledgeGraph(storage="disk", path=str(tmp_path / "g"))
    kg.load_ntriples(_NT, languages=["en"], verbose=False)
    return kg


def _one(kg: KnowledgeGraph, q: str):
    return kg.cypher(q).to_list()


@pytest.mark.parametrize("mode", MODES)
def test_id_is_integer_everywhere(mode, tmp_path):
    kg = _load(mode, tmp_path)
    assert _one(kg, "MATCH (n {nid: 'Q42'}) RETURN n.id AS id") == [{"id": 42}]


@pytest.mark.parametrize("mode", MODES)
def test_nid_is_string_everywhere(mode, tmp_path):
    kg = _load(mode, tmp_path)
    assert _one(kg, "MATCH (n {id: 42}) RETURN n.nid AS nid") == [{"nid": "Q42"}]


@pytest.mark.parametrize("mode", MODES)
def test_lookup_by_integer_id(mode, tmp_path):
    kg = _load(mode, tmp_path)
    assert _one(kg, "MATCH (n {id: 42}) RETURN n.title AS t") == [{"t": "Douglas Adams"}]


@pytest.mark.parametrize("mode", MODES)
def test_lookup_by_nid_string(mode, tmp_path):
    kg = _load(mode, tmp_path)
    assert _one(kg, "MATCH (n {nid: 'Q42'}) RETURN n.title AS t") == [{"t": "Douglas Adams"}]


@pytest.mark.parametrize("mode", MODES)
def test_edge_traversal_by_id(mode, tmp_path):
    kg = _load(mode, tmp_path)
    # Q42 -[:P27]-> Q145 (United Kingdom)
    assert _one(kg, "MATCH (n {id: 42})-[:P27]->(m) RETURN m.title AS t") == [{"t": "United Kingdom"}]
    # …and identically via nid
    assert _one(kg, "MATCH (n {nid: 'Q42'})-[:P27]->(m) RETURN m.title AS t") == [{"t": "United Kingdom"}]


@pytest.mark.parametrize("mode", MODES)
def test_string_qcode_does_not_match_id(mode, tmp_path):
    """`{id: 'Q42'}` no longer coerces — ids are integers; use nid for the string."""
    kg = _load(mode, tmp_path)
    assert _one(kg, "MATCH (n {id: 'Q42'}) RETURN n.title AS t") == []


@pytest.mark.parametrize("mode", MODES)
def test_prefix_string_false_positive_gone(mode, tmp_path):
    """The original wrong-node bug: `{id:'a1'}` must NOT match `UniqueId(1)`."""
    kg = _load(mode, tmp_path)
    assert _one(kg, "MATCH (n {id: 'a1'}) RETURN n.title AS t") == []
    assert _one(kg, "MATCH (n {id: 'x1'}) RETURN n.title AS t") == []


# ── engine-minted ids ────────────────────────────────────────────────
#
# The tests above are about how a *loaded* id is represented. These are
# about the ids the engine hands out when a `CREATE` supplies none, which
# is a different contract: uniqueness there is not opt-in, because the
# caller has asked the engine for an identity rather than declared one.
# (Caller-supplied ids stay permissive by design — CYPHER.md, "Uniqueness
# is opt-in".)


def test_auto_ids_stay_unique_across_save_and_load(tmp_path):
    """Reopening a graph must not restart the id allocator underneath it.

    The allocator was ``Value::UniqueId(node_bound())``, an index-space
    bound rather than a counter: after a load whose index space still held
    the holes left by earlier deletes, ``node_bound()`` stopped advancing
    as new nodes refilled them, so three consecutive ``CREATE``s were all
    handed the *same* id. Duplicate ids are then invisible to
    ``MATCH (n {id: …})`` (one node per id) and are merged outright by WAL
    replay, so this is silent corruption in both directions.
    """
    path = tmp_path / "g.kgl"
    g = KnowledgeGraph()
    for i in range(6):
        g.cypher("CREATE (:T {tag: 'a%d'})" % i)
    g.cypher("MATCH (n:T) WHERE n.tag IN ['a1','a2'] DELETE n")
    g.save(str(path))

    import kglite

    g2 = kglite.load(str(path))
    for i in range(3):
        g2.cypher("CREATE (:T {tag: 'c%d'})" % i)

    ids = [d["id"] for d in g2.cypher("MATCH (n:T) RETURN n.id AS id").to_dicts()]
    assert len(set(ids)) == len(ids), f"engine minted a duplicate id: {sorted(ids)}"


def test_auto_id_does_not_collide_with_a_caller_supplied_id(tmp_path):
    """A caller-supplied id raises the allocator's high-water mark, so the
    engine cannot later mint the same value. Without this, loading one row
    with a sparse id (``5``) and then creating nodes walks the counter up
    onto it."""
    g = KnowledgeGraph()
    g.cypher("CREATE (:T {id: 3, tag: 'explicit'})")
    for i in range(5):
        g.cypher("CREATE (:T {tag: 'auto%d'})" % i)

    ids = [d["id"] for d in g.cypher("MATCH (n:T) RETURN n.id AS id").to_dicts()]
    assert len(set(ids)) == len(ids), f"auto id collided with a caller id: {sorted(ids)}"
