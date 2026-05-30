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
