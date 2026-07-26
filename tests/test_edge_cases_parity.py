"""Phase 10 edge-case parity tests.

Extends Phase 0's deferred edge-case catalogue to cover every storage
backend. Each test builds one graph per mode (memory / mapped / disk),
runs the same queries, and asserts cross-mode agreement.

Scope is the *outer envelope* of the public API under stress — empty
graphs, single-edge graphs, long chains, Unicode, type promotion,
null/NaN parity, and large result sets. Phase 2 already covered the
mutation-level matrix; Phase 10 extends to the observable-query surface.

Run: pytest -m parity tests/test_edge_cases_parity.py
"""

from __future__ import annotations

import math
import sys

import pandas as pd
import pytest

from kglite import KnowledgeGraph
from kglite import load as kglite_load
from tests.conftest import requires_getrusage

pytestmark = pytest.mark.parity

STORAGE_MODES = ("memory", "mapped", "disk")


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


def _kg_for(mode: str, tmp_path) -> KnowledgeGraph:
    path = str(tmp_path / f"kg_{mode}") if mode == "disk" else None
    return _new_kg(mode, path=path)


def _rows(result) -> list[dict]:
    try:
        rows = [dict(r) for r in result]
    except TypeError:
        rows = list(result)
    return sorted(rows, key=lambda r: repr(sorted(r.items())))


# ─── 1. Zero-node graph ─────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_zero_node_describe_schema_find_cypher(mode, tmp_path):
    kg = _kg_for(mode, tmp_path)

    describe = kg.describe()
    assert 'nodes="0"' in describe
    assert 'edges="0"' in describe
    assert "<types>" in describe

    schema = kg.schema()
    assert schema["node_types"] == {}
    assert schema["connection_types"] == {}
    assert schema["node_count"] == 0
    assert schema["edge_count"] == 0

    assert list(kg.find("anything")) == []
    assert list(kg.cypher("MATCH (n) RETURN n")) == []
    assert list(kg.cypher("MATCH (a)-[r]->(b) RETURN a, r, b")) == []


# ─── 2. Single node + single (self-)edge ────────────────────────────────────


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_single_node_single_edge(mode, tmp_path):
    kg = _kg_for(mode, tmp_path)
    kg.add_nodes(pd.DataFrame({"id": [1], "name": ["Solo"]}), "Person", "id", "name")
    kg.add_connections(pd.DataFrame({"s": [1], "t": [1]}), "LOOP", "Person", "s", "Person", "t")

    rows = _rows(kg.cypher("MATCH (a)-[r]->(b) RETURN a.name AS a, b.name AS b"))
    assert rows == [{"a": "Solo", "b": "Solo"}]
    counts = _rows(kg.cypher("MATCH (n) RETURN count(n) AS c"))
    assert counts == [{"c": 1}]


# ─── 3. Long-chain traversal ────────────────────────────────────────────────


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_long_chain_traversal_1000_hops(mode, tmp_path):
    """1,000-hop chain: cypher count terminates, returns 1,000."""
    kg = _kg_for(mode, tmp_path)
    n = 1001  # 1,000 hops between 1,001 nodes
    kg.add_nodes(
        pd.DataFrame({"id": list(range(n)), "name": [f"P{i}" for i in range(n)]}),
        "Person",
        "id",
        "name",
    )
    kg.add_connections(
        pd.DataFrame(
            {
                "s": list(range(n - 1)),
                "t": list(range(1, n)),
                "type": ["KNOWS"] * (n - 1),
            }
        ),
        "KNOWS",
        "Person",
        "s",
        "Person",
        "t",
    )
    rows = _rows(kg.cypher("MATCH (a:Person {id: 0})-[*..1000]->(b) RETURN count(b) AS c"))
    assert rows == [{"c": 1000}]


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_long_chain_traversal_path_materialization_100_hops(mode, tmp_path):
    """100-hop chain: variable-length path returns the right number of
    individual rows (one per matched terminal node), not just `count(b)`.
    Closes the test-coverage gap flagged by 0.9.0 readiness Cluster 7 —
    `count(b)` was being short-circuited by the planner, so the
    1,000-hop test above didn't actually exercise enumeration.

    Capped at 100 hops here because path-materialisation is O(depth)
    per row vs O(1) for the count short-circuit; 1,000 hops would
    timeout on disk."""
    kg = _kg_for(mode, tmp_path)
    n = 101  # 100 hops between 101 nodes
    kg.add_nodes(
        pd.DataFrame({"id": list(range(n)), "name": [f"P{i}" for i in range(n)]}),
        "Person",
        "id",
        "name",
    )
    kg.add_connections(
        pd.DataFrame(
            {
                "s": list(range(n - 1)),
                "t": list(range(1, n)),
                "type": ["KNOWS"] * (n - 1),
            }
        ),
        "KNOWS",
        "Person",
        "s",
        "Person",
        "t",
    )
    # Per-row enumeration — one row per (a, b) match. With a 100-hop
    # cap, b ranges over P1..P100 (100 distinct ends). The parity
    # helper `_rows()` re-sorts by repr() so we assert on the SET of
    # bids (size + membership), not on ordering — `ORDER BY bid` in
    # Cypher returns numeric order, but the harness re-orders
    # lexicographically.
    raw = list(kg.cypher("MATCH (a:Person {id: 0})-[*..100]->(b) RETURN b.id AS bid"))
    bids = sorted(r["bid"] for r in raw)
    assert bids == list(range(1, n)), f"{mode}: missing or extra paths"


# ─── 4. Unicode round-trip ──────────────────────────────────────────────────


UNICODE_NAMES = [
    "Alice",  # BMP ASCII
    "café",  # BMP combining mark
    "𝐀lice",  # Astral (mathematical bold A)
    "👨\u200d👩\u200d👧",  # Emoji ZWJ family sequence
    "مرحبا",  # RTL Arabic
    "",  # empty string
    "🇳🇴",  # Flag sequence (regional indicators)
]


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_unicode_roundtrip(mode, tmp_path):
    kg = _kg_for(mode, tmp_path)
    kg.add_nodes(
        pd.DataFrame({"id": list(range(len(UNICODE_NAMES))), "name": UNICODE_NAMES}),
        "Person",
        "id",
        "name",
    )
    got = list(kg.cypher("MATCH (n:Person) RETURN n.id AS id, n.name AS name ORDER BY n.id"))
    assert [r["name"] for r in got] == UNICODE_NAMES


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_unicode_save_load_roundtrip(mode, tmp_path):
    """Unicode survives .kgl save/load for all backends."""
    kg = _kg_for(mode, tmp_path)
    kg.add_nodes(
        pd.DataFrame({"id": list(range(len(UNICODE_NAMES))), "name": UNICODE_NAMES}),
        "Person",
        "id",
        "name",
    )
    path = str(tmp_path / f"save_{mode}.kgl")
    kg.save(path)
    reloaded = kglite_load(path)
    got = list(reloaded.cypher("MATCH (n:Person) RETURN n.id AS id, n.name AS name ORDER BY n.id"))
    assert [r["name"] for r in got] == UNICODE_NAMES


# ─── 5. Property-type promotion ladder ──────────────────────────────────────


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_type_promotion_int_float_string(mode, tmp_path):
    """Int → int+float → int+float+string on the same property column.

    Each stage is observable through cypher; no data from earlier stages
    is lost. Phase 5 reconciled this across backends; the parity test
    pins the contract.
    """
    kg = _kg_for(mode, tmp_path)
    # Stage 1: ints only
    kg.add_nodes(
        pd.DataFrame({"id": [1, 2], "name": ["A", "B"], "value": [10, 20]}),
        "Person",
        "id",
        "name",
    )
    assert _rows(kg.cypher("MATCH (n:Person) RETURN n.value AS v")) == _rows([{"v": 10}, {"v": 20}])

    # Stage 2: floats added → property column promoted
    kg.add_nodes(
        pd.DataFrame({"id": [3, 4], "name": ["C", "D"], "value": [1.5, 2.5]}),
        "Person",
        "id",
        "name",
    )
    stage2 = _rows(kg.cypher("MATCH (n:Person) RETURN n.value AS v"))
    assert stage2 == _rows([{"v": 10}, {"v": 20}, {"v": 1.5}, {"v": 2.5}])

    # Stage 3: strings added → column holds mixed types
    kg.add_nodes(
        pd.DataFrame({"id": [5, 6], "name": ["E", "F"], "value": ["x", "y"]}),
        "Person",
        "id",
        "name",
    )
    stage3 = _rows(kg.cypher("MATCH (n:Person) RETURN n.value AS v"))
    assert stage3 == _rows([{"v": 10}, {"v": 20}, {"v": 1.5}, {"v": 2.5}, {"v": "x"}, {"v": "y"}])


# ─── 6. Null / NaN / empty-string parity ────────────────────────────────────


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_null_nan_empty_string_properties(mode, tmp_path):
    """None, NaN, and '' round-trip consistently; None/NaN collapse to null."""
    kg = _kg_for(mode, tmp_path)
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["A", "B", "C", "D"],
            "note": ["hello", None, float("nan"), ""],
        }
    )
    kg.add_nodes(df, "Person", "id", "name")

    rows = list(kg.cypher("MATCH (n:Person) RETURN n.id AS id, n.note AS note ORDER BY n.id"))
    assert len(rows) == 4
    assert rows[0]["note"] == "hello"
    # None and NaN both collapse to null (returned as None in Python).
    assert rows[1]["note"] is None
    assert rows[2]["note"] is None
    # Empty string is distinct from null — preserved.
    assert rows[3]["note"] == ""

    # find on None/NaN/empty doesn't crash.
    assert isinstance(kg.find(""), list)

    # schema still reports the node count correctly.
    schema = kg.schema()
    assert schema["node_types"]["Person"]["count"] == 4


# ─── 7. Large-result cypher (100k rows) ─────────────────────────────────────


@requires_getrusage
@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_large_cypher_result_100k_rows(mode, tmp_path):
    """100k rows materialise cleanly across all backends; RSS bounded."""
    import gc
    import resource

    gc.collect()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    kg = _kg_for(mode, tmp_path)
    n = 100_000
    kg.add_nodes(
        pd.DataFrame({"id": list(range(n)), "name": [f"P{i}" for i in range(n)]}),
        "Person",
        "id",
        "name",
    )

    rows = list(kg.cypher("MATCH (n:Person) RETURN n.id AS id, n.name AS name"))
    assert len(rows) == n
    # Spot-check content, not full equality (would OOM the assertion error).
    ids = {r["id"] for r in rows}
    assert ids == set(range(n))

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_delta_bytes = rss_after - rss_before
    # macOS reports bytes; Linux reports KiB. Probe measured ≤175 MB on memory.
    # `sys.platform` rather than `os.uname()`, which does not exist on Windows.
    if sys.platform == "darwin":
        rss_delta_mb = rss_delta_bytes / (1024 * 1024)
    else:
        rss_delta_mb = rss_delta_bytes / 1024
    assert rss_delta_mb < 300, f"100k-row RETURN used {rss_delta_mb:.1f} MB (budget 300)"


# ─── 8. NaN math invariants survive cypher ──────────────────────────────────


@pytest.mark.parametrize("mode", STORAGE_MODES)
def test_nan_not_equal_itself(mode, tmp_path):
    """NaN → null in cypher; WHERE n.score IS NULL captures it."""
    kg = _kg_for(mode, tmp_path)
    kg.add_nodes(
        pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["A", "B", "C"],
                "score": [1.0, float("nan"), None],
            }
        ),
        "Person",
        "id",
        "name",
    )
    null_rows = _rows(kg.cypher("MATCH (n:Person) WHERE n.score IS NULL RETURN n.id AS id"))
    # Both NaN and explicit None must be observed as null.
    assert null_rows == _rows([{"id": 2}, {"id": 3}])
    non_null_rows = _rows(kg.cypher("MATCH (n:Person) WHERE n.score IS NOT NULL RETURN n.id AS id"))
    assert non_null_rows == _rows([{"id": 1}])

    # Cross-mode sanity on math: ensure no silent NaN leakage as real number.
    for r in kg.cypher("MATCH (n:Person) RETURN n.score AS s"):
        s = r["s"]
        assert s is None or (isinstance(s, float) and not math.isnan(s))
