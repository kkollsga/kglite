"""Correctness tests for the Cypher plan cache (keyed on graph_id + version).

The cache reuses a fully-optimized plan for a param-less query against an
unchanged graph. These tests pin the three properties that make it sound:

1. A cache hit returns the same result as the first (freshly-planned) call.
2. A mutation bumps the graph version, invalidating the cache so a re-run
   reflects the new state (no stale plan).
3. Plans never leak across graphs (the graph_id component of the key).
"""

from __future__ import annotations

import pandas as pd
import pytest

import kglite


def _items(ids):
    return pd.DataFrame({"nid": list(ids), "value": [float(i) for i in ids]})


def test_cache_hit_matches_fresh_plan():
    g = kglite.KnowledgeGraph()
    g.add_nodes(_items(range(10)), "Item", "nid")
    q = "MATCH (n:Item) WHERE n.value > 3 RETURN count(n) AS c"
    first = g.cypher(q).to_dicts()  # miss → plan + cache
    second = g.cypher(q).to_dicts()  # hit
    third = g.cypher(q).to_dicts()  # hit
    assert first == second == third
    assert first[0]["c"] == 6  # ids 4..9


def test_mutation_invalidates_cache():
    g = kglite.KnowledgeGraph()
    g.add_nodes(_items(range(5)), "Item", "nid")
    q = "MATCH (n:Item) RETURN count(n) AS c"
    assert g.cypher(q).to_dicts()[0]["c"] == 5  # plans + caches at version V
    assert g.cypher(q).to_dicts()[0]["c"] == 5  # cache hit
    # Mutate — must bump version and invalidate the cached plan.
    g.add_nodes(_items(range(5, 12)), "Item", "nid")
    assert g.cypher(q).to_dicts()[0]["c"] == 12  # fresh plan reflects new nodes


def test_cache_invalidated_by_cypher_write():
    g = kglite.KnowledgeGraph()
    g.add_nodes(_items(range(3)), "Item", "nid")
    q = "MATCH (n:Item) RETURN count(n) AS c"
    assert g.cypher(q).to_dicts()[0]["c"] == 3
    g.cypher(q)  # warm the cache
    g.cypher("CREATE (:Item {nid: 99, value: 99.0})")  # write → version bump
    assert g.cypher(q).to_dicts()[0]["c"] == 4  # reflects the CREATE


def test_no_cross_graph_leakage():
    # Two independent graphs running the identical query text must each get
    # their own result — the graph_id key component prevents collisions even
    # when both sit at the same version.
    a = kglite.KnowledgeGraph()
    a.add_nodes(_items(range(4)), "Item", "nid")
    b = kglite.KnowledgeGraph()
    b.add_nodes(_items(range(20)), "Item", "nid")
    q = "MATCH (n:Item) RETURN count(n) AS c"
    assert a.cypher(q).to_dicts()[0]["c"] == 4
    assert b.cypher(q).to_dicts()[0]["c"] == 20
    # Re-run interleaved (both now potentially cached) — still isolated.
    assert b.cypher(q).to_dicts()[0]["c"] == 20
    assert a.cypher(q).to_dicts()[0]["c"] == 4


def test_divergent_explicit_copies_do_not_share_schema_plans():
    # Explicit copies start at the same version. Advance each exactly once but
    # with a different schema, then warm a schema-dependent plan on only one.
    # Reusing that plan in the other copy would bypass validation entirely.
    original = kglite.KnowledgeGraph()
    original.add_nodes(_items([1]), "Item", "nid")
    copied = original.copy()

    original.add_nodes(
        pd.DataFrame({"nid": [2], "value": [2.0], "original_only": [True]}),
        "Item",
        "nid",
    )
    copied.add_nodes(
        pd.DataFrame({"nid": [3], "value": [3.0], "copy_only": [True]}),
        "Item",
        "nid",
    )

    query = "MATCH (n:Item {original_only: true}) RETURN count(n) AS c"
    assert original.cypher(query).to_dicts()[0]["c"] == 1
    with pytest.raises(kglite.SchemaError, match="original_only"):
        copied.cypher(query)
