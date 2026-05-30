"""KG-2 regression: reserved keywords usable as names.

Reported by kglite-docs (2026-05-30, on 0.10.9): `CONTAINS` (and other
reserved keywords) could not be used as a relationship type, node label, or
property key — the tokenizer classified them as operators before any context
was known, so `CREATE (s)-[:CONTAINS]->(c)` raised a syntax error.

Fix: a "soft keyword" pass. The safe keyword subset (operator / comparison /
sort / set / mutation words like CONTAINS / IN / STARTS / ORDER / MERGE) is
accepted as a NAME in every name-position — relationship types, node labels,
property keys, and property access — across MATCH / CREATE / MERGE / SET /
REMOVE / WHERE and EXISTS subqueries. Structurally load-bearing words
(AND / OR / WHERE / clause keywords) and value keywords (NULL / TRUE / CASE …)
stay reserved and error clearly; the backtick escape hatch still works.

Run: pytest tests/test_cypher_keyword_names.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from kglite import KnowledgeGraph


def test_contains_as_relationship_type_create_and_match():
    g = KnowledgeGraph()
    # The report's intent: CONTAINS usable as a rel type in CREATE + MATCH.
    # Use an inline-edge CREATE so the test exercises rel-type parsing only,
    # not the separate reserved-`id` round-trip behaviour of cypher CREATE.
    g.cypher("CREATE (s:SourceDoc)-[:CONTAINS]->(c:Chunk)")
    fwd = g.cypher("MATCH (s:SourceDoc)-[:CONTAINS]->(c:Chunk) RETURN count(*) AS n").to_list()
    assert fwd == [{"n": 1}]
    # Reverse arrow sees the same edge.
    rev = g.cypher("MATCH (c:Chunk)<-[:CONTAINS]-(s:SourceDoc) RETURN count(*) AS n").to_list()
    assert rev == [{"n": 1}]


def test_contains_as_node_label():
    g = KnowledgeGraph()
    g.cypher("CREATE (n:CONTAINS {id: 1})")
    assert g.cypher("MATCH (n:CONTAINS) RETURN count(n) AS n").to_list() == [{"n": 1}]
    # WHERE label-predicate form too.
    g.cypher("CREATE (m:Other {id: 2})")
    rows = g.cypher("MATCH (n) WHERE n:CONTAINS RETURN count(n) AS n").to_list()
    assert rows == [{"n": 1}]


def test_keyword_as_property_key_create_match_access_set():
    g = KnowledgeGraph()
    g.cypher("CREATE (n:Thing {contains: 5, order: 2})")
    # inline-map filter on a keyword key
    assert g.cypher("MATCH (n:Thing {contains: 5}) RETURN n.contains AS v").to_list() == [{"v": 5}]
    # property access in RETURN + WHERE
    assert g.cypher("MATCH (n:Thing) RETURN n.contains AS v").to_list() == [{"v": 5}]
    assert g.cypher("MATCH (n:Thing) WHERE n.order = 2 RETURN n.order AS v").to_list() == [{"v": 2}]
    # SET a keyword-named property
    assert g.cypher("MATCH (n:Thing) SET n.contains = 9 RETURN n.contains AS v").to_list() == [{"v": 9}]


def test_keyword_relationship_type_in_exists_subquery_parses():
    """`[:CONTAINS]` inside an EXISTS subquery parses identically to a normal
    rel type (the EXISTS re-serializer accepts the soft keyword). Asserted by
    parity with a non-keyword type on the same graph."""
    g = KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame([{"id": 1}, {"id": 2}]),
        "P",
        unique_id_field="id",
        node_title_field="id",
    )
    g.add_connections(
        pd.DataFrame([{"s": 1, "t": 2}]),
        "CONTAINS",
        source_type="P",
        source_id_field="s",
        target_type="P",
        target_id_field="t",
    )
    rows = g.cypher("MATCH (p:P) WHERE EXISTS { (p)-[:CONTAINS]->() } RETURN count(p) AS n").to_list()
    assert rows == [{"n": 1}]


def test_several_safe_keywords_as_rel_types():
    g = KnowledgeGraph()
    g.cypher("CREATE (a:N {id: 1}), (b:N {id: 2})")
    for kw in ("IN", "STARTS", "ENDS", "ORDER", "MERGE", "DELETE"):
        g.cypher(f"MATCH (a:N {{id: 1}}), (b:N {{id: 2}}) CREATE (a)-[:{kw}]->(b)")
        n = g.cypher(f"MATCH ()-[:{kw}]->() RETURN count(*) AS n").to_list()[0]["n"]
        assert n == 1, f"keyword {kw} should be usable as a rel type"


def test_reserved_words_still_error_with_backtick_escape():
    """Load-bearing keywords stay reserved as names (clear error), but the
    backtick escape hatch keeps working."""
    g = KnowledgeGraph()
    # `where` is excluded from the soft set — must error, not silently misparse.
    with pytest.raises(Exception):
        g.cypher("CREATE (n:Q {where: 1})")
    # Backtick escape works for the reserved word.
    g.cypher("CREATE (q:Q {`where`: 7})")
    assert g.cypher("MATCH (n:Q) RETURN n.`where` AS v").to_list() == [{"v": 7}]
