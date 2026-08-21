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
(AND / OR / WHERE / clause keywords) and the value-expression keywords
(CASE / WHEN / END / EXISTS) stay reserved and error clearly; the backtick
escape hatch still works.

The three value literals — NULL / TRUE / FALSE — joined the name-position set
later, per openCypher's ``SchemaName = SymbolicName | ReservedWord``: they are
names after a ``:`` or ``|`` and on the key side of a property map, and stay
literals everywhere else. They are still *not* variable names.

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


class TestKeywordNamesKeepVerbatimCase:
    """Keyword names are case-preserving (Neo4j-parity): the stored name is
    the exact source spelling, not the keyword's canonical uppercase word.
    Before 0.12.16 the parser canonicalised (`{order: 1}` stored key
    `ORDER`); a graph written by an older release may therefore carry
    uppercase keys where a lowercase source now reads the verbatim key."""

    def test_property_map_key_verbatim_through_python(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:T {order: 1})")
        props = g.cypher("MATCH (n:T) RETURN properties(n) AS p").scalar()
        assert props["order"] == 1
        assert "ORDER" not in props

    def test_set_and_read_roundtrip_verbatim(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:T {id: 1})")
        g.cypher("MATCH (n:T) SET n.contains = 5")
        assert g.cypher("MATCH (n:T) RETURN n.contains AS v").scalar() == 5
        props = g.cypher("MATCH (n:T) RETURN properties(n) AS p").scalar()
        assert props["contains"] == 5
        assert "CONTAINS" not in props
        # Mixed-case source spelling is preserved too.
        g.cypher("MATCH (n:T) SET n.Contains = 6")
        props = g.cypher("MATCH (n:T) RETURN properties(n) AS p").scalar()
        assert props["contains"] == 5
        assert props["Contains"] == 6

    def test_map_literal_key_verbatim(self):
        g = KnowledgeGraph()
        assert g.cypher("RETURN {order: 1} AS m").scalar() == {"order": 1}
        assert g.cypher("RETURN {Order: 1} AS m").scalar() == {"Order": 1}

    def test_keyword_names_are_case_sensitive(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:T {order: 1})")
        assert g.cypher("MATCH (n:T) RETURN n.order AS v").scalar() == 1
        assert g.cypher("MATCH (n:T) RETURN n.ORDER AS v").scalar() is None

    def test_rel_type_and_label_verbatim(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (a:contains)-[:contains]->(b:CONTAINS)")
        rows = g.cypher("MATCH (a)-[r:contains]->(b) RETURN type(r) AS t").to_list()
        assert rows == [{"t": "contains"}]
        assert g.cypher("MATCH (n:contains) RETURN count(n) AS c").scalar() == 1
        assert g.cypher("MATCH (n:CONTAINS) RETURN count(n) AS c").scalar() == 1

    def test_keyword_key_inside_match_pattern_map_verbatim(self):
        # The MATCH-pattern re-serializer backticks the verbatim lexeme, so
        # `{contains: 5}` filters on key `contains`, not `CONTAINS`.
        g = KnowledgeGraph()
        # One node carrying BOTH spellings as distinct keys (single CREATE —
        # the schema typo-guard rejects new keys on later CREATEs).
        g.cypher("CREATE (:Thing {contains: 5, `CONTAINS`: 7})")
        assert g.cypher("MATCH (n:Thing {contains: 5}) RETURN count(n) AS c").scalar() == 1
        assert g.cypher("MATCH (n:Thing {CONTAINS: 7}) RETURN count(n) AS c").scalar() == 1
        assert g.cypher("MATCH (n:Thing {CONTAINS: 5}) RETURN count(n) AS c").scalar() == 0

    def test_exists_subquery_pattern_keeps_verbatim_keyword_names(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (p:P {id: 1})-[:contains]->(c:C {order: 2})")
        assert g.cypher("MATCH (p:P) WHERE EXISTS { (p)-[:contains]->({order: 2}) } RETURN count(p) AS c").scalar() == 1
        assert g.cypher("MATCH (p:P) WHERE EXISTS { (p)-[:CONTAINS]->() } RETURN count(p) AS c").scalar() == 0

    def test_backticked_names_unaffected(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:T {`ORDER`: 5})")
        assert g.cypher("MATCH (n:T) RETURN n.`ORDER` AS v").scalar() == 5
        assert g.cypher("MATCH (n:T) RETURN n.order AS v").scalar() is None

    def test_keyword_alias_keeps_source_case(self):
        g = KnowledgeGraph()
        assert g.cypher("RETURN 1 AS Order").columns == ["Order"]
        assert g.cypher("RETURN 1 AS order").columns == ["order"]


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


def test_value_literal_words_are_names_in_name_positions_only():
    """TRUE / FALSE / NULL as label, rel type and property key — and still
    literals in every value position.

    The reported trap was an asymmetry, not a missing feature: ``CREATE
    (:`TRUE` {x: 1})`` succeeded while ``MATCH (n:`TRUE`)`` failed, so the
    label could be minted and never queried. Both spellings, bare and
    backticked, now work in both parsers.
    """
    g = KnowledgeGraph()
    g.cypher("CREATE (:TRUE {id: 1, null: 7, on: true})-[:FALSE]->(:Thing {id: 2})")

    assert g.cypher("MATCH (n:TRUE) RETURN count(n) AS n").scalar() == 1
    assert g.cypher("MATCH (n:`TRUE`) RETURN count(n) AS n").scalar() == 1
    assert g.cypher("MATCH ()-[:FALSE]->(t:Thing) RETURN t.id AS id").scalar() == 2
    assert g.cypher("MATCH (n:TRUE {null: 7}) RETURN n.null AS v").scalar() == 7

    # Value positions are untouched: `on: true` is a boolean, not a name.
    assert g.cypher("MATCH (n:TRUE {on: true}) RETURN n.id AS id").scalar() == 1
    assert g.cypher("MATCH (n:TRUE) WHERE n.on = true RETURN n.id AS id").scalar() == 1
    assert g.cypher("RETURN true AS t").scalar() is True


def test_value_literal_words_are_not_bare_variables_in_either_parser():
    """The symmetry that keeps the trap closed: refused by CREATE *and* MATCH."""
    g = KnowledgeGraph()
    with pytest.raises(Exception):
        g.cypher("CREATE (true:Thing {id: 1})")
    with pytest.raises(Exception):
        g.cypher("MATCH (true:Thing) RETURN 1")
    # Backticked, it is an ordinary variable in both.
    g.cypher("CREATE (`true`:Thing {id: 1})")
    assert g.cypher("MATCH (`true`:Thing) RETURN `true`.id AS id").scalar() == 1


class TestDistinctAsABareVariable:
    """`DISTINCT` bound as a pattern variable and read back inside an aggregate.

    ``kglite-java``'s ``IdentifierPolicyTest`` probes the engine rather than
    copying a word list from prose, and it lists DISTINCT among the words legal
    bare in *every* position. The variable half of that probe —

        MATCH (DISTINCT:Person) RETURN count(DISTINCT) AS c

    — read the ``DISTINCT`` inside ``count(`` as the dedup flag, leaving the
    call with zero arguments; the count arm then indexed ``args[0]`` and
    aborted the host process. DISTINCT is the flag only when an argument
    follows it.
    """

    def test_terminal_distinct_inside_an_aggregate_is_the_variable(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {id: 1})")
        # The exact Java probe. Pre-fix: PanicException, index out of bounds.
        assert g.cypher("MATCH (DISTINCT:Person) RETURN count(DISTINCT) AS c").scalar() == 1
        # ...meaning the same as counting any other bound node variable.
        assert g.cypher("MATCH (n:Person) RETURN count(n) AS c").scalar() == 1

    def test_distinct_distinct_is_the_flag_applied_to_the_variable(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {id: 1})")
        assert g.cypher("MATCH (DISTINCT:Person) RETURN count(DISTINCT DISTINCT) AS c").scalar() == 1

    def test_the_flag_still_deduplicates_and_the_name_still_does_not(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {id: 1, title: 'Ada'})")
        g.cypher("CREATE (:Person {id: 2, title: 'Ada'})")
        # Unchanged: the flag over a property deduplicates.
        assert g.cypher("MATCH (n:Person) RETURN count(DISTINCT n.title) AS c").scalar() == 1
        assert g.cypher("MATCH (n:Person) RETURN count(DISTINCT *) AS c").scalar() == 2
        assert g.cypher("MATCH (n:Person) RETURN count(*) AS c").scalar() == 2
        # The variable of the same name counts its bound rows.
        assert g.cypher("MATCH (DISTINCT:Person) RETURN count(DISTINCT) AS c").scalar() == 2

    @pytest.mark.parametrize("word", ["DISTINCT", "COUNT"])
    def test_the_java_policy_matrix_for_both_words(self, word):
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {id: 1})")
        g.cypher(f"CREATE (:{word} {{id: 2}})")
        assert g.cypher(f"MATCH (n:{word}) RETURN count(n) AS c").scalar() == 1
        assert g.cypher(f"MATCH ({word}:Person) RETURN count({word}) AS c").scalar() == 1
        # The backtick escape stays available.
        assert g.cypher(f"MATCH (`{word}`:Person) RETURN count(`{word}`) AS c").scalar() == 1

    @pytest.mark.parametrize("name", ["count", "sum", "avg", "min", "max", "collect", "median"])
    def test_a_zero_argument_aggregate_is_a_syntax_error(self, name):
        """The shape that reached the blind ``args[0]``.

        Before the fix it did not merely panic in one place: ``count()``
        answered the row count, ``min()`` answered ``True`` and ``collect()``
        and bare ``RETURN count()`` aborted the process.
        """
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {id: 1})")
        with pytest.raises(Exception, match="requires an argument"):
            g.cypher(f"MATCH (n:Person) RETURN {name}() AS c")
        with pytest.raises(Exception, match="requires an argument"):
            g.cypher(f"RETURN {name}()")
