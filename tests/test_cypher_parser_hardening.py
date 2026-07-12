"""Parser-hardening regressions: bounds, budgets, and heuristics.

Locks the fixes for six parser defects found on the openCypher contract
alignment branch:

1. Variable-length bounds — ``*5..2`` (min > max) is a parse error instead of
   a silently empty range; ``*N..`` with N above the default 10-hop ceiling
   raises the ceiling to N instead of producing the empty range (N, 10);
   bare ``*`` keeps its deliberate 10-hop runaway guard (recorded in
   tests/api-baselines/cypher-dialect.json as pattern.var_length_default_cap).
2. Recursion budget — pathologically nested expressions return a clean
   CypherSyntaxError instead of overflowing the stack and killing the
   process. The crash-shape test runs in a subprocess so a regression can
   never take the test runner down with it.
3. Subtraction heuristic — the dash-ambiguity lookahead treats ``-`` as an
   edge only for real edge continuations (``-[``, ``-->``); every other RHS
   (parameters, CASE, unary minus, literals) parses as subtraction.
4. Unary-minus/subscript precedence — ``-list[0]`` is ``-(list[0])``.
   (Golden precedence assertions also live in
   tests/test_cypher_operator_precedence.py.)
5. Unified boolean tower — EXISTS { }, label-checks, and inline pattern
   predicates work identically in expression position (RETURN/WITH/args)
   and predicate position (WHERE), because both share one parser.
6. Soft-keyword map keys — ``{order: 1}`` works in map literals and map
   projections, canonicalised to the uppercase word exactly like pattern
   property maps (KG-2).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import kglite
from kglite import KnowledgeGraph


@pytest.fixture(scope="module")
def chain() -> KnowledgeGraph:
    """Directed 12-edge chain: (:Hop {id: 0})-[:NEXT]->…->(:Hop {id: 12})."""
    g = KnowledgeGraph()
    for i in range(13):
        g.cypher("CREATE (:Hop {id: $i})", params={"i": i})
    g.cypher("MATCH (a:Hop), (b:Hop) WHERE b.id = a.id + 1 CREATE (a)-[:NEXT]->(b)")
    return g


# ============================================================================
# 1. Variable-length path bounds
# ============================================================================


def test_var_length_min_greater_than_max_is_a_parse_error(chain):
    with pytest.raises(kglite.CypherSyntaxError, match=r"minimum hop count \(5\) exceeds maximum \(2\)"):
        chain.cypher("MATCH (:Hop {id: 0})-[:NEXT*5..2]->(n) RETURN n.id")


def test_var_length_open_minimum_above_default_cap_finds_the_path(chain):
    # *11.. used to become the silently empty range (11, 10); the ceiling now
    # rises to the explicit minimum, so the 11-hop path is found.
    ids = chain.cypher("MATCH (:Hop {id: 0})-[:NEXT*11..]->(n) RETURN n.id AS id ORDER BY id").column("id")
    assert ids == [11]


def test_var_length_open_minimum_below_default_cap_keeps_the_cap(chain):
    ids = chain.cypher("MATCH (:Hop {id: 0})-[:NEXT*8..]->(n) RETURN n.id AS id ORDER BY id").column("id")
    assert ids == [8, 9, 10]  # default 10-hop ceiling still applies


def test_var_length_bare_star_stays_capped_at_ten_hops(chain):
    # Deliberate runaway guard — intentional divergence from openCypher's
    # unbounded *; see pattern.var_length_default_cap in the dialect manifest.
    assert chain.cypher("MATCH (:Hop {id: 0})-[:NEXT*]->(n) RETURN count(n) AS c").scalar() == 10


def test_var_length_parse_error_carries_position(chain):
    with pytest.raises(kglite.CypherSyntaxError, match=r"line \d+, col \d+"):
        chain.cypher("MATCH (:Hop {id: 0})-[:NEXT*5..2]->(n) RETURN n.id")


# ============================================================================
# 2. Expression recursion budget
# ============================================================================

BUDGET = 512  # keep in sync with MAX_EXPRESSION_DEPTH in parser/mod.rs


def test_deep_nesting_within_budget_parses_and_executes():
    g = KnowledgeGraph()
    # 400-deep parenthesized expression
    assert g.cypher("RETURN " + "(" * 400 + "1" + ")" * 400 + " AS x").scalar() == 1
    # 400-deep list literal survives parse -> plan -> execute -> conversion
    value = g.cypher("RETURN " + "[" * 400 + "1" + "]" * 400 + " AS x").scalar()
    depth = 0
    while isinstance(value, list):
        value = value[0]
        depth += 1
    assert (depth, value) == (400, 1)
    # 400-deep NOT chain and unary-minus chain
    assert g.cypher("RETURN " + "NOT " * 400 + "false AS x").scalar() is False
    assert g.cypher("RETURN " + "-" * 400 + "5 AS x").scalar() == 5


@pytest.mark.parametrize("opener,closer", [("(", ")"), ("[", "]")], ids=["parens", "lists"])
def test_nesting_past_budget_is_a_clean_syntax_error(opener, closer):
    g = KnowledgeGraph()
    query = "RETURN " + opener * 600 + "1" + closer * 600 + " AS x"
    with pytest.raises(kglite.CypherSyntaxError, match=f"nesting exceeds {BUDGET} levels"):
        g.cypher(query)


def test_not_chain_past_budget_is_a_clean_syntax_error():
    g = KnowledgeGraph()
    with pytest.raises(kglite.CypherSyntaxError, match=f"nesting exceeds {BUDGET} levels"):
        g.cypher("RETURN " + "NOT " * 600 + "false AS x")


@pytest.mark.parametrize("opener,closer", [("(", ")"), ("[", "]")], ids=["parens", "lists"])
def test_pathological_nesting_cannot_kill_the_process(opener, closer):
    # Run in a subprocess: before the recursion budget existed this shape
    # overflowed the stack and aborted the interpreter, so an in-process
    # assertion would take the whole test run down on regression.
    code = (
        "import kglite\n"
        "g = kglite.KnowledgeGraph()\n"
        f"q = 'RETURN ' + {opener!r} * 5000 + '1' + {closer!r} * 5000 + ' AS x'\n"
        "try:\n"
        "    g.cypher(q)\n"
        "    print('UNEXPECTED-SUCCESS')\n"
        "except kglite.CypherSyntaxError as e:\n"
        "    print('CLEAN-ERROR')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"process died (rc={proc.returncode}): {proc.stderr[-300:]}"
    assert proc.stdout.strip() == "CLEAN-ERROR", proc.stdout


# ============================================================================
# 3. Subtraction RHS shapes (dash-ambiguity heuristic)
# ============================================================================


def test_subtraction_accepts_parameter_rhs():
    g = KnowledgeGraph()
    assert g.cypher("RETURN 5 - $p AS x", params={"p": 2}).scalar() == 3


def test_subtraction_accepts_case_rhs():
    g = KnowledgeGraph()
    assert g.cypher("RETURN 10 - CASE WHEN true THEN 1 ELSE 0 END AS x").scalar() == 9


def test_subtraction_accepts_unary_minus_rhs():
    g = KnowledgeGraph()
    assert g.cypher("RETURN 5 - -3 AS x").scalar() == 8


def test_subtraction_accepts_boolean_literal_rhs_and_fails_at_evaluation_not_parse():
    # Parses fine; integer minus boolean is a value-level non-result (null),
    # not a syntax error.
    g = KnowledgeGraph()
    assert g.cypher("RETURN 5 - true AS x").scalar() is None


def test_subtraction_of_parenthesized_variables():
    g = KnowledgeGraph()
    assert g.cypher("WITH 5 AS a, 3 AS b RETURN (a) - (b) AS x").scalar() == 2


def test_dash_still_reads_as_edge_in_pattern_contexts(chain):
    # size((pattern)) and inline pattern predicates keep working: the dash
    # lookahead only yields to real edge continuations (-[ and -->).
    # (Abbreviated edges like (a)-->(b) are a pre-existing engine-wide gap —
    # the core pattern parser requires -[...]- in MATCH too — so only the
    # bracketed form is asserted here.)
    assert chain.cypher("RETURN size((:Hop {id: 0})-[:NEXT]->()) AS c").scalar() == 1
    assert chain.cypher("MATCH (n:Hop) WHERE (n)-[:NEXT]->() AND n.id > 5 RETURN count(n) AS c").scalar() == 6


# ============================================================================
# 4. Unary minus binds looser than subscript/postfix
# ============================================================================


def test_unary_minus_applies_to_indexed_element_not_list():
    g = KnowledgeGraph()
    assert g.cypher("WITH [1, 2] AS l RETURN -l[0] AS x").scalar() == -1
    assert g.cypher("RETURN -[1, 2][0] AS x").scalar() == -1
    assert g.cypher("RETURN -$p[0] AS x", params={"p": [7, 8]}).scalar() == -7
    assert g.cypher("WITH [3] AS l RETURN -head(l) AS x").scalar() == -3


# ============================================================================
# 5. Boolean capabilities are position-independent (one shared tower)
# ============================================================================


def test_exists_subquery_in_return_position(chain):
    rows = chain.cypher(
        "MATCH (n:Hop) WHERE n.id IN [0, 12] RETURN n.id AS id, EXISTS { (n)-[:NEXT]->() } AS has ORDER BY id"
    ).to_list()
    assert rows == [{"id": 0, "has": True}, {"id": 12, "has": False}]


def test_exists_subquery_as_function_argument(chain):
    got = chain.cypher("MATCH (n:Hop {id: 0}) RETURN coalesce(EXISTS { (n)-[:NEXT]->() }, false) AS x").scalar()
    assert got is True


def test_label_check_in_return_position(chain):
    rows = chain.cypher("MATCH (n:Hop {id: 0}) RETURN n:Hop AS yes, n:Missing AS no").to_list()
    assert rows == [{"yes": True, "no": False}]


def test_inline_pattern_predicate_in_return_and_with(chain):
    assert chain.cypher("MATCH (n:Hop {id: 0}) RETURN (n)-[:NEXT]->() AS x").scalar() is True
    assert chain.cypher("MATCH (n:Hop {id: 12}) WITH (n)-[:NEXT]->() AS has RETURN has").scalar() is False


def test_pattern_predicate_composes_with_xor_and_case(chain):
    # XOR / CASE keywords after an inline pattern must terminate the pattern.
    got = chain.cypher("MATCH (n:Hop) WHERE (n)-[:NEXT]->() XOR n.id = 0 RETURN count(n) AS c").scalar()
    assert got == 11  # 12 nodes with outgoing edges, id 0 flipped off by XOR
    labels = chain.cypher(
        "MATCH (n:Hop) WHERE n.id IN [0, 12] "
        "RETURN CASE WHEN (n)-[:NEXT]->() THEN 'yes' ELSE 'no' END AS x ORDER BY n.id"
    ).column("x")
    assert labels == ["yes", "no"]


def test_exists_in_where_still_works(chain):
    got = chain.cypher("MATCH (n:Hop) WHERE EXISTS { (n)-[:NEXT]->() } RETURN count(n) AS c").scalar()
    assert got == 12


def test_parenthesized_arithmetic_followed_by_operator_in_where(chain):
    # The old duplicated predicate tower returned from `( ... )` immediately
    # and choked on a trailing operator; the unified tower keeps parsing.
    got = chain.cypher("MATCH (n:Hop) WHERE (n.id + 1) * 2 > 24 RETURN count(n) AS c").scalar()
    assert got == 1


def test_legacy_exists_property_hint_still_raised(chain):
    with pytest.raises(kglite.CypherSyntaxError, match="IS NOT NULL"):
        chain.cypher("MATCH (n:Hop) WHERE exists(n.id) RETURN n")


# ============================================================================
# 6. Soft keywords as map keys
# ============================================================================


def test_map_literal_accepts_soft_keyword_keys():
    g = KnowledgeGraph()
    got = g.cypher("RETURN {order: 1, contains: 2, count: 3} AS m").scalar()
    # Soft keywords canonicalise to their uppercase word, exactly like
    # pattern property maps (KG-2); plain identifiers keep their case.
    assert got == {"ORDER": 1, "CONTAINS": 2, "count": 3}


def test_map_projection_accepts_soft_keyword_keys():
    g = KnowledgeGraph()
    g.cypher("CREATE (:T {id: 7, `ORDER`: 5})")
    assert g.cypher("MATCH (n:T) RETURN n {.order} AS m").scalar() == {"ORDER": 5}
    assert g.cypher("MATCH (n:T) RETURN n {order: n.id} AS m").scalar() == {"ORDER": 7}
