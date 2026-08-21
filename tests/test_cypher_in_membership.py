"""IN-membership semantics: one golden per evaluation site.

`x IN list` is answered at four independent predicate sites (a fifth, the
`EXISTS` fast path's inline property check, shares the matcher's), and each
one carries its own three-valued (Kleene) implementation:

* the **pattern matcher** (`PropertyMatcher::In`) — a fully-literal or
  plan-time-resolvable ``var.prop IN [...]`` gets pushed into the MATCH
  pattern by index_selection;
* ``Predicate::In`` — a list whose items are not all literals, so no set can
  be folded at plan time;
* ``Predicate::InLiteralSet`` — an all-literal list folded to a set, and the
  rewrite index_selection installs behind a pushed-down param list;
* ``Predicate::InExpression`` — an arbitrary RHS expression (a bare param, a
  list comprehension, a property holding a list).

openCypher's rules are the same at every site::

    NULL IN <anything>                     -> NULL
    x IN [..]      match present           -> true   (NULLs immaterial)
    x IN [..]      no match, list has NULL -> NULL
    x IN [..]      no match, no NULL       -> false
    x IN []                                -> false  (but NULL IN [] -> NULL)

These goldens pin the behaviour so the shared membership-set rewrite cannot
move it. They also pin the numeric coercion rule (`values_equal`): an
`Int64` property matches an integral `Float64` list element and vice versa,
a `UniqueId` (`n.id`) matches a plain integer, and a non-integral float
matches nothing integral.
"""

from __future__ import annotations

import pytest

import kglite


@pytest.fixture(scope="module")
def graph():
    g = kglite.KnowledgeGraph()
    g.cypher(
        """
        CREATE (:Item {id: 1, val: 1, txt: 'a'}),
               (:Item {id: 2, val: 2, txt: 'b'}),
               (:Item {id: 3, val: 3, txt: 'c'}),
               (:Item {id: 4, val: 4, txt: 'd'})
        """
    )
    return g


def rows(graph, query, params=None):
    return graph.cypher(query, params=params).to_list()


def one(graph, query, params=None):
    result = rows(graph, query, params)
    assert len(result) == 1, result
    return next(iter(result[0].values()))


# ──────────────────────────────────────────────────────────────────────────
# Site 1 — pattern matcher (`PropertyMatcher::In`), literal + param lists
# pushed into MATCH by index_selection.
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("clause", "params", "expected"),
    [
        # plain membership
        ("WHERE n.val IN [2, 3]", None, [2, 3]),
        ("WHERE n.val IN $vs", {"vs": [2, 3]}, [2, 3]),
        # a NULL in the list never suppresses a real match, and never
        # promotes a non-match to a match: WHERE keeps only TRUE rows, so a
        # NULL-bearing list behaves exactly like the list without the NULL.
        ("WHERE n.val IN [2, null]", None, [2]),
        ("WHERE n.val IN $vs", {"vs": [2, None]}, [2]),
        ("WHERE n.val IN [null]", None, []),
        # empty list matches nothing
        ("WHERE n.val IN []", None, []),
        ("WHERE n.val IN $vs", {"vs": []}, []),
        # a property that is absent on every node is NULL -> never TRUE
        ("WHERE n.missing IN [1, 2]", None, []),
        ("WHERE n.missing IN [null]", None, []),
        # NOT ... IN: NULL is not TRUE, so NULL-bearing lists erase the
        # complement rows too (three-valued NOT keeps NULL as NULL).
        ("WHERE NOT n.val IN [2, 3]", None, [1, 4]),
        ("WHERE NOT n.val IN [2, null]", None, []),
        # mixed-type lists: only the coercible members can match
        ("WHERE n.val IN [2.0, 'x', true]", None, [2]),
        ("WHERE n.val IN $vs", {"vs": [2.0, "x", True]}, [2]),
        ("WHERE n.val IN [2.5]", None, []),
        # strings
        ("WHERE n.txt IN ['b', 'c']", None, [2, 3]),
        ("WHERE n.txt IN ['b', null]", None, [2]),
        ("WHERE n.txt IN [2]", None, []),
        # id is stored as a UniqueId; an integer literal must still match
        ("WHERE n.id IN [2, 3]", None, [2, 3]),
        ("WHERE n.id IN $vs", {"vs": [2, 3]}, [2, 3]),
        ("WHERE n.id IN [2.0]", None, [2]),
        ("WHERE n.id IN [2.5]", None, []),
        ("WHERE n.id IN [-1]", None, []),
    ],
)
def test_matcher_site_in_semantics(graph, clause, params, expected):
    got = rows(graph, f"MATCH (n:Item) {clause} RETURN n.val AS v ORDER BY v", params)
    assert [r["v"] for r in got] == expected


# ──────────────────────────────────────────────────────────────────────────
# Site 2 — `Predicate::In`: the list carries a non-literal item, so it can
# neither fold to a set nor be pushed into the pattern.
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("clause", "expected"),
    [
        # `n.other` is a per-row expression -> stays Predicate::In
        ("WHERE n.val IN [2, n.val + 10]", [2]),
        ("WHERE n.val IN [n.val - 1]", []),
        ("WHERE n.val IN [3, n.val - 1]", [3]),
        # NULL element from a per-row expression: no match -> NULL, not false
        ("WHERE n.val IN [n.missing]", []),
        ("WHERE NOT n.val IN [n.missing]", []),
        ("WHERE n.val IN [2, n.missing]", [2]),
        ("WHERE NOT n.val IN [2, n.missing]", []),
        # NULL on the left is NULL regardless of the list
        ("WHERE n.missing IN [1, n.val]", []),
        # coercion holds for per-row lists too
        ("WHERE n.val IN [2.0, n.val + 10]", [2]),
    ],
)
def test_predicate_in_site_semantics(graph, clause, expected):
    got = rows(graph, f"MATCH (n:Item) {clause} RETURN n.val AS v ORDER BY v")
    assert [r["v"] for r in got] == expected


# ──────────────────────────────────────────────────────────────────────────
# Site 3 — `Predicate::InExpression`: an arbitrary RHS expression.
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("bind", "clause", "params", "expected"),
    [
        # a list-producing expression on the RHS
        ("[x IN [1, 2, 3] WHERE x > 1 | x]", "WHERE n.val IN lst", None, [2, 3]),
        ("[x IN [1, 2, 3] WHERE x > 1 | x + 10]", "WHERE n.val IN lst", None, []),
        # a list expression that yields NULL elements: no match -> NULL
        ("[x IN [1, 2] | null]", "WHERE n.val IN lst", None, []),
        ("[x IN [1, 2] | null]", "WHERE NOT n.val IN lst", None, []),
        # a NULL-bearing list where a real match exists -> TRUE wins
        ("[2, null]", "WHERE n.val IN lst", None, [2]),
        # NULL RHS -> NULL, never TRUE
        ("n.missing", "WHERE n.val IN lst", None, []),
        ("n.missing", "WHERE NOT n.val IN lst", None, []),
        # NULL LHS -> NULL
        ("[x IN [1, 2] | x]", "WHERE n.missing IN lst", None, []),
        # empty produced list
        ("[x IN [1, 2] WHERE x > 9 | x]", "WHERE n.val IN lst", None, []),
        # param list reached as an expression (LHS is not a bare property,
        # so index_selection cannot push it into the pattern)
        ("$vs", "WHERE n.val + 0 IN lst", {"vs": [2, 3]}, [2, 3]),
        ("$vs", "WHERE n.val + 0 IN lst", {"vs": [2, None]}, [2]),
        ("$vs", "WHERE n.val + 0 IN lst", {"vs": []}, []),
        # coercion through the expression path
        ("[2.0]", "WHERE n.val IN lst", None, [2]),
        ("[2.5]", "WHERE n.val IN lst", None, []),
    ],
)
def test_in_expression_site_semantics(graph, bind, clause, params, expected):
    got = rows(
        graph,
        f"MATCH (n:Item) WITH n, {bind} AS lst {clause} RETURN n.val AS v ORDER BY v",
        params,
    )
    assert [r["v"] for r in got] == expected


# ──────────────────────────────────────────────────────────────────────────
# Site 4 — `Predicate::InLiteralSet`: an all-literal list folded to a set,
# reached where index_selection cannot push it into a pattern (post-WITH).
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("clause", "params", "expected"),
    [
        ("WHERE v IN [2, 3]", None, [2, 3]),
        ("WHERE v IN [2, null]", None, [2]),
        ("WHERE NOT v IN [2, null]", None, []),
        ("WHERE v IN [null]", None, []),
        ("WHERE v IN []", None, []),
        ("WHERE v IN [2.0]", None, [2]),
        ("WHERE v IN [2.5]", None, []),
        ("WHERE v IN [2, 'x', true, 2.0]", None, [2]),
    ],
)
def test_in_literal_set_site_semantics(graph, clause, params, expected):
    got = rows(
        graph,
        f"MATCH (n:Item) WITH n.val AS v {clause} RETURN v ORDER BY v",
        params,
    )
    assert [r["v"] for r in got] == expected


# ──────────────────────────────────────────────────────────────────────────
# Standalone expression form — the three-valued truth table, verbatim.
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("null IN [1, 2]", None),
        ("null IN []", None),
        ("null IN [null]", None),
        ("1 IN [1, null]", True),
        ("2 IN [1, null]", None),
        ("2 IN [1, 3]", False),
        ("2 IN []", False),
        ("NOT (2 IN [1, null])", None),
        # coercion, both directions
        ("2 IN [2.0]", True),
        ("2.0 IN [2]", True),
        ("2 IN [2.5]", False),
        ("2.5 IN [2]", False),
        ("'2' IN [2]", False),
        ("true IN [1]", False),
    ],
)
def test_in_expression_truth_table(graph, expression, expected):
    assert one(graph, f"RETURN {expression} AS value") == expected


# ──────────────────────────────────────────────────────────────────────────
# The same semantics, above the length at which a list is indexed rather
# than scanned. Short lists keep the linear `values_equal` scan, so a
# coercion bug in the index is invisible to every case above — these are the
# ones that see it.
# ──────────────────────────────────────────────────────────────────────────

# Wider than the linear/indexed threshold, and disjoint from every probe.
FILLER = [-1001 - i for i in range(12)]
FILLER_STR = [f"__filler_{i}" for i in range(12)]


def pad(items, filler=None):
    return list(items) + list(FILLER if filler is None else filler)


def literal(items):
    def render(v):
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return "'" + v + "'"
        return repr(v)

    return "[" + ", ".join(render(v) for v in items) + "]"


@pytest.mark.parametrize(
    ("prop", "items", "filler", "expected"),
    [
        # coercion across the numeric family, through the index
        ("val", [2], None, [2]),
        ("val", [2.0], None, [2]),
        ("val", [2.5], None, []),
        ("id", [2], None, [2]),
        ("id", [2.0], None, [2]),
        ("id", [2.5], None, []),
        ("id", [-2], None, []),
        # strings, including the JSON single-element list form
        ("txt", ["b"], FILLER_STR, [2]),
        ("txt", ['["b"]'], FILLER_STR, [2]),
        ("txt", ["z"], FILLER_STR, []),
        # cross-type non-matches must stay non-matches
        ("val", [True], None, []),
        ("txt", [2], None, []),
        # NULL keeps its Kleene meaning at index scale
        ("val", [2, None], None, [2]),
        ("val", [None], None, []),
    ],
)
def test_indexed_list_matches_the_linear_scan(graph, prop, items, filler, expected):
    """Every long-list answer equals the short-list answer for the same items."""
    long_query = f"MATCH (n:Item) WHERE n.{prop} IN {literal(pad(items, filler))} RETURN n.val AS v ORDER BY v"
    short_query = f"MATCH (n:Item) WHERE n.{prop} IN {literal(items)} RETURN n.val AS v ORDER BY v"
    assert [r["v"] for r in rows(graph, long_query)] == expected
    assert [r["v"] for r in rows(graph, short_query)] == expected


@pytest.mark.parametrize(
    ("prop", "items", "filler", "expected"),
    [
        ("val", [2], None, [2]),
        ("val", [2.0], None, [2]),
        ("val", [2.5], None, []),
        ("id", [2], None, [2]),
        ("id", [2.0], None, [2]),
        ("txt", ["b"], FILLER_STR, [2]),
        ("val", [2, None], None, [2]),
        ("val", [None], None, []),
    ],
)
def test_indexed_param_list_matches_the_linear_scan(graph, prop, items, filler, expected):
    query = f"MATCH (n:Item) WHERE n.{prop} IN $vs RETURN n.val AS v ORDER BY v"
    assert [r["v"] for r in rows(graph, query, {"vs": pad(items, filler)})] == expected
    assert [r["v"] for r in rows(graph, query, {"vs": list(items)})] == expected


def test_indexed_post_with_list_matches_the_linear_scan(graph):
    """The post-WITH (InLiteralSet) site, at index scale."""
    for probe_items, want in [([2], [2]), ([2.0], [2]), ([2.5], []), ([2, None], [2])]:
        query = f"MATCH (n:Item) WITH n.val AS v WHERE v IN {literal(pad(probe_items))} RETURN v ORDER BY v"
        assert [r["v"] for r in rows(graph, query)] == want, query


@pytest.mark.parametrize(
    ("expression", "params", "expected"),
    [
        ("2 IN $vs", {"vs": [1, 2]}, True),
        ("2 IN $vs", {"vs": [1, None]}, None),
        ("2 IN $vs", {"vs": []}, False),
        ("2 IN $vs", {"vs": [2.0]}, True),
        ("null IN $vs", {"vs": [1]}, None),
    ],
)
def test_in_param_truth_table(graph, expression, params, expected):
    assert one(graph, f"RETURN {expression} AS value", params) == expected


# ──────────────────────────────────────────────────────────────────────────
# Multiplicity — a repeated list element does not repeat its node.
#
# The index-served `IN` anchors (`{id: IN [...]}` and `{p: IN [...]}` where
# `p` is indexed) are driven by the *list*: one index probe per element. A
# list naming the same node twice therefore used to emit that node once per
# element, so `count(n)` over `[1, 1, 2]` answered 3 where the scan path,
# every other anchor, and Neo4j answer 2. A MATCH binds each node once.
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("clause", "params"),
    [
        ("WHERE n.id IN [1, 1, 2]", None),
        ("WHERE n.id IN $vs", {"vs": [1, 1, 2]}),
        # Equal *values* are not the only way two elements reach one node:
        # the id index coerces across the numeric family, so `1` and `1.0`
        # are distinct elements resolving to the same node.
        ("WHERE n.id IN [1, 1.0, 2]", None),
        # The same element repeated many times is still one row.
        ("WHERE n.id IN [1, 1, 1, 1, 2, 2]", None),
        # A duplicated element that matches nothing changes nothing.
        ("WHERE n.id IN [1, 999, 999, 2]", None),
    ],
)
def test_duplicate_list_entries_bind_each_node_once(graph, clause, params):
    query = f"MATCH (n:Item) {clause} RETURN n.val AS v ORDER BY v"
    assert [r["v"] for r in rows(graph, query, params)] == [1, 2]
    assert one(graph, f"MATCH (n:Item) {clause} RETURN count(n) AS c", params) == 2


def test_duplicate_entries_match_the_unindexed_scan(graph):
    """The control: a property with no index filters each node once, and
    always answered 2. The anchored route must agree with it."""
    for clause, params in [
        ("WHERE n.txt IN ['a', 'a', 'b']", None),
        ("WHERE n.txt IN $vs", {"vs": ["a", "a", "b"]}),
    ]:
        assert one(graph, f"MATCH (n:Item) {clause} RETURN count(n) AS c", params) == 2


def test_duplicate_entries_keep_the_anchor_ordering(graph):
    """Dedup keeps each node's **first** occurrence, so the id anchor still
    answers in list order.

    This is not a Cypher guarantee — the query has no ORDER BY — but it is
    the order this anchor has always returned, and dropping a *later*
    duplicate rather than an earlier one is what keeps it.
    """
    got = rows(graph, "MATCH (n:Item) WHERE n.id IN [3, 1, 3, 2] RETURN n.val AS v")
    assert [r["v"] for r in got] == [3, 1, 2]


def test_duplicate_entries_bind_once_on_an_indexed_property():
    """The sibling arm: `IN` on a **non-id** property that carries a per-type
    index probes that index once per element and concatenated the answers.

    Its own graph — `create_index` would otherwise change the route the
    module-scoped fixture's other goldens take.
    """
    g = kglite.KnowledgeGraph()
    g.cypher("CREATE (:Item {id: 1, txt: 'a'}), (:Item {id: 2, txt: 'b'}), (:Item {id: 3, txt: 'c'})")
    query = "MATCH (n:Item) WHERE n.txt IN ['a', 'a', 'b'] RETURN count(n) AS c"
    assert g.cypher(query).to_list()[0]["c"] == 2
    g.create_index("Item", "txt")
    assert g.cypher(query).to_list()[0]["c"] == 2


def test_duplicate_entries_bind_once_past_the_index_threshold():
    """Long lists take `MembershipSet`'s hash index instead of a linear scan,
    and 0.11.2 fixed a *different* bare-point-lookup defect that only appeared
    past ~64 elements. Cross both boundaries with duplicates present.
    """
    g = kglite.KnowledgeGraph()
    g.cypher("UNWIND range(1, 200) AS i CREATE (:Item {id: i, val: i})")
    ids = list(range(1, 101)) * 2  # 200 elements, 100 distinct
    query = "MATCH (n:Item) WHERE n.id IN $vs RETURN count(n) AS c"
    assert g.cypher(query, params={"vs": ids}).to_list()[0]["c"] == 100
    assert g.cypher(query, params={"vs": ids}, disable_optimizer=True).to_list()[0]["c"] == 100
    got = g.cypher("MATCH (n:Item) WHERE n.id IN $vs RETURN n.val AS v ORDER BY v", params={"vs": ids}).to_list()
    assert [r["v"] for r in got] == list(range(1, 101))


def test_unwind_of_duplicate_ids_still_yields_one_row_per_element(graph):
    """The neighbouring shape that is *not* a defect: UNWIND makes the
    duplicate a driving row, so two rows are correct (Neo4j agrees). The fix
    dedups an anchor's candidate set, never the caller's rows.
    """
    assert one(graph, "UNWIND [1, 1, 2] AS x MATCH (n:Item) WHERE n.id = x RETURN count(n) AS c") == 3
    assert one(graph, "UNWIND [1, 1, 2] AS x MATCH (n:Item {id: x}) RETURN count(n) AS c") == 3
