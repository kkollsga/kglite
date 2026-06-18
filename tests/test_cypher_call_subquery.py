"""CALL { } subqueries — Phase 1 (parser) + Phase 2 (validation) +
Phase 3 (executor: uncorrelated) + Phase 4 (executor: correlated).

Phase 1 ships the parser; Phase 2 adds v1 structural validation
(write / unit / UNION bodies rejected, importing-WITH restrictions,
mutation classification). Phase 3 makes the **uncorrelated** form
(``CALL { ... }`` importing nothing) executable: the body runs exactly
once and its result rows are cartesian-producted with the outer row
stream (§1.1 of ``dev_workfolder/dev-documentation/design/call-subqueries.md``). The
body sees no outer variables (§1.2 rule 1); only its RETURN columns flow
out (§1.2 rule 3); a RETURN alias colliding with an outer variable is a
compile/execution error (§1.2 rule 4).

Phase 4 makes the **correlated** form (leading importing ``WITH``)
executable: the body is planned once and executed once per outer row,
seeded with only the imported variables (preserving each import's
binding kind). Its result rows are inner-joined back to the driving
outer row (§1.1 / §1.3 — zero rows drops the outer row, an aggregating
body always returns one row so the outer row survives); a NULL imported
pattern-anchor yields the empty-match result (§1.3); importing a
variable not in the outer scope errors at execution start.
"""

import pytest

import kglite
from kglite import KnowledgeGraph


@pytest.fixture
def graph():
    g = KnowledgeGraph()
    g.cypher("CREATE (:Person {name: 'Alice'})")
    g.cypher("CREATE (:Person {name: 'Bob'})")
    g.cypher("""
        MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
        CREATE (a)-[:KNOWS]->(b)
    """)
    return g


@pytest.fixture
def friends():
    """4 people in a KNOWS web (Anna, Bo, Cy connected; Dee isolated).

    Anna KNOWS Bo, Bo KNOWS Cy; Dee has no KNOWS edges. Deterministic
    titles for ORDER BY assertions and a guaranteed zero-degree node
    (Dee) for the inner-join / aggregate-zero distinction.
    """
    g = KnowledgeGraph()
    for t in ("Anna", "Bo", "Cy", "Dee"):
        g.cypher("CREATE (:Person {title: $t})", params={"t": t})
    g.cypher("MATCH (a:Person {title:'Anna'}),(b:Person {title:'Bo'}) CREATE (a)-[:KNOWS]->(b)")
    g.cypher("MATCH (a:Person {title:'Bo'}),(b:Person {title:'Cy'}) CREATE (a)-[:KNOWS]->(b)")
    g.cypher("MATCH (a:Person {title:'Anna'}),(b:Person {title:'Cy'}) CREATE (a)-[:KNOWS]->(b)")
    return g


@pytest.fixture
def tagged():
    """2 Person + 3 Tag, deterministic titles for cartesian assertions."""
    g = KnowledgeGraph()
    for n in ("Alice", "Bob"):
        g.cypher("CREATE (:P {title: $t})", params={"t": n})
    for t in ("t1", "t2", "t3"):
        g.cypher("CREATE (:Tag {title: $t})", params={"t": t})
    return g


# ──────────────────────────────────────────────────────────────────
# Phase 3 — uncorrelated execution
# ──────────────────────────────────────────────────────────────────


class TestUncorrelatedCallSubquery:
    def test_leading_call_aggregate(self, tagged):
        """CALL { ... RETURN count(n) AS c } as the first clause → S rows."""
        rows = tagged.cypher("CALL { MATCH (n:P) RETURN count(n) AS c } RETURN c").to_list()
        assert rows == [{"c": 2}]

    def test_outer_cartesian_inner(self, tagged):
        """2 outer (P) × 3 inner (Tag) = 6 rows; only RETURN cols escape."""
        rows = tagged.cypher(
            "MATCH (p:P) CALL { MATCH (x:Tag) RETURN x.title AS t } RETURN p.title AS pt, t ORDER BY pt, t"
        ).to_list()
        assert rows == [
            {"pt": "Alice", "t": "t1"},
            {"pt": "Alice", "t": "t2"},
            {"pt": "Alice", "t": "t3"},
            {"pt": "Bob", "t": "t1"},
            {"pt": "Bob", "t": "t2"},
            {"pt": "Bob", "t": "t3"},
        ]

    def test_body_with_where_order_limit(self, tagged):
        """WHERE / ORDER BY / LIMIT inside the body are honoured."""
        rows = tagged.cypher(
            "CALL { MATCH (x:Tag) WHERE x.title <> 't2' RETURN x.title AS t ORDER BY t DESC LIMIT 1 } RETURN t"
        ).to_list()
        assert rows == [{"t": "t3"}]

    def test_multiple_return_columns(self, tagged):
        rows = tagged.cypher("CALL { MATCH (x:Tag) RETURN x.title AS t, 1 AS one ORDER BY t } RETURN t, one").to_list()
        assert rows == [
            {"t": "t1", "one": 1},
            {"t": "t2", "one": 1},
            {"t": "t3", "one": 1},
        ]

    def test_empty_subquery_result_drops_all_rows(self, tagged):
        """Cartesian with an empty (non-aggregating) subquery → zero rows."""
        rows = tagged.cypher("MATCH (p:P) CALL { MATCH (z:Nope) RETURN z.title AS t } RETURN p.title, t").to_list()
        assert rows == []

    def test_aggregating_body_runs_once(self, tagged):
        """count() body returns one row per outer row, all with the same value."""
        rows = tagged.cypher(
            "MATCH (p:P) CALL { MATCH (x:Tag) RETURN count(x) AS c } RETURN p.title AS pt, c ORDER BY pt"
        ).to_list()
        assert rows == [{"pt": "Alice", "c": 3}, {"pt": "Bob", "c": 3}]

    def test_body_executes_exactly_once_via_uuid(self, tagged):
        """randomUUID() in the body must be the SAME across all output rows
        if the body ran once (determinism probe per the design doc)."""
        rows = tagged.cypher(
            "MATCH (p:P) CALL { RETURN randomUUID() AS u } RETURN p.title AS pt, u ORDER BY pt"
        ).to_list()
        assert len(rows) == 2  # one per outer P row (cartesian with 1 inner row)
        assert len({r["u"] for r in rows}) == 1  # body ran once → one UUID

    def test_nested_uncorrelated_call(self, tagged):
        """A nested uncorrelated CALL { } inside a body (§1.4) executes."""
        rows = tagged.cypher("CALL { CALL { MATCH (x:Tag) RETURN count(x) AS c } RETURN c AS cc } RETURN cc").to_list()
        assert rows == [{"cc": 3}]

    def test_subquery_returning_multiple_rows_after_match(self, tagged):
        """R outer × S inner product where the inner returns N>1 rows."""
        rows = tagged.cypher(
            "MATCH (p:P) WHERE p.title = 'Alice' "
            "CALL { MATCH (x:Tag) RETURN x.title AS t } "
            "RETURN p.title AS pt, t ORDER BY t"
        ).to_list()
        assert rows == [
            {"pt": "Alice", "t": "t1"},
            {"pt": "Alice", "t": "t2"},
            {"pt": "Alice", "t": "t3"},
        ]


class TestUncorrelatedScoping:
    def test_body_does_not_see_outer_variables(self, tagged):
        """§1.2 rule 1 — the body sees a fresh scope. An un-imported outer
        `p` referenced inside the body is a *fresh* unbound variable, so
        `MATCH (p:Tag)` inside matches all Tags, independent of the outer p."""
        rows = tagged.cypher(
            "MATCH (p:P) WHERE p.title = 'Alice' CALL { MATCH (p:Tag) RETURN count(p) AS c } RETURN c"
        ).to_list()
        # The inner `p` is a fresh Tag node, not the outer Person — so the
        # count is the number of Tags (3), not influenced by the outer p.
        assert rows == [{"c": 3}]

    def test_return_alias_collision_with_outer_variable_errors(self, tagged):
        """§1.2 rule 4 — a subquery RETURN alias clashing with an in-scope
        outer variable is an error (Neo4j errors on shadowing)."""
        with pytest.raises(kglite.CypherExecutionError, match="already exists in"):
            tagged.cypher("MATCH (p:P) CALL { MATCH (x:Tag) RETURN x.title AS p } RETURN p")


# ──────────────────────────────────────────────────────────────────
# Phase 4 — correlated execution (per-row inner join)
# ──────────────────────────────────────────────────────────────────


class TestCorrelatedCallSubquery:
    def test_canonical_aggregate_preserves_zero_degree_rows(self, friends):
        """Per-person friend counts; aggregating body keeps the zero-degree
        rows (Cy, Dee) with c=0 (§1.3)."""
        rows = friends.cypher(
            "MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS c } "
            "RETURN p.title AS pt, c ORDER BY pt"
        ).to_list()
        assert rows == [
            {"pt": "Anna", "c": 2},
            {"pt": "Bo", "c": 1},
            {"pt": "Cy", "c": 0},
            {"pt": "Dee", "c": 0},
        ]

    def test_non_aggregating_body_drops_zero_degree_rows(self, friends):
        """Non-aggregating body → inner join: Cy and Dee (no outgoing KNOWS)
        disappear; Anna appears twice (multiplicity preserved, §1.1)."""
        rows = friends.cypher(
            "MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN f.title AS fn } "
            "RETURN p.title AS pt, fn ORDER BY pt, fn"
        ).to_list()
        assert rows == [
            {"pt": "Anna", "fn": "Bo"},
            {"pt": "Anna", "fn": "Cy"},
            {"pt": "Bo", "fn": "Cy"},
        ]

    def test_multi_import(self, friends):
        """Two imports (`WITH p, q`); the body anchors on both."""
        rows = friends.cypher(
            "MATCH (p:Person), (q:Person) WHERE p.title = 'Anna' AND q.title = 'Bo' "
            "CALL { WITH p, q MATCH (p)-[:KNOWS]->(q) RETURN count(*) AS c } "
            "RETURN p.title AS pt, q.title AS qt, c"
        ).to_list()
        # Anna KNOWS Bo → exactly one matching edge.
        assert rows == [{"pt": "Anna", "qt": "Bo", "c": 1}]

    def test_imported_projected_scalar(self, friends):
        """Import a scalar from an outer WITH (not a node) and use it in the
        body's expression — the projected value flows in unchanged."""
        rows = friends.cypher(
            "MATCH (p:Person) WITH p, p.title AS t "
            "CALL { WITH t RETURN toUpper(t) AS up } "
            "RETURN p.title AS pt, up ORDER BY pt"
        ).to_list()
        assert rows == [
            {"pt": "Anna", "up": "ANNA"},
            {"pt": "Bo", "up": "BO"},
            {"pt": "Cy", "up": "CY"},
            {"pt": "Dee", "up": "DEE"},
        ]

    def test_null_import_anchor_aggregating_keeps_row(self, friends):
        """A NULL imported pattern-anchor (§1.3): the aggregating body still
        yields one row (count = 0), so the outer row survives."""
        rows = friends.cypher(
            "MATCH (p:Person) WITH p, null AS x "
            "CALL { WITH x MATCH (x)-[:KNOWS]->(f) RETURN count(f) AS c } "
            "RETURN p.title AS pt, c ORDER BY pt"
        ).to_list()
        assert rows == [
            {"pt": "Anna", "c": 0},
            {"pt": "Bo", "c": 0},
            {"pt": "Cy", "c": 0},
            {"pt": "Dee", "c": 0},
        ]

    def test_null_import_anchor_non_aggregating_drops_row(self, friends):
        """A NULL imported pattern-anchor with a non-aggregating body yields
        zero rows, so every outer row drops (§1.3)."""
        rows = friends.cypher(
            "MATCH (p:Person) WITH p, null AS x "
            "CALL { WITH x MATCH (x)-[:KNOWS]->(f) RETURN f.title AS fn } "
            "RETURN p.title AS pt, fn"
        ).to_list()
        assert rows == []

    def test_null_import_from_optional_match(self, friends):
        """The realistic NULL-import shape: an unmatched leading OPTIONAL
        MATCH binds the node to NULL, then the correlated CALL anchors on
        it. Aggregating body → count 0, row survives."""
        rows = friends.cypher(
            "OPTIONAL MATCH (x:Nope) CALL { WITH x MATCH (x)-[:KNOWS]->(f) RETURN count(f) AS c } RETURN c"
        ).to_list()
        assert rows == [{"c": 0}]

    def test_null_import_non_anchor_scalar_keeps_row(self, friends):
        """A NULL import that is NOT a pattern anchor flows in as NULL; the
        body's expression sees it (coalesce), the row is kept."""
        rows = friends.cypher(
            "MATCH (p:Person) WITH p, null AS x "
            "CALL { WITH x RETURN coalesce(x, 'fallback') AS v } "
            "RETURN p.title AS pt, v ORDER BY pt LIMIT 1"
        ).to_list()
        assert rows == [{"pt": "Anna", "v": "fallback"}]

    def test_import_not_in_scope_errors(self, friends):
        """Importing a variable not bound in the outer scope errors at
        execution start (deferred Phase-2 check that needs the outer scope)."""
        with pytest.raises(kglite.CypherExecutionError, match="not bound in the outer scope"):
            friends.cypher("MATCH (p:Person) CALL { WITH zzz MATCH (zzz)-[:KNOWS]->(f) RETURN count(f) AS c } RETURN c")

    def test_re_returning_imported_name_collides(self, friends):
        """Re-returning an imported variable under the same name is a
        collision (§1.2 rule 4 — Neo4j errors)."""
        with pytest.raises(kglite.CypherExecutionError, match="already exists in"):
            friends.cypher("MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN p AS p } RETURN p")

    def test_correlated_inside_uncorrelated(self, friends):
        """A correlated CALL nested inside an uncorrelated CALL."""
        rows = friends.cypher(
            "CALL { MATCH (p:Person) "
            "CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS c } "
            "RETURN p.title AS pt, c } RETURN pt, c ORDER BY pt"
        ).to_list()
        assert rows == [
            {"pt": "Anna", "c": 2},
            {"pt": "Bo", "c": 1},
            {"pt": "Cy", "c": 0},
            {"pt": "Dee", "c": 0},
        ]

    def test_uncorrelated_inside_correlated(self, friends):
        """An uncorrelated CALL nested inside a correlated body — the inner
        body imports nothing and runs once per outer row."""
        rows = friends.cypher(
            "MATCH (p:Person) WHERE p.title = 'Anna' "
            "CALL { WITH p MATCH (p)-[:KNOWS]->(f) "
            "CALL { MATCH (t:Person) RETURN count(t) AS total } "
            "RETURN f.title AS fn, total } "
            "RETURN p.title AS pt, fn, total ORDER BY fn"
        ).to_list()
        # Anna knows Bo + Cy (2 friends); each carries the global Person count (4).
        assert rows == [
            {"pt": "Anna", "fn": "Bo", "total": 4},
            {"pt": "Anna", "fn": "Cy", "total": 4},
        ]

    def test_determinism_per_row_distinct_uuid(self, friends):
        """Inverse of the Phase-3 determinism probe: randomUUID() in a
        correlated body executes once PER outer row → N distinct values."""
        rows = friends.cypher(
            "MATCH (p:Person) CALL { WITH p RETURN randomUUID() AS u } RETURN p.title AS pt, u ORDER BY pt"
        ).to_list()
        assert len(rows) == 4
        assert len({r["u"] for r in rows}) == 4  # one UUID per outer row

    def test_zero_outer_rows_body_never_runs(self, friends):
        """No outer rows → the body never runs, output is empty."""
        rows = friends.cypher(
            "MATCH (p:Person) WHERE p.title = 'Nobody' "
            "CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS c } "
            "RETURN p.title AS pt, c"
        ).to_list()
        assert rows == []


@pytest.fixture
def likes():
    """3 P nodes (a, b, c) and one T node. a and b LIKE T; c LIKEs nothing.

    Drives the OPTIONAL-MATCH-miss → correlated-CALL shape: for c the
    upstream OPTIONAL MATCH misses, so the imported anchor is declared but
    absent/null on that row. a and b bind the anchor to a real T node.
    """
    g = KnowledgeGraph()
    for t in ("a", "b", "c"):
        g.cypher("CREATE (:P {nid: $t, title: $t})", params={"t": t})
    g.cypher("CREATE (:T {nid: 't1', title: 'T1'})")
    g.cypher("MATCH (p:P {nid:'a'}),(t:T {nid:'t1'}) CREATE (p)-[:LIKES]->(t)")
    g.cypher("MATCH (p:P {nid:'b'}),(t:T {nid:'t1'}) CREATE (p)-[:LIKES]->(t)")
    return g


class TestCorrelatedCallAfterOptionalMatch:
    """The OPTIONAL-MATCH-miss → correlated-CALL shape. The miss leaves the
    imported anchor declared-but-absent (null) on that row, distinct from a
    never-declared (typo'd) import. Static declaredness (the variable is
    bound by a preceding clause) gates the error; per-row seeding decides
    sentinel-vs-real-node so the body runs with the row's actual value.
    """

    def test_missed_optional_match_anchor_aggregating_keeps_row(self, likes):
        """The repro: c has no LIKES edge → x is null on that row → the
        aggregating body returns oc=0 and the row survives (Neo4j semantics,
        previously errored 'not bound in the outer scope')."""
        rows = likes.cypher(
            "MATCH (p:P {nid:'c'}) "
            "OPTIONAL MATCH (p)-[:LIKES]->(x:T) "
            "CALL { WITH x MATCH (x)<-[:LIKES]-(o) RETURN count(o) AS oc } "
            "RETURN p.title AS pt, oc"
        ).to_list()
        assert rows == [{"pt": "c", "oc": 0}]

    def test_mixed_rows_aggregating_per_row_value(self, likes):
        """a, b bind x to a real T (oc = #likers of T1 = 2); c misses → x
        null → oc=0, row kept. Per-row kind decision: real node vs sentinel."""
        rows = likes.cypher(
            "MATCH (p:P) "
            "OPTIONAL MATCH (p)-[:LIKES]->(x:T) "
            "CALL { WITH x MATCH (x)<-[:LIKES]-(o) RETURN count(o) AS oc } "
            "RETURN p.title AS pt, oc ORDER BY pt"
        ).to_list()
        assert rows == [
            {"pt": "a", "oc": 2},
            {"pt": "b", "oc": 2},
            {"pt": "c", "oc": 0},
        ]

    def test_mixed_rows_non_aggregating_drops_null_row(self, likes):
        """Non-aggregating body inner-joins: a and b (x bound) keep their
        rows; c (x null) drops entirely."""
        rows = likes.cypher(
            "MATCH (p:P) "
            "OPTIONAL MATCH (p)-[:LIKES]->(x:T) "
            "CALL { WITH x MATCH (x)<-[:LIKES]-(o) RETURN o.title AS ot } "
            "RETURN p.title AS pt, ot ORDER BY pt, ot"
        ).to_list()
        assert rows == [
            {"pt": "a", "ot": "a"},
            {"pt": "a", "ot": "b"},
            {"pt": "b", "ot": "a"},
            {"pt": "b", "ot": "b"},
        ]

    def test_truly_undeclared_import_still_errors(self, likes):
        """A name never declared by any preceding clause is a typo → error,
        even though no row carries it (same absence as a missed OPTIONAL
        MATCH). Static declaredness, not row-probing, distinguishes them."""
        with pytest.raises(kglite.CypherExecutionError, match="not bound in the outer scope"):
            likes.cypher(
                "MATCH (p:P {nid:'c'}) "
                "OPTIONAL MATCH (p)-[:LIKES]->(x:T) "
                "CALL { WITH zzz MATCH (zzz)<-[:LIKES]-(o) RETURN count(o) AS oc } "
                "RETURN oc"
            )

    def test_explicit_with_null_import_regression(self, likes):
        """`WITH null AS x` (explicit null, x IS declared) still works — the
        declared-set includes x, per-row seeding picks the sentinel."""
        rows = likes.cypher(
            "MATCH (p:P {nid:'c'}) WITH p, null AS x "
            "CALL { WITH x MATCH (x)<-[:LIKES]-(o) RETURN count(o) AS oc } "
            "RETURN p.title AS pt, oc"
        ).to_list()
        assert rows == [{"pt": "c", "oc": 0}]


# ──────────────────────────────────────────────────────────────────
# Phase 1/2 — parse errors and structural rejections (unchanged)
# ──────────────────────────────────────────────────────────────────


class TestCallSubqueryParseErrors:
    def test_missing_closing_brace(self, graph):
        with pytest.raises(kglite.CypherSyntaxError):
            graph.cypher("CALL { MATCH (n:Person) RETURN n")

    @pytest.mark.parametrize(
        "query",
        [
            "MATCH (p:Person) CALL { WITH p AS x MATCH (x) RETURN count(x) AS c } RETURN c",
            "MATCH (p:Person) CALL { WITH p.name MATCH (n) RETURN n } RETURN n",
        ],
    )
    def test_importing_with_violation_is_syntax_error(self, graph, query):
        with pytest.raises(kglite.CypherSyntaxError, match="importing WITH"):
            graph.cypher(query)


class TestCallSubqueryWriteBodyRejected:
    """v1 rejects write clauses inside a CALL { } body (deferred)."""

    @pytest.mark.parametrize(
        "query",
        [
            "MATCH (p:Person) CALL { WITH p CREATE (:Tag {name: p.name}) RETURN p } RETURN p",
            "MATCH (p:Person) CALL { WITH p SET p.seen = true RETURN p } RETURN p",
            "MATCH (p:Person) CALL { WITH p DELETE p RETURN 1 AS x } RETURN x",
            "MATCH (p:Person) CALL { WITH p REMOVE p.name RETURN p } RETURN p",
            "CALL { MERGE (:Tag {name: 'x'}) RETURN 1 AS x } RETURN x",
        ],
    )
    def test_write_in_call_rejected(self, graph, query):
        with pytest.raises(kglite.CypherSyntaxError, match="write clauses.*CALL"):
            graph.cypher(query)

    def test_write_in_call_rejected_even_on_read_only_graph(self):
        g = KnowledgeGraph()
        g.cypher("CREATE (:Person {name: 'Alice'})")
        g.read_only(True)
        with pytest.raises(kglite.CypherSyntaxError, match="write clauses.*CALL"):
            g.cypher("CALL { CREATE (:Tag {name: 'x'}) RETURN 1 AS x } RETURN x")

    def test_nested_write_in_inner_call_rejected(self, graph):
        with pytest.raises(kglite.CypherSyntaxError, match="write clauses.*CALL"):
            graph.cypher("CALL { CALL { CREATE (:Tag) RETURN 1 AS y } RETURN y } RETURN y")


class TestCallSubqueryUnitBodyRejected:
    @pytest.mark.parametrize(
        "query",
        [
            "MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]->(f) } RETURN p",
            "CALL { MATCH (n:Person) WITH n } RETURN n",
        ],
    )
    def test_unit_subquery_rejected(self, graph, query):
        with pytest.raises(kglite.CypherSyntaxError, match="must end with RETURN"):
            graph.cypher(query)


class TestCallSubqueryUnionBodyRejected:
    def test_union_in_call_rejected(self, graph):
        with pytest.raises(kglite.CypherSyntaxError, match="UNION.*inside a CALL"):
            graph.cypher(
                "CALL { MATCH (n:Person) RETURN n.name AS nm UNION MATCH (m:Person) RETURN m.name AS nm } RETURN nm"
            )


class TestCallProcedureUnaffected:
    def test_call_procedure_still_executes(self, graph):
        result = graph.cypher("CALL pagerank() YIELD node, score RETURN node, score")
        rows = result.to_list()
        assert len(rows) == 2  # Alice + Bob
        assert all("score" in row for row in rows)
