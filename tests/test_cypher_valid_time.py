"""The statement prefix ``FOR VALID_TIME AS OF`` and ``cypher(valid_at=...)``.

A context parses, lowers to a per-scope guard template, renders under
``EXPLAIN`` and executes under the filter resolved at its instant (the
execution goldens are in ``test_cypher_valid_time_execution.py``). Two
properties matter as much as the feature: a query without a prefix plans
exactly as before, and a declaration alone changes no plan and no answer.
"""

from __future__ import annotations

import datetime
import re

import pytest

import kglite
from tests import test_cypher_differential as _differential
from tests.test_cypher_differential import (
    DIFFERENTIAL_QUERIES,
    ORDERED_CASES,
    PASS_TRIGGER_CASES,
    _normalize,
)

# The corpus's fixtures, so an entry's fixture name resolves here too.
globals().update(
    {
        name: value
        for name, value in vars(_differential).items()
        if type(value).__name__ == "FixtureFunctionDefinition" or hasattr(value, "_pytestfixturefunction")
    }
)

AS_OF = "FOR VALID_TIME AS OF date('2006-01-01') "


@pytest.fixture
def wells():
    """`Well` and `LICENSED` are declared; `Project` holds the same bound
    properties undeclared."""
    graph = kglite.KnowledgeGraph()
    graph.cypher(
        "CREATE (w1:Well {id: 1, vf: date('2000-01-01'), vt: date('2010-01-01')}),"
        " (w2:Well {id: 2, vf: date('2005-01-01')}),"
        " (f:Project {id: 10, vf: date('2000-01-01')}),"
        " (w1)-[:LICENSED {vf: date('2000-01-01'), vt: date('2020-01-01')}]->(f)"
    ).to_list()
    graph.cypher("CALL db.temporal.declare({node: 'Well', from: 'vf', to: 'vt', convention: 'closed'})").to_list()
    graph.cypher(
        "CALL db.temporal.declare({relationship: 'LICENSED', from: 'vf', to: 'vt', convention: 'half_open'})"
    ).to_list()
    return graph


def _plan(graph, query, **kwargs):
    return [row["operation"] for row in graph.cypher(query, **kwargs).to_list()]


def test_explain_renders_the_context_in_either_order(wells):
    body = "MATCH (w:Well)-[l:LICENSED]->(f:Project) RETURN w.id, f.id"
    before = _plan(wells, f"EXPLAIN {AS_OF}{body}")
    after = _plan(wells, f"{AS_OF}EXPLAIN {body}")
    assert before == after
    assert before[0] == (
        "ValidTimeContext axis=VALID_TIME targets=(:Well [vf, vt] closed), "
        "[:LICENSED [vf, vt] half_open] instant: per execution"
    )
    # The rest of the plan is the plan of the same query under ALL, which
    # lowers to nothing.
    every = _plan(wells, f"EXPLAIN FOR VALID_TIME ALL {body}")
    assert every[0] == "ValidTimeContext axis=VALID_TIME targets=none instant: all"
    plain = every[1:]
    assert [op for op in before[1:] if not op.startswith("OptimizerPass")] == [
        op for op in plain if not op.startswith("OptimizerPass")
    ]


def test_the_template_lists_only_declared_targets(wells):
    """The declared-vs-undeclared twin: `Project` carries the same bound
    properties as `Well` but is not declared."""
    assert _plan(wells, f"EXPLAIN {AS_OF}MATCH (f:Project) RETURN f")[0] == (
        "ValidTimeContext axis=VALID_TIME targets=no declared targets instant: per execution"
    )
    assert (
        "targets=(:Well [vf, vt] closed) instant"
        in _plan(wells, f"EXPLAIN {AS_OF}MATCH (w:Well), (f:Project) RETURN w, f")[0]
    )


AT_2012 = "FOR VALID_TIME AS OF date('2012-01-01') "


@pytest.mark.parametrize(
    "query,ids",
    [
        (f"{AT_2012}MATCH (w:Well) RETURN w.id", [2]),
        (f"PROFILE {AT_2012}MATCH (w:Well) RETURN w.id", [2]),
        (f"{AT_2012}PROFILE MATCH (w:Well) RETURN w.id", [2]),
        (f"{AT_2012}MATCH (w:Well) RETURN w.id UNION MATCH (f:Project) RETURN f.id AS `w.id`", [2, 10]),
    ],
)
def test_execution_runs_under_the_filter(wells, query, ids):
    """Well 1 closed in 2010; the undeclared Project is timeless."""
    assert sorted(row["w.id"] for row in wells.cypher(query).to_list()) == ids


def test_metadata_procedures_run_under_a_context(wells):
    labels = {row["label"] for row in wells.cypher(f"{AS_OF}CALL db.labels()").to_list()}
    assert {"Well", "Project"} <= labels


@pytest.mark.parametrize(
    "query,message",
    [
        ("FOR SYSTEM_TIME AS OF date('2006-01-01') MATCH (n) RETURN n", "axis SYSTEM_TIME is not supported"),
        (f"{AS_OF}MATCH (w:Well) SET w.x = 1", "cannot write"),
        (f"{AS_OF}CALL orphan_node() YIELD node RETURN node", "procedure orphan_node"),
        (
            f"{AS_OF}MATCH (w:Well) RETURN degree(w)",
            "is not available under a valid-time context; count relationships with COUNT",
        ),
        (
            f"{AS_OF}MATCH (w:Well) RETURN outDegree(w)",
            "is not available under a valid-time context; count relationships with COUNT",
        ),
    ],
)
def test_lowering_refusals_stop_explain_too(wells, query, message):
    for text in (query, f"EXPLAIN {query}"):
        with pytest.raises(kglite.KgError, match=message):
            wells.cypher(text).to_list()


def test_no_argument_date_is_today_at_execution(wells):
    query = "FOR VALID_TIME AS OF date() MATCH (w:Well) RETURN w.id"
    assert [row["w.id"] for row in wells.cypher(query).to_list()] == [2]
    assert _plan(wells, f"EXPLAIN {query}")[0].endswith("instant: per execution")
    today = wells.cypher("RETURN date() AS d").to_list()[0]["d"]
    utc = datetime.datetime.now(datetime.timezone.utc).date()
    assert today in (utc, utc - datetime.timedelta(days=1))


@pytest.mark.parametrize(
    "pattern",
    [
        "(a:Project)-[:LICENSED*2]-(b:Project)",
        "p = shortestPath((a:Project)-[:LICENSED*]-(b:Project))",
    ],
)
def test_multi_hop_intermediates_reach_declared_labels(wells, pattern):
    """The hand-expanded twin lists `Well`; so must the multi-hop spelling."""
    expanded = _plan(wells, f"EXPLAIN {AS_OF}MATCH (a:Project)-[:LICENSED]-()-[:LICENSED]-(b:Project) RETURN a")[0]
    assert "(:Well [vf, vt] closed)" in expanded
    assert "(:Well [vf, vt] closed)" in _plan(wells, f"EXPLAIN {AS_OF}MATCH {pattern} RETURN a")[0]


def test_a_graph_without_declarations_refuses_a_context():
    graph = kglite.KnowledgeGraph()
    with pytest.raises(kglite.KgError, match="has none"):
        graph.cypher(f"EXPLAIN {AS_OF}RETURN 1 AS x").to_list()


def test_metadata_procedures_lower_under_a_context(wells):
    plan = _plan(wells, f"EXPLAIN {AS_OF}CALL db.labels()")
    assert plan[0].startswith("ValidTimeContext")


@pytest.mark.parametrize(
    "query,message",
    [
        (f"{AS_OF}{AS_OF}RETURN 1 AS x", "takes one FOR <axis> AS OF context"),
        (f"RETURN 1 AS x UNION {AS_OF}RETURN 2 AS x", "one context per statement"),
        (f"MATCH (w:Well) CALL {{ {AS_OF}MATCH (m) RETURN m }} RETURN w, m", "one context per statement"),
        ("FOR VALID_TIME AS OF w.vf MATCH (w:Well) RETURN w", "takes a constant instant"),
    ],
)
def test_prefix_syntax_errors(wells, query, message):
    with pytest.raises(kglite.CypherSyntaxError, match=message):
        wells.cypher(query)


@pytest.mark.parametrize("arm", ["EXPLAIN", "PROFILE"])
def test_union_arm_rejects_explain(arm):
    """Red before the fix: the arm parsed as a whole statement, so
    `RETURN 1 AS x UNION EXPLAIN RETURN 2 AS x` ran and returned both rows."""
    graph = kglite.KnowledgeGraph()
    with pytest.raises(kglite.CypherSyntaxError, match="must lead the statement"):
        graph.cypher(f"RETURN 1 AS x UNION {arm} RETURN 2 AS x")


@pytest.mark.parametrize(
    "valid_at,literal",
    [
        (datetime.date(2006, 1, 1), "date('2006-01-01')"),
        ("2006-01-01", "date('2006-01-01')"),
        (datetime.datetime(2006, 1, 1, 12, 30), "datetime('2006-01-01T12:30:00')"),
        ("2006-01-01T12:30:00", "datetime('2006-01-01T12:30:00')"),
    ],
)
def test_valid_at_writes_the_prefix(wells, valid_at, literal):
    body = "MATCH (w:Well) RETURN w.id"
    via_kwarg = _plan(wells, f"EXPLAIN {body}", valid_at=valid_at)
    by_hand = _plan(wells, f"FOR VALID_TIME AS OF {literal} EXPLAIN {body}")
    assert via_kwarg == by_hand
    assert via_kwarg[0].startswith("ValidTimeContext axis=VALID_TIME")
    assert sorted(row["w.id"] for row in wells.cypher(body, valid_at=valid_at).to_list()) == [1, 2]
    closed = valid_at.replace(year=2012) if not isinstance(valid_at, str) else valid_at.replace("2006", "2012")
    assert [row["w.id"] for row in wells.cypher(body, valid_at=closed).to_list()] == [2]


def test_valid_at_on_a_query_with_a_prefix_names_both(wells):
    with pytest.raises(ValueError, match=r"already has a FOR .* valid_at= adds another"):
        wells.cypher(f"{AS_OF}MATCH (w:Well) RETURN w", valid_at="2010-01-01")
    with pytest.raises(ValueError, match="valid_at"):
        wells.cypher("MATCH (w:Well) RETURN w", valid_at="next tuesday")


def test_valid_at_none_is_the_default(wells):
    query = "MATCH (w:Well) RETURN w.id AS id ORDER BY id"
    assert wells.cypher(query, valid_at=None).to_list() == wells.cypher(query).to_list()
    every = wells.cypher(f"FOR VALID_TIME ALL {query}").to_list()
    assert every == [{"id": 1}, {"id": 2}]


_DIFFERENTIAL_TRIGGERS = [
    (pass_name, case_id) for pass_name, (source, case_id) in PASS_TRIGGER_CASES.items() if source == "differential"
]
_CASES = {entry[0]: entry for entry in DIFFERENTIAL_QUERIES}
# Mirrors planner/guard.rs; the Rust walk test holds that list against PASSES.
GUARD_SAFE_PASSES = {
    "optimize_nested_queries",
    "lower_fixed_var_length_hops",
    "rewrite_count_bound_var_to_star",
    "hoist_with_where",
    "hoist_where_over_call_subquery",
    "push_where_into_match.1",
    "fold_or_to_in",
    "push_where_into_match.2",
    "extract_pushable_rel_predicates",
    "fold_pass_through_with",
    "fold_aliasing_with",
    "hoist_terminal_return_over_with_top_k",
    "narrow_unwind_source",
    "desugar_multi_match_return_aggregate",
    "reorder_match_clauses",
    "reorder_cyclic_pattern_edges",
    "optimize_pattern_start_node",
    "reorder_match_patterns",
    "push_distinct_into_match",
    "reorder_predicates_by_cost",
    "mark_disjoint_fixed_trails",
    "anchor_element_id",
    "anchor_rel_property_index",
    "push_limit_into_aggregate",
    "fuse_count_short_circuits",
    "fuse_node_scan_aggregate",
    "columnar_scan_aggregate",
    "fuse_node_scan_top_k",
    "fuse_chain_path_count",
    "fuse_match_return_aggregate",
    "fuse_match_with_aggregate",
    "fuse_match_with_aggregate_top_k",
    "fuse_order_by_top_k",
    "fuse_with_top_k",
    "fuse_vector_score_order_limit",
    "fuse_text_bm25_order_limit",
    "push_limit_into_match",
    "fuse_optional_match_aggregate",
    "fuse_spatial_join",
    "fuse_anchored_edge_count",
    "mark_fast_var_length_paths",
}


def _seed_declaration(graph):
    """A declared label nothing in the corpus names, carried as a secondary
    label by one node so every labelled scope reaches it: a context lowers
    (a graph with no declaration refuses one) and guards the plan without
    touching the query."""
    graph.cypher("CREATE (:ZzValidity {vf: date('2000-01-01'), vt: date('2001-01-01')})").to_list()
    graph.cypher("CALL db.temporal.declare({node: 'ZzValidity', from: 'vf', to: 'vt', convention: 'closed'})").to_list()
    graph.cypher("MATCH (n) WHERE NOT n:ZzValidity WITH n LIMIT 1 SET n:ZzValidity").to_list()


def test_guard_allow_list_matches_the_engine():
    assert GUARD_SAFE_PASSES <= set(kglite.cypher_pass_names())


@pytest.mark.parametrize("pass_name,case_id", _DIFFERENTIAL_TRIGGERS)
def test_a_guarded_plan_runs_only_allow_listed_passes(pass_name, case_id, request):
    """Walk every pass's trigger: under a context a denied pass never fires,
    and an allow-listed one still does."""
    _, fixture, query, params = _CASES[case_id]
    graph = request.getfixturevalue(fixture)
    _seed_declaration(graph)
    kwargs = {"params": params} if params else {}
    assert f"OptimizerPass {pass_name}" in _plan(graph, f"EXPLAIN FOR VALID_TIME ALL {query}", **kwargs)
    guarded = _plan(graph, f"EXPLAIN {AS_OF}{query}", **kwargs)
    assert guarded[0].startswith("ValidTimeContext")
    fired = {op.removeprefix("OptimizerPass ") for op in guarded if op.startswith("OptimizerPass ")}
    assert fired <= GUARD_SAFE_PASSES, fired - GUARD_SAFE_PASSES
    if pass_name in GUARD_SAFE_PASSES:
        assert pass_name in fired, guarded


# The corpus's own context entries carry their prefix; the twin is about the
# prefix-less ones.
_PREFIXLESS = [entry for entry in DIFFERENTIAL_QUERIES if not entry[2].startswith("FOR VALID_TIME")]


@pytest.mark.parametrize("name,fixture,query,params", _PREFIXLESS, ids=[e[0] for e in _PREFIXLESS])
def test_a_declaration_changes_no_plan_and_no_answer_under_all(name, fixture, query, params, request):
    """The twin: ``FOR VALID_TIME ALL`` on the seeded graph before and after one
    more declaration gives a byte-identical EXPLAIN and equal rows for every
    prefix-less corpus query. ALL lowers to nothing, whatever is declared (some
    fixtures already declare their own labels; an unprefixed statement would
    default to today there)."""
    graph = request.getfixturevalue(fixture)
    graph.cypher("CREATE (:ZzValidity {vf: date('2000-01-01'), vt: date('2001-01-01')})").to_list()
    kwargs = {"params": params} if params else {}
    order = "ordered" if name in ORDERED_CASES else "bag"

    def observe():
        plan = graph.cypher(f"EXPLAIN FOR VALID_TIME ALL {query}", **kwargs).to_list()
        rows = _normalize(graph.cypher(f"FOR VALID_TIME ALL {query}", **kwargs).to_list(), order=order)
        return plan, rows

    before = observe()
    graph.cypher("CALL db.temporal.declare({node: 'ZzValidity', from: 'vf', to: 'vt', convention: 'closed'})").to_list()
    after = observe()
    assert after[0] == before[0]
    assert after[1] == before[1]


@pytest.mark.parametrize("name,fixture,query,params", _PREFIXLESS, ids=[e[0] for e in _PREFIXLESS])
def test_all_equals_the_unprefixed_statement_on_an_undeclared_graph(name, fixture, query, params, request):
    """On a graph with no declaration the unprefixed statement gets no context
    and ``FOR VALID_TIME ALL`` is a no-op: identical rows and no echo."""
    graph = request.getfixturevalue(fixture)
    if graph.cypher("CALL db.temporal.declarations()").to_list():
        pytest.skip("the fixture declares validity; the default would apply")
    kwargs = {"params": params} if params else {}
    order = "ordered" if name in ORDERED_CASES else "bag"
    plain = graph.cypher(query, **kwargs)
    every = graph.cypher(f"FOR VALID_TIME ALL {query}", **kwargs)
    assert _normalize(every.to_list(), order=order) == _normalize(plain.to_list(), order=order)
    assert plain.diagnostics["temporal"] is None
    assert every.diagnostics["temporal"] is None


# The corpus's variable-length and shortestPath entries, prefix-less.
_VAR_LENGTH = [entry for entry in _PREFIXLESS if re.search(r"\[[^\]]*\*", entry[2]) or "hortestPath" in entry[2]]
# Refusals a context raises before it runs anything: the entry has no
# guarded form to compare.
_CONTEXT_REFUSALS = ("cannot write", "procedure", "not available under a valid-time context", "ambiguous")


def test_the_corpus_has_var_length_entries_to_sync():
    assert len(_VAR_LENGTH) >= 40, len(_VAR_LENGTH)


@pytest.mark.parametrize("name,fixture,query,params", _VAR_LENGTH, ids=[e[0] for e in _VAR_LENGTH])
def test_the_guarded_var_length_expansion_matches_the_plain_one(name, fixture, query, params, request):
    """The guarded expansions are clones of the plain ones. Under a filter
    that admits every element the query can reach — two isolated
    `ZzValidity` nodes, one valid and one not, so the guard runs without
    hiding anything reachable — every corpus variable-length and
    shortestPath entry answers as it does without the prefix. An entry the
    seeded nodes can reach (an untyped zero-hop anchor) is left out."""
    graph = request.getfixturevalue(fixture)
    if graph.cypher("CALL db.temporal.declarations()").to_list():
        pytest.skip("the fixture declares its own intervals")
    kwargs = {"params": params} if params else {}
    order = "ordered" if name in ORDERED_CASES else "bag"
    plain = _normalize(graph.cypher(query, **kwargs).to_list(), order=order)
    graph.cypher(
        "CREATE (:ZzValidity {vf: date('2000-01-01'), vt: date('2001-01-01')}),"
        " (:ZzValidity {vf: date('2000-01-01'), vt: date('2030-01-01')})"
    ).to_list()
    if _normalize(graph.cypher(query, **kwargs).to_list(), order=order) != plain:
        pytest.skip("the seeded nodes are reachable from this entry")
    graph.cypher("CALL db.temporal.declare({node: 'ZzValidity', from: 'vf', to: 'vt', convention: 'closed'})").to_list()
    try:
        guarded = graph.cypher(f"{AS_OF}{query}", **kwargs).to_list()
    except kglite.KgError as error:
        if any(reason in str(error) for reason in _CONTEXT_REFUSALS):
            pytest.skip(f"refused under a context: {error}")
        raise
    assert _normalize(guarded, order=order) == plain
