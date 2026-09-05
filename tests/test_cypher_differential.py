"""Differential test harness for the Cypher optimizer pipeline.

Every query in DIFFERENTIAL_QUERIES is run twice: once with the
optimizer pipeline enabled (the default), once with `disable_optimizer
=True` (every pass skipped). We assert both produce identical row sets
after normalization.

This is the regression mechanism for **silent correctness failures**
(passes that drop or duplicate rows). Historical bugs in this class —
0.8.27 LIMIT pushdown returning fewer rows than asked, 0.8.30
startNode(r) returning wrong endpoints — would all have failed the
appropriate row-equality assertion.

It does NOT catch:

- **Gate misses** (a fusion pass bails when it could fuse): both
  paths produce the same result, just slower. Pinned instead by the
  PASS_TRIGGER_CASES plan-shape assertions below.
- **Execution semantic bugs** that exist in both fast and slow paths
  (rare but real, e.g. 0.8.30 startNode(r) was actually present in both
  paths). Needs absolute goldens — several live below — or cross-mode
  parity.

When fixing a future silent-correctness bug, **add the bug's triggering
query to DIFFERENTIAL_QUERIES** so the regression is permanent.
"""

from __future__ import annotations

import pytest

import kglite

# Each entry is `(name, fixture, query, params)`. The corpus aims to
# exercise:
#
# 1. One query per registered optimizer pass (so each pass's trigger
#    shape is in the corpus by design).
# 2. Historical bug shapes from CHANGELOG entries (0.8.27 +).
# 3. Edge cases that have surprised optimizers in the past: LIMIT 0,
#    OPTIONAL with no match, ORDER BY ties, DISTINCT, parameterized,
#    multi-MATCH chains.
#
# Spatial and embedder-driven trigger oracles live in the specialized suite.
# Vector ordering regressions below use a small registered-vector fixture too,
# so their exact trigger shapes remain available to the pass bisector.
#: The fused shape itself, shared by the clean and the stale fixtures so those
#: two entries differ only in the index state they run against.
TEXT_BM25_TOP_K = "MATCH (d:Doc) RETURN d.id AS id, text_bm25(d, 'body', 'alpha beta') AS s ORDER BY s DESC LIMIT 3"

DIFFERENTIAL_QUERIES: list[tuple[str, str, str, dict | None]] = [
    (
        "vector_whole_type_exact_entry",
        "vector_index_entry_graph",
        "MATCH (d:Doc) RETURN d.id AS id, vector_score(d, 'summary_emb', [1.0,0.0], "
        "{exact:true}) AS s ORDER BY s DESC LIMIT 3",
        None,
    ),
    (
        "vector_whole_type_exact_second_score",
        "vector_index_entry_graph",
        "MATCH (d:Doc) RETURN d.id AS id, vector_score(d, 'summary_emb', [1.0,0.0], "
        "{exact:true}) AS first, vector_score(d, 'summary_emb', [0.0,1.0], {exact:true}) AS "
        "second ORDER BY second DESC LIMIT 2",
        None,
    ),
    (
        "vector_whole_type_index_entry",
        "vector_index_entry_graph",
        "MATCH (d:Doc) RETURN d.id AS id, vector_score(d, 'summary_emb', [1.0,0.0]) AS s ORDER BY s DESC LIMIT 3",
        None,
    ),
    (
        "vector_score_ascending_nulls",
        "vector_order_graph",
        "MATCH (d:Doc) RETURN d.id AS id, vector_score(d, 'summary_emb', [1.0,0.0]) AS s ORDER BY s ASC LIMIT 2",
        None,
    ),
    (
        "vector_score_descending_nulls",
        "vector_order_graph",
        "MATCH (d:Doc) RETURN d.id AS id, vector_score(d, 'summary_emb', [1.0,0.0]) AS s ORDER BY s DESC LIMIT 2",
        None,
    ),
    (
        "vector_score_explicit_nulls_last",
        "vector_order_graph",
        "MATCH (d:Doc) RETURN d.id AS id, vector_score(d, 'summary_emb', [1.0,0.0]) AS s "
        "ORDER BY s DESC NULLS LAST LIMIT 4",
        None,
    ),
    (
        "text_score_ascending_nulls",
        "vector_order_graph",
        "MATCH (d:Doc) RETURN d.id AS id, text_score(d, 'summary', [1.0,0.0]) AS s ORDER BY s ASC LIMIT 2",
        None,
    ),
    (
        "text_score_descending_nulls",
        "vector_order_graph",
        "MATCH (d:Doc) RETURN d.id AS id, text_score(d, 'summary', [1.0,0.0]) AS s ORDER BY s DESC LIMIT 2",
        None,
    ),
    ("simple_match", "small_graph", "MATCH (p:Person) RETURN p.name AS n", None),
    ("simple_match_param", "small_graph", "MATCH (p:Person) WHERE p.age > $min RETURN p.name AS n", {"min": 30}),
    # Dynamic label / relationship type: the parameter is bound before the
    # optimizer runs, so both paths must plan and answer exactly as the
    # literal spelling does — that equivalence is what makes it safe for
    # every pass to keep reading a plain label.
    ("dynamic_label", "small_graph", "MATCH (p:$label) RETURN p.name AS n", {"label": "Person"}),
    (
        "dynamic_label_with_pushdown",
        "small_graph",
        "MATCH (p:$label) WHERE p.age > $min RETURN p.name AS n ORDER BY n",
        {"label": "Person", "min": 30},
    ),
    (
        "dynamic_relationship_type",
        "social_graph",
        "MATCH (a:Person)-[:$type]->(b:Person) RETURN a.name AS a, b.name AS b",
        {"type": "KNOWS"},
    ),
    (
        "dynamic_label_count_fusion",
        "social_graph",
        "MATCH (p:$label) RETURN count(p) AS n",
        {"label": "Person"},
    ),
    ("count_all_typed", "social_graph", "MATCH (p:Person) RETURN count(p) AS n", None),
    ("count_all_untyped", "social_graph", "MATCH (n) RETURN count(n) AS n", None),
    (
        "count_subquery_where_shape",
        "social_graph",
        "RETURN COUNT { (p:Person) WHERE p.age > 30 } AS n",
        None,
    ),
    (
        "count_subquery_cross_join_shape",
        "small_graph",
        "RETURN COUNT { (:Person), (:Person) } AS n",
        None,
    ),
    ("distinct_property", "social_graph", "MATCH (p:Person) RETURN DISTINCT p.city AS c", None),
    ("budget_unwind_shape", "small_graph", "UNWIND [1, 2, 3] AS x RETURN x", None),
    # ── narrow_unwind_source trigger shapes ──
    # The pass lets UNWIND take a dead source list OUT of the row instead of
    # cloning it into every expanded row (the fix for n rows x n-element list
    # quadratic memory). Each shape below is a decision the pass has to get
    # right; a divergence between the optimised and unoptimised paths means it
    # dropped a binding something still reads.
    (
        # Fires: `ns` is dead after the UNWIND.
        "narrow_unwind_source_dead",
        "small_graph",
        "MATCH (p:Person) WITH collect(p.age) AS ns UNWIND ns AS m RETURN m ORDER BY m",
        None,
    ),
    (
        # Bails: `ns` is still read by a later RETURN item.
        "narrow_unwind_source_live",
        "small_graph",
        "MATCH (p:Person) WITH collect(p.age) AS ns UNWIND ns AS m RETURN m, size(ns) AS c ORDER BY m",
        None,
    ),
    (
        # Bails: `RETURN *` names no variable in the AST — the executor expands
        # it from the runtime row, so the source binding is still observable.
        "narrow_unwind_source_return_star",
        "small_graph",
        "MATCH (p:Person) WITH collect(p.age) AS ns UNWIND ns AS m RETURN * ORDER BY m",
        None,
    ),
    (
        # Bails: the second UNWIND re-reads the same list.
        "narrow_unwind_source_double",
        "small_graph",
        "MATCH (p:Person) WITH collect(p.age) AS ns UNWIND ns AS a UNWIND ns AS b RETURN a, b ORDER BY a, b",
        None,
    ),
    (
        # Fires, then the alias is re-aggregated — pins that the moved list is
        # not needed to rebuild an equivalent collection downstream.
        "narrow_unwind_source_recollect",
        "small_graph",
        "MATCH (p:Person) WITH collect(p.age) AS ns UNWIND ns AS m WITH m WHERE m > 0 RETURN collect(m) AS back",
        None,
    ),
    (
        "budget_union_all_shape",
        "small_graph",
        "RETURN 1 AS x UNION ALL RETURN 2 AS x",
        None,
    ),
    (
        "budget_correlated_call_shape",
        "small_graph",
        "UNWIND [1, 2] AS x CALL { WITH x UNWIND [10, 20] AS y RETURN y } RETURN x, y",
        None,
    ),
    (
        "range_i64_terminal_shape",
        "small_graph",
        "RETURN range($start, $end, $step) AS r",
        {"start": -(2**63), "end": -(2**63) + 1, "step": 1},
    ),
    (
        "checked_calendar_shift_shape",
        "small_graph",
        "RETURN add_years(date('2024-02-29'), 1) AS d",
        None,
    ),
    (
        "duration_scale_shape",
        "small_graph",
        "WITH duration({months: 2, days: 3}) * 3 AS d RETURN d.months AS m, d.days AS days",
        None,
    ),
    (
        "boolean_expression_unknown_shape",
        "small_graph",
        "RETURN true OR false AND null AS value",
        None,
    ),
    (
        "membership_unknown_shape",
        "small_graph",
        "RETURN 2 IN [1, null] AS value",
        None,
    ),
    (
        "quantifier_unknown_shape",
        "small_graph",
        "RETURN single(x IN [true, null] WHERE x) AS value",
        None,
    ),
    (
        "list_addition_shape",
        "small_graph",
        "RETURN 0 + [1, 2] + 3 AS value",
        None,
    ),
    # Machine-verified trigger shapes for passes whose older comment-only
    # corpus entries did not actually make the pass fire.
    (
        "trigger_push_limit_into_aggregate",
        "social_graph",
        "MATCH (p:Person) RETURN p.city AS city, count(*) AS n LIMIT 2",
        None,
    ),
    (
        "trigger_push_limit_into_aggregate_with",
        "social_graph",
        "MATCH (p:Person) WITH p.city AS city, count(*) AS n LIMIT 2 RETURN city, n",
        None,
    ),
    ("trigger_anchored_edge_count", "social_graph", "MATCH ({id: 1})-[:KNOWS]->(p) RETURN count(*) AS n", None),
    (
        "trigger_anchored_edge_count_reverse",
        "social_graph",
        "MATCH (p)<-[:KNOWS]-({id: 1}) RETURN count(*) AS n",
        None,
    ),
    ("trigger_count_short_circuit", "social_graph", "MATCH (p:Person) RETURN count(*) AS n", None),
    (
        "count_all_edges_untyped",
        "social_graph",
        "MATCH ()-[r]->() RETURN count(r) AS n",
        None,
    ),
    (
        "property_grouping_duplicate_values",
        "social_graph",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.city AS city, count(b) AS n ORDER BY n DESC LIMIT 10",
        None,
    ),
    (
        "property_grouping_missing_values",
        "social_graph",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.email AS email, count(b) AS n ORDER BY n DESC LIMIT 30",
        None,
    ),
    (
        "property_grouping_target_value",
        "social_graph",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN b.city AS city, count(a) AS n ORDER BY n DESC LIMIT 10",
        None,
    ),
    (
        "property_grouping_other_endpoint_filter",
        "social_graph",
        "MATCH (a:Person {city: 'Oslo'})-[:KNOWS]->(b:Person) "
        "RETURN b.city AS city, count(a) AS n ORDER BY n DESC LIMIT 10",
        None,
    ),
    (
        "property_grouping_relationship_filter",
        "social_graph",
        "MATCH (a:Person)-[r:KNOWS]->(b:Person) WHERE r.since >= 2010 "
        "RETURN b.city AS city, count(r) AS n ORDER BY n DESC LIMIT 10",
        None,
    ),
    (
        "node_grouping_other_endpoint_filter",
        "social_graph",
        "MATCH (a:Person {city: 'Oslo'})-[:KNOWS]->(b:Person) RETURN b, count(a) AS n",
        None,
    ),
    (
        "node_grouping_relationship_filter",
        "social_graph",
        "MATCH (a:Person)-[r:KNOWS]->(b:Person) WHERE r.since >= 2010 RETURN b, count(r) AS n",
        None,
    ),
    (
        "trigger_match_return_aggregate",
        "social_graph",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, count(b) AS n",
        None,
    ),
    (
        "trigger_match_with_aggregate",
        "social_graph",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) WITH a, count(DISTINCT b) AS friends RETURN a, friends",
        None,
    ),
    # ── count(DISTINCT <edge var>) over parallel edges ──
    # The fused DISTINCT path dedups peer NodeIndices, which is not edge
    # identity. `fuse_match_return_aggregate` used to accept the edge variable
    # into that path, so two parallel 1→2 edges counted 1 instead of 2. Both
    # entry points (RETURN-form and WITH-form) reached it.
    (
        "count_distinct_edge_var_parallel",
        "parallel_edge_cycle_graph",
        "MATCH (a:N)-[r:R]->(b:N) RETURN a, count(DISTINCT r) AS c",
        None,
    ),
    (
        "count_distinct_edge_var_parallel_with",
        "parallel_edge_cycle_graph",
        "MATCH (a:N)-[r:R]->(b:N) WITH a, count(DISTINCT r) AS c RETURN a.name AS a, c",
        None,
    ),
    # ── group cap vs a post-projection filter ──
    # `push_limit_into_aggregate` stamped `group_limit_hint` on a WITH whose
    # inline WHERE runs *after* the projection, so the cap froze the group set
    # before the filter could reject the groups that fail it: 2 rows where 5
    # qualified and the LIMIT had room for all 5.
    (
        "with_filtered_aggregate_limit",
        "uneven_group_graph",
        "MATCH (n:T) WITH n.k AS k, collect(n.id) AS ids WHERE size(ids) > 1 LIMIT 5 RETURN k, size(ids) AS c",
        None,
    ),
    (
        "trigger_match_with_top_k",
        "social_graph",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) WITH a, count(b) AS n RETURN a.name AS name, n ORDER BY n DESC LIMIT 3",
        None,
    ),
    (
        "trigger_match_with_top_k_ascending",
        "social_graph",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) WITH a, count(b) AS n RETURN a.name AS name, n ORDER BY n ASC LIMIT 3",
        None,
    ),
    ("trigger_node_scan_aggregate", "social_graph", "MATCH (p:Person) RETURN sum(p.age) AS total", None),
    (
        "fused_property_node_scan_aggregate",
        "social_graph",
        "MATCH (p:Person {city: 'Oslo'}) RETURN p.city AS city, count(*) AS n",
        None,
    ),
    (
        "trigger_node_scan_top_k",
        "social_graph",
        "MATCH (p:Person) RETURN p.name AS name ORDER BY p.age DESC LIMIT 3",
        None,
    ),
    (
        "fused_property_node_scan_top_k",
        "social_graph",
        "MATCH (p:Person {city: 'Oslo'}) RETURN p.name AS name, p.age AS age ORDER BY age DESC LIMIT 2",
        None,
    ),
    # ── fused node-scan compiled property routes (executor/scan_eval.rs) ──
    # The fused scans evaluate group keys, sort keys, aggregate arguments and
    # the surviving WHERE through a plan compiled once per scan. Each shape
    # below is one of the branches that plan has to reproduce: a soft alias, an
    # identity alias, an unpushable string comparison over a nullable column,
    # the three-valued NOT, a text predicate behind its retained safety net, a
    # cross-type scan, and compiled arithmetic.
    (
        "scan_route_soft_alias_group_key",
        "social_graph",
        "MATCH (p:Person) RETURN p.name AS name, count(*) AS n ORDER BY name LIMIT 5",
        None,
    ),
    (
        "scan_route_identity_alias_group_key",
        "social_graph",
        "MATCH (p:Person) RETURN p.id AS i, count(*) AS n ORDER BY i LIMIT 5",
        None,
    ),
    (
        "scan_route_label_soft_alias",
        "multi_label_graph",
        "MATCH (n) RETURN n.label AS l, count(*) AS n ORDER BY l",
        None,
    ),
    (
        "scan_route_not_equals_nullable_string",
        "social_graph",
        "MATCH (p:Person) WHERE p.email <> 'nobody@example.com' RETURN count(*) AS n",
        None,
    ),
    (
        "scan_route_negated_text_over_nullable",
        "social_graph",
        "MATCH (p:Person) WHERE NOT (p.email CONTAINS '@') RETURN count(*) AS n",
        None,
    ),
    (
        "scan_route_text_predicate_safety_net",
        "social_graph",
        "MATCH (p:Person) WHERE p.name STARTS WITH 'Person_1' RETURN p.name AS name ORDER BY name",
        None,
    ),
    (
        "scan_route_or_combined_string_and_numeric",
        "social_graph",
        "MATCH (p:Person) WHERE p.city < 'P' OR p.age > 38 RETURN count(*) AS n",
        None,
    ),
    (
        "scan_route_cross_type_untyped_scan",
        "social_graph",
        "MATCH (n) RETURN n.name AS name, count(*) AS c ORDER BY name LIMIT 5",
        None,
    ),
    (
        "scan_route_compiled_arithmetic_aggregate",
        "social_graph",
        "MATCH (p:Person) RETURN sum(p.age * 2 - 1) AS total, avg(p.age / 2) AS half",
        None,
    ),
    (
        "scan_route_compiled_arithmetic_sort_key",
        "social_graph",
        "MATCH (p:Person) RETURN p.name AS name ORDER BY p.age * -1, p.name LIMIT 4",
        None,
    ),
    (
        "scan_route_folded_constant_aggregate",
        "social_graph",
        "MATCH (p:Person) RETURN sum(1 + 2 + 3) AS total",
        None,
    ),
    ("trigger_generic_top_k", "small_graph", "UNWIND [3, 1, 2] AS x RETURN x ORDER BY x LIMIT 2", None),
    # Mixed-type sort key. Two things here are visible to *this* harness even
    # though it compares row sets, not row order: before the total order the
    # unlimited form aborted the query outright (an intransitive comparator
    # trips `sort_by`'s totality check from 21 rows up), and the LIMIT form
    # let the bounded top-K heap *select* a different set of rows than the
    # full sort. Row ordering itself is pinned absolutely in
    # tests/test_cypher_mixed_type_ordering.py.
    (
        "trigger_mixed_type_sort_key",
        "small_graph",
        "UNWIND [3, 'k3', 1, 'k1', 2, 'k2', 6, 'k6', 4, 'k4', 5, 'k5', 9, 'k9', 7, 'k7', "
        "8, 'k8', 12, 'k12', 10, 'k10', 11, 'k11'] AS x RETURN x ORDER BY x DESC",
        None,
    ),
    (
        "trigger_mixed_type_sort_key_top_k",
        "small_graph",
        "UNWIND [3, 'k3', 1, 'k1', 2, 'k2', 6, 'k6', 4, 'k4', 5, 'k5', 9, 'k9', 7, 'k7', "
        "8, 'k8', 12, 'k12', 10, 'k10', 11, 'k11'] AS x RETURN x ORDER BY x ASC LIMIT 5",
        None,
    ),
    (
        "trigger_predicate_reorder",
        "social_graph",
        "MATCH (p) WHERE EXISTS((p)-[:KNOWS]->()) AND p:Person RETURN p.title AS title",
        None,
    ),
    (
        "trigger_predicate_reorder_or",
        "social_graph",
        "MATCH (p) WHERE EXISTS((p)-[:KNOWS]->()) OR p:Company RETURN p.title AS title",
        None,
    ),
    (
        # Relationship alternation inside EXISTS { } — rejected by the parser
        # until the subquery re-serializer learned `|`.
        "exists_subquery_rel_alternation",
        "social_graph",
        "MATCH (p:Person) WHERE EXISTS { (p)-[:KNOWS|WORKS_AT]->() } RETURN p.title AS title",
        None,
    ),
    (
        "exists_subquery_rel_alternation_negated",
        "social_graph",
        "MATCH (p:Person) WHERE NOT EXISTS { (p)-[:KNOWS|WORKS_AT]->(:Company) } RETURN p.title AS title",
        None,
    ),
    # ── push_where_into_match ──
    ("where_eq", "social_graph", "MATCH (p:Person) WHERE p.city = 'Oslo' RETURN p.name AS n", None),
    ("where_gt", "social_graph", "MATCH (p:Person) WHERE p.age > 30 RETURN p.name AS n", None),
    ("where_and", "social_graph", "MATCH (p:Person) WHERE p.age > 30 AND p.city = 'Bergen' RETURN p.name AS n", None),
    (
        "where_inline_equality_collision",
        "social_graph",
        "MATCH (p:Person {city: 'Oslo'}) WHERE p.city = 'Bergen' AND size(p.name) > 0 RETURN p.name AS n",
        None,
    ),
    (
        "where_inline_prefix_collision",
        "social_graph",
        "MATCH (p:Person {name: 'Person_1'}) WHERE p.name STARTS WITH 'Nope' AND size(p.name) > 0 RETURN p.name AS n",
        None,
    ),
    (
        "where_inline_in_collision",
        "social_graph",
        "MATCH (p:Person {city: 'Oslo'}) WHERE p.city IN ['Bergen'] AND size(p.name) > 0 RETURN p.name AS n",
        None,
    ),
    (
        "where_inline_range_collision",
        "social_graph",
        "MATCH (p:Person {age: 30}) WHERE p.age > 31 AND size(p.name) > 0 RETURN p.name AS n",
        None,
    ),
    (
        "where_same_direction_bound_collision",
        "social_graph",
        "MATCH (p:Person) WHERE p.age > 35 AND p.age > 38 AND size(p.name) > 0 RETURN p.name AS n",
        None,
    ),
    # ── fold_or_to_in ──
    (
        "or_chain_to_in",
        "social_graph",
        "MATCH (p:Person) WHERE p.city = 'Oslo' OR p.city = 'Bergen' OR p.city = 'Stavanger' RETURN p.name AS n",
        None,
    ),
    (
        "or_chain_reversed_literals",
        "social_graph",
        "MATCH (p:Person) WHERE 'Oslo' = p.city OR 'Bergen' = p.city RETURN p.name AS n",
        None,
    ),
    # ── extract_pushable_rel_predicates ──
    (
        "rel_property_filter",
        "social_graph",
        "MATCH (p:Person)-[r:KNOWS]->(q:Person) WHERE r.since > 2017 RETURN p.name AS p, q.name AS q",
        None,
    ),
    (
        "rel_missing_property_not_equals",
        "social_graph",
        "MATCH (p:Person)-[r:KNOWS]->(q:Person) WHERE NOT (r.missing_tag = 'foo') RETURN p.name AS p, q.name AS q",
        None,
    ),
    (
        "rel_null_not_equals_under_not",
        "social_graph",
        "MATCH (p:Person)-[r:KNOWS]->(q:Person) WHERE NOT (r.missing_tag <> 'foo') RETURN p.name AS p, q.name AS q",
        None,
    ),
    (
        "rel_unknown_nested_boolean",
        "social_graph",
        "MATCH (p:Person)-[r:KNOWS]->(q:Person) "
        "WHERE NOT (r.missing_tag = 'foo' AND r.since > 0) RETURN p.name AS p, q.name AS q",
        None,
    ),
    (
        "rel_contains_param_two_hop",
        "social_graph",
        "MATCH (p:Person)-[r:KNOWS]->(q:Person)-[:KNOWS]->(z:Person) "
        "WHERE r.tag CONTAINS $needle RETURN DISTINCT z.name AS n",
        {"needle": "knows_1"},
    ),
    (
        "rel_ends_with_param_two_hop",
        "social_graph",
        "MATCH (p:Person)-[r:KNOWS]->(q:Person)-[:KNOWS]->(z:Person) "
        "WHERE r.tag ENDS WITH $suffix RETURN DISTINCT z.name AS n",
        {"suffix": "_1"},
    ),
    (
        "rel_equality_param",
        "social_graph",
        "MATCH (p:Person)-[r:KNOWS]->(q:Person) WHERE r.since = $year RETURN count(r) AS n",
        {"year": 2016},
    ),
    (
        "rel_not_contains_nullable",
        "social_graph",
        "MATCH (p)-[r:KNOWS]->(q) WHERE NOT (r.tag CONTAINS 'never') RETURN count(r) AS n",
        None,
    ),
    # ── fold_pass_through_with ──
    (
        "pass_through_with",
        "social_graph",
        "MATCH (p:Person) WITH p MATCH (p)-[:KNOWS]->(q:Person) RETURN p.name AS p, q.name AS q",
        None,
    ),
    # ── desugar_multi_match_return_aggregate ──
    # Regression test for the bug found by this harness on first run:
    # `MATCH (p) MATCH (c) RETURN p.city, count(c)` was over-finely
    # grouped (per-p) when the user wrote a per-property aggregation.
    # Fix: WITH groups by the user-specified RETURN expressions, not
    # the source variable. See `desugar_multi_match_return_aggregate`
    # in `simplification.rs`.
    (
        "multi_match_group_agg",
        "social_graph",
        "MATCH (p:Person) MATCH (c:Company) RETURN p.city AS city, count(c) AS n",
        None,
    ),
    (
        "multi_match_two_property_group",
        "social_graph",
        "MATCH (p:Person) MATCH (c:Company) RETURN p.city AS city, p.age AS age, count(c) AS n",
        None,
    ),
    # ── reorder_match_clauses + optimize_pattern_start_node ──
    (
        "two_match_chains",
        "social_graph",
        "MATCH (p:Person)-[:WORKS_AT]->(c:Company) MATCH (p)-[:KNOWS]->(q:Person) "
        "RETURN p.name AS p, c.name AS c, q.name AS q",
        None,
    ),
    (
        "later_shared_id_anchor",
        "social_graph",
        "MATCH (p:Person)-[:WORKS_AT]->(c:Company) MATCH (p)-[:KNOWS]->(q:Person {id: 2}) "
        "RETURN p.name AS p, c.name AS c, q.name AS q",
        None,
    ),
    (
        "anchored_three_hop",
        "social_graph",
        "MATCH (a:Person {person_id: 1})-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) "
        "RETURN a.name AS a, b.name AS b, c.name AS c",
        None,
    ),
    # ── NDV-based selectivity (Tier 0) ──
    # Both ends carry a non-indexed equality on `city`; the optimizer now
    # estimates selectivity via per-(type,property) distinct-value counts and
    # may reverse the pattern to start from the rarer city. Optimised vs naive
    # must return the same rows regardless of which end is chosen as start.
    (
        "ndv_two_end_city_eq",
        "social_graph",
        "MATCH (a:Person {city: 'Oslo'})-[:KNOWS]->(b:Person {city: 'Bergen'}) RETURN a.name AS a, b.name AS b",
        None,
    ),
    # ── cyclic pattern (matcher target_hint fast path) ──
    # `a` reappears at the end → the closing segment is a bound-target check,
    # not a full expansion. Optimised vs naive must agree on the cycle count.
    (
        "knows_triangle_cycle",
        "social_graph",
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)-[:KNOWS]->(a) RETURN count(*) AS n",
        None,
    ),
    # ── relationship-type alternation `[:A|B]` ──
    #
    # `EdgePattern` stores an alternation in `connection_types` and keeps only
    # its FIRST branch in the singular `connection_type` (a documented
    # back-compat hazard on the struct). Four optimizer consumers read the
    # singular field and silently narrowed the pattern to one branch, each
    # producing a wrong answer rather than a slow plan:
    #
    #   * the fused simple counter    (`count_simple_pattern_from_bound`)
    #   * the fused two-hop counter   (`count_two_hop_from_anchor`)
    #   * the anchored-count fusion   (`fuse_anchored_edge_count`)
    #   * `skip_target_type_check`    (endpoint-type guarantee from ONE branch
    #                                  applied to every branch — the only one
    #                                  of the four that corrupts projections
    #                                  rather than counts)
    #
    # The corpus had no alternation entry at all, which is the sole reason it
    # missed a class it was built to catch. Both branch orders are pinned
    # because the bug is order-sensitive: `[:B|A]` and `[:A|B]` returned
    # different answers for the same pattern. The absent-branch entry pins the
    # sharpest case — the singular field naming a type the graph does not have
    # collapsed the whole alternation to zero.
    (
        "alternation_count_forward",
        "social_graph",
        "MATCH (p:Person)-[:KNOWS|WORKS_AT]->(x) RETURN count(*) AS n",
        None,
    ),
    (
        "alternation_count_reversed",
        "social_graph",
        "MATCH (p:Person)-[:WORKS_AT|KNOWS]->(x) RETURN count(*) AS n",
        None,
    ),
    (
        "alternation_count_absent_branch_first",
        "social_graph",
        "MATCH (p:Person)-[:MENTORS|KNOWS|WORKS_AT]->(x) RETURN count(*) AS n",
        None,
    ),
    (
        "alternation_count_absent_branch_last",
        "social_graph",
        "MATCH (p:Person)-[:KNOWS|WORKS_AT|MENTORS]->(x) RETURN count(*) AS n",
        None,
    ),
    (
        "alternation_two_hop_count",
        "social_graph",
        "MATCH (a:Person)-[:KNOWS|WORKS_AT]->(b)-[:KNOWS|WORKS_AT]->(c) RETURN count(*) AS n",
        None,
    ),
    (
        "alternation_undirected_count",
        "social_graph",
        "MATCH (p:Person)-[:KNOWS|WORKS_AT]-(x) RETURN count(*) AS n",
        None,
    ),
    (
        "alternation_grouped_count",
        "social_graph",
        "MATCH (p:Person)-[:KNOWS|WORKS_AT]->(x) RETURN p.name AS a, count(*) AS n",
        None,
    ),
    (
        "alternation_anchored_count",
        "social_graph",
        "MATCH ({id: 1})-[:WORKS_AT|KNOWS]->(v) RETURN count(*) AS n",
        None,
    ),
    # `KNOWS` guarantees a Person target and `WORKS_AT` a Company one, so an
    # endpoint-type guarantee read off the first branch alone is wrong for the
    # second — projections, not just counts.
    (
        "alternation_typed_target_person",
        "social_graph",
        "MATCH (p:Person)-[:KNOWS|WORKS_AT]->(x:Person) RETURN x.name AS b",
        None,
    ),
    (
        "alternation_typed_target_company",
        "social_graph",
        "MATCH (p:Person)-[:KNOWS|WORKS_AT]->(x:Company) RETURN x.name AS b",
        None,
    ),
    # ── push_limit_into_match ──
    ("limit_simple", "social_graph", "MATCH (p:Person) RETURN p.name AS n LIMIT 5", None),
    ("limit_one", "social_graph", "MATCH (p:Person) RETURN p.name AS n LIMIT 1", None),
    ("limit_zero", "social_graph", "MATCH (p:Person) RETURN p.name AS n LIMIT 0", None),
    # ── 0.8.27 bug: multi-MATCH + WHERE on late-bound var + LIMIT ──
    (
        "multi_match_where_limit",
        "social_graph",
        "MATCH (a:Person) MATCH (b:Person) MATCH (c:Person) WHERE c.age > 35 RETURN a.name AS a LIMIT 10",
        None,
    ),
    # ── push_distinct_into_match ──
    (
        "distinct_with_match",
        "social_graph",
        "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN DISTINCT c.name AS c",
        None,
    ),
    # ── fuse_anchored_edge_count ──
    (
        "anchored_edge_count",
        "social_graph",
        "MATCH (p:Person {person_id: 1})-[:KNOWS]->(q:Person) RETURN count(q) AS n",
        None,
    ),
    # ── fuse_count_short_circuits ──
    ("count_distinct_star", "social_graph", "MATCH (p:Person) RETURN count(DISTINCT p) AS n", None),
    # ── fuse_node_scan_aggregate: count(DISTINCT property), plain + grouped ──
    ("count_distinct_prop", "social_graph", "MATCH (p:Person) RETURN count(DISTINCT p.city) AS n", None),
    (
        "count_distinct_prop_grouped",
        "social_graph",
        "MATCH (p:Person) RETURN p.city AS c, count(DISTINCT p.name) AS d",
        None,
    ),
    # ── fuse_node_scan_aggregate: the DISTINCT aggregates it must NOT fuse ──
    # `sum/avg(DISTINCT …)` grouped by a node property is the shape whose
    # surrogate re-bucket merged partial sums into a deduplicated value set;
    # `count(DISTINCT *)` is a row count the inline accumulator folded to 1.
    (
        "sum_distinct_prop_grouped",
        "social_graph",
        "MATCH (p:Person) RETURN p.city AS c, sum(DISTINCT p.age) AS s, avg(DISTINCT p.age) AS a",
        None,
    ),
    ("count_distinct_star_rows", "social_graph", "MATCH (p:Person) RETURN count(DISTINCT *) AS n", None),
    # ── fuse_optional_match_aggregate (0.8.31 bug) ──
    (
        "count_optional_edge_var",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[r:KNOWS]->(:Person) RETURN p.name AS n, count(r) AS k",
        None,
    ),
    # ── fuse_optional_match_aggregate (0.9.6 bug — collect()[slice] over OPTIONAL) ──
    # `aggregates_only_count` fell through `_ => true` for ListSlice/IndexAccess,
    # so `collect(x)[0..3]` was wrongly admitted to the count-only fusion.
    # The fused executor then ran `evaluate_expression` per-row on the
    # substituted (still-containing-collect) expression and the runtime
    # rejected the per-row aggregate call. The query below trips the same
    # admission gate; the `disabled_passes` half of the differential
    # harness exercises the materialised aggregator's correct path so any
    # future regression flags as a memory↔fused divergence.
    (
        "collect_slice_over_optional",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(q:Person) "
        "WITH p, collect(DISTINCT q.name)[0..3] AS first_three "
        "RETURN p.name AS n, first_three ORDER BY n",
        None,
    ),
    (
        "collect_index_over_optional",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(q:Person) "
        "WITH p, collect(DISTINCT q.name)[0] AS first "
        "RETURN p.name AS n, first ORDER BY n",
        None,
    ),
    (
        "sum_over_optional",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(q:Person) "
        "WITH p, sum(q.age) AS total "
        "RETURN p.name AS n, total ORDER BY n",
        None,
    ),
    # ── push_limit_into_aggregate (0.9.6 perf fix — Bug 3 in the user's
    # 124M-node Wikidata report). The aggregator now stops creating
    # new groups once `LIMIT N` distinct keys have been collected;
    # rows for already-collected keys continue to feed their
    # aggregates so collect() / sum() complete correctly. The query
    # below trips the same admission gate; the differential harness
    # confirms the optimised path matches the materialised-then-
    # truncated semantics.
    (
        "limit_into_aggregate_collect",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(q:Person) "
        "WITH p, collect(DISTINCT q.name) AS friends "
        "RETURN p.name AS n, friends LIMIT 3",
        None,
    ),
    (
        "limit_into_aggregate_count",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(q:Person) WITH p, count(q) AS k RETURN p.name AS n, k LIMIT 3",
        None,
    ),
    # ── group cap vs a NodeIndex-keyed surrogate (0.16.10) ──
    # The group set is keyed by the bound node until the resolution pass, so
    # one resolved group spreads over as many surrogates as it has nodes.
    # Capping the surrogate set dropped rows from groups already collected:
    # `count(*)` answered 3 where 10 was the truth and `collect()` returned 3
    # of 10 ids. The projecting WITH is load-bearing — without it
    # `fuse_node_scan_aggregate` absorbs the query and neither aggregation
    # path (nor the hint) is reached.
    (
        "group_cap_nodeprop_key_count",
        "dense_group_graph",
        "MATCH (n:T) WITH n, n.id AS i RETURN n.k AS k, count(*) AS c LIMIT 1",
        None,
    ),
    (
        "group_cap_nodeprop_key_collect",
        "dense_group_graph",
        "MATCH (n:T) WITH n, n.id AS i RETURN n.k AS k, collect(n.id) AS ids LIMIT 1",
        None,
    ),
    # The same shape with a group key that resolves inline: the surrogate set
    # *is* the resolved set, so the cap is exact and this arm must agree too.
    (
        "group_cap_eval_key_count",
        "dense_group_graph",
        "MATCH (n:T) WITH n.k AS kk, n.id AS i RETURN kk AS k, count(*) AS c LIMIT 1",
        None,
    ),
    # ORDER BY between projection and LIMIT MUST disable the
    # optimisation; the differential harness checks that the result
    # is still the proper top-3 by ascending count.
    (
        "limit_with_order_by_no_pushdown",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(q:Person) "
        "WITH p, count(q) AS k "
        "RETURN p.name AS n, k ORDER BY k ASC, n ASC LIMIT 3",
        None,
    ),
    # ── fuse_match_return_aggregate ──
    (
        "global_two_hop_count",
        "social_graph",
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) RETURN count(*) AS paths",
        None,
    ),
    ("group_by_city", "social_graph", "MATCH (p:Person) RETURN p.city AS city, count(p) AS n", None),
    ("group_by_with_sum", "social_graph", "MATCH (p:Person) RETURN p.city AS city, sum(p.salary) AS total", None),
    # Edge-driven group-by where the target node carries a `:Type` label.
    # Pre-fix the planner reversed the pattern to start at :Company, which
    # bailed the FusedMatchReturnAggregate fast path (group_elem_idx=0 with
    # Incoming edge), forcing the slow node-centric scan. On Wikidata this
    # was timeout 122s vs corrected 169ms. The optimised path uses
    # `lookup_peer_counts` keyed by edge target plus `binary_search_idx`
    # against `type_indices[T]` for the type filter; the naive Cypher path
    # iterates everything and produces the same result, so this differential
    # entry doubles as a regression gate for the bypass.
    (
        "edge_groupby_typed_target_top_k",
        "social_graph",
        "MATCH (p:Person)-[:WORKS_AT]->(c:Company) "
        "RETURN c.name AS company, count(p) AS workers "
        "ORDER BY workers DESC, company LIMIT 3",
        None,
    ),
    # Same shape, no ORDER BY+LIMIT — exercises the non-top-K branch of
    # FusedMatchReturnAggregate, which carried the same `group_elem_idx`-only
    # bail as the top-K branch (P1.5 fix). Companion test to the entry above:
    # both paths must agree with the naive walk.
    (
        "edge_groupby_typed_target_no_orderby",
        "social_graph",
        "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN c.name AS company, count(p) AS workers",
        None,
    ),
    # Group at SOURCE side (P2 fix). The persistent peer histogram is keyed
    # by edge target only, so the source-side dual computes counts on the fly
    # via `count_edges_grouped_by_peer(.., Direction::Incoming)`. Same type
    # filter via binary_search_idx applies. Locks in the new fast path's
    # equivalence with the naive walk for both type-anchored and unanchored
    # source.
    (
        "edge_groupby_source_typed",
        "social_graph",
        "MATCH (p:Person)-[:WORKS_AT]->(c) "
        "RETURN p.name AS person, count(c) AS jobs "
        "ORDER BY jobs DESC, person LIMIT 5",
        None,
    ),
    # ORDER BY <agg-expr> form — historically the absorption pass only
    # matched ORDER BY <alias>, so writing the same query as
    # `ORDER BY count(p)` left ORDER BY+LIMIT in the pipeline and the
    # executor materialised every distinct peer (~245k on Wikidata
    # P138). Now both forms fuse equivalently. The differential check
    # is structural (row-set equality), so this entry guards against
    # divergence between the alias-form and the expression-form fast
    # paths.
    (
        "edge_groupby_orderby_expression_form",
        "social_graph",
        "MATCH (p:Person)-[:WORKS_AT]->(c:Company) "
        "RETURN c.name AS company, count(p) "
        "ORDER BY count(p) DESC, company LIMIT 3",
        None,
    ),
    # ORDER BY a non-projected property of a *grouping* variable, after an
    # aggregate. Aggregation rebuilds its output rows, so `p` survives only
    # if the operator carries its binding forward — and the three
    # aggregation operators (streaming, materialized, fused OPTIONAL) used
    # to disagree about that. Where the binding was dropped the sort key
    # evaluated to NULL on every row and the ORDER BY was silently ignored.
    # The optimizer decides which operator runs, so opt-vs-naive is exactly
    # the axis that exposed it. Row-order assertions live in
    # tests/test_cypher_order_by_after_aggregate.py; these entries guard the
    # row-set equality the corpus is responsible for.
    (
        "optional_match_aggregate_order_by_non_projected",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company) "
        "RETURN p.name AS person, count(c) AS jobs ORDER BY p.age DESC",
        None,
    ),
    (
        "optional_match_aggregate_order_by_non_projected_paged",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(f:Person) "
        "RETURN p.name AS person, count(f) AS friends "
        "ORDER BY p.age DESC SKIP 5 LIMIT 5",
        None,
    ),
    # `collect()` is outside the streaming aggregate's reach, so this one
    # takes the materialized path — the same defect with no OPTIONAL MATCH
    # involved at all.
    (
        "collect_aggregate_order_by_non_projected",
        "social_graph",
        "MATCH (p:Person) RETURN p.city AS city, collect(p.name) AS names ORDER BY p.city DESC",
        None,
    ),
    # Aggregate on the EDGE variable (not a node variable). Pre-fix the
    # gate at fuse_match_return_aggregate only accepted count(<other-node>);
    # count(<edge_var>) silently fell out of fusion despite being
    # semantically equivalent for a 3-element pattern (each edge is one
    # other-node binding). Wikidata citation queries are typically
    # written as `(paper)<-[r:P2860]-(citing) ... count(r)`, the natural
    # form, and were dropping into the slow path before this fix.
    (
        "edge_groupby_count_edge_variable",
        "social_graph",
        "MATCH (p:Person)-[r:WORKS_AT]->(c:Company) "
        "RETURN c.name AS company, count(r) AS edges "
        "ORDER BY edges DESC, company LIMIT 3",
        None,
    ),
    # MATCH...WITH variant — exercises `try_fast_with_aggregate_via_histogram`
    # in the executor. Pre-fix this also bailed on group_elem_idx != 2,
    # forcing the per-source enumeration path (3 places in match_clause.rs
    # had the same position-only check; this is the third, after the two
    # in execute_fused_match_return_aggregate). The shape now fuses for
    # both AST orderings.
    (
        "edge_groupby_match_with_aggregate_typed_target",
        "social_graph",
        "MATCH (p:Person)-[:WORKS_AT]->(c:Company) "
        "WITH c, count(p) AS workers "
        "RETURN c.name AS company, workers ORDER BY workers DESC, company LIMIT 3",
        None,
    ),
    # ── fuse_match_with_aggregate + fuse_match_with_aggregate_top_k (0.8.32 bug) ──
    # Secondary sort key (city, n) breaks ties so the row identities are
    # deterministic — without it, both modes return correct counts but
    # which-3-of-4-tied-cities surfaces is implementation-defined.
    (
        "cohort_top_k",
        "social_graph",
        "MATCH (p:Person) WITH p.city AS city, count(p) AS n RETURN city, n ORDER BY n DESC, city LIMIT 3",
        None,
    ),
    (
        "cohort_top_k_property",
        "social_graph",
        "MATCH (p:Person) WITH p, count{(p)-[:KNOWS]->()} AS friends "
        "RETURN p.name AS n, friends ORDER BY friends DESC, n LIMIT 5",
        None,
    ),
    # ── fuse_node_scan_aggregate ──
    ("node_scan_count", "social_graph", "MATCH (n) RETURN count(n) AS n", None),
    # ── fuse_node_scan_top_k + fuse_order_by_top_k ──
    (
        "order_by_limit",
        "social_graph",
        "MATCH (p:Person) RETURN p.name AS n, p.age AS age ORDER BY p.age DESC LIMIT 5",
        None,
    ),
    (
        "order_by_ties",
        "social_graph",
        "MATCH (p:Person) RETURN p.name AS n, p.city AS c ORDER BY p.city, p.name LIMIT 10",
        None,
    ),
    # Multi-key ORDER BY + LIMIT. Before 0.15.14 both top-K passes bailed on
    # more than one sort item, so these shapes only ever ran the full sort;
    # the entries below pin the fused plan against it. `_normalize` sorts
    # rows, so what these catch is the *selected set* — the emitted order is
    # pinned by tests/test_cypher_top_k_ordering.py.
    (
        "order_by_two_keys_limit",
        "social_graph",
        "MATCH (p:Person) RETURN p.name AS n, p.city AS c ORDER BY p.city, p.age LIMIT 7",
        None,
    ),
    (
        "order_by_three_keys_limit",
        "social_graph",
        "MATCH (p:Person) RETURN p.name AS n ORDER BY p.city, p.age DESC, p.name LIMIT 6",
        None,
    ),
    (
        "order_by_two_keys_mixed_directions_limit",
        "social_graph",
        "MATCH (p:Person) RETURN p.name AS n, p.age AS age ORDER BY p.city DESC, p.age ASC LIMIT 8",
        None,
    ),
    # Five Persons share each city, so the LIMIT-4 cut falls inside a
    # first-key tie group and the second key decides which rows are emitted.
    (
        "order_by_tie_on_first_key_resolved_by_second",
        "social_graph",
        "MATCH (p:Person) RETURN p.name AS n ORDER BY p.city, p.name LIMIT 4",
        None,
    ),
    # NULLs in the leading key: `email` is None for odd-numbered persons.
    # DESC defaults to NULLS FIRST, so the NULL-keyed rows ARE the winners —
    # the fused paths used to drop them and return the wrong rows.
    (
        "order_by_null_first_key_desc_limit",
        "social_graph",
        "MATCH (p:Person) RETURN p.name AS n, p.email AS e ORDER BY p.email DESC, p.name LIMIT 5",
        None,
    ),
    (
        "order_by_null_first_key_asc_limit",
        "social_graph",
        "MATCH (p:Person) RETURN p.name AS n, p.email AS e ORDER BY p.email ASC, p.name LIMIT 5",
        None,
    ),
    (
        "order_by_null_second_key_limit",
        "social_graph",
        "MATCH (p:Person) RETURN p.name AS n, p.email AS e ORDER BY p.city, p.email LIMIT 9",
        None,
    ),
    # LIMIT exceeds the number of non-NULL keys: the fused path used to emit
    # fewer rows than the ordinary pipeline.
    (
        "order_by_null_key_limit_beyond_non_null_rows",
        "social_graph",
        "MATCH (p:Person) WHERE p.age > 35 RETURN p.name AS n, p.email AS e ORDER BY p.email DESC, p.name LIMIT 20",
        None,
    ),
    (
        "order_by_explicit_nulls_placement_two_keys",
        "social_graph",
        "MATCH (p:Person) RETURN p.name AS n, p.email AS e ORDER BY p.email DESC NULLS LAST, p.name ASC LIMIT 5",
        None,
    ),
    # ORDER BY over projected aliases (both keys), and over an *expression*
    # of an alias — the latter must not fuse (the alias is unbound before
    # projection); it used to fuse and return zero rows.
    (
        "order_by_projected_aliases_limit",
        "social_graph",
        "MATCH (p:Person) RETURN p.city AS c, p.age AS age ORDER BY c, age DESC LIMIT 6",
        None,
    ),
    (
        "order_by_expression_over_projected_alias_limit",
        "social_graph",
        "MATCH (p:Person) RETURN p.age AS age ORDER BY age + 1 DESC LIMIT 5",
        None,
    ),
    (
        "order_by_two_keys_after_with_limit",
        "social_graph",
        "MATCH (p:Person) WITH p.city AS c, p.age AS age, p.name AS n RETURN n, c, age ORDER BY c, age DESC LIMIT 6",
        None,
    ),
    (
        "order_by_two_keys_over_edges_limit",
        "social_graph",
        "MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a.name AS an, b.name AS bn, r.tag AS tag "
        "ORDER BY r.tag, b.name DESC LIMIT 10",
        None,
    ),
    # ── reorder_predicates_by_cost ──
    (
        "predicate_reorder",
        "social_graph",
        "MATCH (p:Person) WHERE p.salary > 80000 AND p.city = 'Oslo' RETURN p.name AS n",
        None,
    ),
    # ── mark_fast_var_length_paths ──
    # The unguarded fast path used to dedup target nodes during BFS,
    # silently returning fewer rows than per-path Cypher semantics
    # demand. The pass is now gated to fire only when downstream
    # collapses row multiplicity (DISTINCT or distinct-safe aggregate).
    (
        "var_length_no_var_per_path",
        "small_graph",
        # No DISTINCT, no aggregate → slow per-path BFS (3 rows in
        # small_graph: 1→2, 1→3, 1→2→3).
        "MATCH (p:Person {person_id: 1})-[:KNOWS*1..3]->(q:Person) RETURN q.name AS n",
        None,
    ),
    (
        "var_length_no_var_distinct",
        "small_graph",
        # DISTINCT → fast path is safe to fire (2 rows: Bob, Charlie).
        # Both modes dedup at projection so they match either way.
        "MATCH (p:Person {person_id: 1})-[:KNOWS*1..3]->(q:Person) RETURN DISTINCT q.name AS n",
        None,
    ),
    (
        "var_length_no_var_count_distinct",
        "small_graph",
        # count(DISTINCT _) is dedup-safe — the aggregate collapses
        # multiplicities so the fast path's per-target dedup matches.
        "MATCH (p:Person {person_id: 1})-[:KNOWS*1..3]->(q:Person) RETURN count(DISTINCT q) AS n",
        None,
    ),
    (
        "var_length_with_var",
        "small_graph",
        "MATCH (p:Person {person_id: 1})-[r:KNOWS*1..3]->(q:Person) RETURN q.name AS n",
        None,
    ),
    # ── mark_fast_var_length_paths on CYCLIC graphs ──
    # Every var-length fixture above is acyclic with min=1 — the weakest
    # possible witness for a pass whose whole job is deciding when a
    # *distance* BFS may stand in for Cypher's *trail* semantics. On a cycle
    # the two relations come apart, and the un-gated pass shipped both
    # failure directions: it dropped the source node from its own answer
    # (`*1..3` on a triangle returned 2 of 3 nodes) and it answered `min >= 2`
    # patterns with the empty set (`*2..2` undirected on a triangle).
    (
        "var_length_cyclic_source_on_closed_trail",
        "directed_triangle_graph",
        # 1 reaches itself in 3 hops around the cycle, so it is one of its own
        # answers. The distance BFS pre-marks the source visited and cannot
        # see that: it returned [2, 3].
        "MATCH (a:N {id: 1})-[:R*1..3]->(b:N) RETURN DISTINCT b.id AS i",
        None,
    ),
    (
        "var_length_cyclic_undirected_source_on_closed_trail",
        "directed_triangle_graph",
        "MATCH (a:N {id: 1})-[:R*1..3]-(b:N) RETURN DISTINCT b.id AS i",
        None,
    ),
    (
        "var_length_cyclic_undirected_min_two",
        "directed_triangle_graph",
        # Trail-reachable to both peers (1-2-3 forwards, 1-3-2 backwards),
        # distance-reachable to neither. The fast path returned [].
        "MATCH (a:N {id: 1})-[:R*2..2]-(b:N) RETURN DISTINCT b.id AS i",
        None,
    ),
    (
        "var_length_cyclic_count_distinct_min_two",
        "directed_triangle_graph",
        "MATCH (a:N {id: 1})-[:R*2..3]->(b:N) RETURN count(DISTINCT b) AS n",
        None,
    ),
    (
        "var_length_cyclic_zero_hop_control",
        "directed_triangle_graph",
        # `min = 0` already emits the source at zero hops; the control proves
        # the source-inclusion fix does not double-emit it.
        "MATCH (a:N {id: 1})-[:R*0..2]->(b:N) RETURN DISTINCT b.id AS i",
        None,
    ),
    (
        "var_length_second_clause_plain_projection",
        "var_length_diamond_graph",
        # The first consumer is `WITH DISTINCT b`; the second MATCH's consumer
        # is a plain per-path projection. Proving only the first clause and
        # marking every clause returned 2 rows where the graph has 3.
        "MATCH (a:N {id: 1})-[:R*1..1]->(b:N) WITH DISTINCT b MATCH (b)-[:R*1..2]->(c:N) RETURN c.id AS i",
        None,
    ),
    (
        "var_length_second_clause_count_star",
        "var_length_diamond_graph",
        "MATCH (a:N {id: 1})-[:R*1..1]->(b:N) WITH DISTINCT b MATCH (b)-[:R*1..2]->(c:N) RETURN count(*) AS n",
        None,
    ),
    (
        "var_length_distinct_over_row_counting_aggregate",
        "directed_triangle_graph",
        # DISTINCT groups *after* aggregating, so `count(b)` is a per-group row
        # count: it moves with any dropped duplicate (3 -> 2).
        "MATCH (a:N {id: 1})-[:R*1..3]->(b:N) RETURN DISTINCT a.id AS i, count(b) AS n",
        None,
    ),
    (
        "var_length_undirected_parallel_relationships",
        "parallel_edge_cycle_graph",
        # Two relationships between 1 and 2 make a closed trail of length 2,
        # so 1 is one of its own undirected answers. One relationship would
        # not: walking back over the same one is forbidden.
        "MATCH (a:N {id: 1})-[:R*1..2]-(b:N) RETURN DISTINCT b.id AS i",
        None,
    ),
    (
        "var_length_undirected_no_closed_trail",
        "var_length_diamond_graph",
        # Negative control for the same probe: node 1 has one relationship and
        # nothing leads back to it, so it must not appear in its own answer.
        "MATCH (a:N {id: 1})-[:R*1..3]-(b:N) RETURN DISTINCT b.id AS i",
        None,
    ),
    (
        "var_length_undirected_cycle_min_two",
        "parallel_edge_cycle_graph",
        "MATCH (a:N {id: 1})-[:R*2..3]-(b:N) RETURN count(DISTINCT b) AS n",
        None,
    ),
    # ── distinct-target pushdown (part-6 phase V4) ──
    (
        "khop_count_distinct_over_many_seeds",
        "overlapping_khop_graph",
        # The consumer reads only `f`, so the expansion deduplicates targets
        # globally instead of building one row per (seed, target) pair. The
        # answer is the union of the reachable sets, not their sum.
        "MATCH (p:N)-[:R*1..3]->(f:N) WHERE p.id IN [0, 4] RETURN count(DISTINCT f) AS n",
        None,
    ),
    (
        "khop_count_distinct_per_source_when_the_source_escapes",
        "overlapping_khop_graph",
        # Control for the escape analysis: `p` in the projection means the
        # counts stay per-source, which a global target dedup would destroy.
        "MATCH (p:N)-[:R*1..3]->(f:N) WHERE p.id IN [0, 4] RETURN p.id AS i, count(DISTINCT f) AS n ORDER BY i",
        None,
    ),
    (
        "khop_seed_reaches_only_through_already_seen_targets",
        "overlapping_khop_graph",
        # Everything seed 1 adds sits one hop past a node seed 0 already
        # reached: a dedup that pruned the frontier rather than the emitted
        # row would lose all of it.
        "MATCH (p:N)-[:R*1..3]->(f:N) WHERE p.id IN [0, 1] RETURN count(DISTINCT f) AS n",
        None,
    ),
    (
        "khop_count_star_is_not_dedup_safe",
        "overlapping_khop_graph",
        # The negative control: row count is path count here, so no target
        # dedup may run at all.
        "MATCH (p:N)-[:R*1..3]->(f:N) WHERE p.id IN [0, 4] RETURN count(*) AS n",
        None,
    ),
    (
        "distinct_target_with_a_non_pushable_source_predicate",
        "unequal_source_graph",
        # The matcher's per-target dedup keeps one arbitrary source, and this
        # predicate (arithmetic, so not pushed into the pattern) rejects exactly
        # that one — the pass has to be redone without the dedup or the target
        # is lost.
        "MATCH (a:N)-[:R]->(f:N) WHERE a.id + 0 > 1 RETURN DISTINCT f.id AS i",
        None,
    ),
    (
        "comma_patterns_distinct_single_variable",
        "join_then_distinct_graph",
        # The DISTINCT hint used to deduplicate the first pattern before the
        # second joined, dropping the one `a` that satisfies both: answered
        # no rows where the same query without DISTINCT answers one.
        "MATCH (a:N)-[:R]->(f:N), (a)-[:S]->(g:N) RETURN DISTINCT f.id AS i",
        None,
    ),
    # ── the UNWIND spelling of the same pushdown (part-6 phase V4b) ──
    (
        "khop_unwind_count_distinct_over_many_seeds",
        "overlapping_khop_graph",
        # One expansion per driving row, sharing one seen-set across them: the
        # answer is still the union of the reachable sets.
        "UNWIND [0, 4] AS i MATCH (p:N {id: i})-[:R*1..3]->(f:N) RETURN count(DISTINCT f) AS n",
        None,
    ),
    (
        "khop_unwind_count_distinct_per_source_when_the_source_escapes",
        "overlapping_khop_graph",
        # Escape control for the driving-row branch: the counts stay per-seed,
        # which is exactly what a shared seen-set would fold together.
        "UNWIND [0, 4] AS i MATCH (p:N {id: i})-[:R*1..3]->(f:N) "
        "RETURN p.id AS pid, count(DISTINCT f) AS n ORDER BY pid",
        None,
    ),
    (
        "khop_unwind_seed_reaches_only_through_already_seen_targets",
        "overlapping_khop_graph",
        # Sharing the set must skip the emitted row, never the frontier: seed
        # 1 adds only nodes that sit one hop past seed 0's targets.
        "UNWIND [0, 1] AS i MATCH (p:N {id: i})-[:R*1..3]->(f:N) RETURN count(DISTINCT f) AS n",
        None,
    ),
    (
        "khop_unwind_repeated_seed_contributes_no_rows",
        "overlapping_khop_graph",
        # The second driving row reaches nothing new, so under the shared set
        # it emits nothing at all.
        "UNWIND [0, 0] AS i MATCH (p:N {id: i})-[:R*1..3]->(f:N) RETURN count(DISTINCT f) AS n",
        None,
    ),
    (
        "khop_unwind_count_star_is_not_dedup_safe",
        "overlapping_khop_graph",
        # Negative control: row count is path count, so no dedup may run.
        "UNWIND [0, 4] AS i MATCH (p:N {id: i})-[:R*1..3]->(f:N) RETURN count(*) AS n",
        None,
    ),
    (
        "khop_two_match_count_distinct_over_many_seeds",
        "overlapping_khop_graph",
        # The third driving-row shape: `p` bound on the row rather than
        # projected, expanded by a second MATCH clause.
        "MATCH (p:N) WHERE p.id IN [0, 4] MATCH (p)-[:R*1..3]->(f:N) RETURN count(DISTINCT f) AS n",
        None,
    ),
    (
        "var_length_sibling_edge_shares_the_relationship",
        "two_cycle_graph",
        # A marked segment records no trail, so the clause's relationship
        # uniqueness check cannot see the relationships it walked and a
        # sibling edge is free to re-bind one. On 1 ⇄ 2 the segment reaches 1
        # again over both relationships, and the last hop then re-took the one
        # the segment had already used: [1, 2] where trail semantics give [1].
        "MATCH (a:N {id: 1})-[:R*1..2]->(x:N)-[:R]->(c:N) RETURN DISTINCT c.id AS i ORDER BY i",
        None,
    ),
    (
        "var_length_sibling_edge_with_a_disjoint_type",
        "hub_return_graph",
        # The control for the gate above: no relationship can be both `:A` and
        # `:B`, so the segment keeps its mark and stays on the fast path.
        "MATCH (a:N {id: 1})-[:A*1..1]->(x:N)-[:B]->(c:N) RETURN DISTINCT c.id AS i ORDER BY i",
        None,
    ),
    # ── mark_fast_var_length_paths inside EXISTS { … } ──
    # An existence check consumes one boolean, so it is the strongest possible
    # dedup-safe consumer — but the correctness gates that are about the BFS
    # itself (`min <= 1`, no relationship variable) still apply, and so does
    # the witness cap the evaluator now passes down. Every entry below is a
    # cyclic graph, where distance and trail reachability come apart.
    (
        "exists_var_length_min_one",
        "directed_triangle_graph",
        "MATCH (n:N) WHERE EXISTS { (n)-[:R*1..2]->(:N) } RETURN n.id AS i ORDER BY i",
        None,
    ),
    (
        "exists_var_length_min_two",
        "directed_triangle_graph",
        # `min >= 2` never takes the set-based path; the witness cap still
        # stops the per-path expansion at the first complete trail.
        "MATCH (n:N) WHERE EXISTS { (n)-[:R*2..3]->(:N) } RETURN n.id AS i ORDER BY i",
        None,
    ),
    (
        "exists_var_length_undirected",
        "parallel_edge_cycle_graph",
        "MATCH (n:N) WHERE EXISTS { (n)-[:R*1..3]-(:N {id: 1}) } RETURN n.id AS i ORDER BY i",
        None,
    ),
    (
        "not_exists_var_length_witness_absent",
        "directed_triangle_graph",
        # Zero witnesses after the full search — the answer NOT EXISTS needs
        # the cap's uncapped retry to still be able to reach.
        "MATCH (n:N) WHERE NOT EXISTS { (n)-[:R*1..3]->(:N {id: 99}) } RETURN n.id AS i ORDER BY i",
        None,
    ),
    (
        "exists_var_length_in_projection",
        "directed_triangle_graph",
        "MATCH (n:N) RETURN n.id AS i, EXISTS { (n)-[:R*1..2]->(:N {id: 1}) } AS reaches ORDER BY i",
        None,
    ),
    (
        "exists_var_length_with_inner_where",
        "directed_triangle_graph",
        # An inner WHERE forbids the witness cap: the first match may fail it
        # while a later one passes.
        "MATCH (n:N) WHERE EXISTS { (n)-[:R*1..2]->(m:N) WHERE m.id = 1 } RETURN n.id AS i ORDER BY i",
        None,
    ),
    (
        "exists_var_length_two_predicates",
        "directed_triangle_graph",
        # Each EXISTS is capped independently.
        "MATCH (n:N) WHERE EXISTS { (n)-[:R*1..2]->(:N {id: 1}) } "
        "AND NOT EXISTS { (n)-[:R*1..1]->(:N {id: 3}) } RETURN n.id AS i ORDER BY i",
        None,
    ),
    (
        "count_var_length_subquery_keeps_its_trails",
        "directed_triangle_graph",
        # `count { … }` counts rows, so nothing about it is dedup-safe and its
        # segment must NOT be marked — the control for the EXISTS entries.
        "MATCH (n:N) RETURN n.id AS i, COUNT { (n)-[:R*1..2]->(:N) } AS c ORDER BY i",
        None,
    ),
    # ── lower_fixed_var_length_hops ──
    # `*k..k` asks a fixed-length question with a variable-length spelling.
    # The pass rewrites it into k explicit hops so the fixed-pattern machinery
    # (start-node selection, relationship pushdown, the fusion family, the
    # trail and target-type annotations) applies. The naive leg keeps the star
    # spelling, so every entry below is a lowered-vs-variable comparison.
    (
        "lower_two_hop_unanchored_count",
        "var_length_diamond_graph",
        # The shape V2 was measured on: no anchor, so the star spelling could
        # not pick a start node and expanded from every `:N`.
        "MATCH (a:N)-[:R*2..2]->(b:N) RETURN count(*) AS n",
        None,
    ),
    (
        "lower_two_hop_anchored",
        "directed_triangle_graph",
        "MATCH (a:N {id: 1})-[:R*2..2]->(b:N) RETURN DISTINCT b.id AS i",
        None,
    ),
    (
        "three_hop_closed_trail_past_the_lowering_ceiling",
        "directed_triangle_graph",
        # Three hops around the cycle land back on the source: the answer must
        # allow the repeated *node* while still refusing a repeated
        # relationship. Kept after V8 narrowed the lowering ceiling to two
        # hops — this shape is now served by the variable-length expansion
        # rather than by lowered fixed hops, and the closed-trail answer has
        # to be the same either way. `lower_two_hop_anchored` above is what
        # exercises the pass itself.
        "MATCH (a:N {id: 1})-[:R*3..3]->(b:N) RETURN b.id AS i",
        None,
    ),
    (
        "lower_two_hop_undirected_cannot_reverse",
        "var_length_diamond_graph",
        # Node 1 has exactly one relationship. Undirected two hops must leave
        # over 1->2 and continue to 3 or 4 — never walk back over 1->2, which
        # would put 1 in its own answer.
        "MATCH (a:N {id: 1})-[:R*2..2]-(b:N) RETURN DISTINCT b.id AS i",
        None,
    ),
    (
        "lower_two_hop_undirected_parallel_relationships",
        "parallel_edge_cycle_graph",
        # ...and the converse: two distinct relationships between 1 and 2 DO
        # make a closed trail of length 2, so 1 is one of its own answers here.
        # A lowering that deduped by node would drop it; one that ignored
        # relationship identity entirely would report it on the graph above.
        "MATCH (a:N {id: 1})-[:R*2..2]-(b:N) RETURN b.id AS i",
        None,
    ),
    (
        "lower_two_hop_edge_property_filter",
        "social_graph",
        # An inline relationship property is replicated onto every lowered
        # hop, because variable-length semantics require every relationship in
        # the segment to satisfy it.
        "MATCH (a:Person)-[:KNOWS*2..2 {since: 2016}]->(b:Person) RETURN count(*) AS n",
        None,
    ),
    (
        "lower_two_hop_type_alternation",
        "social_graph",
        # Each hop independently accepts any of the alternation's types — the
        # same reading the variable-length expansion takes — so the whole
        # alternation is copied onto each hop rather than split across them.
        "MATCH (a:Person)-[:KNOWS|WORKS_AT*2..2]->(b) RETURN count(*) AS n",
        None,
    ),
    (
        "lower_optional_match_two_hop",
        "var_length_diamond_graph",
        # Null-extension is per row and per pattern, and the lowered pattern
        # binds exactly the same variables (the intermediates are anonymous),
        # so an `a` with no two-hop target keeps its null `j`.
        "MATCH (a:N) OPTIONAL MATCH (a)-[:R*2..2]->(c:N) RETURN a.id AS i, c.id AS j",
        None,
    ),
    (
        "lower_hop_ceiling_at_eight",
        "long_chain_graph",
        "MATCH (a:N {id: 0})-[:R*8..8]->(b:N) RETURN b.id AS i",
        None,
    ),
    (
        "lower_hop_ceiling_declines_nine",
        "long_chain_graph",
        # One hop past the ceiling: the segment stays variable-length, and
        # must still answer what it answered before.
        "MATCH (a:N {id: 0})-[:R*9..9]->(b:N) RETURN b.id AS i",
        None,
    ),
    (
        "lower_declines_bound_relationship_variable",
        "directed_triangle_graph",
        # `r` binds the segment's relationship LIST; individual hop bindings
        # cannot reconstruct it, so the star spelling stays.
        "MATCH (a:N {id: 1})-[r:R*2..2]->(b:N) RETURN size(r) AS hops, b.id AS i",
        None,
    ),
    (
        "lower_declines_path_assignment",
        "directed_triangle_graph",
        "MATCH p = (a:N {id: 1})-[:R*2..2]->(b:N) RETURN [n IN nodes(p) | n.id] AS ids",
        None,
    ),
    # ── intermediate-dedup soundness (matcher, not a pass) ──
    # `push_distinct_into_match` lets the matcher drop partial matches that
    # share an anonymous intermediate node. Two of them are not
    # interchangeable when a later hop can tell them apart, and both ways it
    # can were shipping wrong answers. The `*k..k` lowering makes anonymous
    # intermediates common, which is how they surfaced.
    (
        "distinct_over_anonymous_intermediates_keeps_every_trail",
        "square_cycle_graph",
        # Undirected three hops from 1 reach 2 and 4. The dedup kept only 4:
        # its surviving partial had already consumed the relationship that the
        # last hop to 2 needed.
        "MATCH (a:N {id: 1})-[:R]-()-[:R]-()-[:R]-(b:N) RETURN DISTINCT b.id AS i",
        None,
    ),
    (
        "distinct_over_anonymous_intermediates_var_length_spelling",
        "square_cycle_graph",
        # The same claim through the lowering, which is what put a three-hop
        # anonymous-intermediate pattern in front of the matcher.
        "MATCH (a:N {id: 1})-[:R*3..3]-(b:N) RETURN DISTINCT b.id AS i",
        None,
    ),
    (
        "distinct_over_anonymous_intermediate_with_a_repeated_variable",
        "hub_return_graph",
        # Disjoint hop types record no trail, so the trail guard does not
        # apply — but `(a)` is bound twice, and the dedup kept one `a` for the
        # hub, returning 1 of the 3 rows.
        "MATCH (a:N)-[:A]->()-[:B]->(a) RETURN DISTINCT a.id AS i",
        None,
    ),
    (
        "distinct_over_anonymous_intermediate_stays_deduped_when_it_is_safe",
        "hub_return_graph",
        # Positive control: distinct end variable, no repeat, disjoint types —
        # the dedup is legal here and must still fire.
        "MATCH (a:N)-[:A]->()-[:B]->(b:N) RETURN DISTINCT b.id AS i",
        None,
    ),
    (
        "lower_second_clause_after_with",
        "var_length_diamond_graph",
        # Two lowered segments in one query, the second seeded from the first.
        "MATCH (a:N {id: 1})-[:R*1..1]->(b:N) WITH DISTINCT b MATCH (b)-[:R*2..2]->(c:N) RETURN c.id AS i",
        None,
    ),
    # ── UNION (optimize_nested_queries) ──
    (
        "union_simple",
        "small_graph",
        "MATCH (p:Person) WHERE p.age < 30 RETURN p.name AS n "
        "UNION MATCH (p:Person) WHERE p.age > 40 RETURN p.name AS n",
        None,
    ),
    (
        "optional_no_match",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(c:Company) RETURN p.name AS n, c.name AS c",
        None,
    ),
    (
        "with_chain",
        "social_graph",
        "MATCH (p:Person) WITH p WHERE p.age > 25 WITH p, p.salary AS s RETURN p.name AS n, s",
        None,
    ),
    ("empty_typed_match", "social_graph", "MATCH (n:NoSuchType) RETURN count(n) AS n", None),
    ("skip_and_limit", "social_graph", "MATCH (p:Person) RETURN p.name AS n ORDER BY p.person_id SKIP 5 LIMIT 3", None),
    (
        "union_all",
        "small_graph",
        "MATCH (p:Person) RETURN p.name AS n UNION ALL MATCH (p:Person) RETURN p.name AS n",
        None,
    ),
    (
        "case_simple",
        "social_graph",
        "MATCH (p:Person) RETURN p.name AS n, CASE WHEN p.age > 30 THEN 'old' ELSE 'young' END AS bucket",
        None,
    ),
    (
        "case_chain",
        "social_graph",
        "MATCH (p:Person) RETURN p.name AS n, "
        "CASE WHEN p.age < 25 THEN 'young' WHEN p.age < 35 THEN 'mid' ELSE 'old' END AS bucket",
        None,
    ),
    ("starts_with", "social_graph", "MATCH (p:Person) WHERE p.name STARTS WITH 'Person_1' RETURN p.name AS n", None),
    ("contains", "social_graph", "MATCH (p:Person) WHERE p.name CONTAINS '_1' RETURN p.name AS n", None),
    ("ends_with", "social_graph", "MATCH (p:Person) WHERE p.name ENDS WITH '_5' RETURN p.name AS n", None),
    (
        "multi_hop_contains_distinct",
        "social_graph",
        "MATCH (j:Person)<-[:KNOWS]-(d:Person)-[:WORKS_AT]->(c:Company) "
        "WHERE j.name CONTAINS '_1' RETURN DISTINCT c.name AS company",
        None,
    ),
    (
        "multi_hop_ends_with_param_distinct",
        "social_graph",
        "MATCH (j:Person)<-[:KNOWS]-(d:Person)-[:WORKS_AT]->(c:Company) "
        "WHERE j.name ENDS WITH $suffix RETURN DISTINCT c.name AS company",
        {"suffix": "_5"},
    ),
    ("not_equal", "social_graph", "MATCH (p:Person) WHERE p.city <> 'Oslo' RETURN count(p) AS n", None),
    (
        "range_predicate",
        "social_graph",
        "MATCH (p:Person) WHERE p.age >= 25 AND p.age <= 35 RETURN count(p) AS n",
        None,
    ),
    ("null_check", "social_graph", "MATCH (p:Person) WHERE p.email IS NOT NULL RETURN count(p) AS n", None),
    # ── B1: three-valued NULL semantics in WHERE comparisons ──
    # social_graph has email=None for odd-numbered persons. Each of
    # these triggers a code path that the pre-0.9.52 collapse-to-bool
    # would have surfaced as silent wrong rows.
    (
        "b1_ne_with_null",
        "social_graph",
        "MATCH (p:Person) WHERE p.email <> 'person2@test.com' RETURN count(p) AS n",
        None,
    ),
    (
        "b1_lt_with_null",
        "social_graph",
        "MATCH (p:Person) WHERE p.email < 'zzz' RETURN count(p) AS n",
        None,
    ),
    (
        "b1_not_lt_with_null",
        "social_graph",
        "MATCH (p:Person) WHERE NOT (p.email < 'zzz') RETURN count(p) AS n",
        None,
    ),
    # ── B2: NULL propagation through string predicates under NOT ──
    (
        "b2_not_contains_with_null",
        "social_graph",
        "MATCH (p:Person) WHERE NOT (p.email CONTAINS 'person') RETURN count(p) AS n",
        None,
    ),
    (
        "b2_not_starts_with_with_null",
        "social_graph",
        "MATCH (p:Person) WHERE NOT (p.email STARTS WITH 'person') RETURN count(p) AS n",
        None,
    ),
    (
        "b2_not_ends_with_with_null",
        "social_graph",
        "MATCH (p:Person) WHERE NOT (p.email ENDS WITH 'test.com') RETURN count(p) AS n",
        None,
    ),
    (
        "kleene_or_null_lhs",
        "social_graph",
        "MATCH (p:Person) WHERE p.email = 'never' OR p.city = 'Oslo' RETURN p.name AS n ORDER BY n",
        None,
    ),
    (
        "kleene_and_null_lhs",
        "social_graph",
        "MATCH (p:Person) WHERE p.email <> 'never' AND p.city = 'Oslo' RETURN p.name AS n ORDER BY n",
        None,
    ),
    # ── B5: labels() consumer invariants (single-label model lock-in) ──
    (
        "labels_in",
        "social_graph",
        "MATCH (n) WHERE 'Person' IN labels(n) RETURN count(n) AS n",
        None,
    ),
    (
        "labels_size",
        "social_graph",
        "MATCH (n:Person) RETURN size(labels(n)) AS s ORDER BY s LIMIT 1",
        None,
    ),
    (
        "labels_index",
        "social_graph",
        "MATCH (n:Person) RETURN labels(n)[0] AS l LIMIT 1",
        None,
    ),
    # Map subscript by string key (IndexAccess string-index path,
    # added 0.10.14). Integer index → list; string index → map/node key.
    (
        "map_literal_string_subscript",
        "social_graph",
        "RETURN {x: 1}['x'] AS r",
        None,
    ),
    (
        "node_dynamic_property_subscript",
        "social_graph",
        "MATCH (n:Person) RETURN n['name'] AS r ORDER BY r LIMIT 1",
        None,
    ),
    ("in_list", "social_graph", "MATCH (p:Person) WHERE p.city IN ['Oslo', 'Bergen'] RETURN count(p) AS n", None),
    (
        "empty_in_parameter",
        "social_graph",
        "MATCH (p:Person) WHERE p.city IN $cities RETURN count(p) AS n",
        {"cities": []},
    ),
    (
        "nonindexed_in_opposite_id_anchor",
        "social_graph",
        "MATCH (a:Person)-[:KNOWS]->(b:Person {id: 2}) WHERE a.city IN ['Oslo'] RETURN a.name AS a, b.name AS b",
        None,
    ),
    (
        "predicate_stack",
        "social_graph",
        "MATCH (p:Person) WHERE (p.age > 25 AND p.city = 'Oslo') "
        "OR (p.age > 40 AND p.salary > 90000) RETURN p.name AS n ORDER BY n",
        None,
    ),
    # ── ORDER BY referencing RETURN aliases (regression for fuse_node_scan_top_k bug) ──
    # Before the fix, RETURN <expr> AS h ORDER BY h LIMIT k silently
    # produced empty rows: fuse_node_scan_top_k's sort-key evaluator
    # couldn't resolve RETURN aliases. Caught by the differential harness
    # (probe of broader query shapes); bisected to fuse_node_scan_top_k.
    (
        "string_concat_order_alias",
        "social_graph",
        "MATCH (p:Person) RETURN p.name + '@' + p.city AS handle ORDER BY handle LIMIT 5",
        None,
    ),
    ("order_by_return_alias", "social_graph", "MATCH (p:Person) RETURN p.name AS h ORDER BY h DESC LIMIT 5", None),
    (
        "order_by_expr",
        "social_graph",
        "MATCH (p:Person) RETURN p.name AS n, p.salary AS s ORDER BY p.salary - p.age * 1000 DESC LIMIT 5",
        None,
    ),
    (
        "exists_inline",
        "social_graph",
        "MATCH (p:Person) WHERE EXISTS { (p)-[:KNOWS]->() } RETURN p.name AS n ORDER BY n",
        None,
    ),
    (
        "exists_filter",
        "social_graph",
        "MATCH (p:Person) WHERE EXISTS { (p)-[:WORKS_AT]->(c:Company {industry: 'Tech'}) } "
        "RETURN p.name AS n ORDER BY n",
        None,
    ),
    (
        "not_exists",
        "social_graph",
        "MATCH (p:Person) WHERE NOT EXISTS { (p)-[:KNOWS]->() } RETURN p.name AS n ORDER BY n",
        None,
    ),
    (
        "having_basic",
        "social_graph",
        "MATCH (p:Person) WITH p.city AS c, count(p) AS n WHERE n > 4 RETURN c, n ORDER BY c",
        None,
    ),
    (
        "aggregate_of_aggregate",
        "social_graph",
        "MATCH (p:Person) WITH p.city AS c, count(p) AS n RETURN avg(n) AS avg_per_city, max(n) AS biggest",
        None,
    ),
    (
        "where_after_agg",
        "social_graph",
        "MATCH (p:Person)-[:WORKS_AT]->(c:Company) WITH c, count(p) AS hires "
        "WHERE hires >= 4 RETURN c.name AS n, hires ORDER BY n",
        None,
    ),
    (
        # Typed source, **untyped** group target — the shape that routes to
        # `try_fast_with_aggregate_via_histogram`'s typed-source branch, which
        # reads the disk backend's conn-type edge scan. On a disk graph built by
        # `enable_disk_mode` that scan found nothing (no `conn_type_index_*` is
        # ever written for a converted graph) and the fused path returned zero
        # rows while the naive path returned the right ones. This corpus is
        # in-memory only, so the disk half is pinned by the golden test
        # `test_disk_mutation_roundtrip.py::
        # test_converted_disk_graph_fused_aggregate_golden`; the entry here
        # keeps the *shape* under differential watch.
        "typed_source_untyped_group_count",
        "social_graph",
        "MATCH (p:Person)-[:WORKS_AT]->(c) WITH c, count(p) AS hires RETURN hires ORDER BY hires",
        None,
    ),
    # ── multi-pattern within a single MATCH (regression for self-join + LIMIT bug) ──
    # Before the fix, push_limit_into_match accepted single-MATCH queries
    # but didn't check single-pattern, so multi-pattern + WHERE + LIMIT
    # silently dropped rows. Bisects to push_limit_into_match +
    # optimize_pattern_start_node before the fix. The ORDER BY makes the
    # surfacing deterministic so the test compares row identity.
    (
        "self_join_limit",
        "social_graph",
        "MATCH (p:Person)-[:KNOWS]->(q:Person), (p)-[:KNOWS]->(r:Person) "
        "WHERE q <> r RETURN p.name AS n, q.name AS q, r.name AS r "
        "ORDER BY p.name, q.name, r.name LIMIT 5",
        None,
    ),
    (
        "shortest_typed",
        "social_graph",
        "MATCH p = shortestPath((a:Person {person_id:1})-[:KNOWS*..5]-(b:Person {person_id:10})) RETURN length(p) AS L",
        None,
    ),
    # B4: undirected shortestPath over a graph that has bidirectional
    # neighbours (KNOWS edges chain forward, but the undirected
    # traversal sees both directions). Pre-fix, `filtered_neighbors_undirected`
    # returned duplicate entries; the visited bitmap masked the
    # wrong-answer symptom for shortestPath but DFS-style enumeration
    # paid wasted work per duplicate. Locking the count here guards
    # against a future regression that surfaces the duplicate.
    (
        "shortest_undirected_dense",
        "social_graph",
        "MATCH p = shortestPath((a:Person {person_id:1})-[*..6]-(b:Person {person_id:20})) RETURN length(p) AS L",
        None,
    ),
    # Zero-length variable-length path: `[:R*0..N]` matches the anchor
    # itself at length 0, then each non-zero hop. The 0-length arm has
    # historically been a planner gotcha (it's the only path-pattern
    # shape that admits the anchor into the result set without an edge).
    # Pinning in the corpus so both optimizer paths agree on the result.
    (
        "zero_length_var_path",
        "social_graph",
        "MATCH (a:Person {person_id: 1})-[:KNOWS*0..2]->(b:Person) RETURN b.person_id AS r ORDER BY r",
        None,
    ),
    # ── OPTIONAL MATCH … WHERE (predicate scoped to the clause) ──
    #
    # The predicate lives inside the OPTIONAL MATCH, so the pushdown pass
    # rewrites it into the optional pattern (`push_where_into_match` on the
    # scoped form) while the unoptimised path evaluates it per candidate in
    # the executor. Both must null-extend the same rows: a pushdown that
    # dropped instead of null-extending, or a fusion that counted the
    # excluded candidates, diverges here. Row ORDER BY includes the optional
    # column so NULL placement is pinned.
    (
        "optional_scoped_where_cmp",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(f:Person) WHERE f.age > 35 "
        "RETURN p.name AS n, f.name AS fn ORDER BY n, fn",
        None,
    ),
    (
        "optional_scoped_where_equality",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(f:Person) WHERE f.city = 'Oslo' "
        "RETURN p.name AS n, f.name AS fn ORDER BY n, fn",
        None,
    ),
    (
        "optional_scoped_where_param",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(f:Person) WHERE f.age >= $floor "
        "RETURN p.name AS n, f.name AS fn ORDER BY n, fn",
        {"floor": 38},
    ),
    (
        "optional_scoped_where_correlated",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(f:Person) WHERE f.city = p.city "
        "RETURN p.name AS n, f.name AS fn ORDER BY n, fn",
        None,
    ),
    (
        "optional_scoped_where_outer_only",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(f:Person) WHERE p.age > 35 "
        "RETURN p.name AS n, f.name AS fn ORDER BY n, fn",
        None,
    ),
    (
        "optional_scoped_where_or_chain",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(f:Person) "
        "WHERE f.age = 38 OR f.age = 39 OR f.age = 40 "
        "RETURN p.name AS n, f.name AS fn ORDER BY n, fn",
        None,
    ),
    (
        "optional_scoped_where_count",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(f:Person) WHERE f.age > 35 "
        "RETURN p.name AS n, count(f) AS k ORDER BY n",
        None,
    ),
    (
        "optional_scoped_where_edge_var_count",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[r:KNOWS]->(f:Person) WHERE f.age > 35 "
        "WITH p, count(r) AS k RETURN p.name AS n, k ORDER BY n",
        None,
    ),
    (
        "two_optional_scoped_wheres",
        "social_graph",
        "MATCH (p:Person) "
        "OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company) WHERE c.name = 'TechCorp' "
        "OPTIONAL MATCH (p)-[:KNOWS]->(f:Person) WHERE f.age > 38 "
        "RETURN p.name AS n, c.name AS cn, f.name AS fn ORDER BY n, cn, fn",
        None,
    ),
    (
        "optional_scoped_where_then_with_where",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(f:Person) WHERE f.age > 35 "
        "WITH p, f WHERE p.age < 25 RETURN p.name AS n, f.name AS fn ORDER BY n, fn",
        None,
    ),
    (
        "two_optional_match",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:WORKS_AT]->(c) OPTIONAL MATCH (p)-[:KNOWS]->(f) "
        "RETURN p.name AS n, count(DISTINCT c) AS Cs, count(DISTINCT f) AS Fs "
        "ORDER BY n LIMIT 5",
        None,
    ),
    (
        "arithmetic_agg",
        "social_graph",
        "MATCH (p:Person) RETURN p.city AS c, avg(p.age) AS avg_age, max(p.age) - min(p.age) AS spread ORDER BY c",
        None,
    ),
    (
        "collect_size",
        "social_graph",
        "MATCH (p:Person) WITH p.city AS c, collect(p.name) AS names RETURN c, size(names) AS n ORDER BY c",
        None,
    ),
    ("label_check", "social_graph", "MATCH (n) WHERE n:Person RETURN count(n) AS n", None),
    ("id_function", "social_graph", "MATCH (p:Person) WHERE id(p) IS NOT NULL RETURN count(p) AS n", None),
    (
        "inline_and_where",
        "social_graph",
        "MATCH (p:Person {city: 'Oslo'}) WHERE p.age > 25 RETURN p.name AS n ORDER BY n",
        None,
    ),
    (
        "three_hop_count",
        "social_graph",
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)-[:KNOWS]->(d:Person) RETURN count(*) AS n",
        None,
    ),
    ("with_star", "social_graph", "MATCH (p:Person) WITH * WHERE p.age > 35 RETURN p.name AS n ORDER BY n", None),
    (
        "count_subquery_top_k",
        "social_graph",
        "MATCH (p:Person) WITH p, count{(p)-[:KNOWS]->()} AS deg "
        "WHERE deg > 0 RETURN p.name AS n, deg ORDER BY deg DESC, n LIMIT 5",
        None,
    ),
    (
        "list_comp_after_collect",
        "social_graph",
        "MATCH (p:Person) WITH collect(p.age) AS ages RETURN [a IN ages WHERE a > 30 | a + 1] AS bumped",
        None,
    ),
    (
        "shortest_with_length",
        "social_graph",
        "MATCH p = shortestPath((a:Person {person_id:1})-[:KNOWS*..5]-(b:Person {person_id:10})) "
        "RETURN length(p) AS L, size(nodes(p)) AS hops",
        None,
    ),
    (
        "list_param_in",
        "social_graph",
        "MATCH (p:Person) WHERE p.city IN $cities RETURN p.name AS n ORDER BY n",
        {"cities": ["Oslo", "Bergen"]},
    ),
    # `n.id IN $ids` (param) → index_selection pushes an `id IN [...]` matcher
    # so the scan anchors on the id index (instead of a full type scan +
    # post-filter), and rewrites the surviving WHERE to the O(1) InLiteralSet
    # form. Optimised must equal naive.
    (
        "id_in_param_anchored",
        "social_graph",
        "MATCH (p:Person)-[:KNOWS]-(f:Person) WHERE p.id IN $ids RETURN f.name AS n ORDER BY n",
        {"ids": [3, 7, 11, 15]},
    ),
    # `MATCH (n) WHERE n.id IN $ids RETURN count(n)` — fuse_node_scan_aggregate
    # must BAIL on an id-anchorable WHERE so the id-index anchoring drives the
    # scan instead of a full node sweep. Trigger shape for the bail; optimised
    # (anchored count) must equal naive (full scan).
    (
        "id_in_count_bails_fusion",
        "social_graph",
        "MATCH (p:Person) WHERE p.id IN $ids RETURN count(p) AS n",
        {"ids": [3, 7, 11, 15, 999999]},
    ),
    # `n.id = literal` count must also bail the fusion and anchor.
    (
        "id_eq_count_bails_fusion",
        "social_graph",
        "MATCH (p:Person) WHERE p.id = 7 RETURN count(p) AS n",
        None,
    ),
    # A repeated element in an id-anchored `IN` list must not repeat its node.
    # The anchor is driven by the list — one index probe per element — so
    # `[3, 3, 7]` bound Person 3 twice and `count(p)` answered 3 where the
    # naive scan (which filters each node once) answers 2. Both the count and
    # the row form are pinned: `_normalize` sorts but keeps multiplicity, so
    # the row entry sees the duplicate directly.
    (
        "id_in_duplicate_entries_count",
        "social_graph",
        "MATCH (p:Person) WHERE p.id IN [3, 3, 7, 7, 7] RETURN count(p) AS n",
        None,
    ),
    (
        "id_in_duplicate_entries_rows",
        "social_graph",
        "MATCH (p:Person) WHERE p.id IN [3, 3, 7] RETURN p.name AS n",
        None,
    ),
    # Coercion-equal spellings are two distinct list elements resolving to one
    # node, so deduping the *list* would not have covered this one.
    (
        "id_in_coercion_equal_entries",
        "social_graph",
        "MATCH (p:Person) WHERE p.id IN $ids RETURN p.name AS n",
        {"ids": [3, 3.0, 7]},
    ),
    # The duplicate must survive an expansion too: the anchored start set
    # feeds the expansion, and a doubled seed doubles every row it produces.
    (
        "id_in_duplicate_entries_expanded",
        "social_graph",
        "MATCH (p:Person)-[:KNOWS]->(f:Person) WHERE p.id IN [3, 3, 7] RETURN f.name AS n",
        None,
    ),
    # ── IN membership above the indexing threshold ──
    # Lists longer than the linear/indexed cut-off take the coercion-
    # normalized MembershipSet instead of a per-row `values_equal` scan, and
    # constant folding on the fused MATCH+WHERE path rewrites both the
    # all-literal and the `$param` list into the indexed form. Optimised must
    # equal naive for every element shape the normalization folds: the
    # numeric family, NULL elements, mixed types, and a NULL-valued property.
    (
        "in_big_literal_list",
        "social_graph",
        "MATCH (p:Person) WHERE p.age IN [22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 99] RETURN p.name AS n ORDER BY n",
        None,
    ),
    (
        "in_big_param_list",
        "social_graph",
        "MATCH (p:Person) WHERE p.age IN $ages RETURN p.name AS n ORDER BY n",
        {"ages": [22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 99]},
    ),
    (
        "in_big_list_numeric_coercion",
        "social_graph",
        "MATCH (p:Person) WHERE p.age IN $ages RETURN count(p) AS n",
        {"ages": [22.0, 24.0, 26.5, 28, 30, 32, 34, 36, 38, 40, 99]},
    ),
    (
        "in_big_list_mixed_types",
        "social_graph",
        "MATCH (p:Person) WHERE p.age IN [22, '24', 26.0, true, null, 30, 32, 34, 36, 38, 40] RETURN count(p) AS n",
        None,
    ),
    (
        "in_big_list_with_null_element",
        "social_graph",
        "MATCH (p:Person) WHERE p.age IN [22, 24, 26, 28, 30, 32, 34, 36, 38, 40, null] RETURN p.name AS n ORDER BY n",
        None,
    ),
    (
        "not_in_big_list_with_null_element",
        "social_graph",
        "MATCH (p:Person) WHERE NOT p.age IN [22, 24, 26, 28, 30, 32, 34, 36, 38, 40, null] RETURN count(p) AS n",
        None,
    ),
    # `email` is NULL on odd-numbered persons: a NULL left-hand side must
    # stay UNKNOWN (never TRUE) whichever membership strategy runs.
    (
        "in_big_list_null_property",
        "social_graph",
        "MATCH (p:Person) WHERE p.email IN $emails RETURN count(p) AS n",
        {
            "emails": [
                "person2@test.com",
                "person4@test.com",
                "person6@test.com",
                "person8@test.com",
                "person10@test.com",
                "person12@test.com",
                "person14@test.com",
                "person16@test.com",
                "person18@test.com",
                "person20@test.com",
                None,
            ]
        },
    ),
    # `n.id` is stored as a UniqueId; the list is integral floats, which only
    # match through the numeric normalization.
    (
        "in_big_list_id_float_coercion",
        "social_graph",
        "MATCH (p:Person) WHERE p.id IN $ids RETURN count(p) AS n",
        {"ids": [3.0, 7.0, 11.0, 15.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]},
    ),
    # A list bound by WITH is a general RHS expression (`InExpression`), not a
    # foldable literal — the per-row path, checked against the same corpus.
    (
        "in_list_expression_rhs",
        "social_graph",
        "MATCH (p:Person) WITH p, [22, 24, 26, 28, 30, 32, 34, 36, 38, 40, null] AS ages "
        "WHERE p.age IN ages RETURN count(p) AS n",
        None,
    ),
    (
        "in_big_list_post_with_projection",
        "social_graph",
        "MATCH (p:Person) WITH p.age AS a WHERE a IN [22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 99] RETURN count(a) AS n",
        None,
    ),
    (
        "param_arithmetic",
        "social_graph",
        "MATCH (p:Person) WHERE p.age > $threshold + 5 RETURN count(p) AS n",
        {"threshold": 25},
    ),
    # ── Multi-WITH chain (catches multi-pass WITH folding) ──
    (
        "multi_with_chain",
        "social_graph",
        "MATCH (p:Person) WITH p WHERE p.age > 25 WITH p, p.salary AS s "
        "WHERE s > 80000 WITH p, s ORDER BY s DESC RETURN p.name AS n, s LIMIT 5",
        None,
    ),
    (
        "distinct_order_same_expr",
        "social_graph",
        "MATCH (p:Person) RETURN DISTINCT p.city AS c ORDER BY p.city",
        None,
    ),
    (
        "optional_count_star_group",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company) "
        "WITH p.city AS city, count(c) AS jobs RETURN city, jobs ORDER BY city",
        None,
    ),
    (
        "having_multi_key",
        "social_graph",
        "MATCH (p:Person)-[:KNOWS]->(q:Person) "
        "WITH p.city AS pc, q.city AS qc, count(*) AS edges "
        "WHERE edges > 0 RETURN pc, qc, edges ORDER BY pc, qc",
        None,
    ),
    # ── ORDER BY computed expression on alias (regression for fuse_node_scan_top_k) ──
    (
        "order_by_alias_arithmetic",
        "social_graph",
        "MATCH (p:Person) RETURN p.name AS n, p.age * 2 AS bumped ORDER BY bumped DESC LIMIT 5",
        None,
    ),
    # ── safe LIMIT pushdown over an unfiltered node-only cartesian ──
    (
        "cartesian_node_scans_limit",
        "social_graph",
        "MATCH (p:Person), (c:Company) RETURN p.name AS p, c.name AS c LIMIT 100",
        None,
    ),
    (
        "string_op_filter_order",
        "social_graph",
        "MATCH (p:Person) WHERE p.name STARTS WITH 'Person_' RETURN p.name AS n ORDER BY size(p.name) DESC, n LIMIT 5",
        None,
    ),
    (
        "coalesce_email",
        "social_graph",
        "MATCH (p:Person) RETURN p.name AS n, coalesce(p.email, 'none') AS e ORDER BY n LIMIT 5",
        None,
    ),
    # ── ORDER BY aggregate alias with secondary sort (regression for tie-break) ──
    (
        "order_by_agg_alias_stable",
        "social_graph",
        "MATCH (p:Person) WITH p.city AS city, count(*) AS n RETURN city, n ORDER BY n DESC, city LIMIT 3",
        None,
    ),
    (
        "case_in_agg",
        "social_graph",
        "MATCH (p:Person) RETURN p.city AS c, sum(CASE WHEN p.age > 30 THEN 1 ELSE 0 END) AS olders ORDER BY c",
        None,
    ),
    (
        "nested_func_calls",
        "social_graph",
        "MATCH (p:Person) RETURN p.name AS n, toUpper(p.city) AS c ORDER BY n LIMIT 5",
        None,
    ),
    (
        "not_predicate",
        "social_graph",
        "MATCH (p:Person) WHERE NOT p.city = 'Oslo' RETURN count(p) AS n",
        None,
    ),
    (
        "where_edge_node_mix",
        "social_graph",
        "MATCH (p:Person)-[r:KNOWS]->(q:Person) WHERE r.since > 2017 AND q.age > 25 RETURN count(*) AS n",
        None,
    ),
    (
        "count_subq_in_where",
        "social_graph",
        "MATCH (p:Person) WHERE count{(p)-[:KNOWS]->()} > 2 RETURN p.name AS n ORDER BY n",
        None,
    ),
    # ── integer div/mod at the i64 boundary: in range, so both sides agree.
    # `i64::MIN / -1` and `i64::MIN % -1` are now query errors rather than
    # wrapped values, and an erroring query is not a differential case — the
    # absolute goldens in `executor::tests::semantics` own that contract.
    ("div_at_boundary", "small_graph", "RETURN (-9223372036854775807 - 1) / 2 AS n", None),
    ("mod_at_boundary", "small_graph", "RETURN (-9223372036854775807 - 1) % 7 AS n", None),
    (
        "expr_filter",
        "social_graph",
        "MATCH (p:Person) WHERE p.salary / p.age > 2000 RETURN p.name AS n ORDER BY n LIMIT 5",
        None,
    ),
    (
        "with_expr_filter_sort",
        "social_graph",
        "MATCH (p:Person) WITH p, p.salary - p.age * 1000 AS net "
        "WHERE net > 50000 RETURN p.name AS n, net ORDER BY net DESC, n LIMIT 5",
        None,
    ),
    (
        "multi_optional_having",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(f) "
        "OPTIONAL MATCH (p)-[:WORKS_AT]->(c) "
        "WITH p, count(DISTINCT f) AS friends, count(DISTINCT c) AS jobs "
        "WHERE friends > 0 RETURN p.name AS n, friends, jobs ORDER BY n LIMIT 5",
        None,
    ),
    (
        "cohort_then_match",
        "social_graph",
        "MATCH (p:Person) WITH p ORDER BY p.salary DESC LIMIT 5 "
        "MATCH (p)-[:WORKS_AT]->(c:Company) RETURN p.name AS n, c.name AS c ORDER BY n",
        None,
    ),
    # ── multi-MATCH cartesian + count(*) (regression for desugar fix) ──
    (
        "multi_match_count_star",
        "social_graph",
        "MATCH (p:Person) MATCH (q:Person) WHERE p.person_id < q.person_id AND p.city = q.city RETURN count(*) AS n",
        None,
    ),
    # ── affected_tests procedure (0.9.34) ──
    # Guards against the optimizer rewriting away rows surrounding a CALL
    # to affected_tests — the procedure itself walks IMPORTS inbound to
    # find reachable test files. Trigger shape ships from the plan.
    (
        "affected_tests_simple",
        "file_imports_graph",
        "CALL affected_tests({files: ['src/util.py']}) YIELD test_file, depth "
        "RETURN test_file, depth ORDER BY test_file",
        None,
    ),
    (
        "affected_tests_transitive",
        "file_imports_graph",
        "CALL affected_tests({files: ['src/a.py']}) YIELD test_file RETURN test_file ORDER BY test_file",
        None,
    ),
    # ── path-decomposition functions w/ property-rich nodes() (0.9.35) ──
    # Guards against the optimizer rewriting around a variable-length
    # MATCH that consumes the per-node property dicts from nodes(p).
    (
        "path_unwind_nodes_with_property_access",
        "social_graph",
        "MATCH p = (a:Person {person_id: 1})-[:KNOWS*1..2]->(b:Person) "
        "UNWIND nodes(p) AS n "
        "RETURN n.name AS name ORDER BY name",
        None,
    ),
    # ── refresh_stats() procedure (0.9.35) ──
    # Confirms the optimizer doesn't rewrite around a CALL whose output
    # rows depend on the freshly-computed label-pair triples.
    (
        "refresh_stats_basic",
        "file_imports_graph",
        "CALL refresh_stats() YIELD src_type, edge_type, tgt_type, count "
        "RETURN edge_type, count ORDER BY edge_type, count",
        None,
    ),
    (
        "with_then_multi_pattern_cross_join",
        "social_graph",
        "WITH 1 AS x MATCH (a:Person), (c:Company) RETURN a.name AS a, c.name AS c ORDER BY a, c LIMIT 5",
        None,
    ),
    # Regression: `MATCH (n {id: x.id})` where x is an UNWIND'd map must resolve
    # the member per row (previously matched nothing).
    (
        "unwind_inline_map_member",
        "social_graph",
        "UNWIND [{pid: 1}, {pid: 2}] AS x MATCH (p:Person {person_id: x.pid}) RETURN p.name AS n ORDER BY n",
        None,
    ),
    # ── ready_set() dependency-frontier procedure ──
    # A node is "ready" when every outgoing-KNOWS neighbour satisfies the
    # `done` predicate. Confirms the optimizer leaves the CALL's output
    # untouched (aggregated to a count so the comparison is order-stable).
    (
        "ready_set_basic",
        "social_graph",
        "CALL ready_set({relationship: 'KNOWS', done: 'n.age > 30'}) "
        "YIELD node, dependency_count "
        "RETURN count(node) AS ready, sum(dependency_count) AS deps",
        None,
    ),
    # ── reorder_match_clauses w/ label-pair selectivity (0.9.35) ──
    # Two MATCH clauses where the label-pair cardinalities differ
    # significantly. With the new selectivity-aware branch the planner
    # picks the (Person, WORKS_AT, Company) clause first because
    # WORKS_AT-to-Company is a smaller pair than KNOWS-between-Persons.
    # Optimizer-on vs optimizer-off must produce identical rows.
    (
        "label_pair_reorder_two_match",
        "social_graph",
        "MATCH (p:Person {person_id: 1})-[:KNOWS]->(q:Person) "
        "MATCH (p:Person {person_id: 1})-[:WORKS_AT]->(c:Company) "
        "RETURN q.name AS q, c.name AS c ORDER BY q, c",
        None,
    ),
    # ── db.* schema-introspection procedures ──
    # Pin the canonical YIELD shapes against optimizer rewrites and
    # cross-mode parity. These have no planner pass to validate, but
    # the corpus also serves as the cross-mode oracle.
    (
        "db_labels_basic",
        "social_graph",
        "CALL db.labels() YIELD label RETURN label ORDER BY label",
        None,
    ),
    (
        "db_relationship_types_basic",
        "social_graph",
        "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType",
        None,
    ),
    (
        "db_labels_with_where_postfilter",
        "social_graph",
        "CALL db.labels() YIELD label WITH label WHERE label STARTS WITH 'C' RETURN label ORDER BY label",
        None,
    ),
    (
        "db_property_keys_basic",
        "social_graph",
        "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey ORDER BY propertyKey",
        None,
    ),
    (
        "db_schema_basic",
        "social_graph",
        "CALL db.schema() YIELD nodeType, properties RETURN nodeType, properties ORDER BY nodeType",
        None,
    ),
    # ── Multi-label (secondary-label) read paths ──────────────────────────
    # On a multi-label graph the label-dependent fusions are gated to the
    # general matcher path, so optimised==naive proves the fused fast-paths
    # that DO still fire (e.g. FusedCountTypedNode) agree with the matcher,
    # and that the gates don't drop/duplicate rows. (The matcher path itself
    # is pinned to an independent Python oracle in test_multi_label.py.)
    (
        "ml_count_secondary_label",
        "multi_label_graph",
        "MATCH (n:VIP) RETURN count(n) AS c",
        None,
    ),
    (
        "ml_count_typed_plus_secondary",
        "multi_label_graph",
        "MATCH (n:Person:VIP) RETURN count(n) AS c",
        None,
    ),
    (
        "ml_label_intersection_rows",
        "multi_label_graph",
        "MATCH (n:VIP:Staff) RETURN n.id AS id ORDER BY id",
        None,
    ),
    (
        "ml_secondary_label_property_filter",
        "multi_label_graph",
        "MATCH (n:VIP {name: 'Acme'}) RETURN n.id AS id",
        None,
    ),
    (
        "ml_edge_aggregate_secondary_peer",
        "multi_label_graph",
        "MATCH (a:Person)-[:KNOWS]->(b:VIP) RETURN a.id AS a, count(b) AS c ORDER BY a",
        None,
    ),
    (
        "ml_group_node_secondary",
        "multi_label_graph",
        "MATCH (a)-[:KNOWS]->(v:VIP) RETURN v.id AS v, count(a) AS c ORDER BY v",
        None,
    ),
    (
        "ml_label_predicate_where",
        "multi_label_graph",
        "MATCH (n:Person) WHERE n:VIP RETURN n.id AS id ORDER BY id",
        None,
    ),
    # Finer fusion gate (label-hardening P9): patterns whose types are not
    # themselves secondary labels FUSE on a multi-label graph — these pin
    # the newly-fused paths against the general path.
    (
        "ml_edge_aggregate_safe_types_fuse",
        "multi_label_graph",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.id AS a, count(b) AS c ORDER BY a",
        None,
    ),
    (
        "ml_with_aggregate_safe_types_fuse",
        "multi_label_graph",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) WITH a, count(b) AS c RETURN a.id AS id, c ORDER BY id",
        None,
    ),
    # And the case the gate must still refuse: an extra label on a pattern
    # node (the fused executor drops extra_labels — fusing this would count
    # every Person peer, not just the :VIP ones).
    (
        "ml_edge_aggregate_extra_label_bails",
        "multi_label_graph",
        "MATCH (a:Person)-[:KNOWS]->(b:Person:VIP) RETURN a.id AS a, count(b) AS c ORDER BY a",
        None,
    ),
    # Label alternation `(n:A|B)` (E4): union across primary types and
    # secondary carriers, deduped (P5 is :Person primary + :VIP + :Staff;
    # C1 is :Company primary + :VIP).
    (
        "ml_alt_two_primaries",
        "multi_label_graph",
        "MATCH (n:Person|Company) RETURN n.id AS id ORDER BY id",
        None,
    ),
    (
        "ml_alt_primary_and_secondary_dedup",
        "multi_label_graph",
        "MATCH (n:Person|VIP) RETURN count(n) AS c",
        None,
    ),
    (
        "ml_alt_secondary_pair",
        "multi_label_graph",
        "MATCH (n:VIP|Staff) RETURN n.id AS id ORDER BY id",
        None,
    ),
    (
        "ml_alt_property_filter",
        "multi_label_graph",
        "MATCH (n:Person|Company {name: 'Acme'}) RETURN n.id AS id",
        None,
    ),
    # ── label alternation: count fusion + per-branch index probes (0.16.13) ──
    # Branch-disjointness decides the count fusion; index coverage decides the
    # probe. Both regimes of both are here, on fixtures built for them.
    (
        "alt_count_two_branches",
        "alternation_graph",
        "MATCH (n:Student|Teacher) RETURN count(n) AS c",
        None,
    ),
    (
        "alt_count_three_branches",
        "alternation_graph",
        "MATCH (n:Student|Teacher|Staff) RETURN count(n) AS c",
        None,
    ),
    (
        "alt_count_foreign_secondary_label",
        "alternation_foreign_label_graph",
        "MATCH (n:Student|Teacher) RETURN count(n) AS c",
        None,
    ),
    # Overlapping branches: the fusion must bail, so this pins the count the
    # matcher's union-and-dedup path produces.
    (
        "alt_count_overlapping_branches",
        "alternation_overlap_graph",
        "MATCH (n:Student|Teacher) RETURN count(n) AS c",
        None,
    ),
    (
        "alt_equality_probe_all_branches_indexed",
        "alternation_graph",
        "MATCH (n:Student|Teacher {email: 'tea@x'}) RETURN n.id AS id",
        None,
    ),
    (
        "alt_equality_probe_param",
        "alternation_graph",
        "MATCH (n:Student|Teacher|Staff {email: $e}) RETURN n.id AS id",
        {"e": "sam@x"},
    ),
    # `dept` is indexed on :Student only — the all-or-nothing coverage rule
    # declines rather than dropping :Teacher's rows.
    (
        "alt_equality_partial_index_declines",
        "alternation_graph",
        "MATCH (n:Student|Teacher {dept: 'Sci'}) RETURN n.id AS id ORDER BY id",
        None,
    ),
    # The soundness case: node 1 is primary :Student and secondary :Teacher, so
    # the probe reaches it through both branches and must emit it once.
    (
        "alt_equality_probe_overlapping_branches",
        "alternation_overlap_graph",
        "MATCH (n:Student|Teacher {email: 'ann@x'}) RETURN n.id AS id",
        None,
    ),
    (
        "ml_alt_edge_endpoint",
        "multi_label_graph",
        "MATCH (a:Person)-[:KNOWS]->(b:VIP|Staff) RETURN a.id AS a, b.id AS b ORDER BY a, b",
        None,
    ),
    # Ontology closures (0.16.13). The corpus had no ontology shape at all
    # while the closure probe was structurally unable to fire; these pin the
    # probe against the path it replaces. The `WHERE` spellings are the ones
    # that genuinely differ between the two runs — pushdown moves the equality
    # into the pattern under the optimizer and leaves it a post-filter without.
    (
        "onto_supertype_equality_probe",
        "ontology_closure_graph",
        "MATCH (p:Person {email: 'tea@x'}) RETURN p.id AS id ORDER BY id",
        None,
    ),
    (
        "onto_supertype_equality_other_member",
        "ontology_closure_graph",
        "MATCH (p:Person {email: 'bo@x'}) RETURN p.id AS id ORDER BY id",
        None,
    ),
    (
        "onto_supertype_equality_absent",
        "ontology_closure_graph",
        "MATCH (p:Person {email: 'nobody@x'}) RETURN p.id AS id ORDER BY id",
        None,
    ),
    (
        "onto_supertype_equality_where_pushdown",
        "ontology_closure_graph",
        "MATCH (p:Person) WHERE p.email = 'tea@x' RETURN p.id AS id ORDER BY id",
        None,
    ),
    (
        "onto_supertype_equality_param",
        "ontology_closure_graph",
        "MATCH (p:Person {email: $e}) RETURN p.id AS id ORDER BY id",
        {"e": "ann@x"},
    ),
    (
        "onto_supertype_id_lookup",
        "ontology_closure_graph",
        "MATCH (p:Person {id: 11}) RETURN p.title AS t",
        None,
    ),
    (
        "onto_supertype_id_lookup_absent",
        "ontology_closure_graph",
        "MATCH (p:Person {id: 999}) RETURN p.title AS t",
        None,
    ),
    # Partial coverage: `dept` is indexed on Student only, so the probe must
    # decline wholesale — a union without Teacher's rows would drop id 10.
    (
        "onto_supertype_partial_index_decline",
        "ontology_closure_graph",
        "MATCH (p:Person {dept: 'Sci'}) RETURN p.id AS id ORDER BY id",
        None,
    ),
    # Two properties, one covered one not: the covered index anchors and the
    # rest is a filter.
    (
        "onto_supertype_multi_property",
        "ontology_closure_graph",
        "MATCH (p:Person {email: 'cy@x', dept: 'Art'}) RETURN p.id AS id ORDER BY id",
        None,
    ),
    (
        "onto_supertype_unindexed_property",
        "ontology_closure_graph",
        "MATCH (p:Person {title: 'Uli'}) RETURN p.id AS id ORDER BY id",
        None,
    ),
    (
        "onto_subtype_equality_control",
        "ontology_closure_graph",
        "MATCH (s:Student {email: 'ann@x'}) RETURN s.id AS id ORDER BY id",
        None,
    ),
    (
        "onto_supertype_count",
        "ontology_closure_graph",
        "MATCH (p:Person) RETURN count(p) AS c",
        None,
    ),
    (
        "onto_supertype_expand",
        "ontology_closure_graph",
        "MATCH (p:Person {email: 'ann@x'}) RETURN p.id AS id, p.dept AS d",
        None,
    ),
    # Open label: `:Class` carries `:Person` from outside the closure, so no
    # descendant probe covers it and every shape returns to the scan.
    (
        "onto_open_label_equality",
        "ontology_open_label_graph",
        "MATCH (p:Person {email: 'math@x'}) RETURN p.id AS id ORDER BY id",
        None,
    ),
    (
        "onto_open_label_member_equality",
        "ontology_open_label_graph",
        "MATCH (p:Person {email: 'tea@x'}) RETURN p.id AS id ORDER BY id",
        None,
    ),
    (
        "onto_open_label_count",
        "ontology_open_label_graph",
        "MATCH (p:Person) RETURN count(p) AS c",
        None,
    ),
    # EXISTS fast path over a secondary-carried peer — pinned red-first
    # 2026-08-26: the primary-only compare answered zero rows.
    (
        "ml_exists_secondary_peer",
        "multi_label_graph",
        "MATCH (a:Person) WHERE EXISTS { MATCH (a)-[:KNOWS]->(:VIP) } RETURN a.id AS id ORDER BY id",
        None,
    ),
    # KG-2 soft keywords as names — these don't match the social_graph
    # fixture (no CONTAINS edges / labels), but they must PARSE, plan, and
    # execute consistently under optimised vs naive passes (the optimiser
    # must not choke on a keyword-named rel-type / label / property key).
    (
        "kw_rel_type_in_match",
        "social_graph",
        "MATCH (p:Person)-[:CONTAINS]->(q) RETURN count(q) AS n",
        None,
    ),
    (
        "kw_node_label",
        "social_graph",
        "MATCH (n:CONTAINS) RETURN count(n) AS n",
        None,
    ),
    (
        "kw_property_key",
        "social_graph",
        "MATCH (n {contains: 1}) RETURN count(n) AS n",
        None,
    ),
    (
        "kw_exists_subquery",
        "social_graph",
        "MATCH (p:Person) WHERE EXISTS { (p)-[:CONTAINS]->() } RETURN count(p) AS n",
        None,
    ),
    # ── Trig / math scalar functions (deterministic literal args) ──
    # Constant-foldable trig must produce identical rows with the
    # optimizer on and off — exercises the new sin/cos/atan2/degrees/
    # radians/cot/haversin arms through the folding path. randomUUID()
    # and the local-temporal "now" forms are intentionally excluded
    # (non-deterministic / wall-clock → would flake).
    ("trig_sin_cos", "social_graph", "MATCH (p:Person) RETURN sin(0) AS s, cos(0) AS c LIMIT 1", None),
    (
        "trig_degrees_radians",
        "social_graph",
        "MATCH (p:Person) RETURN degrees(pi()) AS d, radians(180) AS r LIMIT 1",
        None,
    ),
    ("trig_atan2", "social_graph", "MATCH (p:Person) RETURN atan2(1, 1) AS a LIMIT 1", None),
    ("trig_cot_haversin", "social_graph", "MATCH (p:Person) RETURN cot(1) AS c, haversin(0) AS h LIMIT 1", None),
    (
        "trig_on_property",
        "social_graph",
        "MATCH (p:Person) RETURN p.name AS n, sin(radians(p.age)) AS sa ORDER BY n",
        None,
    ),
    (
        "trig_null_propagation",
        "social_graph",
        "MATCH (p:Person) RETURN sin(null) AS s, atan2(null, 1) AS a LIMIT 1",
        None,
    ),
    # ── properties()/keys()/{.*} on an alias-bearing fixture ──
    # `small_graph` loads Person via non-literal id/title fields
    # (add_nodes(..., "person_id", "name")), so each node carries
    # `{id,title}_field_aliases`. properties(n)/keys(n)/n {.*} must
    # surface those recovered columns identically under optimiser-on
    # and optimiser-off (and match the canonical RETURN n shape).
    (
        "properties_aliased_node",
        "small_graph",
        "MATCH (p:Person) RETURN properties(p) AS props ORDER BY props.id",
        None,
    ),
    (
        "keys_aliased_node",
        "small_graph",
        "MATCH (p:Person) RETURN keys(p) AS ks ORDER BY p.person_id",
        None,
    ),
    (
        "map_projection_star_aliased_node",
        "small_graph",
        "MATCH (p:Person) RETURN p {.*} AS m ORDER BY m.id",
        None,
    ),
    # ── CALL { } uncorrelated subqueries ──
    # The body runs once and its rows cartesian-product with the outer
    # stream. CALL { } is opaque to the optimizer passes (the
    # body is optimized once locally), so these entries validate that the
    # run-once + cartesian-combine path is deterministic across the
    # optimizer-on / optimizer-off outer runs.
    (
        "call_uncorrelated_leading_count",
        "social_graph",
        "CALL { MATCH (n:Person) RETURN count(n) AS c } RETURN c",
        None,
    ),
    (
        "call_uncorrelated_cartesian_after_match",
        "social_graph",
        "MATCH (c:Company) CALL { MATCH (n:Person) RETURN count(n) AS pc } RETURN c.name AS cn, pc ORDER BY cn",
        None,
    ),
    (
        "call_uncorrelated_multi_row_inner",
        "social_graph",
        "MATCH (c:Company) WHERE c.name = 'TechCorp' "
        "CALL { MATCH (p:Person) WHERE p.age < 23 RETURN p.name AS pn } "
        "RETURN c.name AS cn, pn ORDER BY pn",
        None,
    ),
    (
        "call_uncorrelated_nested",
        "social_graph",
        "CALL { CALL { MATCH (n:Person) RETURN count(n) AS c } RETURN c AS cc } RETURN cc",
        None,
    ),
    # ── CALL { } correlated subqueries ──
    # The body is planned once and executed per outer row, seeded with the
    # imported variables only. The unoptimized run is the oracle for the
    # optimized run; both must agree on the inner-join cardinality + the
    # per-row aggregate values.
    (
        "call_correlated_aggregate",
        "social_graph",
        "MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS c } "
        "RETURN p.name AS pn, c ORDER BY pn",
        None,
    ),
    (
        "call_correlated_non_aggregating_multiplicity",
        "social_graph",
        "MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN f.name AS fn } "
        "RETURN p.name AS pn, fn ORDER BY pn, fn",
        None,
    ),
    (
        "call_correlated_empty_row_drop",
        "social_graph",
        # Person_20 has zero outgoing KNOWS → dropped by the non-aggregating
        # body's inner join (§1.3).
        "MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN f.name AS fn } RETURN p.name AS pn ORDER BY pn",
        None,
    ),
    (
        "call_correlated_multi_import",
        "social_graph",
        "MATCH (p:Person)-[:WORKS_AT]->(c:Company) "
        "CALL { WITH p, c MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS c2 } "
        "RETURN p.name AS pn, c.name AS cn, c2 ORDER BY pn",
        None,
    ),
    (
        "call_correlated_nested_in_uncorrelated",
        "social_graph",
        "CALL { MATCH (p:Person) WHERE p.age < 23 "
        "CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS c } "
        "RETURN p.name AS pn, c } RETURN pn, c ORDER BY pn",
        None,
    ),
    (
        "call_correlated_after_optional_match_miss",
        "social_graph",
        # An OPTIONAL MATCH that misses for some Persons (no WORKS_AT edge)
        # leaves the imported anchor `c` declared-but-null on those rows;
        # the correlated body anchors on it. Aggregating body → those rows
        # survive with count 0. The naive run is the oracle for the per-row
        # sentinel-vs-real-node seeding decision.
        "MATCH (p:Person) "
        "OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company) "
        "CALL { WITH c MATCH (c)<-[:WORKS_AT]-(co:Person) RETURN count(co) AS colleagues } "
        "RETURN p.name AS pn, colleagues ORDER BY pn",
        None,
    ),
    # ── CALL { } cross-clause barrier ──
    # These shapes would diverge optimized-vs-naive if a planner pass were
    # to treat CallSubquery as transparent — fusing through it, reordering
    # a MATCH across it, or pushing LIMIT/predicates past it. Each pairs a
    # CALL with a downstream/adjacent shape that the pass it targets would
    # otherwise rewrite. The naive (optimizer-off) run is the oracle.
    (
        # push_limit_into_match barrier: a CALL sits between the MATCH and
        # the RETURN+LIMIT, so the LIMIT must NOT be pushed into the MATCH
        # (the CALL's cartesian fan-out changes which rows the LIMIT keeps).
        "call_then_return_limit_barrier",
        "social_graph",
        "MATCH (p:Person) WHERE p.age < 25 "
        "CALL { MATCH (n:Person) RETURN count(n) AS tot } "
        "RETURN p.name AS pn, tot ORDER BY pn LIMIT 3",
        None,
    ),
    (
        # fuse_order_by_top_k barrier: a correlated CALL feeds an outer
        # ORDER BY ... LIMIT. The top-K fusion must see the CALL's output
        # column (`c`), not fuse through to the upstream MATCH.
        "call_correlated_then_order_by_limit",
        "social_graph",
        "MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS c } "
        "RETURN p.name AS pn, c ORDER BY c DESC, pn LIMIT 5",
        None,
    ),
    (
        # desugar_multi_match_return_aggregate / reorder_match_clauses
        # barrier: a CALL sits BETWEEN two MATCHes that the outer RETURN
        # aggregates over. The two MATCHes are NOT adjacent, so neither the
        # multi-match desugar nor the cross-clause MATCH reorder may treat
        # them as a contiguous span.
        "call_between_two_matches_aggregate",
        "social_graph",
        "MATCH (p:Person) WHERE p.age < 24 "
        "CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS fc } "
        "MATCH (p)-[:WORKS_AT]->(co:Company) "
        "RETURN co.name AS cn, sum(fc) AS total ORDER BY cn",
        None,
    ),
    (
        # fold_pass_through_with barrier: a pass-through `WITH p` precedes a
        # correlated `CALL { WITH p ... }`. Folding the WITH must not drop
        # the binding the CALL imports; the collect_clause_variables fix
        # records the CALL's import so the fold's downstream-ref check sees
        # `p` is still needed.
        "with_passthrough_before_correlated_call",
        "social_graph",
        "MATCH (p:Person)-[:WORKS_AT]->(co:Company) WITH p "
        "CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS fc } "
        "RETURN p.name AS pn, fc ORDER BY pn",
        None,
    ),
    (
        # aggregate-after-CALL: the outer RETURN aggregates the per-row
        # multiplicity the non-aggregating CALL produced. fuse_match_*_
        # aggregate must NOT absorb the upstream MATCH through the CALL.
        "aggregate_over_call_multiplicity",
        "social_graph",
        "MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN f.name AS fn } "
        "RETURN p.city AS city, count(fn) AS knows_count ORDER BY city",
        None,
    ),
    (
        # WITH-chain on BOTH sides of the CALL: a WITH narrows before, a
        # WITH re-projects after. Exercises the import-declaredness +
        # fold_pass_through_with interaction around a CALL in the middle of
        # a pipeline.
        "with_chain_around_call",
        "social_graph",
        "MATCH (p:Person) WHERE p.age >= 30 WITH p "
        "CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS fc } "
        "WITH p.city AS city, fc AS fc "
        "RETURN city, sum(fc) AS total ORDER BY city",
        None,
    ),
    (
        # uncorrelated CALL with its OWN body that the body-optimizer can
        # fuse (MATCH+RETURN count) — confirms body optimization (now in
        # the planner pass) agrees with the naive body. The outer LIMIT
        # after the cartesian must not push into the body.
        "call_uncorrelated_body_fusion_then_limit",
        "social_graph",
        "MATCH (c:Company) "
        "CALL { MATCH (n:Person) WHERE n.age > 30 RETURN count(n) AS pc } "
        "RETURN c.name AS cn, pc ORDER BY cn LIMIT 2",
        None,
    ),
    # ── CALL { } Neo4j-conformance shapes ──
    # These target the five shapes called out in the design's §5 Neo4j
    # conformance plan. They flow into scripts/cypher_conformance.py (which
    # imports DIFFERENTIAL_QUERIES) automatically — the next
    # `make neo4j-conformance` run diffs each against a live Neo4j 5. They
    # also run optimized-vs-naive here. Zero divergences expected for v1.
    (
        # Leading uncorrelated: the subquery runs once with no outer driver,
        # producing the single seed row × S subquery rows.
        "call_conf_leading_uncorrelated",
        "social_graph",
        "CALL { MATCH (p:Person) WHERE p.city = 'Oslo' RETURN p.name AS pn } RETURN pn ORDER BY pn",
        None,
    ),
    (
        # Cartesian combine: an outer MATCH × an uncorrelated subquery body
        # → R×S rows. Neo4j's uncorrelated-subquery cartesian semantics.
        "call_conf_cartesian_combine",
        "social_graph",
        "MATCH (c:Company) WHERE c.industry = 'Tech' "
        "CALL { MATCH (p:Person) WHERE p.age < 23 RETURN p.name AS pn } "
        "RETURN c.name AS cn, pn ORDER BY cn, pn",
        None,
    ),
    (
        # Correlated aggregate, count=0 row preserved: Person_20 has zero
        # outgoing KNOWS, but the aggregating body returns count(f)=0 (one
        # row), so the outer row SURVIVES with c=0 (§1.3). Neo4j agrees.
        "call_conf_correlated_aggregate_zero_preserved",
        "social_graph",
        "MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS c } "
        "RETURN p.name AS pn, c ORDER BY pn",
        None,
    ),
    (
        # OPTIONAL MATCH null import: an OPTIONAL MATCH that misses leaves the
        # imported anchor `f` NULL on those rows; the correlated body runs
        # with the NULL binding, the aggregating body yields count=0, the row
        # survives. Matches Neo4j's NULL-import semantics.
        "call_conf_optional_match_null_import",
        "social_graph",
        "MATCH (p:Person) "
        "OPTIONAL MATCH (p)-[:KNOWS]->(f) "
        "CALL { WITH f MATCH (f)-[:WORKS_AT]->(co:Company) RETURN count(co) AS jobs } "
        "RETURN p.name AS pn, jobs ORDER BY pn",
        None,
    ),
    (
        # ORDER BY + LIMIT inside the subquery body: per-row top-K. Each
        # outer Person imports into a body that orders its KNOWS targets by
        # age DESC and keeps the single oldest. Neo4j evaluates the body's
        # ORDER BY/LIMIT independently per outer row.
        "call_conf_order_limit_in_body",
        "social_graph",
        "MATCH (p:Person) WHERE p.age < 25 "
        "CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN f.name AS oldest ORDER BY f.age DESC LIMIT 1 } "
        "RETURN p.name AS pn, oldest ORDER BY pn",
        None,
    ),
    # ── fused count / distinct-hint regression shapes (0.12.x) ──────────
    (
        # push_distinct_into_match with a residual (multi-variable) WHERE:
        # `a.age + b.age > 50` can't be pushed into the pattern, so it is
        # fused into the MATCH as an inline predicate. The distinct-dedup
        # branch of execute_match previously skipped that predicate
        # entirely — the WHERE was silently dropped.
        "distinct_hint_residual_where",
        "social_graph",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.age + b.age > 50 RETURN DISTINCT b.name AS n",
        None,
    ),
    (
        # Fused OPTIONAL MATCH count with a property-filtered peer, on a
        # node that ALSO has edges of another connection type whose peer
        # passes the property filter (Person_1: KNOWS→Person_2 with
        # age=22, plus a WORKS_AT edge). try_count_simple_pattern's slow
        # path previously trusted edges_directed_filtered to filter the
        # connection type — a no-op on memory/mapped storage — so the
        # KNOWS edge was counted under the :WORKS_AT pattern.
        "optional_count_conn_type_postfilter",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:WORKS_AT]->(x {age: 22}) "
        "WITH p, count(x) AS c RETURN p.name AS n, c ORDER BY n",
        None,
    ),
    (
        # Fused OPTIONAL MATCH aggregate + count(*) with unmatched rows:
        # Person_20 has no outgoing KNOWS, so OPTIONAL MATCH emits one
        # null-padded row — count(*) must be 1, count(m) must be 0. The
        # fused operator previously returned match_count (0) for both.
        "optional_count_star_unmatched",
        "social_graph",
        "MATCH (n:Person) OPTIONAL MATCH (n)-[r:KNOWS]->(m) "
        "WITH n, count(*) AS c, count(m) AS cm, count(*) - count(m) AS diff "
        "RETURN n.name AS name, c, cm, diff ORDER BY name",
        None,
    ),
    (
        # Multi-pattern OPTIONAL MATCH + per-variable counts: the fused
        # operator computes ONE match_count summed across patterns, which
        # can't represent per-pattern counts — the fusion gate must bail
        # and leave this to the materialized executor.
        "optional_multi_pattern_count_vars",
        "social_graph",
        "MATCH (n:Person {person_id: 1}) "
        "OPTIONAL MATCH (n)-[:KNOWS]->(a), (n)-[:WORKS_AT]->(b) "
        "WITH n, count(a) AS ca, count(b) AS cb RETURN n.name AS name, ca, cb",
        None,
    ),
    # ── openCypher contract shapes (0.12.x semantics alignment) ─────────
    (
        # Pre-bound relationship variable re-used in a later MATCH: the
        # pattern must bind exactly the carried edge (openCypher re-MATCH
        # identity), not enumerate every KNOWS edge per row.
        "rel_var_rebind_after_with",
        "social_graph",
        "MATCH (:Person {person_id: 1})-[r:KNOWS]->() WITH r, r.since AS s "
        "MATCH (a)-[r]->(b) RETURN a.name AS an, b.name AS bn, s, r.since AS s2 "
        "ORDER BY bn",
        None,
    ),
    (
        # Same contract through a projected relationship VALUE: UNWIND over
        # collect(r) re-binds `r` as a relationship value, which must pin
        # the MATCH to that edge.
        "rel_var_rebind_after_unwind",
        "social_graph",
        "MATCH (:Person {person_id: 1})-[r0:KNOWS]->() WITH collect(r0) AS rels "
        "UNWIND rels AS r MATCH (a)-[r]->(b) "
        "RETURN a.name AS an, b.name AS bn ORDER BY bn",
        None,
    ),
    (
        # Node parallel of the relationship contract above: a node variable
        # carried through WITH re-used in a later MATCH pins the pattern to
        # exactly that node.
        "node_var_rebind_after_with",
        "social_graph",
        "MATCH (n:Person {person_id: 1}) WITH n MATCH (n)-[:WORKS_AT]->(c) "
        "RETURN n.name AS name, c.name AS cn ORDER BY cn",
        None,
    ),
    (
        # Node identity through a projected node VALUE: UNWIND over
        # collect(n) re-binds `n` as a Value::Node, which must pin the MATCH
        # to that node (openCypher re-MATCH identity), not cartesian-join
        # against every WORKS_AT edge.
        "node_var_rebind_after_unwind",
        "social_graph",
        "MATCH (n:Person) WHERE n.person_id <= 2 WITH collect(n) AS ns "
        "UNWIND ns AS n MATCH (n)-[:WORKS_AT]->(c) "
        "RETURN n.name AS name, c.name AS cn ORDER BY name",
        None,
    ),
    (
        # Trail rule across the comma patterns of one EXISTS subquery:
        # Person_1 has exactly one WORKS_AT edge, so the two-pattern EXISTS
        # must be false while the single-pattern one stays true.
        "exists_trail_rule_comma_patterns",
        "social_graph",
        "MATCH (p:Person {person_id: 1}) "
        "RETURN EXISTS { (p)-[r1:WORKS_AT]->(c), (p)-[r2:WORKS_AT]->(d) } AS two, "
        "EXISTS { (p)-[r1:WORKS_AT]->(c) } AS one",
        None,
    ),
    (
        # COUNT subquery mirrors the EXISTS contract above: the value is the
        # number of JOIN rows, with the trail rule across the comma patterns
        # of one subquery. Person_1 has exactly one WORKS_AT edge, so the
        # two-pattern COUNT must be 0 while the single-pattern one counts it.
        "count_subquery_trail_rule_comma_patterns",
        "social_graph",
        "MATCH (p:Person {person_id: 1}) "
        "RETURN COUNT { (p)-[r1:WORKS_AT]->(c), (p)-[r2:WORKS_AT]->(d) } AS two, "
        "COUNT { (p)-[r1:WORKS_AT]->(c) } AS one",
        None,
    ),
    (
        # COUNT subquery join semantics: comma patterns sharing a variable
        # join on it (row count, not a per-pattern sum), and the multi-MATCH
        # subquery form joins independent clause scopes (counts multiply).
        "count_subquery_join_rows",
        "social_graph",
        "RETURN COUNT { (x)-[r1:KNOWS]->(y), (y)-[r2:KNOWS]->(z) } AS chained, "
        "COUNT { MATCH (p:Person {person_id: 1})-[r1:WORKS_AT]->(c) MATCH (a)-[r2:WORKS_AT]->(b) } AS crossed",
        None,
    ),
    (
        # CASE result positions parse at the full expression tower:
        # comparisons and pattern predicates in THEN/ELSE.
        "case_result_predicate_positions",
        "social_graph",
        "MATCH (p:Person {person_id: 1}) "
        "RETURN CASE WHEN true THEN 1 < 2 ELSE false END AS cmp, "
        "CASE WHEN false THEN false ELSE EXISTS { (p)-[:WORKS_AT]->() } END AS pat",
        None,
    ),
    (
        # Abbreviated relationship patterns: --> / -- / <-- are -[]-> /
        # -[]- / <-[]-.
        "abbreviated_edge_forms",
        "social_graph",
        "MATCH (p:Person {person_id: 1})-->(x) WITH count(x) AS out "
        "MATCH (p:Person {person_id: 1})--(y) WITH out, count(y) AS both "
        "MATCH (c:Company {company_id: 100})<--(z) "
        "RETURN out, both, count(z) AS inn",
        None,
    ),
    (
        # Relationship uniqueness (trail rule) across comma patterns of ONE
        # MATCH: Person_1 has exactly one WORKS_AT edge, so two different
        # edge variables anchored on the same node can't both bind it.
        "trail_rule_comma_patterns_named",
        "social_graph",
        "MATCH (a:Person {person_id: 1})-[r1:WORKS_AT]->(c), (a)-[r2:WORKS_AT]->(d) RETURN count(*) AS n",
        None,
    ),
    (
        # Trail rule with anonymous pattern edges — tracked via the match's
        # exact fixed trail, not named bindings.
        "trail_rule_comma_patterns_anonymous",
        "social_graph",
        "MATCH (a:Person {person_id: 1})-[:WORKS_AT]->(c), (a)-[:WORKS_AT]->(d) RETURN count(*) AS n",
        None,
    ),
    (
        # Pairwise-disjoint fixed relationship types cannot reuse an edge, so
        # the planner may omit exact-trail bookkeeping for this shape.
        "disjoint_fixed_relationship_types",
        "social_graph",
        "MATCH (a:Person {person_id: 1})-[:KNOWS]->(b)-[:WORKS_AT]->(c) RETURN DISTINCT c.name",
        None,
    ),
    (
        # Comma patterns join: an empty pattern empties the whole clause.
        # (Regression: the first-MATCH loop re-entered the "first pattern"
        # branch when an earlier pattern produced no rows, fabricating rows
        # that ignored the empty pattern.)
        "comma_pattern_empty_join",
        "social_graph",
        "MATCH (x:Person {person_id: 9999}), (y:Person) RETURN count(*) AS n",
        None,
    ),
    (
        # Multi-pattern OPTIONAL MATCH where BOTH patterns match: openCypher
        # join-then-null-pad semantics make the row set the cross join
        # (3 KNOWS × 1 WORKS_AT), not a per-pattern union.
        "optional_multi_pattern_join_cross",
        "social_graph",
        "MATCH (n:Person {person_id: 1}) "
        "OPTIONAL MATCH (n)-[:KNOWS]->(a), (n)-[:WORKS_AT]->(b) "
        "RETURN a.name AS an, b.name AS bn ORDER BY an, bn",
        None,
    ),
    # ── schema DDL ──
    # `SHOW INDEXES` is the one schema command that classifies as a read, so it
    # is the one that travels the optimizer pipeline. The shared fixtures carry
    # no indexes, so the row set is empty — what this pins is that no pass
    # rewrites or drops the clause, and that it stays on the read engine. The
    # writing DDL statements live in MUTATION_QUERIES, which builds a fresh
    # graph per mode.
    ("show_indexes_ddl", "small_graph", "SHOW INDEXES", None),
    # `SHOW CONSTRAINTS` is the constraint counterpart, and a read for the same
    # reason: it inspects schema state rather than changing it. The shared
    # fixtures declare no constraints, so the row set is empty — what this pins
    # is that no pass rewrites or drops the clause and that it stays off the
    # mutation engine, where it would be rejected.
    ("show_constraints_ddl", "small_graph", "SHOW CONSTRAINTS", None),
    # Dotted access on a map-valued parameter. Pins that no pass mangles
    # the `ExprPropertyAccess` chain the parser now builds over a
    # `Parameter` node (the bracket form was always accepted; the dotted
    # form used to be a syntax error).
    (
        "parameter_map_property_access",
        "small_graph",
        "MATCH (p:Person) WHERE p.city = $filter.city RETURN p.name AS name ORDER BY name",
        {"filter": {"city": "Oslo"}},
    ),
    # A stored string that *is* a single-element JSON list (`'["Oslo"]'`).
    # `values_equal` calls it equal to its inner string, so every equality
    # route must — the index-selection pushdown's byte fast arm was the one
    # that did not, and answered a bare `= 'Oslo'` with one row fewer than the
    # unoptimized plan. No other fixture carries a JSON-list-valued column.
    (
        "json_list_property_equality",
        "json_list_props_graph",
        "MATCH (n:Doc) WHERE n.tag = 'Oslo' RETURN n.title AS t ORDER BY t",
        None,
    ),
    (
        "json_list_property_equality_reversed",
        "json_list_props_graph",
        "MATCH (n:Doc) WHERE n.tag = '[\"Oslo\"]' RETURN n.title AS t ORDER BY t",
        None,
    ),
    (
        "json_list_property_inequality",
        "json_list_props_graph",
        "MATCH (n:Doc) WHERE n.tag <> 'Oslo' RETURN n.title AS t ORDER BY t",
        None,
    ),
    (
        "json_list_property_membership",
        "json_list_props_graph",
        "MATCH (n:Doc) WHERE n.tag IN ['Oslo'] RETURN n.title AS t ORDER BY t",
        None,
    ),
    (
        "json_list_property_inline_equality",
        "json_list_props_graph",
        "MATCH (n:Doc {tag: 'Oslo'}) RETURN n.title AS t ORDER BY t",
        None,
    ),
    # ── vector math over list properties (dot / cosine / norm) ──
    # These read a *list-valued* property, which no other corpus fixture
    # carries. The projected + ORDER BY + LIMIT shape is the one the top-k
    # retention and projection passes rewrite, so a pass that mangles a
    # function call over a list column shows up here as a row-set divergence.
    (
        "vector_cosine_ranking",
        "vector_props_graph",
        "MATCH (d:Doc) RETURN d.title AS t, cosine(d.vec, $q) AS s ORDER BY s DESC, t",
        {"q": [1.0, 0.0]},
    ),
    (
        "vector_cosine_topk",
        "vector_props_graph",
        "MATCH (d:Doc) RETURN d.title AS t, cosine(d.vec, [1, 0]) AS s ORDER BY s DESC, t LIMIT 2",
        None,
    ),
    (
        "vector_dot_in_where",
        "vector_props_graph",
        "MATCH (d:Doc) WHERE dot(d.vec, [1, 0]) > 0.5 RETURN d.title AS t ORDER BY t",
        None,
    ),
    (
        "vector_norm_projection",
        "vector_props_graph",
        "MATCH (d:Doc) RETURN d.title AS t, norm(d.vec) AS n ORDER BY t",
        None,
    ),
    # The zero vector's cosine is NULL: both paths must agree on the null,
    # and on where it sorts.
    (
        "vector_cosine_null_arm",
        "vector_props_graph",
        "MATCH (d:Doc) WHERE cosine(d.vec, [1, 0]) IS NULL RETURN d.title AS t ORDER BY t",
        None,
    ),
    # Two stored vectors against each other — the cross-join shape.
    (
        "vector_pairwise",
        "vector_props_graph",
        "MATCH (a:Doc) MATCH (b:Doc) WHERE a.title < b.title "
        "RETURN a.title AS a, b.title AS b, dot(a.vec, b.vec) AS s ORDER BY a, b",
        None,
    ),
    # ── aggregates nested in map/list literals (fixed 2026-08-15) ──
    # Pre-fix the materialized evaluator only handled top-level aggregates:
    # `RETURN {c: count(*)}` died with "Aggregate function 'count' cannot be
    # used outside of RETURN/WITH" (while sitting in RETURN), and
    # `[count(*)]` wasn't classified as an aggregate projection at all.
    # These also pin the ListLiteral classifier arm, which feeds the
    # planner-simplification aggregation gates — a plan-shape change there
    # must keep both legs in agreement.
    (
        "nested_agg_map_literal",
        "small_graph",
        "MATCH (p:Person) RETURN {name:'people', data: count(*)} AS r",
        None,
    ),
    (
        "nested_agg_list_literal",
        "small_graph",
        "MATCH (p:Person) RETURN [count(*), min(p.age)] AS r",
        None,
    ),
    (
        "nested_agg_collect_slice_in_map",
        "small_graph",
        "MATCH (p:Person) RETURN {data: collect(p.name)[..2]} AS r",
        None,
    ),
    (
        "nested_agg_grouped_map",
        "small_graph",
        "MATCH (p:Person) RETURN p.age AS a, {c: count(*)} AS r ORDER BY a",
        None,
    ),
    (
        "nested_agg_case_result",
        "small_graph",
        "MATCH (p:Person) RETURN CASE WHEN true THEN count(*) ELSE 0 END AS r",
        None,
    ),
    (
        "nested_agg_predicate_expr",
        "small_graph",
        "MATCH (p:Person) RETURN count(*) > 2 AS r",
        None,
    ),
    (
        "nested_agg_negate",
        "small_graph",
        "MATCH (p:Person) RETURN -count(*) AS r",
        None,
    ),
    (
        "nested_agg_empty_rowset",
        "small_graph",
        "MATCH (p:Person) WHERE p.age > 999 RETURN {c: count(*)} AS r",
        None,
    ),
    # Zero-length path assignment (fixed 2026-08-15): p must bind a one-node
    # path, not NULL, on both optimizer legs.
    (
        "zero_length_path_assignment",
        "small_graph",
        "MATCH p = (n:Person) RETURN length(p) AS l, size(nodes(p)) AS c, n.name AS nm ORDER BY nm",
        None,
    ),
    # ── id() literal vs parameter, on cross-type id collisions ───────────
    #
    # Fixed 2026-08-15. The four spellings below all denote "every node whose
    # domain id is 2" and diverged from each other:
    #
    #   * the untyped `{id: X}` anchor returned on the FIRST type whose id
    #     index answered, collapsing a cross-type collision to one arbitrary
    #     node — arbitrary because the type map's key order is a HashMap's
    #     (so the surviving row was not even stable across processes); and
    #   * the anchor read only `PropertyMatcher::Equals`, so the `$param`
    #     spelling fell past it into the full scan — which *is* exhaustive.
    #     Meanwhile `WHERE id(v) = $x` was not pushed into the pattern at all
    #     (try_extract_equality had no `id(v) = $param` arm), so the literal
    #     and the parameter took different plans as well as different rows.
    #     On the reporting graph that read 1 row vs 68.
    #
    # The fixture makes the collision the *normal* case: ids {1, 2} exist under
    # all three labels, so any surviving first-type-wins behaviour answers 1
    # where the corpus expects 3, on both optimizer legs.
    (
        "id_function_literal",
        "cross_type_id_graph",
        "MATCH (v) WHERE id(v) = 2 RETURN v.name AS nm ORDER BY nm",
        None,
    ),
    (
        "id_function_param",
        "cross_type_id_graph",
        "MATCH (v) WHERE id(v) = $x RETURN v.name AS nm ORDER BY nm",
        {"x": 2},
    ),
    (
        "id_pattern_literal",
        "cross_type_id_graph",
        "MATCH (n {id: 2}) RETURN n.name AS nm ORDER BY nm",
        None,
    ),
    (
        "id_pattern_param",
        "cross_type_id_graph",
        "MATCH (n {id: $x}) RETURN n.name AS nm ORDER BY nm",
        {"x": 2},
    ),
    (
        # The anchor as an expansion *start*: two of the three id-1 nodes have
        # an outgoing edge, so a first-type-wins anchor counts 1 instead of 2.
        "id_pattern_edge_anchor",
        "cross_type_id_graph",
        "MATCH (a {id: 1})-[r]->(b) RETURN count(r) AS c",
        None,
    ),
    # ── elementId() slot anchoring (`anchor_element_id`, 2026-08-15) ─────
    #
    # `elementId(v)` is the node's slot, so the predicate names exactly one
    # candidate — but the pattern an IDE sends back after a click carries no
    # label, so the unanchored plan is a full node scan plus a per-row
    # predicate (measured 28 s on a G.V() node-expansion round trip). The pass
    # records the slot as a pre-binding.
    #
    # The anchor is a search-space constraint and the predicate is never
    # removed, which is exactly what these entries pin: the anchored plan must
    # answer what the unanchored one answers, in every shape — including the
    # ones where the pass must decline (a disjunct constrains nothing) and the
    # one where the slot does not exist.
    #
    # `small_graph` slots are 0/1/2 for Alice/Bob/Charlie, in insertion order.
    (
        "element_id_anchor_param",
        "small_graph",
        "MATCH (v) WHERE elementId(v) = $eid RETURN v.name AS nm",
        {"eid": "0"},
    ),
    (
        "element_id_anchor_literal",
        "small_graph",
        "MATCH (v) WHERE elementId(v) = '0' RETURN v.name AS nm",
        None,
    ),
    (
        "element_id_anchor_path",
        "small_graph",
        "MATCH p = (v)--() WHERE elementId(v) = $eid RETURN count(p) AS c",
        {"eid": "0"},
    ),
    (
        # Must NOT anchor: under OR every node is still a candidate, so an
        # anchored plan would answer one row where the naive answers two.
        "element_id_no_anchor_under_or",
        "small_graph",
        "MATCH (v) WHERE elementId(v) = $eid OR v.name = 'Bob' RETURN v.name AS nm ORDER BY nm",
        {"eid": "0"},
    ),
    (
        # A slot past the end of the graph: the pre-binding resolves to no
        # node, which is what the predicate concludes too.
        "element_id_out_of_range",
        "small_graph",
        "MATCH (v) WHERE elementId(v) = $eid RETURN v.name AS nm",
        {"eid": "999999"},
    ),
    # ── LIMIT over the pattern executor's candidate caps (2026-08-20) ────
    #
    # Pushing a LIMIT into the MATCH caps the candidates the pattern executor
    # materialises: `max(limit * 100, 1000)` start nodes and `max(limit * 50,
    # 1000)` intermediates per hop. Both numbers are a *selectivity* guess —
    # neither knows the relationship type — so a start node (or intermediate)
    # whose only matching edge sits past the cap used to be dropped and the
    # query answered with zero rows. The naive plan has no `limit_hint`, so
    # this corpus is exactly the right instrument; what it lacked was a
    # fixture big enough to reach the 1 000 floor. `cap_threshold_graph` is.
    (
        "limit_unlabeled_start_late_source",
        "cap_threshold_graph",
        "MATCH (a)-[:WROTE]->(b) RETURN a.name AS a, b.name AS b LIMIT 1",
        None,
    ),
    (
        "limit_unlabeled_start_late_source_cliff",
        "cap_threshold_graph",
        # `10 * 100` lands exactly on the 1 000 floor, `11 * 100` clears it:
        # the answer used to change between these two.
        "MATCH (a)-[:WROTE]->(b) RETURN a.name AS a, b.name AS b LIMIT 10",
        None,
    ),
    (
        "limit_labeled_sparse_start",
        "cap_threshold_graph",
        "MATCH (p:Paper)-[:CITES]->(q:Paper) RETURN p.name AS p, q.name AS q LIMIT 1",
        None,
    ),
    (
        "limit_undirected_late_source",
        "cap_threshold_graph",
        # LIMIT 2 takes both orientations of the single edge, so the entry does
        # not depend on which one enumerates first.
        "MATCH (a)-[:WROTE]-(b) RETURN a.name AS a, b.name AS b LIMIT 2",
        None,
    ),
    (
        "limit_var_length_late_source",
        "cap_threshold_graph",
        "MATCH (a)-[:WROTE*1..2]->(b) RETURN a.name AS a, b.name AS b LIMIT 1",
        None,
    ),
    (
        "limit_alternation_late_source",
        "cap_threshold_graph",
        "MATCH (a)-[:WROTE|CITES]->(b) RETURN a.name AS a, b.name AS b LIMIT 2",
        None,
    ),
    (
        # Two hops: the venue's 1 200 papers overrun the intermediate-hop cap,
        # and the one paper carrying the second hop is in the overrun.
        "limit_sparse_intermediate_hop",
        "cap_threshold_graph",
        "MATCH (v:Venue)-[:HOSTED]->(p:Paper)-[:REFS]->(q:Paper) RETURN v.name AS v, p.name AS p, q.name AS q LIMIT 1",
        None,
    ),
    (
        # The retry's worst case: nothing matches, so the uncapped pass runs
        # and still finds nothing. Pinned so it stays empty rather than
        # becoming a source of invented rows.
        "limit_missing_relationship_type",
        "cap_threshold_graph",
        "MATCH (a)-[:NOSUCH]->(b) RETURN a.name AS a, b.name AS b LIMIT 1",
        None,
    ),
    (
        # OPTIONAL MATCH + LIMIT drives off the leading MATCH, so the cap never
        # decided this one. Pinned to keep it that way.
        "limit_optional_match_over_cap",
        "cap_threshold_graph",
        "MATCH (a:Author) OPTIONAL MATCH (a)-[:WROTE]->(p:Paper) RETURN a.name AS a, p.name AS p ORDER BY a LIMIT 5",
        None,
    ),
    # ── aggregates over a property holding mixed types ──
    # The fused node-scan aggregate keeps one running sum and one running
    # count per group. It summed only the numeric values but counted every
    # non-null one, then divided them: `[10, 20, 'hello']` averaged to 10.0
    # instead of 15.0, silently, on every RETURN/WITH/grouped shape. It also
    # read `sum()`'s Int64-vs-Float64 choice off `min()`, where a string
    # outranks every number, so the same cell flipped the sum's type. No other
    # corpus fixture carries a mixed-type numeric column, which is why the
    # corpus stayed green through both.
    (
        "mixed_type_aggregate_scan",
        "mixed_type_props_graph",
        "MATCH (n:Sample) RETURN avg(n.v) AS a, sum(n.v) AS s, count(n.v) AS c, min(n.v) AS mn",
        None,
    ),
    (
        "mixed_type_aggregate_grouped",
        "mixed_type_props_graph",
        "MATCH (n:Sample) RETURN n.site AS site, avg(n.v) AS a, sum(n.v) AS s, count(n.v) AS c ORDER BY site",
        None,
    ),
    (
        "mixed_type_aggregate_with_projection",
        "mixed_type_props_graph",
        "MATCH (n:Sample) WITH n RETURN avg(n.v) AS a, sum(n.v) AS s, count(n.v) AS c",
        None,
    ),
    (
        "mixed_type_aggregate_filtered",
        "mixed_type_props_graph",
        "MATCH (n:Sample) WHERE n.w > 0 RETURN avg(n.v) AS a, sum(n.v) AS s",
        None,
    ),
    (
        # Control: `w` is numeric in every row, so this cell must answer the
        # same before and after the numeric-count split. It is what says the
        # entries above are measuring mixed types, not aggregation at large.
        "mixed_type_aggregate_numeric_control",
        "mixed_type_props_graph",
        "MATCH (n:Sample) RETURN n.site AS site, avg(n.w) AS a, sum(n.w) AS s ORDER BY site",
        None,
    ),
    # ── text_bm25 top-k fusion (`fuse_text_bm25_order_limit`) ──
    #
    # The pass claims `RETURN text_bm25(...) AS s ORDER BY s LIMIT k` from the
    # generic top-k operator and serves it from the index's postings. All four
    # shapes below were run against the unoptimized path while the operator was
    # being written; the last two were *divergent* and are here because of it.
    ("text_bm25_top_k", "text_index_graph", TEXT_BM25_TOP_K, None),
    # Nine index documents and nine rows can still be different sets: the
    # absent-body row must win DESC even when the excluded document scores zero.
    (
        "text_bm25_top_k_equal_cardinality_missing_member",
        "text_index_graph",
        "MATCH (d:Doc) WHERE d.id <> 9 RETURN d.id AS id, text_bm25(d, 'body', 'alpha') AS s ORDER BY s DESC LIMIT 1",
        None,
    ),
    # A `WHERE` makes the rows a subset of the corpus, which the postings path
    # cannot answer from (the index ranks documents the subset may not contain)
    # — it declines, and this pins that the decline is still the right answer.
    (
        "text_bm25_top_k_filtered",
        "text_index_graph",
        "MATCH (d:Doc) WHERE d.id > 2 RETURN d.id AS id, text_bm25(d, 'body', 'alpha beta') AS s "
        "ORDER BY s DESC LIMIT 3",
        None,
    ),
    # ASC: least-relevant-first has no postings shortcut. Divergent while the
    # decline formerly fell through to a vector-scoring scan that dropped
    # null rows instead of placing them.
    (
        "text_bm25_top_k_ascending",
        "text_index_graph",
        "MATCH (d:Doc) RETURN d.id AS id, text_bm25(d, 'body', 'alpha') AS s ORDER BY s ASC LIMIT 3",
        None,
    ),
    # A stale index past its auto-refresh limit scores the un-caught-up rows
    # null, and `ORDER BY ... DESC` puts nulls *first* — so the honest answer to
    # this query is mostly nulls. Divergent for the same reason as the ASC case.
    ("text_bm25_top_k_stale_over_limit", "stale_text_index_graph", TEXT_BM25_TOP_K, None),
]


# Sized to cross the candidate caps described at the `cap_threshold_graph`
# corpus entries above. Every other fixture in this file is two or three orders
# of magnitude below the caps' 1 000 floor, which is why the corpus — an
# instrument that would otherwise have caught the bug outright — stayed green
# through it for eight minor versions.
_CAP_FILLERS = 1100
_CAP_PAPERS = 1200


def _build_cap_threshold_graph() -> kglite.KnowledgeGraph:
    """~2 500 nodes over four types, with every edge-carrying source late.

    Insertion order is load-bearing: `Filler` and `Paper` are created first, so
    the `Author` nodes (the only `WROTE` sources) sit past node 2 300 and an
    *unlabeled* start scan reaches them only after the start-node cap. The
    `CITES` source is the last `Paper` created, so a *labeled* start hits the
    same wall inside the type index. And `venue-1` hosts all 1 200 papers while
    only `paper-1` carries a `REFS` edge onward — the edge iteration walks a
    node's edges newest-first, so `paper-1` is the *last* intermediate the
    first hop yields and the intermediate-hop cap is what drops it.
    """
    import pandas as pd

    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {
                "id": [f"f{i}" for i in range(1, _CAP_FILLERS + 1)],
                "name": [f"filler-{i}" for i in range(1, _CAP_FILLERS + 1)],
            }
        ),
        "Filler",
        "id",
        "name",
    )
    g.add_nodes(
        pd.DataFrame(
            {
                "id": [f"p{i}" for i in range(1, _CAP_PAPERS + 1)],
                "name": [f"paper-{i}" for i in range(1, _CAP_PAPERS + 1)],
                "year": [2000 + (i % 20) for i in range(1, _CAP_PAPERS + 1)],
            }
        ),
        "Paper",
        "id",
        "name",
        columns=["year"],
    )
    g.add_nodes(
        pd.DataFrame({"id": ["v1", "v2", "v3"], "name": ["venue-1", "venue-2", "venue-3"]}),
        "Venue",
        "id",
        "name",
    )
    g.add_nodes(
        pd.DataFrame(
            {
                "id": [f"a{i}" for i in range(1, 201)],
                "name": [f"author-{i}" for i in range(1, 201)],
            }
        ),
        "Author",
        "id",
        "name",
    )

    # venue-1 hosts every paper: 1 200 intermediates at the first hop.
    g.add_connections(
        pd.DataFrame({"src": ["v1"] * _CAP_PAPERS, "dst": [f"p{i}" for i in range(1, _CAP_PAPERS + 1)]}),
        "HOSTED",
        "Venue",
        "src",
        "Paper",
        "dst",
    )
    # The graph's only second hop, hanging off the first-created HOSTED edge.
    g.add_connections(pd.DataFrame({"src": ["p1"], "dst": ["p6"]}), "REFS", "Paper", "src", "Paper", "dst")
    # Sole CITES edge, sourced from the LAST paper in the type index.
    g.add_connections(
        pd.DataFrame({"src": [f"p{_CAP_PAPERS}"], "dst": ["p5"]}),
        "CITES",
        "Paper",
        "src",
        "Paper",
        "dst",
    )
    # Sole WROTE edge, sourced from the last node in the graph.
    g.add_connections(pd.DataFrame({"src": ["a200"], "dst": ["p11"]}), "WROTE", "Author", "src", "Paper", "dst")
    return g


@pytest.fixture
def cap_threshold_graph() -> kglite.KnowledgeGraph:
    return _build_cap_threshold_graph()


def _build_text_index_graph(auto_refresh_limit: int | None = None) -> kglite.KnowledgeGraph:
    """Nine documents over a five-word vocabulary, with a BM25 index.

    Deliberately tie-heavy: a vocabulary that small makes identical scores the
    normal case, which is what makes the two paths' tie-breaks comparable at
    all. It also carries the two rows that are not simply "some text" — one
    whose property is absent (never indexed, always null) and one holding the
    empty string (indexed, scores 0.0) — because those are the rows a ranking
    places differently depending on which operator did the placing.
    """
    import pandas as pd

    bodies = [
        "alpha beta gamma",
        "alpha alpha beta",
        "beta gamma delta",
        "alpha",
        "epsilon",
        "alpha beta",
        "gamma gamma",
        # An exact duplicate of doc-6, so two documents tie to the last bit.
        # Tie order is the one thing the two paths break differently if the
        # index's slot ordering ever stops tracking row order.
        "alpha beta",
        "",
    ]
    graph = kglite.KnowledgeGraph()
    graph.add_nodes(
        pd.DataFrame(
            {
                "id": list(range(1, len(bodies) + 1)),
                "title": [f"doc-{i}" for i in range(1, len(bodies) + 1)],
                "body": bodies,
            }
        ),
        "Doc",
        "id",
        "title",
        columns=["body"],
    )
    # No `body` at all: this one scores null, not zero.
    graph.cypher("CREATE (:Doc {id: 10, title: 'doc-10'})")
    graph.build_text_index("Doc", "body", auto_refresh_limit=auto_refresh_limit)
    return graph


def _staled_text_index_graph() -> kglite.KnowledgeGraph:
    """The same corpus, then two creations against a limit of one.

    The delta is over the auto-refresh limit, so a query serves what the index
    has and scores the two new rows null instead of quietly rebuilding.
    """
    graph = _build_text_index_graph(auto_refresh_limit=1)
    graph.cypher("CREATE (:Doc {id: 11, title: 'doc-11', body: 'alpha beta gamma delta'})")
    graph.cypher("CREATE (:Doc {id: 12, title: 'doc-12', body: 'alpha alpha alpha'})")
    return graph


@pytest.fixture
def vector_order_graph() -> kglite.KnowledgeGraph:
    import pandas as pd

    graph = kglite.KnowledgeGraph()
    graph.add_nodes(pd.DataFrame({"id": [0, 1, 2, 3], "summary": ["x"] * 4}), "Doc", "id")
    graph.set_embeddings("Doc", "summary", {0: [1.0, 0.0], 1: [0.0, 1.0], 2: [-1.0, 0.0]})
    return graph


@pytest.fixture
def vector_index_entry_graph(vector_order_graph) -> kglite.KnowledgeGraph:
    # This separated three-vector corpus returns every member; verify that
    # precondition before using ANN output as an exact differential oracle.
    graph = vector_order_graph
    graph.cypher("MATCH (d:Doc {id:3}) DETACH DELETE d")
    graph.build_vector_index("Doc", "summary")
    assert graph.cypher("MATCH (d:Doc) RETURN count(d) AS n").scalar() == 3
    rows = graph.cypher(
        "MATCH (d:Doc) RETURN d.id AS id, vector_score(d, 'summary_emb', [1.0,0.0]) AS s ORDER BY s DESC LIMIT 3"
    )
    assert rows.diagnostics["retrieval"][0]["actual_mode"] == "hnsw"
    assert [row["id"] for row in rows] == [0, 1, 2]
    return graph


@pytest.fixture
def text_index_graph() -> kglite.KnowledgeGraph:
    return _build_text_index_graph()


@pytest.fixture
def stale_text_index_graph() -> kglite.KnowledgeGraph:
    return _staled_text_index_graph()


@pytest.mark.differential
def test_text_index_fixtures_reach_the_shapes_their_entries_assume() -> None:
    """Non-vacuity guard for the four `text_bm25` corpus entries.

    Each one exists for a specific row class — a null score, a zero score, a
    stale delta the query must not fold in. If a fixture stops producing them
    the entries keep passing while testing nothing.
    """
    scores = {
        row["id"]: row["s"]
        for row in _build_text_index_graph()
        .cypher("MATCH (d:Doc) RETURN d.id AS id, text_bm25(d, 'body', 'alpha beta') AS s")
        .to_list()
    }
    assert scores[10] is None, "the property-less doc must score null"
    assert scores[9] == 0.0, "the empty-string doc must be indexed and score zero"
    scored = [s for s in scores.values() if s]
    assert len(set(scored)) < len(scored), "the corpus must produce tied scores"

    row = next(r for r in _staled_text_index_graph().cypher("SHOW INDEXES").to_list() if r["type"] == "FULLTEXT")
    assert row["stale"] and row["delta"] > 1, "the stale fixture must be past its refresh limit"


@pytest.mark.differential
def test_cap_threshold_fixture_crosses_the_candidate_caps() -> None:
    """Non-vacuity guard for the `cap_threshold_graph` corpus entries.

    Those entries compare two plans; if the fixture ever shrinks back under the
    1 000-candidate floor they would compare two *uncapped* plans and agree
    forever. Pin the sizes that make the caps reachable.
    """
    g = _build_cap_threshold_graph()
    assert g.cypher("MATCH (n) RETURN count(n) AS c").to_list()[0]["c"] > 2000
    assert g.cypher("MATCH (p:Paper) RETURN count(p) AS c").to_list()[0]["c"] > 1000
    assert g.cypher("MATCH (v:Venue)-[:HOSTED]->(p:Paper) RETURN count(p) AS c").to_list()[0]["c"] > 1000


@pytest.mark.differential
@pytest.mark.parametrize(
    "query,expected",
    [
        (
            "MATCH (a)-[:WROTE]->(b) RETURN a.name AS a, b.name AS b LIMIT 1",
            [{"a": "author-200", "b": "paper-11"}],
        ),
        (
            "MATCH (a)-[:WROTE]->(b) RETURN a.name AS a, b.name AS b LIMIT 10",
            [{"a": "author-200", "b": "paper-11"}],
        ),
        (
            "MATCH (p:Paper)-[:CITES]->(q:Paper) RETURN p.name AS p, q.name AS q LIMIT 1",
            [{"p": "paper-1200", "q": "paper-5"}],
        ),
        (
            "MATCH (a)-[:WROTE*1..2]->(b) RETURN a.name AS a, b.name AS b LIMIT 1",
            [{"a": "author-200", "b": "paper-11"}],
        ),
        (
            "MATCH (v:Venue)-[:HOSTED]->(p:Paper)-[:REFS]->(q:Paper) "
            "RETURN v.name AS v, p.name AS p, q.name AS q LIMIT 1",
            [{"v": "venue-1", "p": "paper-1", "q": "paper-6"}],
        ),
        (
            "MATCH (a)-[:NOSUCH]->(b) RETURN a.name AS a, b.name AS b LIMIT 1",
            [],
        ),
    ],
    ids=[
        "unlabeled_start",
        "unlabeled_start_at_the_cliff",
        "labeled_sparse_start",
        "var_length",
        "sparse_intermediate_hop",
        "missing_relationship_type",
    ],
)
def test_limit_returns_the_rows_that_exist(query: str, expected: list[dict]) -> None:
    """Absolute goldens for the LIMIT-over-cap shapes.

    The differential entries above compare two plans; these pin the *answer*,
    which is what a user sees. A LIMITed pattern returns the rows the graph
    has — the candidate caps are an optimization, not a bound on the result.
    """
    g = _build_cap_threshold_graph()
    assert g.cypher(query).to_list() == expected


def _edge_graph(nodes: list[int], edges: list[tuple[int, int]]) -> kglite.KnowledgeGraph:
    """`:N` nodes with an `id`/`name`, joined by `:R` relationships.

    One builder for the cyclic var-length fixtures so each one is readable as
    its edge list. Kept out of conftest: the shared fixtures are pinned by
    absolute goldens across the suite, and a cycle added to one of them would
    move those expectations.
    """
    import pandas as pd

    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame({"id": nodes, "name": [f"n{i}" for i in nodes]}), "N", "id", "name")
    g.add_connections(
        pd.DataFrame({"s": [e[0] for e in edges], "t": [e[1] for e in edges]}),
        "R",
        "N",
        "s",
        "N",
        "t",
    )
    return g


@pytest.fixture
def directed_triangle_graph() -> kglite.KnowledgeGraph:
    """1 → 2 → 3 → 1.

    The smallest graph on which distance reachability and Cypher's trail
    reachability disagree: every node is on a closed trail of length 3, and
    undirected `*2..2` reaches both peers while the shortest-distance answer
    reaches neither.
    """
    return _edge_graph([1, 2, 3], [(1, 2), (2, 3), (3, 1)])


@pytest.fixture
def var_length_diamond_graph() -> kglite.KnowledgeGraph:
    """1 → 2, 2 → 3, 2 → 4, 3 → 4 — acyclic, with two routes 2 ⇒ 4.

    The two routes are what make a dropped duplicate observable: a plain
    projection over `(2)-[:R*1..2]->(c)` has three rows, one of them a repeat
    of `4`.
    """
    return _edge_graph([1, 2, 3, 4], [(1, 2), (2, 3), (2, 4), (3, 4)])


@pytest.fixture
def parallel_edge_cycle_graph() -> kglite.KnowledgeGraph:
    """Two parallel 1 → 2 relationships, plus 2 → 3 and 3 → 1.

    Exercises both undirected closed-trail lengths: 1 returns to itself in two
    hops over the parallel pair, and in three around the cycle. The parallel
    pair also makes distinct *edges* and distinct *peers* disagree for node 1,
    which is what `count_distinct_edge_var_parallel` needs.
    """
    return _edge_graph([1, 2, 3], [(1, 2), (1, 2), (2, 3), (3, 1)])


@pytest.fixture
def uneven_group_graph() -> kglite.KnowledgeGraph:
    """`:T` nodes in eight `k` groups: g0-g2 hold one node, g3-g7 hold two.

    Insertion order puts the three singleton groups first, so a group cap of 5
    spends three of its five slots on groups a `size(ids) > 1` filter then
    rejects — leaving 2 rows where all 5 of g3-g7 qualify. That gap is what
    makes an aggregate LIMIT pushed past a post-projection filter observable.
    """
    import pandas as pd

    rows = []
    for k in range(8):
        for _ in range(1 if k < 3 else 2):
            rows.append({"id": len(rows), "k": f"g{k}"})
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame(rows), "T", "id")
    return g


@pytest.fixture
def dense_group_graph() -> kglite.KnowledgeGraph:
    """30 `:T` nodes over three `k` values, round-robin — ten nodes per group.

    Every group is spread across ten distinct nodes, so the group cap's
    NodeIndex-keyed surrogate set is ten times the size of the resolved group
    set. That gap is what let `push_limit_into_aggregate` freeze the surrogate
    set and drop rows belonging to groups it had already collected. A small
    fixture cannot see it: the cap engages only once the row pass has opened
    more surrogate groups than the limit allows.
    """
    import pandas as pd

    rows = [{"id": i, "k": f"g{i % 3}"} for i in range(30)]
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame(rows), "T", "id")
    return g


@pytest.fixture
def overlapping_khop_graph() -> kglite.KnowledgeGraph:
    """Three short cycles bridged so several seeds reach each other's targets.

    0→1→2→3→0, 4→5→6→4 and 7→8→9→7, plus the bridges 1→5 and 2→7. Seeds 0 and
    4 overlap on {5, 6}; seed 1 reaches nothing seed 0 has not already passed
    through, which is what separates "skip the emitted row" from "prune the
    frontier".
    """
    return _edge_graph(
        list(range(10)),
        [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 4), (1, 5), (7, 8), (8, 9), (9, 7), (2, 7)],
    )


@pytest.fixture
def unequal_source_graph() -> kglite.KnowledgeGraph:
    """1 → 3 and 2 → 3: one target, two sources that a predicate can separate."""
    return _edge_graph([1, 2, 3], [(1, 3), (2, 3)])


@pytest.fixture
def join_then_distinct_graph() -> kglite.KnowledgeGraph:
    """1 →`:R` 3 ← `:R` 2, and 2 →`:S` 4.

    Both 1 and 2 reach 3, but only 2 carries the `:S` relationship the second
    comma pattern needs — so which representative a per-target dedup keeps
    decides whether 3 survives the join.
    """
    import pandas as pd

    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame({"id": [1, 2, 3, 4], "name": [f"n{i}" for i in (1, 2, 3, 4)]}), "N", "id", "name")
    g.add_connections(pd.DataFrame({"s": [1, 2], "t": [3, 3]}), "R", "N", "s", "N", "t")
    g.add_connections(pd.DataFrame({"s": [2], "t": [4]}), "S", "N", "s", "N", "t")
    return g


@pytest.fixture
def two_cycle_graph() -> kglite.KnowledgeGraph:
    """1 → 2 and 2 → 1 — two relationships, one in each direction.

    The smallest graph on which a variable-length segment and a sibling edge
    of the same clause can want the same relationship: the segment reaches 1
    again in two hops, and the hop after it can only leave 1 over the
    relationship the segment already consumed.
    """
    return _edge_graph([1, 2], [(1, 2), (2, 1)])


@pytest.fixture
def square_cycle_graph() -> kglite.KnowledgeGraph:
    """1 → 2 → 3 → 4 → 1.

    The smallest graph on which the matcher's intermediate dedup was
    observably wrong: undirected three hops from 1 reach both 2 and 4, but 2
    only over a trail whose intermediate the dedup had already claimed for a
    route that had consumed the relationship the last hop needed.
    """
    return _edge_graph([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4), (4, 1)])


@pytest.fixture
def hub_return_graph() -> kglite.KnowledgeGraph:
    """1, 2, 3 each reach hub 10 over `:A`, and the hub reaches each back over `:B`.

    The relationship types are pairwise disjoint, so no trail is tracked — the
    other way two partial matches on one anonymous intermediate stop being
    interchangeable is a repeated node variable, which this fixture's
    `(a)-[:A]->()-[:B]->(a)` shape supplies.
    """
    import pandas as pd

    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame({"id": [1, 2, 3, 10], "name": [f"n{i}" for i in (1, 2, 3, 10)]}),
        "N",
        "id",
        "name",
    )
    g.add_connections(pd.DataFrame({"s": [1, 2, 3], "t": [10, 10, 10]}), "A", "N", "s", "N", "t")
    g.add_connections(pd.DataFrame({"s": [10, 10, 10], "t": [1, 2, 3]}), "B", "N", "s", "N", "t")
    return g


@pytest.fixture
def long_chain_graph() -> kglite.KnowledgeGraph:
    """0 → 1 → ... → 11, a single directed chain.

    Long enough to answer both sides of the `*k..k` lowering ceiling: `*8..8`
    lowers to eight hops, `*9..9` is one past it and stays variable-length.
    Both have exactly one answer, so a bail cannot hide behind an empty set.
    """
    return _edge_graph(list(range(12)), [(i, i + 1) for i in range(11)])


VAR_LENGTH_HOP_SPECS = ["*0..2", "*1..2", "*1..3", "*2..2", "*2..3", "*3..3"]


@pytest.fixture(scope="module")
def var_length_scale_graph() -> kglite.KnowledgeGraph:
    """3 000 `:N` nodes on a ring, plus a stride-7 chord from every node.

    Small graphs can hide a reachability bug behind their own diameter; this
    one is thick with short cycles, so distance reachability and trail
    reachability come apart everywhere rather than in one hand-built corner.
    The un-gated fast path answered undirected `*3..3` from node 0 with 12 of
    the 16 nodes that are actually reachable, and undirected `*2..3` with 20
    of 24.
    """
    import pandas as pd

    n = 3000
    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame({"id": list(range(n)), "name": [f"n{i}" for i in range(n)]}),
        "N",
        "id",
        "name",
    )
    src = list(range(n)) + list(range(n))
    dst = [(i + 1) % n for i in range(n)] + [(i + 7) % n for i in range(n)]
    g.add_connections(pd.DataFrame({"s": src, "t": dst}), "R", "N", "s", "N", "t")
    return g


@pytest.mark.parametrize("hops", VAR_LENGTH_HOP_SPECS)
@pytest.mark.parametrize("arrow", ["->", "-"], ids=["directed", "undirected"])
def test_var_length_at_scale_matches_the_per_path_answer(
    var_length_scale_graph: kglite.KnowledgeGraph, hops: str, arrow: str
) -> None:
    """Exact-set golden: the optimized expansion answers what the per-path one does.

    The differential corpus above compares plans on graphs small enough to
    read; this compares them where a partial answer looks plausible. Both the
    anchored set and the whole-graph distinct count are asserted — the first
    catches a missing source node, the second catches a systematically
    truncated frontier.
    """
    g = var_length_scale_graph
    anchored = f"MATCH (a:N {{id: 0}})-[:R{hops}]{arrow}(b:N) RETURN DISTINCT b.id AS i"
    optimized = sorted(r["i"] for r in g.cypher(anchored).to_list())
    naive = sorted(r["i"] for r in g.cypher(anchored, disable_optimizer=True).to_list())
    assert optimized == naive, f"{anchored}: {len(optimized)} vs {len(naive)} nodes"

    whole = f"MATCH (a:N)-[:R{hops}]{arrow}(b:N) RETURN count(DISTINCT b) AS n"
    assert g.cypher(whole).to_list() == g.cypher(whole, disable_optimizer=True).to_list()


def _cap_biting_var_length_graph(tail_from: int) -> kglite.KnowledgeGraph:
    """One `:Src`, 1 200 `:Mid`, one `:Tail`; only `Mid[tail_from]` reaches it."""
    import pandas as pd

    mids = 1200
    g = kglite.KnowledgeGraph()
    g.add_nodes(pd.DataFrame({"id": [0], "name": ["src"]}), "Src", "id", "name")
    g.add_nodes(
        pd.DataFrame({"id": list(range(mids)), "name": [f"m{i}" for i in range(mids)]}),
        "Mid",
        "id",
        "name",
    )
    g.add_nodes(pd.DataFrame({"id": [0], "name": ["tail"]}), "Tail", "id", "name")
    g.add_connections(
        pd.DataFrame({"s": [0] * mids, "t": list(range(mids))}),
        "R",
        "Src",
        "s",
        "Mid",
        "t",
    )
    g.add_connections(pd.DataFrame({"s": [tail_from], "t": [0]}), "T", "Mid", "s", "Tail", "t")
    return g


@pytest.mark.parametrize("tail_from", [0, 600, 1199])
@pytest.mark.parametrize(
    "query,kwargs",
    [
        # `disabled_passes` keeps the star spelling variable-length: without it
        # `lower_fixed_var_length_hops` rewrites `*1..1` to a plain hop and
        # these two cases become copies of the fixed control, silently
        # retiring the coverage this test exists for.
        (
            "MATCH (a:Src)-[:R*1..1]->(b:Mid)-[:T]->(c:Tail) RETURN c.name AS n LIMIT 5",
            {"disabled_passes": ["lower_fixed_var_length_hops"]},
        ),
        (
            "MATCH (a:Src)-[:R*1..1]->(b:Mid)-[:T]->(c:Tail) RETURN DISTINCT c.name AS n LIMIT 5",
            {"disabled_passes": ["lower_fixed_var_length_hops"]},
        ),
        (
            "MATCH (a:Src)-[:R*1..2]->(b:Mid)-[:T]->(c:Tail) RETURN c.name AS n LIMIT 5",
            {},
        ),
        (
            "MATCH (a:Src)-[:R*1..1]->(b:Mid)-[:T]->(c:Tail) RETURN c.name AS n LIMIT 5",
            {},
        ),
        ("MATCH (a:Src)-[:R]->(b:Mid)-[:T]->(c:Tail) RETURN c.name AS n LIMIT 5", {}),
    ],
    ids=[
        "var_length",
        "var_length_distinct",
        "var_length_range",
        "var_length_lowered",
        "fixed_control",
    ],
)
def test_var_length_under_limit_returns_the_row_that_exists(tail_from: int, query: str, kwargs: dict) -> None:
    """A variable-length hop is NOT exempt from advisory-cap accounting.

    The intermediate hop's advisory cap (50x `max_matches`, floor 1 000) sits
    *above* the expansion, not inside it: `expand_from_node` ignores its
    `max_results` for a variable-length segment, but the caller still stops
    pushing at the cap and drops the rest. Those 200 dropped `:Mid` rows are
    where the answer lives when `tail_from` is early in relationship order, so
    the pass must report the bite and `execute` must re-run uncapped.
    Suppressing the report for variable-length hops — on the theory that they
    never pre-cap — returns `[]` here for `tail_from=0`.
    """
    g = _cap_biting_var_length_graph(tail_from)
    assert g.cypher(query, **kwargs).to_list() == [{"n": "tail"}]


def _applied_passes(graph: kglite.KnowledgeGraph, query: str, **kwargs) -> set[str]:
    """The optimizer passes that changed the plan for `query`."""
    return {
        row["operation"].removeprefix("OptimizerPass ")
        for row in graph.cypher(f"EXPLAIN {query}", **kwargs).to_list()
        if str(row["operation"]).startswith("OptimizerPass ")
    }


LOWERING_PASS = "lower_fixed_var_length_hops"

LOWERED_SHAPES = [
    "MATCH (a:N {id: 1})-[:R*2..2]->(b:N) RETURN b.id AS i",
    "MATCH (a:N)-[:R*2..2]->(b:N) RETURN count(*) AS n",
    "MATCH (a:N {id: 1})-[:R*2..2]-(b:N) RETURN b.id AS i",
    "MATCH (a:N {id: 1})-[:R*1..1]->(b:N) RETURN b.id AS i",
    "MATCH (a:N) OPTIONAL MATCH (a)-[:R*2..2]->(b:N) RETURN a.id AS i, b.id AS j",
]

UNLOWERED_SHAPES = [
    # Past the two-hop ceiling. The lowered form reaches only the general
    # fixed-hop matcher there, which V8 measured slower than the star spelling
    # on every fixture (1.05x-3.83x) and up to 9x heavier in peak memory.
    "MATCH (a:N {id: 1})-[:R*3..3]->(b:N) RETURN b.id AS i",
    # A two-hop segment beside a plain hop is a three-hop pattern: the budget
    # is per pattern, because the fused counter reads the whole element list.
    "MATCH (a:N {id: 1})-[:R]->(x:N)-[:R*2..2]->(b:N) RETURN b.id AS i",
    # A genuine range has no fixed spelling.
    "MATCH (a:N {id: 1})-[:R*2..3]->(b:N) RETURN b.id AS i",
    "MATCH (a:N {id: 1})-[:R*1..3]->(b:N) RETURN b.id AS i",
    "MATCH (a:N {id: 1})-[:R*]->(b:N) RETURN b.id AS i",
    # `r` binds the relationship LIST.
    "MATCH (a:N {id: 1})-[r:R*2..2]->(b:N) RETURN size(r) AS hops",
    # The path variable consumes the segment's relationship sequence.
    "MATCH p = (a:N {id: 1})-[:R*2..2]->(b:N) RETURN length(p) AS l",
    # Zero-length identity: no hop expresses it.
    "MATCH (a:N {id: 1})-[:R*0..0]->(b:N) RETURN b.id AS i",
]


@pytest.mark.parametrize("query", LOWERED_SHAPES)
def test_fixed_length_var_hops_are_lowered(directed_triangle_graph, query: str) -> None:
    """The pass fires for every `*k..k` spelling inside its window."""
    assert LOWERING_PASS in _applied_passes(directed_triangle_graph, query)


@pytest.mark.parametrize("query", UNLOWERED_SHAPES)
def test_the_lowering_declines_outside_its_window(directed_triangle_graph, query: str) -> None:
    """...and declines outside it. Without this the corpus above could pass
    for the boring reason that the pass stopped firing at all."""
    assert LOWERING_PASS not in _applied_passes(directed_triangle_graph, query)


def test_the_lowering_ceiling_is_observable_from_outside(long_chain_graph) -> None:
    """Two hops lower, three do not — and both answer the same set either way.

    The ceiling is two because that is the fused counter's pattern window and
    the only depth at which V8 measured lowering faster than leaving the star
    alone; past it the lowered form was 1.05x-3.83x slower and up to 9x
    heavier in peak memory. The answers are what must not move.
    """
    at_ceiling = "MATCH (a:N {id: 0})-[:R*2..2]->(b:N) RETURN b.id AS i"
    past_ceiling = "MATCH (a:N {id: 0})-[:R*3..3]->(b:N) RETURN b.id AS i"

    assert LOWERING_PASS in _applied_passes(long_chain_graph, at_ceiling)
    assert LOWERING_PASS not in _applied_passes(long_chain_graph, past_ceiling)
    assert long_chain_graph.cypher(at_ceiling).to_list() == [{"i": 2}]
    assert long_chain_graph.cypher(past_ceiling).to_list() == [{"i": 3}]
    # Non-vacuity: the deep spelling still answers, so "not lowered" is a plan
    # difference and not a query that quietly returns nothing.
    deep = "MATCH (a:N {id: 0})-[:R*9..9]->(b:N) RETURN b.id AS i"
    assert LOWERING_PASS not in _applied_passes(long_chain_graph, deep)
    assert long_chain_graph.cypher(deep).to_list() == [{"i": 9}]


def test_the_lowering_ceiling_counts_the_whole_pattern(long_chain_graph) -> None:
    """Two segments share one budget: 1+1 lowers, 1+2 leaves both alone."""
    fits = "MATCH (a:N {id: 0})-[:R*1..1]->(b:N)-[:R*1..1]->(c:N) RETURN c.id AS i"
    over = "MATCH (a:N {id: 0})-[:R*1..1]->(b:N)-[:R*2..2]->(c:N) RETURN c.id AS i"

    assert LOWERING_PASS in _applied_passes(long_chain_graph, fits)
    assert LOWERING_PASS not in _applied_passes(long_chain_graph, over)
    assert long_chain_graph.cypher(fits).to_list() == [{"i": 2}]
    assert long_chain_graph.cypher(over).to_list() == [{"i": 3}]


def test_anonymous_intermediates_are_only_deduped_when_partials_are_interchangeable(
    square_cycle_graph, hub_return_graph
) -> None:
    """Absolute goldens for the matcher's intermediate dedup.

    `push_distinct_into_match` licenses the matcher to keep one partial match
    per anonymous intermediate node. That is only sound while nothing
    downstream can tell two partials apart, and two things can: the
    relationships they already consumed (Cypher paths are trails), and a node
    variable the pattern binds a second time. Both shipped as silent
    under-returns, and the differential leg alone would not pin the numbers.
    """
    three_hops = "MATCH (a:N {id: 1})-[:R]-()-[:R]-()-[:R]-(b:N) RETURN DISTINCT b.id AS i"
    lowered = "MATCH (a:N {id: 1})-[:R*3..3]-(b:N) RETURN DISTINCT b.id AS i"
    for query in (three_hops, lowered):
        rows = square_cycle_graph.cypher(query).to_list()
        assert sorted(row["i"] for row in rows) == [2, 4], query

    repeated = "MATCH (a:N)-[:A]->()-[:B]->(a) RETURN DISTINCT a.id AS i"
    rows = hub_return_graph.cypher(repeated).to_list()
    assert sorted(row["i"] for row in rows) == [1, 2, 3]

    # The legal case still answers what it always did.
    safe = "MATCH (a:N)-[:A]->()-[:B]->(b:N) RETURN DISTINCT b.id AS i"
    assert sorted(row["i"] for row in hub_return_graph.cypher(safe).to_list()) == [1, 2, 3]


def test_lowered_hops_keep_relationship_uniqueness_on_a_same_type_cycle(
    directed_triangle_graph,
) -> None:
    """The disjoint-types opt-out is type-based, so it can never fire on a
    lowered same-type segment — which is what keeps the rewrite a trail.

    Both claims are asserted from outside: the answer itself, and that the
    pass which would remove the trail bookkeeping declined.
    """
    undirected = "MATCH (a:N {id: 1})-[:R*2..2]-(b:N) RETURN DISTINCT b.id AS i"
    applied = _applied_passes(directed_triangle_graph, undirected)
    assert LOWERING_PASS in applied
    assert "mark_disjoint_fixed_trails" not in applied

    # 1-2-3 forwards and 1-3-2 backwards: both peers, and NOT 1 itself, which
    # would require reversing over the relationship just used.
    rows = directed_triangle_graph.cypher(undirected).to_list()
    assert sorted(row["i"] for row in rows) == [2, 3]


@pytest.fixture
def vector_props_graph() -> kglite.KnowledgeGraph:
    """Nodes carrying a genuinely list-valued numeric property.

    Separate from `small_graph` / `json_list_props_graph` — which are pinned by
    absolute goldens elsewhere — so adding the shape cannot move any existing
    expectation. `zero` is the control for the undefined-cosine arm: its norm
    is 0, so `cosine(...)` on it is NULL while `dot`/`norm` stay numbers.
    """
    import pandas as pd

    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {
                "doc_id": [1, 2, 3, 4],
                "title": ["axis", "diag", "opposite", "zero"],
                "vec": [[1.0, 0.0], [1.0, 1.0], [-1.0, 0.0], [0.0, 0.0]],
            }
        ),
        "Doc",
        "doc_id",
        "title",
        columns=["vec"],
    )
    return g


def build_cross_type_id_graph() -> kglite.KnowledgeGraph:
    """Three labels sharing the same id space: Alpha/Beta/Gamma × ids {1, 2}.

    Ids are unique *within* a type (no intra-type duplicates) and deliberately
    reused *across* types — the shape `MATCH (n {id: 2})` must answer with one
    node per label. Every node carries a distinct `name` so a collapsed result
    names which label survived.

    Two edges, from two different labels' id-1 node, so the anchor-as-expansion
    -start shape counts more than one type's worth of edges:

        Alpha#1 ─LINKS─► Beta#2
        Beta#1  ─LINKS─► Gamma#2

    Kept out of conftest and separate from `small_graph` — which is pinned by
    absolute goldens across the suite — so adding the shape cannot move any
    existing expectation.
    """
    import pandas as pd

    g = kglite.KnowledgeGraph()
    for label in ("Alpha", "Beta", "Gamma"):
        g.add_nodes(
            pd.DataFrame({"nid": [1, 2], "name": [f"{label.lower()}_1", f"{label.lower()}_2"]}),
            label,
            "nid",
            "name",
        )
    g.add_connections(pd.DataFrame({"src": [1], "dst": [2]}), "LINKS", "Alpha", "src", "Beta", "dst")
    g.add_connections(pd.DataFrame({"src": [1], "dst": [2]}), "LINKS", "Beta", "src", "Gamma", "dst")
    return g


@pytest.fixture
def cross_type_id_graph() -> kglite.KnowledgeGraph:
    """See :func:`build_cross_type_id_graph`."""
    return build_cross_type_id_graph()


def build_ontology_closure_graph(open_label: bool = False) -> kglite.KnowledgeGraph:
    """A **materialized ontology closure**: abstract `:Person` over `:Student`
    and `:Teacher`, plus an unrelated `:Class`.

    Index coverage is deliberately mixed, so one fixture carries all three
    closure-probe outcomes:

    * `email` — indexed on **both** members: the probe engages.
    * `dept`  — indexed on `Student` only: partial coverage, wholesale decline
      (a union missing Teacher's rows would silently drop them).
    * `title` — indexed nowhere: plain scan.

    `open_label=True` additionally stamps `:Person` onto the `:Class` node, so
    the managed label is **Open** — a carrier no descendant probe covers, which
    must send every shape back to the scan.

    Kept out of conftest and out of the shared fixtures — which are pinned by
    absolute goldens across the suite — so adding the shape cannot move any
    existing expectation. Materialization stamps secondary labels, and every
    query in this corpus that counts labels reads a fixture this one does not
    touch.
    """
    import pandas as pd

    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["Ann", "Bo", "Cy"],
                "email": ["ann@x", "bo@x", "cy@x"],
                "dept": ["Sci", "Sci", "Art"],
            }
        ),
        "Student",
        "id",
        node_title_field="title",
    )
    g.add_nodes(
        pd.DataFrame(
            {
                "id": [10, 11],
                "title": ["Tea", "Uli"],
                "email": ["tea@x", "uli@x"],
                "dept": ["Sci", "Art"],
            }
        ),
        "Teacher",
        "id",
        node_title_field="title",
    )
    g.add_nodes(
        pd.DataFrame({"id": [100], "title": ["Math"], "email": ["math@x"], "dept": ["Sci"]}),
        "Class",
        "id",
        node_title_field="title",
    )
    g.define_ontology(
        {
            "classes": {
                "Person": {"abstract": True},
                "Student": {"is_a": "Person"},
                "Teacher": {"is_a": "Person"},
            }
        }
    )
    g.materialize_ontology()
    g.create_index("Student", "email")
    g.create_index("Teacher", "email")
    g.create_index("Student", "dept")
    if open_label:
        g.cypher("MATCH (c:Class) SET c:Person")
    return g


def test_ontology_fixture_really_probes_and_declines() -> None:
    """Non-vacuity + absolute goldens for the ontology corpus.

    The differential runs cannot see this on their own: the closure probe is
    a matcher decision, not an optimizer pass, so `disable_optimizer=True`
    takes the same route for the inline-property shapes and both halves would
    agree on a wrong answer. These are the expected *values*, plus the plan
    marker that says the probe is the thing producing them.
    """
    g = build_ontology_closure_graph()

    def probes(query: str) -> bool:
        rows = g.cypher(f"EXPLAIN {query}").to_list()
        return any(str(row["operation"]).startswith("ClosureProbe") for row in rows)

    def ids(query: str) -> list:
        return [row["id"] for row in g.cypher(query).to_list()]

    # Covered on both members → probe, and each member's own value is found.
    assert probes("MATCH (p:Person {email: 'tea@x'}) RETURN p.id")
    assert ids("MATCH (p:Person {email: 'tea@x'}) RETURN p.id AS id") == [10]
    assert ids("MATCH (p:Person {email: 'ann@x'}) RETURN p.id AS id") == [1]
    assert ids("MATCH (p:Person {email: 'nobody@x'}) RETURN p.id AS id") == []
    # The Class node holds `math@x` but not `:Person` — the closure is closed.
    assert ids("MATCH (p:Person {email: 'math@x'}) RETURN p.id AS id") == []

    # Partial coverage → decline, and the uncovered member's rows survive.
    assert not probes("MATCH (p:Person {dept: 'Sci'}) RETURN p.id")
    assert sorted(ids("MATCH (p:Person {dept: 'Sci'}) RETURN p.id AS id")) == [1, 2, 10]

    # Open label → decline everywhere, foreign carrier included in the answer.
    opened = build_ontology_closure_graph(open_label=True)
    assert not any(
        str(row["operation"]).startswith("ClosureProbe")
        for row in opened.cypher("EXPLAIN MATCH (p:Person {email: 'tea@x'}) RETURN p.id").to_list()
    )
    assert [row["id"] for row in opened.cypher("MATCH (p:Person {email: 'math@x'}) RETURN p.id AS id").to_list()] == [
        100
    ]
    assert opened.cypher("MATCH (p:Person) RETURN count(p) AS c").to_list() == [{"c": 6}]


def build_alternation_graph(*, overlap: bool = False, foreign_label: bool = False) -> kglite.KnowledgeGraph:
    """Three indexed sibling labels, for the label-alternation fast paths.

    `email` is indexed on every one of :Student/:Teacher/:Staff, so
    `(n:A|B {email: …})` is fully covered and probes per branch; `dept` is
    indexed on :Student only, so `(n:A|B {dept: …})` is the partial-coverage
    decline in the same fixture.

    `overlap=True` gives node 1 a *secondary* :Teacher on top of its primary
    :Student — legal sibling overlap. It is the reason the branch-sum count
    fusion must bail (5 distinct nodes, branch sum 6) and the reason the probe
    path must dedup (branch :Student's index hit and branch :Teacher's carrier
    scan both name node 1).

    `foreign_label=True` puts a secondary label on a *different* label, so the
    graph has secondary labels but neither alternation branch does — the case
    the per-label disjointness proof must still fuse.
    """
    import pandas as pd

    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["Ann", "Bob", "Cy"],
                "email": ["ann@x", "bob@x", "cy@x"],
                "dept": ["Sci", "Sci", "Hum"],
            }
        ),
        "Student",
        "id",
        node_title_field="title",
    )
    g.add_nodes(
        pd.DataFrame({"id": [10, 11], "title": ["Tea", "Tom"], "email": ["tea@x", "tom@x"], "dept": ["Sci", "Hum"]}),
        "Teacher",
        "id",
        node_title_field="title",
    )
    g.add_nodes(
        pd.DataFrame({"id": [20, 21], "title": ["Sam", "Sue"], "email": ["sam@x", "sue@x"], "dept": ["Ops", "Ops"]}),
        "Staff",
        "id",
        node_title_field="title",
    )
    for label in ("Student", "Teacher", "Staff"):
        g.create_index(label, "email")
    g.create_index("Student", "dept")
    if overlap:
        g.cypher("MATCH (s:Student {id: 1}) SET s:Teacher")
    if foreign_label:
        g.cypher("MATCH (s:Staff {id: 20}) SET s:OnCall")
    return g


def test_alternation_fixture_fuses_and_probes() -> None:
    """Non-vacuity + absolute goldens for the alternation corpus entries.

    Two things the differential runs cannot see for themselves. The count
    fusion IS an optimizer pass, so the corpus compares it against the
    unoptimized path — but nothing there proves the pass *fired*, which the
    EXPLAIN operation name does. The per-branch index probe is a matcher
    decision, not a pass, so `disable_optimizer=True` takes the identical
    route and both halves would agree on a wrong answer; these are its
    expected values.
    """
    clean = build_alternation_graph()
    overlap = build_alternation_graph(overlap=True)
    foreign = build_alternation_graph(foreign_label=True)

    def ops(g: kglite.KnowledgeGraph, query: str) -> list[str]:
        return [str(row["operation"]) for row in g.cypher(f"EXPLAIN {query}").to_list()]

    two = "MATCH (n:Student|Teacher) RETURN count(n) AS c"
    three = "MATCH (n:Student|Teacher|Staff) RETURN count(n) AS c"

    assert "FusedCountLabelUnion :Student|Teacher" in ops(clean, two)
    assert "FusedCountLabelUnion :Student|Teacher|Staff" in ops(clean, three)
    assert clean.cypher(two).to_list() == [{"c": 5}]
    assert clean.cypher(three).to_list() == [{"c": 7}]

    # A secondary label on an unrelated label leaves the branches disjoint.
    assert "FusedCountLabelUnion :Student|Teacher" in ops(foreign, two)
    assert foreign.cypher(two).to_list() == [{"c": 5}]

    # Overlapping branches: no fusion, and node 1 is counted once, not twice.
    assert not any(op.startswith("FusedCountLabelUnion") for op in ops(overlap, two))
    assert overlap.cypher(two).to_list() == [{"c": 5}]

    def ids(g: kglite.KnowledgeGraph, query: str) -> list:
        return [row["id"] for row in g.cypher(query).to_list()]

    # Fully covered equality → per-branch probe; each branch's own value found.
    assert ids(clean, "MATCH (n:Student|Teacher {email: 'tea@x'}) RETURN n.id AS id") == [10]
    assert ids(clean, "MATCH (n:Student|Teacher {email: 'ann@x'}) RETURN n.id AS id") == [1]
    assert ids(clean, "MATCH (n:Student|Teacher {email: 'sam@x'}) RETURN n.id AS id") == []
    assert ids(clean, "MATCH (n:Student|Teacher {email: 'nobody@x'}) RETURN n.id AS id") == []
    # Partial coverage (`dept` is indexed on :Student only) → decline, and the
    # uncovered branch's rows survive.
    assert sorted(ids(clean, "MATCH (n:Student|Teacher {dept: 'Sci'}) RETURN n.id AS id")) == [1, 2, 10]

    # The dedup case: node 1 is reachable through both branches at once.
    assert ids(overlap, "MATCH (n:Student|Teacher {email: 'ann@x'}) RETURN n.id AS id") == [1]
    assert sorted(ids(overlap, "MATCH (n:Student|Teacher {dept: 'Sci'}) RETURN n.id AS id")) == [1, 2, 10]


@pytest.fixture
def alternation_graph() -> kglite.KnowledgeGraph:
    """See :func:`build_alternation_graph`."""
    return build_alternation_graph()


@pytest.fixture
def alternation_overlap_graph() -> kglite.KnowledgeGraph:
    """See :func:`build_alternation_graph` — the `overlap` variant."""
    return build_alternation_graph(overlap=True)


@pytest.fixture
def alternation_foreign_label_graph() -> kglite.KnowledgeGraph:
    """See :func:`build_alternation_graph` — the `foreign_label` variant."""
    return build_alternation_graph(foreign_label=True)


@pytest.fixture
def ontology_closure_graph() -> kglite.KnowledgeGraph:
    """See :func:`build_ontology_closure_graph`."""
    return build_ontology_closure_graph()


@pytest.fixture
def ontology_open_label_graph() -> kglite.KnowledgeGraph:
    """See :func:`build_ontology_closure_graph` — the `open_label` variant."""
    return build_ontology_closure_graph(open_label=True)


def build_mixed_type_props_graph() -> kglite.KnowledgeGraph:
    """`:Sample` nodes whose `v` property holds numbers *and* strings.

    Built with Cypher `CREATE` rather than `add_nodes`: the bulk loader types a
    column once, so a pandas object column of mixed values arrives as all
    strings — the shape that cannot exercise a mixed-type accumulator at all.

    Three sites give three regimes in one fixture:

    * ``north`` — numbers with one string cell (the bug's shape: 3 non-null
      values, 2 of them numeric).
    * ``south`` — one number, one string, and one null.
    * ``east``  — no numeric value at all, so `avg` is null and `sum` is 0.

    `w` is numeric everywhere and is the control column. Kept separate from
    `small_graph`/`social_graph`, which absolute goldens elsewhere pin, so
    adding the shape cannot move an existing expectation.
    """
    g = kglite.KnowledgeGraph()
    g.cypher(
        "CREATE (:Sample {id: 1, name: 'n1', site: 'north', v: 10, w: 1}),"
        "       (:Sample {id: 2, name: 'n2', site: 'north', v: 20, w: 2}),"
        "       (:Sample {id: 3, name: 'n3', site: 'north', v: 'hello', w: 3}),"
        "       (:Sample {id: 4, name: 's1', site: 'south', v: 4, w: 4}),"
        "       (:Sample {id: 5, name: 's2', site: 'south', v: 'n/a', w: 5}),"
        "       (:Sample {id: 6, name: 's3', site: 'south', w: 6}),"
        "       (:Sample {id: 7, name: 'e1', site: 'east', v: 'a', w: 7}),"
        "       (:Sample {id: 8, name: 'e2', site: 'east', v: 'b', w: 8})"
    )
    return g


@pytest.fixture
def mixed_type_props_graph() -> kglite.KnowledgeGraph:
    """See :func:`build_mixed_type_props_graph`."""
    return build_mixed_type_props_graph()


def test_mixed_type_fixture_really_mixes_types() -> None:
    """Non-vacuity guard for the `mixed_type_props_graph` corpus entries.

    They compare two plans; if `v` ever came back as a single type (a loader
    change, a schema coercion), the comparison would still pass while
    measuring nothing. Assert the stored values directly.
    """
    g = build_mixed_type_props_graph()
    values = [row["v"] for row in g.cypher("MATCH (n:Sample) RETURN n.v AS v ORDER BY n.id").to_list()]
    assert [type(v).__name__ for v in values] == [
        "int",
        "int",
        "str",
        "int",
        "str",
        "NoneType",
        "str",
        "str",
    ], values


@pytest.fixture
def json_list_props_graph() -> kglite.KnowledgeGraph:
    """Nodes whose string property holds a JSON-encoded single-element list.

    Kept separate from `small_graph`/`social_graph` — which are pinned by
    absolute goldens all over the suite — so that adding the shape cannot
    change any existing expectation. `stray` is the control: an ordinary
    string that must keep answering by plain byte equality.
    """
    import pandas as pd

    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {
                "doc_id": [1, 2, 3, 4],
                "title": ["listed", "plain", "other", "stray"],
                "tag": ['["Oslo"]', "Oslo", '["Bergen"]', "Trondheim"],
            }
        ),
        "Doc",
        "doc_id",
        "title",
        columns=["tag"],
    )
    return g


# Mutation queries: each test gets its own fresh fixture so state-bleed
# between mutations is impossible. The harness's identity for mutations
# is "optimized result on a fresh fixture == naive result on a fresh
# fixture."
MUTATION_QUERIES: list[tuple[str, str]] = [
    ("create_node", "CREATE (p:Person {person_id: 99, name: 'X', age: 50}) RETURN p.person_id AS pid"),
    ("set_property", "MATCH (p:Person {person_id: 1}) SET p.age = 99 RETURN p.age AS age"),
    (
        "set_map_merge",
        "MATCH (p:Person {person_id: 1}) SET p += {age: 99, active: true} RETURN p.age AS age, p.active AS active",
    ),
    (
        "set_map_replace",
        "MATCH (p:Person {person_id: 1}) SET p = {name: 'A', age: 99} RETURN p.name AS name, p.age AS age",
    ),
    ("set_with_filter", "MATCH (p:Person) WHERE p.age > 30 SET p.bucket = 'old' RETURN count(p) AS n"),
    ("detach_delete", "MATCH (p:Person {person_id: 3}) DETACH DELETE p"),
    ("remove_property", "MATCH (p:Person {person_id: 1}) REMOVE p.name RETURN p.person_id AS pid"),
    # Constraint-enforcement write shapes. The planner must not reorder or fuse
    # these into a form that skips the pre-write constraint gate, and the gate
    # itself must not change a *conforming* write's result. Each shape here is
    # one the gate inspects: a CREATE whose properties it reads, a SET that
    # vacates and re-claims a unique tuple, and a REMOVE of an unconstrained
    # property on a type that carries a constraint.
    (
        "create_node_under_unique_constraint",
        "CREATE (p:Person {person_id: 101, name: 'Fresh', age: 20}) RETURN p.person_id AS pid, p.name AS name",
    ),
    (
        "set_moves_value_then_frees_it",
        "MATCH (p:Person {person_id: 1}) SET p.name = 'Moved' RETURN p.name AS name",
    ),
    (
        "remove_unconstrained_property_on_constrained_type",
        "MATCH (p:Person {person_id: 2}) REMOVE p.age RETURN p.person_id AS pid",
    ),
    (
        "merge_create",
        "MERGE (p:Person {person_id: 100}) ON CREATE SET p.age = 1 RETURN p.person_id AS pid, p.age AS age",
    ),
    ("merge_match", "MERGE (p:Person {person_id: 1}) ON MATCH SET p.touched = true RETURN p.touched AS t"),
    (
        "multi_create",
        "CREATE (a:Person {person_id: 300, name: 'A', age: 10}), "
        "(b:Person {person_id: 301, name: 'B', age: 20}) RETURN count(*) AS n",
    ),
    (
        "match_create_edge",
        "MATCH (a:Person {person_id: 1}), (b:Person {person_id: 2}) CREATE (a)-[:KNOWS_NEW]->(b) RETURN count(*) AS n",
    ),
    (
        "set_rel_property",
        "MATCH (p:Person)-[r:KNOWS]->(q:Person) SET r.since = 2099 RETURN count(r) AS n",
    ),
    (
        "set_rel_map",
        "MATCH (p:Person)-[r:KNOWS]->(q:Person) SET r += {since: 2099, active: true} RETURN count(r) AS n",
    ),
    (
        "remove_rel_property",
        "MATCH (p:Person)-[r:KNOWS]->(q:Person) REMOVE r.since RETURN count(r) AS n",
    ),
    # Schema DDL: schema is graph state, so these route to the mutable engine
    # like any other write. The harness compares the returned rows AND the
    # post-statement node/edge counts across optimized and naive runs, so a
    # pass that mangled a schema clause into a data mutation would show up as a
    # count divergence.
    ("create_index_ddl", "CREATE INDEX FOR (p:Person) ON (p.name)"),
    ("create_index_ddl_if_not_exists", "CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.name)"),
    ("create_composite_index_ddl", "CREATE INDEX FOR (p:Person) ON (p.city, p.age)"),
    ("create_range_index_ddl", "CREATE RANGE INDEX person_age FOR (p:Person) ON (p.age)"),
    ("drop_index_ddl_missing_if_exists", "DROP INDEX Person.name IF EXISTS"),
    # Constraint DDL. Declaring a constraint rebuilds an enforcement structure
    # from live data, so a pass that reordered or duplicated the statement would
    # change post-statement graph state — which this harness compares. The
    # fixture's `name` and `age` are both distinct across all three rows, so each
    # declaration is satisfiable. `person_id` is deliberately avoided: it is the
    # fixture's id field, and uniqueness DDL over `id` is refused.
    ("create_constraint_unique_ddl", "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS UNIQUE"),
    (
        "create_constraint_unique_ddl_if_not_exists",
        "CREATE CONSTRAINT person_name_u IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
    ),
    (
        "create_constraint_composite_unique_ddl",
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE (p.name, p.age) IS UNIQUE",
    ),
    ("create_constraint_not_null_ddl", "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS NOT NULL"),
    ("create_constraint_node_key_ddl", "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS NODE KEY"),
    ("drop_constraint_ddl_missing_if_exists", "DROP CONSTRAINT person_name_u IF EXISTS"),
    # Multi-statement merge/drop sequences live in CONSTRAINT_DDL_SEQUENCES below.
    # ── Shapes whose rollback checkpoint is an undo journal ──────────────
    #
    # Statement atomicity is bought with an inverse-op journal rather than a
    # whole-graph clone, and the journal captures at two seams: the storage
    # backend, and the DirGraph-level choke points for secondary labels and
    # node deletion. The shapes below are the ones that reach the second seam
    # or that mix several seams in one statement, so a capture gap shows up
    # here as a diverging result.
    (
        "create_with_secondary_labels",
        "CREATE (p:Person:Employee:Remote {person_id: 400, name: 'L', age: 33}) RETURN labels(p) AS ls",
    ),
    (
        "set_label",
        "MATCH (p:Person {person_id: 1}) SET p:Employee RETURN labels(p) AS ls",
    ),
    (
        "remove_label",
        "MATCH (p:Person {person_id: 1}) SET p:Employee WITH p REMOVE p:Employee RETURN labels(p) AS ls",
    ),
    (
        "delete_edge_only",
        "MATCH (p:Person)-[r:KNOWS]->(q:Person) DELETE r "
        "WITH 1 AS done MATCH (:Person)-[r2:KNOWS]->(:Person) RETURN count(r2) AS n",
    ),
    (
        "foreach_create",
        "FOREACH (i IN [500, 501, 502] | CREATE (:Person {person_id: i, age: 1})) "
        "WITH 1 AS done MATCH (p:Person) RETURN count(p) AS n",
    ),
    (
        # Three seams in one statement: a property write, a node create, and a
        # detach-delete — the journal must interleave their inverses.
        "multi_clause_set_create_delete",
        "MATCH (p:Person {person_id: 1}) SET p.age = 77 "
        "CREATE (n:Person {person_id: 600, name: 'N', age: 2}) "
        "WITH n MATCH (d:Person {person_id: 2}) DETACH DELETE d "
        "WITH 1 AS done MATCH (p:Person) RETURN count(p) AS n",
    ),
    (
        # Delete then re-create the same identity in one statement: the
        # journal's structural entries must replay in order for the freed slot
        # to be handed back correctly.
        "delete_then_recreate_same_id",
        "MATCH (p:Person {person_id: 3}) DETACH DELETE p "
        "CREATE (q:Person {person_id: 3, name: 'again', age: 44}) "
        "RETURN q.name AS name",
    ),
    (
        # Labels and properties are independent state captured at *different*
        # seams — labels at the DirGraph choke point, properties at the
        # storage backend — and the WAL folds them into separate net maps.
        # Interleaving them in one statement is the shape where a fold that
        # let a property write clobber a label set (or vice versa) diverges.
        "interleaved_label_and_property_writes",
        "MATCH (p:Person {person_id: 1}) SET p:Employee SET p.age = 55 "
        "SET p:Remote SET p.bucket = 'x' WITH p REMOVE p:Employee "
        "RETURN labels(p) AS ls, p.age AS age, p.bucket AS bucket",
    ),
    (
        # Several label writes on one node collapse to a single whole-set WAL
        # op resolved against final state; the same collapse must not lose an
        # intermediate add that survives to the end.
        "many_labels_one_node",
        "MATCH (p:Person {person_id: 2}) SET p:A SET p:B SET p:C WITH p REMOVE p:B RETURN labels(p) AS ls",
    ),
    # ── Write clauses fed an empty binding stream ────────────────────────
    #
    # A clause that produced zero rows must leave every downstream write with
    # nothing to do. What these pin for the optimizer specifically: no pass may
    # drop, hoist, or fuse a write clause out from behind the emptying clause,
    # because doing so would restore the write to a *leading* position — where
    # Cypher's implicit start row legitimately applies and one node really is
    # created. Optimized and naive therefore have to agree on post-state node
    # and edge counts, which is what the harness compares.
    #
    # `person_id: 999` and the `Ghost` label are chosen to match nothing in the
    # fixture; `Ghost` also makes the pattern's node type unknown, the shape the
    # planner emits an "unknown node label … returns no rows" warning for.
    (
        "create_after_empty_inline_map_match",
        "MATCH (p:Person {person_id: 999}) CREATE (t:Task {person_id: 900}) RETURN t.person_id AS pid",
    ),
    (
        "create_after_empty_where_match",
        "MATCH (p:Person) WHERE p.person_id = 999 CREATE (t:Task {person_id: 901}) RETURN t.person_id AS pid",
    ),
    (
        "create_after_empty_match_with",
        "MATCH (p:Person {person_id: 999}) WITH p CREATE (t:Task {person_id: 902}) RETURN t.person_id AS pid",
    ),
    (
        "create_after_empty_relationship_match",
        "MATCH (p:Person)-[:NO_SUCH_EDGE]->(q:Person) CREATE (t:Task {person_id: 903}) RETURN t.person_id AS pid",
    ),
    (
        # The phantom-endpoint shape: `g` binds nothing, so creating the edge
        # would have to invent its target node. Post-state edge count is the
        # assertion that bites.
        "create_edge_after_partially_unmatched_multi_pattern",
        "MATCH (p:Person {person_id: 1}), (g:Ghost {person_id: 999}) "
        "CREATE (p)-[:ASSIGNED_TO]->(g) RETURN count(*) AS n",
    ),
    (
        "merge_after_empty_match",
        "MATCH (p:Person {person_id: 999}) MERGE (t:Task {person_id: 904}) RETURN t.person_id AS pid",
    ),
    (
        "foreach_after_empty_match",
        "MATCH (p:Person {person_id: 999}) FOREACH (i IN [910, 911] | CREATE (:Task {person_id: i})) "
        "WITH 1 AS done MATCH (n) RETURN count(n) AS n",
    ),
    (
        "create_after_unwind_empty_list",
        "UNWIND [] AS x CREATE (t:Task {person_id: 905}) RETURN t.person_id AS pid",
    ),
    (
        # The control living in the corpus: a *leading* CREATE still gets
        # Cypher's implicit start row. Pins that the empty-stream rule was not
        # over-applied to writes that open a query.
        "leading_create_still_runs_once",
        "CREATE (t:Task {person_id: 906}) WITH 1 AS done MATCH (n) RETURN count(n) AS n",
    ),
    (
        # OPTIONAL MATCH is the deliberate opposite: it null-pads rather than
        # emptying, so the CREATE downstream of it must still run.
        "create_after_optional_match_miss",
        "MATCH (p:Person {person_id: 1}) OPTIONAL MATCH (g:Ghost) "
        "CREATE (t:Task {person_id: 907}) RETURN t.person_id AS pid, g AS g",
    ),
]


def _normalize(rows: list[dict]) -> list[tuple]:
    """Sort + canonicalize rows so unordered queries compare equal.

    Modeled on `tests/test_storage_parity.py::_rows()`. Each row becomes
    a tuple of (key, str(value)) pairs sorted by key — handles dict
    ordering and mixed numeric/string types. Final list is sorted so
    queries without ORDER BY are still comparable.
    """
    canonical = [tuple(sorted((k, str(v)) for k, v in row.items())) for row in rows]
    canonical.sort()
    return canonical


@pytest.mark.differential
@pytest.mark.parametrize(
    "name,fixture,query,params",
    DIFFERENTIAL_QUERIES,
    ids=[entry[0] for entry in DIFFERENTIAL_QUERIES],
)
def test_optimized_matches_naive(
    name: str,
    fixture: str,
    query: str,
    params: dict | None,
    request: pytest.FixtureRequest,
) -> None:
    """Run `query` against `fixture` with optimizer on, then off; assert equal rows."""
    g = request.getfixturevalue(fixture)
    kwargs = {"params": params} if params else {}

    naive = _normalize(g.cypher(query, disable_optimizer=True, **kwargs).to_list())
    optimized = _normalize(g.cypher(query, **kwargs).to_list())

    assert optimized == naive, (
        f"Optimizer divergence on `{name}`:\n"
        f"  query:     {query}\n"
        f"  optimized: {optimized[:5]}{'...' if len(optimized) > 5 else ''} ({len(optimized)} rows)\n"
        f"  naive:     {naive[:5]}{'...' if len(naive) > 5 else ''} ({len(naive)} rows)\n"
        f"  diff (in optimized but not naive): {[r for r in optimized if r not in naive][:3]}\n"
        f"  diff (in naive but not optimized): {[r for r in naive if r not in optimized][:3]}\n"
        f"To bisect: rerun with disabled_passes=[<one pass at a time>] until divergence resolves.\n"
        f"Pass list: kglite.cypher_pass_names()"
    )

    # Every pass must remain an optional optimization. Run the complete corpus
    # with each pass disabled in isolation so one pass cannot silently become
    # load-bearing for another pass's correctness. This is intentionally the
    # full corpus rather than a representative cohort: every registered pass ×
    # the current corpus is cheap, and failures identify the exact query/pass
    # pair.
    for pass_name in kglite.cypher_pass_names():
        isolated = _normalize(g.cypher(query, disabled_passes=[pass_name], **kwargs).to_list())
        assert isolated == naive, (
            f"Single-pass isolation divergence on `{name}` with `{pass_name}` disabled:\n"
            f"  query:     {query}\n"
            f"  isolated:  {isolated[:5]}{'...' if len(isolated) > 5 else ''} ({len(isolated)} rows)\n"
            f"  naive:     {naive[:5]}{'...' if len(naive) > 5 else ''} ({len(naive)} rows)\n"
            f"  diff (in isolated but not naive): {[r for r in isolated if r not in naive][:3]}\n"
            f"  diff (in naive but not isolated): {[r for r in naive if r not in isolated][:3]}"
        )


# ── Known divergences (xfail) ────────────────────────────────────────
#
# Shapes that diverge between optimized and naive and whose fix is not yet
# available. They land here as permanent regression tests: when the fix lands,
# flip xfail → expected pass and the test starts protecting it.

KNOWN_DIVERGENT: list[tuple[str, str, str, str]] = [
    # Empty: every divergence the harness has surfaced is now fixed and
    # tracked as a regular passing entry above. Future bugs the harness
    # finds land here when the fix needs design discussion or is
    # blocked; otherwise they go straight to DIFFERENTIAL_QUERIES with
    # the fix in the same commit.
]


# Machine-readable ownership: every registered optimizer pass names one query
# that must make the pass change an EXPLAIN plan. Schema-dependent passes live
# in test_cypher_specialized_optimizer; all others point into the differential
# corpus above. The applied-pass trace makes this stronger than comment-only
# coverage: a gate regression that silently stops firing fails CI.
PASS_TRIGGER_CASES: dict[str, tuple[str, str]] = {
    "fuse_text_bm25_order_limit": ("differential", "text_bm25_top_k"),
    "optimize_nested_queries": ("differential", "call_uncorrelated_body_fusion_then_limit"),
    "lower_fixed_var_length_hops": ("differential", "lower_two_hop_unanchored_count"),
    "rewrite_count_bound_var_to_star": ("differential", "count_all_typed"),
    "push_where_into_match.1": ("differential", "where_eq"),
    "anchor_element_id": ("differential", "element_id_anchor_param"),
    "fold_or_to_in": ("differential", "or_chain_to_in"),
    "push_where_into_match.2": ("differential", "or_chain_to_in"),
    "extract_pushable_rel_predicates": ("differential", "rel_property_filter"),
    "fold_pass_through_with": ("differential", "pass_through_with"),
    "narrow_unwind_source": ("differential", "narrow_unwind_source_dead"),
    "desugar_multi_match_return_aggregate": ("differential", "multi_match_group_agg"),
    "fuse_spatial_join": ("specialized", "spatial_join"),
    "reorder_match_clauses": ("specialized", "reorder_match_clauses"),
    "reorder_cyclic_pattern_edges": ("specialized", "reorder_cyclic_pattern_edges"),
    "optimize_pattern_start_node": ("specialized", "optimize_pattern_start_node"),
    "reorder_match_patterns": ("specialized", "reorder_match_patterns"),
    "push_limit_into_match": ("differential", "limit_simple"),
    "push_limit_into_aggregate": ("differential", "trigger_push_limit_into_aggregate"),
    "push_distinct_into_match": ("differential", "distinct_with_match"),
    "fuse_anchored_edge_count": ("differential", "trigger_anchored_edge_count"),
    "fuse_count_short_circuits": ("differential", "trigger_count_short_circuit"),
    "fuse_optional_match_aggregate": ("differential", "count_optional_edge_var"),
    "fuse_match_return_aggregate": ("differential", "trigger_match_return_aggregate"),
    "fuse_match_with_aggregate": ("differential", "trigger_match_with_aggregate"),
    "fuse_match_with_aggregate_top_k": ("differential", "trigger_match_with_top_k"),
    "fuse_node_scan_aggregate": ("differential", "trigger_node_scan_aggregate"),
    "fuse_node_scan_top_k": ("differential", "trigger_node_scan_top_k"),
    "fuse_vector_score_order_limit": ("specialized", "vector_score_top_k"),
    "fuse_order_by_top_k": ("differential", "trigger_generic_top_k"),
    "reorder_predicates_by_cost": ("differential", "trigger_predicate_reorder"),
    "mark_fast_var_length_paths": ("differential", "var_length_no_var_distinct"),
    "mark_disjoint_fixed_trails": (
        "differential",
        "disjoint_fixed_relationship_types",
    ),
    "mark_skip_target_type_check": ("differential", "anchored_three_hop"),
}

# Independent shapes for shared-corpus passes which otherwise had only one
# positive trigger. These are deliberately different branches (reversed
# equality operands, WITH vs RETURN, reverse direction, ascending vs
# descending, OR vs AND) so a narrow gate regression cannot hide behind two
# near-identical queries.
PASS_SECONDARY_TRIGGER_CASES: dict[str, str] = {
    # The parameter spelling of the pushed equality: `WHERE id(v) = $x` had no
    # extraction arm at all until 2026-08-15, so `push_where_into_match.1`
    # fired for the literal and silently declined the parameter. One pass maps
    # to exactly one secondary case here, and `.1` had none.
    "push_where_into_match.1": "id_function_param",
    # A different arm of the same pass: the undirected spelling on a cyclic
    # graph, where relationship uniqueness across the lowered hops is what
    # keeps the answer right.
    "lower_fixed_var_length_hops": "lower_two_hop_undirected_parallel_relationships",
    "fold_or_to_in": "or_chain_reversed_literals",
    "push_where_into_match.2": "or_chain_reversed_literals",
    "desugar_multi_match_return_aggregate": "multi_match_two_property_group",
    "push_limit_into_aggregate": "trigger_push_limit_into_aggregate_with",
    "fuse_anchored_edge_count": "trigger_anchored_edge_count_reverse",
    "fuse_match_with_aggregate_top_k": "trigger_match_with_top_k_ascending",
    "reorder_predicates_by_cost": "trigger_predicate_reorder_or",
}


def test_every_registered_pass_has_a_trigger_case() -> None:
    assert set(PASS_TRIGGER_CASES) == set(kglite.cypher_pass_names())
    differential_ids = {entry[0] for entry in DIFFERENTIAL_QUERIES}
    specialized_ids = {
        "spatial_join",
        "vector_score_top_k",
        "text_score_top_k",
        "reorder_match_clauses",
        "reorder_cyclic_pattern_edges",
        "optimize_pattern_start_node",
        "reorder_match_patterns",
    }

    for source, case_id in PASS_TRIGGER_CASES.values():
        available = differential_ids if source == "differential" else specialized_ids
        assert case_id in available


@pytest.mark.parametrize(
    "pass_name,case_id",
    [(pass_name, case_id) for pass_name, (source, case_id) in PASS_TRIGGER_CASES.items() if source == "differential"],
)
def test_registered_pass_changes_its_trigger_plan(pass_name, case_id, request) -> None:
    cases = {entry[0]: entry for entry in DIFFERENTIAL_QUERIES}
    _, fixture, query, params = cases[case_id]
    graph = request.getfixturevalue(fixture)
    kwargs = {"params": params} if params else {}
    plan = graph.cypher(f"EXPLAIN {query}", **kwargs).to_list()
    operations = [row["operation"] for row in plan]
    assert f"OptimizerPass {pass_name}" in operations


@pytest.mark.parametrize("pass_name,case_id", PASS_SECONDARY_TRIGGER_CASES.items())
def test_thin_pass_changes_an_independent_trigger_plan(pass_name, case_id, request) -> None:
    cases = {entry[0]: entry for entry in DIFFERENTIAL_QUERIES}
    _, fixture, query, params = cases[case_id]
    graph = request.getfixturevalue(fixture)
    kwargs = {"params": params} if params else {}
    plan = graph.cypher(f"EXPLAIN {query}", **kwargs).to_list()
    operations = [row["operation"] for row in plan]
    assert f"OptimizerPass {pass_name}" in operations


@pytest.mark.differential
@pytest.mark.skipif(
    not KNOWN_DIVERGENT,
    reason="no known divergences pending — corpus is clean (this is the desired state)",
)
@pytest.mark.parametrize(
    "name,fixture,query,reason",
    KNOWN_DIVERGENT,
    ids=[entry[0] for entry in KNOWN_DIVERGENT],
)
def test_known_divergences(
    name: str,
    fixture: str,
    query: str,
    reason: str,
    request: pytest.FixtureRequest,
) -> None:
    """Documented divergence — xfail'd until fixed.

    Once a fix lands, the test starts passing and pytest will flag the
    xfail-as-passing — that's the signal to remove the entry from
    KNOWN_DIVERGENT and let it run as a regular regression test.
    """
    pytest.xfail(f"Known divergence: {reason}")
    # Unreachable, but documents what we'd assert when fixed:
    g = request.getfixturevalue(fixture)
    assert _normalize(g.cypher(query).to_list()) == _normalize(g.cypher(query, disable_optimizer=True).to_list())


# ── Mutation differential ────────────────────────────────────────────
#
# Mutations write to graph state, so each mode needs its own freshly-
# built graph (we can't reuse a pytest fixture — within one test
# invocation it caches and returns the same instance on every call).


def _build_mutation_graph() -> kglite.KnowledgeGraph:
    """Fresh small_graph clone, built without going through a pytest
    fixture so successive calls produce independent instances."""
    import pandas as pd

    g = kglite.KnowledgeGraph()
    g.add_nodes(
        pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "age": [28, 35, 42],
                "city": ["Oslo", "Bergen", "Oslo"],
            }
        ),
        "Person",
        "person_id",
        "name",
    )
    g.add_connections(
        pd.DataFrame(
            {
                "from_id": [1, 2, 1],
                "to_id": [2, 3, 3],
                "since": [2020, 2019, 2021],
            }
        ),
        "KNOWS",
        "Person",
        "from_id",
        "Person",
        "to_id",
        columns=["since"],
    )
    return g


@pytest.mark.differential
@pytest.mark.parametrize("name,query", MUTATION_QUERIES, ids=[entry[0] for entry in MUTATION_QUERIES])
def test_mutation_optimized_matches_naive(name: str, query: str) -> None:
    """For each mutation, build two independent graphs, run the query
    on each (one optimized, one naive), and assert both the returned
    rows AND the post-mutation graph state (node + edge counts) match.
    Catches passes that mishandle mutation clauses by comparing the
    side effect on graph state, not just the cypher return value."""
    g_opt = _build_mutation_graph()
    rows_opt = _normalize(g_opt.cypher(query).to_list())
    nodes_opt = g_opt.cypher("MATCH (n) RETURN count(n) AS c").to_list()[0]["c"]
    edges_opt = g_opt.cypher("MATCH ()-[r]->() RETURN count(r) AS c").to_list()[0]["c"]

    g_naive = _build_mutation_graph()
    rows_naive = _normalize(g_naive.cypher(query, disable_optimizer=True).to_list())
    nodes_naive = g_naive.cypher("MATCH (n) RETURN count(n) AS c").to_list()[0]["c"]
    edges_naive = g_naive.cypher("MATCH ()-[r]->() RETURN count(r) AS c").to_list()[0]["c"]

    assert rows_opt == rows_naive, f"Mutation `{name}` rows: opt={rows_opt}, naive={rows_naive}"
    assert nodes_opt == nodes_naive, f"Mutation `{name}` post-state node count: opt={nodes_opt}, naive={nodes_naive}"
    assert edges_opt == edges_naive, f"Mutation `{name}` post-state edge count: opt={edges_opt}, naive={edges_naive}"


# ---------------------------------------------------------------------------
# Columnar (consolidated-store) write shapes
# ---------------------------------------------------------------------------
#
# Properties are columnar from construction, so the corpora above already write
# through a column store — but only ever one carrying its growth history. A
# save / vacuum / unspill consolidates that into a single contiguous run of
# rows, which is the shape a loaded graph carries and therefore the one a real
# deployment writes to for the rest of its life.
#
# The shapes here are the ones that path touches: a multi-row `SET` and
# `REMOVE` against a consolidated type (which write through the per-type master
# store and then re-point every node's handle), `MERGE … ON MATCH SET` over
# several rows (one `execute_set` per row), and `SET n = {…}` (which enumerates
# the node's existing keys before clearing them). `SET n.name` is kept separate
# because a title-aliased property is *not* in the store — it takes the
# node-private copy-on-write route instead, which is a different code path with
# different failure modes.
#
# These assert more than the other mutation corpora: alongside rows and post-
# statement counts, a **probe query reads the written values back**. A columnar
# write that lands in the master but never reaches the nodes (or vice versa)
# produces identical counts and identical returned rows, and is only visible on
# the next read.
COLUMNAR_MUTATION_QUERIES: list[tuple[str, str, str]] = [
    (
        "columnar_set_multi_row",
        "MATCH (p:Person) SET p.bucket = 'x' RETURN count(p) AS n",
        "MATCH (p:Person) RETURN p.person_id AS pid, p.bucket AS bucket ORDER BY pid",
    ),
    (
        "columnar_set_single_row",
        "MATCH (p:Person {person_id: 1}) SET p.age = 99 RETURN p.age AS age",
        "MATCH (p:Person) RETURN p.person_id AS pid, p.age AS age ORDER BY pid",
    ),
    (
        "columnar_remove_multi_row",
        "MATCH (p:Person) REMOVE p.city RETURN count(p) AS n",
        "MATCH (p:Person) RETURN p.person_id AS pid, p.city AS city ORDER BY pid",
    ),
    (
        "columnar_remove_single_row",
        "MATCH (p:Person {person_id: 2}) REMOVE p.age RETURN p.person_id AS pid",
        "MATCH (p:Person) RETURN p.person_id AS pid, p.age AS age ORDER BY pid",
    ),
    (
        "columnar_set_map_merge",
        "MATCH (p:Person {person_id: 1}) SET p += {age: 99, active: true} RETURN p.age AS age",
        "MATCH (p:Person) RETURN p.person_id AS pid, p.age AS age, p.active AS active ORDER BY pid",
    ),
    (
        # `SET n = {…}` enumerates the node's existing property keys to clear
        # them — the path that once read an empty key set on a saved graph.
        "columnar_set_map_replace",
        "MATCH (p:Person {person_id: 1}) SET p = {name: 'A', age: 99} RETURN p.age AS age",
        "MATCH (p:Person) RETURN p.person_id AS pid, p.age AS age, p.city AS city ORDER BY pid",
    ),
    (
        # One `execute_set` per matched row: R rows, R clauses.
        "columnar_merge_on_match_set_multi_row",
        "MATCH (p:Person) WITH collect(p.person_id) AS ids UNWIND ids AS pid "
        "MERGE (q:Person {person_id: pid}) ON MATCH SET q.touched = true RETURN count(q) AS n",
        "MATCH (p:Person) RETURN p.person_id AS pid, p.touched AS touched ORDER BY pid",
    ),
    (
        "columnar_merge_on_match_set_single_row",
        "MERGE (p:Person {person_id: 1}) ON MATCH SET p.touched = true RETURN p.touched AS t",
        "MATCH (p:Person) RETURN p.person_id AS pid, p.touched AS touched ORDER BY pid",
    ),
    (
        # `name` is the title alias, so it lands in the store's reserved
        # `__title__` column rather than in a schema slot — a different write
        # route from every other case here.
        "columnar_set_title_alias_multi_row",
        "MATCH (p:Person) SET p.name = 'Same' RETURN count(p) AS n",
        "MATCH (p:Person) RETURN p.person_id AS pid, p.name AS name ORDER BY pid",
    ),
    (
        "columnar_set_then_remove_same_property",
        "MATCH (p:Person) SET p.tmp = 1 REMOVE p.tmp RETURN count(p) AS n",
        "MATCH (p:Person) RETURN p.person_id AS pid, p.tmp AS tmp ORDER BY pid",
    ),
]


def _build_columnar_mutation_graph() -> kglite.KnowledgeGraph:
    """`_build_mutation_graph()` after the consolidation pass a save runs.

    The two are the same shape: properties are columnar from construction, and
    the consolidation pass — which `save()`, `vacuum()` and `unspill()` all
    run — is idempotent over a graph already in that shape. Kept as its own
    builder because the corpus below exercises the *consolidated* store (one
    contiguous run of rows, no growth history), which is what a loaded graph
    carries.
    """
    g = _build_mutation_graph()
    g.unspill()
    return g


@pytest.mark.differential
def test_columnar_mutation_fixture_is_actually_columnar() -> None:
    """Non-vacuity guard for the corpus below.

    The corpus is only about columnar storage if its fixture reads properties
    out of a column store. This used to be asserted as a *difference* — the
    plain builder owning zero rows and the consolidated one three — because a
    graph became columnar only when it was saved. Both arms carry the rows now,
    so the guard asserts the property directly on each.
    """
    fresh = _build_mutation_graph()
    assert fresh.graph_info()["columnar_total_rows"] == 3, (
        "the fixture must read its properties through a column store, or every "
        "case below proves nothing about columnar storage"
    )
    columnar = _build_columnar_mutation_graph()
    assert columnar.graph_info()["columnar_total_rows"] == 3, (
        "consolidation must leave all three Person rows in the store"
    )


@pytest.mark.differential
@pytest.mark.parametrize(
    "name,query,probe",
    COLUMNAR_MUTATION_QUERIES,
    ids=[entry[0] for entry in COLUMNAR_MUTATION_QUERIES],
)
def test_columnar_mutation_optimized_matches_naive(name: str, query: str, probe: str) -> None:
    """Optimized and naive must agree on a *consolidated* store — in the returned
    rows, in the post-statement counts, and in what a later read observes."""
    g_opt = _build_columnar_mutation_graph()
    rows_opt = _normalize(g_opt.cypher(query).to_list())
    nodes_opt = g_opt.cypher("MATCH (n) RETURN count(n) AS c").to_list()[0]["c"]
    edges_opt = g_opt.cypher("MATCH ()-[r]->() RETURN count(r) AS c").to_list()[0]["c"]
    probe_opt = _normalize(g_opt.cypher(probe).to_list())

    g_naive = _build_columnar_mutation_graph()
    rows_naive = _normalize(g_naive.cypher(query, disable_optimizer=True).to_list())
    nodes_naive = g_naive.cypher("MATCH (n) RETURN count(n) AS c").to_list()[0]["c"]
    edges_naive = g_naive.cypher("MATCH ()-[r]->() RETURN count(r) AS c").to_list()[0]["c"]
    probe_naive = _normalize(g_naive.cypher(probe, disable_optimizer=True).to_list())

    assert rows_opt == rows_naive, f"Columnar `{name}` rows: opt={rows_opt}, naive={rows_naive}"
    assert nodes_opt == nodes_naive, f"Columnar `{name}` node count: opt={nodes_opt}, naive={nodes_naive}"
    assert edges_opt == edges_naive, f"Columnar `{name}` edge count: opt={edges_opt}, naive={edges_naive}"
    assert probe_opt == probe_naive, f"Columnar `{name}` post-write read: opt={probe_opt}, naive={probe_naive}"


@pytest.mark.differential
@pytest.mark.parametrize(
    "name,query,probe",
    COLUMNAR_MUTATION_QUERIES,
    ids=[entry[0] for entry in COLUMNAR_MUTATION_QUERIES],
)
def test_columnar_mutation_matches_row_storage(name: str, query: str, probe: str) -> None:
    """The same write must produce the same observable on a consolidated store
    as on one still carrying its growth history.

    This is the arm that catches a columnar write landing in one replica of the
    type's column store and not the other: optimized-vs-naive agrees whenever
    *both* paths lose the write, so it cannot see that class on its own.
    """
    g_row = _build_mutation_graph()
    rows_row = _normalize(g_row.cypher(query).to_list())
    probe_row = _normalize(g_row.cypher(probe).to_list())

    g_col = _build_columnar_mutation_graph()
    rows_col = _normalize(g_col.cypher(query).to_list())
    probe_col = _normalize(g_col.cypher(probe).to_list())

    assert rows_col == rows_row, f"Columnar `{name}` rows differ from row storage: {rows_col} vs {rows_row}"
    assert probe_col == probe_row, (
        f"Columnar `{name}` post-write read differs from row storage: {probe_col} vs {probe_row}"
    )


# ---------------------------------------------------------------------------
# Constraint DDL sequences
# ---------------------------------------------------------------------------
#
# Separate for one mechanical reason: a schema command is a standalone
# statement, so the shapes that matter here — declaring one half of a
# node key on a property that already carries the other, then dropping one half
# again — need *several* statements, which MUTATION_QUERIES cannot express.
#
# What these pin: constraint declaration is state accumulation, not replacement.
# Comparing `SHOW CONSTRAINTS` after the sequence catches a divergence in what
# is *reported*; the enforcement probes catch a divergence in what is actually
# in force. Both halves are needed — the audit that produced this corpus found a
# listing reporting NODE_KEY for a pair whose presence half was not enforced,
# and a report whose constraint name changed across a save/load round-trip.
CONSTRAINT_DDL_SEQUENCES: list[tuple[str, list[str], list[str]]] = [
    (
        "not_null_over_existing_unique",
        [
            "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS NOT NULL",
        ],
        # Both halves in force: a duplicate name and a missing name are refused.
        ["CREATE (p:Person {person_id: 900, name: 'Alice', age: 1})", "CREATE (p:Person {person_id: 901, age: 1})"],
    ),
    (
        "unique_over_existing_not_null",
        [
            "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS NOT NULL",
            "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS UNIQUE",
        ],
        ["CREATE (p:Person {person_id: 900, name: 'Alice', age: 1})", "CREATE (p:Person {person_id: 901, age: 1})"],
    ),
    (
        "drop_the_unique_half_of_a_pair",
        [
            "CREATE CONSTRAINT pn_u FOR (p:Person) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT pn_nn FOR (p:Person) REQUIRE p.name IS NOT NULL",
            "DROP CONSTRAINT pn_u",
        ],
        ["CREATE (p:Person {person_id: 901, age: 1})"],
    ),
    (
        "drop_the_not_null_half_of_a_pair",
        [
            "CREATE CONSTRAINT pn_u FOR (p:Person) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT pn_nn FOR (p:Person) REQUIRE p.name IS NOT NULL",
            "DROP CONSTRAINT pn_nn",
        ],
        ["CREATE (p:Person {person_id: 900, name: 'Alice', age: 1})"],
    ),
]


def _constraint_state(g, probes: list[str]) -> tuple[list[dict], list[bool]]:
    """What the graph reports, and what it actually enforces.

    A fix to either half alone is not a fix, so the differential compares both.
    """
    reported = _normalize(g.cypher("SHOW CONSTRAINTS").to_list())
    enforced = []
    for probe in probes:
        try:
            g.cypher(probe)
            enforced.append(False)
        except Exception:
            enforced.append(True)
    return reported, enforced


@pytest.mark.differential
@pytest.mark.parametrize(
    "name,statements,probes",
    CONSTRAINT_DDL_SEQUENCES,
    ids=[entry[0] for entry in CONSTRAINT_DDL_SEQUENCES],
)
def test_constraint_ddl_sequence_optimized_matches_naive(name: str, statements: list[str], probes: list[str]) -> None:
    opt_reported, opt_enforced = _constraint_state(
        _run_constraint_sequence(statements, disable_optimizer=False), probes
    )
    naive_reported, naive_enforced = _constraint_state(
        _run_constraint_sequence(statements, disable_optimizer=True), probes
    )

    assert opt_reported == naive_reported, f"`{name}` SHOW CONSTRAINTS: opt={opt_reported}, naive={naive_reported}"
    assert opt_enforced == naive_enforced, f"`{name}` enforcement: opt={opt_enforced}, naive={naive_enforced}"
    # Every sequence above leaves at least one constraint declared and every
    # probe violating it, so a run that silently enforced nothing is a failure
    # rather than a vacuous pass.
    assert opt_reported, f"`{name}` declared no constraint at all"
    assert all(opt_enforced), f"`{name}` reported constraints it does not enforce: {opt_enforced}"


def _run_constraint_sequence(statements: list[str], *, disable_optimizer: bool):
    g = _build_mutation_graph()
    for statement in statements:
        g.cypher(statement, disable_optimizer=disable_optimizer)
    return g


# ---------------------------------------------------------------------------
# LOAD CSV
# ---------------------------------------------------------------------------
#
# Separate from the corpora above for one mechanical reason: every LOAD CSV
# query needs a real file, so the query text carries a `{csv}`
# placeholder the runner substitutes with a per-test `tmp_path` file. It cannot
# live in DIFFERENTIAL_QUERIES because that list is also consumed verbatim by
# `scripts/bolt_conformance.py`, where LOAD CSV is denied by design (a Bolt
# client is a remote caller — see `executor/load_csv.rs`).
#
# What these pin: LOAD CSV is an opaque barrier to every optimizer pass, and
# the batch driver produces the same answer as the naive plan. Row counts
# straddle the 1000-row batch size on purpose, so a pass that reordered or
# fused across the clause would diverge rather than merely run slower.
LOAD_CSV_QUERIES: list[tuple[str, str]] = [
    (
        "load_csv_headers_return",
        "LOAD CSV WITH HEADERS FROM '{csv}' AS row RETURN row.name AS name",
    ),
    (
        "load_csv_no_headers_index",
        "LOAD CSV FROM '{csv}' AS row RETURN row[0] AS id",
    ),
    (
        "load_csv_where_filter",
        "LOAD CSV WITH HEADERS FROM '{csv}' AS row WITH row WHERE toInteger(row.id) % 7 = 0 RETURN row.id AS id",
    ),
    (
        "load_csv_aggregate_barrier",
        "LOAD CSV WITH HEADERS FROM '{csv}' AS row RETURN count(*) AS n",
    ),
    (
        "load_csv_group_aggregate",
        "LOAD CSV WITH HEADERS FROM '{csv}' AS row RETURN row.city AS city, count(*) AS n ORDER BY city",
    ),
    (
        "load_csv_order_by_limit",
        "LOAD CSV WITH HEADERS FROM '{csv}' AS row RETURN row.name AS name ORDER BY name DESC LIMIT 5",
    ),
    (
        "load_csv_distinct",
        "LOAD CSV WITH HEADERS FROM '{csv}' AS row RETURN DISTINCT row.city AS city ORDER BY city",
    ),
    (
        "load_csv_create_ingest",
        "LOAD CSV WITH HEADERS FROM '{csv}' AS row CREATE (:Imported {id: toInteger(row.id), name: row.name})",
    ),
    (
        "load_csv_merge_dedupe",
        "LOAD CSV WITH HEADERS FROM '{csv}' AS row MERGE (:City {id: row.city})",
    ),
    (
        "load_csv_match_join",
        "LOAD CSV WITH HEADERS FROM '{csv}' AS row "
        "WITH toInteger(row.id) AS pid MATCH (p:Person {person_id: pid}) "
        "RETURN p.name AS name ORDER BY name",
    ),
    (
        "load_csv_unwind_after",
        "LOAD CSV WITH HEADERS FROM '{csv}' AS row UNWIND [row.city, row.name] AS token RETURN count(token) AS n",
    ),
]

# Straddles the engine's 1000-row batch size so batch-boundary bugs surface.
_LOAD_CSV_ROWS = 1000 * 2 + 3


def _write_differential_csv(path) -> str:
    import csv as csv_module

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv_module.writer(fh)
        writer.writerow(["id", "name", "city"])
        for i in range(_LOAD_CSV_ROWS):
            writer.writerow([i + 1, f"Name{i:05d}", f"City{i % 4}"])
    return str(path)


@pytest.mark.differential
@pytest.mark.parametrize("name,query", LOAD_CSV_QUERIES, ids=[entry[0] for entry in LOAD_CSV_QUERIES])
def test_load_csv_optimized_matches_naive(name: str, query: str, tmp_path) -> None:
    """Optimized and naive plans must agree on rows and on post-state, with the
    CSV batch driver in play on both sides."""
    csv_path = _write_differential_csv(tmp_path / "differential.csv")
    resolved = query.replace("{csv}", csv_path)

    g_opt = _build_mutation_graph()
    rows_opt = _normalize(g_opt.cypher(resolved).to_list())
    nodes_opt = g_opt.cypher("MATCH (n) RETURN count(n) AS c").to_list()[0]["c"]
    edges_opt = g_opt.cypher("MATCH ()-[r]->() RETURN count(r) AS c").to_list()[0]["c"]

    g_naive = _build_mutation_graph()
    rows_naive = _normalize(g_naive.cypher(resolved, disable_optimizer=True).to_list())
    nodes_naive = g_naive.cypher("MATCH (n) RETURN count(n) AS c").to_list()[0]["c"]
    edges_naive = g_naive.cypher("MATCH ()-[r]->() RETURN count(r) AS c").to_list()[0]["c"]

    assert rows_opt == rows_naive, f"LOAD CSV `{name}` rows: opt={rows_opt}, naive={rows_naive}"
    assert nodes_opt == nodes_naive, f"LOAD CSV `{name}` node count: opt={nodes_opt}, naive={nodes_naive}"
    assert edges_opt == edges_naive, f"LOAD CSV `{name}` edge count: opt={edges_opt}, naive={edges_naive}"
