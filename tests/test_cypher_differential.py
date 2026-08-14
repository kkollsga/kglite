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
  paths produce the same result, just slower. Needs plan-shape or perf
  regression testing — covered by follow-ups.
- **Execution semantic bugs** that exist in both fast and slow paths
  (rare but real, e.g. 0.8.30 startNode(r) was actually present in both
  paths). Needs cross-mode parity (cypher vs. fluent vs. naive).

When fixing a future silent-correctness bug, **add the bug's triggering
query to DIFFERENTIAL_QUERIES** so the regression is permanent.
"""

from __future__ import annotations

import pytest

import kglite

# ── Corpus ───────────────────────────────────────────────────────────
#
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
# The corpus deliberately skips vector_score / text_score and spatial
# fusion — those depend on registered embedders or geometry data and
# don't exist in the shared fixtures. They warrant a separate harness
# that builds purpose-specific fixtures.
DIFFERENTIAL_QUERIES: list[tuple[str, str, str, dict | None]] = [
    # ── basic shapes ──
    ("simple_match", "small_graph", "MATCH (p:Person) RETURN p.name AS n", None),
    ("simple_match_param", "small_graph", "MATCH (p:Person) WHERE p.age > $min RETURN p.name AS n", {"min": 30}),
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
    # First key ties on every row (all Persons share the type), so the
    # emitted set is decided entirely by the second key.
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
    # ── UNION (optimize_nested_queries) ──
    (
        "union_simple",
        "small_graph",
        "MATCH (p:Person) WHERE p.age < 30 RETURN p.name AS n "
        "UNION MATCH (p:Person) WHERE p.age > 40 RETURN p.name AS n",
        None,
    ),
    # ── edge cases ──
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
    # ── UNION ALL ──
    (
        "union_all",
        "small_graph",
        "MATCH (p:Person) RETURN p.name AS n UNION ALL MATCH (p:Person) RETURN p.name AS n",
        None,
    ),
    # ── expression shapes ──
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
    # ── Kleene AND/OR with NULL operand ──
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
    # ── EXISTS / NOT EXISTS subqueries ──
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
    # ── HAVING / multi-WITH ──
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
    # ── shortest path ──
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
    # ── multiple OPTIONAL MATCH ──
    (
        "two_optional_match",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:WORKS_AT]->(c) OPTIONAL MATCH (p)-[:KNOWS]->(f) "
        "RETURN p.name AS n, count(DISTINCT c) AS Cs, count(DISTINCT f) AS Fs "
        "ORDER BY n LIMIT 5",
        None,
    ),
    # ── arithmetic + collect ──
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
    # ── label check / id() function ──
    ("label_check", "social_graph", "MATCH (n) WHERE n:Person RETURN count(n) AS n", None),
    ("id_function", "social_graph", "MATCH (p:Person) WHERE id(p) IS NOT NULL RETURN count(p) AS n", None),
    # ── inline pattern + WHERE ──
    (
        "inline_and_where",
        "social_graph",
        "MATCH (p:Person {city: 'Oslo'}) WHERE p.age > 25 RETURN p.name AS n ORDER BY n",
        None,
    ),
    # ── 3-hop chain ──
    (
        "three_hop_count",
        "social_graph",
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)-[:KNOWS]->(d:Person) RETURN count(*) AS n",
        None,
    ),
    # ── WITH * project everything ──
    ("with_star", "social_graph", "MATCH (p:Person) WITH * WHERE p.age > 35 RETURN p.name AS n ORDER BY n", None),
    # ── count{...} subquery + ORDER BY + LIMIT ──
    (
        "count_subquery_top_k",
        "social_graph",
        "MATCH (p:Person) WITH p, count{(p)-[:KNOWS]->()} AS deg "
        "WHERE deg > 0 RETURN p.name AS n, deg ORDER BY deg DESC, n LIMIT 5",
        None,
    ),
    # ── List comprehension after collect aggregate ──
    (
        "list_comp_after_collect",
        "social_graph",
        "MATCH (p:Person) WITH collect(p.age) AS ages RETURN [a IN ages WHERE a > 30 | a + 1] AS bumped",
        None,
    ),
    # ── Path operations (length / nodes / relationships) ──
    (
        "shortest_with_length",
        "social_graph",
        "MATCH p = shortestPath((a:Person {person_id:1})-[:KNOWS*..5]-(b:Person {person_id:10})) "
        "RETURN length(p) AS L, size(nodes(p)) AS hops",
        None,
    ),
    # ── Parameterized list in IN ──
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
    # ── Parameterized scalar with arithmetic ──
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
    # ── DISTINCT + ORDER BY same expression ──
    (
        "distinct_order_same_expr",
        "social_graph",
        "MATCH (p:Person) RETURN DISTINCT p.city AS c ORDER BY p.city",
        None,
    ),
    # ── OPTIONAL MATCH + count(*) + GROUP BY ──
    (
        "optional_count_star_group",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company) "
        "WITH p.city AS city, count(c) AS jobs RETURN city, jobs ORDER BY city",
        None,
    ),
    # ── HAVING expression with multi-key GROUP ──
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
    # ── COUNT(*) with multi-MATCH ──
    (
        "multi_match_count_star",
        "social_graph",
        "MATCH (p:Person) MATCH (q:Person) WHERE p.person_id < q.person_id AND p.city = q.city RETURN count(*) AS n",
        None,
    ),
    # ── safe LIMIT pushdown over an unfiltered node-only cartesian ──
    (
        "cartesian_node_scans_limit",
        "social_graph",
        "MATCH (p:Person), (c:Company) RETURN p.name AS p, c.name AS c LIMIT 100",
        None,
    ),
    # ── String operations + WHERE + ORDER BY ──
    (
        "string_op_filter_order",
        "social_graph",
        "MATCH (p:Person) WHERE p.name STARTS WITH 'Person_' RETURN p.name AS n ORDER BY size(p.name) DESC, n LIMIT 5",
        None,
    ),
    # ── coalesce / IS NOT NULL filter ──
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
    # ── CASE inside aggregate ──
    (
        "case_in_agg",
        "social_graph",
        "MATCH (p:Person) RETURN p.city AS c, sum(CASE WHEN p.age > 30 THEN 1 ELSE 0 END) AS olders ORDER BY c",
        None,
    ),
    # ── nested function calls ──
    (
        "nested_func_calls",
        "social_graph",
        "MATCH (p:Person) RETURN p.name AS n, toUpper(p.city) AS c ORDER BY n LIMIT 5",
        None,
    ),
    # ── NOT predicate ──
    (
        "not_predicate",
        "social_graph",
        "MATCH (p:Person) WHERE NOT p.city = 'Oslo' RETURN count(p) AS n",
        None,
    ),
    # ── WHERE with edge property AND node property ──
    (
        "where_edge_node_mix",
        "social_graph",
        "MATCH (p:Person)-[r:KNOWS]->(q:Person) WHERE r.since > 2017 AND q.age > 25 RETURN count(*) AS n",
        None,
    ),
    # ── count{} subquery in WHERE ──
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
    # ── arithmetic expression in WHERE ──
    (
        "expr_filter",
        "social_graph",
        "MATCH (p:Person) WHERE p.salary / p.age > 2000 RETURN p.name AS n ORDER BY n LIMIT 5",
        None,
    ),
    # ── WITH expression alias as filter then sort ──
    (
        "with_expr_filter_sort",
        "social_graph",
        "MATCH (p:Person) WITH p, p.salary - p.age * 1000 AS net "
        "WHERE net > 50000 RETURN p.name AS n, net ORDER BY net DESC, n LIMIT 5",
        None,
    ),
    # ── multi-OPTIONAL with HAVING-style filter ──
    (
        "multi_optional_having",
        "social_graph",
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(f) "
        "OPTIONAL MATCH (p)-[:WORKS_AT]->(c) "
        "WITH p, count(DISTINCT f) AS friends, count(DISTINCT c) AS jobs "
        "WHERE friends > 0 RETURN p.name AS n, friends, jobs ORDER BY n LIMIT 5",
        None,
    ),
    # ── WITH chain with re-entered MATCH (cohort then expansion) ──
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
    # ── String op + ORDER BY ──
    (
        "string_op_filter_order",
        "social_graph",
        "MATCH (p:Person) WHERE p.name STARTS WITH 'Person_' RETURN p.name AS n ORDER BY size(p.name) DESC, n LIMIT 5",
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
    # ── multi-pattern MATCH after a seeded pipeline must cross-join ──
    (
        "with_then_multi_pattern_cross_join",
        "social_graph",
        "WITH 1 AS x MATCH (a:Person), (c:Company) RETURN a.name AS a, c.name AS c ORDER BY a, c LIMIT 5",
        None,
    ),
    # ── inline pattern referencing an UNWIND map member (`{id: x.id}`) ──
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
    # ── Phase A.3 — db.* schema-introspection procedures ──
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
    # ── CALL { } uncorrelated subqueries (Phase 3) ──
    # The body runs once and its rows cartesian-product with the outer
    # stream. CALL { } is opaque to the optimizer passes this phase (the
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
    # ── CALL { } correlated subqueries (Phase 4) ──
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
    # ── CALL { } cross-clause barrier (Phase 5) ──
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
    # ── CALL { } Neo4j-conformance shapes (Phase 6) ──
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
]


# Mutation queries: each test gets its own fresh fixture so state-bleed
# between mutations is impossible. The harness's identity for mutations
# is "optimized result on a fresh fixture == naive result on a fresh
# fixture." Lives separate from DIFFERENTIAL_QUERIES because of the
# fresh-fixture-per-test requirement.
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
    # Declaring one half of a node key on a property that already carries the
    # other must *add* to it rather than replace it, so the post-statement graph
    # state carries both. A pass that dropped or reordered the second statement
    # would leave only one half enforced — which this harness sees as diverging
    # post-state once a later write is rejected on one path and not the other.
    # Both orders, because the merge is order-sensitive code.
    # Multi-statement constraint shapes live in CONSTRAINT_DDL_SEQUENCES below:
    # a schema command is a standalone statement, so a merge (declaring one half
    # of a node key over the other) cannot be expressed as one query here.
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
    # full corpus rather than a representative cohort: 30 passes × the current
    # corpus is cheap, and failures identify the exact query/pass pair.
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
# These shapes diverge between optimized and naive but the divergence
# was discovered by the harness on first run. They land here as
# permanent regression tests: when a fix lands, flip xfail → expected
# pass and the test starts protecting the fix.

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
    "optimize_nested_queries": ("differential", "call_uncorrelated_body_fusion_then_limit"),
    "rewrite_count_bound_var_to_star": ("differential", "count_all_typed"),
    "push_where_into_match.1": ("differential", "where_eq"),
    "fold_or_to_in": ("differential", "or_chain_to_in"),
    "push_where_into_match.2": ("differential", "or_chain_to_in"),
    "extract_pushable_rel_predicates": ("differential", "rel_property_filter"),
    "fold_pass_through_with": ("differential", "pass_through_with"),
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
# Building the graph inline is verbose but gives us isolation.


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
# Columnar (saved-graph) write shapes
# ---------------------------------------------------------------------------
#
# Every corpus above runs on a *fresh* graph, whose nodes hold row storage
# (`Map` / `Compact`). The moment a graph is saved it converts to per-type
# column stores and keeps that shape for the rest of its life — so the write
# paths a real deployment exercises are the columnar ones, and until D1 none of
# them were in this corpus.
#
# The shapes here are the ones the D1 programme touches: a multi-row `SET` and
# `REMOVE` against a saved type (which write through the per-type master store
# and then re-point every node's handle), `MERGE … ON MATCH SET` over several
# rows (one `execute_set` per row), and `SET n = {…}` (which enumerates the
# node's existing keys before clearing them). `SET n.name` is kept separate
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
        # them — the path that read an empty key set on a saved graph before D1.
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
    """Optimized and naive must agree on a *saved* graph — in the returned
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
    """The same write must produce the same observable on a saved graph as on a
    fresh one.

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
# A fourth corpus, separate for one mechanical reason: a schema command is a
# standalone statement, so the shapes that matter here — declaring one half of a
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
# A third corpus, separate from the two above for one mechanical reason: every
# LOAD CSV query needs a real file, so the query text carries a `{csv}`
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
