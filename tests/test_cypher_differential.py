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
    ("trigger_anchored_edge_count", "social_graph", "MATCH ({id: 1})-[:KNOWS]->(p) RETURN count(*) AS n", None),
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
    ("trigger_generic_top_k", "small_graph", "UNWIND [3, 1, 2] AS x RETURN x ORDER BY x LIMIT 2", None),
    (
        "trigger_predicate_reorder",
        "social_graph",
        "MATCH (p) WHERE EXISTS((p)-[:KNOWS]->()) AND p:Person RETURN p.title AS title",
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
    # ── reorder_match_clauses + optimize_pattern_start_node ──
    (
        "two_match_chains",
        "social_graph",
        "MATCH (p:Person)-[:WORKS_AT]->(c:Company) MATCH (p)-[:KNOWS]->(q:Person) "
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
    # ── integer div/mod overflow wraps (i64::MIN / -1) instead of panicking ──
    ("div_overflow_wraps", "small_graph", "RETURN (-9223372036854775807 - 1) / -1 AS n", None),
    ("mod_overflow_wraps", "small_graph", "RETURN (-9223372036854775807 - 1) % -1 AS n", None),
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


# Per-pass bisection check: a representative cohort query must produce
# correct rows when ANY single pass is disabled. Catches passes that
# silently became load-bearing for correctness (a fusion pass should
# never affect rows — only speed). When a pass appears here that would
# affect correctness in isolation, that's a real bug to fix.
@pytest.mark.differential
@pytest.mark.parametrize("pass_name", kglite.cypher_pass_names())
def test_disabling_single_pass_preserves_correctness(pass_name: str, social_graph) -> None:
    """Each pass, disabled in isolation, must produce the same rows as the naive baseline."""
    query = "MATCH (p:Person) WITH p.city AS city, count(p) AS n RETURN city, n ORDER BY n DESC LIMIT 5"
    baseline = _normalize(social_graph.cypher(query, disable_optimizer=True).to_list())
    actual = _normalize(social_graph.cypher(query, disabled_passes=[pass_name]).to_list())
    assert actual == baseline, (
        f"Disabling pass `{pass_name}` produced different rows:\n  baseline: {baseline}\n  actual:   {actual}"
    )


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
