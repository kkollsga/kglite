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
    # ── push_where_into_match ──
    ("where_eq", "social_graph", "MATCH (p:Person) WHERE p.city = 'Oslo' RETURN p.name AS n", None),
    ("where_gt", "social_graph", "MATCH (p:Person) WHERE p.age > 30 RETURN p.name AS n", None),
    ("where_and", "social_graph", "MATCH (p:Person) WHERE p.age > 30 AND p.city = 'Bergen' RETURN p.name AS n", None),
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
    ("not_equal", "social_graph", "MATCH (p:Person) WHERE p.city <> 'Oslo' RETURN count(p) AS n", None),
    (
        "range_predicate",
        "social_graph",
        "MATCH (p:Person) WHERE p.age >= 25 AND p.age <= 35 RETURN count(p) AS n",
        None,
    ),
    ("null_check", "social_graph", "MATCH (p:Person) WHERE p.email IS NOT NULL RETURN count(p) AS n", None),
    ("in_list", "social_graph", "MATCH (p:Person) WHERE p.city IN ['Oslo', 'Bergen'] RETURN count(p) AS n", None),
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
]


# Mutation queries: each test gets its own fresh fixture so state-bleed
# between mutations is impossible. The harness's identity for mutations
# is "optimized result on a fresh fixture == naive result on a fresh
# fixture." Lives separate from DIFFERENTIAL_QUERIES because of the
# fresh-fixture-per-test requirement.
MUTATION_QUERIES: list[tuple[str, str]] = [
    ("create_node", "CREATE (p:Person {person_id: 99, name: 'X', age: 50}) RETURN p.person_id AS pid"),
    ("set_property", "MATCH (p:Person {person_id: 1}) SET p.age = 99 RETURN p.age AS age"),
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


@pytest.mark.differential
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
