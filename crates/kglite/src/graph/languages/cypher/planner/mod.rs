//! Cypher query optimizer.
//!
//! Split (Phase 9):
//! - [`join_order`] — pattern-start node selection, selectivity-based reordering
//! - [`index_selection`] — predicate pushdown into MATCH, equality/comparison helpers
//! - [`cost_model`] — predicate / expression cost heuristics
//! - [`simplification`] — fold_or_to_in, push LIMIT/DISTINCT, rewrite_text_score
//! - [`fusion`] — multi-clause fusion (MATCH+RETURN+AGG, top-K, …)

use super::ast::*;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::PatternElement;
use crate::graph::schema::DirGraph;
use std::collections::{HashMap, HashSet};

pub mod cost_model;
pub mod fusion;
pub mod index_selection;
pub mod join_order;
pub mod rel_predicate_pushdown;
pub mod schema_check;
pub mod simplification;

use cost_model::reorder_predicates_by_cost;
use fusion::{
    fuse_anchored_edge_count, fuse_count_short_circuits, fuse_match_return_aggregate,
    fuse_match_with_aggregate, fuse_match_with_aggregate_top_k, fuse_node_scan_aggregate,
    fuse_node_scan_top_k, fuse_optional_match_aggregate, fuse_order_by_top_k, fuse_spatial_join,
    fuse_vector_score_order_limit, mark_return_lazy_eligible,
};
use index_selection::push_where_into_match;
use join_order::{optimize_pattern_start_node, reorder_match_clauses, reorder_match_patterns};
use rel_predicate_pushdown::extract_pushable_rel_predicates;
use simplification::{
    desugar_multi_match_return_aggregate, fold_or_to_in, fold_pass_through_with,
    push_distinct_into_match, push_limit_into_aggregate, push_limit_into_match,
};

/// Carries the per-call inputs every pass might need. Passing this once
/// through the registry loop is cheaper than threading three positional
/// arguments through 25+ wrapper fns, and adding a new dependency means
/// extending this struct rather than every wrapper signature.
pub struct PassCtx<'a> {
    pub graph: &'a DirGraph,
    pub params: &'a HashMap<String, Value>,
    pub disabled: &'a HashSet<String>,
}

type PassFn = fn(&mut CypherQuery, &PassCtx);

/// The optimizer pipeline as a single source of truth. Order is
/// load-bearing — comments on individual entries call out cross-pass
/// dependencies. Adding a new pass: write the impl, write a `pass_*`
/// wrapper, register here with a unique name, doc-comment the wrapper,
/// add at least one query to `tests/test_cypher_differential.py`.
pub const PASSES: &[(&str, PassFn)] = &[
    ("optimize_nested_queries", pass_optimize_nested_queries),
    ("push_where_into_match.1", pass_push_where_into_match),
    ("fold_or_to_in", pass_fold_or_to_in),
    // second push_where pass: catches IN predicates created by fold_or_to_in
    ("push_where_into_match.2", pass_push_where_into_match),
    (
        "extract_pushable_rel_predicates",
        pass_extract_pushable_rel_predicates,
    ),
    // strip pass-through WITH BEFORE cross-clause MATCH reorder so the
    // latter sees a contiguous Match-Match span when a `WITH p` sat between.
    ("fold_pass_through_with", pass_fold_pass_through_with),
    // rewrites Match-Match-Return(group, agg) so the aggregate-fusion +
    // top-K pipeline can pick it up.
    (
        "desugar_multi_match_return_aggregate",
        pass_desugar_multi_match_return_aggregate,
    ),
    ("fuse_spatial_join", pass_fuse_spatial_join),
    // O(1) cost-proxy reorder. Runs BEFORE pattern_start_node so reversal
    // sees the post-reorder clause sequence and tracks bound_vars correctly.
    ("reorder_match_clauses", pass_reorder_match_clauses),
    (
        "optimize_pattern_start_node",
        pass_optimize_pattern_start_node,
    ),
    ("reorder_match_patterns", pass_reorder_match_patterns),
    ("push_limit_into_match", pass_push_limit_into_match),
    ("push_limit_into_aggregate", pass_push_limit_into_aggregate),
    ("push_distinct_into_match", pass_push_distinct_into_match),
    ("fuse_anchored_edge_count", pass_fuse_anchored_edge_count),
    ("fuse_count_short_circuits", pass_fuse_count_short_circuits),
    (
        "fuse_optional_match_aggregate",
        pass_fuse_optional_match_aggregate,
    ),
    (
        "fuse_match_return_aggregate",
        pass_fuse_match_return_aggregate,
    ),
    ("fuse_match_with_aggregate", pass_fuse_match_with_aggregate),
    // top-K absorption AFTER fuse_match_with_aggregate (which produces
    // FusedMatchWithAggregate) but BEFORE fuse_order_by_top_k (which would
    // otherwise consume the downstream RETURN+ORDER BY+LIMIT).
    (
        "fuse_match_with_aggregate_top_k",
        pass_fuse_match_with_aggregate_top_k,
    ),
    ("fuse_node_scan_aggregate", pass_fuse_node_scan_aggregate),
    ("fuse_node_scan_top_k", pass_fuse_node_scan_top_k),
    (
        "fuse_vector_score_order_limit",
        pass_fuse_vector_score_order_limit,
    ),
    ("fuse_order_by_top_k", pass_fuse_order_by_top_k),
    (
        "reorder_predicates_by_cost",
        pass_reorder_predicates_by_cost,
    ),
    (
        "mark_fast_var_length_paths",
        pass_mark_fast_var_length_paths,
    ),
    (
        "mark_skip_target_type_check",
        pass_mark_skip_target_type_check,
    ),
];

/// Returns true iff `name` is a registered pass name. PyAPI uses this to
/// reject typos in the `disabled_passes` kwarg before they silently
/// suppress nothing.
pub fn is_known_pass(name: &str) -> bool {
    PASSES.iter().any(|(n, _)| *n == name)
}

/// Returns every registered pass name. Used by the PyAPI's
/// `disable_optimizer=True` shortcut, which expands to "disable everything".
pub fn all_pass_names() -> Vec<String> {
    PASSES.iter().map(|(n, _)| n.to_string()).collect()
}

/// Annotate the top-level query's terminal RETURN with `lazy_eligible`
/// when no downstream operator forces row materialisation. Called once
/// after `optimize`, never recursively, so nested UNION arms don't get
/// marked (their results pass through the union machinery, which expects
/// fully evaluated rows).
pub fn mark_lazy_eligibility(query: &mut CypherQuery) {
    // Don't mark when the top-level query contains a UNION — the union
    // machinery merges materialised rows.
    if query.clauses.iter().any(|c| matches!(c, Clause::Union(_))) {
        return;
    }
    // Don't mark for mutation queries — CREATE/SET/DELETE/REMOVE/MERGE go
    // through `execute_mutable`, which doesn't read the lazy descriptor
    // and would produce empty rows.
    if query.clauses.iter().any(|c| {
        matches!(
            c,
            Clause::Create(_)
                | Clause::Set(_)
                | Clause::Delete(_)
                | Clause::Remove(_)
                | Clause::Merge(_)
        )
    }) {
        return;
    }
    mark_return_lazy_eligible(query);
}

/// Run the optimizer pipeline. Equivalent to `optimize_with_disabled`
/// with no passes disabled. Kept as the primary entry point so most
/// callers (executor, transactions, mutations) don't need to think about
/// the disable knob.
pub fn optimize(query: &mut CypherQuery, graph: &DirGraph, params: &HashMap<String, Value>) {
    optimize_with_disabled(query, graph, params, empty_disabled_set());
}

/// Process-lifetime empty `HashSet<String>` used as the no-knob default.
/// Avoids a fresh `HashSet::new()` allocation on every cypher call —
/// negligible per-call (no heap alloc on empty), but the static is
/// clearer about intent and removes per-call stack-frame setup.
pub fn empty_disabled_set() -> &'static HashSet<String> {
    static EMPTY: std::sync::OnceLock<HashSet<String>> = std::sync::OnceLock::new();
    EMPTY.get_or_init(HashSet::new)
}

/// Run the optimizer pipeline, skipping any pass whose name is in
/// `disabled`. Diagnostic hook for the differential test harness and
/// `cypher(..., disabled_passes=[...])` kwarg — production callers should
/// use the no-knob `optimize()` wrapper.
pub fn optimize_with_disabled(
    query: &mut CypherQuery,
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    disabled: &HashSet<String>,
) {
    let ctx = PassCtx {
        graph,
        params,
        disabled,
    };
    for (name, pass_fn) in PASSES {
        if disabled.contains(*name) {
            continue;
        }
        pass_fn(query, &ctx);
        #[cfg(debug_assertions)]
        debug_check_invariants(query, name);
    }
}

/// Sanity checks on the post-pass IR. Debug-only — release builds pay
/// nothing. Catches the class of bug where pass X corrupts the IR and a
/// downstream pass or the executor crashes 200 lines later with a
/// confusing error. Each check is permissive (only catches definitely-
/// invalid shapes); we'd rather miss a subtle bug than panic on a valid
/// query the writer of an invariant didn't anticipate.
#[cfg(debug_assertions)]
fn debug_check_invariants(query: &CypherQuery, after_pass_name: &str) {
    if let Err(msg) = check_match_patterns_non_empty(query) {
        panic!("Pass `{after_pass_name}` produced invalid IR: {msg}");
    }
    if let Err(msg) = check_return_with_items_non_empty(query) {
        panic!("Pass `{after_pass_name}` produced invalid IR: {msg}");
    }
    if let Err(msg) = check_limit_skip_nonnegative(query) {
        panic!("Pass `{after_pass_name}` produced invalid IR: {msg}");
    }
}

/// Every Match / OptionalMatch must have at least one pattern, and each
/// pattern at least one element. Catches passes that delete the last
/// pattern but leave the clause shell.
#[cfg(debug_assertions)]
fn check_match_patterns_non_empty(query: &CypherQuery) -> Result<(), String> {
    for (idx, clause) in query.clauses.iter().enumerate() {
        let mc = match clause {
            Clause::Match(m) | Clause::OptionalMatch(m) => m,
            _ => continue,
        };
        if mc.patterns.is_empty() {
            return Err(format!("Match clause at index {idx} has no patterns"));
        }
        for (pi, p) in mc.patterns.iter().enumerate() {
            if p.elements.is_empty() {
                return Err(format!(
                    "Match clause at index {idx}, pattern {pi} has no elements"
                ));
            }
        }
    }
    Ok(())
}

/// Return / With must project at least one item. Catches passes that
/// leave a stub Return after consuming its only item into a fused clause.
#[cfg(debug_assertions)]
fn check_return_with_items_non_empty(query: &CypherQuery) -> Result<(), String> {
    for (idx, clause) in query.clauses.iter().enumerate() {
        match clause {
            Clause::Return(r) if r.items.is_empty() => {
                return Err(format!("Return clause at index {idx} has no items"));
            }
            Clause::With(w) if w.items.is_empty() => {
                return Err(format!("With clause at index {idx} has no items"));
            }
            _ => {}
        }
    }
    Ok(())
}

/// Literal LIMIT / SKIP values must be non-negative. Catches passes
/// that synthesize a literal hint (e.g. fusion top-K) and forget to
/// clamp at zero. Non-literal values (parameters, expressions) are left
/// alone — the executor handles those at runtime.
#[cfg(debug_assertions)]
fn check_limit_skip_nonnegative(query: &CypherQuery) -> Result<(), String> {
    for (idx, clause) in query.clauses.iter().enumerate() {
        match clause {
            Clause::Limit(l) => {
                if let Expression::Literal(Value::Int64(n)) = &l.count {
                    if *n < 0 {
                        return Err(format!(
                            "Limit clause at index {idx} has negative literal {n}"
                        ));
                    }
                }
            }
            Clause::Skip(s) => {
                if let Expression::Literal(Value::Int64(n)) = &s.count {
                    if *n < 0 {
                        return Err(format!(
                            "Skip clause at index {idx} has negative literal {n}"
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

// Note: a `check_terminal_return_position` invariant was prototyped here
// and removed — the parser legitimately produces `RETURN ... WHERE ...`
// for queries where the WHERE syntactically trails the RETURN (test:
// test_edge_properties.py). Without a clear oracle for "what's a valid
// post-RETURN clause", a position check creates false positives. The
// non-empty-patterns and non-empty-items checks above stay because they
// have unambiguous oracles.

// ── Pass wrappers ──────────────────────────────────────────────────
// Each wrapper is the registry-facing entry point for one optimizer
// pass. Adding a new pass: write the impl in the appropriate
// sub-module, add a wrapper here with a doc-comment in the standard
// shape, register it in `PASSES`, add at least one query to
// `tests/test_cypher_differential.py::DIFFERENTIAL_QUERIES`.

/// **Pass:** `optimize_nested_queries` — Recurse the optimizer into
/// every nested query (UNION right-arms today; subqueries when added).
/// Inherits the parent's `disabled` set so diagnostic toggles propagate
/// to the inner planner pipeline.
fn pass_optimize_nested_queries(query: &mut CypherQuery, ctx: &PassCtx) {
    for clause in &mut query.clauses {
        if let Clause::Union(ref mut u) = clause {
            optimize_with_disabled(&mut u.query, ctx.graph, ctx.params, ctx.disabled);
        }
    }
}

/// **Pass:** `push_where_into_match` — Move comparison predicates from
/// a trailing `WHERE` clause into the preceding `MATCH`'s
/// `PropertyMatcher`. The matcher applies them during pattern expansion
/// instead of evaluating them per row, pruning the search early. Runs
/// twice in the pipeline (before and after `fold_or_to_in`) so IN
/// predicates synthesized by the OR fold also get pushed.
fn pass_push_where_into_match(query: &mut CypherQuery, ctx: &PassCtx) {
    push_where_into_match(query, ctx.params)
}

/// **Pass:** `fold_or_to_in` — Rewrite `(a.x = v1 OR a.x = v2 OR ...)`
/// chains into `a.x IN [v1, v2, ...]`. Lets the second
/// `push_where_into_match` push the synthesized IN as a single
/// equality-set matcher.
fn pass_fold_or_to_in(query: &mut CypherQuery, _ctx: &PassCtx) {
    fold_or_to_in(query)
}

/// **Pass:** `extract_pushable_rel_predicates` — Inline edge-side
/// predicates (`type(r) = 'X'`, `r.prop OP literal`, `startNode(r) =
/// peer`) from a trailing WHERE into the edge's `rel_predicate`. The
/// matcher applies them during expansion, before per-edge bindings are
/// allocated. WHY-BAIL: predicates referencing unbound vars stay in WHERE.
fn pass_extract_pushable_rel_predicates(query: &mut CypherQuery, _ctx: &PassCtx) {
    extract_pushable_rel_predicates(query)
}

/// **Pass:** `fold_pass_through_with` — Strip `WITH x AS x` /
/// pass-through `WITH *` clauses that don't reshape the row stream.
/// Removing them lets `reorder_match_clauses` see contiguous Match
/// spans for cross-clause reorder; otherwise the WITH would block.
fn pass_fold_pass_through_with(query: &mut CypherQuery, _ctx: &PassCtx) {
    fold_pass_through_with(query)
}

/// **Pass:** `desugar_multi_match_return_aggregate` — Rewrite
/// `MATCH ... MATCH ... RETURN <group>, <agg>` into the equivalent
/// `MATCH ... MATCH ... WITH <group>, <agg> RETURN <project>` so the
/// aggregate-fusion + top-K pipeline can pick it up. The WITH groups
/// by the user-specified RETURN expressions (per-property), not by the
/// source variable (which would over-finely group when the property
/// has duplicates across instances).
fn pass_desugar_multi_match_return_aggregate(query: &mut CypherQuery, _ctx: &PassCtx) {
    desugar_multi_match_return_aggregate(query)
}

/// **Pass:** `fuse_spatial_join` — Specialize `MATCH ... WHERE
/// contains(geom_a, geom_b)` into a spatial-join iterator that uses
/// the spatial index instead of a cartesian product + per-pair filter.
fn pass_fuse_spatial_join(query: &mut CypherQuery, ctx: &PassCtx) {
    fuse_spatial_join(query, ctx.graph)
}

/// **Pass:** `reorder_match_clauses` — Reorder adjacent `MATCH` clauses
/// by connection-type total counts (O(1) cost proxy) so the smaller
/// driver runs first. Runs BEFORE `optimize_pattern_start_node` so the
/// reversal sees the post-reorder sequence and tracks `bound_vars`
/// correctly.
fn pass_reorder_match_clauses(query: &mut CypherQuery, ctx: &PassCtx) {
    reorder_match_clauses(query, ctx.graph)
}

/// **Pass:** `optimize_pattern_start_node` — For 3+-element patterns,
/// reverse the pattern so iteration starts from the most-selective node
/// (typically id-anchored or smallest-cardinality type). Reduces the
/// front of the join from O(N) to O(1) when one end is anchored.
fn pass_optimize_pattern_start_node(query: &mut CypherQuery, ctx: &PassCtx) {
    optimize_pattern_start_node(query, ctx.graph)
}

/// **Pass:** `reorder_match_patterns` — Reorder multiple comma-
/// separated patterns within one `MATCH` clause by size/type
/// selectivity. Sibling of `reorder_match_clauses` but operates within
/// a single MATCH.
fn pass_reorder_match_patterns(query: &mut CypherQuery, ctx: &PassCtx) {
    reorder_match_patterns(query, ctx.graph)
}

/// **Pass:** `push_limit_into_match` — Mark the trailing `LIMIT N` as
/// an early-stop hint on the preceding `MATCH` so the executor can
/// short-circuit pattern expansion. WHY-BAIL: requires single-MATCH
/// queries (multi-MATCH + WHERE on late-bound var produced silent row
/// drops in 0.8.27 — see CHANGELOG).
fn pass_push_limit_into_match(query: &mut CypherQuery, ctx: &PassCtx) {
    push_limit_into_match(query, ctx.graph)
}

/// **Pass:** `push_limit_into_aggregate` — Stamp `group_limit_hint`
/// on a `RETURN/WITH` that has both group keys and aggregates when the
/// next clause is a literal `LIMIT N`. The aggregator stops creating
/// new groups after `N` distinct keys; rows for already-collected keys
/// continue to feed their aggregates. WHY-BAIL: ORDER BY between
/// projection and LIMIT changes which N rows survive (need every group
/// to find the top N), so the pass leaves those queries to the
/// materialised path. DISTINCT / HAVING also bail. The trailing LIMIT
/// clause stays in the plan as a hard cap.
fn pass_push_limit_into_aggregate(query: &mut CypherQuery, ctx: &PassCtx) {
    push_limit_into_aggregate(query, ctx.graph)
}

/// **Pass:** `push_distinct_into_match` — Mark `RETURN DISTINCT` /
/// `WITH DISTINCT` as a hint on the preceding MATCH so the executor
/// can dedup during expansion instead of materializing all rows first.
fn pass_push_distinct_into_match(query: &mut CypherQuery, _ctx: &PassCtx) {
    push_distinct_into_match(query)
}

/// **Pass:** `fuse_anchored_edge_count` — Specialize
/// `MATCH (id:VAL)-[r:T]->(v) RETURN count(*)` into an O(1) anchored
/// edge lookup using the connection type's edge count metadata.
fn pass_fuse_anchored_edge_count(query: &mut CypherQuery, ctx: &PassCtx) {
    fuse_anchored_edge_count(query, ctx.graph)
}

/// **Pass:** `fuse_count_short_circuits` — Merge `RETURN count(DISTINCT *)`
/// with the preceding COUNT/GROUP BY when both can be evaluated in the
/// same pass.
fn pass_fuse_count_short_circuits(query: &mut CypherQuery, _ctx: &PassCtx) {
    fuse_count_short_circuits(query)
}

/// **Pass:** `fuse_optional_match_aggregate` — Fuse
/// `OPTIONAL MATCH ... RETURN <agg>` into a single
/// `FusedOptionalMatchAggregate` clause that counts matches per input
/// row without materializing intermediate per-row expansions. WHY-BAIL:
/// gate growing — most recently extended in 0.8.31 to recognize edge
/// vars (`count(r)`) as local-to-OPT.
fn pass_fuse_optional_match_aggregate(query: &mut CypherQuery, _ctx: &PassCtx) {
    fuse_optional_match_aggregate(query)
}

/// **Pass:** `fuse_match_return_aggregate` — Fuse
/// `MATCH ... RETURN <group_keys>, <agg>` into
/// `FusedMatchReturnAggregate`, building the GROUP-BY hash map inline
/// during pattern expansion.
fn pass_fuse_match_return_aggregate(query: &mut CypherQuery, _ctx: &PassCtx) {
    fuse_match_return_aggregate(query)
}

/// **Pass:** `fuse_match_with_aggregate` — Like
/// `fuse_match_return_aggregate`, but for `MATCH ... WITH <group>,
/// <agg>` (pipeline continues after WITH). Emits
/// `FusedMatchWithAggregate`.
fn pass_fuse_match_with_aggregate(query: &mut CypherQuery, _ctx: &PassCtx) {
    fuse_match_with_aggregate(query)
}

/// **Pass:** `fuse_match_with_aggregate_top_k` — Absorb a downstream
/// `ORDER BY <agg> LIMIT k` into a preceding
/// `FusedMatchWithAggregate`, replacing full sort with heap-pruned
/// top-K (O(n log k) instead of O(n log n)). Must run AFTER
/// `fuse_match_with_aggregate` and BEFORE `fuse_order_by_top_k`.
fn pass_fuse_match_with_aggregate_top_k(query: &mut CypherQuery, _ctx: &PassCtx) {
    fuse_match_with_aggregate_top_k(query)
}

/// **Pass:** `fuse_node_scan_aggregate` — Untyped `MATCH (n) RETURN
/// <agg>` → specialized scan-only aggregate that walks the node store
/// once without producing intermediate row tuples.
fn pass_fuse_node_scan_aggregate(query: &mut CypherQuery, _ctx: &PassCtx) {
    fuse_node_scan_aggregate(query)
}

/// **Pass:** `fuse_node_scan_top_k` — `MATCH (n:Type) RETURN n LIMIT k`
/// → specialized scan that returns the first k nodes of the type
/// without going through the pattern executor.
fn pass_fuse_node_scan_top_k(query: &mut CypherQuery, _ctx: &PassCtx) {
    fuse_node_scan_top_k(query)
}

/// **Pass:** `fuse_vector_score_order_limit` — `MATCH ...
/// vector_score(...) ORDER BY score LIMIT k` → top-K via a vector-
/// score min-heap. Projects RETURN expressions only for the k surviving
/// rows.
fn pass_fuse_vector_score_order_limit(query: &mut CypherQuery, _ctx: &PassCtx) {
    fuse_vector_score_order_limit(query)
}

/// **Pass:** `fuse_order_by_top_k` — Generic ORDER BY + LIMIT fusion
/// for any preceding clause that didn't already absorb top-K. Heap-
/// pruned top-K replaces full sort + truncate.
fn pass_fuse_order_by_top_k(query: &mut CypherQuery, _ctx: &PassCtx) {
    fuse_order_by_top_k(query)
}

/// **Pass:** `reorder_predicates_by_cost` — Within a WHERE clause,
/// reorder predicates by estimated evaluation cost so cheap predicates
/// short-circuit AND/OR chains before expensive ones run.
fn pass_reorder_predicates_by_cost(query: &mut CypherQuery, _ctx: &PassCtx) {
    reorder_predicates_by_cost(query)
}

/// **Pass:** `mark_fast_var_length_paths` — When a variable-length
/// edge `[:T*1..N]` has no path assignment and no edge variable AND
/// the downstream RETURN/WITH is `DISTINCT` or composed of dedup-safe
/// aggregates (`min/max/count(DISTINCT)/collect(DISTINCT)`), mark
/// `needs_path_info=false` so the executor uses a fast BFS with
/// global target-node dedup. The downstream-safety check is critical:
/// row count is implicit path count, so dedup-by-target silently
/// drops rows when the user wrote a plain per-path projection like
/// `RETURN q.name`. WHY-BAIL: anything else stays on the slow per-
/// path BFS — correct, just not as fast.
fn pass_mark_fast_var_length_paths(query: &mut CypherQuery, _ctx: &PassCtx) {
    mark_fast_var_length_paths(query)
}

/// **Pass:** `mark_skip_target_type_check` — When connection-type
/// metadata guarantees an edge's target node type, mark the edge as
/// `skip_target_type_check=true` so the executor doesn't redundantly
/// re-verify the type during BFS. Saves one slab dereference per
/// visited node.
fn pass_mark_skip_target_type_check(query: &mut CypherQuery, ctx: &PassCtx) {
    mark_skip_target_type_check(query, ctx.graph)
}

/// Mark variable-length edges that don't need path tracking.
///
/// When a MATCH clause has no path assignments (`p = ...`) and the edge
/// has no named variable (`[r:T*1..N]`), AND the query's downstream
/// projection is provably indifferent to row multiplicity, the executor
/// can use a fast BFS with global target-node dedup instead of tracking
/// every distinct path.
///
/// The "indifferent to row multiplicity" check is critical: row count
/// is itself an implicit count of paths in Cypher's semantics, so
/// dedup-by-target silently drops rows when the user wrote a plain
/// per-path projection like `RETURN q.name`. The fast path is only
/// safe when the downstream is `DISTINCT`, or every projection is an
/// aggregate (multiplicity collapses inside the aggregate).
///
/// Caught by `tests/test_cypher_differential.py::var_length_no_var`,
/// which previously xfail'd because the un-gated fast path returned
/// 2 rows where Neo4j semantics demand 3.
fn mark_fast_var_length_paths(query: &mut CypherQuery) {
    if !downstream_is_dedup_safe(query) {
        return;
    }
    for clause in &mut query.clauses {
        let mc = match clause {
            Clause::Match(mc) | Clause::OptionalMatch(mc) => mc,
            _ => continue,
        };

        // If there are path assignments, path info is needed for all patterns
        if !mc.path_assignments.is_empty() {
            continue;
        }

        for pattern in &mut mc.patterns {
            for element in &mut pattern.elements {
                if let PatternElement::Edge(ep) = element {
                    if ep.var_length.is_some() && ep.variable.is_none() {
                        ep.needs_path_info = false;
                    }
                }
            }
        }
    }
}

/// Returns true iff the query's first downstream projection collapses
/// row multiplicity. The fast var-length BFS dedups by target node;
/// that's only correct when the surrounding query doesn't depend on
/// per-path row counts.
///
/// Two safe cases:
/// - `RETURN/WITH DISTINCT` — row tuples are deduped at projection
///   anyway, so a fast-path target-dedup is consistent.
/// - `RETURN/WITH` whose every item is an aggregate — multiplicity is
///   collapsed by the aggregate. (`count(*)` over the matches: paths
///   would count differently than targets, so we don't allow `count(*)`
///   here unless it's `count(DISTINCT target)` — but the simpler check
///   "every item is an aggregate" handles `count(DISTINCT target)`,
///   `sum(target.x)`, etc. uniformly. Plain `count(*)` over var-length
///   matches is a real semantic question; we conservatively reject
///   non-DISTINCT `count(*)` by requiring DISTINCT-aware aggregates.)
///
/// Conservative anywhere else: we'd rather skip the optimization than
/// silently drop rows.
fn downstream_is_dedup_safe(query: &CypherQuery) -> bool {
    for clause in &query.clauses {
        match clause {
            Clause::Return(r) => {
                if r.distinct {
                    return true;
                }
                let all_agg_distinct = !r.items.is_empty()
                    && r.items
                        .iter()
                        .all(|item| is_distinct_safe_aggregate(&item.expression));
                return all_agg_distinct;
            }
            Clause::With(w) => {
                if w.distinct {
                    return true;
                }
                let all_agg_distinct = !w.items.is_empty()
                    && w.items
                        .iter()
                        .all(|item| is_distinct_safe_aggregate(&item.expression));
                return all_agg_distinct;
            }
            _ => continue,
        }
    }
    false
}

/// True when an expression is an aggregate that's invariant to row
/// multiplicity: `count(DISTINCT _)`, `min/max(_)`, `collect(DISTINCT _)`.
/// Plain `count(_)` and `sum(_)` would shift with row count, so they
/// don't qualify.
fn is_distinct_safe_aggregate(expr: &Expression) -> bool {
    if let Expression::FunctionCall {
        name,
        args: _,
        distinct,
    } = expr
    {
        let nm = name.to_lowercase();
        if matches!(nm.as_str(), "min" | "max") {
            return true;
        }
        if *distinct && matches!(nm.as_str(), "count" | "collect") {
            return true;
        }
    }
    false
}

/// Skip node type checks when the connection type metadata guarantees the target type.
///
/// For a pattern like `(a:Person)-[:AUTHORED]->(b:Paper)`, if `AUTHORED` edges
/// only ever connect Person→Paper, then checking `node_weight(target).node_type`
/// in the BFS inner loop is redundant. This saves one `StableDiGraph` slab
/// dereference per visited node.
fn mark_skip_target_type_check(query: &mut CypherQuery, graph: &DirGraph) {
    use crate::graph::core::pattern_matching::EdgeDirection;

    for clause in &mut query.clauses {
        let mc = match clause {
            Clause::Match(mc) | Clause::OptionalMatch(mc) => mc,
            _ => continue,
        };

        for pattern in &mut mc.patterns {
            let elements = &mut pattern.elements;
            // Walk elements in triples: Node, Edge, Node
            let len = elements.len();
            for i in 0..len {
                if i + 2 >= len {
                    break;
                }
                // Extract edge and target node info without overlapping borrows
                let (conn_type, direction, target_node_type) = {
                    let edge = match &elements[i + 1] {
                        PatternElement::Edge(ep) => ep,
                        _ => continue,
                    };
                    let target = match &elements[i + 2] {
                        PatternElement::Node(np) => np,
                        _ => continue,
                    };
                    match (&edge.connection_type, edge.direction, &target.node_type) {
                        (Some(ct), dir, Some(nt)) => (ct.clone(), dir, nt.clone()),
                        _ => continue,
                    }
                };

                // Look up connection type metadata
                if let Some(info) = graph.connection_type_metadata.get(&conn_type) {
                    let guaranteed = match direction {
                        EdgeDirection::Outgoing => {
                            info.target_types.len() == 1
                                && info.target_types.contains(&target_node_type)
                        }
                        EdgeDirection::Incoming => {
                            info.source_types.len() == 1
                                && info.source_types.contains(&target_node_type)
                        }
                        EdgeDirection::Both => false, // can't guarantee for bidirectional
                    };
                    if guaranteed {
                        if let PatternElement::Edge(ep) = &mut elements[i + 1] {
                            ep.skip_target_type_check = true;
                        }
                    }
                }
            }
        }
    }
}

// Historical note: the fusion docstrings for `FusedCountAll`,
// `FusedCountByType`, `FusedCountEdgesByType`, and
// `FusedCountAnchoredEdges` moved to their respective fuse functions in
// `src/graph/languages/cypher/planner/fusion.rs` during the Phase 9
// split. See those functions for the current prose.

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "planner_tests.rs"]
mod tests;
