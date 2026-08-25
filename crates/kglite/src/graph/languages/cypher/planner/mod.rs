//! Cypher query optimizer.

use super::ast::*;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::PatternElement;
use crate::graph::schema::DirGraph;
use std::collections::{HashMap, HashSet};

mod annotations;
mod invariants;
#[cfg(debug_assertions)]
use invariants::debug_check_invariants;
pub mod cost_model;
pub mod fusion;
pub mod index_selection;
pub mod join_order;
mod node_anchor;
pub mod rel_predicate_pushdown;
pub mod schema_check;
pub mod simplification;
mod var_length_lowering;

use annotations::{
    pass_mark_disjoint_fixed_trails, pass_mark_fast_var_length_paths,
    pass_mark_skip_target_type_check,
};
use cost_model::reorder_predicates_by_cost;
use fusion::{
    fuse_anchored_edge_count, fuse_count_short_circuits, fuse_match_return_aggregate,
    fuse_match_with_aggregate, fuse_match_with_aggregate_top_k, fuse_node_scan_aggregate,
    fuse_node_scan_top_k, fuse_optional_match_aggregate, fuse_order_by_top_k, fuse_spatial_join,
    fuse_text_bm25_order_limit, fuse_vector_score_order_limit, mark_return_lazy_eligible,
};
use index_selection::push_where_into_match;
use join_order::{
    optimize_pattern_start_node, reorder_cyclic_pattern_edges, reorder_match_clauses,
    reorder_match_patterns,
};
use node_anchor::anchor_element_id;
use rel_predicate_pushdown::extract_pushable_rel_predicates_with_params;
use var_length_lowering::lower_fixed_var_length_hops;

use simplification::{
    desugar_multi_match_return_aggregate, fold_or_to_in, fold_pass_through_with,
    narrow_unwind_source, push_distinct_into_match, push_limit_into_aggregate,
    push_limit_into_match, rewrite_count_bound_var_to_star,
};

/// Carries the per-call inputs every pass might need, so a new dependency
/// extends this struct rather than 25+ wrapper signatures.
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
///
/// ## `CALL { }` (CallSubquery) barrier audit
///
/// `Clause::CallSubquery` is an OPAQUE barrier to every pass below. A
/// subquery's per-row cardinality is unknown at plan time and a
/// correlated body depends on its seeded input — so NO pass may move a
/// clause across it, fuse a window through it, or push a LIMIT/predicate
/// into or past it. Every pass is safe today; only two are not safe
/// purely by shape:
///
/// - `optimize_nested_queries` — **recurses by design.** Owns body
///   optimization; import-aware (disables seed-ignoring fusion for
///   anchored correlated bodies).
/// - `fold_pass_through_with` — **guarded.** Its downstream-ref check
///   records a CallSubquery's import names + body refs (see
///   `collect_clause_variables`), so a `WITH` a correlated CALL depends
///   on is never folded away.
///
/// Every other pass either matches a fixed clause-adjacency window
/// (which a CallSubquery breaks), rewrites/reorders WITHIN one clause,
/// or fires only on a whole-query shape (`[Match, Return]`, len 2); the
/// clause walks hit `_ => continue` or end their span at `_ => break`,
/// and the downstream-dedup-safety scan stops at the first Return/With,
/// which a CallSubquery is not. Two conservative details: prior-scope
/// helpers under-report CALL outputs, so pushdown under-pushes; and
/// `anchor_element_id` writes only a hint while leaving its predicate
/// standing, so a hint written against a body it could not see changes
/// no answer.
pub const PASSES: &[(&str, PassFn)] = &[
    ("optimize_nested_queries", pass_optimize_nested_queries),
    // `*k..k` → k explicit hops. Runs FIRST among the structural rewrites:
    // every pass below bails on `var_length.is_some()`, so a segment lowered
    // any later would inherit none of them.
    (
        "lower_fixed_var_length_hops",
        pass_lower_fixed_var_length_hops,
    ),
    // count(bound node/edge var) → count(*): runs early so the rewritten
    // count(*) reaches the count-fusion + light-row MATCH paths.
    (
        "rewrite_count_bound_var_to_star",
        pass_rewrite_count_bound_var_to_star,
    ),
    ("push_where_into_match.1", pass_push_where_into_match),
    ("fold_or_to_in", pass_fold_or_to_in),
    // second push_where pass: catches IN predicates created by fold_or_to_in
    ("push_where_into_match.2", pass_push_where_into_match),
    // Slot anchors from elementId() equality. Must precede the join-order
    // passes: they read `node_anchors` to score an anchored variable as
    // fully selective.
    ("anchor_element_id", pass_anchor_element_id),
    (
        "extract_pushable_rel_predicates",
        pass_extract_pushable_rel_predicates,
    ),
    // strip pass-through WITH BEFORE cross-clause MATCH reorder so the
    // latter sees a contiguous Match-Match span when a `WITH p` sat between.
    ("fold_pass_through_with", pass_fold_pass_through_with),
    // Runs AFTER fold_pass_through_with: folding a pass-through WITH changes
    // which clauses sit downstream of an UNWIND, and this pass's whole job is
    // to read that downstream set. Running it first would decide against a
    // stale clause list.
    ("narrow_unwind_source", pass_narrow_unwind_source),
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
    // Re-root simple cyclic patterns at their most-selective node BEFORE
    // pattern_start_node (which can't help a cycle — both ends are the same
    // variable, so its reverse is a no-op).
    (
        "reorder_cyclic_pattern_edges",
        pass_reorder_cyclic_pattern_edges,
    ),
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
    // Same three-clause shape as fuse_order_by_top_k, so it must run first or
    // the generic pass takes it: a bail here is a fall-through to that pass,
    // never a lost answer.
    (
        "fuse_text_bm25_order_limit",
        pass_fuse_text_bm25_order_limit,
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
        "mark_disjoint_fixed_trails",
        pass_mark_disjoint_fixed_trails,
    ),
    (
        "mark_skip_target_type_check",
        pass_mark_skip_target_type_check,
    ),
];

/// PyAPI uses this to reject typos in the `disabled_passes` kwarg before
/// they silently suppress nothing.
pub fn is_known_pass(name: &str) -> bool {
    PASSES.iter().any(|(n, _)| *n == name)
}

/// Backs the PyAPI's `disable_optimizer=True` shortcut, which expands to
/// "disable everything".
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

/// Run the optimizer pipeline with no passes disabled. The entry point
/// production callers (executor, transactions, mutations) should use.
pub fn optimize(query: &mut CypherQuery, graph: &DirGraph, params: &HashMap<String, Value>) {
    optimize_with_disabled(query, graph, params, empty_disabled_set());
}

/// Process-lifetime empty `HashSet<String>` used as the no-knob default.
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
    query.optimizer_tags.clear();
    let ctx = PassCtx {
        graph,
        params,
        disabled,
    };
    for (name, pass_fn) in PASSES {
        if disabled.contains(*name) {
            continue;
        }
        let before = query.explain.then(|| format!("{:?}", query.clauses));
        pass_fn(query, &ctx);
        if before.is_some_and(|snapshot| snapshot != format!("{:?}", query.clauses)) {
            query.optimizer_tags.push((*name).to_string());
        }
        #[cfg(debug_assertions)]
        debug_check_invariants(query, name);
    }
}

/// **Pass:** `lower_fixed_var_length_hops` — **Precondition:** a
/// `MATCH` / `OPTIONAL MATCH` with no path assignment. **Pattern
/// matched:** a relationship element whose `var_length` is `(k, k)`,
/// `k >= 1`. **Rewrite:** `k` copies of that element with
/// `var_length` cleared, separated by anonymous unlabelled nodes; type
/// alternations, inline relationship properties and direction are
/// replicated onto every copy, which is what variable-length semantics
/// already demand of each relationship in the segment. **Why:** every
/// pass below — relationship pushdown, start-node selection, the fusion
/// family, `mark_disjoint_fixed_trails`, `mark_skip_target_type_check` —
/// declines a variable-length element, so the star spelling of a
/// fixed-length question paid for none of them (measured 3.4x to 17x
/// inside the one-or-two-hop window; the table is in
/// `var_length_lowering`).
///
/// **Why-bail:** `min != max`; `k == 0` (the zero-length identity is not
/// a hop); `k` or the pattern's post-lowering hop count above the
/// module's ceiling; a bound relationship variable (`r` binds the
/// *list*); a path assignment anywhere in the clause; a pre-existing
/// `edge_filter` or unresolved `-[:$type]->` slot (neither can be set at
/// this point in `PASSES` — they are "the pipeline moved" guards).
///
/// The trail-semantics argument, and why the disjoint-types opt-out
/// cannot fire on a lowered segment, are in `var_length_lowering`'s
/// module docs.
fn pass_lower_fixed_var_length_hops(query: &mut CypherQuery, _ctx: &PassCtx) {
    lower_fixed_var_length_hops(query)
}

/// **Pass:** `optimize_nested_queries` — Recurse the optimizer into
/// every nested query: UNION right-arms and `CALL { }` subquery bodies.
/// Inherits the parent's `disabled` set so diagnostic toggles propagate
/// to the inner planner pipeline — including the `disable_optimizer=True`
/// expansion, which puts every pass name (this one among them) into
/// `disabled`. When THIS pass is itself disabled the recursion never
/// runs, so a fully-disabled optimizer leaves bodies un-optimized too,
/// making the differential corpus's optimized-vs-naive comparison
/// meaningful for subquery bodies.
///
/// This pass OWNS `CALL { }` body optimization (the executor runs the
/// body exactly as planned here). Two body shapes are optimized
/// differently:
///
/// - **Uncorrelated body** (`import.is_empty()`) or a correlated body
///   whose patterns do NOT anchor on an imported variable: the full
///   pipeline runs. A graph-global aggregate in such a body is genuinely
///   the same value for every outer row, so the seed-ignoring fused
///   operators are correct.
/// - **Correlated body whose patterns anchor on an imported variable**
///   (`!import_pattern_anchors(body, import).is_empty()`): the
///   [`seed_ignoring_fusion_passes`] are disabled for that body — they
///   emit plan-time-anchored operators that ignore the per-row seed and
///   would return the GLOBAL count for every outer row. Disabling them
///   leaves a plain `Match`/`Return` that honours the seeded binding via
///   CSR adjacency (§3.2). The disable is unioned with the inherited
///   `disabled` set so an outer toggle still propagates.
fn pass_optimize_nested_queries(query: &mut CypherQuery, ctx: &PassCtx) {
    for clause in &mut query.clauses {
        match clause {
            Clause::Union(ref mut u) => {
                optimize_with_disabled(&mut u.query, ctx.graph, ctx.params, ctx.disabled);
            }
            Clause::CallSubquery {
                ref import,
                ref mut body,
            } => {
                let anchors = import_pattern_anchors(body, import);
                if anchors.is_empty() {
                    optimize_with_disabled(body, ctx.graph, ctx.params, ctx.disabled);
                } else {
                    // Union the seed-ignoring set with the inherited
                    // disabled set so both the per-row-correctness disable
                    // AND any outer diagnostic toggle apply to the body.
                    let mut merged = ctx.disabled.clone();
                    merged.extend(seed_ignoring_fusion_passes().iter().cloned());
                    optimize_with_disabled(body, ctx.graph, ctx.params, &merged);
                }
            }
            _ => {}
        }
    }
}

/// The subset of `import` names that appear as a `MATCH` / `OPTIONAL
/// MATCH` pattern element in a correlated `CALL { }` body (so the body
/// anchors on the seeded binding). Non-empty ⇒ the seed-ignoring fusion
/// passes must be disabled when optimizing the body, and (in the
/// executor) a NULL value for any of these names empties the per-row
/// pipeline (§1.3 of the design doc).
///
/// Only the body's OWN clauses are scanned — a nested `CALL { }` re-binds
/// its own imports from its own seed, so its patterns are not this body's
/// concern.
pub(crate) fn import_pattern_anchors(body: &CypherQuery, import: &[String]) -> Vec<String> {
    let mut anchors: Vec<String> = Vec::new();
    for clause in &body.clauses {
        let patterns = match clause {
            Clause::Match(m) | Clause::OptionalMatch(m) => &m.patterns,
            _ => continue,
        };
        for pattern in patterns {
            for elem in &pattern.elements {
                let var = match elem {
                    PatternElement::Node(np) => np.variable.as_ref(),
                    PatternElement::Edge(ep) => ep.variable.as_ref(),
                };
                if let Some(v) = var {
                    if import.iter().any(|name| name == v) && !anchors.iter().any(|a| a == v) {
                        anchors.push(v.clone());
                    }
                }
            }
        }
    }
    anchors
}

/// The optimizer passes that emit a graph-global / plan-time-anchored
/// operator (`FusedCount*`, `FusedMatch*Aggregate`, `FusedNodeScan*`)
/// which IGNORES the incoming seed row. Disabled when a correlated body
/// anchors on an imported variable (see [`pass_optimize_nested_queries`]).
///
/// These names MUST stay in sync with `PASSES`; each is a registered pass
/// name. A future `fuse_call_subquery_aggregate` pass (design §Q7) would
/// be the correct seed-AWARE replacement and would NOT belong here.
pub(crate) fn seed_ignoring_fusion_passes() -> &'static HashSet<String> {
    static PASSES_SET: std::sync::OnceLock<HashSet<String>> = std::sync::OnceLock::new();
    PASSES_SET.get_or_init(|| {
        [
            "fuse_anchored_edge_count",
            "fuse_count_short_circuits",
            "fuse_optional_match_aggregate",
            "fuse_match_return_aggregate",
            "fuse_match_with_aggregate",
            "fuse_match_with_aggregate_top_k",
            "fuse_node_scan_aggregate",
            "fuse_node_scan_top_k",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    })
}

/// **Pass:** `push_where_into_match` — Move comparison predicates from
/// a trailing `WHERE` clause into the preceding `MATCH`'s
/// `PropertyMatcher`. The matcher applies them during pattern expansion
/// instead of evaluating them per row, pruning the search early.
///
/// Two sources of predicate, one rewrite: a trailing `Clause::Where` after
/// a `MATCH`/`OPTIONAL MATCH`, and an `OPTIONAL MATCH`'s own
/// `MatchClause::where_clause`. The second is safe because under clause
/// scoping a candidate the predicate rejects and a candidate the pattern
/// never produced are the same outcome — the row is null-extended either
/// way — so moving the test from the predicate into the pattern cannot
/// change which rows survive. The safety-net rule is unchanged in both
/// homes: a partial push leaves the whole predicate standing, and a full
/// push keeps it as the filter every non-pattern-matcher path still relies
/// on.
fn pass_push_where_into_match(query: &mut CypherQuery, ctx: &PassCtx) {
    push_where_into_match(query, ctx.params)
}

/// **Pass:** `anchor_element_id` — Record the slot named by
/// `WHERE elementId(v) = <literal|$param>` on the MATCH clause, so the
/// executor seeds it as a pre-binding instead of scanning for it.
///
/// **Precondition:** a `Clause::Match`/`Clause::OptionalMatch` with a WHERE —
/// its own (the scoped `OPTIONAL MATCH` form) or the adjacent standalone one.
///
/// **Pattern matched:** an `Equals` comparison, either operand order, between
/// `elementId(v)` — where `v` is a node variable of *this* clause's patterns —
/// and a value that parses to a non-negative slot. Only the predicate's `And`
/// spine is descended.
///
/// **Rewrite:** pushes `(v, NodeIndex)` onto `MatchClause::node_anchors`. The
/// predicate is left standing, so this narrows the candidate set without
/// owning the answer: an out-of-range or stale slot resolves to no node, which
/// is what the retained predicate would have concluded.
///
/// **Why-bail:** `Or`/`Not`/`Xor` are not descended (a disjunct constrains
/// nothing, and a negation inverts the reasoning); a non-numeric, negative or
/// unbound value does not name a slot; a variable this clause does not bind
/// belongs to another clause's search space. Conflicting anchors on one
/// variable keep the first — two of them cannot both hold, and the predicate
/// rejects the loser.
fn pass_anchor_element_id(query: &mut CypherQuery, ctx: &PassCtx) {
    anchor_element_id(query, ctx.params)
}

/// **Pass:** `fold_or_to_in` — Rewrite `(a.x = v1 OR a.x = v2 OR ...)`
/// chains into `a.x IN [v1, v2, ...]`. Lets the second
/// `push_where_into_match` push the synthesized IN as a single
/// equality-set matcher.
fn pass_fold_or_to_in(query: &mut CypherQuery, _ctx: &PassCtx) {
    fold_or_to_in(query)
}

/// **Pass:** `rewrite_count_bound_var_to_star` — rewrite non-distinct
/// `count(v)` to `count(*)` when `v` is a mandatorily-bound node/edge variable
/// (so always non-null). Avoids per-row node materialization and heavy binding
/// retention on deep-path counts. WHY-BAIL: DISTINCT, OPTIONAL-bound `v`, or any
/// `WITH` present. Column name preserved via alias.
fn pass_rewrite_count_bound_var_to_star(query: &mut CypherQuery, _ctx: &PassCtx) {
    rewrite_count_bound_var_to_star(query)
}

/// **Pass:** `extract_pushable_rel_predicates` — Inline edge-side
/// predicates (`type(r) = 'X'`, `r.prop OP literal`, `startNode(r) =
/// peer`) from a trailing WHERE into the edge's `rel_predicate`. The
/// matcher applies them during expansion, before per-edge bindings are
/// allocated. WHY-BAIL: predicates referencing unbound vars stay in WHERE.
fn pass_extract_pushable_rel_predicates(query: &mut CypherQuery, ctx: &PassCtx) {
    extract_pushable_rel_predicates_with_params(query, ctx.params)
}

/// **Pass:** `fold_pass_through_with` — Strip `WITH x AS x` /
/// pass-through `WITH *` clauses that don't reshape the row stream.
/// Removing them lets `reorder_match_clauses` see contiguous Match
/// spans for cross-clause reorder; otherwise the WITH would block.
fn pass_fold_pass_through_with(query: &mut CypherQuery, _ctx: &PassCtx) {
    fold_pass_through_with(query)
}

/// **Pass:** `narrow_unwind_source` — mark `UNWIND <var> AS alias` whose
/// source binding is dead after the clause, letting the executor take the list
/// out of the row by move instead of cloning it into all `n` expanded rows.
///
/// **Precondition:** the UNWIND source is a bare variable (a computed source is
/// never bound in the row). **Pattern:** no clause after the UNWIND mentions
/// the source variable, and every such clause has fully enumerable references.
/// **Rewrite:** sets `UnwindClause::consume_source`; adds/removes/reorders
/// nothing and cannot change results. **Why bail:** a write / fused /
/// procedure clause downstream, whose references are not enumerable, or any
/// downstream mention of the variable — either way the binding may still be
/// read, so the copy is kept.
///
/// Fixes the quadratic memory of `WITH collect(x) AS xs UNWIND xs AS y`:
/// `n` rows each retaining an `n`-element copy of the same list.
fn pass_narrow_unwind_source(query: &mut CypherQuery, _ctx: &PassCtx) {
    narrow_unwind_source(query)
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
/// contains(geom_a, geom_b)` into a spatial-join iterator that probes a
/// per-query R-tree over the container type instead of running a
/// cartesian product + per-pair filter. WHY-BAIL: a graph with secondary
/// labels — the R-tree is built from primary `type_indices` only, so it
/// would miss secondary-labelled nodes.
fn pass_fuse_spatial_join(query: &mut CypherQuery, ctx: &PassCtx) {
    fuse_spatial_join(query, ctx.graph)
}

/// **Pass:** `reorder_match_clauses` — Reorder adjacent `MATCH` clauses
/// by connection-type total counts (O(1) cost proxy) so the smaller
/// driver runs first.
fn pass_reorder_match_clauses(query: &mut CypherQuery, ctx: &PassCtx) {
    reorder_match_clauses(query, ctx.graph)
}

/// **Pass:** `reorder_cyclic_pattern_edges` — Re-root a simple cyclic pattern
/// (a ring whose start variable repeats at the end) at its most-selective node,
/// orienting the walk so the cheaper incident edge drives first. Turns the
/// cycle-closing segment into an O(1) bound-target check in the matcher.
/// Shape-gated: only fires on simple rings of clean single-typed edges and only
/// on a clear (≥4×) selectivity win, leaving every acyclic pattern unchanged.
fn pass_reorder_cyclic_pattern_edges(query: &mut CypherQuery, ctx: &PassCtx) {
    reorder_cyclic_pattern_edges(query, ctx.graph)
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
/// queries (multi-MATCH + WHERE on a late-bound var silently drops
/// rows).
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
/// materialised path. DISTINCT, `RETURN … HAVING` and a `WITH`'s inline
/// `WHERE`/`HAVING` also bail — the filter runs *after* the projection,
/// so a capped group set would drop qualifying groups the LIMIT still
/// had room for. The trailing LIMIT clause stays in the plan as a hard
/// cap.
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

/// **Pass:** `fuse_count_short_circuits` — Answer a `MATCH` + `RETURN`
/// count query from type-bucket / CSR metadata instead of expanding the
/// pattern: `MATCH (n[:Type])` or `MATCH ()-[r[:T]]->()` with
/// `count(*)`/`count(var)`, optionally paired with a `type(r)`/`labels(n)`
/// group key. WHY-BAIL: `DISTINCT`, `HAVING`, a path assignment, more than
/// one pattern, an inline property filter or a repeated variable, a
/// multi-label `(n:A:B)` (needs an intersection the O(1) bucket count
/// cannot express), and undirected or `[:A|B]` edges.
fn pass_fuse_count_short_circuits(query: &mut CypherQuery, ctx: &PassCtx) {
    fuse_count_short_circuits(query, ctx.graph.has_secondary_labels, ctx.graph)
}

/// **Pass:** `fuse_optional_match_aggregate` — Fuse
/// `OPTIONAL MATCH ... RETURN <agg>` into a single
/// `FusedOptionalMatchAggregate` clause that counts matches per input
/// row without materializing intermediate per-row expansions. WHY-BAIL:
/// edge vars (`count(r)`) count as local-to-OPT; multi-pattern clauses
/// and a clause-owned `WHERE` (`OPTIONAL MATCH … WHERE …`) bail, the latter
/// because the fused counter counts a pattern's matches with no hook to
/// test a predicate per candidate.
fn pass_fuse_optional_match_aggregate(query: &mut CypherQuery, _ctx: &PassCtx) {
    fuse_optional_match_aggregate(query)
}

/// **Pass:** `fuse_match_return_aggregate` — Fuse
/// `MATCH ... RETURN <group_keys>, <agg>` into
/// `FusedMatchReturnAggregate`, building the GROUP-BY hash map inline
/// during pattern expansion.
fn pass_fuse_match_return_aggregate(query: &mut CypherQuery, ctx: &PassCtx) {
    fuse_match_return_aggregate(query, ctx.graph)
}

/// **Pass:** `fuse_match_with_aggregate` — Like
/// `fuse_match_return_aggregate`, but for `MATCH ... WITH <group>,
/// <agg>` (pipeline continues after WITH). Emits
/// `FusedMatchWithAggregate`.
fn pass_fuse_match_with_aggregate(query: &mut CypherQuery, ctx: &PassCtx) {
    fuse_match_with_aggregate(query, ctx.graph)
}

/// **Pass:** `fuse_match_with_aggregate_top_k` — Absorb a downstream
/// `ORDER BY <agg> LIMIT k` into a preceding
/// `FusedMatchWithAggregate`, replacing full sort with heap-pruned
/// top-K (O(n log k) instead of O(n log n)).
fn pass_fuse_match_with_aggregate_top_k(query: &mut CypherQuery, _ctx: &PassCtx) {
    fuse_match_with_aggregate_top_k(query)
}

/// **Pass:** `fuse_node_scan_aggregate` — Untyped `MATCH (n) RETURN
/// <agg>` → specialized scan-only aggregate that walks the node store
/// once without producing intermediate row tuples.
fn pass_fuse_node_scan_aggregate(query: &mut CypherQuery, ctx: &PassCtx) {
    fuse_node_scan_aggregate(query, ctx.params)
}

/// **Pass:** `fuse_node_scan_top_k` — `MATCH (n:Type) RETURN n LIMIT k`
/// → specialized scan that returns the first k nodes of the type
/// without going through the pattern executor.
fn pass_fuse_node_scan_top_k(query: &mut CypherQuery, ctx: &PassCtx) {
    fuse_node_scan_top_k(query, ctx.params)
}

/// **Pass:** `fuse_vector_score_order_limit` — `MATCH ...
/// vector_score(...) ORDER BY score LIMIT k` → top-K via a vector-
/// score min-heap. Projects RETURN expressions only for the k surviving
/// rows.
fn pass_fuse_vector_score_order_limit(query: &mut CypherQuery, _ctx: &PassCtx) {
    fuse_vector_score_order_limit(query)
}

/// **Pass:** `fuse_text_bm25_order_limit` — `RETURN ... text_bm25(...) AS s
/// ORDER BY s DESC LIMIT k` → one postings-driven top-k operator.
fn pass_fuse_text_bm25_order_limit(query: &mut CypherQuery, _ctx: &PassCtx) {
    fuse_text_bm25_order_limit(query)
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

#[cfg(test)]
#[path = "planner_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "planner_fusion_tests.rs"]
mod fusion_tests;
