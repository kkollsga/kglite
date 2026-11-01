//! Which optimizer passes may run on a scope that carries a guard template
//! (a statement prefixed `FOR VALID_TIME AS OF`). Default-deny: a pass runs
//! under a guard only when it is listed in [`GUARD_SAFE_PASSES`]. Every name
//! in `PASSES` is in exactly one of the two lists with the reason for its
//! verdict, and a test fails when a new pass is in neither.
//!
//! The premise the verdicts rest on: the guard is applied where the pattern
//! matcher accepts an element, so a pass is safe when every row it leaves in
//! the plan still comes out of the matcher. A pass that answers from a
//! store, an index, a count or a precomputed frontier without the matcher
//! would bypass the guard and is denied until it is given a guarded form.

/// Passes that run under a guard, each with why it is safe.
pub(super) const GUARD_SAFE_PASSES: &[(&str, &str)] = &[
    (
        "optimize_nested_queries",
        "recursion only; each nested scope carries its own template",
    ),
    (
        "lower_fixed_var_length_hops",
        "makes the intermediate hops explicit nodes, which the matcher then accepts",
    ),
    ("rewrite_count_bound_var_to_star", "expression rewrite"),
    (
        "hoist_with_where",
        "moves a predicate; rows still come from the matcher",
    ),
    (
        "hoist_where_over_call_subquery",
        "moves a predicate ahead of a read-only CALL whose body keeps its own guard; \
         rows still come from the matcher",
    ),
    (
        "push_where_into_match.1",
        "a predicate into the matcher; candidates are still accepted by it",
    ),
    ("fold_or_to_in", "expression rewrite"),
    (
        "push_where_into_match.2",
        "a predicate into the matcher; candidates are still accepted by it",
    ),
    (
        "extract_pushable_rel_predicates",
        "an edge predicate into the edge filter; independent of the guard",
    ),
    ("fold_pass_through_with", "clause structure only"),
    ("fold_aliasing_with", "clause structure only"),
    (
        "hoist_terminal_return_over_with_top_k",
        "clause structure only",
    ),
    ("narrow_unwind_source", "clause structure only"),
    (
        "desugar_multi_match_return_aggregate",
        "shape rewrite; it only feeds the fusion passes, each with its own verdict",
    ),
    (
        "reorder_match_clauses",
        "order only; guards on every element make the order irrelevant to the answer",
    ),
    (
        "reorder_cyclic_pattern_edges",
        "order only; guards on every element make the order irrelevant to the answer",
    ),
    (
        "optimize_pattern_start_node",
        "order only; guards on every element make the order irrelevant to the answer",
    ),
    (
        "reorder_match_patterns",
        "order only; guards on every element make the order irrelevant to the answer",
    ),
    (
        "push_distinct_into_match",
        "a dedup hint inside the matcher, applied to rows it accepted",
    ),
    ("reorder_predicates_by_cost", "WHERE evaluation order only"),
    (
        "mark_disjoint_fixed_trails",
        "trail bookkeeping only; no element is skipped",
    ),
    (
        "anchor_element_id",
        "the anchor is a pre-binding, which the matcher's candidate guard re-tests",
    ),
    (
        "anchor_rel_property_index",
        "an inline edge matcher; index seeds pass the candidate guard and every hop is tested",
    ),
    (
        "push_limit_into_aggregate",
        "a group cap over rows the matcher already admitted",
    ),
    (
        "fuse_count_short_circuits",
        "its counts have a guarded form that tests every node and relationship \
         (execute_fused_count_guarded)",
    ),
    (
        "fuse_node_scan_aggregate",
        "its candidates come from the matcher's guarded node scan",
    ),
    (
        "columnar_scan_aggregate",
        "a marker only; the executor declines when the statement's filter may hide the \
         label, so a guarded query runs the matcher's guarded scan",
    ),
    (
        "fuse_node_scan_top_k",
        "its candidates come from the matcher's guarded node scan",
    ),
    (
        "fuse_order_by_top_k",
        "a bounded heap over rows the matcher already admitted",
    ),
    (
        "fuse_with_top_k",
        "the same three operators as the RETURN-shape fusions, over rows the matcher \
         already admitted; only the projection scope differs",
    ),
    (
        "fuse_chain_path_count",
        "each hop's relationships and peers are put to the filter (admits_hop) and the \
         start nodes come from the guarded matcher scan, so the DP counts exactly the \
         paths the matcher would admit, and the distinct form sweeps the same admitted \
         relationships forward and backward (chain_count.rs)",
    ),
    (
        "fuse_match_return_aggregate",
        "3-element patterns only: its group scan, per-node counters and node lists \
         all test the filter (guarded_ops.rs, fused_match.rs); the two-hop shape stays unfused",
    ),
    (
        "fuse_match_with_aggregate",
        "3-element patterns only: the group scan is the matcher's, the counts are the \
         guarded per-node counters, and the peer histograms decline under a filter \
         (fused_match.rs, guarded_ops.rs)",
    ),
    (
        "fuse_match_with_aggregate_top_k",
        "a bounded trim of counts the guarded operator already produced",
    ),
    (
        "fuse_vector_score_order_limit",
        "its entry ranks only the nodes the filter admits and its per-clause route \
         ranks rows the matcher admitted (retrieval_mask.rs)",
    ),
    (
        "fuse_text_bm25_order_limit",
        "its entry and per-clause route rank only admitted documents, under the \
         admitted corpus's statistics (retrieval_mask.rs)",
    ),
    (
        "push_limit_into_match",
        "it only stamps a row cap on the MATCH, and every cap site counts rows the guard \
         has already admitted: candidates are guarded before truncation, index seeds \
         through guard_seeds with the capped-pass retry, and a hop is tested before it \
         is pushed (matcher_guard.rs, matcher_expansion.rs)",
    ),
    (
        "fuse_optional_match_aggregate",
        "its driving rows come from the guarded MATCH, each row's count is the guarded \
         per-node counter (the bound node, every relationship and the peer are tested), \
         and a pattern the counter cannot take runs the filtered matcher \
         (fused_match.rs, guarded_ops.rs)",
    ),
    (
        "fuse_spatial_join",
        "the container list the R-tree is built from and the probe loop both keep only \
         nodes the filter admits, and the residual predicate runs through the filtered \
         executor (spatial_join.rs)",
    ),
    (
        "fuse_anchored_edge_count",
        "the plan holds the id literal, not a node or an instant; the executor resolves \
         the anchors visible at the statement's instant and counts only relationships \
         the filter admits with their far endpoint (guarded_ops.rs)",
    ),
    (
        "mark_fast_var_length_paths",
        "the distance frontier has a guarded form that tests every relationship and \
         node it crosses before marking it (matcher_var_length_guarded.rs)",
    ),
];

/// Passes that never run under a guard, each with what it would bypass. The
/// planner reads only the allow-list; this list is the verdict record the
/// walk test holds every registered pass against.
#[cfg(test)]
const GUARD_DENIED_PASSES: &[(&str, &str)] = &[
    (
        "mark_skip_target_type_check",
        "skips the node-type check that label guards rely on",
    ),
    (
        "fuse_unwind_valid_at",
        "scans the pattern once without the statement's filter and keeps matches by \
         a mask of its own, so it would ignore the filter on every other element",
    ),
];

/// Planner work outside `PASSES` that a guarded scope also skips, each with
/// what it would bypass. The planner tests `CypherQuery::context` /
/// `CypherQuery::guard` at the call site; this list is its verdict record.
#[cfg(test)]
const GUARD_DENIED_PREPASSES: &[(&str, &str)] = &[(
    "fold_constant_inline_maps",
    "evaluates a row-independent inline-map value (`{n: COUNT { (:Well) }}`) at plan \
     time with an executor that has no filter, and caches the literal with the plan",
)];

/// Whether pass `name` may run on a guarded scope.
pub(crate) fn is_safe(name: &str) -> bool {
    GUARD_SAFE_PASSES.iter().any(|(safe, _)| *safe == name)
}

#[cfg(test)]
mod tests {
    use super::super::PASSES;
    use super::*;
    use std::collections::HashSet;

    /// Every registered pass has exactly one verdict, and every listed name
    /// is a registered pass — a new pass in neither list fails here.
    #[test]
    fn every_pass_has_exactly_one_guard_verdict() {
        let safe: HashSet<&str> = GUARD_SAFE_PASSES.iter().map(|(n, _)| *n).collect();
        let denied: HashSet<&str> = GUARD_DENIED_PASSES.iter().map(|(n, _)| *n).collect();
        assert_eq!(safe.len(), GUARD_SAFE_PASSES.len(), "duplicate safe entry");
        assert_eq!(
            denied.len(),
            GUARD_DENIED_PASSES.len(),
            "duplicate denied entry"
        );
        assert!(
            safe.is_disjoint(&denied),
            "{:?}",
            safe.intersection(&denied)
        );
        let registered: HashSet<&str> = PASSES.iter().map(|(n, _)| *n).collect();
        for name in &registered {
            assert!(
                safe.contains(name) || denied.contains(name),
                "pass '{name}' has no guard verdict: add it to GUARD_SAFE_PASSES or \
                 GUARD_DENIED_PASSES in planner/guard.rs"
            );
        }
        for name in safe.union(&denied) {
            assert!(
                registered.contains(name),
                "'{name}' is not a registered pass"
            );
        }
        for (name, _) in GUARD_DENIED_PREPASSES {
            assert!(
                !registered.contains(name),
                "'{name}' is a registered pass: give it a verdict in the pass lists"
            );
        }
        let all = GUARD_SAFE_PASSES
            .iter()
            .chain(GUARD_DENIED_PASSES)
            .chain(GUARD_DENIED_PREPASSES);
        for (name, reason) in all {
            assert!(!reason.is_empty(), "'{name}' has no reason");
        }
        assert!(!is_safe("mark_skip_target_type_check"));
        assert!(is_safe("optimize_nested_queries"));
    }
}
