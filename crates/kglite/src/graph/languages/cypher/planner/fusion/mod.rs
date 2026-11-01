//! Multi-clause fusion passes — rewrite MATCH+RETURN+AGG, top-K, ORDER BY+LIMIT
//! into specialised physical plans.
//! Note: an earlier draft of this module exposed
//! `match_clause_has_edge_filter` and bailed every fused pass when any
//! edge carried an inline filter. That regressed unfiltered cohort
//! queries by ~250× — the fused histogram fast path got thrown away
//! even though it was still safe to use. The current design keeps
//! fusion enabled and has each fused count helper apply the filter
//! inline (`try_count_simple_pattern`, `try_count_distinct_peers`) or
//! bail itself (`try_fast_with_aggregate_via_histogram`). See those
//! helpers for the details.

mod aggregate;
mod chain_count;
mod columnar_scan;
mod count;
mod spatial;
mod topk;
mod valid_at_scan;
mod with_top_k;
#[cfg(test)]
mod with_top_k_tests;

pub(super) use aggregate::*;
pub(super) use chain_count::*;
pub(crate) use columnar_scan::columnar_count_shape;
pub(super) use columnar_scan::pass_columnar_scan_aggregate;
pub(super) use count::*;
pub(super) use spatial::*;
pub(super) use topk::*;
pub(super) use valid_at_scan::*;
pub(super) use with_top_k::*;

/// True when a projection carries a `*`, which no fused operator can project.
///
/// `*` names no item in the AST: the executor expands it from the *runtime
/// row's* bindings (`executor/return_clause.rs::expand_wildcards`), which a
/// fused operator's own projection never builds. Fusing it projects the
/// literal `Star` expression instead, which evaluates to the `1` that exists
/// so `count(*)` has an argument. Two passes have shipped that bug:
/// `MATCH (p:Person) RETURN * ORDER BY p.age DESC LIMIT 2` answered
/// `[{'*': 1}, {'*': 1}]` while the same query without the LIMIT answered the
/// rows, and `MATCH (n:N) RETURN *, count(*) AS c` fused to a column called
/// `*` after the mixed-`*` expansion started producing real columns
/// everywhere else. The unoptimised plan is right in both, so the
/// differential corpus is the detector and carries both shapes.
///
/// Gates that classify items one at a time need this explicitly: a bare
/// `Star` is not an aggregate, so a check shaped "not an aggregate, therefore
/// a group key" accepts it.
pub(super) fn projection_has_wildcard(
    items: &[crate::graph::languages::cypher::ast::ReturnItem],
) -> bool {
    items.iter().any(|item| {
        matches!(
            item.expression,
            crate::graph::languages::cypher::ast::Expression::Star
        )
    })
}

/// Whether `pattern` is one node whose inline map holds only values the
/// node scan can test as they stand. An expression value (`{id: 19 + 1}`) is
/// resolved per row by the unfused path; the node scan would compare against
/// the unevaluated expression and match nothing. Outside a valid-time guard
/// `fold_constant_inline_maps` has already folded every constant one.
pub(super) fn is_single_scannable_node(
    clause: &crate::graph::languages::cypher::ast::MatchClause,
) -> bool {
    use crate::graph::core::pattern_matching::{PatternElement, PropertyMatcher};
    let [pattern] = clause.patterns.as_slice() else {
        return false;
    };
    let [PatternElement::Node(node)] = pattern.elements.as_slice() else {
        return false;
    };
    clause.path_assignments.is_empty()
        && !node.properties.as_ref().is_some_and(|props| {
            props
                .values()
                .any(|m| matches!(m, PropertyMatcher::EqualsExpr(_)))
        })
}

/// Finer multi-label fusion gate, shared by the fusions whose executors
/// filter typed nodes via `binary_search` on the primary `type_indices`
/// slice (edge aggregates) or build an R-tree from it (spatial join), and
/// which drop `extra_labels` from the pattern. Such an executor is blind to
/// secondary-labelled nodes, so a pattern is unsafe to fuse iff it carries
/// extra labels, or names a type that also exists as a secondary label —
/// every other pattern on a multi-label graph fuses with full correctness.
/// This replaces the global `has_secondary_labels` bail, which cost 71x /
/// 33x (aggregate / spatial, measured) on every such query the moment one
/// label existed anywhere in the graph.
pub(super) fn multi_label_fuse_unsafe(
    graph: &crate::graph::schema::DirGraph,
    np: &crate::graph::core::pattern_matching::NodePattern,
) -> bool {
    if !graph.has_secondary_labels {
        return false;
    }
    if np.multi_label_constrained() {
        return true;
    }
    np.node_type.as_deref().is_some_and(|node_type| {
        graph
            .secondary_label_index
            .contains_key(&crate::graph::schema::InternedKey::from_str(node_type))
    })
}

/// Whether `pat` has an untyped `{id: V}` / `{id: $p}` node that is not a
/// grouping key of the projection (`group_vars`: the variables its
/// non-aggregate items read).
///
/// The aggregate operators scan the *group* endpoint — every node when it is
/// untyped — and test the other end per candidate, so an id anchor on the far
/// end costs a whole-graph scan (65-71 ms where the matcher's id seed answers
/// in 0.1 ms) and counts every same-id node of a type, where the matcher
/// resolves one node per type (the duplicate-id contract). Such a pattern is
/// left to the matcher. An anchor that is itself the group key stays fused:
/// its group set comes from the matcher's own id seed.
pub(super) fn has_ungrouped_id_anchor(
    pat: &crate::graph::core::pattern_matching::Pattern,
    group_vars: &[&str],
) -> bool {
    use crate::graph::core::pattern_matching::{PatternElement, PropertyMatcher};
    pat.elements.iter().any(|element| {
        let PatternElement::Node(np) = element else {
            return false;
        };
        let anchored = np.node_type.is_none()
            && np.properties.as_ref().is_some_and(|props| {
                matches!(
                    props.get("id"),
                    Some(PropertyMatcher::Equals(_) | PropertyMatcher::EqualsParam(_))
                )
            });
        anchored
            && !np
                .variable
                .as_deref()
                .is_some_and(|variable| group_vars.contains(&variable))
    })
}

/// The variables the non-aggregate items of a projection read.
pub(super) fn grouping_variables(
    items: &[crate::graph::languages::cypher::ast::ReturnItem],
) -> Vec<&str> {
    use crate::graph::languages::cypher::ast::{is_aggregate_expression, Expression};
    items
        .iter()
        .filter(|item| !is_aggregate_expression(&item.expression))
        .filter_map(|item| match &item.expression {
            Expression::Variable(variable) | Expression::PropertyAccess { variable, .. } => {
                Some(variable.as_str())
            }
            _ => None,
        })
        .collect()
}
