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
mod count;
mod spatial;
mod topk;

pub(super) use aggregate::*;
pub(super) use count::*;
pub(super) use spatial::*;
pub(super) use topk::*;

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
