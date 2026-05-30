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
