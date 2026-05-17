//! `CALL refresh_stats() YIELD src_type, edge_type, tgt_type, count`
//!
//! Operator-callable recomputation of the label-pair edge-count
//! cardinality cache (0.9.35). Forces a fresh O(E) walk of all edges
//! and yields one row per `(src_type, edge_type, tgt_type)` triple
//! with its current count. Useful after bulk loads that bypass the
//! mutation paths the cache invalidator listens on, and as a manual
//! "what does the planner think the schema looks like right now?"
//! diagnostic.
//!
//! Pattern follows `affected_tests` (0.9.34) — same yield validation,
//! same per-column-optional model, same dispatch shape from
//! `call_clause.rs`.
//!
//! @procedure: refresh_stats

use std::collections::HashMap;

use super::super::ast::YieldItem;
use super::super::result::ResultRow;
use crate::datatypes::values::Value;
use crate::graph::dir_graph::DirGraph;

const PROC: &str = "refresh_stats";

pub(super) fn execute_refresh_stats(
    graph: &DirGraph,
    _params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    // Each YIELD column is optional individually — caller can ask for
    // just `count` (totals view), just the triple, or both.
    let src_var = yield_alias(yield_items, "src_type");
    let edge_var = yield_alias(yield_items, "edge_type");
    let tgt_var = yield_alias(yield_items, "tgt_type");
    let count_var = yield_alias(yield_items, "count");
    if src_var.is_none() && edge_var.is_none() && tgt_var.is_none() && count_var.is_none() {
        return Err(format!(
            "CALL {PROC}: must YIELD at least one of \
             'src_type', 'edge_type', 'tgt_type', 'count'."
        ));
    }

    // Force fresh recompute: invalidate then read. The invalidate +
    // lazy-recompute combo guarantees the next read walks all edges,
    // even if the cache was warm from a prior call.
    graph.invalidate_edge_type_counts_cache();
    let mut triples = graph.get_or_compute_type_connectivity();

    // Sort for deterministic output — agents calling this expect
    // stable row order so they can diff between calls.
    triples.sort_by(|a, b| {
        a.src
            .cmp(&b.src)
            .then(a.conn.cmp(&b.conn))
            .then(a.tgt.cmp(&b.tgt))
    });

    let mut rows = Vec::with_capacity(triples.len());
    for t in triples {
        let mut row = ResultRow::new();
        if let Some(name) = &src_var {
            row.projected.insert(name.clone(), Value::String(t.src));
        }
        if let Some(name) = &edge_var {
            row.projected.insert(name.clone(), Value::String(t.conn));
        }
        if let Some(name) = &tgt_var {
            row.projected.insert(name.clone(), Value::String(t.tgt));
        }
        if let Some(name) = &count_var {
            row.projected
                .insert(name.clone(), Value::Int64(t.count as i64));
        }
        rows.push(row);
    }
    Ok(rows)
}

/// Return the alias the caller gave a particular YIELD column, or `None`
/// if they didn't ask for it. Unknown YIELD names are rejected upstream
/// by the executor's `valid_yields` check, so we don't need to validate
/// names here.
fn yield_alias(yield_items: &[YieldItem], expected: &str) -> Option<String> {
    yield_items
        .iter()
        .find(|y| y.name == expected)
        .map(|item| item.alias.clone().unwrap_or_else(|| expected.to_string()))
}
