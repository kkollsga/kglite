//! Heap-pruned top-K operator.
//!
//! Replaces the full sort + truncate path
//! ([`super::super::CypherExecutor::execute_order_by`] +
//! [`super::super::CypherExecutor::execute_limit`]) for streaming
//! pipelines that end in `ORDER BY <expr> [ASC|DESC] LIMIT k`.
//!
//! The operator maintains a `BinaryHeap` of capacity K. For each upstream
//! row, it evaluates the sort-key expressions, compares against the
//! heap's current worst element, and either pushes or discards. At the
//! end it drains the heap in sorted order. Wall-clock complexity is
//! O(n log k) instead of O(n log n), and peak memory is O(k) result-row
//! references — which matters when the upstream cardinality is in the
//! tens of millions but K is a few dozen.

use super::super::super::ast::OrderItem;
use super::super::super::result::ResultRow;
use super::super::ordering::{SortSpec, TopKCollector};
use super::super::CypherExecutor;
use super::RowStream;
use crate::datatypes::values::Value;

/// Consume `upstream` and emit at most `limit` rows in the order
/// requested by `order_items`. Eager: drains the upstream fully before
/// emitting the result. Pipeline-wise this is fine because top-K is
/// always followed by a downstream consumer that sees only K rows.
///
/// `executor` is borrowed for expression evaluation. The folded sort
/// expressions are evaluated against each row once.
pub fn apply<'q>(
    executor: &'q CypherExecutor<'q>,
    upstream: RowStream<'q>,
    order_items: &[OrderItem],
    limit: usize,
) -> Result<RowStream<'q>, String> {
    let columns = upstream.columns_owned();

    if limit == 0 {
        // Drain and discard upstream so any propagated errors still
        // surface, but emit zero rows. Callers expect this to behave
        // like a pure post-pipeline LIMIT 0.
        for row in upstream {
            row?;
        }
        return Ok(RowStream::from_vec(Vec::new(), columns));
    }

    // Pre-fold sort-key expressions once. Constant folding turns
    // `now() + p.year` into a partially-evaluated form that
    // `evaluate_expression` resolves cheaply per row.
    let folded_exprs: Vec<_> = order_items
        .iter()
        .map(|item| executor.fold_constants_expr(&item.expression))
        .collect();

    let specs: Vec<SortSpec> = order_items.iter().map(SortSpec::from_order_item).collect();

    let mut collector: TopKCollector<ResultRow> = TopKCollector::new(specs, limit);

    for (seq, row) in upstream.enumerate() {
        let row = row?;

        let sort_keys: Vec<Value> = folded_exprs
            .iter()
            .map(|expr| {
                executor
                    .evaluate_expression(expr, &row)
                    .unwrap_or(Value::Null)
            })
            .collect();

        if collector.accepts(&sort_keys, seq) {
            collector.push(sort_keys, seq, row);
        }
    }

    // Best entry first, ties in upstream order.
    let rows: Vec<ResultRow> = collector
        .into_sorted()
        .into_iter()
        .map(|(_, row)| row)
        .collect();

    Ok(RowStream::from_vec(rows, columns))
}
