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

use super::super::super::ast::{NullsPlacement, OrderItem};
use super::super::super::result::ResultRow;
use super::super::CypherExecutor;
use super::RowStream;
use crate::datatypes::values::Value;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Per-key sort spec: ASC/DESC + nulls placement. Pre-resolved at
/// stream-build time so the inner loop is a pure `&[SortSpec]` walk.
/// 0.9.0 §2.
#[derive(Clone, Copy)]
struct SortSpec {
    ascending: bool,
    nulls: NullsPlacement,
}

/// A sort-key tuple paired with the row it belongs to. The `Ord` impl
/// orders entries so that the heap's *root* is the entry the operator
/// wants to evict first when the heap grows past K.
///
/// `specs[i].ascending == true` for ascending (smaller is better, evict
/// largest), `false` for descending (larger is better, evict smallest).
struct HeapEntry {
    sort_keys: Vec<Value>,
    specs: std::sync::Arc<[SortSpec]>,
    row: ResultRow,
}

impl HeapEntry {
    /// Compare two entries respecting the ASC/DESC direction and NULLS
    /// placement of each key. Returns `Ordering` from the perspective
    /// of "which entry is better according to the user's ORDER BY" —
    /// better entries compare *Less* so that `BinaryHeap` (max-heap)
    /// keeps the worst at the root.
    fn cmp_better_first(&self, other: &Self) -> Ordering {
        for (i, spec) in self.specs.iter().enumerate() {
            let a = self.sort_keys.get(i).unwrap_or(&Value::Null);
            let b = other.sort_keys.get(i).unwrap_or(&Value::Null);

            // Explicit NULLS placement. NULL First → NULL is "better"
            // (sorts earlier in result, evicted last). NULL Last → NULL
            // is "worse".
            let a_null = matches!(a, Value::Null);
            let b_null = matches!(b, Value::Null);
            match (a_null, b_null) {
                (true, true) => continue,
                (true, false) => {
                    return match spec.nulls {
                        NullsPlacement::First => Ordering::Less,
                        NullsPlacement::Last => Ordering::Greater,
                    };
                }
                (false, true) => {
                    return match spec.nulls {
                        NullsPlacement::First => Ordering::Greater,
                        NullsPlacement::Last => Ordering::Less,
                    };
                }
                (false, false) => {}
            }

            let raw =
                crate::graph::core::filtering::compare_values(a, b).unwrap_or(Ordering::Equal);
            // `raw == Less` means a is smaller. For ascending order,
            // smaller is "better" (closer to top of final result).
            // We want better entries to compare `Less` so the
            // max-heap root holds the worst entry.
            let oriented = if spec.ascending { raw } else { raw.reverse() };
            if oriented != Ordering::Equal {
                return oriented;
            }
        }
        Ordering::Equal
    }
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp_better_first(other) == Ordering::Equal
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap, so root holds the entry with the
        // largest `Ord`. We want root to hold the *worst* entry (so
        // eviction on overflow drops the worst). `cmp_better_first`
        // returns `Less` when self is better — passing it through
        // unchanged means worse self ranks Greater, which BinaryHeap
        // surfaces at root.
        self.cmp_better_first(other)
    }
}

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

    let specs: std::sync::Arc<[SortSpec]> = order_items
        .iter()
        .map(|item| SortSpec {
            ascending: item.ascending,
            nulls: item.effective_nulls(),
        })
        .collect::<Vec<_>>()
        .into();

    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(limit + 1);

    for row in upstream {
        let row = row?;

        let sort_keys: Vec<Value> = folded_exprs
            .iter()
            .map(|expr| {
                executor
                    .evaluate_expression(expr, &row)
                    .unwrap_or(Value::Null)
            })
            .collect();

        let entry = HeapEntry {
            sort_keys,
            specs: specs.clone(),
            row,
        };

        if heap.len() < limit {
            heap.push(entry);
        } else {
            // The heap's root is the *worst* entry currently kept. If
            // the candidate is better than the root, swap them.
            //
            // `entry < root` (in the `Ord` sense) means entry's
            // `cmp_better_first` is `Less` -> entry IS better. Push
            // entry, pop the now-worst.
            if let Some(root) = heap.peek() {
                if entry.cmp(root) == Ordering::Less {
                    heap.push(entry);
                    heap.pop();
                }
            }
        }
    }

    // Drain heap into a vector sorted with the best entry first.
    // `into_sorted_vec` returns ascending in `Ord` terms — and our
    // `Ord` ranks better entries `Less`, so ascending == best-first.
    let entries = heap.into_sorted_vec();
    let rows: Vec<ResultRow> = entries.into_iter().map(|e| e.row).collect();

    Ok(RowStream::from_vec(rows, columns))
}
