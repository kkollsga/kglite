//! Streaming-pipeline recognizer + assembler.
//!
//! Called by the driver in [`super::super::CypherExecutor::execute`]
//! before each materialized-path single-clause dispatch. If the
//! recognizer matches a clause run the streaming path can absorb, it
//! consumes those clauses, runs the streaming pipeline, and returns the
//! resulting `ResultSet` plus the number of clauses absorbed. The
//! driver then advances its index by that count and continues.
//!
//! # Phase 1 shapes
//! - **A.** `RETURN/WITH(group, agg)` whose RETURN/WITH consists of
//!   pure variable / property-access group keys plus inline-able
//!   aggregates (`count`, `sum`, `avg`, `min`, `max`, with optional
//!   `DISTINCT`). Streaming aggregate replaces materialize-then-bucket.
//! - **B.** A streaming-aggregate clause optionally followed by
//!   `ORDER BY <expr> [ASC|DESC] LIMIT k` (no DISTINCT, no HAVING,
//!   positive literal LIMIT). Heap top-K replaces full sort + truncate.
//!
//! Anything else returns `None` and the materialized executor handles
//! the clause as before.

use super::super::super::ast::{
    is_aggregate_expression, Clause, Expression, LimitClause, OrderByClause, OrderItem,
    ReturnClause, WhereClause,
};
use super::super::super::result::ResultSet;
use super::super::CypherExecutor;
use super::{aggregate, heap_top_k, RowStream};
use crate::datatypes::values::Value;

/// Outcome of a recognition attempt.
pub(crate) struct StreamingRun {
    /// Number of clauses absorbed by the streaming path. The driver
    /// advances its loop index by this count.
    pub absorbed: usize,
    pub result: ResultSet,
}

/// What [`try_run_streaming`] returns to the driver. `Absorbed` means
/// the pipeline ran successfully; `Bailed` returns the input
/// `ResultSet` unchanged so the caller can pass it to the materialized
/// executor.
pub(crate) enum StreamingOutcome {
    Absorbed(StreamingRun),
    Bailed(ResultSet),
}

/// Try to recognize and run a streaming clause run. `clauses` is the
/// remaining clause slice starting from the next clause to execute;
/// `result_set` is the materialized prefix the streaming path consumes
/// as its source stream.
///
/// Returns `StreamingOutcome::Bailed(result_set)` when no shape
/// matches, so the caller never loses ownership of the input. Returns
/// `Err(_)` only when a recognized pipeline fails mid-execution.
pub(crate) fn try_run_streaming<'q>(
    executor: &'q CypherExecutor<'q>,
    clauses: &[Clause],
    result_set: ResultSet,
) -> Result<StreamingOutcome, String> {
    // The first absorbed clause must be either a WITH(group, agg) or
    // a RETURN(group, agg). Anything else: bail.
    if clauses.is_empty() {
        return Ok(StreamingOutcome::Bailed(result_set));
    }

    let (return_clause_owned, is_with, with_where) = match &clauses[0] {
        Clause::With(w) => {
            // WITH delegates to the same agg machinery as RETURN.
            let rc = ReturnClause {
                items: w.items.clone(),
                distinct: w.distinct,
                having: None,
                lazy_eligible: false,
                group_limit_hint: w.group_limit_hint,
            };
            (rc, true, w.where_clause.clone())
        }
        Clause::Return(rc) => (rc.clone(), false, None),
        _ => return Ok(StreamingOutcome::Bailed(result_set)),
    };

    // Must contain at least one aggregate item — otherwise the
    // materialized projection path is fine.
    let has_agg = return_clause_owned
        .items
        .iter()
        .any(|item| is_aggregate_expression(&item.expression));
    if !has_agg {
        return Ok(StreamingOutcome::Bailed(result_set));
    }

    // RETURN-side guards: streaming path bails on DISTINCT-on-RETURN,
    // HAVING, and lazy-eligible (lazy_eligible is set only when there
    // are no aggregates anyway, so this is belt-and-suspenders).
    if return_clause_owned.having.is_some() {
        return Ok(StreamingOutcome::Bailed(result_set));
    }

    // Try to compile the aggregate specs. If anything is unsupported
    // (collect/std/etc., arithmetic on aggregates, complex group keys),
    // bail.
    let (group_indices, agg_indices, specs) =
        match aggregate::try_compile_specs(&return_clause_owned) {
            Ok(t) => t,
            Err(_) => return Ok(StreamingOutcome::Bailed(result_set)),
        };

    // Look for an optional follow-up `ORDER BY → LIMIT` we can fuse via
    // heap top-K. Only fire when *both* clauses are present; an ORDER
    // BY without LIMIT still materializes everything, so the
    // materialized sort path is fine.
    let (order_items, limit, top_k_clauses) = match find_top_k(&clauses[1..]) {
        Some((items, n, count)) => (Some(items), Some(n), count),
        None => (None, None, 0),
    };

    // Build the streaming pipeline from the materialized upstream.
    let upstream = RowStream::from_result_set(result_set);
    let mut current = aggregate::apply(
        executor,
        upstream,
        &return_clause_owned,
        &group_indices,
        &agg_indices,
        &specs,
    )?;

    let mut absorbed = 1usize; // the WITH/RETURN clause

    if let (Some(items), Some(n)) = (order_items, limit) {
        current = heap_top_k::apply(executor, current, &items, n)?;
        absorbed += top_k_clauses;
    }

    let mut result = current.drain()?;

    // WITH ... WHERE: apply the post-projection WHERE on the
    // materialized result, mirroring `execute_with`.
    if is_with {
        if let Some(wc) = with_where {
            result = executor.execute_where(&wc, result)?;
        }
    }

    Ok(StreamingOutcome::Absorbed(StreamingRun {
        absorbed,
        result,
    }))
}

/// Pattern-match an `OrderBy → Limit` tail. Returns the order items, the
/// resolved K, and the number of clauses consumed (always 2 on success).
fn find_top_k(clauses: &[Clause]) -> Option<(Vec<OrderItem>, usize, usize)> {
    if clauses.len() < 2 {
        return None;
    }
    let order = match &clauses[0] {
        Clause::OrderBy(OrderByClause { items }) => items.clone(),
        _ => return None,
    };
    let limit_count = match &clauses[1] {
        Clause::Limit(LimitClause { count }) => count,
        _ => return None,
    };
    // Limit must be a positive literal integer for top-K. Param /
    // expression LIMITs require eval-with-row, which the streaming
    // path doesn't currently set up.
    let n = match limit_count {
        Expression::Literal(Value::Int64(n)) if *n >= 0 => *n as usize,
        _ => return None,
    };
    Some((order, n, 2))
}

// Ensure we don't accidentally drop `WhereClause` import — it's part of
// the WITH ... WHERE handling above.
#[allow(dead_code)]
fn _silence_where_clause(_: &WhereClause) {}
