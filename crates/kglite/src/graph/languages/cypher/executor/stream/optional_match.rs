//! Streaming OPTIONAL MATCH.
//!
//! Lazy flat-map over the upstream stream: for each upstream row, run
//! the pattern once and yield either the expanded rows or a single
//! NULL-padded row when no expansion exists. Mirrors
//! [`super::super::CypherExecutor::execute_optional_match`] but never
//! materializes the cross product as a `Vec<ResultRow>`.
//!
//! Phase 1 doesn't use this for the leading-OPTIONAL-MATCH-with-empty-upstream
//! case — the streaming pipeline only enters when a regular MATCH is the
//! first absorbed clause, so the upstream is always non-empty here. The
//! materialized executor still owns first-clause-OPTIONAL semantics.

// The shape recognizer in `super::pipeline` does not yet emit the
// streaming OPTIONAL MATCH operator — that wiring lands later in Phase 1
// (or in Phase 2 once multi-MATCH absorption is in place). The
// implementation lives here in the meantime so subsequent commits can
// flip on the shape without re-introducing the operator.
#![allow(dead_code)]

use super::super::super::ast::MatchClause;
use super::super::super::result::ResultRow;
use super::super::CypherExecutor;
use super::RowStream;
use crate::datatypes::values::Value;
use crate::graph::core::pattern_matching::PatternElement;

/// State machine for streaming OPTIONAL MATCH. Holds the upstream
/// iterator plus a buffer of expanded rows for the row we're currently
/// fanning out from. When the buffer empties, we pull the next upstream
/// row and re-fill it.
struct StreamingOptionalMatch<'q> {
    executor: &'q CypherExecutor<'q>,
    clause: MatchClause,
    upstream: RowStream<'q>,
    /// Pending expansions for the current upstream row (already drained
    /// from the upstream iterator). Yielded one at a time.
    pending: Vec<ResultRow>,
    /// Set once we've observed the upstream iterator is exhausted —
    /// avoids one extra `next()` call per future yield.
    upstream_done: bool,
}

impl<'q> Iterator for StreamingOptionalMatch<'q> {
    type Item = Result<ResultRow, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(row) = self.pending.pop() {
                return Some(Ok(row));
            }
            if self.upstream_done {
                return None;
            }
            // Pull the next upstream row, expand, refill `pending`. We
            // push expansions in reverse so `pop()` yields them in
            // input-order.
            let row = match self.upstream.next() {
                Some(Ok(r)) => r,
                Some(Err(e)) => return Some(Err(e)),
                None => {
                    self.upstream_done = true;
                    return None;
                }
            };

            match self.executor.stream_expand_optional(&self.clause, &row) {
                Ok(mut expanded) => {
                    if expanded.is_empty() {
                        // No matches: keep the upstream row, NULL-pad
                        // any variables introduced by the pattern. Same
                        // shape as `execute_optional_match` line 221-223.
                        let mut keep = row;
                        null_pad_pattern_vars(&mut keep, &self.clause);
                        self.pending.push(keep);
                    } else {
                        // Reverse so `pop()` yields the first match first.
                        expanded.reverse();
                        self.pending = expanded;
                    }
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

/// Build the streaming OPTIONAL MATCH operator. Borrows `executor` for
/// the pipeline lifetime.
pub fn apply<'q>(
    executor: &'q CypherExecutor<'q>,
    upstream: RowStream<'q>,
    clause: MatchClause,
) -> RowStream<'q> {
    let columns = upstream.columns_owned();
    RowStream::new(
        StreamingOptionalMatch {
            executor,
            clause,
            upstream,
            pending: Vec::new(),
            upstream_done: false,
        },
        columns,
    )
}

/// NULL-pad pattern-introduced variables on `row`. Matches the
/// "OPTIONAL MATCH yields no rows -> keep upstream row with NULL
/// projections for pattern vars" semantics in `execute_optional_match`.
fn null_pad_pattern_vars(row: &mut ResultRow, clause: &MatchClause) {
    for pattern in &clause.patterns {
        for elem in &pattern.elements {
            match elem {
                PatternElement::Node(np) => {
                    if let Some(ref var) = np.variable {
                        if !row.node_bindings.contains_key(var) && !row.projected.contains_key(var)
                        {
                            row.projected.insert(var.clone(), Value::Null);
                        }
                    }
                }
                PatternElement::Edge(ep) => {
                    if let Some(ref var) = ep.variable {
                        if !row.edge_bindings.contains_key(var) && !row.projected.contains_key(var)
                        {
                            row.projected.insert(var.clone(), Value::Null);
                        }
                    }
                }
            }
        }
    }
}
