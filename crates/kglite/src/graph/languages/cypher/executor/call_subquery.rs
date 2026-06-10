//! Cypher executor — `CALL { ... }` subquery execution.
//!
//! Phase 3 ships the **uncorrelated** path (`import.is_empty()`): the
//! body runs exactly once via a fresh sub-executor over the same graph,
//! and its result rows are cartesian-producted with the outer row stream
//! (§1.1 of `dev-documentation/design/call-subqueries.md`). The body sees
//! NO outer variables (§1.2 rule 1 — a fresh, empty executor scope); only
//! the body's terminal `RETURN` columns flow back into the outer scope.
//!
//! The correlated path (`!import.is_empty()`) is Phase 4 — it still
//! returns a clean not-yet-executable error here.

use super::*;
use crate::datatypes::values::Value;

impl<'a> CypherExecutor<'a> {
    /// Execute a `CALL { ... }` subquery clause.
    ///
    /// Dispatches on correlation: an empty `import` is the uncorrelated
    /// case (run-once + cartesian); a non-empty `import` is correlated
    /// (Phase 4) and returns a clean deferral error.
    pub(super) fn execute_call_subquery(
        &self,
        import: &[String],
        body: &CypherQuery,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        self.check_deadline()?;

        if !import.is_empty() {
            return Err(
                "correlated CALL { } subqueries (importing outer variables via a leading \
                 WITH) are not yet executable (planned for a future release); the \
                 uncorrelated form CALL { ... } that imports nothing already works"
                    .to_string(),
            );
        }

        self.execute_uncorrelated_call_subquery(body, result_set)
    }

    /// Uncorrelated `CALL { }`: run the body once, fan the outer rows out
    /// against every subquery row (cartesian product, §1.1).
    fn execute_uncorrelated_call_subquery(
        &self,
        body: &CypherQuery,
        result_set: ResultSet,
    ) -> Result<ResultSet, String> {
        // Run the body exactly once in a fresh executor scope seeded with
        // NO outer bindings (§1.2 rule 1). Reuse this executor's graph,
        // params, and deadline so the subquery honours the outer timeout.
        //
        // Phase 3 optimizes the body locally on first use; Phase 5 moves
        // this into `pass_optimize_nested_queries` so it happens once at
        // plan time instead of at execution time.
        // TODO(phase5): drop this local optimize; recurse the planner pass
        // into CallSubquery bodies and rely on the pre-optimized AST.
        let mut planned = body.clone();
        crate::graph::languages::cypher::planner::optimize(&mut planned, self.graph, self.params);

        let sub = CypherExecutor::with_params(self.graph, self.params, self.deadline)
            .with_streaming(self.streaming);
        let sub_result = sub.execute(&planned)?;

        // The body must terminate in RETURN (parser-enforced, §1.4), so a
        // lazy descriptor is never produced here — the body is not lazy-
        // marked. Defensive: if it somehow were, materialise eagerly is not
        // possible without the graph-side resolver, so treat the absence of
        // eager rows as zero rows. In practice `sub_result.rows` is populated.
        let sub_columns = sub_result.columns;
        let sub_rows = sub_result.rows;

        // §1.2 rule 4 — a subquery RETURN alias must not clash with a
        // variable already in the outer scope. For the uncorrelated case
        // the outer scope is whatever the preceding clauses bound; check
        // against the current result_set's columns and any per-row
        // bindings. We probe the first row (all rows share the same
        // binding key shape within a result set).
        if let Some(first) = result_set.rows.first() {
            for col in &sub_columns {
                let collides = first.node_bindings.contains_key(col)
                    || first.edge_bindings.contains_key(col)
                    || first.path_bindings.contains_key(col)
                    || first.projected.contains_key(col);
                if collides {
                    return Err(format!(
                        "CALL {{ }} subquery returns a column `{col}` that already exists in \
                         the outer scope; rename the subquery's RETURN alias (Neo4j errors on \
                         shadowing an outer variable)"
                    ));
                }
            }
        }

        // Cartesian product: every outer row × every subquery row. The
        // subquery's RETURN columns become new projected bindings on each
        // combined row (§1.1 / §1.2 rule 3 — only RETURN columns escape).
        let outer_rows = result_set.rows;
        let mut combined_rows: Vec<ResultRow> = Vec::new();

        if outer_rows.is_empty() {
            // Leading CALL { } (no preceding clause produced rows): the
            // executor has not seeded an empty row for a CallSubquery
            // first-clause, so the result is simply the S subquery rows.
            // R = 1 implicit empty outer row × S subquery rows = S rows.
            combined_rows.reserve(sub_rows.len());
            for sub_row in &sub_rows {
                combined_rows.push(subquery_row_to_result_row(sub_row, &sub_columns));
            }
        } else {
            // R × S. For each outer row we emit one combined row per
            // subquery row. To avoid an extra clone, the *last* subquery
            // pairing reuses (moves) the outer row instead of cloning it,
            // so we clone exactly (S-1) times per outer row rather than S.
            // When S == 0 the outer row is dropped entirely (cartesian with
            // an empty subquery result → zero rows, §1.3 / inner join).
            let s = sub_rows.len();
            combined_rows.reserve(outer_rows.len().saturating_mul(s));
            for outer_row in outer_rows {
                if s == 0 {
                    continue;
                }
                for sub_row in &sub_rows[..s - 1] {
                    let mut row = outer_row.clone();
                    splice_subquery_columns(&mut row, sub_row, &sub_columns);
                    combined_rows.push(row);
                }
                // Last subquery row: move the outer row in (no clone).
                let mut row = outer_row;
                splice_subquery_columns(&mut row, &sub_rows[s - 1], &sub_columns);
                combined_rows.push(row);
            }
        }

        // Carry forward outer columns + the subquery's RETURN columns. The
        // outer columns are only set once a RETURN/WITH ran upstream; for a
        // mid-pipeline CALL { } after a MATCH, `result_set.columns` may be
        // empty (columns get assigned by the terminal RETURN). We append the
        // subquery columns so a later RETURN can reference them.
        let mut columns = result_set.columns;
        for col in &sub_columns {
            if !columns.contains(col) {
                columns.push(col.clone());
            }
        }

        Ok(ResultSet {
            rows: combined_rows,
            columns,
            lazy_return_items: None,
        })
    }
}

/// Build a fresh `ResultRow` carrying only the subquery's RETURN columns
/// as projected values (used for the leading-CALL case where there is no
/// outer row to splice onto).
fn subquery_row_to_result_row(sub_row: &[Value], sub_columns: &[String]) -> ResultRow {
    let mut projected = Bindings::with_capacity(sub_columns.len());
    for (col, val) in sub_columns.iter().zip(sub_row.iter()) {
        projected.insert(col.clone(), val.clone());
    }
    ResultRow::from_projected(projected)
}

/// Splice the subquery's RETURN columns into an existing outer row's
/// projected bindings (the cartesian-pairing case).
fn splice_subquery_columns(row: &mut ResultRow, sub_row: &[Value], sub_columns: &[String]) {
    for (col, val) in sub_columns.iter().zip(sub_row.iter()) {
        row.projected.insert(col.clone(), val.clone());
    }
}
