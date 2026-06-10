//! Cypher executor — `CALL { ... }` subquery execution.
//!
//! Phase 3 ships the **uncorrelated** path (`import.is_empty()`): the
//! body runs exactly once via a fresh sub-executor over the same graph,
//! and its result rows are cartesian-producted with the outer row stream
//! (§1.1 of `dev-documentation/design/call-subqueries.md`). The body sees
//! NO outer variables (§1.2 rule 1 — a fresh, empty executor scope); only
//! the body's terminal `RETURN` columns flow back into the outer scope.
//!
//! Phase 4 ships the **correlated** path (`!import.is_empty()`, Strategy
//! B1 / §4): the body is planned ONCE, then executed once per outer row
//! against a seed carrying ONLY the imported variables — preserving each
//! import's binding kind (node → node binding, edge → edge binding,
//! projected value → projected). The subquery's result rows are
//! inner-joined back to *that* outer row; zero rows drops the outer row
//! (§1.3), an aggregating body always returns one row (count = 0) so the
//! outer row survives.

use super::*;
use crate::datatypes::values::Value;

impl<'a> CypherExecutor<'a> {
    /// Execute a `CALL { ... }` subquery clause.
    ///
    /// Dispatches on correlation: an empty `import` is the uncorrelated
    /// case (run-once + cartesian); a non-empty `import` is correlated
    /// (per-row inner join over the imported variables).
    pub(super) fn execute_call_subquery(
        &self,
        import: &[String],
        body: &CypherQuery,
        result_set: ResultSet,
        declared: &std::collections::HashSet<String>,
    ) -> Result<ResultSet, String> {
        self.check_deadline()?;

        if !import.is_empty() {
            return self.execute_correlated_call_subquery(import, body, result_set, declared);
        }

        self.execute_uncorrelated_call_subquery(body, result_set)
    }

    /// Correlated `CALL { WITH … }`: run the planned-once body per outer
    /// row, seeded with only the imported variables (§1.2 rule 1), and
    /// inner-join the sub-results back to each driving outer row (§1.1 /
    /// §1.3).
    fn execute_correlated_call_subquery(
        &self,
        import: &[String],
        body: &CypherQuery,
        result_set: ResultSet,
        declared: &std::collections::HashSet<String>,
    ) -> Result<ResultSet, String> {
        // Import validation (deferred from Phase 2 — needs the outer scope,
        // available only now). Every imported name must be *declared* by a
        // clause preceding the CALL — NOT merely "present as a binding on
        // the first row". An upstream OPTIONAL MATCH that missed leaves its
        // variable declared-but-absent (null) on that row; the engine
        // represents the miss as the binding being absent from the row, so
        // probing a row can't tell "never declared" (typo → error) from
        // "declared upstream, null here" (must seed null per NULL-import
        // semantics, §1.3). Static declaredness — computed from the
        // preceding clauses at the dispatch site — is the correct oracle.
        //
        // This validation runs before the empty-rows short-circuit so a
        // typo'd import is reported even when the outer stream is empty.
        for name in import {
            if !declared.contains(name) {
                return Err(format!(
                    "CALL {{ }} subquery imports variable `{name}` via its leading WITH, but \
                     `{name}` is not bound in the outer scope at the CALL; import only \
                     variables introduced by an earlier MATCH / WITH / UNWIND"
                ));
            }
        }

        let outer_rows = result_set.rows;

        // No outer rows → nothing to drive the subquery. Carry columns
        // forward so a later RETURN still type-checks; the body never runs.
        if outer_rows.is_empty() {
            return Ok(ResultSet {
                rows: Vec::new(),
                columns: result_set.columns,
                lazy_return_items: None,
            });
        }

        // Plan the body ONCE outside the row loop (§3: never re-plan per
        // row). The same optimized AST drives every per-row execution.
        //
        // Correlated subtlety (Phase-4 boundary with Phase-5): the
        // graph-global aggregate/count-collapse passes fold a pattern such
        // as `MATCH (p)-[:KNOWS]->(f) RETURN count(f)` into a seed-ignoring
        // `FusedCountTypedEdge :KNOWS` (the *total* KNOWS count) because,
        // optimised in isolation, the body has no upstream clause binding
        // `p`. With `p` seeded per row, those fused operators ignore the
        // seed and return the global count for every outer row. Until
        // Phase 5 makes the fusion passes import-aware, disable exactly the
        // seed-ignoring collapse passes whenever a body pattern references
        // an imported variable — leaving a plain `Match`/`Return` that
        // anchors on the seeded binding (CSR adjacency, §3.2). Bodies that
        // never reference an import keep the full pipeline (their global
        // aggregate is genuinely correct — same value for every outer row).
        // Imported names used as a MATCH/OPTIONAL-MATCH pattern anchor in
        // the body. Drives two things: (a) disable seed-ignoring fusion
        // when non-empty, and (b) per-row NULL-anchor detection below.
        let anchor_imports = import_pattern_anchors(body, import);

        let mut planned = body.clone();
        if anchor_imports.is_empty() {
            crate::graph::languages::cypher::planner::optimize(
                &mut planned,
                self.graph,
                self.params,
            );
        } else {
            crate::graph::languages::cypher::planner::optimize_with_disabled(
                &mut planned,
                self.graph,
                self.params,
                seed_ignoring_fusion_passes(),
            );
        }

        // One sub-executor, reused across every outer row. It holds only
        // graph/params refs + fresh per-query caches (regex/spatial), so
        // reuse lets those caches warm across rows instead of being thrown
        // away per row. The deadline is inherited so a long correlated CALL
        // honours the outer timeout.
        let sub = CypherExecutor::with_params(self.graph, self.params, self.deadline)
            .with_streaming(self.streaming);

        // Run the body once for the first outer row to learn the subquery's
        // RETURN columns, then check those columns for an outer-scope
        // collision (§1.2 rule 4) — including a re-returned imported name.
        let mut combined_rows: Vec<ResultRow> = Vec::new();
        let mut sub_columns: Option<Vec<String>> = None;

        for outer_row in outer_rows.into_iter() {
            // Deadline check inside the per-row loop — a 100k-outer-row
            // correlated CALL must remain cancellable.
            self.check_deadline()?;

            // NULL-anchor handling (§1.3): if an imported variable that the
            // body uses as a pattern anchor is NULL on this outer row (e.g.
            // an unmatched upstream OPTIONAL MATCH), every anchored match
            // produces no rows. Seed the body with an EMPTY pipeline (zero
            // rows) rather than a one-row null binding: a non-aggregating
            // body then yields zero rows (outer row drops), while an
            // aggregating body still yields exactly one row (e.g.
            // `count() = 0`, outer row survives) — matching Neo4j. A NULL
            // scalar import that is NOT a pattern anchor stays in the seed
            // as projected-null (the body's expressions see null).
            let seed = self.seed_row_from_imports(&outer_row, import, &anchor_imports);
            let seed_set = ResultSet {
                rows: vec![seed],
                columns: Vec::new(),
                lazy_return_items: None,
            };
            // The body is optimized but NOT lazy-marked (`mark_lazy_eligibility`
            // runs only on the top-level query, never on a subquery body), so
            // `finalize_result` yields eager `Vec<Vec<Value>>` rows here.
            let body_set = sub.execute_clauses(&planned, seed_set)?;
            let body_result = sub.finalize_result(body_set)?;

            // First row establishes + validates the subquery's columns.
            if sub_columns.is_none() {
                for col in &body_result.columns {
                    let collides = outer_row.node_bindings.contains_key(col)
                        || outer_row.edge_bindings.contains_key(col)
                        || outer_row.path_bindings.contains_key(col)
                        || outer_row.projected.contains_key(col);
                    if collides {
                        return Err(format!(
                            "CALL {{ }} subquery returns a column `{col}` that already exists in \
                             the outer scope; rename the subquery's RETURN alias (re-returning an \
                             imported variable under the same name is a collision in Neo4j)"
                        ));
                    }
                }
                sub_columns = Some(body_result.columns.clone());
            }
            let cols = sub_columns.as_deref().unwrap();

            // Inner join: zero sub-rows drops the outer row (§1.3). For the
            // last sub-row reuse (move) the outer row; clone for the rest —
            // mirrors the uncorrelated cartesian path's move-on-last.
            let s = body_result.rows.len();
            if s == 0 {
                continue;
            }
            for sub_row in &body_result.rows[..s - 1] {
                let mut row = outer_row.clone();
                splice_subquery_columns(&mut row, sub_row, cols);
                combined_rows.push(row);
            }
            let mut row = outer_row;
            splice_subquery_columns(&mut row, &body_result.rows[s - 1], cols);
            combined_rows.push(row);
        }

        // Carry outer columns forward + append the subquery's RETURN
        // columns so a later RETURN can reference them. When every outer
        // row dropped (sub_columns never set), fall back to the outer
        // columns only.
        let mut columns = result_set.columns;
        if let Some(cols) = sub_columns {
            for col in cols {
                if !columns.contains(&col) {
                    columns.push(col);
                }
            }
        }

        Ok(ResultSet {
            rows: combined_rows,
            columns,
            lazy_return_items: None,
        })
    }

    /// Build a fresh seed row carrying ONLY the imported variables (§1.2
    /// rule 1), preserving each import's binding kind so the body can use
    /// it correctly: a node import seeds a node binding (so `MATCH (p)-[]->`
    /// expands from it via CSR adjacency, §3.2), an edge seeds an edge
    /// binding, a path seeds a path binding, and a projected scalar seeds a
    /// projected value (a NULL non-anchor scalar flows through as
    /// projected-null so the body's expressions see null).
    ///
    /// A node imported as a node binding is preferred over the same name
    /// also living in `projected`; the kind that anchors pattern matching
    /// wins.
    ///
    /// **NULL / absent pattern-anchor (§1.3).** An imported name that is
    /// NULL *or* entirely absent on the outer row (an upstream OPTIONAL
    /// MATCH that missed leaves its variable absent from the row's
    /// bindings — the engine's representation of a null) is decided per
    /// row:
    ///
    /// - If the body uses it as a pattern anchor (`anchor_imports`), seed
    ///   a node binding to an out-of-range *sentinel* `NodeIndex` (one past
    ///   the graph's node count). The body's anchored expansion walks that
    ///   node's (empty) adjacency and finds nothing — a non-aggregating
    ///   body yields zero rows (the outer row drops) while an aggregating
    ///   body yields the empty-aggregate value (`count() = 0`, the outer
    ///   row survives). This reproduces Neo4j's "pattern match against a
    ///   NULL node produces no rows" without a real null-node type:
    ///   `node_weight(sentinel)` returns `None`, so any property read on it
    ///   is NULL too.
    /// - Otherwise seed projected-null so the body's expressions see null.
    ///
    /// The kind is decided **per row** — `x` may be a real node on row 1
    /// (seeded as a node binding) and null on row 2 (sentinel / projected-
    /// null), since the body is planned once but seeded once per row.
    fn seed_row_from_imports(
        &self,
        outer_row: &ResultRow,
        import: &[String],
        anchor_imports: &[String],
    ) -> ResultRow {
        let mut seed = ResultRow::with_capacity(import.len(), 0, 0);
        for name in import {
            if let Some(idx) = outer_row.node_bindings.get(name) {
                seed.node_bindings.insert(name.clone(), *idx);
            } else if let Some(edge) = outer_row.edge_bindings.get(name) {
                seed.edge_bindings.insert(name.clone(), *edge);
            } else if let Some(path) = outer_row.path_bindings.get(name) {
                seed.path_bindings.insert(name.clone(), path.clone());
            } else {
                // Either a projected scalar, a projected NULL, or entirely
                // absent (OPTIONAL MATCH miss — declared but unbound on this
                // row). A non-null projected scalar flows through unchanged;
                // null/absent routes through the NULL-import decision.
                match outer_row.projected.get(name) {
                    Some(val) if !matches!(val, Value::Null) => {
                        seed.projected.insert(name.clone(), val.clone());
                    }
                    _ => self.seed_null_import(&mut seed, name, anchor_imports),
                }
            }
        }
        seed
    }

    /// Seed a single NULL/absent import into `seed`, deciding its kind: a
    /// sentinel node binding when the body anchors a pattern on it (so the
    /// anchored match yields nothing), else projected-null. Factored out so
    /// the projected-null and absent-binding paths share one decision.
    fn seed_null_import(&self, seed: &mut ResultRow, name: &str, anchor_imports: &[String]) {
        if anchor_imports.iter().any(|a| a == name) {
            let sentinel = petgraph::graph::NodeIndex::new(self.graph.graph.node_count());
            seed.node_bindings.insert(name.to_string(), sentinel);
        } else {
            seed.projected.insert(name.to_string(), Value::Null);
        }
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

/// The subset of `import` names that appear as a `MATCH` / `OPTIONAL
/// MATCH` pattern element in the correlated body (so the body anchors on
/// the seeded binding). Non-empty ⇒ the seed-ignoring fusion passes must
/// be disabled, and a NULL value for any of these names empties the
/// per-row pipeline (§1.3).
///
/// Only the body's own clauses are scanned — a nested `CALL { }` re-binds
/// its own imports from its own seed, so its patterns are not this body's
/// concern.
fn import_pattern_anchors(body: &CypherQuery, import: &[String]) -> Vec<String> {
    let mut anchors: Vec<String> = Vec::new();
    for clause in &body.clauses {
        let patterns = match clause {
            Clause::Match(m) | Clause::OptionalMatch(m) => &m.patterns,
            _ => continue,
        };
        for pattern in patterns {
            for elem in &pattern.elements {
                let var = match elem {
                    PatternElement::Node(np) => np.variable.as_ref(),
                    PatternElement::Edge(ep) => ep.variable.as_ref(),
                };
                if let Some(v) = var {
                    if import.iter().any(|name| name == v) && !anchors.iter().any(|a| a == v) {
                        anchors.push(v.clone());
                    }
                }
            }
        }
    }
    anchors
}

/// The optimizer passes that emit a graph-global / plan-time-anchored
/// operator (`FusedCount*`, `FusedMatch*Aggregate`, `FusedNodeScan*`)
/// which IGNORES the incoming seed row. Disabled when a correlated body
/// anchors on an imported variable, so the body runs as a plain
/// `Match`/`Return` that honours the seed. Process-lifetime set —
/// built once.
fn seed_ignoring_fusion_passes() -> &'static std::collections::HashSet<String> {
    static PASSES: std::sync::OnceLock<std::collections::HashSet<String>> =
        std::sync::OnceLock::new();
    PASSES.get_or_init(|| {
        [
            "fuse_anchored_edge_count",
            "fuse_count_short_circuits",
            "fuse_optional_match_aggregate",
            "fuse_match_return_aggregate",
            "fuse_match_with_aggregate",
            "fuse_match_with_aggregate_top_k",
            "fuse_node_scan_aggregate",
            "fuse_node_scan_top_k",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    })
}
