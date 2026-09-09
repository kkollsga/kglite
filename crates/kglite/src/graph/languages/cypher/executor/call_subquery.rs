//! Cypher executor — `CALL { ... }` subquery execution.
//!
//! Every read subquery body is planned once and executed once per incoming
//! row. The seed carries only the selected imports — preserving each
//! import's binding kind (node → node binding, edge → edge binding,
//! projected value → projected). The subquery's result rows are
//! inner-joined back to *that* outer row; zero rows drops the outer row
//! (§1.3), an aggregating body always returns one row (count = 0) so the
//! outer row survives.
//!
//! Body OPTIMIZATION belongs to the planner: this module does not
//! optimize bodies. `planner::pass_optimize_nested_queries` recurses into
//! every `CALL { }` body once at plan time (import-aware: it disables the
//! seed-ignoring fusion passes for correlated bodies that anchor on an
//! imported variable) and the executor runs the body exactly as planned.
//! The executor still re-derives arm-local pattern anchors (re-exported from
//! the planner) for per-row NULL-anchor detection (§1.3).

use super::*;
use crate::datatypes::values::Value;

impl<'a> CypherExecutor<'a> {
    /// Execute a `CALL { ... }` subquery clause.
    ///
    pub(super) fn execute_call_subquery(
        &self,
        import: &CallSubqueryImport,
        body: &CypherQuery,
        result_set: ResultSet,
        declared: &std::collections::HashSet<String>,
    ) -> Result<ResultSet, String> {
        self.check_deadline()?;
        let mut imports = match import {
            CallSubqueryImport::Legacy(names) | CallSubqueryImport::Named(names) => names.clone(),
            CallSubqueryImport::All => declared.iter().cloned().collect(),
            CallSubqueryImport::Empty => Vec::new(),
        };
        imports.sort();

        // Validate before inspecting runtime rows: an emptied outer stream
        // must not hide a typo or an output-name collision.
        for name in &imports {
            if !declared.contains(name) {
                return Err(format!(
                    "CALL {{ }} subquery imports variable `{name}`, but `{name}` is not bound in \
                     the outer scope at the CALL"
                ));
            }
        }
        let globally_scoped = matches!(
            import,
            CallSubqueryImport::Named(_) | CallSubqueryImport::All
        );
        let output_columns = subquery_output_columns(body, &imports, globally_scoped)?;
        for col in &output_columns {
            if declared.contains(col) {
                return Err(format!(
                    "CALL {{ }} subquery returns a column `{col}` that already exists in the \
                     outer scope; rename the subquery's RETURN alias"
                ));
            }
        }

        self.execute_per_row_call_subquery(
            &imports,
            globally_scoped,
            body,
            &output_columns,
            result_set,
            declared,
        )
    }

    /// Correlated `CALL { WITH … }`: run the planned-once body per outer
    /// row, seeded with only the imported variables (§1.2 rule 1), and
    /// inner-join the sub-results back to each driving outer row (§1.1 /
    /// §1.3).
    fn execute_per_row_call_subquery(
        &self,
        import: &[String],
        globally_scoped: bool,
        body: &CypherQuery,
        output_columns: &[String],
        result_set: ResultSet,
        declared: &std::collections::HashSet<String>,
    ) -> Result<ResultSet, String> {
        let outer_rows = result_set.rows;

        // No outer rows → nothing drives the body, but its statically declared
        // RETURN columns remain part of the result schema.
        if outer_rows.is_empty() {
            let mut columns = result_set.columns;
            for col in output_columns {
                if !columns.contains(col) {
                    columns.push(col.clone());
                }
            }
            return Ok(ResultSet {
                rows: Vec::new(),
                columns,
                lazy_return_items: None,
            });
        }

        // The body is ALREADY optimized — the planner's
        // `pass_optimize_nested_queries` recurses into every `CALL { }`
        // body once at plan time (§3.1: never re-plan per row), with the
        // seed-ignoring fusion passes disabled when the body anchors on an
        // imported variable (so a per-row `MATCH (p)-[:KNOWS]->(f) RETURN
        // count(f)` honours the seeded `p` via CSR adjacency rather than
        // collapsing to the global KNOWS count). The executor runs the
        // body exactly as planned; it does NOT re-optimize.
        //
        // One sub-executor, reused across every outer row. It holds only
        // graph/params refs + fresh per-query caches (regex/spatial), so
        // reuse lets those caches warm across rows instead of being thrown
        // away per row. The deadline is inherited so a long correlated CALL
        // honours the outer timeout.
        let sub = CypherExecutor::with_params(self.graph, self.params, self.deadline)
            .with_streaming(self.streaming)
            .with_parallel(self.parallel)
            .with_cancel(self.cancel)
            .with_budget(self.budget.clone());

        // Run the body once for the first outer row to learn the subquery's
        // RETURN columns, then check those columns for an outer-scope
        // collision (§1.2 rule 4) — including a re-returned imported name.
        // The sub-executor's warnings are drained onto this one once the loop
        // is done (see `absorb_warnings`) — a per-row body raises the same
        // warning on every row, and `warn` de-duplicates it there.
        let mut combined_rows: Vec<ResultRow> = Vec::new();
        let mut sub_columns: Option<Vec<String>> = None;

        for (outer_index, outer_row) in outer_rows.into_iter().enumerate() {
            // Poll the driving-row loop itself: a body that returns one row
            // never enters the cloned-subrow loop below, but a large join
            // must still observe both deadlines and cooperative cancellation.
            self.check_interrupt_periodic(outer_index)?;

            // NULL-anchor handling (§1.3): if an imported variable that the
            // current set arm uses as a pattern anchor is NULL on this outer row (e.g.
            // an unmatched upstream OPTIONAL MATCH), every anchored match
            // produces no rows. Seed the body with an EMPTY pipeline (zero
            // rows) rather than a one-row null binding: a non-aggregating
            // body then yields zero rows (outer row drops), while an
            // aggregating body still yields exactly one row (e.g.
            // `count() = 0`, outer row survives) — matching Neo4j. A NULL
            // scalar import that is NOT a pattern anchor stays in the seed
            // as projected-null (the body's expressions see null).
            let set_seed = SubquerySetSeed {
                outer_row: &outer_row,
                imports: import,
                arm_local: !globally_scoped
                    && body.clauses.iter().any(|c| matches!(c, Clause::Union(_))),
            };
            let (seed, arm_imports) = self.seed_subquery_set_arm(set_seed, body);
            let seed_set = ResultSet {
                rows: vec![seed.clone()],
                columns: Vec::new(),
                lazy_return_items: None,
            };
            // The body is optimized but NOT lazy-marked (`mark_lazy_eligibility`
            // runs only on the top-level query, never on a subquery body), so
            // `finalize_result` yields eager `Vec<Vec<Value>>` rows here.
            let body_set = if globally_scoped {
                sub.execute_clauses_preserving(body, seed_set, &seed, import, set_seed)?
            } else {
                sub.execute_clauses(body, seed_set, &arm_imports, set_seed)?
            };
            let body_result = sub.finalize_result(body_set)?;

            // First row establishes the subquery's columns. Static schema
            // validation already checked collisions, while this runtime
            // guard protects callers that construct ASTs directly.
            if sub_columns.is_none() {
                let columns = if body_result.rows.is_empty() {
                    output_columns
                } else {
                    &body_result.columns
                };
                for col in columns {
                    if declared.contains(col) {
                        return Err(format!(
                            "CALL {{ }} subquery returns a column `{col}` that already exists in \
                             the outer scope; rename the subquery's RETURN alias (re-returning an \
                             imported variable under the same name is a collision in Neo4j)"
                        ));
                    }
                }
                sub_columns = Some(columns.to_vec());
            }
            let cols = sub_columns.as_deref().unwrap();

            // Inner join: zero sub-rows drops the outer row (§1.3). For the
            // last sub-row reuse (move) the outer row; clone for the rest —
            // avoids cloning the outer row for the final pairing.
            let s = body_result.rows.len();
            if s == 0 {
                continue;
            }
            self.budget
                .reserve_rows(combined_rows.len(), s, "CALL subquery row join")?;
            for (sub_idx, sub_row) in body_result.rows[..s - 1].iter().enumerate() {
                self.check_interrupt_periodic(sub_idx)?;
                let mut row = outer_row.clone();
                splice_subquery_columns(&mut row, sub_row, cols);
                combined_rows.push(row);
            }
            let mut row = outer_row;
            splice_subquery_columns(&mut row, &body_result.rows[s - 1], cols);
            combined_rows.push(row);
        }

        self.absorb_warnings(&sub);

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

    /// Build one arm's seed directly from the original outer row. Modern
    /// scope clauses expose every selected import to every arm. Legacy arms
    /// expose only the variables named by that arm's leading importing WITH;
    /// the CALL-level list is merely the union of candidates copied from the
    /// outer row.
    pub(super) fn seed_subquery_set_arm(
        &self,
        set_seed: SubquerySetSeed<'_>,
        arm: &CypherQuery,
    ) -> (ResultRow, Vec<String>) {
        let imports = if set_seed.arm_local {
            legacy_arm_imports(arm)
        } else {
            set_seed.imports.to_vec()
        };
        let anchors = import_pattern_anchors_in_arm(arm, &imports);
        let row = self.seed_row_from_imports(set_seed.outer_row, &imports, &anchors);
        (row, imports)
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
}

pub(crate) fn subquery_output_columns(
    body: &CypherQuery,
    imports: &[String],
    globally_scoped: bool,
) -> Result<Vec<String>, String> {
    let globals = if globally_scoped { imports } else { &[] };
    subquery_set_output_columns(&body.clauses, imports, globals)
}

pub(crate) fn subquery_set_output_columns(
    clauses: &[Clause],
    imports: &[String],
    globals: &[String],
) -> Result<Vec<String>, String> {
    let columns = subquery_arm_output_columns(clauses, imports, globals)?;
    if let Some(Clause::Union(set)) = clauses.iter().find(|c| matches!(c, Clause::Union(_))) {
        let right = subquery_set_output_columns(&set.query.clauses, imports, globals)?;
        if columns != right {
            let operator = match set.kind {
                SetOpKind::Union => "UNION",
                SetOpKind::Intersect => "INTERSECT",
                SetOpKind::Except => "EXCEPT",
            };
            return Err(format!(
                "All sub queries in a {operator} must have the same return column names \
                 (left side {columns:?} != right side {right:?})."
            ));
        }
    }
    Ok(columns)
}

pub(crate) fn subquery_arm_output_columns(
    clauses: &[Clause],
    imports: &[String],
    globals: &[String],
) -> Result<Vec<String>, String> {
    let mut scope = imports.to_vec();
    for clause in clauses {
        match clause {
            Clause::Return(ret) => {
                if ret.items.len() == 1
                    && matches!(ret.items[0].expression, Expression::Star)
                    && ret.items[0].alias.is_none()
                {
                    return Ok(scope);
                }
                return Ok(ret.items.iter().map(return_item_column_name).collect());
            }
            Clause::Union(_) => break,
            Clause::With(with) => project_static_scope(&mut scope, with, globals),
            Clause::Match(matched) | Clause::OptionalMatch(matched) => {
                extend_match_scope(&mut scope, matched)
            }
            Clause::Unwind(unwind) => push_unique(&mut scope, &unwind.alias),
            Clause::LoadCsv(load) => push_unique(&mut scope, &load.variable),
            Clause::Call(call) => {
                for item in &call.yield_items {
                    push_unique(&mut scope, item.alias.as_ref().unwrap_or(&item.name));
                }
            }
            Clause::CallSubquery { import, body } => {
                let nested_imports = match import {
                    CallSubqueryImport::Legacy(names) | CallSubqueryImport::Named(names) => {
                        names.clone()
                    }
                    CallSubqueryImport::All => scope.clone(),
                    CallSubqueryImport::Empty => Vec::new(),
                };
                let nested_globals = matches!(
                    import,
                    CallSubqueryImport::Named(_) | CallSubqueryImport::All
                );
                for column in subquery_output_columns(body, &nested_imports, nested_globals)? {
                    push_unique(&mut scope, &column);
                }
            }
            _ => {}
        }
    }
    Ok(Vec::new())
}

fn project_static_scope(scope: &mut Vec<String>, with: &WithClause, globals: &[String]) {
    if !with
        .items
        .iter()
        .any(|item| matches!(item.expression, Expression::Star))
    {
        scope.clear();
    }
    for item in &with.items {
        if !matches!(item.expression, Expression::Star) {
            push_unique(scope, &return_item_column_name(item));
        }
    }
    for global in globals {
        push_unique(scope, global);
    }
}

fn extend_match_scope(scope: &mut Vec<String>, matched: &MatchClause) {
    for pattern in &matched.patterns {
        for element in &pattern.elements {
            let variable = match element {
                PatternElement::Node(node) => node.variable.as_ref(),
                PatternElement::Edge(edge) => edge.variable.as_ref(),
            };
            if let Some(variable) = variable {
                push_unique(scope, variable);
            }
        }
    }
    for path in &matched.path_assignments {
        push_unique(scope, &path.variable);
    }
}

fn push_unique(scope: &mut Vec<String>, name: &str) {
    if !scope.iter().any(|existing| existing == name) {
        scope.push(name.to_string());
    }
}

/// Splice the subquery's RETURN columns into an existing outer row's
/// projected bindings (the cartesian-pairing case).
fn splice_subquery_columns(row: &mut ResultRow, sub_row: &[Value], sub_columns: &[String]) {
    for (col, val) in sub_columns.iter().zip(sub_row.iter()) {
        row.projected.insert(col.clone(), val.clone());
    }
}

// `import_pattern_anchors` and `seed_ignoring_fusion_passes` live in the
// planner (`planner::mod`) — they encode the plan-time seed-ignoring-fusion
// decision, which the planner OWNS. The executor re-uses
// `import_pattern_anchors` for per-row NULL-anchor detection via the
// planner re-export.
fn legacy_arm_imports(arm: &CypherQuery) -> Vec<String> {
    let Some(Clause::With(with_clause)) = arm.clauses.first() else {
        return Vec::new();
    };
    with_clause
        .items
        .iter()
        .filter_map(|item| match (&item.expression, &item.alias) {
            (Expression::Variable(name), None) => Some(name.clone()),
            _ => None,
        })
        .collect()
}

use crate::graph::languages::cypher::planner::import_pattern_anchors_in_arm;
