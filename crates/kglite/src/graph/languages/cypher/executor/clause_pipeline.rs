//! Read-only clause-pipeline orchestration.

use super::*;
use crate::graph::core::pattern_matching::column_scan;
use std::collections::HashSet;
use std::time::Instant;

#[derive(Clone, Copy)]
struct PipelineScope<'declared, 'row> {
    preserved: Option<(&'row ResultRow, &'row [String])>,
    initial_declared: &'declared HashSet<String>,
    set_seed: Option<SubquerySetSeed<'row>>,
}

impl CypherExecutor<'_> {
    /// [`Self::execute`] with the retention cap named explicitly.
    ///
    /// `execute_union` calls this with `None`: a UNION arm is an *input* to
    /// the statement's result, not the result, so capping it would drop rows
    /// the set operation still has to see — `A EXCEPT B` with a truncated `B`
    /// keeps rows that should have been excluded. Only the top-level call
    /// passes the executor's own `row_limit`.
    pub(super) fn execute_with_cap(
        &self,
        query: &CypherQuery,
        row_limit: Option<usize>,
    ) -> Result<CypherResult, String> {
        let mut profile_stats: Vec<ClauseStats> = Vec::new();
        let initial_declared = HashSet::new();
        let mut result_set = self.execute_clauses_profiled(
            query,
            ResultSet::new(),
            Some(&mut profile_stats),
            None,
            &initial_declared,
            None,
        )?;

        // Applied before `finalize_result`, so rows past the cap are never
        // projected into cells: the cap bounds what the caller retains *and*
        // the work of building what they would have discarded.
        let capped = apply_row_limit(&mut result_set.rows, row_limit);
        let mut result = self.finalize_result(result_set)?;
        helpers::apply_column_display(query, &mut result);
        stamp_row_limit(&mut result, capped);
        result.stats = None;
        if query.profile {
            result.profile = Some(profile_stats);
        }
        self.attach_runtime_diagnostics(&mut result);
        Ok(result)
    }

    /// Drive a read-only `LOAD CSV` pipeline: strip the leading clause, then
    /// run the remaining clauses once per bounded batch of CSV rows and
    /// concatenate the outputs.
    ///
    /// Only reached for read-only queries — `LOAD CSV … CREATE/MERGE/SET`
    /// routes to the mutable engine, which has its own batch driver over the
    /// same [`load_csv::drive`] helper.
    fn execute_load_csv_pipeline(
        &self,
        query: &CypherQuery,
        load: &LoadCsvClause,
        profile: Option<&mut Vec<ClauseStats>>,
        preserved: Option<(&ResultRow, &[String])>,
        initial_declared: &HashSet<String>,
    ) -> Result<ResultSet, String> {
        let empty = ResultRow::new();
        let eval_row = preserved.map_or(&empty, |(source, _)| source);
        let source = self.evaluate_expression(&load.source, eval_row)?;
        let barrier = load_csv::batching_barrier(&query.clauses[1..]);

        // The suffix is executed as its own query so the driver loop reuses
        // the ordinary clause dispatch, fusion, and streaming machinery
        // untouched. Cloning the clause list costs one allocation per
        // `LOAD CSV` query — never per batch, never on any other path.
        let suffix = CypherQuery {
            clauses: query.clauses[1..].to_vec(),
            explain: false,
            profile: query.profile,
            output_format: query.output_format,
            optimizer_tags: Vec::new(),
            context: None,
            // The batches run the scope's clauses, so they keep its guard.
            guard: query.guard.clone(),
            suppress_default: query.suppress_default,
            column_display: Vec::new(),
        };
        let mut suffix_declared = initial_declared.clone();
        suffix_declared.insert(load.variable.clone());

        let mut merged_profile: Vec<ClauseStats> = Vec::new();
        let result = load_csv::drive(
            load,
            &source,
            &self.csv_import,
            barrier.as_deref(),
            &self.budget,
            |mut seed| {
                if let Some((source, names)) = preserved {
                    restore_scoped_imports(&mut seed, source, names);
                }
                let mut batch_profile = Vec::new();
                let out = self.execute_clauses_profiled(
                    &suffix,
                    seed,
                    if query.profile {
                        Some(&mut batch_profile)
                    } else {
                        None
                    },
                    preserved,
                    &suffix_declared,
                    None,
                )?;
                write::merge_profile(&mut merged_profile, batch_profile);
                Ok(out)
            },
        )?;

        if let Some(stats) = profile {
            stats.push(ClauseStats {
                clause_name: clause_display_name(&query.clauses[0]),
                rows_in: 0,
                rows_out: merged_profile.first().map_or(0, |first| first.rows_in),
                elapsed_us: 0,
            });
            stats.extend(merged_profile);
        }
        Ok(result)
    }

    /// Run a query's clause pipeline from a seed result set, without
    /// PROFILE accounting. Thin wrapper for the subquery body path.
    pub(super) fn execute_clauses(
        &self,
        query: &CypherQuery,
        initial: ResultSet,
        initial_declared: &HashSet<String>,
        set_seed: SubquerySetSeed<'_>,
    ) -> Result<ResultSet, String> {
        self.execute_clauses_profiled(query, initial, None, None, initial_declared, Some(set_seed))
    }

    pub(super) fn execute_clauses_preserving(
        &self,
        query: &CypherQuery,
        initial: ResultSet,
        source: &ResultRow,
        names: &[String],
        initial_declared: &HashSet<String>,
        set_seed: SubquerySetSeed<'_>,
    ) -> Result<ResultSet, String> {
        self.execute_clauses_profiled(
            query,
            initial,
            None,
            Some((source, names)),
            initial_declared,
            Some(set_seed),
        )
    }

    fn execute_with_preserving(
        &self,
        clause: &WithClause,
        result_set: ResultSet,
        source: &ResultRow,
        names: &[String],
        dropped: &[String],
    ) -> Result<ResultSet, String> {
        let mut projection = clause.clone();
        projection.where_clause = None;
        let mut result = self.execute_with_retaining(&projection, result_set, dropped)?;
        restore_scoped_imports(&mut result, source, names);
        if let Some(where_clause) = &clause.where_clause {
            result = self.execute_where(where_clause, result)?;
        }
        Ok(result)
    }

    /// Run a query's clause pipeline starting from a caller-provided
    /// `initial` result set, returning the final `ResultSet` (not yet
    /// finalised into a `CypherResult`).
    ///
    /// `execute` calls this with an empty `initial` and an opt-in
    /// `profile` accumulator. A correlated `CALL { ... }` subquery calls
    /// it via `execute_clauses` with a single seed row carrying the
    /// imported bindings (and `profile = None`), so the body's first
    /// `MATCH` expands from the bound outer node/edge.
    pub(super) fn execute_clauses_profiled(
        &self,
        query: &CypherQuery,
        initial: ResultSet,
        mut profile: Option<&mut Vec<ClauseStats>>,
        preserved: Option<(&ResultRow, &[String])>,
        initial_declared: &HashSet<String>,
        set_seed: Option<SubquerySetSeed<'_>>,
    ) -> Result<ResultSet, String> {
        // `LOAD CSV` drives the clauses that follow it over bounded row
        // batches instead of running as a clause, so peak memory never scales
        // with file size. See `executor/load_csv.rs`.
        if let Some(Clause::LoadCsv(load)) = query.clauses.first() {
            return self.execute_load_csv_pipeline(
                query,
                load,
                profile,
                preserved,
                initial_declared,
            );
        }

        let mut result_set = initial;
        let profiling = query.profile;

        // Clauses already consumed: a WHERE folded into the preceding MATCH,
        // or a run absorbed by the streaming pipeline below.
        let mut skip_clause = vec![false; query.clauses.len()];

        for (i, clause) in query.clauses.iter().enumerate() {
            if skip_clause[i] {
                continue;
            }
            self.check_deadline()?;
            // Seed first-clause row consumers with one empty row so standalone
            // expressions (e.g. `WITH [1,2,3] AS l`, `RETURN 1+2`, or a
            // leading ordinary procedure CALL) can be evaluated.
            // Only for the very first clause — a WITH after an empty MATCH
            // must stay empty.
            if i == 0
                && result_set.rows.is_empty()
                && matches!(
                    clause,
                    Clause::With(_)
                        | Clause::Filter(_)
                        | Clause::Unwind(_)
                        | Clause::Return(_)
                        | Clause::Call(_)
                        | Clause::CallSubquery { .. }
                )
            {
                result_set.rows.push(ResultRow::new());
            }

            // If a prior clause produced 0 rows, MATCH/OPTIONAL MATCH cannot
            // extend an empty pipeline — short-circuit to 0 rows.
            if i > 0
                && result_set.rows.is_empty()
                && matches!(clause, Clause::Match(_) | Clause::OptionalMatch(_))
            {
                if let Some(stats) = profile.as_deref_mut() {
                    stats.push(ClauseStats {
                        clause_name: clause_display_name(clause),
                        rows_in: 0,
                        rows_out: 0,
                        elapsed_us: 0,
                    });
                }
                continue;
            }

            if i == 0 && !profiling && result_set.rows.is_empty() && result_set.columns.is_empty() {
                if let Some(result) = self.try_retrieval_entry(&query.clauses)? {
                    let operator = if matches!(query.clauses[1], Clause::FusedTextBm25TopK { .. }) {
                        "FusedTextBm25TopK"
                    } else {
                        "vector retrieval"
                    };
                    self.budget.check_rows(result.rows.len(), operator)?;
                    result_set = result;
                    skip_clause[1] = true;
                    continue;
                }
            }

            // WHERE-into-MATCH fusion: when MATCH is followed by WHERE, pass the
            // WHERE predicate to execute_match for inline filtering during expansion.
            // This prevents materializing millions of rows that WHERE would discard.
            // A single-pattern MATCH folds in both positions: the opening one,
            // and one that joins incoming rows (`UNWIND … MATCH … WHERE`), where
            // the predicate then runs on each driving row's matches before they
            // are collected instead of after all of them are. A multi-pattern
            // MATCH can refer to variables that are not bound during expansion.
            let folded_inline_where = if let Clause::Match(mc) = clause {
                if mc.patterns.len() == 1
                    && (result_set.rows.is_empty() || joined_match_folds_where(mc, preserved))
                {
                    if let Some(Clause::Where(w)) = query.clauses.get(i + 1) {
                        skip_clause[i + 1] = true;
                        Some(self.fold_constants_pred(&w.predicate))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };
            let inline_where = folded_inline_where.as_ref();

            if self.streaming
                && !profiling
                && i == 0
                && preserved.is_none()
                && result_set.rows.is_empty()
            {
                if let Some(streamed) =
                    self.stream_leading_match(query, clause, inline_where, &mut skip_clause)?
                {
                    result_set = streamed;
                    continue;
                }
            }

            // Streaming-pipeline path: absorb a contiguous run of clauses.
            // A bail returns the input unchanged for materialized dispatch.
            // It also runs under a valid-time guard: it starts at a clause
            // other than a MATCH, so it reads only rows an earlier clause
            // matched, and it evaluates their expressions through this
            // (filtered) executor, as the materialized projection does.
            if self.streaming
                && !profiling
                && inline_where.is_none()
                && !matches!(clause, Clause::Match(_) | Clause::OptionalMatch(_))
                && !(preserved.is_some()
                    && matches!(clause, Clause::With(w) if w.where_clause.is_some()))
            {
                match stream::pipeline::try_run_streaming(self, &query.clauses[i..], result_set)? {
                    stream::pipeline::StreamingOutcome::Absorbed(run) => {
                        for off in 1..run.absorbed {
                            if i + off < skip_clause.len() {
                                skip_clause[i + off] = true;
                            }
                        }
                        result_set = run.result;
                        if let Some((source, names)) = preserved {
                            restore_scoped_imports(&mut result_set, source, names);
                        }
                        self.budget
                            .check_rows(result_set.rows.len(), "streaming pipeline")?;
                        continue;
                    }
                    stream::pipeline::StreamingOutcome::Bailed(rs) => result_set = rs,
                }
            }

            let rows_in = profiling.then_some(result_set.rows.len());
            let started = profiling.then(Instant::now);
            column_scan::take_route();
            result_set = self.execute_pipeline_clause(
                query,
                i,
                clause,
                result_set,
                inline_where,
                PipelineScope {
                    preserved,
                    initial_declared,
                    set_seed,
                },
            )?;

            let elapsed = started.map(|start| start.elapsed());
            if let (Some(elapsed), Some(rows_in)) = (elapsed, rows_in) {
                let name = profiled_clause_name(clause, inline_where.is_some());
                if let Some(stats) = profile.as_deref_mut() {
                    stats.push(ClauseStats {
                        clause_name: name,
                        rows_in,
                        rows_out: result_set.rows.len(),
                        elapsed_us: elapsed.as_micros() as u64,
                    });
                }
            }

            if let Some((source, names)) = preserved {
                restore_scoped_imports(&mut result_set, source, names);
            }

            self.budget
                .check_rows(result_set.rows.len(), &clause_display_name(clause))?;
        }

        Ok(result_set)
    }

    /// A leading MATCH feeds a streamable aggregate row by row, so its matches
    /// are never all held at once. The source is this executor's own matcher
    /// (guarded under a valid-time context) and the aggregate evaluates
    /// through this executor, so a context filters the streamed rows exactly
    /// as the materialized route does. `None` leaves every clause to the
    /// materialized dispatch; on a run, the clauses it absorbed are marked in
    /// `skip_clause` and the aggregate's result is returned.
    fn stream_leading_match(
        &self,
        query: &CypherQuery,
        clause: &Clause,
        inline_where: Option<&Predicate>,
        skip_clause: &mut [bool],
    ) -> Result<Option<ResultSet>, String> {
        let Clause::Match(mc) = clause else {
            return Ok(None);
        };
        let agg_at = 1 + usize::from(inline_where.is_some());
        let tail = query.clauses.get(agg_at..).unwrap_or(&[]);
        let Some(run) = match_stream::try_stream_first_match(self, mc, inline_where, tail)? else {
            return Ok(None);
        };
        for flag in skip_clause.iter_mut().skip(agg_at).take(run.absorbed) {
            *flag = true;
        }
        self.budget
            .check_rows(run.result.rows.len(), "streaming pipeline")?;
        Ok(Some(run.result))
    }

    fn execute_pipeline_clause(
        &self,
        query: &CypherQuery,
        index: usize,
        clause: &Clause,
        result_set: ResultSet,
        inline_where: Option<&Predicate>,
        scope: PipelineScope<'_, '_>,
    ) -> Result<ResultSet, String> {
        let PipelineScope {
            preserved,
            initial_declared,
            set_seed,
        } = scope;
        self.debug_assert_matcher_route(clause, query);
        if let Clause::Match(m) = clause {
            self.execute_match(m, result_set, inline_where)
        } else if let (Clause::With(w), Some((source, names))) = (clause, preserved) {
            let dropped = super::projection_scope::dropped_reads(&query.clauses, index);
            let mut result =
                self.execute_with_preserving(w, result_set, source, names, &dropped)?;
            super::projection_scope::strip_retained(&query.clauses, index, &mut result);
            Ok(result)
        } else if let Clause::CallSubquery { import, body } = clause {
            let declared = declared_scope_before(query, index, initial_declared, preserved);
            self.execute_call_subquery(import, body, result_set, &declared)
        } else if let Clause::Union(union) = clause {
            if let Some(seed) = set_seed {
                self.execute_seeded_union(union, result_set, seed, preserved)
            } else {
                self.execute_union(union, result_set)
            }
        } else if let Clause::Return(r) = clause {
            let retain = order_by_scope_after(&query.clauses, index);
            if let Some(set_seed) = set_seed.filter(|_| {
                r.items.len() == 1
                    && matches!(r.items[0].expression, Expression::Star)
                    && r.items[0].alias.is_none()
            }) {
                let globals = preserved.map(|(_, names)| names).unwrap_or(&[]);
                let columns = super::call_subquery::subquery_arm_output_columns(
                    &query.clauses,
                    set_seed.imports,
                    globals,
                )?;
                let mut expanded = r.clone();
                expanded.items = columns
                    .into_iter()
                    .map(|name| ReturnItem {
                        expression: Expression::Variable(name.clone()),
                        alias: Some(name),
                        display: None,
                    })
                    .collect();
                self.execute_return_retaining(&expanded, result_set, &retain)
            } else if result_set.rows.is_empty()
                && r.items
                    .iter()
                    .any(|item| matches!(item.expression, Expression::Star) && item.alias.is_none())
            {
                // No row to read the names `*` stands for out of: name the
                // columns from the query's static scope instead, so a
                // zero-row `RETURN *` reports `['n']` rather than `['*']`.
                let columns = super::call_subquery::subquery_arm_output_columns(
                    &query.clauses[..=index],
                    &[],
                    &[],
                )?;
                let mut expanded = r.clone();
                expanded.items = columns
                    .into_iter()
                    .map(|name| {
                        r.items
                            .iter()
                            .find(|item| {
                                !matches!(item.expression, Expression::Star)
                                    && return_item_column_name(item) == name
                            })
                            .cloned()
                            .unwrap_or(ReturnItem {
                                expression: Expression::Variable(name.clone()),
                                alias: Some(name),
                                display: None,
                            })
                    })
                    .collect();
                self.execute_return_retaining(&expanded, result_set, &retain)
            } else {
                self.execute_return_retaining(r, result_set, &retain)
            }
        } else {
            self.execute_scoped_read_clause(&query.clauses, index, clause, result_set)
        }
    }
}

/// Whether a MATCH that joins incoming rows may take its following `WHERE`
/// inline. Left out: a row cap (counted after the filter, which the join's
/// matcher cap is not), a distinct-by-node hint (its cross-row dedup is
/// licensed only without a predicate), path variables (bound after the
/// join, so a predicate reading one would see none), and a `CALL { }` body
/// running with imports restored around each clause.
fn joined_match_folds_where(mc: &MatchClause, preserved: Option<(&ResultRow, &[String])>) -> bool {
    preserved.is_none()
        && mc.limit_hint.is_none()
        && mc.distinct_node_hint.is_none()
        && mc.path_assignments.is_empty()
        && mc.where_clause.is_none()
}

/// The variable names an `ORDER BY` immediately following `clauses[i]` reads.
///
/// `ORDER BY` executes after the projection that precedes it, so the return
/// projection retains these otherwise-hidden sort keys until ordering runs.
pub(in super::super) fn order_by_scope_after(clauses: &[Clause], i: usize) -> Vec<String> {
    let Some(Clause::OrderBy(order_by)) = clauses.get(i + 1) else {
        return Vec::new();
    };
    let mut names = HashSet::new();
    for item in &order_by.items {
        crate::graph::languages::cypher::planner::simplification::collect_expression_refs(
            &item.expression,
            &mut names,
        );
    }
    names.into_iter().collect()
}

/// Reattach modern CALL-scope imports after a body clause projected or
/// aggregated them away. Bindings retain their original node/edge/path/value
/// representation; output columns are intentionally unchanged.
fn restore_scoped_imports(result_set: &mut ResultSet, source: &ResultRow, names: &[String]) {
    for row in &mut result_set.rows {
        for name in names {
            if let Some(value) = source.node_bindings.get(name) {
                row.node_bindings.insert(name.clone(), *value);
            }
            if let Some(value) = source.edge_bindings.get(name) {
                row.edge_bindings.insert(name.clone(), *value);
            }
            if let Some(value) = source.path_bindings.get(name) {
                row.path_bindings.insert(name.clone(), value.clone());
            }
            if let Some(value) = source.projected.get(name) {
                row.projected.insert(name.clone(), value.clone());
            }
        }
    }
}

/// Declared outer scope before the indexed CALL clause. OPTIONAL MATCH misses
/// can leave a declared variable absent from a row, so row bindings alone are
/// insufficient for correlated-import validation.
fn declared_scope_before(
    query: &CypherQuery,
    index: usize,
    initial_declared: &HashSet<String>,
    preserved: Option<(&ResultRow, &[String])>,
) -> HashSet<String> {
    let mut declared = initial_declared.clone();
    for prior in &query.clauses[..index] {
        crate::graph::languages::cypher::planner::simplification::advance_visible_variable_scope(
            &mut declared,
            prior,
        );
    }
    if let Some((_, names)) = preserved {
        declared.extend(names.iter().cloned());
    }
    declared
}

/// The PROFILE label of `clause`: its plan name, the fused-WHERE note, and the
/// route a fused scan reported (`[column kernel]` or the reason it declined).
fn profiled_clause_name(clause: &Clause, fused_where: bool) -> String {
    let name = if fused_where {
        format!("{} + Where (fused)", clause_display_name(clause))
    } else {
        clause_display_name(clause)
    };
    match column_scan::take_route() {
        Some(route) => format!("{name} [{route}]"),
        None => name,
    }
}
