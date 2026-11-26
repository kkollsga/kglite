//! Cypher mutation execution — execute_mutable + per-clause helpers
//! (execute_create, execute_set, execute_remove, execute_merge). DELETE lives
//! in [`super::delete_clause`] and MERGE's match arm in
//! [`super::merge_pattern`].

use super::super::ast::*;
use super::super::result::*;
use super::bound_writes::BoundWrites;
use super::columnar_write::{
    set_via_column_master, write_column_master, ColumnMasterWrite, MasterCell, PriorCell,
};
use super::edge_property_write::{remove_edge_property, set_edge_property};
use super::id_assignment::{set_after_create, set_existing_id, supply_id_from_set};
use super::identity_fields::{
    check_identity_uniqueness, create_identity, remove_write_field, CreatedIdentity,
    IdentityAliases, TitlesAreIds,
};
use super::projected_targets::{
    is_null_write_target, projected_node_target, promote_projected_relationships,
    refresh_written_projections, remove_clause_variables, resolve_node_write_target,
    set_clause_variables,
};
use super::set_row::{apply_node_property_set, NodePropertySet, SetMemos};
use super::write_scope::{enforce_edge_write_scope, enforce_node_write_scope, enforce_write_scope};
use super::{clause_display_name, delete_clause, merge_pattern, schema_ddl, CypherExecutor};
use crate::datatypes::value_depth::check_value;
pub(crate) mod create_support;
use create_support::{note_created_id, reject_abstract_create};
// The relationship-identity tests address DELETE's slot retirement through the
// mutation module rather than the clause file that owns it.
use crate::datatypes::values::Value;
use crate::graph::algorithms::Interrupt;
use crate::graph::features::temporal::{check_new_node, EmptyIntervals};
use crate::graph::ontology::node_gate::Tally;
use crate::graph::schema::{DirGraph, EdgeData, InternedKey};
use crate::graph::storage::{GraphRead, GraphWrite};
use petgraph::graph::NodeIndex;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Check if a query contains any mutation clauses.
///
/// Recurses into nested sub-pipelines (`CALL { ... }` bodies and `UNION` arms)
/// so a write buried inside one routes the whole query to the mutation path —
/// see [`clause_is_mutation`] for why mis-classifying one is a correctness bug.
pub fn is_mutation_query(query: &CypherQuery) -> bool {
    query.clauses.iter().any(clause_is_mutation)
}

/// Whether a statement contains a CDC lifecycle procedure whose shared log
/// must be deep-snapshotted for statement rollback.
pub(crate) fn mutates_cdc_configuration(query: &CypherQuery) -> bool {
    fn clause_mutates_cdc(clause: &Clause) -> bool {
        match clause {
            Clause::Call(call) => {
                let name = call.procedure_name.to_lowercase();
                name.starts_with("db.cdc.")
                    && super::procedure_registry::is_mutating_procedure(&name)
            }
            Clause::CallSubquery { body, .. } | Clause::Union(UnionClause { query: body, .. }) => {
                mutates_cdc_configuration(body)
            }
            Clause::Foreach { body, .. } => body.iter().any(clause_mutates_cdc),
            _ => false,
        }
    }
    query.clauses.iter().any(clause_mutates_cdc)
}

/// True if `clause` is itself a write clause or contains a write clause in a
/// nested sub-pipeline.
///
/// The single classifier deciding read engine (`executor/mod.rs`) vs mutable
/// engine (`execute_mutable`). A new clause that can mutate — or whose *body*
/// can — must add an arm here that recurses into its body. Miss it and the
/// query is mis-routed to the read engine, where its writes are silently
/// rejected.
pub(crate) fn clause_is_mutation(clause: &Clause) -> bool {
    match clause {
        Clause::Create(_)
        | Clause::Set(_)
        | Clause::Delete(_)
        | Clause::Remove(_)
        | Clause::Merge(_) => true,
        Clause::CallSubquery { body, .. } => is_mutation_query(body),
        // The change-capture lifecycle verbs (`db.cdc.enable` /
        // `db.cdc.disable`) change graph state, so they route to the write
        // engine behind the same read-only and rollback guards as any other
        // mutation. The registry is the classifier — a name list here would be
        // a second one.
        Clause::Call(call) => {
            super::procedure_registry::is_mutating_procedure(&call.procedure_name.to_lowercase())
        }
        Clause::Union(u) => is_mutation_query(&u.query),
        // FOREACH's body holds only update clauses, so it always routes to the
        // mutable engine (matching Neo4j); a degenerate empty-body FOREACH is a
        // harmless no-op there rather than an error on the read path.
        Clause::Foreach { .. } => true,
        // Schema is graph state, so index and constraint DDL are mutations:
        // that is what puts them behind the read-only-graph guard, the
        // read-only-transaction guard, and the rollback checkpoint. The `SHOW`
        // forms are reads and stay on the read engine.
        Clause::Schema(SchemaCommand::ShowIndexes)
        | Clause::Schema(SchemaCommand::ShowProcedures { .. })
        | Clause::Schema(SchemaCommand::ShowFunctions { .. })
        | Clause::Schema(SchemaCommand::ShowOntology)
        | Clause::Schema(SchemaCommand::Constraint(ConstraintCommand::Show)) => false,
        Clause::Schema(_) => true,
        _ => false,
    }
}

/// Execute a mutation query — the write-path counterpart of
/// `CypherExecutor::execute()`.
pub fn execute_mutable(
    graph: &mut DirGraph,
    query: &CypherQuery,
    params: HashMap<String, Value>,
    interrupt: Interrupt,
) -> Result<CypherResult, String> {
    execute_mutable_bounded(graph, query, params, interrupt, None)
}

/// The result-shaping knobs a mutation carries from `ExecuteOptions`. Bundled
/// so the two `Option<usize>`s — which mean opposite things — cannot be
/// swapped at a call site.
#[derive(Clone, Copy)]
pub(crate) struct MutationLimits {
    /// Work budget; exceeding it fails the statement.
    pub max_work_units: Option<usize>,
    /// Retention cap on the trailing RETURN's rows; exceeding it truncates.
    /// The writes themselves always run in full — a cap on what is *reported*
    /// is not a cap on what is *changed*, and `MutationStats` still counts
    /// every write.
    pub row_limit: Option<usize>,
    /// Hide the virtual `title` / `type` keys from the maps the statement's
    /// reads and trailing `RETURN` build (`ExecuteOptions::hide_structural_keys`).
    pub hide_structural_keys: bool,
}

/// Invariant context for one mutation run — everything the clause loop reads
/// but never changes. Bundled so [`run_clause_pipeline`] can be called once
/// per `LOAD CSV` batch without a ten-argument signature.
pub(super) struct MutationCtx<'a> {
    pub diagnostics: &'a std::sync::Mutex<QueryDiagnostics>,
    pub params: &'a HashMap<String, Value>,
    pub interrupt: &'a Interrupt,
    pub budget: &'a super::budget::ExecutionBudget,
    pub profiling: bool,
    pub csv_import: &'a super::load_csv::CsvImportPolicy,
    pub hide_structural_keys: bool,
    pub relationship_identities: &'a std::sync::Arc<
        std::sync::Mutex<super::relationship_identity::StatementRelationshipIdentities>,
    >,
    pub embedding_service:
        Option<&'a crate::graph::edge_embedding_generation::EmbeddingExecutionService<'a>>,
    /// Clauses that precede the pipeline being run — non-empty only for the
    /// `LOAD CSV` suffix, where the stripped `LOAD CSV … AS row` still
    /// declares `row` for correlated-subquery import validation.
    pub leading: &'a [Clause],
    /// The statement's writes onto declared types whose interval is empty,
    /// reported as one warning once the statement succeeds.
    pub empty_intervals: &'a std::sync::Mutex<EmptyIntervals>,
}

/// The owned half of the mutation context: everything [`finalize_mutation`]
/// needs to build the executor for a trailing `RETURN`. Grouped for the same
/// reason [`MutationCtx`] groups the borrowed state — passed individually they
/// push `finalize_mutation` to nine parameters.
struct FinalizeCtx {
    params: HashMap<String, Value>,
    interrupt: Interrupt,
    budget: super::budget::ExecutionBudget,
    row_limit: Option<usize>,
}

impl MutationCtx<'_> {
    fn absorb_runtime(&self, executor: &CypherExecutor<'_>) {
        let mut result = CypherResult::empty();
        executor.attach_runtime_diagnostics(&mut result);
        if let Some(d) = result.diagnostics {
            let mut target = self.diagnostics.lock().unwrap_or_else(|e| e.into_inner());
            for record in d.retrieval {
                if !target.retrieval.contains(&record) {
                    target.retrieval.push(record);
                }
            }
            for warning in d.warnings {
                if !target.warnings.contains(&warning) {
                    target.warnings.push(warning);
                }
            }
        }
    }

    /// Variables declared by everything before `clauses[i]`, including the
    /// stripped leading clauses.
    fn declared_before(&self, clauses: &[Clause], i: usize) -> std::collections::HashSet<String> {
        use crate::graph::languages::cypher::planner::simplification::{
            advance_visible_variable_scope, visible_variables,
        };
        let mut visible = visible_variables(self.leading);
        for clause in &clauses[..i] {
            advance_visible_variable_scope(&mut visible, clause);
        }
        visible
    }
}

/// Poll the abort check before a write row, reporting through the read
/// path's [`check_interrupt`](super::check_interrupt), so a timed-out write
/// says "timed out" and a cancelled one says "cancelled".
///
/// Every row, not the read path's 4,096-unit cadence: a write row is never
/// reliably microsecond-cheap — on disk one SET row clones the touched
/// column, and a row's value expression can be arbitrarily slow — so a count
/// cadence gave no time bound (a 2,025-row SET polled once and overran a
/// 180 s deadline to 402 s). One clock read and one relaxed atomic load per
/// row bounds the overrun to one row's work.
#[inline]
pub(super) fn check_interrupt_row(interrupt: &Interrupt) -> Result<(), String> {
    #[cfg(test)]
    TEST_WRITE_POLLS.with(|polls| polls.set(polls.get() + 1));
    super::check_interrupt(interrupt)
}

#[cfg(test)]
thread_local! {
    /// Polls [`check_interrupt_row`] performed on this thread —
    /// the timing-free meter `write_deadline_tests` reads per statement.
    static TEST_WRITE_POLLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Read and reset this thread's write-poll count.
#[cfg(test)]
pub(super) fn take_write_polls() -> usize {
    TEST_WRITE_POLLS.with(|polls| polls.replace(0))
}

/// The evaluator for a write clause's value expressions. It carries the
/// statement's deadline and cancel flag, so a slow expression inside one row
/// (`range()`, a list comprehension, a `COUNT { }`) polls them as a read
/// would, and the statement's budget, which it charges as a read's projection
/// does — cumulatively across the statement's rows.
pub(super) fn row_evaluator<'a>(
    graph: &'a DirGraph,
    ctx: &WriteClauseCtx<'a>,
) -> CypherExecutor<'a> {
    CypherExecutor::with_params_and_budget(
        graph,
        ctx.params,
        ctx.interrupt.deadline,
        ctx.budget.clone(),
    )
    .with_cancel(ctx.interrupt.cancel)
}

/// Mutable execution with the same row/collection budget used by reads.
/// Session/binding entry points use this; the low-level unbounded primitive
/// above remains useful to Rust callers that do not request a cap.
pub(crate) fn execute_mutable_bounded(
    graph: &mut DirGraph,
    query: &CypherQuery,
    params: HashMap<String, Value>,
    interrupt: Interrupt,
    max_work_units: Option<usize>,
) -> Result<CypherResult, String> {
    execute_mutable_with_csv(
        graph,
        query,
        params,
        interrupt,
        MutationLimits {
            max_work_units,
            row_limit: None,
            hide_structural_keys: false,
        },
        &super::load_csv::CsvImportPolicy::Denied,
        None,
    )
}

/// Mutable execution with an explicit `LOAD CSV` filesystem capability.
///
/// The capability defaults to denied in every other entry point, so granting
/// it is always visible at the call site.
pub(crate) fn execute_mutable_with_csv(
    graph: &mut DirGraph,
    query: &CypherQuery,
    params: HashMap<String, Value>,
    interrupt: Interrupt,
    limits: MutationLimits,
    csv_import: &super::load_csv::CsvImportPolicy,
    embedding_service: Option<
        &crate::graph::edge_embedding_generation::EmbeddingExecutionService<'_>,
    >,
) -> Result<CypherResult, String> {
    // Lowering refuses a writing statement under a context; this catches a
    // caller that executes without optimizing.
    crate::graph::languages::cypher::valid_time::check_executable(query)?;

    let budget = super::budget::ExecutionBudget::new(limits.max_work_units);

    let diagnostics = std::sync::Mutex::new(QueryDiagnostics::default());
    let relationship_identities = std::sync::Arc::new(std::sync::Mutex::new(
        super::relationship_identity::StatementRelationshipIdentities::new(),
    ));
    let empty_intervals = std::sync::Mutex::new(EmptyIntervals::default());
    let mut stats = MutationStats::default();
    let profiling = query.profile;
    let mut profile_stats: Vec<ClauseStats> = Vec::new();
    // Enforced node and relationship rules are judged on what the statement touched once
    // its clauses have run, so a SET later in the statement can repair an
    // earlier CREATE. A stale buffer from a statement that failed midway is
    // dropped here.
    graph.ontology_touched.clear();
    graph.clear_required_touched();
    graph.ontology_touched_edges.clear();
    if !graph.ontology_tx_deferred {
        // A statement outside a transaction is its own: drop what an earlier
        // one left behind.
        graph.ontology_tx.clear();
    }
    let mut ontology_tally = Tally::default();

    // `LOAD CSV` drives the rest of the pipeline over bounded row batches
    // instead of being executed as a clause — the whole point is that peak
    // memory must not scale with file size. See `executor/load_csv.rs`.
    let result_set = if let Some(Clause::LoadCsv(load)) = query.clauses.first() {
        let suffix = &query.clauses[1..];
        let ctx = MutationCtx {
            diagnostics: &diagnostics,
            params: &params,
            interrupt: &interrupt,
            budget: &budget,
            profiling,
            csv_import,
            hide_structural_keys: limits.hide_structural_keys,
            relationship_identities: &relationship_identities,
            embedding_service,
            leading: &query.clauses[..1],
            empty_intervals: &empty_intervals,
        };
        let source = {
            let executor = CypherExecutor::with_params(graph, &params, interrupt.deadline)
                .with_cancel(interrupt.cancel)
                .with_budget(budget.clone());
            executor.evaluate_expression(&load.source, &ResultRow::new())?
        };
        let barrier = super::load_csv::batching_barrier(suffix);
        super::load_csv::drive(
            load,
            &source,
            csv_import,
            barrier.as_deref(),
            &budget,
            |seed| {
                let mut batch_profile = Vec::new();
                let out =
                    run_clause_pipeline(graph, suffix, seed, &ctx, &mut stats, &mut batch_profile)?;
                graph.judge_statement_writes(&mut ontology_tally)?;
                merge_profile(&mut profile_stats, batch_profile);
                Ok(out)
            },
        )?
    } else {
        let ctx = MutationCtx {
            diagnostics: &diagnostics,
            params: &params,
            interrupt: &interrupt,
            budget: &budget,
            profiling,
            csv_import,
            hide_structural_keys: limits.hide_structural_keys,
            relationship_identities: &relationship_identities,
            embedding_service,
            leading: &[],
            empty_intervals: &empty_intervals,
        };
        let out = run_clause_pipeline(
            graph,
            &query.clauses,
            ResultSet::new(),
            &ctx,
            &mut stats,
            &mut profile_stats,
        )?;
        graph.judge_statement_writes(&mut ontology_tally)?;
        out
    };
    // Outside a transaction the statement is the whole unit, so the "must
    // exist" rules are judged here, inside the statement checkpoint; a
    // transaction working copy leaves them to its commit.
    if graph.ontology_tx_judges_here() {
        graph.judge_transaction_end_into(&mut ontology_tally)?;
    }

    let mut result = finalize_mutation(
        graph,
        query,
        FinalizeCtx {
            params,
            interrupt,
            budget,
            row_limit: limits.row_limit,
        },
        result_set,
        stats,
        profiling.then_some(profile_stats),
    )?;
    let d = diagnostics.into_inner().unwrap_or_else(|e| e.into_inner());
    let target = result
        .diagnostics
        .get_or_insert_with(QueryDiagnostics::default);
    target.retrieval.extend(d.retrieval);
    target.warnings.extend(d.warnings);
    let empty = empty_intervals
        .into_inner()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(warning) = empty.warning() {
        super::retrieval_diagnostics::record_warning(&mut target.warnings, warning);
    }
    for warning in ontology_tally.warnings() {
        super::retrieval_diagnostics::record_warning(&mut target.warnings, warning);
    }
    Ok(result)
}

/// Sum one batch's PROFILE rows into the accumulator.
///
/// Every batch runs the identical clause list, so entries align by index and
/// the merged row reads as the clause's total across the whole file.
pub(super) fn merge_profile(acc: &mut Vec<ClauseStats>, batch: Vec<ClauseStats>) {
    for (index, entry) in batch.into_iter().enumerate() {
        match acc.get_mut(index) {
            Some(existing) => {
                existing.rows_in += entry.rows_in;
                existing.rows_out += entry.rows_out;
                existing.elapsed_us += entry.elapsed_us;
            }
            None => acc.push(entry),
        }
    }
}

/// Run `clauses` against `graph`, starting from `seed`.
///
/// Separate from `execute_mutable_with_csv` so the `LOAD CSV` driver can call
/// it once per row batch: `seed` is empty for an ordinary query and holds one
/// batch of CSV rows otherwise.
fn run_clause_pipeline(
    graph: &mut DirGraph,
    clauses: &[Clause],
    seed: ResultSet,
    ctx: &MutationCtx<'_>,
    stats: &mut MutationStats,
    profile_stats: &mut Vec<ClauseStats>,
) -> Result<ResultSet, String> {
    let params = ctx.params;
    let interrupt = ctx.interrupt;
    let budget = ctx.budget;
    let profiling = ctx.profiling;
    let mut result_set = seed;

    // `result_set.rows.is_empty()` means two opposite things. Before any clause
    // has run it means "no binding stream exists yet", and Cypher's implicit
    // single empty row applies — a leading `CREATE` runs once. After a clause
    // has run it means "the stream exists and holds zero rows", and every
    // downstream clause must produce zero rows and no side effects. Only the
    // pipeline can tell the two apart; a clause re-deriving it from emptiness is
    // what made `MATCH`-finds-nothing + `CREATE` fabricate a row and create an
    // unbound node. `leading` is non-empty only for the `LOAD CSV` driver, whose
    // batch rows are an already-established stream.
    let mut stream_established = !ctx.leading.is_empty();
    let mut write_ctx = WriteClauseCtx {
        params,
        interrupt,
        budget,
        relationship_identities: ctx.relationship_identities,
        empty_intervals: ctx.empty_intervals,
        next_set: None,
    };

    for (i, clause) in clauses.iter().enumerate() {
        // The mutation is atomic: aborting here discards the in-flight
        // changes, leaving the graph unchanged.
        super::check_interrupt(interrupt)?;
        // Materialize Cypher's implicit start row for the clauses that consume
        // one. Deliberately lazy rather than seeding `ResultSet::new()` up
        // front: MATCH/OPTIONAL MATCH select their leading (scan) form over
        // their correlated (extend-each-row) form by testing `rows.is_empty()`,
        // so handing them a row would change how they plan. Same seed the
        // read-only path applies in `executor/mod.rs`.
        if !stream_established && clause_needs_implicit_row(clause) {
            result_set.rows.push(ResultRow::new());
        }

        let rows_in = if profiling { result_set.rows.len() } else { 0 };
        let start = if profiling {
            Some(Instant::now())
        } else {
            None
        };

        // An established-but-empty stream cannot be extended: MATCH has nothing
        // to join against, and OPTIONAL MATCH null-pads incoming rows of which
        // there are none. Short-circuit rather than dispatch, so neither reaches
        // its leading form and re-scans the graph.
        if stream_established
            && result_set.rows.is_empty()
            && matches!(clause, Clause::Match(_) | Clause::OptionalMatch(_))
        {
            if let Some(s) = start {
                profile_stats.push(ClauseStats {
                    clause_name: clause_display_name(clause),
                    rows_in,
                    rows_out: 0,
                    elapsed_us: s.elapsed().as_micros() as u64,
                });
            }
            continue;
        }

        write_ctx.next_set = set_after_create(clause, clauses.get(i + 1));
        let mut wrote = true;
        match clause {
            Clause::Create(create) => {
                result_set = execute_create(graph, create, result_set, stats, &write_ctx)?;
            }
            Clause::Set(set) => {
                apply_set_clause(graph, set, &mut result_set, stats, &write_ctx)?;
            }
            Clause::Delete(del) => {
                delete_clause::execute_delete(
                    graph,
                    del,
                    &result_set,
                    stats,
                    interrupt,
                    &mut ctx.relationship_identities.lock().expect("identity lock"),
                )?;
            }
            Clause::Remove(rem) => {
                apply_remove_clause(graph, rem, &mut result_set, stats, &write_ctx)?;
            }
            Clause::Merge(merge) => {
                result_set = execute_merge(graph, merge, result_set, stats, &write_ctx)?;
                // MERGE may run ON MATCH SET / ON CREATE SET; flush as above.
                GraphWrite::flush_pending_writes(&mut graph.graph);
            }
            Clause::Foreach {
                variable,
                list,
                body,
            } => {
                execute_foreach(graph, variable, list, body, &result_set, stats, &write_ctx)?;
                GraphWrite::flush_pending_writes(&mut graph.graph);
            }
            // Correlated CALL { } import validation needs the declared outer
            // scope (variables bound by clauses 0..i), distinct from the
            // bindings present in any single row.
            Clause::CallSubquery { import, body } => {
                let executor = CypherExecutor::with_params(graph, params, interrupt.deadline)
                    .with_cancel(interrupt.cancel)
                    .with_budget(budget.clone())
                    .with_csv_import(ctx.csv_import.clone())
                    .with_hidden_structural_keys(ctx.hide_structural_keys)
                    .with_relationship_identities(Some(ctx.relationship_identities.clone()));
                let declared = ctx.declared_before(clauses, i);
                result_set = executor.execute_call_subquery(import, body, result_set, &declared)?;
                ctx.absorb_runtime(&executor);
            }
            // Change-capture lifecycle verbs run here, not on the read engine
            // (see `clause_is_mutation`); read CDC procedures fall through to
            // the read executor below like every other `CALL`.
            Clause::Call(call)
                if super::procedure_registry::is_mutating_procedure(
                    &call.procedure_name.to_lowercase(),
                ) =>
            {
                result_set = super::mutating_call::execute(
                    graph,
                    call,
                    result_set,
                    super::mutating_call::MutatingCallCtx {
                        params,
                        interrupt,
                        budget,
                        identities: ctx.relationship_identities,
                        service: ctx.embedding_service,
                        diagnostics: ctx.diagnostics,
                    },
                )?;
            }
            // Schema DDL runs here, not on the read engine (see
            // `clause_is_mutation`); the `SHOW` forms classify as reads and
            // never reach this arm.
            Clause::Schema(command) => {
                schema_ddl::execute_schema_mutation(graph, command, stats, interrupt)?;
                if let Some(warning) = schema_ddl::created_index_warning(graph, command) {
                    let mut target = ctx.diagnostics.lock().unwrap_or_else(|e| e.into_inner());
                    super::retrieval_diagnostics::record_warning(&mut target.warnings, warning);
                }
            }
            _ => {
                wrote = false;
                let executor = CypherExecutor::with_params(graph, params, interrupt.deadline)
                    .with_cancel(interrupt.cancel)
                    .with_budget(budget.clone())
                    .with_csv_import(ctx.csv_import.clone())
                    .with_hidden_structural_keys(ctx.hide_structural_keys)
                    .with_relationship_identities(Some(ctx.relationship_identities.clone()));
                result_set = if let Clause::Return(r) = clause {
                    // Same ORDER BY scope carry as the read loop: the write
                    // engine dispatches read clauses one at a time and would
                    // otherwise lose a WITH alias the sort key reads.
                    let retain = super::order_by_scope_after(clauses, i);
                    executor.execute_return_retaining(r, result_set, &retain)?
                } else {
                    executor.execute_scoped_read_clause(clauses, i, clause, result_set)?
                };
                ctx.absorb_runtime(&executor);
            }
        }

        // The graph version moves only at commit; a read later in this
        // statement must not answer from what the cache held before the write.
        if wrote {
            crate::graph::features::temporal::endpoint_index::invalidate(graph);
        }
        budget.check_rows(result_set.rows.len(), &clause_display_name(clause))?;
        super::mutation_support::check_budget(budget, stats)?;

        if let Some(s) = start {
            profile_stats.push(ClauseStats {
                clause_name: clause_display_name(clause),
                rows_in,
                rows_out: result_set.rows.len(),
                elapsed_us: s.elapsed().as_micros() as u64,
            });
        }

        // From here on, `rows.is_empty()` means "zero rows", never "not started".
        super::mutation_support::stamp_relationships(&mut result_set, ctx.relationship_identities);

        stream_established = true;
    }

    Ok(result_set)
}

/// Does `clause` consume Cypher's implicit single empty start row when it opens
/// a query?
///
/// True for the clauses that produce output per incoming row and so need one to
/// act on with nothing before them. MATCH/OPTIONAL MATCH are absent on purpose:
/// they open a query by scanning, not by extending a row.
///
/// Only ever consulted while the stream is unestablished, so it cannot
/// resurrect a stream that a preceding clause emptied.
fn clause_needs_implicit_row(clause: &Clause) -> bool {
    matches!(
        clause,
        Clause::With(_)
            | Clause::Filter(_)
            | Clause::Unwind(_)
            | Clause::Call(_)
            | Clause::CallSubquery { .. }
            | Clause::Create(_)
            | Clause::Merge(_)
            | Clause::Foreach { .. }
    )
}

/// Flush staged writes and project the pipeline's rows into a `CypherResult`.
///
/// Outside [`run_clause_pipeline`] so the flush and the projection happen
/// exactly once per query, even when the `LOAD CSV` driver ran the pipeline
/// many times.
fn finalize_mutation(
    graph: &mut DirGraph,
    query: &CypherQuery,
    ctx: FinalizeCtx,
    mut result_set: ResultSet,
    stats: MutationStats,
    profile: Option<Vec<ClauseStats>>,
) -> Result<CypherResult, String> {
    // Flush pending mutation state into the steady-state stores so both the
    // trailing RETURN and any later read-only query observe this query's
    // writes. No-op on memory/mapped (writes land in `StableDiGraph`
    // directly); on disk it drains `node_mut_cache`/`edge_mut_cache` into
    // `column_stores`/`edge_properties`. Without it, Cypher SET on a
    // disk-backed graph appeared to no-op until the next mutation or save.
    GraphWrite::flush_pending_writes(&mut graph.graph);

    let has_return = query.clauses.iter().any(|c| matches!(c, Clause::Return(_)));
    // A query that ends in an updating clause has no result rows, whatever a
    // preceding WITH projected. A trailing procedure CALL keeps its YIELD rows.
    let ends_in_update = matches!(
        query.clauses.last(),
        Some(
            Clause::Create(_)
                | Clause::Merge(_)
                | Clause::Set(_)
                | Clause::Remove(_)
                | Clause::Delete(_)
                | Clause::Foreach { .. }
        )
    );

    if has_return || (!ends_in_update && !result_set.columns.is_empty()) {
        let FinalizeCtx {
            params,
            interrupt,
            budget,
            row_limit,
        } = ctx;
        // The writes are already applied and counted; the cap only decides how
        // many of the RETURN's rows come back. Same placement as the read
        // path: before projection, after every clause. See `apply_row_limit`.
        let capped = super::apply_row_limit(&mut result_set.rows, row_limit);
        let executor = CypherExecutor::with_params(graph, &params, interrupt.deadline)
            .with_cancel(interrupt.cancel)
            .with_budget(budget);
        let mut result = executor.finalize_result(result_set)?;
        crate::graph::languages::cypher::result::clear_published_relationship_incarnations(
            &mut result,
        );
        super::helpers::apply_column_display(query, &mut result);
        super::stamp_row_limit(&mut result, capped);
        result.stats = Some(stats);
        result.profile = profile;
        Ok(result)
    } else {
        Ok(CypherResult {
            columns: Vec::new(),
            rows: Vec::new(),
            stats: Some(stats),
            profile,
            diagnostics: None,
            lazy: None,
        })
    }
}

/// Execute a `FOREACH (var IN list | body)` loop.
///
/// For each incoming row, evaluate `list` in that row's context and run
/// `body`'s update clauses once per element with `variable` bound to it.
/// The outer row set is a side-effect input only — it is not modified
/// and body bindings do not propagate out. Zero incoming rows means the
/// body never runs; a standalone FOREACH still runs once, over the implicit
/// start row the pipeline seeds for it.
fn execute_foreach(
    graph: &mut DirGraph,
    variable: &str,
    list: &Expression,
    body: &[Clause],
    outer: &ResultSet,
    stats: &mut MutationStats,
    ctx: &WriteClauseCtx<'_>,
) -> Result<(), String> {
    let interrupt = ctx.interrupt;
    let budget = ctx.budget;
    let folded =
        (!outer.rows.is_empty()).then(|| row_evaluator(graph, ctx).fold_constants_expr(list));
    let list = folded.as_ref().unwrap_or(list);
    for row in outer.rows.iter() {
        check_interrupt_row(interrupt)?;
        // Evaluate the list in this row's context (read-only borrow of the
        // graph, dropped before the per-element mutations below).
        let list_val = row_evaluator(graph, ctx).evaluate_expression(list, row)?;
        let items = match list_val {
            Value::List(items) => items,
            // FOREACH over null is a no-op (Neo4j semantics).
            Value::Null => continue,
            other => {
                return Err(format!("FOREACH expects a list, got {}", other.type_name()));
            }
        };

        budget.check_work(items.len(), "FOREACH")?;

        for item in items.into_iter() {
            check_interrupt_row(interrupt)?;
            let mut elem_row = row.clone();
            elem_row.projected.insert(variable.to_string(), item);
            let mut elem_set = ResultSet {
                rows: vec![elem_row],
                columns: outer.columns.clone(),
                lazy_return_items: None,
            };
            for bclause in body {
                elem_set = apply_foreach_body_clause(graph, bclause, elem_set, stats, ctx)?;
            }
        }
    }
    Ok(())
}

/// The statement-invariant inputs every write clause shares — built once per
/// pipeline run and handed to every write clause and each FOREACH body clause.
pub(super) struct WriteClauseCtx<'a> {
    pub(super) params: &'a HashMap<String, Value>,
    pub(super) interrupt: &'a Interrupt,
    pub(super) budget: &'a super::budget::ExecutionBudget,
    pub(super) relationship_identities: &'a std::sync::Arc<
        std::sync::Mutex<super::relationship_identity::StatementRelationshipIdentities>,
    >,
    pub(super) empty_intervals: &'a std::sync::Mutex<EmptyIntervals>,
    /// The SET clause directly after the CREATE being run, which may name the
    /// ids of the nodes it creates ([`supply_id_from_set`]).
    pub(super) next_set: Option<&'a SetClause>,
}

impl WriteClauseCtx<'_> {
    /// The statement's empty-interval count, for a write check to add to.
    pub(super) fn empty_intervals(&self) -> std::sync::MutexGuard<'_, EmptyIntervals> {
        self.empty_intervals
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// [`check_new_node`], counting into the statement's empty intervals.
    fn check_new_node<'l>(
        &self,
        graph: &DirGraph,
        labels: impl IntoIterator<Item = &'l str>,
        row: (&Value, impl Fn(&str) -> Option<Value>),
    ) -> Result<(), String> {
        check_new_node(graph, labels, row, &mut self.empty_intervals())
    }
}

/// Apply one clause inside a FOREACH body. Only update clauses and nested
/// FOREACH are valid (the parser enforces this; the catch-all is a guard).
/// Mirrors the per-clause mutation handling (incl. disk flush/sync) from
/// `run_clause_pipeline`.
fn apply_foreach_body_clause(
    graph: &mut DirGraph,
    clause: &Clause,
    mut result_set: ResultSet,
    stats: &mut MutationStats,
    ctx: &WriteClauseCtx<'_>,
) -> Result<ResultSet, String> {
    let interrupt = ctx.interrupt;
    let relationship_identities = ctx.relationship_identities;
    // The flush is per element, not per clause: on disk a property read in the
    // same or a later iteration (e.g. `coalesce(n.hits, 0)`) reads the type's
    // column store, which only reflects a staged write once
    // `flush_pending_writes` has drained `node_mut_cache`. Reducing it safely
    // needs the disk read path to consult the mut-cache, a deeper storage
    // change. (Memory mode pays ~nothing.)
    match clause {
        Clause::Create(create) => execute_create(graph, create, result_set, stats, ctx),
        Clause::Set(set) => {
            apply_set_clause(graph, set, &mut result_set, stats, ctx)?;
            Ok(result_set)
        }
        Clause::Delete(del) => {
            delete_clause::execute_delete(
                graph,
                del,
                &result_set,
                stats,
                interrupt,
                &mut relationship_identities.lock().expect("identity lock"),
            )?;
            GraphWrite::flush_pending_writes(&mut graph.graph);
            Ok(result_set)
        }
        Clause::Remove(rem) => {
            apply_remove_clause(graph, rem, &mut result_set, stats, ctx)?;
            Ok(result_set)
        }
        Clause::Merge(merge) => {
            let rs = execute_merge(graph, merge, result_set, stats, ctx)?;
            GraphWrite::flush_pending_writes(&mut graph.graph);
            Ok(rs)
        }
        Clause::Foreach {
            variable,
            list,
            body,
        } => {
            execute_foreach(graph, variable, list, body, &result_set, stats, ctx)?;
            Ok(result_set)
        }
        other => Err(format!(
            "FOREACH body may only contain update clauses, got {}",
            clause_display_name(other)
        )),
    }
}

fn execute_create(
    graph: &mut DirGraph,
    create: &CreateClause,
    existing: ResultSet,
    stats: &mut MutationStats,
    ctx: &WriteClauseCtx<'_>,
) -> Result<ResultSet, String> {
    // On disk, node properties route through the per-type ColumnStore in
    // `DirGraph::insert_node_routed`, which writes into the store the backend
    // owns — no second copy to sync, unlike the SET/REMOVE staged-write path.
    // One CREATE per incoming row, and nothing at all for zero rows. A leading
    // CREATE gets its single empty row from the pipeline's implicit-start-row
    // seed (`clause_needs_implicit_row`) — this function must not synthesize
    // one, because from here an empty `existing` is indistinguishable from a
    // preceding MATCH that found nothing.
    let folded = (!existing.rows.is_empty())
        .then(|| super::write_folding::fold_create(&row_evaluator(graph, ctx), create));
    execute_folded_create(
        graph,
        folded.as_ref().unwrap_or(create),
        existing,
        stats,
        ctx,
    )
}

/// [`execute_create`] of a clause whose values are already folded — MERGE's
/// create arm, which folded its pattern once for all its rows.
fn execute_folded_create(
    graph: &mut DirGraph,
    create: &CreateClause,
    existing: ResultSet,
    stats: &mut MutationStats,
    ctx: &WriteClauseCtx<'_>,
) -> Result<ResultSet, String> {
    let source_rows = existing.rows;

    let mut new_rows = Vec::with_capacity(source_rows.len());
    let mut titles_are_ids = TitlesAreIds::default();

    for row in source_rows.iter() {
        check_interrupt_row(ctx.interrupt)?;
        let mut new_row = row.clone();

        // Positional element -> NodeIndex record for the pattern part being
        // walked. Reused (cleared) per part; see `create_pattern_edges`.
        let mut element_nodes: Vec<Option<NodeIndex>> = Vec::new();

        for pattern in &create.patterns {
            // First pass: create all new nodes.
            //
            // The variable map is `new_row.node_bindings` itself, NOT a map
            // rebuilt per comma-separated part, so a variable introduced in one
            // part is a *reference* in every later part of the same CREATE.
            // Rebuilding it per part made each part blind to its predecessors:
            // `CREATE (a:T {id:5}), (b:T {id:7}), (b)-[:E]->(a)` silently
            // fabricated two anonymous nodes and wired the edge between those.
            element_nodes.clear();
            element_nodes.resize(pattern.elements.len(), None);

            for (pos, element) in pattern.elements.iter().enumerate() {
                if let CreateElement::Node(node_pat) = element {
                    // If the variable is already bound — by a prior MATCH, by
                    // an earlier part of this same CREATE, or as a projected
                    // node VALUE — this occurrence references that node
                    // instead of creating a second one.
                    if let Some(var) = node_pat.variable.as_deref() {
                        if let Some(&bound) = new_row.node_bindings.get(var) {
                            element_nodes[pos] = Some(bound);
                            continue;
                        }
                        if let Some(projected) = projected_node_target(graph, &new_row, var)? {
                            element_nodes[pos] = Some(projected);
                            continue;
                        }
                    }

                    let node_idx =
                        create_node(graph, node_pat, &new_row, stats, ctx, &mut titles_are_ids)?;

                    // Recorded by position as well as by name: an anonymous
                    // endpoint has no name to record under — see
                    // `create_pattern_edges`.
                    element_nodes[pos] = Some(node_idx);
                    if let Some(var) = node_pat.variable.as_deref() {
                        new_row.node_bindings.insert(var.to_string(), node_idx);
                    }
                }
            }

            // Second pass: create edges.
            create_pattern_edges(graph, pattern, &element_nodes, &mut new_row, stats, ctx)?;
        }

        new_rows.push(new_row);
    }

    if stats.relationships_created > 0 {
        graph.invalidate_edge_type_counts_cache();
        // Defensive: build the CSR if these edges landed in the deferred-build
        // pending set (no-op on memory/mapped and when nothing is pending —
        // individual Cypher edges normally go straight to disk overflow and are
        // already visible).
        graph.ensure_disk_edges_built()?;
    }

    Ok(ResultSet {
        rows: new_rows,
        columns: existing.columns,
        lazy_return_items: None,
    })
}

/// Create every edge of one CREATE pattern part, wiring endpoints from the
/// positional record the node pass produced.
///
/// Elements alternate `[Node, Edge, Node, Edge, Node, …]` (the parser enforces
/// both the alternation and the node-terminated shape), so the endpoints of the
/// edge at index `i` are the elements at `i - 1` and `i + 1`. Endpoints are
/// resolved from `element_nodes` — by *position*, not by name — because an
/// anonymous endpoint has no name to resolve by.
fn create_pattern_edges(
    graph: &mut DirGraph,
    pattern: &CreatePattern,
    element_nodes: &[Option<NodeIndex>],
    new_row: &mut ResultRow,
    stats: &mut MutationStats,
    ctx: &WriteClauseCtx<'_>,
) -> Result<(), String> {
    let mut i = 1;
    while i < pattern.elements.len() {
        if let CreateElement::Edge(edge_pat) = &pattern.elements[i] {
            let source_idx = resolve_create_node_idx(pattern, element_nodes, i - 1)?;
            let target_idx = resolve_create_node_idx(pattern, element_nodes, i + 1)?;

            let (actual_source, actual_target) = match edge_pat.direction {
                CreateEdgeDirection::Outgoing => (source_idx, target_idx),
                CreateEdgeDirection::Incoming => (target_idx, source_idx),
            };

            // Edge creation is write-scoped by the ONE-ENDPOINT rule
            // (`enforce_edge_write_scope`): at least one endpoint's stored type
            // must be in the whitelist. Linking two *existing* (MATCH-bound)
            // nodes mutates neither — it reads both endpoints — so the central
            // agent-contract pattern (link a runtime `Task` to a managed
            // `AlgorithmSpec`) stays allowed under a scope that excludes the
            // managed type. What the rule blocks is an edge between two nodes
            // the role owns nothing in (the `FORGED` repro). A *newly created*
            // endpoint is in scope by construction — its `create_node` enforces
            // the scope. Gated before anything is registered or written, so a
            // refusal leaves the schema exactly as it found it.
            enforce_edge_write_scope(
                graph,
                &edge_pat.connection_type,
                actual_source,
                actual_target,
            )?;

            // Endpoint types — needed for both the schema-lock check
            // and the connection-type metadata upsert below.
            let src_type = graph
                .node_view(actual_source)
                .map(|n| n.get_node_type_ref(&graph.interner).to_string())
                .unwrap_or_default();
            let tgt_type = graph
                .node_view(actual_target)
                .map(|n| n.get_node_type_ref(&graph.interner).to_string())
                .unwrap_or_default();

            if graph.schema_locked {
                crate::graph::mutation::validation::validate_edge_creation(
                    &edge_pat.connection_type,
                    &src_type,
                    &tgt_type,
                    &graph.connection_type_metadata,
                    &graph.node_type_metadata,
                )?;
            }

            // A property evaluating to null is *not written*, matching what
            // `CREATE (:N {x: null})` does for nodes (the columnar node store
            // skips null cells, so `keys(n)` never reports one). An edge's
            // properties are a plain key/value vector with no such skip, so the
            // filter has to happen here, or `keys(r)` reports a property the
            // same literal did not create on a node and the metadata upsert
            // below registers a phantom `"Null"`-typed property that
            // `schema_text()`/`connection_types()` advertise forever.
            // `check_rel_row` judges `Some(Value::Null)` and `None` identically,
            // so relationship constraints are unaffected.
            let mut edge_props = HashMap::new();
            {
                let executor = row_evaluator(graph, ctx);
                for (key, expr) in &edge_pat.properties {
                    let val = executor.evaluate_expression(expr, new_row)?;
                    check_value(&val)?;
                    if matches!(val, Value::Null) {
                        continue;
                    }
                    edge_props.insert(key.clone(), val);
                }
            }
            crate::graph::session::snapshot_property_values(&graph.graph, edge_props.values_mut());
            let source_type = graph
                .graph
                .node_type_of(actual_source)
                .and_then(|key| graph.interner.try_resolve(key));
            crate::graph::features::temporal::coerce_new_edge_bounds(
                graph,
                (&edge_pat.connection_type, source_type),
                &mut edge_props,
            );
            edge_props.retain(|_, value| !matches!(value, Value::Null));
            // Freshness provenance: stamp `updated_at` if this edge type
            // opted in (before metadata/EdgeData pick up the props).
            graph.inject_edge_provenance(&edge_pat.connection_type, &mut edge_props);

            // Declared relationship constraints, gated *before* anything is
            // registered or written: registering first would teach the schema a
            // connection type — and a property shape — that no successful write
            // ever produced, and leave `describe()` advertising it. It also
            // keeps the change log free of phantoms, since every capture on this
            // path happens downstream of here.
            if graph.has_rel_constraints() {
                graph.check_new_rel_row(&edge_pat.connection_type, |property| {
                    edge_props.get(property).cloned()
                })?;
            }
            crate::graph::features::temporal::check_new_edge(
                graph,
                &edge_pat.connection_type,
                (actual_source, actual_target),
                |property| edge_props.get(property).cloned(),
                &mut ctx.empty_intervals(),
            )?;

            // Register both the lightweight cache (for `has_connection_type`)
            // AND the metadata map: the metadata is what `connection_types()`,
            // the planner's schema check and the columnar edge-store save read.
            // Without it a Cypher-created relationship type was treated as
            // "unknown" (spurious warnings) and — on a columnar graph — its
            // edges were silently dropped on `save()`, since that store
            // serializes by registered connection type. (SimulatoRS, 0.12.1.)
            graph.register_connection_type(edge_pat.connection_type.clone());
            let prop_types: HashMap<String, String> = edge_props
                .iter()
                .map(|(k, v)| (k.clone(), v.type_name().to_string()))
                .collect();
            graph.upsert_connection_type_metadata(
                &edge_pat.connection_type,
                &src_type,
                &tgt_type,
                prop_types,
            );
            stats.relationships_created += 1;

            let edge_data = EdgeData::new(
                edge_pat.connection_type.clone(),
                edge_props,
                &mut graph.interner,
            );
            let edge_index =
                GraphWrite::add_edge(&mut graph.graph, actual_source, actual_target, edge_data);
            crate::graph::index_freshness::write_hooks::note_edge_created(graph, edge_index);
            graph.note_required_edge(&edge_pat.connection_type, edge_index);
            graph.note_ontology_edge_touch(edge_index);
            graph.note_tx_edge_added(edge_index);

            if let Some(ref var) = edge_pat.variable {
                new_row.edge_bindings.insert(
                    var.clone(),
                    EdgeBinding {
                        incarnation: None,
                        source: actual_source,
                        target: actual_target,
                        edge_index,
                    },
                );
            }
        }
        i += 2;
    }
    Ok(())
}

/// Evaluate a CREATE pattern's property expressions against `row`. The
/// evaluator borrows `graph` immutably and is dropped before the caller writes.
fn evaluate_properties(
    graph: &DirGraph,
    properties: &[(String, Expression)],
    row: &ResultRow,
    ctx: &WriteClauseCtx<'_>,
) -> Result<HashMap<String, Value>, String> {
    let executor = row_evaluator(graph, ctx);
    properties
        .iter()
        .map(|(key, expr)| {
            let value = executor.evaluate_expression(expr, row)?;
            check_value(&value)?;
            Ok((key.clone(), value))
        })
        .collect()
}

fn create_node(
    graph: &mut DirGraph,
    node_pat: &CreateNodePattern,
    row: &ResultRow,
    stats: &mut MutationStats,
    ctx: &WriteClauseCtx<'_>,
    titles_are_ids: &mut TitlesAreIds,
) -> Result<petgraph::graph::NodeIndex, String> {
    let mut properties = evaluate_properties(graph, &node_pat.properties, row, ctx)?;
    let label = node_pat.label.clone().unwrap_or_else(|| "Node".to_string());
    crate::graph::features::temporal::coerce_new_node_bounds(
        graph,
        (&label, node_pat.extra_labels.iter().map(String::as_str)),
        &mut properties,
    );

    // Identity fields, under whichever spelling this node type declares — see
    // [`IdentityAliases`] and [`create_identity`].
    let aliases = IdentityAliases::for_type(graph, &label);
    supply_id_from_set(graph, ctx, (node_pat, row), &aliases, &mut properties);
    crate::graph::session::snapshot_property_values(
        &graph.graph,
        properties
            .iter_mut()
            .filter(|(key, _)| aliases.canonical(key.as_str()) != "id")
            .map(|(_, value)| value),
    );
    let CreatedIdentity {
        id,
        title,
        title_supplied,
    } = create_identity(
        graph,
        node_pat,
        &label,
        &aliases,
        &mut properties,
        titles_are_ids,
    )?;

    check_identity_uniqueness(graph, &label, &id)?;

    // A constraint declared on the type's *own* id/title spelling (`REQUIRE
    // p.name IS UNIQUE` on a type loaded with `node_title_field='name'`) reads
    // the value this CREATE is putting in the identity field — that spelling no
    // longer has a property key of its own, the value having been promoted.
    // The title alias answers only when the caller *supplied* the title: the
    // id or `<Label>_<id>` fallback is engine-minted, and a `REQUIRE … IS NOT
    // NULL` on the type's title column is asking the caller for a value.
    let constraint_read = |property: &str| -> Option<Value> {
        if aliases.id_field() == Some(property) {
            return (!matches!(id, Value::Null)).then(|| id.clone());
        }
        if aliases.title_field() == Some(property) {
            return (title_supplied && !matches!(title, Value::Null)).then(|| title.clone());
        }
        match property {
            "id" => (!matches!(id, Value::Null)).then(|| id.clone()),
            "title" => (!matches!(title, Value::Null)).then(|| title.clone()),
            other => match properties.get(other) {
                Some(Value::Null) | None => None,
                Some(value) => Some(value.clone()),
            },
        }
    };
    // Presence is judged at statement end, once a `SET n = $props` has run:
    // the node is noted after the insert (`DirGraph::judge_required_touched`).
    // Declared property types. Checked here — before the schema-lock
    // validation further down — so a user who declared a type constraint gets
    // the constraint's own error rather than the generic observed-metadata one
    // for the same value.
    let typed = graph.check_property_types(&label, constraint_read);
    if let Err(violation) = typed {
        return Err(graph.record_constraint_violation(*violation));
    }
    // Declared structured shapes gate CREATE the same way (never WAL
    // replay — replay does not route through this executor).
    if !graph.property_shapes.is_empty() {
        for (shape_property, shape) in graph.shapes_for_type(&label) {
            if let Some(value) = constraint_read(&shape_property) {
                shape.check(&shape_property, &value)?;
            }
        }
    }
    let ancestors = crate::graph::features::temporal::ancestor_labels(graph, &label);
    let labels = std::iter::once(label.as_str())
        .chain(node_pat.extra_labels.iter().map(String::as_str))
        .chain(ancestors.iter().copied());
    ctx.check_new_node(graph, labels, (&id, constraint_read))?;
    let unique_claims = graph.unique_claims(&label, constraint_read);
    let unique = graph.check_unique_claims(&unique_claims, None);
    if let Err(violation) = unique {
        return Err(graph.record_constraint_violation(*violation));
    }

    // Cloned for the incremental id-index maintenance below, which cannot have
    // it: `insert_node_routed` moves the original.
    let pk_id = Some(id.clone());

    // Role-scoped write guard (integrity): reject CREATE of a node type
    // outside the active write whitelist, before any storage mutation.
    enforce_write_scope(graph, &label)?;

    // An abstract ontology class is not creatable — after the security
    // guard (a caller outside the whitelist must not learn the subtype
    // list), before the schema-lock check (which would accept a declared
    // class and lose this message).
    reject_abstract_create(graph, &label)?;

    if graph.schema_locked {
        crate::graph::mutation::validation::validate_node_creation(
            &label,
            &properties,
            &graph.node_type_metadata,
            graph.schema_definition.as_ref(),
            &|property| {
                std::iter::once(&label)
                    .chain(&node_pat.extra_labels)
                    .any(|l| graph.temporal.names_node_bound(l, property))
            },
        )?;
    }

    // Last abort point: a single-node CREATE can run with no rollback
    // checkpoint (`can_skip_rollback_checkpoint`), so no deadline error may
    // follow the insert.
    super::check_interrupt(ctx.interrupt)?;

    // Every backend writes id/title/properties through the per-type
    // ColumnStore — see `DirGraph::insert_node_routed`.
    let node_idx = graph.insert_node_routed(id, title, &label, properties);
    graph.note_ontology_touch(node_idx);
    graph.note_required_node(&label, node_idx);

    // Update type_indices. `bucket_was_new` feeds statement rollback: undoing
    // the append is not enough if this CREATE also *introduced* the type —
    // an emptied-but-present bucket still shows up in `describe()` as a
    // zero-count type.
    let bucket_was_new = !graph.type_indices.contains_key(&label);
    graph.type_indices.push_to_type(&label, node_idx);
    // One of the two creation funnels the freshness watermark cannot see
    // through on its own: a `CREATE` into a slot petgraph recycled lands
    // *below* the watermark and would otherwise read as already-indexed.
    crate::graph::index_freshness::write_hooks::note_node_created(graph, node_idx, &label);
    if let Some(journal) = graph.graph.undo_journal_mut() {
        journal.note_bucket_appended(
            crate::graph::storage::undo::BucketId::NodeType(label.clone()),
            node_idx,
            bucket_was_new,
        );
    }

    note_created_id(graph, &label, pk_id, node_idx);

    // Files the node into every index of its type — including the ones its
    // UNIQUE constraints own, which is what claims the tuples validated above.
    graph.update_property_indices_for_add(&label, node_idx);

    ensure_type_metadata(graph, &label);

    // Apply secondary labels from `CREATE (n:A:B:C)` patterns. The
    // first label is the primary type (set by `insert_node_routed`
    // above); the rest are added through the choke-point API so the
    // secondary_label_index stays in sync.
    for extra in &node_pat.extra_labels {
        let key = graph.interner.get_or_intern(extra);
        graph.add_node_label(node_idx, key);
    }

    // Materialized-ontology closure maintenance: a new node of a declared
    // type carries its ancestors from birth. After the explicit labels, so
    // `CREATE (:Student:Person)` normalizes to the same set regardless of
    // order (`add_node_label` is idempotent).
    if !graph.managed_labels.is_empty() && !graph.suppress_ontology_stamp {
        let type_key = graph.interner.get_or_intern(&label);
        let ancestors: Vec<_> = graph.ontology_ancestors_of(type_key).to_vec();
        for ancestor in ancestors {
            graph.add_node_label(node_idx, ancestor);
        }
    }

    stats.nodes_created += 1;

    Ok(node_idx)
}

/// Give the created type an entry in `node_type_metadata`, so a type whose every
/// CREATE carried no property still shows up in `describe()` / the saved schema
/// (what the Python `add_nodes` API does in maintain.rs).
///
/// Only the empty entry: property-type *inference* belongs to
/// `DirGraph::register_property_types`, which runs inside `insert_node_routed`
/// from the property map in hand. Reading the node back to infer types (what
/// this used to do) could never find a key that call had not already registered,
/// and was wrong on disk, where the columnar store is not synced to the read
/// side until the end of the clause — so the read-back saw no properties at all.
fn ensure_type_metadata(graph: &mut DirGraph, node_type: &str) {
    if graph.node_type_metadata.contains_key(node_type) {
        return;
    }
    graph
        .node_type_metadata_mut()
        .entry(node_type.to_string())
        .or_default();
}

fn value_type_name(v: &Value) -> String {
    v.type_name().to_string()
}

/// Resolve a CREATE edge endpoint at `pos` to its NodeIndex.
///
/// Both error arms are structural (a non-node element at an endpoint position,
/// or a position the node pass never visited); the parser's alternation check
/// makes them unreachable from user input, and they stay as defensive
/// diagnostics rather than an index panic.
fn resolve_create_node_idx(
    pattern: &CreatePattern,
    element_nodes: &[Option<NodeIndex>],
    pos: usize,
) -> Result<NodeIndex, String> {
    match pattern.elements.get(pos) {
        Some(CreateElement::Node(node_pat)) => {
            element_nodes.get(pos).copied().flatten().ok_or_else(|| {
                match node_pat.variable.as_deref() {
                    Some(name) => format!("Unbound variable '{}' in CREATE edge", name),
                    None => "Unresolved anonymous node in CREATE edge".to_string(),
                }
            })
        }
        _ => Err("CREATE edge endpoints must be node patterns".to_string()),
    }
}

/// The node's type name when that type opted into `updated_at` stamping, else
/// `None`. The read is scoped so the borrow ends before the caller's `&mut`
/// writes.
fn auto_timestamp_type_of(graph: &DirGraph, node_idx: NodeIndex) -> Option<String> {
    let node_type = {
        graph
            .graph
            .node_view(node_idx)
            .map(|n| n.node_type_str(&graph.interner).to_string())
    };
    node_type.filter(|nt| graph.auto_timestamp_for(nt))
}

/// Every property key currently set on `variable`'s binding — the clear-list
/// for `SET n = {…}` / `SET r = {…}`.
fn existing_property_keys(
    graph: &crate::graph::dir_graph::DirGraph,
    row: &ResultRow,
    variable: &str,
) -> Vec<String> {
    // A projected node value is a `SET x = {…}` target like any bound node, so
    // its stored keys have to reach the clear-list too — otherwise the replace
    // form kept every property the map did not mention.
    let node_target =
        row.node_bindings
            .get(variable)
            .copied()
            .or_else(|| match row.projected.get(variable) {
                Some(Value::Node(node)) => Some(NodeIndex::new(node.id as usize)),
                Some(Value::NodeRef(index)) => Some(NodeIndex::new(*index as usize)),
                _ => None,
            });
    if let Some(node_idx) = node_target.as_ref() {
        let mut keys: Vec<String> = graph
            .graph
            .node_view(*node_idx)
            .map(|node| {
                node.properties_cloned(&graph.interner)
                    .into_keys()
                    .collect()
            })
            .unwrap_or_default();
        // Resolved through `NodeView::title()`, not off the inline field: that
        // field is the `Null` sentinel on every columnar node — which is now
        // every node — so reading it directly would drop `name` from the
        // clear-list altogether and let a title survive `SET n = {…}`.
        if graph
            .graph
            .node_view(*node_idx)
            .is_some_and(|node| !matches!(*node.title(), Value::Null))
            && !keys.iter().any(|key| key == "name" || key == "title")
        {
            keys.push("name".to_string());
        }
        keys
    } else if let Some(edge) = row.edge_bindings.get(variable) {
        graph
            .graph
            .edge_weight(edge.edge_index)
            .map(|edge| {
                edge.properties_cloned(&graph.interner)
                    .into_keys()
                    .collect()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    }
}

/// Clear a node's title and stored `name` together, as `REMOVE n.name` /
/// `REMOVE n.title` do. Returns the removed title and `(title, name)` as they
/// stood before: `name` is the stored property if the node has one and the
/// title otherwise, so the two can sit in different index buckets.
fn clear_name_and_title(
    graph: &mut crate::graph::dir_graph::DirGraph,
    node_idx: NodeIndex,
) -> (Option<Value>, (Option<Value>, Option<Value>)) {
    let old = graph.graph.get_node_title(node_idx).unwrap_or(Value::Null);
    let title = (!matches!(old, Value::Null)).then_some(old);
    let key = InternedKey::from_str("name");
    let stored_name = graph
        .graph
        .get_node_property(node_idx, key)
        .filter(|value| !matches!(value, Value::Null));
    let name = stored_name.or_else(|| title.clone());
    GraphWrite::set_node_title(&mut graph.graph, node_idx, Value::Null);
    GraphWrite::remove_node_property(&mut graph.graph, node_idx, key);
    (title.clone(), (title, name))
}

/// Vacate the buckets a [`clear_name_and_title`] left a node in: its `name`
/// from the `name` indexes and its title from the title indexes.
fn refile_cleared_name_title(
    graph: &mut crate::graph::dir_graph::DirGraph,
    node_type: &str,
    node_idx: NodeIndex,
    (title, name): &(Option<Value>, Option<Value>),
) {
    if let Some(name) = name {
        graph.update_property_indices_for_remove(node_type, node_idx, "name", name);
    }
    if let Some(title) = title {
        graph.update_property_indices_for_remove(node_type, node_idx, "title", title);
    }
}

/// Write one property straight onto a node, bypassing the columnar master fast
/// path.
///
/// The fall-through for row storage, for `title` / `name` (a routing detail —
/// see [`set_via_column_master`]), and for a columnar node whose type the
/// backend has no store for. Returns whether the node existed.
pub(super) fn set_node_property_direct(
    graph: &mut crate::graph::dir_graph::DirGraph,
    node_idx: NodeIndex,
    property: &str,
    value: Value,
) -> bool {
    if graph.graph.node_weight(node_idx).is_none() {
        return false;
    }
    let set_title = |graph: &mut crate::graph::dir_graph::DirGraph, v: Value| {
        // Through the backend, not onto the inline field: a columnar node's
        // title belongs in its store's reserved column, and every node is
        // columnar from construction.
        GraphWrite::set_node_title(&mut graph.graph, node_idx, v);
    };
    match property {
        "title" => set_title(graph, value),
        // "name" maps to title in Cypher reads; update both for consistency.
        "name" => {
            set_title(graph, value.clone());
            let key = graph.interner.get_or_intern("name");
            GraphWrite::set_node_property(&mut graph.graph, node_idx, key, value);
        }
        _ => {
            let key = graph.interner.get_or_intern(property);
            GraphWrite::set_node_property(&mut graph.graph, node_idx, key, value);
        }
    }
    true
}

/// Row-loop control signal from [`execute_property_set_item`]: `SkipRow`
/// maps to the openCypher null-target no-op (`continue` in the caller).
enum PropSetFlow {
    Applied,
    SkipRow,
}

/// One `SET var.prop[...path...] = expr` against one result row — extracted
/// from `execute_set`, which is at its complexity ceiling; the argument set
/// is that loop's working state.
fn execute_property_set_item<'a>(
    graph: &mut DirGraph,
    row: &ResultRow,
    (variable, property, path, expression): (
        &'a String,
        &'a String,
        &'a [crate::graph::languages::cypher::ast::SetPathStep],
        &'a Expression,
    ),
    ctx: &WriteClauseCtx<'_>,
    stats: &mut MutationStats,
    (memos, bound_writes): (&mut SetMemos<'a>, &mut BoundWrites),
    (edges_to_stamp, nodes_to_stamp): (
        &mut std::collections::HashSet<petgraph::graph::EdgeIndex>,
        &mut HashMap<NodeIndex, String>,
    ),
) -> Result<PropSetFlow, String> {
    if !path.is_empty() && row.edge_bindings.contains_key(variable) {
        return Err(format!(
            "Nested SET paths are supported on node properties only; '{variable}' is a \
             relationship"
        ));
    }
    // Relationship property SET is its own path — edges carry none of the
    // node id/type guards, columnar routing or index maintenance below.
    if set_edge_property(
        graph,
        row,
        (variable, property, expression),
        ctx,
        stats,
        edges_to_stamp,
    )? {
        if let Some(binding) = row.edge_bindings.get(variable) {
            bound_writes.edge_property(graph, binding.edge_index, property);
        }
        return Ok(PropSetFlow::Applied);
    }

    // `type` / `node_type` / `label` are soft aliases: SET writes a stored
    // property that wins on read (KG-1); the primary label never changes.

    // Resolve the node — a live binding or a projected node value. A
    // null-valued target (OPTIONAL MATCH miss) makes this row's write a no-op
    // per openCypher.
    let Some(node_idx) = resolve_node_write_target(graph, row, variable, "SET")? else {
        return Ok(PropSetFlow::SkipRow);
    };

    // Evaluate the expression (borrows graph immutably). A nested l-value
    // (`o.line_items[2].qty`) does its whole-value read-modify-write in
    // set_path.rs, with the same evaluator.
    let value = {
        let executor = row_evaluator(graph, ctx);
        let value = executor.evaluate_expression(expression, row)?;
        check_value(&value)?;
        if path.is_empty() {
            value
        } else {
            super::set_path::read_modify(graph, &executor, node_idx, property, path, value, row)?
        }
    };

    if property == "id" {
        set_existing_id(graph, node_idx, property, &value)?;
        return Ok(PropSetFlow::Applied);
    }
    apply_node_property_set(
        graph,
        NodePropertySet {
            node_idx,
            property: property.as_str(),
            value,
        },
        memos,
        stats,
        nodes_to_stamp,
    )?;
    bound_writes.node_property(graph, node_idx, property);
    Ok(PropSetFlow::Applied)
}

/// Index bookkeeping and following items must read the completed disk write.
/// Gate before backend dispatch so memory/mapped need no uniqueness check.
pub(super) fn flush_disk_item_writes(graph: &mut DirGraph) {
    if graph.graph.is_disk() {
        GraphWrite::flush_pending_writes(&mut graph.graph);
    }
}

/// One SET clause over the rows the pipeline handed it, shared by the clause
/// pipeline and the FOREACH body. Projected relationship values the clause
/// names become verified bindings first. The flush afterwards is load-bearing
/// on disk: SET stages into `node_mut_cache`, and without it a later clause's
/// `node_weight` (a trailing RETURN's projection included) reads
/// `column_stores` and sees the pre-SET values.
fn apply_set_clause(
    graph: &mut DirGraph,
    set: &SetClause,
    result_set: &mut ResultSet,
    stats: &mut MutationStats,
    ctx: &WriteClauseCtx<'_>,
) -> Result<(), String> {
    let promoted = {
        let identities = ctx.relationship_identities.lock().expect("identity lock");
        promote_projected_relationships(graph, result_set, &set_clause_variables(set), &identities)?
    };
    let rows = promoted.as_ref().unwrap_or(&*result_set);
    execute_set(graph, set, rows, stats, ctx)?;
    GraphWrite::flush_pending_writes(&mut graph.graph);
    refresh_written_projections(graph, result_set, &set_clause_variables(set));
    Ok(())
}

/// [`apply_set_clause`] for REMOVE — same promotion, same disk-staging flush.
fn apply_remove_clause(
    graph: &mut DirGraph,
    remove: &RemoveClause,
    result_set: &mut ResultSet,
    stats: &mut MutationStats,
    ctx: &WriteClauseCtx<'_>,
) -> Result<(), String> {
    let promoted = {
        let identities = ctx.relationship_identities.lock().expect("identity lock");
        let variables = remove_clause_variables(remove);
        promote_projected_relationships(graph, result_set, &variables, &identities)?
    };
    let rows = promoted.as_ref().unwrap_or(&*result_set);
    execute_remove(graph, remove, rows, stats, ctx.interrupt)?;
    GraphWrite::flush_pending_writes(&mut graph.graph);
    refresh_written_projections(graph, result_set, &remove_clause_variables(remove));
    Ok(())
}

/// One `SET` clause over `result_set`, its row-invariant values folded once
/// (`write_folding`), refused when it leaves an element breaking a validity
/// declaration ([`BoundWrites`]).
fn execute_set(
    graph: &mut DirGraph,
    set: &SetClause,
    result_set: &ResultSet,
    stats: &mut MutationStats,
    ctx: &WriteClauseCtx<'_>,
) -> Result<(), String> {
    if result_set.rows.is_empty() {
        return Ok(());
    }
    let folded = SetClause {
        items: super::write_folding::fold_set_items(&row_evaluator(graph, ctx), &set.items),
    };
    execute_folded_set(graph, &folded, result_set, stats, ctx)
}

/// [`execute_set`] of a clause whose values are already folded — MERGE's
/// `ON CREATE` / `ON MATCH SET`, folded once for all its rows.
fn execute_folded_set(
    graph: &mut DirGraph,
    set: &SetClause,
    result_set: &ResultSet,
    stats: &mut MutationStats,
    ctx: &WriteClauseCtx<'_>,
) -> Result<(), String> {
    let mut bound_writes = BoundWrites::default();
    execute_set_items(graph, set, result_set, ctx, stats, &mut bound_writes)?;
    bound_writes.check(graph, &mut ctx.empty_intervals())
}

fn execute_set_items(
    graph: &mut DirGraph,
    set: &SetClause,
    result_set: &ResultSet,
    ctx: &WriteClauseCtx<'_>,
    stats: &mut MutationStats,
    bound_writes: &mut BoundWrites,
) -> Result<(), String> {
    let interrupt = ctx.interrupt;
    // Freshness provenance: nodes (of opted-in types) modified by this SET get a
    // single `updated_at` bump after the loop (engine-managed reserved key) —
    // collected here so multiple property writes on one node stamp it once.
    let mut nodes_to_stamp: std::collections::HashMap<NodeIndex, String> =
        std::collections::HashMap::new();
    // Edges (of opted-in connection types) modified by this SET — bumped once
    // after the loop, same as nodes.
    let mut edges_to_stamp: std::collections::HashSet<petgraph::graph::EdgeIndex> =
        std::collections::HashSet::new();
    // Statement-scoped memo: everything a row's bookkeeping asks that is
    // constant across the statement's rows (see `set_row::SetMemos`).
    let mut memos = SetMemos::default();

    for row in result_set.rows.iter() {
        check_interrupt_row(interrupt)?;
        for item in &set.items {
            match item {
                SetItem::Property {
                    variable,
                    property,
                    path,
                    expression,
                } => {
                    let outcome = execute_property_set_item(
                        graph,
                        row,
                        (variable, property, path, expression),
                        ctx,
                        stats,
                        (&mut memos, &mut *bound_writes),
                        (&mut edges_to_stamp, &mut nodes_to_stamp),
                    )?;
                    if matches!(outcome, PropSetFlow::SkipRow) {
                        continue;
                    }
                }
                SetItem::Map {
                    variable,
                    expression,
                    replace,
                } => {
                    let value = row_evaluator(graph, ctx).evaluate_expression(expression, row)?;
                    check_value(&value)?;
                    let Value::Map(map) = value else {
                        return Err(format!(
                            "SET {} {} expects a map expression",
                            variable,
                            if *replace { "=" } else { "+=" }
                        ));
                    };
                    if !row.node_bindings.contains_key(variable)
                        && !row.edge_bindings.contains_key(variable)
                        && projected_node_target(graph, row, variable)?.is_none()
                    {
                        // Null target (OPTIONAL MATCH miss): no-op for this row.
                        if is_null_write_target(row, variable) {
                            continue;
                        }
                        return Err(format!("Variable '{}' is not bound in SET", variable));
                    }

                    let one_row = ResultSet {
                        rows: vec![row.clone()],
                        columns: Vec::new(),
                        lazy_return_items: None,
                    };

                    if *replace {
                        let existing_keys = existing_property_keys(graph, row, variable);
                        let removals: Vec<RemoveItem> = existing_keys
                            .into_iter()
                            .filter(|key| {
                                if key == "name" || key == "title" {
                                    !map.contains_key("name") && !map.contains_key("title")
                                } else {
                                    !map.contains_key(key)
                                }
                            })
                            .map(|property| RemoveItem::Property {
                                variable: variable.clone(),
                                property,
                            })
                            .collect();
                        if !removals.is_empty() {
                            execute_remove(
                                graph,
                                &RemoveClause { items: removals },
                                &one_row,
                                stats,
                                interrupt,
                            )?;
                        }
                    }

                    let properties: Vec<SetItem> = map
                        .into_iter()
                        .map(|(property, value)| SetItem::Property {
                            variable: variable.clone(),
                            property: property.to_string(),
                            path: Vec::new(),
                            expression: Expression::Literal(value),
                        })
                        .collect();
                    if !properties.is_empty() {
                        execute_set_items(
                            graph,
                            &SetClause { items: properties },
                            &one_row,
                            ctx,
                            stats,
                            bound_writes,
                        )?;
                    }
                }
                SetItem::Label {
                    variable, label, ..
                } => set_node_label(
                    graph,
                    row,
                    (variable, label),
                    stats,
                    (&mut nodes_to_stamp, &mut *bound_writes),
                )?,
            }
            flush_disk_item_writes(graph);
        }
    }

    stamp_node_provenance(graph, &nodes_to_stamp);

    stamp_edge_provenance(graph, &edges_to_stamp);

    Ok(())
}

/// Stamp the reserved provenance keys (`updated_at` + the caller's
/// `git_sha`/`modified_by`) on every edge of an opted-in type a `SET`
/// modified, once per edge.
fn stamp_edge_provenance(
    graph: &mut DirGraph,
    edges_to_stamp: &std::collections::HashSet<petgraph::graph::EdgeIndex>,
) {
    if edges_to_stamp.is_empty() {
        return;
    }
    let provenance = graph.provenance_props();
    let interned: Vec<(InternedKey, Value)> = provenance
        .iter()
        .map(|(k, v)| (graph.interner.get_or_intern(k), v.clone()))
        .collect();
    let mut stamped_types: std::collections::HashSet<InternedKey> =
        std::collections::HashSet::new();
    for edge_index in edges_to_stamp {
        let rel_index_pre = graph.rel_index_capture(*edge_index);
        if let Some(EdgeData {
            connection_type,
            properties: edge_props,
            ..
        }) = GraphWrite::edge_weight_mut(&mut graph.graph, *edge_index)
        {
            stamped_types.insert(*connection_type);
            for (key, val) in &interned {
                if let Some((_, existing)) = edge_props.iter_mut().find(|(ek, _)| ek == key) {
                    *existing = val.clone();
                } else {
                    edge_props.push((*key, val.clone()));
                }
            }
        }
        graph.refresh_rel_indexes(rel_index_pre);
        for (name, _) in &provenance {
            crate::graph::index_freshness::write_hooks::note_edge_property_written(
                graph,
                *edge_index,
                Some(name),
            );
        }
    }
    // Once per stamped type, as `stamp_node_provenance` does: a type opted in
    // after its edges were written, or a key its creating write did not carry,
    // is otherwise stamped without its catalogue knowing the key.
    for type_key in stamped_types {
        let conn_type = graph.interner.resolve(type_key).to_string();
        for (name, value) in &provenance {
            graph.record_connection_property_type(&conn_type, name, value.type_name());
        }
    }
}

/// Apply one `SET n:Label` item to a row.
fn set_node_label(
    graph: &mut DirGraph,
    row: &ResultRow,
    item: (&str, &str),
    stats: &mut MutationStats,
    (nodes_to_stamp, bound_writes): (&mut HashMap<NodeIndex, String>, &mut BoundWrites),
) -> Result<(), String> {
    let (variable, label) = item;
    // Null target (OPTIONAL MATCH miss): no-op for this row.
    let Some(node_idx) = resolve_node_write_target(graph, row, variable, "SET")? else {
        return Ok(());
    };
    // Adding a secondary label is a write to the node, judged by its *stored*
    // type — the label being added never widens the scope (that is the
    // label-smuggling defence).
    enforce_node_write_scope(graph, node_idx)?;
    let key = graph.interner.get_or_intern(label);
    if graph.add_node_label(node_idx, key) {
        bound_writes.node_label(graph, node_idx, label);
        stats.properties_set += 1;
        // A label add is a modification — bump `updated_at` if the node's type
        // opted in (same post-loop stamp as a property SET).
        if let Some(nt) = auto_timestamp_type_of(graph, node_idx) {
            nodes_to_stamp.insert(node_idx, nt);
        }
    }
    Ok(())
}

/// Stamp the reserved provenance keys on every node a `SET` modified.
fn stamp_node_provenance(graph: &mut DirGraph, nodes_to_stamp: &HashMap<NodeIndex, String>) {
    // updated_at + the caller's git_sha/modified_by, once per node of an
    // opted-in type, on one clock read for the whole SET. Writes through the
    // in-memory columnar master (fast path) or the per-node setter; the
    // type-schema slot + metadata are registered so they persist. No
    // equality-index update — provenance is range-queried, not equality-matched.
    if !nodes_to_stamp.is_empty() {
        let prov = graph.provenance_props();
        for (node_idx, node_type) in nodes_to_stamp {
            // Scoped so the read ends before the &mut writes below.
            let columnar_row_id = {
                graph
                    .graph
                    .node_weight(*node_idx)
                    .and_then(|n| n.properties.columnar_row_id())
            };
            let type_key = InternedKey::from_str(node_type);
            for &(pname, ref pval) in &prov {
                let key = graph.interner.get_or_intern(pname);
                // Read-check first — see the note in
                // `set_row::finish_node_property_write`.
                let needs_key = graph
                    .type_schemas
                    .get(node_type)
                    .is_some_and(|schema| schema.slot(key).is_none());
                if needs_key {
                    if let Some(schema_arc) = graph.type_schemas_mut().get_mut(node_type) {
                        Arc::make_mut(schema_arc).add_key(key);
                    }
                }
                let wrote = set_via_column_master(
                    graph,
                    ColumnMasterWrite {
                        node_idx: *node_idx,
                        node_type,
                        type_key,
                        property: pname,
                        key,
                        value: pval,
                        row_id: columnar_row_id,
                    },
                );
                if !wrote {
                    GraphWrite::set_node_property(&mut graph.graph, *node_idx, key, pval.clone());
                }
            }
            // A property SET reached this stamp through
            // `finish_node_property_write`, which already notified; `SET n:Label`
            // did not, and this is still a write to the row a disk graph's
            // persistent bundles snapshot.
            crate::graph::index_freshness::write_hooks::note_property_written(
                graph, *node_idx, node_type, None,
            );
        }
        // The catalogue entry for each provenance key is a fact about the
        // *type*, not about the node — so it is recorded once per stamped type
        // rather than once per stamped node per key.
        let stamped_types: std::collections::HashSet<&String> = nodes_to_stamp.values().collect();
        let prop_types: HashMap<String, String> = prov
            .iter()
            .map(|(pname, pval)| (pname.to_string(), value_type_name(pval)))
            .collect();
        for node_type in stamped_types {
            graph.upsert_node_type_metadata(node_type, prop_types.clone());
        }
    }
}

fn execute_remove(
    graph: &mut DirGraph,
    remove: &RemoveClause,
    result_set: &ResultSet,
    stats: &mut MutationStats,
    interrupt: &Interrupt,
) -> Result<(), String> {
    for row in result_set.rows.iter() {
        check_interrupt_row(interrupt)?;
        for item in &remove.items {
            match item {
                RemoveItem::Property { variable, property } => {
                    // Relationship property REMOVE is its own path, for the
                    // same reasons the SET counterpart is.
                    if remove_edge_property(graph, row, variable, property, stats)? {
                        flush_disk_item_writes(graph);
                        continue;
                    }

                    if property == "id" {
                        return Err("Cannot REMOVE node id — it is immutable".to_string());
                    }

                    // A null-valued target (OPTIONAL MATCH miss) makes this
                    // row's REMOVE a no-op per openCypher.
                    let Some(node_idx) = resolve_node_write_target(graph, row, variable, "REMOVE")?
                    else {
                        continue;
                    };

                    // Read node_type before mutable borrow (for index update)
                    let node_type_str = graph
                        .node_view(node_idx)
                        .map(|n| n.get_node_type_ref(&graph.interner).to_string())
                        .unwrap_or_default();

                    // Role-scoped write guard, on the node's *stored* type —
                    // stripping a property off a node is a write to it, so it
                    // is judged exactly as `SET n.p` is. Before every gate and
                    // every write below, so a refusal leaves storage untouched.
                    enforce_write_scope(graph, &node_type_str)?;

                    let write_field = remove_write_field(graph, &node_type_str, property.as_str())?;

                    // Declared NOT NULL / UNIQUE gates. Removing a required
                    // property is a violation; removing part of a unique tuple
                    // just vacates it. Planned before the write so a rejection
                    // leaves storage untouched.
                    graph
                        .check_property_write_deferring(
                            &node_type_str,
                            node_idx,
                            property,
                            None,
                            true,
                        )
                        .map_err(|violation| violation.to_string())?;
                    graph.note_required_node(&node_type_str, node_idx);

                    let is_disk = graph.graph.is_disk();

                    // In-memory Columnar: clear through the graph master, the
                    // same chokepoint `execute_set` writes through. Going
                    // through the node's own `Arc<ColumnStore>` instead (what
                    // `PropertyStorage::remove` does) is both wrong and slow:
                    // `Arc::make_mut` forks the node's store, so the master
                    // keeps the removed value and can resurrect it with no save
                    // involved, and the fork is a full `ColumnStore` clone per
                    // node removed — O(R x N) for R rows of an N-node type.
                    // title/name are excluded as a routing detail, exactly as in
                    // [`set_via_column_master`].
                    let mut cleared_via_master = None;
                    if !is_disk && write_field != "name" && write_field != "title" {
                        let columnar_row_id = {
                            graph
                                .graph
                                .node_weight(node_idx)
                                .and_then(|n| n.properties.columnar_row_id())
                        };
                        if let Some(row_id) = columnar_row_id {
                            let key = graph.interner.get_or_intern(write_field);
                            // Same primitive `SET` uses, for the same reason:
                            // it captures the pre-statement store into the undo
                            // journal *before* mutating. Writing the master
                            // directly leaves a failed statement nothing to roll
                            // back to, now that the backend is the store's sole
                            // owner and the write lands in place.
                            cleared_via_master = write_column_master(
                                graph,
                                MasterCell {
                                    node_type: &node_type_str,
                                    type_key: InternedKey::from_str(&node_type_str),
                                    node_idx,
                                    row_id,
                                    key,
                                    value: &Value::Null,
                                },
                                PriorCell::Read,
                            );
                        }
                    }

                    // `(title, name)` as they stood before a `name` / `title`
                    // removal, which clears both: `name` is the stored property
                    // if the node has one and the title otherwise, so the two
                    // can sit in different buckets.
                    let mut cleared_name_title: Option<(Option<Value>, Option<Value>)> = None;
                    let removed_value = if let Some(prior) = cleared_via_master {
                        prior
                    } else if graph.graph.node_weight(node_idx).is_some() {
                        if write_field == "name" || write_field == "title" {
                            // Read the *resolved* title (a columnar node keeps
                            // it in its store's reserved column, with the
                            // inline field on the `Null` sentinel) and clear it
                            // through the same backend seam a `SET` writes it
                            // through, so the removal reaches the store instead
                            // of nulling an already-null field.
                            let (removed, cleared) = clear_name_and_title(graph, node_idx);
                            cleared_name_title = Some(cleared);
                            removed
                        } else {
                            // The backend picks the right removal semantics per
                            // storage: disk stages a `Null` write so its flush
                            // propagates the removal; row storage drops the key.
                            let key = InternedKey::from_str(write_field);
                            GraphWrite::remove_node_property(&mut graph.graph, node_idx, key)
                        }
                    } else {
                        None
                    };

                    // Composite maintenance reads the current tuple from disk columns.
                    flush_disk_item_writes(graph);
                    if let Some(cleared) = &cleared_name_title {
                        refile_cleared_name_title(graph, &node_type_str, node_idx, cleared);
                    }
                    if let Some(old_val) = removed_value {
                        stats.properties_removed += 1;
                        if cleared_name_title.is_none() {
                            graph.update_property_indices_for_remove(
                                &node_type_str,
                                node_idx,
                                property,
                                &old_val,
                            );
                        }
                        // Stripping the indexed property is a content change
                        // like any other; the next refresh will find no string
                        // and drop the document.
                        crate::graph::index_freshness::write_hooks::note_property_written(
                            graph,
                            node_idx,
                            &node_type_str,
                            Some(write_field),
                        );
                    }
                    graph.note_ontology_touch(node_idx);
                }
                RemoveItem::Label {
                    variable, label, ..
                } => {
                    let Some(&node_idx) = row.node_bindings.get(variable) else {
                        // Null target (OPTIONAL MATCH miss): no-op for this row.
                        if is_null_write_target(row, variable) {
                            continue;
                        }
                        return Err(format!(
                            "Variable '{}' not bound to a node in REMOVE",
                            variable
                        ));
                    };
                    // A secondary label is node state; removing one is a write
                    // to the node, judged by its stored type.
                    enforce_node_write_scope(graph, node_idx)?;
                    let key = graph.interner.get_or_intern(label);
                    if graph.remove_node_label(node_idx, key)? {
                        stats.properties_removed += 1;
                    }
                }
            }
            flush_disk_item_writes(graph);
        }
    }

    Ok(())
}

fn execute_merge(
    graph: &mut DirGraph,
    merge: &MergeClause,
    existing: ResultSet,
    stats: &mut MutationStats,
    ctx: &WriteClauseCtx<'_>,
) -> Result<ResultSet, String> {
    // As in `execute_create`: one MERGE per incoming row, none for zero rows.
    // The implicit start row for a leading MERGE is seeded by the pipeline.
    let source_rows = existing.rows;
    let folded = (!source_rows.is_empty())
        .then(|| super::write_folding::fold_merge(&row_evaluator(graph, ctx), merge));
    let merge = folded.as_ref().unwrap_or(merge);

    let mut new_rows = Vec::with_capacity(source_rows.len());
    let mut row_pattern = merge_row_scratch(&merge.pattern);

    for mut new_row in source_rows.into_iter() {
        check_interrupt_row(ctx.interrupt)?;
        // (Block-scoped: the executor's borrow of `graph` must end before the
        // &mut mutation calls below.)
        let pattern = {
            let executor = row_evaluator(graph, ctx);
            evaluate_merge_pattern(graph, &executor, &merge.pattern, &mut row_pattern, &new_row)?
        };
        let matches = merge_pattern::try_match_merge_pattern(graph, pattern, &new_row, ctx)?;

        if !matches.is_empty() {
            // One output row per match; the pattern's bindings differ per row.
            for bound_row in matches {
                let mut matched_row = new_row.clone();
                for (var, idx) in &bound_row.node_bindings {
                    matched_row.node_bindings.insert(var.clone(), *idx);
                }
                for (var, binding) in &bound_row.edge_bindings {
                    matched_row.edge_bindings.insert(var.clone(), *binding);
                }

                if let Some(ref set_items) = merge.on_match {
                    matched_row = apply_merge_set(graph, set_items, matched_row, stats, ctx)?;
                }
                new_rows.push(matched_row);
            }
            continue;
        }

        let create_clause = CreateClause {
            patterns: vec![pattern.clone()],
        };
        let temp_rs = ResultSet {
            rows: vec![new_row.clone()],
            columns: existing.columns.clone(),
            lazy_return_items: None,
        };
        let created = execute_folded_create(graph, &create_clause, temp_rs, stats, ctx)?;

        if let Some(created_row) = created.rows.into_iter().next() {
            for (var, idx) in created_row.node_bindings {
                new_row.node_bindings.insert(var, idx);
            }
            for (var, binding) in created_row.edge_bindings {
                new_row.edge_bindings.insert(var, binding);
            }
        }

        if let Some(ref set_items) = merge.on_create {
            new_row = apply_merge_set(graph, set_items, new_row, stats, ctx)?;
        }

        new_rows.push(new_row);
    }

    Ok(ResultSet {
        rows: new_rows,
        columns: existing.columns,
        lazy_return_items: None,
    })
}

/// A MERGE's `ON MATCH` / `ON CREATE` SET over one row. The row comes back
/// with the node and relationship values the SET names re-read from the
/// store, as after a plain SET clause: a projected node (`UNWIND collect(n) AS
/// x`) is a snapshot, and `RETURN x.p` must show the written value.
fn apply_merge_set(
    graph: &mut DirGraph,
    set_items: &[SetItem],
    row: ResultRow,
    stats: &mut MutationStats,
    ctx: &WriteClauseCtx<'_>,
) -> Result<ResultRow, String> {
    let set_clause = SetClause {
        items: set_items.to_vec(),
    };
    let mut rows = ResultSet {
        rows: vec![row],
        columns: Vec::new(),
        lazy_return_items: None,
    };
    execute_folded_set(graph, &set_clause, &rows, stats, ctx)?;
    GraphWrite::flush_pending_writes(&mut graph.graph);
    refresh_written_projections(graph, &mut rows, &set_clause_variables(&set_clause));
    Ok(rows.rows.pop().expect("one row in, one row out"))
}

/// The clause's pattern, cloned once, whose row-dependent property slots
/// [`evaluate_merge_pattern`] overwrites with each row's values. `None` when
/// `fold_merge` already made every property a literal: rows then use the
/// clause's pattern itself.
fn merge_row_scratch(pattern: &CreatePattern) -> Option<CreatePattern> {
    let all_literal = pattern.elements.iter().all(|element| {
        merge_element_properties(element)
            .iter()
            .all(|(_, expression)| matches!(expression, Expression::Literal(_)))
    });
    (!all_literal).then(|| pattern.clone())
}

fn merge_element_properties(element: &CreateElement) -> &[(String, Expression)] {
    match element {
        CreateElement::Node(node) => &node.properties,
        CreateElement::Edge(edge) => &edge.properties,
    }
}

/// One row's MERGE pattern with every property already a value: each
/// row-dependent expression is evaluated once here and handed to both the
/// match and the create arm as a literal, so the statement's budget is charged
/// once per row and a non-deterministic key (`rand()`) is created with the
/// value the match looked for. The values land in `scratch` (from
/// [`merge_row_scratch`]) in place, so matching a row allocates no pattern or
/// name (the create arm still clones the pattern into its clause, as before);
/// with no scratch every property is a literal and `pattern` is returned.
///
/// Equality against null is undefined, so a null-bearing MERGE key identifies
/// neither a match nor a safe entity to create — refused here. The non-identity
/// values are snapshotted before the null test, as both arms store and compare
/// them; the arms snapshot again, which leaves a resolved value unchanged.
fn evaluate_merge_pattern<'p>(
    graph: &DirGraph,
    executor: &CypherExecutor<'_>,
    pattern: &'p CreatePattern,
    scratch: &'p mut Option<CreatePattern>,
    row: &ResultRow,
) -> Result<&'p CreatePattern, String> {
    for (position, element) in pattern.elements.iter().enumerate() {
        let identity = match element {
            CreateElement::Node(node) => Some(IdentityAliases::for_type(
                graph,
                node.label.as_deref().unwrap_or("Node"),
            )),
            CreateElement::Edge(_) => None,
        };
        for (slot, (name, expression)) in merge_element_properties(element).iter().enumerate() {
            let mut value = executor.evaluate_expression(expression, row)?;
            check_value(&value)?;
            let is_identity = identity
                .as_ref()
                .is_some_and(|aliases| aliases.canonical(name) == "id");
            if !is_identity {
                crate::graph::session::snapshot_property_values(
                    &graph.graph,
                    std::iter::once(&mut value),
                );
            }
            if matches!(value, Value::Null) {
                return Err(format!("MERGE cannot use null for property '{}'", name));
            }
            if matches!(expression, Expression::Literal(_)) {
                continue;
            }
            if let Some(scratch) = scratch.as_mut() {
                let properties = match &mut scratch.elements[position] {
                    CreateElement::Node(node) => &mut node.properties,
                    CreateElement::Edge(edge) => &mut edge.properties,
                };
                properties[slot].1 = Expression::Literal(value);
            }
        }
    }
    Ok(match scratch {
        Some(scratch) => scratch,
        None => pattern,
    })
}

#[cfg(test)]
#[path = "write_remove_columnar_tests.rs"]
mod remove_columnar_tests;

#[cfg(test)]
#[path = "write_rel_constraint_tests.rs"]
mod rel_constraint_tests;

#[cfg(test)]
#[path = "write_update_tail_tests.rs"]
mod update_tail_tests;

#[cfg(test)]
#[path = "write_mutation_query_tests.rs"]
mod is_mutation_query_tests;

#[cfg(test)]
#[path = "write_deadline_tests.rs"]
mod deadline_tests;

#[cfg(test)]
#[path = "write_merge_scratch_tests.rs"]
mod merge_scratch_tests;
