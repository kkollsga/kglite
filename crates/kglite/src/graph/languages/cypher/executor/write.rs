//! Cypher mutation execution — execute_mutable + per-clause helpers
//! (execute_create, execute_set, execute_delete, execute_remove, execute_merge).

use super::super::ast::*;
use super::super::result::*;
use super::columnar_write::{
    set_via_column_master, write_column_master, ColumnMasterWrite, MasterCell, PriorCell,
};
use super::edge_property_write::{remove_edge_property, set_edge_property};
use super::identity_fields::{
    check_identity_uniqueness, create_identity, merge_expected_props, remove_write_field,
    CreatedIdentity, IdentityAliases,
};
use super::set_row::{apply_node_property_set, NodePropertySet, SetMemos};
use super::write_scope::{
    enforce_bound_edge_write_scope, enforce_edge_write_scope, enforce_node_write_scope,
    enforce_write_scope,
};
use super::{clause_display_name, schema_ddl, CypherExecutor};
use crate::datatypes::values::Value;
use crate::graph::algorithms::Interrupt;
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

/// Run a change-capture lifecycle procedure (`db.cdc.enable` / `db.cdc.disable`)
/// on the write engine.
///
/// Split out of the clause pipeline, which is at its complexity ceiling.
fn execute_cdc_lifecycle_call(
    graph: &mut DirGraph,
    call: &crate::graph::languages::cypher::ast::CallClause,
    params: &HashMap<String, Value>,
    interrupt: &Interrupt,
    budget: &super::budget::ExecutionBudget,
) -> Result<ResultSet, String> {
    let proc_name = call.procedure_name.to_lowercase();
    // The registry answers "what does this yield?" for both engines, so a bare
    // CALL expands and an unknown column is refused exactly as on the read path.
    let yield_items = super::call_clause::resolve_yield_items(
        &proc_name,
        &call.procedure_name,
        &call.yield_items,
    )?;
    // Arguments are evaluated by the read executor: a CALL argument is an
    // ordinary expression over no rows, and duplicating that evaluation here is
    // how the two paths would start accepting different argument forms.
    let params_map = {
        let executor = CypherExecutor::with_params(graph, params, interrupt.deadline)
            .with_cancel(interrupt.cancel)
            .with_budget(budget.clone());
        executor.extract_call_params(&call.parameters)?
    };
    let rows = if proc_name.starts_with("table.") {
        super::table_procedures::execute_table_procedure(
            graph,
            &proc_name,
            &params_map,
            &yield_items,
        )?
    } else {
        super::cdc_procedures::execute_mutating_procedure(
            graph,
            &proc_name,
            &params_map,
            &yield_items,
        )?
    };
    Ok(ResultSet {
        columns: yield_items
            .iter()
            .map(|item| item.alias.clone().unwrap_or_else(|| item.name.clone()))
            .collect(),
        rows,
        lazy_return_items: None,
    })
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
}

/// Invariant context for one mutation run — everything the clause loop reads
/// but never changes. Bundled so [`run_clause_pipeline`] can be called once
/// per `LOAD CSV` batch without a ten-argument signature.
pub(super) struct MutationCtx<'a> {
    pub params: &'a HashMap<String, Value>,
    pub interrupt: &'a Interrupt,
    pub budget: &'a super::budget::ExecutionBudget,
    pub profiling: bool,
    pub csv_import: &'a super::load_csv::CsvImportPolicy,
    /// Clauses that precede the pipeline being run — non-empty only for the
    /// `LOAD CSV` suffix, where the stripped `LOAD CSV … AS row` still
    /// declares `row` for correlated-subquery import validation.
    pub leading: &'a [Clause],
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
    /// Variables declared by everything before `clauses[i]`, including the
    /// stripped leading clauses.
    fn declared_before(&self, clauses: &[Clause], i: usize) -> std::collections::HashSet<String> {
        use crate::graph::languages::cypher::planner::simplification::declared_variables;
        let mut declared = declared_variables(self.leading);
        declared.extend(declared_variables(&clauses[..i]));
        declared
    }
}

/// Poll the abort check at the read path's cadence — every
/// `INTERRUPT_POLL_INTERVAL` units, starting before the first — and report
/// through the read path's [`check_interrupt`](super::check_interrupt), so a
/// timed-out write says "timed out" and a cancelled one says "cancelled".
#[inline]
fn check_interrupt_periodic(interrupt: &Interrupt, iteration: usize) -> Result<(), String> {
    if iteration & (super::INTERRUPT_POLL_INTERVAL - 1) == 0 {
        super::check_interrupt(interrupt)?;
    }
    Ok(())
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
        },
        &super::load_csv::CsvImportPolicy::Denied,
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
) -> Result<CypherResult, String> {
    // Arena guard for the whole mutation: holds the query count so every
    // materializing read (`get_node` → `node_weight`) inside a mutation clause
    // is guard-covered. Owned counter handle — coexists with `&mut`.
    let _arena_guard = graph.graph.begin_query();

    let budget = super::budget::ExecutionBudget::new(limits.max_work_units);

    let mut stats = MutationStats::default();
    let profiling = query.profile;
    let mut profile_stats: Vec<ClauseStats> = Vec::new();

    // `LOAD CSV` drives the rest of the pipeline over bounded row batches
    // instead of being executed as a clause — the whole point is that peak
    // memory must not scale with file size. See `executor/load_csv.rs`.
    let result_set = if let Some(Clause::LoadCsv(load)) = query.clauses.first() {
        let suffix = &query.clauses[1..];
        let ctx = MutationCtx {
            params: &params,
            interrupt: &interrupt,
            budget: &budget,
            profiling,
            csv_import,
            leading: &query.clauses[..1],
        };
        let source = {
            let executor = CypherExecutor::with_params(graph, &params, interrupt.deadline)
                .with_cancel(interrupt.cancel)
                .with_budget(budget.clone());
            executor.evaluate_expression(&load.source, &ResultRow::new())?
        };
        let barrier = super::load_csv::batching_barrier(suffix);
        super::load_csv::drive(load, &source, csv_import, barrier.as_deref(), |seed| {
            let mut batch_profile = Vec::new();
            let out =
                run_clause_pipeline(graph, suffix, seed, &ctx, &mut stats, &mut batch_profile)?;
            merge_profile(&mut profile_stats, batch_profile);
            Ok(out)
        })?
    } else {
        let ctx = MutationCtx {
            params: &params,
            interrupt: &interrupt,
            budget: &budget,
            profiling,
            csv_import,
            leading: &[],
        };
        run_clause_pipeline(
            graph,
            &query.clauses,
            ResultSet::new(),
            &ctx,
            &mut stats,
            &mut profile_stats,
        )?
    };

    finalize_mutation(
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
    )
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

    for (i, clause) in clauses.iter().enumerate() {
        // The mutation is atomic: aborting here discards the in-flight
        // changes, leaving the graph unchanged.
        super::check_interrupt(&interrupt)?;
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

        match clause {
            Clause::Create(create) => {
                result_set = execute_create(graph, create, result_set, params, stats, interrupt)?;
            }
            Clause::Set(set) => {
                execute_set(graph, set, &result_set, params, stats, interrupt)?;
                // Flush staged writes so any subsequent clause's reads
                // (including a trailing RETURN's projection) observe the SET.
                // On disk, SET stages into `node_mut_cache`; without the flush
                // the next `node_weight` reads `column_stores` and returns the
                // pre-SET values.
                GraphWrite::flush_pending_writes(&mut graph.graph);
            }
            Clause::Delete(del) => {
                execute_delete(graph, del, &result_set, stats, interrupt)?;
            }
            Clause::Remove(rem) => {
                execute_remove(graph, rem, &result_set, stats, interrupt)?;
                // Same rationale as SET — REMOVE stages on disk too.
                GraphWrite::flush_pending_writes(&mut graph.graph);
            }
            Clause::Merge(merge) => {
                result_set = execute_merge(graph, merge, result_set, params, stats, interrupt)?;
                // MERGE may run ON MATCH SET / ON CREATE SET; flush as above.
                GraphWrite::flush_pending_writes(&mut graph.graph);
            }
            Clause::Foreach {
                variable,
                list,
                body,
            } => {
                execute_foreach(
                    graph,
                    variable,
                    list,
                    body,
                    &result_set,
                    params,
                    stats,
                    interrupt,
                    budget,
                )?;
                GraphWrite::flush_pending_writes(&mut graph.graph);
            }
            // Correlated CALL { } import validation needs the declared outer
            // scope (variables bound by clauses 0..i), distinct from the
            // bindings present in any single row.
            Clause::CallSubquery { import, body } => {
                let executor = CypherExecutor::with_params(graph, params, interrupt.deadline)
                    .with_cancel(interrupt.cancel)
                    .with_budget(budget.clone())
                    .with_csv_import(ctx.csv_import.clone());
                let declared = ctx.declared_before(clauses, i);
                result_set = executor.execute_call_subquery(import, body, result_set, &declared)?;
            }
            // Change-capture lifecycle verbs run here, not on the read engine
            // (see `clause_is_mutation`); read CDC procedures fall through to
            // the read executor below like every other `CALL`.
            Clause::Call(call)
                if super::procedure_registry::is_mutating_procedure(
                    &call.procedure_name.to_lowercase(),
                ) =>
            {
                result_set = execute_cdc_lifecycle_call(graph, call, params, interrupt, budget)?;
            }
            // Schema DDL runs here, not on the read engine (see
            // `clause_is_mutation`); the `SHOW` forms classify as reads and
            // never reach this arm.
            Clause::Schema(command) => {
                schema_ddl::execute_schema_mutation(graph, command, stats, interrupt)?;
            }
            _ => {
                let executor = CypherExecutor::with_params(graph, params, interrupt.deadline)
                    .with_cancel(interrupt.cancel)
                    .with_budget(budget.clone())
                    .with_csv_import(ctx.csv_import.clone());
                result_set = executor.execute_single_clause(clause, result_set)?;
            }
        }

        budget.check_rows(result_set.rows.len(), &clause_display_name(clause))?;
        let mutation_units = stats
            .nodes_created
            .checked_add(stats.relationships_created)
            .and_then(|n| n.checked_add(stats.properties_set))
            .and_then(|n| n.checked_add(stats.nodes_deleted))
            .and_then(|n| n.checked_add(stats.relationships_deleted))
            .and_then(|n| n.checked_add(stats.properties_removed))
            .ok_or_else(|| "Mutation work counter overflow".to_string())?;
        budget.check_work(mutation_units, "mutation clauses")?;

        if let Some(s) = start {
            profile_stats.push(ClauseStats {
                clause_name: clause_display_name(clause),
                rows_in,
                rows_out: result_set.rows.len(),
                elapsed_us: s.elapsed().as_micros() as u64,
            });
        }

        // From here on, `rows.is_empty()` means "zero rows", never "not started".
        stream_established = true;
    }

    Ok(result_set)
}

/// Does `clause` consume Cypher's implicit single empty start row when it opens
/// a query?
///
/// True for the clauses that produce output per incoming row and so need one to
/// act on with nothing before them. Read clauses are absent on purpose —
/// MATCH/OPTIONAL MATCH open a query by scanning, not by extending a row.
///
/// Only ever consulted while the stream is unestablished, so it cannot
/// resurrect a stream that a preceding clause emptied.
fn clause_needs_implicit_row(clause: &Clause) -> bool {
    matches!(
        clause,
        Clause::With(_)
            | Clause::Unwind(_)
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

    if has_return || !result_set.columns.is_empty() {
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
#[allow(clippy::too_many_arguments)]
fn execute_foreach(
    graph: &mut DirGraph,
    variable: &str,
    list: &Expression,
    body: &[Clause],
    outer: &ResultSet,
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
    interrupt: &Interrupt,
    budget: &super::budget::ExecutionBudget,
) -> Result<(), String> {
    for (row_idx, row) in outer.rows.iter().enumerate() {
        check_interrupt_periodic(interrupt, row_idx)?;
        // Evaluate the list in this row's context (read-only borrow of the
        // graph, dropped before the per-element mutations below).
        let list_val = {
            let executor = CypherExecutor::with_params(graph, params, interrupt.deadline)
                .with_cancel(interrupt.cancel);
            executor.evaluate_expression(list, row)?
        };
        let items = match list_val {
            Value::List(items) => items,
            // FOREACH over null is a no-op (Neo4j semantics).
            Value::Null => continue,
            other => {
                return Err(format!("FOREACH expects a list, got {}", other.type_name()));
            }
        };

        budget.check_work(items.len(), "FOREACH")?;

        for (item_idx, item) in items.into_iter().enumerate() {
            check_interrupt_periodic(interrupt, item_idx)?;
            let mut elem_row = row.clone();
            elem_row.projected.insert(variable.to_string(), item);
            let mut elem_set = ResultSet {
                rows: vec![elem_row],
                columns: outer.columns.clone(),
                lazy_return_items: None,
            };
            for bclause in body {
                elem_set = apply_foreach_body_clause(
                    graph, bclause, elem_set, params, stats, interrupt, budget,
                )?;
            }
        }
    }
    Ok(())
}

/// Apply one clause inside a FOREACH body. Only update clauses and nested
/// FOREACH are valid (the parser enforces this; the catch-all is a guard).
/// Mirrors the per-clause mutation handling (incl. disk flush/sync) from
/// `run_clause_pipeline`.
fn apply_foreach_body_clause(
    graph: &mut DirGraph,
    clause: &Clause,
    result_set: ResultSet,
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
    interrupt: &Interrupt,
    budget: &super::budget::ExecutionBudget,
) -> Result<ResultSet, String> {
    // The flush is per element, not per clause: on disk a property read in the
    // same or a later iteration (e.g. `coalesce(n.hits, 0)`) reads the type's
    // column store, which only reflects a staged write once
    // `flush_pending_writes` has drained `node_mut_cache`. Reducing it safely
    // needs the disk read path to consult the mut-cache, a deeper storage
    // change. (Memory mode pays ~nothing.)
    match clause {
        Clause::Create(create) => {
            execute_create(graph, create, result_set, params, stats, interrupt)
        }
        Clause::Set(set) => {
            execute_set(graph, set, &result_set, params, stats, interrupt)?;
            GraphWrite::flush_pending_writes(&mut graph.graph);
            Ok(result_set)
        }
        Clause::Delete(del) => {
            execute_delete(graph, del, &result_set, stats, interrupt)?;
            GraphWrite::flush_pending_writes(&mut graph.graph);
            Ok(result_set)
        }
        Clause::Remove(rem) => {
            execute_remove(graph, rem, &result_set, stats, interrupt)?;
            GraphWrite::flush_pending_writes(&mut graph.graph);
            Ok(result_set)
        }
        Clause::Merge(merge) => {
            let rs = execute_merge(graph, merge, result_set, params, stats, interrupt)?;
            GraphWrite::flush_pending_writes(&mut graph.graph);
            Ok(rs)
        }
        Clause::Foreach {
            variable,
            list,
            body,
        } => {
            execute_foreach(
                graph,
                variable,
                list,
                body,
                &result_set,
                params,
                stats,
                interrupt,
                budget,
            )?;
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
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
    interrupt: &Interrupt,
) -> Result<ResultSet, String> {
    // On disk, node properties route through the per-type ColumnStore in
    // `DirGraph::insert_node_routed`, which writes into the store the backend
    // owns — no second copy to sync, unlike the SET/REMOVE staged-write path.
    // One CREATE per incoming row, and nothing at all for zero rows. A leading
    // CREATE gets its single empty row from the pipeline's implicit-start-row
    // seed (`clause_needs_implicit_row`) — this function must not synthesize
    // one, because from here an empty `existing` is indistinguishable from a
    // preceding MATCH that found nothing.
    let source_rows = existing.rows;

    let mut new_rows = Vec::with_capacity(source_rows.len());

    for (row_idx, row) in source_rows.iter().enumerate() {
        check_interrupt_periodic(interrupt, row_idx)?;
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
                    // If the variable is already bound — by a prior MATCH or by
                    // an earlier part of this same CREATE — this occurrence
                    // references that node instead of creating a second one.
                    if let Some(var) = node_pat.variable.as_deref() {
                        if let Some(&bound) = new_row.node_bindings.get(var) {
                            element_nodes[pos] = Some(bound);
                            continue;
                        }
                    }

                    let node_idx = create_node(graph, node_pat, &new_row, params, stats)?;

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
            create_pattern_edges(graph, pattern, &element_nodes, &mut new_row, params, stats)?;
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
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
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
                let executor = CypherExecutor::with_params(graph, params, None);
                for (key, expr) in &edge_pat.properties {
                    let val = executor.evaluate_expression(expr, new_row)?;
                    if matches!(val, Value::Null) {
                        continue;
                    }
                    edge_props.insert(key.clone(), val);
                }
            }
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
                graph.check_rel_row(&edge_pat.connection_type, |property| {
                    edge_props.get(property).cloned()
                })?;
            }

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

            if let Some(ref var) = edge_pat.variable {
                new_row.edge_bindings.insert(
                    var.clone(),
                    EdgeBinding {
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

/// Refuse creating a node whose primary type is a declared **abstract**
/// class, naming the concrete declared subtypes (built only on the cold
/// path).
pub(crate) fn reject_abstract_create(graph: &DirGraph, label: &str) -> Result<(), String> {
    let Some(decl) = graph.ontology.classes.get(label) else {
        return Ok(());
    };
    if !decl.is_abstract {
        return Ok(());
    }
    let mut concrete: Vec<&str> = graph
        .ontology
        .classes
        .iter()
        .filter(|(name, d)| {
            !d.is_abstract && graph.ontology.ancestors(name).iter().any(|a| a == label)
        })
        .map(|(name, _)| name.as_str())
        .collect();
    concrete.sort_unstable();
    Err(if concrete.is_empty() {
        format!("'{label}' is an abstract ontology class and cannot hold nodes")
    } else {
        format!(
            "'{label}' is an abstract ontology class — create one of: {}",
            concrete.join(", ")
        )
    })
}

fn create_node(
    graph: &mut DirGraph,
    node_pat: &CreateNodePattern,
    row: &ResultRow,
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
) -> Result<petgraph::graph::NodeIndex, String> {
    // Evaluate property expressions (borrow graph immutably, then drop)
    let mut properties = HashMap::new();
    {
        let executor = CypherExecutor::with_params(graph, params, None);
        for (key, expr) in &node_pat.properties {
            let val = executor.evaluate_expression(expr, row)?;
            properties.insert(key.clone(), val);
        }
    }

    let label = node_pat.label.clone().unwrap_or_else(|| "Node".to_string());

    // Identity fields, under whichever spelling this node type declares — see
    // [`IdentityAliases`] and [`create_identity`].
    let aliases = IdentityAliases::for_type(graph, &label);
    let CreatedIdentity {
        id,
        title,
        title_supplied,
    } = create_identity(graph, node_pat, &label, &aliases, &mut properties)?;

    check_identity_uniqueness(graph, &label, &id)?;

    // A constraint declared on the type's *own* id/title spelling (`REQUIRE
    // p.name IS UNIQUE` on a type loaded with `node_title_field='name'`) reads
    // the value this CREATE is putting in the identity field — that spelling no
    // longer has a property key of its own, the value having been promoted.
    // The title alias answers only when the caller *supplied* the title: the
    // `<Label>_<n>` fallback is engine-minted, and a `REQUIRE … IS NOT NULL` on
    // the type's title column is asking the caller for a value.
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
    // Bind before reporting: `record_constraint_violation` needs `&mut graph`,
    // so the immutable borrow held by the check must end at the semicolon.
    let required = graph.check_required_fields(&label, constraint_read);
    if let Err(violation) = required {
        return Err(graph.record_constraint_violation(*violation));
    }
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
        )?;
    }

    // Every backend writes id/title/properties through the per-type
    // ColumnStore — see `DirGraph::insert_node_routed`.
    let node_idx = graph.insert_node_routed(id, title, &label, properties);

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

    // Keep the id-index consistent. Whenever the index is already cached — and
    // therefore complete — insert into it, so a sequential CREATE (e.g.
    // UNWIND … CREATE) stays O(1)/node instead of paying an O(n)
    // rebuild-per-node. When it isn't cached, invalidate for lazy rebuild: the
    // `contains_key` guard is what stops us building a *partial* entry that
    // `build_id_index` would later short-circuit on and trust as complete.
    //
    // Deliberately independent of whether a primary key is declared — the
    // declaration governs *uniqueness enforcement*, not index freshness, and
    // gating maintenance on it dropped an undeclared type's whole cached id
    // index on every single CREATE. Nothing about duplicate ids is protected by
    // invalidating: a rebuild and an incremental insert collapse a duplicate
    // identically.
    match pk_id {
        Some(idv) if graph.id_indices.contains_key(&label) => {
            graph
                .id_indices
                .entry_or_default(label.clone())
                .insert(idv, node_idx);
        }
        _ => {
            graph.id_indices.remove(&label);
        }
    }

    graph.update_property_indices_for_add(&label, node_idx);
    // Claim the unique tuples validated above, now that the node exists.
    graph.commit_unique_claims(&unique_claims, node_idx);

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

fn get_create_node_variable(element: &CreateElement) -> Option<&str> {
    match element {
        CreateElement::Node(np) => np.variable.as_deref(),
        _ => None,
    }
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
/// `None`. Arena guard: `node_view` materializes on the disk backend, so the
/// read is scoped and the borrow ends before the caller's `&mut` writes.
fn auto_timestamp_type_of(graph: &DirGraph, node_idx: NodeIndex) -> Option<String> {
    let node_type = {
        let _arena_guard = graph.graph.begin_query();
        graph
            .graph
            .node_view(node_idx)
            .map(|n| n.node_type_str(&graph.interner).to_string())
    };
    node_type.filter(|nt| graph.auto_timestamp_for(nt))
}

/// True when `variable` is a bound-but-null write target on this row —
/// e.g. an unmatched OPTIONAL MATCH variable (no binding at all) or an
/// explicit NULL projection. openCypher: SET / REMOVE on a NULL target is
/// a no-op for that row, mirroring how DELETE already skips NULLs. A
/// *truly undefined* name never reaches here — the planner's scope
/// validation (`validate_scope`) rejects it before execution — so any
/// remaining non-entity target that isn't NULL is a genuine type error
/// and the caller keeps returning its descriptive error for it.
fn is_null_write_target(row: &ResultRow, variable: &str) -> bool {
    !row.node_bindings.contains_key(variable)
        && !row.edge_bindings.contains_key(variable)
        && !row.path_bindings.contains_key(variable)
        && matches!(row.projected.get(variable), None | Some(Value::Null))
}

/// Every property key currently set on `variable`'s binding — the clear-list
/// for `SET n = {…}` / `SET r = {…}`.
fn existing_property_keys(
    graph: &crate::graph::dir_graph::DirGraph,
    row: &ResultRow,
    variable: &str,
) -> Vec<String> {
    // Arena guard: node_weight/edge_weight materialize on the disk backend
    // (protocol in disk/graph.rs); scoped so the borrow ends before the
    // caller's &mut.
    let _arena_guard = graph.graph.begin_query();
    if let Some(node_idx) = row.node_bindings.get(variable) {
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
/// from `execute_set`, which is at its complexity ceiling.
// One row-loop arm extracted whole from execute_set (complexity ceiling);
// the argument set IS that loop's working state.
#[allow(clippy::too_many_arguments)]
fn execute_property_set_item<'a>(
    graph: &mut DirGraph,
    row: &ResultRow,
    (variable, property, path, expression): (
        &'a String,
        &'a String,
        &'a [crate::graph::languages::cypher::ast::SetPathStep],
        &'a Expression,
    ),
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
    memos: &mut SetMemos<'a>,
    edges_to_stamp: &mut std::collections::HashSet<petgraph::graph::EdgeIndex>,
    nodes_to_stamp: &mut HashMap<NodeIndex, String>,
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
        params,
        stats,
        edges_to_stamp,
    )? {
        return Ok(PropSetFlow::Applied);
    }

    if property == "id" {
        return Err("Cannot SET node id — it is immutable".to_string());
    }
    if property == "type" || property == "node_type" || property == "label" {
        return Err("Cannot SET node type via property assignment".to_string());
    }

    // Resolve the node. A null-valued target (OPTIONAL MATCH miss) makes this
    // row's write a no-op per openCypher.
    let Some(node_idx) = row.node_bindings.get(variable) else {
        if is_null_write_target(row, variable) {
            return Ok(PropSetFlow::SkipRow);
        }
        return Err(format!(
            "Variable '{}' not bound to a node in SET",
            variable
        ));
    };

    // Evaluate the expression (borrows graph immutably)
    let value = {
        let executor = CypherExecutor::with_params(graph, params, None);
        executor.evaluate_expression(expression, row)?
    };

    // Nested l-value (`o.line_items[2].qty`): the whole-value
    // read-modify-write lives in set_path.rs.
    let value = if path.is_empty() {
        value
    } else {
        super::set_path::read_modify(graph, *node_idx, property, path, value, params, row)?
    };

    apply_node_property_set(
        graph,
        NodePropertySet {
            node_idx: *node_idx,
            property: property.as_str(),
            value,
        },
        memos,
        stats,
        nodes_to_stamp,
    )?;
    Ok(PropSetFlow::Applied)
}

fn execute_set(
    graph: &mut DirGraph,
    set: &SetClause,
    result_set: &ResultSet,
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
    interrupt: &Interrupt,
) -> Result<(), String> {
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

    for (row_idx, row) in result_set.rows.iter().enumerate() {
        check_interrupt_periodic(interrupt, row_idx)?;
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
                        params,
                        stats,
                        &mut memos,
                        &mut edges_to_stamp,
                        &mut nodes_to_stamp,
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
                    let value = {
                        let executor = CypherExecutor::with_params(graph, params, None);
                        executor.evaluate_expression(expression, row)?
                    };
                    let Value::Map(map) = value else {
                        return Err(format!(
                            "SET {} {} expects a map expression",
                            variable,
                            if *replace { "=" } else { "+=" }
                        ));
                    };
                    if !row.node_bindings.contains_key(variable)
                        && !row.edge_bindings.contains_key(variable)
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
                        execute_set(
                            graph,
                            &SetClause { items: properties },
                            &one_row,
                            params,
                            stats,
                            interrupt,
                        )?;
                    }
                }
                SetItem::Label {
                    variable, label, ..
                } => set_node_label(graph, row, (variable, label), stats, &mut nodes_to_stamp)?,
            }
        }
    }

    stamp_node_provenance(graph, &nodes_to_stamp);

    // Edge freshness provenance: bump the reserved keys (updated_at + caller
    // git_sha/modified_by) once per modified edge of an opted-in type.
    if !edges_to_stamp.is_empty() {
        let interned: Vec<(InternedKey, Value)> = graph
            .provenance_props()
            .into_iter()
            .map(|(k, v)| (graph.interner.get_or_intern(k), v))
            .collect();
        for edge_index in &edges_to_stamp {
            if let Some(EdgeData {
                properties: edge_props,
                ..
            }) = GraphWrite::edge_weight_mut(&mut graph.graph, *edge_index)
            {
                for (key, val) in &interned {
                    if let Some((_, existing)) = edge_props.iter_mut().find(|(ek, _)| ek == key) {
                        *existing = val.clone();
                    } else {
                        edge_props.push((*key, val.clone()));
                    }
                }
            }
        }
    }

    Ok(())
}

/// Apply one `SET n:Label` item to a row.
fn set_node_label(
    graph: &mut DirGraph,
    row: &ResultRow,
    item: (&str, &str),
    stats: &mut MutationStats,
    nodes_to_stamp: &mut HashMap<NodeIndex, String>,
) -> Result<(), String> {
    let (variable, label) = item;
    let Some(&node_idx) = row.node_bindings.get(variable) else {
        // Null target (OPTIONAL MATCH miss): no-op for this row.
        if is_null_write_target(row, variable) {
            return Ok(());
        }
        return Err(format!(
            "Variable '{}' not bound to a node in SET",
            variable
        ));
    };
    // Adding a secondary label is a write to the node, judged by its *stored*
    // type — the label being added never widens the scope (that is the
    // label-smuggling defence).
    enforce_node_write_scope(graph, node_idx)?;
    let key = graph.interner.get_or_intern(label);
    if graph.add_node_label(node_idx, key) {
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
            // Arena guard: node_weight materializes on the disk backend
            // (protocol in disk/graph.rs); scoped so the borrow ends before
            // the &mut writes below.
            let columnar_row_id = {
                let _arena_guard = graph.graph.begin_query();
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

fn execute_delete(
    graph: &mut DirGraph,
    delete: &DeleteClause,
    result_set: &ResultSet,
    stats: &mut MutationStats,
    interrupt: &Interrupt,
) -> Result<(), String> {
    use std::collections::HashSet;

    let mut nodes_to_delete: HashSet<petgraph::graph::NodeIndex> = HashSet::new();
    // Collected as a set up front because the Phase-2 "node still has
    // relationships" check must see the WHOLE statement's deletions:
    // openCypher deletes are statement-atomic, so `DELETE r, n` is legal
    // when `r` is the only relationship attached to `n`.
    let mut deleted_edges: HashSet<petgraph::graph::EdgeIndex> = HashSet::new();

    // Phase 1: collect all nodes and edges to delete across all rows, and
    // authorize each against the role-scoped write whitelist as it is first
    // seen. Authorization belongs *here*, not at the commit in Phase 3: a
    // refusal must return before any storage mutation (the refusal-before-
    // mutation norm stated at `set_row.rs`'s `enforce_write_scope` call), so a
    // statement whose 500th row is out of scope does not delete the first 499.
    // The check is per distinct node/edge, not per row — a `HashSet::insert`
    // that returns false has already been judged.
    for (row_idx, row) in result_set.rows.iter().enumerate() {
        check_interrupt_periodic(interrupt, row_idx)?;
        for expr in &delete.expressions {
            let var_name = match expr {
                Expression::Variable(name) => name,
                other => return Err(format!("DELETE expects variable names, got {:?}", other)),
            };

            if let Some(&node_idx) = row.node_bindings.get(var_name) {
                if nodes_to_delete.insert(node_idx) {
                    enforce_node_write_scope(graph, node_idx)?;
                }
            } else if let Some(edge_binding) = row.edge_bindings.get(var_name) {
                if deleted_edges.insert(edge_binding.edge_index) {
                    enforce_bound_edge_write_scope(graph, edge_binding)?;
                }
            } else {
                // Not bound to a node/edge. A node VALUE (NodeRef from
                // WITH / collect) is still deletable; anything else is NULL
                // — e.g. an unmatched OPTIONAL MATCH variable — and
                // openCypher ignores NULL in DELETE (so the idiomatic
                // single-statement cascade `MATCH (root) OPTIONAL MATCH
                // (root)-->(child) DETACH DELETE root, child` works even
                // when a branch is empty). Skip it.
                match row.projected.get(var_name) {
                    Some(Value::NodeRef(i)) => {
                        let node_idx = petgraph::graph::NodeIndex::new(*i as usize);
                        if nodes_to_delete.insert(node_idx) {
                            enforce_node_write_scope(graph, node_idx)?;
                        }
                    }
                    // A materialised node value (`collect(n)` / `RETURN n`) is
                    // deletable too — this is the load-bearing case for
                    // `FOREACH (e IN collect(n) | DETACH DELETE e)`, where the
                    // loop variable is bound in `projected` as a `Value::Node`,
                    // not a `NodeRef`. Both `NodeValue` constructors
                    // (`materialize_node_value` + the Variable-resolution path)
                    // set `id` to the petgraph index, so it resolves the same
                    // way as `NodeRef`. (Without this arm, DELETE inside FOREACH
                    // over a collected list was a silent no-op.)
                    Some(Value::Node(nv)) => {
                        let node_idx = petgraph::graph::NodeIndex::new(nv.id as usize);
                        if nodes_to_delete.insert(node_idx) {
                            enforce_node_write_scope(graph, node_idx)?;
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Phase 2: for plain DELETE (not DETACH), verify no node keeps edges.
    // Relationships deleted by THIS statement don't count — openCypher
    // deletes are statement-atomic, so `MATCH (a)-[r]->(b) DELETE r, a`
    // succeeds when `r` covers every relationship attached to `a`.
    if !delete.detach {
        // Arena guard: node_weight (and the disk backend's edge iteration)
        // materialize into the query arena (protocol in disk/graph.rs);
        // scoped so the borrow ends before Phase 3's &mut commits.
        let _arena_guard = graph.graph.begin_query();
        for (node_count, &node_idx) in nodes_to_delete.iter().enumerate() {
            check_interrupt_periodic(interrupt, node_count)?;
            let has_edges = graph
                .graph
                .edges_directed(node_idx, petgraph::Direction::Outgoing)
                .any(|e| !deleted_edges.contains(&e.id()))
                || graph
                    .graph
                    .edges_directed(node_idx, petgraph::Direction::Incoming)
                    .any(|e| !deleted_edges.contains(&e.id()));
            if has_edges {
                let name = graph
                    .graph
                    .node_view(node_idx)
                    .map(|n| {
                        n.get_field_ref("name")
                            .or_else(|| n.get_field_ref("title"))
                            .map(|v| match v.as_ref() {
                                // Bare string, not the quoted Display form
                                // (and never the old Debug `String("…")`).
                                Value::String(s) => s.clone(),
                                other => other.to_string(),
                            })
                            .unwrap_or_else(|| format!("index {}", node_idx.index()))
                    })
                    .unwrap_or_else(|| "unknown".to_string());
                return Err(format!(
                    "Cannot delete node '{}' because it still has relationships. Use DETACH DELETE to delete the node and all its relationships.",
                    name
                ));
            }
        }
    }

    // Phase 3: infallible commit of the preflighted edge set. Deliberately
    // non-interruptible: once deletion begins, completing it preserves atomic
    // statement semantics without an O(graph) rollback checkpoint.
    for edge_index in deleted_edges.iter().copied() {
        GraphWrite::remove_edge(&mut graph.graph, edge_index);
        stats.relationships_deleted += 1;
    }

    // Phase 3's explicit edge-variable deletes still need cache
    // invalidation (`detach_delete_nodes` only covers its own edges).
    if stats.relationships_deleted > 0 {
        graph.invalidate_edge_type_counts_cache();
        graph.connection_types.clear();
    }

    // Phase 4-7: DETACH-delete the nodes — incident edges, the nodes,
    // and index cleanup. For a plain DELETE, Phase 2 has verified the
    // nodes carry no edges, so none are removed here. Shared with
    // `purge_provisional_nodes` via `maintain::detach_delete_nodes`.
    //
    // Write scope: the incident edges removed here are **collateral of an
    // already-authorized node delete** and are deliberately not re-checked per
    // far endpoint. Re-checking would be both wrong and expensive — wrong
    // because a node the role may delete cannot be left behind as a dangling
    // half-edge just because it points at a type the role may not write, and
    // expensive because it is an O(degree) type resolution per deleted node.
    // Phase 1 authorized every node in `nodes_to_delete`; that authorization
    // covers everything attached to them.
    let (nodes_deleted, edges_removed) =
        crate::graph::mutation::maintain::detach_delete_nodes(graph, &nodes_to_delete);
    stats.nodes_deleted += nodes_deleted;
    stats.relationships_deleted += edges_removed;

    Ok(())
}

fn execute_remove(
    graph: &mut DirGraph,
    remove: &RemoveClause,
    result_set: &ResultSet,
    stats: &mut MutationStats,
    interrupt: &Interrupt,
) -> Result<(), String> {
    for (row_idx, row) in result_set.rows.iter().enumerate() {
        check_interrupt_periodic(interrupt, row_idx)?;
        for item in &remove.items {
            match item {
                RemoveItem::Property { variable, property } => {
                    // Relationship property REMOVE is its own path, for the
                    // same reasons the SET counterpart is.
                    if remove_edge_property(graph, row, variable, property, stats)? {
                        continue;
                    }

                    if property == "id" {
                        return Err("Cannot REMOVE node id — it is immutable".to_string());
                    }
                    if property == "type" || property == "node_type" || property == "label" {
                        return Err("Cannot REMOVE node type".to_string());
                    }

                    // A null-valued target (OPTIONAL MATCH miss) makes this
                    // row's REMOVE a no-op per openCypher.
                    let Some(node_idx) = row.node_bindings.get(variable) else {
                        if is_null_write_target(row, variable) {
                            continue;
                        }
                        return Err(format!(
                            "Variable '{}' not bound to a node in REMOVE",
                            variable
                        ));
                    };

                    // Read node_type before mutable borrow (for index update)
                    let node_type_str = graph
                        .node_view(*node_idx)
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
                    let constraint_plan = graph
                        .plan_property_write(&node_type_str, *node_idx, property, None)
                        .map_err(|violation| violation.to_string())?;

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
                            let _arena_guard = graph.graph.begin_query();
                            graph
                                .graph
                                .node_weight(*node_idx)
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
                                    node_idx: *node_idx,
                                    row_id,
                                    key,
                                    value: &Value::Null,
                                },
                                PriorCell::Read,
                            );
                        }
                    }

                    let removed_value = if let Some(prior) = cleared_via_master {
                        prior
                    } else if graph.graph.node_weight(*node_idx).is_some() {
                        if write_field == "name" || write_field == "title" {
                            // Read the *resolved* title (a columnar node keeps
                            // it in its store's reserved column, with the
                            // inline field on the `Null` sentinel) and clear it
                            // through the same backend seam a `SET` writes it
                            // through, so the removal reaches the store instead
                            // of nulling an already-null field.
                            let old = graph.graph.get_node_title(*node_idx).unwrap_or(Value::Null);
                            GraphWrite::set_node_title(&mut graph.graph, *node_idx, Value::Null);
                            let key = InternedKey::from_str("name");
                            GraphWrite::remove_node_property(&mut graph.graph, *node_idx, key);
                            (!matches!(old, Value::Null)).then_some(old)
                        } else {
                            // The backend picks the right removal semantics per
                            // storage: disk stages a `Null` write so its flush
                            // propagates the removal; row storage drops the key.
                            let key = InternedKey::from_str(write_field);
                            GraphWrite::remove_node_property(&mut graph.graph, *node_idx, key)
                        }
                    } else {
                        None
                    };

                    if let Some(old_val) = removed_value {
                        stats.properties_removed += 1;
                        graph.update_property_indices_for_remove(
                            &node_type_str,
                            *node_idx,
                            property,
                            &old_val,
                        );
                        // Stripping the indexed property is a content change
                        // like any other; the next refresh will find no string
                        // and drop the document.
                        crate::graph::index_freshness::write_hooks::note_property_written(
                            graph,
                            *node_idx,
                            &node_type_str,
                            Some(write_field),
                        );
                    }
                    graph.apply_property_write_plan(&constraint_plan, *node_idx);
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
        }
    }

    Ok(())
}

fn execute_merge(
    graph: &mut DirGraph,
    merge: &MergeClause,
    existing: ResultSet,
    params: &HashMap<String, Value>,
    stats: &mut MutationStats,
    interrupt: &Interrupt,
) -> Result<ResultSet, String> {
    // As in `execute_create`: one MERGE per incoming row, none for zero rows.
    // The implicit start row for a leading MERGE is seeded by the pipeline.
    let source_rows = existing.rows;

    let mut new_rows = Vec::with_capacity(source_rows.len());

    for (row_idx, mut new_row) in source_rows.into_iter().enumerate() {
        check_interrupt_periodic(interrupt, row_idx)?;
        // Equality against null is undefined, so a null-bearing MERGE key
        // cannot identify either a match or a safe entity to create.
        // (Block-scoped: the executor holds the disk arena guard, whose
        // borrow of `graph` must end before the &mut mutation calls below.)
        {
            let executor = CypherExecutor::with_params(graph, params, None);
            for element in &merge.pattern.elements {
                let properties = match element {
                    CreateElement::Node(node) => &node.properties,
                    CreateElement::Edge(edge) => &edge.properties,
                };
                for (name, expression) in properties {
                    if matches!(
                        executor.evaluate_expression(expression, &new_row)?,
                        Value::Null
                    ) {
                        return Err(format!("MERGE cannot use null for property '{}'", name));
                    }
                }
            }
        }
        let matched = try_match_merge_pattern(graph, &merge.pattern, &new_row, params)?;

        if let Some(bound_row) = matched {
            for (var, idx) in &bound_row.node_bindings {
                new_row.node_bindings.insert(var.clone(), *idx);
            }
            for (var, binding) in &bound_row.edge_bindings {
                new_row.edge_bindings.insert(var.clone(), *binding);
            }

            if let Some(ref set_items) = merge.on_match {
                let set_clause = SetClause {
                    items: set_items.clone(),
                };
                let temp_rs = ResultSet {
                    rows: vec![new_row.clone()],
                    columns: Vec::new(),
                    lazy_return_items: None,
                };
                execute_set(graph, &set_clause, &temp_rs, params, stats, interrupt)?;
            }
        } else {
            let create_clause = CreateClause {
                patterns: vec![merge.pattern.clone()],
            };
            let temp_rs = ResultSet {
                rows: vec![new_row.clone()],
                columns: existing.columns.clone(),
                lazy_return_items: None,
            };
            let created = execute_create(graph, &create_clause, temp_rs, params, stats, interrupt)?;

            if let Some(created_row) = created.rows.into_iter().next() {
                for (var, idx) in created_row.node_bindings {
                    new_row.node_bindings.insert(var, idx);
                }
                for (var, binding) in created_row.edge_bindings {
                    new_row.edge_bindings.insert(var, binding);
                }
            }

            if let Some(ref set_items) = merge.on_create {
                let set_clause = SetClause {
                    items: set_items.clone(),
                };
                let temp_rs = ResultSet {
                    rows: vec![new_row.clone()],
                    columns: Vec::new(),
                    lazy_return_items: None,
                };
                execute_set(graph, &set_clause, &temp_rs, params, stats, interrupt)?;
            }
        }

        new_rows.push(new_row);
    }

    Ok(ResultSet {
        rows: new_rows,
        columns: existing.columns,
        lazy_return_items: None,
    })
}

/// Returns the bound row when the pattern already exists, `None` when it does
/// not (the caller then CREATEs it).
fn try_match_merge_pattern(
    graph: &DirGraph,
    pattern: &CreatePattern,
    row: &ResultRow,
    params: &HashMap<String, Value>,
) -> Result<Option<ResultRow>, String> {
    let executor = CypherExecutor::with_params(graph, params, None);

    match pattern.elements.len() {
        1 => {
            // Node-only MERGE: (var:Label {key: val, ...})
            if let CreateElement::Node(node_pat) = &pattern.elements[0] {
                // If variable is already bound from prior MATCH, it's already matched
                if let Some(ref var) = node_pat.variable {
                    if let Some(&existing_idx) = row.node_bindings.get(var) {
                        if graph.graph.node_view(existing_idx).is_some() {
                            let mut result_row = ResultRow::new();
                            result_row.node_bindings.insert(var.clone(), existing_idx);
                            return Ok(Some(result_row));
                        }
                    }
                }

                let label = node_pat.label.as_deref().unwrap_or("Node");

                // The id/property/composite indexes and `type_indices` are
                // keyed by PRIMARY type. If `label` also occurs as a
                // secondary label on some node, those structures miss the
                // secondary-labelled candidates and would falsely report
                // "no match" → MERGE creates a duplicate. In that case skip
                // the index short-circuits and scan the full primary∪secondary
                // candidate set (`nodes_with_label`). The common case (label
                // has no secondary occurrences) keeps every index fast path.
                let label_has_secondary = graph.has_secondary_labels
                    && graph
                        .secondary_label_index
                        .contains_key(&crate::graph::schema::InternedKey::from_str(label));

                let expected_props = merge_expected_props(&executor, node_pat, row, graph)?;

                let node_matches_all = |idx: NodeIndex, props: &[(&str, Value)]| -> bool {
                    if let Some(node) = graph.graph.node_view(idx) {
                        props.iter().all(|(key, expected)| {
                            let value = if *key == "name" || *key == "title" {
                                node.get_field_ref("title")
                            } else {
                                node.get_field_ref(key)
                            };
                            value.as_deref() == Some(expected)
                        })
                    } else {
                        false
                    }
                };

                let build_result = |idx: NodeIndex| -> ResultRow {
                    let mut result_row = ResultRow::new();
                    if let Some(ref var) = node_pat.variable {
                        result_row.node_bindings.insert(var.clone(), idx);
                    }
                    result_row
                };

                if !label_has_secondary {
                    // 1. If pattern contains "id" property, use O(1) id_index lookup
                    if let Some((_, id_value)) = expected_props.iter().find(|(k, _)| *k == "id") {
                        if let Some(idx) = graph.lookup_by_id_readonly(label, id_value) {
                            if expected_props.len() == 1 || node_matches_all(idx, &expected_props) {
                                return Ok(Some(build_result(idx)));
                            }
                        }
                        return Ok(None);
                    }

                    // 2. Single non-id property: try property index.
                    // Probed under the pattern's own spelling —
                    // `lookup_by_index` resolves a type's registered
                    // title/id alias itself. The old `name`/`title` → `title`
                    // remap was not that: on a type whose `name` is an
                    // ordinary stored property distinct from the title, it
                    // asked the title index a question about `name` and
                    // MERGE created a duplicate of the node it missed.
                    if expected_props.len() == 1 {
                        let (key, ref value) = expected_props[0];
                        if let Some(candidates) = graph.lookup_by_index(label, key, value) {
                            for &idx in &candidates {
                                if node_matches_all(idx, &expected_props) {
                                    return Ok(Some(build_result(idx)));
                                }
                            }
                            return Ok(None);
                        }
                        // No index — fall through to linear scan
                    }

                    // 3. Multi-property: try composite index
                    if expected_props.len() >= 2 {
                        // Composite lookup excludes id/name/title: they use
                        // special storage, not ordinary property columns.
                        let mut indexable: Vec<(&str, &Value)> = expected_props
                            .iter()
                            .filter(|(k, _)| *k != "id" && *k != "name" && *k != "title")
                            .map(|(k, v)| (*k, v))
                            .collect();
                        if indexable.len() >= 2 {
                            indexable.sort_by(|a, b| a.0.cmp(b.0));
                            let names: Vec<String> =
                                indexable.iter().map(|(k, _)| k.to_string()).collect();
                            let values: Vec<Value> =
                                indexable.iter().map(|(_, v)| (*v).clone()).collect();
                            if let Some(candidates) =
                                graph.lookup_by_composite_index(label, &names, &values)
                            {
                                for &idx in &candidates {
                                    if node_matches_all(idx, &expected_props) {
                                        return Ok(Some(build_result(idx)));
                                    }
                                }
                                return Ok(None);
                            }
                        }
                    }
                }

                // 4. Fall back to linear scan (no index covers the pattern, or
                // `label` has secondary occurrences). `nodes_with_label` unions
                // primary + secondary candidates (and is the identical
                // `type_indices` clone when no secondary labels exist).
                for idx in graph.nodes_with_label(label) {
                    if node_matches_all(idx, &expected_props) {
                        return Ok(Some(build_result(idx)));
                    }
                }
                Ok(None)
            } else {
                Err("MERGE pattern must start with a node".to_string())
            }
        }
        3 => {
            // Relationship MERGE: (a)-[r:TYPE]->(b)
            let source_var = get_create_node_variable(&pattern.elements[0]);
            let target_var = get_create_node_variable(&pattern.elements[2]);

            let source_idx = source_var
                .and_then(|v| row.node_bindings.get(v).copied())
                .ok_or("MERGE path: source node must be bound by prior MATCH")?;
            let target_idx = target_var
                .and_then(|v| row.node_bindings.get(v).copied())
                .ok_or("MERGE path: target node must be bound by prior MATCH")?;

            if let CreateElement::Edge(edge_pat) = &pattern.elements[1] {
                let (actual_src, actual_tgt) = match edge_pat.direction {
                    CreateEdgeDirection::Outgoing => (source_idx, target_idx),
                    CreateEdgeDirection::Incoming => (target_idx, source_idx),
                };

                let interned_ct = InternedKey::from_str(&edge_pat.connection_type);
                let matching_edge = graph
                    .graph
                    .edges_directed(actual_src, petgraph::Direction::Outgoing)
                    .find(|e| {
                        e.target() == actual_tgt && e.weight().connection_type == interned_ct
                    });

                if let Some(edge_ref) = matching_edge {
                    let mut result_row = ResultRow::new();
                    if let Some(ref var) = edge_pat.variable {
                        result_row.edge_bindings.insert(
                            var.clone(),
                            EdgeBinding {
                                source: actual_src,
                                target: actual_tgt,
                                edge_index: edge_ref.id(),
                            },
                        );
                    }
                    Ok(Some(result_row))
                } else {
                    Ok(None)
                }
            } else {
                Err("Expected edge in MERGE path pattern".to_string())
            }
        }
        _ => Err("MERGE supports single-node or single-edge patterns only".to_string()),
    }
}

#[cfg(test)]
#[path = "write_remove_columnar_tests.rs"]
mod remove_columnar_tests;

#[cfg(test)]
#[path = "write_rel_constraint_tests.rs"]
mod rel_constraint_tests;

#[cfg(test)]
mod is_mutation_query_tests {
    use super::super::super::ast::*;
    use super::is_mutation_query;

    fn query(clauses: Vec<Clause>) -> CypherQuery {
        CypherQuery {
            clauses,
            explain: false,
            profile: false,
            output_format: OutputFormat::Default,
            optimizer_tags: Vec::new(),
        }
    }

    fn create_clause() -> Clause {
        Clause::Create(CreateClause {
            patterns: Vec::new(),
        })
    }

    fn return_clause() -> Clause {
        Clause::Return(ReturnClause {
            items: Vec::new(),
            distinct: false,
            having: None,
            lazy_eligible: false,
            group_limit_hint: None,
        })
    }

    #[test]
    fn plain_read_is_not_a_mutation() {
        assert!(!is_mutation_query(&query(vec![return_clause()])));
    }

    #[test]
    fn top_level_write_is_a_mutation() {
        assert!(is_mutation_query(&query(vec![create_clause()])));
    }

    #[test]
    fn write_inside_call_subquery_body_is_a_mutation() {
        // CALL { CREATE (...) RETURN ... } — the body carries a write,
        // so the enclosing query must classify as a mutation even though
        // the outer clause is a CallSubquery (not itself a write clause).
        let body = Box::new(query(vec![create_clause(), return_clause()]));
        let call = Clause::CallSubquery {
            import: Vec::new(),
            body,
        };
        assert!(is_mutation_query(&query(vec![call, return_clause()])));
    }

    #[test]
    fn nested_write_inside_call_subquery_body_is_a_mutation() {
        // CALL { CALL { CREATE (...) RETURN ... } RETURN ... } — recursion
        // must reach an arbitrarily-deep nested body.
        let inner = Box::new(query(vec![create_clause(), return_clause()]));
        let inner_call = Clause::CallSubquery {
            import: Vec::new(),
            body: inner,
        };
        let outer = Box::new(query(vec![inner_call, return_clause()]));
        let outer_call = Clause::CallSubquery {
            import: Vec::new(),
            body: outer,
        };
        assert!(is_mutation_query(&query(vec![outer_call])));
    }

    #[test]
    fn read_only_call_subquery_body_is_not_a_mutation() {
        let body = Box::new(query(vec![return_clause()]));
        let call = Clause::CallSubquery {
            import: Vec::new(),
            body,
        };
        assert!(!is_mutation_query(&query(vec![call, return_clause()])));
    }

    #[test]
    fn write_inside_union_arm_is_a_mutation() {
        // UNION arms also recurse — a write in either arm makes the
        // query a mutation.
        let arm = Box::new(query(vec![create_clause(), return_clause()]));
        let union = Clause::Union(UnionClause {
            all: false,
            query: arm,
            kind: SetOpKind::Union,
        });
        assert!(is_mutation_query(&query(vec![return_clause(), union])));
    }
}
