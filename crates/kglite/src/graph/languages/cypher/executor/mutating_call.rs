//! Per-row dispatch for procedures that mutate graph state.

use super::relationship_identity::StatementRelationshipIdentities;
use super::CypherExecutor;
use crate::datatypes::values::Value;
use crate::graph::algorithms::Interrupt;
use crate::graph::edge_embedding_generation::EmbeddingExecutionService;
use crate::graph::languages::cypher::ast::CallClause;
use crate::graph::languages::cypher::result::{QueryDiagnostics, ResultRow, ResultSet};
use crate::graph::schema::DirGraph;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub(super) struct MutatingCallCtx<'a, 'service> {
    pub(super) params: &'a HashMap<String, Value>,
    pub(super) interrupt: &'a Interrupt,
    pub(super) budget: &'a super::budget::ExecutionBudget,
    pub(super) identities: &'a Arc<Mutex<StatementRelationshipIdentities>>,
    pub(super) service: Option<&'service EmbeddingExecutionService<'service>>,
    /// The statement's warning sink, for advisories a procedure raises.
    pub(super) diagnostics: &'a Mutex<QueryDiagnostics>,
}

pub(super) fn execute(
    graph: &mut DirGraph,
    call: &CallClause,
    existing: ResultSet,
    ctx: MutatingCallCtx<'_, '_>,
) -> Result<ResultSet, String> {
    let name = call.procedure_name.to_lowercase();
    let resolved = CallClause {
        procedure_name: call.procedure_name.clone(),
        parameters: call.parameters.clone(),
        yield_items: super::call_clause::resolve_yield_items(
            &name,
            &call.procedure_name,
            &call.yield_items,
        )?,
    };
    super::call_clause::reject_call_yield_collisions(&existing, &resolved)?;
    let mut columns = existing.columns;
    super::call_clause::append_call_yield_columns(&mut columns, &resolved);
    let mut joined = Vec::new();
    for outer in existing.rows {
        super::check_interrupt(ctx.interrupt)?;
        // The argument expressions belong to this statement, so they evaluate
        // under its relationship identities: an inline `relationships(p)[0]`
        // must carry the statement token the procedures check, and a binding
        // retired earlier in the statement must read as retired here too.
        let args = CypherExecutor::with_params(graph, ctx.params, ctx.interrupt.deadline)
            .with_cancel(ctx.interrupt.cancel)
            .with_budget(ctx.budget.clone())
            .with_relationship_identities(Some(ctx.identities.clone()))
            .extract_call_params(&resolved.parameters, &outer)?;
        let rows = dispatch(graph, &name, &args, &resolved, &ctx)?;
        ctx.budget.check_work(rows.len(), &format!("CALL {name}"))?;
        ctx.budget
            .reserve_rows(joined.len(), rows.len(), &format!("CALL {name} row join"))?;
        joined.extend(
            rows.into_iter()
                .map(|row| super::call_clause::join_call_row(&outer, row)),
        );
    }
    Ok(ResultSet {
        columns,
        rows: joined,
        lazy_return_items: None,
    })
}

fn dispatch(
    graph: &mut DirGraph,
    name: &str,
    params: &HashMap<String, Value>,
    call: &CallClause,
    ctx: &MutatingCallCtx<'_, '_>,
) -> Result<Vec<ResultRow>, String> {
    let (identities, service) = (ctx.identities, ctx.service);
    if name.starts_with("table.") {
        super::table_procedures::execute_table_procedure(graph, name, params, &call.yield_items)
    } else if name.starts_with("db.relationship_text_index.") {
        super::edge_text_index_procedures::execute(graph, name, params, &call.yield_items)
    } else if name.starts_with("db.node_text_index.") {
        super::node_text_index_procedures::execute(graph, name, params, &call.yield_items)
    } else if name.starts_with("db.node_embeddings.") {
        super::node_embedding_procedures::execute(graph, name, params, &call.yield_items, service)
    } else if name.starts_with("db.relationship_embeddings.") {
        super::edge_embedding_procedures::execute(
            graph,
            name,
            params,
            &call.yield_items,
            identities,
            service,
        )
    } else if name.starts_with("db.temporal.") {
        let yields = &call.yield_items;
        super::temporal_procedures::execute(graph, name, params, yields, ctx.diagnostics)
    } else {
        super::cdc_procedures::execute_mutating_procedure(graph, name, params, &call.yield_items)
    }
}
