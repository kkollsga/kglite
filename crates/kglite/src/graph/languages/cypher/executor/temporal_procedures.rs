//! `db.temporal.*` — validity-interval declarations in Cypher, over
//! `kglite::api::temporal`.
//!
//! ```text
//! CALL db.temporal.declare({node: 'Field', from: 'vf', to: 'vt', convention: 'closed'})
//! CALL db.temporal.declare({relationship: 'HAS_LICENSEE', source_type: 'Field',
//!                           from: 'vf', to: 'vt', convention: 'half_open'})
//! CALL db.temporal.undeclare({relationship: 'HAS_LICENSEE', source_type: 'Field'})
//! CALL db.temporal.declarations()
//! ```

use std::collections::HashMap;
use std::sync::Mutex;

use crate::datatypes::values::Value;
use crate::graph::features::temporal::declarations::{declare, list, undeclare};
use crate::graph::features::temporal::{IntervalConvention, TemporalTarget};
use crate::graph::languages::cypher::ast::YieldItem;
use crate::graph::languages::cypher::result::{QueryDiagnostics, ResultRow};
use crate::graph::schema::DirGraph;

use super::edge_embedding_procedures::{optional_string, require_string, yield_row};
use super::procedure_params::reject_unknown_keys;

/// The mutating procedures: declare and undeclare. A declaration's abutment
/// advisory goes to `diagnostics`, the statement's warning sink.
pub(super) fn execute(
    graph: &mut DirGraph,
    proc_name: &str,
    params: &HashMap<String, Value>,
    yields: &[YieldItem],
    diagnostics: &Mutex<QueryDiagnostics>,
) -> Result<Vec<ResultRow>, String> {
    reject_unknown_keys(
        &format!("CALL {proc_name}"),
        params.keys().map(String::as_str),
        accepted_keys(proc_name),
    )?;
    let target = target(params, proc_name)?;
    let values = match proc_name {
        "db.temporal.declare" => {
            let from = require_string(params, "from", proc_name)?;
            let to = require_string(params, "to", proc_name)?;
            let convention = convention(params, proc_name)?;
            let report = declare(graph, &target, &from, &to, convention)
                .map_err(|error| format!("CALL {proc_name}: {error}"))?;
            if let Some(warning) = report.warning {
                let mut sink = diagnostics.lock().unwrap_or_else(|e| e.into_inner());
                super::retrieval_diagnostics::record_warning(&mut sink.warnings, warning);
            }
            HashMap::from([
                ("declared", Value::Boolean(report.changed)),
                ("rows", Value::Int64(report.rows as i64)),
                ("abutting_rows", count(report.abutting_rows)),
            ])
        }
        "db.temporal.undeclare" => {
            HashMap::from([("undeclared", Value::Boolean(undeclare(graph, &target)))])
        }
        other => unreachable!("non-temporal procedure routed here: {other}"),
    };
    Ok(vec![yield_row(values, yields)])
}

fn count(value: Option<usize>) -> Value {
    value.map_or(Value::Null, |n| Value::Int64(n as i64))
}

/// Exactly one of `node` / `relationship`; `source_type` only with the
/// latter.
fn target(params: &HashMap<String, Value>, proc_name: &str) -> Result<TemporalTarget, String> {
    let node = optional_string(params, "node", proc_name)?;
    let relationship = optional_string(params, "relationship", proc_name)?;
    let source_type = optional_string(params, "source_type", proc_name)?;
    match (node, relationship) {
        (Some(label), None) => {
            if source_type.is_some() {
                return Err(format!(
                    "CALL {proc_name}: 'source_type' applies to a relationship target, not a \
                     node label"
                ));
            }
            Ok(TemporalTarget::Node(label))
        }
        (None, Some(rel_type)) => Ok(TemporalTarget::Relationship {
            rel_type,
            source_type,
        }),
        (Some(_), Some(_)) => Err(format!(
            "CALL {proc_name}: give either 'node' or 'relationship', not both"
        )),
        (None, None) => Err(format!(
            "CALL {proc_name}: name the target kind: {{node: 'Label'}} or {{relationship: \
             'TYPE'[, source_type: 'Label']}}"
        )),
    }
}

fn convention(
    params: &HashMap<String, Value>,
    proc_name: &str,
) -> Result<IntervalConvention, String> {
    let text = require_string(params, "convention", proc_name)?;
    IntervalConvention::parse(&text).ok_or_else(|| {
        format!(
            "CALL {proc_name}: convention '{text}' is not one of 'closed' (the to day is \
             still valid) or 'half_open' (the to day is the first day no longer valid)"
        )
    })
}

/// `db.temporal.declarations()` — one row per declaration.
pub(super) fn declarations(
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yields: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    let proc_name = "db.temporal.declarations";
    reject_unknown_keys(
        &format!("CALL {proc_name}"),
        params.keys().map(String::as_str),
        accepted_keys(proc_name),
    )?;
    Ok(list(graph)
        .into_iter()
        .map(|info| {
            let (kind, name, source_type) = match info.target {
                TemporalTarget::Node(label) => ("node", label, None),
                TemporalTarget::Relationship {
                    rel_type,
                    source_type,
                } => ("relationship", rel_type, source_type),
            };
            yield_row(
                HashMap::from([
                    ("kind", Value::String(kind.into())),
                    ("name", Value::String(name)),
                    (
                        "source_type",
                        source_type.map_or(Value::Null, Value::String),
                    ),
                    ("from", Value::String(info.config.valid_from)),
                    ("to", Value::String(info.config.valid_to)),
                    (
                        "convention",
                        Value::String(info.config.convention.as_str().into()),
                    ),
                    ("abutting_rows", count(info.abutting_rows)),
                    ("ambiguous", Value::Boolean(info.ambiguous)),
                ]),
                yields,
            )
        })
        .collect())
}

/// Every parameter each `db.temporal.*` procedure reads — also the
/// "Accepted:" line of the unknown-key refusal.
pub(super) fn accepted_keys(proc_name: &str) -> &'static [&'static str] {
    match proc_name {
        "db.temporal.declare" => &[
            "node",
            "relationship",
            "source_type",
            "from",
            "to",
            "convention",
        ],
        "db.temporal.undeclare" => &["node", "relationship", "source_type"],
        "db.temporal.declarations" => &[],
        other => unreachable!("non-temporal procedure routed here: {other}"),
    }
}
