//! Code-analysis and maintenance procedures exposed through Cypher `CALL`.

use std::collections::HashMap;

use super::super::ast::YieldItem;
use super::super::result::ResultRow;
use crate::datatypes::values::Value;
use crate::graph::schema::DirGraph;

/// Dispatch analysis procedures after shared CALL validation.
pub(super) fn execute_analysis_procedure(
    proc_name: &str,
    graph: &DirGraph,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    match proc_name {
        "affected_tests" => {
            super::affected_tests::execute_affected_tests(graph, params, yield_items)
        }
        "rev_diff" => super::rev_procedures::execute_rev_diff(graph, params, yield_items),
        "dead_code" => super::dead_code::execute_dead_code(graph, params, yield_items),
        "refresh_stats" => super::refresh_stats::execute_refresh_stats(graph, params, yield_items),
        _ => unreachable!("non-analysis procedure routed to analysis dispatcher: {proc_name}"),
    }
}
