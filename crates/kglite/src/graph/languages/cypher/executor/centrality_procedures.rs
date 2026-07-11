//! Centrality and community-detection `CALL` procedures.

use std::collections::{HashMap, HashSet};

use petgraph::graph::NodeIndex;

use super::helpers::{
    call_param_bool, call_param_f64, call_param_opt_string, call_param_opt_usize,
    call_param_string_list, call_param_usize,
};
use super::{CypherExecutor, ResultRow};
use crate::datatypes::values::Value;
use crate::graph::languages::cypher::ast::YieldItem;

/// Dispatch the centrality/community procedure family after shared CALL
/// validation and scope construction.
pub(super) fn execute_centrality_procedure(
    executor: &CypherExecutor<'_>,
    proc_name: &str,
    params: &HashMap<String, Value>,
    scope: Option<&HashSet<NodeIndex>>,
    streaming_community: bool,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    match proc_name {
        "pagerank" => {
            let damping = call_param_f64(params, "damping_factor", 0.85);
            let max_iter = call_param_usize(params, "max_iterations", 100);
            let tolerance = call_param_f64(params, "tolerance", 1e-6);
            let conn = call_param_string_list(params, "connection_types");
            let results = crate::graph::algorithms::graph_algorithms::pagerank(
                executor.graph,
                damping,
                max_iter,
                tolerance,
                conn.as_deref(),
                scope,
                executor.interrupt(),
            )?;
            executor.centrality_to_rows(&results, yield_items)
        }
        "betweenness" | "betweenness_centrality" => {
            let normalized = call_param_bool(params, "normalized", true);
            let sample_size = call_param_opt_usize(params, "sample_size");
            let conn = call_param_string_list(params, "connection_types");
            let results = crate::graph::algorithms::graph_algorithms::betweenness_centrality(
                executor.graph,
                normalized,
                sample_size,
                conn.as_deref(),
                scope,
                executor.interrupt(),
            )?;
            executor.centrality_to_rows(&results, yield_items)
        }
        "degree" | "degree_centrality" => {
            let normalized = call_param_bool(params, "normalized", true);
            let conn = call_param_string_list(params, "connection_types");
            let results = crate::graph::algorithms::graph_algorithms::degree_centrality(
                executor.graph,
                normalized,
                conn.as_deref(),
                scope,
                executor.interrupt(),
            )?;
            executor.centrality_to_rows(&results, yield_items)
        }
        "closeness" | "closeness_centrality" => {
            let normalized = call_param_bool(params, "normalized", true);
            let sample_size = call_param_opt_usize(params, "sample_size");
            let conn = call_param_string_list(params, "connection_types");
            let results = crate::graph::algorithms::graph_algorithms::closeness_centrality(
                executor.graph,
                normalized,
                sample_size,
                conn.as_deref(),
                scope,
                executor.interrupt(),
            )?;
            executor.centrality_to_rows(&results, yield_items)
        }
        "louvain" | "louvain_communities" => {
            let resolution = call_param_f64(params, "resolution", 1.0);
            let weight_prop = call_param_opt_string(params, "weight_property");
            let conn = call_param_string_list(params, "connection_types");
            let result = crate::graph::algorithms::graph_algorithms::louvain_communities(
                executor.graph,
                weight_prop.as_deref(),
                resolution,
                conn.as_deref(),
                scope,
                if streaming_community {
                    crate::graph::algorithms::Interrupt::default()
                } else {
                    executor.interrupt()
                },
            )?;
            executor.community_result_to_rows(&result, yield_items)
        }
        "leiden" | "leiden_communities" => {
            let resolution = call_param_f64(params, "resolution", 1.0);
            let weight_prop = call_param_opt_string(params, "weight_property");
            let conn = call_param_string_list(params, "connection_types");
            let result = crate::graph::algorithms::graph_algorithms::leiden_communities(
                executor.graph,
                weight_prop.as_deref(),
                resolution,
                conn.as_deref(),
                scope,
                if streaming_community {
                    crate::graph::algorithms::Interrupt::default()
                } else {
                    executor.interrupt()
                },
            )?;
            executor.community_result_to_rows(&result, yield_items)
        }
        "label_propagation" => {
            let max_iter = call_param_usize(params, "max_iterations", 100);
            let conn = call_param_string_list(params, "connection_types");
            let result = crate::graph::algorithms::graph_algorithms::label_propagation(
                executor.graph,
                max_iter,
                conn.as_deref(),
                scope,
                executor.interrupt(),
            )?;
            executor.community_result_to_rows(&result, yield_items)
        }
        _ => unreachable!("non-centrality procedure routed to centrality dispatcher: {proc_name}"),
    }
}
