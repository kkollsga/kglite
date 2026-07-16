//! Centrality and community-detection `CALL` procedures.

use std::collections::{HashMap, HashSet};

use petgraph::graph::NodeIndex;

use super::helpers::{
    call_param_bool, call_param_f64, call_param_opt_string, call_param_opt_usize,
    call_param_string_list, call_param_usize,
};
use super::{CypherExecutor, ResultRow};
use crate::datatypes::values::Value;
use crate::graph::algorithms::graph_algorithms as ga;
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
            let results = ga::pagerank(
                executor.graph,
                &ga::PagerankOptions {
                    damping_factor: damping,
                    max_iterations: max_iter,
                    tolerance,
                    connection_types: conn.as_deref(),
                    scope,
                    interrupt: executor.interrupt(),
                },
            )?;
            executor.centrality_to_rows(&results, yield_items)
        }
        "betweenness" | "betweenness_centrality" => {
            let normalized = call_param_bool(params, "normalized", true);
            let sample_size = call_param_opt_usize(params, "sample_size");
            let conn = call_param_string_list(params, "connection_types");
            let results = ga::betweenness_centrality(
                executor.graph,
                &ga::CentralityOptions {
                    normalized,
                    sample_size,
                    connection_types: conn.as_deref(),
                    scope,
                    interrupt: executor.interrupt(),
                },
            )?;
            executor.centrality_to_rows(&results, yield_items)
        }
        "degree" | "degree_centrality" => {
            let normalized = call_param_bool(params, "normalized", true);
            let conn = call_param_string_list(params, "connection_types");
            let results = ga::degree_centrality(
                executor.graph,
                &ga::DegreeCentralityOptions {
                    normalized,
                    connection_types: conn.as_deref(),
                    scope,
                    interrupt: executor.interrupt(),
                },
            )?;
            executor.centrality_to_rows(&results, yield_items)
        }
        "closeness" | "closeness_centrality" => {
            let normalized = call_param_bool(params, "normalized", true);
            let sample_size = call_param_opt_usize(params, "sample_size");
            let conn = call_param_string_list(params, "connection_types");
            let results = ga::closeness_centrality(
                executor.graph,
                &ga::CentralityOptions {
                    normalized,
                    sample_size,
                    connection_types: conn.as_deref(),
                    scope,
                    interrupt: executor.interrupt(),
                },
            )?;
            executor.centrality_to_rows(&results, yield_items)
        }
        "louvain" | "louvain_communities" => {
            let resolution = call_param_f64(params, "resolution", 1.0);
            let weight_prop = call_param_opt_string(params, "weight_property");
            let conn = call_param_string_list(params, "connection_types");
            let result = ga::louvain_communities(
                executor.graph,
                &ga::CommunityOptions {
                    weight_property: weight_prop.as_deref(),
                    resolution,
                    connection_types: conn.as_deref(),
                    scope,
                    interrupt: if streaming_community {
                        crate::graph::algorithms::Interrupt::default()
                    } else {
                        executor.interrupt()
                    },
                },
            )?;
            executor.community_result_to_rows(&result, yield_items)
        }
        "leiden" | "leiden_communities" => {
            let resolution = call_param_f64(params, "resolution", 1.0);
            let weight_prop = call_param_opt_string(params, "weight_property");
            let conn = call_param_string_list(params, "connection_types");
            let result = ga::leiden_communities(
                executor.graph,
                &ga::CommunityOptions {
                    weight_property: weight_prop.as_deref(),
                    resolution,
                    connection_types: conn.as_deref(),
                    scope,
                    interrupt: if streaming_community {
                        crate::graph::algorithms::Interrupt::default()
                    } else {
                        executor.interrupt()
                    },
                },
            )?;
            executor.community_result_to_rows(&result, yield_items)
        }
        "label_propagation" => {
            let max_iter = call_param_usize(params, "max_iterations", 100);
            let conn = call_param_string_list(params, "connection_types");
            let result = ga::label_propagation(
                executor.graph,
                &ga::LabelPropagationOptions {
                    max_iterations: max_iter,
                    connection_types: conn.as_deref(),
                    scope,
                    interrupt: executor.interrupt(),
                },
            )?;
            executor.community_result_to_rows(&result, yield_items)
        }
        _ => unreachable!("non-centrality procedure routed to centrality dispatcher: {proc_name}"),
    }
}
