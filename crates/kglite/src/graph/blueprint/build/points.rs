//! Point columns: a spec property typed `point` is loaded as the text
//! `point(lat, lon)` (a frame has no point column) and converted to
//! `Value::Point` here, once the rows are in. Relationship properties have no
//! such pass, so an edge column typed `point` stays text.

use super::specs::FlatSpec;
use crate::datatypes::values::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::mutation::property_updates;
use petgraph::graph::NodeIndex;

/// `point(lat, lon)` as written by the lossless CSV export.
pub(crate) fn parse_point(text: &str) -> Option<Value> {
    let inner = text.trim().strip_prefix("point(")?.strip_suffix(')')?;
    let (lat, lon) = inner.split_once(',')?;
    Some(Value::Point {
        lat: lat.trim().parse().ok()?,
        lon: lon.trim().parse().ok()?,
    })
}

const BATCH: usize = 100_000;

pub(super) fn convert_point_columns(
    graph: &mut DirGraph,
    specs: &[&FlatSpec],
) -> Result<(), String> {
    for spec in specs {
        for (column, keyword) in &spec.spec.properties {
            if keyword != "point" {
                continue;
            }
            let indices: Vec<NodeIndex> = graph
                .type_indices
                .get(&spec.node_type)
                .map(|nodes| nodes.iter().collect())
                .unwrap_or_default();
            for chunk in indices.chunks(BATCH) {
                let updates: Vec<(Option<NodeIndex>, Value)> = {
                    use crate::graph::storage::GraphRead;
                    chunk
                        .iter()
                        .filter_map(|&idx| {
                            let node = graph.graph.node_view(idx)?;
                            match node.get_property(column)?.as_ref() {
                                Value::String(s) => parse_point(s).map(|p| (Some(idx), p)),
                                _ => None,
                            }
                        })
                        .collect()
                };
                if !updates.is_empty() {
                    property_updates::update_node_properties(graph, &updates, column).map_err(
                        |e| format!("node '{}': point column '{column}': {e}", spec.node_type),
                    )?;
                }
            }
        }
    }
    Ok(())
}
