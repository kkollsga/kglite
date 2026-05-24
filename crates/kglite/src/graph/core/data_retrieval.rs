// src/graph/data_retrieval.rs
use crate::datatypes::values::{format_value, Value};
use crate::graph::schema::{CurrentSelection, DirGraph, NodeInfo};
use crate::graph::storage::GraphRead;
use petgraph::graph::NodeIndex;
use std::borrow::Cow;
use std::collections::HashMap;

#[derive(Debug)]
pub struct LevelNodes {
    pub parent_title: String,
    pub parent_id: Option<Value>,
    pub parent_idx: Option<NodeIndex>,
    pub parent_type: Option<String>,
    pub nodes: Vec<NodeInfo>,
}

#[derive(Debug)]
pub struct LevelValues {
    pub parent_title: String,
    pub values: Vec<Vec<Value>>,
}

pub fn get_nodes(
    graph: &DirGraph,
    selection: &CurrentSelection,
    level_index: Option<usize>,
    indices: Option<&[usize]>,
    max_nodes: Option<usize>,
) -> Vec<LevelNodes> {
    // If specific indices are provided, do direct lookup
    if let Some(idx) = indices {
        let mut direct_nodes = Vec::new();
        for &index in idx {
            if let Some(node_idx) = NodeIndex::new(index).into() {
                if let Some(node) = graph.get_node(node_idx) {
                    let node_info = node.to_node_info(&graph.interner);
                    direct_nodes.push(node_info);
                    if let Some(max) = max_nodes {
                        if direct_nodes.len() >= max {
                            break;
                        }
                    }
                }
            }
        }

        if !direct_nodes.is_empty() {
            return vec![LevelNodes {
                parent_title: "Direct Lookup".to_string(),
                parent_id: None,
                parent_idx: None,
                parent_type: None,
                nodes: direct_nodes,
            }];
        }
        return Vec::new();
    }

    // Check if selection is effectively empty (no nodes selected)
    // Note: CurrentSelection always has at least one level, so we check if the level is empty
    let selection_is_empty = if selection.get_level_count() > 0 {
        let level_idx = selection.get_level_count().saturating_sub(1);
        selection
            .get_level(level_idx)
            .map(|l| l.node_count() == 0)
            .unwrap_or(true)
    } else {
        true
    };

    // Check if any query operations have been applied (type_filter, filter, traverse, etc.)
    let has_query_operations = !selection.get_execution_plan().is_empty();

    // If selection is empty AND no query operations were applied, return all regular nodes.
    // If selection is empty BUT query operations were applied (e.g., filter matched 0 nodes),
    // return empty to respect the query result.
    if selection_is_empty && !has_query_operations {
        let mut all_nodes = Vec::new();
        for node_idx in GraphRead::node_indices(&graph.graph) {
            if let Some(node) = graph.get_node(node_idx) {
                let node_info = node.to_node_info(&graph.interner);
                all_nodes.push(node_info);
                if let Some(max) = max_nodes {
                    if all_nodes.len() >= max {
                        break;
                    }
                }
            }
        }

        if !all_nodes.is_empty() {
            return vec![LevelNodes {
                parent_title: "Root".to_string(),
                parent_id: None,
                parent_idx: None,
                parent_type: None,
                nodes: all_nodes,
            }];
        }
        return Vec::new();
    }

    let level_idx = level_index.unwrap_or_else(|| selection.get_level_count().saturating_sub(1));
    let mut result = Vec::new();

    if let Some(level) = selection.get_level(level_idx) {
        for (parent, children) in level.iter_groups() {
            let mut nodes = Vec::new();

            for &child_idx in children {
                if let Some(node) = graph.get_node(child_idx) {
                    let node_info = node.to_node_info(&graph.interner);
                    nodes.push(node_info);
                    if let Some(max) = max_nodes {
                        if nodes.len() >= max {
                            break;
                        }
                    }
                }
            }

            // Always create an entry for the parent, even if nodes is empty
            let (parent_title, parent_id, parent_type) = match parent {
                Some(p) => {
                    if let Some(node) = graph.get_node(*p) {
                        (
                            node.get_field_ref("title")
                                .as_deref()
                                .and_then(|v| match v {
                                    Value::String(s) => Some(s.clone()),
                                    _ => None,
                                })
                                .unwrap_or_else(|| "Unknown".to_string()),
                            node.get_field_ref("id").map(Cow::into_owned),
                            Some(node.get_node_type_ref(&graph.interner).to_string()),
                        )
                    } else {
                        ("Unknown".to_string(), None, None)
                    }
                }
                None => ("Root".to_string(), None, None),
            };

            result.push(LevelNodes {
                parent_title,
                parent_id,
                parent_idx: parent.map(|p| p),
                parent_type,
                nodes,
            });
        }
    }
    result
}

pub fn get_property_values(
    graph: &DirGraph,
    selection: &CurrentSelection,
    level_index: Option<usize>,
    properties: &[&str],
    indices: Option<&[usize]>,
    max_nodes: Option<usize>,
) -> Vec<LevelValues> {
    let level_idx = level_index.unwrap_or_else(|| selection.get_level_count().saturating_sub(1));
    let mut result = Vec::new();

    if let Some(level) = selection.get_level(level_idx) {
        for (parent, children) in level.iter_groups() {
            let filtered_children: Vec<NodeIndex> = match indices {
                Some(idx) => children
                    .iter()
                    .filter(|&c| idx.contains(&c.index()))
                    .take(max_nodes.unwrap_or(usize::MAX))
                    .cloned()
                    .collect(),
                None => children
                    .iter()
                    .take(max_nodes.unwrap_or(usize::MAX))
                    .cloned()
                    .collect(),
            };

            // Always create values vector, even if empty
            let values: Vec<Vec<Value>> = filtered_children
                .iter()
                .map(|&idx| {
                    properties
                        .iter()
                        .map(|&prop| {
                            graph
                                .get_node(idx)
                                .and_then(|node| node.get_field_ref(prop))
                                .map(Cow::into_owned)
                                .unwrap_or(Value::Null)
                        })
                        .collect()
                })
                .collect();

            // Get parent title even if there are no children
            let parent_title = match parent {
                Some(p) => {
                    if let Some(node) = graph.get_node(*p) {
                        if let Some(Value::String(title)) = node.get_field_ref("title").as_deref() {
                            title.clone()
                        } else {
                            "Unknown".to_string()
                        }
                    } else {
                        "Unknown".to_string()
                    }
                }
                None => "Root".to_string(),
            };

            // Always add to result, even with empty values
            result.push(LevelValues {
                parent_title,
                values,
            });
        }
    }
    result
}

#[derive(Debug)]
pub struct UniqueValues {
    pub parent_title: String,
    pub parent_idx: Option<NodeIndex>,
    pub values: Vec<Value>,
}

pub fn get_unique_values(
    graph: &DirGraph,
    selection: &CurrentSelection,
    property: &str,
    level_index: Option<usize>,
    group_by_parent: bool,
    indices: Option<&[usize]>,
) -> Vec<UniqueValues> {
    let level_idx = level_index.unwrap_or_else(|| selection.get_level_count().saturating_sub(1));
    let mut result = Vec::new();

    if let Some(level) = selection.get_level(level_idx) {
        if group_by_parent {
            for (parent, children) in level.iter_groups() {
                let filtered_children: Vec<NodeIndex> = match indices {
                    Some(idx) => children
                        .iter()
                        .filter(|&c| idx.contains(&c.index()))
                        .cloned()
                        .collect(),
                    None => children.clone(),
                };

                let mut unique_values = std::collections::HashSet::new();

                for &idx in &filtered_children {
                    if let Some(node) = graph.get_node(idx) {
                        if let Some(value) = node.get_field_ref(property) {
                            unique_values.insert(value.into_owned());
                        }
                    }
                }

                let parent_title = match parent {
                    Some(p) => {
                        if let Some(node) = graph.get_node(*p) {
                            if let Some(Value::String(title)) =
                                node.get_field_ref("title").as_deref()
                            {
                                title.clone()
                            } else {
                                "Unknown".to_string()
                            }
                        } else {
                            "Unknown".to_string()
                        }
                    }
                    None => "Root".to_string(),
                };

                result.push(UniqueValues {
                    parent_title,
                    parent_idx: parent.map(|p| p),
                    values: unique_values.into_iter().collect(),
                });
            }
        } else {
            let mut all_unique_values = std::collections::HashSet::new();

            for (_, children) in level.iter_groups() {
                let filtered_children: Vec<NodeIndex> = match indices {
                    Some(idx) => children
                        .iter()
                        .filter(|&c| idx.contains(&c.index()))
                        .cloned()
                        .collect(),
                    None => children.clone(),
                };

                for &idx in &filtered_children {
                    if let Some(node) = graph.get_node(idx) {
                        if let Some(value) = node.get_field_ref(property) {
                            all_unique_values.insert(value.into_owned());
                        }
                    }
                }
            }

            result.push(UniqueValues {
                parent_title: "All".to_string(),
                parent_idx: None,
                values: all_unique_values.into_iter().collect(),
            });
        }
    }

    result
}

pub fn format_unique_values_for_storage(
    values: &[UniqueValues],
    max_length: Option<usize>,
) -> Vec<(Option<NodeIndex>, Value)> {
    values
        .iter()
        .map(|unique_values| {
            let mut value_list: Vec<String> = unique_values
                .values
                .iter()
                .map(|v| {
                    // Get formatted value
                    let formatted = format_value(v);

                    // Remove quotes from strings (if present)
                    match v {
                        Value::String(_) => {
                            // The format_value function wraps strings in quotes
                            // We need to remove the opening and closing quotes
                            if formatted.starts_with('"') && formatted.ends_with('"') {
                                formatted[1..formatted.len() - 1].to_string()
                            } else {
                                formatted
                            }
                        }
                        _ => formatted,
                    }
                })
                .collect::<Vec<String>>();

            value_list.sort();
            value_list.dedup();

            if let Some(max_len) = max_length {
                if value_list.len() > max_len {
                    println!(
                        "Warning: Truncating value list from {} to {} items for parent: {}",
                        value_list.len(),
                        max_len,
                        unique_values.parent_title
                    );
                    value_list.truncate(max_len);
                }
            }

            // Join with comma and space
            (
                unique_values.parent_idx,
                Value::String(value_list.join(", ")),
            )
        })
        .collect()
}

#[derive(Debug)]
#[allow(clippy::type_complexity)]
pub struct ConnectionInfo {
    pub node_id: Value,
    pub node_title: String,
    pub node_type: String,
    pub incoming: Vec<(
        String,
        Value,
        Value,
        HashMap<String, Value>,
        Option<HashMap<String, Value>>,
    )>, // (type, id, title, conn_props, node_props)
    pub outgoing: Vec<(
        String,
        Value,
        Value,
        HashMap<String, Value>,
        Option<HashMap<String, Value>>,
    )>, // (type, id, title, conn_props, node_props)
}

#[derive(Debug)]
pub struct LevelConnections {
    pub parent_title: String,
    pub parent_id: Option<Value>,
    pub parent_idx: Option<NodeIndex>,
    pub parent_type: Option<String>,
    pub connections: Vec<ConnectionInfo>,
}

pub fn get_connections(
    graph: &DirGraph,
    selection: &CurrentSelection,
    level_index: Option<usize>,
    indices: Option<&[usize]>,
    include_node_properties: bool,
) -> Vec<LevelConnections> {
    let level_idx = level_index.unwrap_or_else(|| selection.get_level_count().saturating_sub(1));
    let mut result = Vec::new();

    if let Some(level) = selection.get_level(level_idx) {
        // Handle direct lookup if indices provided
        let nodes = if let Some(idx) = indices {
            idx.iter()
                .filter_map(|&i| NodeIndex::new(i).into())
                .collect::<Vec<_>>()
        } else {
            level.get_all_nodes()
        };

        // If using direct indices, create a single level
        let groups = if indices.is_some() {
            vec![(None, nodes)]
        } else {
            level.iter_groups().map(|(p, c)| (*p, c.clone())).collect()
        };

        for (parent, children) in groups {
            let mut level_connections = Vec::new();

            for node_idx in children {
                if let Some(node) = graph.get_node(node_idx) {
                    let node_title = node.title();
                    let title_str = match &*node_title {
                        Value::String(s) => s.clone(),
                        _ => "Unknown".to_string(),
                    };

                    let mut incoming = Vec::new();
                    let mut outgoing = Vec::new();

                    // Collect incoming connections
                    for edge_ref in graph
                        .graph
                        .edges_directed(node_idx, petgraph::Direction::Incoming)
                    {
                        if let Some(source_node) = graph.get_node(edge_ref.source()) {
                            let edge_data = edge_ref.weight();
                            let node_props = if include_node_properties {
                                Some(source_node.properties_cloned(&graph.interner))
                            } else {
                                None
                            };
                            incoming.push((
                                edge_data.connection_type_str(&graph.interner).to_string(),
                                source_node
                                    .get_field_ref("id")
                                    .map(Cow::into_owned)
                                    .unwrap_or(Value::Null),
                                source_node
                                    .get_field_ref("title")
                                    .map(Cow::into_owned)
                                    .unwrap_or(Value::Null),
                                edge_data.properties_cloned(&graph.interner),
                                node_props,
                            ));
                        }
                    }

                    // Collect outgoing connections
                    for edge_ref in graph
                        .graph
                        .edges_directed(node_idx, petgraph::Direction::Outgoing)
                    {
                        if let Some(target_node) = graph.get_node(edge_ref.target()) {
                            let edge_data = edge_ref.weight();
                            let node_props = if include_node_properties {
                                Some(target_node.properties_cloned(&graph.interner))
                            } else {
                                None
                            };
                            outgoing.push((
                                edge_data.connection_type_str(&graph.interner).to_string(),
                                target_node
                                    .get_field_ref("id")
                                    .map(Cow::into_owned)
                                    .unwrap_or(Value::Null),
                                target_node
                                    .get_field_ref("title")
                                    .map(Cow::into_owned)
                                    .unwrap_or(Value::Null),
                                edge_data.properties_cloned(&graph.interner),
                                node_props,
                            ));
                        }
                    }

                    if !incoming.is_empty() || !outgoing.is_empty() {
                        level_connections.push(ConnectionInfo {
                            node_id: node.id().into_owned(),
                            node_title: title_str,
                            node_type: node.node_type_str(&graph.interner).to_string(),
                            incoming,
                            outgoing,
                        });
                    }
                }
            }

            // Rest of the function remains the same
            let (parent_title, parent_id, parent_type) = if indices.is_some() {
                ("Direct Lookup".to_string(), None, None)
            } else {
                match parent {
                    Some(p) => {
                        if let Some(node) = graph.get_node(p) {
                            (
                                node.get_field_ref("title")
                                    .as_deref()
                                    .and_then(|v| match v {
                                        Value::String(s) => Some(s.clone()),
                                        _ => None,
                                    })
                                    .unwrap_or_else(|| "Unknown".to_string()),
                                node.get_field_ref("id").map(Cow::into_owned),
                                Some(node.get_node_type_ref(&graph.interner).to_string()),
                            )
                        } else {
                            ("Unknown".to_string(), None, None)
                        }
                    }
                    None => ("Root".to_string(), None, None),
                }
            };

            result.push(LevelConnections {
                parent_title,
                parent_id,
                parent_idx: parent,
                parent_type,
                connections: level_connections,
            });
        }
    }
    result
}
