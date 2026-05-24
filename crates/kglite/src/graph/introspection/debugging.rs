// src/graph/debugging.rs
use crate::datatypes::values::Value;
use crate::graph::schema::{CurrentSelection, DirGraph, SelectionOperation};

pub fn get_schema_string(graph: &DirGraph) -> String {
    let mut schema_string = String::from("Graph Schema:\n");
    let mut has_metadata = false;

    // ── Node type metadata ──────────────────────────────────────────────
    let mut node_types: Vec<&String> = graph.node_type_metadata.keys().collect();
    node_types.sort();

    for node_type in node_types {
        has_metadata = true;
        schema_string.push_str(&format!("  Node Type: {}\n", node_type));

        if let Some(props) = graph.node_type_metadata.get(node_type.as_str()) {
            if !props.is_empty() {
                schema_string.push_str("    Properties:\n");
                let mut prop_keys: Vec<&String> = props.keys().collect();
                prop_keys.sort();
                for key in prop_keys {
                    if let Some(type_name) = props.get(key.as_str()) {
                        schema_string.push_str(&format!("      - {}: {}\n", key, type_name));
                    }
                }
            }
        }
        schema_string.push('\n');
    }

    // ── Connection type metadata ────────────────────────────────────────
    let mut conn_types: Vec<&String> = graph.connection_type_metadata.keys().collect();
    conn_types.sort();

    for conn_type in conn_types {
        has_metadata = true;
        if let Some(info) = graph.connection_type_metadata.get(conn_type.as_str()) {
            let sources: Vec<&String> = info.source_types.iter().collect();
            let targets: Vec<&String> = info.target_types.iter().collect();
            schema_string.push_str(&format!(
                "  Connection Type: {} ({} -> {})\n",
                conn_type,
                sources
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
                targets
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
            ));

            if !info.property_types.is_empty() {
                schema_string.push_str("    Properties:\n");
                let mut prop_keys: Vec<&String> = info.property_types.keys().collect();
                prop_keys.sort();
                for key in prop_keys {
                    if let Some(type_name) = info.property_types.get(key.as_str()) {
                        schema_string.push_str(&format!("      - {}: {}\n", key, type_name));
                    }
                }
            }
            schema_string.push('\n');
        }
    }

    if !has_metadata {
        schema_string.push_str("  No schema metadata found.\n");
    }

    schema_string
}

pub fn get_selection_string(graph: &DirGraph, selection: &CurrentSelection) -> String {
    let mut output = String::from("Selection State:\n");
    if selection.get_level_count() == 0 {
        output.push_str("    No active selections\n");
        return output;
    }

    // Iterate through levels using level count and get_level
    for level_idx in 0..selection.get_level_count() {
        let level = selection
            .get_level(level_idx)
            .expect("Level should exist as we checked count");

        output.push_str(&format!("\nLevel {}:\n", level_idx));

        // Print operations that created this level
        if !level.operations.is_empty() {
            output.push_str("    Operations:\n");
            for op in &level.operations {
                match op {
                    SelectionOperation::Filter(conditions) => {
                        output.push_str("        Filter: ");
                        let conditions_str: Vec<String> = conditions
                            .iter()
                            .map(|(key, condition)| format!("{}={:?}", key, condition))
                            .collect();
                        output.push_str(&conditions_str.join(", "));
                        output.push('\n');
                    }
                    SelectionOperation::Sort(sort_fields) => {
                        output.push_str("        Sort: ");
                        let sorts_str: Vec<String> = sort_fields
                            .iter()
                            .map(|(field, ascending)| {
                                format!("{} {}", field, if *ascending { "↑" } else { "↓" })
                            })
                            .collect();
                        output.push_str(&sorts_str.join(", "));
                        output.push('\n');
                    }
                    SelectionOperation::Traverse {
                        connection_type,
                        direction,
                        max_nodes,
                    } => {
                        let mut traverse_parts = vec![format!("type={}", connection_type)];
                        if let Some(dir) = direction {
                            traverse_parts.push(format!("dir={}", dir));
                        }
                        if let Some(max) = max_nodes {
                            traverse_parts.push(format!("max={}", max));
                        }
                        output.push_str("        Traverse: ");
                        output.push_str(&traverse_parts.join(", "));
                        output.push('\n');
                    }
                    SelectionOperation::Custom(description) => {
                        output.push_str("        Custom: ");
                        output.push_str(description);
                        output.push('\n');
                    }
                }
            }
        }

        // Print selection structure
        output.push_str("    Selections:\n");
        for (parent, children) in level.iter_groups() {
            match parent {
                Some(parent_idx) => {
                    if let Some(parent_node) = graph.get_node(*parent_idx) {
                        let parent_title = parent_node.title();
                        let title_str = match &*parent_title {
                            Value::String(s) => s.clone(),
                            _ => format!("{:?}", &*parent_title),
                        };
                        output.push_str(&format!(
                            "        Parent [{}]: {} - {}\n",
                            parent_idx.index(),
                            parent_node.node_type_str(&graph.interner),
                            title_str
                        ));
                    }
                }
                None => output.push_str("        Root Selection\n"),
            }

            // Add children info with node type if available
            if !children.is_empty() {
                let first_child = graph.get_node(children[0]);
                let child_type = first_child.map(|node| node.node_type_str(&graph.interner));

                let indices: Vec<String> = children
                    .iter()
                    .take(5)
                    .map(|idx| idx.index().to_string())
                    .collect();

                let indices_str = if children.len() > 5 {
                    format!("[{}, ...]", indices.join(", "))
                } else {
                    format!("[{}]", indices.join(", "))
                };

                output.push_str(&format!(
                    "            Children: {} {}{}\n",
                    children.len(),
                    child_type.map(|t| format!("({})", t)).unwrap_or_default(),
                    if children.len() <= 5 {
                        format!(": {}", indices_str)
                    } else {
                        format!(", e.g. {}", indices_str)
                    }
                ));
            }
        }
    }
    output
}
