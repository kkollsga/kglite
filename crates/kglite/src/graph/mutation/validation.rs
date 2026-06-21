// src/graph/validation.rs
//! Schema validation module for validating graph data against a defined schema.

use crate::datatypes::values::Value;
use crate::graph::schema::{
    ConnectionTypeInfo, DirGraph, NodeSchemaDefinition, SchemaDefinition, ValidationError,
};
use crate::graph::storage::GraphRead;
use std::collections::HashMap;

/// Validate the graph against the provided schema definition
pub fn validate_graph(
    graph: &DirGraph,
    schema: &SchemaDefinition,
    strict: bool, // If true, report undefined types as errors
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // Validate nodes
    errors.extend(validate_nodes(graph, schema, strict));

    // Validate connections
    errors.extend(validate_connections(graph, schema, strict));

    errors
}

/// Validate all nodes against the schema
fn validate_nodes(
    graph: &DirGraph,
    schema: &SchemaDefinition,
    strict: bool,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // Check each node type defined in schema
    for (node_type, node_schema) in &schema.node_schemas {
        if let Some(node_indices) = graph.type_indices.get(node_type) {
            for node_idx in node_indices.iter() {
                if let Some(node) = graph.get_node(node_idx) {
                    errors.extend(validate_single_node(node, node_type, node_schema));
                }
            }
        }
    }

    // Check for undefined node types (strict mode)
    if strict {
        for (node_type, node_indices) in graph.type_indices.iter() {
            if !schema.node_schemas.contains_key(node_type) {
                errors.push(ValidationError::UndefinedNodeType {
                    node_type: node_type.to_string(),
                    count: node_indices.len(),
                });
            }
        }
    }

    errors
}

/// Validate a single node against its schema
fn validate_single_node(
    node: &crate::graph::schema::NodeData,
    node_type: &str,
    schema: &NodeSchemaDefinition,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    let node_title = node.title();
    let title = match &*node_title {
        Value::String(s) => s.clone(),
        _ => format!("{:?}", &*node_title),
    };
    // Check required fields
    for required_field in &schema.required_fields {
        // Skip built-in fields that are always present
        if required_field == "id" || required_field == "title" || required_field == "type" {
            continue;
        }

        let has_field = node
            .get_property(required_field)
            .map(|v| !matches!(*v, Value::Null))
            .unwrap_or(false);

        if !has_field {
            errors.push(ValidationError::MissingRequiredField {
                node_type: node_type.to_string(),
                node_title: title.clone(),
                field: required_field.clone(),
            });
        }
    }

    // Check field types
    for (field, expected_type) in &schema.field_types {
        if let Some(value) = node.get_property(field) {
            if !value_matches_type(&value, expected_type) {
                errors.push(ValidationError::TypeMismatch {
                    node_type: node_type.to_string(),
                    node_title: title.clone(),
                    field: field.clone(),
                    expected_type: expected_type.clone(),
                    actual_type: get_value_type_name(&value),
                });
            }
        }
    }

    errors
}

/// Validate all connections against the schema
fn validate_connections(
    graph: &DirGraph,
    schema: &SchemaDefinition,
    strict: bool,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    let mut connection_type_counts: HashMap<String, usize> = HashMap::new();

    // Iterate through all edges
    let g = &graph.graph;
    for edge_ref in g.edge_references() {
        let edge_data = edge_ref.weight();
        let connection_type = edge_data.connection_type_str(&graph.interner);

        // Count connection types for strict mode check
        *connection_type_counts
            .entry(connection_type.to_string())
            .or_insert(0) += 1;

        // If there's a schema for this connection type, validate it
        if let Some(conn_schema) = schema.connection_schemas.get(connection_type) {
            let source_idx = edge_ref.source();
            let target_idx = edge_ref.target();

            // Get source and target node info
            let (source_type, source_title) = get_node_info(graph, source_idx);
            let (target_type, target_title) = get_node_info(graph, target_idx);

            // Validate endpoint types
            if source_type != conn_schema.source_type || target_type != conn_schema.target_type {
                errors.push(ValidationError::InvalidConnectionEndpoint {
                    connection_type: connection_type.to_string(),
                    expected_source: conn_schema.source_type.clone(),
                    expected_target: conn_schema.target_type.clone(),
                    actual_source: source_type,
                    actual_target: target_type,
                });
            }

            // Validate required properties
            for required_prop in &conn_schema.required_properties {
                let has_prop = edge_data
                    .get_property(required_prop)
                    .map(|v| !matches!(v, Value::Null))
                    .unwrap_or(false);

                if !has_prop {
                    errors.push(ValidationError::MissingConnectionProperty {
                        connection_type: connection_type.to_string(),
                        source_title: source_title.clone(),
                        target_title: target_title.clone(),
                        property: required_prop.clone(),
                    });
                }
            }
        }
    }

    // Check for undefined connection types (strict mode)
    if strict {
        for (conn_type, count) in connection_type_counts {
            if !schema.connection_schemas.contains_key(&conn_type) {
                errors.push(ValidationError::UndefinedConnectionType {
                    connection_type: conn_type,
                    count,
                });
            }
        }
    }

    errors
}

/// Get node type and title from a node index
fn get_node_info(graph: &DirGraph, node_idx: petgraph::graph::NodeIndex) -> (String, String) {
    match graph.get_node(node_idx) {
        Some(node) => {
            let node_title = node.title();
            let title_str = match &*node_title {
                Value::String(s) => s.clone(),
                _ => format!("{:?}", &*node_title),
            };
            (node.node_type_str(&graph.interner).to_string(), title_str)
        }
        None => ("Unknown".to_string(), "Unknown".to_string()),
    }
}

/// Check if a value matches the expected type.
/// Handles both user-facing names ("string", "integer") and metadata names ("String", "Int64").
pub fn value_matches_type(value: &Value, expected_type: &str) -> bool {
    match expected_type.to_lowercase().as_str() {
        "string" | "str" => matches!(value, Value::String(_)),
        "integer" | "int" | "i64" | "int64" => {
            matches!(value, Value::Int64(_) | Value::UniqueId(_))
        }
        "float" | "double" | "f64" | "number" | "float64" => {
            matches!(
                value,
                Value::Float64(_) | Value::Int64(_) | Value::UniqueId(_)
            )
        }
        "boolean" | "bool" => matches!(value, Value::Boolean(_)),
        "datetime" => matches!(value, Value::DateTime(_) | Value::Timestamp(_)),
        "date" => matches!(value, Value::DateTime(_)),
        "timestamp" => matches!(value, Value::Timestamp(_)),
        "uniqueid" => matches!(value, Value::Int64(_) | Value::UniqueId(_)),
        "point" => matches!(value, Value::Point { .. }),
        "null" => matches!(value, Value::Null),
        "any" => true,
        _ => true, // Unknown types default to valid (permissive)
    }
}

/// Get a human-readable type name for a value.
pub fn get_value_type_name(value: &Value) -> String {
    match value {
        Value::String(_) => "string".to_string(),
        Value::Int64(_) => "integer".to_string(),
        Value::Float64(_) => "float".to_string(),
        Value::Boolean(_) => "boolean".to_string(),
        Value::DateTime(_) => "datetime".to_string(),
        Value::Timestamp(_) => "timestamp".to_string(),
        Value::UniqueId(_) => "integer".to_string(),
        Value::Point { .. } => "point".to_string(),
        Value::Duration { .. } => "duration".to_string(),
        Value::Null => "null".to_string(),
        Value::NodeRef(_) => "noderef".to_string(),
        // Phase A.1 — these typically appear in query results, not as
        // stored properties. Validation that sees them is likely
        // surfacing a bug; classify for the error message.
        Value::List(_) => "list".to_string(),
        Value::Map(_) => "map".to_string(),
        Value::Node(_) => "node".to_string(),
        Value::Relationship(_) => "relationship".to_string(),
        Value::Path(_) => "path".to_string(),
    }
}

// ============================================================================
// Edit distance (Levenshtein) for "did you mean?" suggestions
// ============================================================================

/// Compute Levenshtein edit distance between two strings (case-insensitive).
pub fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().flat_map(|c| c.to_lowercase()).collect();
    let b: Vec<char> = b.chars().flat_map(|c| c.to_lowercase()).collect();
    let (m, n) = (a.len(), b.len());
    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut curr = vec![0; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Format a "did you mean 'X'?" suffix, or empty string if no close match.
pub fn did_you_mean(input: &str, candidates: &[&str]) -> String {
    if candidates.is_empty() {
        return String::new();
    }
    let threshold = input.len().clamp(2, 4);
    let mut best: Option<(&str, usize)> = None;
    for &c in candidates {
        let d = edit_distance(input, c);
        if d > 0 && d <= threshold && (best.is_none() || d < best.unwrap().1) {
            best = Some((c, d));
        }
    }
    match best {
        Some((suggestion, _)) => format!(" Did you mean '{}'?", suggestion),
        None => String::new(),
    }
}

// ============================================================================
// Schema-lock validation: pre-mutation checks for CREATE, SET, MERGE
// ============================================================================

/// Built-in fields that are always valid on any node type.
const BUILTIN_FIELDS: &[&str] = &["id", "title", "name", "type"];

/// Validate a node creation against the locked schema.
///
/// Checks:
/// 1. Node type (label) must exist in `node_type_metadata`
/// 2. All property names must be known for that type
/// 3. Property value types must match
/// 4. If `SchemaDefinition` exists, required fields must be present
pub fn validate_node_creation(
    label: &str,
    properties: &HashMap<String, Value>,
    node_type_metadata: &HashMap<String, HashMap<String, String>>,
    schema_def: Option<&SchemaDefinition>,
) -> Result<(), String> {
    // 1. Check node type exists
    let type_props = match node_type_metadata.get(label) {
        Some(props) => props,
        None => {
            let types: Vec<&str> = node_type_metadata.keys().map(|s| s.as_str()).collect();
            let hint = did_you_mean(label, &types);
            let mut valid_list: Vec<&str> = types;
            valid_list.sort();
            return Err(format!(
                "Schema violation: Unknown node type '{}'.{}\n  Valid types: {}",
                label,
                hint,
                valid_list.join(", ")
            ));
        }
    };

    // 2. Check property names and types
    for (prop_name, prop_value) in properties {
        if BUILTIN_FIELDS.contains(&prop_name.as_str()) {
            continue;
        }
        if matches!(prop_value, Value::Null) {
            continue;
        }
        if let Some(expected_type) = type_props.get(prop_name) {
            // 3. Check type matches
            if !value_matches_type(prop_value, expected_type) {
                return Err(format!(
                    "Schema violation: {}.{} expects {}, got {}.",
                    label,
                    prop_name,
                    normalize_type_name(expected_type),
                    get_value_type_name(prop_value)
                ));
            }
        } else {
            let known: Vec<&str> = type_props.keys().map(|s| s.as_str()).collect();
            let hint = did_you_mean(prop_name, &known);
            let mut sorted: Vec<&str> = known;
            sorted.sort();
            return Err(format!(
                "Schema violation: Unknown property '{}' on {}.{}\n  Valid properties: {}",
                prop_name,
                label,
                hint,
                sorted.join(", ")
            ));
        }
    }

    // 4. Required fields (only if SchemaDefinition is available)
    if let Some(schema) = schema_def {
        if let Some(node_schema) = schema.node_schemas.get(label) {
            for required in &node_schema.required_fields {
                if BUILTIN_FIELDS.contains(&required.as_str()) {
                    continue;
                }
                let has_field = properties
                    .get(required)
                    .map(|v| !matches!(v, Value::Null))
                    .unwrap_or(false);
                if !has_field {
                    return Err(format!(
                        "Schema violation: {} nodes require '{}'. Properties provided: {}.",
                        label,
                        required,
                        if properties.is_empty() {
                            "(none)".to_string()
                        } else {
                            let mut keys: Vec<&str> =
                                properties.keys().map(|s| s.as_str()).collect();
                            keys.sort();
                            keys.join(", ")
                        }
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Validate an edge creation against the locked schema.
///
/// Checks:
/// 1. Edge type must exist in `connection_type_metadata`
/// 2. Source node type must be in the allowed source_types
/// 3. Target node type must be in the allowed target_types
pub fn validate_edge_creation(
    edge_type: &str,
    source_type: &str,
    target_type: &str,
    connection_type_metadata: &HashMap<String, ConnectionTypeInfo>,
    node_type_metadata: &HashMap<String, HashMap<String, String>>,
) -> Result<(), String> {
    // 1. Check edge type exists
    let conn_info = match connection_type_metadata.get(edge_type) {
        Some(info) => info,
        None => {
            let types: Vec<&str> = connection_type_metadata
                .keys()
                .map(|s| s.as_str())
                .collect();
            let hint = did_you_mean(edge_type, &types);
            let mut sorted: Vec<&str> = types;
            sorted.sort();
            return Err(format!(
                "Schema violation: Unknown edge type '{}'.{}\n  Valid edge types: {}",
                edge_type,
                hint,
                sorted.join(", ")
            ));
        }
    };

    // Also verify source/target node types exist (catch wrong node type early)
    if !node_type_metadata.contains_key(source_type) {
        let types: Vec<&str> = node_type_metadata.keys().map(|s| s.as_str()).collect();
        let hint = did_you_mean(source_type, &types);
        return Err(format!(
            "Schema violation: Unknown source node type '{}' for {} edge.{}",
            source_type, edge_type, hint
        ));
    }
    if !node_type_metadata.contains_key(target_type) {
        let types: Vec<&str> = node_type_metadata.keys().map(|s| s.as_str()).collect();
        let hint = did_you_mean(target_type, &types);
        return Err(format!(
            "Schema violation: Unknown target node type '{}' for {} edge.{}",
            target_type, edge_type, hint
        ));
    }

    // 2-3. Check source/target types are valid for this edge type
    if !conn_info.source_types.contains(source_type)
        || !conn_info.target_types.contains(target_type)
    {
        let sources: Vec<&str> = conn_info.source_types.iter().map(|s| s.as_str()).collect();
        let targets: Vec<&str> = conn_info.target_types.iter().map(|s| s.as_str()).collect();
        return Err(format!(
            "Schema violation: {} edges connect {} -> {}, not {} -> {}.",
            edge_type,
            sources.join("|"),
            targets.join("|"),
            source_type,
            target_type
        ));
    }

    Ok(())
}

/// Validate a SET property operation against the locked schema.
///
/// Checks:
/// 1. Property name must be known for the node type
/// 2. Value type must match the expected type
pub fn validate_property_set(
    node_type: &str,
    property: &str,
    value: &Value,
    node_type_metadata: &HashMap<String, HashMap<String, String>>,
) -> Result<(), String> {
    // Built-in fields are always allowed
    if BUILTIN_FIELDS.contains(&property) {
        return Ok(());
    }

    // Null values are always allowed (clearing a property)
    if matches!(value, Value::Null) {
        return Ok(());
    }

    let type_props = match node_type_metadata.get(node_type) {
        Some(props) => props,
        None => return Ok(()), // Unknown type — don't block SET on matched nodes
    };

    // 1. Check property exists
    if let Some(expected_type) = type_props.get(property) {
        // 2. Check type matches
        if !value_matches_type(value, expected_type) {
            return Err(format!(
                "Schema violation: {}.{} expects {}, got {}.",
                node_type,
                property,
                normalize_type_name(expected_type),
                get_value_type_name(value)
            ));
        }
    } else {
        let known: Vec<&str> = type_props.keys().map(|s| s.as_str()).collect();
        let hint = did_you_mean(property, &known);
        let mut sorted: Vec<&str> = known;
        sorted.sort();
        return Err(format!(
            "Schema violation: Unknown property '{}' on {}.{}\n  Valid properties: {}",
            property,
            node_type,
            hint,
            sorted.join(", ")
        ));
    }

    Ok(())
}

/// Normalize metadata type names to user-friendly names for error messages.
fn normalize_type_name(type_name: &str) -> &str {
    match type_name.to_lowercase().as_str() {
        "string" | "str" => "string",
        "int64" | "integer" | "int" | "i64" => "integer",
        "float64" | "float" | "double" | "f64" | "number" => "float",
        "boolean" | "bool" => "boolean",
        "datetime" | "date" => "datetime",
        "uniqueid" => "integer",
        "point" => "point",
        _ => type_name,
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;

    #[test]
    fn test_value_matches_type() {
        assert!(value_matches_type(
            &Value::String("test".to_string()),
            "string"
        ));
        assert!(value_matches_type(&Value::Int64(42), "integer"));
        assert!(value_matches_type(&Value::Float64(3.14), "float"));
        assert!(value_matches_type(&Value::Boolean(true), "boolean"));
        assert!(!value_matches_type(
            &Value::String("test".to_string()),
            "integer"
        ));
    }

    #[test]
    fn test_value_matches_metadata_types() {
        // Metadata stores types as "Int64", "Float64", "String", etc.
        assert!(value_matches_type(&Value::Int64(42), "Int64"));
        assert!(value_matches_type(&Value::Float64(3.14), "Float64"));
        assert!(value_matches_type(
            &Value::String("x".to_string()),
            "String"
        ));
        assert!(value_matches_type(&Value::UniqueId(1), "UniqueId"));
        assert!(!value_matches_type(
            &Value::String("x".to_string()),
            "Int64"
        ));
    }

    #[test]
    fn test_edit_distance() {
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("abc", ""), 3);
        assert_eq!(edit_distance("abc", "abc"), 0);
        assert_eq!(edit_distance("Paper", "Papier"), 1); // case-insensitive
        assert_eq!(edit_distance("Person", "Persom"), 1);
    }

    #[test]
    fn test_did_you_mean() {
        let candidates = vec!["Paper", "Person", "Software", "Grant"];
        assert!(did_you_mean("Papier", &candidates).contains("Paper"));
        assert!(did_you_mean("Persom", &candidates).contains("Person"));
        assert!(did_you_mean("ZZZZZZZZZ", &candidates).is_empty());
    }
}
