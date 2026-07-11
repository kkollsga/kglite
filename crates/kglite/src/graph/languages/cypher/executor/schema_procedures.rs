//! Schema-introspection procedures exposed through Cypher `CALL`.

use std::collections::HashMap;

use super::call_clause::{compute_property_stats, indexes_to_rows, names_to_rows};
use super::helpers::call_param_string;
use super::{CypherExecutor, ResultRow};
use crate::datatypes::values::Value;
use crate::graph::languages::cypher::ast::YieldItem;
use crate::graph::storage::GraphRead;

/// Dispatch schema procedures after shared CALL validation.
pub(super) fn execute_schema_procedure(
    executor: &CypherExecutor<'_>,
    proc_name: &str,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    match proc_name {
        "db.labels" => {
            let labels =
                crate::graph::introspection::schema_overview::collect_labels(executor.graph);
            Ok(names_to_rows(&labels, yield_items))
        }
        "db.relationshiptypes" => {
            let names = crate::graph::introspection::schema_overview::collect_relationship_types(
                executor.graph,
            );
            Ok(names_to_rows(&names, yield_items))
        }
        "db.indexes" => {
            let indexes = crate::graph::introspection::schema_overview::collect_indexes_structured(
                executor.graph,
            );
            Ok(indexes_to_rows(&indexes, yield_items))
        }
        "db.propertykeys" => {
            let keys =
                crate::graph::introspection::schema_overview::collect_property_keys(executor.graph);
            Ok(names_to_rows(&keys, yield_items))
        }
        "db.schema" => {
            let schema =
                crate::graph::introspection::schema_overview::compute_schema(executor.graph);
            let mut rows = Vec::with_capacity(schema.node_types.len());
            for (node_type, overview) in &schema.node_types {
                let mut props: Vec<String> = overview.properties.keys().cloned().collect();
                props.sort();
                let mut row = ResultRow::new();
                for item in yield_items {
                    let alias = item.alias.as_deref().unwrap_or(&item.name);
                    let value = match item.name.as_str() {
                        "nodeType" => Value::String(node_type.clone()),
                        "properties" => {
                            Value::List(props.iter().cloned().map(Value::String).collect())
                        }
                        _ => continue,
                    };
                    row.projected.insert(alias.to_string(), value);
                }
                rows.push(row);
            }
            Ok(rows)
        }
        "db.graph_stats" => {
            let node_count = executor.graph.graph.node_count() as i64;
            let edge_count = executor.graph.graph.edge_count() as i64;
            let label_count =
                crate::graph::introspection::schema_overview::collect_labels(executor.graph).len()
                    as i64;
            let relationship_type_count =
                crate::graph::introspection::schema_overview::collect_relationship_types(
                    executor.graph,
                )
                .len() as i64;
            let mut row = ResultRow::new();
            for item in yield_items {
                let alias = item.alias.as_deref().unwrap_or(&item.name);
                let value = match item.name.as_str() {
                    "node_count" => Value::Int64(node_count),
                    "edge_count" => Value::Int64(edge_count),
                    "label_count" => Value::Int64(label_count),
                    "relationship_type_count" => Value::Int64(relationship_type_count),
                    _ => continue,
                };
                row.projected.insert(alias.to_string(), value);
            }
            Ok(vec![row])
        }
        "db.property_stats" | "db.property_uniqueness" => {
            let node_type = call_param_string(params, "node_type")
                .ok_or_else(|| format!("{proc_name}() requires a `node_type` string param"))?;
            let property = call_param_string(params, "property")
                .ok_or_else(|| format!("{proc_name}() requires a `property` string param"))?;
            let (value_count, null_count, distinct_count) =
                compute_property_stats(executor, &node_type, &property)?;
            let mut row = ResultRow::new();
            for item in yield_items {
                let alias = item.alias.as_deref().unwrap_or(&item.name);
                let value = match (proc_name, item.name.as_str()) {
                    ("db.property_stats", "value_count") => Value::Int64(value_count),
                    ("db.property_stats", "null_count") => Value::Int64(null_count),
                    ("db.property_stats", "distinct_count") => Value::Int64(distinct_count),
                    ("db.property_uniqueness", "is_unique") => {
                        Value::Boolean(value_count > 0 && value_count == distinct_count)
                    }
                    ("db.property_uniqueness", "violation_count") => {
                        Value::Int64(value_count.saturating_sub(distinct_count))
                    }
                    ("db.property_uniqueness", "distinct_count") => Value::Int64(distinct_count),
                    _ => continue,
                };
                row.projected.insert(alias.to_string(), value);
            }
            Ok(vec![row])
        }
        _ => unreachable!("non-schema procedure routed to schema dispatcher: {proc_name}"),
    }
}
