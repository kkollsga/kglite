//! Schema-introspection procedures exposed through Cypher `CALL`.

use std::collections::HashMap;

use super::call_clause::{
    compute_property_stats, constraints_to_rows, indexes_to_rows, names_to_rows,
};
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
        "db.constraints" => {
            let constraints =
                crate::graph::introspection::schema_overview::collect_constraints_structured(
                    executor.graph,
                );
            Ok(constraints_to_rows(&constraints, yield_items))
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
        "db.schema.visualization" => {
            // Neo4j's schema-graph shape: ONE row, two columns — a virtual
            // Node per label (properties: name, indexes, constraints) and a
            // virtual Relationship per (source label, type, target label)
            // combination. Neo4j Browser's schema tab renders exactly this.
            //
            // The relationship set is the cross-product of each connection
            // type's observed source and target label sets — the same
            // approximation Neo4j makes: a multi-endpoint type fans out, and
            // a (src, tgt) pair that never co-occurred on one edge can still
            // appear. Virtual ids are label ordinals, not graph ids.
            use crate::datatypes::values::{NodeValue, RelValue};
            use std::collections::BTreeMap;

            let graph = executor.graph;
            let labels = crate::graph::introspection::schema_overview::collect_labels(graph);
            let indexes =
                crate::graph::introspection::schema_overview::collect_indexes_structured(graph);
            let constraints =
                crate::graph::introspection::schema_overview::collect_constraints_structured(graph);

            let mut id_of: BTreeMap<&str, u32> = BTreeMap::new();
            let mut nodes: Vec<Value> = Vec::with_capacity(labels.len());
            for (ordinal, label) in labels.iter().enumerate() {
                id_of.insert(label.as_str(), ordinal as u32);
                let names_on =
                    |all: Vec<String>| Value::List(all.into_iter().map(Value::String).collect());
                let index_names: Vec<String> = indexes
                    .iter()
                    .filter(|info| info.labels_or_types.iter().any(|l| l == label))
                    .map(|info| info.name.clone())
                    .collect();
                let constraint_names: Vec<String> = constraints
                    .iter()
                    .filter(|info| info.labels_or_types.iter().any(|l| l == label))
                    .map(|info| info.name.clone())
                    .collect();
                let mut properties = BTreeMap::new();
                properties.insert("name".to_string(), Value::String(label.clone()));
                properties.insert("indexes".to_string(), names_on(index_names));
                properties.insert("constraints".to_string(), names_on(constraint_names));
                nodes.push(Value::Node(Box::new(NodeValue {
                    id: ordinal as u32,
                    labels: vec![label.clone()],
                    properties,
                })));
            }

            let stats =
                crate::graph::introspection::schema_overview::compute_connection_type_stats(graph);
            let mut relationships: Vec<Value> = Vec::new();
            let mut rel_id: u32 = 0;
            for stat in &stats {
                for source in &stat.source_types {
                    for target in &stat.target_types {
                        if let (Some(&start_id), Some(&end_id)) =
                            (id_of.get(source.as_str()), id_of.get(target.as_str()))
                        {
                            relationships.push(Value::Relationship(Box::new(RelValue {
                                id: rel_id,
                                start_id,
                                end_id,
                                rel_type: stat.connection_type.clone(),
                                properties: BTreeMap::new(),
                            })));
                            rel_id += 1;
                        }
                    }
                }
            }

            let mut row = ResultRow::new();
            for item in yield_items {
                let alias = item.alias.as_deref().unwrap_or(&item.name);
                let value = match item.name.as_str() {
                    "nodes" => Value::List(nodes.clone()),
                    "relationships" => Value::List(relationships.clone()),
                    _ => continue,
                };
                row.projected.insert(alias.to_string(), value);
            }
            Ok(vec![row])
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
