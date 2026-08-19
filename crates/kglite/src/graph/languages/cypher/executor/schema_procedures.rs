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
        "db.schema.visualization" => schema_visualization_rows(executor, yield_items),
        "db.schema.nodetypeproperties" | "db.schema.reltypeproperties" => {
            schema_type_properties_rows(
                executor,
                proc_name == "db.schema.nodetypeproperties",
                false,
                yield_items,
            )
        }
        // The two APOC schema names, served as deliberate compatibility
        // shims: same data as the db.schema.* pair, under APOC's column set —
        // including the rel side's sourceNodeLabels/targetNodeLabels, which
        // the db.schema.* contract lacks and which clients (G.V(), measured
        // 2026-08-15) require to draw schema-graph edges at all. Scoped to
        // exactly these two names: no other apoc.* resolves.
        "apoc.meta.nodetypeproperties" | "apoc.meta.reltypeproperties" => {
            schema_type_properties_rows(
                executor,
                proc_name == "apoc.meta.nodetypeproperties",
                true,
                yield_items,
            )
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

/// `db.schema.visualization()` — extracted from the dispatcher to keep it a
/// table of thin delegations.
fn schema_visualization_rows(
    executor: &CypherExecutor<'_>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
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
    let indexes = crate::graph::introspection::schema_overview::collect_indexes_structured(graph);
    let constraints =
        crate::graph::introspection::schema_overview::collect_constraints_structured(graph);

    let mut id_of: BTreeMap<&str, u32> = BTreeMap::new();
    let mut nodes: Vec<Value> = Vec::with_capacity(labels.len());
    for (ordinal, label) in labels.iter().enumerate() {
        id_of.insert(label.as_str(), ordinal as u32);
        let names_on = |all: Vec<String>| Value::List(all.into_iter().map(Value::String).collect());
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
        let properties = crate::datatypes::PropMap::from_iter([
            ("name", Value::String(label.clone())),
            ("indexes", names_on(index_names)),
            ("constraints", names_on(constraint_names)),
        ]);
        nodes.push(Value::Node(Box::new(NodeValue {
            id: ordinal as u32,
            labels: vec![label.clone()],
            properties,
        })));
    }

    let stats = crate::graph::introspection::schema_overview::compute_connection_type_stats(graph);
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
                        properties: crate::datatypes::PropMap::new(),
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

/// `db.schema.nodeTypeProperties()` / `relTypeProperties()` — extracted from
/// the dispatcher to keep it a table of thin delegations.
fn schema_type_properties_rows(
    executor: &CypherExecutor<'_>,
    node_side: bool,
    apoc_shape: bool,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    // Neo4j's typed-schema pair — measured as the calls G.V()'s
    // data-model load makes (2026-08-15; it falls back to
    // apoc.meta.* when these are absent and then gives up). One row
    // per (type, property) with Neo4j's spelling: the type wrapped
    // as :`Name`, propertyTypes as a one-element list, and a
    // property-less type still emits one row with null propertyName
    // so the type itself is visible to the client.
    //
    // `apoc_shape` serves the same data under APOC's column set —
    // crucially including `sourceNodeLabels`/`targetNodeLabels` on the rel
    // side, which the db.schema.* contract does not carry. Clients build
    // their schema GRAPH (which labels connect via which types) exclusively
    // from those columns: G.V()'s schema view had 98 disconnected boxes and
    // its no-code path picker offered zero (A)-[R]->(B) paths until these
    // two names answered. In apoc shape the rel side emits one row per
    // observed (source, type, target) pairing.

    struct RowSpec<'a> {
        type_name: &'a str,
        labels: Option<&'a str>,
        endpoints: Option<(&'a str, &'a str)>,
        prop: Option<(&'a str, &'a str)>,
        count: usize,
    }

    let mut rows = Vec::new();
    let mut push_row = |spec: RowSpec<'_>| {
        let mut row = ResultRow::new();
        for item in yield_items {
            let alias = item.alias.as_deref().unwrap_or(&item.name);
            let value = match item.name.as_str() {
                "nodeType" | "relType" => Value::String(format!(":`{}`", spec.type_name)),
                "nodeLabels" => match spec.labels {
                    Some(l) => Value::List(vec![Value::String(l.to_string())]),
                    None => Value::Null,
                },
                "sourceNodeLabels" => match spec.endpoints {
                    Some((s, _)) => Value::List(vec![Value::String(s.to_string())]),
                    None => Value::List(Vec::new()),
                },
                "targetNodeLabels" => match spec.endpoints {
                    Some((_, t)) => Value::List(vec![Value::String(t.to_string())]),
                    None => Value::List(Vec::new()),
                },
                "totalObservations" => Value::Int64(spec.count as i64),
                "propertyObservations" => match spec.prop {
                    Some(_) => Value::Int64(spec.count as i64),
                    None => Value::Int64(0),
                },
                "propertyName" => match spec.prop {
                    Some((name, _)) => Value::String(name.to_string()),
                    None => Value::Null,
                },
                "propertyTypes" => match spec.prop {
                    Some((_, ty)) => {
                        Value::List(vec![Value::String(neo4j_type_name(ty).to_string())])
                    }
                    None => Value::Null,
                },
                // KGLite does not track per-property presence
                // enforcement here; false matches Neo4j's answer
                // for unconstrained properties.
                "mandatory" => Value::Boolean(false),
                _ => continue,
            };
            row.projected.insert(alias.to_string(), value);
        }
        rows.push(row);
    };

    if node_side {
        let schema = crate::graph::introspection::schema_overview::compute_schema(executor.graph);
        for (node_type, overview) in &schema.node_types {
            // A property whose every observed value is null has no type to
            // report. In Neo4j's model a null property is an absent one, so
            // such a row cannot exist there — and emitting our "Null" type
            // string breaks strict clients (G.V()'s schema builder rejects
            // the whole payload over one unknown type name; measured
            // 2026-08-15 on a graph with four all-null columns).
            let mut props: Vec<(&String, &String)> = overview
                .properties
                .iter()
                .filter(|(_, ty)| ty.as_str() != "Null")
                .collect();
            props.sort();
            if props.is_empty() {
                push_row(RowSpec {
                    type_name: node_type,
                    labels: Some(node_type),
                    endpoints: None,
                    prop: None,
                    count: overview.count,
                });
            }
            for (name, ty) in props {
                push_row(RowSpec {
                    type_name: node_type,
                    labels: Some(node_type),
                    endpoints: None,
                    prop: Some((name, ty)),
                    count: overview.count,
                });
            }
        }
    } else {
        let stats = crate::graph::introspection::schema_overview::compute_connection_type_stats(
            executor.graph,
        );
        for stat in &stats {
            let mut props: Vec<(&String, String)> = stat
                .property_names
                .iter()
                .map(|name| {
                    let ty = executor
                        .graph
                        .connection_type_metadata
                        .get(&stat.connection_type)
                        .and_then(|info| info.property_types.get(name))
                        .cloned()
                        .unwrap_or_else(|| "Any".to_string());
                    (name, ty)
                })
                // Same all-null rule as the node side.
                .filter(|(_, ty)| ty != "Null")
                .collect();
            props.sort();

            // apoc shape: one row set per observed (source, target) pairing;
            // db.schema shape: one per type (the contract has no endpoint
            // columns, so pairings would be indistinguishable duplicates).
            let pairings: Vec<Option<(&str, &str)>> = if apoc_shape {
                let mut pairs = Vec::new();
                for source in &stat.source_types {
                    for target in &stat.target_types {
                        pairs.push(Some((source.as_str(), target.as_str())));
                    }
                }
                if pairs.is_empty() {
                    pairs.push(None);
                }
                pairs
            } else {
                vec![None]
            };

            for endpoints in pairings {
                if props.is_empty() {
                    push_row(RowSpec {
                        type_name: &stat.connection_type,
                        labels: None,
                        endpoints,
                        prop: None,
                        count: stat.count,
                    });
                }
                for (name, ty) in &props {
                    push_row(RowSpec {
                        type_name: &stat.connection_type,
                        labels: None,
                        endpoints,
                        prop: Some((name, ty)),
                        count: stat.count,
                    });
                }
            }
        }
    }
    Ok(rows)
}

/// Map a KGLite type string to Neo4j's schema-procedure spelling, so clients
/// that switch on the type name (G.V(), Browser) recognize it. Unknown names
/// pass through verbatim — honest, and visibly ours.
fn neo4j_type_name(kglite_type: &str) -> &str {
    match kglite_type {
        "Int64" | "Int32" | "UniqueId" => "Long",
        "Float64" | "Float32" => "Double",
        "String" => "String",
        "Boolean" => "Boolean",
        "DateTime" => "DateTime",
        "Date" => "Date",
        "List" => "List",
        "Map" => "Map",
        "Point" => "Point",
        other => other,
    }
}
