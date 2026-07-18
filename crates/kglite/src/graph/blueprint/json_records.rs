//! `from_records` — build a graph from an inline JSON records spec.
//!
//! A JSON-native sibling to [`crate::graph::blueprint::build`]: instead of a
//! blueprint pointing at CSV files on disk, the caller passes node and
//! connection records inline. Agent-authored graphs are JSON-native, so this
//! is the natural ingestion path for them.
//!
//! The spec shape (all `records` are arrays of flat JSON objects):
//!
//! ```json
//! {
//!   "nodes": [
//!     { "type": "Person", "id_field": "id", "title_field": "name",
//!       "conflict_handling": "update",
//!       "records": [ {"id": 1, "name": "Alice", "aliases": ["a", "b"]} ] }
//!   ],
//!   "connections": [
//!     { "type": "KNOWS", "source_type": "Person", "source_id_field": "from",
//!       "target_type": "Person", "target_id_field": "to",
//!       "records": [ {"from": 1, "to": 2, "since": 2020} ] }
//!   ],
//!   "on_missing_endpoint": "vivify"
//! }
//! ```
//!
//! Column types are **inferred** from the record values (across all rows, via
//! [`DataFrame::from_cypher_rows`]), so a JSON array becomes a native list
//! property, an integer an `Int64`, etc. All graph mutation reuses the
//! existing engine — [`maintain::add_nodes`] and [`maintain::add_connections`]
//! (the latter's Pass-A/B/C endpoint vivification fills in missing endpoints) —
//! so there is no duplicated mutation logic.

use crate::datatypes::values::{DataFrame, Value};
use crate::graph::mutation::maintain;
use crate::graph::DirGraph;
use serde_json::Value as Json;

/// Summary of a `from_records` build.
#[derive(Debug, Default, Clone)]
pub struct RecordsReport {
    pub nodes_added: usize,
    pub edges_added: usize,
    /// Connection records omitted by `on_missing_endpoint: "drop"` because
    /// at least one endpoint was null or absent from the graph.
    pub edges_dropped_missing_endpoint: usize,
    pub node_types: Vec<String>,
    pub connection_types: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MissingEndpointPolicy {
    Vivify,
    Drop,
    Error,
}

/// Build (or extend) `graph` from an inline JSON records spec. See the module
/// docs for the spec shape. Returns a per-build summary.
pub fn from_records(graph: &mut DirGraph, spec: &Json) -> Result<RecordsReport, String> {
    let obj = spec
        .as_object()
        .ok_or_else(|| "from_records: top-level JSON must be an object".to_string())?;

    let endpoint_policy = parse_endpoint_policy(obj.get("on_missing_endpoint"))?;
    if endpoint_policy == MissingEndpointPolicy::Error {
        // `error` is an all-or-nothing ingestion mode. Build against a
        // backend-aware transaction fork and publish it only after every node
        // and connection spec has succeeded.
        let mut working = graph.fork_transaction();
        let report = load_records(&mut working, obj, endpoint_policy)?;
        *graph = working;
        return Ok(report);
    }

    load_records(graph, obj, endpoint_policy)
}

fn load_records(
    graph: &mut DirGraph,
    obj: &serde_json::Map<String, Json>,
    endpoint_policy: MissingEndpointPolicy,
) -> Result<RecordsReport, String> {
    let mut report = RecordsReport::default();

    // ── Nodes ────────────────────────────────────────────────────────────
    if let Some(nodes) = obj.get("nodes") {
        let arr = nodes
            .as_array()
            .ok_or_else(|| "from_records: 'nodes' must be an array".to_string())?;
        for (i, node_spec) in arr.iter().enumerate() {
            load_node_spec(graph, node_spec, i, &mut report)?;
        }
    }

    // ── Connections ──────────────────────────────────────────────────────
    if let Some(conns) = obj.get("connections") {
        let arr = conns
            .as_array()
            .ok_or_else(|| "from_records: 'connections' must be an array".to_string())?;
        for (i, conn_spec) in arr.iter().enumerate() {
            load_connection_spec(graph, conn_spec, i, endpoint_policy, &mut report)?;
        }
    }

    Ok(report)
}

fn load_node_spec(
    graph: &mut DirGraph,
    spec: &Json,
    idx: usize,
    report: &mut RecordsReport,
) -> Result<(), String> {
    let ctx = || format!("from_records: nodes[{}]", idx);
    let node_type = required_str(spec, "type", &ctx)?;
    let id_field = required_str(spec, "id_field", &ctx)?;
    let title_field = optional_str(spec, "title_field");
    let conflict_handling = optional_str(spec, "conflict_handling");

    let records = records_array(spec, &ctx)?;
    if records.is_empty() {
        return Ok(()); // nothing to add for this type
    }

    // The id field always leads the column order so a record missing it
    // still produces a (null id) row that add_nodes' validity check catches.
    let (columns, rows) = records_to_columns_rows(records, &[&id_field], &ctx)?;
    let df = DataFrame::from_cypher_rows(columns, rows).map_err(|e| format!("{}: {}", ctx(), e))?;

    let rep = maintain::add_nodes(
        graph,
        df,
        node_type.clone(),
        id_field,
        title_field,
        conflict_handling,
    )
    .map_err(|e| format!("{}: {}", ctx(), e))?;

    report.nodes_added += rep.nodes_created + rep.nodes_updated;
    report.node_types.push(node_type);
    Ok(())
}

fn load_connection_spec(
    graph: &mut DirGraph,
    spec: &Json,
    idx: usize,
    endpoint_policy: MissingEndpointPolicy,
    report: &mut RecordsReport,
) -> Result<(), String> {
    let ctx = || format!("from_records: connections[{}]", idx);
    let connection_type = required_str(spec, "type", &ctx)?;
    let source_type = required_str(spec, "source_type", &ctx)?;
    let source_id_field = required_str(spec, "source_id_field", &ctx)?;
    let target_type = required_str(spec, "target_type", &ctx)?;
    let target_id_field = required_str(spec, "target_id_field", &ctx)?;

    let records = records_array(spec, &ctx)?;
    if records.is_empty() {
        return Ok(());
    }

    let (columns, rows) =
        records_to_columns_rows(records, &[&source_id_field, &target_id_field], &ctx)?;
    let df = DataFrame::from_cypher_rows(columns, rows).map_err(|e| format!("{}: {}", ctx(), e))?;

    match endpoint_policy {
        MissingEndpointPolicy::Vivify => {
            let rep = maintain::add_connections(
                graph,
                df,
                connection_type.clone(),
                source_type,
                source_id_field,
                target_type,
                target_id_field,
                None,
                None,
                None,
            )
            .map_err(|e| format!("{}: {}", ctx(), e))?;
            report.edges_added += rep.connections_created;
        }
        MissingEndpointPolicy::Drop | MissingEndpointPolicy::Error => {
            let edge_context = EdgeFrameContext {
                connection_type: &connection_type,
                source_type: &source_type,
                source_id_field: &source_id_field,
                target_type: &target_type,
                target_id_field: &target_id_field,
                connection_idx: idx,
                endpoint_policy,
            };
            let edge_specs = edge_specs_from_frame(graph, &df, &edge_context)?;
            let rep = maintain::add_edges_from_specs(graph, edge_specs)
                .map_err(|e| format!("{}: {}", ctx(), e))?;
            report.edges_added += rep.connections_created;
            report.edges_dropped_missing_endpoint += rep.skipped_missing_endpoint;
        }
    }
    report.connection_types.push(connection_type);
    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn parse_endpoint_policy(value: Option<&Json>) -> Result<MissingEndpointPolicy, String> {
    match value {
        None => Ok(MissingEndpointPolicy::Vivify),
        Some(Json::String(s)) if s == "vivify" => Ok(MissingEndpointPolicy::Vivify),
        Some(Json::String(s)) if s == "drop" => Ok(MissingEndpointPolicy::Drop),
        Some(Json::String(s)) if s == "error" => Ok(MissingEndpointPolicy::Error),
        Some(Json::String(s)) => Err(format!(
            "from_records: unknown on_missing_endpoint mode '{s}'; expected 'vivify', 'drop', or 'error'"
        )),
        Some(_) => Err(
            "from_records: 'on_missing_endpoint' must be 'vivify', 'drop', or 'error'"
                .to_string(),
        ),
    }
}

struct EdgeFrameContext<'a> {
    connection_type: &'a str,
    source_type: &'a str,
    source_id_field: &'a str,
    target_type: &'a str,
    target_id_field: &'a str,
    connection_idx: usize,
    endpoint_policy: MissingEndpointPolicy,
}

fn edge_specs_from_frame(
    graph: &DirGraph,
    frame: &DataFrame,
    context: &EdgeFrameContext<'_>,
) -> Result<Vec<maintain::EdgeSpec>, String> {
    let property_columns: Vec<String> = frame
        .get_column_names()
        .into_iter()
        .filter(|name| name != context.source_id_field && name != context.target_id_field)
        .collect();
    let mut specs = Vec::with_capacity(frame.row_count());

    for row_idx in 0..frame.row_count() {
        let source_id = frame
            .get_value(row_idx, context.source_id_field)
            .unwrap_or(Value::Null);
        let target_id = frame
            .get_value(row_idx, context.target_id_field)
            .unwrap_or(Value::Null);

        if context.endpoint_policy == MissingEndpointPolicy::Error {
            validate_endpoint(
                graph,
                context.source_type,
                context.source_id_field,
                &source_id,
                context.connection_idx,
                row_idx,
                "source",
            )?;
            validate_endpoint(
                graph,
                context.target_type,
                context.target_id_field,
                &target_id,
                context.connection_idx,
                row_idx,
                "target",
            )?;
        }

        let properties = property_columns
            .iter()
            .filter_map(|name| {
                frame
                    .get_value(row_idx, name)
                    .filter(|value| !matches!(value, Value::Null))
                    .map(|value| (name.clone(), value))
            })
            .collect();
        specs.push(maintain::EdgeSpec {
            source_type: context.source_type.to_string(),
            source_id,
            target_type: context.target_type.to_string(),
            target_id,
            edge_type: context.connection_type.to_string(),
            properties,
        });
    }

    Ok(specs)
}

fn validate_endpoint(
    graph: &DirGraph,
    node_type: &str,
    id_field: &str,
    id: &Value,
    connection_idx: usize,
    row_idx: usize,
    endpoint_name: &str,
) -> Result<(), String> {
    let ctx = format!("from_records: connections[{connection_idx}].records[{row_idx}]");
    if matches!(id, Value::Null) {
        return Err(format!(
            "{ctx}: {endpoint_name} endpoint id field '{id_field}' is null"
        ));
    }
    if graph.lookup_by_id_readonly(node_type, id).is_none() {
        return Err(format!(
            "{ctx}: {endpoint_name} endpoint {node_type}({id}) does not exist"
        ));
    }
    Ok(())
}

fn required_str(spec: &Json, key: &str, ctx: &impl Fn() -> String) -> Result<String, String> {
    spec.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| format!("{}: missing required string field '{}'", ctx(), key))
}

fn optional_str(spec: &Json, key: &str) -> Option<String> {
    spec.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn records_array<'a>(spec: &'a Json, ctx: &impl Fn() -> String) -> Result<&'a Vec<Json>, String> {
    spec.get("records")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("{}: 'records' must be an array", ctx()))
}

/// Flatten an array of JSON objects into `(column_names, rows)`. Column order
/// is `required` fields first (in the given order), then every other key in
/// first-seen order. A record missing a column yields `Value::Null` there.
fn records_to_columns_rows(
    records: &[Json],
    required: &[&str],
    ctx: &impl Fn() -> String,
) -> Result<(Vec<String>, Vec<Vec<Value>>), String> {
    let mut columns: Vec<String> = required.iter().map(|s| s.to_string()).collect();
    let mut seen: std::collections::HashSet<String> = columns.iter().cloned().collect();
    for rec in records {
        let obj = rec
            .as_object()
            .ok_or_else(|| format!("{}: every record must be a JSON object", ctx()))?;
        for key in obj.keys() {
            if seen.insert(key.clone()) {
                columns.push(key.clone());
            }
        }
    }

    let rows = records
        .iter()
        .map(|rec| {
            let obj = rec.as_object().expect("validated above");
            columns
                .iter()
                .map(|col| obj.get(col).map(json_to_value).unwrap_or(Value::Null))
                .collect()
        })
        .collect();

    Ok((columns, rows))
}

/// Recursive JSON → [`Value`]. Arrays become native `Value::List`, objects
/// become `Value::Map`; scalars map to their natural typed variant.
fn json_to_value(j: &Json) -> Value {
    match j {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Boolean(*b),
        Json::Number(n) => n
            .as_i64()
            .map(Value::Int64)
            .or_else(|| n.as_f64().map(Value::Float64))
            .unwrap_or(Value::Null),
        Json::String(s) => Value::String(s.clone()),
        Json::Array(items) => Value::List(items.iter().map(json_to_value).collect()),
        Json::Object(map) => Value::Map(
            map.iter()
                .map(|(k, v)| (k.clone(), json_to_value(v)))
                .collect(),
        ),
    }
}

#[cfg(test)]
#[path = "json_records_tests.rs"]
mod tests;
