//! Keyed row mutations for table-valued properties: `CALL table.upsert` /
//! `CALL table.delete`.
//!
//! Both are engine-side read-modify-writes of a `list<map>` property keyed
//! by one column, so concurrent application-level read→modify→write races
//! cannot lose updates: the merge happens under the statement, and the
//! rebuilt value flows through the normal property-SET path (write scope,
//! constraints, declared shapes, WAL, CDC — the whole gauntlet).
//!
//! ```cypher
//! CALL table.upsert({type: 'Order', id: 'order-1', property: 'line_items',
//!                    key: 'sku', row: {sku: 'a-1', qty: 8}})
//!   YIELD action, rows
//! CALL table.delete({type: 'Order', id: 'order-1', property: 'line_items',
//!                    key: 'sku', value: 'a-1'}) YIELD removed, rows
//! ```
//!
//! `upsert` replaces the first row whose `key` cell equals `row[key]`
//! (whole-row replace, not merge) or appends when none matches; `delete`
//! removes every matching row.

use std::collections::HashMap;

use petgraph::graph::NodeIndex;

use crate::datatypes::values::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::languages::cypher::ast::YieldItem;
use crate::graph::languages::cypher::result::MutationStats;
use crate::graph::languages::cypher::result::ResultRow;

use super::set_row::{apply_node_property_set, NodePropertySet, SetMemos};

pub(crate) fn execute_table_procedure(
    graph: &mut DirGraph,
    proc_name: &str,
    params: &HashMap<String, Value>,
    yield_items: &[YieldItem],
) -> Result<Vec<ResultRow>, String> {
    let node_type = require_str(params, "type", proc_name)?;
    let property = require_str(params, "property", proc_name)?;
    let key = require_str(params, "key", proc_name)?;
    let id = params
        .get("id")
        .cloned()
        .ok_or_else(|| format!("CALL {proc_name}: missing parameter 'id'"))?;

    graph.build_id_index(&node_type);
    let Some(node_idx) = graph.lookup_by_id(&node_type, &id) else {
        return Err(format!(
            "CALL {proc_name}: no {node_type} node with id {id}"
        ));
    };

    let mut rows = read_rows(graph, node_idx, &property)?;
    let (action, removed) = match proc_name {
        "table.upsert" => {
            let row = match params.get("row") {
                Some(Value::Map(map)) => Value::Map(map.clone()),
                Some(other) => {
                    return Err(format!(
                        "CALL table.upsert: 'row' must be a map, got {}",
                        other.type_name()
                    ));
                }
                None => return Err("CALL table.upsert: missing parameter 'row'".to_string()),
            };
            let Value::Map(ref row_map) = row else {
                unreachable!()
            };
            let Some(key_value) = row_map.get(&key) else {
                return Err(format!(
                    "CALL table.upsert: the row carries no '{key}' cell to key on"
                ));
            };
            let slot = rows
                .iter()
                .position(|r| matches!(r, Value::Map(m) if m.get(&key) == Some(key_value)));
            match slot {
                Some(i) => {
                    rows[i] = row;
                    ("updated", 0)
                }
                None => {
                    rows.push(row);
                    ("inserted", 0)
                }
            }
        }
        "table.delete" => {
            let value = params
                .get("value")
                .cloned()
                .ok_or_else(|| "CALL table.delete: missing parameter 'value'".to_string())?;
            let before = rows.len();
            rows.retain(|r| !matches!(r, Value::Map(m) if m.get(&key) == Some(&value)));
            ("deleted", before - rows.len())
        }
        other => unreachable!("non-table procedure routed here: {other}"),
    };

    let row_count = rows.len();
    let mut memos = SetMemos::default();
    let mut stats = MutationStats::default();
    let mut stamp: HashMap<NodeIndex, String> = HashMap::new();
    apply_node_property_set(
        graph,
        NodePropertySet {
            node_idx,
            property: &property,
            value: Value::List(rows),
        },
        &mut memos,
        &mut stats,
        &mut stamp,
    )?;

    let alias = |name: &str| {
        yield_items
            .iter()
            .find(|y| y.name == name)
            .map(|y| y.alias.clone().unwrap_or_else(|| name.to_string()))
    };
    let mut out_row = ResultRow::new();
    if let Some(a) = alias("action") {
        out_row
            .projected
            .insert(a, Value::String(action.to_string()));
    }
    if let Some(a) = alias("removed") {
        out_row.projected.insert(a, Value::Int64(removed as i64));
    }
    if let Some(a) = alias("rows") {
        out_row.projected.insert(a, Value::Int64(row_count as i64));
    }
    Ok(vec![out_row])
}

fn read_rows(graph: &DirGraph, node_idx: NodeIndex, property: &str) -> Result<Vec<Value>, String> {
    use crate::graph::storage::GraphRead;
    let _guard = graph.graph.begin_query();
    let key = crate::graph::schema::InternedKey::from_str(property);
    match graph.graph.get_node_property(node_idx, key) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::List(items)) => Ok(items),
        Some(other) => Err(format!(
            "table procedure target '{property}' holds {} — expected a list of maps",
            other.type_name()
        )),
    }
}

fn require_str(params: &HashMap<String, Value>, name: &str, proc: &str) -> Result<String, String> {
    match params.get(name) {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(other) => Err(format!(
            "CALL {proc}: '{name}' must be a string, got {}",
            other.type_name()
        )),
        None => Err(format!("CALL {proc}: missing parameter '{name}'")),
    }
}
