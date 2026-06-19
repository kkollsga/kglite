// src/datatypes/py_out.rs
use super::values::Value;
use crate::graph::schema::NodeInfo;
use kglite_core::api::fluent::PropertyStats;
use kglite_core::api::fluent::StatResult;
use kglite_core::api::fluent::{LevelConnections, LevelNodes, LevelValues, UniqueValues};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::IntoPyObjectExt;
use std::collections::HashMap;

pub fn nodeinfo_to_pydict(py: Python, node: &NodeInfo) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("type", &node.node_type)?;
    dict.set_item("title", value_to_py(py, &node.title)?)?;
    dict.set_item("id", value_to_py(py, &node.id)?)?;

    // Always merge properties directly into the main dictionary
    for (k, v) in &node.properties {
        dict.set_item(k, value_to_py(py, v)?)?;
    }

    Ok(dict.into())
}

pub fn value_to_py(py: Python, value: &Value) -> PyResult<Py<PyAny>> {
    match value {
        Value::String(s) => s.clone().into_py_any(py),
        Value::Float64(f) => f.into_py_any(py),
        Value::Int64(i) => i.into_py_any(py),
        Value::Boolean(b) => b.into_py_any(py),
        Value::UniqueId(u) => u.into_py_any(py),
        Value::DateTime(d) => d.format("%Y-%m-%d").to_string().into_py_any(py),
        Value::Point { lat, lon } => {
            let dict = PyDict::new(py);
            dict.set_item("latitude", lat)?;
            dict.set_item("longitude", lon)?;
            Ok(dict.into_any().unbind())
        }
        Value::Null => Ok(py.None()),
        // NodeRef should be resolved before reaching Python; fallback to index
        Value::NodeRef(idx) => idx.into_py_any(py),
        // Value::Duration → Python dict {months, days, seconds}.
        // 0.9.0 Cluster 2.
        Value::Duration {
            months,
            days,
            seconds,
        } => {
            let dict = PyDict::new(py);
            dict.set_item("months", months)?;
            dict.set_item("days", days)?;
            dict.set_item("seconds", seconds)?;
            Ok(dict.into_any().unbind())
        }
        // Phase A.1 / C4 — native conversion of the collection /
        // graph-entity variants. Replaces the prior JSON-string-
        // inference hack (PreProcessedValue in py_convert.rs).
        Value::List(items) => {
            let py_items: PyResult<Vec<Py<PyAny>>> =
                items.iter().map(|v| value_to_py(py, v)).collect();
            Ok(PyList::new(py, py_items?)?.into_any().unbind())
        }
        Value::Map(entries) => {
            let dict = PyDict::new(py);
            for (k, v) in entries {
                dict.set_item(k, value_to_py(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
        Value::Node(node_val) => {
            let dict = PyDict::new(py);
            dict.set_item("id", node_val.id)?;
            // labels mirror Neo4j/Bolt shape: list of strings
            let py_labels = PyList::new(py, &node_val.labels)?;
            dict.set_item("labels", py_labels)?;
            // properties as a nested dict — recursive value_to_py
            // means nested Nodes/Lists/Maps round-trip cleanly.
            let props_dict = PyDict::new(py);
            for (k, v) in &node_val.properties {
                props_dict.set_item(k, value_to_py(py, v)?)?;
            }
            dict.set_item("properties", props_dict)?;
            Ok(dict.into_any().unbind())
        }
        Value::Relationship(rel_val) => {
            let dict = PyDict::new(py);
            dict.set_item("id", rel_val.id)?;
            dict.set_item("start", rel_val.start_id)?;
            dict.set_item("end", rel_val.end_id)?;
            dict.set_item("type", &rel_val.rel_type)?;
            let props_dict = PyDict::new(py);
            for (k, v) in &rel_val.properties {
                props_dict.set_item(k, value_to_py(py, v)?)?;
            }
            dict.set_item("properties", props_dict)?;
            Ok(dict.into_any().unbind())
        }
        Value::Path(path_val) => {
            let dict = PyDict::new(py);
            // Recurse into the Node/Rel conversions above so the
            // path nests cleanly.
            let py_nodes: PyResult<Vec<Py<PyAny>>> = path_val
                .nodes
                .iter()
                .map(|n| value_to_py(py, &Value::Node(Box::new(n.clone()))))
                .collect();
            let py_rels: PyResult<Vec<Py<PyAny>>> = path_val
                .rels
                .iter()
                .map(|r| value_to_py(py, &Value::Relationship(Box::new(r.clone()))))
                .collect();
            dict.set_item("nodes", PyList::new(py, py_nodes?)?)?;
            dict.set_item("relationships", PyList::new(py, py_rels?)?)?;
            Ok(dict.into_any().unbind())
        }
    }
}

/// Convert a HashMap<String, Value> to a Python dict
pub fn hashmap_to_pydict<'py>(
    py: Python<'py>,
    map: &HashMap<String, Value>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (k, v) in map {
        dict.set_item(k, value_to_py(py, v)?)?;
    }
    Ok(dict)
}

pub fn convert_stats_for_python(stats: Vec<PropertyStats>) -> PyResult<Py<PyAny>> {
    Python::attach(|py| {
        let dict = PyDict::new(py);

        let parent_idx = PyList::empty(py);
        let parent_type = PyList::empty(py);
        let parent_title = PyList::empty(py);
        let parent_id = PyList::empty(py);
        let property_name = PyList::empty(py);
        let value_type = PyList::empty(py);
        let children = PyList::empty(py);
        let count = PyList::empty(py);
        let valid_count = PyList::empty(py);
        let sum_val = PyList::empty(py);
        let avg = PyList::empty(py);
        let min_val = PyList::empty(py);
        let max_val = PyList::empty(py);

        for stat in stats {
            parent_idx.append(
                stat.parent_idx
                    .map(|idx| idx.index().into_pyobject(py).unwrap().into_any().unbind())
                    .unwrap_or_else(|| py.None()),
            )?;
            parent_type.append(stat.parent_type.unwrap_or_default())?;
            parent_title.append(
                stat.parent_title
                    .map_or_else(|| py.None(), |v| value_to_py(py, &v).unwrap()),
            )?;
            parent_id.append(
                stat.parent_id
                    .map_or_else(|| py.None(), |v| value_to_py(py, &v).unwrap()),
            )?;
            property_name.append(stat.property_name)?;
            value_type.append(stat.value_type)?;
            children.append(stat.children)?;
            count.append(stat.count)?;
            valid_count.append(stat.valid_count)?;

            if stat.is_numeric {
                sum_val.append(
                    stat.sum
                        .map(|v| v.into_pyobject(py).unwrap().into_any().unbind())
                        .unwrap_or_else(|| py.None()),
                )?;
                avg.append(
                    stat.avg
                        .map(|v| v.into_pyobject(py).unwrap().into_any().unbind())
                        .unwrap_or_else(|| py.None()),
                )?;
                min_val.append(
                    stat.min
                        .map(|v| v.into_pyobject(py).unwrap().into_any().unbind())
                        .unwrap_or_else(|| py.None()),
                )?;
                max_val.append(
                    stat.max
                        .map(|v| v.into_pyobject(py).unwrap().into_any().unbind())
                        .unwrap_or_else(|| py.None()),
                )?;
            } else {
                sum_val.append(py.None())?;
                avg.append(py.None())?;
                min_val.append(py.None())?;
                max_val.append(py.None())?;
            }
        }

        dict.set_item("parent_idx", parent_idx)?;
        dict.set_item("parent_type", parent_type)?;
        dict.set_item("parent_title", parent_title)?;
        dict.set_item("parent_id", parent_id)?;
        dict.set_item("property", property_name)?;
        dict.set_item("value_type", value_type)?;
        dict.set_item("children_count", children)?;
        dict.set_item("property_count", count)?;
        dict.set_item("valid_count", valid_count)?;
        dict.set_item("sum", sum_val)?;
        dict.set_item("avg", avg)?;
        dict.set_item("min", min_val)?;
        dict.set_item("max", max_val)?;

        Ok(dict.into())
    })
}

pub fn level_nodes_to_pydict(
    py: Python,
    level_nodes: &[LevelNodes],
    parent_key: Option<&str>,
    parent_info: Option<bool>,
    flatten_single_parent: Option<bool>,
) -> PyResult<Py<PyAny>> {
    // Default to true if not specified
    let should_flatten = flatten_single_parent.unwrap_or(true);

    // If there's only one parent and flatten_single_parent is true, return just the nodes
    if should_flatten && level_nodes.len() == 1 {
        let group = &level_nodes[0];

        if parent_info.unwrap_or(false) && group.parent_idx.is_some() {
            // When parent info is requested, still return a dict but with a simpler structure
            let parent_dict = PyDict::new(py);

            if let Some(ref type_str) = group.parent_type {
                parent_dict.set_item("type", type_str)?;
            }
            parent_dict.set_item("title", &group.parent_title)?;
            if let Some(ref id) = group.parent_id {
                parent_dict.set_item("id", value_to_py(py, id)?)?;
            }

            let nodes: Vec<Py<PyAny>> = group
                .nodes
                .iter()
                .map(|node| nodeinfo_to_pydict(py, node))
                .collect::<PyResult<_>>()?;
            parent_dict.set_item("nodes", nodes)?;

            return Ok(parent_dict.into());
        } else {
            // Just return the list of nodes
            let nodes: Vec<Py<PyAny>> = group
                .nodes
                .iter()
                .map(|node| nodeinfo_to_pydict(py, node))
                .collect::<PyResult<_>>()?;
            return Ok(PyList::new(py, nodes)?.into());
        }
    }

    // Original behavior for multiple parents
    let result = PyDict::new(py);
    let mut seen_keys = std::collections::HashMap::new();

    for group in level_nodes {
        let base_key = match parent_key {
            Some("idx") => {
                if let Some(idx) = group.parent_idx {
                    format!("{}", idx.index())
                } else {
                    String::from("no_idx")
                }
            }
            Some("id") => {
                if let Some(ref id) = group.parent_id {
                    match id {
                        Value::String(s) => s.clone(),
                        Value::Int64(i) => i.to_string(),
                        Value::Float64(f) => f.to_string(),
                        Value::UniqueId(u) => u.to_string(),
                        _ => format!("{:?}", id),
                    }
                } else {
                    String::from("no_id")
                }
            }
            _ => {
                if !group.parent_title.is_empty() {
                    group.parent_title.clone()
                } else {
                    String::from("no_title")
                }
            }
        };

        let key = {
            let count = seen_keys.entry(base_key.clone()).or_insert(0);
            *count += 1;

            if *count > 1 {
                format!("{}_{}", base_key, count)
            } else {
                base_key
            }
        };

        let value: Py<PyAny> = if parent_info.unwrap_or(false) && group.parent_idx.is_some() {
            let parent_dict = PyDict::new(py);

            if let Some(ref type_str) = group.parent_type {
                parent_dict.set_item("type", type_str)?;
            }
            parent_dict.set_item("title", &group.parent_title)?;
            if let Some(ref id) = group.parent_id {
                parent_dict.set_item("id", value_to_py(py, id)?)?;
            }

            let nodes: Vec<Py<PyAny>> = group
                .nodes
                .iter()
                .map(|node| nodeinfo_to_pydict(py, node))
                .collect::<PyResult<_>>()?;
            parent_dict.set_item("children", nodes)?;

            parent_dict.into()
        } else {
            let nodes: Vec<Py<PyAny>> = group
                .nodes
                .iter()
                .map(|node| nodeinfo_to_pydict(py, node))
                .collect::<PyResult<_>>()?;
            PyList::new(py, nodes)?.into()
        };

        result.set_item(key, value)?;
    }

    Ok(result.into())
}

pub fn level_values_to_pydict(
    py: Python,
    level_values: &[LevelValues],
    flatten_single_parent: Option<bool>,
) -> PyResult<Py<PyAny>> {
    let should_flatten = flatten_single_parent.unwrap_or(true);

    // If single parent and flatten requested, return flat list of tuples
    if should_flatten && level_values.len() == 1 {
        let group = &level_values[0];
        let values: Vec<Py<PyAny>> = group
            .values
            .iter()
            .map(|vec_values| {
                let tuple_values: Vec<Py<PyAny>> = vec_values
                    .iter()
                    .map(|v| value_to_py(py, v))
                    .collect::<PyResult<_>>()?;
                Ok(PyTuple::new(py, &tuple_values)?.into())
            })
            .collect::<PyResult<_>>()?;
        return Ok(PyList::new(py, values)?.into());
    }

    let result = PyDict::new(py);

    for group in level_values {
        let values: Vec<Py<PyAny>> = group
            .values
            .iter()
            .map(|vec_values| {
                let tuple_values: Vec<Py<PyAny>> = vec_values
                    .iter()
                    .map(|v| value_to_py(py, v))
                    .collect::<PyResult<_>>()?;
                Ok(PyTuple::new(py, &tuple_values)?.into())
            })
            .collect::<PyResult<_>>()?;

        result.set_item(&group.parent_title, values)?;
    }

    Ok(result.into())
}

pub fn level_single_values_to_pydict(
    py: Python,
    level_values: &[LevelValues],
    flatten_single_parent: Option<bool>,
) -> PyResult<Py<PyAny>> {
    let should_flatten = flatten_single_parent.unwrap_or(true);

    // If single parent and flatten requested, return flat list
    if should_flatten && level_values.len() == 1 {
        let group = &level_values[0];
        let values: Vec<Py<PyAny>> = group
            .values
            .iter()
            .map(|vec_values| value_to_py(py, &vec_values[0]))
            .collect::<PyResult<_>>()?;
        return Ok(PyList::new(py, values)?.into());
    }

    let result = PyDict::new(py);

    for group in level_values {
        let values: Vec<Py<PyAny>> = group
            .values
            .iter()
            .map(|vec_values| value_to_py(py, &vec_values[0]))
            .collect::<PyResult<_>>()?;

        result.set_item(&group.parent_title, values)?;
    }

    Ok(result.into())
}

pub fn level_connections_to_pydict(
    py: Python,
    connections: &[LevelConnections],
    parent_info: Option<bool>,
    flatten_single_parent: Option<bool>,
) -> PyResult<Py<PyAny>> {
    // Default to true if not specified
    let should_flatten = flatten_single_parent.unwrap_or(true);

    // If there's only one parent and flatten_single_parent is true, return just the connections
    if should_flatten && connections.len() == 1 {
        let level = &connections[0];
        let connections_dict = PyDict::new(py);

        // Add parent info if requested
        if parent_info.unwrap_or(false) {
            if let Some(parent_id) = &level.parent_id {
                connections_dict.set_item("parent_id", value_to_py(py, parent_id)?)?;
            }
            if let Some(parent_type) = &level.parent_type {
                connections_dict.set_item("parent_type", parent_type)?;
            }
            if let Some(parent_idx) = &level.parent_idx {
                connections_dict.set_item("parent_idx", parent_idx.index())?;
            }
            connections_dict.set_item("parent_title", &level.parent_title)?;
        }

        // Add all connections directly to the main dictionary
        for conn in &level.connections {
            let node_dict = PyDict::new(py);
            node_dict.set_item("node_id", value_to_py(py, &conn.node_id)?)?;
            node_dict.set_item("type", &conn.node_type)?;

            let incoming_dict = PyDict::new(py);
            for (conn_type, id, title, conn_props, node_props) in &conn.incoming {
                if !incoming_dict.contains(conn_type)? {
                    incoming_dict.set_item(conn_type, PyDict::new(py))?;
                }

                let conn_type_item = incoming_dict.get_item(conn_type)?;
                let conn_type_any = conn_type_item.unwrap();
                let conn_type_dict = conn_type_any.cast::<PyDict>()?;

                let node_info = PyDict::new(py);
                node_info.set_item("node_id", value_to_py(py, id)?)?;
                node_info.set_item("connection_properties", hashmap_to_pydict(py, conn_props)?)?;
                if let Some(props) = node_props {
                    node_info.set_item("node_properties", hashmap_to_pydict(py, props)?)?;
                }

                match title {
                    Value::String(t) => conn_type_dict.set_item(t, node_info)?,
                    _ => conn_type_dict.set_item("Unknown", node_info)?,
                }
            }
            node_dict.set_item("incoming", incoming_dict)?;

            let outgoing_dict = PyDict::new(py);
            for (conn_type, id, title, conn_props, node_props) in &conn.outgoing {
                if !outgoing_dict.contains(conn_type)? {
                    outgoing_dict.set_item(conn_type, PyDict::new(py))?;
                }

                let conn_type_item = outgoing_dict.get_item(conn_type)?;
                let conn_type_any = conn_type_item.unwrap();
                let conn_type_dict = conn_type_any.cast::<PyDict>()?;

                let node_info = PyDict::new(py);
                node_info.set_item("node_id", value_to_py(py, id)?)?;
                node_info.set_item("connection_properties", hashmap_to_pydict(py, conn_props)?)?;
                if let Some(props) = node_props {
                    node_info.set_item("node_properties", hashmap_to_pydict(py, props)?)?;
                }

                match title {
                    Value::String(t) => conn_type_dict.set_item(t, node_info)?,
                    _ => conn_type_dict.set_item("Unknown", node_info)?,
                }
            }
            node_dict.set_item("outgoing", outgoing_dict)?;

            connections_dict.set_item(&conn.node_title, node_dict)?;
        }

        return Ok(connections_dict.into());
    }

    // Original behavior for multiple parents
    let result = PyDict::new(py);

    for level in connections {
        let group_dict = PyDict::new(py);

        if parent_info.unwrap_or(false) {
            if let Some(parent_id) = &level.parent_id {
                group_dict.set_item("parent_id", value_to_py(py, parent_id)?)?;
            }
            if let Some(parent_type) = &level.parent_type {
                group_dict.set_item("parent_type", parent_type)?;
            }
            if let Some(parent_idx) = &level.parent_idx {
                group_dict.set_item("parent_idx", parent_idx.index())?;
            }
        }

        let connections_dict = PyDict::new(py);
        for conn in &level.connections {
            let node_dict = PyDict::new(py);
            node_dict.set_item("node_id", value_to_py(py, &conn.node_id)?)?;
            node_dict.set_item("type", &conn.node_type)?;

            let incoming_dict = PyDict::new(py);
            for (conn_type, id, title, conn_props, node_props) in &conn.incoming {
                if !incoming_dict.contains(conn_type)? {
                    incoming_dict.set_item(conn_type, PyDict::new(py))?;
                }

                let conn_type_item = incoming_dict.get_item(conn_type)?;
                let conn_type_any = conn_type_item.unwrap();
                let conn_type_dict = conn_type_any.cast::<PyDict>()?;

                let node_info = PyDict::new(py);
                node_info.set_item("node_id", value_to_py(py, id)?)?;
                node_info.set_item("connection_properties", hashmap_to_pydict(py, conn_props)?)?;
                if let Some(props) = node_props {
                    node_info.set_item("node_properties", hashmap_to_pydict(py, props)?)?;
                }

                match title {
                    Value::String(t) => conn_type_dict.set_item(t, node_info)?,
                    _ => conn_type_dict.set_item("Unknown", node_info)?,
                }
            }
            node_dict.set_item("incoming", incoming_dict)?;

            let outgoing_dict = PyDict::new(py);
            for (conn_type, id, title, conn_props, node_props) in &conn.outgoing {
                if !outgoing_dict.contains(conn_type)? {
                    outgoing_dict.set_item(conn_type, PyDict::new(py))?;
                }

                let conn_type_item = outgoing_dict.get_item(conn_type)?;
                let conn_type_any = conn_type_item.unwrap();
                let conn_type_dict = conn_type_any.cast::<PyDict>()?;

                let node_info = PyDict::new(py);
                node_info.set_item("node_id", value_to_py(py, id)?)?;
                node_info.set_item("connection_properties", hashmap_to_pydict(py, conn_props)?)?;
                if let Some(props) = node_props {
                    node_info.set_item("node_properties", hashmap_to_pydict(py, props)?)?;
                }

                match title {
                    Value::String(t) => conn_type_dict.set_item(t, node_info)?,
                    _ => conn_type_dict.set_item("Unknown", node_info)?,
                }
            }
            node_dict.set_item("outgoing", outgoing_dict)?;

            connections_dict.set_item(&conn.node_title, node_dict)?;
        }
        group_dict.set_item("connections", connections_dict)?;

        result.set_item(&level.parent_title, group_dict)?;
    }

    Ok(result.into())
}

pub fn level_unique_values_to_pydict(py: Python, values: &[UniqueValues]) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);
    for unique_values in values {
        let py_values: PyResult<Vec<Py<PyAny>>> = unique_values
            .values
            .iter()
            .map(|v| value_to_py(py, v))
            .collect();
        result.set_item(&unique_values.parent_title, PyList::new(py, py_values?)?)?;
    }
    Ok(result.into())
}

pub fn convert_computation_results_for_python(results: Vec<StatResult>) -> PyResult<Py<PyAny>> {
    Python::attach(|py| {
        let dict = PyDict::new(py);

        // Convert and insert each result within the GIL scope
        for (i, result) in results.iter().enumerate() {
            // Get a key from parent_title or generate one
            let key = match &result.parent_title {
                Some(title) => title.clone(),
                None => {
                    if let Some(idx) = result.parent_idx {
                        format!("node_{}", idx.index())
                    } else {
                        format!("result_{}", i)
                    }
                }
            };

            // Insert the value or null for errors
            if result.error_msg.is_some() {
                // Add null for error cases
                dict.set_item(&key, py.None())?;
            } else {
                // Process successful results
                match &result.value {
                    Value::Int64(i) => {
                        dict.set_item(&key, i)?;
                    }
                    Value::Float64(f) => {
                        dict.set_item(&key, f)?;
                    }
                    Value::UniqueId(u) => {
                        dict.set_item(&key, u)?;
                    }
                    Value::Null => {
                        // Add explicit null value
                        dict.set_item(&key, py.None())?;
                    }
                    _ => {
                        // Add null for unsupported types
                        dict.set_item(&key, py.None())?;
                    }
                }
            }
        }

        // Return empty dict if no results (not an error - traversal may have found nothing)
        Ok(dict.into())
    })
}

pub fn string_pairs_to_pydict(py: Python, pairs: &[(String, String)]) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);

    for (key, value) in pairs {
        result.set_item(key, value)?;
    }

    Ok(result.into())
}

/// Convert pattern matching results to a Python list of dictionaries
pub fn pattern_matches_to_pylist(
    py: Python,
    matches: &[kglite_core::api::fluent::PatternMatch],
    interner: &crate::graph::schema::StringInterner,
) -> PyResult<Py<PyAny>> {
    use kglite_core::api::fluent::MatchBinding;

    let result = PyList::empty(py);

    for pattern_match in matches {
        let match_dict = PyDict::new(py);

        for (var_name, binding) in &pattern_match.bindings {
            let binding_dict = PyDict::new(py);

            match binding {
                MatchBinding::Node {
                    node_type,
                    title,
                    id,
                    properties,
                    ..
                } => {
                    binding_dict.set_item("type", node_type)?;
                    binding_dict.set_item("title", title)?;
                    binding_dict.set_item("id", value_to_py(py, id)?)?;

                    // Add properties
                    let props_dict = PyDict::new(py);
                    for (key, value) in properties {
                        props_dict.set_item(key, value_to_py(py, value)?)?;
                    }
                    binding_dict.set_item("properties", props_dict)?;
                }
                MatchBinding::NodeRef(index) => {
                    binding_dict.set_item("index", index.index())?;
                }
                MatchBinding::Edge {
                    source,
                    target,
                    edge_index,
                    connection_type,
                    properties,
                } => {
                    binding_dict.set_item("source_idx", source.index())?;
                    binding_dict.set_item("target_idx", target.index())?;
                    binding_dict.set_item("edge_index", edge_index.index())?;
                    binding_dict.set_item("connection_type", interner.resolve(*connection_type))?;
                    let props_dict = PyDict::new(py);
                    for (key, value) in properties {
                        props_dict.set_item(key, value_to_py(py, value)?)?;
                    }
                    binding_dict.set_item("properties", props_dict)?;
                }
                MatchBinding::VariableLengthPath {
                    source,
                    target,
                    hops,
                    path,
                } => {
                    binding_dict.set_item("source_idx", source.index())?;
                    binding_dict.set_item("target_idx", target.index())?;
                    binding_dict.set_item("hops", *hops)?;

                    // Add path as list of (node_idx, connection_type) tuples
                    let path_list = PyList::empty(py);
                    for (node_idx, conn_type) in path {
                        let step_dict = PyDict::new(py);
                        step_dict.set_item("node_idx", node_idx.index())?;
                        step_dict.set_item("connection_type", interner.resolve(*conn_type))?;
                        path_list.append(step_dict)?;
                    }
                    binding_dict.set_item("path", path_list)?;
                }
            }

            match_dict.set_item(var_name, binding_dict)?;
        }

        result.append(match_dict)?;
    }

    Ok(result.into())
}
