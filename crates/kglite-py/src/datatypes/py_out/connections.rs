use super::keys::presentation_keys;
use super::{hashmap_to_pydict, value_to_py, Value};
use kglite_core::api::fluent::LevelConnections;
use kglite_core::api::storage::GraphBackend;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::{BTreeMap, HashMap};

type Properties = HashMap<String, Value>;
type Connection = (String, Value, Value, Properties, Option<Properties>);

fn canonical_properties(properties: &Properties) -> Vec<(&String, &Value)> {
    let mut entries: Vec<_> = properties.iter().collect();
    entries.sort_by_key(|(name, _)| *name);
    entries
}

fn connection_node_to_pydict<'py>(
    py: Python<'py>,
    graph: &GraphBackend,
    id: &Value,
    connection_properties: &Properties,
    node_properties: Option<&Properties>,
) -> PyResult<Bound<'py, PyDict>> {
    let node_info = PyDict::new(py);
    node_info.set_item("node_id", value_to_py(py, id)?)?;
    node_info.set_item(
        "connection_properties",
        hashmap_to_pydict(py, graph, connection_properties)?,
    )?;
    if let Some(properties) = node_properties {
        node_info.set_item("node_properties", hashmap_to_pydict(py, graph, properties)?)?;
    }
    Ok(node_info)
}

fn direction_to_pydict<'py>(
    py: Python<'py>,
    graph: &GraphBackend,
    connections: &[Connection],
) -> PyResult<Bound<'py, PyDict>> {
    let mut by_type: BTreeMap<&str, Vec<&Connection>> = BTreeMap::new();
    for connection in connections {
        by_type.entry(&connection.0).or_default().push(connection);
    }
    let result = PyDict::new(py);
    for (connection_type, edges) in by_type {
        let keys = presentation_keys(
            edges
                .iter()
                .map(|(_, id, title, properties, node_properties)| {
                    let label = match title {
                        Value::String(title) => title.clone(),
                        _ => "Unknown".to_owned(),
                    };
                    (
                        label,
                        (
                            id,
                            canonical_properties(properties),
                            node_properties.as_ref().map(canonical_properties),
                        ),
                    )
                })
                .collect(),
            &[],
        );
        let endpoints = PyDict::new(py);
        for ((_, id, _, properties, node_properties), key) in edges.into_iter().zip(keys) {
            endpoints.set_item(
                key,
                connection_node_to_pydict(py, graph, id, properties, node_properties.as_ref())?,
            )?;
        }
        result.set_item(connection_type, endpoints)?;
    }
    Ok(result)
}

fn parent_metadata<'py>(
    py: Python<'py>,
    level: &LevelConnections,
    include_title: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    if let Some(id) = &level.parent_id {
        result.set_item("parent_id", value_to_py(py, id)?)?;
    }
    if let Some(kind) = &level.parent_type {
        result.set_item("parent_type", kind)?;
    }
    if let Some(index) = level.parent_idx {
        result.set_item("parent_idx", index.index())?;
    }
    if include_title {
        result.set_item("parent_title", &level.parent_title)?;
    }
    Ok(result)
}

fn nodes_to_pydict<'py>(
    py: Python<'py>,
    graph: &GraphBackend,
    level: &LevelConnections,
    result: &Bound<'py, PyDict>,
) -> PyResult<()> {
    // A flattened result shares its dict with parent metadata. Reserve those
    // emitted keys as well as genuine node titles before assigning suffixes.
    let metadata: Vec<String> = result
        .keys()
        .iter()
        .map(|key| key.extract())
        .collect::<PyResult<_>>()?;
    let reserved: Vec<&str> = metadata.iter().map(String::as_str).collect();
    let keys = presentation_keys(
        level
            .connections
            .iter()
            .map(|node| (node.node_title.clone(), (&node.node_type, &node.node_id)))
            .collect(),
        &reserved,
    );
    for (node, key) in level.connections.iter().zip(keys) {
        let row = PyDict::new(py);
        row.set_item("node_id", value_to_py(py, &node.node_id)?)?;
        row.set_item("type", &node.node_type)?;
        row.set_item("incoming", direction_to_pydict(py, graph, &node.incoming)?)?;
        row.set_item("outgoing", direction_to_pydict(py, graph, &node.outgoing)?)?;
        result.set_item(key, row)?;
    }
    Ok(())
}

pub fn level_connections_to_pydict(
    py: Python,
    graph: &GraphBackend,
    connections: &[LevelConnections],
    parent_info: Option<bool>,
    flatten_single_parent: Option<bool>,
) -> PyResult<Py<PyAny>> {
    if flatten_single_parent.unwrap_or(true) && connections.len() == 1 {
        let level = &connections[0];
        let result = if parent_info.unwrap_or(false) {
            parent_metadata(py, level, true)?
        } else {
            PyDict::new(py)
        };
        nodes_to_pydict(py, graph, level, &result)?;
        return Ok(result.into());
    }
    let keys = presentation_keys(
        connections
            .iter()
            .map(|level| {
                (
                    level.parent_title.clone(),
                    level.parent_idx.map(|index| index.index()),
                )
            })
            .collect(),
        &[],
    );
    let result = PyDict::new(py);
    for (level, key) in connections.iter().zip(keys) {
        let group = if parent_info.unwrap_or(false) {
            parent_metadata(py, level, false)?
        } else {
            PyDict::new(py)
        };
        let nodes = PyDict::new(py);
        nodes_to_pydict(py, graph, level, &nodes)?;
        group.set_item("connections", nodes)?;
        result.set_item(key, group)?;
    }
    Ok(result.into())
}
