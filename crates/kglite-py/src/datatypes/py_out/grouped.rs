use super::keys::presentation_keys;
use super::Value;
use super::{graph_value_to_py, nodeinfo_to_pydict, value_to_py};
use kglite_core::api::fluent::{LevelNodes, LevelValues, StatResult, UniqueValues};
use kglite_core::api::storage::GraphBackend;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

fn node_group_label(group: &LevelNodes, parent_key: Option<&str>) -> String {
    match parent_key {
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
    }
}

fn node_group_value(
    py: Python,
    graph: &GraphBackend,
    group: &LevelNodes,
    parent_info: bool,
    flattened: bool,
) -> PyResult<Py<PyAny>> {
    let nodes: Vec<Py<PyAny>> = group
        .nodes
        .iter()
        .map(|node| nodeinfo_to_pydict(py, graph, node))
        .collect::<PyResult<_>>()?;
    if !parent_info || group.parent_idx.is_none() {
        return Ok(PyList::new(py, nodes)?.into());
    }
    let result = PyDict::new(py);
    if let Some(kind) = &group.parent_type {
        result.set_item("type", kind)?;
    }
    result.set_item("title", &group.parent_title)?;
    if let Some(id) = &group.parent_id {
        result.set_item("id", value_to_py(py, id)?)?;
    }
    // The existing flattened metadata shape uses nodes; grouped uses children.
    result.set_item(
        if flattened { "nodes" } else { "children" },
        PyList::new(py, nodes)?,
    )?;
    Ok(result.into())
}

pub fn level_nodes_to_pydict(
    py: Python,
    graph: &GraphBackend,
    level_nodes: &[LevelNodes],
    parent_key: Option<&str>,
    parent_info: Option<bool>,
    flatten_single_parent: Option<bool>,
) -> PyResult<Py<PyAny>> {
    if flatten_single_parent.unwrap_or(true) && level_nodes.len() == 1 {
        return node_group_value(
            py,
            graph,
            &level_nodes[0],
            parent_info.unwrap_or(false),
            true,
        );
    }
    let keys = presentation_keys(
        level_nodes
            .iter()
            .map(|group| {
                (
                    node_group_label(group, parent_key),
                    group.parent_idx.map(|idx| idx.index()),
                )
            })
            .collect(),
        &[],
    );
    let result = PyDict::new(py);
    for (group, key) in level_nodes.iter().zip(keys) {
        result.set_item(
            key,
            node_group_value(py, graph, group, parent_info.unwrap_or(false), false)?,
        )?;
    }
    Ok(result.into())
}

pub fn level_values_to_pydict(
    py: Python,
    graph: &GraphBackend,
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
                    .map(|v| graph_value_to_py(py, graph, v))
                    .collect::<PyResult<_>>()?;
                Ok(PyTuple::new(py, &tuple_values)?.into())
            })
            .collect::<PyResult<_>>()?;
        return Ok(PyList::new(py, values)?.into());
    }

    let result = PyDict::new(py);

    let keys = presentation_keys(
        level_values
            .iter()
            .map(|group| (group.parent_title.clone(), &group.values))
            .collect(),
        &[],
    );
    for (group, key) in level_values.iter().zip(keys) {
        let values: Vec<Py<PyAny>> = group
            .values
            .iter()
            .map(|vec_values| {
                let tuple_values: Vec<Py<PyAny>> = vec_values
                    .iter()
                    .map(|v| graph_value_to_py(py, graph, v))
                    .collect::<PyResult<_>>()?;
                Ok(PyTuple::new(py, &tuple_values)?.into())
            })
            .collect::<PyResult<_>>()?;

        result.set_item(key, values)?;
    }

    Ok(result.into())
}

pub fn level_single_values_to_pydict(
    py: Python,
    graph: &GraphBackend,
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
            .map(|vec_values| graph_value_to_py(py, graph, &vec_values[0]))
            .collect::<PyResult<_>>()?;
        return Ok(PyList::new(py, values)?.into());
    }

    let result = PyDict::new(py);

    let keys = presentation_keys(
        level_values
            .iter()
            .map(|group| (group.parent_title.clone(), &group.values))
            .collect(),
        &[],
    );
    for (group, key) in level_values.iter().zip(keys) {
        let values: Vec<Py<PyAny>> = group
            .values
            .iter()
            .map(|vec_values| graph_value_to_py(py, graph, &vec_values[0]))
            .collect::<PyResult<_>>()?;

        result.set_item(key, values)?;
    }

    Ok(result.into())
}

pub fn level_unique_values_to_pydict(
    py: Python,
    graph: &GraphBackend,
    values: &[UniqueValues],
) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);
    let keys = presentation_keys(
        values
            .iter()
            .map(|group| {
                (
                    group.parent_title.clone(),
                    group.parent_idx.map(|idx| idx.index()),
                )
            })
            .collect(),
        &[],
    );
    for (unique_values, key) in values.iter().zip(keys) {
        let py_values: PyResult<Vec<Py<PyAny>>> = unique_values
            .values
            .iter()
            .map(|v| graph_value_to_py(py, graph, v))
            .collect();
        result.set_item(key, PyList::new(py, py_values?)?)?;
    }
    Ok(result.into())
}

pub fn convert_computation_results_for_python(results: Vec<StatResult>) -> PyResult<Py<PyAny>> {
    Python::attach(|py| {
        let dict = PyDict::new(py);

        let keys = presentation_keys(
            results
                .iter()
                .enumerate()
                .map(|(i, result)| {
                    let label = result.parent_title.clone().unwrap_or_else(|| {
                        result.parent_idx.map_or_else(
                            || format!("result_{i}"),
                            |idx| format!("node_{}", idx.index()),
                        )
                    });
                    (
                        label,
                        (
                            result.parent_idx.map(|idx| idx.index()),
                            result.node_idx.map(|idx| idx.index()),
                            &result.value,
                        ),
                    )
                })
                .collect(),
            &[],
        );
        for (result, key) in results.iter().zip(keys) {
            if result.error_msg.is_some() {
                dict.set_item(&key, py.None())?;
            } else {
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
                        dict.set_item(&key, py.None())?;
                    }
                    _ => {
                        dict.set_item(&key, py.None())?;
                    }
                }
            }
        }

        // Return empty dict if no results (not an error - traversal may have found nothing)
        Ok(dict.into())
    })
}
