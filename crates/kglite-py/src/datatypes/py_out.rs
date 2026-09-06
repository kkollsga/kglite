// src/datatypes/py_out.rs
use super::values::Value;
use kglite_core::api::fluent::PropertyStats;
use kglite_core::api::storage::GraphBackend;
use kglite_core::api::NodeInfo;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::IntoPyObjectExt;
use std::collections::HashMap;

mod connections;
mod grouped;
mod keys;

pub use connections::level_connections_to_pydict;
pub use grouped::{
    convert_computation_results_for_python, level_nodes_to_pydict, level_single_values_to_pydict,
    level_unique_values_to_pydict, level_values_to_pydict,
};

pub fn nodeinfo_to_pydict(
    py: Python,
    graph: &GraphBackend,
    node: &NodeInfo,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("type", &node.node_type)?;
    dict.set_item("title", graph_value_to_py(py, graph, &node.title)?)?;
    dict.set_item("id", value_to_py(py, &node.id)?)?;

    for (k, v) in &node.properties {
        dict.set_item(k, graph_value_to_py(py, graph, v)?)?;
    }

    Ok(dict.into())
}

pub fn graph_value_to_py(py: Python, graph: &GraphBackend, value: &Value) -> PyResult<Py<PyAny>> {
    let value = kglite_core::api::session::resolve_noderef_value(graph, value);
    value_to_py(py, &value)
}

pub fn value_to_py(py: Python, value: &Value) -> PyResult<Py<PyAny>> {
    match value {
        Value::String(s) => s.clone().into_py_any(py),
        Value::Float64(f) => f.into_py_any(py),
        Value::Int64(i) => i.into_py_any(py),
        Value::Boolean(b) => b.into_py_any(py),
        Value::UniqueId(u) => u.into_py_any(py),
        Value::DateTime(d) => d.format("%Y-%m-%d").to_string().into_py_any(py),
        // Timestamp → native Python datetime.datetime (pyo3 chrono bridge).
        Value::Timestamp(dt) => dt.into_py_any(py),
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
        // Native conversion of the collection / graph-entity variants —
        // no JSON-string inference (see py_convert.rs).
        Value::List(items) => {
            let py_items: PyResult<Vec<Py<PyAny>>> =
                items.iter().map(|v| value_to_py(py, v)).collect();
            Ok(PyList::new(py, py_items?)?.into_any().unbind())
        }
        Value::Map(entries) => {
            let dict = PyDict::new(py);
            for (k, v) in entries.iter() {
                dict.set_item(k, value_to_py(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
        Value::Node(node_val) => Ok(node_to_py(py, node_val)?.into_any().unbind()),
        Value::Relationship(rel_val) => Ok(rel_to_py(py, rel_val)?.into_any().unbind()),
        Value::Path(path_val) => {
            let dict = PyDict::new(py);
            // Convert the members in place. This used to route each element
            // back through `value_to_py` as `Value::Node(Box::new(n.clone()))`,
            // which deep-cloned every node and relationship on the path — and
            // boxed it — purely to re-enter a match arm. A path of k hops paid
            // 2k+1 whole-entity clones to produce the same dicts.
            let py_nodes: PyResult<Vec<Py<PyAny>>> = path_val
                .nodes
                .iter()
                .map(|n| node_to_py(py, n).map(|d| d.into_any().unbind()))
                .collect();
            let py_rels: PyResult<Vec<Py<PyAny>>> = path_val
                .rels
                .iter()
                .map(|r| rel_to_py(py, r).map(|d| d.into_any().unbind()))
                .collect();
            dict.set_item("nodes", PyList::new(py, py_nodes?)?)?;
            dict.set_item("relationships", PyList::new(py, py_rels?)?)?;
            Ok(dict.into_any().unbind())
        }
    }
}

/// A materialised node as the Python dict `{id, labels, properties}`.
///
/// Borrows the node: the `Value::Node` arm and the path members both reach it
/// without constructing an owned `Value` first.
fn node_to_py<'py>(
    py: Python<'py>,
    node_val: &kglite_core::api::NodeValue,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", node_val.id)?;
    // labels mirror Neo4j/Bolt shape: list of strings
    dict.set_item("labels", PyList::new(py, &node_val.labels)?)?;
    // properties as a nested dict — recursive value_to_py means nested
    // Nodes/Lists/Maps round-trip cleanly.
    let props_dict = PyDict::new(py);
    for (k, v) in node_val.properties.iter() {
        props_dict.set_item(k, value_to_py(py, v)?)?;
    }
    dict.set_item("properties", props_dict)?;
    Ok(dict)
}

/// A materialised relationship as `{id, start, end, type, properties}`.
fn rel_to_py<'py>(
    py: Python<'py>,
    rel_val: &kglite_core::api::RelValue,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", rel_val.id)?;
    dict.set_item("start", rel_val.start_id)?;
    dict.set_item("end", rel_val.end_id)?;
    dict.set_item("type", &rel_val.rel_type)?;
    let props_dict = PyDict::new(py);
    for (k, v) in rel_val.properties.iter() {
        props_dict.set_item(k, value_to_py(py, v)?)?;
    }
    dict.set_item("properties", props_dict)?;
    Ok(dict)
}

pub fn hashmap_to_pydict<'py>(
    py: Python<'py>,
    graph: &GraphBackend,
    map: &HashMap<String, Value>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (k, v) in map {
        dict.set_item(k, graph_value_to_py(py, graph, v)?)?;
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

pub fn string_pairs_to_pydict(py: Python, pairs: &[(String, String)]) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);

    for (key, value) in pairs {
        result.set_item(key, value)?;
    }

    Ok(result.into())
}

/// Convert pattern matching results to a Python list of dictionaries.
///
/// Takes the graph (not just the interner): `MatchBinding::Edge` is
/// index-only, so edge properties are resolved here — once per *returned*
/// binding — instead of being cloned into every candidate binding on the
/// matcher's expansion hot path.
pub fn pattern_matches_to_pylist(
    py: Python,
    matches: &[kglite_core::api::fluent::PatternMatch],
    graph: &kglite_core::api::DirGraph,
) -> PyResult<Py<PyAny>> {
    use kglite_core::api::fluent::MatchBinding;
    use kglite_core::api::GraphRead;

    let interner = &graph.interner;

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
                } => {
                    binding_dict.set_item("source_idx", source.index())?;
                    binding_dict.set_item("target_idx", target.index())?;
                    binding_dict.set_item("edge_index", edge_index.index())?;
                    binding_dict.set_item("connection_type", interner.resolve(*connection_type))?;
                    let props_dict = PyDict::new(py);
                    if let Some(edge_data) = graph.graph.edge_weight(*edge_index) {
                        for (key, value) in edge_data.properties_cloned(interner) {
                            props_dict.set_item(key, value_to_py(py, &value)?)?;
                        }
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

                    // Add exact path hops while retaining the established
                    // node/type fields exposed by the fluent API.
                    let path_list = PyList::empty(py);
                    for hop in path {
                        let step_dict = PyDict::new(py);
                        step_dict.set_item("node_idx", hop.node.index())?;
                        step_dict.set_item("edge_index", hop.edge.index())?;
                        step_dict
                            .set_item("connection_type", interner.resolve(hop.connection_type))?;
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
