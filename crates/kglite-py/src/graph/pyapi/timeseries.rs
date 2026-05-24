// src/graph/timeseries.rs
//
// PyO3 methods for timeseries operations on KnowledgeGraph.

use crate::datatypes::py_in;
use crate::graph::features::timeseries::{NodeTimeseries, TimeseriesConfig};
use crate::graph::storage::GraphRead;
use crate::graph::{get_graph_mut, KnowledgeGraph};
use chrono::NaiveDate;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

#[pymethods]
impl KnowledgeGraph {
    // ── Config ────────────────────────────────────────────────────────────

    /// Configure timeseries metadata for a node type.
    ///
    /// `resolution` declares the time granularity: "year", "month", or "day".
    /// `channels` lists known channel names (informational).
    /// `units` maps channel names to unit strings (e.g. {"oil": "MSm3"}).
    /// `bin_type` describes what values represent: "total", "mean", or "sample".
    #[pyo3(signature = (node_type, *, resolution, channels=None, units=None, bin_type=None))]
    fn set_timeseries(
        &mut self,
        node_type: String,
        resolution: String,
        channels: Option<Vec<String>>,
        units: Option<HashMap<String, String>>,
        bin_type: Option<String>,
    ) -> PyResult<()> {
        crate::graph::features::timeseries::validate_resolution(&resolution)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let graph = get_graph_mut(&mut self.inner);
        graph.timeseries_configs.insert(
            node_type,
            TimeseriesConfig {
                resolution,
                channels: channels.unwrap_or_default(),
                units: units.unwrap_or_default(),
                bin_type,
            },
        );
        Ok(())
    }

    /// Get timeseries configuration for a node type, or all types.
    #[pyo3(signature = (node_type=None))]
    fn timeseries_config(&self, py: Python<'_>, node_type: Option<String>) -> PyResult<Py<PyAny>> {
        let graph = &self.inner;
        if let Some(nt) = node_type {
            match graph.timeseries_configs.get(&nt) {
                Some(config) => ts_config_to_py(py, config),
                None => Ok(py.None()),
            }
        } else {
            if graph.timeseries_configs.is_empty() {
                return Ok(py.None());
            }
            let result = PyDict::new(py);
            for (nt, config) in &graph.timeseries_configs {
                result.set_item(nt, ts_config_to_py(py, config)?)?;
            }
            Ok(result.into_any().unbind())
        }
    }

    // ── Per-node data ─────────────────────────────────────────────────────

    /// Set the sorted time index for a specific node.
    ///
    /// Accepts either:
    /// - A list of date strings: `["2020-01", "2020-02"]`
    /// - A list of integer lists for backwards compat: `[[2020, 1], [2020, 2]]`
    ///
    /// If the node already has a timeseries, this replaces its time index
    /// and clears all channels (since they would no longer align).
    #[pyo3(signature = (node_id, keys))]
    fn set_time_index(
        &mut self,
        node_id: &Bound<'_, PyAny>,
        keys: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let date_keys = parse_keys_from_python(keys)?;

        crate::graph::features::timeseries::validate_keys_sorted(&date_keys)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let graph = get_graph_mut(&mut self.inner);
        let id_val = py_in::py_value_to_value(node_id)?;
        let node_idx = find_node_by_id(graph, &id_val)?;

        graph.timeseries_store.insert(
            node_idx.index(),
            NodeTimeseries {
                keys: date_keys,
                channels: HashMap::new(),
            },
        );

        Ok(())
    }

    /// Add a timeseries channel to a node.
    ///
    /// The node must already have a time index set (via `set_time_index` or `add_timeseries`).
    /// The values length must match the time index length.
    #[pyo3(signature = (node_id, channel_name, values))]
    fn add_ts_channel(
        &mut self,
        node_id: &Bound<'_, PyAny>,
        channel_name: String,
        values: Vec<f64>,
    ) -> PyResult<()> {
        let graph = get_graph_mut(&mut self.inner);
        let id_val = py_in::py_value_to_value(node_id)?;
        let node_idx = find_node_by_id(graph, &id_val)?;

        let ts = graph
            .timeseries_store
            .get_mut(&node_idx.index())
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Node has no time index. Call set_time_index() first.",
                )
            })?;

        crate::graph::features::timeseries::validate_channel_length(
            ts.keys.len(),
            values.len(),
            &channel_name,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        // Update config channels list
        if let Some(node) = graph.graph.node_weight(node_idx) {
            let nt = node.node_type_str(&graph.interner).to_string();
            if let Some(config) = graph.timeseries_configs.get_mut(&nt) {
                if !config.channels.contains(&channel_name) {
                    config.channels.push(channel_name.clone());
                }
            }
        }

        ts.channels.insert(channel_name, values);
        Ok(())
    }

    /// Bulk-load timeseries data from a DataFrame.
    ///
    /// Groups rows by `fk`, sorts by `time_key`, and attaches the resulting
    /// timeseries to matching nodes (found by node ID).
    ///
    /// `time_key` accepts either:
    /// - A list of column names for composite keys: `["year", "month"]` — combined into NaiveDate
    /// - A single column name (as 1-element list): `["date"]` — parsed as date strings
    ///
    /// `channels` accepts either:
    /// - a list of column names (used as channel names): `["oil", "gas"]`
    /// - a dict mapping channel names to column names: `{"oil": "prfOilCol"}`
    ///
    /// `units` optionally maps channel names to unit strings, merged into config.
    #[pyo3(signature = (node_type, *, data, fk, time_key, channels, resolution=None, units=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_timeseries(
        &mut self,
        py: Python<'_>,
        node_type: String,
        data: &Bound<'_, PyAny>,
        fk: String,
        time_key: Vec<String>,
        channels: &Bound<'_, PyAny>,
        resolution: Option<String>,
        units: Option<HashMap<String, String>>,
    ) -> PyResult<Py<PyAny>> {
        // Resolve resolution: parameter > existing config > auto-detect
        let graph_ref = &self.inner;
        let resolved_resolution = if let Some(r) = resolution {
            crate::graph::features::timeseries::validate_resolution(&r)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
            r
        } else if let Some(config) = graph_ref.timeseries_configs.get(&node_type) {
            config.resolution.clone()
        } else {
            // Auto-detect from time_key column count
            match time_key.len() {
                1 => "month".to_string(), // default for single-column date strings
                2 => "month".to_string(),
                3 => "day".to_string(),
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Cannot auto-detect resolution. Pass resolution= or call set_timeseries() first.",
                    ));
                }
            }
        };

        // Parse channels: dict or list
        let channel_map: Vec<(String, String)> = if let Ok(d) = data
            .py()
            .import("builtins")?
            .call_method1("isinstance", (channels, data.py().get_type::<PyDict>()))
            .and_then(|r| r.extract::<bool>())
        {
            if d {
                let dict = channels.cast::<PyDict>()?;
                dict.iter()
                    .map(|(k, v)| Ok((k.extract::<String>()?, v.extract::<String>()?)))
                    .collect::<PyResult<Vec<_>>>()?
            } else {
                let list: Vec<String> = channels.extract()?;
                list.into_iter().map(|c| (c.clone(), c)).collect()
            }
        } else {
            let list: Vec<String> = channels.extract()?;
            list.into_iter().map(|c| (c.clone(), c)).collect()
        };

        // Extract columns from DataFrame as Python lists
        let fk_col: Vec<Py<PyAny>> = data.get_item(&fk)?.call_method0("tolist")?.extract()?;

        // Build NaiveDate keys from time columns
        let n_rows = fk_col.len();
        let date_keys: Vec<NaiveDate> = if time_key.len() == 1 {
            // Single column: parse as date strings
            let raw: Vec<String> = data
                .get_item(&time_key[0])?
                .call_method1("astype", ("str",))?
                .call_method0("tolist")?
                .extract()?;
            raw.iter()
                .map(|s| {
                    crate::graph::features::timeseries::parse_date_query(s)
                        .map(|(d, _)| d)
                        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
                })
                .collect::<PyResult<Vec<_>>>()?
        } else {
            // Multiple columns: combine year + month [+ day] → NaiveDate
            let mut int_cols: Vec<Vec<i64>> = Vec::with_capacity(time_key.len());
            for tk in &time_key {
                let col: Vec<i64> = data.get_item(tk)?.call_method0("tolist")?.extract()?;
                int_cols.push(col);
            }
            (0..n_rows)
                .map(|i| {
                    let year = int_cols[0][i] as i32;
                    let month = if int_cols.len() > 1 {
                        int_cols[1][i] as u32
                    } else {
                        1
                    };
                    let day = if int_cols.len() > 2 {
                        int_cols[2][i] as u32
                    } else {
                        1
                    };
                    crate::graph::features::timeseries::date_from_ymd(year, month, day)
                        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
                })
                .collect::<PyResult<Vec<_>>>()?
        };

        let mut value_cols: Vec<(String, Vec<f64>)> = Vec::with_capacity(channel_map.len());
        for (ch_name, col_name) in &channel_map {
            let col: Vec<f64> = data.get_item(col_name)?.call_method0("tolist")?.extract()?;
            value_cols.push((ch_name.clone(), col));
        }

        // Group row indices by FK value
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, fk_val) in fk_col.iter().enumerate() {
            let key = fk_val.bind(py).str()?.to_string();
            groups.entry(key).or_default().push(i);
        }

        // Build ID index for this node type
        let graph = get_graph_mut(&mut self.inner);
        graph.build_id_index(&node_type);

        let mut nodes_loaded = 0usize;
        let mut total_records = 0usize;

        for (fk_str, row_indices) in &groups {
            // Look up node by FK value (try as string, int, or unique_id)
            let node_idx = {
                let id_str = crate::datatypes::values::Value::String(fk_str.clone());
                if let Some(idx) = graph.lookup_by_id_normalized(&node_type, &id_str) {
                    idx
                } else if let Ok(n) = fk_str.parse::<i64>() {
                    let id_int = crate::datatypes::values::Value::Int64(n);
                    if let Some(idx) = graph.lookup_by_id_normalized(&node_type, &id_int) {
                        idx
                    } else {
                        continue; // No matching node
                    }
                } else {
                    continue; // No matching node
                }
            };

            // Sort row indices by date key
            let mut sorted_rows: Vec<usize> = row_indices.clone();
            sorted_rows.sort_by(|&a, &b| date_keys[a].cmp(&date_keys[b]));

            // Build NaiveDate keys
            let keys: Vec<NaiveDate> = sorted_rows.iter().map(|&ri| date_keys[ri]).collect();

            // Build channels
            let mut channels_data: HashMap<String, Vec<f64>> = HashMap::new();
            for (ch_name, col) in &value_cols {
                let values: Vec<f64> = sorted_rows.iter().map(|&ri| col[ri]).collect();
                channels_data.insert(ch_name.clone(), values);
            }

            graph.timeseries_store.insert(
                node_idx.index(),
                NodeTimeseries {
                    keys,
                    channels: channels_data,
                },
            );

            nodes_loaded += 1;
            total_records += sorted_rows.len();
        }

        // Auto-configure / update TimeseriesConfig
        let channel_names: Vec<String> = channel_map.iter().map(|(n, _)| n.clone()).collect();
        let existing = graph.timeseries_configs.get(&node_type);
        let mut merged_channels = existing.map(|c| c.channels.clone()).unwrap_or_default();
        for ch in &channel_names {
            if !merged_channels.contains(ch) {
                merged_channels.push(ch.clone());
            }
        }
        let mut merged_units = existing.map(|c| c.units.clone()).unwrap_or_default();
        if let Some(u) = units {
            for (k, v) in u {
                merged_units.insert(k, v);
            }
        }
        let bin_type = existing.and_then(|c| c.bin_type.clone());

        graph.timeseries_configs.insert(
            node_type,
            TimeseriesConfig {
                resolution: resolved_resolution,
                channels: merged_channels,
                units: merged_units,
                bin_type,
            },
        );

        // Return summary dict
        let result = PyDict::new(py);
        result.set_item("nodes_loaded", nodes_loaded)?;
        result.set_item("total_records", total_records)?;
        result.set_item("total_rows", n_rows)?;
        Ok(result.into_any().unbind())
    }

    // ── Retrieval ─────────────────────────────────────────────────────────

    /// Extract timeseries data for a node.
    ///
    /// Range arguments `start`/`end` are date strings (e.g. "2020", "2020-2").
    /// If `channel` is given, returns `{keys: [...], values: [...]}`.
    /// Otherwise returns `{keys: [...], channels: {name: [...], ...}}`.
    /// Returns None if the node has no timeseries data.
    #[pyo3(signature = (node_id, channel=None, start=None, end=None))]
    fn timeseries(
        &self,
        py: Python<'_>,
        node_id: &Bound<'_, PyAny>,
        channel: Option<String>,
        start: Option<String>,
        end: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let graph = &self.inner;
        let id_val = py_in::py_value_to_value(node_id)?;
        let node_idx = find_node_by_id_readonly(graph, &id_val)?;

        let ts = match graph.get_node_timeseries(node_idx.index()) {
            Some(ts) => ts,
            None => return Ok(py.None()),
        };

        // Parse date strings and compute range
        let start_date = start
            .as_ref()
            .map(|s| crate::graph::features::timeseries::parse_date_query(s).map(|(d, _)| d))
            .transpose()
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        let end_date = end
            .as_ref()
            .map(|s| {
                crate::graph::features::timeseries::parse_date_query(s)
                    .map(|(d, prec)| crate::graph::features::timeseries::expand_end(d, prec))
            })
            .transpose()
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let (lo, hi) =
            crate::graph::features::timeseries::find_range(&ts.keys, start_date, end_date);
        let keys_slice = &ts.keys[lo..hi];

        let result = PyDict::new(py);
        let keys_py = PyList::new(
            py,
            keys_slice.iter().map(|d| d.format("%Y-%m-%d").to_string()),
        )?;
        result.set_item("keys", keys_py)?;

        if let Some(ch_name) = channel {
            match ts.channels.get(&ch_name) {
                Some(values) => {
                    let values_py = PyList::new(py, &values[lo..hi])?;
                    result.set_item("values", values_py)?;
                }
                None => {
                    return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                        "Channel '{}' not found",
                        ch_name
                    )));
                }
            }
        } else {
            let channels_dict = PyDict::new(py);
            for (name, values) in &ts.channels {
                let values_py = PyList::new(py, &values[lo..hi])?;
                channels_dict.set_item(name, values_py)?;
            }
            result.set_item("channels", channels_dict)?;
        }

        Ok(result.into_any().unbind())
    }

    /// Get the time index for a node as ISO date strings, or None.
    #[pyo3(signature = (node_id,))]
    fn time_index(&self, py: Python<'_>, node_id: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let graph = &self.inner;
        let id_val = py_in::py_value_to_value(node_id)?;
        let node_idx = find_node_by_id_readonly(graph, &id_val)?;

        match graph.get_node_timeseries(node_idx.index()) {
            Some(ts) => {
                let keys_py =
                    PyList::new(py, ts.keys.iter().map(|d| d.format("%Y-%m-%d").to_string()))?;
                Ok(keys_py.into_any().unbind())
            }
            None => Ok(py.None()),
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Convert a TimeseriesConfig to a Python dict.
fn ts_config_to_py(py: Python<'_>, config: &TimeseriesConfig) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("resolution", &config.resolution)?;
    if !config.channels.is_empty() {
        let channels = PyList::new(py, &config.channels)?;
        dict.set_item("channels", channels)?;
    }
    if !config.units.is_empty() {
        let units_dict = PyDict::new(py);
        for (k, v) in &config.units {
            units_dict.set_item(k, v)?;
        }
        dict.set_item("units", units_dict)?;
    }
    if let Some(bt) = &config.bin_type {
        dict.set_item("bin_type", bt)?;
    }
    Ok(dict.into_any().unbind())
}

use crate::datatypes::values::Value;
use crate::graph::schema::DirGraph;
use petgraph::graph::NodeIndex;

/// Parse time index keys from Python — accepts either list[str] or list[list[int]].
fn parse_keys_from_python(keys: &Bound<'_, PyAny>) -> PyResult<Vec<NaiveDate>> {
    // Try as list of strings first
    if let Ok(strings) = keys.extract::<Vec<String>>() {
        return strings
            .iter()
            .map(|s| {
                crate::graph::features::timeseries::parse_date_query(s)
                    .map(|(d, _)| d)
                    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
            })
            .collect();
    }

    // Try as list of list[int] (backwards compat)
    if let Ok(int_lists) = keys.extract::<Vec<Vec<i64>>>() {
        return int_lists
            .iter()
            .map(|parts| {
                let year = parts.first().copied().unwrap_or(2000) as i32;
                let month = parts.get(1).copied().unwrap_or(1) as u32;
                let day = parts.get(2).copied().unwrap_or(1) as u32;
                crate::graph::features::timeseries::date_from_ymd(year, month, day)
                    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
            })
            .collect();
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "keys must be a list of date strings or a list of integer lists",
    ))
}

/// Find a node by its ID value, scanning all type indices (mutable).
fn find_node_by_id(graph: &mut DirGraph, id: &Value) -> PyResult<NodeIndex> {
    let types: Vec<String> = graph.type_indices.keys().map(|s| s.to_string()).collect();
    for nt in &types {
        graph.build_id_index(nt);
        if let Some(idx) = graph.lookup_by_id_normalized(nt, id) {
            return Ok(idx);
        }
    }
    Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
        "No node found with id {:?}",
        id
    )))
}

/// Find a node by its ID value, scanning all type indices (read-only).
fn find_node_by_id_readonly(graph: &DirGraph, id: &Value) -> PyResult<NodeIndex> {
    for nt in graph.type_indices.keys() {
        if let Some(idx) = graph.lookup_by_id_normalized(nt, id) {
            return Ok(idx);
        }
    }
    Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
        "No node found with id {:?}",
        id
    )))
}
