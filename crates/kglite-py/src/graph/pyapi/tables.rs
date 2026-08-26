//! Table-valued property helpers: DataFrame ⇄ `list<map>` with fidelity.
//!
//! The stored value is a plain list of maps — queryable with today's Cypher
//! (`UNWIND o.line_items`, `o.line_items[2].qty`) — while column order,
//! dtypes, and nullability live in the graph's `table_property_meta`
//! registry ([`TablePropertyMeta`]) and are restored only when a DataFrame
//! is reconstructed. Writes route through `session::execute_mut`, so write
//! scope, constraints, declared shapes, WAL, and CDC all apply exactly as
//! for any other `SET`.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use kglite_core::api::session::{execute_mut, execute_read, CsvImportPolicy, ExecuteOptions};
use kglite_core::api::{table_meta_key, TablePropertyMeta};
use kglite_core::datatypes::values::Value;
use kglite_core::datatypes::PropMap;

use crate::datatypes::on_invalid::OnInvalid;
use crate::datatypes::{py_in, py_out};
use crate::graph::{get_graph_mut, KnowledgeGraph};

fn table_opts(params: &HashMap<String, Value>) -> ExecuteOptions<'_> {
    ExecuteOptions {
        params,
        deadline: None,
        max_rows: None,
        lazy_eligible: false,
        parallel: false,
        disabled_passes: None,
        embedder: None,
        value_codecs: None,
        cancel: None,
        write_scope: None,
        git_sha: None,
        modified_by: None,
        csv_import: CsvImportPolicy::LocalFilesystem,
    }
}

fn scalar_count(result: &kglite_core::api::cypher::CypherResult) -> usize {
    match result.rows.first().and_then(|r| r.first()) {
        Some(Value::Int64(n)) => *n as usize,
        _ => 0,
    }
}

/// The identifier rule for the type/property names interpolated into the
/// helper's internal Cypher — refuses anything that would need quoting.
fn require_plain_identifier(what: &str, name: &str) -> PyResult<()> {
    let ok = !name.is_empty()
        && name.chars().all(|c| c.is_alphanumeric() || c == '_')
        && !name.chars().next().unwrap().is_numeric();
    if ok {
        Ok(())
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{what} {name:?} must be a plain identifier (letters, digits, '_')"
        )))
    }
}

#[pymethods]
impl KnowledgeGraph {
    /// Store a DataFrame as a table-valued property (a queryable list of maps).
    #[pyo3(signature = (node_type, node_id, property, data))]
    fn set_table_property(
        &mut self,
        py: Python<'_>,
        node_type: &str,
        node_id: &Bound<'_, PyAny>,
        property: &str,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        require_plain_identifier("node_type", node_type)?;
        require_plain_identifier("property", property)?;
        let columns: Vec<String> = data
            .getattr("columns")
            .and_then(|c| c.call_method0("tolist"))
            .and_then(|c| c.extract())
            .map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "set_table_property expects a pandas DataFrame (no .columns.tolist())",
                )
            })?;
        if columns.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "set_table_property: the DataFrame has no columns",
            ));
        }
        let dtypes_obj = data.getattr("dtypes")?;
        let mut meta = TablePropertyMeta {
            columns: columns.clone(),
            ..Default::default()
        };
        for name in &columns {
            if name.contains('\u{1f}') {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "set_table_property: column name {name:?} contains the reserved \
                     separator U+001F"
                )));
            }
            let dtype: String = dtypes_obj.get_item(name)?.str()?.extract()?;
            meta.dtypes.insert(name.clone(), dtype);
        }

        // The frame converter gives typed columns + per-cell nulls with the
        // same value semantics add_nodes uses — one dialect, one error set.
        // OnInvalid::Error, not Warn: a table property is a fidelity
        // contract, and a silently-dropped bad cell breaks it.
        let frame = py_in::pandas_to_dataframe_with_options(
            data,
            &[],
            &columns,
            None,
            false,
            OnInvalid::Error,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "set_table_property({property}): {e}"
            ))
        })?;

        let mut rows: Vec<Value> = Vec::with_capacity(frame.row_count());
        for row_idx in 0..frame.row_count() {
            let mut pairs: Vec<(kglite_core::datatypes::PropKey, Value)> = Vec::new();
            for name in &columns {
                match frame.get_value(row_idx, name) {
                    Some(Value::Null) | None => {
                        if !meta.nullable.contains(name) {
                            meta.nullable.push(name.clone());
                        }
                    }
                    Some(v) => pairs.push((name.as_str().into(), v)),
                }
            }
            rows.push(Value::Map(PropMap::from_pairs(pairs)));
        }
        let row_count = rows.len();

        let mut params: HashMap<String, Value> = HashMap::new();
        params.insert("nid".to_string(), py_in::py_value_to_value(node_id)?);
        params.insert("rows".to_string(), Value::List(rows));
        let query = format!(
            "MATCH (n:{node_type} {{id: $nid}}) SET n.{property} = $rows RETURN count(n) AS c"
        );
        let graph = get_graph_mut(&mut self.inner);
        let outcome = execute_mut(graph, &query, &table_opts(&params))
            .map_err(crate::error_py::kg_to_pyerr)?;
        if scalar_count(&outcome.result) == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "set_table_property: no {node_type} node with id {node_id}"
            )));
        }
        get_graph_mut(&mut self.inner)
            .table_property_meta
            .insert(table_meta_key(node_type, property), meta);
        self.commit_wal()?;
        let _ = py;
        Ok(row_count)
    }

    /// Reconstruct a table-valued property as a pandas DataFrame, restoring
    /// stored column order and dtypes.
    #[pyo3(signature = (node_type, node_id, property))]
    fn get_table_property(
        &self,
        py: Python<'_>,
        node_type: &str,
        node_id: &Bound<'_, PyAny>,
        property: &str,
    ) -> PyResult<Py<PyAny>> {
        require_plain_identifier("node_type", node_type)?;
        require_plain_identifier("property", property)?;
        let mut params: HashMap<String, Value> = HashMap::new();
        params.insert("nid".to_string(), py_in::py_value_to_value(node_id)?);
        let query = format!(
            "MATCH (n:{node_type} {{id: $nid}}) RETURN n.{property} AS rows, count(n) AS c"
        );
        let outcome = execute_read(&self.inner, &query, &table_opts(&params))
            .map_err(crate::error_py::kg_to_pyerr)?;
        let row = outcome.result.rows.first().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "get_table_property: no {node_type} node with id {node_id}"
            ))
        })?;
        let stored = row.first().cloned().unwrap_or(Value::Null);
        let rows_py = py_out::value_to_py(py, &stored)?;
        if matches!(stored, Value::Null) {
            // A node without the property reconstructs as an empty frame
            // (with the registered columns, when known).
            let empty = PyList::empty(py);
            return build_frame(py, empty.as_any(), self, node_type, property);
        }
        build_frame(py, rows_py.bind(py), self, node_type, property)
    }
}

fn build_frame(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    kg: &KnowledgeGraph,
    node_type: &str,
    property: &str,
) -> PyResult<Py<PyAny>> {
    let pandas = py.import("pandas")?;
    let meta = kg
        .inner
        .table_property_meta
        .get(&table_meta_key(node_type, property));
    let kwargs = PyDict::new(py);
    if let Some(meta) = meta {
        kwargs.set_item("columns", PyList::new(py, &meta.columns)?)?;
    }
    let df = pandas
        .getattr("DataFrame")
        .and_then(|ctor| ctor.call((rows,), Some(&kwargs)))?;
    if let Some(meta) = meta {
        // Restore dtypes; columns that held nulls use pandas nullable
        // dtypes so NaN-coerced integers come back as integers.
        for (name, dtype) in &meta.dtypes {
            let target = if meta.nullable.contains(name) {
                match dtype.as_str() {
                    "int64" => "Int64".to_string(),
                    "bool" => "boolean".to_string(),
                    other => other.to_string(),
                }
            } else {
                dtype.clone()
            };
            if let Ok(col) = df.get_item(name) {
                if let Ok(cast) = col.call_method1("astype", (target.as_str(),)) {
                    df.set_item(name, cast)?;
                }
            }
        }
    }
    Ok(df.into())
}
