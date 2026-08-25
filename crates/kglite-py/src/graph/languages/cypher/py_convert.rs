// src/graph/cypher/py_convert.rs
// Convert result data to Python objects.
// Used by ResultView for lazy conversion and by to_df=True direct paths.

use crate::datatypes::py_out;
use crate::datatypes::values::Value;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::IntoPyObjectExt;

// ========================================================================
// Row values reach Python as bare `Value`s
// ========================================================================
//
// No JSON-string inference happens on the way: a `Value::String("[...]")`
// or `Value::String("{...}")` is never re-parsed via `serde_json::from_str`,
// so a user-set property value of `"[shopping list]"` is not silently
// re-typed as a list. Native `Value::List` / `Value::Map` / `Value::Node` /
// `Value::Relationship` / `Value::Path` flow straight through
// `py_out::value_to_py`.

// ========================================================================
// Stats conversion
// ========================================================================

/// Convert MutationStats to a Python dict.
pub fn stats_to_py<'py>(
    py: Python<'py>,
    stats: &super::MutationStats,
) -> PyResult<Bound<'py, PyDict>> {
    let stats_dict = PyDict::new(py);
    stats_dict.set_item("nodes_created", stats.nodes_created)?;
    stats_dict.set_item("relationships_created", stats.relationships_created)?;
    stats_dict.set_item("properties_set", stats.properties_set)?;
    stats_dict.set_item("nodes_deleted", stats.nodes_deleted)?;
    stats_dict.set_item("relationships_deleted", stats.relationships_deleted)?;
    stats_dict.set_item("properties_removed", stats.properties_removed)?;
    stats_dict.set_item("indexes_added", stats.indexes_added)?;
    stats_dict.set_item("indexes_removed", stats.indexes_removed)?;
    stats_dict.set_item("constraints_added", stats.constraints_added)?;
    stats_dict.set_item("constraints_removed", stats.constraints_removed)?;
    Ok(stats_dict)
}

// ========================================================================
// DataFrame conversion (used by to_df=True shortcut and ResultView::to_df)
// ========================================================================

/// Convert result rows to a pandas DataFrame.
pub fn rows_to_dataframe(
    py: Python<'_>,
    columns: &[String],
    rows: &[Vec<Value>],
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    let col_order = PyList::empty(py);

    let col_keys: Vec<Py<PyAny>> = columns
        .iter()
        .map(|col| col.clone().into_py_any(py))
        .collect::<PyResult<_>>()?;

    for (i, key) in col_keys.iter().enumerate() {
        let col_list = PyList::empty(py);
        for row in rows {
            if let Some(pv) = row.get(i) {
                col_list.append(py_out::value_to_py(py, pv)?)?;
            } else {
                col_list.append(py.None())?;
            }
        }
        dict.set_item(key, col_list)?;
        col_order.append(key)?;
    }

    let pd = py.import("pandas")?;

    if rows.is_empty() {
        let kwargs = PyDict::new(py);
        kwargs.set_item("columns", col_order)?;
        return pd
            .call_method("DataFrame", (), Some(&kwargs))
            .map(|df| df.unbind());
    }

    let kwargs = PyDict::new(py);
    kwargs.set_item("columns", col_order)?;
    pd.call_method("DataFrame", (dict,), Some(&kwargs))
        .map(|df| df.unbind())
}
