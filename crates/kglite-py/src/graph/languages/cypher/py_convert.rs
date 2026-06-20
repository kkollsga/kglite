// src/graph/cypher/py_convert.rs
// Convert pre-processed result data to Python objects.
// Used by ResultView for lazy conversion and by to_df=True direct paths.

use crate::datatypes::py_out;
use crate::datatypes::values::Value;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::IntoPyObjectExt;

// ========================================================================
// PreProcessedValue — the core data type for ResultView rows
// ========================================================================
//
// Phase A.1 / C7a — the standalone `json_value_to_py` helper that
// recursively converted a `serde_json::Value` to a Python object is
// deleted along with `ParsedJson`. The JSON-string inference hack
// it served (C4 removal target) is gone, so no caller remains.
// Native `Value::List` / `Value::Map` / `Value::Node` / ... flow
// straight through `py_out::value_to_py` instead.

/// Wraps a Value for the Python conversion boundary.
///
/// Phase A.1 / C7a — collapsed to a single `Plain(Value)` variant
/// after the C4 removal of the JSON-string inference hack made the
/// `ParsedJson(serde_json::Value)` variant dead. Kept as a newtype-
/// style enum (rather than an `pub struct(Value)` or just `Value`)
/// because the existing public API surface in `result_view.rs` and
/// `kg_core.rs` consumes `PreProcessedValue` by pattern match — a
/// later cleanup can collapse this enum entirely once those sites
/// migrate to bare `Value`.
#[derive(Clone)]
pub enum PreProcessedValue {
    /// The only variant. Convert via py_out::value_to_py.
    Plain(Value),
}

/// Convert a pre-processed value to a Python object.
pub fn preprocessed_value_to_py(py: Python<'_>, pv: &PreProcessedValue) -> PyResult<Py<PyAny>> {
    match pv {
        PreProcessedValue::Plain(v) => py_out::value_to_py(py, v),
    }
}

// ========================================================================
// Pre-processing: Value → PreProcessedValue (runs without GIL)
// ========================================================================

/// Wrap owned Value rows for the Python conversion boundary.
///
/// Phase A.1 / C4 — the prior JSON-string inference hack here (detect
/// `Value::String("[...]")` / `Value::String("{...}")` and re-parse via
/// `serde_json::from_str` into `PreProcessedValue::ParsedJson`) is
/// removed. Native `Value::List` / `Value::Map` / `Value::Node` /
/// `Value::Relationship` / `Value::Path` now flow through
/// `py_out::value_to_py` directly — no inference, no mis-parse risk
/// (a user-set property value of `"[shopping list]"` no longer gets
/// silently re-typed as a list).
///
/// `PreProcessedValue::ParsedJson` is preserved as a variant for now
/// (other call sites reference it); slated for removal in the C7a
/// slim-down pass once the dead path is fully audited.
pub fn preprocess_values_owned(rows: Vec<Vec<Value>>) -> Vec<Vec<PreProcessedValue>> {
    rows.into_iter()
        .map(|row| row.into_iter().map(PreProcessedValue::Plain).collect())
        .collect()
}

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
    Ok(stats_dict)
}

// ========================================================================
// DataFrame conversion (used by to_df=True shortcut and ResultView::to_df)
// ========================================================================

/// Convert pre-processed rows to a pandas DataFrame.
pub fn preprocessed_result_to_dataframe(
    py: Python<'_>,
    columns: &[String],
    rows: &[Vec<PreProcessedValue>],
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
                col_list.append(preprocessed_value_to_py(py, pv)?)?;
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
