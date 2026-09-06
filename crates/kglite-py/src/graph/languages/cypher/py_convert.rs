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
    let native_dtypes = PyDict::new(py);

    let col_keys: Vec<Py<PyAny>> = columns
        .iter()
        .map(|col| col.clone().into_py_any(py))
        .collect::<PyResult<_>>()?;

    for (i, key) in col_keys.iter().enumerate() {
        let col_list = PyList::empty(py);
        let mut kinds = 0_u8;
        for row in rows {
            let value = row.get(i);
            kinds |= dataframe_value_kind(value);
            if let Some(pv) = value {
                col_list.append(py_out::value_to_py(py, pv)?)?;
            } else {
                col_list.append(py.None())?;
            }
        }
        native_dtypes.set_item(key, dataframe_integer_dtype(kinds))?;
        dict.set_item(key, col_list)?;
        col_order.append(key)?;
    }

    crate::datatypes::pandas_out::dataframe(
        py,
        dict.as_any(),
        Some(&col_order),
        None,
        Some(&native_dtypes),
    )
}

fn dataframe_integer_dtype(kinds: u8) -> Option<&'static str> {
    match kinds {
        3 => Some("Int64"),
        5 | 7 => Some("object"),
        _ => None,
    }
}

// Match emitted Python scalar types, not core predicate equality. UniqueId and
// the unresolved NodeRef fallback emit integers; absent cells emit None.
fn dataframe_value_kind(value: Option<&Value>) -> u8 {
    match value {
        Some(Value::Int64(_) | Value::UniqueId(_) | Value::NodeRef(_)) => 1,
        Some(Value::Null) | None => 2,
        _ => 4,
    }
}

#[cfg(test)]
mod dataframe_dtype_tests {
    use super::*;

    #[test]
    fn hints_follow_emitted_integer_null_and_other_types() {
        for value in [
            Value::Int64(i64::MAX),
            Value::UniqueId(7),
            Value::NodeRef(9),
        ] {
            let integer = dataframe_value_kind(Some(&value));
            assert_eq!(integer, 1);
            assert_eq!(dataframe_integer_dtype(integer), None);
            assert_eq!(
                dataframe_integer_dtype(integer | dataframe_value_kind(None)),
                Some("Int64")
            );
            assert_eq!(
                dataframe_integer_dtype(integer | dataframe_value_kind(Some(&Value::Null))),
                Some("Int64")
            );
            for other in [
                Value::Boolean(true),
                Value::Float64(1.5),
                Value::String("x".into()),
            ] {
                let mixed = integer | dataframe_value_kind(Some(&other));
                assert_eq!(dataframe_integer_dtype(mixed), Some("object"));
                assert_eq!(dataframe_integer_dtype(mixed | 2), Some("object"));
            }
        }
        for kind in [0, 2, 4, 6] {
            assert_eq!(dataframe_integer_dtype(kind), None);
        }
    }
}
