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

    let numpy = ColumnBuffer::numpy(py);

    for (i, key) in col_keys.iter().enumerate() {
        let buffer = numpy.as_ref().and_then(|_| ColumnBuffer::of(rows, i));
        if let (Some(numpy), Some(buffer)) = (numpy.as_ref(), buffer) {
            // Every cell of this column is the same unboxed numeric kind, so
            // the exact dtype pandas would infer from the boxed list is known
            // here and the whole column travels as raw bytes: no per-cell
            // Python object, and no per-cell dtype inference on arrival.
            dict.set_item(
                key,
                crate::datatypes::pandas_out::numeric_column(
                    py,
                    numpy,
                    &buffer.bytes(rows, i),
                    buffer.numpy_dtype(),
                )?,
            )?;
            native_dtypes.set_item(key, None::<&str>)?;
            col_order.append(key)?;
            continue;
        }
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

/// The fixed-width numpy layout a whole column can travel in.
///
/// Only the three kinds whose boxed-list form pandas infers to exactly this
/// dtype qualify, and only when **every** cell is that kind: one NULL, one
/// short row, or one cell of another type and the column stays on the boxed
/// path, where `dataframe_integer_dtype` decides between `Int64`, `object` and
/// pandas' own inference. So the frame is value- and dtype-identical either
/// way — this is a transport change, not a policy one.
#[derive(Clone, Copy)]
enum ColumnBuffer {
    Int64,
    Float64,
    Bool,
}

impl ColumnBuffer {
    /// numpy, or `None` when it cannot be imported — then every column takes
    /// the boxed path. pandas requires numpy, so this is unreachable in
    /// practice; it costs one cached import per frame to not depend on that.
    fn numpy(py: Python<'_>) -> Option<Bound<'_, PyModule>> {
        py.import("numpy").ok()
    }

    /// The layout column `i` shares, or `None` for a mixed, nullable, ragged
    /// or empty column. Empty is excluded deliberately: a zero-length typed
    /// array would fix a dtype the boxed path leaves to pandas.
    fn of(rows: &[Vec<Value>], i: usize) -> Option<Self> {
        let mut kind: Option<Self> = None;
        for row in rows {
            let cell = match row.get(i)? {
                Value::Int64(_) | Value::UniqueId(_) | Value::NodeRef(_) => Self::Int64,
                Value::Float64(_) => Self::Float64,
                Value::Boolean(_) => Self::Bool,
                _ => return None,
            };
            match kind {
                None => kind = Some(cell),
                Some(seen) if seen.numpy_dtype() == cell.numpy_dtype() => {}
                Some(_) => return None,
            }
        }
        kind
    }

    /// Native-endian, matching the dtype name below — the bytes never leave
    /// this machine, so there is nothing to byte-swap for.
    fn bytes(self, rows: &[Vec<Value>], i: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(rows.len() * self.width());
        for row in rows {
            match (self, row.get(i)) {
                (Self::Int64, Some(Value::Int64(v))) => out.extend_from_slice(&v.to_ne_bytes()),
                (Self::Int64, Some(Value::UniqueId(v) | Value::NodeRef(v))) => {
                    out.extend_from_slice(&i64::from(*v).to_ne_bytes());
                }
                (Self::Float64, Some(Value::Float64(v))) => out.extend_from_slice(&v.to_ne_bytes()),
                (Self::Bool, Some(Value::Boolean(v))) => out.push(u8::from(*v)),
                // `of` proved every cell above; a mismatch here would be a
                // classifier/writer disagreement, not user data.
                _ => unreachable!("column layout was proven by ColumnBuffer::of"),
            }
        }
        out
    }

    fn width(self) -> usize {
        match self {
            Self::Int64 | Self::Float64 => 8,
            Self::Bool => 1,
        }
    }

    fn numpy_dtype(self) -> &'static str {
        match self {
            Self::Int64 => "int64",
            Self::Float64 => "float64",
            Self::Bool => "bool",
        }
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
