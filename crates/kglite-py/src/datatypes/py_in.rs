// src/datatypes/py_in.rs
use super::type_conversions::{to_bool, to_datetime, to_f64, to_i64, to_u32};
use super::values::{ColumnData, ColumnType, DataFrame, FilterCondition, Value};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::Bound;
use std::collections::HashMap;

pub fn pydict_to_filter_conditions(
    dict: &Bound<'_, PyDict>,
) -> PyResult<HashMap<String, FilterCondition>> {
    dict.iter()
        .map(|(k, v)| {
            let key = k.extract::<String>()?;
            let condition = match v.cast::<PyDict>() {
                // Handle operator-based condition
                Ok(op_dict) => {
                    let (op, val) = op_dict.iter().next().ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>("Empty operator dictionary")
                    })?;

                    let op_str = op.extract::<String>()?;
                    parse_operator_condition(&op_str, &val)?
                }
                // Handle direct value match (equivalent to ==)
                Err(_) => FilterCondition::Equals(py_value_to_value(&v)?),
            };

            Ok((key, condition))
        })
        .collect()
}

fn parse_operator_condition(op: &str, val: &Bound<'_, PyAny>) -> PyResult<FilterCondition> {
    match op {
        "==" => {
            if val.cast::<PyList>().is_ok() {
                parse_in_condition(val)
            } else {
                Ok(FilterCondition::Equals(py_value_to_value(val)?))
            }
        }
        "!=" => Ok(FilterCondition::NotEquals(py_value_to_value(val)?)),
        ">" => Ok(FilterCondition::GreaterThan(py_value_to_value(val)?)),
        ">=" => Ok(FilterCondition::GreaterThanEquals(py_value_to_value(val)?)),
        "<" => Ok(FilterCondition::LessThan(py_value_to_value(val)?)),
        "<=" => Ok(FilterCondition::LessThanEquals(py_value_to_value(val)?)),
        "in" => parse_in_condition(val),
        "between" => parse_between_condition(val),
        "is_null" => Ok(FilterCondition::IsNull),
        "is_not_null" => Ok(FilterCondition::IsNotNull),
        "contains" => Ok(FilterCondition::Contains(py_value_to_value(val)?)),
        "starts_with" => Ok(FilterCondition::StartsWith(py_value_to_value(val)?)),
        "ends_with" => Ok(FilterCondition::EndsWith(py_value_to_value(val)?)),
        "regex" | "=~" => {
            let pattern = val.extract::<String>().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "'regex' operator requires a string pattern",
                )
            })?;
            // Validate the regex at parse time
            regex::Regex::new(&pattern).map_err(|e| -> PyErr {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(format!(
                    "Invalid regex pattern '{}': {}",
                    pattern, e
                )))
            })?;
            Ok(FilterCondition::Regex(pattern))
        }
        // Negated operators: not_contains, not_starts_with, not_ends_with, not_in, not_regex
        "not_contains" => Ok(FilterCondition::Not(Box::new(FilterCondition::Contains(
            py_value_to_value(val)?,
        )))),
        "not_starts_with" => Ok(FilterCondition::Not(Box::new(FilterCondition::StartsWith(
            py_value_to_value(val)?,
        )))),
        "not_ends_with" => Ok(FilterCondition::Not(Box::new(FilterCondition::EndsWith(
            py_value_to_value(val)?,
        )))),
        "not_in" => Ok(FilterCondition::Not(Box::new(parse_in_condition(val)?))),
        "not_regex" => {
            let pattern = val.extract::<String>().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "'not_regex' operator requires a string pattern",
                )
            })?;
            regex::Regex::new(&pattern).map_err(|e| -> PyErr {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(format!(
                    "Invalid regex pattern '{}': {}",
                    pattern, e
                )))
            })?;
            Ok(FilterCondition::Not(Box::new(FilterCondition::Regex(
                pattern,
            ))))
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported operator: {}. Supported: ==, !=, >, >=, <, <=, in, between, \
             is_null, is_not_null, contains, starts_with, ends_with, regex, \
             not_contains, not_starts_with, not_ends_with, not_in, not_regex",
            op
        ))),
    }
}

fn parse_between_condition(val: &Bound<'_, PyAny>) -> PyResult<FilterCondition> {
    match val.cast::<PyList>() {
        Ok(list) => {
            let items: Vec<_> = list.iter().collect();
            if items.len() != 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "'between' operator requires a list of exactly 2 values [min, max]",
                ));
            }
            let min = py_value_to_value(&items[0])?;
            let max = py_value_to_value(&items[1])?;
            Ok(FilterCondition::Between(min, max))
        }
        Err(_) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "'between' operator requires a list of [min, max] values",
        )),
    }
}

fn parse_in_condition(val: &Bound<'_, PyAny>) -> PyResult<FilterCondition> {
    match val.cast::<PyList>() {
        Ok(list) => {
            let values: PyResult<Vec<Value>> =
                list.iter().map(|item| py_value_to_value(&item)).collect();
            Ok(FilterCondition::In(values?))
        }
        Err(_) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "'in' operator requires a list value",
        )),
    }
}

fn convert_pandas_series(series: &Bound<'_, PyAny>, col_type: ColumnType) -> PyResult<ColumnData> {
    let length = series.len()?;

    // Get the null mask from pandas — this correctly handles None, np.nan, pd.NA, pd.NaT
    // regardless of pandas version or dtype backend (object, string[python], string[pyarrow], etc.)
    let null_mask: Vec<bool> = series
        .call_method0("isna")?
        .call_method0("tolist")?
        .extract()?;

    // Convert series to Python list once — PyList.get_item() is O(1) C-level array access,
    // whereas Series.get_item() goes through pandas' label-aware indexing with dtype dispatch.
    // For 15M cells this saves ~10-20s of Python↔Rust FFI overhead.
    let py_list = series.call_method0("tolist")?;

    match col_type {
        ColumnType::Float64 => {
            // Fast path: try batch extraction (works when column has no mixed types)
            match py_list.extract::<Vec<Option<f64>>>() {
                Ok(mut values) => {
                    // Apply null mask — pandas NaN extracts as Some(NaN), not None
                    for (i, &is_null) in null_mask.iter().enumerate() {
                        if is_null {
                            values[i] = None;
                        }
                    }
                    Ok(ColumnData::Float64(values))
                }
                Err(_) => {
                    // Fallback: per-element with null mask (mixed types or special values)
                    let py_list = py_list.cast::<PyList>()?;
                    let mut vec = Vec::with_capacity(length);
                    for (i, &is_null) in null_mask.iter().enumerate() {
                        if is_null {
                            vec.push(None);
                        } else {
                            let item = py_list.get_item(i)?;
                            vec.push(to_f64(&item));
                        }
                    }
                    Ok(ColumnData::Float64(vec))
                }
            }
        }
        ColumnType::Boolean => {
            // Fast path: try batch extraction
            match py_list.extract::<Vec<Option<bool>>>() {
                Ok(mut values) => {
                    // Apply null mask — pd.NA may not extract as None
                    for (i, &is_null) in null_mask.iter().enumerate() {
                        if is_null {
                            values[i] = None;
                        }
                    }
                    Ok(ColumnData::Boolean(values))
                }
                Err(_) => {
                    let py_list = py_list.cast::<PyList>()?;
                    let mut vec = Vec::with_capacity(length);
                    for (i, &is_null) in null_mask.iter().enumerate() {
                        if is_null {
                            vec.push(None);
                        } else {
                            let item = py_list.get_item(i)?;
                            vec.push(to_bool(&item));
                        }
                    }
                    Ok(ColumnData::Boolean(vec))
                }
            }
        }
        ColumnType::String => {
            // Fast path: try batch extraction
            match py_list.extract::<Vec<Option<String>>>() {
                Ok(mut values) => {
                    // Apply null mask for safety
                    for (i, &is_null) in null_mask.iter().enumerate() {
                        if is_null {
                            values[i] = None;
                        }
                    }
                    Ok(ColumnData::String(values))
                }
                Err(_) => {
                    let py_list = py_list.cast::<PyList>()?;
                    let mut vec = Vec::with_capacity(length);
                    for (i, &is_null) in null_mask.iter().enumerate() {
                        if is_null {
                            vec.push(None);
                        } else {
                            let item = py_list.get_item(i)?;
                            vec.push(item.str().ok().map(|s| s.to_string()));
                        }
                    }
                    Ok(ColumnData::String(vec))
                }
            }
        }
        ColumnType::Int64 => {
            // Int64 columns may contain numpy int64 which doesn't batch-extract to i64,
            // so use per-element with PyList (still much faster than Series indexing)
            let py_list = py_list.cast::<PyList>()?;
            let mut vec = Vec::with_capacity(length);
            for (i, &is_null) in null_mask.iter().enumerate() {
                if is_null {
                    vec.push(None);
                } else {
                    let item = py_list.get_item(i)?;
                    vec.push(to_i64(&item));
                }
            }
            Ok(ColumnData::Int64(vec))
        }
        ColumnType::UniqueId => {
            let py_list = py_list.cast::<PyList>()?;
            let mut vec = Vec::with_capacity(length);
            for (i, &is_null) in null_mask.iter().enumerate() {
                if is_null {
                    vec.push(None);
                } else {
                    let item = py_list.get_item(i)?;
                    vec.push(to_u32(&item));
                }
            }
            Ok(ColumnData::UniqueId(vec))
        }
        ColumnType::DateTime => {
            // DateTime needs custom parsing — use PyList for O(1) access
            let py_list = py_list.cast::<PyList>()?;
            let mut vec = Vec::with_capacity(length);
            for (i, &is_null) in null_mask.iter().enumerate() {
                if is_null {
                    vec.push(None);
                } else {
                    let item = py_list.get_item(i)?;
                    vec.push(to_datetime(&item));
                }
            }
            Ok(ColumnData::DateTime(vec))
        }
    }
}

pub fn pandas_to_dataframe(
    df: &Bound<'_, PyAny>,
    unique_id_fields: &[String],
    column_names: &[String],
    column_types: Option<&Bound<'_, PyDict>>,
) -> PyResult<DataFrame> {
    pandas_to_dataframe_with_options(df, unique_id_fields, column_names, column_types, false)
}

/// Same as [`pandas_to_dataframe`] but with extra knobs.
///
/// `nullable_int_downcast`: when true, every Float64 column whose
/// non-null values are all integer-valued (`v.fract() == 0.0` and within
/// `i64` range) is downcast to Int64. Pandas auto-promotes nullable
/// integer columns to float64 when nulls are present, which then
/// surfaces in queries as `"2.0"` instead of `2`. Off by default — opt
/// in via `add_nodes(nullable_int_downcast=True)`.
pub fn pandas_to_dataframe_with_options(
    df: &Bound<'_, PyAny>,
    unique_id_fields: &[String],
    column_names: &[String],
    column_types: Option<&Bound<'_, PyDict>>,
    nullable_int_downcast: bool,
) -> PyResult<DataFrame> {
    // Reset index to ensure contiguous positional access.
    // Handles filtered/deduped DataFrames with non-contiguous indexes.
    let kwargs = pyo3::types::PyDict::new(df.py());
    kwargs.set_item("drop", true)?;
    let df = df.call_method("reset_index", (), Some(&kwargs))?;

    let df_columns = df.getattr("columns")?;
    let all_column_names: Vec<String> = df_columns.extract()?;

    let mut df_out = DataFrame::new(Vec::new());

    Python::attach(|_py| {
        for col_name in column_names {
            // Skip if column doesn't exist in the dataframe
            if !all_column_names.contains(col_name) {
                continue;
            }

            // Use bracket notation df[col_name] instead of df.col_name to avoid
            // conflicts with DataFrame attributes like 'shape', 'index', 'columns', etc.
            let series = df.get_item(col_name).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Failed to access column '{}': {}. If using a reserved name like 'shape', 'index', or 'columns', please rename your column.", col_name, e)
                )
            })?;

            // Determine column type - check custom type mapping first
            let col_type = if let Some(type_dict) = column_types {
                // Get item returns Result<Option<...>, ...> so we need to handle both
                let maybe_type_value = type_dict.get_item(col_name)?;

                if let Some(type_value) = maybe_type_value {
                    // Extract type string from the dictionary
                    let type_str = type_value.extract::<String>()?;

                    // Parse the requested type
                    match type_str.to_lowercase().as_str() {
                        "uniqueid" | "unique_id" | "id" => ColumnType::UniqueId,
                        "int" | "int64" | "integer" => ColumnType::Int64,
                        "float" | "float64" | "double" => ColumnType::Float64,
                        "str" | "string" | "text" => ColumnType::String,
                        "bool" | "boolean" => ColumnType::Boolean,
                        "date" | "datetime" | "timestamp" => ColumnType::DateTime,
                        _ => {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                                "Unsupported column type '{}' specified for column '{}'",
                                type_str, col_name
                            )))
                        }
                    }
                } else if unique_id_fields.contains(col_name) {
                    // Auto-detect: string/object columns keep String type rather than
                    // forcing UniqueId (which silently drops non-numeric string IDs)
                    let detected = determine_column_type(&series, col_name)?;
                    match detected {
                        ColumnType::String => ColumnType::String,
                        _ => ColumnType::UniqueId,
                    }
                } else {
                    // Fall back to auto-detection if not in custom mapping
                    determine_column_type(&series, col_name)?
                }
            } else if unique_id_fields.contains(col_name) {
                // Auto-detect: string/object columns keep String type
                let detected = determine_column_type(&series, col_name)?;
                match detected {
                    ColumnType::String => ColumnType::String,
                    _ => ColumnType::UniqueId,
                }
            } else {
                // No custom mapping provided, use auto-detection
                determine_column_type(&series, col_name)?
            };

            let data = convert_pandas_series(&series, col_type.clone())?;
            // Optional: downcast Float64 columns whose non-null values are
            // all integer-valued. Pandas turns nullable int columns into
            // float64 when nulls are present; this restores the integer
            // shape so downstream queries don't see "2.0" for what was
            // originally `2`.
            let (final_type, final_data) = if nullable_int_downcast {
                try_downcast_float_to_int(col_type, data)
            } else {
                (col_type, data)
            };
            df_out
                .add_column(col_name.clone(), final_type, final_data)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        }
        Ok(())
    })?;

    Ok(df_out)
}

/// If `data` is `ColumnData::Float64` with all non-null values
/// integer-valued and within `i64` range, return an `Int64` version.
/// Otherwise pass through unchanged.
fn try_downcast_float_to_int(col_type: ColumnType, data: ColumnData) -> (ColumnType, ColumnData) {
    if !matches!(col_type, ColumnType::Float64) {
        return (col_type, data);
    }
    let ColumnData::Float64(values) = data else {
        return (col_type, data);
    };
    // Empty column: no signal — keep as float to be safe.
    if values.iter().all(|v| v.is_none()) {
        return (ColumnType::Float64, ColumnData::Float64(values));
    }
    let i64_min = i64::MIN as f64;
    let i64_max = i64::MAX as f64;
    let convertible = values.iter().all(|opt| match opt {
        None => true,
        Some(v) => v.is_finite() && v.fract() == 0.0 && *v >= i64_min && *v <= i64_max,
    });
    if !convertible {
        return (ColumnType::Float64, ColumnData::Float64(values));
    }
    let int_values: Vec<Option<i64>> = values.iter().map(|opt| opt.map(|v| v as i64)).collect();
    (ColumnType::Int64, ColumnData::Int64(int_values))
}

// Helper function to determine column type from pandas series
fn determine_column_type(series: &Bound<'_, PyAny>, col_name: &str) -> PyResult<ColumnType> {
    let dtype = series.getattr("dtype").map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Could not determine type for column '{}'. The data may not be a valid pandas Series. \
                If your column is named 'shape', 'index', 'columns', 'values', or 'dtype', \
                please rename it as these conflict with pandas DataFrame attributes.",
            col_name
        ))
    })?;
    let type_str = dtype.str()?.to_string();

    match type_str.as_str() {
        "int64" | "int32" | "int16" | "int8" | "Int64" | "Int32" | "Int16" | "Int8" => {
            Ok(ColumnType::Int64)
        }
        "float64" | "float32" | "Float64" | "Float32" => Ok(ColumnType::Float64),
        "bool" | "boolean" => Ok(ColumnType::Boolean),
        s if s.starts_with("datetime64") => Ok(ColumnType::DateTime),
        "object" => {
            // Object dtype may contain booleans mixed with None (common in code_tree
            // metadata like is_test, is_abstract).  Use pandas' own type inference
            // which handles skipna correctly.
            let pd = series.py().import("pandas")?.getattr("api")?.getattr("types")?;
            let kwargs = pyo3::types::PyDict::new(series.py());
            kwargs.set_item("skipna", true)?;
            let inferred: String = pd
                .call_method("infer_dtype", (series,), Some(&kwargs))?
                .extract()?;
            if inferred == "boolean" {
                return Ok(ColumnType::Boolean);
            }
            Ok(ColumnType::String)
        }
        "string" | "str" | "string[python]" | "string[pyarrow]" => {
            Ok(ColumnType::String)
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unsupported column type '{}' for column '{}'. Supported types: int, float, bool, datetime, string/object.", type_str, col_name)
        )),
    }
}

pub fn ensure_columns(
    df_columns: &[String],
    default_columns: &[&str],
    columns_to_include: Option<&Bound<'_, PyList>>,
    columns_to_skip: Option<&Bound<'_, PyList>>,
    enforce_columns: Option<bool>,
) -> PyResult<Vec<String>> {
    // Process columns_to_skip if present
    let skip_cols = match columns_to_skip {
        Some(list) => list
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<PyResult<Vec<String>>>()?,
        None => Vec::new(),
    };

    // Process user-specified columns if present
    if let Some(include_list) = columns_to_include {
        let mut result: Vec<String> = include_list
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<PyResult<_>>()?;

        // Always add default columns
        for &field in default_columns {
            if !result.contains(&field.to_string()) && !skip_cols.contains(&field.to_string()) {
                result.push(field.to_string());
            }
        }

        // Filter out any skipped columns
        result.retain(|col| !skip_cols.contains(col));

        Ok(result)
    } else if enforce_columns.unwrap_or(false) {
        // If enforce_columns is true and no columns specified, use only default columns
        let mut result = Vec::new();

        for &field in default_columns {
            if !skip_cols.contains(&field.to_string()) {
                result.push(field.to_string());
            }
        }

        Ok(result)
    } else {
        // Use all dataframe columns except skipped ones
        let mut result = df_columns
            .iter()
            .filter(|col| !skip_cols.contains(col))
            .cloned()
            .collect::<Vec<String>>();

        // Ensure default columns are included
        for &field in default_columns {
            if !result.contains(&field.to_string()) && !skip_cols.contains(&field.to_string()) {
                result.push(field.to_string());
            }
        }

        Ok(result)
    }
}

pub fn parse_sort_fields(
    sort_spec: &Bound<'_, PyAny>,
    ascending: Option<bool>,
) -> PyResult<Vec<(String, bool)>> {
    // Handle single field with explicit direction
    if let Some(asc) = ascending {
        let field = sort_spec.extract::<String>()?;
        return Ok(vec![(field, asc)]);
    }

    // Handle single string
    if let Ok(field) = sort_spec.extract::<String>() {
        return Ok(vec![(field, true)]);
    }

    // Handle single tuple
    if let Ok((field, ascending)) = sort_spec.extract::<(String, bool)>() {
        return Ok(vec![(field, ascending)]);
    }

    // Handle list of sort specifications
    let sort_list = sort_spec.cast::<PyList>()?;
    if sort_list.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Sort specification cannot be empty",
        ));
    }

    // Process each item in the list
    sort_list
        .iter()
        .map(|item| {
            if let Ok(field) = item.extract::<String>() {
                Ok((field, true))
            } else if let Ok((field, ascending)) = item.extract::<(String, bool)>() {
                Ok((field, ascending))
            } else if let Ok(list) = item.cast::<PyList>() {
                if list.len() != 2 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "List specification must have exactly 2 elements",
                    ));
                }
                Ok((
                    list.get_item(0)?.extract::<String>()?,
                    list.get_item(1)?.extract::<bool>()?,
                ))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid sort specification",
                ))
            }
        })
        .collect()
}

pub fn py_value_to_value(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    if value.is_none() {
        return Ok(Value::Null);
    }

    // Check bool FIRST - Python bool is a subclass of int, so must check before numeric
    if value.is_instance_of::<pyo3::types::PyBool>() {
        if let Ok(b) = value.extract::<bool>() {
            return Ok(Value::Boolean(b));
        }
    }

    // Try numeric types before string (they fail fast without allocation)
    if let Ok(i) = value.extract::<i64>() {
        return Ok(Value::Int64(i));
    }
    if let Ok(f) = value.extract::<f64>() {
        return Ok(Value::Float64(f));
    }

    // String extraction is expensive (allocates), so try last among common types
    if let Ok(s) = value.extract::<String>() {
        return Ok(Value::String(s));
    }

    // Fallback for other types
    if let Ok(u) = value.extract::<u32>() {
        return Ok(Value::UniqueId(u));
    }

    // Handle lists (e.g. embedding vectors for Cypher params) — serialize as JSON string
    if let Ok(list) = value.cast::<pyo3::types::PyList>() {
        let items: Vec<String> = list
            .iter()
            .map(|item| {
                if let Ok(f) = item.extract::<f64>() {
                    format!("{}", f)
                } else if let Ok(s) = item.extract::<String>() {
                    format!("\"{}\"", s)
                } else if let Ok(i) = item.extract::<i64>() {
                    format!("{}", i)
                } else {
                    "null".to_string()
                }
            })
            .collect();
        return Ok(Value::String(format!("[{}]", items.join(", "))));
    }

    Ok(Value::Null)
}
