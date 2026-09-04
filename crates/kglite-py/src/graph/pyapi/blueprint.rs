//! PyO3 entry for the Rust blueprint loader.
//!
//! Thin wrapper: returns the populated `KnowledgeGraph` plus the output
//! path declared in the blueprint (if any). Save and `lock_schema` are
//! invoked from the Python shim using the existing `KnowledgeGraph`
//! methods — avoids duplicating the v3 save pipeline here.

use crate::datatypes::on_invalid::OnInvalid;
use crate::datatypes::py_in;
use crate::graph::KnowledgeGraph;
use kglite_core::api::blueprint;
use kglite_core::datatypes::values::{ColumnType, DataFrame};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;
use std::sync::Arc;

/// Parse a JSON blueprint and build a `KnowledgeGraph` from its inputs.
///
/// Returns `(graph, output_path_or_none)` — the Python shim saves and applies
/// `lock_schema` on top. Exposed as `kglite.kglite.from_blueprint_rust` to
/// avoid colliding with the user-facing `kglite.from_blueprint` wrapper.
#[pyfunction]
#[pyo3(signature = (blueprint_path, *, verbose=false, storage=None, path=None, frames=None))]
pub fn from_blueprint_rust(
    py: Python<'_>,
    blueprint_path: String,
    verbose: bool,
    storage: Option<&str>,
    path: Option<&str>,
    frames: Option<&Bound<'_, PyDict>>,
) -> PyResult<(KnowledgeGraph, Option<String>)> {
    let bp_path = Path::new(&blueprint_path).to_path_buf();
    if !bp_path.exists() {
        return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "Blueprint file not found: {}",
            bp_path.display()
        )));
    }

    // Parsed under the GIL, before the build is detached: converting a pandas
    // frame needs both the GIL and the blueprint's declared types, and the
    // core decides which frames are missing or unexpected — this side only
    // marshals what it was handed.
    let parsed = blueprint::load_blueprint_file(&bp_path)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let inputs = convert_frames(&parsed, frames)?;

    let bp_dir = bp_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();

    let (kg, report, output_path) = py
        .detach(
            || -> Result<(KnowledgeGraph, blueprint::BuildReport, Option<std::path::PathBuf>), String> {
                // Construct the backing DirGraph with the requested storage
                // mode via the shared core builder (one mode vocabulary across
                // wheel / servers / C ABI). Empty string is treated as default.
                let mode = match storage {
                    None | Some("") => kglite_core::api::storage::StorageMode::Memory,
                    Some(s) => kglite_core::api::storage::StorageMode::parse(s)?,
                };
                let mut graph =
                    kglite_core::api::storage::new_dir_graph_in_mode(mode, path.map(Path::new))?;

                let (report, output_path) =
                    blueprint::from_blueprint(&mut graph, parsed, &bp_dir, inputs)?;

                let kg = KnowledgeGraph {
                    inner: Arc::new(graph),
                    cursor: crate::graph::CursorState::new(),
                    embedder: None,
                    default_timeout_ms: None,
                    default_max_work_units: None,
                    default_row_limit: None,
                    lifecycle: crate::graph::GraphLifecycle::detached(),
                };
                Ok((kg, report, output_path))
            },
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    print!("{}", report.render_text(verbose));
    if !report.warnings.is_empty() {
        if verbose {
            for w in &report.warnings {
                eprintln!("warning: {}", w);
            }
        } else {
            // Compact summary so callers running silent still know data
            // quality issues exist.
            eprintln!(
                "{} blueprint warning(s) — pass verbose=True for details.",
                report.warnings.len()
            );
        }
    }
    for e in &report.errors {
        eprintln!("error: {}", e);
    }

    Ok((kg, output_path.map(|p| p.to_string_lossy().into_owned())))
}

/// Convert every frame the caller passed into a core `DataFrame`, typed by
/// what the blueprint declares for the input of that name.
///
/// A frame is coerced to the blueprint's declared property types — that is the
/// contract, and it is what makes a `frames=` build produce the same graph as a
/// CSV of the same data. Where the blueprint declares nothing, the frame's own
/// dtype is kept and reported to the loader as a known column type.
///
/// Names are not checked here: a declared-but-missing or passed-but-undeclared
/// frame is the core's error to phrase, and it phrases it once for every
/// binding.
fn convert_frames(
    parsed: &blueprint::Blueprint,
    frames: Option<&Bound<'_, PyDict>>,
) -> PyResult<blueprint::BuildInputs> {
    let mut inputs = blueprint::BuildInputs::default();
    let Some(frames) = frames else {
        return Ok(inputs);
    };
    for (key, value) in frames.iter() {
        let name: String = key.extract().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "frames= keys must be strings naming a 'files' entry",
            )
        })?;
        inputs
            .frames
            .insert(name.clone(), to_dataframe(parsed, &name, &value)?);
    }
    Ok(inputs)
}

fn to_dataframe(
    parsed: &blueprint::Blueprint,
    name: &str,
    df: &Bound<'_, PyAny>,
) -> PyResult<DataFrame> {
    let columns: Vec<String> = df
        .getattr("columns")
        .and_then(|c| c.extract())
        .map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "frames['{name}'] is not a DataFrame with string column names — pass a pandas \
                 DataFrame (or anything with .to_pandas()) whose columns are strings"
            ))
        })?;

    let declared = blueprint::declared_column_types(parsed, name)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let types = PyDict::new(df.py());
    for (column, ct) in &declared {
        if let Some(keyword) = pandas_type_keyword(ct) {
            types.set_item(column, keyword)?;
        }
    }

    py_in::pandas_to_dataframe_with_options(
        df,
        // No id column is special here: the loader keys nodes off the string
        // table like it does for a CSV, so nothing needs a `UniqueId` column.
        &[],
        &columns,
        Some(&types),
        // Off, matching `add_nodes`. A pandas integer column carrying nulls is
        // a float64 column, and downcasting it here would also turn a genuine
        // float column of whole numbers into ints — declare `"int"` in the
        // blueprint to get an integer property back.
        false,
        // A frame is the caller's own data, and a mixed-dtype column silently
        // stringified is exactly the surprise `on_invalid` exists to refuse.
        OnInvalid::Error,
    )
}

/// The type name `py_in` reads for a blueprint type. The blueprint's keyword
/// vocabulary is wider (`str`, `integer`, `validFrom`, …) and `py_in`'s is a
/// different set, so the `ColumnType` in between is what the two agree on.
fn pandas_type_keyword(ct: &ColumnType) -> Option<&'static str> {
    match ct {
        ColumnType::String => Some("string"),
        ColumnType::Int64 => Some("int"),
        ColumnType::Float64 => Some("float"),
        ColumnType::Boolean => Some("bool"),
        ColumnType::DateTime => Some("date"),
        ColumnType::List => Some("list"),
        // `map_blueprint_type` yields none of these, so the arm is unreachable
        // from a blueprint; leaving the column untyped is the honest fallback.
        ColumnType::UniqueId | ColumnType::Timestamp | ColumnType::Map => None,
    }
}

/// Build a `KnowledgeGraph` from an inline JSON records spec (nodes +
/// connections), no CSV files on disk. JSON-native sibling to
/// `from_blueprint_rust`. Returns the populated graph; the Python shim handles
/// optional save / lock_schema. Exposed as `kglite.kglite.from_records_rust`.
#[pyfunction]
#[pyo3(signature = (records_json, *, storage=None, path=None, on_missing_endpoint=None))]
pub fn from_records_rust(
    py: Python<'_>,
    records_json: String,
    storage: Option<&str>,
    path: Option<&str>,
    on_missing_endpoint: Option<&str>,
) -> PyResult<KnowledgeGraph> {
    let mut spec: serde_json::Value = serde_json::from_str(&records_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid JSON: {}", e)))?;
    let spec_obj = spec.as_object_mut().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("from_records: top-level JSON must be an object")
    })?;
    // Only when the caller passed one. The spec's own `on_missing_endpoint`
    // is a documented top-level key, and inserting the argument's default
    // unconditionally overwrote it — a spec asking for `drop` vivified in
    // silence.
    if let Some(policy) = on_missing_endpoint {
        spec_obj.insert(
            "on_missing_endpoint".to_string(),
            serde_json::Value::String(policy.to_string()),
        );
    }

    let kg = py
        .detach(|| -> Result<KnowledgeGraph, String> {
            let mode = match storage {
                None | Some("") => kglite_core::api::storage::StorageMode::Memory,
                Some(s) => kglite_core::api::storage::StorageMode::parse(s)?,
            };
            let mut graph =
                kglite_core::api::storage::new_dir_graph_in_mode(mode, path.map(Path::new))?;

            blueprint::from_records(&mut graph, &spec)?;

            Ok(KnowledgeGraph {
                inner: Arc::new(graph),
                cursor: crate::graph::CursorState::new(),
                embedder: None,
                default_timeout_ms: None,
                default_max_work_units: None,
                default_row_limit: None,
                lifecycle: crate::graph::GraphLifecycle::detached(),
            })
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok(kg)
}
