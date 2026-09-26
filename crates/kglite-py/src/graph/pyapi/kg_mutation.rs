//! KnowledgeGraph #[pymethods]: node + connection ingestion.
//!
//! PyO3 merges multiple `#[pymethods] impl` blocks at class-registration
//! time, so splitting them across files is purely structural — no runtime
//! impact.

use crate::datatypes::on_invalid::{self, OnInvalid};
use crate::datatypes::py_in;
use crate::datatypes::values::{DataFrame, Value};
use crate::graph::languages::cypher;
use crate::graph::{
    get_graph_mut, parse_inline_timeseries, parse_spatial_column_types,
    parse_temporal_column_types, resolve_noderefs, EmbeddingColumnData, InlineTimeseriesConfig,
    KnowledgeGraph, TimeSpec,
};
use kglite_core::api::mutation::{NodeOperationReport, OperationReport};
use kglite_core::api::DirGraph;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::Bound;
use std::collections::HashMap;
use std::sync::Arc;

// ─── add_nodes phase helpers ────────────────────────────────────────────────
//
// The `add_nodes` pymethod is a thin orchestrator: every phase it runs is a
// private free function below, called in order.

struct InlineConfig {
    ts_config: Option<InlineTimeseriesConfig>,
    embedding_columns: Vec<String>,
    column_list: Vec<String>,
}

/// Map a bulk-write failure onto the typed error it deserves.
///
/// The engine's write channel is `Result<_, String>`, but a constraint refusal
/// parks the structured violation on the graph alongside the message it
/// produced. Recovering it here is what makes a bulk write raise
/// `kglite.ConstraintViolationError` rather than the generic `ArgumentError`.
pub(super) fn bulk_write_err(graph: &mut DirGraph, message: String) -> pyo3::PyErr {
    let error = graph
        .take_constraint_error(&message)
        .unwrap_or(crate::error::KgError::Argument(message));
    crate::error_py::kg_to_pyerr(error)
}

/// Run a bulk write off-GIL and raise its typed error.
///
/// [`detach_mutation`]'s sibling for the paths that can raise a constraint
/// violation: the graph is threaded through so [`bulk_write_err`] can recover
/// the parked violation once the detached borrow has ended.
fn detach_bulk_write<T, F>(py: Python<'_>, graph: &mut DirGraph, f: F) -> PyResult<T>
where
    F: pyo3::marker::Ungil + Send + FnOnce(&mut DirGraph) -> Result<T, String>,
    T: pyo3::marker::Ungil + Send,
{
    let outcome = py.detach(|| f(graph));
    outcome.map_err(|message| bulk_write_err(graph, message))
}

/// Run a pure-Rust batch-mutation closure with the GIL released, mapping
/// the engine's `String` error to the typed `kglite.*` exception.
///
/// The apply phase of a bulk loader operates purely on already-converted Rust
/// data (`DataFrame`, `&mut DirGraph`), so holding the GIL through it starves
/// every other Python thread for the duration of a bulk insert. Detach only
/// spans like this one — anything touching `Bound`/`PyAny` must stay attached.
fn detach_mutation<T, F>(py: Python<'_>, f: F) -> PyResult<T>
where
    F: pyo3::marker::Ungil + Send + FnOnce() -> Result<T, String>,
    T: pyo3::marker::Ungil + Send,
{
    py.detach(f)
        .map_err(|e| crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e)))
}

fn validate_interner_names<'a>(
    graph: &DirGraph,
    names: impl IntoIterator<Item = &'a str>,
) -> PyResult<()> {
    graph
        .interner
        .validate_names(names)
        .map(|_| ())
        .map_err(|e| crate::error_py::kg_to_pyerr(crate::error::KgError::from(e)))
}

fn collect_bulk_connection_names(
    connections: &Bound<'_, PyList>,
    loaded_types: &std::collections::HashSet<String>,
    filter_to_loaded: bool,
) -> PyResult<Vec<String>> {
    let mut names = Vec::new();
    for item in connections.iter() {
        let spec = item.cast::<PyDict>()?;
        let required = |key: &str| -> PyResult<Bound<'_, PyAny>> {
            spec.get_item(key)?.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                    "Missing '{key}' in relationship spec"
                ))
            })
        };
        let source_type: String = required("source_type")?.extract()?;
        let target_type: String = required("target_type")?.extract()?;
        let connection_name: String = required("connection_name")?.extract()?;
        let data = required("data")?;
        if filter_to_loaded
            && (!loaded_types.contains(&source_type) || !loaded_types.contains(&target_type))
        {
            continue;
        }
        names.extend([source_type, target_type, connection_name]);
        names.extend(data.getattr("columns")?.extract::<Vec<String>>()?);
    }
    Ok(names)
}

/// Build the internal DataFrame for a connection ingest from a pandas
/// frame (the `data` mode shared by `add_relationships` and
/// `replace_relationships`). Returns the columnar DataFrame plus any
/// temporal-edge config auto-detected from `validFrom`/`validTo` column
/// types — the caller records that as the type's temporal config.
// Mirrors add_relationships' keyword surface one-to-one; a params struct would
// just re-spell the pyo3 signature.
#[allow(clippy::too_many_arguments)]
fn build_connection_df_from_pandas(
    data: &Bound<'_, PyAny>,
    source_id_field: &str,
    target_id_field: &str,
    source_title_field: Option<&str>,
    target_title_field: Option<&str>,
    columns: Option<&Bound<'_, PyList>>,
    skip_columns: Option<&Bound<'_, PyList>>,
    column_types: Option<&Bound<'_, PyDict>>,
    on_invalid: OnInvalid,
) -> PyResult<(DataFrame, Option<kglite_core::api::TemporalConfig>)> {
    let df_cols = data.getattr("columns")?;
    let all_columns: Vec<String> = df_cols.extract()?;

    let mut default_cols = vec![source_id_field, target_id_field];
    if let Some(src_title) = source_title_field {
        default_cols.push(src_title);
    }
    if let Some(tgt_title) = target_title_field {
        default_cols.push(tgt_title);
    }

    // Auto-include columns mentioned in column_types (e.g. temporal date columns)
    let mut column_type_cols: Vec<String> = Vec::new();
    if let Some(type_dict) = column_types {
        for key in type_dict.keys() {
            column_type_cols.push(key.extract()?);
        }
    }
    for col in &column_type_cols {
        default_cols.push(col.as_str());
    }

    // Match add_nodes: without an explicit whitelist, preserve every DataFrame
    // column except those named in skip_columns. Passing columns=[...] keeps
    // the explicit whitelist behavior.
    let column_list = py_in::ensure_columns(
        &all_columns,
        &default_cols,
        columns,
        skip_columns,
        Some(false),
    )?;

    // Parse temporal column_types (validFrom/validTo → datetime)
    let py = data.py();
    let (temporal_cfg, cleaned_types) = if let Some(type_dict) = column_types {
        let (tcfg, cleaned) = parse_temporal_column_types(py, type_dict)?;
        (tcfg, Some(cleaned))
    } else {
        (None, None)
    };
    let effective_types = cleaned_types.as_ref().map(|d| d.bind(py).clone());

    let df_result = py_in::pandas_to_dataframe_with_options(
        data,
        &[source_id_field.to_string(), target_id_field.to_string()],
        &column_list,
        effective_types.as_ref(),
        false,
        on_invalid,
    )?;
    Ok((df_result, temporal_cfg))
}

/// Run a read-only connection `query` against `graph` and turn its result rows
/// into the columnar frame the edge loader takes, stamping `extra_properties`
/// onto every row as constant columns.
///
/// The query-mode counterpart of [`build_connection_df_from_pandas`].
fn build_connection_df_from_query(
    graph: &Arc<DirGraph>,
    query_str: &str,
    extra_properties: Option<&Bound<'_, PyDict>>,
) -> PyResult<DataFrame> {
    let mut parsed = parse_read_only_connection_query(query_str)?;
    let empty_params = HashMap::new();
    // Run the same planner optimizations as g.cypher() — otherwise pushdowns
    // (including correlated-equality) don't fire here.
    cypher::optimize(&mut parsed, graph, &empty_params);
    let cypher_result = {
        let executor = cypher::CypherExecutor::with_params(graph, &empty_params, None);
        executor.execute(&parsed)
    }
    .map_err(|e| {
        crate::error_py::kg_to_pyerr(crate::error::KgError::CypherExecution {
            message: format!("Cypher execution error in relationship query: {}", e),
            position: None,
        })
    })?;

    // Resolve NodeRef values to actual IDs/titles
    let mut rows = cypher_result.rows;
    resolve_noderefs(&graph.graph, &mut rows);

    let mut df_result =
        DataFrame::from_cypher_rows(cypher_result.columns, rows).map_err(|e| -> PyErr {
            crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(format!(
                "Failed to convert query results to DataFrame: {}",
                e
            )))
        })?;

    if let Some(props_dict) = extra_properties {
        for (key, val) in props_dict.iter() {
            let col_name: String = key.extract()?;
            let value = py_in::py_value_to_value(&val)?;
            df_result
                .add_constant_column(col_name.clone(), value)
                .map_err(|e| -> PyErr {
                    crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(format!(
                        "Failed to add extra_property '{}': {}",
                        col_name, e
                    )))
                })?;
        }
    }
    Ok(df_result)
}

fn validate_connection_input_mode(
    has_data: bool,
    has_query: bool,
    has_extra_properties: bool,
    has_columns: bool,
    has_skip_columns: bool,
    has_column_types: bool,
) -> PyResult<()> {
    if has_data && has_query {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cannot specify both 'data' and 'query'. Use one or the other.",
        ));
    }
    if !has_data && !has_query {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Must specify either 'data' (DataFrame) or 'query' (Cypher query string).",
        ));
    }
    if has_data && has_extra_properties {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "extra_properties is only supported with query mode, not data mode.",
        ));
    }
    if has_query && has_columns {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "'columns' is only supported with data mode, not query mode.",
        ));
    }
    if has_query && has_skip_columns {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "'skip_columns' is only supported with data mode, not query mode.",
        ));
    }
    if has_query && has_column_types {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "'column_types' is only supported with data mode, not query mode.",
        ));
    }
    Ok(())
}

/// Parse the `query=` form of `add_relationships` / `replace_relationships`,
/// enforcing the two things this entry point requires of it: the query must be
/// read-only, and it must not reference parameters — this route accepts none.
///
/// Resolving dynamic labels against an empty parameter map is what turns
/// `MATCH (n:$label)` here into an actionable "Missing parameter" error rather
/// than a pattern that silently matches nothing.
fn parse_read_only_connection_query(query: &str) -> PyResult<cypher::CypherQuery> {
    let mut parsed = cypher::parse_cypher(query).map_err(|e| -> PyErr {
        crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(format!(
            "Cypher syntax error in query: {}",
            e
        )))
    })?;
    if cypher::is_mutation_query(&parsed) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "The 'query' parameter must be read-only (for example MATCH...RETURN); \
             mutation clauses are not allowed here.",
        ));
    }
    cypher::dynamic_labels::resolve(&mut parsed, &HashMap::new())
        .map_err(crate::error_py::kg_to_pyerr)?;
    Ok(parsed)
}

/// Shared body of `add_relationships` (replace=false) and
/// `replace_relationships` (replace=true). The two methods are identical
/// except for the core call: `replace` first prunes the existing edges
/// of `connection_type` from the source nodes present in the input, so
/// the result is "set this node's edges of this type to exactly this
/// list" rather than "add to them". Both modes (`data` DataFrame /
/// `query` Cypher) and every option behave the same across the two.
#[allow(clippy::too_many_arguments)]
fn write_connections(
    py: Python<'_>,
    kg: &mut KnowledgeGraph,
    replace: bool,
    data: Option<&Bound<'_, PyAny>>,
    connection_type: String,
    source_type: String,
    source_id_field: String,
    target_type: String,
    target_id_field: String,
    source_title_field: Option<String>,
    target_title_field: Option<String>,
    columns: Option<&Bound<'_, PyList>>,
    skip_columns: Option<&Bound<'_, PyList>>,
    conflict_handling: Option<String>,
    column_types: Option<&Bound<'_, PyDict>>,
    query: Option<String>,
    extra_properties: Option<&Bound<'_, PyDict>>,
    git_sha: Option<String>,
    modified_by: Option<String>,
    on_invalid: &str,
    convention: Option<&str>,
) -> PyResult<Py<PyAny>> {
    kg.check_durable_owner()?;
    let on_invalid = OnInvalid::parse(on_invalid)?;
    let convention = crate::graph::parse_interval_convention(convention)?;
    let has_data = data.as_ref().map(|d| !d.is_none()).unwrap_or(false);
    validate_connection_input_mode(
        has_data,
        query.is_some(),
        extra_properties.is_some(),
        columns.is_some(),
        skip_columns.is_some(),
        column_types.is_some(),
    )?;

    // ── Query path: run Cypher, convert to internal DataFrame ──
    if let Some(query_str) = query {
        refuse_convention_without_interval(convention, None)?;
        // Execute read-only against a cloned Arc, so no mutable borrow is held
        // while the query runs.
        let inner_clone = kg.inner.clone();
        let df_result = build_connection_df_from_query(&inner_clone, &query_str, extra_properties)?;

        refuse_unusable_rows_if_asked(
            on_invalid,
            "add_relationships",
            &df_result,
            &[&source_id_field, &target_id_field],
            None,
        )?;

        let mut names = vec![
            connection_type.as_str(),
            source_type.as_str(),
            target_type.as_str(),
        ];
        let frame_names = df_result.get_column_names();
        names.extend(frame_names.iter().map(String::as_str));
        validate_interner_names(&inner_clone, names)?;

        // Drop the Arc clone so Arc::make_mut in get_graph_mut doesn't
        // need to deep-copy the entire graph (refcount goes back to 1).
        drop(inner_clone);

        let graph = get_graph_mut(&mut kg.inner);

        // Everything past this point is pure Rust — run off-GIL.
        let result = detach_bulk_write(py, graph, |graph| {
            graph.with_write_provenance(git_sha.as_deref(), modified_by.as_deref(), |graph| {
                if replace {
                    kglite_core::api::mutation::replace_connections(
                        graph,
                        df_result,
                        connection_type.clone(),
                        source_type,
                        source_id_field,
                        target_type,
                        target_id_field,
                        source_title_field,
                        target_title_field,
                        conflict_handling,
                    )
                } else {
                    kglite_core::api::mutation::add_connections(
                        graph,
                        df_result,
                        connection_type.clone(),
                        source_type,
                        source_id_field,
                        target_type,
                        target_id_field,
                        source_title_field,
                        target_title_field,
                        conflict_handling,
                    )
                }
            })
        })?;

        kg.cursor.selection.clear();
        return finish_connection_write(kg, &result, &connection_type, on_invalid);
    }

    // ── Data path: pandas DataFrame logic ──
    let data = data.unwrap(); // Safe: validated above that has_data is true

    let (df_result, temporal_cfg) = build_connection_df_from_pandas(
        data,
        &source_id_field,
        &target_id_field,
        source_title_field.as_deref(),
        target_title_field.as_deref(),
        columns,
        skip_columns,
        column_types,
        on_invalid,
    )?;
    refuse_unusable_rows_if_asked(
        on_invalid,
        "add_relationships",
        &df_result,
        &[&source_id_field, &target_id_field],
        Some(data),
    )?;

    let mut names = vec![
        connection_type.as_str(),
        source_type.as_str(),
        target_type.as_str(),
    ];
    let frame_names = df_result.get_column_names();
    names.extend(frame_names.iter().map(String::as_str));
    validate_interner_names(&kg.inner, names)?;

    refuse_convention_without_interval(convention, temporal_cfg.as_ref())?;
    let graph = get_graph_mut(&mut kg.inner);
    let declaration = temporal_cfg
        .map(|cfg| {
            let target = kglite_core::api::temporal::TemporalTarget::Relationship {
                rel_type: connection_type.clone(),
                source_type: Some(source_type.clone()),
            };
            kglite_core::api::temporal::declare_from_column_types(
                graph,
                target,
                &cfg.valid_from,
                &cfg.valid_to,
                convention,
                &df_result,
            )
        })
        .transpose()
        .map_err(|e| crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e)))?;

    // The converted frame is pure Rust — apply the batch off-GIL.
    let result = detach_bulk_write(py, graph, |graph| {
        graph.with_write_provenance(git_sha.as_deref(), modified_by.as_deref(), |graph| {
            if replace {
                kglite_core::api::mutation::replace_connections(
                    graph,
                    df_result,
                    connection_type.clone(),
                    source_type,
                    source_id_field,
                    target_type,
                    target_id_field,
                    source_title_field,
                    target_title_field,
                    conflict_handling,
                )
            } else {
                kglite_core::api::mutation::add_connections(
                    graph,
                    df_result,
                    connection_type.clone(),
                    source_type,
                    source_id_field,
                    target_type,
                    target_id_field,
                    source_title_field,
                    target_title_field,
                    conflict_handling,
                )
            }
        })
    });
    let (result, declared) = finish_declaration(py, graph, declaration, result)?;

    kg.cursor.selection.clear();

    // Disk mode: build CSR from pending edges so queries work immediately
    let graph = get_graph_mut(&mut kg.inner);
    graph
        .ensure_disk_edges_built()
        .map_err(pyo3::exceptions::PyOSError::new_err)?;

    let report = finish_connection_write(kg, &result, &connection_type, on_invalid)?;
    declared.map(|()| report)
}

/// Shared tail of both `write_connections` paths: make the edges durable, file
/// the operation report, and marshal it for Python.
///
/// Shared by both paths so the WAL flush — the step most easily forgotten —
/// lives in one place rather than two that must stay in sync.
fn finish_connection_write(
    kg: &mut KnowledgeGraph,
    result: &kglite_core::api::mutation::ConnectionOperationReport,
    connection_type: &str,
    on_invalid: OnInvalid,
) -> PyResult<Py<PyAny>> {
    kg.commit_wal()?;
    kg.add_report(OperationReport::ConnectionOperation(result.clone()));
    KnowledgeGraph::connection_report_to_py(result, connection_type, on_invalid)
}

/// `convention=` qualifies the validity interval `validFrom`/`validTo` column
/// types declare; a call that declares none has nothing for it to apply to.
fn refuse_convention_without_interval(
    convention: Option<kglite_core::api::temporal::IntervalConvention>,
    interval: Option<&kglite_core::api::TemporalConfig>,
) -> PyResult<()> {
    if convention.is_some() && interval.is_none() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "convention applies to the validity interval that validFrom/validTo column_types \
             declare, and this call's column_types name none.",
        ));
    }
    Ok(())
}

/// Close a load's declaration: withdraw it when the write failed, else
/// validate it, raising its abutment advisory as a `UserWarning`. A refusal at
/// that point comes back beside the write's result rather than in place of
/// it, so the caller still commits what was written before raising it.
fn finish_declaration<T>(
    py: Python<'_>,
    graph: &mut DirGraph,
    declaration: Option<kglite_core::api::temporal::LoadDeclaration>,
    written: PyResult<T>,
) -> PyResult<(T, PyResult<()>)> {
    let Some(declaration) = declaration else {
        return written.map(|written| (written, Ok(())));
    };
    let written = match written {
        Ok(written) => written,
        Err(err) => {
            declaration.abandon(graph);
            return Err(err);
        }
    };
    let declared = declaration
        .finish(graph)
        .map_err(|e| crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e)))
        .and_then(|report| crate::graph::warn_declaration(py, &report));
    Ok((written, declared))
}

/// Refuse a loader call whose input carries rows no id can be read from, when
/// `on_invalid` asked for a refusal.
///
/// Runs on the *converted* frame and before any mutation, so the refusal is
/// total: an `on_invalid="error"` load either happens or leaves the graph
/// exactly as it was — unlike the tolerated modes, where the usable rows land
/// and the rest are counted. `raw_frame` is the caller's own DataFrame, read
/// only to quote the offending cell as they wrote it.
fn refuse_unusable_rows_if_asked(
    on_invalid: OnInvalid,
    loader: &str,
    df: &DataFrame,
    id_columns: &[&str],
    raw_frame: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    if !on_invalid.raises() {
        return Ok(());
    }
    let Some(bad) = on_invalid::scan_unusable_rows(df, id_columns) else {
        return Ok(());
    };
    let raw = on_invalid::raw_cell_repr(raw_frame, &bad.first_column, bad.first_row);
    Err(on_invalid::unusable_rows_err(
        loader,
        df.row_count(),
        &bad,
        &raw,
    ))
}

fn parse_inline_config<'py>(
    data: &Bound<'py, PyAny>,
    unique_id_field: &str,
    node_title_field: Option<&str>,
    columns: Option<&Bound<'py, PyList>>,
    skip_columns: Option<&Bound<'py, PyList>>,
    column_types: Option<&Bound<'py, PyDict>>,
    timeseries: Option<&Bound<'py, PyDict>>,
) -> PyResult<InlineConfig> {
    let ts_config = timeseries.map(parse_inline_timeseries).transpose()?;

    let mut embedding_columns: Vec<String> = Vec::new();
    if let Some(type_dict) = column_types {
        for (key, value) in type_dict.iter() {
            let col_name: String = key.extract()?;
            let type_str: String = value.extract()?;
            if type_str.to_lowercase() == "embedding" {
                embedding_columns.push(col_name);
            }
        }
    }

    let df_cols = data.getattr("columns")?;
    let all_columns: Vec<String> = df_cols.extract()?;

    let mut default_cols = vec![unique_id_field];
    if let Some(title_field) = node_title_field {
        default_cols.push(title_field);
    }

    let mut column_list = py_in::ensure_columns(
        &all_columns,
        &default_cols,
        columns,
        skip_columns,
        Some(false),
    )?;
    if !embedding_columns.is_empty() {
        column_list.retain(|c| !embedding_columns.contains(c));
    }
    if let Some(ref ts_cfg) = ts_config {
        let ts_cols = ts_cfg.all_columns();
        column_list.retain(|c| !ts_cols.contains(c));
    }

    Ok(InlineConfig {
        ts_config,
        embedding_columns,
        column_list,
    })
}

fn extract_embedding_pairs<'py>(
    data: &Bound<'py, PyAny>,
    unique_id_field: &str,
    embedding_columns: &[String],
) -> PyResult<EmbeddingColumnData> {
    if embedding_columns.is_empty() {
        return Ok(Vec::new());
    }
    // `.iloc`: positional. `Series[i]` reads the index *label* `i`, which is
    // absent from a frame whose index is not 0..n.
    let id_series = data.get_item(unique_id_field)?.getattr("iloc")?;
    let nrows: usize = data.getattr("shape")?.get_item(0)?.extract()?;
    let mut result = Vec::with_capacity(embedding_columns.len());

    for emb_col in embedding_columns {
        let series = data.get_item(emb_col)?.getattr("iloc")?;
        let mut pairs = Vec::with_capacity(nrows);
        let mut rows = py_in::F32Rows::default();
        for i in 0..nrows {
            let id_val = py_in::py_value_to_value(&id_series.get_item(i)?)?;
            let emb_val = rows.extract(&series.get_item(i)?)?;
            pairs.push((id_val, emb_val));
        }
        result.push((emb_col.clone(), pairs));
    }
    Ok(result)
}

struct ConvertedFrame {
    df: DataFrame,
    spatial_cfg: Option<kglite_core::api::SpatialConfig>,
    temporal_cfg: Option<kglite_core::api::TemporalConfig>,
}

fn convert_dataframe<'py>(
    data: &Bound<'py, PyAny>,
    unique_id_field: &str,
    column_list: &[String],
    ts_config: Option<&InlineTimeseriesConfig>,
    column_types: Option<&Bound<'py, PyDict>>,
    nullable_int_downcast: bool,
    on_invalid: OnInvalid,
) -> PyResult<ConvertedFrame> {
    let py = data.py();
    let (spatial_cfg, cleaned_after_spatial) = match column_types {
        Some(type_dict) => {
            let (cfg, cleaned) = parse_spatial_column_types(py, type_dict)?;
            (cfg, Some(cleaned))
        }
        None => (None, None),
    };

    let (temporal_cfg, cleaned_types) = match cleaned_after_spatial.as_ref() {
        Some(cleaned) => {
            let (tcfg, final_cleaned) = parse_temporal_column_types(py, cleaned.bind(py))?;
            (tcfg, Some(final_cleaned))
        }
        None => (None, cleaned_after_spatial),
    };

    let effective_types = cleaned_types.as_ref().map(|d| d.bind(py).clone());

    // When timeseries is present, deduplicate rows (keep first per unique_id) for static props.
    let data_for_nodes: std::borrow::Cow<'_, Bound<'py, PyAny>> = if ts_config.is_some() {
        let kwargs = PyDict::new(py);
        kwargs.set_item("subset", vec![unique_id_field])?;
        kwargs.set_item("keep", "first")?;
        let deduped = data.call_method("drop_duplicates", (), Some(&kwargs))?;
        std::borrow::Cow::Owned(deduped)
    } else {
        std::borrow::Cow::Borrowed(data)
    };

    let df = py_in::pandas_to_dataframe_with_options(
        &data_for_nodes,
        std::slice::from_ref(&unique_id_field.to_string()),
        column_list,
        effective_types.as_ref(),
        nullable_int_downcast,
        on_invalid,
    )?;

    // `data_for_nodes`, not `data`: when a timeseries config deduplicates the
    // input, that is the frame the converted rows line up with positionally.
    refuse_unusable_rows_if_asked(
        on_invalid,
        "add_nodes",
        &df,
        &[unique_id_field],
        Some(&data_for_nodes),
    )?;

    Ok(ConvertedFrame {
        df,
        spatial_cfg,
        temporal_cfg,
    })
}

struct NodeBatchInput {
    df: DataFrame,
    node_type: String,
    unique_id_field: String,
    node_title_field: Option<String>,
    conflict_handling: Option<String>,
}

/// Apply the converted node batch with the GIL released — the DataFrame is
/// already pure Rust at this point, so the insert runs off-GIL.
fn apply_node_batch(
    py: Python<'_>,
    graph: &mut DirGraph,
    input: NodeBatchInput,
    provenance: (Option<String>, Option<String>),
) -> PyResult<NodeOperationReport> {
    let (git_sha, modified_by) = provenance;
    let outcome = py.detach(|| {
        graph.with_write_provenance(git_sha.as_deref(), modified_by.as_deref(), |graph| {
            kglite_core::api::mutation::add_nodes(
                graph,
                input.df,
                input.node_type,
                input.unique_id_field,
                input.node_title_field,
                input.conflict_handling,
            )
        })
    });
    // The mutable borrow ends with the detach closure, so the structured
    // violation `add_nodes` parked is recoverable here: a constraint failure
    // raises `kglite.ConstraintViolationError`, anything else keeps the
    // existing `ArgumentError`.
    outcome.map_err(|message| bulk_write_err(graph, message))
}

fn store_extracted_embeddings(
    graph: &mut DirGraph,
    node_type: &str,
    embedding_data: &EmbeddingColumnData,
) {
    if embedding_data.is_empty() {
        return;
    }
    graph.build_id_index(node_type);
    for (emb_col, pairs) in embedding_data {
        let dimension = pairs.first().map(|(_, v)| v.len()).unwrap_or(0);
        if dimension == 0 {
            continue;
        }
        // add_nodes() accepts a column already named `<col>_emb` as well as the
        // bare source column, so the suffix is only minted when it is missing.
        let store_key = if emb_col.ends_with("_emb") {
            emb_col.clone()
        } else {
            kglite_core::api::embeddings::store_name(emb_col)
        };
        let mut store = kglite_core::api::storage::EmbeddingStore::new(dimension);
        store.data.reserve(pairs.len() * dimension);
        for (id_val, vec) in pairs {
            if vec.len() != dimension {
                continue;
            }
            if let Some(node_idx) = graph.lookup_by_id(node_type, id_val) {
                store.set_embedding(node_idx.index(), vec);
            }
        }
        if store.len() > 0 {
            graph
                .embeddings
                .insert((node_type.to_string(), store_key), store);
        }
    }
}

/// A uniform set of secondary labels for every row of an `add_nodes` batch,
/// with the batch's ids, read before any row is written so that applying them
/// afterwards cannot fail.
struct PreparedLabels {
    ids: Vec<Value>,
    labels: Vec<kglite_core::api::InternedKey>,
}

fn prepare_batch_labels<'py>(
    graph: &mut DirGraph,
    data: &Bound<'py, PyAny>,
    unique_id_field: &str,
    labels: &[String],
) -> PyResult<PreparedLabels> {
    let labels = labels
        .iter()
        .map(|l| {
            graph
                .interner
                .try_get_or_intern(l)
                .map_err(|e| crate::error_py::kg_to_pyerr(crate::error::KgError::from(e)))
        })
        .collect::<PyResult<_>>()?;
    // Positional, through `tolist()`: `Series[i]` looks up the index *label*,
    // which raised `KeyError` on a frame whose index is not 0..n.
    let ids = data
        .get_item(unique_id_field)?
        .call_method0("tolist")?
        .try_iter()?
        .map(|id| py_in::py_value_to_value(&id?))
        .collect::<PyResult<_>>()?;
    Ok(PreparedLabels { ids, labels })
}

/// Stamp every label on each batch node, found by id. Idempotent — a label
/// already present (or equal to the primary type) is a no-op.
fn apply_batch_labels(graph: &mut DirGraph, node_type: &str, prepared: &PreparedLabels) {
    graph.build_id_index(node_type);
    // One id-resolution pass, then one bulk stamp per label — the nested
    // per-row × per-label add_node_label loop this replaces was quadratic
    // in the batch size.
    let indices: Vec<_> = prepared
        .ids
        .iter()
        .filter_map(|id| graph.lookup_by_id(node_type, id))
        .collect();
    for &key in &prepared.labels {
        graph.add_node_labels_bulk(&indices, key);
    }
}

/// An `add_nodes` call's inline timeseries, read and validated in full before
/// any row is written, so a cell it cannot read refuses the whole call instead
/// of raising after the nodes have landed.
struct PreparedTimeseries {
    cfg: InlineTimeseriesConfig,
    resolution: String,
    time_keys: Vec<chrono::NaiveDate>,
    value_cols: Vec<(String, Vec<f64>)>,
    /// Row indexes by the row's foreign-key text.
    groups: HashMap<String, Vec<usize>>,
}

/// `None` for an empty frame, which loads no timeseries.
fn prepare_timeseries<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    fk_field: &str,
    ts_cfg: InlineTimeseriesConfig,
) -> PyResult<Option<PreparedTimeseries>> {
    let n_rows: usize = data.getattr("shape")?.get_item(0)?.extract()?;
    if n_rows == 0 {
        return Ok(None);
    }

    let fk_col: Vec<Py<PyAny>> = data.get_item(fk_field)?.call_method0("tolist")?.extract()?;

    let time_keys: Vec<chrono::NaiveDate> = match &ts_cfg.time {
        TimeSpec::StringColumn(col_name) => {
            let raw: Vec<String> = data
                .get_item(col_name)?
                .call_method1("astype", ("str",))?
                .call_method0("tolist")?
                .extract()?;
            raw.iter()
                .map(|s| kglite_core::api::timeseries::parse_date_query(s).map(|(d, _)| d))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e: String| -> PyErr {
                    crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
                })?
        }
        TimeSpec::SeparateColumns(col_names) => {
            let mut int_cols: Vec<Vec<i64>> = Vec::with_capacity(col_names.len());
            for cn in col_names {
                let col: Vec<i64> = data.get_item(cn)?.call_method0("tolist")?.extract()?;
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
                    kglite_core::api::timeseries::date_from_ymd(year, month, day)
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e: String| -> PyErr {
                    crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
                })?
        }
    };

    let resolved_resolution = if let Some(ref r) = ts_cfg.resolution {
        kglite_core::api::timeseries::validate_resolution(r).map_err(|e: String| -> PyErr {
            crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
        })?;
        r.clone()
    } else {
        match &ts_cfg.time {
            TimeSpec::SeparateColumns(cols) => match cols.len() {
                1 => "year".to_string(),
                2 => "month".to_string(),
                _ => "day".to_string(),
            },
            TimeSpec::StringColumn(_) => "month".to_string(),
        }
    };

    let mut value_cols: Vec<(String, Vec<f64>)> = Vec::with_capacity(ts_cfg.channels.len());
    for ch_name in &ts_cfg.channels {
        let col: Vec<f64> = data.get_item(ch_name)?.call_method0("tolist")?.extract()?;
        value_cols.push((ch_name.clone(), col));
    }

    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, fk_val) in fk_col.iter().enumerate() {
        let key = fk_val.bind(py).str()?.to_string();
        groups.entry(key).or_default().push(i);
    }

    Ok(Some(PreparedTimeseries {
        cfg: ts_cfg,
        resolution: resolved_resolution,
        time_keys,
        value_cols,
        groups,
    }))
}

/// Attach a [`PreparedTimeseries`] to the nodes the load wrote. Nothing here
/// can fail: every read of the input happened in [`prepare_timeseries`].
fn apply_timeseries(
    py: Python<'_>,
    graph: &mut DirGraph,
    node_type: &str,
    prepared: PreparedTimeseries,
) {
    let PreparedTimeseries {
        cfg: ts_cfg,
        resolution: resolved_resolution,
        time_keys,
        value_cols,
        groups,
    } = prepared;
    graph.build_id_index(node_type);

    let mut ts_nodes_loaded = 0usize;
    for (fk_str, row_indices) in &groups {
        let node_idx = {
            let id_str = Value::String(fk_str.clone());
            if let Some(idx) = graph.lookup_by_id_normalized(node_type, &id_str) {
                idx
            } else if let Ok(n) = fk_str.parse::<i64>() {
                let id_int = Value::Int64(n);
                if let Some(idx) = graph.lookup_by_id_normalized(node_type, &id_int) {
                    idx
                } else {
                    continue;
                }
            } else {
                continue;
            }
        };

        let mut sorted = row_indices.clone();
        sorted.sort_by(|&a, &b| time_keys[a].cmp(&time_keys[b]));

        let keys: Vec<chrono::NaiveDate> = sorted.iter().map(|&i| time_keys[i]).collect();
        let channels: HashMap<String, Vec<f64>> = value_cols
            .iter()
            .map(|(name, col)| (name.clone(), sorted.iter().map(|&i| col[i]).collect()))
            .collect();

        graph.set_node_timeseries(
            node_idx,
            kglite_core::api::timeseries::NodeTimeseries { keys, channels },
        );
        ts_nodes_loaded += 1;
    }

    let existing = graph.timeseries_configs.get(node_type);
    let mut merged_channels = existing.map(|c| c.channels.clone()).unwrap_or_default();
    for ch in &ts_cfg.channels {
        if !merged_channels.contains(ch) {
            merged_channels.push(ch.clone());
        }
    }
    let mut merged_units = existing.map(|c| c.units.clone()).unwrap_or_default();
    for (k, v) in ts_cfg.units {
        merged_units.insert(k, v);
    }
    let bin_type = existing.and_then(|c| c.bin_type.clone());

    graph.set_timeseries_config(
        node_type,
        kglite_core::api::timeseries::TimeseriesConfig {
            resolution: resolved_resolution,
            channels: merged_channels,
            units: merged_units,
            bin_type,
        },
    );

    if ts_nodes_loaded == 0 && !groups.is_empty() {
        let msg = std::ffi::CString::new(format!(
            "add_nodes: timeseries data found for {} groups but no matching nodes were created",
            groups.len()
        ))
        .unwrap_or_default();
        let _ = PyErr::warn(
            py,
            py.get_type::<pyo3::exceptions::PyUserWarning>().as_any(),
            msg.as_c_str(),
            1,
        );
    }
}

/// Marshal an `add_nodes` report, warning about skipped rows unless
/// `on_invalid` asked for silence. The counts are in the dict either way.
fn build_node_report_dict<'py>(
    py: Python<'py>,
    result: &NodeOperationReport,
    on_invalid: OnInvalid,
) -> PyResult<Py<PyAny>> {
    let report_dict = PyDict::new(py);
    report_dict.set_item("operation", &result.operation_type)?;
    report_dict.set_item("timestamp", result.timestamp.to_rfc3339())?;
    report_dict.set_item("nodes_created", result.nodes_created)?;
    report_dict.set_item("nodes_updated", result.nodes_updated)?;
    report_dict.set_item("nodes_skipped", result.nodes_skipped)?;
    report_dict.set_item("processing_time_ms", result.processing_time_ms)?;

    let has_errors = !result.errors.is_empty() || result.nodes_skipped > 0;
    if !result.errors.is_empty() {
        report_dict.set_item("errors", &result.errors)?;
    }
    report_dict.set_item("has_errors", has_errors)?;

    // Silent skips on bulk loads were a recurring footgun — surface them at
    // warn level rather than only in last_report().
    if has_errors && on_invalid.warns() {
        let total = result.nodes_created + result.nodes_updated + result.nodes_skipped;
        let detail = if result.errors.is_empty() {
            String::new()
        } else {
            format!(" {}", result.errors.join("; "))
        };
        let msg = if result.nodes_skipped > 0 {
            format!(
                "add_nodes: {} of {} rows skipped.{}",
                result.nodes_skipped, total, detail
            )
        } else {
            format!("add_nodes: completed with errors.{}", detail)
        };
        let cmsg = std::ffi::CString::new(msg).unwrap_or_default();
        let _ = PyErr::warn(
            py,
            py.get_type::<pyo3::exceptions::PyUserWarning>().as_any(),
            cmsg.as_c_str(),
            1,
        );
    }

    Ok(report_dict.into())
}

/// The report a managed reload gets back when it addresses a `runtime`-layer
/// type: the write is skipped, and this says so.
///
/// It carries the **full** `add_nodes` report shape — `operation`,
/// `timestamp`, the three counts, `processing_time_ms`, `has_errors` — plus
/// the two skip-specific keys. Shipping a differently-shaped dict from one
/// arm of the same method made every caller that reads `report["has_errors"]`
/// (the documented way to check a load) raise `KeyError` on exactly the path
/// where the caller most needs a readable answer.
///
/// `has_errors` is `False`: the skip is the contract working, not a failure.
/// `nodes_skipped` stays 0 because it counts *rows* rejected within a
/// performed load; nothing here was loaded at all, and `skipped_runtime_layer`
/// is the key that says so.
fn skipped_runtime_layer_report(py: Python<'_>, node_type: &str) -> PyResult<Py<PyAny>> {
    let report = PyDict::new(py);
    report.set_item("operation", "add_nodes")?;
    report.set_item("timestamp", chrono::Utc::now().to_rfc3339())?;
    report.set_item("nodes_created", 0)?;
    report.set_item("nodes_updated", 0)?;
    report.set_item("nodes_skipped", 0)?;
    report.set_item("processing_time_ms", 0.0)?;
    report.set_item("has_errors", false)?;
    report.set_item("skipped_runtime_layer", true)?;
    report.set_item("node_type", node_type)?;
    report.set_item(
        "message",
        format!("'{node_type}' is a runtime-owned type — skipped in managed reload"),
    )?;
    Ok(report.into_any().unbind())
}

/// Build the report dict returned by `extend`. Mirrors the
/// `build_node_report_dict` style (snake_case count keys + `has_errors`
/// + optional `errors`) so users see a familiar shape.
fn build_extend_report_dict<'py>(
    py: Python<'py>,
    result: &kglite_core::api::mutation::ExtendReport,
) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("operation", "extend")?;
    d.set_item("nodes_created", result.nodes_created)?;
    d.set_item("nodes_updated", result.nodes_updated)?;
    d.set_item("nodes_skipped", result.nodes_skipped)?;
    d.set_item("edges_created", result.edges_created)?;
    d.set_item("edges_updated", result.edges_updated)?;
    d.set_item("edges_skipped", result.edges_skipped)?;
    d.set_item("node_types_merged", result.node_types_merged)?;
    d.set_item("connection_types_merged", result.connection_types_merged)?;
    d.set_item("labels_unioned", result.labels_unioned)?;
    d.set_item("processing_time_ms", result.processing_time_ms)?;
    let has_errors = !result.errors.is_empty() || result.nodes_skipped > 0;
    if !result.errors.is_empty() {
        d.set_item("errors", &result.errors)?;
    }
    d.set_item("has_errors", has_errors)?;
    Ok(d.into())
}

#[pymethods]
impl KnowledgeGraph {
    /// Create an empty graph in the given storage mode.
    ///
    /// **Never durable, and there is deliberately no `durable` argument.**
    /// This produces a *detached* graph — `GraphLifecycle::detached()`, so no
    /// `source_path` and nowhere for a write-ahead log to live. `kglite.open`
    /// is the durable entry point: it binds the graph to a path and defaults
    /// to `DurabilityLevel::Full`. The asymmetry is structural, not a
    /// defaulting inconsistency, but it means `KnowledgeGraph(storage=
    /// "mapped")` and `kglite.open(new_path, storage="mapped")` differ in
    /// whether every commit is logged — worth knowing before comparing them.
    #[new]
    #[pyo3(signature = (*, storage=None, path=None))]
    fn new(storage: Option<&str>, path: Option<&str>) -> PyResult<Self> {
        use kglite_core::api::GraphRead;
        let mut graph = Self::construct(storage, path)?;
        // A disk graph lives in `path`: that directory is its origin exactly as
        // it is for `kglite.load(path)`, so a bare `save()` writes back there
        // instead of refusing for want of a path.
        if graph.inner.graph.is_disk() {
            graph.lifecycle.source_path = path.map(std::path::PathBuf::from);
        }
        Ok(graph)
    }
}

impl KnowledgeGraph {
    /// Build a fresh `KnowledgeGraph` for the given storage mode, creating
    /// disk-backed state at `path` when `storage="disk"`. Shared by the
    /// `#[new]` Python constructor and the `kglite.open(path)` load-or-create
    /// pyfunction. `source_path` is left `None` here — callers that want the
    /// graph to remember an origin file set it after construction.
    pub(crate) fn construct(storage: Option<&str>, path: Option<&str>) -> PyResult<Self> {
        // Mode selection + backend wiring lives in core
        // (`kglite::api::storage::new_dir_graph_in_mode`) so the wheel, the
        // bolt/mcp servers (`--storage`), and the C ABI
        // (`kglite_graph_new_in_mode`) all share one mode vocabulary.
        let graph = match storage {
            Some(mode_str) => {
                let mode =
                    kglite_core::api::storage::StorageMode::parse(mode_str).map_err(|e| {
                        crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
                    })?;
                kglite_core::api::storage::new_dir_graph_in_mode(
                    mode,
                    path.map(std::path::Path::new),
                )
                .map_err(|e| crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e)))?
            }
            None => DirGraph::new(),
        };

        Ok(KnowledgeGraph {
            inner: Arc::new(graph),
            cursor: crate::graph::CursorState::new(),
            embedder: None,
            default_timeout_ms: None,
            default_max_work_units: None,
            default_row_limit: None,
            lifecycle: crate::graph::GraphLifecycle::detached(),
        })
    }
}

#[pymethods]
impl KnowledgeGraph {
    /// Add nodes from a pandas DataFrame.
    #[pyo3(signature = (data, node_type, unique_id_field, node_title_field=None, columns=None, conflict_handling=None, skip_columns=None, column_types=None, timeseries=None, nullable_int_downcast=false, labels=None, managed_reload=false, git_sha=None, modified_by=None, on_invalid="warn", convention=None))]
    // The public Python loader exposes independently optional ingestion controls.
    #[allow(clippy::too_many_arguments)]
    fn add_nodes(
        &mut self,
        data: &Bound<'_, PyAny>,
        node_type: String,
        unique_id_field: String,
        node_title_field: Option<String>,
        columns: Option<&Bound<'_, PyList>>,
        conflict_handling: Option<String>,
        skip_columns: Option<&Bound<'_, PyList>>,
        column_types: Option<&Bound<'_, PyDict>>,
        timeseries: Option<&Bound<'_, PyDict>>,
        nullable_int_downcast: bool,
        labels: Option<Vec<String>>,
        managed_reload: bool,
        git_sha: Option<String>,
        modified_by: Option<String>,
        on_invalid: &str,
        convention: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        self.check_durable_owner()?;
        let py = data.py();
        let on_invalid = OnInvalid::parse(on_invalid)?;
        let convention = crate::graph::parse_interval_convention(convention)?;
        // Managed-reload guard: a managed reload (research rebuilding from
        // source) must never write a `runtime`-layer type (agent-owned). Skip
        // it as a no-op + report, so disjoint ownership is enforced, not
        // trusted. Undeclared / `managed` types proceed normally.
        if managed_reload && self.inner.layer_for(&node_type) == Some("runtime") {
            return skipped_runtime_layer_report(py, &node_type);
        }
        let parsed = parse_inline_config(
            data,
            &unique_id_field,
            node_title_field.as_deref(),
            columns,
            skip_columns,
            column_types,
            timeseries,
        )?;
        let embedding_data =
            extract_embedding_pairs(data, &unique_id_field, &parsed.embedding_columns)?;
        let converted = convert_dataframe(
            data,
            &unique_id_field,
            &parsed.column_list,
            parsed.ts_config.as_ref(),
            column_types,
            nullable_int_downcast,
            on_invalid,
        )?;

        let mut names = vec![node_type.as_str()];
        names.extend(parsed.column_list.iter().map(String::as_str));
        if let Some(label_list) = labels.as_ref() {
            names.extend(label_list.iter().map(String::as_str));
        }
        validate_interner_names(&self.inner, names)?;

        let graph = get_graph_mut(&mut self.inner);
        refuse_convention_without_interval(convention, converted.temporal_cfg.as_ref())?;
        // Every refusal comes before the declaration: once declared, only
        // `finish_declaration` may end the call. Everything after the write is
        // infallible, or a raise would leave the rows in memory but out of the
        // write-ahead log.
        let timeseries = parsed
            .ts_config
            .map(|cfg| prepare_timeseries(py, data, &unique_id_field, cfg))
            .transpose()?
            .flatten();
        let labels = labels
            .as_ref()
            .filter(|list| !list.is_empty())
            .map(|list| prepare_batch_labels(graph, data, &unique_id_field, list))
            .transpose()?;
        let declaration = converted
            .temporal_cfg
            .map(|cfg| {
                kglite_core::api::temporal::declare_from_column_types(
                    graph,
                    kglite_core::api::temporal::TemporalTarget::Node(node_type.clone()),
                    &cfg.valid_from,
                    &cfg.valid_to,
                    convention,
                    &converted.df,
                )
            })
            .transpose()
            .map_err(|e| crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e)))?;
        let result = apply_node_batch(
            py,
            graph,
            NodeBatchInput {
                df: converted.df,
                node_type: node_type.clone(),
                unique_id_field: unique_id_field.clone(),
                node_title_field,
                conflict_handling,
            },
            (git_sha, modified_by),
        );
        let (result, declared) = finish_declaration(py, graph, declaration, result)?;
        if let Some(cfg) = converted.spatial_cfg {
            graph.set_spatial_config(&node_type, cfg);
        }
        store_extracted_embeddings(graph, &node_type, &embedding_data);
        if let Some(prepared) = timeseries {
            apply_timeseries(py, graph, &node_type, prepared);
        }
        if let Some(prepared) = &labels {
            apply_batch_labels(graph, &node_type, prepared);
        }

        self.cursor.selection.clear();
        self.commit_wal()?;
        self.add_report(OperationReport::NodeOperation(result.clone()));
        declared?;

        Python::attach(|py| build_node_report_dict(py, &result, on_invalid))
    }

    /// Merge another KnowledgeGraph into this one, in place, with its declarations.
    #[pyo3(signature = (other, conflict_handling=None))]
    fn extend(
        &mut self,
        other: &Bound<'_, KnowledgeGraph>,
        conflict_handling: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        self.check_durable_owner()?;
        let py = other.py();

        // Clone the source's Arc<DirGraph> up front and release the
        // borrow, keeping the source strictly read-only. `g.extend(g)`
        // (self-extend) hits the `&mut self` borrow already held by this
        // call, so `try_borrow` fails — fall back to cloning self's own
        // Arc. Either way `source_arc` keeps the original DirGraph alive,
        // so the `Arc::make_mut` inside `get_graph_mut` clones on the
        // self-extend path: we read the original and write a fresh copy.
        let source_arc = match other.try_borrow() {
            Ok(other_ref) => Arc::clone(&other_ref.inner),
            Err(_) => Arc::clone(&self.inner),
        };

        // Surface the embedding-store limitation before mutating.
        if !source_arc.embeddings.is_empty() {
            let store_count = source_arc.embeddings.len();
            let msg = format!(
                "extend: the source graph has {} embedding store(s) which are NOT merged. \
                 Re-run set_embeddings()/add_embeddings() on the merged graph to rebuild them.",
                store_count
            );
            let cmsg = std::ffi::CString::new(msg).unwrap_or_default();
            let _ = PyErr::warn(
                py,
                py.get_type::<pyo3::exceptions::PyUserWarning>().as_any(),
                cmsg.as_c_str(),
                1,
            );
        }

        let graph = get_graph_mut(&mut self.inner);
        let result =
            kglite_core::api::mutation::extend_graph(graph, &source_arc, conflict_handling)
                .map_err(|message| bulk_write_err(graph, message))?;

        self.cursor.selection.clear();
        self.commit_wal()?;
        build_extend_report_dict(py, &result)
    }

    /// Add relationships from a DataFrame or read-only Cypher query.
    #[pyo3(signature = (data, connection_type, source_type, source_id_field, target_type, target_id_field, source_title_field=None, target_title_field=None, columns=None, skip_columns=None, conflict_handling=None, column_types=None, query=None, extra_properties=None, git_sha=None, modified_by=None, on_invalid="warn", convention=None))]
    // The public Python loader supports DataFrame and query modes with optional controls.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn add_relationships(
        &mut self,
        py: Python<'_>,
        data: Option<&Bound<'_, PyAny>>,
        connection_type: String,
        source_type: String,
        source_id_field: String,
        target_type: String,
        target_id_field: String,
        source_title_field: Option<String>,
        target_title_field: Option<String>,
        columns: Option<&Bound<'_, PyList>>,
        skip_columns: Option<&Bound<'_, PyList>>,
        conflict_handling: Option<String>,
        column_types: Option<&Bound<'_, PyDict>>,
        query: Option<String>,
        extra_properties: Option<&Bound<'_, PyDict>>,
        git_sha: Option<String>,
        modified_by: Option<String>,
        on_invalid: &str,
        convention: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        write_connections(
            py,
            self,
            false,
            data,
            connection_type,
            source_type,
            source_id_field,
            target_type,
            target_id_field,
            source_title_field,
            target_title_field,
            columns,
            skip_columns,
            conflict_handling,
            column_types,
            query,
            extra_properties,
            git_sha,
            modified_by,
            on_invalid,
            convention,
        )
    }

    /// Replace each input source node's relationships of a type with the input's — an atomic edge upsert.
    #[pyo3(signature = (data, connection_type, source_type, source_id_field, target_type, target_id_field, source_title_field=None, target_title_field=None, columns=None, skip_columns=None, conflict_handling=None, column_types=None, query=None, extra_properties=None, git_sha=None, modified_by=None, on_invalid="warn", convention=None))]
    // The same loader arguments add_relationships takes.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn replace_relationships(
        &mut self,
        py: Python<'_>,
        data: Option<&Bound<'_, PyAny>>,
        connection_type: String,
        source_type: String,
        source_id_field: String,
        target_type: String,
        target_id_field: String,
        source_title_field: Option<String>,
        target_title_field: Option<String>,
        columns: Option<&Bound<'_, PyList>>,
        skip_columns: Option<&Bound<'_, PyList>>,
        conflict_handling: Option<String>,
        column_types: Option<&Bound<'_, PyDict>>,
        query: Option<String>,
        extra_properties: Option<&Bound<'_, PyDict>>,
        git_sha: Option<String>,
        modified_by: Option<String>,
        on_invalid: &str,
        convention: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        write_connections(
            py,
            self,
            true,
            data,
            connection_type,
            source_type,
            source_id_field,
            target_type,
            target_id_field,
            source_title_field,
            target_title_field,
            columns,
            skip_columns,
            conflict_handling,
            column_types,
            query,
            extra_properties,
            git_sha,
            modified_by,
            on_invalid,
            convention,
        )
    }

    // ========================================================================
    // Connector API Methods (Bulk Loading)
    // ========================================================================

    /// Get the set of node types that exist in the graph.
    ///
    /// Returns:
    ///     List of node type names present in the graph.
    ///
    /// Example:
    ///     ```python
    ///     graph.add_nodes(df, 'Person', 'id', 'name')
    ///     graph.add_nodes(df2, 'Company', 'id', 'name')
    ///     print(graph.node_types)  # ['Person', 'Company']
    ///     ```
    #[getter]
    fn node_types(&self) -> Vec<String> {
        self.inner.get_node_types()
    }

    /// Add a secondary label to a batch of nodes by id.
    ///
    /// Secondary labels are queryable via Cypher (`MATCH (n:Label)`)
    /// and surfaced by `labels(n)`. The primary type (set by
    /// `add_nodes(node_type=...)`) is immutable — recreate or migrate
    /// a node to change it.
    ///
    /// Args:
    ///     node_type: Primary type of the nodes to label.
    ///     ids: List of node ids (the unique_id_field values).
    ///     label: Secondary label to add.
    ///
    /// Returns:
    ///     dict with ``labelled`` (count of nodes the label was newly
    ///     added to) and ``skipped`` (ids that don't exist as
    ///     ``node_type`` nodes). Idempotent — re-adding a label that's
    ///     already present is counted in ``skipped``.
    fn add_label(
        &mut self,
        py: Python<'_>,
        node_type: &str,
        ids: &Bound<'_, PyList>,
        label: &str,
    ) -> PyResult<Py<PyAny>> {
        self.check_durable_owner()?;
        validate_interner_names(&self.inner, [label])?;
        let g = get_graph_mut(&mut self.inner);
        if !g.type_indices.contains_key(node_type) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Node type '{}' does not exist in the graph",
                node_type
            )));
        }
        g.build_id_index(node_type);
        let key = g
            .interner
            .try_get_or_intern(label)
            .map_err(|e| crate::error_py::kg_to_pyerr(crate::error::KgError::from(e)))?;
        // Resolve ids first, then stamp through the bulk path — the
        // add_node_label loop this replaces was O(n²) over the id list.
        let mut missing = 0usize;
        let mut indices = Vec::with_capacity(ids.len());
        for item in ids.iter() {
            let id_val = py_in::py_value_to_value(&item)?;
            match g.lookup_by_id(node_type, &id_val) {
                Some(idx) => indices.push(idx),
                None => missing += 1,
            }
        }
        let (labelled, bulk_skipped) = g.add_node_labels_bulk(&indices, key);
        let skipped = missing + bulk_skipped;
        self.commit_wal()?;
        let result = PyDict::new(py);
        result.set_item("labelled", labelled)?;
        result.set_item("skipped", skipped)?;
        Ok(result.into())
    }

    /// Remove a secondary label from a batch of nodes by id.
    ///
    /// Errors if `label` is the primary type, which is immutable —
    /// recreate or migrate a node to change it.
    ///
    /// Args:
    ///     node_type: Primary type of the nodes.
    ///     ids: List of node ids.
    ///     label: Secondary label to remove.
    ///
    /// Returns:
    ///     dict with ``removed`` (count of nodes the label was
    ///     actually removed from) and ``skipped`` (ids that don't
    ///     exist, or didn't have the label).
    fn remove_label(
        &mut self,
        py: Python<'_>,
        node_type: &str,
        ids: &Bound<'_, PyList>,
        label: &str,
    ) -> PyResult<Py<PyAny>> {
        self.check_durable_owner()?;
        validate_interner_names(&self.inner, [label])?;
        let g = get_graph_mut(&mut self.inner);
        if !g.type_indices.contains_key(node_type) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Node type '{}' does not exist in the graph",
                node_type
            )));
        }
        g.build_id_index(node_type);
        let key = g
            .interner
            .try_get_or_intern(label)
            .map_err(|e| crate::error_py::kg_to_pyerr(crate::error::KgError::from(e)))?;
        let mut removed = 0usize;
        let mut skipped = 0usize;
        for item in ids.iter() {
            let id_val = py_in::py_value_to_value(&item)?;
            match g.lookup_by_id(node_type, &id_val) {
                Some(idx) => match g.remove_node_label(idx, key) {
                    Ok(true) => removed += 1,
                    Ok(false) => skipped += 1,
                    Err(e) => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e));
                    }
                },
                None => skipped += 1,
            }
        }
        self.commit_wal()?;
        let result = PyDict::new(py);
        result.set_item("removed", removed)?;
        result.set_item("skipped", skipped)?;
        Ok(result.into())
    }

    /// Add multiple node types at once from a list of node specifications.
    ///
    /// This enables bulk loading of nodes from data sources that provide
    /// standardized node specifications.
    ///
    /// Args:
    ///     nodes: List of dicts, each containing:
    ///         - 'node_type': str - The type/label for these nodes
    ///         - 'unique_id_field': str - Column name for unique ID
    ///         - 'node_title_field': str - Column name for display title
    ///         - 'data': DataFrame - The node data
    ///
    /// Returns:
    ///     Dict mapping node_type to count of nodes added
    ///
    /// Example:
    ///     ```python
    ///     nodes = [
    ///         {'node_type': 'Person', 'unique_id_field': 'id',
    ///          'node_title_field': 'name', 'data': people_df},
    ///         {'node_type': 'Company', 'unique_id_field': 'id',
    ///          'node_title_field': 'name', 'data': companies_df},
    ///     ]
    ///     stats = graph.add_nodes_bulk(nodes)
    ///     # {'Person': 100, 'Company': 50}
    ///     ```
    #[pyo3(signature = (nodes, *, git_sha=None, modified_by=None))]
    fn add_nodes_bulk(
        &mut self,
        py: Python<'_>,
        nodes: &Bound<'_, PyList>,
        git_sha: Option<String>,
        modified_by: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        self.check_durable_owner()?;
        let result_dict = PyDict::new(py);

        for item in nodes.iter() {
            let spec = item.cast::<PyDict>()?;

            let node_type: String = spec
                .get_item("node_type")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                        "Missing 'node_type' in node spec",
                    )
                })?
                .extract()?;
            let unique_id_field: String = spec
                .get_item("unique_id_field")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                        "Missing 'unique_id_field' in node spec",
                    )
                })?
                .extract()?;
            let node_title_field: String = spec
                .get_item("node_title_field")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                        "Missing 'node_title_field' in node spec",
                    )
                })?
                .extract()?;
            let data = spec.get_item("data")?.ok_or_else(|| {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(
                    "Missing 'data' in node spec".to_string(),
                ))
            })?;

            let df_cols = data.getattr("columns")?;
            let all_columns: Vec<String> = df_cols.extract()?;

            let df_result = py_in::pandas_to_dataframe(
                &data,
                std::slice::from_ref(&unique_id_field),
                &all_columns,
                None,
            )?;

            let graph = get_graph_mut(&mut self.inner);

            // Converted frame is pure Rust — apply off-GIL.
            let report = detach_mutation(py, || {
                graph.with_write_provenance(git_sha.as_deref(), modified_by.as_deref(), |graph| {
                    kglite_core::api::mutation::add_nodes(
                        graph,
                        df_result,
                        node_type.clone(),
                        unique_id_field,
                        Some(node_title_field),
                        None,
                    )
                })
            })?;

            result_dict.set_item(&node_type, report.nodes_created + report.nodes_updated)?;
        }

        self.cursor.selection.clear();
        self.commit_wal()?;
        Ok(result_dict.into())
    }

    /// Load several relationship types at once from a list of specs.
    #[pyo3(signature = (connections, *, git_sha=None, modified_by=None))]
    pub(super) fn add_relationships_bulk(
        &mut self,
        py: Python<'_>,
        connections: &Bound<'_, PyList>,
        git_sha: Option<String>,
        modified_by: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        self.add_relationships_internal(
            py,
            connections,
            false,
            git_sha.as_deref(),
            modified_by.as_deref(),
        )
    }

    /// Load relationship specs, skipping those whose source or target node type is not loaded.
    #[pyo3(signature = (connections, *, git_sha=None, modified_by=None))]
    pub(super) fn add_relationships_from_source(
        &mut self,
        py: Python<'_>,
        connections: &Bound<'_, PyList>,
        git_sha: Option<String>,
        modified_by: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        self.add_relationships_internal(
            py,
            connections,
            true,
            git_sha.as_deref(),
            modified_by.as_deref(),
        )
    }
}

/// Plain `impl`, deliberately outside `#[pymethods]`: this is the Rust-side
/// body shared by [`KnowledgeGraph::add_relationships_bulk`] and
/// [`KnowledgeGraph::add_relationships_from_source`]. Inside a `#[pymethods]`
/// block PyO3 would export it as a public Python method that
/// `kglite/__init__.pyi` does not declare.
impl KnowledgeGraph {
    fn add_relationships_internal(
        &mut self,
        py: Python<'_>,
        connections: &Bound<'_, PyList>,
        filter_to_loaded: bool,
        git_sha: Option<&str>,
        modified_by: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        self.check_durable_owner()?;
        let result_dict = PyDict::new(py);
        let loaded_types: std::collections::HashSet<String> = if filter_to_loaded {
            self.inner.all_node_types().into_iter().collect()
        } else {
            std::collections::HashSet::new()
        };
        let names = collect_bulk_connection_names(connections, &loaded_types, filter_to_loaded)?;
        validate_interner_names(&self.inner, names.iter().map(String::as_str))?;

        for item in connections.iter() {
            let spec = item.cast::<PyDict>()?;

            let source_type: String = spec
                .get_item("source_type")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                        "Missing 'source_type' in relationship spec",
                    )
                })?
                .extract()?;
            let target_type: String = spec
                .get_item("target_type")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                        "Missing 'target_type' in relationship spec",
                    )
                })?
                .extract()?;
            let connection_name: String = spec
                .get_item("connection_name")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                        "Missing 'connection_name' in relationship spec",
                    )
                })?
                .extract()?;
            let data = spec.get_item("data")?.ok_or_else(|| {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(
                    "Missing 'data' in relationship spec".to_string(),
                ))
            })?;

            if filter_to_loaded
                && (!loaded_types.contains(&source_type) || !loaded_types.contains(&target_type))
            {
                continue;
            }

            // Standardized column names for connector API
            let source_id_field = "source_id".to_string();
            let target_id_field = "target_id".to_string();

            let df_cols = data.getattr("columns")?;
            let all_columns: Vec<String> = df_cols.extract()?;

            if !all_columns.contains(&source_id_field) {
                return Err(crate::error_py::kg_to_pyerr(
                    crate::error::KgError::Argument(format!(
                    "Relationship spec for '{}' missing required 'source_id' column. Available: [{}]",
                    connection_name,
                    all_columns.join(", ")
                )),
                ));
            }
            if !all_columns.contains(&target_id_field) {
                return Err(crate::error_py::kg_to_pyerr(
                    crate::error::KgError::Argument(format!(
                    "Relationship spec for '{}' missing required 'target_id' column. Available: [{}]",
                    connection_name,
                    all_columns.join(", ")
                )),
                ));
            }

            let df_result = py_in::pandas_to_dataframe(
                &data,
                &[source_id_field.clone(), target_id_field.clone()],
                &all_columns,
                None,
            )?;

            let graph = get_graph_mut(&mut self.inner);

            // Converted frame is pure Rust — apply off-GIL.
            let report = detach_mutation(py, || {
                graph.with_write_provenance(git_sha, modified_by, |graph| {
                    kglite_core::api::mutation::add_connections(
                        graph,
                        df_result,
                        connection_name.clone(),
                        source_type,
                        source_id_field,
                        target_type,
                        target_id_field,
                        None, // source_title_field
                        None, // target_title_field
                        None, // conflict_handling
                    )
                })
            })?;

            result_dict.set_item(&connection_name, report.connections_created)?;
        }

        self.cursor.selection.clear();
        self.commit_wal()?;
        Ok(result_dict.into())
    }
}
