//! KnowledgeGraph #[pymethods]: node + connection ingestion.
//!
//! Part of the Phase 9 split of the kg_methods.rs monolith (5,419 lines
//! single pymethods block). PyO3 merges multiple `#[pymethods] impl`
//! blocks at class-registration time, so the split is purely structural —
//! no runtime impact.

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
use kglite_core::api::GraphRead;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::Bound;
use std::collections::HashMap;
use std::sync::Arc;

// ─── add_nodes phase helpers ────────────────────────────────────────────────
//
// `add_nodes` is the most-touched user-facing function in the project. It
// runs eight independent phases over its inputs (parse config → extract
// embeddings → convert DataFrame → apply batch → register feature configs
// → store embeddings → apply timeseries → finalize). Each phase is a
// private free function below; the pymethod itself is a thin orchestrator.

struct InlineConfig {
    ts_config: Option<InlineTimeseriesConfig>,
    embedding_columns: Vec<String>,
    column_list: Vec<String>,
}

/// Run a pure-Rust batch-mutation closure with the GIL released, mapping
/// the engine's `String` error to the typed `kglite.*` exception.
///
/// The apply phase of `add_nodes` / `add_connections` operates purely on
/// already-converted Rust data (`DataFrame`, `&mut DirGraph`), so holding
/// the GIL through it starves every other Python thread for the duration
/// of a bulk insert. Detach only spans like this one — anything touching
/// `Bound`/`PyAny` must stay attached.
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
                    "Missing '{key}' in connection spec"
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
/// frame (the `data` mode shared by `add_connections` and
/// `replace_connections`). Returns the columnar DataFrame plus any
/// temporal-edge config auto-detected from `validFrom`/`validTo` column
/// types — the caller merges that into `graph.temporal_edge_configs`.
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

    let df_result = py_in::pandas_to_dataframe(
        data,
        &[source_id_field.to_string(), target_id_field.to_string()],
        &column_list,
        effective_types.as_ref(),
    )?;
    Ok((df_result, temporal_cfg))
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

/// Shared body of `add_connections` (replace=false) and
/// `replace_connections` (replace=true). The two methods are identical
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
) -> PyResult<Py<PyAny>> {
    use crate::datatypes::values::DataFrame as KgDataFrame;

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
        // Parse the cypher query
        let mut parsed = cypher::parse_cypher(&query_str).map_err(|e| -> PyErr {
            crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(format!(
                "Cypher syntax error in query: {}",
                e
            )))
        })?;

        // Reject mutation queries — the query must be read-only
        if cypher::is_mutation_query(&parsed) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "The 'query' parameter must be a read-only query (MATCH...RETURN). \
                 CREATE/SET/DELETE/MERGE are not allowed here.",
            ));
        }

        // Execute read-only: clone Arc, execute without holding mutable borrow
        let inner_clone = kg.inner.clone();
        let empty_params = HashMap::new();
        // Run the same planner optimizations as g.cypher() — otherwise
        // pushdowns (including correlated-equality) don't fire here.
        cypher::optimize(&mut parsed, &inner_clone, &empty_params);
        let cypher_result = {
            let executor = cypher::CypherExecutor::with_params(&inner_clone, &empty_params, None);
            executor.execute(&parsed)
        }
        .map_err(|e| {
            crate::error_py::kg_to_pyerr(crate::error::KgError::CypherExecution {
                message: format!("Cypher execution error in connection query: {}", e),
                position: None,
            })
        })?;

        // Resolve NodeRef values to actual IDs/titles
        let mut rows = cypher_result.rows;
        resolve_noderefs(&inner_clone.graph, &mut rows);

        // Convert row-oriented Cypher result to columnar DataFrame
        let mut df_result =
            KgDataFrame::from_cypher_rows(cypher_result.columns, rows).map_err(|e| -> PyErr {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(format!(
                    "Failed to convert query results to DataFrame: {}",
                    e
                )))
            })?;

        // Apply extra_properties as constant columns
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
        let result = detach_mutation(py, || {
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
        kg.add_report(OperationReport::ConnectionOperation(result.clone()));

        return KnowledgeGraph::connection_report_to_py(&result, &connection_type);
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
    )?;

    let mut names = vec![
        connection_type.as_str(),
        source_type.as_str(),
        target_type.as_str(),
    ];
    let frame_names = df_result.get_column_names();
    names.extend(frame_names.iter().map(String::as_str));
    validate_interner_names(&kg.inner, names)?;

    let graph = get_graph_mut(&mut kg.inner);

    // The converted frame is pure Rust — apply the batch off-GIL.
    let result = detach_mutation(py, || {
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

    // Merge temporal config into graph (auto-detected from validFrom/validTo column types)
    if let Some(cfg) = temporal_cfg {
        graph
            .temporal_edge_configs
            .entry(connection_type.clone())
            .or_default()
            .push(cfg);
    }

    kg.cursor.selection.clear();

    // Disk mode: build CSR from pending edges so queries work immediately
    let graph = get_graph_mut(&mut kg.inner);
    graph
        .ensure_disk_edges_built()
        .map_err(pyo3::exceptions::PyOSError::new_err)?;

    kg.add_report(OperationReport::ConnectionOperation(result.clone()));

    KnowledgeGraph::connection_report_to_py(&result, &connection_type)
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
    let id_series = data.get_item(unique_id_field)?;
    let nrows: usize = data.getattr("shape")?.get_item(0)?.extract()?;
    let mut result = Vec::with_capacity(embedding_columns.len());

    for emb_col in embedding_columns {
        let series = data.get_item(emb_col)?;
        let mut pairs = Vec::with_capacity(nrows);
        for i in 0..nrows {
            let id_val = py_in::py_value_to_value(&id_series.get_item(i)?)?;
            let emb_val: Vec<f32> = series.get_item(i)?.extract()?;
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
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    unique_id_field: &str,
    column_list: &[String],
    ts_config: Option<&InlineTimeseriesConfig>,
    column_types: Option<&Bound<'py, PyDict>>,
    nullable_int_downcast: bool,
) -> PyResult<ConvertedFrame> {
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
    )?;

    Ok(ConvertedFrame {
        df,
        spatial_cfg,
        temporal_cfg,
    })
}

/// Apply the converted node batch with the GIL released — the DataFrame is
/// already pure Rust at this point, so the insert runs off-GIL.
struct NodeBatchInput {
    df: DataFrame,
    node_type: String,
    unique_id_field: String,
    node_title_field: Option<String>,
    conflict_handling: Option<String>,
}

fn apply_node_batch(
    py: Python<'_>,
    graph: &mut DirGraph,
    input: NodeBatchInput,
    provenance: (Option<String>, Option<String>),
) -> PyResult<NodeOperationReport> {
    let (git_sha, modified_by) = provenance;
    detach_mutation(py, || {
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
    })
}

fn register_feature_configs(
    graph: &mut DirGraph,
    node_type: &str,
    spatial_cfg: Option<kglite_core::api::SpatialConfig>,
    temporal_cfg: Option<kglite_core::api::TemporalConfig>,
) {
    if let Some(cfg) = spatial_cfg {
        graph.spatial_configs.insert(node_type.to_string(), cfg);
    }
    if let Some(cfg) = temporal_cfg {
        graph
            .temporal_node_configs
            .insert(node_type.to_string(), cfg);
    }
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
        let store_key = if emb_col.ends_with("_emb") {
            emb_col.clone()
        } else {
            format!("{}_emb", emb_col)
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

/// Apply a uniform set of secondary labels to every row in the batch.
/// Reads the unique_id_field column from the original DataFrame and
/// looks each id up in the graph, applying every label in `labels`.
/// Idempotent — if a label is already present (or equals the primary
/// type), `DirGraph::add_node_label` no-ops.
fn apply_batch_labels<'py>(
    graph: &mut DirGraph,
    node_type: &str,
    data: &Bound<'py, PyAny>,
    unique_id_field: &str,
    labels: &[String],
) -> PyResult<()> {
    graph.build_id_index(node_type);
    let label_keys: Vec<kglite_core::api::InternedKey> = labels
        .iter()
        .map(|l| {
            graph
                .interner
                .try_get_or_intern(l)
                .map_err(|e| crate::error_py::kg_to_pyerr(crate::error::KgError::from(e)))
        })
        .collect::<PyResult<_>>()?;
    let id_series = data.get_item(unique_id_field)?;
    let nrows: usize = data.getattr("shape")?.get_item(0)?.extract()?;
    for i in 0..nrows {
        let id_val = py_in::py_value_to_value(&id_series.get_item(i)?)?;
        if let Some(node_idx) = graph.lookup_by_id(node_type, &id_val) {
            for &key in &label_keys {
                graph.add_node_label(node_idx, key);
            }
        }
    }
    Ok(())
}

fn apply_timeseries<'py>(
    py: Python<'py>,
    graph: &mut DirGraph,
    node_type: &str,
    data: &Bound<'py, PyAny>,
    fk_field: &str,
    ts_cfg: InlineTimeseriesConfig,
) -> PyResult<()> {
    let n_rows: usize = data.getattr("shape")?.get_item(0)?.extract()?;
    if n_rows == 0 {
        return Ok(());
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

        graph.timeseries_store.insert(
            node_idx.index(),
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

    graph.timeseries_configs.insert(
        node_type.to_string(),
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

    Ok(())
}

fn build_node_report_dict<'py>(
    py: Python<'py>,
    result: &NodeOperationReport,
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

    // Emit a Python warning whenever the report carries any skips or
    // errors. Silent skips on bulk loads were a recurring footgun —
    // surface them at warn level so the user sees them without needing
    // to inspect last_report().
    if has_errors {
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
    #[new]
    #[pyo3(signature = (*, storage=None, path=None))]
    fn new(storage: Option<&str>, path: Option<&str>) -> PyResult<Self> {
        Self::construct(storage, path)
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
            default_max_rows: None,
            lifecycle: crate::graph::GraphLifecycle::detached(),
        })
    }
}

#[pymethods]
impl KnowledgeGraph {
    /// Add nodes from a pandas DataFrame.
    ///
    /// Args:
    ///     data: DataFrame containing node data.
    ///     node_type: Label for this set of nodes (e.g. 'Person').
    ///     unique_id_field: Column used as unique identifier. String and integer IDs
    ///         are auto-detected from the DataFrame dtype.
    ///     node_title_field: Column used as display title. Defaults to unique_id_field.
    ///     columns: Whitelist of columns to include. None = all.
    ///     conflict_handling: 'update' (default), 'replace', 'skip', or 'preserve'.
    ///     skip_columns: Columns to exclude from properties.
    ///     column_types: Override column type detection: {'col': 'string'|'integer'|'float'|'datetime'|'uniqueid'}.
    ///     nullable_int_downcast: When True, Float64 columns whose non-null
    ///         values are all integer-valued (e.g. `pd.NA`-bearing ints that
    ///         pandas auto-promoted to float64) are silently downcast to Int64.
    ///         Default False — explicit opt-in protects existing callers.
    ///
    /// Returns:
    ///     dict with 'nodes_created', 'nodes_updated', 'nodes_skipped',
    ///     'processing_time_ms', 'has_errors', and optionally 'errors'.
    #[pyo3(signature = (data, node_type, unique_id_field, node_title_field=None, columns=None, conflict_handling=None, skip_columns=None, column_types=None, timeseries=None, nullable_int_downcast=false, labels=None, managed_reload=false, git_sha=None, modified_by=None))]
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
    ) -> PyResult<Py<PyAny>> {
        let py = data.py();
        // Managed-reload guard: a managed reload (research rebuilding from
        // source) must never write a `runtime`-layer type (agent-owned). Skip
        // it as a no-op + report, so disjoint ownership is enforced, not
        // trusted. Undeclared / `managed` types proceed normally.
        if managed_reload && self.inner.layer_for(&node_type) == Some("runtime") {
            let report = PyDict::new(py);
            report.set_item("nodes_created", 0)?;
            report.set_item("nodes_updated", 0)?;
            report.set_item("skipped_runtime_layer", true)?;
            report.set_item("node_type", &node_type)?;
            report.set_item(
                "message",
                format!("'{node_type}' is a runtime-owned type — skipped in managed reload"),
            )?;
            return Ok(report.into_any().unbind());
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
            py,
            data,
            &unique_id_field,
            &parsed.column_list,
            parsed.ts_config.as_ref(),
            column_types,
            nullable_int_downcast,
        )?;

        let mut names = vec![node_type.as_str()];
        names.extend(parsed.column_list.iter().map(String::as_str));
        if let Some(label_list) = labels.as_ref() {
            names.extend(label_list.iter().map(String::as_str));
        }
        validate_interner_names(&self.inner, names)?;

        let graph = get_graph_mut(&mut self.inner);
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
        )?;
        register_feature_configs(
            graph,
            &node_type,
            converted.spatial_cfg,
            converted.temporal_cfg,
        );
        store_extracted_embeddings(graph, &node_type, &embedding_data);
        if let Some(ts_cfg) = parsed.ts_config {
            apply_timeseries(py, graph, &node_type, data, &unique_id_field, ts_cfg)?;
        }
        if let Some(label_list) = labels.as_ref() {
            if !label_list.is_empty() {
                apply_batch_labels(graph, &node_type, data, &unique_id_field, label_list)?;
            }
        }

        self.cursor.selection.clear();
        if graph.graph.is_disk() {
            graph.sync_disk_column_stores();
        }
        self.add_report(OperationReport::NodeOperation(result.clone()));

        Python::attach(|py| build_node_report_dict(py, &result))
    }

    /// Merge another KnowledgeGraph into this one, in place.
    ///
    /// A native alternative to round-tripping through CSV export/import
    /// when building a graph incrementally from multiple sources (or
    /// merging two ``.kgl`` files loaded into memory). The *other* graph
    /// is read-only and never mutated.
    ///
    /// Semantics
    /// ---------
    /// - **Node identity** is ``(node_type, id)`` — the same key the id
    ///   index uses. ``id`` is the canonical integer node id in every
    ///   storage mode. When a node in ``other`` matches an existing node
    ///   here, the conflict is resolved by ``conflict_handling`` (same
    ///   vocabulary as ``add_nodes``):
    ///
    ///   - ``'update'`` (default) — merge properties, ``other`` wins on
    ///     conflicts; title is overwritten.
    ///   - ``'replace'`` — replace all properties and title with
    ///     ``other``'s.
    ///   - ``'skip'`` — leave the existing node untouched.
    ///   - ``'preserve'`` — merge properties, existing values win;
    ///     title kept unless currently null.
    ///   - ``'sum'`` — adds numeric property values on **edges**; for
    ///     **node** properties it acts as ``update`` (matches
    ///     ``ConflictHandling::Sum`` in ``add_nodes`` / ``add_connections``).
    ///
    /// - **Secondary labels** (multi-label, since 0.10.5) are *unioned*
    ///   onto the matched/created node — never removed. Idempotent.
    /// - **Property schemas** merge: a property present in ``other`` but
    ///   not here extends this graph's type schema (same path
    ///   ``add_nodes`` uses for new columns).
    /// - **Edges** dedup on ``(connection_type, source, target)``: an
    ///   edge that already exists here is **not** duplicated — its
    ///   properties merge per ``conflict_handling``. Exact-duplicate
    ///   edges present in both graphs are therefore created once, not
    ///   twice. (This is stricter than petgraph's raw parallel-edge
    ///   capability, mirroring ``add_connections``' dedup so a merge
    ///   never silently doubles shared edges.)
    ///
    /// Scope limits (v1)
    /// -----------------
    /// - **In-memory only.** Both graphs must use the default in-memory
    ///   storage; ``storage='mapped'`` / ``'disk'`` graphs raise an
    ///   error suggesting the export/import path.
    /// - **Embeddings are NOT merged.** If ``other`` has any embedding
    ///   stores a warning is emitted — re-run ``set_embeddings`` /
    ///   ``add_embeddings`` after the merge to rebuild them here.
    /// - **Self-extend** (``g.extend(g)``) is a deliberate no-op for
    ///   creation: every node/edge already matches itself, so the result
    ///   is property-merge-against-self (a no-op under every mode but
    ///   ``replace``, which rewrites each node with its own values).
    ///   Reported as 0 created, N updated.
    /// - **Locks.** Like ``add_nodes`` / ``add_connections``, this bulk
    ///   path does not consult ``schema_locked`` / ``read_only`` (those
    ///   gate the Cypher write path only).
    ///
    /// Args:
    ///     other: KnowledgeGraph to merge into this one (read-only).
    ///     conflict_handling: 'update' (default), 'replace', 'skip',
    ///         'preserve', or 'sum'.
    ///
    /// Returns:
    ///     dict with 'nodes_created', 'nodes_updated', 'nodes_skipped',
    ///     'edges_created', 'edges_skipped', 'node_types_merged',
    ///     'connection_types_merged', 'labels_unioned',
    ///     'processing_time_ms', 'has_errors', and optionally 'errors'.
    #[pyo3(signature = (other, conflict_handling=None))]
    fn extend(
        &mut self,
        other: &Bound<'_, KnowledgeGraph>,
        conflict_handling: Option<String>,
    ) -> PyResult<Py<PyAny>> {
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
                .map_err(|e: String| -> PyErr {
                    crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(e))
                })?;

        self.cursor.selection.clear();
        build_extend_report_dict(py, &result)
    }

    /// Add connections (edges) between existing nodes.
    ///
    /// Two modes — supply **either** `data` (a pandas DataFrame) **or** `query`
    /// (a Cypher string whose RETURN columns provide source/target IDs):
    ///
    /// ```python
    /// # From DataFrame (existing API):
    /// graph.add_connections(df, "KNOWS", "Person", "src_id", "Person", "tgt_id")
    ///
    /// # From Cypher query (new):
    /// graph.add_connections(
    ///     None, "ENCLOSES", "Play", "play_id", "StructuralElement", "struct_id",
    ///     query="MATCH (p:Play), (s:StructuralElement) WHERE contains(p, s) "
    ///           "RETURN DISTINCT p.id AS play_id, s.id AS struct_id",
    /// )
    ///
    /// # With extra static properties stamped onto every edge:
    /// graph.add_connections(
    ///     None, "HC_IN_FORMATION", "Discovery", "src", "Stratigraphy", "tgt",
    ///     query="MATCH ... RETURN d.id AS src, s.id AS tgt",
    ///     extra_properties={"hc_rank": 1},
    /// )
    /// ```
    ///
    /// Args:
    ///     data: DataFrame containing connection data, or None when using query.
    ///     connection_type: Label for this connection type (e.g. 'KNOWS').
    ///     source_type: Node type of the source nodes.
    ///     source_id_field: Column containing source node IDs.
    ///     target_type: Node type of the target nodes.
    ///     target_id_field: Column containing target node IDs.
    ///     source_title_field: Optional column to update source node titles.
    ///     target_title_field: Optional column to update target node titles.
    ///     columns: Optional edge-property whitelist (data mode only). None keeps all
    ///         non-skipped DataFrame columns, matching add_nodes.
    ///     skip_columns: Columns to exclude from edge properties (data mode only).
    ///     conflict_handling: 'update' (default), 'replace', 'skip', or 'preserve'.
    ///     column_types: Override column type detection (data mode only).
    ///     query: Cypher query string (alternative to data). Must be a read-only
    ///         query that RETURNs columns matching source_id_field and target_id_field.
    ///     extra_properties: Dict of static properties to add to every edge created
    ///         from the query results (query mode only).
    ///
    /// Returns:
    ///     dict with 'connections_created', 'connections_skipped',
    ///     'processing_time_ms', 'has_errors', and optionally 'errors'.
    #[pyo3(signature = (data, connection_type, source_type, source_id_field, target_type, target_id_field, source_title_field=None, target_title_field=None, columns=None, skip_columns=None, conflict_handling=None, column_types=None, query=None, extra_properties=None, git_sha=None, modified_by=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_connections(
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
        )
    }

    /// Replace a node's outgoing edges of a given type, then add the
    /// supplied edges — an atomic edge upsert.
    ///
    /// Unlike `add_connections` (add-only), this **prunes** first: for
    /// every source node that appears in `data` (or the `query` result),
    /// its existing edges *of `connection_type`* are removed, then the
    /// edges described by the input are added. Edges from sources not in
    /// the input, and edges of other types from the same sources, are
    /// untouched. The prune + add run in one call, so there is no
    /// clear-then-add window that could leave a node edgeless on failure.
    ///
    /// Use it to re-sync a derived edge set — "the current MENTIONS of
    /// exactly these documents are this list" — idempotently:
    ///
    /// ```python
    /// # First sync: doc 1 → [A, B]
    /// graph.replace_connections(df_ab, "MENTIONS", "Doc", "doc", "Entity", "ent")
    /// # Re-sync doc 1 → [B, C]: the stale 1→A edge is pruned, 1→C added.
    /// graph.replace_connections(df_bc, "MENTIONS", "Doc", "doc", "Entity", "ent")
    /// ```
    ///
    /// Accepts every argument `add_connections` does (including `query`
    /// mode and `extra_properties`) with identical semantics; only the
    /// prune-first behaviour differs.
    ///
    /// Args:
    ///     data: DataFrame containing connection data, or None when using query.
    ///     connection_type: Label for the connection type to replace (e.g. 'MENTIONS').
    ///     source_type: Node type of the source nodes.
    ///     source_id_field: Column containing source node IDs.
    ///     target_type: Node type of the target nodes.
    ///     target_id_field: Column containing target node IDs.
    ///     source_title_field: Optional column to update source node titles.
    ///     target_title_field: Optional column to update target node titles.
    ///     columns: Optional edge-property whitelist (data mode only). None keeps all
    ///         non-skipped DataFrame columns, matching add_nodes.
    ///     skip_columns: Columns to exclude from edge properties (data mode only).
    ///     conflict_handling: 'update' (default), 'replace', 'skip', or 'preserve'.
    ///     column_types: Override column type detection (data mode only).
    ///     query: Cypher query string (alternative to data). Must be read-only.
    ///     extra_properties: Static properties stamped onto every edge (query mode only).
    ///
    /// Returns:
    ///     dict with 'connections_created', 'connections_skipped',
    ///     'processing_time_ms', 'has_errors', and optionally 'errors'.
    #[pyo3(signature = (data, connection_type, source_type, source_id_field, target_type, target_id_field, source_title_field=None, target_title_field=None, columns=None, skip_columns=None, conflict_handling=None, column_types=None, query=None, extra_properties=None, git_sha=None, modified_by=None))]
    #[allow(clippy::too_many_arguments)]
    fn replace_connections(
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
        )
    }

    // ========================================================================
    // Connector API Methods (Bulk Loading)
    // ========================================================================

    /// Get the set of node types that exist in the graph.
    ///
    /// Returns:
    ///     List of node type names (excludes internal SchemaNode type)
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
    /// `add_nodes(node_type=...)`) is immutable via this API — to
    /// retype a node, use `SET n.type = 'NewType'`.
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
        let mut labelled = 0usize;
        let mut skipped = 0usize;
        for item in ids.iter() {
            let id_val = py_in::py_value_to_value(&item)?;
            match g.lookup_by_id(node_type, &id_val) {
                Some(idx) => {
                    if g.add_node_label(idx, key) {
                        labelled += 1;
                    } else {
                        skipped += 1;
                    }
                }
                None => skipped += 1,
            }
        }
        let result = PyDict::new(py);
        result.set_item("labelled", labelled)?;
        result.set_item("skipped", skipped)?;
        Ok(result.into())
    }

    /// Remove a secondary label from a batch of nodes by id.
    ///
    /// Errors if `label` is the primary type — use `SET n.type` to
    /// retype a node.
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

            // Get columns from dataframe
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
        Ok(result_dict.into())
    }

    /// Add multiple connection types at once from a list of connection specifications.
    ///
    /// This enables bulk loading of connections from data sources that provide
    /// standardized connection specifications with 'source_id' and 'target_id' columns.
    ///
    /// Args:
    ///     connections: List of dicts, each containing:
    ///         - 'source_type': str - Node type of source nodes
    ///         - 'target_type': str - Node type of target nodes
    ///         - 'connection_name': str - The connection/edge type
    ///         - 'data': DataFrame - Must have 'source_id' and 'target_id' columns
    ///
    /// Returns:
    ///     Dict mapping connection_name to count of connections added
    ///
    /// Example:
    ///     ```python
    ///     connections = [
    ///         {'source_type': 'Person', 'target_type': 'Company',
    ///          'connection_name': 'WORKS_AT', 'data': works_df},
    ///         {'source_type': 'Person', 'target_type': 'Person',
    ///          'connection_name': 'KNOWS', 'data': knows_df},
    ///     ]
    ///     stats = graph.add_connections_bulk(connections)
    ///     # {'WORKS_AT': 500, 'KNOWS': 1200}
    ///     ```
    #[pyo3(signature = (connections, *, git_sha=None, modified_by=None))]
    fn add_connections_bulk(
        &mut self,
        py: Python<'_>,
        connections: &Bound<'_, PyList>,
        git_sha: Option<String>,
        modified_by: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        self.add_connections_internal(
            py,
            connections,
            false,
            git_sha.as_deref(),
            modified_by.as_deref(),
        )
    }

    /// Add connections, automatically filtering to only those where
    /// both source and target node types exist in the graph.
    ///
    /// This enables data sources to provide ALL possible connections,
    /// and kglite selects only the valid ones based on loaded node types.
    ///
    /// Args:
    ///     connections: List of dicts, each containing:
    ///         - 'source_type': str - Node type of source nodes
    ///         - 'target_type': str - Node type of target nodes
    ///         - 'connection_name': str - The connection/edge type
    ///         - 'data': DataFrame - Must have 'source_id' and 'target_id' columns
    ///
    /// Returns:
    ///     Dict mapping connection_name to count of connections added
    ///     (only includes connections that were actually loaded)
    ///
    /// Example:
    ///     ```python
    ///     # Data source provides all possible connections
    ///     all_connections = data_source.get_all_connections()
    ///
    /// ```text
    /// # Graph only has Person and Company loaded
    /// # This will skip connections involving other node types
    /// stats = graph.add_connections_from_source(all_connections)
    /// ```
    /// ```
    #[pyo3(signature = (connections, *, git_sha=None, modified_by=None))]
    fn add_connections_from_source(
        &mut self,
        py: Python<'_>,
        connections: &Bound<'_, PyList>,
        git_sha: Option<String>,
        modified_by: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        self.add_connections_internal(
            py,
            connections,
            true,
            git_sha.as_deref(),
            modified_by.as_deref(),
        )
    }

    /// Internal helper for bulk connection loading
    fn add_connections_internal(
        &mut self,
        py: Python<'_>,
        connections: &Bound<'_, PyList>,
        filter_to_loaded: bool,
        git_sha: Option<&str>,
        modified_by: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let result_dict = PyDict::new(py);
        let loaded_types: std::collections::HashSet<String> = if filter_to_loaded {
            self.inner.get_node_types().into_iter().collect()
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
                        "Missing 'source_type' in connection spec",
                    )
                })?
                .extract()?;
            let target_type: String = spec
                .get_item("target_type")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                        "Missing 'target_type' in connection spec",
                    )
                })?
                .extract()?;
            let connection_name: String = spec
                .get_item("connection_name")?
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                        "Missing 'connection_name' in connection spec",
                    )
                })?
                .extract()?;
            let data = spec.get_item("data")?.ok_or_else(|| {
                crate::error_py::kg_to_pyerr(crate::error::KgError::Argument(
                    "Missing 'data' in connection spec".to_string(),
                ))
            })?;

            // Skip if filtering and types not loaded
            if filter_to_loaded
                && (!loaded_types.contains(&source_type) || !loaded_types.contains(&target_type))
            {
                continue;
            }

            // Standardized column names for connector API
            let source_id_field = "source_id".to_string();
            let target_id_field = "target_id".to_string();

            // Get columns from dataframe
            let df_cols = data.getattr("columns")?;
            let all_columns: Vec<String> = df_cols.extract()?;

            // Verify required columns exist
            if !all_columns.contains(&source_id_field) {
                return Err(crate::error_py::kg_to_pyerr(
                    crate::error::KgError::Argument(format!(
                    "Connection spec for '{}' missing required 'source_id' column. Available: [{}]",
                    connection_name,
                    all_columns.join(", ")
                )),
                ));
            }
            if !all_columns.contains(&target_id_field) {
                return Err(crate::error_py::kg_to_pyerr(
                    crate::error::KgError::Argument(format!(
                    "Connection spec for '{}' missing required 'target_id' column. Available: [{}]",
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
        Ok(result_dict.into())
    }
}
