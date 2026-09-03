//! Phase 4: foreign-key edges declared on node CSVs, plus the implicit
//! `parent` → `OF_{PARENT}` edges, buffered or streamed.

use super::super::filter::apply_filter;
use super::super::table::{read_csv_chunks, ListMisparseTally, RawCsv};
use super::super::timeseries as ts;
use super::super::typing::map_blueprint_type;
use super::cache::CsvCache;
use super::nodes::{node_chunk_size, should_stream_spec};
use super::specs::FlatSpec;
use super::table_ops::subset_rows;
use super::BuildReport;
use crate::datatypes::values::DataFrame;
use crate::graph::mutation::maintain;
use crate::graph::schema::DirGraph;
use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};
use std::path::Path;

struct PreppedFkEdges {
    source_type: String,
    /// Source PK column name (which may be a synthesised `_type_id` for `pk: "auto"`).
    pk: String,
    /// Pre-built edge DataFrames, one per declared FK edge, in blueprint
    /// insertion order (critical for `skip_existence_check` parity with the
    /// old Python loader).
    edges: Vec<PreppedFkEdge>,
    /// Spec-level errors (e.g. missing FK column); surfaced after the serial
    /// consumer runs.
    errors: Vec<String>,
    /// Spec-level warnings (list cells that were probably meant as several
    /// values), surfaced alongside the errors.
    warnings: Vec<String>,
}

struct PreppedFkEdge {
    edge_type: String,
    target_type: String,
    target_col: String,
    df: DataFrame,
}

fn prep_fk_edges(spec: &FlatSpec, root: &Path, cache: &CsvCache) -> Option<PreppedFkEdges> {
    let csv_rel = spec.spec.csv.as_deref()?;

    let mut fk_edges: IndexMap<String, super::super::schema::FkEdge> = spec
        .spec
        .connections
        .fk_edges
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if let (Some(parent_type), Some(parent_fk)) = (&spec.spec.parent, &spec.spec.parent_fk) {
        let edge_type = format!("OF_{}", parent_type.to_uppercase());
        fk_edges.entry(edge_type).or_insert_with(|| {
            super::super::schema::FkEdge::plain(parent_type.clone(), parent_fk.clone())
        });
    }
    if fk_edges.is_empty() {
        return None;
    }

    let raw_rc = cache.get(root, csv_rel).ok()?;
    let mut raw: RawCsv = (*raw_rc).clone_raw();
    if !spec.spec.filter.is_empty() {
        apply_filter(&mut raw, &spec.spec.filter);
    }
    if let Some(tspec) = &spec.spec.timeseries {
        ts::drop_zero_time_components(&mut raw, tspec);
    }
    let raw_pk = spec.spec.pk.clone().unwrap_or_else(|| "id".to_string());
    let pk = if raw_pk == "auto" {
        let synth = format!("_{}_id", spec.node_type);
        let n = raw.row_count();
        let values: Vec<String> = (1..=n).map(|i| i.to_string()).collect();
        raw.headers.push(synth.clone());
        for (r, row) in raw.rows.iter_mut().enumerate() {
            row.push(values[r].clone());
            raw.nulls[r].push(false);
        }
        synth
    } else {
        raw_pk
    };

    let mut built = Vec::new();
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    for (edge_type, edge) in &fk_edges {
        let Some(fk_idx) = raw.col_index(&edge.fk) else {
            errors.push(format!(
                "[{}] FK column '{}' not found for edge {}",
                spec.node_type, edge.fk, edge_type
            ));
            continue;
        };
        let Some(pk_idx) = raw.col_index(&pk) else {
            errors.push(format!(
                "[{}] pk column '{}' not found for edge {}",
                spec.node_type, pk, edge_type
            ));
            continue;
        };
        let props = match fk_edge_properties(edge_type, &spec.node_type, edge, &pk) {
            Ok(p) => p,
            Err(e) => {
                errors.push(e);
                continue;
            }
        };

        let mut misparses = ListMisparseTally::default();
        let frame = match fk_edge_frame(&raw, &pk, pk_idx, edge, fk_idx, &props, &mut misparses) {
            Ok(Some(frame)) => frame,
            Ok(None) => continue,
            Err(e) => {
                errors.push(format!(
                    "[{}] failed to build edge DataFrame for {}: {}",
                    spec.node_type, edge_type, e
                ));
                continue;
            }
        };
        warnings.extend(misparses.into_warnings(&format!(
            "fk_edge '{edge_type}' (node '{}')",
            spec.node_type
        )));
        for col in &frame.missing_properties {
            errors.push(missing_fk_property_error(&spec.node_type, edge_type, col));
        }
        built.push(PreppedFkEdge {
            edge_type: edge_type.clone(),
            target_type: edge.target.clone(),
            target_col: frame.target_col,
            df: frame.df,
        });
    }

    Some(PreppedFkEdges {
        source_type: spec.node_type.clone(),
        pk,
        edges: built,
        errors,
        warnings,
    })
}

fn missing_fk_property_error(node_type: &str, edge_type: &str, column: &str) -> String {
    format!(
        "[{node_type}] fk_edge {edge_type}: property column '{column}' not found in the \
         source CSV — the edge is built without it"
    )
}

/// What one FK edge attaches to each edge besides the two ids: the source
/// columns, their declared types and the name each lands under. Validated
/// once per edge, then reused for every table (chunk) the edge is built from.
struct FkEdgeProperties {
    columns: Vec<String>,
    /// Keyed by CSV column name, like a junction's `property_types` — the
    /// rename applies to the output name only.
    declared: HashMap<String, String>,
    rename: HashMap<String, String>,
}

/// Validate one FK edge's `properties` / `property_types` / `rename` against
/// the id columns the frame already carries. Same rules as a junction's:
/// a rename key must be a declared property, an id column is not renamable,
/// and no two columns may land under one name.
fn fk_edge_properties(
    edge_type: &str,
    node_type: &str,
    edge: &super::super::schema::FkEdge,
    pk: &str,
) -> Result<FkEdgeProperties, String> {
    let target_col = fk_target_col(pk, &edge.fk);
    let mut columns: Vec<String> = Vec::new();
    for col in &edge.properties {
        if col == pk || col == &edge.fk {
            return Err(format!(
                "[{node_type}] fk_edge {edge_type}: property '{col}' is an id column \
                 (pk '{pk}', fk '{}'); the edge already carries it",
                edge.fk
            ));
        }
        if !columns.contains(col) {
            columns.push(col.clone());
        }
    }

    let mut rename: HashMap<String, String> = HashMap::new();
    for (col, new_name) in &edge.rename {
        if col == pk || col == &edge.fk {
            return Err(format!(
                "[{node_type}] fk_edge {edge_type}: rename of fk column '{col}' is not \
                 supported — 'fk' and 'pk' name the CSV columns"
            ));
        }
        if !columns.contains(col) {
            return Err(format!(
                "[{node_type}] fk_edge {edge_type}: rename key '{col}' is not in 'properties'"
            ));
        }
        let collides = new_name == pk
            || new_name == &target_col
            || columns.iter().any(|c| c == new_name && c != col)
            || rename.values().any(|v| v == new_name);
        if collides {
            return Err(format!(
                "[{node_type}] fk_edge {edge_type}: rename target '{new_name}' collides with \
                 another column"
            ));
        }
        rename.insert(col.clone(), new_name.clone());
    }

    // An unrecognized type keyword falls through to inference, and
    // `validation::unknown_property_type_warnings` already names it.
    let declared = edge
        .property_types
        .iter()
        .filter(|(_, ty)| map_blueprint_type(ty).is_some())
        .map(|(col, ty)| (col.clone(), ty.clone()))
        .collect();
    Ok(FkEdgeProperties {
        columns,
        declared,
        rename,
    })
}

struct FkEdgeFrame {
    target_col: String,
    df: DataFrame,
    /// Declared property columns this table does not have.
    missing_properties: Vec<String>,
}

/// One FK edge's frame, built from one raw table: the target column's name
/// plus the DataFrame `connect` consumes (source id, target id, and any
/// declared edge properties). `Ok(None)` when the table contributes no edge —
/// every FK cell in it was null.
///
/// Shared by the buffered and the streaming loader so the two cannot drift:
/// a chunk is just a shorter table, and both paths must derive an edge from
/// one the same way.
fn fk_edge_frame(
    raw: &RawCsv,
    pk: &str,
    pk_idx: usize,
    edge: &super::super::schema::FkEdge,
    fk_idx: usize,
    props: &FkEdgeProperties,
    misparses: &mut ListMisparseTally,
) -> Result<Option<FkEdgeFrame>, String> {
    let cols = build_fk_columns(raw, pk, &edge.fk, pk_idx, fk_idx);
    if cols.src.is_empty() {
        return Ok(None);
    }
    let mut df = build_edge_df(pk, &cols.target_col, cols.src, cols.tgt)?;

    let mut missing_properties = Vec::new();
    let mut present = Vec::new();
    for col in &props.columns {
        if raw.col_index(col).is_some() {
            present.push(col.clone());
        } else {
            missing_properties.push(col.clone());
        }
    }
    if !present.is_empty() {
        // Property values must follow the rows the ids came from: a row whose
        // FK was null produced no edge, and its properties must not slide onto
        // the next row's.
        let subset;
        let source: &RawCsv = if cols.rows.len() == raw.row_count() {
            raw
        } else {
            subset = subset_rows(raw, &cols.rows);
            &subset
        };
        super::super::typing::append_typed_columns(
            &mut df,
            source,
            &present,
            &props.declared,
            &props.rename,
            misparses,
        )?;
    }

    Ok(Some(FkEdgeFrame {
        target_col: cols.target_col,
        df,
        missing_properties,
    }))
}

/// The edge frame's target column name. A self-reference (`fk == pk`) needs a
/// synthesised one so the source and target columns differ.
fn fk_target_col(pk: &str, fk: &str) -> String {
    if pk == fk {
        format!("_target_{}", fk)
    } else {
        fk.to_string()
    }
}

struct FkColumns {
    target_col: String,
    src: Vec<Option<String>>,
    tgt: Vec<Option<String>>,
    /// Indices into `raw.rows` of the rows behind `src`/`tgt`, so property
    /// columns can be built from exactly the rows that produced an edge.
    rows: Vec<usize>,
}

fn build_fk_columns(raw: &RawCsv, pk: &str, fk: &str, pk_idx: usize, fk_idx: usize) -> FkColumns {
    let target_col = fk_target_col(pk, fk);
    let mut src = Vec::new();
    let mut tgt = Vec::new();
    let mut rows = Vec::new();
    // Keep only rows with a non-null target id.
    if pk == fk {
        for (r, row) in raw.rows.iter().enumerate() {
            if raw.nulls[r][pk_idx] {
                continue;
            }
            src.push(Some(row[pk_idx].clone()));
            tgt.push(Some(row[pk_idx].clone()));
            rows.push(r);
        }
    } else {
        for (r, row) in raw.rows.iter().enumerate() {
            if raw.nulls[r][fk_idx] {
                continue;
            }
            let src_val = if raw.nulls[r][pk_idx] {
                None
            } else {
                Some(row[pk_idx].clone())
            };
            src.push(src_val);
            tgt.push(Some(row[fk_idx].clone()));
            rows.push(r);
        }
    }
    FkColumns {
        target_col,
        src,
        tgt,
        rows,
    }
}

pub(super) fn load_fk_edges(
    graph: &mut DirGraph,
    specs: &[&FlatSpec],
    root: &Path,
    cache: &CsvCache,
    report: &mut BuildReport,
) -> Result<(), String> {
    use rayon::prelude::*;
    let profile = std::env::var("KGLITE_BLUEPRINT_PROFILE").is_ok();

    // Same predicate as node streaming, so a spec's nodes and FK edges
    // either both stream or both buffer. Mixing the two for one spec would
    // re-introduce the cache requirement streaming exists to drop.
    let (streamable, buffered): (Vec<&FlatSpec>, Vec<&FlatSpec>) = specs
        .iter()
        .copied()
        .partition(|s| should_stream_spec(s, root));

    // Buffered path: parallel prep, serial connect.
    let t_par = std::time::Instant::now();
    let prepped: Vec<Option<PreppedFkEdges>> = buffered
        .par_iter()
        .map(|spec| prep_fk_edges(spec, root, cache))
        .collect();
    let t_par_ms = t_par.elapsed().as_millis();

    let t_serial = std::time::Instant::now();
    let mut t_connect = std::time::Duration::ZERO;
    for result in prepped {
        let Some(pfx) = result else { continue };
        for err in pfx.errors {
            report.errors.push(err);
        }
        report.warnings.extend(pfx.warnings);
        for edge in pfx.edges {
            let t_c = std::time::Instant::now();
            let count = connect(
                graph,
                edge.df,
                &edge.edge_type,
                &pfx.source_type,
                &pfx.pk,
                &edge.target_type,
                &edge.target_col,
                report,
                maintain::InitialLoad::Detect,
            )?;
            t_connect += t_c.elapsed();
            *report
                .edges_by_type
                .entry(edge.edge_type.clone())
                .or_insert(0) += count;
        }
    }

    // Streaming path: same chain, one chunk at a time.
    let t_stream = std::time::Instant::now();
    for spec in &streamable {
        if let Err(e) = load_streamed_fk_edges(graph, spec, root, report) {
            report.errors.push(e);
        }
    }
    let t_stream_ms = t_stream.elapsed().as_millis();

    if profile {
        eprintln!(
            "    fk parallel prep: {} ms | serial connect: {} ms | streaming ({} specs): {} ms | serial total: {} ms",
            t_par_ms,
            t_connect.as_millis(),
            streamable.len(),
            t_stream_ms,
            t_serial.elapsed().as_millis(),
        );
    }
    Ok(())
}

/// Streaming FK-edge loader. Mirrors `load_streamed_node_spec`
/// row-handling, but each chunk emits one `connect()` call per declared
/// FK edge, built with the same `build_fk_columns` + `build_edge_df`
/// primitives the buffered path uses.
///
/// The auto-pk counter advances in lock-step with
/// `load_streamed_node_spec`'s counter so source ids match across
/// the node + FK phases (both apply the same filter to the same CSV
/// in the same chunk order).
fn load_streamed_fk_edges(
    graph: &mut DirGraph,
    spec: &FlatSpec,
    root: &Path,
    report: &mut BuildReport,
) -> Result<(), String> {
    let Some(csv_rel) = spec.spec.csv.as_deref() else {
        return Ok(());
    };

    // Declared edges plus the implicit `OF_{PARENT}` edge for any spec
    // that declares both `parent` and `parent_fk`.
    let mut fk_edges: IndexMap<String, super::super::schema::FkEdge> = spec
        .spec
        .connections
        .fk_edges
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if let (Some(parent_type), Some(parent_fk)) = (&spec.spec.parent, &spec.spec.parent_fk) {
        let edge_type = format!("OF_{}", parent_type.to_uppercase());
        fk_edges.entry(edge_type).or_insert_with(|| {
            super::super::schema::FkEdge::plain(parent_type.clone(), parent_fk.clone())
        });
    }
    if fk_edges.is_empty() {
        return Ok(());
    }

    let csv_path = root.join(csv_rel);
    let chunk_size = node_chunk_size();
    // Decided before the first chunk and reused for all of them: chunking this
    // CSV bounds peak RAM, so it must not decide which rows become their own
    // edge. See `maintain::InitialLoad`.
    let initial_load: HashMap<String, maintain::InitialLoad> = fk_edges
        .keys()
        .map(|edge_type| {
            let unseen = !graph.connection_type_metadata.contains_key(edge_type);
            (edge_type.clone(), maintain::InitialLoad::Preset(unseen))
        })
        .collect();
    let chunks = read_csv_chunks(&csv_path, chunk_size)
        .map_err(|e| format!("[{}] {}", spec.node_type, e))?;

    let raw_pk = spec.spec.pk.clone().unwrap_or_else(|| "id".to_string());
    let (pk, is_auto_pk) = if raw_pk == "auto" {
        (format!("_{}_id", spec.node_type), true)
    } else {
        (raw_pk, false)
    };
    let mut auto_pk_counter: u64 = 1;

    // Validated once per edge, not once per chunk: a bad `rename` is a
    // property of the spec, and an edge carrying one is skipped whole rather
    // than half-built. An edge missing from this map is one such.
    let mut edge_props: IndexMap<String, FkEdgeProperties> = IndexMap::new();
    for (edge_type, edge) in &fk_edges {
        match fk_edge_properties(edge_type, &spec.node_type, edge, &pk) {
            Ok(props) => {
                edge_props.insert(edge_type.clone(), props);
            }
            Err(e) => report.errors.push(e),
        }
    }

    // Track per-edge missing-column errors so we report each at most
    // once instead of once per chunk.
    let mut reported_missing_fk: HashSet<String> = HashSet::new();
    let mut reported_missing_pk: HashSet<String> = HashSet::new();
    let mut reported_missing_prop: HashSet<(String, String)> = HashSet::new();
    // One tally per edge across every chunk of this CSV — the junction
    // loader's reason applies here too.
    let mut misparses: IndexMap<String, ListMisparseTally> = IndexMap::new();

    for chunk_result in chunks {
        let mut raw = chunk_result.map_err(|e| format!("[{}] {}", spec.node_type, e))?;
        if !spec.spec.filter.is_empty() {
            apply_filter(&mut raw, &spec.spec.filter);
        }
        if raw.row_count() == 0 {
            continue;
        }
        if is_auto_pk {
            raw.headers.push(pk.clone());
            for r in 0..raw.row_count() {
                raw.rows[r].push(auto_pk_counter.to_string());
                raw.nulls[r].push(false);
                auto_pk_counter += 1;
            }
        }

        let Some(pk_idx) = raw.col_index(&pk) else {
            for edge_type in fk_edges.keys() {
                if reported_missing_pk.insert(edge_type.clone()) {
                    report.errors.push(format!(
                        "[{}] pk column '{}' not found for edge {}",
                        spec.node_type, pk, edge_type
                    ));
                }
            }
            continue;
        };

        for (edge_type, edge) in &fk_edges {
            let Some(props) = edge_props.get(edge_type) else {
                continue;
            };
            let Some(fk_idx) = raw.col_index(&edge.fk) else {
                if reported_missing_fk.insert(edge_type.clone()) {
                    report.errors.push(format!(
                        "[{}] FK column '{}' not found for edge {}",
                        spec.node_type, edge.fk, edge_type
                    ));
                }
                continue;
            };
            let tally = misparses.entry(edge_type.clone()).or_default();
            let frame = match fk_edge_frame(&raw, &pk, pk_idx, edge, fk_idx, props, tally) {
                Ok(Some(frame)) => frame,
                Ok(None) => continue,
                Err(e) => {
                    report.errors.push(format!(
                        "[{}] failed to build edge DataFrame for {}: {}",
                        spec.node_type, edge_type, e
                    ));
                    continue;
                }
            };
            for col in &frame.missing_properties {
                if reported_missing_prop.insert((edge_type.clone(), col.clone())) {
                    report
                        .errors
                        .push(missing_fk_property_error(&spec.node_type, edge_type, col));
                }
            }
            let (target_col, df) = (frame.target_col, frame.df);
            let count = connect(
                graph,
                df,
                edge_type,
                &spec.node_type,
                &pk,
                &edge.target,
                &target_col,
                report,
                initial_load[edge_type],
            )?;
            *report.edges_by_type.entry(edge_type.clone()).or_insert(0) += count;
        }
    }
    for (edge_type, tally) in misparses {
        report.warnings.extend(tally.into_warnings(&format!(
            "fk_edge '{edge_type}' (node '{}')",
            spec.node_type
        )));
    }
    Ok(())
}

fn build_edge_df(
    src_name: &str,
    tgt_name: &str,
    src: Vec<Option<String>>,
    tgt: Vec<Option<String>>,
) -> Result<DataFrame, String> {
    // Decide column types: try i64, fall back to string.
    let src_type = infer_id_type(&src);
    let tgt_type = infer_id_type(&tgt);
    let mut df = DataFrame::new(Vec::new());
    add_id_column(&mut df, src_name, src, src_type)?;
    add_id_column(&mut df, tgt_name, tgt, tgt_type)?;
    Ok(df)
}

pub(super) fn infer_id_type(vals: &[Option<String>]) -> crate::datatypes::values::ColumnType {
    let mut all_int = true;
    for v in vals {
        let Some(s) = v else { continue };
        let t = s.trim();
        if t.is_empty() {
            continue;
        }
        if t.parse::<i64>().is_ok() {
            continue;
        }
        if let Ok(f) = t.parse::<f64>() {
            if f.is_finite() && f.fract() == 0.0 {
                continue;
            }
        }
        all_int = false;
        break;
    }
    if all_int {
        crate::datatypes::values::ColumnType::Int64
    } else {
        crate::datatypes::values::ColumnType::String
    }
}

pub(super) fn add_id_column(
    df: &mut DataFrame,
    name: &str,
    vals: Vec<Option<String>>,
    col_type: crate::datatypes::values::ColumnType,
) -> Result<(), String> {
    use crate::datatypes::values::{ColumnData, ColumnType};
    let data = match col_type {
        ColumnType::Int64 => {
            let ints: Vec<Option<i64>> = vals
                .iter()
                .map(|v| {
                    v.as_ref().and_then(|s| {
                        let t = s.trim();
                        if t.is_empty() {
                            None
                        } else if let Ok(i) = t.parse::<i64>() {
                            Some(i)
                        } else if let Ok(f) = t.parse::<f64>() {
                            if f.is_finite()
                                && f.fract() == 0.0
                                && f >= i64::MIN as f64
                                && f <= i64::MAX as f64
                            {
                                Some(f as i64)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                })
                .collect();
            ColumnData::Int64(ints)
        }
        _ => ColumnData::String(
            vals.into_iter()
                .map(|v| v.filter(|s| !s.is_empty()))
                .collect(),
        ),
    };
    df.add_column(name.to_string(), col_type, data)
}

// Thin adapter: the parameter list mirrors `add_connections_with_initial_load`.
#[allow(clippy::too_many_arguments)]
pub(super) fn connect(
    graph: &mut DirGraph,
    df: DataFrame,
    connection_type: &str,
    source_type: &str,
    source_id_field: &str,
    target_type: &str,
    target_id_field: &str,
    report: &mut BuildReport,
    initial_load: maintain::InitialLoad,
) -> Result<usize, String> {
    match maintain::add_connections_with_initial_load(
        graph,
        df,
        connection_type.to_string(),
        source_type.to_string(),
        source_id_field.to_string(),
        target_type.to_string(),
        target_id_field.to_string(),
        None,
        None,
        None,
        initial_load,
    ) {
        Ok(r) => {
            if r.connections_skipped > 0 {
                let detail = r.errors.join("; ");
                report.warnings.push(format!(
                    "[{}] -[{}]-> {}: {} skipped ({})",
                    source_type, connection_type, target_type, r.connections_skipped, detail
                ));
            }
            if r.stubs_vivified > 0 {
                report.warnings.push(format!(
                    "[{}] -[{}]-> {}: {} stub node(s) vivified for missing endpoints",
                    source_type, connection_type, target_type, r.stubs_vivified
                ));
            }
            Ok(r.connections_created)
        }
        Err(e) => {
            report
                .errors
                .push(format!("[{}] edge {}: {}", source_type, connection_type, e));
            Ok(0)
        }
    }
}
