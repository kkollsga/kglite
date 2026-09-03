//! Phases 2 and 3: core node specs and the sub-node specs flattened out of
//! them, loaded either buffered (whole file) or streamed in row chunks.

use super::super::filter::apply_filter;
use super::super::geometry::{convert_geojson, has_spatial_properties, spatial_targets};
use super::super::input::InputRegistry;
use super::super::table::{ListMisparseTally, RawCsv};
use super::super::timeseries as ts;
use super::super::typing::{map_blueprint_type, typed_dataframe};
use super::cache::CsvCache;
use super::prepass;
use super::specs::FlatSpec;
use super::table_ops::dedupe_by_pk;
use super::BuildReport;
use crate::datatypes::values::DataFrame;
use crate::graph::mutation::maintain;
use crate::graph::schema::{DirGraph, SpatialConfig};
use std::collections::{HashMap, HashSet};

/// Everything a single node spec produces — computed off-thread by
/// `prep_node_spec`, then consumed sequentially by `load_node_specs`.
struct PreppedNode {
    node_type: String,
    pk: String,
    title_arg: Option<String>,
    df: DataFrame,
    spatial_config: Option<SpatialConfig>,
    /// Diagnostics raised while typing the columns. Prep runs under rayon and
    /// has no `&mut BuildReport`, so they ride back to the serial phase.
    warnings: Vec<String>,
    /// Full raw (pre-dedup) CSV + resolved timeseries spec, if this type
    /// has `timeseries` declared. Kept because `apply_timeseries` needs
    /// every row, not just the dedup'd node DataFrame.
    timeseries: Option<(RawCsv, ts::ResolvedTimeseries)>,
}

fn prep_node_spec(
    spec: &FlatSpec,
    registry: &InputRegistry,
    cache: &CsvCache,
) -> Result<Option<PreppedNode>, String> {
    if spec.is_manual {
        return Ok(None);
    }
    let Some(input) = spec.input.as_deref() else {
        return Ok(None);
    };
    let raw_rc = match cache.get(registry, input) {
        Ok(r) => r,
        Err(e) => return Err(format!("[{}] {}", spec.node_type, e)),
    };
    let mut raw: RawCsv = (*raw_rc).clone_raw();

    if !spec.spec.filter.is_empty() {
        apply_filter(&mut raw, &spec.spec.filter);
    }
    if let Some(tspec) = &spec.spec.timeseries {
        ts::drop_zero_time_components(&mut raw, tspec);
    }

    let pk = spec.spec.pk.clone().unwrap_or_else(|| "id".to_string());
    let (pk, synth_pk_values) = if pk == "auto" {
        let synth = format!("_{}_id", spec.node_type);
        let n = raw.row_count();
        let values: Vec<String> = (1..=n).map(|i| i.to_string()).collect();
        (synth, Some(values))
    } else {
        (pk, None)
    };
    if let Some(vals) = &synth_pk_values {
        raw.headers.push(pk.clone());
        for (r, row) in raw.rows.iter_mut().enumerate() {
            row.push(vals[r].clone());
            raw.nulls[r].push(false);
        }
    }

    let title_field = spec.spec.title.clone().unwrap_or_else(|| pk.clone());

    // Geometry conversion (GeoJSON → WKT + centroid, in-place on raw)
    let has_geo = has_spatial_properties(&spec.spec.properties);
    let targets = if has_geo {
        let t = spatial_targets(&spec.spec.properties);
        convert_geojson(&mut raw, &t)?;
        Some(t)
    } else {
        None
    };

    let ts_resolved = if let Some(tspec) = &spec.spec.timeseries {
        Some(ts::resolve(tspec, &raw)?)
    } else {
        None
    };

    // Dedup for node creation only (timeseries keeps the full row set).
    let raw_for_nodes = if ts_resolved.is_some() {
        dedupe_by_pk(&raw, &pk)
    } else {
        raw.clone_raw()
    };

    let skip_set: HashSet<&String> = spec.spec.skipped.iter().collect();
    let ts_excluded: HashSet<String> = ts_resolved
        .as_ref()
        .map(|r| r.excluded_columns.iter().cloned().collect())
        .unwrap_or_default();
    let geometry_passthrough: HashSet<String> = HashSet::from_iter(["_geometry".to_string()]);
    let parent_fk_skip: HashSet<String> = match &spec.spec.parent_fk {
        Some(pfk) if !spec.spec.properties.contains_key(pfk) => HashSet::from_iter([pfk.clone()]),
        _ => HashSet::new(),
    };

    let mut declared: HashMap<String, String> = HashMap::new();
    for (col, ty) in &spec.spec.properties {
        if map_blueprint_type(ty).is_some() {
            declared.insert(col.clone(), ty.clone());
        }
    }

    let keep: Vec<String> = raw
        .headers
        .iter()
        .filter(|h| {
            !skip_set.contains(h)
                && !ts_excluded.contains(h.as_str())
                && !geometry_passthrough.contains(h.as_str())
                && !parent_fk_skip.contains(h.as_str())
                || *h == &pk
                || *h == &title_field
        })
        .cloned()
        .collect();
    let mut seen = HashSet::new();
    let keep: Vec<String> = keep
        .into_iter()
        .filter(|h| seen.insert(h.clone()))
        .collect();

    let mut misparses = ListMisparseTally::default();
    let df = typed_dataframe(
        &raw_for_nodes,
        &keep,
        &declared,
        &HashMap::new(),
        &mut misparses,
    )?;
    let warnings = misparses.into_warnings(&format!("node '{}'", spec.node_type));

    let title_arg = if title_field != pk {
        Some(title_field.clone())
    } else {
        None
    };

    let spatial_config = if has_geo {
        let tgt = targets.unwrap_or_default();
        let mut cfg = SpatialConfig {
            geometry: tgt.wkt,
            ..Default::default()
        };
        if let (Some(lat), Some(lon)) = (tgt.lat, tgt.lon) {
            cfg.location = Some((lat, lon));
        }
        Some(cfg)
    } else {
        None
    };

    let timeseries = ts_resolved.map(|r| (raw, r));

    Ok(Some(PreppedNode {
        node_type: spec.node_type.clone(),
        pk,
        title_arg,
        df,
        spatial_config,
        warnings,
        timeseries,
    }))
}

pub(super) fn load_node_specs(
    graph: &mut DirGraph,
    specs: &[FlatSpec],
    registry: &InputRegistry,
    cache: &CsvCache,
    report: &mut BuildReport,
    _phase_name: &str,
) -> Result<(), String> {
    use rayon::prelude::*;
    let profile = std::env::var("KGLITE_BLUEPRINT_PROFILE").is_ok();

    // Streamable specs run a per-chunk `Source::chunks → typed_dataframe
    // → add_nodes` loop that bounds peak RAM by chunk size. Everything else
    // (timeseries, spatial, manual, and anything below the size threshold)
    // keeps the parallel-prep path — `should_stream_spec` carries why small
    // CSVs stay buffered.
    let (buffered, streamable): (Vec<&FlatSpec>, Vec<&FlatSpec>) =
        specs.iter().partition(|s| !should_stream_spec(s, registry));

    // Buffered path: parallel prep, serial dispatch.
    let t_par = std::time::Instant::now();
    let prepped: Vec<Result<Option<PreppedNode>, String>> = buffered
        .par_iter()
        .map(|spec| prep_node_spec(spec, registry, cache))
        .collect();
    let t_par_ms = t_par.elapsed().as_millis();

    let t_serial = std::time::Instant::now();
    let mut t_add = std::time::Duration::ZERO;
    let mut t_ts = std::time::Duration::ZERO;
    for (spec, result) in buffered.iter().zip(prepped) {
        let node = match result {
            Ok(Some(n)) => n,
            Ok(None) => continue,
            Err(e) => {
                report.errors.push(e);
                continue;
            }
        };

        report.warnings.extend(node.warnings);

        let t_a = std::time::Instant::now();
        let rep = maintain::add_nodes(
            graph,
            node.df,
            node.node_type.clone(),
            node.pk.clone(),
            node.title_arg,
            None,
        )
        .map_err(|e| format!("add_nodes '{}': {}", node.node_type, e))?;
        t_add += t_a.elapsed();

        let count = rep.nodes_created + rep.nodes_updated;
        *report
            .nodes_by_type
            .entry(node.node_type.clone())
            .or_insert(0) += count;

        if let Some(cfg) = node.spatial_config {
            graph.spatial_configs.insert(node.node_type.clone(), cfg);
        }

        if let Some((raw, resolved)) = node.timeseries {
            let t_t = std::time::Instant::now();
            apply_timeseries(graph, &spec.node_type, &node.pk, &raw, &resolved)?;
            t_ts += t_t.elapsed();
        }
    }

    // Streaming path: per-spec errors land in `report.errors` (parity with
    // the buffered path) — missing CSVs / type mismatches must not abort
    // the build.
    let t_stream = std::time::Instant::now();
    for spec in &streamable {
        if let Err(e) = load_streamed_node_spec(graph, spec, registry, report) {
            report.errors.push(e);
        }
    }
    let t_stream_ms = t_stream.elapsed().as_millis();

    if profile {
        eprintln!(
            "    parallel prep: {} ms | serial add_nodes: {} ms | timeseries: {} ms | streaming ({} specs): {} ms | serial total: {} ms",
            t_par_ms,
            t_add.as_millis(),
            t_ts.as_millis(),
            streamable.len(),
            t_stream_ms,
            t_serial.elapsed().as_millis(),
        );
    }
    Ok(())
}

/// True iff this spec's row shape is compatible with the
/// streaming loader. Independent of file size — the size gate is
/// applied separately via `should_stream_spec`.
///
/// Returns false for:
/// - manual specs (no input — synthesised from FK targets)
/// - timeseries specs (need full row set for grouping + dedup-by-pk)
/// - spatial specs (geometry conversion mutates RawCsv in-place)
///
/// `pk: "auto"` stays streamable — a per-spec counter hands each chunk
/// a dense id range matching the buffered path's row order (see
/// `load_streamed_node_spec`).
fn is_streamable_node_spec(spec: &FlatSpec) -> bool {
    if spec.is_manual {
        return false;
    }
    if spec.input.is_none() {
        return false;
    }
    if spec.spec.timeseries.is_some() {
        return false;
    }
    if has_spatial_properties(&spec.spec.properties) {
        return false;
    }
    true
}

/// True iff this spec should actually flow through the streaming
/// loader on the current build. Combines the semantic eligibility
/// check (`is_streamable_node_spec`) with a size gate so
/// small/medium inputs stay on the (faster) buffered path.
///
/// The streaming dispatch carries ~20% overhead per spec vs the
/// buffered parallel-prep on a single 500K-row CSV — fine on
/// 50M-row CSVs where the streaming RAM bound is the point, not
/// worth paying on the KB-to-~50MB CSVs typical of current graphs.
/// Threshold default: 100 MB, tunable via
/// `KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB`.
///
/// Falls back to the buffered path for a source that cannot be chunked at
/// all, and for one whose size is unknown — a file that can't be stat'd is
/// usually one that isn't there, and the buffered path's error reporting
/// handles that.
pub(super) fn should_stream_spec(spec: &FlatSpec, registry: &InputRegistry) -> bool {
    if !is_streamable_node_spec(spec) {
        return false;
    }
    let Some(input) = spec.input.as_deref() else {
        return false;
    };
    let Ok(source) = registry.get(input) else {
        return false;
    };
    if !source.can_chunk() {
        return false;
    }
    match source.size_hint() {
        Some(bytes) => bytes >= streaming_threshold_bytes(),
        None => false,
    }
}

fn streaming_threshold_bytes() -> u64 {
    let mb = std::env::var("KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(100);
    mb.saturating_mul(1024 * 1024)
}

/// Streaming node-spec loader: `filter` + `typed_dataframe` per chunk,
/// then `maintain::add_nodes`. `add_nodes` is upsert-by-id, so successive
/// chunks accumulate cleanly into the same node type without resurrecting
/// the per-spec working clone the buffered path needs.
fn load_streamed_node_spec(
    graph: &mut DirGraph,
    spec: &FlatSpec,
    registry: &InputRegistry,
    report: &mut BuildReport,
) -> Result<(), String> {
    let Some(input) = spec.input.as_deref() else {
        return Ok(());
    };
    let chunk_size = node_chunk_size();
    let source = registry
        .get(input)
        .map_err(|e| format!("[{}] {}", spec.node_type, e))?;

    let raw_pk = spec.spec.pk.clone().unwrap_or_else(|| "id".to_string());
    let (pk, is_auto_pk) = if raw_pk == "auto" {
        (format!("_{}_id", spec.node_type), true)
    } else {
        (raw_pk, false)
    };
    let title_field = spec.spec.title.clone().unwrap_or_else(|| pk.clone());
    let title_arg = if title_field != pk {
        Some(title_field.clone())
    } else {
        None
    };

    let skip_set: HashSet<&String> = spec.spec.skipped.iter().collect();
    let parent_fk_skip: HashSet<String> = match &spec.spec.parent_fk {
        Some(pfk) if !spec.spec.properties.contains_key(pfk) => HashSet::from_iter([pfk.clone()]),
        _ => HashSet::new(),
    };
    let mut declared: HashMap<String, String> = HashMap::new();
    for (col, ty) in &spec.spec.properties {
        if map_blueprint_type(ty).is_some() {
            declared.insert(col.clone(), ty.clone());
        }
    }
    if is_auto_pk {
        // The synthesised pk is a dense 1..N counter, which the buffered path
        // infers as Int64 over any row set; naming it here keeps it out of the
        // pre-pass, whose chunks do not carry the column at all.
        declared.insert(pk.clone(), "int".to_string());
    }

    // Types for the kept columns the blueprint did not declare, resolved over
    // the whole input before the first row is loaded — per-chunk inference
    // would make a column's type depend on the chunk size.
    let prepared = prepass::prepare_chunks(source, chunk_size, &declared, |raw| {
        if !spec.spec.filter.is_empty() {
            apply_filter(raw, &spec.spec.filter);
        }
        streaming_keep_list(raw, &pk, &title_field, &skip_set, &parent_fk_skip)
    })
    .map_err(|e| format!("[{}] {}", spec.node_type, e))?;
    if let Some(w) = prepass::prepass_warning(&format!("node '{}'", spec.node_type), &prepared) {
        report.warnings.push(w);
    }
    let prepass::Prepared {
        resolved, chunks, ..
    } = prepared;
    declared.extend(resolved);

    // Per-spec auto-pk counter. Plain `u64` (not atomic) is fine because
    // chunks are processed serially within a spec, and the first id is 1 to
    // match the buffered path's `1..=n`.
    let mut auto_pk_counter: u64 = 1;
    // One tally for the whole CSV: a malformed list column is malformed in
    // every chunk, and a per-chunk warning would repeat it once per 250k rows.
    let mut misparses = ListMisparseTally::default();

    for chunk_result in chunks {
        let mut raw = chunk_result.map_err(|e| format!("[{}] {}", spec.node_type, e))?;
        if !spec.spec.filter.is_empty() {
            apply_filter(&mut raw, &spec.spec.filter);
        }
        if raw.row_count() == 0 {
            continue;
        }
        if is_auto_pk {
            // Ids span auto_pk_counter .. + row_count, so the counter
            // advances by each chunk's post-filter row count and the total
            // assignment matches the buffered path's 1..=N.
            raw.headers.push(pk.clone());
            for r in 0..raw.row_count() {
                raw.rows[r].push(auto_pk_counter.to_string());
                raw.nulls[r].push(false);
                auto_pk_counter += 1;
            }
        }
        let keep = streaming_keep_list(&raw, &pk, &title_field, &skip_set, &parent_fk_skip);
        let df = typed_dataframe(&raw, &keep, &declared, &HashMap::new(), &mut misparses)
            .map_err(|e| format!("[{}] {}", spec.node_type, e))?;
        let rep = maintain::add_nodes(
            graph,
            df,
            spec.node_type.clone(),
            pk.clone(),
            title_arg.clone(),
            None,
        )
        .map_err(|e| format!("add_nodes '{}': {}", spec.node_type, e))?;
        let count = rep.nodes_created + rep.nodes_updated;
        *report
            .nodes_by_type
            .entry(spec.node_type.clone())
            .or_insert(0) += count;
    }
    report
        .warnings
        .extend(misparses.into_warnings(&format!("node '{}'", spec.node_type)));
    Ok(())
}

/// Default streaming chunk size for node inputs. ~250K rows × ~15
/// cols × ~30B avg string ≈ 110 MB peak per chunk — bounds RAM
/// for large CSVs without paying the multi-chunk dispatch
/// overhead on common medium files (1-spec, ≤250K rows fits in
/// one chunk so the streaming path matches the buffered path
/// in `add_nodes` / `connect()` call count).
///
/// The junction-edge loader keeps its own 100K default because
/// junction CSVs typically span far more rows and have tighter
/// per-row memory than node CSVs.
pub(super) fn node_chunk_size() -> usize {
    std::env::var("KGLITE_BLUEPRINT_NODE_CHUNK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(250_000)
}

/// Build the keep-list for the streaming node loader. Mirrors the
/// buffered keep-list in `prep_node_spec` minus the timeseries-only
/// exclusion (streaming specs never have timeseries).
fn streaming_keep_list(
    raw: &RawCsv,
    pk: &str,
    title_field: &str,
    skip_set: &HashSet<&String>,
    parent_fk_skip: &HashSet<String>,
) -> Vec<String> {
    let geometry_passthrough: HashSet<&str> = HashSet::from_iter(["_geometry"]);
    let keep: Vec<String> = raw
        .headers
        .iter()
        .filter(|h| {
            !skip_set.contains(h)
                && !geometry_passthrough.contains(h.as_str())
                && !parent_fk_skip.contains(h.as_str())
                || h.as_str() == pk
                || h.as_str() == title_field
        })
        .cloned()
        .collect();
    let mut seen = HashSet::new();
    keep.into_iter()
        .filter(|h| seen.insert(h.clone()))
        .collect()
}

fn apply_timeseries(
    graph: &mut DirGraph,
    node_type: &str,
    pk_col: &str,
    raw: &RawCsv,
    resolved: &ts::ResolvedTimeseries,
) -> Result<(), String> {
    let per_node = ts::build_node_timeseries(raw, pk_col, resolved)?;

    graph.build_id_index(node_type);
    for (key_str, node_ts) in per_node {
        let str_val = crate::datatypes::values::Value::String(key_str.clone());
        let node_idx = graph
            .lookup_by_id_normalized(node_type, &str_val)
            .or_else(|| {
                key_str.parse::<i64>().ok().and_then(|i| {
                    graph.lookup_by_id_normalized(
                        node_type,
                        &crate::datatypes::values::Value::Int64(i),
                    )
                })
            });
        let Some(idx) = node_idx else { continue };
        graph.timeseries_store.insert(idx.index(), node_ts);
    }

    let merged = ts::merge_config(graph.timeseries_configs.get(node_type), resolved);
    graph
        .timeseries_configs
        .insert(node_type.to_string(), merged);
    Ok(())
}
