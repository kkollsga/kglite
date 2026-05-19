//! Blueprint build orchestrator: JSON + CSVs → populated `DirGraph`.
//!
//! Phase order mirrors the Python loader:
//!   1. Manual nodes — types without a CSV, synthesised from FK values
//!      referring to that type.
//!   2. Core nodes — top-level node types with CSVs.
//!   3. Sub-nodes — types declared inside a parent spec's `sub_nodes`.
//!   4. FK edges — single-column foreign keys on node CSVs (plus
//!      implicit `parent` → `OF_{PARENT}` edges).
//!   5. Junction edges — many-to-many CSVs with two FK columns + optional
//!      property columns.

use super::csv_loader::{
    map_blueprint_type, read_csv_chunks, read_csv_raw, typed_dataframe, RawCsv,
};
use super::filter::apply_filter;
use super::geometry::{convert_geojson, has_spatial_properties, spatial_targets};
use super::schema::{Blueprint, NodeSpec};
use super::timeseries as ts;
use crate::datatypes::values::DataFrame;
use crate::graph::mutation::maintain;
use crate::graph::schema::{DirGraph, SpatialConfig};
use indexmap::IndexMap;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

pub struct BuildReport {
    pub nodes_by_type: BTreeMap<String, usize>,
    pub edges_by_type: BTreeMap<String, usize>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

pub fn build(
    graph: &mut DirGraph,
    blueprint: Blueprint,
    blueprint_dir: &Path,
) -> Result<BuildReport, String> {
    let root = blueprint
        .settings
        .input_root
        .as_deref()
        .map(|r| {
            if Path::new(r).is_absolute() {
                PathBuf::from(r)
            } else {
                blueprint_dir.join(r)
            }
        })
        .unwrap_or_else(|| blueprint_dir.to_path_buf());

    let mut report = BuildReport {
        nodes_by_type: BTreeMap::new(),
        edges_by_type: BTreeMap::new(),
        warnings: Vec::new(),
        errors: Vec::new(),
    };

    let profile = std::env::var("KGLITE_BLUEPRINT_PROFILE").is_ok();
    let t0 = std::time::Instant::now();

    let (core_specs, sub_specs) = collect_specs(&blueprint.nodes);
    if profile {
        eprintln!(
            "  collect_specs: {} ms ({} core + {} sub)",
            t0.elapsed().as_millis(),
            core_specs.len(),
            sub_specs.len()
        );
    }

    // Phase 0: pre-parse node + sub-node CSV paths in parallel so later
    // phases hit the cache without blocking on disk I/O.
    //
    // E3 note: junction-edge CSVs are NOT pre-parsed any more. They're
    // streamed via `read_csv_chunks` inside `load_junction_edges`. Pre-
    // caching them would defeat the streaming memory bound.
    let csv_cache: CsvCache = CsvCache::default();
    let mut all_csv_paths: Vec<String> = Vec::new();
    for s in core_specs.iter().chain(sub_specs.iter()) {
        if let Some(p) = s.spec.csv.as_deref() {
            all_csv_paths.push(p.to_string());
        }
    }
    all_csv_paths.sort();
    all_csv_paths.dedup();
    let t_preparse = std::time::Instant::now();
    parse_in_parallel(&all_csv_paths, &root, &csv_cache);
    if profile {
        eprintln!(
            "  parse_in_parallel: {} ms ({} distinct files)",
            t_preparse.elapsed().as_millis(),
            all_csv_paths.len()
        );
    }

    // Phase 1: manual nodes.
    let t = std::time::Instant::now();
    load_manual_nodes(graph, &core_specs, &sub_specs, &root, &mut report)?;
    if profile {
        eprintln!("  load_manual_nodes: {} ms", t.elapsed().as_millis());
    }

    let t = std::time::Instant::now();
    load_node_specs(
        graph,
        &core_specs,
        &root,
        &csv_cache,
        &mut report,
        "core nodes",
    )?;
    if profile {
        eprintln!("  load_core_nodes: {} ms", t.elapsed().as_millis());
    }
    let t = std::time::Instant::now();
    load_node_specs(
        graph,
        &sub_specs,
        &root,
        &csv_cache,
        &mut report,
        "sub-nodes",
    )?;
    if profile {
        eprintln!("  load_sub_nodes: {} ms", t.elapsed().as_millis());
    }

    // Register parent types for sub-nodes
    for sub in &sub_specs {
        if let Some(parent) = &sub.parent {
            if graph.type_indices.contains_key(&sub.node_type)
                && graph.type_indices.contains_key(parent)
            {
                graph
                    .parent_types
                    .insert(sub.node_type.clone(), parent.clone());
            }
        }
    }

    // Phase 4: FK edges
    let all_specs: Vec<&FlatSpec> = core_specs.iter().chain(sub_specs.iter()).collect();
    let t = std::time::Instant::now();
    load_fk_edges(graph, &all_specs, &root, &csv_cache, &mut report)?;
    if profile {
        eprintln!("  load_fk_edges: {} ms", t.elapsed().as_millis());
    }

    // Phase 5: junction edges
    let t = std::time::Instant::now();
    load_junction_edges(graph, &all_specs, &root, &csv_cache, &mut report)?;
    if profile {
        eprintln!("  load_junction_edges: {} ms", t.elapsed().as_millis());
        eprintln!("  TOTAL build: {} ms", t0.elapsed().as_millis());
    }

    Ok(report)
}

// ─── Spec flattening ──────────────────────────────────────────────────────

/// Flattened view of one node spec with parent info carried along.
pub struct FlatSpec {
    pub node_type: String,
    pub spec: NodeSpec,
    pub parent: Option<String>,
    pub is_manual: bool,
}

fn collect_specs(nodes: &IndexMap<String, NodeSpec>) -> (Vec<FlatSpec>, Vec<FlatSpec>) {
    let mut core = Vec::new();
    let mut subs = Vec::new();
    for (name, spec) in nodes {
        let is_manual = spec.csv.is_none();
        core.push(FlatSpec {
            node_type: name.clone(),
            spec: clone_without_subs(spec),
            parent: None,
            is_manual,
        });
        for (sub_name, sub_spec) in &spec.sub_nodes {
            // Sub-nodes keep their raw `parent` field untouched — the
            // enclosing type name is recorded on `FlatSpec.parent` so we
            // can call `set_parent_type` without also generating an
            // implicit OF_PARENT edge (that is reserved for top-level
            // specs that explicitly declare `parent` + `parent_fk`).
            let sub_clone = clone_without_subs(sub_spec);
            subs.push(FlatSpec {
                node_type: sub_name.clone(),
                spec: sub_clone,
                parent: Some(name.clone()),
                is_manual: false,
            });
        }
    }
    (core, subs)
}

fn clone_without_subs(spec: &NodeSpec) -> NodeSpec {
    NodeSpec {
        csv: spec.csv.clone(),
        pk: spec.pk.clone(),
        title: spec.title.clone(),
        parent: spec.parent.clone(),
        parent_fk: spec.parent_fk.clone(),
        properties: spec.properties.clone(),
        skipped: spec.skipped.clone(),
        filter: spec.filter.clone(),
        connections: super::schema::Connections {
            fk_edges: spec
                .connections
                .fk_edges
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        super::schema::FkEdge {
                            target: v.target.clone(),
                            fk: v.fk.clone(),
                        },
                    )
                })
                .collect(),
            junction_edges: spec
                .connections
                .junction_edges
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        super::schema::JunctionEdge {
                            csv: v.csv.clone(),
                            source_fk: v.source_fk.clone(),
                            target: v.target.clone(),
                            target_fk: v.target_fk.clone(),
                            properties: v.properties.clone(),
                            property_types: v.property_types.clone(),
                        },
                    )
                })
                .collect(),
        },
        sub_nodes: IndexMap::new(),
        timeseries: spec
            .timeseries
            .as_ref()
            .map(|t| super::schema::TimeseriesSpec {
                time_key: match &t.time_key {
                    super::schema::TimeKey::Single(s) => super::schema::TimeKey::Single(s.clone()),
                    super::schema::TimeKey::Composite(m) => {
                        super::schema::TimeKey::Composite(m.clone())
                    }
                },
                channels: t.channels.clone(),
                resolution: t.resolution.clone(),
                units: t.units.clone(),
            }),
    }
}

// ─── CSV cache ────────────────────────────────────────────────────────────

/// Cache of raw CSVs keyed by relative path. Populated in parallel at the
/// start of the build (see `parse_in_parallel`) so serial phases that read
/// the same CSV (e.g. node load + FK edges + junction edges) never block
/// on disk.
#[derive(Default)]
struct CsvCache {
    inner: std::sync::Mutex<HashMap<String, std::sync::Arc<RawCsv>>>,
}

impl CsvCache {
    fn get(&self, root: &Path, rel: &str) -> Result<std::sync::Arc<RawCsv>, String> {
        {
            let guard = self.inner.lock().unwrap();
            if let Some(hit) = guard.get(rel) {
                return Ok(hit.clone());
            }
        }
        let full = root.join(rel);
        let raw = read_csv_raw(&full)?;
        let arc = std::sync::Arc::new(raw);
        self.inner
            .lock()
            .unwrap()
            .insert(rel.to_string(), arc.clone());
        Ok(arc)
    }

    fn insert(&self, rel: &str, raw: RawCsv) {
        self.inner
            .lock()
            .unwrap()
            .insert(rel.to_string(), std::sync::Arc::new(raw));
    }
}

/// Parse all given CSV paths in parallel, populating the cache. Failures
/// are silently skipped — the caller will see the `Err` again when it tries
/// to look up that path serially (and can emit a targeted error then).
fn parse_in_parallel(paths: &[String], root: &Path, cache: &CsvCache) {
    use rayon::prelude::*;
    paths.par_iter().for_each(|rel| {
        let full = root.join(rel);
        if let Ok(raw) = read_csv_raw(&full) {
            cache.insert(rel, raw);
        }
    });
}

// ─── Phase 1: manual nodes ────────────────────────────────────────────────

fn load_manual_nodes(
    graph: &mut DirGraph,
    core: &[FlatSpec],
    subs: &[FlatSpec],
    root: &Path,
    report: &mut BuildReport,
) -> Result<(), String> {
    let manual: Vec<&FlatSpec> = core.iter().filter(|s| s.is_manual).collect();
    if manual.is_empty() {
        return Ok(());
    }

    for ms in &manual {
        let mut distinct: HashSet<String> = HashSet::new();
        // Scan every spec's fk_edges for targets pointing at this manual type.
        for spec in core.iter().chain(subs.iter()) {
            let Some(csv) = spec.spec.csv.as_deref() else {
                continue;
            };
            for (_, edge) in &spec.spec.connections.fk_edges {
                if edge.target != ms.node_type {
                    continue;
                }
                let full = root.join(csv);
                let raw = match read_csv_raw(&full) {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                if let Some(fk_idx) = raw.col_index(&edge.fk) {
                    for (r, row) in raw.rows.iter().enumerate() {
                        if raw.nulls[r][fk_idx] {
                            continue;
                        }
                        let trimmed = row[fk_idx].trim();
                        if !trimmed.is_empty() {
                            distinct.insert(trimmed.to_string());
                        }
                    }
                }
            }
        }

        if distinct.is_empty() {
            continue;
        }

        let pk = ms.spec.pk.clone().unwrap_or_else(|| "name".to_string());
        let title = ms.spec.title.clone().unwrap_or_else(|| pk.clone());

        // Build a tiny single-column (or two-column) DataFrame by hand.
        let mut df = DataFrame::new(Vec::new());
        let values: Vec<String> = distinct.into_iter().collect();
        let col_type_strings = vec![Some(String::from("string")); values.len()]
            .iter()
            .all(|_| true);
        let _ = col_type_strings; // silence unused
        let data = crate::datatypes::values::ColumnData::String(
            values.iter().cloned().map(Some).collect(),
        );
        df.add_column(
            pk.clone(),
            crate::datatypes::values::ColumnType::String,
            data,
        )
        .map_err(|e| format!("manual nodes: {}", e))?;
        if title != pk {
            let data2 = crate::datatypes::values::ColumnData::String(
                values.into_iter().map(Some).collect(),
            );
            df.add_column(
                title.clone(),
                crate::datatypes::values::ColumnType::String,
                data2,
            )
            .map_err(|e| format!("manual nodes: {}", e))?;
        }

        let title_field = if title != pk {
            Some(title.clone())
        } else {
            None
        };
        let result = maintain::add_nodes(graph, df, ms.node_type.clone(), pk, title_field, None)
            .map_err(|e| format!("manual nodes '{}': {}", ms.node_type, e))?;

        let count = result.nodes_created + result.nodes_updated;
        report.nodes_by_type.insert(ms.node_type.clone(), count);
    }

    Ok(())
}

// ─── Phase 2 + 3: node loading ────────────────────────────────────────────

/// Everything a single node spec produces — computed off-thread by
/// `prep_node_spec`, then consumed sequentially by `load_node_specs`.
struct PreppedNode {
    node_type: String,
    pk: String,
    title_arg: Option<String>,
    df: DataFrame,
    spatial_config: Option<SpatialConfig>,
    /// Full raw (pre-dedup) CSV + resolved timeseries spec, if this type
    /// has `timeseries` declared. Kept because `apply_timeseries` needs
    /// every row, not just the dedup'd node DataFrame.
    timeseries: Option<(RawCsv, ts::ResolvedTimeseries)>,
}

fn prep_node_spec(
    spec: &FlatSpec,
    root: &Path,
    cache: &CsvCache,
) -> Result<Option<PreppedNode>, String> {
    if spec.is_manual {
        return Ok(None);
    }
    let Some(csv_rel) = spec.spec.csv.as_deref() else {
        return Ok(None);
    };
    let raw_rc = match cache.get(root, csv_rel) {
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

    // Handle pk: "auto"
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

    let df = typed_dataframe(&raw_for_nodes, &keep, &declared)?;

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
        timeseries,
    }))
}

fn load_node_specs(
    graph: &mut DirGraph,
    specs: &[FlatSpec],
    root: &Path,
    cache: &CsvCache,
    report: &mut BuildReport,
    _phase_name: &str,
) -> Result<(), String> {
    use rayon::prelude::*;
    let profile = std::env::var("KGLITE_BLUEPRINT_PROFILE").is_ok();

    // F1: split specs by streaming eligibility. Streamable specs run
    // through a per-chunk `read_csv_chunks → typed_dataframe → add_nodes`
    // loop that bounds peak RAM by chunk size. Buffered specs (timeseries,
    // spatial, pk:"auto", manual) keep the parallel-prep path because
    // they need random access to the full row set during prep.
    let (buffered, streamable): (Vec<&FlatSpec>, Vec<&FlatSpec>) =
        specs.iter().partition(|s| !is_streamable_node_spec(s));

    // Buffered path: parallel prep + serial dispatch (existing behaviour).
    let t_par = std::time::Instant::now();
    let prepped: Vec<Result<Option<PreppedNode>, String>> = buffered
        .par_iter()
        .map(|spec| prep_node_spec(spec, root, cache))
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

    // Streaming path: serial per-spec dispatch, per-chunk add_nodes.
    // Per-spec errors land in `report.errors` (parity with the buffered
    // path) — missing CSVs / type mismatches must not abort the build.
    let t_stream = std::time::Instant::now();
    for spec in &streamable {
        if let Err(e) = load_streamed_node_spec(graph, spec, root, report) {
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

/// True iff this spec can be loaded via `load_streamed_node_spec`.
///
/// Returns false for:
/// - manual specs (no CSV — synthesised from FK targets)
/// - timeseries specs (need full row set for grouping + dedup-by-pk)
/// - spatial specs (geometry conversion mutates RawCsv in-place)
/// - `pk: "auto"` specs (need row-count for synthesised id range —
///   F2 lifts this restriction via an `AtomicU64` counter)
fn is_streamable_node_spec(spec: &FlatSpec) -> bool {
    if spec.is_manual {
        return false;
    }
    if spec.spec.csv.is_none() {
        return false;
    }
    if spec.spec.timeseries.is_some() {
        return false;
    }
    if has_spatial_properties(&spec.spec.properties) {
        return false;
    }
    if spec.spec.pk.as_deref() == Some("auto") {
        return false;
    }
    true
}

/// Streaming node-spec loader: reads the CSV in chunks via
/// `read_csv_chunks`, applies `filter` + `typed_dataframe` per chunk,
/// then dispatches via `maintain::add_nodes`. `add_nodes` is
/// upsert-by-id (`nodes_created`/`nodes_updated`), so successive
/// chunks accumulate cleanly into the same node type without
/// resurrecting the per-spec working clone the buffered path needs.
fn load_streamed_node_spec(
    graph: &mut DirGraph,
    spec: &FlatSpec,
    root: &Path,
    report: &mut BuildReport,
) -> Result<(), String> {
    let Some(csv_rel) = spec.spec.csv.as_deref() else {
        return Ok(());
    };
    let csv_path = root.join(csv_rel);
    let chunk_size = node_chunk_size();
    let chunks = read_csv_chunks(&csv_path, chunk_size)
        .map_err(|e| format!("[{}] {}", spec.node_type, e))?;

    let pk = spec.spec.pk.clone().unwrap_or_else(|| "id".to_string());
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

    for chunk_result in chunks {
        let mut raw = chunk_result.map_err(|e| format!("[{}] {}", spec.node_type, e))?;
        if !spec.spec.filter.is_empty() {
            apply_filter(&mut raw, &spec.spec.filter);
        }
        if raw.row_count() == 0 {
            continue;
        }
        let keep = streaming_keep_list(&raw, &pk, &title_field, &skip_set, &parent_fk_skip);
        let df = typed_dataframe(&raw, &keep, &declared)
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
    Ok(())
}

/// Default streaming chunk size for node CSVs. ~100K rows × ~20 cols
/// × ~24B avg string ≈ 50 MB peak per chunk — well under any
/// reasonable RAM budget. Configurable via env var for perf experiments.
fn node_chunk_size() -> usize {
    std::env::var("KGLITE_BLUEPRINT_NODE_CHUNK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000)
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

// ─── Phase 4: FK edges ────────────────────────────────────────────────────

struct PreppedFkEdges {
    /// Source type (borrowed from the FlatSpec).
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
}

struct PreppedFkEdge {
    edge_type: String,
    target_type: String,
    target_col: String,
    df: DataFrame,
}

fn prep_fk_edges(spec: &FlatSpec, root: &Path, cache: &CsvCache) -> Option<PreppedFkEdges> {
    let csv_rel = spec.spec.csv.as_deref()?;

    let mut fk_edges: IndexMap<String, super::schema::FkEdge> = spec
        .spec
        .connections
        .fk_edges
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                super::schema::FkEdge {
                    target: v.target.clone(),
                    fk: v.fk.clone(),
                },
            )
        })
        .collect();
    if let (Some(parent_type), Some(parent_fk)) = (&spec.spec.parent, &spec.spec.parent_fk) {
        let edge_type = format!("OF_{}", parent_type.to_uppercase());
        fk_edges.entry(edge_type).or_insert(super::schema::FkEdge {
            target: parent_type.clone(),
            fk: parent_fk.clone(),
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

        let (target_col, src_vals, tgt_vals) =
            build_fk_columns(&raw, &pk, &edge.fk, pk_idx, fk_idx);
        if src_vals.is_empty() {
            continue;
        }

        let Ok(df) = build_edge_df(&pk, &target_col, src_vals, tgt_vals) else {
            errors.push(format!(
                "[{}] failed to build edge DataFrame for {}",
                spec.node_type, edge_type
            ));
            continue;
        };
        built.push(PreppedFkEdge {
            edge_type: edge_type.clone(),
            target_type: edge.target.clone(),
            target_col,
            df,
        });
    }

    Some(PreppedFkEdges {
        source_type: spec.node_type.clone(),
        pk,
        edges: built,
        errors,
    })
}

fn build_fk_columns(
    raw: &RawCsv,
    pk: &str,
    fk: &str,
    pk_idx: usize,
    fk_idx: usize,
) -> (String, Vec<Option<String>>, Vec<Option<String>>) {
    // Keep only rows with a non-null target id (matches Python at loader.py:468).
    if pk == fk {
        // Self-reference: synthesise _target_{fk} so source/target column names differ.
        let target_col = format!("_target_{}", fk);
        let mut src = Vec::new();
        let mut tgt = Vec::new();
        for (r, row) in raw.rows.iter().enumerate() {
            if raw.nulls[r][pk_idx] {
                continue;
            }
            src.push(Some(row[pk_idx].clone()));
            tgt.push(Some(row[pk_idx].clone()));
        }
        (target_col, src, tgt)
    } else {
        let mut src = Vec::new();
        let mut tgt = Vec::new();
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
        }
        (fk.to_string(), src, tgt)
    }
}

fn load_fk_edges(
    graph: &mut DirGraph,
    specs: &[&FlatSpec],
    root: &Path,
    cache: &CsvCache,
    report: &mut BuildReport,
) -> Result<(), String> {
    use rayon::prelude::*;
    let profile = std::env::var("KGLITE_BLUEPRINT_PROFILE").is_ok();

    // Parallel prep: per-spec CSV clone + filter + pk handling + per-edge
    // DataFrame construction. Mutation is serial below.
    let t_par = std::time::Instant::now();
    let prepped: Vec<Option<PreppedFkEdges>> = specs
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
            )?;
            t_connect += t_c.elapsed();
            *report
                .edges_by_type
                .entry(edge.edge_type.clone())
                .or_insert(0) += count;
        }
    }

    if profile {
        eprintln!(
            "    fk parallel prep: {} ms | serial connect: {} ms | serial total: {} ms",
            t_par_ms,
            t_connect.as_millis(),
            t_serial.elapsed().as_millis(),
        );
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

fn infer_id_type(vals: &[Option<String>]) -> crate::datatypes::values::ColumnType {
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

fn add_id_column(
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
                .map(|v| v.and_then(|s| if s.is_empty() { None } else { Some(s) }))
                .collect(),
        ),
    };
    df.add_column(name.to_string(), col_type, data)
}

#[allow(clippy::too_many_arguments)]
fn connect(
    graph: &mut DirGraph,
    df: DataFrame,
    connection_type: &str,
    source_type: &str,
    source_id_field: &str,
    target_type: &str,
    target_id_field: &str,
    report: &mut BuildReport,
) -> Result<usize, String> {
    match maintain::add_connections(
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
    ) {
        Ok(r) => {
            if r.connections_skipped > 0 {
                let detail = r.errors.join("; ");
                report.warnings.push(format!(
                    "[{}] -[{}]-> {}: {} skipped ({})",
                    source_type, connection_type, target_type, r.connections_skipped, detail
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

// ─── Phase 5: junction edges (streaming, E3+) ─────────────────────────────
//
// The pre-E3 `prep_junction_edges` (cached buffered prep) was removed
// in favour of `load_junction_edges`' inline streaming loop above. The
// streaming path bounds peak RAM at chunk_size × cols × avg_string_len
// regardless of the junction CSV's total size — critical for the
// 10M+ row junction tables (e.g. SEC HOLDS at full-universe scale).

/// Junction-edge chunk size. ~100K rows × ~10 columns × ~20B avg
/// string ≈ 20 MB peak per chunk, well under any reasonable RAM
/// budget for the build host. Configurable via env var for
/// performance experiments.
fn junction_chunk_size() -> usize {
    std::env::var("KGLITE_BLUEPRINT_JUNCTION_CHUNK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000)
}

fn load_junction_edges(
    graph: &mut DirGraph,
    specs: &[&FlatSpec],
    root: &Path,
    _cache: &CsvCache,
    report: &mut BuildReport,
) -> Result<(), String> {
    let chunk_size = junction_chunk_size();
    let profile = std::env::var("KGLITE_BLUEPRINT_PROFILE").is_ok();
    let t_total = std::time::Instant::now();

    // E3: stream junction CSVs in chunks. For each (spec, junction edge):
    //   open chunked iterator → per chunk: typed_dataframe + connect()
    //   The OS page cache handles any repeat reads (none here — each
    //   junction CSV is only read once).
    // The legacy CsvCache + parse_in_parallel path is no longer used
    // for junctions; node specs still rely on the cache.
    for spec in specs {
        for (edge_type, junc) in &spec.spec.connections.junction_edges {
            let csv_path = root.join(&junc.csv);

            // Build the keep-columns list once per junction; we'll
            // apply it to every chunk.
            let mut keep: Vec<String> = vec![junc.source_fk.clone(), junc.target_fk.clone()];
            for p in &junc.properties {
                if !keep.contains(p) {
                    keep.push(p.clone());
                }
            }
            let mut declared: HashMap<String, String> = HashMap::new();
            for (col, ty) in &junc.property_types {
                if map_blueprint_type(ty).is_some() {
                    declared.insert(col.clone(), ty.clone());
                }
            }

            let chunks = match read_csv_chunks(&csv_path, chunk_size) {
                Ok(it) => it,
                Err(e) => {
                    report.errors.push(format!("junction {}: {}", edge_type, e));
                    continue;
                }
            };

            for chunk_result in chunks {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        report.errors.push(format!("junction {}: {}", edge_type, e));
                        continue;
                    }
                };
                // Only keep columns present in the chunk's headers.
                let chunk_keep: Vec<String> = keep
                    .iter()
                    .filter(|p| chunk.col_index(p).is_some())
                    .cloned()
                    .collect();
                if chunk_keep.is_empty() {
                    continue;
                }
                let df = match typed_dataframe(&chunk, &chunk_keep, &declared) {
                    Ok(df) => df,
                    Err(e) => {
                        report.errors.push(format!("junction {}: {}", edge_type, e));
                        continue;
                    }
                };
                let count = connect(
                    graph,
                    df,
                    edge_type,
                    &spec.node_type,
                    &junc.source_fk,
                    &junc.target,
                    &junc.target_fk,
                    report,
                )?;
                *report.edges_by_type.entry(edge_type.clone()).or_insert(0) += count;
            }
        }
    }

    if profile {
        eprintln!(
            "    streaming junction edges total: {} ms (chunk_size={})",
            t_total.elapsed().as_millis(),
            chunk_size,
        );
    }
    Ok(())
}

// ─── Helpers ──────────────────────────────────────────────────────────────

impl RawCsv {
    fn clone_raw(&self) -> RawCsv {
        RawCsv {
            headers: self.headers.clone(),
            rows: self.rows.clone(),
            nulls: self.nulls.clone(),
        }
    }
}

/// Keep only the first row per unique pk value. Used for timeseries specs:
/// one node per carrier, time samples stored separately.
fn dedupe_by_pk(raw: &RawCsv, pk_col: &str) -> RawCsv {
    let Some(idx) = raw.col_index(pk_col) else {
        return raw.clone_raw();
    };
    let mut seen: HashSet<String> = HashSet::new();
    let mut new_rows = Vec::new();
    let mut new_nulls = Vec::new();
    for r in 0..raw.row_count() {
        if raw.nulls[r][idx] {
            new_rows.push(raw.rows[r].clone());
            new_nulls.push(raw.nulls[r].clone());
            continue;
        }
        let key = raw.rows[r][idx].clone();
        if seen.insert(key) {
            new_rows.push(raw.rows[r].clone());
            new_nulls.push(raw.nulls[r].clone());
        }
    }
    RawCsv {
        headers: raw.headers.clone(),
        rows: new_rows,
        nulls: new_nulls,
    }
}
