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
//!   6. Provisional purge — under `settings.auto_purge`, stub nodes that
//!      no real row ever promoted are dropped with their edges.

use super::csv_loader::{
    map_blueprint_type, read_csv_chunks, read_csv_raw, typed_dataframe, ListMisparseTally, RawCsv,
};
use super::filter::apply_filter;
use super::geometry::{convert_geojson, has_spatial_properties, spatial_targets};
use super::schema::{Blueprint, NodeSpec};
use super::timeseries as ts;
use crate::datatypes::values::DataFrame;
use crate::graph::mutation::maintain;
use crate::graph::schema::{DirGraph, SpatialConfig, PROVISIONAL_KEY};
use indexmap::IndexMap;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

pub struct BuildReport {
    pub nodes_by_type: BTreeMap<String, usize>,
    pub edges_by_type: BTreeMap<String, usize>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    /// Provisional stub nodes dropped by `settings.auto_purge`.
    pub provisional_purged: usize,
}

pub fn build(
    graph: &mut DirGraph,
    mut blueprint: Blueprint,
    blueprint_dir: &Path,
) -> Result<BuildReport, String> {
    // Validate the compute pipeline before any phase touches data, so bad
    // expressions / dangling type refs / misplaced aggregate functions fail
    // at load time rather than midway through the build.
    super::validation::validate_compute(&blueprint)?;

    // Captured before `apply_compute` rewrites the blueprint: the warning is
    // about what the *author* wrote, and a compute-derived spec inherits its
    // source's stray keys, which would report the same typo under a type name
    // that appears in no file.
    let unknown_keys = super::validation::unknown_key_warnings(&blueprint);

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

    // Compute primitives run as a CSV-shaping pre-phase: each op reads its
    // source CSV, writes `<root>/computed/*.csv`, and repoints the blueprint
    // at the new files, so the load phases below consume the augmented
    // blueprint as if compute didn't exist.
    super::compute::apply_compute(&mut blueprint, &root)?;

    let mut report = BuildReport {
        nodes_by_type: BTreeMap::new(),
        edges_by_type: BTreeMap::new(),
        warnings: Vec::new(),
        errors: Vec::new(),
        provisional_purged: 0,
    };
    report.warnings.extend(unknown_keys);
    report
        .warnings
        .extend(super::validation::unknown_property_type_warnings(
            &blueprint,
        ));

    let profile = std::env::var("KGLITE_BLUEPRINT_PROFILE").is_ok();
    let t0 = std::time::Instant::now();

    let (core_specs, sub_specs) = collect_specs(&blueprint.nodes);
    // `_provisional` is the reserved auto-vivification marker — a node
    // spec must not declare a property of that name.
    for spec in core_specs.iter().chain(sub_specs.iter()) {
        if spec.spec.properties.contains_key(PROVISIONAL_KEY) {
            return Err(format!(
                "node type '{}': property '{}' is reserved (auto-vivification marker)",
                spec.node_type, PROVISIONAL_KEY
            ));
        }
    }
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
    // Junction-edge CSVs and streamable node specs are deliberately
    // excluded — both are read on demand via `read_csv_chunks`, and
    // pre-caching them would pin a full `RawCsv` for the whole build,
    // re-introducing the RAM ceiling streaming exists to avoid.
    let csv_cache: CsvCache = CsvCache::default();
    let mut buffered_csv_paths: Vec<String> = Vec::new();
    for s in core_specs.iter().chain(sub_specs.iter()) {
        if should_stream_spec(s, &root) {
            continue;
        }
        if let Some(p) = s.spec.csv.as_deref() {
            buffered_csv_paths.push(p.to_string());
        }
    }
    buffered_csv_paths.sort();
    buffered_csv_paths.dedup();
    let t_preparse = std::time::Instant::now();
    parse_in_parallel(&buffered_csv_paths, &root, &csv_cache);
    if profile {
        eprintln!(
            "  parse_in_parallel: {} ms ({} distinct files, streamed specs excluded)",
            t_preparse.elapsed().as_millis(),
            buffered_csv_paths.len()
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

    for sub in &sub_specs {
        if let Some(parent) = &sub.parent {
            if graph.type_indices.contains_key(&sub.node_type)
                && graph.type_indices.contains_key(parent)
            {
                graph
                    .parent_types_mut()
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
    }

    // Phase 6: a provisional stub that no real node row ever promoted is a
    // dangling reference; `auto_purge` discards it and its edges.
    if blueprint.settings.auto_purge {
        let t = std::time::Instant::now();
        let (purged, _edges) = maintain::purge_provisional_nodes(graph);
        report.provisional_purged = purged;
        if profile {
            eprintln!(
                "  purge_provisional: {} ms ({} purged)",
                t.elapsed().as_millis(),
                purged
            );
        }
    }
    // Phase 6b: declared secondary labels. After the edge phases and the
    // purge, not straight after the node phases: an endpoint a CSV never
    // provided is vivified as a stub *during* the edge phases, and a blueprint
    // owns every node of the types it declares — labelling earlier would make
    // `MATCH (:Place)` miss exactly the rows that arrived via an edge.
    let t = std::time::Instant::now();
    stamp_declared_labels(graph, &all_specs, &mut report)?;
    if profile {
        eprintln!("  stamp_declared_labels: {} ms", t.elapsed().as_millis());
    }

    // Phase 7: ontology install + gate. Runs after every load phase and
    // before the caller can save, so an `enforcement: error` violation
    // means no `.kgl` is ever written (the save lives in the callers).
    if let Some(ontology_path) = &blueprint.ontology {
        apply_ontology_gate(graph, ontology_path, blueprint_dir, &mut report)?;
    }

    if profile {
        eprintln!("  TOTAL build: {} ms", t0.elapsed().as_millis());
    }

    Ok(report)
}

/// Stamp each spec's declared `labels` on every node of its type.
///
/// One `add_node_labels_bulk` call per (type, label): the bulk path merges
/// into the label bucket once, where a per-node loop would be quadratic. The
/// primary type name is skipped by that call itself, so a spec listing its own
/// type among its labels is a no-op and not a duplicate.
fn stamp_declared_labels(
    graph: &mut DirGraph,
    specs: &[&FlatSpec],
    report: &mut BuildReport,
) -> Result<(), String> {
    for spec in specs {
        if spec.spec.labels.is_empty() {
            continue;
        }
        // Interning is fallible (name length / count ceilings), so validate
        // the whole set before stamping any of it: a half-labelled type is
        // worse than a refused build.
        maintain::preflight_interner_names(graph, spec.spec.labels.iter().map(String::as_str))
            .map_err(|e| format!("node '{}': labels: {}", spec.node_type, e))?;
        let Some(nodes) = graph.type_indices.get(&spec.node_type) else {
            report.warnings.push(format!(
                "node '{}': declares labels {:?} but the build produced no nodes of that type",
                spec.node_type, spec.spec.labels
            ));
            continue;
        };
        let indices: Vec<petgraph::graph::NodeIndex> = nodes.iter().collect();
        for label in &spec.spec.labels {
            let key = graph.interner.get_or_intern(label);
            graph.add_node_labels_bulk(&indices, key);
        }
    }
    Ok(())
}

/// Install the referenced ontology document and run the declaration audit
/// with per-declaration severity: `advisory` stays silent (available via
/// `CALL`s on demand), `warn` lands one summary line per rule in the build
/// report, `error` collects across ALL rules and then fails once — the
/// report-all-then-fail shape, so one rebuild fixes everything rather than
/// one rebuild per violation.
fn apply_ontology_gate(
    graph: &mut DirGraph,
    ontology_path: &str,
    blueprint_dir: &Path,
    report: &mut BuildReport,
) -> Result<(), String> {
    let resolved = if Path::new(ontology_path).is_absolute() {
        PathBuf::from(ontology_path)
    } else {
        blueprint_dir.join(ontology_path)
    };
    let text = std::fs::read_to_string(&resolved)
        .map_err(|e| format!("ontology: cannot read {}: {e}", resolved.display()))?;
    let store = crate::graph::ontology::ontology_from_json(&text)
        .map_err(|e| format!("ontology {}: {e}", resolved.display()))?;
    report.warnings.extend(
        graph
            .define_ontology(store)?
            .into_iter()
            .map(|w| format!("ontology: {w}")),
    );

    use crate::graph::languages::cypher::executor::ontology_procedures::audit_counts;
    use crate::graph::ontology::Enforcement;
    let mut errors: Vec<String> = Vec::new();
    for line in audit_counts(graph)? {
        if line.violations == 0 && line.exempted == 0 {
            continue;
        }
        // The exempted tail is reported even when it leaves zero violations:
        // an exemption that silently absorbs every flagged row would make a
        // passing gate indistinguishable from a clean graph.
        let tail = if line.exempted > 0 {
            format!(" (+{} exempted)", line.exempted)
        } else {
            String::new()
        };
        let summary = format!(
            "{}: {}/{} ({:.1}%) violations{tail}",
            line.rule, line.violations, line.total, line.pct
        );
        match line.severity {
            Enforcement::Advisory => {}
            Enforcement::Warn => report.warnings.push(format!("ontology: {summary}")),
            Enforcement::Error if line.violations > 0 => errors.push(summary),
            Enforcement::Error => report.warnings.push(format!("ontology: {summary}")),
        }
    }
    if !errors.is_empty() {
        return Err(format!(
            "ontology gate failed — {} contract(s) violated:\n  {}\nFix the data (or lower \
             the declaration's enforcement) and rebuild; drill into each rule with the \
             matching CALL procedure (e.g. CALL type_domain_violation()).",
            errors.len(),
            errors.join("\n  ")
        ));
    }
    Ok(())
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

/// The flattening pass's per-type copy: everything the spec declares except
/// its `sub_nodes`, which are flattened into their own `FlatSpec`s.
///
/// Struct-update syntax on purpose — a field-by-field copy silently drops any
/// field added to `NodeSpec` later, and the loss shows up as a directive the
/// blueprint declares and the build ignores.
fn clone_without_subs(spec: &NodeSpec) -> NodeSpec {
    NodeSpec {
        sub_nodes: IndexMap::new(),
        ..spec.clone()
    }
}

// ─── CSV cache ────────────────────────────────────────────────────────────

/// Cache of raw CSVs keyed by relative path. Populated in parallel at the
/// start of the build (see `parse_in_parallel`) so serial phases that read
/// the same CSV (node load + FK edges) never block on disk. Junction edges
/// bypass it entirely — see `load_junction_edges`.
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

        let mut df = DataFrame::new(Vec::new());
        let values: Vec<Option<String>> = distinct.into_iter().map(Some).collect();
        // The id column is typed by exactly the rule the FK-edge frame uses
        // (`infer_id_type`), because these two frames must agree: the edge
        // resolves its endpoint against the ids this call creates. Typed
        // `String` unconditionally, a numeric FK produced a `"10"` node here
        // and an unmatched `10` endpoint there — so the edge vivified a second
        // node and the type carried two per value.
        let id_type = infer_id_type(&values);
        add_id_column(&mut df, &pk, values.clone(), id_type)
            .map_err(|e| format!("manual nodes: {}", e))?;
        if title != pk {
            // The title is a display string whatever the id is.
            let data2 = crate::datatypes::values::ColumnData::String(values);
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

    // Streamable specs run a per-chunk `read_csv_chunks → typed_dataframe
    // → add_nodes` loop that bounds peak RAM by chunk size. Everything else
    // (timeseries, spatial, manual, and anything below the size threshold)
    // keeps the parallel-prep path — `should_stream_spec` carries why small
    // CSVs stay buffered.
    let (buffered, streamable): (Vec<&FlatSpec>, Vec<&FlatSpec>) =
        specs.iter().partition(|s| !should_stream_spec(s, root));

    // Buffered path: parallel prep, serial dispatch.
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

/// True iff this spec's row shape is compatible with the
/// streaming loader. Independent of file size — the size gate is
/// applied separately via `should_stream_spec`.
///
/// Returns false for:
/// - manual specs (no CSV — synthesised from FK targets)
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
    if spec.spec.csv.is_none() {
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
/// check (`is_streamable_node_spec`) with a file-size gate so
/// small/medium CSVs stay on the (faster) buffered path.
///
/// The streaming dispatch carries ~20% overhead per spec vs the
/// buffered parallel-prep on a single 500K-row CSV — fine on
/// 50M-row CSVs where the streaming RAM bound is the point, not
/// worth paying on the KB-to-~50MB CSVs typical of current graphs.
/// Threshold default: 100 MB, tunable via
/// `KGLITE_BLUEPRINT_STREAMING_THRESHOLD_MB`.
///
/// Falls back to the buffered path when the file can't be stat'd —
/// that path's error reporting handles missing files.
fn should_stream_spec(spec: &FlatSpec, root: &Path) -> bool {
    if !is_streamable_node_spec(spec) {
        return false;
    }
    let Some(csv_rel) = spec.spec.csv.as_deref() else {
        return false;
    };
    let path = root.join(csv_rel);
    match std::fs::metadata(&path) {
        Ok(m) => m.len() >= streaming_threshold_bytes(),
        Err(_) => false,
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

/// Default streaming chunk size for node CSVs. ~250K rows × ~15
/// cols × ~30B avg string ≈ 110 MB peak per chunk — bounds RAM
/// for large CSVs without paying the multi-chunk dispatch
/// overhead on common medium files (1-spec, ≤250K rows fits in
/// one chunk so the streaming path matches the buffered path
/// in `add_nodes` / `connect()` call count).
///
/// The junction-edge loader keeps its own 100K default because
/// junction CSVs typically span far more rows and have tighter
/// per-row memory than node CSVs.
fn node_chunk_size() -> usize {
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

// ─── Phase 4: FK edges ────────────────────────────────────────────────────

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

    let mut fk_edges: IndexMap<String, super::schema::FkEdge> = spec
        .spec
        .connections
        .fk_edges
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if let (Some(parent_type), Some(parent_fk)) = (&spec.spec.parent, &spec.spec.parent_fk) {
        let edge_type = format!("OF_{}", parent_type.to_uppercase());
        fk_edges.entry(edge_type).or_insert_with(|| {
            super::schema::FkEdge::plain(parent_type.clone(), parent_fk.clone())
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
    edge: &super::schema::FkEdge,
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
    edge: &super::schema::FkEdge,
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
        super::csv_loader::append_typed_columns(
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

/// A row-subset of `raw`, carrying the source row numbers so a diagnostic
/// still names a row the author can find.
fn subset_rows(raw: &RawCsv, rows: &[usize]) -> RawCsv {
    RawCsv {
        headers: raw.headers.clone(),
        rows: rows.iter().map(|&r| raw.rows[r].clone()).collect(),
        nulls: rows.iter().map(|&r| raw.nulls[r].clone()).collect(),
        row_ids: rows.iter().map(|&r| raw.row_id(r)).collect(),
    }
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

fn load_fk_edges(
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
    let mut fk_edges: IndexMap<String, super::schema::FkEdge> = spec
        .spec
        .connections
        .fk_edges
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if let (Some(parent_type), Some(parent_fk)) = (&spec.spec.parent, &spec.spec.parent_fk) {
        let edge_type = format!("OF_{}", parent_type.to_uppercase());
        fk_edges.entry(edge_type).or_insert_with(|| {
            super::schema::FkEdge::plain(parent_type.clone(), parent_fk.clone())
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
                .map(|v| v.filter(|s| !s.is_empty()))
                .collect(),
        ),
    };
    df.add_column(name.to_string(), col_type, data)
}

// Thin adapter: the parameter list mirrors `add_connections_with_initial_load`.
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

// ─── Phase 5: junction edges (streaming) ──────────────────────────────────
//
// Streaming bounds peak RAM at chunk_size × cols × avg_string_len whatever
// the junction CSV's total size — the 10M+ row junction tables (e.g. SEC
// HOLDS at full-universe scale) are why.

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

    // Junction CSVs are streamed, never cached: each is read exactly once,
    // so the `CsvCache` that node specs rely on would buy nothing here —
    // hence the unused `_cache` parameter.
    for spec in specs {
        for (edge_type, junc) in &spec.spec.connections.junction_edges {
            load_one_junction_edge(graph, spec, edge_type, junc, root, chunk_size, report)?;
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

/// Stream one junction CSV into edges. One `connect` call per (chunk, target
/// type) — a single-target junction, which is every one written before the
/// union form, has exactly one group and takes the same path it always did.
fn load_one_junction_edge(
    graph: &mut DirGraph,
    spec: &FlatSpec,
    edge_type: &str,
    junc: &super::schema::JunctionEdge,
    root: &Path,
    chunk_size: usize,
    report: &mut BuildReport,
) -> Result<(), String> {
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

    let rename = match junction_rename_map(edge_type, junc, &keep) {
        Ok(map) => map,
        Err(e) => {
            report.errors.push(e);
            return Ok(());
        }
    };
    let routing = match target_routing(edge_type, junc) {
        Ok(r) => r,
        Err(e) => {
            report.errors.push(e);
            return Ok(());
        }
    };

    // Decided before the first chunk and reused for all of them: streaming
    // this CSV bounds peak RAM, so it must not decide which of its rows become
    // parallel edges. Per-type splitting rides on the same decision — every
    // group of every chunk gets this one value. See `maintain::InitialLoad`.
    let initial_load =
        maintain::InitialLoad::Preset(!graph.connection_type_metadata.contains_key(edge_type));

    let csv_path = root.join(&junc.csv);
    let chunks = match read_csv_chunks(&csv_path, chunk_size) {
        Ok(it) => it,
        Err(e) => {
            report.errors.push(format!("junction {}: {}", edge_type, e));
            return Ok(());
        }
    };

    // One tally per junction CSV, not per chunk — see the node loader's for
    // why. Same for the unroutable-row counts.
    let mut misparses = ListMisparseTally::default();
    let mut unroutable: BTreeMap<String, usize> = BTreeMap::new();
    let mut reported_missing_type_column = false;

    for chunk_result in chunks {
        let chunk = match chunk_result {
            Ok(c) => c,
            Err(e) => {
                report.errors.push(format!("junction {}: {}", edge_type, e));
                continue;
            }
        };
        let chunk_keep: Vec<String> = keep
            .iter()
            .filter(|p| chunk.col_index(p).is_some())
            .cloned()
            .collect();
        if chunk_keep.is_empty() {
            continue;
        }
        if let TargetRouting::Column { column, .. } = &routing {
            if chunk.col_index(column).is_none() {
                if !reported_missing_type_column {
                    reported_missing_type_column = true;
                    report.errors.push(format!(
                        "junction {edge_type}: target_type_column '{column}' not found in the CSV"
                    ));
                }
                continue;
            }
        }

        for (target_type, rows) in
            group_rows_by_target(graph, &chunk, junc, &routing, &declared, &mut unroutable)
        {
            let whole_chunk = rows.len() == chunk.row_count();
            let subset;
            let source: &RawCsv = if whole_chunk {
                &chunk
            } else {
                subset = subset_rows(&chunk, &rows);
                &subset
            };
            let df = match typed_dataframe(source, &chunk_keep, &declared, &rename, &mut misparses)
            {
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
                target_type,
                &junc.target_fk,
                report,
                initial_load,
            )?;
            *report
                .edges_by_type
                .entry(edge_type.to_string())
                .or_insert(0) += count;
        }
    }
    report.warnings.extend(misparses.into_warnings(&format!(
        "junction '{edge_type}' (node '{}')",
        spec.node_type
    )));
    for (value, count) in unroutable {
        report.warnings.push(format!(
            "junction '{edge_type}' (node '{}'): {count} row(s) name target type '{value}', \
             which is not in this edge's 'target' list ({}); they built no edge",
            spec.node_type,
            junc.target.join(", ")
        ));
    }
    Ok(())
}

/// How each row of a junction CSV picks its target node type.
enum TargetRouting<'a> {
    /// One declared type: every row goes to it.
    Single(&'a str),
    /// A CSV column names the type per row.
    Column {
        types: &'a [String],
        column: &'a str,
    },
    /// No column: the declared types are probed in declaration order and the
    /// first that already has the row's target id wins.
    Probe { types: &'a [String] },
}

fn target_routing<'a>(
    edge_type: &str,
    junc: &'a super::schema::JunctionEdge,
) -> Result<TargetRouting<'a>, String> {
    if junc.target.is_empty() {
        return Err(format!("junction {edge_type}: 'target' names no node type"));
    }
    let Some(column) = junc.target_type_column.as_deref() else {
        return Ok(if junc.target.len() == 1 {
            TargetRouting::Single(&junc.target[0])
        } else {
            TargetRouting::Probe {
                types: &junc.target,
            }
        });
    };
    if column == junc.source_fk || column == junc.target_fk {
        return Err(format!(
            "junction {edge_type}: target_type_column '{column}' is an fk column — it names the \
             column holding each row's target *type*, not its id"
        ));
    }
    Ok(TargetRouting::Column {
        types: &junc.target,
        column,
    })
}

/// One chunk's rows grouped by the target type they route to, in declaration
/// order, skipping types no row picked.
///
/// A row naming a type outside the declaration is counted in `unroutable` and
/// left out: the loader cannot guess which of the declared types the author
/// meant, and picking one would attach the edge to a node of the wrong type
/// — or invent a stub under it. Under the probe form there is no such row: an
/// id no declared type has takes the first declared type, where the existing
/// missing-endpoint policy vivifies its stub, because dropping it would lose
/// an edge the author did declare.
fn group_rows_by_target<'a>(
    graph: &DirGraph,
    chunk: &RawCsv,
    junc: &'a super::schema::JunctionEdge,
    routing: &TargetRouting<'a>,
    declared: &HashMap<String, String>,
    unroutable: &mut BTreeMap<String, usize>,
) -> Vec<(&'a str, Vec<usize>)> {
    let types: &[String] = match routing {
        TargetRouting::Single(t) => return vec![(t, (0..chunk.row_count()).collect())],
        TargetRouting::Column { types, .. } | TargetRouting::Probe { types } => types,
    };
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); types.len()];

    match routing {
        TargetRouting::Single(_) => unreachable!("returned above"),
        TargetRouting::Column { column, .. } => {
            let col_idx = chunk.col_index(column).expect("caller checked presence");
            for r in 0..chunk.row_count() {
                let value = &chunk.rows[r][col_idx];
                match types.iter().position(|t| t == value) {
                    Some(i) => groups[i].push(r),
                    None => *unroutable.entry(value.clone()).or_default() += 1,
                }
            }
        }
        TargetRouting::Probe { .. } => {
            let ids = typed_target_ids(chunk, &junc.target_fk, declared);
            for (r, id) in ids.iter().enumerate() {
                let owner = id.as_ref().and_then(|v| {
                    types
                        .iter()
                        .position(|t| graph.id_indices.lookup(t, v).is_some())
                });
                groups[owner.unwrap_or(0)].push(r);
            }
        }
    }

    types
        .iter()
        .zip(groups)
        .filter(|(_, rows)| !rows.is_empty())
        .map(|(t, rows)| (t.as_str(), rows))
        .collect()
}

/// The chunk's target-id column, typed the way the edge frame will type it,
/// so a probe compares like with like: an id index keyed by `Int64` never
/// matches the raw `"10"` string the CSV holds.
fn typed_target_ids(
    chunk: &RawCsv,
    target_fk: &str,
    declared: &HashMap<String, String>,
) -> Vec<Option<crate::datatypes::values::Value>> {
    let keep = vec![target_fk.to_string()];
    // Discarded: the real frame types the same column again and tallies it
    // there, and counting a misparse twice would double the warning's count.
    let mut scratch = ListMisparseTally::default();
    match typed_dataframe(chunk, &keep, declared, &HashMap::new(), &mut scratch) {
        Ok(df) => (0..chunk.row_count())
            .map(|r| df.get_value_by_index(r, 0))
            .collect(),
        Err(_) => vec![None; chunk.row_count()],
    }
}

/// Validate a junction's `rename:` map and return it as the lookup
/// `typed_dataframe` takes. Keys must be property columns (fk columns keep
/// their spelling — `connect` finds them by name), and targets must not
/// collide with any kept column or another target.
fn junction_rename_map(
    edge_type: &str,
    junc: &crate::graph::blueprint::schema::JunctionEdge,
    keep: &[String],
) -> Result<HashMap<String, String>, String> {
    let mut rename: HashMap<String, String> = HashMap::new();
    for (col, new_name) in &junc.rename {
        if col == &junc.source_fk || col == &junc.target_fk {
            return Err(format!(
                "junction {edge_type}: rename of fk column '{col}' is not supported — \
                 source_fk/target_fk name the CSV columns"
            ));
        }
        if !junc.properties.contains(col) {
            return Err(format!(
                "junction {edge_type}: rename key '{col}' is not in 'properties'"
            ));
        }
        let collides = keep.iter().any(|k| k == new_name && k != col)
            || rename.values().any(|v| v == new_name);
        if collides {
            return Err(format!(
                "junction {edge_type}: rename target '{new_name}' collides with another column"
            ));
        }
        rename.insert(col.clone(), new_name.clone());
    }
    Ok(rename)
}

// ─── Helpers ──────────────────────────────────────────────────────────────

impl RawCsv {
    fn clone_raw(&self) -> RawCsv {
        RawCsv {
            headers: self.headers.clone(),
            rows: self.rows.clone(),
            nulls: self.nulls.clone(),
            row_ids: self.row_ids.clone(),
        }
    }
}

/// Keep only the first row per unique pk value; rows with a null pk all
/// pass through. Used for timeseries specs: one node per carrier, time
/// samples stored separately.
fn dedupe_by_pk(raw: &RawCsv, pk_col: &str) -> RawCsv {
    let Some(idx) = raw.col_index(pk_col) else {
        return raw.clone_raw();
    };
    let mut seen: HashSet<String> = HashSet::new();
    let mut new_rows = Vec::new();
    let mut new_nulls = Vec::new();
    let mut new_row_ids = Vec::new();
    for r in 0..raw.row_count() {
        if raw.nulls[r][idx] {
            new_rows.push(raw.rows[r].clone());
            new_nulls.push(raw.nulls[r].clone());
            new_row_ids.push(raw.row_id(r));
            continue;
        }
        let key = raw.rows[r][idx].clone();
        if seen.insert(key) {
            new_rows.push(raw.rows[r].clone());
            new_nulls.push(raw.nulls[r].clone());
            new_row_ids.push(raw.row_id(r));
        }
    }
    RawCsv {
        headers: raw.headers.clone(),
        rows: new_rows,
        nulls: new_nulls,
        row_ids: new_row_ids,
    }
}

#[cfg(test)]
#[path = "build_spec_clone_tests.rs"]
mod spec_clone_tests;
