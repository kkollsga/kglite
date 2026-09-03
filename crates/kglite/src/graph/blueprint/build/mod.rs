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
//!
//! Each phase lives in its own submodule; this file holds only the
//! orchestration, the report type and the two whole-build passes
//! (`stamp_declared_labels`, `apply_ontology_gate`).

mod cache;
mod fk;
mod junction;
mod manual;
mod nodes;
mod specs;
mod table_ops;

pub use specs::FlatSpec;

use cache::{parse_in_parallel, CsvCache};
use fk::load_fk_edges;
use junction::load_junction_edges;
use manual::load_manual_nodes;
use nodes::{load_node_specs, should_stream_spec};
use specs::collect_specs;

use super::schema::Blueprint;
use crate::graph::mutation::maintain;
use crate::graph::schema::{DirGraph, PROVISIONAL_KEY};
use std::collections::BTreeMap;
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
