//! Blueprint build orchestrator: JSON + input tables → populated `DirGraph`.
//!
//! Phase order mirrors the Python loader:
//!   1. Manual nodes — types without an input, synthesised from FK values
//!      referring to that type.
//!   2. Core nodes — top-level node types with inputs.
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
mod prepass;
mod specs;
mod table_ops;

pub use specs::FlatSpec;

use cache::{parse_in_parallel, CsvCache, IdTypeCache};
use fk::load_fk_edges;
use junction::load_junction_edges;
use manual::load_manual_nodes;
use nodes::{load_node_specs, should_stream_spec};
use specs::collect_specs;

use super::input::{csv::CsvFile, frame::FrameSource, resolve_input_path, InputRegistry};
use super::schema::{Blueprint, FileSpec};
use crate::datatypes::values::DataFrame;
use crate::graph::mutation::maintain;
use crate::graph::schema::{DirGraph, PROVISIONAL_KEY};
use indexmap::IndexMap;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// Inputs a build cannot read from disk, handed in by the caller.
///
/// A `files` entry declaring `"format": "frame"` binds to `frames[<entry
/// name>]`. The frame is consumed exactly like a read file — same specs,
/// filters, chunked junction loading, dedupe regime and warnings — by being
/// coerced to the blueprint's declared property types through the string
/// table; see [`super::input::frame`] for the canonical text rules. Files
/// stream; frames do not, because they were already materialised before the
/// build began.
#[derive(Default)]
pub struct BuildInputs {
    pub frames: HashMap<String, DataFrame>,
}

pub struct BuildReport {
    /// Edge writes the blueprint *attempted*, per type. With the default
    /// conflict handling a duplicate input row increments this and adds no
    /// edge, so it is the input-row tally, not the graph's.
    pub nodes_by_type: BTreeMap<String, usize>,
    pub edges_by_type: BTreeMap<String, usize>,
    /// Edges the built graph actually holds, per type — the number a
    /// `MATCH ()-[r]->() RETURN count(r)` returns. Read back once at the end
    /// of the build, because the difference from `edges_by_type` is the
    /// dedupe count every caller's report wants to name.
    pub edges_actual: BTreeMap<String, usize>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    /// Provisional stub nodes dropped by `settings.auto_purge`.
    pub provisional_purged: usize,
}

impl BuildReport {
    /// The build summary as text, in the shape the wheel has printed since
    /// 0.9.1. Empty when `verbose` is false — a silent build prints nothing.
    ///
    /// Lives here rather than in a wrapper because every binding formats the
    /// same numbers the same way; only *where* the text goes is per-binding.
    pub fn render_text(&self, verbose: bool) -> String {
        use std::fmt::Write;
        if !verbose {
            return String::new();
        }
        let mut out = String::new();
        let n_total: usize = self.nodes_by_type.values().sum();
        let e_actual: usize = self.edges_actual.values().sum();
        let e_input: usize = self.edges_by_type.values().sum();
        let _ = writeln!(out, "Loading blueprint...");
        for (t, n) in &self.nodes_by_type {
            let _ = writeln!(out, "  {}: {} nodes", t, n);
        }
        for (t, n_input) in &self.edges_by_type {
            let n_actual = self.edges_actual.get(t).copied().unwrap_or(0);
            if n_actual == *n_input {
                let _ = writeln!(out, "  [{}]: {} edges", t, n_actual);
            } else {
                let _ = writeln!(
                    out,
                    "  [{}]: {} edges ({} input rows, {} deduped)",
                    t,
                    n_actual,
                    n_input,
                    n_input.saturating_sub(n_actual),
                );
            }
        }
        if e_actual == e_input {
            let _ = writeln!(
                out,
                "Loaded {} nodes ({} types), {} edges ({} types)",
                n_total,
                self.nodes_by_type.len(),
                e_actual,
                self.edges_by_type.len(),
            );
        } else {
            let _ = writeln!(
                out,
                "Loaded {} nodes ({} types), {} edges ({} types) — {} input rows, {} deduped",
                n_total,
                self.nodes_by_type.len(),
                e_actual,
                self.edges_by_type.len(),
                e_input,
                e_input.saturating_sub(e_actual),
            );
        }
        if self.provisional_purged > 0 {
            let _ = writeln!(
                out,
                "  auto_purge: dropped {} unpromoted provisional stub node(s)",
                self.provisional_purged
            );
        }
        out
    }
}

pub fn build(
    graph: &mut DirGraph,
    mut blueprint: Blueprint,
    blueprint_dir: &Path,
    mut inputs: BuildInputs,
) -> Result<BuildReport, String> {
    // Validate the compute pipeline before any phase touches data, so bad
    // expressions / dangling type refs / misplaced aggregate functions fail
    // at load time rather than midway through the build.
    super::validation::validate_compute(&blueprint)?;
    // Every spec's input resolves to exactly one declared entry, before that
    // declaration becomes a registry below.
    super::validation::validate_inputs(&blueprint)?;

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
        edges_actual: BTreeMap::new(),
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

    // Phase 0: declare every input, then pre-read the buffered ones.
    let (registry, csv_cache, frame_warnings) = prepare_inputs(
        &blueprint.files,
        &core_specs,
        &sub_specs,
        &root,
        &mut inputs,
        profile,
    )?;
    report.warnings.extend(frame_warnings);
    // Filled by the streamed node phase, read by the streamed FK phase — see
    // `IdTypeCache`.
    let id_types = IdTypeCache::default();

    // Phase 1: manual nodes.
    let t = std::time::Instant::now();
    load_manual_nodes(
        graph,
        &core_specs,
        &sub_specs,
        &registry,
        &csv_cache,
        &mut report,
    )?;
    if profile {
        eprintln!("  load_manual_nodes: {} ms", t.elapsed().as_millis());
    }

    let t = std::time::Instant::now();
    load_node_specs(
        graph,
        &core_specs,
        &registry,
        &csv_cache,
        &id_types,
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
        &registry,
        &csv_cache,
        &id_types,
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
    load_fk_edges(
        graph,
        &all_specs,
        &registry,
        &csv_cache,
        &id_types,
        &mut report,
    )?;
    if profile {
        eprintln!("  load_fk_edges: {} ms", t.elapsed().as_millis());
    }

    // Phase 5: junction edges
    let t = std::time::Instant::now();
    load_junction_edges(graph, &all_specs, &registry, &csv_cache, &mut report)?;
    if profile {
        eprintln!("  load_junction_edges: {} ms", t.elapsed().as_millis());
    }

    finish_build(
        graph,
        &blueprint,
        blueprint_dir,
        &all_specs,
        &mut report,
        profile,
    )?;

    if profile {
        eprintln!("  TOTAL build: {} ms", t0.elapsed().as_millis());
    }

    Ok(report)
}

/// Phases 6-7 plus the report's read-back: everything that happens once the
/// load phases have put every row in the graph.
fn finish_build(
    graph: &mut DirGraph,
    blueprint: &Blueprint,
    blueprint_dir: &Path,
    all_specs: &[&FlatSpec],
    report: &mut BuildReport,
    profile: bool,
) -> Result<(), String> {
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
    stamp_declared_labels(graph, all_specs, report)?;
    if profile {
        eprintln!("  stamp_declared_labels: {} ms", t.elapsed().as_millis());
    }

    // Phase 7: ontology install + gate. Runs after every load phase and
    // before the caller can save, so an `enforcement: error` violation
    // means no `.kgl` is ever written (the save lives in the callers).
    if let Some(ontology_path) = &blueprint.ontology {
        apply_ontology_gate(graph, ontology_path, blueprint_dir, report)?;
    }

    // The graph's own counts, read back once: `edges_by_type` counts attempted
    // writes, and the difference is the dedupe number every caller's summary
    // reports. The accessor memoises, so the first later query reuses it.
    report.edges_actual = graph
        .get_edge_type_counts()
        .iter()
        .map(|(t, n)| (t.clone(), *n))
        .collect();
    Ok(())
}

/// Declare every input the flattened specs name and pre-read the buffered
/// ones in parallel, so serial phases that read the same input (node load +
/// FK edges) never block on I/O.
///
/// Junction-edge inputs and streamable node specs are deliberately excluded
/// from the pre-read — both are consumed on demand in chunks, and caching
/// them would pin a full `RawCsv` for the whole build, re-introducing the RAM
/// ceiling streaming exists to avoid.
fn prepare_inputs(
    files: &IndexMap<String, FileSpec>,
    core_specs: &[FlatSpec],
    sub_specs: &[FlatSpec],
    root: &Path,
    inputs: &mut BuildInputs,
    profile: bool,
) -> Result<(InputRegistry, CsvCache, Vec<String>), String> {
    let (registry, frame_warnings) =
        build_input_registry(files, core_specs, sub_specs, root, inputs)?;

    let cache = CsvCache::default();
    let mut buffered_inputs: Vec<String> = Vec::new();
    for s in core_specs.iter().chain(sub_specs.iter()) {
        if should_stream_spec(s, &registry) {
            continue;
        }
        if let Some(name) = s.input.as_deref() {
            buffered_inputs.push(name.to_string());
        }
    }
    buffered_inputs.sort();
    buffered_inputs.dedup();
    let t_preparse = std::time::Instant::now();
    parse_in_parallel(&buffered_inputs, &registry, &cache);
    if profile {
        eprintln!(
            "  parse_in_parallel: {} ms ({} distinct files, streamed specs excluded)",
            t_preparse.elapsed().as_millis(),
            buffered_inputs.len()
        );
    }
    Ok((registry, cache, frame_warnings))
}

/// Declare every input the build can read: one entry per `files` declaration,
/// plus one synthetic entry per `csv` shorthand, which is an input declared
/// inline and named by its own path.
///
/// Called after `apply_compute`, because that pre-phase repoints specs at the
/// files it generated (always as `csv` shorthands, so they arrive here as
/// synthetic entries like any other).
///
/// The synthetic key is the shorthand string exactly as the blueprint wrote
/// it, so two spellings of one file (`x.csv` and `./x.csv`) stay two inputs,
/// as they always have; the cache is keyed the same way and inherits the same
/// behaviour. Declarations are inserted first and the registry keeps the first
/// source per name — a shorthand naming a declared entry's file is that one
/// entry, and validation has already refused a name that would mean two
/// different files.
fn build_input_registry(
    files: &IndexMap<String, FileSpec>,
    core_specs: &[FlatSpec],
    sub_specs: &[FlatSpec],
    root: &Path,
    inputs: &mut BuildInputs,
) -> Result<(InputRegistry, Vec<String>), String> {
    let mut registry = InputRegistry::default();
    let mut frame_warnings: Vec<String> = Vec::new();
    for (name, file) in files {
        if file.format == "frame" {
            // Taken, not borrowed: the frame is this input's rows for the rest
            // of the build, and what is left over afterwards is a frame the
            // blueprint never declared.
            let Some(df) = inputs.frames.remove(name) else {
                return Err(format!(
                    "frame '{name}' is declared in `files` but was not passed in `frames=`"
                ));
            };
            let (source, warnings) = FrameSource::new(name, &df);
            frame_warnings.extend(warnings);
            registry.insert(name.clone(), Box::new(source));
            continue;
        }
        // A `path`-less entry of a file-backed format is refused by
        // `validate_inputs`.
        let Some(path) = file.path.as_deref() else {
            continue;
        };
        registry.insert(
            name.clone(),
            // `validate_inputs` has already checked the format is registered,
            // and `frame` was handled above, so what is left reads a file.
            Box::new(CsvFile::new(
                resolve_input_path(root, path),
                path.to_string(),
            )),
        );
    }
    if !inputs.frames.is_empty() {
        let mut names: Vec<&str> = inputs.frames.keys().map(String::as_str).collect();
        names.sort_unstable();
        let list = names
            .iter()
            .map(|n| format!("'{n}'"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "frames= contains {list}, which {} not declared in `files`",
            if names.len() == 1 { "is" } else { "are" }
        ));
    }
    let declare = |name: &str, registry: &mut InputRegistry| {
        registry.insert(
            name,
            Box::new(CsvFile::new(
                resolve_input_path(root, name),
                name.to_string(),
            )),
        );
    };
    for spec in core_specs.iter().chain(sub_specs.iter()) {
        if let Some(name) = spec.spec.csv.as_deref() {
            declare(name, &mut registry);
        }
        for junc in spec.spec.connections.junction_edges.values() {
            if let Some(name) = junc.csv.as_deref() {
                declare(name, &mut registry);
            }
        }
    }
    Ok((registry, frame_warnings))
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

#[cfg(test)]
mod frame_input_tests {
    use super::*;
    use crate::datatypes::values::{ColumnData, ColumnType};

    fn bp(json: &str) -> Blueprint {
        serde_json::from_str(json).expect("fixture parses")
    }

    fn one_int_frame() -> DataFrame {
        let mut df = DataFrame::new(Vec::new());
        df.add_column(
            "id".to_string(),
            ColumnType::Int64,
            ColumnData::Int64(vec![Some(1), Some(2)]),
        )
        .unwrap();
        df
    }

    fn frames(pairs: Vec<(&str, DataFrame)>) -> BuildInputs {
        BuildInputs {
            frames: pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        }
    }

    const ONE_FRAME_SPEC: &str = r#"{"files": {"rows": {"format": "frame"}},
        "nodes": {"Person": {"file": "rows", "pk": "id"}}}"#;

    #[test]
    fn a_declared_frame_that_was_not_passed_names_itself() {
        let mut graph = DirGraph::new();
        let err = build(
            &mut graph,
            bp(ONE_FRAME_SPEC),
            Path::new("."),
            BuildInputs::default(),
        )
        .err()
        .expect("a declared frame with no rows cannot build");
        assert_eq!(
            err,
            "frame 'rows' is declared in `files` but was not passed in `frames=`"
        );
    }

    #[test]
    fn a_passed_frame_that_was_not_declared_names_itself() {
        let mut graph = DirGraph::new();
        let err = build(
            &mut graph,
            bp(ONE_FRAME_SPEC),
            Path::new("."),
            frames(vec![("rows", one_int_frame()), ("extra", one_int_frame())]),
        )
        .err()
        .expect("a frame the blueprint never declared cannot be read");
        assert_eq!(
            err,
            "frames= contains 'extra', which is not declared in `files`"
        );
    }

    #[test]
    fn several_undeclared_frames_are_all_named() {
        let mut graph = DirGraph::new();
        let err = build(
            &mut graph,
            bp(ONE_FRAME_SPEC),
            Path::new("."),
            frames(vec![
                ("rows", one_int_frame()),
                ("z", one_int_frame()),
                ("a", one_int_frame()),
            ]),
        )
        .err()
        .expect("undeclared frames cannot be read");
        assert_eq!(
            err,
            "frames= contains 'a', 'z', which are not declared in `files`"
        );
    }

    /// The frame is bound by the `files` entry's name, not by anything in the
    /// spec — a build that found its rows produces its nodes.
    #[test]
    fn a_supplied_frame_builds_its_nodes() {
        let mut graph = DirGraph::new();
        let report = build(
            &mut graph,
            bp(ONE_FRAME_SPEC),
            Path::new("."),
            frames(vec![("rows", one_int_frame())]),
        )
        .unwrap_or_else(|e| panic!("a supplied frame builds: {e}"));
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        assert_eq!(report.nodes_by_type.get("Person"), Some(&2));
    }

    /// A column whose type the blueprint vocabulary cannot hold is reported
    /// once, in the build report, not swallowed by the registry.
    #[test]
    fn a_frame_column_outside_the_vocabulary_warns_in_the_report() {
        let mut df = one_int_frame();
        df.add_column(
            "m".to_string(),
            ColumnType::Map,
            ColumnData::Map(vec![None, None]),
        )
        .unwrap();
        let mut graph = DirGraph::new();
        let report = build(
            &mut graph,
            bp(ONE_FRAME_SPEC),
            Path::new("."),
            frames(vec![("rows", df)]),
        )
        .unwrap_or_else(|e| panic!("a supplied frame builds: {e}"));
        assert!(
            report
                .warnings
                .iter()
                .any(|w| w.contains("frame 'rows' column 'm' has type map")),
            "{:?}",
            report.warnings
        );
    }
}

#[cfg(test)]
mod render_text_tests {
    use super::*;

    fn report() -> BuildReport {
        BuildReport {
            nodes_by_type: [("Person".to_string(), 3), ("Org".to_string(), 1)]
                .into_iter()
                .collect(),
            edges_by_type: [("KNOWS".to_string(), 5), ("WORKS_AT".to_string(), 2)]
                .into_iter()
                .collect(),
            edges_actual: [("KNOWS".to_string(), 4), ("WORKS_AT".to_string(), 2)]
                .into_iter()
                .collect(),
            warnings: vec!["ignored".to_string()],
            errors: vec!["ignored".to_string()],
            provisional_purged: 2,
        }
    }

    /// The exact lines the wheel has printed since 0.9.1 — golden, because
    /// `tests/test_blueprint.py` asserts on substrings of them.
    #[test]
    fn verbose_text_is_the_summary_the_wheel_prints() {
        let expected = [
            "Loading blueprint...",
            "  Org: 1 nodes",
            "  Person: 3 nodes",
            "  [KNOWS]: 4 edges (5 input rows, 1 deduped)",
            "  [WORKS_AT]: 2 edges",
            "Loaded 4 nodes (2 types), 6 edges (2 types) \u{2014} 7 input rows, 1 deduped",
            "  auto_purge: dropped 2 unpromoted provisional stub node(s)",
            "",
        ]
        .join("\n");
        assert_eq!(report().render_text(true), expected);
    }

    /// Nothing at all when the caller did not ask: warnings and errors are a
    /// separate channel and a silent build stays silent.
    #[test]
    fn a_non_verbose_report_renders_nothing() {
        assert_eq!(report().render_text(false), "");
    }

    /// With no dedupe, neither the per-type line nor the summary carries the
    /// input-row annotation.
    #[test]
    fn text_without_dedupe_carries_no_input_row_annotation() {
        let mut r = report();
        r.edges_actual
            .insert("KNOWS".to_string(), 5)
            .expect("the fixture has this type");
        r.provisional_purged = 0;
        let expected = [
            "Loading blueprint...",
            "  Org: 1 nodes",
            "  Person: 3 nodes",
            "  [KNOWS]: 5 edges",
            "  [WORKS_AT]: 2 edges",
            "Loaded 4 nodes (2 types), 7 edges (2 types)",
            "",
        ]
        .join("\n");
        assert_eq!(r.render_text(true), expected);
    }
}
