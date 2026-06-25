//! PyO3 entry for the Rust blueprint loader.
//!
//! Thin wrapper: returns the populated `KnowledgeGraph` plus the output
//! path declared in the blueprint (if any). Save and `lock_schema` are
//! invoked from the Python shim using the existing `KnowledgeGraph`
//! methods — avoids duplicating the v3 save pipeline here.

use crate::graph::KnowledgeGraph;
use kglite_core::api::blueprint;
use kglite_core::api::mutation::OperationReports;
use kglite_core::api::TemporalContext;
use kglite_core::api::{CowSelection, DirGraph};
use pyo3::prelude::*;
use std::path::Path;
use std::sync::Arc;

/// Parse a JSON blueprint and build a `KnowledgeGraph` from its CSVs.
///
/// Returns `(graph, output_path_or_none)` — the Python shim saves and
/// applies `lock_schema` on top. Exposed as `kglite.kglite.from_blueprint_rust`
/// to avoid colliding with the user-facing `kglite.from_blueprint` wrapper.
#[pyfunction]
#[pyo3(signature = (blueprint_path, *, verbose=false, storage=None, path=None))]
pub fn from_blueprint_rust(
    py: Python<'_>,
    blueprint_path: String,
    verbose: bool,
    storage: Option<&str>,
    path: Option<&str>,
) -> PyResult<(KnowledgeGraph, Option<String>)> {
    let bp_path = Path::new(&blueprint_path).to_path_buf();
    if !bp_path.exists() {
        return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "Blueprint file not found: {}",
            bp_path.display()
        )));
    }

    let (kg, output_path) = py
        .detach(
            || -> Result<(KnowledgeGraph, Option<std::path::PathBuf>), String> {
                // Construct the backing DirGraph with the requested storage
                // mode via the shared core builder (one mode vocabulary across
                // wheel / servers / C ABI). Empty string is treated as default.
                let mode = match storage {
                    None | Some("") => kglite_core::api::storage::StorageMode::Memory,
                    Some(s) => kglite_core::api::storage::StorageMode::parse(s)?,
                };
                let mut graph =
                    kglite_core::api::storage::new_dir_graph_in_mode(mode, path.map(Path::new))?;

                // Parse blueprint
                let blueprint = blueprint::load_blueprint_file(&bp_path)?;
                let output_path = blueprint
                    .settings
                    .resolved_output(bp_path.parent().unwrap_or_else(|| Path::new(".")));

                // Run the build
                let bp_dir = bp_path
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .to_path_buf();
                let report = blueprint::build(&mut graph, blueprint, &bp_dir)?;

                if verbose {
                    let n_total: usize = report.nodes_by_type.values().sum();
                    // 0.9.1 #1: report.edges_by_type counts ATTEMPTED
                    // edge writes from the blueprint pipeline. With the
                    // default Update conflict handling, each duplicate
                    // input row increments the counter but only one
                    // edge ends up in the graph — so the report total
                    // overcounts vs `MATCH ()-[r]->() RETURN count(r)`.
                    // Query the actual graph edge counts here so the
                    // verbose log matches reality. Difference is
                    // surfaced as "(N input rows, M deduped)" when
                    // non-zero.
                    let actual_counts = graph.get_edge_type_counts();
                    let e_actual: usize = actual_counts.values().sum();
                    let e_input: usize = report.edges_by_type.values().sum();
                    println!("Loading blueprint...");
                    for (t, n) in &report.nodes_by_type {
                        println!("  {}: {} nodes", t, n);
                    }
                    for (t, n_input) in &report.edges_by_type {
                        let n_actual = actual_counts.get(t).copied().unwrap_or(0);
                        if n_actual == *n_input {
                            println!("  [{}]: {} edges", t, n_actual);
                        } else {
                            println!(
                                "  [{}]: {} edges ({} input rows, {} deduped)",
                                t,
                                n_actual,
                                n_input,
                                n_input.saturating_sub(n_actual),
                            );
                        }
                    }
                    if e_actual == e_input {
                        println!(
                            "Loaded {} nodes ({} types), {} edges ({} types)",
                            n_total,
                            report.nodes_by_type.len(),
                            e_actual,
                            report.edges_by_type.len(),
                        );
                    } else {
                        println!(
                            "Loaded {} nodes ({} types), {} edges ({} types) — \
                             {} input rows, {} deduped",
                            n_total,
                            report.nodes_by_type.len(),
                            e_actual,
                            report.edges_by_type.len(),
                            e_input,
                            e_input.saturating_sub(e_actual),
                        );
                    }
                    if report.provisional_purged > 0 {
                        println!(
                            "  auto_purge: dropped {} unpromoted provisional stub node(s)",
                            report.provisional_purged
                        );
                    }
                }
                if !report.warnings.is_empty() {
                    if verbose {
                        for w in &report.warnings {
                            eprintln!("warning: {}", w);
                        }
                    } else {
                        // Compact summary so callers running silent
                        // still know data quality issues exist.
                        eprintln!(
                            "{} blueprint warning(s) — pass verbose=True for details.",
                            report.warnings.len()
                        );
                    }
                }
                if !report.errors.is_empty() {
                    for e in &report.errors {
                        eprintln!("error: {}", e);
                    }
                }

                let kg = KnowledgeGraph {
                    inner: Arc::new(graph),
                    cursor: crate::graph::CursorState::new(),
                    embedder: None,
                    default_timeout_ms: None,
                    default_max_rows: None,
                    lifecycle: crate::graph::GraphLifecycle::detached(),
                };
                Ok((kg, output_path))
            },
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok((kg, output_path.map(|p| p.to_string_lossy().into_owned())))
}

/// Build a `KnowledgeGraph` from an inline JSON records spec (nodes +
/// connections), no CSV files on disk. JSON-native sibling to
/// `from_blueprint_rust`. Returns the populated graph; the Python shim handles
/// optional save / lock_schema. Exposed as `kglite.kglite.from_records_rust`.
#[pyfunction]
#[pyo3(signature = (records_json, *, storage=None, path=None))]
pub fn from_records_rust(
    py: Python<'_>,
    records_json: String,
    storage: Option<&str>,
    path: Option<&str>,
) -> PyResult<KnowledgeGraph> {
    let spec: serde_json::Value = serde_json::from_str(&records_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid JSON: {}", e)))?;

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
                default_max_rows: None,
                lifecycle: crate::graph::GraphLifecycle::detached(),
            })
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok(kg)
}
