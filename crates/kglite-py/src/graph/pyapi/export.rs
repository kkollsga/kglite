// Export #[pymethods] — extracted from mod.rs

use pyo3::prelude::*;
use std::path::{Path, PathBuf};

use crate::graph::KnowledgeGraph;
use kglite_core::api::CurrentSelection;

#[pymethods]
impl KnowledgeGraph {
    // ========================================================================
    // Export Methods
    // ========================================================================

    /// Export the graph or current selection to files in the specified format.
    #[pyo3(signature = (path, format=None, selection_only=None))]
    fn export(
        &self,
        path: &str,
        format: Option<&str>,
        selection_only: Option<bool>,
    ) -> PyResult<()> {
        // Infer format from extension if not specified
        let fmt = format.unwrap_or_else(|| {
            if path.ends_with(".graphml") {
                "graphml"
            } else if path.ends_with(".gexf") {
                "gexf"
            } else if path.ends_with(".json") {
                "d3"
            } else if path.ends_with(".csv") {
                "csv"
            } else if path.ends_with(".sql") {
                "sqlite"
            } else {
                "graphml" // Default
            }
        });

        let selection: Option<&CurrentSelection> =
            crate::graph::resolve_export_selection(self, selection_only);

        match fmt {
            "graphml" => {
                let content = kglite_core::api::io::to_graphml(&self.inner, selection)
                    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
                std::fs::write(path, content)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            }
            "gexf" => {
                let content = kglite_core::api::io::to_gexf(&self.inner, selection)
                    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
                std::fs::write(path, content)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            }
            "d3" | "json" => {
                let content = kglite_core::api::io::to_d3_json(&self.inner, selection)
                    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
                std::fs::write(path, content)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            }
            "csv" => {
                let (nodes_path, edges_path) = paired_csv_paths(Path::new(path))?;
                let (nodes_csv, edges_csv) =
                    kglite_core::api::io::to_csv(&self.inner, selection)
                        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

                std::fs::write(&nodes_path, nodes_csv)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

                std::fs::write(&edges_path, edges_csv)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            }
            "sqlite" => {
                let content = kglite_core::api::io::to_sqlite_dump(&self.inner, selection)
                    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
                std::fs::write(path, content)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown export format: '{}'. Supported: graphml, gexf, d3, json, csv, sqlite",
                    fmt
                )));
            }
        }

        Ok(())
    }

    /// Export graph data to a CSV directory tree with a re-import blueprint.
    #[pyo3(signature = (path, selection_only=None, verbose=false))]
    fn export_csv(
        &self,
        path: &str,
        selection_only: Option<bool>,
        verbose: bool,
    ) -> PyResult<Py<PyAny>> {
        // Check if selection actually has nodes (not just levels)
        // Same pattern as export_string() — avoids empty export when
        // add_nodes creates a selection level with 0 nodes.
        let selection: Option<&CurrentSelection> =
            crate::graph::resolve_export_selection(self, selection_only);

        let summary = kglite_core::api::io::to_csv_dir(
            &self.inner,
            path,
            selection,
            &self.inner.parent_types,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyIOError, _>)?;

        if verbose {
            for line in &summary.log_lines {
                println!("{}", line);
            }
        }

        Python::attach(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("output_dir", &summary.output_dir)?;

            let nodes_dict = pyo3::types::PyDict::new(py);
            for (k, v) in &summary.nodes {
                nodes_dict.set_item(k, v)?;
            }
            dict.set_item("nodes", nodes_dict)?;

            let conn_dict = pyo3::types::PyDict::new(py);
            for (k, v) in &summary.connections {
                conn_dict.set_item(k, v)?;
            }
            dict.set_item("connections", conn_dict)?;

            dict.set_item("files_written", summary.files_written)?;

            Ok(dict.into())
        })
    }

    /// Deterministic, human-readable text projection of the whole graph (nodes
    /// by type sorted by id; edges sorted by endpoints). Stable across
    /// save/load — the canonical form behind the `.kgl` git ``textconv`` diff
    /// filter (also available as ``kglite export-text <file>``). Reserved
    /// provenance keys (``updated_at``/``git_sha``) are omitted so diffs aren't
    /// swamped by per-write metadata churn.
    fn to_text(&self) -> String {
        kglite_core::api::io::to_text(&self.inner)
    }

    /// Export to a string instead of a file.
    ///
    /// Useful for web APIs or further processing.
    ///
    /// Args:
    ///     format: Export format (graphml, gexf, d3, json, sqlite).
    ///         Default: "json". `export()` has no string to inspect either way,
    ///         so it infers the format from the *path* extension (falling back
    ///         to graphml); `export_string()` has no path, so it defaults to
    ///         the format a string return is most often fed to — JSON.
    ///     selection_only: If True, export only selected nodes
    ///
    /// Returns:
    ///     The exported data as a string
    ///
    /// Note:
    ///     "csv" is a file-only format — it writes two files (nodes and edges),
    ///     which a single string cannot carry — so `export_string('csv')` is
    ///     rejected. Use `export(path, format='csv')`.
    ///
    /// Note:
    ///     If selection_only is not specified:
    ///     - If there's a non-empty selection, exports only selected nodes
    ///     - If selection is empty, exports the entire graph
    ///     Use selection_only=True to force selection export (may be empty)
    ///     Use selection_only=False to always export the entire graph
    #[pyo3(signature = (format=None, selection_only=None))]
    fn export_string(
        &self,
        format: Option<&str>,
        selection_only: Option<bool>,
    ) -> PyResult<String> {
        let selection: Option<&CurrentSelection> =
            crate::graph::resolve_export_selection(self, selection_only);
        let format = format.unwrap_or("json");

        match format {
            "graphml" => kglite_core::api::io::to_graphml(&self.inner, selection)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>),
            "gexf" => kglite_core::api::io::to_gexf(&self.inner, selection)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>),
            "d3" | "json" => kglite_core::api::io::to_d3_json(&self.inner, selection)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>),
            "sqlite" => kglite_core::api::io::to_sqlite_dump(&self.inner, selection)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>),
            "csv" => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "export_string() cannot produce 'csv': the CSV export writes two \
                 files (nodes and edges), which one string cannot carry. Use \
                 export(path, format='csv') instead."
                    .to_string(),
            )),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown export format: '{}'. Supported: graphml, gexf, d3, json, sqlite",
                format
            ))),
        }
    }
}

/// Resolve both sibling destinations before serializing or writing either file.
fn paired_csv_paths(path: &Path) -> PyResult<(PathBuf, PathBuf)> {
    let stem = path
        .file_stem()
        .filter(|stem| !stem.is_empty())
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("CSV export requires a file basename")
        })?;
    let mut nodes = stem.to_os_string();
    nodes.push("_nodes.csv");
    let mut edges = stem.to_os_string();
    edges.push("_edges.csv");
    Ok((path.with_file_name(nodes), path.with_file_name(edges)))
}

#[cfg(test)]
mod path_tests {
    use super::*;

    #[test]
    fn csv_sibling_paths_use_only_the_final_basename() {
        for input in [
            "folder.csv/graph",
            "folder.csv/graph.csv",
            "folder.csv/graph.CSV",
            "folder.csv/graph.data",
        ] {
            let (nodes, edges) = paired_csv_paths(Path::new(input)).unwrap();
            assert_eq!(nodes, Path::new("folder.csv/graph_nodes.csv"));
            assert_eq!(edges, Path::new("folder.csv/graph_edges.csv"));
            assert_ne!(nodes, edges);
        }
    }
}
