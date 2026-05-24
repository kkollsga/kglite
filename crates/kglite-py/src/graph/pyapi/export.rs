// Export #[pymethods] — extracted from mod.rs

use pyo3::prelude::*;

use crate::graph::schema::CurrentSelection;
use crate::graph::KnowledgeGraph;

#[pymethods]
impl KnowledgeGraph {
    // ========================================================================
    // Export Methods
    // ========================================================================

    /// Export the graph or current selection to a file in the specified format.
    ///
    /// Supported formats:
    /// - "graphml" - GraphML XML format (Gephi, yEd, Cytoscape)
    /// - "gexf" - GEXF XML format (Gephi native)
    /// - "d3" or "json" - D3.js compatible JSON format
    /// - "csv" - CSV format (creates two files: path_nodes.csv and path_edges.csv)
    ///
    /// Args:
    ///     path: Output file path
    ///     format: Export format (default: inferred from file extension)
    ///     selection_only: If True, export only selected nodes (default: True if selection exists)
    ///
    /// Example:
    ///     ```python
    ///     # Export entire graph to GraphML
    ///     graph.export('output.graphml')
    ///
    /// ```text
    /// # Export selection to D3 format
    /// graph.select('Field').expand(hops=2).export('fields.json', format='d3')
    ///
    /// # Export to GEXF for Gephi
    /// graph.export('network.gexf', format='gexf')
    /// ```
    /// ```
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
            } else {
                "graphml" // Default
            }
        });

        // Determine if we should use selection
        let use_selection = selection_only.unwrap_or(self.selection.get_level_count() > 0);
        let selection: Option<&CurrentSelection> = if use_selection {
            Some(&self.selection) // Deref coercion: &CowSelection -> &CurrentSelection
        } else {
            None
        };

        match fmt {
            "graphml" => {
                let content = crate::graph::io::export::to_graphml(&self.inner, selection)
                    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
                std::fs::write(path, content)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            }
            "gexf" => {
                let content = crate::graph::io::export::to_gexf(&self.inner, selection)
                    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
                std::fs::write(path, content)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            }
            "d3" | "json" => {
                let content = crate::graph::io::export::to_d3_json(&self.inner, selection)
                    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
                std::fs::write(path, content)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            }
            "csv" => {
                let (nodes_csv, edges_csv) =
                    crate::graph::io::export::to_csv(&self.inner, selection)
                        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

                // Write nodes file
                let nodes_path = path.replace(".csv", "_nodes.csv");
                std::fs::write(&nodes_path, nodes_csv)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

                // Write edges file
                let edges_path = path.replace(".csv", "_edges.csv");
                std::fs::write(&edges_path, edges_csv)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown export format: '{}'. Supported: graphml, gexf, d3, json, csv",
                    fmt
                )));
            }
        }

        Ok(())
    }

    /// Export graph data to an organized CSV directory tree.
    ///
    /// Creates a directory with:
    /// - `nodes/<Type>.csv` for each node type (with all properties as columns)
    /// - `connections/<Type>.csv` for each connection type
    /// - `blueprint.json` for round-trip re-import via `from_blueprint()`
    ///
    /// Sub-nodes (types registered with `set_parent_type()`) are nested
    /// under their parent type's folder: `nodes/<Parent>/<Child>.csv`.
    ///
    /// Args:
    ///     path: Output directory path (created if it doesn't exist)
    ///     selection_only: If True, export only selected nodes and their connections.
    ///         Default: True if a selection exists, False otherwise.
    ///     verbose: If True, print progress information during export.
    ///
    /// Returns:
    ///     A dict with export summary:
    ///     - "output_dir": str — the output directory path
    ///     - "nodes": dict[str, int] — node counts per type
    ///     - "connections": dict[str, int] — connection counts per type
    ///     - "files_written": int — total files written
    ///
    /// Example:
    ///     ```python
    ///     # Export entire graph
    ///     result = graph.export_csv('output/')
    ///
    /// ```text
    /// # Export only selected nodes
    /// result = graph.select('Person').export_csv('output/')
    ///
    /// # With progress output
    /// result = graph.export_csv('output/', verbose=True)
    /// ```
    /// ```
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
        let selection_has_nodes = if self.selection.get_level_count() > 0 {
            let level_idx = self.selection.get_level_count().saturating_sub(1);
            self.selection
                .get_level(level_idx)
                .map(|l| l.node_count() > 0)
                .unwrap_or(false)
        } else {
            false
        };
        let use_selection = match selection_only {
            Some(true) => true,
            Some(false) => false,
            None => selection_has_nodes,
        };
        let selection: Option<&CurrentSelection> = if use_selection {
            Some(&self.selection)
        } else {
            None
        };

        let summary = crate::graph::io::export::to_csv_dir(
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

    /// Export to a string instead of a file.
    ///
    /// Useful for web APIs or further processing.
    ///
    /// Args:
    ///     format: Export format (graphml, gexf, d3, json)
    ///     selection_only: If True, export only selected nodes
    ///
    /// Returns:
    ///     The exported data as a string
    ///
    /// Note:
    ///     If selection_only is not specified:
    ///     - If there's a non-empty selection, exports only selected nodes
    ///     - If selection is empty, exports the entire graph
    ///     Use selection_only=True to force selection export (may be empty)
    ///     Use selection_only=False to always export the entire graph
    #[pyo3(signature = (format, selection_only=None))]
    fn export_string(&self, format: &str, selection_only: Option<bool>) -> PyResult<String> {
        // Check if selection has actual nodes
        let selection_has_nodes = if self.selection.get_level_count() > 0 {
            let level_idx = self.selection.get_level_count().saturating_sub(1);
            self.selection
                .get_level(level_idx)
                .map(|l| l.node_count() > 0)
                .unwrap_or(false)
        } else {
            false
        };

        // Default behavior: use selection only if it has nodes
        // If selection_only is explicitly set, respect that
        let use_selection = match selection_only {
            Some(true) => true,          // User explicitly wants selection only
            Some(false) => false,        // User explicitly wants full graph
            None => selection_has_nodes, // Auto: use selection if it has nodes
        };

        let selection: Option<&CurrentSelection> = if use_selection {
            Some(&self.selection) // Deref coercion
        } else {
            None
        };

        match format {
            "graphml" => crate::graph::io::export::to_graphml(&self.inner, selection)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>),
            "gexf" => crate::graph::io::export::to_gexf(&self.inner, selection)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>),
            "d3" | "json" => crate::graph::io::export::to_d3_json(&self.inner, selection)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown export format: '{}'. Supported: graphml, gexf, d3, json",
                format
            ))),
        }
    }
}
