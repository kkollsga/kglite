// src/graph/export.rs
//! Export graph data to various visualization formats

use crate::datatypes::values::Value;
use crate::graph::schema::{CurrentSelection, DirGraph};
use crate::graph::storage::GraphRead;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

/// Export the graph (or selection) to GraphML format.
///
/// GraphML is an XML-based format supported by many graph visualization tools
/// including Gephi, yEd, and Cytoscape.
pub fn to_graphml(
    graph: &DirGraph,
    selection: Option<&CurrentSelection>,
) -> Result<String, String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let mut xml = String::with_capacity(64 * 1024); // Pre-allocate 64KB

    // XML header
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\"\n");
    xml.push_str("         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n");
    xml.push_str("         xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns\n");
    xml.push_str("         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">\n");

    // Define attribute keys for nodes
    xml.push_str(
        "  <key id=\"node_type\" for=\"node\" attr.name=\"type\" attr.type=\"string\"/>\n",
    );
    xml.push_str(
        "  <key id=\"node_title\" for=\"node\" attr.name=\"title\" attr.type=\"string\"/>\n",
    );
    xml.push_str("  <key id=\"node_id\" for=\"node\" attr.name=\"id\" attr.type=\"string\"/>\n");
    xml.push_str("  <key id=\"node_properties\" for=\"node\" attr.name=\"properties\" attr.type=\"string\"/>\n");

    // Define attribute keys for edges
    xml.push_str("  <key id=\"edge_type\" for=\"edge\" attr.name=\"connection_type\" attr.type=\"string\"/>\n");
    xml.push_str("  <key id=\"edge_properties\" for=\"edge\" attr.name=\"properties\" attr.type=\"string\"/>\n");

    xml.push_str("  <graph id=\"G\" edgedefault=\"directed\">\n");

    // Determine which nodes to export
    let node_indices: Vec<_> = if let Some(sel) = selection {
        let level_idx = sel.get_level_count().saturating_sub(1);
        if let Some(level) = sel.get_level(level_idx) {
            level.get_all_nodes()
        } else {
            {
                let g = &graph.graph;
                g.node_indices().collect()
            }
        }
    } else {
        {
            let g = &graph.graph;
            g.node_indices().collect()
        }
    };

    let node_set: std::collections::HashSet<_> = node_indices.iter().copied().collect();

    // Export nodes
    for &idx in &node_indices {
        if let Some(node) = graph.graph.node_weight(idx) {
            xml.push_str(&format!("    <node id=\"n{}\">\n", idx.index()));
            xml.push_str(&format!(
                "      <data key=\"node_type\">{}</data>\n",
                escape_xml(node.node_type_str(&graph.interner))
            ));
            xml.push_str(&format!(
                "      <data key=\"node_title\">{}</data>\n",
                escape_xml(&crate::datatypes::values::raw_string(&node.title()))
            ));
            xml.push_str(&format!(
                "      <data key=\"node_id\">{}</data>\n",
                escape_xml(&crate::datatypes::values::raw_string(&node.id()))
            ));

            // Serialize properties as JSON
            if node.property_count() > 0 {
                let props_json = properties_to_json(node.property_iter(&graph.interner));
                xml.push_str(&format!(
                    "      <data key=\"node_properties\">{}</data>\n",
                    escape_xml(&props_json)
                ));
            }

            xml.push_str("    </node>\n");
        }
    }

    // Export edges (only between selected nodes)
    let mut edge_id = 0;
    for &source_idx in &node_indices {
        for edge in {
            let g = &graph.graph;
            g.edges(source_idx)
        } {
            let target_idx = edge.target();

            // Only include edge if target is in selection
            if node_set.contains(&target_idx) {
                xml.push_str(&format!(
                    "    <edge id=\"e{}\" source=\"n{}\" target=\"n{}\">\n",
                    edge_id,
                    source_idx.index(),
                    target_idx.index()
                ));
                xml.push_str(&format!(
                    "      <data key=\"edge_type\">{}</data>\n",
                    escape_xml(edge.weight().connection_type_str(&graph.interner))
                ));

                if edge.weight().property_count() > 0 {
                    let props_json =
                        properties_to_json(edge.weight().property_iter(&graph.interner));
                    xml.push_str(&format!(
                        "      <data key=\"edge_properties\">{}</data>\n",
                        escape_xml(&props_json)
                    ));
                }

                xml.push_str("    </edge>\n");
                edge_id += 1;
            }
        }
    }

    xml.push_str("  </graph>\n");
    xml.push_str("</graphml>\n");

    Ok(xml)
}

/// Export the graph (or selection) to D3.js compatible JSON format.
///
/// This format is designed for use with D3.js force-directed graph visualizations.
/// The output is a JSON object with "nodes" and "links" arrays.
pub fn to_d3_json(
    graph: &DirGraph,
    selection: Option<&CurrentSelection>,
) -> Result<String, String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    // Determine which nodes to export
    let node_indices: Vec<_> = if let Some(sel) = selection {
        let level_idx = sel.get_level_count().saturating_sub(1);
        if let Some(level) = sel.get_level(level_idx) {
            level.get_all_nodes()
        } else {
            {
                let g = &graph.graph;
                g.node_indices().collect()
            }
        }
    } else {
        {
            let g = &graph.graph;
            g.node_indices().collect()
        }
    };

    let node_set: std::collections::HashSet<_> = node_indices.iter().copied().collect();

    // Build index mapping (old index -> array position)
    let mut index_map: HashMap<usize, usize> = HashMap::with_capacity(node_indices.len());
    for (pos, &idx) in node_indices.iter().enumerate() {
        index_map.insert(idx.index(), pos);
    }

    // Build nodes array
    let mut nodes_json = Vec::with_capacity(node_indices.len());
    for &idx in &node_indices {
        if let Some(node) = graph.graph.node_weight(idx) {
            let mut obj = String::from("{");
            obj.push_str(&format!("\"id\":{},", json_value(&node.id())));
            obj.push_str(&format!(
                "\"type\":{},",
                json_string(node.node_type_str(&graph.interner))
            ));
            obj.push_str(&format!("\"title\":{}", json_value(&node.title())));

            // Add select properties (not all to keep output clean)
            for (key, value) in node.property_iter(&graph.interner) {
                if key != "id" && key != "title" && key != "type" {
                    obj.push_str(&format!(",{}:{}", json_string(key), json_value(value)));
                }
            }

            obj.push('}');
            nodes_json.push(obj);
        }
    }

    // Build links array
    let mut links_json = Vec::new();
    for &source_idx in &node_indices {
        for edge in {
            let g = &graph.graph;
            g.edges(source_idx)
        } {
            let target_idx = edge.target();

            if node_set.contains(&target_idx) {
                if let (Some(&source_pos), Some(&target_pos)) = (
                    index_map.get(&source_idx.index()),
                    index_map.get(&target_idx.index()),
                ) {
                    let mut link = String::from("{");
                    link.push_str(&format!("\"source\":{},", source_pos));
                    link.push_str(&format!("\"target\":{},", target_pos));
                    link.push_str(&format!(
                        "\"type\":{}",
                        json_string(edge.weight().connection_type_str(&graph.interner))
                    ));

                    // Add edge properties
                    for (key, value) in edge.weight().property_iter(&graph.interner) {
                        link.push_str(&format!(",{}:{}", json_string(key), json_value(value)));
                    }

                    link.push('}');
                    links_json.push(link);
                }
            }
        }
    }

    // Build final JSON
    let mut result = String::with_capacity(32 * 1024);
    result.push_str("{\n  \"nodes\": [\n    ");
    result.push_str(&nodes_json.join(",\n    "));
    result.push_str("\n  ],\n  \"links\": [\n    ");
    result.push_str(&links_json.join(",\n    "));
    result.push_str("\n  ]\n}");

    Ok(result)
}

/// Export to GEXF format (Gephi native format).
///
/// GEXF is the native format for Gephi and supports dynamic graphs,
/// hierarchies, and rich attribute types.
pub fn to_gexf(graph: &DirGraph, selection: Option<&CurrentSelection>) -> Result<String, String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let mut xml = String::with_capacity(64 * 1024);

    // XML header
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<gexf xmlns=\"http://www.gexf.net/1.2draft\"\n");
    xml.push_str("      xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n");
    xml.push_str("      xsi:schemaLocation=\"http://www.gexf.net/1.2draft http://www.gexf.net/1.2draft/gexf.xsd\"\n");
    xml.push_str("      version=\"1.2\">\n");
    xml.push_str("  <meta>\n");
    xml.push_str("    <creator>kglite</creator>\n");
    xml.push_str("    <description>Exported from KnowledgeGraph</description>\n");
    xml.push_str("  </meta>\n");
    xml.push_str("  <graph mode=\"static\" defaultedgetype=\"directed\">\n");

    // Define node attributes
    xml.push_str("    <attributes class=\"node\">\n");
    xml.push_str("      <attribute id=\"0\" title=\"type\" type=\"string\"/>\n");
    xml.push_str("      <attribute id=\"1\" title=\"title\" type=\"string\"/>\n");
    xml.push_str("    </attributes>\n");

    // Define edge attributes
    xml.push_str("    <attributes class=\"edge\">\n");
    xml.push_str("      <attribute id=\"0\" title=\"connection_type\" type=\"string\"/>\n");
    xml.push_str("    </attributes>\n");

    // Determine which nodes to export
    let node_indices: Vec<_> = if let Some(sel) = selection {
        let level_idx = sel.get_level_count().saturating_sub(1);
        if let Some(level) = sel.get_level(level_idx) {
            level.get_all_nodes()
        } else {
            {
                let g = &graph.graph;
                g.node_indices().collect()
            }
        }
    } else {
        {
            let g = &graph.graph;
            g.node_indices().collect()
        }
    };

    let node_set: std::collections::HashSet<_> = node_indices.iter().copied().collect();

    // Export nodes
    xml.push_str("    <nodes>\n");
    for &idx in &node_indices {
        if let Some(node) = graph.graph.node_weight(idx) {
            let title_str = crate::datatypes::values::raw_string(&node.title());
            xml.push_str(&format!(
                "      <node id=\"{}\" label=\"{}\">\n",
                idx.index(),
                escape_xml(&title_str)
            ));
            xml.push_str("        <attvalues>\n");
            xml.push_str(&format!(
                "          <attvalue for=\"0\" value=\"{}\"/>\n",
                escape_xml(node.node_type_str(&graph.interner))
            ));
            xml.push_str(&format!(
                "          <attvalue for=\"1\" value=\"{}\"/>\n",
                escape_xml(&title_str)
            ));
            xml.push_str("        </attvalues>\n");
            xml.push_str("      </node>\n");
        }
    }
    xml.push_str("    </nodes>\n");

    // Export edges
    xml.push_str("    <edges>\n");
    let mut edge_id = 0;
    for &source_idx in &node_indices {
        for edge in {
            let g = &graph.graph;
            g.edges(source_idx)
        } {
            let target_idx = edge.target();

            if node_set.contains(&target_idx) {
                xml.push_str(&format!(
                    "      <edge id=\"{}\" source=\"{}\" target=\"{}\">\n",
                    edge_id,
                    source_idx.index(),
                    target_idx.index()
                ));
                xml.push_str("        <attvalues>\n");
                xml.push_str(&format!(
                    "          <attvalue for=\"0\" value=\"{}\"/>\n",
                    escape_xml(edge.weight().connection_type_str(&graph.interner))
                ));
                xml.push_str("        </attvalues>\n");
                xml.push_str("      </edge>\n");
                edge_id += 1;
            }
        }
    }
    xml.push_str("    </edges>\n");

    xml.push_str("  </graph>\n");
    xml.push_str("</gexf>\n");

    Ok(xml)
}

/// Export to CSV format (nodes and edges as separate content).
///
/// Deterministic, human-readable text projection of a graph's full contents —
/// the canonical form behind the `.kgl` git `textconv` diff filter. Stable
/// across save/load: nodes are grouped by type then sorted by **id** (not the
/// petgraph index, which isn't reload-stable), edges sorted by
/// `(type, source_id, target_id)`. So `git diff` over two `.kgl` snapshots shows
/// real content changes, not reordering noise. Reserved provenance keys
/// (`updated_at`/`git_sha`/`modified_by`) are omitted — they change on every
/// write and would swamp the diff (and they're engine metadata, not data).
pub fn to_text(graph: &DirGraph) -> String {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    use crate::datatypes::values::raw_string;
    use crate::graph::schema::is_reserved_provenance_key;

    let g = &graph.graph;
    let mut out = String::new();

    let fmt_props = |pairs: BTreeMap<String, String>| -> String {
        pairs
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(", ")
    };

    // Nodes, grouped by type (sorted), then by id.
    let mut by_type: BTreeMap<String, Vec<petgraph::graph::NodeIndex>> = BTreeMap::new();
    for idx in g.node_indices() {
        if let Some(node) = g.node_weight(idx) {
            by_type
                .entry(node.node_type_str(&graph.interner).to_string())
                .or_default()
                .push(idx);
        }
    }
    for (ntype, mut idxs) in by_type {
        idxs.sort_by_key(|&i| {
            g.node_weight(i)
                .map(|n| raw_string(&n.id()))
                .unwrap_or_default()
        });
        out.push_str(&format!("# {} ({} node(s))\n", ntype, idxs.len()));
        for i in idxs {
            let Some(node) = g.node_weight(i) else {
                continue;
            };
            let (id, title) = (raw_string(&node.id()), raw_string(&node.title()));
            // Use the canonical node projection (`materialize_node_value`) for
            // properties: it is backend-consistent (handles the columnar store
            // where `property_iter` yields nothing), recovers alias columns
            // (a `name`/title-aliased field shows the same in-memory and after a
            // save/load), and already omits reserved provenance keys. Skip the
            // id/title/type virtuals — they're printed explicitly above.
            let mut props: BTreeMap<String, String> = BTreeMap::new();
            if let Some(nv) =
                crate::graph::languages::cypher::executor::helpers::materialize_node_value(i, graph)
            {
                for (k, v) in nv.properties {
                    if k != "id" && k != "title" && k != "type" {
                        props.insert(k, raw_string(&v));
                    }
                }
            }
            let propstr = fmt_props(props);
            out.push_str(&format!("  {id} | {title}"));
            if !propstr.is_empty() {
                out.push_str(&format!(" | {propstr}"));
            }
            out.push('\n');
        }
        out.push('\n');
    }

    // Edges, sorted by (type, source_id, target_id).
    let mut edges: Vec<(String, String, String, String)> = Vec::new();
    for src in g.node_indices() {
        let Some(src_node) = g.node_weight(src) else {
            continue;
        };
        let src_id = raw_string(&src_node.id());
        for e in g.edges(src) {
            let Some(tgt_node) = g.node_weight(e.target()) else {
                continue;
            };
            let w = e.weight();
            let mut props: BTreeMap<String, String> = BTreeMap::new();
            for (k, v) in w.property_iter(&graph.interner) {
                if !is_reserved_provenance_key(k) {
                    props.insert(k.to_string(), raw_string(v));
                }
            }
            edges.push((
                w.connection_type_str(&graph.interner).to_string(),
                src_id.clone(),
                raw_string(&tgt_node.id()),
                fmt_props(props),
            ));
        }
    }
    edges.sort();
    if !edges.is_empty() {
        out.push_str(&format!("# edges ({})\n", edges.len()));
        for (conn, src, tgt, props) in edges {
            out.push_str(&format!("  ({src})-[{conn}]->({tgt})"));
            if !props.is_empty() {
                out.push_str(&format!(" {{{props}}}"));
            }
            out.push('\n');
        }
    }
    out
}

/// Returns a tuple of (nodes_csv, edges_csv).
pub fn to_csv(
    graph: &DirGraph,
    selection: Option<&CurrentSelection>,
) -> Result<(String, String), String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    // Determine which nodes to export
    let node_indices: Vec<_> = if let Some(sel) = selection {
        let level_idx = sel.get_level_count().saturating_sub(1);
        if let Some(level) = sel.get_level(level_idx) {
            level.get_all_nodes()
        } else {
            {
                let g = &graph.graph;
                g.node_indices().collect()
            }
        }
    } else {
        {
            let g = &graph.graph;
            g.node_indices().collect()
        }
    };

    let node_set: std::collections::HashSet<_> = node_indices.iter().copied().collect();

    // Build nodes CSV
    let mut nodes_csv = String::from("id,type,title\n");
    for &idx in &node_indices {
        if let Some(node) = graph.graph.node_weight(idx) {
            nodes_csv.push_str(&format!(
                "{},{},{}\n",
                idx.index(),
                escape_csv(node.node_type_str(&graph.interner)),
                escape_csv(&crate::datatypes::values::raw_string(&node.title()))
            ));
        }
    }

    // Build edges CSV
    let mut edges_csv = String::from("source,target,type\n");
    for &source_idx in &node_indices {
        for edge in {
            let g = &graph.graph;
            g.edges(source_idx)
        } {
            let target_idx = edge.target();

            if node_set.contains(&target_idx) {
                edges_csv.push_str(&format!(
                    "{},{},{}\n",
                    source_idx.index(),
                    target_idx.index(),
                    escape_csv(edge.weight().connection_type_str(&graph.interner))
                ));
            }
        }
    }

    Ok((nodes_csv, edges_csv))
}

/// Metadata for a single connection type in the blueprint.
/// (source_type, target_type, property_columns, property_type_map)
type ConnMeta = (String, String, Vec<String>, BTreeMap<String, String>);

/// Summary of a CSV directory export.
pub struct ExportSummary {
    /// Output directory path.
    pub output_dir: String,
    /// Node counts per type: type_name → row count.
    pub nodes: BTreeMap<String, usize>,
    /// Connection counts per type: connection_type → row count.
    pub connections: BTreeMap<String, usize>,
    /// Total files written (CSVs + blueprint.json).
    pub files_written: usize,
    /// Log lines for verbose output.
    pub log_lines: Vec<String>,
}

/// Export the graph (or selection) to an organized CSV directory tree.
///
/// Creates:
/// - `nodes/<Type>.csv` for each node type (sub-nodes nested under parent)
/// - `connections/<Type>.csv` for each connection type
/// - `blueprint.json` for round-trip re-import via `from_blueprint()`
pub fn to_csv_dir(
    graph: &DirGraph,
    output_dir: &str,
    selection: Option<&CurrentSelection>,
    parent_types: &HashMap<String, String>,
) -> Result<ExportSummary, String> {
    // Arena guard: disk-backed node/edge reads materialize into the query
    // arena (protocol in disk/graph.rs); no-op on memory/mapped.
    let _arena_guard = graph.graph.begin_query();
    let output = Path::new(output_dir);
    let mut log_lines = Vec::new();
    log_lines.push(format!("Exporting to {}...", output_dir));

    // ── 1. Collect selected node indices ─────────────────────────
    let node_indices = selected_node_indices(graph, selection);
    let node_set: HashSet<_> = node_indices.iter().copied().collect();

    // ── 2. Group nodes by type ───────────────────────────────────
    // type_name → Vec<NodeIndex>
    let mut nodes_by_type: BTreeMap<String, Vec<petgraph::graph::NodeIndex>> = BTreeMap::new();
    for &idx in &node_indices {
        if let Some(node) = graph.graph.node_weight(idx) {
            nodes_by_type
                .entry(node.node_type_str(&graph.interner).to_string())
                .or_default()
                .push(idx);
        }
    }

    // ── 3. Collect edges between selected nodes, grouped by type ─
    // conn_type → Vec<(source_idx, target_idx, edge_ref)>
    struct EdgeInfo {
        source_idx: petgraph::graph::NodeIndex,
        target_idx: petgraph::graph::NodeIndex,
        properties: HashMap<String, Value>,
    }
    let mut edges_by_type: BTreeMap<String, Vec<EdgeInfo>> = BTreeMap::new();
    for &source_idx in &node_indices {
        for edge in {
            let g = &graph.graph;
            g.edges(source_idx)
        } {
            let target_idx = edge.target();
            if node_set.contains(&target_idx) {
                let w = edge.weight();
                edges_by_type
                    .entry(w.connection_type_str(&graph.interner).to_string())
                    .or_default()
                    .push(EdgeInfo {
                        source_idx,
                        target_idx,
                        properties: w.properties_cloned(&graph.interner),
                    });
            }
        }
    }

    // ── 4. Create directories ────────────────────────────────────
    let nodes_dir = output.join("nodes");
    let connections_dir = output.join("connections");
    std::fs::create_dir_all(&nodes_dir)
        .map_err(|e| format!("Failed to create nodes directory: {}", e))?;
    if !edges_by_type.is_empty() {
        std::fs::create_dir_all(&connections_dir)
            .map_err(|e| format!("Failed to create connections directory: {}", e))?;
    }

    // Create sub-node directories
    for parent in parent_types.values() {
        if nodes_by_type.contains_key(parent) {
            let sub_dir = nodes_dir.join(parent);
            std::fs::create_dir_all(&sub_dir)
                .map_err(|e| format!("Failed to create sub-node directory: {}", e))?;
        }
    }

    let mut summary = ExportSummary {
        output_dir: output_dir.to_string(),
        nodes: BTreeMap::new(),
        connections: BTreeMap::new(),
        files_written: 0,
        log_lines: Vec::new(),
    };

    // ── 5. Write node CSVs ───────────────────────────────────────
    // Track property columns and types per node type for the blueprint
    let mut node_type_columns: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut node_type_prop_types: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();

    for (node_type, indices) in &nodes_by_type {
        // First pass: collect the union of all property names
        let mut prop_names: BTreeSet<String> = BTreeSet::new();
        for &idx in indices {
            if let Some(node) = graph.graph.node_weight(idx) {
                for key in node.property_keys(&graph.interner) {
                    prop_names.insert(key.to_string());
                }
            }
        }
        let prop_cols: Vec<String> = prop_names.into_iter().collect();

        // Infer property types from first non-null value of each property
        let mut prop_types: BTreeMap<String, String> = BTreeMap::new();
        for col in &prop_cols {
            for &idx in indices {
                if let Some(node) = graph.graph.node_weight(idx) {
                    if let Some(val) = node.get_property(col) {
                        if !matches!(*val, Value::Null) {
                            prop_types.insert(col.clone(), value_type_name(&val));
                            break;
                        }
                    }
                }
            }
        }

        // Build CSV
        let mut csv = String::with_capacity(4096);
        // Header: id, title, then properties
        csv.push_str("id,title");
        for col in &prop_cols {
            csv.push(',');
            csv.push_str(&escape_csv(col));
        }
        csv.push('\n');

        // Rows
        for &idx in indices {
            if let Some(node) = graph.graph.node_weight(idx) {
                csv.push_str(&escape_csv(&crate::datatypes::values::raw_string(
                    &node.id(),
                )));
                csv.push(',');
                csv.push_str(&escape_csv(&crate::datatypes::values::raw_string(
                    &node.title(),
                )));
                for col in &prop_cols {
                    csv.push(',');
                    if let Some(val) = node.get_property(col) {
                        csv.push_str(&escape_csv(&crate::datatypes::values::raw_string(&val)));
                    }
                }
                csv.push('\n');
            }
        }

        // Determine file path (nested under parent if sub-node)
        let csv_path = if let Some(parent) = parent_types.get(node_type) {
            nodes_dir.join(parent).join(format!("{}.csv", node_type))
        } else {
            nodes_dir.join(format!("{}.csv", node_type))
        };

        let relative_path = csv_path
            .strip_prefix(output)
            .unwrap_or(&csv_path)
            .to_string_lossy()
            .to_string();

        std::fs::write(&csv_path, &csv)
            .map_err(|e| format!("Failed to write {}: {}", relative_path, e))?;

        log_lines.push(format!(
            "  {}: {} nodes, {} properties",
            relative_path,
            indices.len(),
            prop_cols.len()
        ));

        summary.nodes.insert(node_type.clone(), indices.len());
        summary.files_written += 1;
        node_type_columns.insert(node_type.clone(), prop_cols);
        node_type_prop_types.insert(node_type.clone(), prop_types);
    }

    // ── 6. Write connection CSVs ─────────────────────────────────
    // Track connection metadata for blueprint
    let mut conn_meta: BTreeMap<String, ConnMeta> = BTreeMap::new(); // conn_type -> (source_type, target_type, prop_cols, prop_types)

    for (conn_type, edges) in &edges_by_type {
        // Collect property names across all edges of this type
        let mut prop_names: BTreeSet<String> = BTreeSet::new();
        for edge in edges {
            for key in edge.properties.keys() {
                prop_names.insert(key.clone());
            }
        }
        let prop_cols: Vec<String> = prop_names.into_iter().collect();

        // Infer property types
        let mut prop_types: BTreeMap<String, String> = BTreeMap::new();
        for col in &prop_cols {
            for edge in edges {
                if let Some(val) = edge.properties.get(col) {
                    if !matches!(val, Value::Null) {
                        prop_types.insert(col.clone(), value_type_name(val));
                        break;
                    }
                }
            }
        }

        // Detect source and target types (use first edge)
        let source_type = edges
            .first()
            .and_then(|e| graph.graph.node_weight(e.source_idx))
            .map(|n| n.node_type_str(&graph.interner).to_string())
            .unwrap_or_default();
        let target_type = edges
            .first()
            .and_then(|e| graph.graph.node_weight(e.target_idx))
            .map(|n| n.node_type_str(&graph.interner).to_string())
            .unwrap_or_default();

        // Build CSV
        let mut csv = String::with_capacity(4096);
        csv.push_str("source_id,source_type,target_id,target_type");
        for col in &prop_cols {
            csv.push(',');
            csv.push_str(&escape_csv(col));
        }
        csv.push('\n');

        for edge in edges {
            let source_id = graph
                .graph
                .node_weight(edge.source_idx)
                .map(|n| crate::datatypes::values::raw_string(&n.id()))
                .unwrap_or_default();
            let src_type = graph
                .graph
                .node_weight(edge.source_idx)
                .map(|n| n.node_type_str(&graph.interner).to_string())
                .unwrap_or_default();
            let target_id = graph
                .graph
                .node_weight(edge.target_idx)
                .map(|n| crate::datatypes::values::raw_string(&n.id()))
                .unwrap_or_default();
            let tgt_type = graph
                .graph
                .node_weight(edge.target_idx)
                .map(|n| n.node_type_str(&graph.interner).to_string())
                .unwrap_or_default();

            csv.push_str(&escape_csv(&source_id));
            csv.push(',');
            csv.push_str(&escape_csv(&src_type));
            csv.push(',');
            csv.push_str(&escape_csv(&target_id));
            csv.push(',');
            csv.push_str(&escape_csv(&tgt_type));
            for col in &prop_cols {
                csv.push(',');
                if let Some(val) = edge.properties.get(col) {
                    csv.push_str(&escape_csv(&crate::datatypes::values::raw_string(val)));
                }
            }
            csv.push('\n');
        }

        let csv_path = connections_dir.join(format!("{}.csv", conn_type));
        let relative_path = csv_path
            .strip_prefix(output)
            .unwrap_or(&csv_path)
            .to_string_lossy()
            .to_string();

        std::fs::write(&csv_path, &csv)
            .map_err(|e| format!("Failed to write {}: {}", relative_path, e))?;

        log_lines.push(format!("  {}: {} edges", relative_path, edges.len()));

        summary.connections.insert(conn_type.clone(), edges.len());
        summary.files_written += 1;
        conn_meta.insert(
            conn_type.clone(),
            (source_type, target_type, prop_cols, prop_types),
        );
    }

    // ── 7. Generate blueprint.json ───────────────────────────────
    let blueprint = build_blueprint(
        &nodes_by_type,
        &node_type_columns,
        &node_type_prop_types,
        parent_types,
        &conn_meta,
        output,
    );
    let blueprint_path = output.join("blueprint.json");
    std::fs::write(&blueprint_path, &blueprint)
        .map_err(|e| format!("Failed to write blueprint.json: {}", e))?;
    log_lines.push("  blueprint.json".to_string());
    summary.files_written += 1;

    // Summary line
    let total_nodes: usize = summary.nodes.values().sum();
    let total_edges: usize = summary.connections.values().sum();
    log_lines.push(format!(
        "Done: {} nodes, {} edges, {} files written",
        total_nodes, total_edges, summary.files_written
    ));

    summary.log_lines = log_lines;
    Ok(summary)
}

/// Extract the selected node indices (or all nodes if no selection).
fn selected_node_indices(
    graph: &DirGraph,
    selection: Option<&CurrentSelection>,
) -> Vec<petgraph::graph::NodeIndex> {
    let g = &graph.graph;
    if let Some(sel) = selection {
        let level_idx = sel.get_level_count().saturating_sub(1);
        if let Some(level) = sel.get_level(level_idx) {
            level.get_all_nodes()
        } else {
            g.node_indices().collect()
        }
    } else {
        g.node_indices().collect()
    }
}

/// Map a Value variant to a blueprint property type string.
fn value_type_name(value: &Value) -> String {
    match value {
        Value::String(_) => "string".to_string(),
        Value::Int64(_) => "int".to_string(),
        Value::Float64(_) => "float".to_string(),
        Value::Boolean(_) => "bool".to_string(),
        Value::DateTime(_) => "date".to_string(),
        // No blueprint "timestamp" column type; export as ISO string so the
        // value survives CSV round-trip as text.
        Value::Timestamp(_) => "string".to_string(),
        Value::UniqueId(_) => "int".to_string(),
        Value::Point { .. } => "string".to_string(), // serialized as string in CSV
        // Durations are query-time-only — CSV export serializes as string.
        Value::Duration { .. } => "string".to_string(),
        Value::Null => "string".to_string(),
        Value::NodeRef(_) => "int".to_string(),
        // Phase A.1 — collection / graph-entity values serialize as
        // JSON-ish strings in CSV (round-trip is best-effort).
        Value::List(_)
        | Value::Map(_)
        | Value::Node(_)
        | Value::Relationship(_)
        | Value::Path(_) => "string".to_string(),
    }
}

/// Build a blueprint.json string for round-trip re-import.
fn build_blueprint(
    nodes_by_type: &BTreeMap<String, Vec<petgraph::graph::NodeIndex>>,
    _node_type_columns: &BTreeMap<String, Vec<String>>,
    node_type_prop_types: &BTreeMap<String, BTreeMap<String, String>>,
    parent_types: &HashMap<String, String>,
    conn_meta: &BTreeMap<String, ConnMeta>,
    _output_dir: &Path,
) -> String {
    let mut json = String::with_capacity(4096);
    json.push_str("{\n  \"settings\": {\n    \"root\": \".\"\n  },\n  \"nodes\": {");

    let mut first_node = true;
    for node_type in nodes_by_type.keys() {
        if !first_node {
            json.push(',');
        }
        first_node = false;

        // Determine CSV path relative to output_dir
        let csv_rel = if let Some(parent) = parent_types.get(node_type) {
            format!("nodes/{}/{}.csv", parent, node_type)
        } else {
            format!("nodes/{}.csv", node_type)
        };

        json.push_str(&format!("\n    {}: {{\n", json_string(node_type)));
        json.push_str(&format!("      \"csv\": {},\n", json_string(&csv_rel)));
        json.push_str("      \"pk\": \"id\",\n");
        json.push_str("      \"title\": \"title\"");

        // Properties
        if let Some(prop_types) = node_type_prop_types.get(node_type) {
            if !prop_types.is_empty() {
                json.push_str(",\n      \"properties\": {");
                let mut first_prop = true;
                for (col, typ) in prop_types {
                    if !first_prop {
                        json.push(',');
                    }
                    first_prop = false;
                    json.push_str(&format!(
                        "\n        {}: {}",
                        json_string(col),
                        json_string(typ)
                    ));
                }
                json.push_str("\n      }");
            }
        }

        // Sub-node parent reference
        if let Some(parent) = parent_types.get(node_type) {
            json.push_str(&format!(",\n      \"parent\": {}", json_string(parent)));
        }

        // Connection definitions for this node type
        let node_conns: Vec<_> = conn_meta
            .iter()
            .filter(|(_, (src, _, _, _))| src == node_type)
            .collect();

        if !node_conns.is_empty() {
            json.push_str(",\n      \"connections\": {\n        \"junction_edges\": {");
            let mut first_conn = true;
            for (conn_type, (_, target_type, prop_cols, _prop_types)) in &node_conns {
                if !first_conn {
                    json.push(',');
                }
                first_conn = false;
                let conn_csv = format!("connections/{}.csv", conn_type);
                json.push_str(&format!(
                    "\n          {}: {{\n            \"csv\": {},\n            \"source_fk\": \"source_id\",\n            \"target\": {},\n            \"target_fk\": \"target_id\"",
                    json_string(conn_type),
                    json_string(&conn_csv),
                    json_string(target_type)
                ));
                // Edge properties (exclude the 4 standard columns)
                if !prop_cols.is_empty() {
                    json.push_str(",\n            \"properties\": [");
                    let mut first_p = true;
                    for p in prop_cols.iter() {
                        if !first_p {
                            json.push_str(", ");
                        }
                        first_p = false;
                        json.push_str(&json_string(p));
                    }
                    json.push(']');
                }
                json.push_str("\n          }");
            }
            json.push_str("\n        }\n      }");
        }

        json.push_str("\n    }");
    }

    json.push_str("\n  }\n}\n");

    // Pretty-format via serde_json for consistency (the string is already valid JSON)
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json) {
        if let Ok(pretty) = serde_json::to_string_pretty(&parsed) {
            return pretty;
        }
    }
    json
}

// Helper functions

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn json_string(s: &str) -> String {
    format!(
        "\"{}\"",
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    )
}

fn json_value(value: &Value) -> String {
    match value {
        Value::String(s) => json_string(s),
        Value::Int64(n) => n.to_string(),
        Value::Float64(f) => {
            if f.is_nan() || f.is_infinite() {
                "null".to_string()
            } else {
                f.to_string()
            }
        }
        Value::Boolean(b) => b.to_string(),
        Value::DateTime(dt) => json_string(&dt.to_string()),
        Value::Timestamp(dt) => json_string(&dt.format("%Y-%m-%dT%H:%M:%S").to_string()),
        Value::UniqueId(id) => id.to_string(),
        Value::Point { lat, lon } => format!("{{\"lat\":{},\"lon\":{}}}", lat, lon),
        Value::Duration {
            months,
            days,
            seconds,
        } => format!(
            "{{\"months\":{},\"days\":{},\"seconds\":{}}}",
            months, days, seconds
        ),
        Value::Null => "null".to_string(),
        Value::NodeRef(idx) => idx.to_string(),
        // Phase A.1 — JSON-encode the new variants recursively via
        // serde_json (List, Map, Node, Relationship, Path all derive
        // Serialize). Fallback to "null" if serialisation fails (it
        // shouldn't — these are owned structures with no cycles).
        Value::List(_)
        | Value::Map(_)
        | Value::Node(_)
        | Value::Relationship(_)
        | Value::Path(_) => serde_json::to_string(value).unwrap_or_else(|_| "null".to_string()),
    }
}

fn properties_to_json<'a>(properties: impl Iterator<Item = (&'a str, &'a Value)>) -> String {
    let pairs: Vec<String> = properties
        .map(|(k, v)| format!("{}:{}", json_string(k), json_value(v)))
        .collect();
    format!("{{{}}}", pairs.join(","))
}
