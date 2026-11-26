//! Export graph data to various visualization formats

use crate::datatypes::values::Value;
use crate::graph::schema::{CurrentSelection, DirGraph};
use crate::graph::storage::GraphRead;
use std::collections::{BTreeMap, HashMap};

mod csv_tree;
mod encoding;
#[cfg(feature = "rdf")]
pub mod kg_vocab;
pub mod manifest;
pub(crate) mod typed_json;
pub use csv_tree::to_csv_dir;
#[cfg(feature = "rdf")]
mod rdf_writer;
#[cfg(feature = "rdf")]
pub use rdf_writer::{to_rdf, RdfExportOptions, RdfExportSummary, RdfFormat};
mod paths;
use encoding::{escape_csv, escape_xml, json_string};
pub use manifest::ExportManifest;

/// Export the graph (or selection) to GraphML format.
///
/// GraphML is an XML-based format supported by many graph visualization tools
/// including Gephi, yEd, and Cytoscape.
pub fn to_graphml(
    graph: &DirGraph,
    selection: Option<&CurrentSelection>,
) -> Result<String, String> {
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
    // `label` is the display name Gephi / yEd / Cytoscape render; the `title`
    // key above is kglite's own name for the same string and those readers
    // ignore it, so without this an import shows the synthetic `n0` element
    // ids. Both are emitted: consumers already reading `title` keep working.
    xml.push_str(
        "  <key id=\"node_label\" for=\"node\" attr.name=\"label\" attr.type=\"string\"/>\n",
    );
    xml.push_str("  <key id=\"node_properties\" for=\"node\" attr.name=\"properties\" attr.type=\"string\"/>\n");

    // Define attribute keys for edges
    xml.push_str("  <key id=\"edge_type\" for=\"edge\" attr.name=\"connection_type\" attr.type=\"string\"/>\n");
    xml.push_str("  <key id=\"edge_properties\" for=\"edge\" attr.name=\"properties\" attr.type=\"string\"/>\n");
    xml.push_str(
        "  <key id=\"edge_label\" for=\"edge\" attr.name=\"label\" attr.type=\"string\"/>\n",
    );

    xml.push_str("  <graph id=\"G\" edgedefault=\"directed\">\n");

    let node_indices = selected_node_indices(graph, selection);

    let node_set: std::collections::HashSet<_> = node_indices.iter().copied().collect();

    // Export nodes
    for &idx in &node_indices {
        if let Some(node) = graph.graph.node_view(idx) {
            xml.push_str(&format!("    <node id=\"n{}\">\n", idx.index()));
            xml.push_str(&format!(
                "      <data key=\"node_type\">{}</data>\n",
                escape_xml(node.node_type_str(&graph.interner))?
            ));
            let title = escape_xml(&crate::datatypes::values::raw_string(&node.title()))?;
            xml.push_str(&format!("      <data key=\"node_title\">{title}</data>\n"));
            xml.push_str(&format!("      <data key=\"node_label\">{title}</data>\n"));
            xml.push_str(&format!(
                "      <data key=\"node_id\">{}</data>\n",
                escape_xml(&crate::datatypes::values::raw_string(&node.id()))?
            ));

            // Serialize properties as JSON
            // This read used to be `node.property_iter(..)`,
            // which yields *nothing* for `PropertyStorage::Columnar` while
            // `property_count()` reports the real count — so every node of a
            // saved graph got an empty `{}` here. `NodeView` enumeration is
            // complete for every storage variant.
            if node.property_count() > 0 {
                let props_json =
                    properties_to_json_owned(node.property_pairs_named(&graph.interner));
                xml.push_str(&format!(
                    "      <data key=\"node_properties\">{}</data>\n",
                    escape_xml(&props_json)?
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
                let conn_type = escape_xml(edge.weight().connection_type_str(&graph.interner))?;
                xml.push_str(&format!(
                    "      <data key=\"edge_type\">{conn_type}</data>\n"
                ));
                xml.push_str(&format!(
                    "      <data key=\"edge_label\">{conn_type}</data>\n"
                ));

                if edge.weight().property_count() > 0 {
                    let props_json =
                        properties_to_json(edge.weight().property_iter(&graph.interner));
                    xml.push_str(&format!(
                        "      <data key=\"edge_properties\">{}</data>\n",
                        escape_xml(&props_json)?
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
    let node_indices = selected_node_indices(graph, selection);

    let node_set: std::collections::HashSet<_> = node_indices.iter().copied().collect();

    // Build index mapping (old index -> array position)
    let mut index_map: HashMap<usize, usize> = HashMap::with_capacity(node_indices.len());
    for (pos, &idx) in node_indices.iter().enumerate() {
        index_map.insert(idx.index(), pos);
    }

    // Build nodes array
    let mut nodes_json = Vec::with_capacity(node_indices.len());
    for &idx in &node_indices {
        if let Some(node) = graph.graph.node_view(idx) {
            let mut obj = String::from("{");
            obj.push_str(&format!("\"id\":{},", json_value(&node.id())));
            obj.push_str(&format!(
                "\"type\":{},",
                json_string(node.node_type_str(&graph.interner))
            ));
            obj.push_str(&format!("\"title\":{}", json_value(&node.title())));

            // Canonical node fields take precedence over conflicting properties.
            // `property_iter` yielded nothing for columnar (saved) graphs,
            // silently dropping every property here.
            for (key, value) in node.property_pairs_named(&graph.interner) {
                if key != "id" && key != "title" && key != "type" {
                    obj.push_str(&format!(",{}:{}", json_string(&key), json_value(&value)));
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
                        if !matches!(key, "source" | "target" | "type") {
                            link.push_str(&format!(",{}:{}", json_string(key), json_value(value)));
                        }
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

    let node_indices = selected_node_indices(graph, selection);

    let node_set: std::collections::HashSet<_> = node_indices.iter().copied().collect();

    // Export nodes
    xml.push_str("    <nodes>\n");
    for &idx in &node_indices {
        if let Some(node) = graph.graph.node_view(idx) {
            let title_str = crate::datatypes::values::raw_string(&node.title());
            xml.push_str(&format!(
                "      <node id=\"{}\" label=\"{}\">\n",
                idx.index(),
                escape_xml(&title_str)?
            ));
            xml.push_str("        <attvalues>\n");
            xml.push_str(&format!(
                "          <attvalue for=\"0\" value=\"{}\"/>\n",
                escape_xml(node.node_type_str(&graph.interner))?
            ));
            xml.push_str(&format!(
                "          <attvalue for=\"1\" value=\"{}\"/>\n",
                escape_xml(&title_str)?
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
                    escape_xml(graph.interner.resolve(edge.connection_type()))?
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
        if let Some(node) = g.node_view(idx) {
            by_type
                .entry(node.node_type_str(&graph.interner).to_string())
                .or_default()
                .push(idx);
        }
    }
    for (ntype, mut idxs) in by_type {
        idxs.sort_by_key(|&i| {
            g.node_view(i)
                .map(|n| raw_string(&n.id()))
                .unwrap_or_default()
        });
        out.push_str(&format!("# {} ({} node(s))\n", ntype, idxs.len()));
        for i in idxs {
            let Some(node) = g.node_view(i) else {
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
                for (k, v) in nv.properties.iter() {
                    if k != "id" && k != "title" && k != "type" {
                        props.insert(k.to_string(), raw_string(v));
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
        let Some(src_node) = g.node_view(src) else {
            continue;
        };
        let src_id = raw_string(&src_node.id());
        for e in g.edges(src) {
            let Some(tgt_node) = g.node_view(e.target()) else {
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
    let node_indices = selected_node_indices(graph, selection);

    let node_set: std::collections::HashSet<_> = node_indices.iter().copied().collect();

    // Build nodes CSV
    let mut nodes_csv = String::from("id,type,title\n");
    for &idx in &node_indices {
        if let Some(node) = graph.graph.node_view(idx) {
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
                    escape_csv(graph.interner.resolve(edge.connection_type()))
                ));
            }
        }
    }

    Ok((nodes_csv, edges_csv))
}

/// Extract the selected node indices (or all nodes if no selection).
pub(crate) fn selected_node_indices(
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
        Value::Timestamp(dt) => json_string(&dt.format("%Y-%m-%dT%H:%M:%S%.f").to_string()),
        Value::UniqueId(id) => id.to_string(),
        Value::Point { lat, lon } => format!(
            "{{\"lat\":{},\"lon\":{}}}",
            json_value(&Value::Float64(*lat)),
            json_value(&Value::Float64(*lon))
        ),
        Value::Duration {
            months,
            days,
            seconds,
            nanos: 0,
        } => format!(
            "{{\"months\":{},\"days\":{},\"seconds\":{}}}",
            months, days, seconds
        ),
        Value::Duration {
            months,
            days,
            seconds,
            nanos,
        } => format!(
            "{{\"months\":{},\"days\":{},\"seconds\":{},\"nanoseconds\":{}}}",
            months, days, seconds, nanos
        ),
        Value::LocalTime(n) => json_string(&crate::datatypes::time_of_day::format_clock(*n)),
        Value::Time { nanos, offset_secs } => json_string(
            &crate::datatypes::time_of_day::format_time(*nanos, *offset_secs),
        ),
        Value::ZonedDateTime {
            utc,
            offset_secs,
            zone,
        } => json_string(&crate::datatypes::zoned_datetime::format_zoned_in(
            *utc,
            *offset_secs,
            *zone,
        )),
        Value::Null => "null".to_string(),
        Value::NodeRef(idx) => idx.to_string(),
        // JSON-encode these variants recursively via
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

/// Owned-pair variant, for node reads that resolve through a
/// [`NodeView`](crate::graph::storage::NodeView) — columnar values are
/// constructed on read, so they cannot be handed out as `&Value`.
fn properties_to_json_owned(properties: Vec<(String, Value)>) -> String {
    let pairs: Vec<String> = properties
        .into_iter()
        .map(|(k, v)| format!("{}:{}", json_string(&k), json_value(&v)))
        .collect();
    format!("{{{}}}", pairs.join(","))
}

#[cfg(test)]
#[path = "export_tests.rs"]
mod export_tests;
