//! The lossless CSV directory export behind `export_csv`.
//!
//! Layout: `nodes/<Type>.csv` (sub-nodes under their parent's directory), one
//! `connections/<REL>.csv` per relationship type — or `<REL>.<Source>.csv`
//! per source type when a relationship leaves several — plus `blueprint.json`
//! (what `from_blueprint` reads) and `manifest.json` (the
//! [`ExportManifest`], applied after the build for what no spec key can say).
//!
//! Two passes, both streaming: the manifest scan types every column, then each
//! CSV is written row by row through a buffered writer, so memory does not grow
//! with the graph's size.
//!
//! Cell grammar (see `blueprint::typing::exact` for the reader side): null is
//! the empty cell; strings are `text` (empty string `\e`, a leading `\`
//! doubled); ints/floats/bools/dates as their text; timestamps ISO 8601
//! without a zone; durations `{"months","days","seconds"}`; lists and maps
//! JSON; points `point(lat, lon)`. A date, timestamp, duration, point or
//! non-finite float *inside* a list or map is a tagged JSON object (see
//! [`super::typed_json`]), so nested values keep their type.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use petgraph::graph::NodeIndex;
use serde_json::{json, Map, Value as Json};

use super::encoding::escape_csv;
use super::manifest::{ColumnKind, ExportManifest, Scope};
use super::paths::ExportPaths;
use super::typed_json::to_json;
use crate::datatypes::values::{raw_string, Value};
use crate::graph::blueprint::typing::exact::encode_text;
use crate::graph::dir_graph::DirGraph;
use crate::graph::schema::CurrentSelection;
use crate::graph::storage::GraphRead;

/// Summary of a CSV directory export.
pub struct ExportSummary {
    /// Output directory path.
    pub output_dir: String,
    /// Node counts per type: type_name → row count.
    pub nodes: BTreeMap<String, usize>,
    /// Connection counts per type: connection_type → row count.
    pub connections: BTreeMap<String, usize>,
    /// Total files written (CSVs + blueprint.json + manifest.json).
    pub files_written: usize,
    /// Log lines for verbose output.
    pub log_lines: Vec<String>,
}

const STANDARD_EDGE_COLUMNS: [&str; 4] = ["source_id", "source_type", "target_id", "target_type"];

/// The blueprint type keyword a column of `kind` is declared with.
fn keyword(kind: ColumnKind) -> &'static str {
    match kind {
        ColumnKind::UniqueId | ColumnKind::Int64 => "int",
        ColumnKind::Float64 => "float",
        ColumnKind::Boolean => "bool",
        ColumnKind::DateTime => "date",
        ColumnKind::Timestamp => "timestamp",
        ColumnKind::Duration => "duration",
        ColumnKind::LocalTime => "localtime",
        ColumnKind::Time => "time",
        ColumnKind::ZonedDateTime => "zoneddatetime",
        ColumnKind::List => "list",
        ColumnKind::Map => "map",
        ColumnKind::Point => "point",
        ColumnKind::String | ColumnKind::Null | ColumnKind::Mixed => "text",
    }
}

/// The keyword of an id column: ids are never escaped, so a string id is
/// plain `string`.
fn id_keyword(kind: ColumnKind) -> &'static str {
    match keyword(kind) {
        "text" => "string",
        other => other,
    }
}

/// One cell: `None` is null (the empty cell).
fn encode_cell(value: &Value, kind: ColumnKind) -> Option<String> {
    if matches!(value, Value::Null | Value::NodeRef(_)) {
        return None;
    }
    Some(match (kind, value) {
        (ColumnKind::Int64 | ColumnKind::UniqueId, Value::Int64(n)) => n.to_string(),
        (ColumnKind::Int64 | ColumnKind::UniqueId, Value::UniqueId(n)) => n.to_string(),
        (ColumnKind::Float64, Value::Float64(f)) => f.to_string(),
        (ColumnKind::Boolean, Value::Boolean(b)) => b.to_string(),
        (ColumnKind::DateTime, Value::DateTime(d)) => d.to_string(),
        (ColumnKind::Timestamp, Value::Timestamp(t)) => {
            t.format("%Y-%m-%dT%H:%M:%S%.f").to_string()
        }
        (
            ColumnKind::Duration,
            Value::Duration {
                months,
                days,
                seconds,
                nanos,
            },
        ) => super::typed_json::duration_fields(*months, *days, *seconds, *nanos).to_string(),
        (ColumnKind::Point, Value::Point { lat, lon }) => format!("point({lat}, {lon})"),
        (ColumnKind::List | ColumnKind::Map, v @ (Value::List(_) | Value::Map(_))) => {
            to_json(v).to_string()
        }
        (_, Value::String(s)) => encode_text(s),
        (_, v) => encode_text(&raw_string(v)),
    })
}

/// A title cell. A type whose titles mix kinds writes each as its JSON
/// spelling, which [`ExportManifest::restore_mixed_titles`] reads back; every
/// other type writes the plain cell.
fn encode_title(value: &Value, kind: ColumnKind) -> Option<String> {
    if kind == ColumnKind::Mixed && !matches!(value, Value::Null | Value::NodeRef(_)) {
        return Some(to_json(value).to_string());
    }
    encode_cell(value, kind)
}

/// An id cell: ids are written as their text, never escaped.
fn encode_id(value: &Value) -> String {
    raw_string(value)
}

fn push_cell(line: &mut String, cell: &str) {
    line.push_str(&escape_csv(cell));
}

fn open(output: &Path, relative: &str) -> Result<BufWriter<File>, String> {
    File::create(output.join(relative))
        .map(BufWriter::new)
        .map_err(|e| format!("Failed to write {relative}: {e}"))
}

fn write_all(w: &mut BufWriter<File>, relative: &str, text: &str) -> Result<(), String> {
    w.write_all(text.as_bytes())
        .map_err(|e| format!("Failed to write {relative}: {e}"))
}

/// The four standard edge columns as written. A CSV whose edge properties
/// include one of the standard names prefixes all four with `_kg_`, so a
/// property keeps its own name and the blueprint's `source_fk` / `target_fk` /
/// `target_type_column` say which column is which.
fn standard_columns<'a>(properties: impl Iterator<Item = &'a String>) -> [String; 4] {
    let clash = properties
        .into_iter()
        .any(|p| STANDARD_EDGE_COLUMNS.contains(&p.as_str()));
    STANDARD_EDGE_COLUMNS.map(|name| {
        if clash {
            format!("_kg_{name}")
        } else {
            name.to_string()
        }
    })
}

/// Key of one relationship CSV in [`ExportPaths`].
fn connection_key(rel: &str, source: &str, sources_of_rel: usize) -> String {
    if sources_of_rel > 1 {
        format!("{rel}.{source}")
    } else {
        rel.to_string()
    }
}

fn write_node_csv(
    graph: &DirGraph,
    scope: &Scope,
    manifest: &ExportManifest,
    node_type: &str,
    output: &Path,
    relative: &str,
) -> Result<usize, String> {
    let info = &manifest.node_types[node_type];
    let props: Vec<(&String, ColumnKind)> = info.properties.iter().map(|(k, v)| (k, *v)).collect();
    let mut out = open(output, relative)?;
    let mut line = String::from("id,title");
    for (name, _) in &props {
        line.push(',');
        push_cell(&mut line, name);
    }
    line.push('\n');
    write_all(&mut out, relative, &line)?;

    let mut written = 0usize;
    if let Some(nodes) = graph.type_indices.get(node_type) {
        for idx in nodes.iter() {
            if !scope.contains(idx) {
                continue;
            }
            let Some(node) = graph.graph.node_view(idx) else {
                continue;
            };
            line.clear();
            push_cell(&mut line, &encode_id(&node.id()));
            line.push(',');
            if let Some(cell) = encode_title(&node.title(), info.title_kind) {
                push_cell(&mut line, &cell);
            }
            for (name, kind) in &props {
                line.push(',');
                if let Some(value) = node.get_property(name) {
                    if let Some(cell) = encode_cell(&value, *kind) {
                        push_cell(&mut line, &cell);
                    }
                }
            }
            line.push('\n');
            write_all(&mut out, relative, &line)?;
            written += 1;
        }
    }
    out.flush()
        .map_err(|e| format!("Failed to write {relative}: {e}"))?;
    Ok(written)
}

/// Where the relationship CSVs go and how they are keyed.
struct Layout<'a> {
    keys: &'a BTreeMap<(String, String), String>,
    paths: &'a ExportPaths,
    output: &'a Path,
}

/// One pass over a source type's nodes feeding every relationship CSV that
/// leaves it.
fn write_source_edges(
    graph: &DirGraph,
    scope: &Scope,
    manifest: &ExportManifest,
    source_type: &str,
    layout: &Layout<'_>,
) -> Result<BTreeMap<String, usize>, String> {
    struct Sink {
        out: BufWriter<File>,
        relative: String,
        columns: Vec<(String, ColumnKind)>,
        written: usize,
    }
    let mut sinks: HashMap<String, Sink> = HashMap::new();
    for ((rel, source), key) in layout.keys {
        if source != source_type {
            continue;
        }
        let relative = layout.paths.connection(key).to_string();
        let info = &manifest.relationship_types[rel].sources[source];
        let columns: Vec<(String, ColumnKind)> = info
            .properties
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        let mut out = open(layout.output, &relative)?;
        let mut header = standard_columns(info.properties.keys()).join(",");
        for (name, _) in &columns {
            header.push(',');
            push_cell(&mut header, name);
        }
        header.push('\n');
        write_all(&mut out, &relative, &header)?;
        sinks.insert(
            rel.clone(),
            Sink {
                out,
                relative,
                columns,
                written: 0,
            },
        );
    }
    let mut line = String::new();
    if let Some(nodes) = graph.type_indices.get(source_type) {
        for idx in nodes.iter() {
            if !scope.contains(idx) {
                continue;
            }
            let Some(src) = graph.graph.node_view(idx) else {
                continue;
            };
            let source_id = encode_id(&src.id());
            for edge in graph.graph.edges(idx) {
                let target: NodeIndex = edge.target();
                if !scope.contains(target) {
                    continue;
                }
                let w = edge.weight();
                let Some(sink) = sinks.get_mut(w.connection_type_str(&graph.interner)) else {
                    continue;
                };
                let Some(tgt) = graph.graph.node_view(target) else {
                    continue;
                };
                line.clear();
                push_cell(&mut line, &source_id);
                line.push(',');
                push_cell(&mut line, source_type);
                line.push(',');
                push_cell(&mut line, &encode_id(&tgt.id()));
                line.push(',');
                push_cell(&mut line, tgt.node_type_str(&graph.interner));
                let props = w.properties_cloned(&graph.interner);
                for (name, kind) in &sink.columns {
                    line.push(',');
                    if let Some(cell) = props.get(name).and_then(|v| encode_cell(v, *kind)) {
                        push_cell(&mut line, &cell);
                    }
                }
                line.push('\n');
                write_all(&mut sink.out, &sink.relative, &line)?;
                sink.written += 1;
            }
        }
    }
    let mut counts = BTreeMap::new();
    for (rel, mut sink) in sinks {
        sink.out
            .flush()
            .map_err(|e| format!("Failed to write {}: {e}", sink.relative))?;
        counts.insert(rel, sink.written);
    }
    Ok(counts)
}

fn temporal_json(from: &str, to: &str, convention: &str) -> Json {
    json!({"from": from, "to": to, "convention": convention})
}

fn build_blueprint(
    manifest: &ExportManifest,
    paths: &ExportPaths,
    keys: &BTreeMap<(String, String), String>,
) -> String {
    let mut nodes = Map::new();
    for (name, info) in &manifest.node_types {
        let mut spec = Map::new();
        spec.insert("csv".into(), json!(paths.node(name)));
        spec.insert("pk".into(), json!("id"));
        spec.insert("title".into(), json!("title"));
        let mut properties = Map::new();
        properties.insert("id".into(), json!(id_keyword(info.id_kind)));
        properties.insert("title".into(), json!(keyword(info.title_kind)));
        for (prop, kind) in &info.properties {
            properties.insert(prop.clone(), json!(keyword(*kind)));
        }
        spec.insert("properties".into(), Json::Object(properties));
        if !info.labels.is_empty() {
            spec.insert("labels".into(), json!(info.labels));
        }
        if let Some(parent) = &info.parent {
            spec.insert("parent".into(), json!(parent));
        }
        if let Some(d) = manifest
            .temporal
            .iter()
            .find(|d| d.target == "node" && &d.name == name)
        {
            spec.insert(
                "temporal".into(),
                temporal_json(&d.from, &d.to, &d.convention),
            );
        }
        let mut junctions = Map::new();
        for (rel, rel_info) in &manifest.relationship_types {
            let Some(src) = rel_info.sources.get(name) else {
                continue;
            };
            let key = &keys[&(rel.clone(), name.clone())];
            let mut j = Map::new();
            j.insert("csv".into(), json!(paths.connection(key)));
            let std = standard_columns(src.properties.keys());
            j.insert("source_fk".into(), json!(std[0]));
            j.insert("target_fk".into(), json!(std[2]));
            if src.targets.len() > 1 {
                j.insert("target".into(), json!(src.targets));
                j.insert("target_type_column".into(), json!(std[3]));
            } else {
                j.insert("target".into(), json!(src.targets[0]));
            }
            if !src.properties.is_empty() {
                let columns: Vec<&String> = src.properties.keys().collect();
                j.insert("properties".into(), json!(columns));
                // No edge-side point pass: a point column stays text.
                let types: Map<String, Json> = src
                    .properties
                    .iter()
                    .map(|(p, k)| {
                        let k = if *k == ColumnKind::Point {
                            ColumnKind::String
                        } else {
                            *k
                        };
                        (p.clone(), json!(keyword(k)))
                    })
                    .collect();
                j.insert("property_types".into(), Json::Object(types));
            }
            if let Some(d) = manifest.temporal.iter().find(|d| {
                d.target == "relationship"
                    && &d.name == rel
                    && d.source_type.as_deref() == Some(name.as_str())
            }) {
                j.insert(
                    "temporal".into(),
                    temporal_json(&d.from, &d.to, &d.convention),
                );
            }
            junctions.insert(rel.clone(), Json::Object(j));
        }
        if !junctions.is_empty() {
            spec.insert(
                "connections".into(),
                json!({ "junction_edges": Json::Object(junctions) }),
            );
        }
        nodes.insert(name.clone(), Json::Object(spec));
    }
    let blueprint = json!({
        "settings": {"root": ".", "manifest": "manifest.json"},
        "nodes": Json::Object(nodes),
    });
    serde_json::to_string_pretty(&blueprint).expect("a blueprint serializes")
}

/// Export the graph (or selection) to an organized CSV directory tree.
///
/// Creates:
/// - `nodes/<Type>.csv` for each node type (sub-nodes nested under parent)
/// - `connections/<REL>[.<Source>].csv` for each relationship type and source
/// - `blueprint.json` for lossless re-import via `from_blueprint()`
/// - `manifest.json`, the [`ExportManifest`] the blueprint's
///   `settings.manifest` points at
pub fn to_csv_dir(
    graph: &DirGraph,
    output_dir: &str,
    selection: Option<&CurrentSelection>,
    parent_types: &HashMap<String, String>,
) -> Result<ExportSummary, String> {
    if output_dir.trim().is_empty() {
        return Err("export_csv: output_dir must not be empty".to_string());
    }
    let output = Path::new(output_dir);
    let scope = Scope::new(graph, selection);
    let manifest = ExportManifest::build_in(graph, &scope, parent_types)?;
    // A CSV id column holds one kind: `{id: 1}` and `{id: '1'}` would both
    // read back as the text `1` and become one node.
    if let Some((name, _)) = manifest
        .node_types
        .iter()
        .find(|(_, info)| info.id_kind == ColumnKind::Mixed)
    {
        return Err(format!(
            "export_csv: the ids of node type '{name}' mix kinds (for example 1 and '1'), \
             which a CSV id column cannot keep apart. Export as RDF (export_rdf) or save a \
             .kgl file instead, or give the type ids of one kind."
        ));
    }

    let mut keys: BTreeMap<(String, String), String> = BTreeMap::new();
    for (rel, info) in &manifest.relationship_types {
        for source in info.sources.keys() {
            keys.insert(
                (rel.clone(), source.clone()),
                connection_key(rel, source, info.sources.len()),
            );
        }
    }
    let paths = ExportPaths::new(manifest.node_types.keys(), keys.values(), parent_types)?;

    std::fs::create_dir_all(output.join("nodes"))
        .map_err(|e| format!("Failed to create nodes directory: {e}"))?;
    if !keys.is_empty() {
        std::fs::create_dir_all(output.join("connections"))
            .map_err(|e| format!("Failed to create connections directory: {e}"))?;
    }
    for relative_directory in paths.node_directories() {
        std::fs::create_dir_all(output.join(relative_directory))
            .map_err(|e| format!("Failed to create sub-node directory: {e}"))?;
    }

    let mut summary = ExportSummary {
        output_dir: output_dir.to_string(),
        nodes: BTreeMap::new(),
        connections: BTreeMap::new(),
        files_written: 0,
        log_lines: vec![format!("Exporting to {output_dir}...")],
    };
    for (node_type, info) in &manifest.node_types {
        let relative = paths.node(node_type);
        let n = write_node_csv(graph, &scope, &manifest, node_type, output, relative)?;
        summary.log_lines.push(format!(
            "  {relative}: {n} nodes, {} properties",
            info.properties.len()
        ));
        summary.nodes.insert(node_type.clone(), n);
        summary.files_written += 1;
    }
    let source_types: std::collections::BTreeSet<&String> =
        keys.keys().map(|(_, source)| source).collect();
    for source_type in source_types {
        let layout = Layout {
            keys: &keys,
            paths: &paths,
            output,
        };
        let counts = write_source_edges(graph, &scope, &manifest, source_type, &layout)?;
        for (rel, n) in counts {
            let key = &keys[&(rel.clone(), source_type.clone())];
            summary
                .log_lines
                .push(format!("  {}: {n} edges", paths.connection(key)));
            *summary.connections.entry(rel).or_default() += n;
            summary.files_written += 1;
        }
    }

    std::fs::write(
        output.join("blueprint.json"),
        build_blueprint(&manifest, &paths, &keys),
    )
    .map_err(|e| format!("Failed to write blueprint.json: {e}"))?;
    std::fs::write(output.join("manifest.json"), manifest.to_json())
        .map_err(|e| format!("Failed to write manifest.json: {e}"))?;
    summary.log_lines.push("  blueprint.json".to_string());
    summary.log_lines.push("  manifest.json".to_string());
    summary.files_written += 2;

    let total_nodes: usize = summary.nodes.values().sum();
    let total_edges: usize = summary.connections.values().sum();
    summary.log_lines.push(format!(
        "Done: {total_nodes} nodes, {total_edges} edges, {} files written",
        summary.files_written
    ));
    Ok(summary)
}
