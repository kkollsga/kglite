//! The RDF export behind `export_rdf`: N-Quads or TriG, RDF 1.2 flavoured.
//!
//! Layout and vocabulary come from [`kg_vocab`]; the mapping is
//! - a node → `<base>node/<Type>/<id>` with `rdf:type <base>type/<Type>`,
//!   `rdfs:label` for the title and `<base>prop/<name>` per property;
//! - an edge → the plain statement `s <base>rel/<TYPE> o`, written once per
//!   edge; an edge with properties also gets a reifier
//!   `_:eN rdf:reifies <<( s p o )>>` carrying `_:eN <base>prop/<name> value`;
//! - the [`ExportManifest`] → one `kg:manifest` statement in `<base>meta`.
//!
//! Property values are typed literals: ints `xsd:integer`, floats
//! `xsd:double`, booleans `xsd:boolean`, dates `xsd:date`, timestamps
//! `xsd:dateTime` (no zone), points `geo:wktLiteral` (`POINT(lon lat)`),
//! durations `xsd:duration`, lists and maps `kg:json` (JSON; a typed value
//! inside one is a tagged object, see [`super::typed_json`]). A duration
//! whose months, days and seconds disagree in sign has no `xsd:duration`
//! spelling and is written as `kg:duration` (`months,days,seconds`). A
//! language-map property is an ordinary map and is written as `kg:json`.
//!
//! No prefixes are declared: the importer recovers names from the IRI
//! fragments. Nodes sharing an id (valid-time versions) are told apart by a
//! `;<node index>` suffix on all but the node the id index answers for.
//!
//! The file is written statement by statement through a buffered writer, so
//! memory does not grow with the graph's size.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufWriter, Write};

use oxrdf::vocab::{rdf, rdfs, xsd};
use oxrdf::{
    BlankNode, GraphNameRef, Literal, NamedNode, NamedNodeRef, NamedOrBlankNode, Quad, Term, Triple,
};
use oxttl::{NQuadsSerializer, TriGSerializer};
use petgraph::graph::NodeIndex;

use super::kg_vocab::{
    encode_segment, DUPLICATE_MARK, KG_DURATION, KG_JSON, KG_MANIFEST, META_GRAPH, NODE_PATH,
    PROP_PATH, REL_PATH, TYPE_PATH,
};
use super::manifest::{ColumnKind, ExportManifest, Scope};
use super::typed_json::to_json;
use crate::datatypes::values::{raw_string, Value};
use crate::graph::dir_graph::DirGraph;
use crate::graph::io::rdf::well_known_namespace;
use crate::graph::schema::CurrentSelection;
use crate::graph::storage::GraphRead;

const GEO_WKT: &str = "http://www.opengis.net/ont/geosparql#wktLiteral";
const SCHEMA_VALID_FROM: &str = "http://schema.org/validFrom";
const SCHEMA_VALID_THROUGH: &str = "http://schema.org/validThrough";
/// Base IRI used when the caller names none.
pub const DEFAULT_BASE: &str = "https://kglite.example/";

/// Serialisation of an RDF export.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RdfFormat {
    NQuads,
    TriG,
}

impl RdfFormat {
    /// `"nq"` / `"nquads"` or `"trig"`.
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "nq" | "nquads" | "n-quads" => Some(Self::NQuads),
            "trig" => Some(Self::TriG),
            _ => None,
        }
    }

    /// The format a file extension names.
    pub fn from_path(path: &str) -> Option<Self> {
        path.rsplit('.').next().and_then(Self::parse)
    }
}

/// Options of [`to_rdf`].
#[derive(Clone, Debug)]
pub struct RdfExportOptions {
    pub format: RdfFormat,
    /// IRI prefix of every generated IRI; ends with `/` or `#`.
    pub base: String,
    /// Also write `schema:validFrom` / `schema:validThrough` for the bounds
    /// of declared valid-time intervals. Those statements re-import as
    /// ordinary `schema__validFrom` / `schema__validThrough` properties.
    pub schema_org: bool,
}

impl Default for RdfExportOptions {
    fn default() -> Self {
        Self {
            format: RdfFormat::NQuads,
            base: DEFAULT_BASE.to_string(),
            schema_org: false,
        }
    }
}

/// Summary of an RDF export.
pub struct RdfExportSummary {
    pub output_path: String,
    /// Node counts per type.
    pub nodes: BTreeMap<String, usize>,
    /// Edge counts per relationship type.
    pub connections: BTreeMap<String, usize>,
    /// Statements written, manifest included.
    pub statements: u64,
}

enum Sink {
    NQuads(oxttl::nquads::WriterNQuadsSerializer<BufWriter<File>>),
    TriG(Box<oxttl::trig::WriterTriGSerializer<BufWriter<File>>>),
}

impl Sink {
    fn write(&mut self, quad: &Quad) -> Result<(), String> {
        match self {
            Sink::NQuads(s) => s.serialize_quad(quad),
            Sink::TriG(s) => s.serialize_quad(quad),
        }
        .map_err(|e| format!("Failed to write RDF: {e}"))
    }

    fn finish(self) -> Result<(), String> {
        let mut out = match self {
            Sink::NQuads(s) => s.finish(),
            Sink::TriG(s) => s
                .finish()
                .map_err(|e| format!("Failed to write RDF: {e}"))?,
        };
        out.flush().map_err(|e| format!("Failed to write RDF: {e}"))
    }
}

/// The IRIs and statements of one export.
struct Emitter<'a> {
    base: &'a str,
    sink: Sink,
    statements: u64,
    reifiers: u64,
    xsd_integer: NamedNode,
    xsd_double: NamedNode,
    xsd_boolean: NamedNode,
    xsd_date: NamedNode,
    xsd_date_time: NamedNode,
    xsd_time: NamedNode,
    xsd_duration: NamedNode,
    geo_wkt: NamedNode,
    kg_json: NamedNode,
    kg_duration: NamedNode,
}

fn iri(text: String) -> NamedNode {
    // Every variable part is percent-encoded and the base was validated.
    NamedNode::new_unchecked(text)
}

impl<'a> Emitter<'a> {
    fn new(base: &'a str, sink: Sink) -> Self {
        let named = |n: NamedNodeRef<'_>| n.into_owned();
        Self {
            base,
            sink,
            statements: 0,
            reifiers: 0,
            xsd_integer: named(xsd::INTEGER),
            xsd_double: named(xsd::DOUBLE),
            xsd_boolean: named(xsd::BOOLEAN),
            xsd_date: named(xsd::DATE),
            xsd_date_time: named(xsd::DATE_TIME),
            xsd_time: named(xsd::TIME),
            xsd_duration: named(xsd::DURATION),
            geo_wkt: iri(GEO_WKT.to_string()),
            kg_json: iri(KG_JSON.to_string()),
            kg_duration: iri(KG_DURATION.to_string()),
        }
    }

    fn quad(
        &mut self,
        subject: NamedOrBlankNode,
        predicate: NamedNode,
        object: Term,
        graph: Option<&NamedNode>,
    ) -> Result<(), String> {
        let graph_name = graph.map_or(GraphNameRef::DefaultGraph, GraphNameRef::from);
        let quad = Quad::new(subject, predicate, object, graph_name);
        self.statements += 1;
        self.sink.write(&quad)
    }

    fn type_iri(&self, node_type: &str) -> NamedNode {
        iri(format!(
            "{}{TYPE_PATH}{}",
            self.base,
            encode_segment(node_type)
        ))
    }

    fn prop_iri(&self, name: &str) -> NamedNode {
        iri(format!("{}{PROP_PATH}{}", self.base, encode_segment(name)))
    }

    fn rel_iri(&self, rel: &str) -> NamedNode {
        iri(format!("{}{REL_PATH}{}", self.base, encode_segment(rel)))
    }

    /// The node's IRI: `<base>node/<Type>/<id>`, plus `;<index>` when another
    /// node of the type is the one the id index answers for.
    fn node_iri(
        &self,
        graph: &DirGraph,
        manifest: &ExportManifest,
        idx: NodeIndex,
    ) -> Option<NamedNode> {
        let view = graph.graph.node_view(idx)?;
        let node_type = view.node_type_str(&graph.interner);
        let id = view.id();
        let mixed = manifest
            .node_types
            .get(node_type)
            .is_some_and(|t| t.id_kind == ColumnKind::Mixed);
        let mut segment = String::new();
        if mixed {
            segment.push_str(match &*id {
                Value::Int64(_) | Value::UniqueId(_) => "i-",
                Value::String(_) => "s-",
                _ => "o-",
            });
        }
        segment.push_str(&raw_string(&id));
        let mut text = format!(
            "{}{NODE_PATH}{}/{}",
            self.base,
            encode_segment(node_type),
            encode_segment(&segment)
        );
        if matches!(graph.id_indices.lookup(node_type, &id), Some(winner) if winner != idx) {
            text.push(DUPLICATE_MARK);
            text.push_str(&idx.index().to_string());
        }
        Some(iri(text))
    }

    /// The typed literal of a value; `None` for null.
    fn literal(&self, value: &Value) -> Option<Literal> {
        let typed = |text: String, dt: &NamedNode| Literal::new_typed_literal(text, dt.clone());
        Some(match value {
            Value::Null | Value::NodeRef(_) => return None,
            Value::String(s) => Literal::new_simple_literal(s.clone()),
            Value::Int64(n) => typed(n.to_string(), &self.xsd_integer),
            Value::UniqueId(n) => typed(n.to_string(), &self.xsd_integer),
            Value::Float64(f) => typed(
                if f.is_nan() {
                    "NaN".to_string()
                } else if f.is_infinite() {
                    if *f > 0.0 { "INF" } else { "-INF" }.to_string()
                } else {
                    f.to_string()
                },
                &self.xsd_double,
            ),
            Value::Boolean(b) => typed(b.to_string(), &self.xsd_boolean),
            Value::DateTime(d) => typed(d.to_string(), &self.xsd_date),
            Value::Timestamp(t) => typed(
                t.format("%Y-%m-%dT%H:%M:%S%.f").to_string(),
                &self.xsd_date_time,
            ),
            // `xsd:time` always spells the seconds; the offset marks a `Time`
            // and its absence a `LocalTime`, which is how the reader tells them apart.
            Value::LocalTime(n) => typed(xsd_time_lexical(*n, None), &self.xsd_time),
            Value::Time { nanos, offset_secs } => {
                typed(xsd_time_lexical(*nanos, Some(*offset_secs)), &self.xsd_time)
            }
            // RDF has no zone names: a named zone is written as the offset it had.
            Value::ZonedDateTime {
                utc, offset_secs, ..
            } => typed(xsd_zoned_lexical(*utc, *offset_secs), &self.xsd_date_time),
            Value::Point { lat, lon } => typed(format!("POINT({lon} {lat})"), &self.geo_wkt),
            // A sub-second part is an `xsd:duration` fraction when the sign is
            // uniform; the mixed-sign case takes the lossless `kg:duration` form.
            Value::Duration {
                months,
                days,
                seconds,
                nanos: 0,
            } => match duration_lexical(*months, *days, *seconds) {
                Some(text) => typed(text, &self.xsd_duration),
                None => typed(format!("{months},{days},{seconds}"), &self.kg_duration),
            },
            Value::Duration {
                months,
                days,
                seconds,
                nanos,
            } => match duration_lexical_fine(*months, *days, *seconds, *nanos) {
                Some(text) => typed(text, &self.xsd_duration),
                None => typed(
                    format!("{months},{days},{seconds},{nanos}"),
                    &self.kg_duration,
                ),
            },
            Value::List(_) | Value::Map(_) => typed(to_json(value).to_string(), &self.kg_json),
            other => Literal::new_simple_literal(raw_string(other)),
        })
    }
}

/// The `xsd:time` lexical form: `hh:mm:ss[.fraction]` with trailing fraction
/// zeros trimmed, then `Z` or `±hh:mm` when an offset is given.
/// `2020-01-02T03:04:05.25+02:00`: the seconds are always spelled and the offset
/// is whole minutes. A sub-minute offset has no `xsd:dateTime` spelling, so that
/// instant is written at `Z`.
fn xsd_zoned_lexical(utc: chrono::NaiveDateTime, offset_secs: i32) -> String {
    use crate::datatypes::time_of_day::format_offset;
    use crate::datatypes::zoned_datetime::wall_clock;
    let offset_secs = if offset_secs % 60 == 0 {
        offset_secs
    } else {
        0
    };
    format!(
        "{}{}",
        wall_clock(utc, offset_secs).format("%Y-%m-%dT%H:%M:%S%.f"),
        format_offset(offset_secs)
    )
}

fn xsd_time_lexical(nanos: u64, offset_secs: Option<i32>) -> String {
    let secs = nanos / 1_000_000_000;
    let sub = nanos % 1_000_000_000;
    let mut text = format!("{:02}:{:02}:{:02}", secs / 3600, secs / 60 % 60, secs % 60);
    if sub != 0 {
        let fraction = format!("{sub:09}");
        text.push('.');
        text.push_str(fraction.trim_end_matches('0'));
    }
    if let Some(offset) = offset_secs {
        // A sub-minute offset has no xsd spelling; it is written with seconds,
        // which this crate's reader accepts and a strict xsd validator would not.
        text.push_str(&crate::datatypes::time_of_day::format_offset(offset));
    }
    text
}

/// `P{y}Y{m}M{d}DT{s}S` for a duration whose fields agree in sign; `None`
/// when they do not, since `xsd:duration` has one sign for the whole value.
fn duration_lexical(months: i32, days: i32, seconds: i64) -> Option<String> {
    let negative = months < 0 || days < 0 || seconds < 0;
    if negative && (months > 0 || days > 0 || seconds > 0) {
        return None;
    }
    let (months, days, seconds) = (
        months.unsigned_abs(),
        days.unsigned_abs(),
        seconds.unsigned_abs(),
    );
    let mut text = String::from(if negative { "-P" } else { "P" });
    if months / 12 > 0 {
        text.push_str(&format!("{}Y", months / 12));
    }
    if months % 12 > 0 {
        text.push_str(&format!("{}M", months % 12));
    }
    if days > 0 {
        text.push_str(&format!("{days}D"));
    }
    if seconds > 0 || text.len() == 1 + usize::from(negative) {
        text.push_str(&format!("T{seconds}S"));
    }
    Some(text)
}

/// `xsd:duration` for a duration with a non-zero `nanos` (the sign rides on
/// `seconds`, nanos is 1..1e9). A negative total spells `-(s + f)` as
/// `(-seconds - 1)` whole seconds and `1e9 - nanos` fractional ones. `None`
/// when months or days disagree in sign with the total.
fn duration_lexical_fine(months: i32, days: i32, seconds: i64, nanos: i32) -> Option<String> {
    let negative = seconds < 0;
    if negative && (months > 0 || days > 0) || !negative && (months < 0 || days < 0) {
        return None;
    }
    let (whole, frac) = if negative {
        (
            seconds.checked_neg()?.checked_sub(1)?,
            1_000_000_000 - nanos,
        )
    } else {
        (seconds, nanos)
    };
    let mut text = if months == 0 && days == 0 {
        String::from(if negative { "-P" } else { "P" })
    } else {
        duration_lexical(months, days, 0)?
    };
    let frac = format!("{frac:09}");
    text.push_str(&format!("T{whole}.{}S", frac.trim_end_matches('0')));
    Some(text)
}

fn validate_base(base: &str) -> Result<(), String> {
    NamedNode::new(base).map_err(|e| format!("Invalid RDF export base '{base}': {e}"))?;
    if !(base.ends_with('/') || base.ends_with('#')) {
        return Err(format!("RDF export base '{base}' must end with '/' or '#'"));
    }
    if let Some(ns) = well_known_namespace(base) {
        return Err(format!(
            "RDF export base '{base}' lies inside the well-known namespace {ns}, \
             where a re-import would rename types and properties; choose another base"
        ));
    }
    Ok(())
}

/// Valid-time bounds declared for `node_type`, as (from, to) property names.
fn node_bounds<'m>(manifest: &'m ExportManifest, node_type: &str) -> Vec<(&'m str, &'m str)> {
    let labels = manifest.node_types.get(node_type).map(|t| &t.labels);
    manifest
        .temporal
        .iter()
        .filter(|d| {
            d.target == "node"
                && (d.name == node_type || labels.is_some_and(|l| l.contains(&d.name)))
        })
        .map(|d| (d.from.as_str(), d.to.as_str()))
        .collect()
}

/// Valid-time bounds declared for `rel` leaving `source_type`.
fn rel_bounds<'m>(
    manifest: &'m ExportManifest,
    rel: &str,
    source_type: &str,
) -> Vec<(&'m str, &'m str)> {
    manifest
        .temporal
        .iter()
        .filter(|d| {
            d.target == "relationship"
                && d.name == rel
                && d.source_type.as_deref().is_none_or(|s| s == source_type)
        })
        .map(|d| (d.from.as_str(), d.to.as_str()))
        .collect()
}

/// One edge as it is written: its ends, properties and declared bounds.
struct EdgeRow<'a> {
    subject: &'a NamedNode,
    predicate: &'a NamedNode,
    object: &'a NamedNode,
    props: &'a HashMap<String, Value>,
    columns: &'a [&'a String],
    bounds: &'a [(&'a str, &'a str)],
}

impl Emitter<'_> {
    /// `schema:validFrom` / `schema:validThrough` for each declared bound.
    fn write_alias(
        &mut self,
        subject: &NamedOrBlankNode,
        bounds: &[(&str, &str)],
        value_of: &dyn Fn(&str) -> Option<Value>,
    ) -> Result<(), String> {
        for (from, to) in bounds {
            for (name, predicate) in [(from, SCHEMA_VALID_FROM), (to, SCHEMA_VALID_THROUGH)] {
                if let Some(lit) = value_of(name).and_then(|v| self.literal(&v)) {
                    self.quad(
                        subject.clone(),
                        iri(predicate.to_string()),
                        lit.into(),
                        None,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn write_edge(&mut self, edge: &EdgeRow<'_>, schema_org: bool) -> Result<(), String> {
        let EdgeRow {
            subject,
            predicate,
            object,
            props,
            columns,
            bounds,
        } = edge;
        self.quad(
            (*subject).clone().into(),
            (*predicate).clone(),
            (*object).clone().into(),
            None,
        )?;
        let literals: Vec<(&String, Literal)> = columns
            .iter()
            .filter_map(|name| Some((*name, self.literal(props.get(*name)?)?)))
            .collect();
        if literals.is_empty() {
            return Ok(());
        }
        self.reifiers += 1;
        let reifier: NamedOrBlankNode =
            BlankNode::new_unchecked(format!("e{}", self.reifiers)).into();
        let term = Term::Triple(Box::new(Triple::new(
            (*subject).clone(),
            (*predicate).clone(),
            (*object).clone(),
        )));
        self.quad(reifier.clone(), rdf::REIFIES.into_owned(), term, None)?;
        for (name, lit) in literals {
            self.quad(reifier.clone(), self.prop_iri(name), lit.into(), None)?;
        }
        if schema_org {
            self.write_alias(&reifier, bounds, &|n| props.get(n).cloned())?;
        }
        Ok(())
    }
}

/// Export the graph (or selection) as N-Quads or TriG to `path`.
pub fn to_rdf(
    graph: &DirGraph,
    path: &str,
    selection: Option<&CurrentSelection>,
    parent_types: &HashMap<String, String>,
    options: &RdfExportOptions,
) -> Result<RdfExportSummary, String> {
    if path.trim().is_empty() {
        return Err("export_rdf: path must not be empty".to_string());
    }
    validate_base(&options.base)?;
    let scope = Scope::new(graph, selection);
    let manifest = ExportManifest::build_in(graph, &scope, parent_types)?;
    // The id index is built lazily: until a lookup has run it answers `None`,
    // which would leave every version of a repeated id on one IRI.
    for node_type in manifest.node_types.keys() {
        graph.ensure_id_index(node_type);
    }
    let file = File::create(path).map_err(|e| format!("Failed to create {path}: {e}"))?;
    let out = BufWriter::new(file);
    let sink = match options.format {
        RdfFormat::NQuads => Sink::NQuads(NQuadsSerializer::new().for_writer(out)),
        RdfFormat::TriG => Sink::TriG(Box::new(TriGSerializer::new().for_writer(out))),
    };
    let mut emit = Emitter::new(&options.base, sink);

    let meta = iri(format!("{}{META_GRAPH}", options.base));
    let manifest_literal = Literal::new_typed_literal(manifest.to_json(), emit.kg_json.clone());
    emit.quad(
        meta.clone().into(),
        iri(KG_MANIFEST.to_string()),
        manifest_literal.into(),
        Some(&meta),
    )?;

    let rdf_type = rdf::TYPE.into_owned();
    let label = rdfs::LABEL.into_owned();
    let mut summary = RdfExportSummary {
        output_path: path.to_string(),
        nodes: BTreeMap::new(),
        connections: BTreeMap::new(),
        statements: 0,
    };
    for (node_type, info) in &manifest.node_types {
        let type_iri = emit.type_iri(node_type);
        let columns: Vec<&String> = info.properties.keys().collect();
        let bounds = node_bounds(&manifest, node_type);
        let mut written = 0usize;
        let Some(nodes) = graph.type_indices.get(node_type) else {
            continue;
        };
        for idx in nodes.iter() {
            if !scope.contains(idx) {
                continue;
            }
            let (Some(view), Some(subject)) = (
                graph.graph.node_view(idx),
                emit.node_iri(graph, &manifest, idx),
            ) else {
                continue;
            };
            let subject_term: NamedOrBlankNode = subject.clone().into();
            emit.quad(
                subject_term.clone(),
                rdf_type.clone(),
                type_iri.clone().into(),
                None,
            )?;
            if let Some(lit) = emit.literal(&view.title()) {
                emit.quad(subject_term.clone(), label.clone(), lit.into(), None)?;
            }
            for name in &columns {
                if let Some(lit) = view.get_property(name).and_then(|v| emit.literal(&v)) {
                    emit.quad(subject_term.clone(), emit.prop_iri(name), lit.into(), None)?;
                }
            }
            if options.schema_org {
                emit.write_alias(&subject_term, &bounds, &|n| {
                    view.get_property(n).map(std::borrow::Cow::into_owned)
                })?;
            }
            for edge in graph.graph.edges(idx) {
                let target = edge.target();
                if !scope.contains(target) {
                    continue;
                }
                let Some(object) = emit.node_iri(graph, &manifest, target) else {
                    continue;
                };
                let weight = edge.weight();
                let rel = weight.connection_type_str(&graph.interner);
                let Some(source_info) = manifest
                    .relationship_types
                    .get(rel)
                    .and_then(|r| r.sources.get(node_type))
                else {
                    continue;
                };
                let rel_columns: Vec<&String> = source_info.properties.keys().collect();
                let rel_bounds = rel_bounds(&manifest, rel, node_type);
                let props = weight.properties_cloned(&graph.interner);
                let predicate = emit.rel_iri(rel);
                let row = EdgeRow {
                    subject: &subject,
                    predicate: &predicate,
                    object: &object,
                    props: &props,
                    columns: &rel_columns,
                    bounds: &rel_bounds,
                };
                emit.write_edge(&row, options.schema_org)?;
                *summary.connections.entry(rel.to_string()).or_default() += 1;
            }
            written += 1;
        }
        summary.nodes.insert(node_type.clone(), written);
    }
    summary.statements = emit.statements;
    emit.sink.finish()?;
    Ok(summary)
}

#[cfg(test)]
#[path = "rdf_writer_tests.rs"]
mod rdf_writer_tests;
