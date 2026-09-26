//! Declarations through a `.kgl` save and load: the `temporal_declarations`
//! entries, the legacy keys an older build reads, and files that carry only
//! the legacy keys.

use std::collections::HashMap;
use std::sync::Arc;

use serde::Deserialize;

use super::declarations::{declare, list, TemporalTarget};
use super::eval::IntervalConvention::{Closed, HalfOpen};
use super::persist::PersistedDeclaration;
use crate::datatypes::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::features::temporal::declarations::TemporalDeclarations;
use crate::graph::io::file::{load_kgl_bytes, prepare_kgl_write, write_kgl_to};
use crate::graph::schema::TemporalConfig;
use crate::graph::session::execute::{execute_mut, ExecuteOptions};

fn graph(queries: &[&str]) -> DirGraph {
    let mut graph = DirGraph::new();
    let params: HashMap<String, Value> = HashMap::new();
    for query in queries {
        execute_mut(&mut graph, query, &ExecuteOptions::eager(&params))
            .unwrap_or_else(|e| panic!("{query}: {e}"));
    }
    graph
}

fn rel(rel_type: &str, source: Option<&str>) -> TemporalTarget {
    TemporalTarget::Relationship {
        rel_type: rel_type.into(),
        source_type: source.map(str::to_string),
    }
}

fn config(from: &str, to: &str) -> TemporalConfig {
    TemporalConfig {
        valid_from: from.into(),
        valid_to: to.into(),
        ..TemporalConfig::default()
    }
}

/// Append an unkeyed closed config the way builds before declarations did,
/// unvalidated — the state a file carrying only the legacy keys loads into.
fn legacy_push_edge(graph: &mut DirGraph, rel_type: &str, config: TemporalConfig) {
    graph
        .temporal
        .edges
        .entry(rel_type.to_string())
        .or_default()
        .push(config);
}

fn encode(graph: DirGraph) -> Vec<u8> {
    let mut graph = Arc::new(graph);
    prepare_kgl_write(&mut graph);
    let mut bytes = Vec::new();
    write_kgl_to(&graph, &mut bytes).unwrap();
    bytes
}

/// The metadata block's JSON text.
fn metadata(bytes: &[u8]) -> String {
    let len = u32::from_le_bytes(bytes[9..13].try_into().unwrap()) as usize;
    String::from_utf8(bytes[13..13 + len].to_vec()).unwrap()
}

/// The raw JSON text of one top-level metadata key, `None` when absent.
fn key<'m>(metadata: &'m str, name: &str) -> Option<&'m str> {
    let tag = format!("\"{name}\":");
    let start = metadata.find(&tag)? + tag.len();
    let mut stream =
        serde_json::Deserializer::from_str(&metadata[start..]).into_iter::<serde_json::Value>();
    stream.next()?.ok()?;
    Some(&metadata[start..start + stream.byte_offset()])
}

const LICENSEES: &[&str] = &[
    "CREATE (:Field {id: 1, vf: '2000-01-01', vt: '2010-12-31'}), (:Licence {id: 10}), (:Company {id: 100})",
    "MATCH (f:Field), (c:Company) CREATE (f)-[:HAS_LICENSEE {ff: '2000-01-01', ft: '2009-12-31'}]->(c)",
    "MATCH (l:Licence), (c:Company) CREATE (l)-[:HAS_LICENSEE {lf: '1990-01-01', lt: '1999-12-31'}]->(c)",
    "MATCH (f:Field), (c:Company) CREATE (f)-[:OPERATES {of: '2001-01-01', ot: '2002-01-01'}]->(c)",
    "MATCH (l:Licence), (c:Company) CREATE (l)-[:OPERATES {of: '2003-01-01', ot: null}]->(c)",
    "MATCH (f:Field), (c:Company) CREATE (f)-[:AUDITS {af: '2001-01-01', at: '2001-06-30'}]->(c)",
];

/// Closed node; keyed closed and keyed half-open with different properties
/// (`HAS_LICENSEE`); unkeyed closed beside keyed closed with the same
/// properties (`OPERATES`); unkeyed half-open (`AUDITS`); two different
/// unkeyed closed configs in `set_temporal` order (`SUPPLIES`).
fn declared() -> DirGraph {
    let mut g = graph(LICENSEES);
    let node = TemporalTarget::Node("Field".into());
    declare(&mut g, &node, "vf", "vt", Closed).unwrap();
    let field = rel("HAS_LICENSEE", Some("Field"));
    declare(&mut g, &field, "ff", "ft", Closed).unwrap();
    let licence = rel("HAS_LICENSEE", Some("Licence"));
    declare(&mut g, &licence, "lf", "lt", HalfOpen).unwrap();
    for source in [None, Some("Field")] {
        declare(&mut g, &rel("OPERATES", source), "of", "ot", Closed).unwrap();
    }
    declare(&mut g, &rel("AUDITS", None), "af", "at", HalfOpen).unwrap();
    legacy_push_edge(&mut g, "SUPPLIES", config("sf", "st"));
    legacy_push_edge(&mut g, "SUPPLIES", config("pf", "pt"));
    g
}

#[test]
fn every_declaration_is_written_under_its_own_key_in_list_order() {
    let json = metadata(&encode(declared()));
    assert_eq!(
        key(&json, "temporal_declarations").unwrap(),
        concat!(
            "[",
            r#"{"abutting_rows":0,"convention":"closed","from":"vf","kind":"node","name":"Field","to":"vt"},"#,
            r#"{"abutting_rows":0,"convention":"half_open","from":"af","kind":"relationship","name":"AUDITS","to":"at"},"#,
            r#"{"abutting_rows":0,"convention":"closed","from":"ff","kind":"relationship","name":"HAS_LICENSEE","source_type":"Field","to":"ft"},"#,
            r#"{"abutting_rows":0,"convention":"half_open","from":"lf","kind":"relationship","name":"HAS_LICENSEE","source_type":"Licence","to":"lt"},"#,
            r#"{"abutting_rows":0,"convention":"closed","from":"of","kind":"relationship","name":"OPERATES","source_type":"Field","to":"ot"},"#,
            r#"{"abutting_rows":0,"convention":"closed","from":"of","kind":"relationship","name":"OPERATES","to":"ot"},"#,
            r#"{"convention":"closed","from":"sf","kind":"relationship","name":"SUPPLIES","to":"st"},"#,
            r#"{"convention":"closed","from":"pf","kind":"relationship","name":"SUPPLIES","to":"pt"}"#,
            "]"
        )
    );
}

#[test]
fn the_legacy_keys_hold_closed_unkeyed_configs_in_order_without_new_fields() {
    let json = metadata(&encode(declared()));
    assert_eq!(
        key(&json, "temporal_node_configs").unwrap(),
        r#"{"Field":{"valid_from":"vf","valid_to":"vt"}}"#
    );
    // HAS_LICENSEE and OPERATES: a keyed config. AUDITS: half-open.
    assert_eq!(
        key(&json, "temporal_edge_configs").unwrap(),
        r#"{"SUPPLIES":[{"valid_from":"sf","valid_to":"st"},{"valid_from":"pf","valid_to":"pt"}]}"#
    );
}

#[test]
fn a_graph_without_declarations_writes_no_new_key() {
    let json = metadata(&encode(graph(LICENSEES)));
    assert_eq!(key(&json, "temporal_declarations"), None);
    assert_eq!(key(&json, "temporal_node_configs"), Some("{}"));
    assert_eq!(key(&json, "temporal_edge_configs"), Some("{}"));
}

#[test]
fn declarations_round_trip_with_conventions_sources_and_counts() {
    let before = declared();
    let expected = list(&before);
    let loaded = load_kgl_bytes(&encode(before)).unwrap();
    assert_eq!(list(&loaded), expected);
    let counted = |info: &super::declarations::DeclarationInfo| {
        info.abutting_rows == Some(0)
            || matches!(&info.target, TemporalTarget::Relationship { rel_type, .. } if rel_type == "SUPPLIES")
    };
    assert!(expected.iter().all(counted));
    let ambiguous: Vec<bool> = expected.iter().map(|info| info.ambiguous).collect();
    assert_eq!(
        ambiguous,
        [false, false, false, false, false, false, true, true]
    );
}

/// The metadata shape builds before `temporal_declarations` deserialised:
/// the two legacy maps, each config with only its two property names.
#[derive(Deserialize)]
struct OlderMetadata {
    #[serde(default)]
    temporal_node_configs: HashMap<String, OlderConfig>,
    #[serde(default)]
    temporal_edge_configs: HashMap<String, Vec<OlderConfig>>,
}

#[derive(Deserialize, Debug, PartialEq)]
struct OlderConfig {
    valid_from: String,
    valid_to: String,
}

#[test]
fn an_older_reader_sees_the_closed_unkeyed_mirror_and_nothing_half_open_or_keyed() {
    let old: OlderMetadata = serde_json::from_str(&metadata(&encode(declared()))).unwrap();
    let older = |from: &str, to: &str| OlderConfig {
        valid_from: from.into(),
        valid_to: to.into(),
    };
    assert_eq!(old.temporal_node_configs.len(), 1);
    assert_eq!(old.temporal_node_configs["Field"], older("vf", "vt"));
    assert_eq!(old.temporal_edge_configs.len(), 1);
    assert_eq!(
        old.temporal_edge_configs["SUPPLIES"],
        vec![older("sf", "st"), older("pf", "pt")],
        "both, in order, so an older build's first match is what it always was"
    );
    for omitted in ["AUDITS", "HAS_LICENSEE", "OPERATES"] {
        assert!(
            !old.temporal_edge_configs.contains_key(omitted),
            "{omitted}"
        );
    }
}

#[test]
fn a_half_open_node_declaration_is_not_mirrored() {
    let mut g = graph(LICENSEES);
    let node = TemporalTarget::Node("Field".into());
    declare(&mut g, &node, "vf", "vt", HalfOpen).unwrap();
    let json = metadata(&encode(g));
    assert_eq!(key(&json, "temporal_node_configs"), Some("{}"));
    assert!(key(&json, "temporal_declarations")
        .unwrap()
        .contains(r#""convention":"half_open""#));
}

fn from_legacy(edges: Vec<TemporalConfig>) -> TemporalDeclarations {
    TemporalDeclarations::from_file(
        Vec::new(),
        (
            HashMap::new(),
            HashMap::from([("HAS_LICENSEE".to_string(), edges)]),
        ),
    )
}

#[test]
fn a_legacy_list_repeating_one_config_reads_as_that_config_once() {
    let store = from_legacy(vec![config("a", "b"), config("a", "b")]);
    assert_eq!(store.edges("HAS_LICENSEE"), &[config("a", "b")]);
    assert!(!store.is_ambiguous("HAS_LICENSEE"));
    let mut g = DirGraph::new();
    g.temporal = store;
    assert_eq!(
        key(&metadata(&encode(g)), "temporal_edge_configs"),
        Some(r#"{"HAS_LICENSEE":[{"valid_from":"a","valid_to":"b"}]}"#),
        "written back once"
    );
}

#[test]
fn a_legacy_list_of_different_unkeyed_configs_keeps_both_and_is_ambiguous() {
    let mut g = DirGraph::new();
    g.temporal = from_legacy(vec![config("a", "b"), config("c", "d")]);
    assert_eq!(
        g.temporal.edges("HAS_LICENSEE"),
        &[config("a", "b"), config("c", "d")],
        "both kept, in their order, so the first-match choice is unchanged"
    );
    let listed = list(&g);
    assert_eq!(listed.len(), 2);
    assert!(listed.iter().all(|info| info.ambiguous));
    // Still ambiguous after a save and load through the new key, and
    // mirrored whole into the legacy key.
    let bytes = encode(g);
    assert_eq!(
        key(&metadata(&bytes), "temporal_edge_configs"),
        Some(
            r#"{"HAS_LICENSEE":[{"valid_from":"a","valid_to":"b"},{"valid_from":"c","valid_to":"d"}]}"#
        )
    );
    let loaded = load_kgl_bytes(&bytes).unwrap();
    assert!(list(&loaded).iter().all(|info| info.ambiguous));
}

#[test]
fn a_keyed_declaration_beside_one_unkeyed_config_is_not_ambiguous() {
    let mut g = graph(LICENSEES);
    legacy_push_edge(&mut g, "HAS_LICENSEE", config("ff", "ft"));
    let keyed = rel("HAS_LICENSEE", Some("Licence"));
    declare(&mut g, &keyed, "lf", "lt", Closed).unwrap();
    assert!(list(&g).iter().all(|info| !info.ambiguous));
}

#[test]
fn the_new_key_wins_over_the_legacy_keys_when_both_are_present() {
    let entries: Vec<PersistedDeclaration> = serde_json::from_str(
        r#"[{"kind":"relationship","name":"HAS_LICENSEE","from":"x","to":"y","convention":"half_open"}]"#,
    )
    .unwrap();
    let store = TemporalDeclarations::from_file(
        entries,
        (
            HashMap::new(),
            HashMap::from([("HAS_LICENSEE".to_string(), vec![config("a", "b")])]),
        ),
    );
    let only = store.edges("HAS_LICENSEE");
    assert_eq!(only.len(), 1);
    assert_eq!(
        (only[0].valid_from.as_str(), only[0].convention),
        ("x", HalfOpen)
    );
}

#[test]
fn an_entry_this_build_cannot_read_is_dropped_and_an_unknown_field_ignored() {
    #[derive(Deserialize)]
    struct Holder {
        #[serde(deserialize_with = "super::persist::lenient_entries")]
        entries: Vec<PersistedDeclaration>,
    }
    let holder: Holder = serde_json::from_str(
        r#"{"entries":[
            {"kind":"node","name":"A","from":"f","to":"t","convention":"weekly"},
            {"kind":"node","name":"B","from":"f","to":"t","convention":"closed","grain":"day"}
        ]}"#,
    )
    .unwrap();
    let store = TemporalDeclarations::from_file(holder.entries, Default::default());
    assert!(store.node("A").is_none());
    assert_eq!(store.node("B"), Some(&config("f", "t")));
}
