//! The declaration store through `kglite::api::temporal`: what a declaration
//! accepts, what it refuses (naming the offending element), what it counts,
//! and when it changes the graph version.

use std::collections::HashMap;

use super::declarations::{declare, declare_loaded, list, undeclare, TemporalTarget};
use super::eval::IntervalConvention::{self, Closed, HalfOpen};
use crate::datatypes::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::session::execute::{execute_mut, ExecuteOptions};

fn run(graph: &mut DirGraph, query: &str) {
    let params: HashMap<String, Value> = HashMap::new();
    execute_mut(graph, query, &ExecuteOptions::eager(&params))
        .unwrap_or_else(|e| panic!("{query}: {e}"));
}

fn graph(queries: &[&str]) -> DirGraph {
    let mut graph = DirGraph::new();
    for query in queries {
        run(&mut graph, query);
    }
    graph
}

fn node(label: &str) -> TemporalTarget {
    TemporalTarget::Node(label.into())
}

fn rel(rel_type: &str, source: Option<&str>) -> TemporalTarget {
    TemporalTarget::Relationship {
        rel_type: rel_type.into(),
        source_type: source.map(str::to_string),
    }
}

fn err(
    graph: &mut DirGraph,
    target: &TemporalTarget,
    from: &str,
    to: &str,
    convention: IntervalConvention,
) -> String {
    declare(graph, target, from, to, convention)
        .expect_err("declaration unexpectedly accepted")
        .to_string()
}

const STATUS: &str = "UNWIND [
    {id: 1, vf: '2000-01-01', vt: '2010-05-31'},
    {id: 2, vf: '2010-06-01', vt: '2019-12-31'},
    {id: 3, vf: '2020-01-01', vt: null}
  ] AS r CREATE (:Status {id: r.id, vf: r.vf, vt: r.vt})";

/// Two sources whose licensee periods use different properties, and one
/// field whose second period starts the day its first ends.
const LICENSEES: &[&str] = &[
    "CREATE (:Field {id: 1}), (:Field {id: 2}), (:Licence {id: 10}), (:Company {id: 100})",
    "MATCH (f:Field {id: 1}), (c:Company) CREATE (f)-[:HAS_LICENSEE {ff: '2000-01-01', ft: '2009-12-31'}]->(c)",
    "MATCH (f:Field {id: 1}), (c:Company) CREATE (f)-[:HAS_LICENSEE {ff: '2009-12-31', ft: null}]->(c)",
    "MATCH (f:Field {id: 2}), (c:Company) CREATE (f)-[:HAS_LICENSEE {ff: '2009-12-31', ft: '2011-01-01'}]->(c)",
    "MATCH (l:Licence), (c:Company) CREATE (l)-[:HAS_LICENSEE {lf: '1990-01-01', lt: '1999-12-31'}]->(c)",
];

#[test]
fn a_node_declaration_round_trips_through_list() {
    let mut g = graph(&[STATUS]);
    let before = g.version();
    let report = declare(&mut g, &node("Status"), "vf", "vt", Closed).unwrap();
    assert!(report.changed);
    assert_eq!(report.rows, 3);
    assert_eq!(report.abutting_rows, Some(0));
    assert_eq!(report.warning, None);
    assert_eq!(g.version(), before + 1);
    let listed = list(&g);
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].target, node("Status"));
    assert_eq!(listed[0].config.valid_from, "vf");
    assert_eq!(listed[0].config.convention, Closed);
    assert_eq!(listed[0].abutting_rows, Some(0));
}

#[test]
fn an_identical_redeclaration_is_a_no_op_and_a_different_one_conflicts() {
    let mut g = graph(&[STATUS]);
    declare(&mut g, &node("Status"), "vf", "vt", Closed).unwrap();
    let version = g.version();
    let again = declare(&mut g, &node("Status"), "vf", "vt", Closed).unwrap();
    assert!(!again.changed);
    assert_eq!(g.version(), version, "a no-op must not bump the version");
    let message = err(&mut g, &node("Status"), "vf", "vt", HalfOpen);
    assert!(message.contains("already declared"), "{message}");
    assert!(message.contains("convention 'closed'"), "{message}");
}

#[test]
fn unknown_targets_and_properties_are_refused_by_name() {
    let mut g = graph(&[STATUS]);
    let message = err(&mut g, &node("Nope"), "vf", "vt", Closed);
    assert!(message.contains("no node label 'Nope'"), "{message}");
    let message = err(&mut g, &node("Status"), "vf", "valid_to", Closed);
    assert!(
        message.contains("property 'valid_to' does not exist on node label 'Status'"),
        "{message}"
    );
    let mut g = graph(LICENSEES);
    let message = err(&mut g, &rel("NOPE", None), "ff", "ft", Closed);
    assert!(message.contains("no relationship type 'NOPE'"), "{message}");
    let message = err(
        &mut g,
        &rel("HAS_LICENSEE", Some("Company")),
        "ff",
        "ft",
        Closed,
    );
    assert!(
        message.contains("no relationships from source type 'Company'"),
        "{message}"
    );
    assert!(list(&g).is_empty());
}

#[test]
fn a_dirty_bound_is_refused_naming_its_element() {
    let mut g = graph(&[STATUS, "CREATE (:Status {id: 4, vf: 'someday', vt: null})"]);
    let message = err(&mut g, &node("Status"), "vf", "vt", Closed);
    assert!(message.contains("node '4'"), "{message}");
    assert!(message.contains("property 'vf'"), "{message}");
    assert!(message.contains("'someday'"), "{message}");

    let mut g = graph(LICENSEES);
    run(
        &mut g,
        "MATCH (f:Field {id: 2}), (c:Company) CREATE (f)-[:HAS_LICENSEE {ff: 2009, ft: null}]->(c)",
    );
    let message = err(
        &mut g,
        &rel("HAS_LICENSEE", Some("Field")),
        "ff",
        "ft",
        Closed,
    );
    assert!(
        message.contains("HAS_LICENSEE relationship from node '2' to node '100'"),
        "{message}"
    );
    assert!(message.contains("2009 (INTEGER)"), "{message}");
    assert!(list(&g).is_empty(), "a refused declaration stores nothing");
}

#[test]
fn an_inverted_row_is_refused_and_an_empty_one_under_half_open() {
    let mut g = graph(&["CREATE (:Span {id: 7, vf: '2010-01-01', vt: '2009-01-01'})"]);
    let message = err(&mut g, &node("Span"), "vf", "vt", Closed);
    assert!(message.contains("node '7'"), "{message}");
    assert!(message.contains("is after the to bound"), "{message}");

    let mut g = graph(&["CREATE (:Span {id: 8, vf: '2010-01-01', vt: '2010-01-01'})"]);
    let message = err(&mut g, &node("Span"), "vf", "vt", HalfOpen);
    assert!(message.contains("node '8'"), "{message}");
    assert!(declare(&mut g, &node("Span"), "vf", "vt", Closed).is_ok());
}

#[test]
fn abutting_rows_are_counted_and_warned_about_only_when_closed() {
    let mut g = graph(LICENSEES);
    let report = declare(
        &mut g,
        &rel("HAS_LICENSEE", Some("Field")),
        "ff",
        "ft",
        Closed,
    )
    .unwrap();
    // Field 1's first period ends the day its second begins. Field 2's period
    // starts that same day, but on another source node, so it is not counted.
    assert_eq!(report.rows, 3);
    assert_eq!(report.abutting_rows, Some(1));
    let warning = report
        .warning
        .expect("a closed declaration with abutting rows warns");
    assert!(
        warning.starts_with(
            "1 of 3 rows of relationship type 'HAS_LICENSEE' from source type 'Field'"
        ),
        "{warning}"
    );
    assert!(warning.contains("'half_open'"), "{warning}");

    let mut g = graph(LICENSEES);
    let report = declare(
        &mut g,
        &rel("HAS_LICENSEE", Some("Field")),
        "ff",
        "ft",
        HalfOpen,
    )
    .unwrap();
    assert_eq!(report.abutting_rows, Some(1));
    assert_eq!(report.warning, None);
}

#[test]
fn source_keyed_declarations_coexist_and_list_in_lookup_order() {
    let mut g = graph(LICENSEES);
    let unkeyed = declare(&mut g, &rel("HAS_LICENSEE", None), "ff", "ft", Closed).unwrap();
    assert_eq!(
        unkeyed.rows, 4,
        "with no keyed declaration it covers every source"
    );
    declare(
        &mut g,
        &rel("HAS_LICENSEE", Some("Licence")),
        "lf",
        "lt",
        HalfOpen,
    )
    .unwrap();
    declare(
        &mut g,
        &rel("HAS_LICENSEE", Some("Field")),
        "ff",
        "ft",
        Closed,
    )
    .unwrap();
    let listed: Vec<_> = list(&g)
        .into_iter()
        .map(|info| (info.target, info.config.valid_from, info.config.convention))
        .collect();
    assert_eq!(
        listed,
        vec![
            (rel("HAS_LICENSEE", Some("Field")), "ff".to_string(), Closed),
            (
                rel("HAS_LICENSEE", Some("Licence")),
                "lf".to_string(),
                HalfOpen
            ),
            (rel("HAS_LICENSEE", None), "ff".to_string(), Closed),
        ],
        "keyed declarations list before the unkeyed fallback"
    );
}

/// The unkeyed declaration is the type-wide fallback: it is stored beside
/// keyed ones whatever their properties, validates only the sources without
/// one, and conflicts only with another unkeyed declaration.
#[test]
fn an_unkeyed_declaration_is_the_fallback_and_its_own_key() {
    let mut g = graph(&[
        "CREATE (:A {id: 1}), (:B {id: 2}), (:C {id: 3})",
        "MATCH (a:A), (c:C) CREATE (a)-[:R {vf: '2000', vt: '2001', wf: 'junk'}]->(c)",
        "MATCH (b:B), (c:C) CREATE (b)-[:R {wf: '2002', wt: '2003'}]->(c)",
    ]);
    declare(&mut g, &rel("R", Some("A")), "vf", "vt", Closed).unwrap();
    let fallback = declare(&mut g, &rel("R", None), "wf", "wt", Closed).unwrap();
    assert!(fallback.changed);
    assert_eq!(fallback.rows, 1, "A's rows belong to A's keyed declaration");
    assert!(
        !declare(&mut g, &rel("R", None), "wf", "wt", Closed)
            .unwrap()
            .changed
    );
    let message = err(&mut g, &rel("R", None), "vf", "vt", Closed);
    assert!(
        message.contains("relationship type 'R' is already declared"),
        "{message}"
    );
    let message = err(&mut g, &rel("R", Some("A")), "wf", "wt", Closed);
    assert!(
        message.contains("from source type 'A' is already declared"),
        "{message}"
    );
    // A keyed declaration identical to the fallback is still its own key.
    assert!(
        declare(&mut g, &rel("R", Some("B")), "wf", "wt", Closed)
            .unwrap()
            .changed
    );
    assert_eq!(list(&g).len(), 3);
}

/// An edge takes its source's keyed config, and the unkeyed one only when
/// its source has none — whatever order they were declared in.
#[test]
fn an_edge_takes_its_sources_keyed_config_before_the_fallback() {
    use super::{is_temporally_valid_multi, TemporalConfig};
    use crate::graph::schema::InternedKey;
    let config = |from: &str, to: &str, source: Option<&str>| TemporalConfig {
        valid_from: from.into(),
        valid_to: to.into(),
        convention: HalfOpen,
        source_type: source.map(str::to_string),
    };
    let s = |t: &str| Value::String(t.into());
    // Unkeyed first in the list: the keyed one must still win for Field.
    let configs = [config("f", "t", None), config("f", "t", Some("Field"))];
    let mut configs_closed_fallback = configs.clone();
    configs_closed_fallback[0].convention = Closed;
    let props = [
        (InternedKey::from_str("f"), s("2000-01-01")),
        (InternedKey::from_str("t"), s("2009-12-31")),
    ];
    let day = chrono::NaiveDate::from_ymd_opt(2009, 12, 31).unwrap();
    let field = Some(InternedKey::from_str("Field"));
    let licence = Some(InternedKey::from_str("Licence"));
    assert!(!is_temporally_valid_multi(&props, &configs_closed_fallback, field, &day).unwrap());
    assert!(is_temporally_valid_multi(&props, &configs_closed_fallback, licence, &day).unwrap());
}

#[test]
fn a_loader_may_name_a_column_it_wrote_entirely_null() {
    let mut g = graph(&["UNWIND [1, 2] AS i CREATE (:Open {id: i, vf: '2000-01-01', vt: null})"]);
    let message = err(&mut g, &node("Open"), "vf", "vt", Closed);
    assert!(
        message.contains("property 'vt' does not exist"),
        "{message}"
    );
    let report = declare_loaded(&mut g, &node("Open"), "vf", "vt", Closed, &["vf", "vt"]).unwrap();
    assert_eq!(report.rows, 2);
    // The list vouches only for what it names.
    let mut g = graph(&["CREATE (:Open {id: 1, vf: '2000-01-01'})"]);
    let message = declare_loaded(&mut g, &node("Open"), "vf", "vtt", Closed, &["vf", "vt"])
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("property 'vtt' does not exist"),
        "{message}"
    );

    let mut g = graph(&[
        "CREATE (:A {id: 1}), (:C {id: 2})",
        "MATCH (a:A), (c:C) CREATE (a)-[:R {vf: '2000-01-01', vt: null}]->(c)",
    ]);
    let message = err(&mut g, &rel("R", None), "vf", "vt", Closed);
    assert!(
        message.contains("property 'vt' does not exist on relationship type 'R'"),
        "{message}"
    );
    assert!(declare_loaded(&mut g, &rel("R", None), "vf", "vt", Closed, &["vt"]).is_ok());
}

#[test]
fn a_secondary_label_is_a_node_target() {
    let mut g = graph(&[
        "CREATE (:Status:Tracked {id: 1, vf: '2000-01-01', vt: '2001-01-01'})",
        "CREATE (:Other:Tracked {id: 2, vf: '2001-01-01', vt: null})",
    ]);
    let report = declare(&mut g, &node("Tracked"), "vf", "vt", Closed).unwrap();
    assert_eq!(report.rows, 2);
    assert_eq!(report.abutting_rows, Some(1));
    assert_eq!(list(&g)[0].target, node("Tracked"));
}

#[test]
fn undeclare_removes_and_bumps_only_when_something_was_declared() {
    let mut g = graph(LICENSEES);
    declare(
        &mut g,
        &rel("HAS_LICENSEE", Some("Field")),
        "ff",
        "ft",
        Closed,
    )
    .unwrap();
    let version = g.version();
    assert!(!undeclare(&mut g, &rel("HAS_LICENSEE", None)));
    assert_eq!(g.version(), version);
    assert!(undeclare(&mut g, &rel("HAS_LICENSEE", Some("Field"))));
    assert_eq!(g.version(), version + 1);
    assert!(list(&g).is_empty());
}

#[test]
fn a_disk_graph_is_validated_without_growing_the_edge_arena() {
    let mut g = graph(LICENSEES);
    g.enable_disk_mode().unwrap();
    let report = declare(
        &mut g,
        &rel("HAS_LICENSEE", Some("Field")),
        "ff",
        "ft",
        Closed,
    )
    .unwrap();
    assert_eq!((report.rows, report.abutting_rows), (3, Some(1)));
    let disk = g.graph.as_disk().expect("disk-backed");
    assert_eq!(disk.edge_arena_len(), 0);
    assert_eq!(disk.node_arena_len(), 0);
}
