//! `declare_defaulted` and `declare_from_column_types`: the defaulted
//! convention, the checks a load passes before it writes, and the declaration
//! it leaves.

use std::collections::HashMap;

use chrono::NaiveDate;

use super::declarations::{declare, list, TemporalTarget};
use super::eval::IntervalConvention::{Closed, HalfOpen};
use super::loader::{declare_defaulted, declare_from_column_types};
use crate::datatypes::{DataFrame, Value};
use crate::graph::dir_graph::DirGraph;
use crate::graph::mutation::maintain::{add_connections, add_nodes};
use crate::graph::session::execute::{execute_mut, ExecuteOptions};

fn day(text: &str) -> Value {
    Value::DateTime(NaiveDate::parse_from_str(text, "%Y-%m-%d").unwrap())
}

fn status_frame(rows: &[(i64, Value, Value)]) -> DataFrame {
    let rows = rows
        .iter()
        .map(|(id, vf, vt)| vec![Value::Int64(*id), vf.clone(), vt.clone()])
        .collect();
    let columns = ["id", "vf", "vt"].map(str::to_string).to_vec();
    DataFrame::from_cypher_rows(columns, rows).unwrap()
}

fn load_statuses(graph: &mut DirGraph, frame: DataFrame) {
    add_nodes(graph, frame, "Status".into(), "id".into(), None, None).unwrap();
}

fn status() -> TemporalTarget {
    TemporalTarget::Node("Status".into())
}

fn conventions(graph: &DirGraph) -> Vec<(TemporalTarget, super::eval::IntervalConvention)> {
    list(graph)
        .into_iter()
        .map(|info| (info.target, info.config.convention))
        .collect()
}

#[test]
fn a_repeated_load_leaves_one_declaration_and_validates_nothing() {
    let mut graph = DirGraph::new();
    let first = status_frame(&[(1, day("2000-01-01"), day("2005-01-01"))]);
    let pending =
        declare_from_column_types(&mut graph, status(), "vf", "vt", None, &first).unwrap();
    load_statuses(&mut graph, first);
    let report = pending.finish(&mut graph).unwrap();
    assert!(report.changed);
    assert_eq!(report.rows, 1);

    let version = graph.version();
    let second = status_frame(&[(2, day("2006-01-01"), Value::Null)]);
    let pending =
        declare_from_column_types(&mut graph, status(), "vf", "vt", None, &second).unwrap();
    load_statuses(&mut graph, second);
    let report = pending.finish(&mut graph).unwrap();
    assert!(!report.changed);
    assert_eq!(report.rows, 0);
    assert_eq!(conventions(&graph), vec![(status(), Closed)]);
    // Only the load's own write bumped the version.
    assert_eq!(graph.version(), version + 1);
}

#[test]
fn a_conflicting_load_is_refused_before_it_writes() {
    let mut graph = DirGraph::new();
    load_statuses(
        &mut graph,
        status_frame(&[(1, day("2000-01-01"), day("2005-01-01"))]),
    );
    declare(&mut graph, &status(), "vf", "vt", HalfOpen).unwrap();
    let frame = status_frame(&[(2, day("2006-01-01"), Value::Null)]);
    let err = declare_from_column_types(&mut graph, status(), "vf", "vt", Some(Closed), &frame)
        .err()
        .expect("a different convention is a conflict");
    assert!(err.contains("already declared"), "{err}");
    // Without a convention the half-open declaration is kept.
    let pending =
        declare_from_column_types(&mut graph, status(), "vf", "vt", None, &frame).unwrap();
    assert!(!pending.finish(&mut graph).unwrap().changed);
    assert_eq!(conventions(&graph), vec![(status(), HalfOpen)]);
}

#[test]
fn an_inverted_row_of_the_load_is_refused_by_position() {
    let mut graph = DirGraph::new();
    let frame = status_frame(&[
        (1, day("2000-01-01"), day("2005-01-01")),
        (2, day("2010-01-01"), day("2009-01-01")),
    ]);
    let err = declare_from_column_types(&mut graph, status(), "vf", "vt", None, &frame)
        .err()
        .expect("inverted row");
    assert!(err.starts_with("row 1 of the load, "), "{err}");
    assert!(err.contains("is after the to bound"), "{err}");
    assert!(list(&graph).is_empty());
}

#[test]
fn a_stored_dirty_bound_refuses_the_load_before_it_writes() {
    let mut graph = DirGraph::new();
    let params: HashMap<String, Value> = HashMap::new();
    execute_mut(
        &mut graph,
        "CREATE (:Status {id: 9, vf: 'someday', vt: null})",
        &ExecuteOptions::eager(&params),
    )
    .unwrap();
    let frame = status_frame(&[(1, day("2000-01-01"), Value::Null)]);
    let err = declare_from_column_types(&mut graph, status(), "vf", "vt", None, &frame)
        .err()
        .expect("dirty stored bound");
    assert!(err.contains("node '9'") && err.contains("someday"), "{err}");
    assert!(list(&graph).is_empty());
}

#[test]
fn a_first_load_with_every_period_open_declares_its_relationship_type() {
    let mut graph = DirGraph::new();
    load_statuses(&mut graph, status_frame(&[(1, Value::Null, Value::Null)]));
    let rows = vec![vec![
        Value::Int64(1),
        Value::Int64(1),
        day("2000-01-01"),
        Value::Null,
    ]];
    let columns = ["src", "tgt", "vf", "vt"].map(str::to_string).to_vec();
    let frame = DataFrame::from_cypher_rows(columns, rows).unwrap();
    let pending = declare_from_column_types(
        &mut graph,
        next(Some("Status")),
        "vf",
        "vt",
        Some(HalfOpen),
        &frame,
    )
    .unwrap();
    add_connections(
        &mut graph,
        frame,
        "NEXT".into(),
        "Status".into(),
        "src".into(),
        "Status".into(),
        "tgt".into(),
        None,
        None,
        None,
    )
    .unwrap();
    let report = pending.finish(&mut graph).unwrap();
    assert!(report.changed);
    assert_eq!(conventions(&graph), vec![(next(None), HalfOpen)]);
}

#[test]
fn abandoning_withdraws_a_declaration_made_before_the_load() {
    let mut graph = DirGraph::new();
    load_statuses(
        &mut graph,
        status_frame(&[(1, day("2000-01-01"), day("2005-01-01"))]),
    );
    let frame = status_frame(&[(2, day("2006-01-01"), Value::Null)]);
    let pending =
        declare_from_column_types(&mut graph, status(), "vf", "vt", None, &frame).unwrap();
    assert_eq!(list(&graph).len(), 1);
    pending.abandon(&mut graph);
    assert!(list(&graph).is_empty());
}

#[test]
fn declare_defaulted_keeps_a_declaration_naming_the_same_properties() {
    let mut graph = DirGraph::new();
    load_statuses(
        &mut graph,
        status_frame(&[(1, day("2000-01-01"), day("2005-01-01"))]),
    );
    declare(&mut graph, &status(), "vf", "vt", HalfOpen).unwrap();
    let report = declare_defaulted(&mut graph, &status(), "vf", "vt", None).unwrap();
    assert!(!report.changed);
    assert_eq!(conventions(&graph), vec![(status(), HalfOpen)]);
    let err = declare_defaulted(&mut graph, &status(), "vf", "vt", Some(Closed)).unwrap_err();
    assert!(err.contains("already declared"), "{err}");
}

/// The type already exists from another source, so this load merges; the
/// declaration is installed before it writes, keeping its two periods for
/// one pair apart instead of folding them into an inverted interval.
#[test]
fn a_new_sources_first_load_onto_an_existing_type_merges_under_the_key() {
    let mut graph = DirGraph::new();
    load_statuses(
        &mut graph,
        status_frame(&[(1, Value::Null, Value::Null), (2, Value::Null, Value::Null)]),
    );
    let params: HashMap<String, Value> = HashMap::new();
    execute_mut(
        &mut graph,
        "CREATE (:Other {id: 5})-[:NEXT]->(:Target {id: 6})",
        &ExecuteOptions::eager(&params),
    )
    .unwrap();
    let rows = vec![
        vec![
            Value::Int64(1),
            Value::Int64(2),
            day("2000-01-01"),
            day("2005-01-01"),
        ],
        vec![
            Value::Int64(1),
            Value::Int64(2),
            day("2010-01-01"),
            Value::Null,
        ],
    ];
    let columns = ["src", "tgt", "vf", "vt"].map(str::to_string).to_vec();
    let frame = DataFrame::from_cypher_rows(columns, rows).unwrap();
    let pending =
        declare_from_column_types(&mut graph, next(Some("Status")), "vf", "vt", None, &frame)
            .unwrap();
    let report = add_connections(
        &mut graph,
        frame,
        "NEXT".into(),
        "Status".into(),
        "src".into(),
        "Status".into(),
        "tgt".into(),
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(report.connections_created, 2);
    assert!(pending.finish(&mut graph).unwrap().changed);
    assert_eq!(conventions(&graph), vec![(next(None), Closed)]);
}

fn next(source: Option<&str>) -> TemporalTarget {
    TemporalTarget::Relationship {
        rel_type: "NEXT".into(),
        source_type: source.map(str::to_string),
    }
}

/// A load declares the type-wide declaration — what every build before
/// source types applied — unless its source needs one of its own.
#[test]
fn a_relationship_load_declares_for_its_source_only_when_it_must() {
    let mut graph = DirGraph::new();
    load_statuses(&mut graph, status_frame(&[(1, Value::Null, Value::Null)]));
    let params: HashMap<String, Value> = HashMap::new();
    execute_mut(
        &mut graph,
        "MATCH (s:Status) CREATE (s)-[:NEXT {af: '2000-01-01', bf: '2000-01-01', t: null}]->
         (o:Other {id: 7})-[:NEXT {af: '2000-01-01', bf: '2000-01-01', t: '2001-01-01'}]->(s)",
        &ExecuteOptions::eager(&params),
    )
    .unwrap();
    let empty = DataFrame::from_cypher_rows(vec![], vec![]).unwrap();
    let declared = |graph: &mut DirGraph, source: &str, from: &str| {
        declare_from_column_types(graph, next(Some(source)), from, "t", None, &empty)
            .and_then(|pending| pending.finish(graph))
            .unwrap_or_else(|e| panic!("{source} {from}: {e}"));
    };
    declared(&mut graph, "Status", "af");
    // Same properties from another source: the type-wide one covers it.
    declared(&mut graph, "Other", "af");
    // Other properties: declared for the source.
    declared(&mut graph, "Other", "bf");
    // And that source's own declaration now governs its loads.
    declared(&mut graph, "Other", "bf");
    let targets: Vec<_> = list(&graph).into_iter().map(|info| info.target).collect();
    assert_eq!(targets, vec![next(Some("Other")), next(None)]);
}
