//! The bulk loaders' merge key on a declared temporal relationship type: a
//! row whose `from` bound differs from every stored relationship's between the
//! same endpoints is a new, parallel relationship; the same bound merges as
//! before. Undeclared types keep merging on the endpoints alone.

use std::collections::HashMap;

use chrono::NaiveDate;

use super::declarations::{declare, TemporalTarget};
use super::eval::IntervalConvention::Closed;
use crate::datatypes::{DataFrame, Value};
use crate::graph::dir_graph::DirGraph;
use crate::graph::introspection::reporting::ConnectionOperationReport;
use crate::graph::mutation::maintain::{
    add_connections, add_edges_from_specs, add_nodes, replace_connections, EdgeSpec,
};
use crate::graph::session::execute::{execute_mut, execute_read, ExecuteOptions};

fn day(text: &str) -> Value {
    Value::DateTime(NaiveDate::parse_from_str(text, "%Y-%m-%d").unwrap())
}

fn docs(graph: &mut DirGraph, node_type: &str) {
    let rows = vec![vec![Value::Int64(1)], vec![Value::Int64(2)]];
    let df = DataFrame::from_cypher_rows(vec!["id".to_string()], rows).unwrap();
    add_nodes(
        graph,
        df,
        node_type.to_string(),
        "id".to_string(),
        None,
        None,
    )
    .unwrap();
}

fn frame(rows: Vec<(Value, Value)>, extra: Option<Value>) -> DataFrame {
    let mut columns = vec!["src", "tgt", "vf", "vt"];
    if extra.is_some() {
        columns.push("x");
    }
    let rows = rows
        .into_iter()
        .map(|(vf, vt)| {
            let mut row = vec![Value::Int64(1), Value::Int64(2), vf, vt];
            row.extend(extra.clone());
            row
        })
        .collect();
    let columns = columns.into_iter().map(str::to_string).collect();
    DataFrame::from_cypher_rows(columns, rows).unwrap()
}

fn load(
    graph: &mut DirGraph,
    rows: Vec<(Value, Value)>,
    mode: Option<&str>,
) -> Result<ConnectionOperationReport, String> {
    load_frame(graph, frame(rows, None), "Doc", mode)
}

fn load_frame(
    graph: &mut DirGraph,
    df: DataFrame,
    source: &str,
    mode: Option<&str>,
) -> Result<ConnectionOperationReport, String> {
    add_connections(
        graph,
        df,
        "IN".to_string(),
        source.to_string(),
        "src".to_string(),
        "Doc".to_string(),
        "tgt".to_string(),
        None,
        None,
        mode.map(str::to_string),
    )
}

fn counts(report: &ConnectionOperationReport) -> (usize, usize) {
    (report.connections_created, report.connections_updated)
}

fn periods(graph: &DirGraph) -> Vec<Vec<Value>> {
    let params = HashMap::new();
    execute_read(
        graph,
        "MATCH ()-[r:IN]->(:Doc) RETURN r.vf AS vf, r.vt AS vt ORDER BY vf",
        &ExecuteOptions::eager(&params),
    )
    .unwrap()
    .result
    .rows
}

fn in_type(source: Option<&str>) -> TemporalTarget {
    TemporalTarget::Relationship {
        rel_type: "IN".into(),
        source_type: source.map(str::to_string),
    }
}

/// `Doc` 1 → 2 holds `IN` for 2000..2005; `declared` then declares `IN`.
fn first_period(declared: Option<TemporalTarget>) -> DirGraph {
    let mut graph = DirGraph::new();
    docs(&mut graph, "Doc");
    load(
        &mut graph,
        vec![(day("2000-01-01"), day("2005-01-01"))],
        None,
    )
    .unwrap();
    if let Some(target) = declared {
        declare(&mut graph, &target, "vf", "vt", Closed).unwrap();
    }
    graph
}

const MODES: [Option<&str>; 6] = [
    None,
    Some("update"),
    Some("replace"),
    Some("preserve"),
    Some("skip"),
    Some("sum"),
];

#[test]
fn a_later_period_on_a_declared_type_is_a_parallel_relationship_in_every_mode() {
    for target in [in_type(None), in_type(Some("Doc"))] {
        for mode in MODES {
            let mut graph = first_period(Some(target.clone()));
            let report = load(&mut graph, vec![(day("2010-01-01"), Value::Null)], mode).unwrap();
            assert_eq!(counts(&report), (1, 0), "{target:?} {mode:?}");
            assert_eq!(
                periods(&graph),
                vec![
                    vec![day("2000-01-01"), day("2005-01-01")],
                    vec![day("2010-01-01"), Value::Null],
                ],
                "{target:?} {mode:?}"
            );
        }
    }
}

/// The undeclared behaviour this change must keep, mode by mode.
#[test]
fn an_undeclared_type_still_merges_on_the_endpoints() {
    let (old, new) = (day("2000-01-01"), day("2010-01-01"));
    let old_end = day("2005-01-01");
    for (mode, expected, reported) in [
        (None, vec![new.clone(), old_end.clone()], (0, 1)),
        (Some("update"), vec![new.clone(), old_end.clone()], (0, 1)),
        (Some("replace"), vec![new.clone(), Value::Null], (0, 1)),
        (Some("preserve"), vec![old.clone(), old_end.clone()], (0, 1)),
        (Some("skip"), vec![old.clone(), old_end.clone()], (0, 0)),
        (Some("sum"), vec![new.clone(), old_end.clone()], (0, 1)),
    ] {
        let mut graph = first_period(None);
        let report = load(&mut graph, vec![(new.clone(), Value::Null)], mode).unwrap();
        assert_eq!(counts(&report), reported, "{mode:?}");
        assert_eq!(periods(&graph), vec![expected], "{mode:?}");
    }
}

#[test]
fn the_same_from_bound_merges_as_before() {
    let mut graph = first_period(Some(in_type(None)));
    load(&mut graph, vec![(day("2010-01-01"), Value::Null)], None).unwrap();
    // Closing the open period.
    let report = load(
        &mut graph,
        vec![(day("2010-01-01"), day("2015-01-01"))],
        Some("update"),
    )
    .unwrap();
    assert_eq!(counts(&report), (0, 1));
    // An identical re-load.
    let report = load(
        &mut graph,
        vec![(day("2010-01-01"), day("2015-01-01"))],
        None,
    )
    .unwrap();
    assert_eq!(counts(&report), (0, 1));
    let report = load(
        &mut graph,
        vec![(day("2010-01-01"), day("2016-01-01"))],
        Some("skip"),
    )
    .unwrap();
    assert_eq!(counts(&report), (0, 0));
    assert_eq!(
        periods(&graph),
        vec![
            vec![day("2000-01-01"), day("2005-01-01")],
            vec![day("2010-01-01"), day("2015-01-01")],
        ]
    );
}

#[test]
fn two_new_periods_in_one_call_are_two_relationships() {
    let mut graph = first_period(Some(in_type(None)));
    let report = load(
        &mut graph,
        vec![
            (day("2006-01-01"), day("2009-12-31")),
            (day("2010-01-01"), Value::Null),
        ],
        None,
    )
    .unwrap();
    assert_eq!(counts(&report), (2, 0));
    assert_eq!(periods(&graph).len(), 3);
}

#[test]
fn a_declaration_keyed_to_another_source_leaves_this_source_merging() {
    let mut graph = first_period(None);
    docs(&mut graph, "Other");
    load_frame(
        &mut graph,
        frame(vec![(day("2001-01-01"), Value::Null)], None),
        "Other",
        None,
    )
    .unwrap();
    declare(&mut graph, &in_type(Some("Other")), "vf", "vt", Closed).unwrap();
    let report = load(&mut graph, vec![(day("2010-01-01"), Value::Null)], None).unwrap();
    assert_eq!(counts(&report), (0, 1));
    // Other → Doc now takes the key.
    let report = load_frame(
        &mut graph,
        frame(vec![(day("2011-01-01"), Value::Null)], None),
        "Other",
        None,
    )
    .unwrap();
    assert_eq!(counts(&report), (1, 0));
}

#[test]
fn replace_connections_keeps_each_period_of_the_frame() {
    let mut graph = first_period(Some(in_type(None)));
    let report = replace_connections(
        &mut graph,
        frame(
            vec![
                (day("2000-01-01"), day("2005-01-01")),
                (day("2010-01-01"), Value::Null),
            ],
            None,
        ),
        "IN".to_string(),
        "Doc".to_string(),
        "src".to_string(),
        "Doc".to_string(),
        "tgt".to_string(),
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(counts(&report), (2, 0));
    assert_eq!(periods(&graph).len(), 2);
}

#[test]
fn the_spec_path_keys_a_declared_type_on_its_from_bound() {
    let spec = |vf: &str| EdgeSpec {
        source_type: "Doc".into(),
        source_id: Value::Int64(1),
        target_type: "Doc".into(),
        target_id: Value::Int64(2),
        edge_type: "IN".into(),
        properties: HashMap::from([("vf".to_string(), day(vf))]),
    };
    let mut graph = first_period(Some(in_type(Some("Doc"))));
    let report = add_edges_from_specs(&mut graph, vec![spec("2010-01-01")]).unwrap();
    assert_eq!(
        (report.connections_created, report.connections_updated),
        (1, 0)
    );
    let report = add_edges_from_specs(&mut graph, vec![spec("2010-01-01")]).unwrap();
    assert_eq!(
        (report.connections_created, report.connections_updated),
        (0, 1)
    );
    assert_eq!(periods(&graph).len(), 2);
}

/// The constraint gate models the same key the loader merges on: a new
/// period lacking a NOT NULL property is a new relationship, not a merge
/// that keeps the stored value.
#[test]
fn the_constraint_gate_judges_a_new_period_as_its_own_relationship() {
    let mut graph = DirGraph::new();
    docs(&mut graph, "Doc");
    load_frame(
        &mut graph,
        frame(
            vec![(day("2000-01-01"), day("2005-01-01"))],
            Some(Value::Int64(7)),
        ),
        "Doc",
        None,
    )
    .unwrap();
    let params: HashMap<String, Value> = HashMap::new();
    execute_mut(
        &mut graph,
        "CREATE CONSTRAINT FOR ()-[r:IN]-() REQUIRE r.x IS NOT NULL",
        &ExecuteOptions::eager(&params),
    )
    .unwrap();
    // Undeclared: the row merges into the stored edge, which keeps x.
    let report = load(&mut graph, vec![(day("2010-01-01"), Value::Null)], None).unwrap();
    assert_eq!(counts(&report), (0, 1));

    let mut graph = DirGraph::new();
    docs(&mut graph, "Doc");
    load_frame(
        &mut graph,
        frame(
            vec![(day("2000-01-01"), day("2005-01-01"))],
            Some(Value::Int64(7)),
        ),
        "Doc",
        None,
    )
    .unwrap();
    execute_mut(
        &mut graph,
        "CREATE CONSTRAINT FOR ()-[r:IN]-() REQUIRE r.x IS NOT NULL",
        &ExecuteOptions::eager(&params),
    )
    .unwrap();
    declare(&mut graph, &in_type(None), "vf", "vt", Closed).unwrap();
    let err = load(&mut graph, vec![(day("2010-01-01"), Value::Null)], None).unwrap_err();
    assert!(err.contains('x'), "{err}");
    assert_eq!(periods(&graph).len(), 1);
    // The same period merges, keeping x.
    let report = load(
        &mut graph,
        vec![(day("2000-01-01"), day("2006-01-01"))],
        None,
    )
    .unwrap();
    assert_eq!(counts(&report), (0, 1));
}
