//! Writes onto a declared type answer to the declaration's row rule: a load
//! is judged before it writes, by what each row leaves under its conflict
//! mode; a Cypher `CREATE` before its insert; a `SET` once its clause has
//! applied every item.

use std::collections::HashMap;

use chrono::NaiveDate;

use super::declarations::{declare, list, TemporalTarget};
use super::eval::IntervalConvention::{Closed, HalfOpen};
use crate::datatypes::{DataFrame, Value};
use crate::graph::dir_graph::DirGraph;
use crate::graph::mutation::maintain::{add_connections, add_nodes};
use crate::graph::session::execute::{execute_mut, ExecuteOptions};
use crate::graph::storage::GraphRead;

fn day(text: &str) -> Value {
    Value::DateTime(NaiveDate::parse_from_str(text, "%Y-%m-%d").unwrap())
}

fn run(graph: &mut DirGraph, query: &str) -> Result<(), String> {
    let params: HashMap<String, Value> = HashMap::new();
    execute_mut(graph, query, &ExecuteOptions::eager(&params))
        .map(|_| ())
        .map_err(|e| e.to_string())
}

fn frame(columns: &[&str], rows: Vec<Vec<Value>>) -> DataFrame {
    let columns = columns.iter().map(|c| c.to_string()).collect();
    DataFrame::from_cypher_rows(columns, rows).unwrap()
}

/// `Status {id: 1}` holds [2000-01-01, 2010-01-01], declared `convention`.
fn statuses(convention: super::eval::IntervalConvention) -> DirGraph {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "CREATE (:Status {id: 1, vf: date('2000-01-01'), vt: date('2010-01-01')})",
    )
    .unwrap();
    declare(
        &mut graph,
        &TemporalTarget::Node("Status".into()),
        "vf",
        "vt",
        convention,
    )
    .unwrap();
    graph
}

fn status_count(graph: &mut DirGraph) -> usize {
    graph.type_indices.get("Status").map_or(0, |b| b.len())
}

fn load(graph: &mut DirGraph, rows: DataFrame, mode: Option<&str>) -> Result<(), String> {
    add_nodes(
        graph,
        rows,
        "Status".into(),
        "id".into(),
        None,
        mode.map(str::to_string),
    )
    .map(|_| ())
}

#[test]
fn a_load_row_is_judged_before_the_load_writes() {
    let mut graph = statuses(HalfOpen);
    let rows = frame(
        &["id", "vf", "vt"],
        vec![
            vec![Value::Int64(2), day("2011-01-01"), Value::Null],
            vec![Value::Int64(3), day("2012-01-01"), day("2011-01-01")],
        ],
    );
    let err = load(&mut graph, rows, None).unwrap_err();
    assert!(err.starts_with("row 1 (0-based) of the load, "), "{err}");
    assert!(err.contains("is after the to bound"), "{err}");
    assert_eq!(status_count(&mut graph), 1, "nothing written");
}

#[test]
fn an_empty_half_open_load_row_is_written_and_counted() {
    let mut graph = statuses(HalfOpen);
    let rows = frame(
        &["id", "vf", "vt"],
        vec![
            vec![Value::Int64(2), day("2011-01-01"), Value::Null],
            vec![Value::Int64(3), day("2012-01-01"), day("2012-01-01")],
            vec![Value::Int64(4), day("2013-01-01"), day("2013-01-01")],
        ],
    );
    load(&mut graph, rows, None).unwrap();
    assert_eq!(status_count(&mut graph), 4);
    assert_eq!(list(&graph)[0].empty_rows, Some(2));
}

#[test]
fn an_empty_half_open_cypher_write_is_kept_with_one_warning() {
    let mut graph = statuses(HalfOpen);
    let params: HashMap<String, Value> = HashMap::new();
    let result = execute_mut(
        &mut graph,
        "UNWIND [2, 3, 4] AS i CREATE (:Status {id: i, vf: date('2011-01-01'), \
         vt: CASE WHEN i = 4 THEN null ELSE date('2011-01-01') END})",
        &ExecuteOptions::eager(&params),
    )
    .unwrap();
    let warnings = result
        .result
        .diagnostics
        .map(|d| d.warnings)
        .unwrap_or_default();
    let empty: Vec<&String> = warnings
        .iter()
        .filter(|w| w.contains("empty interval"))
        .collect();
    assert_eq!(empty.len(), 1, "one warning per statement: {warnings:?}");
    assert!(
        empty[0]
            .starts_with("2 of 3 rows written have an empty interval under convention 'half_open'"),
        "{}",
        empty[0]
    );
    assert!(empty[0].contains("the first is node '2'"), "{}", empty[0]);
    assert_eq!(status_count(&mut graph), 4);
    assert_eq!(list(&graph)[0].empty_rows, Some(2));

    // A SET that leaves an empty interval is kept and warned about too.
    let result = execute_mut(
        &mut graph,
        "MATCH (s:Status {id: 1}) SET s.vt = s.vf",
        &ExecuteOptions::eager(&params),
    )
    .unwrap();
    let warnings = result
        .result
        .diagnostics
        .map(|d| d.warnings)
        .unwrap_or_default();
    assert!(
        warnings.iter().any(
            |w| w.starts_with("1 of 1 rows written have an empty interval")
                && w.contains("node '1'")
        ),
        "{warnings:?}"
    );
    assert_eq!(list(&graph)[0].empty_rows, Some(3));
    // An inverted SET is still refused.
    let err = run(
        &mut graph,
        "MATCH (s:Status {id: 1}) SET s.vt = date('1999-01-01')",
    )
    .unwrap_err();
    assert!(err.contains("is after the to bound"), "{err}");
}

#[test]
fn an_update_row_is_judged_with_the_bound_it_keeps() {
    let mut graph = statuses(Closed);
    // Only `vt` arrives; the stored `vf` 2000-01-01 is after it.
    let rows = || {
        frame(
            &["id", "vt"],
            vec![vec![Value::Int64(1), day("1999-01-01")]],
        )
    };
    let err = load(&mut graph, rows(), None).unwrap_err();
    assert!(err.contains("is after the to bound"), "{err}");
    // `preserve` keeps the stored `vt`, so the row changes nothing.
    load(&mut graph, rows(), Some("preserve")).unwrap();
    // `skip` leaves the node as it is.
    load(&mut graph, rows(), Some("skip")).unwrap();
    // `replace` drops the stored `vf`: an open start is valid.
    load(&mut graph, rows(), Some("replace")).unwrap();
}

#[test]
fn a_relationship_load_is_judged_by_its_rows() {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "CREATE (:Project {id: 1})-[:OP {vf: date('2000-01-01'), vt: date('2001-01-01')}]->(:Co {id: 2})",
    )
    .unwrap();
    declare(
        &mut graph,
        &TemporalTarget::Relationship {
            rel_type: "OP".into(),
            source_type: None,
        },
        "vf",
        "vt",
        Closed,
    )
    .unwrap();
    let rows = frame(
        &["s", "t", "vf", "vt"],
        vec![
            vec![
                Value::Int64(1),
                Value::Int64(2),
                day("2002-01-01"),
                day("2003-01-01"),
            ],
            vec![
                Value::Int64(1),
                Value::Int64(2),
                day("2005-01-01"),
                day("2004-01-01"),
            ],
        ],
    );
    let err = add_connections(
        &mut graph,
        rows,
        "OP".into(),
        "Project".into(),
        "s".into(),
        "Co".into(),
        "t".into(),
        None,
        None,
        None,
    )
    .unwrap_err();
    assert!(err.starts_with("row 1 (0-based) of the load, "), "{err}");
    assert_eq!(graph.graph.edge_count(), 1, "nothing written");
}

#[test]
fn a_set_is_judged_on_the_interval_its_clause_leaves() {
    let mut graph = statuses(Closed);
    // Item by item the first write inverts the interval; the pair does not.
    run(
        &mut graph,
        "MATCH (s:Status {id: 1}) SET s.vf = date('2015-01-01'), s.vt = date('2016-01-01')",
    )
    .unwrap();
    let err = run(
        &mut graph,
        "MATCH (s:Status {id: 1}) SET s.vt = date('2014-01-01')",
    )
    .unwrap_err();
    assert!(err.contains("node '1', the from bound"), "{err}");
    let err = run(
        &mut graph,
        "MATCH (s:Status {id: 1}) SET s += {vt: 'someday'}",
    )
    .unwrap_err();
    assert!(err.contains("node '1', property 'vt'"), "{err}");
    run(&mut graph, "MATCH (s:Status {id: 1}) SET s.vt = null").unwrap();
    assert_eq!(list(&graph)[0].empty_rows, Some(0));
    assert_eq!(list(&graph)[0].unreadable_rows, Some(0));
}

#[test]
fn a_create_is_judged_before_it_inserts() {
    let mut graph = statuses(HalfOpen);
    let err = run(
        &mut graph,
        "CREATE (:Status {id: 2, vf: date('2011-01-01'), vt: date('2010-01-01')})",
    )
    .unwrap_err();
    assert!(err.contains("node '2', the from bound"), "{err}");
    assert!(
        err.contains("is after the to bound")
            && err.contains("an inverted interval under convention 'half_open'"),
        "{err}"
    );
    assert_eq!(status_count(&mut graph), 1);
    // A secondary label carries its declaration too.
    let err = run(
        &mut graph,
        "CREATE (:Other:Status {id: 3, vf: date('2011-01-01'), vt: date('2010-01-01')})",
    )
    .unwrap_err();
    assert!(err.contains("node '3'"), "{err}");
    // Gaining a declared label answers to it.
    run(
        &mut graph,
        "CREATE (:Other {id: 4, vf: date('2011-01-01'), vt: date('2010-01-01')})",
    )
    .unwrap();
    let err = run(&mut graph, "MATCH (o:Other {id: 4}) SET o:Status").unwrap_err();
    assert!(err.contains("node '4'"), "{err}");
}

fn stamp(text: &str) -> Value {
    Value::Timestamp(chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S").unwrap())
}

#[test]
fn a_half_open_timestamp_from_after_a_date_to_midnight_is_inverted_not_empty() {
    // [2011-01-01T12:00, 2011-01-01): the `to` day is the first day no longer
    // valid, so the interval ends at its midnight, before the `from`.
    let mut graph = statuses(HalfOpen);
    let rows = frame(
        &["id", "vf", "vt"],
        vec![vec![
            Value::Int64(2),
            stamp("2011-01-01T12:00:00"),
            day("2011-01-01"),
        ]],
    );
    let err = load(&mut graph, rows, None).unwrap_err();
    assert!(err.contains("is after the to bound"), "{err}");
    assert_eq!(status_count(&mut graph), 1, "nothing written");
}

#[test]
fn a_half_open_timestamp_from_at_a_date_to_midnight_is_empty() {
    let mut graph = statuses(HalfOpen);
    let rows = frame(
        &["id", "vf", "vt"],
        vec![vec![
            Value::Int64(2),
            stamp("2011-01-01T00:00:00"),
            day("2011-01-01"),
        ]],
    );
    load(&mut graph, rows, None).unwrap();
    assert_eq!(list(&graph)[0].empty_rows, Some(1));
}

#[test]
fn a_closed_timestamp_from_within_a_date_to_day_is_neither_inverted_nor_empty() {
    // Closed: the `to` day is the last valid day, so T12:00 of it is valid.
    let mut graph = statuses(Closed);
    let rows = frame(
        &["id", "vf", "vt"],
        vec![vec![
            Value::Int64(2),
            stamp("2011-01-01T12:00:00"),
            day("2011-01-01"),
        ]],
    );
    load(&mut graph, rows, None).unwrap();
    assert_eq!(list(&graph)[0].empty_rows, Some(0));
}

#[test]
fn a_half_open_date_from_with_a_same_day_timestamp_to_is_not_empty() {
    let mut graph = statuses(HalfOpen);
    let rows = frame(
        &["id", "vf", "vt"],
        vec![vec![
            Value::Int64(2),
            day("2011-01-01"),
            stamp("2011-01-01T06:00:00"),
        ]],
    );
    load(&mut graph, rows, None).unwrap();
    assert_eq!(list(&graph)[0].empty_rows, Some(0));
}

#[test]
fn iso_text_bounds_of_a_declared_type_are_stored_as_dates() {
    let mut graph = statuses(Closed);
    let rows = frame(
        &["id", "vf", "vt"],
        vec![
            vec![
                Value::Int64(2),
                Value::String("2011-01-01".into()),
                Value::String("2012-01-01".into()),
            ],
            vec![
                Value::Int64(3),
                Value::String("2011-06-01".into()),
                Value::Null,
            ],
        ],
    );
    load(&mut graph, rows, None).unwrap();
    let bounds = |id: i64| {
        let idx = graph
            .id_indices
            .lookup("Status", &Value::Int64(id))
            .unwrap();
        (
            crate::graph::features::temporal::node_bound(&graph, idx, "vf"),
            crate::graph::features::temporal::node_bound(&graph, idx, "vt"),
        )
    };
    assert_eq!(bounds(2), (day("2011-01-01"), day("2012-01-01")));
    assert_eq!(bounds(3), (day("2011-06-01"), Value::Null));
    let recorded = graph.get_node_type_metadata("Status").unwrap();
    assert!(
        !recorded.get("vf").unwrap().eq_ignore_ascii_case("string"),
        "{recorded:?}"
    );
}

#[test]
fn an_unparsable_text_bound_is_still_refused() {
    let mut graph = statuses(Closed);
    let rows = frame(
        &["id", "vf", "vt"],
        vec![vec![
            Value::Int64(2),
            Value::String("not a date".into()),
            Value::Null,
        ]],
    );
    let err = load(&mut graph, rows, None).unwrap_err();
    assert!(err.contains("row 0"), "{err}");
    assert_eq!(status_count(&mut graph), 1);
}
