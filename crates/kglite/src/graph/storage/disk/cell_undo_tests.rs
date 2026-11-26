//! A failed disk statement rolls back exactly through the cell journal: cells,
//! titles, grown schema, appended rows and the column *types* a write demoted.
//! (That a `SET` statement copies no column is `flush_tests`.)
use crate::datatypes::Value;
use crate::graph::schema::DirGraph;
use crate::graph::session::execute::{execute_mut, ExecuteOptions};
use crate::graph::storage::column_store::{column_clones, reset_column_clones, TypedColumn};
use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};
use crate::graph::storage::GraphRead;
use std::collections::HashMap;

fn run(graph: &mut DirGraph, query: &str) -> Result<(), String> {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// A row fails the statement at `i = 250`: the duration's month count does not
/// fit, after 249 rows' worth of writes.
const FAIL_AT_250: &str = "CASE WHEN i = 250 THEN duration({months: 2147483648}) ELSE i END";

fn prop(graph: &DirGraph, id: i64, key: &str) -> Option<Value> {
    let idx = graph
        .graph
        .node_indices()
        .find(|i| graph.graph.get_node_id(*i) == Some(Value::Int64(id)))
        .expect("node exists");
    graph
        .graph
        .node_view(idx)
        .and_then(|n| n.get_property_value(key))
}

/// 2,000 staff: `id`, an int64 column `grade`, a string column `name` and a
/// string title.
fn staff_graph(dir: &std::path::Path) -> DirGraph {
    let mut graph = new_dir_graph_in_mode(StorageMode::Disk, Some(dir)).expect("disk graph");
    run(
        &mut graph,
        "UNWIND range(1, 2000) AS i \
         CREATE (:Staff {id: i, title: 'Employee ' + toString(i), grade: 0, name: 'n' + toString(i)})",
    )
    .unwrap();
    assert!(graph.graph.is_disk());
    graph
}

fn column_kind(graph: &DirGraph, key: &str) -> &'static str {
    let type_key = crate::graph::schema::InternedKey::from_str("Staff");
    let store = graph.graph.column_store(type_key).expect("Staff store");
    let slot = store
        .slot(crate::graph::schema::InternedKey::from_str(key))
        .expect("column exists");
    match store.column(slot as usize).expect("column") {
        TypedColumn::Int64 { .. } => "int64",
        TypedColumn::Str { .. } => "str",
        TypedColumn::Mixed { .. } => "mixed",
        _ => "other",
    }
}

fn title_kind(graph: &DirGraph) -> &'static str {
    let type_key = crate::graph::schema::InternedKey::from_str("Staff");
    let store = graph.graph.column_store(type_key).expect("Staff store");
    match store.title_column_ref() {
        Some(TypedColumn::Str { .. }) => "str",
        Some(TypedColumn::Mixed { .. }) => "mixed",
        Some(_) => "other",
        None => "none",
    }
}

#[test]
fn a_failed_disk_set_only_statement_restores_cells_titles_and_schema() {
    let dir = tempfile::tempdir().unwrap();
    let mut graph = staff_graph(dir.path());
    run(&mut graph, "MATCH (n:Staff {id: 2}) SET n.grade = 7").unwrap();
    let before_keys = graph.type_schemas.get("Staff").map(|s| s.len());
    let err = run(
        &mut graph,
        &format!(
            "UNWIND range(1, 300) AS i MATCH (n:Staff {{id: i}}) \
             SET n.grade = i * 10, n.fresh = i, n.title = 'changed', n.extra = {FAIL_AT_250}"
        ),
    );
    assert!(
        err.is_err(),
        "the statement must fail after its first writes"
    );
    for (id, grade) in [(1, 0), (2, 7), (249, 0), (250, 0), (300, 0)] {
        assert_eq!(
            prop(&graph, id, "grade"),
            Some(Value::Int64(grade)),
            "grade of {id}"
        );
        assert_eq!(prop(&graph, id, "fresh"), None, "fresh of {id}");
        assert_eq!(prop(&graph, id, "extra"), None, "extra of {id}");
        assert_eq!(
            prop(&graph, id, "title"),
            Some(Value::String(format!("Employee {id}"))),
            "title of {id}"
        );
    }
    assert_eq!(
        graph.type_schemas.get("Staff").map(|s| s.len()),
        before_keys
    );
    // The graph stays writable and the next statement lands.
    run(&mut graph, "MATCH (n:Staff {id: 3}) SET n.grade = 3").unwrap();
    assert_eq!(prop(&graph, 3, "grade"), Some(Value::Int64(3)));
    assert_eq!(prop(&graph, 4, "grade"), Some(Value::Int64(0)));
}

/// A `SET` the int64 column cannot hold demotes it to `Mixed`; rolled back, the
/// column must be the int64 column again — a `Mixed` column has no file
/// representation, so leaving it behind changes what the next save writes.
#[test]
fn a_rolled_back_demotion_restores_the_column_type() {
    let dir = tempfile::tempdir().unwrap();
    let mut graph = staff_graph(dir.path());
    assert_eq!(column_kind(&graph, "grade"), "int64");
    let err = run(
        &mut graph,
        &format!(
            "UNWIND range(1, 300) AS i MATCH (n:Staff {{id: i}}) \
             SET n.grade = CASE WHEN i = 100 THEN 'unknown' ELSE i END, n.extra = {FAIL_AT_250}"
        ),
    );
    assert!(err.is_err());
    assert_eq!(
        column_kind(&graph, "grade"),
        "int64",
        "the demotion was not undone"
    );
    for id in [1, 99, 100, 101, 300] {
        assert_eq!(
            prop(&graph, id, "grade"),
            Some(Value::Int64(0)),
            "grade of {id}"
        );
    }
    // Non-vacuity: the same write that succeeds does demote, so the assertion
    // above is about the rollback and not about a column that never changed.
    run(
        &mut graph,
        "MATCH (n:Staff {id: 100}) SET n.grade = 'unknown'",
    )
    .unwrap();
    assert_eq!(column_kind(&graph, "grade"), "mixed");
    assert_eq!(
        prop(&graph, 100, "grade"),
        Some(Value::String("unknown".into()))
    );
}

#[test]
fn a_rolled_back_title_demotion_restores_the_title_column() {
    let dir = tempfile::tempdir().unwrap();
    let mut graph = staff_graph(dir.path());
    let before = title_kind(&graph);
    assert_eq!(before, "str");
    let err = run(
        &mut graph,
        &format!(
            "UNWIND range(1, 300) AS i MATCH (n:Staff {{id: i}}) \
             SET n.title = CASE WHEN i = 100 THEN 42 ELSE 'changed' END, n.extra = {FAIL_AT_250}"
        ),
    );
    assert!(err.is_err());
    assert_eq!(
        title_kind(&graph),
        before,
        "the title demotion was not undone"
    );
    assert_eq!(
        prop(&graph, 100, "title"),
        Some(Value::String("Employee 100".into()))
    );
}

fn row_count(graph: &DirGraph, node_type: &str) -> Option<u32> {
    let type_key = crate::graph::schema::InternedKey::from_str(node_type);
    graph.graph.column_store(type_key).map(|s| s.row_count())
}

fn staff_count(graph: &mut DirGraph) -> i64 {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    let out = execute_mut(graph, "MATCH (n:Staff) RETURN count(n) AS c", &opts).unwrap();
    match out.result.rows[0].first() {
        Some(Value::Int64(c)) => *c,
        other => panic!("unexpected count {other:?}"),
    }
}

/// A `CREATE` of 300 staff that fails at the 250th: nothing of it may remain —
/// not a row, not a node, not a column it introduced or a type it demoted.
#[test]
fn a_failed_disk_create_truncates_its_rows_and_columns() {
    let dir = tempfile::tempdir().unwrap();
    let mut graph = staff_graph(dir.path());
    let rows_before = row_count(&graph, "Staff");
    let keys_before = graph.type_schemas.get("Staff").map(|s| s.len());
    let err = run(
        &mut graph,
        &format!(
            "UNWIND range(1, 300) AS i CREATE (:Staff {{id: 5000 + i, name: 'x', \
             grade: CASE WHEN i = 100 THEN 'unknown' ELSE {FAIL_AT_250} END}})"
        ),
    );
    assert!(err.is_err());
    assert_eq!(
        row_count(&graph, "Staff"),
        rows_before,
        "appended rows remain"
    );
    assert_eq!(
        graph.type_schemas.get("Staff").map(|s| s.len()),
        keys_before
    );
    assert_eq!(staff_count(&mut graph), 2000);
    assert_eq!(
        column_kind(&graph, "grade"),
        "int64",
        "the demotion was not undone"
    );
    // The vacated rows are reused, and the graph is consistent afterwards.
    run(&mut graph, "CREATE (:Staff {id: 9001, grade: 5})").unwrap();
    assert_eq!(row_count(&graph, "Staff"), rows_before.map(|n| n + 1));
    assert_eq!(prop(&graph, 9001, "grade"), Some(Value::Int64(5)));
    assert_eq!(staff_count(&mut graph), 2001);

    // Non-vacuity: the same rows without the failing one do append and demote,
    // so the assertions above were about the rollback.
    run(
        &mut graph,
        "UNWIND range(1, 300) AS i CREATE (:Staff {id: 5000 + i, name: 'x', \
         grade: CASE WHEN i = 100 THEN 'unknown' ELSE i END})",
    )
    .unwrap();
    assert_eq!(staff_count(&mut graph), 2301);
    assert_eq!(column_kind(&graph, "grade"), "mixed");
}

/// A statement that creates the type's store is undone by dropping the store: an
/// empty-but-present one is observable.
#[test]
fn a_failed_disk_create_of_a_new_type_leaves_no_store_behind() {
    let dir = tempfile::tempdir().unwrap();
    let mut graph = staff_graph(dir.path());
    let err = run(
        &mut graph,
        &format!("UNWIND range(1, 300) AS i CREATE (:Contractor {{id: i, v: {FAIL_AT_250}}})"),
    );
    assert!(err.is_err());
    assert_eq!(row_count(&graph, "Contractor"), None);
    run(&mut graph, "CREATE (:Contractor {id: 1, v: 1})").unwrap();
    assert_eq!(row_count(&graph, "Contractor"), Some(1));
}

#[test]
fn a_failed_disk_merge_that_creates_and_matches_restores_both_branches() {
    let dir = tempfile::tempdir().unwrap();
    let mut graph = staff_graph(dir.path());
    let rows_before = row_count(&graph, "Staff");
    let err = run(
        &mut graph,
        &format!(
            "UNWIND range(1, 300) AS i MERGE (n:Staff {{id: 1900 + i}}) \
             ON CREATE SET n.grade = {FAIL_AT_250} \
             ON MATCH SET n.grade = 500 + i, n.name = 'renamed', n.extra = i"
        ),
    );
    assert!(err.is_err());
    assert_eq!(row_count(&graph, "Staff"), rows_before);
    assert_eq!(staff_count(&mut graph), 2000);
    for id in [1901, 2000] {
        assert_eq!(
            prop(&graph, id, "grade"),
            Some(Value::Int64(0)),
            "matched row {id}"
        );
        assert_eq!(prop(&graph, id, "extra"), None, "matched row {id}");
    }
}

/// A deleted node comes back on its slot, with its edges and its cells, when a
/// later clause of the statement fails.
#[test]
fn a_failed_disk_delete_restores_the_nodes() {
    let dir = tempfile::tempdir().unwrap();
    let mut graph = staff_graph(dir.path());
    run(
        &mut graph,
        "MATCH (a:Staff {id: 1}), (b:Staff {id: 2}) CREATE (a)-[:REPORTS_TO {since: 2020}]->(b)",
    )
    .unwrap();
    let err = run(
        &mut graph,
        &format!("MATCH (n:Staff) WHERE n.id <= 300 DETACH DELETE n WITH count(n) AS c CREATE (:Blocked {{v: {FAIL_AT_250}}})"),
    );
    assert!(err.is_err());
    assert_eq!(staff_count(&mut graph), 2000);
    assert_eq!(prop(&graph, 2, "name"), Some(Value::String("n2".into())));
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    let out = execute_mut(
        &mut graph,
        "MATCH (:Staff {id: 1})-[r:REPORTS_TO]->(:Staff {id: 2}) RETURN r.since AS s",
        &opts,
    )
    .unwrap();
    assert_eq!(out.result.rows.len(), 1, "the edge was not restored");
}

/// A `CREATE` statement appends in place: no column of the type is copied
/// however large the type is (the deep copy per touched column was the cost).
#[test]
fn a_disk_create_statement_copies_no_column() {
    let dir = tempfile::tempdir().unwrap();
    let mut graph = staff_graph(dir.path());
    run(
        &mut graph,
        "CREATE (:Staff {id: 7000, grade: 1, name: 'warm'})",
    )
    .unwrap();
    reset_column_clones();
    run(
        &mut graph,
        "UNWIND range(1, 200) AS i CREATE (:Staff {id: 7000 + i, grade: i, name: 'x'})",
    )
    .unwrap();
    assert_eq!(column_clones(), 0, "an append copied a column");
    assert_eq!(staff_count(&mut graph), 2201);
}
