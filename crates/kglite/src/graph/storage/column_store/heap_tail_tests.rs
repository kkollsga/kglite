//! Heap tails: a transaction's appends to a large type under a writer overlay
//! copy no column, read back on every route, roll back, save, and fold into
//! the store when the overlay folds (`tail.rs`, "Heap tails").

use super::tail::HEAP_TAIL_MIN_ROWS;
use super::{column_clones, reset_column_clones};
use crate::datatypes::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::session::{execute_mut, execute_read, CommitOutcome, ExecuteOptions, Session};
use std::collections::HashMap;

const ROWS: u32 = HEAP_TAIL_MIN_ROWS + 1_000;

fn execute(graph: &mut DirGraph, query: &str) -> Result<(), String> {
    execute_mut(graph, query, &ExecuteOptions::eager(&HashMap::new()))
        .map(|_| ())
        .map_err(|e| e.to_string())
}

fn one(graph: &DirGraph, query: &str) -> Value {
    let out = execute_read(graph, query, &ExecuteOptions::eager(&HashMap::new()))
        .unwrap_or_else(|e| panic!("{query}: {e}"));
    out.result.rows[0][0].clone()
}

fn seeded_graph() -> DirGraph {
    let mut graph = DirGraph::new();
    execute(
        &mut graph,
        &format!(
            "UNWIND range(0, {}) AS i CREATE (:Item {{id: i, title: 'n' + toString(i), \
             score: i, tag: 'base'}})",
            ROWS - 1
        ),
    )
    .unwrap();
    graph
}

fn seed() -> Session {
    Session::new(seeded_graph())
}

fn heap_tail(graph: &DirGraph) -> bool {
    graph.column_store("Item").unwrap().has_heap_tail()
}

/// Every read route a tail row must answer on: the id point lookup, a
/// filtered scan the column fast path would serve, a count and an aggregate.
fn assert_reads(graph: &DirGraph, created: &[i64]) {
    let rows = i64::from(ROWS) + created.len() as i64;
    assert_eq!(
        one(graph, "MATCH (n:Item) RETURN count(n)"),
        Value::Int64(rows)
    );
    for &id in created {
        assert_eq!(
            one(
                graph,
                &format!("MATCH (n:Item {{id: {id}}}) RETURN n.title")
            ),
            Value::String(format!("t{id}"))
        );
    }
    assert_eq!(
        one(graph, "MATCH (n:Item) WHERE n.tag = 'tx' RETURN count(n)"),
        Value::Int64(created.len() as i64)
    );
    assert_eq!(
        one(
            graph,
            &format!("MATCH (n:Item) WHERE n.score >= {ROWS} RETURN sum(n.score)")
        ),
        Value::Int64(created.iter().sum())
    );
}

fn create(id: i64) -> String {
    format!("CREATE (:Item {{id: {id}, title: 't{id}', score: {id}, tag: 'tx'}})")
}

#[test]
fn a_transaction_append_on_a_large_type_copies_no_column() {
    let session = seed();
    let mut created = Vec::new();
    for id in [i64::from(ROWS), i64::from(ROWS) + 1] {
        let mut tx = session.begin();
        let working = tx.working_mut().unwrap();
        reset_column_clones();
        execute(working, &create(id)).unwrap();
        assert_eq!(column_clones(), 0, "the append copied a shared column");
        assert!(heap_tail(working));
        created.push(id);
        assert_reads(working, &created);
        assert!(matches!(
            session.commit(tx, true),
            CommitOutcome::Committed { .. }
        ));
        let published = session.snapshot();
        assert!(!heap_tail(&published), "the commit's fold left the tail");
        assert_reads(&published, &created);
    }
}

#[test]
fn a_small_type_or_an_unshared_store_takes_no_tail() {
    let mut graph = DirGraph::new();
    execute(
        &mut graph,
        "UNWIND range(0, 99) AS i CREATE (:Item {id: i})",
    )
    .unwrap();
    let session = Session::new(graph);
    let mut tx = session.begin();
    execute(tx.working_mut().unwrap(), "CREATE (:Item {id: 100})").unwrap();
    assert!(!heap_tail(tx.working_mut().unwrap()), "below the row floor");

    let mut big = seeded_graph();
    execute(&mut big, &create(i64::from(ROWS))).unwrap();
    assert!(!heap_tail(&big), "no overlay, no tail");
}

#[test]
fn a_failed_statement_in_a_transaction_rolls_its_tail_rows_back() {
    let session = seed();
    let mut tx = session.begin();
    let working = tx.working_mut().unwrap();
    let id = i64::from(ROWS);
    execute(working, &create(id)).unwrap();
    let error = execute(
        working,
        &format!(
            "UNWIND [1, 2] AS i CREATE (:Item {{id: {} + i, tag: 'tx', \
             score: CASE WHEN i = 2 THEN duration({{months: 2147483648}}) ELSE i END}})",
            id
        ),
    );
    assert!(error.is_err());
    assert_reads(working, &[id]);
    execute(working, &create(id + 5)).unwrap();
    assert_reads(working, &[id, id + 5]);
    assert!(matches!(
        session.commit(tx, true),
        CommitOutcome::Committed { .. }
    ));
    assert_reads(&session.snapshot(), &[id, id + 5]);
}

#[test]
fn a_held_reader_keeps_the_tail_until_it_drops_and_a_save_holds_every_row() {
    let session = seed();
    let holder = session.snapshot();
    let id = i64::from(ROWS);
    let mut tx = session.begin();
    execute(tx.working_mut().unwrap(), &create(id)).unwrap();
    assert!(matches!(
        session.commit(tx, true),
        CommitOutcome::Committed { .. }
    ));
    let published = session.snapshot();
    assert!(published.graph.is_forked());
    assert!(heap_tail(&published));
    assert_reads(&published, &[id]);
    // The compiled scan filter still serves the base rows; the tail row
    // reaches the result through the row route.
    use crate::graph::core::pattern_matching::column_filter::{reset_rows_filtered, rows_filtered};
    reset_rows_filtered();
    assert_eq!(
        one(&published, "MATCH (n:Item) WHERE n.tag = 'tx' RETURN n.id"),
        Value::Int64(id)
    );
    assert!(
        rows_filtered() >= ROWS as usize,
        "the compiled filter declined"
    );
    assert_eq!(
        one(&holder, "MATCH (n:Item) RETURN count(n)"),
        Value::Int64(i64::from(ROWS))
    );

    let mut bytes = Vec::new();
    crate::graph::io::file::write_kgl_to(&published, &mut bytes).unwrap();
    let loaded = crate::graph::io::file::load_kgl_bytes(&bytes).unwrap();
    assert_reads(&loaded, &[id]);

    drop((holder, published));
    let mut tx = session.begin();
    execute(tx.working_mut().unwrap(), &create(id + 1)).unwrap();
    assert!(matches!(
        session.commit(tx, true),
        CommitOutcome::Committed { .. }
    ));
    let compacted = session.snapshot();
    assert!(!compacted.graph.is_forked());
    assert!(!heap_tail(&compacted));
    assert_reads(&compacted, &[id, id + 1]);
}

fn commit(session: &Session, query: &str) {
    let mut tx = session.begin();
    execute(tx.working_mut().unwrap(), query).unwrap();
    assert!(matches!(
        session.commit(tx, true),
        CommitOutcome::Committed { .. }
    ));
}

fn recorded(graph: &DirGraph, key: &str) -> Option<String> {
    Some(
        graph
            .get_node_type_metadata("Item")?
            .get(key)?
            .to_lowercase(),
    )
}

/// `(live, saved and loaded, after the fold)` record of `w`.
fn record_after(
    session: &Session,
    holder: Option<std::sync::Arc<DirGraph>>,
) -> [Option<String>; 3] {
    let published = session.snapshot();
    assert_eq!(heap_tail(&published), holder.is_some());
    let live = recorded(&published, "w");
    let mut bytes = Vec::new();
    crate::graph::io::file::write_kgl_to(&published, &mut bytes).unwrap();
    let loaded = recorded(
        &crate::graph::io::file::load_kgl_bytes(&bytes).unwrap(),
        "w",
    );
    drop((holder, published));
    commit(session, &create(i64::from(ROWS) + 9));
    let folded = session.snapshot();
    assert!(!heap_tail(&folded));
    [live, loaded, recorded(&folded, "w")]
}

/// A key first written to tail rows has its column only in the tail. An
/// integer written after a float widens that column to Mixed, so the type
/// record becomes `mixed` — as it does on the store without a tail.
#[test]
fn a_tail_only_float_key_records_mixed_after_an_integer() {
    let id = i64::from(ROWS);
    let run = |hold: bool| {
        let session = seed();
        let holder = hold.then(|| session.snapshot());
        commit(&session, &create(id));
        commit(&session, &create(id + 1));
        commit(
            &session,
            &format!("MATCH (n:Item {{id: {id}}}) SET n.w = 1.5"),
        );
        commit(
            &session,
            &format!("MATCH (n:Item {{id: {}}}) SET n.w = 2", id + 1),
        );
        record_after(&session, holder)
    };
    let control = run(false);
    assert_eq!(control, [(); 3].map(|()| Some("mixed".to_string())));
    assert_eq!(run(true), control);
}

/// The `add_nodes` route to the same record: a frame bringing the float, then
/// one bringing an integer, both landing in the tail.
#[test]
fn add_nodes_into_a_tail_records_mixed_after_an_integer() {
    use crate::datatypes::values::{ColumnData, ColumnType, DataFrame};
    let frame = |id: i64, w: Option<f64>, int_w: Option<i64>| {
        let mut df = DataFrame::new(Vec::new());
        df.add_column(
            "id".into(),
            ColumnType::Int64,
            ColumnData::Int64(vec![Some(id)]),
        )
        .unwrap();
        match (w, int_w) {
            (Some(w), _) => df
                .add_column(
                    "w".into(),
                    ColumnType::Float64,
                    ColumnData::Float64(vec![Some(w)]),
                )
                .unwrap(),
            (_, w) => df
                .add_column("w".into(), ColumnType::Int64, ColumnData::Int64(vec![w]))
                .unwrap(),
        }
        df
    };
    let id = i64::from(ROWS);
    let run = |hold: bool| {
        let session = seed();
        let holder = hold.then(|| session.snapshot());
        for df in [frame(id, Some(1.5), None), frame(id + 1, None, Some(2))] {
            let mut tx = session.begin();
            crate::graph::mutation::maintain::add_nodes(
                tx.working_mut().unwrap(),
                df,
                "Item".into(),
                "id".into(),
                None,
                None,
            )
            .unwrap();
            assert!(matches!(
                session.commit(tx, true),
                CommitOutcome::Committed { .. }
            ));
        }
        record_after(&session, holder)
    };
    let control = run(false);
    assert_eq!(control, [(); 3].map(|()| Some("mixed".to_string())));
    assert_eq!(run(true), control);
}

/// A heap tail's id column widens on its own: an `Int64` id past `u32` in a
/// tail of a compact-id type. The store's id kind is the kind its folded
/// column will have (`Int64`), not the base part's — an external writer
/// opens its id column from it.
#[test]
fn a_widened_tail_id_column_reports_the_folded_kind() {
    use crate::datatypes::values::{ColumnData, ColumnType, DataFrame};
    let mut graph = DirGraph::new();
    let mut df = DataFrame::new(Vec::new());
    df.add_column(
        "id".into(),
        ColumnType::UniqueId,
        ColumnData::UniqueId((0..ROWS).map(Some).collect()),
    )
    .unwrap();
    crate::graph::mutation::maintain::add_nodes(
        &mut graph,
        df,
        "Item".into(),
        "id".into(),
        None,
        None,
    )
    .unwrap();
    let session = Session::new(graph);
    let kind = |graph: &DirGraph| graph.column_store("Item").unwrap().id_type_str();
    assert_eq!(kind(&session.snapshot()), Some("uniqueid"));
    let holder = session.snapshot();
    commit(&session, "CREATE (:Item {id: 1099511627776})");
    let published = session.snapshot();
    assert!(heap_tail(&published));
    assert_eq!(kind(&published), Some("int64"));
    drop((holder, published));
    commit(&session, "CREATE (:Item {id: 1099511627777})");
    let folded = session.snapshot();
    assert!(!heap_tail(&folded));
    assert_eq!(kind(&folded), Some("int64"));
}

/// `add_nodes` under a held reader on a large type appends into a tail: no
/// shared column is copied, the rows and a column only they carry read back,
/// and the fold after the reader lets go leaves no tail.
#[test]
fn add_nodes_under_a_held_reader_copies_no_column() {
    use crate::datatypes::values::{ColumnData, ColumnType, DataFrame};
    let session = seed();
    let holder = session.snapshot();
    let ids: Vec<i64> = (0..3).map(|i| i64::from(ROWS) + i).collect();
    let mut df = DataFrame::new(Vec::new());
    df.add_column(
        "id".into(),
        ColumnType::Int64,
        ColumnData::Int64(ids.iter().copied().map(Some).collect()),
    )
    .unwrap();
    df.add_column(
        "title".into(),
        ColumnType::String,
        ColumnData::String(ids.iter().map(|id| Some(format!("t{id}"))).collect()),
    )
    .unwrap();
    df.add_column(
        "fresh".into(),
        ColumnType::Int64,
        ColumnData::Int64(ids.iter().map(|id| Some(id * 10)).collect()),
    )
    .unwrap();
    let mut tx = session.begin();
    reset_column_clones();
    crate::graph::mutation::maintain::add_nodes(
        tx.working_mut().unwrap(),
        df,
        "Item".into(),
        "id".into(),
        Some("title".into()),
        None,
    )
    .unwrap();
    assert_eq!(column_clones(), 0, "add_nodes copied a shared column");
    assert!(heap_tail(tx.working_mut().unwrap()));
    assert!(matches!(
        session.commit(tx, true),
        CommitOutcome::Committed { .. }
    ));
    let read = |graph: &DirGraph| {
        (
            one(graph, "MATCH (n:Item) RETURN count(n)"),
            one(
                graph,
                "MATCH (n:Item) WHERE n.fresh IS NOT NULL RETURN collect(n.fresh)",
            ),
        )
    };
    let expected = (
        Value::Int64(i64::from(ROWS) + 3),
        Value::List(ids.iter().map(|id| Value::Int64(id * 10)).collect()),
    );
    assert_eq!(read(&session.snapshot()), expected);
    drop(holder);
    commit(&session, "MATCH (n:Item {id: 0}) SET n.score = 1");
    let folded = session.snapshot();
    assert!(!folded.graph.is_forked());
    assert!(!heap_tail(&folded));
    assert_eq!(read(&folded), expected);
}
