//! Routing regression for [`CypherExecutor::try_fast_with_aggregate_via_histogram`]:
//! its typed-source branch reads the disk graph directly, and a disk graph
//! opened `durable=True` (or with `cdc::enable`) is wrapped in
//! `GraphBackend::Recording`. Matching `GraphBackend::Disk` there bailed the
//! whole fast path — the caller's per-source fallback returns the same rows,
//! so the forfeit is silent.

use super::*;
use crate::datatypes::{DataFrame, Value};
use crate::graph::languages::cypher::ast::{Clause, WithClause};
use crate::graph::languages::cypher::parse_cypher;
use std::collections::HashMap;
use tempfile::TempDir;

/// Two `Person`s pointing at one `City` via `VISITED`.
///
/// The connections are added **after** `enable_disk_mode`, deliberately: they
/// route through overflow, and `save_disk` seals them into the CSR *with* a
/// `conn_type_index_*` — the indexed shape the fast path's typed branch reads.
/// `converted_disk_graph` below is the index-less counterpart.
fn disk_graph(dir: &TempDir) -> DirGraph {
    let people = DataFrame::from_cypher_rows(
        vec!["id".into(), "title".into()],
        vec![
            vec![Value::Int64(1), Value::String("p1".into())],
            vec![Value::Int64(2), Value::String("p2".into())],
        ],
    )
    .unwrap();
    let cities = DataFrame::from_cypher_rows(
        vec!["id".into(), "title".into()],
        vec![vec![Value::Int64(10), Value::String("Oslo".into())]],
    )
    .unwrap();
    let visits = DataFrame::from_cypher_rows(
        vec!["src".into(), "tgt".into()],
        vec![
            vec![Value::Int64(1), Value::Int64(10)],
            vec![Value::Int64(2), Value::Int64(10)],
        ],
    )
    .unwrap();

    let mut graph = DirGraph::new();
    crate::graph::mutation::maintain::add_nodes(
        &mut graph,
        people,
        "Person".to_string(),
        "id".to_string(),
        Some("title".to_string()),
        None,
    )
    .unwrap();
    crate::graph::mutation::maintain::add_nodes(
        &mut graph,
        cities,
        "City".to_string(),
        "id".to_string(),
        Some("title".to_string()),
        None,
    )
    .unwrap();
    graph.enable_disk_mode().unwrap();
    crate::graph::mutation::maintain::add_connections(
        &mut graph,
        visits,
        "VISITED".to_string(),
        "Person".to_string(),
        "src".to_string(),
        "City".to_string(),
        "tgt".to_string(),
        None,
        None,
        None,
    )
    .unwrap();
    graph.save_disk(dir.path().to_str().unwrap()).unwrap();
    graph
}

/// `MATCH (p:Person)-[:VISITED]->(c) WITH c, count(p) AS n` — the typed-source
/// shape whose fast path reads the disk graph.
fn typed_scan_aggregate() -> (Pattern, WithClause) {
    let query =
        parse_cypher("MATCH (p:Person)-[:VISITED]->(c) WITH c, count(p) AS n RETURN c, n").unwrap();
    let mut pattern = None;
    let mut with_clause = None;
    for clause in query.clauses {
        match clause {
            Clause::Match(mc) => pattern = Some(mc.patterns[0].clone()),
            Clause::With(wc) => with_clause = Some(wc),
            _ => {}
        }
    }
    (pattern.unwrap(), with_clause.unwrap())
}

/// `Some(rows)` iff the histogram fast path ran; `None` is the bail the defect
/// produced.
fn run_fast_path(graph: &DirGraph) -> Option<Vec<ResultRow>> {
    let (pattern, with_clause) = typed_scan_aggregate();
    let params: HashMap<String, Value> = HashMap::new();
    let executor = CypherExecutor::with_params(graph, &params, None);
    executor
        .try_fast_with_aggregate_via_histogram(
            &pattern,
            &with_clause,
            &["c".to_string(), "n".to_string()],
            "c",
            2,
            &[0],
            &[1],
        )
        .unwrap()
}

fn counts_of(rows: &[ResultRow]) -> Vec<i64> {
    rows.iter()
        .map(|row| match row.projected.get("n") {
            Some(Value::Int64(n)) => *n,
            other => panic!("expected an Int64 count, got {other:?}"),
        })
        .collect()
}

#[test]
fn a_capture_wrapped_disk_graph_still_takes_the_typed_scan_fast_path() {
    let dir = TempDir::new().unwrap();
    let mut graph = disk_graph(&dir);
    let bare = run_fast_path(&graph).expect("the bare disk graph must take the fast path");
    assert_eq!(counts_of(&bare), vec![2]);

    graph.graph.wrap_for_durability();
    assert!(
        graph.graph.is_recording(),
        "the wrap must have taken effect, or this test asserts nothing"
    );
    assert!(
        graph.graph.as_disk().is_some(),
        "a durability-wrapped disk graph is still a disk graph"
    );

    let wrapped = run_fast_path(&graph)
        .expect("a durability-wrapped disk graph must still take the fast path");
    assert_eq!(counts_of(&wrapped), counts_of(&bare));
}

/// The same two `Person`s and one `City`, with the connections added **before**
/// `enable_disk_mode` — so the conversion carries every edge into the CSR with
/// no `conn_type_index_*` and nothing in overflow. This is what
/// `enable_disk_mode()` on an already-populated in-memory graph produces, and
/// `save_disk` + reload preserves it (no index file is written).
fn converted_disk_graph(dir: &TempDir) -> DirGraph {
    let people = DataFrame::from_cypher_rows(
        vec!["id".into(), "title".into()],
        vec![
            vec![Value::Int64(1), Value::String("a".into())],
            vec![Value::Int64(2), Value::String("b".into())],
        ],
    )
    .unwrap();
    let cities = DataFrame::from_cypher_rows(
        vec!["id".into(), "title".into()],
        vec![vec![Value::Int64(10), Value::String("Oslo".into())]],
    )
    .unwrap();
    let visits = DataFrame::from_cypher_rows(
        vec!["src".into(), "tgt".into()],
        vec![
            vec![Value::Int64(1), Value::Int64(10)],
            vec![Value::Int64(2), Value::Int64(10)],
        ],
    )
    .unwrap();

    let mut graph = DirGraph::new();
    crate::graph::mutation::maintain::add_nodes(
        &mut graph,
        people,
        "Person".to_string(),
        "id".to_string(),
        Some("title".to_string()),
        None,
    )
    .unwrap();
    crate::graph::mutation::maintain::add_nodes(
        &mut graph,
        cities,
        "City".to_string(),
        "id".to_string(),
        Some("title".to_string()),
        None,
    )
    .unwrap();
    crate::graph::mutation::maintain::add_connections(
        &mut graph,
        visits,
        "VISITED".to_string(),
        "Person".to_string(),
        "src".to_string(),
        "City".to_string(),
        "tgt".to_string(),
        None,
        None,
        None,
    )
    .unwrap();
    graph.enable_disk_mode().unwrap();
    graph.save_disk(dir.path().to_str().unwrap()).unwrap();
    graph
}

/// The typed-source fast path reads the disk edge scan, which walked only the
/// persisted `conn_type_index_*`. A converted disk graph has none, so the scan
/// visited nothing and the aggregate returned **zero rows** while the
/// unoptimised path returned one — a silent wrong answer, not a bail.
#[test]
fn a_converted_disk_graph_aggregates_over_its_csr_edges() {
    let dir = TempDir::new().unwrap();
    let graph = converted_disk_graph(&dir);

    assert!(
        graph
            .graph
            .as_disk()
            .expect("disk mode")
            .conn_type_index_types
            .is_empty(),
        "the conversion builds no conn-type index, or this test asserts nothing"
    );

    let rows = run_fast_path(&graph).expect("the converted disk graph must take the fast path");
    assert_eq!(
        counts_of(&rows),
        vec![2],
        "both VISITED edges must reach the aggregate"
    );
}

/// End-to-end through the optimizer pipeline, against the unoptimised path as
/// oracle. The differential corpus is in-memory only, so this shape's disk half
/// has to be pinned here.
#[test]
fn the_full_statement_returns_rows_on_a_converted_disk_graph() {
    let dir = TempDir::new().unwrap();
    let graph = converted_disk_graph(&dir);
    let text = "MATCH (p:Person)-[:VISITED]->(c) WITH c, count(p) AS n RETURN n";
    let params: HashMap<String, Value> = HashMap::new();

    let mut fused = parse_cypher(text).unwrap();
    crate::graph::languages::cypher::optimize(&mut fused, &graph, &params);
    let optimized = CypherExecutor::with_params(&graph, &params, None)
        .execute(&fused)
        .unwrap();

    let unoptimized = CypherExecutor::with_params(&graph, &params, None)
        .execute(&parse_cypher(text).unwrap())
        .unwrap();

    assert_eq!(
        unoptimized.rows,
        vec![vec![Value::Int64(2)]],
        "the unoptimised path is the oracle"
    );
    assert_eq!(
        optimized.rows, unoptimized.rows,
        "the planner must not diverge from the unoptimised path"
    );
}
