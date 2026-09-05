//! Structural attribution of first versus repeated append ownership.
//! These counters count copies, not elapsed time or allocated bytes.
use super::{execute_mut, execute_read, ExecuteOptions, Session};
use crate::datatypes::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::storage::column_store::{
    column_clones, column_store_clones, column_store_row_pushes, reset_column_clones,
    reset_column_store_clones, reset_column_store_row_pushes,
};
use std::collections::HashMap;
use std::sync::Arc;

fn execute(graph: &mut DirGraph, query: &str) {
    execute_mut(graph, query, &ExecuteOptions::eager(&HashMap::new())).unwrap();
}

fn seed(width: usize) -> DirGraph {
    let mut graph = DirGraph::new();
    let extra = (1..width).map(|i| format!(", p{i}:i")).collect::<String>();
    for label in ["Person", "Other"] {
        execute(&mut graph,&format!("UNWIND range(0,7) AS i CREATE (:{label} {{id:i,title:toString(i),city:'seed'{extra}}})"));
    }
    graph
}

fn append(graph: &mut DirGraph, label: &str, id: usize) -> (usize, usize, usize) {
    reset_column_clones();
    reset_column_store_clones();
    reset_column_store_row_pushes();
    execute(
        graph,
        &format!("CREATE (:{label} {{id:{id},title:'{id}',city:'new'}})"),
    );
    (
        column_store_clones(),
        column_clones(),
        column_store_row_pushes(),
    )
}

fn rows(graph: &DirGraph, label: &str, width: usize, count: usize) {
    let extra = (1..width).map(|i| format!(",n.p{i}")).collect::<String>();
    let actual = execute_read(
        graph,
        &format!("MATCH (n:{label}) RETURN n.id,n.title,n.city{extra} ORDER BY n.id"),
        &ExecuteOptions::eager(&HashMap::new()),
    )
    .unwrap()
    .result
    .rows;
    let expected: Vec<Vec<Value>> = (0..count)
        .map(|id| {
            let mut row = vec![
                Value::Int64(id as i64),
                Value::String(id.to_string()),
                Value::String(if id < 8 { "seed" } else { "new" }.into()),
            ];
            row.extend((1..width).map(|_| {
                if id < 8 {
                    Value::Int64(id as i64)
                } else {
                    Value::Null
                }
            }));
            row
        })
        .collect();
    assert_eq!(actual, expected);
}

#[test]
fn transaction_append_copy_cost_resets_per_populated_type() {
    for width in [1, 4, 16] {
        let session = Session::new(seed(width));
        let mut tx = session.begin();
        let working = tx.working_mut().unwrap();
        assert_eq!(
            Arc::strong_count(working.column_store("Person").unwrap()),
            2
        );
        // Literal title also remains a property; the two identity sidecars
        // are additional to city, title and the width-1 extra properties.
        assert_eq!(
            working.column_store("Person").unwrap().column_count(),
            width + 1
        );
        let first = append(working, "Person", 8);
        assert_eq!(
            first,
            (1, width + 3, 1),
            "first append clones the store and each property/identity column"
        );
        assert_eq!(
            Arc::strong_count(working.column_store("Person").unwrap()),
            1
        );
        let second = append(working, "Person", 9);
        assert_eq!(second, (0, 0, 1));
        let other = append(working, "Other", 8);
        assert_eq!(other, (1, width + 3, 1));
        let empty = append(working, "Empty", 8);
        assert_eq!(empty, (0, 0, 1));
        rows(working, "Person", width, 10);
        rows(working, "Other", width, 9);
        rows(&session.snapshot(), "Person", width, 8);
        rows(&session.snapshot(), "Other", width, 8);
        println!("COPY_PROBE width={width} first={first:?} second={second:?} other={other:?} empty={empty:?}");
    }
}

#[test]
fn owned_append_reuses_existing_column_buffers() {
    for width in [1, 4, 16] {
        let mut graph = seed(width);
        assert_eq!(Arc::strong_count(graph.column_store("Person").unwrap()), 1);
        for (label, id) in [("Person", 8), ("Person", 9), ("Other", 8), ("Empty", 8)] {
            assert_eq!(append(&mut graph, label, id), (0, 0, 1));
        }
        rows(&graph, "Person", width, 10);
        rows(&graph, "Other", width, 9);
    }
}
