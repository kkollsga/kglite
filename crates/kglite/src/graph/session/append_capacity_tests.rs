use super::{execute_mut, execute_read, CommitOutcome, ExecuteOptions, Session};
use crate::datatypes::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::storage::mapped::mmap_vec::{heap_append_growths, reset_heap_append_growths};
use std::collections::HashMap;

fn run(graph: &mut DirGraph, query: &str) {
    execute_mut(graph, query, &ExecuteOptions::eager(&HashMap::new())).unwrap();
}
fn exact(graph: &DirGraph, label: &str, count: usize, width: usize) {
    let extras = (1..width).map(|i| format!(",n.p{i}")).collect::<String>();
    let actual = execute_read(
        graph,
        &format!("MATCH (n:{label}) RETURN n.id,n.title,n.city{extras} ORDER BY n.id"),
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
                Value::String(if id < 32 { "seed" } else { "new" }.into()),
            ];
            row.extend((1..width).map(|_| {
                if id < 32 {
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
fn shared_append_does_not_regrow_just_cloned_column_buffers() {
    for width in [1, 16] {
        let mut graph = DirGraph::new();
        let extras = (1..width).map(|i| format!(",p{i}:i")).collect::<String>();
        for label in ["Person", "Other"] {
            run(&mut graph,&format!("UNWIND range(0,31) AS i CREATE (:{label} {{id:i,title:toString(i),city:'seed'{extras}}})"));
        }
        let session = Session::new(graph);
        let old = session.snapshot();
        let mut tx = session.begin();
        for (label, id) in [("Person", 32), ("Person", 33), ("Other", 32)] {
            reset_heap_append_growths();
            run(
                tx.working_mut().unwrap(),
                &format!("CREATE (:{label} {{id:{id},title:'{id}',city:'new'}})"),
            );
            assert_eq!(
                heap_append_growths(),
                0,
                "copy capacity must cover first and second append"
            );
        }
        exact(tx.current().unwrap(), "Person", 34, width);
        exact(tx.current().unwrap(), "Other", 33, width);
        exact(&old, "Person", 32, width);
        exact(&old, "Other", 32, width);
        assert!(matches!(
            session.commit(tx, true),
            CommitOutcome::Committed { .. }
        ));
        exact(&session.snapshot(), "Person", 34, width);
        exact(&old, "Person", 32, width);
    }
}
