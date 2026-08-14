//! `detach_delete_nodes` removing its doomed members by position.
//!
//! The store-level mechanics live in
//! `storage/disk/type_index_positional_tests.rs`; these pin what the delete
//! path itself must preserve when it uses them — bucket order (the scan order
//! of an un-`ORDER BY`'d `MATCH`) and the fallback to the full-bucket retain
//! whenever a doomed member cannot be located.

use super::*;
use crate::graph::session::execute::{execute_mut, ExecuteOptions};

fn run(graph: &mut DirGraph, query: &str) {
    let params = std::collections::HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("setup query failed: {query}: {e}"));
}

fn bucket(graph: &DirGraph, node_type: &str) -> Vec<NodeIndex> {
    graph
        .type_indices
        .get(node_type)
        .map(|members| members.to_vec())
        .unwrap_or_default()
}

fn seeded(count: i64) -> DirGraph {
    let mut graph = DirGraph::new();
    for id in 0..count {
        run(&mut graph, &format!("CREATE (:Item {{id: {id}, v: {id}}})"));
    }
    graph.build_id_index("Item");
    graph
}

fn delete(graph: &mut DirGraph, ids: &[i64]) -> (usize, usize) {
    let doomed: HashSet<NodeIndex> = ids
        .iter()
        .map(|id| {
            graph
                .lookup_by_id_readonly("Item", &Value::Int64(*id))
                .unwrap_or_else(|| panic!("id {id} must be indexed"))
        })
        .collect();
    detach_delete_nodes(graph, &doomed)
}

/// Deleting from the middle must leave every survivor in its original slot —
/// a swap-remove would be cheaper and would silently reorder the label scan.
#[test]
fn a_middle_delete_keeps_the_surviving_bucket_order() {
    let mut graph = seeded(6);
    let before = bucket(&graph, "Item");
    let doomed = graph
        .lookup_by_id_readonly("Item", &Value::Int64(2))
        .unwrap();

    assert_eq!(delete(&mut graph, &[2]), (1, 0));

    let expected: Vec<NodeIndex> = before.into_iter().filter(|idx| *idx != doomed).collect();
    assert_eq!(bucket(&graph, "Item"), expected);
}

/// Several members at once, drawn from the head, middle and tail.
#[test]
fn a_multi_delete_keeps_the_surviving_bucket_order() {
    let mut graph = seeded(8);
    let before = bucket(&graph, "Item");
    let doomed: HashSet<NodeIndex> = [0i64, 3, 7]
        .iter()
        .map(|id| {
            graph
                .lookup_by_id_readonly("Item", &Value::Int64(*id))
                .unwrap()
        })
        .collect();

    assert_eq!(delete(&mut graph, &[0, 3, 7]), (3, 0));

    let expected: Vec<NodeIndex> = before
        .into_iter()
        .filter(|idx| !doomed.contains(idx))
        .collect();
    assert_eq!(bucket(&graph, "Item"), expected);
    for id in [1i64, 2, 4, 5, 6] {
        assert!(graph
            .lookup_by_id_readonly("Item", &Value::Int64(id))
            .is_some());
    }
    for id in [0i64, 3, 7] {
        assert!(graph
            .lookup_by_id_readonly("Item", &Value::Int64(id))
            .is_none());
    }
}

/// A bucket that has lost the sortedness invariant — a create reusing a freed
/// slot appends it out of order — must still delete correctly, via the retain.
///
/// The whole positional path is a fast lane over an invariant that a single
/// delete-then-create breaks, so the fallback is not a corner: it is the state
/// every long-lived graph reaches.
#[test]
fn an_out_of_order_bucket_still_deletes_correctly() {
    let mut graph = seeded(3);
    // Free slot 1, then reuse it: the new node is appended last, so the bucket
    // reads [0, 2, 1].
    assert_eq!(delete(&mut graph, &[1]), (1, 0));
    run(&mut graph, "CREATE (:Item {id: 9, v: 9})");
    let reused = graph
        .lookup_by_id_readonly("Item", &Value::Int64(9))
        .unwrap();
    let before = bucket(&graph, "Item");
    assert_eq!(
        before.last(),
        Some(&reused),
        "the reused slot must be appended, not sorted in — otherwise this test \
         is not exercising the fallback it exists for"
    );
    assert!(
        before.windows(2).any(|pair| pair[0] > pair[1]),
        "the bucket must be out of order for this test to mean anything"
    );

    assert_eq!(delete(&mut graph, &[9]), (1, 0));

    let expected: Vec<NodeIndex> = before.into_iter().filter(|idx| *idx != reused).collect();
    assert_eq!(bucket(&graph, "Item"), expected);
    assert!(graph
        .lookup_by_id_readonly("Item", &Value::Int64(9))
        .is_none());
    for id in [0i64, 2] {
        assert!(graph
            .lookup_by_id_readonly("Item", &Value::Int64(id))
            .is_some());
    }
}

/// Deleting every member of a type empties its bucket rather than leaving
/// dangling indices a later scan would materialize.
#[test]
fn deleting_every_member_empties_the_bucket() {
    let mut graph = seeded(4);
    assert_eq!(delete(&mut graph, &[0, 1, 2, 3]), (4, 0));
    assert!(bucket(&graph, "Item").is_empty());
    assert_eq!(graph.graph.node_count(), 0);
}

/// A delete large enough to cross [`POSITIONAL_MAX_SHARE`] hands the bucket
/// back to the retain — and must produce exactly the same bucket.
///
/// The crossover exists because locating `k` members costs `k log N` probes
/// into a bucket far larger than cache: past a share of the type it loses to
/// the single linear pass it replaces. This pins that the choice is a cost
/// decision only, invisible in the result.
#[test]
fn a_mass_delete_falls_back_to_the_retain_with_the_same_result() {
    let mut graph = DirGraph::new();
    let rows: Vec<Vec<Value>> = (0..2_000i64).map(|id| vec![Value::Int64(id)]).collect();
    let frame = DataFrame::from_cypher_rows(vec!["id".to_string()], rows).unwrap();
    add_nodes(
        &mut graph,
        frame,
        "Item".to_string(),
        "id".to_string(),
        Some("id".to_string()),
        None,
    )
    .unwrap();
    let before = bucket(&graph, "Item");
    assert!(before.len() > POSITIONAL_MIN_BUCKET);

    // 100 of 2000 — above 1/32nd, so the positional path is declined.
    let doomed_ids: Vec<i64> = (0..100).map(|i| i * 20).collect();
    let doomed: HashSet<NodeIndex> = doomed_ids
        .iter()
        .map(|id| {
            graph
                .lookup_by_id_readonly("Item", &Value::Int64(*id))
                .unwrap()
        })
        .collect();
    assert!(
        doomed.len().saturating_mul(POSITIONAL_MAX_SHARE) > before.len(),
        "this delete must be above the crossover or it tests the other arm"
    );

    assert_eq!(detach_delete_nodes(&mut graph, &doomed), (100, 0));

    let expected: Vec<NodeIndex> = before
        .into_iter()
        .filter(|idx| !doomed.contains(idx))
        .collect();
    assert_eq!(bucket(&graph, "Item"), expected);
    assert!(graph
        .lookup_by_id_readonly("Item", &Value::Int64(0))
        .is_none());
    assert!(graph
        .lookup_by_id_readonly("Item", &Value::Int64(1))
        .is_some());
}

/// The delete must reach *every* affected type's bucket, not just the first.
#[test]
fn a_multi_type_delete_edits_each_bucket_independently() {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "CREATE (:Item {id: 1}), (:Item {id: 2}), (:Tag {id: 10}), (:Tag {id: 11})",
    );
    graph.build_id_index("Item");
    graph.build_id_index("Tag");
    let item = graph
        .lookup_by_id_readonly("Item", &Value::Int64(1))
        .unwrap();
    let tag = graph
        .lookup_by_id_readonly("Tag", &Value::Int64(11))
        .unwrap();
    let item_survivor = graph
        .lookup_by_id_readonly("Item", &Value::Int64(2))
        .unwrap();
    let tag_survivor = graph
        .lookup_by_id_readonly("Tag", &Value::Int64(10))
        .unwrap();

    let doomed: HashSet<NodeIndex> = [item, tag].into_iter().collect();
    assert_eq!(detach_delete_nodes(&mut graph, &doomed), (2, 0));

    assert_eq!(bucket(&graph, "Item"), vec![item_survivor]);
    assert_eq!(bucket(&graph, "Tag"), vec![tag_survivor]);
}
