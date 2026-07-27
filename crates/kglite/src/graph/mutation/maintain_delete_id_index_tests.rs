//! Regression tests for in-place id-index maintenance on the delete path.
//!
//! Split out of `maintain.rs` to keep that file under the source-quality
//! line ceiling, matching the existing `maintain_edge_spec_tests.rs`.

use super::*;
use crate::graph::session::execute::{execute_mut, ExecuteOptions};

fn run(graph: &mut DirGraph, query: &str) {
    let params = std::collections::HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("setup query failed: {query}: {e}"));
}

/// Three `Person` nodes with distinct ids and a warm id index.
fn seeded_people() -> DirGraph {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "CREATE (:Person {id: 1, name: 'a'}), (:Person {id: 2, name: 'b'}), \
         (:Person {id: 3, name: 'c'})",
    );
    graph.build_id_index("Person");
    assert!(graph.id_indices.contains_key("Person"));
    graph
}

/// Deleting one node must edit the type's id index in place, not drop it.
///
/// Dropping it makes a single-node delete O(N_type): the next id lookup
/// rebuilds the whole map via `compute_id_index`, one node-weight read and
/// `Value` clone per node of the type. That is the delete-side twin of the
/// incremental maintenance the create path already does.
///
/// Asserted through `IdIndexStore::lookup`, the **non-building** read, so
/// the self-healing `lookup_or_build` path cannot mask a dropped index:
/// after a drop the survivor probe returns `None` and this fails.
#[test]
fn delete_edits_id_index_in_place() {
    let mut graph = seeded_people();
    let doomed = graph
        .lookup_by_id_readonly("Person", &Value::Int64(2))
        .expect("id 2 must be indexed");
    let survivor = graph
        .lookup_by_id_readonly("Person", &Value::Int64(3))
        .expect("id 3 must be indexed");

    let mut to_delete = HashSet::new();
    to_delete.insert(doomed);
    assert_eq!(detach_delete_nodes(&mut graph, &to_delete), (1, 0));

    // The index survived the delete: a non-building lookup still resolves
    // an untouched id. This is the assertion the fix exists for.
    assert_eq!(
        graph.id_indices.lookup("Person", &Value::Int64(3)),
        Some(survivor),
        "the delete must leave the id index usable, not force a rebuild"
    );
    // And it is correct: the deleted id is gone, the count dropped by one.
    assert_eq!(graph.id_indices.lookup("Person", &Value::Int64(2)), None);
    assert_eq!(graph.id_indices.overlay_len("Person"), Some(2));
    // The self-healing path agrees with the in-place edit.
    assert!(graph
        .lookup_by_id_readonly("Person", &Value::Int64(2))
        .is_none());
    assert_eq!(
        graph.lookup_by_id_readonly("Person", &Value::Int64(1)),
        graph.id_indices.lookup("Person", &Value::Int64(1))
    );
}

/// Duplicate ids must still force a full rebuild.
///
/// In-place eviction removes exactly the ids it is handed; a rebuild
/// re-derives the map from the survivors. They disagree only when two live
/// nodes share an id: the index holds one of them, and deleting *that* one
/// must leave the other reachable — which only a rebuild achieves.
/// `detach_delete_nodes` detects this in O(1) by comparing the index length
/// against the type's live node count.
///
/// Removing the `evictable` guard makes this fail: the id-1 entry is
/// evicted, the index stays "built", and the shadowed survivor becomes
/// permanently unreachable by id.
#[test]
fn delete_with_duplicate_ids_falls_back_to_rebuild() {
    let mut graph = DirGraph::new();
    // No primary key declared, so duplicate ids are admitted.
    run(&mut graph, "CREATE (:Person {id: 1, name: 'first'})");
    run(&mut graph, "CREATE (:Person {id: 1, name: 'second'})");
    run(&mut graph, "CREATE (:Person {id: 2, name: 'other'})");
    graph.build_id_index("Person");

    // Two nodes share id 1, so the index collapsed them: three nodes, two
    // entries. That inequality is exactly what the guard keys on.
    assert_eq!(graph.type_indices.get("Person").map(|m| m.len()), Some(3));
    assert_eq!(graph.id_indices.overlay_len("Person"), Some(2));

    let indexed = graph
        .lookup_by_id_readonly("Person", &Value::Int64(1))
        .expect("id 1 resolves to one of the two");
    let mut to_delete = HashSet::new();
    to_delete.insert(indexed);
    assert_eq!(detach_delete_nodes(&mut graph, &to_delete), (1, 0));

    // The shadowed duplicate is now the only node with id 1, and must be
    // reachable by id.
    let survivor = graph
        .lookup_by_id_readonly("Person", &Value::Int64(1))
        .expect("the shadowed duplicate must remain reachable by id");
    assert_ne!(survivor, indexed);
    assert!(graph
        .lookup_by_id_readonly("Person", &Value::Int64(2))
        .is_some());
}

/// An unbuilt id index must stay unbuilt.
///
/// Exercised directly against `IdIndexStore`, **not** through
/// `detach_delete_nodes`: the `evictable` guard already excludes any type
/// that is not overlay-resident, so routing this through a delete would
/// never reach `evict_entries` and the test would pass no matter what
/// `evict_entries` did. This is the store-level contract the delete path
/// relies on — a half-populated entry would be trusted as complete by
/// `build_id_index`, which short-circuits whenever an entry exists.
#[test]
fn evicting_an_absent_id_index_does_not_materialize_it() {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "CREATE (:Person {id: 1, name: 'a'}), (:Person {id: 2, name: 'b'})",
    );
    graph.build_id_index("Person");
    let doomed = graph
        .lookup_by_id_readonly("Person", &Value::Int64(1))
        .unwrap();
    graph.id_indices.remove("Person");
    assert!(!graph.id_indices.contains_key("Person"));

    let evicted = graph
        .id_indices
        .evict_entries("Person", &[(Value::Int64(1), doomed)]);

    assert!(!evicted, "an absent index cannot be edited in place");
    assert!(
        !graph.id_indices.contains_key("Person"),
        "an absent index must not be partially materialized by an evict"
    );
    // The later rebuild is therefore complete: both nodes are still live.
    assert!(graph
        .lookup_by_id_readonly("Person", &Value::Int64(2))
        .is_some());
    assert_eq!(graph.id_indices.overlay_len("Person"), Some(2));
}

/// A multi-type delete must edit each affected type's index independently.
#[test]
fn detach_delete_edits_every_affected_type() {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "CREATE (i:Issue {id: 1}), (c:Comment {id: 10}), (c2:Comment {id: 11})",
    );
    run(
        &mut graph,
        "MATCH (i:Issue {id: 1}), (c:Comment {id: 10}) CREATE (c)-[:ON]->(i)",
    );
    run(
        &mut graph,
        "MATCH (i:Issue {id: 1}), (c:Comment {id: 11}) CREATE (c)-[:ON]->(i)",
    );
    assert_eq!(graph.graph.edge_count(), 2);
    graph.build_id_index("Issue");
    graph.build_id_index("Comment");

    let issue = graph
        .lookup_by_id_readonly("Issue", &Value::Int64(1))
        .unwrap();
    let comment = graph
        .lookup_by_id_readonly("Comment", &Value::Int64(10))
        .unwrap();
    let survivor = graph
        .lookup_by_id_readonly("Comment", &Value::Int64(11))
        .unwrap();

    let mut to_delete = HashSet::new();
    to_delete.insert(issue);
    to_delete.insert(comment);
    let (nodes, edges) = detach_delete_nodes(&mut graph, &to_delete);
    assert_eq!((nodes, edges), (2, 2));

    // Both indices survived, both are correct.
    assert_eq!(graph.id_indices.lookup("Issue", &Value::Int64(1)), None);
    assert_eq!(graph.id_indices.overlay_len("Issue"), Some(0));
    assert_eq!(graph.id_indices.lookup("Comment", &Value::Int64(10)), None);
    assert_eq!(
        graph.id_indices.lookup("Comment", &Value::Int64(11)),
        Some(survivor),
        "an untouched sibling must stay indexed without a rebuild"
    );
    assert_eq!(graph.id_indices.overlay_len("Comment"), Some(1));
}
