//! Relationship embedding carry: identity by endpoint ids, parallel groups by
//! a named key, refusals by name, `.kgle` v4 round trip and durable import.

use std::collections::HashMap;
use std::sync::Arc;

use super::*;
use crate::graph::edge_embeddings::edge_store_key;
use crate::graph::io::file::{
    export_embeddings_to_file, import_embeddings_from_file, load_file, save_graph,
};
use crate::graph::session::execute::{execute_mut, ExecuteOptions};
use crate::graph::session::{CommitOutcome, Session};
use crate::graph::wal::DurabilityLevel;

fn run(graph: &mut DirGraph, query: &str) {
    let params = HashMap::new();
    execute_mut(graph, query, &ExecuteOptions::eager(&params))
        .unwrap_or_else(|e| panic!("query failed: {query}: {e}"));
}

/// Docs 1, 2, 3; a singleton `CLAIMS` 1→2 (uid 's') and a parallel group of
/// two `CLAIMS` 1→3 (uids 'p1', 'p2'). `swapped` creates the parallel members
/// in the opposite order, so the target's slot order disagrees with the
/// source's — the pre-mortem's misattachment shape.
fn graph(swapped: bool, uids: [&str; 2]) -> DirGraph {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "CREATE (:Doc {id: 1}), (:Doc {id: 2}), (:Doc {id: 3})",
    );
    run(
        &mut graph,
        "MATCH (a:Doc {id: 1}), (b:Doc {id: 2}) CREATE (a)-[:CLAIMS {uid: 's', text: 'single'}]->(b)",
    );
    let order = if swapped { [1, 0] } else { [0, 1] };
    for i in order {
        run(
            &mut graph,
            &format!(
                "MATCH (a:Doc {{id: 1}}), (c:Doc {{id: 3}}) \
                 CREATE (a)-[:CLAIMS {{uid: '{}', text: 't{i}'}}]->(c)",
                uids[i]
            ),
        );
    }
    graph
}

fn edge_by_uid(graph: &DirGraph, uid: &str) -> EdgeIndex {
    graph
        .graph
        .edge_indices()
        .find(|edge| {
            graph
                .graph
                .edge_weight(*edge)
                .as_deref()
                .and_then(|w| w.get_property("uid"))
                == Some(&Value::String(uid.into()))
        })
        .unwrap_or_else(|| panic!("no CLAIMS edge with uid {uid}"))
}

/// A source store with a distinct vector, text hash per member, and a model id.
fn source() -> DirGraph {
    let mut graph = graph(false, ["p1", "p2"]);
    let mut store = EdgeEmbeddingStore::new(2, Some("cosine"));
    for (uid, vector, hash) in [
        ("s", [1.0, 0.0], 11),
        ("p1", [0.0, 1.0], 22),
        ("p2", [0.6, 0.8], 33),
    ] {
        store.install_wal_vector(edge_by_uid(&graph, uid), &vector, Some(hash));
    }
    store.set_model_id(Some("stub/model".into()));
    graph
        .edge_embeddings
        .insert(edge_store_key("CLAIMS", "text"), store);
    graph
}

fn keys() -> RelationshipKeys {
    HashMap::from([("CLAIMS".to_string(), "uid".to_string())])
}

/// `(uid → (vector, hash))` for every stored member of the target store.
fn cells(graph: &DirGraph) -> Vec<(String, Vec<f32>, Option<u64>)> {
    let store = &graph.edge_embeddings[&edge_store_key("CLAIMS", "text")];
    let mut out: Vec<_> = store
        .edges()
        .map(|edge| {
            let uid = match graph.graph.edge_weight(edge).unwrap().get_property("uid") {
                Some(Value::String(uid)) => uid.clone(),
                other => panic!("uid: {other:?}"),
            };
            (
                uid,
                store.get(edge).unwrap().to_vec(),
                store.text_hash(edge),
            )
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn expected_cells() -> Vec<(String, Vec<f32>, Option<u64>)> {
    vec![
        ("p1".into(), vec![0.0, 1.0], Some(22)),
        ("p2".into(), vec![0.6, 0.8], Some(33)),
        ("s".into(), vec![1.0, 0.0], Some(11)),
    ]
}

#[test]
fn a_parallel_group_without_a_key_is_refused_by_name() {
    let error = extract_edge_stores(&source(), &RelationshipKeys::new()).unwrap_err();
    assert!(error.contains("'CLAIMS.text'"), "{error}");
    assert!(
        error.contains("2 'CLAIMS' relationships connect (Doc id=1) to (Doc id=3)"),
        "{error}"
    );
    assert!(error.contains("relationship_keys"), "{error}");
}

#[test]
fn a_key_carries_each_member_to_its_twin_despite_swapped_order() {
    let src = source();
    let mut dst = graph(true, ["p1", "p2"]);
    // The swap is real: the first-created member differs between the graphs.
    assert_ne!(edge_by_uid(&src, "p1"), edge_by_uid(&dst, "p1"));

    let report = dst
        .copy_embeddings_with_relationships_from(&src, &keys())
        .unwrap();
    assert_eq!(
        report.relationships,
        RelationshipCarryStats {
            stores: 1,
            carried: 3,
            skipped: 0,
            dropped_stores: 0,
            installed: vec![("CLAIMS".to_string(), "text".to_string())],
        }
    );
    assert_eq!(cells(&dst), expected_cells());
    let store = &dst.edge_embeddings[&edge_store_key("CLAIMS", "text")];
    assert_eq!(store.model_id(), Some("stub/model"));
    assert_eq!(store.metric(), Some("cosine"));
}

#[test]
fn an_ambiguous_target_group_is_refused_before_anything_is_written() {
    let mut src = source();
    crate::graph::embeddings::set_embeddings(
        &mut src,
        "Doc",
        "title",
        None,
        [(Value::Int64(1), vec![1.0, 0.0])],
    )
    .unwrap();
    // The target group repeats a key value, so the key cannot choose a member.
    let mut dst = graph(false, ["p1", "p1"]);
    let error = dst
        .copy_embeddings_with_relationships_from(&src, &keys())
        .unwrap_err();
    assert!(
        error.contains("2 'CLAIMS' relationships connect (Doc id=1) to (Doc id=3)"),
        "{error}"
    );
    assert!(error.contains("repeats the value"), "{error}");
    assert!(dst.edge_embeddings.is_empty());
    assert!(
        dst.embeddings.is_empty(),
        "no node store may be copied either"
    );
}

#[test]
fn a_keyless_vector_meeting_a_target_parallel_group_is_refused() {
    // Source: one CLAIMS 1→3 (a singleton, carried without a key).
    let mut src = DirGraph::new();
    run(&mut src, "CREATE (:Doc {id: 1}), (:Doc {id: 3})");
    run(
        &mut src,
        "MATCH (a:Doc {id: 1}), (c:Doc {id: 3}) CREATE (a)-[:CLAIMS {uid: 'p1'}]->(c)",
    );
    let only = edge_by_uid(&src, "p1");
    crate::graph::edge_embeddings::upsert_edge_embeddings(
        &mut src,
        "CLAIMS",
        "text",
        vec![(only, vec![1.0, 0.0])],
        None,
    )
    .unwrap();
    let mut dst = graph(false, ["p1", "p2"]);
    let error = dst
        .copy_embeddings_with_relationships_from(&src, &RelationshipKeys::new())
        .unwrap_err();
    assert!(error.contains("names no key value"), "{error}");
    assert!(dst.edge_embeddings.is_empty());
}

#[test]
fn relationships_the_target_lacks_are_skipped_and_an_unmatched_store_dropped() {
    let src = source();
    let mut dst = DirGraph::new();
    run(&mut dst, "CREATE (:Doc {id: 1}), (:Doc {id: 2})");
    run(
        &mut dst,
        "MATCH (a:Doc {id: 1}), (b:Doc {id: 2}) CREATE (a)-[:CLAIMS {uid: 's'}]->(b)",
    );
    let report = dst
        .copy_embeddings_with_relationships_from(&src, &keys())
        .unwrap();
    assert_eq!(
        (report.relationships.carried, report.relationships.skipped),
        (1, 2)
    );

    let mut empty = DirGraph::new();
    run(&mut empty, "CREATE (:Doc {id: 9})");
    let report = empty
        .copy_embeddings_with_relationships_from(&src, &keys())
        .unwrap();
    assert_eq!(
        report.relationships,
        RelationshipCarryStats {
            stores: 0,
            carried: 0,
            skipped: 3,
            dropped_stores: 1,
            installed: Vec::new(),
        }
    );
    assert!(empty.edge_embeddings.is_empty());
}

#[test]
fn a_kgle_round_trip_carries_the_store_and_its_provenance() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("e.kgle").to_string_lossy().into_owned();
    let stats = export_embeddings_to_file(&source(), &path, None, &keys()).unwrap();
    assert_eq!(
        (stats.relationship_stores, stats.relationship_embeddings),
        (1, 3)
    );
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 4);

    // The key travels in the file, so the import names none.
    let mut dst = graph(true, ["p1", "p2"]);
    let imported = import_embeddings_from_file(&mut dst, &path, &RelationshipKeys::new()).unwrap();
    assert_eq!(
        (
            imported.relationships.stores,
            imported.relationships.carried
        ),
        (1, 3)
    );
    assert_eq!(cells(&dst), expected_cells());
}

#[test]
fn an_export_that_meets_an_unkeyed_parallel_group_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("e.kgle");
    let error = export_embeddings_to_file(
        &source(),
        path.to_str().unwrap(),
        None,
        &RelationshipKeys::new(),
    )
    .err()
    .expect("refused");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    assert!(!path.exists(), "a refused export must not leave a file");
}

/// A durable graph journals the import: a session that commits it and then
/// ends without a checkpoint recovers the vectors from the WAL alone.
#[test]
fn a_durable_import_survives_a_crash_shaped_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let kgle = dir.path().join("e.kgle").to_string_lossy().into_owned();
    export_embeddings_to_file(&source(), &kgle, None, &keys()).unwrap();
    let path = dir.path().join("app.kgl").to_string_lossy().into_owned();
    {
        let mut checkpoint = Arc::new(graph(true, ["p1", "p2"]));
        save_graph(&mut checkpoint, &path).expect("checkpoint");
    }
    {
        let session =
            Session::open_durable(load_file(&path).unwrap(), &path, DurabilityLevel::Full).unwrap();
        let mut tx = session.begin();
        import_embeddings_from_file(
            tx.working_mut().expect("writable"),
            &kgle,
            &RelationshipKeys::new(),
        )
        .unwrap();
        match session.commit(tx, true) {
            CommitOutcome::Committed { .. } => {}
            other => panic!("commit refused: {other:?}"),
        }
    }
    let recovered =
        Session::open_durable(load_file(&path).unwrap(), &path, DurabilityLevel::Full).unwrap();
    let snapshot = recovered.snapshot();
    assert_eq!(cells(&snapshot), expected_cells());
    assert_eq!(
        snapshot.edge_embeddings[&edge_store_key("CLAIMS", "text")].model_id(),
        Some("stub/model")
    );
}
