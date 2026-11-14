use super::vector_index::*;
use super::*;
use crate::datatypes::Value;
use crate::graph::embedding_hints::Surface;
use crate::graph::schema::{EdgeData, NodeData};
use crate::graph::storage::GraphWrite;
use std::collections::HashMap;

fn fixture() -> (DirGraph, EdgeIndex, EdgeIndex, EdgeIndex) {
    let mut graph = DirGraph::new();
    let source = GraphWrite::add_node(
        &mut graph.graph,
        NodeData::new(
            Value::Int64(1),
            Value::String("source".into()),
            "Doc".into(),
            HashMap::new(),
            &mut graph.interner,
        ),
    );
    let mut edges = Vec::new();
    for ordinal in 0..3 {
        let target = GraphWrite::add_node(
            &mut graph.graph,
            NodeData::new(
                Value::Int64(ordinal + 2),
                Value::String(format!("target-{ordinal}")),
                "Doc".into(),
                HashMap::new(),
                &mut graph.interner,
            ),
        );
        edges.push(GraphWrite::add_edge(
            &mut graph.graph,
            source,
            target,
            EdgeData::new("CLAIMS".into(), HashMap::new(), &mut graph.interner),
        ));
    }
    upsert_edge_embeddings(
        &mut graph,
        "CLAIMS",
        "text",
        vec![
            (edges[0], vec![1.0, 0.0]),
            (edges[1], vec![0.0, 1.0]),
            (edges[2], vec![-1.0, 0.0]),
        ],
        Some("cosine"),
    )
    .unwrap();
    (graph, edges[0], edges[1], edges[2])
}

#[test]
fn explicit_index_query_matches_exact_and_status_uses_text_property() {
    let (mut graph, first, _, _) = fixture();
    let exact = query_edge_embeddings(
        &graph,
        "CLAIMS",
        "text",
        &[1.0, 0.0],
        EdgeVectorQueryOptions {
            top_k: 2,
            exact: true,
            metric: None,
        },
    )
    .unwrap();
    assert_eq!(exact.hits[0].edge, first);
    assert_eq!(exact.search_method, "exact");

    build_edge_vector_index(
        &mut graph,
        "CLAIMS",
        "text",
        EdgeVectorIndexOptions::default(),
    )
    .unwrap();
    let approximate = query_edge_embeddings(
        &graph,
        "CLAIMS",
        "text",
        &[1.0, 0.0],
        EdgeVectorQueryOptions {
            top_k: 2,
            exact: false,
            metric: None,
        },
    )
    .unwrap();
    assert_eq!(approximate.hits, exact.hits);
    assert_eq!(approximate.search_method, "hnsw");
    assert_eq!(
        list_edge_vector_indexes(&graph),
        vec![EdgeVectorIndexStatus {
            connection_type: "CLAIMS".into(),
            text_property: "text".into(),
            built: true,
            stale: false,
            delta: 0,
            unembedded: 0,
        }]
    );
}

#[test]
fn unsupported_build_is_atomic_and_metric_mismatch_falls_back() {
    let (mut graph, _, _, _) = fixture();
    let before = graph.version();
    let error = build_edge_vector_index(
        &mut graph,
        "CLAIMS",
        "text",
        EdgeVectorIndexOptions {
            metric: Some("poincare".into()),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(error.contains("not supported by HNSW"));
    assert_eq!(graph.version(), before);
    assert!(!list_edge_vector_indexes(&graph)[0].built);

    let error = build_edge_vector_index(
        &mut graph,
        "CLAIMS",
        "text",
        EdgeVectorIndexOptions {
            m: Some(1),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(error.contains("at least 2"));
    assert_eq!(graph.version(), before);
    assert!(!list_edge_vector_indexes(&graph)[0].built);
}

#[test]
fn build_and_drop_roll_back_without_rebuilding_vectors() {
    let (mut graph, _, _, _) = fixture();
    let checkpoint = crate::graph::dir_graph::rollback::StatementCheckpoint::open(&mut graph);
    build_edge_vector_index(
        &mut graph,
        "CLAIMS",
        "text",
        EdgeVectorIndexOptions::default(),
    )
    .unwrap();
    assert!(list_edge_vector_indexes(&graph)[0].built);
    checkpoint.rollback(&mut graph);
    assert!(!list_edge_vector_indexes(&graph)[0].built);

    build_edge_vector_index(
        &mut graph,
        "CLAIMS",
        "text",
        EdgeVectorIndexOptions::default(),
    )
    .unwrap();
    let checkpoint = crate::graph::dir_graph::rollback::StatementCheckpoint::open(&mut graph);
    assert!(drop_edge_vector_index(&mut graph, "CLAIMS", "text").unwrap());
    assert!(!list_edge_vector_indexes(&graph)[0].built);
    checkpoint.rollback(&mut graph);
    assert!(list_edge_vector_indexes(&graph)[0].built);
}

#[test]
fn vector_mutation_marks_index_stale_and_exact_fallback_remains_correct() {
    let (mut graph, first, second, _) = fixture();
    build_edge_vector_index(
        &mut graph,
        "CLAIMS",
        "text",
        EdgeVectorIndexOptions {
            auto_refresh_limit: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    upsert_edge_embeddings(
        &mut graph,
        "CLAIMS",
        "text",
        vec![(second, vec![1.0, 0.0])],
        None,
    )
    .unwrap();
    let status = &list_edge_vector_indexes(&graph)[0];
    assert!(status.built);
    assert!(status.stale);
    assert_eq!(status.delta, 1);
    let result = query_edge_embeddings(
        &graph,
        "CLAIMS",
        "text",
        &[1.0, 0.0],
        EdgeVectorQueryOptions {
            top_k: 2,
            exact: false,
            metric: None,
        },
    )
    .unwrap();
    assert_eq!(result.search_method, "exact");
    assert_eq!(
        result.hits.iter().map(|hit| hit.edge).collect::<Vec<_>>(),
        vec![first, second]
    );
}

#[test]
fn wal_declaration_rebuilds_and_drops_after_vectors_are_present() {
    let (mut recovered, _, _, _) = fixture();
    crate::graph::mutation::wal_replay::apply_frames(
        &mut recovered,
        &[crate::graph::wal::WalFrame {
            lsn: 1,
            ops: vec![crate::graph::wal::MutationOp::SetEdgeVectorIndex {
                conn_type: "CLAIMS".into(),
                text_column: "text".into(),
                metric: Some("cosine".into()),
                m: Some(8),
                ef_construction: Some(32),
                ef_search: Some(16),
                auto_refresh_limit: Some(4),
                present: true,
            }],
        }],
        0,
    )
    .unwrap();
    assert!(list_edge_vector_indexes(&recovered)[0].built);

    crate::graph::mutation::wal_replay::apply_frames(
        &mut recovered,
        &[crate::graph::wal::WalFrame {
            lsn: 2,
            ops: vec![crate::graph::wal::MutationOp::SetEdgeVectorIndex {
                conn_type: "CLAIMS".into(),
                text_column: "text".into(),
                metric: None,
                m: None,
                ef_construction: None,
                ef_search: None,
                auto_refresh_limit: None,
                present: false,
            }],
        }],
        1,
    )
    .unwrap();
    assert!(!list_edge_vector_indexes(&recovered)[0].built);
}

#[test]
fn malformed_wal_index_options_refuse_without_installing_an_index() {
    let (mut recovered, _, _, _) = fixture();
    let result = crate::graph::mutation::wal_replay::apply_frames(
        &mut recovered,
        &[crate::graph::wal::WalFrame {
            lsn: 1,
            ops: vec![crate::graph::wal::MutationOp::SetEdgeVectorIndex {
                conn_type: "CLAIMS".into(),
                text_column: "text".into(),
                metric: Some("cosine".into()),
                m: Some(8),
                ef_construction: Some(0),
                ef_search: Some(16),
                auto_refresh_limit: Some(4),
                present: true,
            }],
        }],
        0,
    );
    assert!(result.unwrap_err().contains("greater than 0"));
    assert!(!list_edge_vector_indexes(&recovered)[0].built);
}

/// Same shape as [`fixture`], but the store is created without a declared
/// metric — the state `db.relationship_embeddings.set` leaves behind when the caller
/// names no metric, and the one an explicit build metric may claim.
fn fixture_without_declared_metric() -> (DirGraph, EdgeIndex) {
    let mut graph = DirGraph::new();
    let source = GraphWrite::add_node(
        &mut graph.graph,
        NodeData::new(
            Value::Int64(1),
            Value::String("source".into()),
            "Doc".into(),
            HashMap::new(),
            &mut graph.interner,
        ),
    );
    let mut edges = Vec::new();
    for ordinal in 0..3 {
        let target = GraphWrite::add_node(
            &mut graph.graph,
            NodeData::new(
                Value::Int64(ordinal + 2),
                Value::String(format!("target-{ordinal}")),
                "Doc".into(),
                HashMap::new(),
                &mut graph.interner,
            ),
        );
        edges.push(GraphWrite::add_edge(
            &mut graph.graph,
            source,
            target,
            EdgeData::new("CLAIMS".into(), HashMap::new(), &mut graph.interner),
        ));
    }
    upsert_edge_embeddings(
        &mut graph,
        "CLAIMS",
        "text",
        vec![
            (edges[0], vec![3.0, 0.0]),
            (edges[1], vec![0.0, 1.0]),
            (edges[2], vec![-1.0, 0.0]),
        ],
        None,
    )
    .unwrap();
    (graph, edges[0])
}

/// A build metric the store does not contradict becomes the store's metric.
///
/// Before the fix the store kept its `None` (which resolves to cosine) beside a
/// euclidean index, so every metric-less query resolved cosine, failed the
/// index's metric check and fell back to the exact scan for good — while
/// `list` reported a metric nothing used.
#[test]
fn explicit_build_metric_becomes_the_store_metric_and_serves_default_queries() {
    let (mut graph, _) = fixture_without_declared_metric();
    let report = build_edge_vector_index(
        &mut graph,
        "CLAIMS",
        "text",
        EdgeVectorIndexOptions {
            metric: Some("euclidean".into()),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(report.metric, "euclidean");

    let store = graph
        .edge_embeddings
        .get(&("CLAIMS".to_string(), "text_emb".to_string()))
        .unwrap();
    assert_eq!(store.metric(), Some("euclidean"));

    let served = query_edge_embeddings(
        &graph,
        "CLAIMS",
        "text",
        &[3.0, 0.0],
        EdgeVectorQueryOptions {
            top_k: 2,
            exact: false,
            metric: None,
        },
    )
    .unwrap();
    assert_eq!(served.search_method, "hnsw");
}

/// The store's declared metric wins over a contradicting build argument, and
/// says so: the vectors were written under it, and an index answering under
/// another one is the defect above wearing an explicit store metric.
#[test]
fn build_metric_contradicting_the_store_is_refused_and_changes_nothing() {
    let (mut graph, _, _, _) = fixture();
    let before = graph.version();
    let error = build_edge_vector_index(
        &mut graph,
        "CLAIMS",
        "text",
        EdgeVectorIndexOptions {
            metric: Some("euclidean".into()),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(error.contains("declares metric 'cosine'"), "{error}");
    assert!(error.contains("requested 'euclidean'"), "{error}");
    assert_eq!(graph.version(), before);
    assert!(!list_edge_vector_indexes(&graph)[0].built);
    assert_eq!(
        graph
            .edge_embeddings
            .get(&("CLAIMS".to_string(), "text_emb".to_string()))
            .unwrap()
            .metric(),
        Some("cosine")
    );
}

/// The metric a build persists is statement state like any other: a rolled-back
/// statement leaves the store scoring the way it did before.
#[test]
fn rolled_back_build_restores_the_stores_prior_metric() {
    let (mut graph, _) = fixture_without_declared_metric();
    let checkpoint = crate::graph::dir_graph::rollback::StatementCheckpoint::open(&mut graph);
    build_edge_vector_index(
        &mut graph,
        "CLAIMS",
        "text",
        EdgeVectorIndexOptions {
            metric: Some("euclidean".into()),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        graph
            .edge_embeddings
            .get(&("CLAIMS".to_string(), "text_emb".to_string()))
            .unwrap()
            .metric(),
        Some("euclidean")
    );
    checkpoint.rollback(&mut graph);
    assert_eq!(
        graph
            .edge_embeddings
            .get(&("CLAIMS".to_string(), "text_emb".to_string()))
            .unwrap()
            .metric(),
        None
    );
    assert!(!list_edge_vector_indexes(&graph)[0].built);
}

/// Recovery reproduces the store the build left behind, metric included — so a
/// reopened graph does not go back to resolving cosine against a euclidean
/// index.
#[test]
fn wal_replay_restores_the_metric_the_index_was_built_for() {
    let (mut recovered, _) = fixture_without_declared_metric();
    crate::graph::mutation::wal_replay::apply_frames(
        &mut recovered,
        &[crate::graph::wal::WalFrame {
            lsn: 1,
            ops: vec![crate::graph::wal::MutationOp::SetEdgeVectorIndex {
                conn_type: "CLAIMS".into(),
                text_column: "text".into(),
                metric: Some("euclidean".into()),
                m: Some(8),
                ef_construction: Some(32),
                ef_search: Some(16),
                auto_refresh_limit: Some(4),
                present: true,
            }],
        }],
        0,
    )
    .unwrap();
    assert_eq!(
        recovered
            .edge_embeddings
            .get(&("CLAIMS".to_string(), "text_emb".to_string()))
            .unwrap()
            .metric(),
        Some("euclidean")
    );
    let served = query_edge_embeddings(
        &recovered,
        "CLAIMS",
        "text",
        &[3.0, 0.0],
        EdgeVectorQueryOptions {
            top_k: 2,
            exact: false,
            metric: None,
        },
    )
    .unwrap();
    assert_eq!(served.search_method, "hnsw");
}

/// A refresh with no index to refresh refuses, naming the store and the build
/// call. It answered `0` — "nothing outstanding" — which is exactly what an
/// agent reads after a relationship delete has dropped the index to `none`.
#[test]
fn refresh_without_an_index_refuses_and_names_the_build_call() {
    let (mut graph, _, _, _) = fixture();
    let error = refresh_edge_vector_index(&graph, "CLAIMS", "text", Surface::Cypher).unwrap_err();
    assert!(error.contains("'CLAIMS.text_emb'"), "{error}");
    assert!(
        error.contains(
            "db.relationship_embeddings.build_index({type: 'CLAIMS', text_column: 'text'})"
        ),
        "{error}"
    );

    build_edge_vector_index(
        &mut graph,
        "CLAIMS",
        "text",
        EdgeVectorIndexOptions::default(),
    )
    .unwrap();
    assert_eq!(
        refresh_edge_vector_index(&graph, "CLAIMS", "text", Surface::Cypher),
        Ok(0)
    );
    assert!(drop_edge_vector_index(&mut graph, "CLAIMS", "text").unwrap());
    assert!(refresh_edge_vector_index(&graph, "CLAIMS", "text", Surface::Cypher).is_err());

    // Read-only does not turn the refusal back into a silent zero.
    graph.read_only = true;
    assert!(refresh_edge_vector_index(&graph, "CLAIMS", "text", Surface::Cypher).is_err());
}

/// `fixture()` plus two `SUPPORTS` relationships in their own store: one whose
/// vector ties exactly with `CLAIMS`' first, one in between.
fn two_type_fixture(supports_metric: Option<&str>) -> (DirGraph, EdgeIndex, EdgeIndex) {
    let (mut graph, first, _, _) = fixture();
    let source = graph.graph.edge_endpoints(first).unwrap().0;
    let mut supports = Vec::new();
    for _ in 0..2 {
        let target = GraphWrite::add_node(
            &mut graph.graph,
            NodeData::new(
                Value::Int64(90 + supports.len() as i64),
                Value::String("support".into()),
                "Doc".into(),
                HashMap::new(),
                &mut graph.interner,
            ),
        );
        supports.push(GraphWrite::add_edge(
            &mut graph.graph,
            source,
            target,
            EdgeData::new("SUPPORTS".into(), HashMap::new(), &mut graph.interner),
        ));
    }
    upsert_edge_embeddings(
        &mut graph,
        "SUPPORTS",
        "text",
        vec![(supports[0], vec![1.0, 0.0]), (supports[1], vec![0.8, 0.6])],
        supports_metric,
    )
    .unwrap();
    (graph, first, supports[1])
}

fn stores_query(
    graph: &DirGraph,
    types: &[&str],
    metric: Option<&str>,
    top_k: usize,
) -> Result<Vec<EdgeStoreQueryHit>, String> {
    let types: Vec<String> = types.iter().map(|t| t.to_string()).collect();
    query_edge_embedding_stores(
        graph,
        &types,
        "text",
        &[1.0, 0.0],
        EdgeVectorQueryOptions {
            top_k,
            exact: false,
            metric: metric.map(str::to_string),
        },
        Surface::Cypher,
        None,
    )
}

#[test]
fn several_stores_merge_into_one_top_k_ordered_by_score_type_then_slot() {
    let (graph, claims_first, supports_between) = two_type_fixture(Some("cosine"));
    let hits = stores_query(&graph, &["CLAIMS", "SUPPORTS"], None, 3).unwrap();
    let summary: Vec<(&str, f64)> = hits
        .iter()
        .map(|hit| (hit.rel_type.as_str(), (hit.score * 100.0).round() / 100.0))
        .collect();
    // The 1.0 tie is broken by type name, then the 0.8 SUPPORTS hit.
    assert_eq!(
        summary,
        vec![("CLAIMS", 1.0), ("SUPPORTS", 1.0), ("SUPPORTS", 0.8)]
    );
    assert_eq!(hits[0].edge, claims_first);
    assert_eq!(hits[2].edge, supports_between);
    assert!(hits.iter().all(|hit| hit.search_method == "exact"));
    // One type is exactly the single-store answer.
    let single = stores_query(&graph, &["CLAIMS"], None, 2).unwrap();
    assert!(single.iter().all(|hit| hit.rel_type == "CLAIMS"));
    assert_eq!(single.len(), 2);
}

#[test]
fn a_merge_across_metrics_or_a_missing_store_is_refused_by_name() {
    let (graph, _, _) = two_type_fixture(Some("euclidean"));
    let error = stores_query(&graph, &["CLAIMS", "SUPPORTS"], None, 3).unwrap_err();
    assert!(
        error.contains("'CLAIMS.text' (metric 'cosine')")
            && error.contains("'SUPPORTS.text' (metric 'euclidean')"),
        "{error}"
    );
    // One metric named for every store puts the scores on one scale.
    assert_eq!(
        stores_query(&graph, &["CLAIMS", "SUPPORTS"], Some("cosine"), 3)
            .unwrap()
            .len(),
        3
    );
    let error = stores_query(&graph, &["CLAIMS", "MISSING"], None, 3).unwrap_err();
    assert!(
        error.contains("No relationship embedding store 'MISSING.text'"),
        "{error}"
    );
}

#[test]
fn relationship_embeddings_reads_the_store_by_endpoint_ids_in_a_stable_order() {
    use crate::graph::edge_embeddings::carry::{relationship_embeddings, RelationshipKeys};
    let (mut graph, first, _, _) = fixture();
    let (source, target) = graph.graph.edge_endpoints(first).unwrap();
    // A second CLAIMS relationship between the same two nodes: a parallel group.
    let parallel = GraphWrite::add_edge(
        &mut graph.graph,
        source,
        target,
        EdgeData::new(
            "CLAIMS".into(),
            HashMap::from([("uid".to_string(), Value::String("b".into()))]),
            &mut graph.interner,
        ),
    );
    upsert_edge_embeddings(
        &mut graph,
        "CLAIMS",
        "text",
        vec![(parallel, vec![0.5, 0.5])],
        None,
    )
    .unwrap();

    let rows = relationship_embeddings(&graph, "CLAIMS", "text", &RelationshipKeys::new()).unwrap();
    let addresses: Vec<(Value, Value, Option<Value>, Vec<f32>)> = rows
        .iter()
        .map(|row| {
            (
                row.source_id.clone(),
                row.target_id.clone(),
                row.key.clone(),
                row.vector.clone(),
            )
        })
        .collect();
    assert_eq!(
        addresses,
        vec![
            (Value::Int64(1), Value::Int64(2), None, vec![1.0, 0.0]),
            (Value::Int64(1), Value::Int64(2), None, vec![0.5, 0.5]),
            (Value::Int64(1), Value::Int64(3), None, vec![0.0, 1.0]),
            (Value::Int64(1), Value::Int64(4), None, vec![-1.0, 0.0]),
        ]
    );
    assert!(rows
        .iter()
        .all(|row| row.source_type == "Doc" && row.target_type == "Doc"));

    // A key named for the type must tell the group apart: the first member has none.
    let keys = RelationshipKeys::from([("CLAIMS".to_string(), "uid".to_string())]);
    let error = relationship_embeddings(&graph, "CLAIMS", "text", &keys).unwrap_err();
    assert!(
        error.contains("2 'CLAIMS' relationships connect"),
        "{error}"
    );
    assert!(error.contains("has no 'uid' value"), "{error}");

    let error =
        relationship_embeddings(&graph, "CLAIMS", "nope", &RelationshipKeys::new()).unwrap_err();
    assert!(
        error.contains("No relationship embedding store 'CLAIMS.nope'"),
        "{error}"
    );
}

/// Every index listing calls `list_edge_vector_indexes`, and so does every
/// `db.index.fulltext.queryNodes`. A graph with no relationship embedding store
/// must answer without visiting its edges: with 300,000 of them the walk cost
/// milliseconds per call (about 10 ns per edge in release), where the answer
/// is empty. The minimum of 20 calls is compared with a bound the walk
/// exceeds on any machine and the early return stays far under.
#[test]
fn listing_with_no_edge_store_does_not_walk_the_edges() {
    let mut graph = DirGraph::new();
    let hub = GraphWrite::add_node(
        &mut graph.graph,
        NodeData::new(
            Value::Int64(0),
            Value::String("hub".into()),
            "Doc".into(),
            HashMap::new(),
            &mut graph.interner,
        ),
    );
    for _ in 0..300_000 {
        GraphWrite::add_edge(
            &mut graph.graph,
            hub,
            hub,
            EdgeData::new("CITES".into(), HashMap::new(), &mut graph.interner),
        );
    }
    let fastest = (0..20)
        .map(|_| {
            let started = std::time::Instant::now();
            assert!(list_edge_vector_indexes(&graph).is_empty());
            started.elapsed()
        })
        .min()
        .unwrap();
    assert!(
        fastest < std::time::Duration::from_micros(500),
        "listing with no edge store took {fastest:?}: it walked the edges"
    );
}
