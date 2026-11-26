use super::DirGraph;
use crate::datatypes::Value;
use crate::graph::edge_embeddings::{
    edge_store_key, remove_edge_embeddings, remove_edge_with_embeddings, upsert_edge_embeddings,
    PersistedEdgeEmbeddingStore,
};
use crate::graph::io::file::{load_file, save_graph};
use crate::graph::schema::{EdgeData, NodeData};
use crate::graph::storage::{GraphRead, GraphWrite};
use petgraph::graph::EdgeIndex;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;

fn graph_with_relationships() -> (DirGraph, EdgeIndex, EdgeIndex, EdgeIndex) {
    let mut graph = DirGraph::new();
    let a = GraphWrite::add_node(
        &mut graph.graph,
        NodeData::new(
            Value::Int64(1),
            Value::String("a".into()),
            "Doc".into(),
            HashMap::new(),
            &mut graph.interner,
        ),
    );
    let b = GraphWrite::add_node(
        &mut graph.graph,
        NodeData::new(
            Value::Int64(2),
            Value::String("b".into()),
            "Doc".into(),
            HashMap::new(),
            &mut graph.interner,
        ),
    );
    let first = GraphWrite::add_edge(
        &mut graph.graph,
        a,
        b,
        EdgeData::new("ASSERTS".into(), HashMap::new(), &mut graph.interner),
    );
    let parallel = GraphWrite::add_edge(
        &mut graph.graph,
        a,
        b,
        EdgeData::new("ASSERTS".into(), HashMap::new(), &mut graph.interner),
    );
    let self_loop = GraphWrite::add_edge(
        &mut graph.graph,
        a,
        a,
        EdgeData::new("MENTIONS".into(), HashMap::new(), &mut graph.interner),
    );
    (graph, first, parallel, self_loop)
}

fn save_disk(mut graph: DirGraph, dir: &std::path::Path) {
    if !graph.graph.is_disk() {
        graph.enable_disk_mode().unwrap();
    }
    let mut graph = Arc::new(graph);
    save_graph(&mut graph, dir.to_str().unwrap()).unwrap();
}

fn only_edge_of_type(graph: &DirGraph, relationship_type: &str) -> EdgeIndex {
    let matches: Vec<_> = graph
        .graph
        .edge_indices()
        .filter(|edge| {
            graph
                .graph
                .edge_weight(*edge)
                .is_some_and(|data| data.connection_type_str(&graph.interner) == relationship_type)
        })
        .collect();
    assert_eq!(matches.len(), 1, "expected one {relationship_type} edge");
    matches[0]
}

fn edge_property_format(dir: &std::path::Path) -> u64 {
    let resolved = crate::graph::storage::disk::generation::resolve_snapshot(dir).unwrap();
    let metadata: serde_json::Value = serde_json::from_slice(
        &std::fs::read(resolved.snapshot_dir.join("disk_graph_meta.json")).unwrap(),
    )
    .unwrap();
    metadata["edge_properties_format"].as_u64().unwrap()
}

#[test]
fn disk_round_trip_preserves_parallel_and_self_loop_vectors() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("graph");
    let (mut graph, first, parallel, self_loop) = graph_with_relationships();
    upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(first, vec![1.0, 0.0]), (parallel, vec![0.0, 1.0])],
        Some("cosine"),
    )
    .unwrap();
    upsert_edge_embeddings(
        &mut graph,
        "MENTIONS",
        "description",
        vec![(self_loop, vec![0.5, 0.5])],
        Some("cosine"),
    )
    .unwrap();

    save_disk(graph, &dir);
    assert_eq!(edge_property_format(&dir), 3);
    let loaded = load_file(dir.to_str().unwrap()).unwrap();

    let assertions = &loaded.edge_embeddings[&edge_store_key("ASSERTS", "description")];
    assert_eq!(assertions.get(first), Some(&[1.0, 0.0][..]));
    assert_eq!(assertions.get(parallel), Some(&[0.0, 1.0][..]));
    let mentions = &loaded.edge_embeddings[&edge_store_key("MENTIONS", "description")];
    assert_eq!(mentions.get(self_loop), Some(&[0.5, 0.5][..]));
}

/// A graph with no edge vectors must keep writing format 2: every reader up to
/// 0.17.12 refuses a snapshot stamped 3 with "unsupported edge property format
/// 3", so promoting the format unconditionally would make ordinary graphs
/// unreadable by installed versions.
#[test]
fn node_only_disk_snapshot_stays_format_two_without_sidecar() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("graph");
    let (graph, _, _, _) = graph_with_relationships();

    save_disk(graph, &dir);

    let resolved = crate::graph::storage::disk::generation::resolve_snapshot(&dir).unwrap();
    assert_eq!(edge_property_format(&dir), 2);
    assert!(!resolved
        .snapshot_dir
        .join("edge_embeddings.bin.zst")
        .exists());
    load_file(dir.to_str().unwrap()).unwrap();
}

/// A declared-but-empty store still carries dimension and metric that only
/// format 3 records, so the bump is required here even with no vectors left —
/// and 3 is exactly the value older readers refuse ("unsupported edge property
/// format 3"), which is what keeps them from silently dropping the store.
#[test]
fn empty_declared_edge_store_requires_format_three_and_round_trips() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("graph");
    let (mut graph, first, _, _) = graph_with_relationships();
    upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(first, vec![1.0, 0.0])],
        Some("cosine"),
    )
    .unwrap();
    remove_edge_embeddings(&mut graph, "ASSERTS", "description", &[first]).unwrap();

    save_disk(graph, &dir);
    assert_eq!(edge_property_format(&dir), 3);
    let loaded = load_file(dir.to_str().unwrap()).unwrap();
    let store = &loaded.edge_embeddings[&edge_store_key("ASSERTS", "description")];
    assert!(store.is_empty());
    assert_eq!(store.dimension(), 2);
    assert_eq!(store.metric(), Some("cosine"));
}

#[test]
fn save_time_compaction_preserves_survivor_and_overflow_vectors() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("graph");
    let (mut graph, removed, survivor, self_loop) = graph_with_relationships();
    upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(survivor, vec![0.0, 1.0])],
        None,
    )
    .unwrap();
    let endpoints = graph.graph.edge_endpoints(removed).unwrap();
    remove_edge_with_embeddings(&mut graph, removed).unwrap();
    remove_edge_with_embeddings(&mut graph, self_loop).unwrap();
    graph.enable_disk_mode().unwrap();
    let overflow = GraphWrite::add_edge(
        &mut graph.graph,
        endpoints.0,
        endpoints.1,
        EdgeData::new("FOLLOWS".into(), HashMap::new(), &mut graph.interner),
    );
    upsert_edge_embeddings(
        &mut graph,
        "FOLLOWS",
        "description",
        vec![(overflow, vec![0.25, 0.75])],
        None,
    )
    .unwrap();

    save_disk(graph, &dir);
    let loaded = load_file(dir.to_str().unwrap()).unwrap();
    let assertion = only_edge_of_type(&loaded, "ASSERTS");
    let follows = only_edge_of_type(&loaded, "FOLLOWS");
    assert_eq!(
        loaded.edge_embeddings[&edge_store_key("ASSERTS", "description")].get(assertion),
        Some(&[0.0, 1.0][..])
    );
    assert_eq!(
        loaded.edge_embeddings[&edge_store_key("FOLLOWS", "description")].get(follows),
        Some(&[0.25, 0.75][..])
    );
}

#[test]
fn reopened_disk_graph_preserves_embeddings_through_append_delete_and_save_as() {
    let tmp = tempfile::tempdir().unwrap();
    let first_dir = tmp.path().join("first");
    let second_dir = tmp.path().join("second");
    let (mut graph, first, parallel, self_loop) = graph_with_relationships();
    upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(first, vec![1.0, 0.0]), (parallel, vec![0.0, 1.0])],
        None,
    )
    .unwrap();
    save_disk(graph, &first_dir);

    let loaded = load_file(first_dir.to_str().unwrap()).unwrap();
    let mut graph = match Arc::try_unwrap(loaded) {
        Ok(graph) => graph,
        Err(_) => panic!("load returns one graph owner"),
    };
    let endpoints = graph.graph.edge_endpoints(parallel).unwrap();
    remove_edge_with_embeddings(&mut graph, first).unwrap();
    remove_edge_with_embeddings(&mut graph, self_loop).unwrap();
    let appended = GraphWrite::add_edge(
        &mut graph.graph,
        endpoints.0,
        endpoints.1,
        EdgeData::new("FOLLOWS".into(), HashMap::new(), &mut graph.interner),
    );
    upsert_edge_embeddings(
        &mut graph,
        "FOLLOWS",
        "description",
        vec![(appended, vec![0.25, 0.75])],
        None,
    )
    .unwrap();
    save_disk(graph, &second_dir);

    let reloaded = load_file(second_dir.to_str().unwrap()).unwrap();
    let assertion = only_edge_of_type(&reloaded, "ASSERTS");
    let follows = only_edge_of_type(&reloaded, "FOLLOWS");
    assert_eq!(
        reloaded.edge_embeddings[&edge_store_key("ASSERTS", "description")].get(assertion),
        Some(&[0.0, 1.0][..])
    );
    assert_eq!(
        reloaded.edge_embeddings[&edge_store_key("FOLLOWS", "description")].get(follows),
        Some(&[0.25, 0.75][..])
    );
}

#[test]
fn disk_round_trip_preserves_model_metric_and_text_hash() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("graph");
    let (mut graph, first, _, _) = graph_with_relationships();
    let decoded = std::collections::BTreeMap::from([(
        edge_store_key("ASSERTS", "description"),
        PersistedEdgeEmbeddingStore::fixture(
            2,
            vec![1.0, 0.0],
            vec![first.index()],
            Some("dot_product".to_string()),
            Some("model-v1".to_string()),
            HashMap::from([(first.index(), 42)]),
        ),
    )]);
    graph.edge_embeddings =
        crate::graph::edge_embeddings::validate_decoded_edge_embedding_stores(&graph, decoded)
            .unwrap();
    save_disk(graph, &dir);

    let loaded = load_file(dir.to_str().unwrap()).unwrap();
    let store = &loaded.edge_embeddings[&edge_store_key("ASSERTS", "description")];
    assert_eq!(store.metric(), Some("dot_product"));
    assert_eq!(store.model_id(), Some("model-v1"));
    assert_eq!(store.text_hash(first), Some(42));
}

#[test]
fn format_three_without_required_sidecar_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("graph");
    let (mut graph, first, _, _) = graph_with_relationships();
    upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(first, vec![1.0, 0.0])],
        None,
    )
    .unwrap();
    save_disk(graph, &dir);
    let resolved = crate::graph::storage::disk::generation::resolve_snapshot(&dir).unwrap();
    std::fs::remove_file(resolved.snapshot_dir.join("edge_embeddings.bin.zst")).unwrap();

    let error = load_file(dir.to_str().unwrap())
        .err()
        .expect("missing required sidecar must fail");
    assert!(error
        .to_string()
        .contains("required relationship-embedding sidecar is missing"));
}

#[derive(Serialize)]
struct PersistedFixture {
    dimension: usize,
    data: Vec<f32>,
    edge_slots: Vec<usize>,
    metric: Option<String>,
    model_id: Option<String>,
    text_hashes: HashMap<usize, u64>,
}

fn overwrite_edge_sidecar(
    dir: &std::path::Path,
    payload: &std::collections::BTreeMap<(String, String), PersistedFixture>,
) {
    let resolved = crate::graph::storage::disk::generation::resolve_snapshot(dir).unwrap();
    let framed = crate::graph::io::file::encode_disk_serde(payload).unwrap();
    let compressed = zstd::encode_all(framed.as_slice(), 3).unwrap();
    std::fs::write(
        resolved.snapshot_dir.join("edge_embeddings.bin.zst"),
        compressed,
    )
    .unwrap();
}

#[test]
fn corrupt_disk_edge_store_payloads_are_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("graph");
    let (mut graph, first, _, _) = graph_with_relationships();
    upsert_edge_embeddings(
        &mut graph,
        "ASSERTS",
        "description",
        vec![(first, vec![1.0, 0.0])],
        None,
    )
    .unwrap();
    save_disk(graph, &dir);

    overwrite_edge_sidecar(&dir, &std::collections::BTreeMap::new());
    let empty_error = load_file(dir.to_str().unwrap())
        .err()
        .expect("format 3 with an empty decoded store map must fail")
        .to_string();
    assert!(empty_error.contains("contains no stores"), "{empty_error}");

    let cases = [
        (
            "non-finite",
            ("ASSERTS", vec![f32::NAN, 0.0], vec![first.index()]),
            "finite",
        ),
        (
            "wrong width",
            ("ASSERTS", vec![1.0], vec![first.index()]),
            "cardinality",
        ),
        (
            "dead slot",
            ("ASSERTS", vec![1.0, 0.0], vec![9999]),
            "not live",
        ),
        (
            "wrong type",
            ("MENTIONS", vec![1.0, 0.0], vec![first.index()]),
            "expected 'MENTIONS'",
        ),
    ];
    for (name, (relationship_type, data, edge_slots), expected) in cases {
        let payload = std::collections::BTreeMap::from([(
            (relationship_type.to_string(), "description_emb".to_string()),
            PersistedFixture {
                dimension: 2,
                data,
                edge_slots,
                metric: Some("cosine".to_string()),
                model_id: Some("test-model".to_string()),
                text_hashes: HashMap::new(),
            },
        )]);
        overwrite_edge_sidecar(&dir, &payload);
        let error = load_file(dir.to_str().unwrap())
            .err()
            .unwrap_or_else(|| panic!("{name} sidecar must fail"));
        let message = error.to_string();
        assert!(message.contains("edge_embeddings.bin.zst"), "{message}");
        assert!(message.contains(expected), "{name}: {message}");
    }
}

#[test]
fn decoded_store_validation_is_atomic_before_disk_install() {
    let (graph, first, parallel, _) = graph_with_relationships();
    let valid = PersistedEdgeEmbeddingStore::fixture(
        2,
        vec![1.0, 0.0],
        vec![first.index()],
        None,
        None,
        HashMap::new(),
    );
    let invalid = PersistedEdgeEmbeddingStore::fixture(
        2,
        vec![f32::NAN, 0.0],
        vec![parallel.index()],
        None,
        None,
        HashMap::new(),
    );
    let decoded = std::collections::BTreeMap::from([
        (edge_store_key("ASSERTS", "description"), valid),
        (edge_store_key("ASSERTS", "summary"), invalid),
    ]);

    assert!(
        crate::graph::edge_embeddings::validate_decoded_edge_embedding_stores(&graph, decoded,)
            .is_err()
    );
    assert!(graph.edge_embeddings.is_empty());
}
