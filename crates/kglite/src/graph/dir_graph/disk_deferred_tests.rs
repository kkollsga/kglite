use super::DirGraph;
use crate::datatypes::{DataFrame, Value};
use crate::graph::mutation::maintain;
use crate::graph::storage::GraphRead;
use std::sync::Arc;
use tempfile::TempDir;

fn add_doc(graph: &mut DirGraph, id: i64, email: &str) -> Result<(), String> {
    let frame = DataFrame::from_cypher_rows(
        vec!["id".into(), "title".into(), "email".into()],
        vec![vec![
            Value::Int64(id),
            Value::String(format!("doc-{id}")),
            Value::String(email.into()),
        ]],
    )
    .unwrap();
    maintain::add_nodes(
        graph,
        frame,
        "Doc".into(),
        "id".into(),
        Some("title".into()),
        None,
    )
    .map(|_| ())
}

fn saved_graph(path: &str, indexed: bool) -> DirGraph {
    let mut graph = DirGraph::new();
    add_doc(&mut graph, 1, "first").unwrap();
    if indexed {
        graph.create_index("Doc", "email");
        graph.create_range_index("Doc", "email");
        graph.create_composite_index("Doc", &["email", "title"]);
        graph.create_unique_constraint("Doc", &["email"]).unwrap();
    }
    graph.enable_disk_mode().unwrap();
    graph.save_disk(path).unwrap();
    graph
}

fn load_owned(path: &str) -> DirGraph {
    match Arc::try_unwrap(crate::graph::io::file::load_file(path).unwrap()) {
        Ok(graph) => graph,
        Err(_) => panic!("fresh load unexpectedly shared"),
    }
}

#[test]
fn direct_disk_append_materializes_empty_deferred_state() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    let _original = saved_graph(path, false);
    let mut graph = load_owned(path);
    assert!(graph.indexes_deferred());
    add_doc(&mut graph, 2, "second").unwrap();
    assert!(!graph.indexes_deferred());
    assert_eq!(graph.graph.node_count(), 2);
}

#[test]
fn direct_disk_append_restores_declared_indexes_and_unique_enforcement() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    let _original = saved_graph(path, true);
    let mut graph = load_owned(path);
    let held = graph.clone();
    assert!(graph.indexes_deferred());
    assert!(graph.property_indices.is_empty());
    assert!(graph.unique_indices.is_empty());
    assert_eq!(
        graph.list_unique_constraints(),
        held.list_unique_constraints()
    );
    let error = add_doc(&mut graph, 2, "first").expect_err("duplicate must be refused");
    assert!(error.contains("UNIQUE"), "{error}");
    assert_eq!(graph.graph.node_count(), 1);
    assert!(!graph.indexes_deferred());
    assert!(!graph.range_indices.is_empty());
    assert!(!graph.composite_indices.is_empty());
    add_doc(&mut graph, 2, "second").unwrap();
    let second = graph
        .lookup_by_id_readonly("Doc", &Value::Int64(2))
        .unwrap();
    assert_eq!(
        graph.lookup_by_index("Doc", "email", &Value::String("second".into())),
        Some(vec![second])
    );
    assert!(held.indexes_deferred());
    assert!(held.unique_indices.is_empty());
    assert_eq!(held.graph.node_count(), 1);
}

#[test]
fn direct_disk_property_update_materializes_deferred_indexes() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    let _original = saved_graph(path, true);
    let mut graph = load_owned(path);
    let node = graph
        .lookup_by_id_readonly("Doc", &Value::Int64(1))
        .unwrap();
    maintain::update_node_properties(
        &mut graph,
        &[(Some(node), Value::String("changed".into()))],
        "email",
    )
    .unwrap();
    assert!(!graph.indexes_deferred());
    assert_eq!(
        graph.lookup_by_index("Doc", "email", &Value::String("changed".into())),
        Some(vec![node])
    );
    assert!(graph
        .lookup_by_index("Doc", "email", &Value::String("first".into()))
        .unwrap()
        .is_empty());
    assert!(add_doc(&mut graph, 2, "changed").is_err());
    add_doc(&mut graph, 2, "first").unwrap();
}

#[test]
fn blocked_direct_disk_write_preserves_deferred_state_until_retry() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    let mut first = saved_graph(path, true);
    let mut second = load_owned(path);
    add_doc(&mut first, 3, "third").unwrap();
    let error = add_doc(&mut second, 2, "second").expect_err("second writer must fail");
    assert!(error.contains("active writer"), "{error}");
    assert!(second.indexes_deferred());
    assert!(second.property_indices.is_empty());
    assert!(second.unique_indices.is_empty());
    assert_eq!(second.graph.node_count(), 1);
    first.save_disk(path).unwrap();
    add_doc(&mut second, 2, "second").unwrap();
    assert!(!second.indexes_deferred());
    assert_eq!(second.graph.node_count(), 2);
    assert!(add_doc(&mut second, 4, "first").is_err());
    assert_eq!(second.graph.node_count(), 2);
}

fn check_portable_direct_mutation(mode: crate::graph::storage::mode::StorageMode) {
    use crate::graph::io::file::{load_file_with, save_graph, LoadOptions};
    use crate::graph::storage::mode::live_storage_mode;

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("indexed.kgl");
    let path_str = path.to_str().unwrap();
    let mut original = DirGraph::new();
    add_doc(&mut original, 1, "first").unwrap();
    original.create_index("Doc", "email");
    original
        .create_unique_constraint("Doc", &["email"])
        .unwrap();
    let mut original = Arc::new(original);
    save_graph(&mut original, path_str).unwrap();
    drop(original);
    let saved_bytes = std::fs::read(&path).unwrap();
    let loaded = load_file_with(
        path_str,
        &LoadOptions::new()
            .with_storage(mode)
            .with_defer_index_rebuild(true),
    )
    .unwrap();
    let mut graph = match Arc::try_unwrap(loaded) {
        Ok(graph) => graph,
        Err(_) => panic!("fresh portable load unexpectedly shared"),
    };
    assert_eq!(live_storage_mode(&graph), mode);
    assert!(graph.indexes_deferred());
    assert!(graph.property_indices.is_empty());
    assert!(graph.unique_indices.is_empty());
    let held = graph.clone();
    let error = add_doc(&mut graph, 2, "first").expect_err("duplicate must be refused");
    assert!(error.contains("UNIQUE"), "{error}");
    assert!(!graph.indexes_deferred());
    assert_eq!(graph.graph.node_count(), 1);
    add_doc(&mut graph, 2, "second").unwrap();
    let second = graph
        .lookup_by_id_readonly("Doc", &Value::Int64(2))
        .unwrap();
    assert_eq!(
        graph.lookup_by_index("Doc", "email", &Value::String("second".into())),
        Some(vec![second])
    );
    maintain::update_node_properties(
        &mut graph,
        &[(Some(second), Value::String("changed".into()))],
        "email",
    )
    .unwrap();
    assert_eq!(
        graph.lookup_by_index("Doc", "email", &Value::String("changed".into())),
        Some(vec![second])
    );
    assert!(add_doc(&mut graph, 3, "changed").is_err());
    assert_eq!(graph.graph.node_count(), 2);
    assert!(held.indexes_deferred());
    assert!(held.unique_indices.is_empty());
    assert_eq!(held.graph.node_count(), 1);
    assert_eq!(std::fs::read(&path).unwrap(), saved_bytes);
}

#[test]
fn direct_memory_loaded_mutation_materializes_deferred_indexes() {
    check_portable_direct_mutation(crate::graph::storage::mode::StorageMode::Memory);
}

#[test]
fn direct_mapped_loaded_mutation_materializes_deferred_indexes() {
    check_portable_direct_mutation(crate::graph::storage::mode::StorageMode::Mapped);
}

fn assert_saved_index_declarations(graph: &DirGraph) {
    assert_eq!(
        graph.property_index_keys,
        vec![("Doc".into(), "email".into())]
    );
    assert_eq!(graph.range_index_keys, vec![("Doc".into(), "email".into())]);
    assert_eq!(
        graph.composite_index_keys,
        vec![("Doc".into(), vec!["email".into(), "title".into()])]
    );
    assert_eq!(
        graph.list_unique_constraints(),
        vec![("Doc".into(), vec!["email".into()])]
    );
    assert_eq!(
        graph.list_composite_indexes(),
        vec![("Doc".into(), vec!["email".into(), "title".into()])]
    );
}

#[test]
fn disk_save_snapshots_all_live_index_declarations() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    let original = saved_graph(path, true);
    assert_eq!(original.property_indices.len(), 1);
    assert_eq!(original.range_indices.len(), 1);
    assert_eq!(original.composite_indices.len(), 1);
    let loaded = load_owned(path);
    assert_saved_index_declarations(&loaded);
    assert!(loaded.indexes_deferred());
    assert!(loaded.property_indices.is_empty());
    assert!(loaded.range_indices.is_empty());
    assert!(loaded.composite_indices.is_empty());
    assert!(loaded.unique_indices.is_empty());
}

#[test]
fn deferred_disk_resave_preserves_declarations_without_materializing() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    let _original = saved_graph(path, true);
    let mut loaded = load_owned(path);
    assert_saved_index_declarations(&loaded);
    let held = loaded.clone();
    loaded.save_disk(path).unwrap();
    assert_saved_index_declarations(&loaded);
    assert!(loaded.indexes_deferred());
    assert!(loaded.property_indices.is_empty());
    assert!(loaded.range_indices.is_empty());
    assert!(loaded.composite_indices.is_empty());
    assert!(loaded.unique_indices.is_empty());
    assert_saved_index_declarations(&held);
    assert!(held.indexes_deferred());
    let reloaded = load_owned(path);
    assert_saved_index_declarations(&reloaded);
    assert!(reloaded.indexes_deferred());
    assert_eq!(reloaded.graph.node_count(), 1);
}
