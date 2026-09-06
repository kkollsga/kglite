//! Direct index-state oracles: a predicate scan fallback must not mask stale tuples.

use crate::datatypes::Value;
use crate::graph::io::file::{load_file, save_graph};
use crate::graph::schema::{DirGraph, InternedKey};
use crate::graph::session::{execute_mut, ExecuteOptions};
use crate::graph::storage::{GraphRead, GraphWrite};
use std::collections::HashMap;
use std::sync::Arc;

fn saved_disk_graph() -> (tempfile::TempDir, Arc<DirGraph>) {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("graph");
    let params = HashMap::new();
    let options = ExecuteOptions::new(&params);
    let mut graph = DirGraph::new();
    execute_mut(
        &mut graph,
        "CREATE (:T {id:1,a:'old',b:'fixed'}),(:T {id:2,a:'other',b:'fixed'})",
        &options,
    )
    .unwrap();
    graph.enable_disk_mode().unwrap();
    let mut handle = Arc::new(graph);
    save_graph(&mut handle, path.to_str().unwrap()).unwrap();
    drop(handle);
    (tmp, load_file(path.to_str().unwrap()).unwrap())
}

#[test]
fn disk_set_and_remove_publish_before_composite_index_readback() {
    let (_tmp, mut handle) = saved_disk_graph();
    let params = HashMap::new();
    let options = ExecuteOptions::new(&params);
    let graph = Arc::make_mut(&mut handle);
    assert!(graph.graph.is_disk());
    assert_eq!(graph.create_composite_index("T", &["a", "b"]), 2);
    let properties = ["a".to_string(), "b".to_string()];
    let tuple = |a: Value| [a, Value::String("fixed".to_string())];
    let old = tuple(Value::String("old".to_string()));
    let changed = tuple(Value::String("new".to_string()));
    let removed = tuple(Value::Null);
    let member = graph
        .lookup_by_composite_index("T", &properties, &old)
        .unwrap();
    assert_eq!(member.len(), 1);

    execute_mut(graph, "MATCH(n:T {id:1}) SET n.a='new'", &options).unwrap();
    assert_eq!(
        graph.lookup_by_composite_index("T", &properties, &changed),
        Some(member.clone())
    );
    assert!(graph
        .lookup_by_composite_index("T", &properties, &old)
        .unwrap_or_default()
        .is_empty());

    execute_mut(graph, "MATCH(n:T {id:1}) REMOVE n.a", &options).unwrap();
    assert_eq!(
        graph.lookup_by_composite_index("T", &properties, &removed),
        Some(member)
    );
    assert!(graph
        .lookup_by_composite_index("T", &properties, &changed)
        .unwrap_or_default()
        .is_empty());
}

#[test]
fn disk_clear_returns_committed_or_staged_previous_value() {
    let (_tmp, mut handle) = saved_disk_graph();
    let graph = Arc::make_mut(&mut handle);
    let key = InternedKey::from_str("a");
    let node = petgraph::graph::NodeIndex::new(0);
    assert_eq!(
        graph.graph.get_node_property(node, key),
        Some(Value::String("old".to_string()))
    );
    assert_eq!(
        GraphWrite::clear_node_property(&mut graph.graph, node, key),
        Some(Value::String("old".to_string()))
    );
    assert_eq!(
        GraphWrite::clear_node_property(&mut graph.graph, node, key),
        Some(Value::Null)
    );
    GraphWrite::flush_pending_writes(&mut graph.graph);
    assert_eq!(
        GraphWrite::clear_node_property(&mut graph.graph, node, key),
        None
    );
    GraphWrite::set_node_property(
        &mut graph.graph,
        node,
        key,
        Value::String("staged".to_string()),
    );
    assert_eq!(
        GraphWrite::remove_node_property(&mut graph.graph, node, key),
        Some(Value::String("staged".to_string()))
    );
    GraphWrite::flush_pending_writes(&mut graph.graph);
    assert_eq!(graph.graph.get_node_property(node, key), None);
    let type_key = InternedKey::from_str("T");
    let columns = graph.graph.column_store(type_key).unwrap().column_count();
    assert_eq!(
        GraphWrite::remove_node_property(&mut graph.graph, node, InternedKey::from_str("missing")),
        None
    );
    GraphWrite::flush_pending_writes(&mut graph.graph);
    let store = graph.graph.column_store(type_key).unwrap();
    assert_eq!(store.column_count(), columns);
    assert!(store.slot(InternedKey::from_str("missing")).is_none());
}

fn assert_removed_property_and_untouched_row(graph: &DirGraph) {
    let _query = graph.graph.begin_query();
    let first = graph
        .graph
        .node_view(petgraph::graph::NodeIndex::new(0))
        .unwrap();
    let second = graph
        .graph
        .node_view(petgraph::graph::NodeIndex::new(1))
        .unwrap();
    let key = InternedKey::from_str("a");
    assert_eq!(first.get_property_value("a"), None);
    assert_eq!(first.str_prop_eq(key, "old"), None);
    assert!(!first.has_property("a"));
    assert_eq!(
        first.property_pairs(),
        vec![(InternedKey::from_str("b"), Value::String("fixed".into()))]
    );
    assert_eq!(first.property_key_set(), vec![InternedKey::from_str("b")]);
    assert_eq!(first.property_count(), 1);
    assert_eq!(
        second.get_property_value("a"),
        Some(Value::String("other".into()))
    );
    assert_eq!(
        second.get_property_value("b"),
        Some(Value::String("fixed".into()))
    );
}

#[test]
fn saved_disk_null_writes_preserve_reads_snapshots_and_resave() {
    for operation in ["REMOVE n.a", "SET n.a=null"] {
        let (tmp, mut handle) = saved_disk_graph();
        let held = Arc::clone(&handle);
        let params = HashMap::new();
        let options = ExecuteOptions::new(&params);
        execute_mut(
            Arc::make_mut(&mut handle),
            &format!("MATCH(n:T {{id:1}}) {operation}"),
            &options,
        )
        .unwrap();
        assert_removed_property_and_untouched_row(&handle);
        assert_eq!(
            held.graph.get_node_property(
                petgraph::graph::NodeIndex::new(0),
                InternedKey::from_str("a")
            ),
            Some(Value::String("old".into()))
        );
        let output = tmp.path().join("resaved");
        save_graph(&mut handle, output.to_str().unwrap()).unwrap();
        drop(handle);
        let reopened = load_file(output.to_str().unwrap()).unwrap();
        assert_removed_property_and_untouched_row(&reopened);
    }
}

#[test]
fn saved_disk_null_rollback_and_replacement_restore_exact_value() {
    let (_tmp, mut handle) = saved_disk_graph();
    let params = HashMap::new();
    let options = ExecuteOptions::new(&params);
    let graph = Arc::make_mut(&mut handle);
    assert!(execute_mut(graph, "MATCH(n:T {id:1}) SET n.a=null,n.id=9", &options).is_err());
    let first = petgraph::graph::NodeIndex::new(0);
    let key = InternedKey::from_str("a");
    assert_eq!(
        graph.graph.get_node_property(first, key),
        Some(Value::String("old".into()))
    );
    execute_mut(graph, "MATCH(n:T {id:1}) REMOVE n.a", &options).unwrap();
    assert_removed_property_and_untouched_row(graph);
    execute_mut(graph, "MATCH(n:T {id:1}) SET n.a='restored'", &options).unwrap();
    assert_eq!(
        graph.graph.get_node_property(first, key),
        Some(Value::String("restored".into()))
    );
    execute_mut(graph, "MATCH(n:T {id:1}) SET n.a=null", &options).unwrap();
    assert_removed_property_and_untouched_row(graph);
}

#[test]
fn saved_disk_property_overrides_survive_save_without_readback() {
    let (tmp, mut handle) = saved_disk_graph();
    let params = HashMap::new();
    let options = ExecuteOptions::new(&params);
    execute_mut(
        Arc::make_mut(&mut handle),
        "MATCH(n:T {id:1}) SET n.a=null,n.fresh='added'",
        &options,
    )
    .unwrap();
    // Persist before checking merged reads so this independently detects stale bytes.
    let output = tmp.path().join("saved-with-overrides");
    save_graph(&mut handle, output.to_str().unwrap()).unwrap();
    drop(handle);
    let reopened = load_file(output.to_str().unwrap()).unwrap();
    let first = petgraph::graph::NodeIndex::new(0);
    let second = petgraph::graph::NodeIndex::new(1);
    assert_eq!(
        reopened
            .graph
            .get_node_property(first, InternedKey::from_str("fresh")),
        Some(Value::String("added".into()))
    );
    assert_eq!(
        reopened
            .graph
            .get_node_property(first, InternedKey::from_str("a")),
        None
    );
    assert_eq!(
        reopened
            .graph
            .get_node_property(second, InternedKey::from_str("a")),
        Some(Value::String("other".into()))
    );
    assert_eq!(
        reopened
            .graph
            .get_node_property(second, InternedKey::from_str("fresh")),
        None
    );
}

#[test]
fn untouched_saved_disk_resave_to_new_directory_preserves_complete_rows() {
    let (tmp, mut handle) = saved_disk_graph();
    let output = tmp.path().join("untouched-resaved");
    save_graph(&mut handle, output.to_str().unwrap()).unwrap();
    drop(handle);
    let reopened = load_file(output.to_str().unwrap()).unwrap();
    let _query = reopened.graph.begin_query();
    for (row, expected) in [(0, "old"), (1, "other")] {
        let node = reopened
            .graph
            .node_view(petgraph::graph::NodeIndex::new(row))
            .unwrap();
        assert_eq!(node.id().as_ref(), &Value::Int64(row as i64 + 1));
        assert_eq!(
            node.get_property_value("a"),
            Some(Value::String(expected.into()))
        );
        assert_eq!(
            node.get_property_value("b"),
            Some(Value::String("fixed".into()))
        );
        assert_eq!(node.property_count(), 2);
    }
}

#[test]
fn saved_disk_title_clear_preserves_sentinels_snapshots_rollback_and_save() {
    use crate::graph::schema::PropertyStorage;
    use crate::graph::storage::property_storage::ColumnarRow;

    let (tmp, mut handle) = saved_disk_graph();
    let params = HashMap::new();
    let options = ExecuteOptions::new(&params);
    let first = petgraph::graph::NodeIndex::new(0);
    execute_mut(
        Arc::make_mut(&mut handle),
        "MATCH(n:T) SET n.title='kept'",
        &options,
    )
    .unwrap();
    let path = tmp.path().join("title-base");
    save_graph(&mut handle, path.to_str().unwrap()).unwrap();
    drop(handle);
    let mut handle = load_file(path.to_str().unwrap()).unwrap();
    {
        let graph = Arc::make_mut(&mut handle);
        let scratch = GraphWrite::node_weight_mut(&mut graph.graph, first).unwrap();
        scratch.properties = PropertyStorage::Columnar(ColumnarRow::new(0));
        scratch.title = Value::Null;
        GraphWrite::flush_pending_writes(&mut graph.graph);
        assert_eq!(
            graph.graph.get_node_title(first),
            Some(Value::String("kept".into()))
        );
        assert!(execute_mut(graph, "MATCH(n:T {id:1}) REMOVE n.title,n.id", &options).is_err());
        assert_eq!(
            graph.graph.get_node_title(first),
            Some(Value::String("kept".into()))
        );
    }
    let held = Arc::clone(&handle);
    let result = execute_mut(
        Arc::make_mut(&mut handle),
        "MATCH(n:T {id:1}) REMOVE n.title,n.title",
        &options,
    )
    .unwrap();
    assert_eq!(result.result.stats.unwrap().properties_removed, 1);
    assert_eq!(handle.graph.get_node_title(first), None);
    assert_eq!(
        held.graph.get_node_title(first),
        Some(Value::String("kept".into()))
    );
    assert_eq!(
        handle
            .graph
            .get_node_title(petgraph::graph::NodeIndex::new(1)),
        Some(Value::String("kept".into()))
    );
    let output = tmp.path().join("title-cleared");
    save_graph(&mut handle, output.to_str().unwrap()).unwrap();
    drop(handle);
    let loaded = load_file(output.to_str().unwrap()).unwrap();
    assert_eq!(loaded.graph.get_node_title(first), None);
    assert_eq!(
        loaded
            .graph
            .get_node_title(petgraph::graph::NodeIndex::new(1)),
        Some(Value::String("kept".into()))
    );
}

#[test]
fn forked_missing_property_remove_does_not_grow_schema() {
    let params = HashMap::new();
    let options = ExecuteOptions::new(&params);
    let mut graph = DirGraph::new();
    execute_mut(&mut graph, "CREATE(:T{id:1,a:'kept'})", &options).unwrap();
    let session = crate::graph::session::Session::new(graph);
    let held = session.snapshot();
    let mut working = session.write();
    let graph = &mut *working;
    assert!(graph.graph.is_forked());
    let type_key = InternedKey::from_str("T");
    let count = graph.graph.column_store(type_key).unwrap().column_count();
    assert_eq!(
        GraphWrite::remove_node_property(
            &mut graph.graph,
            petgraph::graph::NodeIndex::new(0),
            InternedKey::from_str("missing")
        ),
        None
    );
    assert_eq!(
        graph.graph.column_store(type_key).unwrap().column_count(),
        count
    );
    assert_eq!(
        held.graph.column_store(type_key).unwrap().column_count(),
        count
    );
}
