//! Lifecycle tests for the graph-side text index: build, score, drop, and the
//! four ways an index can go wrong when the graph moves under it (delete,
//! `NodeIndex` reuse, vacuum, fork).

use super::*;
use crate::datatypes::Value;
use crate::graph::introspection::schema_overview::collect_indexes_structured;
use crate::graph::schema::NodeData;
use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};
use crate::graph::storage::GraphWrite;
use std::collections::{HashMap, HashSet};

/// Add one `Doc` node with `body` text, returning its index.
fn push_doc(graph: &mut DirGraph, id: i64, title: &str, body: Value) -> NodeIndex {
    let mut props = HashMap::new();
    props.insert("body".to_string(), body);
    let data = NodeData::new(
        Value::Int64(id),
        Value::String(title.to_string()),
        "Doc".to_string(),
        props,
        &mut graph.interner,
    );
    let idx = GraphWrite::add_node(&mut graph.graph, data);
    graph
        .type_indices
        .entry_or_default("Doc".to_string())
        .push(idx);
    // What both production creation funnels do at the same point — without it
    // these fixtures would exercise a graph no `CREATE` can produce, and the
    // recycled-slot detection would never be reached.
    crate::graph::index_freshness::write_hooks::note_node_created(graph, idx, "Doc");
    idx
}

/// A three-document corpus with one term unique to the middle document.
fn corpus(graph: &mut DirGraph) -> Vec<NodeIndex> {
    vec![
        push_doc(graph, 1, "a", Value::String("the quick brown fox".into())),
        push_doc(
            graph,
            2,
            "b",
            Value::String("a quick brown marmoset appears".into()),
        ),
        push_doc(graph, 3, "c", Value::String("slow green turtles".into())),
    ]
}

fn memory_corpus() -> (DirGraph, Vec<NodeIndex>) {
    let mut graph = DirGraph::new();
    let nodes = corpus(&mut graph);
    graph.build_id_index("Doc");
    (graph, nodes)
}

fn store<'g>(graph: &'g DirGraph, property: &str) -> &'g TextIndexStore {
    graph
        .text_indexes
        .get(&index_key("Doc", property))
        .expect("index built")
}

/// Score every node against `query`, in node order.
fn scores(graph: &DirGraph, nodes: &[NodeIndex], query: &str) -> Vec<Option<f64>> {
    let store = store(graph, "body");
    let prepared = store.prepare_query(query);
    nodes.iter().map(|&n| store.score(n, &prepared)).collect()
}

#[test]
fn build_indexes_every_string_document_and_reports_its_shape() {
    let (mut graph, nodes) = memory_corpus();
    let before = graph.version();

    let report = build_text_index(&mut graph, "Doc", "body", None).expect("build");

    assert_eq!(report.indexed, 3);
    assert_eq!(report.skipped, 0);
    assert_eq!(report.terms, 10, "the union of the three documents' terms");
    assert!(graph.version() > before, "a build is a write");

    // "marmoset" occurs in exactly one document, so only that row scores.
    let marmoset = scores(&graph, &nodes, "marmoset");
    assert_eq!(marmoset[0], Some(0.0));
    assert!(marmoset[1].expect("indexed") > 0.0);
    assert_eq!(marmoset[2], Some(0.0));

    assert_eq!(store(&graph, "body").resolved_field(), "body");
    assert!(store(&graph, "body").validate().is_ok());
}

#[test]
fn an_absent_or_non_string_property_is_skipped_but_an_empty_string_is_a_document() {
    let mut graph = DirGraph::new();
    let indexed = push_doc(&mut graph, 1, "a", Value::String("real text".into()));
    let empty = push_doc(&mut graph, 2, "b", Value::String(String::new()));
    let numeric = push_doc(&mut graph, 3, "c", Value::Int64(42));
    let absent = push_doc(&mut graph, 4, "d", Value::Null);

    let report = build_text_index(&mut graph, "Doc", "body", None).expect("build");

    assert_eq!(report.indexed, 2, "the string and the empty string");
    assert_eq!(report.skipped, 2, "the number and the null");
    let store = store(&graph, "body");
    assert!(store.contains_node(indexed));
    assert!(
        store.contains_node(empty),
        "an empty string is a document with no terms, not a missing document"
    );
    assert!(!store.contains_node(numeric), "a number is not text");
    assert!(!store.contains_node(absent));

    // The empty document still counts in the corpus, which is what makes an
    // unindexed row (None) different from a zero-scoring one (Some(0.0)).
    assert_eq!(store.documents(), 2);
    let prepared = store.prepare_query("real");
    assert_eq!(store.score(empty, &prepared), Some(0.0));
    assert_eq!(store.score(numeric, &prepared), None);
}

#[test]
fn the_indexed_value_is_read_through_the_title_alias() {
    let mut graph = DirGraph::new();
    let node = push_doc(&mut graph, 1, "quick brown fox", Value::Null);
    graph
        .title_field_aliases_mut()
        .insert("Doc".to_string(), "name".to_string());

    let report = build_text_index(&mut graph, "Doc", "name", None).expect("build");

    assert_eq!(report.indexed, 1, "'name' resolves onto the title column");
    let store = store(&graph, "name");
    assert_eq!(store.resolved_field(), "title");
    let prepared = store.prepare_query("fox");
    assert!(store.score(node, &prepared).expect("indexed") > 0.0);
}

#[test]
fn a_rebuild_replaces_the_index_and_picks_up_the_new_text() {
    let (mut graph, nodes) = memory_corpus();
    build_text_index(&mut graph, "Doc", "body", None).expect("build");
    assert!(scores(&graph, &nodes, "marmoset")[1].expect("indexed") > 0.0);

    assert!(graph.set_node_property(nodes[1], "body", Value::String("entirely new words".into())));
    build_text_index(&mut graph, "Doc", "body", None).expect("rebuild");

    assert_eq!(
        scores(&graph, &nodes, "marmoset")[1],
        Some(0.0),
        "a rebuild is the refresh; the old term must be gone"
    );
    assert!(scores(&graph, &nodes, "entirely")[1].expect("indexed") > 0.0);
    assert_eq!(graph.text_indexes.len(), 1, "a rebuild replaces, not adds");
    assert!(store(&graph, "body").validate().is_ok());
}

#[test]
fn drop_reports_whether_an_index_existed() {
    let (mut graph, _) = memory_corpus();
    build_text_index(&mut graph, "Doc", "body", None).expect("build");
    assert!(has_text_index(&graph, "Doc", "body"));

    assert!(drop_text_index(&mut graph, "Doc", "body"));
    assert!(!has_text_index(&graph, "Doc", "body"));
    assert!(
        !drop_text_index(&mut graph, "Doc", "body"),
        "dropping nothing reports nothing"
    );
}

#[test]
fn an_unknown_type_or_an_unindexable_property_errors_loudly() {
    let (mut graph, _) = memory_corpus();

    let unknown_type = build_text_index(&mut graph, "Nope", "body", None).unwrap_err();
    assert!(
        unknown_type.contains("Unknown node type 'Nope'"),
        "{unknown_type}"
    );

    let unknown_property = build_text_index(&mut graph, "Doc", "bdoy", None).unwrap_err();
    assert!(
        unknown_property.contains("No 'Doc' node carries text or a string/null list for 'bdoy'"),
        "{unknown_property}"
    );
    assert!(
        graph.text_indexes.is_empty(),
        "a refused build installs nothing"
    );
}

#[test]
fn a_type_with_no_nodes_yet_builds_an_empty_index() {
    let mut graph = DirGraph::new();
    graph.type_indices.entry_or_default("Doc".to_string());

    let report = build_text_index(&mut graph, "Doc", "body", None).expect("declare before ingest");

    assert_eq!(report.indexed, 0);
    assert_eq!(store(&graph, "body").documents(), 0);
}

#[test]
fn disk_mode_refuses_and_names_the_modes_that_work() {
    let dir = tempfile::tempdir().unwrap();
    let mut graph = new_dir_graph_in_mode(StorageMode::Disk, Some(dir.path())).expect("disk graph");
    assert!(GraphRead::is_disk(&graph.graph));

    let err = build_text_index(&mut graph, "Doc", "body", None).unwrap_err();

    assert!(
        err.contains("not supported on a disk-backed graph"),
        "{err}"
    );
    assert!(err.contains("'mapped'"), "{err}");
    assert!(graph.text_indexes.is_empty());
}

#[test]
fn mapped_mode_ranks_identically_to_memory() {
    let (memory, memory_nodes) = {
        let (mut graph, nodes) = memory_corpus();
        build_text_index(&mut graph, "Doc", "body", None).expect("build");
        (graph, nodes)
    };

    let mut mapped = new_dir_graph_in_mode(StorageMode::Mapped, None).expect("mapped graph");
    let mapped_nodes = corpus(&mut mapped);
    mapped.build_id_index("Doc");
    build_text_index(&mut mapped, "Doc", "body", None).expect("build");

    assert_eq!(
        scores(&memory, &memory_nodes, "quick brown"),
        scores(&mapped, &mapped_nodes, "quick brown"),
        "the two portable backends must rank the same corpus identically"
    );
}

#[test]
fn deleting_a_node_prunes_its_document() {
    let (mut graph, nodes) = memory_corpus();
    build_text_index(&mut graph, "Doc", "body", None).expect("build");

    crate::graph::mutation::maintain::detach_delete_nodes(&mut graph, &HashSet::from([nodes[1]]));

    let store = store(&graph, "body");
    assert!(!store.contains_node(nodes[1]));
    assert_eq!(store.documents(), 2);
    assert!(
        store.validate().is_ok(),
        "the two views must still agree after a prune"
    );
    assert_eq!(
        store.score(nodes[1], &store.prepare_query("marmoset")),
        None,
        "a deleted node is unindexed, not zero-scoring"
    );
}

/// The riskiest coupling in the lane: `StableDiGraph` hands a freed
/// `NodeIndex` to the next node created, and a text document is addressed *by*
/// that index. A document left behind is therefore not stale bookkeeping — it
/// is inherited, and the new node comes back scoring content it never had.
#[test]
fn a_reused_node_index_does_not_inherit_the_deleted_document() {
    let (mut graph, nodes) = memory_corpus();
    build_text_index(&mut graph, "Doc", "body", None).expect("build");
    let doomed = nodes[1];

    crate::graph::mutation::maintain::detach_delete_nodes(&mut graph, &HashSet::from([doomed]));
    let reused = push_doc(
        &mut graph,
        99,
        "fresh",
        Value::String("completely unrelated".into()),
    );

    assert_eq!(
        reused, doomed,
        "the fixture only proves anything if petgraph actually recycled the index"
    );
    let store = store(&graph, "body");
    assert_eq!(
        store.score(reused, &store.prepare_query("marmoset")),
        None,
        "the new node is unindexed — it must not inherit the deleted document's score"
    );
    assert!(!store.contains_node(reused));
    assert!(store.validate().is_ok());
}

#[test]
fn vacuum_drops_text_indexes_wholesale() {
    let (mut graph, nodes) = memory_corpus();
    crate::graph::mutation::maintain::detach_delete_nodes(&mut graph, &HashSet::from([nodes[0]]));
    build_text_index(&mut graph, "Doc", "body", None).expect("build");
    assert!(has_text_index(&graph, "Doc", "body"));

    graph.vacuum();

    assert!(
        !has_text_index(&graph, "Doc", "body"),
        "vacuum remaps every NodeIndex, so every document would point at the wrong node"
    );
}

#[test]
fn a_fork_indexes_independently_of_its_parent() {
    let (mut parent, nodes) = memory_corpus();
    build_text_index(&mut parent, "Doc", "body", None).expect("build");

    let mut fork = parent.independent_copy();
    crate::graph::mutation::maintain::detach_delete_nodes(&mut fork, &HashSet::from([nodes[1]]));

    assert_eq!(store(&fork, "body").documents(), 2);
    assert_eq!(
        store(&parent, "body").documents(),
        3,
        "the parent's index must not see the fork's delete"
    );
    assert!(drop_text_index(&mut fork, "Doc", "body"));
    assert!(
        has_text_index(&parent, "Doc", "body"),
        "and must not see the fork's drop"
    );
}

#[test]
fn show_indexes_reports_a_text_index_under_its_canonical_name() {
    let (mut graph, _) = memory_corpus();
    graph.create_index("Doc", "body");
    build_text_index(&mut graph, "Doc", "body", None).expect("build");

    let rows = collect_indexes_structured(&graph);
    let text: Vec<_> = rows
        .iter()
        .filter(|info| info.kind == crate::graph::introspection::schema_overview::IndexKind::Text)
        .collect();

    assert_eq!(text.len(), 1);
    assert_eq!(text[0].name, "Doc.body");
    assert_eq!(text[0].properties, vec!["body".to_string()]);
    assert_eq!(text[0].kind.neo4j_type(), "FULLTEXT");
    assert_eq!(
        rows.iter().filter(|info| info.name == "Doc.body").count(),
        2,
        "the equality index and the text index share one canonical name"
    );
}

#[test]
fn list_text_indexes_is_ordered_by_type_then_property() {
    let mut graph = DirGraph::new();
    corpus(&mut graph);
    push_doc(&mut graph, 4, "d", Value::String("another".into()));
    build_text_index(&mut graph, "Doc", "body", None).expect("build");
    build_text_index(&mut graph, "Doc", "title", None).expect("build");

    let listed: Vec<&str> = list_text_indexes(&graph)
        .iter()
        .map(|(_, property, _)| *property)
        .collect();

    assert_eq!(listed, vec!["body", "title"]);
}
