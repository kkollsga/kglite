//! Freshness of the persistent disk index bundles.
//!
//! A `property_index_*` / `global_index_*` bundle is an mmap snapshot: nothing
//! maintains it after the build, so the only sound thing it can say about a
//! graph that has moved is *nothing*. These cases pin `None` ("unknown, go
//! scan") rather than `Some(vec![])` ("proven empty") for every way a disk
//! graph can move under a bundle, and pin the query answer that a scan then
//! produces.
//!
//! Deep-scan 2026-09-07 items 2+3 (lane report F2/F3): before this, a
//! save+load armed the auto-built `title`/`nid` globals and every later
//! `name`/`title` lookup missed the rows added since, and a user
//! `create_index` was trusted as authoritative forever.

use crate::datatypes::{DataFrame, Value};
use crate::graph::session::execute::{execute_mut, execute_read, ExecuteOptions};
use crate::graph::DirGraph;
use std::collections::HashMap;
use tempfile::TempDir;

/// `Doc` rows as `(id, name, tag)`; `name` is the title column, so `n.name`
/// and `n.title` both resolve through the auto-built global `title` bundle.
fn add_docs(graph: &mut DirGraph, rows: &[(i64, &str, &str)]) {
    let frame = DataFrame::from_cypher_rows(
        vec!["id".to_string(), "name".to_string(), "tag".to_string()],
        rows.iter()
            .map(|(id, name, tag)| {
                vec![
                    Value::Int64(*id),
                    Value::String((*name).to_string()),
                    Value::String((*tag).to_string()),
                ]
            })
            .collect(),
    )
    .unwrap();
    crate::graph::mutation::maintain::add_nodes(
        graph,
        frame,
        "Doc".to_string(),
        "id".to_string(),
        Some("name".to_string()),
        None,
    )
    .unwrap();
}

fn run(graph: &mut DirGraph, query: &str) {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("{query}: {e}"));
}

fn rows(graph: &DirGraph, query: &str) -> usize {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_read(graph, query, &opts)
        .unwrap_or_else(|e| panic!("{query}: {e}"))
        .result
        .rows
        .len()
}

/// A published disk graph at `path` holding `rows`, live on the mapped
/// generation — the state `kglite.open(path)` reaches.
fn published_disk_graph(path: &std::path::Path, docs: &[(i64, &str, &str)]) -> DirGraph {
    let mut graph = DirGraph::new();
    if !docs.is_empty() {
        add_docs(&mut graph, docs);
    }
    graph.enable_disk_mode().unwrap();
    graph.save_disk(path.to_str().unwrap()).unwrap();
    graph
}

fn typed_lookup(graph: &DirGraph, property: &str, value: &str) -> Option<Vec<usize>> {
    graph
        .graph
        .as_disk()
        .expect("disk backend")
        .lookup_property_eq("Doc", property, value)
        .map(|hits| hits.into_iter().map(|idx| idx.index()).collect())
}

fn global_lookup(graph: &DirGraph, property: &str, value: &str) -> Option<Vec<usize>> {
    graph
        .graph
        .as_disk()
        .expect("disk backend")
        .lookup_global_eq(property, value)
        .map(|hits| hits.into_iter().map(|idx| idx.index()).collect())
}

/// F3, index level: a row created after the build is not in the bundle, so the
/// bundle cannot answer "no such row".
#[test]
fn a_creation_after_the_build_makes_the_typed_bundle_decline() {
    let dir = TempDir::new().unwrap();
    let mut graph = published_disk_graph(dir.path(), &[(1, "a", "c0"), (2, "b", "c1")]);
    let (_count, persistent) = graph.create_property_index_routed("Doc", "tag").unwrap();
    assert!(persistent, "a disk create_index builds the mmap bundle");
    assert_eq!(
        typed_lookup(&graph, "tag", "c1"),
        Some(vec![1]),
        "a bundle built over the current graph answers"
    );

    add_docs(&mut graph, &[(3, "c", "c1")]);

    assert_eq!(
        typed_lookup(&graph, "tag", "c1"),
        None,
        "the bundle no longer covers every row, so it must decline to the scan"
    );
    assert_eq!(
        rows(&graph, "MATCH (n:Doc) WHERE n.tag = 'c1' RETURN n.id"),
        2,
        "the scan finds both c1 rows"
    );
}

/// F3, via the matcher: a `SET` moves a row between buckets the bundle still
/// records under the old value.
#[test]
fn a_set_on_the_indexed_property_makes_the_typed_bundle_decline() {
    let dir = TempDir::new().unwrap();
    let mut graph = published_disk_graph(dir.path(), &[(1, "a", "c0"), (2, "b", "c1")]);
    graph.create_property_index_routed("Doc", "tag").unwrap();

    run(&mut graph, "MATCH (n:Doc) WHERE n.id = 1 SET n.tag = 'c1'");

    assert_eq!(
        typed_lookup(&graph, "tag", "c1"),
        None,
        "the row moved into 'c1' after the build"
    );
    assert_eq!(
        rows(&graph, "MATCH (n:Doc) WHERE n.tag = 'c1' RETURN n.id"),
        2
    );
    assert_eq!(
        rows(&graph, "MATCH (n:Doc) WHERE n.tag = 'c0' RETURN n.id"),
        0,
        "and the row it left is no longer under the old value"
    );
}

/// F2: every disk `save()` auto-builds the global `title` bundle, so a
/// save+load arms this on a graph whose owner never asked for an index.
#[test]
fn a_creation_after_a_load_makes_the_global_title_bundle_decline() {
    let dir = TempDir::new().unwrap();
    drop(published_disk_graph(dir.path(), &[]));
    let mut handle = crate::graph::io::file::load_file(dir.path().to_str().unwrap()).unwrap();
    let graph = crate::graph::handle::make_dir_graph_mut(&mut handle);

    add_docs(graph, &[(6, "nm_6", "c0"), (19, "nm_7", "c1")]);

    assert_eq!(
        global_lookup(graph, "title", "nm_7"),
        None,
        "the bundle was built over an empty graph"
    );
    assert_eq!(
        rows(graph, "MATCH (n:Doc) WHERE n.name = 'nm_7' RETURN n.id"),
        1
    );
    assert_eq!(
        rows(graph, "MATCH (n:Doc {name: 'nm_7'}) RETURN n.id"),
        1,
        "the map form routes through the same global bundle"
    );
    assert_eq!(
        rows(graph, "MATCH (n:Doc) WHERE n.title = 'nm_7' RETURN n.id"),
        1
    );
}

/// A `SET` of the title column after a load is the same defect through the
/// global bundle's other arm.
#[test]
fn a_title_set_after_a_load_makes_the_global_bundle_decline() {
    let dir = TempDir::new().unwrap();
    drop(published_disk_graph(
        dir.path(),
        &[(6, "nm_6", "c0"), (19, "nm_6", "c1")],
    ));
    let mut handle = crate::graph::io::file::load_file(dir.path().to_str().unwrap()).unwrap();
    let graph = crate::graph::handle::make_dir_graph_mut(&mut handle);

    run(graph, "MATCH (n:Doc) WHERE n.id = 19 SET n.name = 'nm_7'");

    assert_eq!(global_lookup(graph, "title", "nm_7"), None);
    assert_eq!(
        rows(graph, "MATCH (n:Doc) WHERE n.name = 'nm_7' RETURN n.id"),
        1
    );
}

/// Deletion must mark the bundle stale on its own. Relying on the tombstone
/// filter downstream would leave the *value* half of a delete+re-add — a slot
/// recycled under a different value — answered from the stale bundle.
#[test]
fn a_delete_and_re_add_under_a_new_value_makes_the_typed_bundle_decline() {
    let dir = TempDir::new().unwrap();
    let mut graph = published_disk_graph(dir.path(), &[(1, "a", "c0"), (2, "b", "c1")]);
    graph.create_property_index_routed("Doc", "tag").unwrap();

    run(&mut graph, "MATCH (n:Doc) WHERE n.id = 2 DETACH DELETE n");
    assert_eq!(
        typed_lookup(&graph, "tag", "c1"),
        None,
        "a deletion is a change the bundle does not carry"
    );

    add_docs(&mut graph, &[(4, "d", "c0")]);
    assert_eq!(
        rows(&graph, "MATCH (n:Doc) WHERE n.tag = 'c0' RETURN n.id"),
        2
    );
    assert_eq!(
        rows(&graph, "MATCH (n:Doc) WHERE n.tag = 'c1' RETURN n.id"),
        0
    );
}

/// `reindex()` is the documented repair verb, and on a disk graph it rebuilt
/// only the empty heap maps — a measured no-op (F2).
#[test]
fn reindex_rebuilds_the_disk_bundles() {
    let dir = TempDir::new().unwrap();
    let mut graph = published_disk_graph(dir.path(), &[(1, "a", "c0"), (2, "b", "c1")]);
    graph.create_property_index_routed("Doc", "tag").unwrap();
    add_docs(&mut graph, &[(3, "c", "c1")]);
    assert_eq!(typed_lookup(&graph, "tag", "c1"), None);

    graph.reindex();

    let hits = typed_lookup(&graph, "tag", "c1").expect("the rebuilt bundle answers again");
    assert_eq!(hits.len(), 2, "and it covers the row added since the build");
    assert_eq!(
        global_lookup(&graph, "title", "c").map(|hits| hits.len()),
        Some(1),
        "the global bundles are rebuilt too"
    );
}

/// A stale bundle must not cross a generation: the next process would open it
/// with no memory that it was stale.
#[test]
fn a_save_never_carries_a_stale_bundle_into_the_new_generation() {
    let source = TempDir::new().unwrap();
    let destination = TempDir::new().unwrap();
    let mut graph = published_disk_graph(source.path(), &[(1, "a", "c0"), (2, "b", "c1")]);
    graph.create_property_index_routed("Doc", "tag").unwrap();
    add_docs(&mut graph, &[(3, "c", "c1")]);

    let mut handle = std::sync::Arc::new(graph);
    crate::graph::io::file::save_graph(&mut handle, destination.path().to_str().unwrap()).unwrap();
    drop(handle);
    let reloaded = crate::graph::io::file::load_file(destination.path().to_str().unwrap()).unwrap();

    match typed_lookup(&reloaded, "tag", "c1") {
        Some(hits) => assert_eq!(hits.len(), 2, "a carried bundle must be complete"),
        None => {}
    }
    assert_eq!(
        rows(&reloaded, "MATCH (n:Doc) WHERE n.tag = 'c1' RETURN n.id"),
        2
    );
}

/// save → load → mutate → save → load: the generation carry seen twice, which
/// is what a long-running ingest server does.
#[test]
fn generation_carry_survives_a_second_save_and_load() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap().to_string();
    drop(published_disk_graph(dir.path(), &[(1, "nm_1", "c0")]));

    let mut handle = crate::graph::io::file::load_file(&path).unwrap();
    add_docs(
        crate::graph::handle::make_dir_graph_mut(&mut handle),
        &[(2, "nm_2", "c1")],
    );
    crate::graph::io::file::save_graph(&mut handle, &path).unwrap();
    drop(handle);

    let reloaded = crate::graph::io::file::load_file(&path).unwrap();
    assert_eq!(
        global_lookup(&reloaded, "title", "nm_2").map(|hits| hits.len()),
        Some(1),
        "the save rebuilt the global title bundle over the final node set"
    );
    assert_eq!(
        rows(&reloaded, "MATCH (n:Doc) WHERE n.name = 'nm_2' RETURN n.id"),
        1
    );
    assert_eq!(
        rows(&reloaded, "MATCH (n:Doc) WHERE n.name = 'nm_1' RETURN n.id"),
        1
    );
}
