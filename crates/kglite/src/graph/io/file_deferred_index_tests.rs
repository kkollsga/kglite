//! Deferred load-time index rebuild: the contract that a `.kgl` loaded with
//! `defer_index_rebuild` answers *identically* to one loaded eagerly.
//!
//! The bug class these guard is the 0.16.13 soft-alias one — an index consulted
//! as authoritative while it is incomplete. The deferral's whole safety
//! argument is that it never produces that state: while deferred the four index
//! maps are empty, which is what a graph declaring no index looks like, and a
//! miss there falls back to a scan. Every test below is an instance of the same
//! question: *can a reader tell?*

use super::*;
use crate::api::cypher::CypherResult;
use crate::graph::dir_graph::DirGraph;
use crate::graph::io::file::defer_indexes::DeferIndexRebuild;
use crate::graph::session::execute::{execute_mut, execute_read, ExecuteOptions};
use std::collections::HashMap;

fn run(graph: &mut DirGraph, query: &str) {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("setup query failed: {query}: {e}"));
}

fn read(graph: &DirGraph, query: &str) -> CypherResult {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_read(graph, query, &opts)
        .unwrap_or_else(|e| panic!("query failed: {query}: {e}"))
        .result
}

/// Rows as comparable text, so an eager and a deferred load can be asserted
/// equal without depending on `Value`'s `PartialEq` across column shapes.
fn rows(result: &CypherResult) -> Vec<String> {
    let mut out: Vec<String> = result.rows.iter().map(|row| format!("{:?}", row)).collect();
    out.sort();
    out
}

/// 60 `:Item` nodes carrying every index family the load path rebuilds:
/// a UNIQUE constraint on `sku`, equality indexes on `category` and `region`,
/// a composite over both, and a range index on `score`.
fn indexed_fixture() -> DirGraph {
    let mut graph = DirGraph::new();
    for i in 0..60u32 {
        run(
            &mut graph,
            &format!(
                "CREATE (:Item {{id: {i}, sku: 'sku-{i}', category: 'cat-{}', \
                 region: 'reg-{}', score: {i}}})",
                i % 5,
                i % 3
            ),
        );
    }
    run(
        &mut graph,
        "CREATE CONSTRAINT item_sku FOR (n:Item) REQUIRE n.sku IS UNIQUE",
    );
    run(&mut graph, "CREATE INDEX FOR (n:Item) ON (n.category)");
    run(&mut graph, "CREATE INDEX FOR (n:Item) ON (n.region)");
    run(
        &mut graph,
        "CREATE INDEX FOR (n:Item) ON (n.category, n.region)",
    );
    run(&mut graph, "CREATE RANGE INDEX FOR (n:Item) ON (n.score)");
    graph
}

fn write_fixture(dir: &std::path::Path) -> String {
    let path = dir.join("indexed.kgl");
    let mut arc = Arc::new(indexed_fixture());
    prepare_save(&mut arc);
    Arc::make_mut(&mut arc).enable_columnar();
    write_kgl(&arc, path.to_str().unwrap()).unwrap();
    path.to_str().unwrap().to_string()
}

fn load_eager(path: &str) -> Arc<DirGraph> {
    let _scope = DeferIndexRebuild::scoped(false);
    load_file(path).unwrap()
}

fn load_deferred(path: &str) -> Arc<DirGraph> {
    let _scope = DeferIndexRebuild::scoped(true);
    let graph = load_file(path).unwrap();
    assert!(
        graph.indexes_deferred(),
        "the scoped override must have produced a deferred load"
    );
    graph
}

/// Every read shape the four index families can serve, plus the scan shapes
/// they must not change.
const SHAPES: &[&str] = &[
    "MATCH (n:Item {category: 'cat-2'}) RETURN n.sku AS sku",
    "MATCH (n:Item) WHERE n.category = 'cat-3' RETURN n.sku AS sku",
    "MATCH (n:Item {category: 'cat-1', region: 'reg-2'}) RETURN n.sku AS sku",
    "MATCH (n:Item) WHERE n.category = 'cat-0' AND n.region = 'reg-1' RETURN n.sku AS sku",
    "MATCH (n:Item) WHERE n.score > 40 RETURN n.sku AS sku",
    "MATCH (n:Item) WHERE n.score >= 10 AND n.score < 20 RETURN n.sku AS sku",
    "MATCH (n:Item {sku: 'sku-17'}) RETURN n.id AS id",
    // A value no row carries: the arm that returns rows when the index is
    // absent and *proven empty* when it is present. The two must agree.
    "MATCH (n:Item {category: 'nope'}) RETURN n.sku AS sku",
    "MATCH (n:Item) WHERE n.category IN ['cat-1', 'cat-4'] RETURN n.sku AS sku",
    "MATCH (n:Item) RETURN count(n) AS c",
];

#[test]
fn deferred_load_answers_every_indexed_shape_identically() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path());
    let eager = load_eager(&path);
    let deferred = load_deferred(&path);

    for query in SHAPES {
        let expected = read(&eager, query);
        let actual = read(&deferred, query);
        assert_eq!(
            rows(&expected),
            rows(&actual),
            "deferred load diverged from eager on: {query}"
        );
        // A shape that returns nothing on both sides proves nothing about
        // index equivalence, so the corpus must not silently become vacuous.
        if !query.contains("'nope'") {
            assert!(
                !expected.rows.is_empty(),
                "fixture query returns no rows, making the comparison vacuous: {query}"
            );
        }
    }
}

/// The laziness itself: nothing is built at load, and the declarations that
/// describe what to build survive intact.
#[test]
fn deferred_load_builds_nothing_and_keeps_every_declaration() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path());
    let eager = load_eager(&path);
    let deferred = load_deferred(&path);

    assert!(
        !eager.property_indices.is_empty()
            && !eager.composite_indices.is_empty()
            && !eager.range_indices.is_empty()
            && !eager.unique_indices.is_empty(),
        "the eager control must actually build all four families"
    );
    assert!(
        deferred.property_indices.is_empty()
            && deferred.composite_indices.is_empty()
            && deferred.range_indices.is_empty()
            && deferred.unique_indices.is_empty(),
        "a deferred load must build nothing"
    );

    assert_eq!(deferred.property_index_keys, eager.property_index_keys);
    assert_eq!(deferred.composite_index_keys, eager.composite_index_keys);
    assert_eq!(deferred.range_index_keys, eager.range_index_keys);

    // ...and materializing on demand reaches the eager state exactly.
    let mut materialized = deferred;
    assert!(Arc::make_mut(&mut materialized).materialize_indexes());
    assert!(!materialized.indexes_deferred());
    assert_eq!(
        materialized.list_indexes().len(),
        eager.list_indexes().len()
    );
    assert_eq!(
        materialized.list_composite_indexes(),
        eager.list_composite_indexes()
    );
    assert_eq!(
        materialized.list_unique_constraints(),
        eager.list_unique_constraints()
    );
    assert!(!Arc::make_mut(&mut materialized).materialize_indexes());
}

/// While deferred, no index may report itself as *present*. This is the
/// invariant the whole design rests on: the matcher treats a value-miss on a
/// covered index as proven-empty, so a `has_index` that answered from the
/// declaration list would turn every indexed lookup into zero rows.
#[test]
fn deferred_state_never_claims_an_index_is_present() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path());
    let deferred = load_deferred(&path);

    assert!(!deferred.has_index("Item", "category"));
    assert!(!deferred.has_composite_index("Item", &["category".to_string(), "region".to_string()]));
    assert!(deferred.list_indexes().is_empty());
    assert!(deferred.list_composite_indexes().is_empty());
    assert!(!deferred.has_unique_constraints());
    assert!(deferred
        .lookup_by_index("Item", "category", &Value::String("cat-1".into()))
        .is_none());
    assert!(deferred
        .lookup_by_composite_index(
            "Item",
            &["category".to_string(), "region".to_string()],
            &[Value::String("cat-1".into()), Value::String("reg-1".into())]
        )
        .is_none());
}

/// A write arriving before the build must leave the index correct. The build
/// runs at `&mut` acquisition, *before* the write, so the write is filed
/// incrementally — building afterwards from live data would be correct too, but
/// building in the middle would double-count the row.
#[test]
fn write_before_first_indexed_read_leaves_indexes_correct() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path());
    let mut deferred = load_deferred(&path);

    let graph = crate::graph::handle::make_dir_graph_mut(&mut deferred);
    assert!(
        !graph.indexes_deferred(),
        "acquiring a mutable graph must materialize before the caller writes"
    );
    run(
        graph,
        "CREATE (:Item {id: 900, sku: 'sku-900', category: 'cat-2', \
         region: 'reg-1', score: 99})",
    );
    run(
        graph,
        "MATCH (n:Item {sku: 'sku-16'}) SET n.category = 'cat-2'",
    );

    // Bucket membership, not just row counts: a double-filed node shows up as a
    // duplicated row here and nowhere else.
    let hits = read(
        &deferred,
        "MATCH (n:Item {category: 'cat-2'}) RETURN n.sku AS sku",
    );
    let mut skus = rows(&hits);
    skus.dedup();
    assert_eq!(
        skus.len(),
        hits.rows.len(),
        "an indexed lookup returned a node twice — the index was filed into mid-build"
    );

    // The same answer a scan gives.
    let scanned = read(
        &deferred,
        "MATCH (n:Item) WHERE n.category = 'cat-2' RETURN n.sku AS sku",
    );
    assert_eq!(rows(&hits), rows(&scanned));
    // 12 rows at `cat-2` in the fixture, plus the created one, plus the moved one.
    assert_eq!(hits.rows.len(), 14, "fixture drifted");
}

/// The unique constraint must still reject a duplicate. Deferred, the
/// enforcement map is empty and `unique_claims` would report no claims — so
/// this is the test that proves the write path materializes first.
#[test]
fn deferred_load_still_enforces_unique_constraints() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path());
    let mut deferred = load_deferred(&path);

    let graph = crate::graph::handle::make_dir_graph_mut(&mut deferred);
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    let outcome = execute_mut(
        graph,
        "CREATE (:Item {id: 901, sku: 'sku-17', category: 'cat-0', region: 'reg-0', score: 1})",
        &opts,
    );
    assert!(
        outcome.is_err(),
        "a duplicate `sku` was admitted after a deferred load — the unique \
         constraint was not materialized before the write"
    );
    // And a non-duplicate still lands.
    execute_mut(
        graph,
        "CREATE (:Item {id: 902, sku: 'sku-902', category: 'cat-0', region: 'reg-0', score: 1})",
        &opts,
    )
    .unwrap();
}

/// Saving a deferred graph must persist the declarations, not the empty maps.
/// `populate_index_keys` snapshots the live maps; without its deferred guard
/// this save writes a file with no indexes and no constraint names — silent
/// loss of every DDL declaration the user made.
#[test]
fn saving_a_deferred_graph_preserves_declarations_byte_for_byte() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path());

    let resave = |graph: Arc<DirGraph>, name: &str| -> Vec<u8> {
        let out = dir.path().join(name);
        let mut arc = graph;
        prepare_save(&mut arc);
        crate::graph::handle::make_dir_graph_mut_preserving_lineage(&mut arc).enable_columnar();
        write_kgl(&arc, out.to_str().unwrap()).unwrap();
        std::fs::read(&out).unwrap()
    };

    let from_eager = resave(load_eager(&path), "eager.kgl");
    let deferred = load_deferred(&path);
    let from_deferred = resave(deferred, "deferred.kgl");
    assert_eq!(
        from_eager, from_deferred,
        "a deferred graph must re-save to the same bytes as an eager one"
    );

    // ...and the re-saved file still carries working indexes and constraints.
    let reloaded = load_eager(dir.path().join("deferred.kgl").to_str().unwrap());
    assert!(reloaded.has_index("Item", "category"));
    assert!(reloaded.has_unique_constraints());
    assert_eq!(reloaded.list_unique_constraints().len(), 1);
}

/// Concurrency. Materialization only ever happens behind a `&mut DirGraph`,
/// which Rust makes exclusive — so unlike `id_indices` (whose `&self` lazy
/// build needs a lock) there is no window in which two builders can race. What
/// *can* happen concurrently is many readers of the same deferred `Arc`, and
/// they must all agree with the eager answer.
#[test]
fn concurrent_first_readers_agree_with_eager() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path());
    let expected: Vec<Vec<String>> = {
        let eager = load_eager(&path);
        SHAPES.iter().map(|q| rows(&read(&eager, q))).collect()
    };
    let deferred = load_deferred(&path);

    std::thread::scope(|scope| {
        for _ in 0..8 {
            let graph = Arc::clone(&deferred);
            let expected = &expected;
            scope.spawn(move || {
                for (query, want) in SHAPES.iter().zip(expected.iter()) {
                    assert_eq!(&rows(&read(&graph, query)), want, "racing readers: {query}");
                }
            });
        }
    });
    assert!(
        deferred.indexes_deferred(),
        "reads must not have materialized anything"
    );
}

/// A second writer arriving after the first has materialized must not rebuild,
/// and must see the first writer's rows through the index.
#[test]
fn materialization_happens_once_and_later_writes_are_indexed() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path());
    let mut graph = load_deferred(&path);

    run(
        crate::graph::handle::make_dir_graph_mut(&mut graph),
        "CREATE (:Item {id: 903, sku: 'sku-903', category: 'cat-9', region: 'reg-0', score: 1})",
    );
    assert!(!graph.indexes_deferred());
    run(
        crate::graph::handle::make_dir_graph_mut(&mut graph),
        "CREATE (:Item {id: 904, sku: 'sku-904', category: 'cat-9', region: 'reg-0', score: 2})",
    );
    let hits = read(
        &graph,
        "MATCH (n:Item {category: 'cat-9'}) RETURN n.sku AS sku",
    );
    assert_eq!(hits.rows.len(), 2);
}

/// DDL against a deferred graph reached without `make_dir_graph_mut` — the
/// route an embedder holding an owned `DirGraph` takes. The entry point itself
/// materializes, so `DROP INDEX` reports the truth instead of "no such index".
#[test]
fn ddl_on_an_owned_deferred_graph_materializes_first() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path());
    let deferred = load_deferred(&path);
    let mut owned = Arc::try_unwrap(deferred).unwrap_or_else(|arc| (*arc).clone());
    assert!(owned.indexes_deferred());

    assert_eq!(owned.drop_index("Item", "category"), Ok(true));
    assert!(!owned.indexes_deferred());
    assert!(
        owned.has_index("Item", "region"),
        "the untouched index survived"
    );
}
