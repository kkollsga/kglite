//! `MappedGraph`'s lazy property indexes must answer for the graph as it is
//! *now*, and only for rows they actually cover.
//!
//! Both halves are silent-wrong-answer classes, because
//! `lookup_by_property_eq` is authoritative at its caller: the matcher returns
//! the block's answer verbatim (`core/pattern_matching/matcher.rs`) and only a
//! `None` — which the backend reports as an *empty* block — falls through to a
//! scan. A block that is stale, or that covers half its type's rows, is
//! therefore a wrong `MATCH` result rather than a slow one.
//!
//! The fixture is the shape that makes the index live at all: nodes whose
//! properties are row-storage (`PropertyStorage::Map`) on a `Mapped` backend.
//! `kglite::api::io::load_rdf` builds exactly that when handed a mapped
//! `DirGraph` (`io/rdf/loader.rs` → `NodeData::new` → `GraphWrite::add_node`);
//! the wheel and the C ABI both hand it a `DirGraph::new()`, so today that
//! pairing is reachable from the Rust API only. An all-columnar mapped graph
//! indexes nothing — see [`a_columnar_row_keeps_its_types_index_off`].

use std::collections::HashMap;

use crate::datatypes::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::schema::NodeData;
use crate::graph::session::{execute_mut, execute_read, ExecuteOptions};
use crate::graph::storage::{GraphRead, GraphWrite};

fn run(graph: &mut DirGraph, query: &str) {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("query failed: {query}: {e}"));
}

fn ids(graph: &DirGraph, query: &str) -> Vec<Value> {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_read(graph, query, &opts)
        .unwrap_or_else(|e| panic!("{query}: {e}"))
        .result
        .rows
        .into_iter()
        .filter_map(|r| r.into_iter().next())
        .collect()
}

/// `n` row-storage `Person` nodes on a `Mapped` backend, `name` = `name-<i>`.
fn mapped_row_storage_people(n: i64) -> DirGraph {
    let mut g = crate::graph::storage::mode::new_dir_graph_in_mode(
        crate::graph::storage::mode::StorageMode::Mapped,
        None,
    )
    .expect("mapped backend must be constructible");
    for i in 1..=n {
        let mut props = HashMap::new();
        props.insert("name".to_string(), Value::String(format!("name-{i}")));
        let nd = NodeData::new(
            Value::Int64(i),
            Value::String(format!("t{i}")),
            "Person".to_string(),
            props,
            &mut g.interner,
        );
        let idx = GraphWrite::add_node(&mut g.graph, nd);
        g.type_indices
            .entry_or_default("Person".to_string())
            .push(idx);
        g.id_indices
            .entry_or_default("Person".to_string())
            .insert(Value::Int64(i), idx);
    }
    assert!(
        g.graph
            .lookup_by_property_eq("Person", "name", "name-1")
            .is_some_and(|hits| !hits.is_empty()),
        "fixture must produce a live property index, or every assertion below \
         passes through a full scan and tests nothing"
    );
    g
}

/// `SET` through the shared property writers must not leave the index mapping
/// the overwritten value.
///
/// Red-first: without the `note_property_write` hook the first assertion
/// returns `[Int64(1)]` (the pre-`SET` value still resolves) and the second
/// returns `[]` (the written value does not).
#[test]
fn a_set_invalidates_the_mapped_property_index() {
    let mut g = mapped_row_storage_people(3);
    assert_eq!(
        ids(&g, "MATCH (p:Person {name:'name-1'}) RETURN p.id"),
        vec![Value::Int64(1)],
        "precondition: the index answers before the write"
    );

    run(&mut g, "MATCH (p:Person) WHERE p.id = 1 SET p.name = 'Bob'");

    assert!(
        ids(&g, "MATCH (p:Person {name:'name-1'}) RETURN p.id").is_empty(),
        "the overwritten value must stop matching"
    );
    assert_eq!(
        ids(&g, "MATCH (p:Person {name:'Bob'}) RETURN p.id"),
        vec![Value::Int64(1)],
        "the written value must match"
    );
}

/// The `REMOVE` route (`remove_node_property`) is a separate writer in the
/// same macro and invalidates on its own account.
#[test]
fn a_remove_invalidates_the_mapped_property_index() {
    let mut g = mapped_row_storage_people(3);
    assert_eq!(
        ids(&g, "MATCH (p:Person {name:'name-2'}) RETURN p.id"),
        vec![Value::Int64(2)]
    );

    run(&mut g, "MATCH (p:Person) WHERE p.id = 2 REMOVE p.name");

    assert!(
        ids(&g, "MATCH (p:Person {name:'name-2'}) RETURN p.id").is_empty(),
        "a removed property must stop matching"
    );
}

/// The global (untyped) index is built by the same function and dropped by the
/// same hook, so an untyped pattern must see the write too.
#[test]
fn a_set_invalidates_the_mapped_global_property_index() {
    let mut g = mapped_row_storage_people(3);
    assert_eq!(
        g.graph
            .lookup_by_property_eq_any_type("name", "name-3")
            .expect("global index must be live for the fixture"),
        vec![petgraph::graph::NodeIndex::new(2)],
        "precondition: the global index answers before the write"
    );

    run(&mut g, "MATCH (p:Person) WHERE p.id = 3 SET p.name = 'Zoe'");

    assert_eq!(
        g.graph.lookup_by_property_eq_any_type("name", "name-3"),
        Some(Vec::new()),
        "the overwritten value must stop resolving through the global index"
    );
    assert_eq!(
        g.graph.lookup_by_property_eq_any_type("name", "Zoe"),
        Some(vec![petgraph::graph::NodeIndex::new(2)]),
        "the written value must resolve through the global index"
    );
}

/// A columnar row's values live in the type's `ColumnStore`, which the index
/// build does not read — so a type with one turns its index **off** rather
/// than publishing a block that covers only the row-storage half.
///
/// Red-first: without the bail, `MATCH (p:Person {name:'name-9'})` returns
/// `[]` for a node that exists, because the block built from the two
/// row-storage nodes is non-empty and the matcher trusts it verbatim.
#[test]
fn a_columnar_row_keeps_its_types_index_off() {
    let mut g = mapped_row_storage_people(2);
    run(&mut g, "CREATE (:Person {id: 9, name: 'name-9'})");

    assert!(
        g.graph
            .lookup_by_property_eq("Person", "name", "name-1")
            .is_none(),
        "a partially-covered index must report 'no index' so the matcher scans"
    );
    assert_eq!(
        ids(&g, "MATCH (p:Person {name:'name-9'}) RETURN p.name"),
        vec![Value::String("name-9".into())],
        "the columnar row must still be found"
    );
    assert_eq!(
        ids(&g, "MATCH (p:Person {name:'name-1'}) RETURN p.id"),
        vec![Value::Int64(1)],
        "and the row-storage nodes must still be found"
    );
}
