//! The schema half of a rollback: what the shell clone carries, and what it
//! must cost.
//!
//! The journal reverses the O(V+E) half; the *schema* half — the property
//! catalogue a statement grows on its way in — comes back only from the shell
//! clone. Since P2 that clone is a pointer copy of six `Arc`-shared maps
//! instead of a deep copy of them, so both halves of the contract need pinning
//! here:
//!
//! - **Restore fidelity** — a failed statement that introduced a property, a
//!   node type, or a connection type leaves none of them behind, live *or* in
//!   the file a later `save()` writes.
//! - **Cost** — the shell copies no map, and a statement forks a map only when
//!   it actually changes that map, at most once.

use super::*;
use crate::graph::cow::{reset_schema_map_forks, schema_map_forks};

// ─────────────────────────────────────────────────────────────────────
// Restore fidelity
// ─────────────────────────────────────────────────────────────────────

/// A failed `SET` that introduced a property must not leave it in the type's
/// metadata or in its shared `TypeSchema`.
///
/// The metadata is what `describe()`/`schema()` publish and what `save()`
/// writes, and it is also what the columnar column set is derived from on load
/// — so a leaked key is not cosmetic: it becomes a real, null-backfilled
/// column in the next `.kgl`.
#[test]
fn a_failed_set_restores_a_grown_property_catalogue() {
    let mut graph = seeded();
    assert!(
        !graph.node_type_metadata["Item"].contains_key("color"),
        "the fixture must not already declare the property, or this is vacuous"
    );

    assert_rolls_back(
        &mut graph,
        "MATCH (n:Item {id: 1}) SET n.color = 'red' \
         WITH n MATCH (m:Item {id: 2}) SET m.qty = duration({months: 2147483648})",
        None,
    );

    assert!(
        !graph.node_type_metadata["Item"].contains_key("color"),
        "a rolled-back SET left `color` in Item's property catalogue"
    );
}

/// The same statement on the columnar shape, where the property write also
/// grows the master store's schema — two independent restores that have to
/// agree.
#[test]
fn a_failed_columnar_set_restores_a_grown_property_catalogue() {
    let mut graph = seeded_columnar();

    assert_rolls_back(
        &mut graph,
        "MATCH (n:Item {id: 1}) SET n.color = 'red' \
         WITH n MATCH (m:Item {id: 2}) SET m.qty = duration({months: 2147483648})",
        None,
    );

    assert!(!graph.node_type_metadata["Item"].contains_key("color"));
}

/// A failed `CREATE` of a *new node type* must leave no trace of the type —
/// not in `node_type_metadata`, not in `type_schemas`.
#[test]
fn a_failed_create_restores_a_grown_type_catalogue() {
    let mut graph = seeded();
    assert!(!graph.node_type_metadata.contains_key("Widget"));

    assert_rolls_back(
        &mut graph,
        "CREATE (w:Widget {id: 1, name: 'w', spin: 3}) \
         WITH w MATCH (m:Item {id: 2}) SET m.qty = duration({months: 2147483648})",
        None,
    );

    assert!(
        !graph.node_type_metadata.contains_key("Widget"),
        "a rolled-back CREATE left the type in the property catalogue"
    );
    assert!(
        !graph.type_schemas.contains_key("Widget"),
        "a rolled-back CREATE left the type's shared TypeSchema behind"
    );
}

/// The edge side: a failed statement that introduced a connection type, an
/// endpoint pair, or an edge property must restore all three.
///
/// `connection_type_metadata` is a single map whose *values* carry the
/// endpoint sets and the property catalogue, so a shell that restored the keys
/// but shared the values would pass a keys-only check and fail this one.
#[test]
fn a_failed_edge_write_restores_connection_metadata() {
    let mut graph = seeded();
    assert!(!graph.connection_type_metadata.contains_key("RELATES"));
    assert!(
        !graph.connection_type_metadata["LINKS"]
            .property_types
            .contains_key("note"),
        "the fixture must not already declare the edge property"
    );

    // A new connection type, and a new property on an existing one.
    assert_rolls_back(
        &mut graph,
        "MATCH (a:Item {id: 1}), (b:Item {id: 3}) \
         CREATE (a)-[:RELATES {note: 'x'}]->(b) \
         WITH a MATCH (m:Item {id: 2}) SET m.qty = duration({months: 2147483648})",
        None,
    );
    assert!(
        !graph.connection_type_metadata.contains_key("RELATES"),
        "a rolled-back CREATE left the connection type behind"
    );

    assert_rolls_back(
        &mut graph,
        "MATCH (a:Item {id: 1})-[r:LINKS]->(b:Item {id: 2}) SET r.note = 'x' \
         WITH a MATCH (m:Item {id: 2}) SET m.qty = duration({months: 2147483648})",
        None,
    );
    assert!(
        !graph.connection_type_metadata["LINKS"]
            .property_types
            .contains_key("note"),
        "a rolled-back SET left an edge property in the connection catalogue"
    );
}

/// The metadata is *persisted*, so the restore has to hold through a
/// save/load round trip as well as in memory.
///
/// A live-only check cannot see a leak that lands in the file: the two are
/// written from the same map here, but `save()` also derives the columnar
/// column set from it, so a leaked key reappears as a materialized column with
/// null in every row. That is the shape of the bug this arm exists for.
#[test]
fn a_rolled_back_schema_growth_does_not_survive_save_and_load() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("rollback.kgl");
    let path = path.to_str().expect("utf-8 path");

    let mut graph = seeded();
    expect_failure(
        &mut graph,
        "CREATE (w:Widget {id: 1, name: 'w'}) \
         WITH w MATCH (n:Item {id: 1}) SET n.color = 'red' \
         WITH n MATCH (m:Item {id: 2}) SET m.qty = duration({months: 2147483648})",
        None,
    );

    let mut handle = std::sync::Arc::new(graph);
    crate::graph::io::file::save_graph(&mut handle, path).expect("save");
    let reloaded = crate::graph::io::file::load_file(path).expect("load");

    assert!(
        !reloaded.node_type_metadata.contains_key("Widget"),
        "a rolled-back type was written into the .kgl"
    );
    assert!(
        !reloaded.node_type_metadata["Item"].contains_key("color"),
        "a rolled-back property was written into the .kgl"
    );
}

/// A multi-row `SET` that introduces a property records it in the catalogue
/// **once**, and the record is right — after the statement commits, and absent
/// after it fails.
///
/// P6 hoisted the `(type, property) → value type` upsert out of the per-row
/// body and behind a statement-scoped memo, so this pins both ends of that
/// move: the catalogue still gains the key (a memo that skipped the *first*
/// row would silently stop declaring properties, and the columnar column set
/// is derived from this map on save), and a failed statement still leaves none
/// of it behind (the memo must not outlive the writes it describes).
#[test]
fn a_multi_row_set_records_a_new_property_once_and_correctly() {
    let mut graph = wide_rows_into(DirGraph::new());
    assert!(
        !graph.node_type_metadata["Item"].contains_key("color"),
        "precondition: the property must be new, or this is vacuous"
    );

    run(&mut graph, "MATCH (n:Item) SET n.color = 'red'");
    assert_eq!(
        graph.node_type_metadata["Item"]
            .get("color")
            .map(String::as_str),
        Some("String"),
        "a 200-row SET must declare the property it introduced"
    );

    // And the rollback half, on a property this statement introduces.
    let mut graph = wide_rows_into(DirGraph::new());
    assert_rolls_back(
        &mut graph,
        "MATCH (n:Item) SET n.color = 'red' \
         WITH n LIMIT 1 MATCH (m:Item {id: 2}) \
         SET m.qty = duration({months: 2147483648})",
        None,
    );
    assert!(
        !graph.node_type_metadata["Item"].contains_key("color"),
        "a rolled-back multi-row SET left the property in the catalogue"
    );
}

/// Rows whose values have **different** type names still take the catalogue
/// with them: last write wins, exactly as it did per row.
///
/// The statement-scoped memo remembers the type name it recorded, not merely
/// that it recorded something — a first-wins memo would leave the catalogue
/// claiming `int64` for a property whose surviving rows are strings, and the
/// column that metadata derives on save would be typed from the wrong end.
#[test]
fn a_set_with_mixed_value_types_records_the_last_one() {
    let mut graph = wide_rows_into(DirGraph::new());

    run(
        &mut graph,
        "MATCH (n:Item) WITH n ORDER BY n.qty \
         SET n.mixed = CASE WHEN n.qty < 100 THEN 1 ELSE 'late' END",
    );

    assert_eq!(
        graph.node_type_metadata["Item"]
            .get("mixed")
            .map(String::as_str),
        Some("String"),
        "the catalogue must carry the type name of the last row written, which \
         is what the per-row upsert this replaced recorded"
    );
}

/// The four O(types) maps no Cypher statement writes — aliases and parent
/// types — restore too.
///
/// They are unreachable from the statement-level arms above (only `add_nodes`
/// and the Python introspection API write them), but they are `Arc`-shared by
/// the shell exactly like the others, so a restore that dropped one would go
/// unnoticed. Exercised directly against `schema_shell`/`restore_schema_shell`.
#[test]
fn the_shell_restores_the_alias_and_parent_maps() {
    let mut graph = seeded();
    graph
        .id_field_aliases_mut()
        .insert("Item".to_string(), "npdid".to_string());
    graph
        .title_field_aliases_mut()
        .insert("Item".to_string(), "label".to_string());
    graph
        .parent_types_mut()
        .insert("Tag".to_string(), "Item".to_string());

    let shell = graph.schema_shell();

    graph.id_field_aliases_mut().clear();
    graph.title_field_aliases_mut().clear();
    graph.parent_types_mut().clear();
    graph
        .node_type_metadata_mut()
        .insert("Ghost".to_string(), HashMap::new());

    graph.restore_schema_shell(shell);

    assert_eq!(
        graph.id_field_aliases.get("Item").map(String::as_str),
        Some("npdid")
    );
    assert_eq!(
        graph.title_field_aliases.get("Item").map(String::as_str),
        Some("label")
    );
    assert_eq!(
        graph.parent_types.get("Tag").map(String::as_str),
        Some("Item")
    );
    assert!(!graph.node_type_metadata.contains_key("Ghost"));
}

// ─────────────────────────────────────────────────────────────────────
// Cost model
// ─────────────────────────────────────────────────────────────────────

/// A 200-type × 50-property schema — the shape the shell clone is O(·) in.
///
/// Deliberately node-free: the cost this pins is per *statement*, not per row,
/// and 10,000 metadata cells make a deep copy unmistakable while the graph
/// stays a unit test.
fn wide_schema() -> DirGraph {
    let mut graph = DirGraph::new();
    for t in 0..200 {
        let mut props: HashMap<String, String> = HashMap::new();
        for c in 0..50 {
            props.insert(format!("p{c}"), "Int64".to_string());
        }
        graph.upsert_node_type_metadata(&format!("T{t}"), props);
    }
    graph.rebuild_type_schemas();
    graph
}

/// The shell copies nothing: opening and closing a checkpoint on a wide schema
/// must fork no schema map at all.
///
/// This is the P2 defect, inverted. Before it, `schema_shell` was a full
/// `DirGraph::clone()` and every mutating statement deep-copied all 10,000
/// metadata cells — 224 µs of pure overhead on a statement that wrote nothing.
/// A fork counter, rather than a timing, is what makes that regression visible
/// in a unit test: the copy is invisible to `BACKEND_CLONE_NODES`,
/// `JOURNAL_NODE_PRE_IMAGES` and `COLUMN_STORE_CLONES` alike.
#[test]
fn a_statement_that_changes_no_schema_forks_no_map() {
    let mut graph = wide_schema();
    // A statement that matches nothing still opens (and closes) a checkpoint.
    reset_schema_map_forks();
    run(&mut graph, "MATCH (n:T0 {id: -1}) SET n.p0 = 7");
    assert_eq!(
        schema_map_forks(),
        0,
        "a statement that changes no schema must copy no schema map; the shell \
         clone is a pointer copy"
    );

    // And one that does write, but writes only keys the catalogue already
    // holds. `qty` is declared by the seeding below, so the upsert is a hit.
    let mut graph = seeded();
    reset_schema_map_forks();
    run(&mut graph, "MATCH (n:Item {id: 1}) SET n.qty = 99");
    assert_eq!(
        schema_map_forks(),
        0,
        "an upsert whose keys are all already declared must not fork the map"
    );
    // Non-vacuity: the write landed.
    assert_eq!(item_prop(&graph, 1, "qty"), Some(Value::Int64(99)));
}

/// A statement that *does* grow the schema forks each map it changes exactly
/// once, however many rows it writes.
///
/// One fork per statement is the whole CoW contract: the shell holds the
/// second handle, so the first write forks and every later write in the same
/// statement mutates the fork in place.
#[test]
fn a_schema_growing_statement_forks_each_touched_map_once() {
    let mut graph = wide_rows_into(DirGraph::new());

    reset_schema_map_forks();
    // 200 rows, one new property: `node_type_metadata` is upserted per row.
    run(&mut graph, "MATCH (n:Item) SET n.color = 'red'");
    let forks = schema_map_forks();
    assert_eq!(
        forks, 3,
        "a 200-row SET that introduces one property must fork exactly the three \
         pieces of schema state it changes — node_type_metadata, type_schemas, \
         and the interner, which sees the name `color` for the first time — once \
         each, not once per row; saw {forks}"
    );

    // The next statement re-declares nothing, so it forks nothing.
    reset_schema_map_forks();
    run(&mut graph, "MATCH (n:Item) SET n.color = 'blue'");
    assert_eq!(
        schema_map_forks(),
        0,
        "the second pass over the same property must fork no map"
    );
}

/// The fork counter is not vacuous: a deliberate deep copy registers.
#[test]
fn the_fork_counter_sees_a_real_copy() {
    let mut graph = wide_schema();
    reset_schema_map_forks();
    let _second_handle = graph.node_type_metadata.clone();
    graph
        .node_type_metadata_mut()
        .insert("New".to_string(), HashMap::new());
    assert_eq!(
        schema_map_forks(),
        1,
        "writing through a shared handle must copy, and must be counted"
    );
}
