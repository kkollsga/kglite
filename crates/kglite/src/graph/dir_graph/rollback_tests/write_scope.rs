//! The `write_scope` perimeter, judged from inside the engine.
//!
//! The Python suite (`tests/test_write_scope.py`) pins the user-visible
//! boundary; this file pins the property that boundary rests on and that a
//! result-set comparison cannot see: **a refused write mutates nothing**.
//!
//! Every refusal below is checked against the full rollback fingerprint —
//! petgraph slot identity, index bucket order, column masters, the version
//! counter — so a "refusal" that got as far as touching storage and then had
//! its damage undone by the journal is not mistaken for a refusal that
//! happened first. For `DELETE` that is load-bearing: the authorization sits
//! in the collection phase precisely so a statement whose later row is out of
//! scope never unlinks its earlier, in-scope ones.

use super::*;

/// The seeded fixture holds `Item` and `Tag` nodes joined by `LINKS`
/// (Item→Item) and `TAGGED` (Item→Tag).
const ITEM_ONLY: Option<&[&str]> = Some(&["Item"]);
const TAG_ONLY: Option<&[&str]> = Some(&["Tag"]);
const NOTHING: Option<&[&str]> = Some(&[]);

fn counts(graph: &DirGraph) -> (usize, usize) {
    (graph.graph.node_count(), graph.graph.edge_count())
}

// ── node writes: DELETE / REMOVE / SET label ─────────────────────────

#[test]
fn node_delete_outside_the_scope_is_refused() {
    let mut graph = seeded();
    let before = counts(&graph);
    let error = expect_failure(&mut graph, "MATCH (n:Tag) DETACH DELETE n", ITEM_ONLY);
    assert!(
        error.contains("write scope violation: node type 'Tag'"),
        "unexpected error: {error}"
    );
    assert_eq!(counts(&graph), before);
}

#[test]
fn a_refused_delete_does_not_take_the_in_scope_rows_with_it() {
    // The whole-graph delete matches Items (in scope) *and* Tags (not). The
    // refusal must land before the first Item is unlinked — this is what
    // authorizing at collection time buys over authorizing at commit time.
    let mut graph = seeded();
    let before = counts(&graph);
    expect_failure(&mut graph, "MATCH (n) DETACH DELETE n", ITEM_ONLY);
    assert_eq!(counts(&graph), before);
    assert_rolls_back(&mut graph, "MATCH (n) DETACH DELETE n", ITEM_ONLY);
}

#[test]
fn remove_property_outside_the_scope_is_refused() {
    let mut graph = seeded();
    assert_rolls_back(&mut graph, "MATCH (t:Tag) REMOVE t.name", ITEM_ONLY);
}

#[test]
fn remove_label_outside_the_scope_is_refused() {
    let mut graph = seeded();
    assert_rolls_back(&mut graph, "MATCH (t:Tag {id: 1}) REMOVE t:Hot", ITEM_ONLY);
}

#[test]
fn set_label_outside_the_scope_is_refused() {
    let mut graph = seeded();
    assert_rolls_back(&mut graph, "MATCH (t:Tag {id: 2}) SET t:Warm", ITEM_ONLY);
}

#[test]
fn the_stored_type_decides_not_the_pattern_label() {
    // Label smuggling: an added secondary label cannot talk a node into scope,
    // because every node gate reads the node's stored type.
    let mut graph = seeded();
    run(&mut graph, "MATCH (t:Tag {id: 1}) SET t:Item");
    assert_rolls_back(
        &mut graph,
        "MATCH (n:Item {id: 1}) DETACH DELETE n",
        TAG_ONLY,
    );
}

// ── relationship writes: one endpoint in scope is enough ─────────────

#[test]
fn relationship_write_with_one_endpoint_in_scope_is_allowed() {
    // TAGGED runs Item→Tag. Under a Tag-only scope the source is out of scope
    // and the target is in it, which is enough for every relationship verb.
    let mut graph = seeded();
    let params = HashMap::new();
    let owned: HashSet<String> = ["Tag".to_string()].into_iter().collect();
    let mut opts = ExecuteOptions::eager(&params);
    opts.write_scope = Some(&owned);
    for query in [
        "MATCH ()-[r:TAGGED]->() SET r.note = 'ok'",
        "MATCH ()-[r:TAGGED]->() REMOVE r.note",
        "MATCH ()-[r:TAGGED]->() DELETE r",
    ] {
        execute_mut(&mut graph, query, &opts)
            .unwrap_or_else(|e| panic!("{query} must be allowed under a Tag scope: {e}"));
    }
}

#[test]
fn relationship_write_with_neither_endpoint_in_scope_is_refused() {
    let mut graph = seeded();
    // LINKS runs Item→Item, so a Tag-only scope owns neither endpoint.
    let error = expect_failure(&mut graph, "MATCH ()-[r:LINKS]->() DELETE r", TAG_ONLY);
    assert!(
        error.contains(
            "write scope violation: relationship 'LINKS' connects 'Item' to 'Item' and \
             neither endpoint type is in the allowed write set (Tag)"
        ),
        "unexpected error: {error}"
    );
    assert_rolls_back(&mut graph, "MATCH ()-[r:LINKS]->() DELETE r", TAG_ONLY);
    assert_rolls_back(
        &mut graph,
        "MATCH ()-[r:LINKS]->() SET r.weight = 1",
        TAG_ONLY,
    );
    assert_rolls_back(
        &mut graph,
        "MATCH ()-[r:LINKS]->() REMOVE r.weight",
        TAG_ONLY,
    );
}

#[test]
fn edge_create_between_two_out_of_scope_nodes_is_refused() {
    let mut graph = seeded();
    assert_rolls_back(
        &mut graph,
        "MATCH (a:Item {id: 1}), (b:Item {id: 3}) CREATE (a)-[:FORGED]->(b)",
        TAG_ONLY,
    );
    // The refusal must also leave the schema clean: a connection type the
    // graph never accepted must not show up in its metadata.
    assert!(
        !graph.connection_type_metadata.contains_key("FORGED"),
        "a refused edge CREATE registered its connection type"
    );
}

#[test]
fn detach_delete_collateral_edges_are_authorized_by_the_node() {
    // Item 1 carries a TAGGED edge to a Tag, which is out of scope. Deleting
    // the Item is authorized, and its incident edges go with it — the far
    // endpoint's type is deliberately not re-checked.
    let mut graph = seeded();
    let (nodes, edges) = counts(&graph);
    let params = HashMap::new();
    let owned: HashSet<String> = ["Item".to_string()].into_iter().collect();
    let mut opts = ExecuteOptions::eager(&params);
    opts.write_scope = Some(&owned);
    execute_mut(&mut graph, "MATCH (n:Item {id: 1}) DETACH DELETE n", &opts)
        .expect("deleting an in-scope node must carry its edges with it");
    assert_eq!(counts(&graph), (nodes - 1, edges - 2));
}

// ── an empty whitelist denies everything ─────────────────────────────

#[test]
fn an_empty_scope_denies_every_verb() {
    for query in [
        "CREATE (:Item {id: 9})",
        "MATCH (n:Item {id: 1}) SET n.qty = 1",
        "MATCH (n:Item {id: 1}) SET n:Warm",
        "MATCH (n:Item {id: 1}) REMOVE n.name",
        "MATCH (n:Item {id: 3}) DELETE n",
        "MATCH ()-[r:LINKS]->() DELETE r",
        "MATCH ()-[r:LINKS]->() SET r.weight = 1",
        "MATCH ()-[r:LINKS]->() REMOVE r.weight",
        "MATCH (a:Item {id: 1}), (b:Item {id: 3}) CREATE (a)-[:FORGED]->(b)",
    ] {
        let mut graph = seeded();
        let error = expect_failure(&mut graph, query, NOTHING);
        assert!(
            error.contains("write scope violation"),
            "{query} must be refused by an empty scope, got: {error}"
        );
    }
}
