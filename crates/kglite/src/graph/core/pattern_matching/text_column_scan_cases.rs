//! One named case per state the text column driver must get right: each
//! builds the state, asserts the driver against brute force over every
//! matcher and mode, and then pins the specific answer the state is about, so
//! that a driver and a brute force that agree on a wrong thing still fail.

use super::text_column_scan::{TextAnswer, TextDecline, TextMode};
use super::text_column_scan_tests::*;
use crate::datatypes::Value;
use crate::graph::algorithms::Interrupt;
use crate::graph::dir_graph::DirGraph;
use crate::graph::io::file::{
    load_file_with, save_graph_with_layout, with_floors, LoadOptions, MapPolicy, SaveLayout,
};
use crate::graph::session::{CommitOutcome, Session};
use crate::graph::storage::mode::StorageMode;
use crate::graph::storage::GraphRead;
use crate::graph::text_scan::matcher::{Matcher, Pattern};
use petgraph::graph::NodeIndex;
use std::collections::HashMap;
use std::sync::Arc;

fn lit(text: &str) -> Matcher {
    Matcher::compile(&Pattern::Literal {
        text: text.into(),
        case_insensitive: false,
    })
    .unwrap()
}

fn found(graph: &DirGraph, ty: &str, prop: &str, text: &str) -> Vec<NodeIndex> {
    match drive(graph, ty, prop, &lit(text), None, TextMode::Exists)
        .unwrap()
        .unwrap_or_else(|d| panic!("declined: {d:?}"))
    {
        TextAnswer::Nodes(nodes) => nodes,
        other => panic!("{other:?}"),
    }
}

fn ids_of(graph: &DirGraph, nodes: &[NodeIndex]) -> Vec<i64> {
    nodes
        .iter()
        .map(|n| match graph.graph.get_node_id(*n) {
            Some(Value::Int64(i)) => i,
            Some(Value::UniqueId(i)) => i64::from(i),
            other => panic!("{other:?}"),
        })
        .collect()
}

fn exact(graph: &DirGraph, ty: &str, prop: &str) {
    for scope in [None, Some(&[][..])] {
        assert_exact(graph, ty, prop, scope, &[], "case");
    }
    let all: Vec<NodeIndex> = graph.graph.node_indices().collect();
    assert_exact(graph, ty, prop, Some(&all), &[], "case, whole-graph scope");
}

fn docs() -> DirGraph {
    let mut g = DirGraph::new();
    for (id, a) in [
        (1, "alpha one"),
        (2, "no match"),
        (3, "two alpha alpha"),
        (4, "alp"),
        (5, "ha tail"),
    ] {
        run(
            &mut g,
            &format!("CREATE (:A {{id: {id}, a: $a}})"),
            &param("a", a),
        )
        .unwrap();
    }
    g
}

#[test]
fn a_deleted_nodes_row_lingers_in_the_bytes_and_is_not_matched() {
    let mut g = docs();
    assert_eq!(ids_of(&g, &found(&g, "A", "a", "alpha")), [1, 3]);
    exec(&mut g, "MATCH (n:A {id: 1}) DELETE n");
    // Row 0's bytes still say "alpha one"; only the live row 2 matches.
    assert_eq!(ids_of(&g, &found(&g, "A", "a", "alpha")), [3]);
    exec(&mut g, "MATCH (n:A {id: 3}) DELETE n");
    assert!(found(&g, "A", "a", "alpha").is_empty());
    exact(&g, "A", "a");
}

#[test]
fn a_node_created_after_a_delete_is_reported_with_its_own_row() {
    let mut g = docs();
    exec(&mut g, "MATCH (n:A {id: 2}) DELETE n");
    run(
        &mut g,
        "CREATE (:A {id: 6, a: $a})",
        &param("a", "late alpha"),
    )
    .unwrap();
    let nodes = found(&g, "A", "a", "alpha");
    // Bucket order: the new node is last.
    assert_eq!(ids_of(&g, &nodes), [1, 3, 6]);
    exact(&g, "A", "a");
}

#[test]
fn a_relocated_row_is_matched_by_its_new_value_and_not_its_old_bytes() {
    let mut g = docs();
    run(
        &mut g,
        "MATCH (n:A {id: 1}) SET n.a = $v",
        &param("v", "a longer replacement value with beta inside"),
    )
    .unwrap();
    assert_eq!(ids_of(&g, &found(&g, "A", "a", "alpha")), [3]);
    let nodes = found(&g, "A", "a", "beta");
    assert_eq!(ids_of(&g, &nodes), [1]);
    // The offset is in the new value.
    let got = drive(
        &g,
        "A",
        "a",
        &lit("beta"),
        None,
        TextMode::Hits { max: None },
    )
    .unwrap()
    .unwrap();
    let TextAnswer::Hits { hits, .. } = got else {
        panic!()
    };
    assert_eq!((hits[0].offset, hits[0].len), (32, 4));
    exact(&g, "A", "a");
}

#[test]
fn a_same_length_edit_is_visible_in_the_bytes_without_an_overlay() {
    let mut g = docs();
    run(
        &mut g,
        "MATCH (n:A {id: 2}) SET n.a = $v",
        &param("v", "ALPHA no"),
    )
    .unwrap();
    run(
        &mut g,
        "MATCH (n:A {id: 1}) SET n.a = $v",
        &param("v", "xxxxx one"),
    )
    .unwrap();
    assert_eq!(
        column_state(&g, "A", "a").1,
        0,
        "heap bytes are overwritten in place"
    );
    assert_eq!(ids_of(&g, &found(&g, "A", "a", "ALPHA")), [2]);
    assert_eq!(ids_of(&g, &found(&g, "A", "a", "alpha")), [3]);
    exact(&g, "A", "a");
}

#[test]
fn a_value_set_to_null_or_removed_stops_matching_and_a_null_can_gain_one() {
    let mut g = docs();
    exec(&mut g, "MATCH (n:A {id: 1}) SET n.a = null");
    exec(&mut g, "MATCH (n:A {id: 3}) REMOVE n.a");
    assert!(found(&g, "A", "a", "alpha").is_empty());
    run(
        &mut g,
        "MATCH (n:A {id: 1}) SET n.a = $v",
        &param("v", "alpha again"),
    )
    .unwrap();
    assert_eq!(ids_of(&g, &found(&g, "A", "a", "alpha")), [1]);
    exact(&g, "A", "a");
}

#[test]
fn a_value_written_over_a_null_in_a_mapped_column_is_served() {
    let mut g = big_a_graph();
    exec(&mut g, "MATCH (n:A {id: 1000}) SET n.b = null");
    let dir = tempfile::tempdir().unwrap();
    let mut loaded = save_and_load_mapped(g, dir.path());
    let (mapped, _) = column_state(&loaded, "A", "b");
    assert!(mapped, "the column is a mapped section");
    run(
        Arc::make_mut(&mut loaded),
        "MATCH (n:A {id: 1000}) SET n.b = $v",
        &param("v", "gamma"),
    )
    .unwrap();
    run(
        Arc::make_mut(&mut loaded),
        "MATCH (n:A {id: 1}) SET n.b = null",
        &HashMap::new(),
    )
    .unwrap();
    assert_eq!(column_state(&loaded, "A", "b"), (true, 1));
    assert_eq!(ids_of(&loaded, &found(&loaded, "A", "b", "gamma")), [1000]);
    exact(&loaded, "A", "b");
}

#[test]
fn secondary_label_carriers_decline_and_an_unrelated_label_does_not() {
    let mut g = docs();
    run(
        &mut g,
        "CREATE (:B {id: 10, a: $a})",
        &param("a", "alpha b"),
    )
    .unwrap();
    exec(&mut g, "MATCH (n:B {id: 10}) SET n:L");
    assert!(drive(&g, "A", "a", &lit("alpha"), None, TextMode::Exists)
        .unwrap()
        .is_ok());
    exec(&mut g, "MATCH (n:B {id: 10}) SET n:A");
    let declined = drive(&g, "A", "a", &lit("alpha"), None, TextMode::Exists).unwrap();
    assert_eq!(declined, Err(TextDecline::SecondaryLabel));
    // The carrier's own type is unaffected.
    assert!(drive(&g, "B", "a", &lit("alpha"), None, TextMode::Exists)
        .unwrap()
        .is_ok());
}

#[test]
fn a_scope_drops_foreign_dead_and_repeated_nodes_and_keeps_its_own_order() {
    let mut g = docs();
    run(
        &mut g,
        "CREATE (:B {id: 20, a: $a})",
        &param("a", "alpha foreign"),
    )
    .unwrap();
    let node = |g: &DirGraph, id: i64| {
        *found(g, "A", "a", "alpha")
            .iter()
            .find(|n| ids_of(g, &[**n])[0] == id)
            .unwrap()
    };
    let (n3, n1) = (node(&g, 3), node(&g, 1));
    let foreign = g.type_indices.get("B").unwrap().to_vec()[0];
    exec(&mut g, "MATCH (n:A {id: 5}) DELETE n");
    let dead = NodeIndex::new(900);
    let scope = [n3, foreign, n1, dead, n3, NodeIndex::new(0)];
    let got = drive(&g, "A", "a", &lit("alpha"), Some(&scope), TextMode::Exists)
        .unwrap()
        .unwrap();
    // `NodeIndex 0` is node id 1's own slot: repeated, so reported once.
    assert_eq!(got, TextAnswer::Nodes(vec![n3, n1]));
    assert_exact(&g, "A", "a", Some(&scope), &[], "scope case");
}

#[test]
fn an_empty_scope_does_not_touch_the_column() {
    let g = docs();
    let expired = Interrupt::from_deadline(Some(
        std::time::Instant::now() - std::time::Duration::from_secs(1),
    ));
    let ask = |scope: Option<&[NodeIndex]>| {
        super::text_column_scan::text_column_scan(
            &g,
            &super::text_column_scan::TextScanRequest {
                node_type: "A",
                property: "a",
                matcher: &lit("alpha"),
                scope,
                mode: TextMode::Exists,
                interrupt: expired,
                serial: false,
            },
        )
    };
    // A scan would be stopped by the expired deadline; an empty scope never
    // reaches one.
    assert_eq!(ask(Some(&[])), Ok(Ok(TextAnswer::Nodes(vec![]))));
    assert!(
        ask(None).is_err(),
        "the deadline is polled by every real scan"
    );
    let one = g.type_indices.get("A").unwrap().to_vec();
    assert!(ask(Some(&one)).is_err());
    // Nodes of another type are not a scope either.
    let mut with_b = docs();
    run(
        &mut with_b,
        "CREATE (:B {id: 9, a: $a})",
        &param("a", "alpha"),
    )
    .unwrap();
    let b = with_b.type_indices.get("B").unwrap().to_vec();
    assert_eq!(
        drive(&with_b, "A", "a", &lit("alpha"), Some(&b), TextMode::Exists),
        Ok(Ok(TextAnswer::Nodes(vec![])))
    );
}

#[test]
fn an_interrupt_reaches_the_engines_timeout_and_cancel_wording() {
    use std::sync::atomic::AtomicBool;
    let g = docs();
    let past = std::time::Instant::now() - std::time::Duration::from_secs(1);
    for (interrupt, expect) in [
        (Interrupt::from_deadline(Some(past)), "timed out"),
        (
            Interrupt {
                deadline: None,
                cancel: Some(Box::leak(Box::new(AtomicBool::new(true)))),
            },
            "cancelled",
        ),
    ] {
        let err = super::text_column_scan::text_column_scan(
            &g,
            &super::text_column_scan::TextScanRequest {
                node_type: "A",
                property: "a",
                matcher: &lit("alpha"),
                scope: None,
                mode: TextMode::Count,
                interrupt,
                serial: false,
            },
        )
        .unwrap_err();
        let message = err.message(&interrupt);
        assert!(message.to_lowercase().contains(expect), "{message}");
    }
}

#[test]
fn identity_fields_and_soft_aliases_decline_and_other_names_answer() {
    let mut g = docs();
    for (prop, want) in [
        ("id", TextDecline::IdentityField),
        ("title", TextDecline::IdentityField),
        ("name", TextDecline::SoftAlias),
        ("label", TextDecline::SoftAlias),
        ("type", TextDecline::SoftAlias),
        ("node_type", TextDecline::SoftAlias),
    ] {
        let got = drive(&g, "A", prop, &lit("alpha"), None, TextMode::Exists).unwrap();
        assert_eq!(got, Err(want), "{prop}");
    }
    // A declared title spelling resolves to the title field.
    g.declare_title_field_alias("A", "a");
    assert_eq!(
        drive(&g, "A", "a", &lit("alpha"), None, TextMode::Exists).unwrap(),
        Err(TextDecline::IdentityField)
    );
}

#[test]
fn a_property_the_type_lacks_matches_nothing_and_a_missing_type_declines() {
    let g = docs();
    assert_eq!(
        drive(&g, "A", "absent", &lit("alpha"), None, TextMode::Count).unwrap(),
        Ok(TextAnswer::Count(0))
    );
    assert_eq!(
        drive(&g, "Nope", "a", &lit("alpha"), None, TextMode::Count).unwrap(),
        Err(TextDecline::NoStore)
    );
}

#[test]
fn a_column_holding_strings_beside_other_kinds_declines_and_a_numeric_one_answers_empty() {
    let mut g = docs();
    exec(&mut g, "CREATE (:N {id: 7, n: 1})");
    assert_eq!(
        drive(&g, "N", "n", &lit("1"), None, TextMode::Exists).unwrap(),
        Ok(TextAnswer::Nodes(vec![])),
        "an Int64 column holds no string"
    );
    exec(&mut g, "MATCH (n:A {id: 1}) SET n.a = 5");
    assert_eq!(
        drive(&g, "A", "a", &lit("alpha"), None, TextMode::Exists).unwrap(),
        Err(TextDecline::NotStringColumn)
    );
}

#[test]
fn a_disk_graph_declines() {
    let mut g = docs();
    g.enable_disk_mode().unwrap();
    assert_eq!(
        drive(&g, "A", "a", &lit("alpha"), None, TextMode::Exists).unwrap(),
        Err(TextDecline::Disk)
    );
}

#[test]
fn a_statement_that_fails_after_writing_leaves_the_column_exact() {
    let mut g = docs();
    let failed = run(
        &mut g,
        "MATCH (n:A {id: 2}) SET n.a = 'alpha from a rolled back statement' SET n.id = 99",
        &HashMap::new(),
    );
    assert!(failed.is_err());
    assert_eq!(ids_of(&g, &found(&g, "A", "a", "alpha")), [1, 3]);
    exact(&g, "A", "a");
}

// ── mapped columns and heap tails ────────────────────────────────────

/// 40 rows of 8 KB in `b`: a column a save writes as a mappable section.
fn big_a_graph() -> DirGraph {
    let mut g = DirGraph::new();
    for i in 0..40 {
        let mut v = "z".repeat(4000);
        if i % 5 == 0 {
            v.push_str(" alpha ");
        }
        v.push_str(&"z".repeat(4000));
        let params = HashMap::from([("v".to_string(), Value::String(v))]);
        run(
            &mut g,
            &format!("CREATE (:A {{id: {}, a: 'row', b: $v}})", i + 1000),
            &params,
        )
        .unwrap();
    }
    run(
        &mut g,
        "CREATE (:A {id: 1, a: 'row', b: 'alpha small'})",
        &HashMap::new(),
    )
    .unwrap();
    g
}

fn save_and_load_mapped(g: DirGraph, dir: &std::path::Path) -> Arc<DirGraph> {
    let path = dir.join("big.kgl");
    let mut arc = Arc::new(g);
    with_floors(10, 10, || {
        save_graph_with_layout(&mut arc, path.to_str().unwrap(), true, SaveLayout::Auto)
    })
    .unwrap();
    load_file_with(
        path.to_str().unwrap(),
        &LoadOptions::new()
            .with_storage(StorageMode::Mapped)
            .with_columns(MapPolicy::Mapped),
    )
    .unwrap()
}

#[test]
fn a_mapped_column_with_edited_rows_is_served_beside_its_stale_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let mut g = save_and_load_mapped(big_a_graph(), dir.path());
    assert_eq!(column_state(&g, "A", "b"), (true, 0));
    let before = found(&g, "A", "b", "alpha");
    assert_eq!(before.len(), 9, "eight big rows and the small one");
    // Replace a matching row and a non-matching one.
    run(
        Arc::make_mut(&mut g),
        "MATCH (n:A {id: 1000}) SET n.b = $v",
        &param("v", "now plain"),
    )
    .unwrap();
    run(
        Arc::make_mut(&mut g),
        "MATCH (n:A {id: 1001}) SET n.b = $v",
        &param("v", "now alpha"),
    )
    .unwrap();
    assert_eq!(column_state(&g, "A", "b"), (true, 2));
    let after = found(&g, "A", "b", "alpha");
    let mut ids = ids_of(&g, &after);
    ids.sort_unstable();
    assert_eq!(ids, [1, 1001, 1005, 1010, 1015, 1020, 1025, 1030, 1035]);
    exact(&g, "A", "b");
}

#[test]
fn a_heap_tail_is_scanned_beside_its_base() {
    const ROWS: usize = 16_500;
    let mut base = DirGraph::new();
    exec(
        &mut base,
        &format!(
            "UNWIND range(0, {}) AS i CREATE (:Item {{id: i, s: 'row ' + toString(i) + \
             CASE WHEN i % 97 = 0 THEN ' alpha' ELSE '' END}})",
            ROWS - 1
        ),
    );
    let session = Session::new(base);
    let mut tx = session.begin();
    let working = tx.working_mut().unwrap();
    run(
        working,
        "CREATE (:Item {id: 90000, s: $s})",
        &param("s", "tail alpha"),
    )
    .unwrap();
    run(
        working,
        "CREATE (:Item {id: 90001, s: $s})",
        &param("s", "tail plain"),
    )
    .unwrap();
    assert!(
        working.column_store("Item").unwrap().has_heap_tail(),
        "the append started a heap tail"
    );
    // An edit in the base and one in the tail.
    run(
        working,
        "MATCH (n:Item {id: 5}) SET n.s = $s",
        &param("s", "edited alpha in the base"),
    )
    .unwrap();
    run(
        working,
        "MATCH (n:Item {id: 90001}) SET n.s = $s",
        &param("s", "edited alpha in the tail"),
    )
    .unwrap();
    run(working, "MATCH (n:Item {id: 97}) DELETE n", &HashMap::new()).unwrap();
    let nodes = found(working, "Item", "s", "alpha");
    let ids = ids_of(working, &nodes);
    assert!(ids.contains(&90000) && ids.contains(&90001) && ids.contains(&5));
    assert!(!ids.contains(&97));
    assert_exact(working, "Item", "s", None, &[], "heap tail");
    let all: Vec<NodeIndex> = working.type_indices.get("Item").unwrap().to_vec();
    let scope: Vec<NodeIndex> = all.iter().rev().step_by(7).copied().collect();
    assert_exact(working, "Item", "s", Some(&scope), &[], "heap tail, scoped");
    assert!(matches!(
        session.commit(tx, true),
        CommitOutcome::Committed { .. }
    ));
}

#[test]
fn a_shuffled_scope_with_repeats_gives_the_same_nodes_in_scope_order() {
    let mut g = docs();
    exec(&mut g, "MATCH (n:A {id: 3}) SET n.a = 'a longer alpha replacement'");
    let all = g.type_indices.get("A").unwrap().to_vec();
    let scope: Vec<NodeIndex> = [4usize, 0, 2, 4, 1, 0, 3, 2, 2]
        .iter()
        .map(|&i| all[i])
        .collect();
    let got = drive(&g, "A", "a", &lit("alpha"), Some(&scope), TextMode::Exists)
        .unwrap()
        .unwrap();
    assert_eq!(got, TextAnswer::Nodes(vec![all[0], all[2]]));
    assert_exact(&g, "A", "a", Some(&scope), &[], "shuffled scope");
}
