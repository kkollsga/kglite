//! Replay of journaled validity-interval declarations.
use super::*;
use crate::datatypes::Value;
use crate::graph::features::temporal::persist::JournaledDeclaration;
use crate::graph::features::temporal::{self, IntervalConvention, TemporalTarget};
use crate::graph::schema::TemporalConfig;
use crate::graph::wal::MutationOp;

fn config(from: &str, to: &str, source_type: Option<&str>) -> TemporalConfig {
    TemporalConfig {
        valid_from: from.into(),
        valid_to: to.into(),
        convention: IntervalConvention::HalfOpen,
        source_type: source_type.map(str::to_string),
    }
}

fn rel(source_type: Option<&str>) -> TemporalTarget {
    TemporalTarget::Relationship {
        rel_type: "R".into(),
        source_type: source_type.map(str::to_string),
    }
}

fn op(target: &TemporalTarget, config: Option<(&TemporalConfig, Option<usize>)>) -> MutationOp {
    let change = JournaledDeclaration::new(target, config);
    MutationOp::SetTemporalDeclaration {
        declaration_json: serde_json::to_string(&change).unwrap(),
    }
}

fn raw(json: &str) -> MutationOp {
    MutationOp::SetTemporalDeclaration {
        declaration_json: json.into(),
    }
}

fn replay(g: &mut DirGraph, ops: Vec<MutationOp>) {
    let frames = vec![WalFrame { lsn: 1, ops }];
    assert_eq!(apply_frames(g, &frames, 0).unwrap(), 1);
}

fn node_row(id: i64) -> MutationOp {
    MutationOp::UpsertNode {
        node_type: "A".into(),
        id: Value::Int64(id),
        title: Value::String(format!("a{id}")),
        properties: vec![],
    }
}

#[test]
fn a_declaration_replays_with_its_count() {
    let mut g = DirGraph::new();
    let node = TemporalTarget::Node("A".into());
    let cfg = config("vf", "vt", None);
    replay(&mut g, vec![op(&node, Some((&cfg, Some(3))))]);
    assert_eq!(temporal::node_config(&g, "A"), Some(&cfg));
    let listed = temporal::list(&g);
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].abutting_rows, Some(3));
}

#[test]
fn declare_then_undeclare_in_one_log_replays_absent() {
    let mut g = DirGraph::new();
    let node = TemporalTarget::Node("A".into());
    let cfg = config("vf", "vt", None);
    replay(&mut g, vec![op(&node, Some((&cfg, None))), op(&node, None)]);
    assert!(temporal::list(&g).is_empty());
}

#[test]
fn an_undeclare_removes_a_checkpointed_declaration() {
    let mut g = DirGraph::new();
    let node = TemporalTarget::Node("A".into());
    g.temporal.apply_journaled(JournaledDeclaration::new(
        &node,
        Some((&config("vf", "vt", None), None)),
    ));
    replay(&mut g, vec![op(&node, None)]);
    assert!(temporal::list(&g).is_empty());
}

#[test]
fn keyed_and_unkeyed_keys_fold_apart() {
    let mut g = DirGraph::new();
    let keyed = config("kf", "kt", Some("S"));
    let unkeyed = config("uf", "ut", None);
    replay(
        &mut g,
        vec![
            op(&rel(Some("S")), Some((&keyed, None))),
            op(&rel(None), Some((&unkeyed, None))),
            op(&rel(None), None),
        ],
    );
    assert_eq!(temporal::edge_configs(&g, "R"), [keyed]);
}

#[test]
fn a_redeclaration_replaces_its_key() {
    let mut g = DirGraph::new();
    let first = config("a", "b", Some("S"));
    let second = config("c", "d", Some("S"));
    replay(
        &mut g,
        vec![
            op(&rel(Some("S")), Some((&first, None))),
            op(&rel(Some("S")), None),
            op(&rel(Some("S")), Some((&second, None))),
        ],
    );
    assert_eq!(temporal::edge_configs(&g, "R"), [second]);
}

#[test]
fn an_unknown_field_is_ignored() {
    let mut g = DirGraph::new();
    replay(
        &mut g,
        vec![raw(
            r#"{"kind":"node","name":"A","later":true,"config":{"from":"vf","to":"vt","convention":"closed","grain":"day"}}"#,
        )],
    );
    let cfg = temporal::node_config(&g, "A").expect("declared");
    assert_eq!(
        (cfg.valid_from.as_str(), cfg.valid_to.as_str()),
        ("vf", "vt")
    );
    assert_eq!(cfg.convention, IntervalConvention::Closed);
}

#[test]
fn a_malformed_record_is_skipped_and_the_frame_still_replays() {
    let mut g = DirGraph::new();
    let node = TemporalTarget::Node("B".into());
    let cfg = config("vf", "vt", None);
    replay(
        &mut g,
        vec![
            raw(r#"{"kind":"node","name":"A","config":{"from":"#),
            node_row(1),
            op(&node, Some((&cfg, None))),
        ],
    );
    assert!(temporal::node_config(&g, "A").is_none());
    assert_eq!(temporal::node_config(&g, "B"), Some(&cfg));
    assert!(g.lookup_by_id("A", &Value::Int64(1)).is_some());
}
