use super::*;
use crate::datatypes::Value;
use crate::graph::schema::InternedKey;
use crate::graph::storage::GraphRead;
use crate::graph::wal::MutationOp;

fn frame(lsn: u64, ops: Vec<MutationOp>) -> WalFrame {
    WalFrame { lsn, ops }
}

fn upsert_node(id: i64, title: &str, props: Vec<(&str, Value)>) -> MutationOp {
    MutationOp::UpsertNode {
        node_type: "Person".into(),
        id: Value::Int64(id),
        title: Value::String(title.into()),
        properties: props.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
    }
}

fn knows(src: i64, tgt: i64) -> MutationOp {
    MutationOp::UpsertEdge {
        conn_type: "KNOWS".into(),
        src_type: "Person".into(),
        src_id: Value::Int64(src),
        tgt_type: "Person".into(),
        tgt_id: Value::Int64(tgt),
        properties: vec![],
    }
}

fn prop(g: &mut DirGraph, id: i64, key: &str) -> Option<Value> {
    let idx = g.lookup_by_id("Person", &Value::Int64(id))?;
    g.graph
        .node_view(idx)
        .and_then(|n| n.get_field_ref(key).map(|c| c.into_owned()))
}

#[test]
fn replays_upserts_and_edge() {
    let mut g = DirGraph::new();
    let frames = vec![frame(
        1,
        vec![
            upsert_node(1, "Alice", vec![("age", Value::Int64(30))]),
            upsert_node(2, "Bob", vec![]),
            knows(1, 2),
        ],
    )];
    let max = apply_frames(&mut g, &frames, 0).unwrap();
    assert_eq!(max, 1);
    assert_eq!(g.graph.node_count(), 2);
    assert_eq!(g.graph.edge_count(), 1);
    assert_eq!(prop(&mut g, 1, "age"), Some(Value::Int64(30)));
}

#[test]
fn later_upsert_replaces_properties() {
    let mut g = DirGraph::new();
    let frames = vec![
        frame(
            1,
            vec![upsert_node(1, "Alice", vec![("age", Value::Int64(30))])],
        ),
        frame(
            2,
            vec![upsert_node(1, "Alice", vec![("age", Value::Int64(41))])],
        ),
    ];
    apply_frames(&mut g, &frames, 0).unwrap();
    assert_eq!(
        g.graph.node_count(),
        1,
        "same (type,id) is upserted, not duplicated"
    );
    assert_eq!(prop(&mut g, 1, "age"), Some(Value::Int64(41)));
}

#[test]
fn remove_node_deletes_it_and_its_edges() {
    let mut g = DirGraph::new();
    let frames = vec![
        frame(
            1,
            vec![
                upsert_node(1, "Alice", vec![]),
                upsert_node(2, "Bob", vec![]),
                knows(1, 2),
            ],
        ),
        frame(
            2,
            vec![MutationOp::RemoveNode {
                node_type: "Person".into(),
                id: Value::Int64(2),
            }],
        ),
    ];
    apply_frames(&mut g, &frames, 0).unwrap();
    assert_eq!(g.graph.node_count(), 1);
    assert_eq!(
        g.graph.edge_count(),
        0,
        "incident edge removed with the node"
    );
    assert!(g.lookup_by_id("Person", &Value::Int64(2)).is_none());
}

/// Recovery replays a node removal through `detach_delete_nodes`, so the
/// embedding prune rides along: a `.kgl` saved before the delete plus a
/// WAL carrying it must not reload a graph whose store still holds the
/// removed node's vector — the freed index is handed to the next node
/// created and would inherit it.
#[test]
fn replayed_node_removal_prunes_the_embedding_store() {
    let mut g = DirGraph::new();
    apply_frames(
        &mut g,
        &[frame(
            1,
            vec![
                upsert_node(1, "Alice", vec![]),
                upsert_node(2, "Bob", vec![]),
            ],
        )],
        0,
    )
    .unwrap();
    let report = crate::graph::embeddings::set_embeddings(
        &mut g,
        "Person",
        "name",
        None,
        [
            (Value::Int64(1), vec![1.0f32, 0.0]),
            (Value::Int64(2), vec![0.0, 1.0]),
        ],
    )
    .expect("seed embeddings");
    assert_eq!(report.embeddings_stored, 2);
    let doomed = g
        .lookup_by_id("Person", &Value::Int64(2))
        .expect("Bob is present");

    apply_frames(
        &mut g,
        &[frame(
            2,
            vec![MutationOp::RemoveNode {
                node_type: "Person".into(),
                id: Value::Int64(2),
            }],
        )],
        1,
    )
    .unwrap();

    let store = g
        .embeddings
        .get(&("Person".to_string(), "name_emb".to_string()))
        .expect("store");
    assert_eq!(store.len(), 1);
    assert_eq!(store.get_embedding(doomed.index()), None);
    assert_eq!(store.validate_shape(), Ok(()));
}

#[test]
fn remove_edge_keeps_endpoints() {
    let mut g = DirGraph::new();
    let frames = vec![
        frame(
            1,
            vec![
                upsert_node(1, "Alice", vec![]),
                upsert_node(2, "Bob", vec![]),
                knows(1, 2),
            ],
        ),
        frame(
            2,
            vec![MutationOp::RemoveEdge {
                conn_type: "KNOWS".into(),
                src_type: "Person".into(),
                src_id: Value::Int64(1),
                tgt_type: "Person".into(),
                tgt_id: Value::Int64(2),
            }],
        ),
    ];
    apply_frames(&mut g, &frames, 0).unwrap();
    assert_eq!(g.graph.node_count(), 2, "endpoints survive an edge remove");
    assert_eq!(g.graph.edge_count(), 0);
}

#[test]
fn frames_at_or_below_checkpoint_are_skipped() {
    let mut g = DirGraph::new();
    let frames = vec![
        frame(1, vec![upsert_node(1, "Old", vec![])]),
        frame(2, vec![upsert_node(2, "New", vec![])]),
    ];
    // Checkpoint already folded in lsn 1; only replay lsn 2.
    let max = apply_frames(&mut g, &frames, 1).unwrap();
    assert_eq!(max, 2);
    assert!(g.lookup_by_id("Person", &Value::Int64(1)).is_none());
    assert!(g.lookup_by_id("Person", &Value::Int64(2)).is_some());
}

/// Secondary labels a node carries in `labels(n)` order. The exact
/// list, not a set: `DirGraph::node_labels` promises primary-first then
/// name-sorted, and replay must not degrade that to arbitrary order.
fn labels_of(g: &mut DirGraph, id: i64) -> Vec<String> {
    let idx = g
        .lookup_by_id("Person", &Value::Int64(id))
        .expect("node must exist");
    g.node_labels(idx)
        .into_iter()
        .map(|k| g.interner.resolve(k).to_string())
        .collect()
}

fn set_labels(id: i64, labels: &[&str]) -> MutationOp {
    MutationOp::SetNodeLabels {
        node_type: "Person".into(),
        id: Value::Int64(id),
        labels: labels.iter().map(|s| s.to_string()).collect(),
    }
}

/// The regression this op exists for: before `SetNodeLabels`, a node's
/// properties survived replay and its secondary labels silently did
/// not.
#[test]
fn replay_restores_secondary_labels_in_exact_order() {
    let mut g = DirGraph::new();
    let frames = vec![frame(
        1,
        vec![
            upsert_node(1, "Alice", vec![("age", Value::Int64(30))]),
            // Logged unsorted on purpose: ordering is replay's job.
            set_labels(1, &["Manager", "Employee"]),
        ],
    )];
    apply_frames(&mut g, &frames, 0).unwrap();

    assert_eq!(
        labels_of(&mut g, 1),
        vec!["Person", "Employee", "Manager"],
        "primary first, then secondaries sorted by name"
    );
    assert_eq!(prop(&mut g, 1, "age"), Some(Value::Int64(30)));
    assert!(g.has_secondary_labels, "fast-skip flag must be set");
    // The label index is the candidate source for `MATCH (n:Employee)`.
    assert_eq!(g.nodes_with_label("Employee").len(), 1);
}

/// A whole-set op reconciles: labels present in the checkpoint but
/// absent from the log are removed, which is what makes `REMOVE
/// n:Label` recoverable.
#[test]
fn replay_removes_labels_the_log_dropped() {
    let mut g = DirGraph::new();
    apply_frames(
        &mut g,
        &[frame(
            1,
            vec![upsert_node(1, "Alice", vec![]), set_labels(1, &["A", "B"])],
        )],
        0,
    )
    .unwrap();
    assert_eq!(labels_of(&mut g, 1), vec!["Person", "A", "B"]);

    // A later frame carries only "B" — "A" was removed in the session.
    apply_frames(&mut g, &[frame(2, vec![set_labels(1, &["B"])])], 1).unwrap();
    assert_eq!(labels_of(&mut g, 1), vec!["Person", "B"]);
    assert!(
        g.nodes_with_label("A").is_empty(),
        "the dropped label must leave no index residue"
    );
}

/// Emptying the set clears the fast-skip flag, so a graph whose last
/// label was removed pays no secondary-label scan cost after recovery.
#[test]
fn replay_to_an_empty_label_set_clears_the_flag() {
    let mut g = DirGraph::new();
    apply_frames(
        &mut g,
        &[
            frame(
                1,
                vec![upsert_node(1, "Alice", vec![]), set_labels(1, &["A"])],
            ),
            frame(2, vec![set_labels(1, &[])]),
        ],
        0,
    )
    .unwrap();
    assert_eq!(labels_of(&mut g, 1), vec!["Person"]);
    assert!(!g.has_secondary_labels);
}

/// Labels and properties are independent state: an `UpsertNode` logged
/// after a label set (a later `SET n.age = …`) must not wipe the
/// labels, in either fold order.
#[test]
fn property_upsert_does_not_clobber_labels() {
    for reversed in [false, true] {
        let mut ops = vec![
            upsert_node(1, "Alice", vec![]),
            set_labels(1, &["Employee"]),
            upsert_node(1, "Alice", vec![("age", Value::Int64(41))]),
        ];
        if reversed {
            ops.swap(1, 2);
        }
        let mut g = DirGraph::new();
        apply_frames(&mut g, &[frame(1, ops)], 0).unwrap();
        assert_eq!(
            labels_of(&mut g, 1),
            vec!["Person", "Employee"],
            "{reversed}"
        );
        assert_eq!(prop(&mut g, 1, "age"), Some(Value::Int64(41)), "{reversed}");
    }
}

/// A node deleted later in the log must not be resurrected by its own
/// label op.
#[test]
fn label_set_for_a_removed_node_is_skipped() {
    let mut g = DirGraph::new();
    let frames = vec![frame(
        1,
        vec![
            upsert_node(1, "Alice", vec![]),
            set_labels(1, &["Employee"]),
            MutationOp::RemoveNode {
                node_type: "Person".into(),
                id: Value::Int64(1),
            },
        ],
    )];
    apply_frames(&mut g, &frames, 0).unwrap();
    assert_eq!(g.graph.node_count(), 0);
    assert!(g.nodes_with_label("Employee").is_empty());
}

#[test]
fn replaying_labels_twice_is_idempotent() {
    let frames = vec![frame(
        1,
        vec![
            upsert_node(1, "Alice", vec![]),
            set_labels(1, &["Employee", "Manager"]),
        ],
    )];
    let mut g = DirGraph::new();
    apply_frames(&mut g, &frames, 0).unwrap();
    apply_frames(&mut g, &frames, 0).unwrap();
    assert_eq!(labels_of(&mut g, 1), vec!["Person", "Employee", "Manager"]);
    assert_eq!(
        g.nodes_with_label("Employee").len(),
        1,
        "no duplicate bucket entry"
    );
}

/// Replay must work on a `mapped` graph, not only the heap default.
/// Asserted here rather than from Python because the storage mode is not
/// observable through the Python surface — a silent downgrade to memory
/// would make an end-to-end mapped test pass vacuously.
///
/// It works for a structural reason worth pinning: `MappedGraph` mutates
/// the same petgraph `StableDiGraph` as `MemoryGraph` and differs only in
/// its derived mmap-backed indexes, so `apply_frames`' `maintain::*` calls
/// reach it unchanged.
#[test]
fn replays_onto_a_mapped_graph() {
    use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};
    let mut g = new_dir_graph_in_mode(StorageMode::Mapped, None).unwrap();
    assert!(g.graph.is_mapped(), "fixture must really be mapped");

    let frames = vec![
        frame(
            1,
            vec![
                upsert_node(1, "Alice", vec![("age", Value::Int64(30))]),
                upsert_node(2, "Bob", vec![]),
                knows(1, 2),
                set_labels(1, &["Employee"]),
            ],
        ),
        frame(
            2,
            vec![MutationOp::RemoveNode {
                node_type: "Person".into(),
                id: Value::Int64(2),
            }],
        ),
    ];
    apply_frames(&mut g, &frames, 0).unwrap();

    assert!(g.graph.is_mapped(), "replay must not switch the backend");
    assert_eq!(g.graph.node_count(), 1);
    assert_eq!(g.graph.edge_count(), 0, "edge went with the removed node");
    assert_eq!(labels_of(&mut g, 1), vec!["Person", "Employee"]);
    assert_eq!(prop(&mut g, 1, "age"), Some(Value::Int64(30)));
}

/// A property whose values differ in type across nodes must replay with
/// every value's type intact. Folding routes a whole node_type's rows
/// through one `DataFrame`, whose columns are singly-typed, so a mixed
/// column used to resolve to `String` (or `Float64` for an int/float
/// mix) and rewrite every cell in it.
#[test]
fn mixed_typed_property_keeps_every_value_type() {
    use chrono::NaiveDate;
    let date = NaiveDate::from_ymd_opt(2020, 1, 2).unwrap();
    let cases: Vec<(i64, Value)> = vec![
        (1, Value::Int64(1)),
        (2, Value::String("two".into())),
        (3, Value::Float64(3.5)),
        (4, Value::Boolean(true)),
        (5, Value::DateTime(date)),
    ];
    let mut g = DirGraph::new();
    let frames: Vec<WalFrame> = cases
        .iter()
        .enumerate()
        .map(|(i, (id, v))| {
            frame(
                i as u64 + 1,
                vec![upsert_node(*id, "n", vec![("mixedish", v.clone())])],
            )
        })
        .collect();
    apply_frames(&mut g, &frames, 0).unwrap();
    for (id, expected) in &cases {
        assert_eq!(
            prop(&mut g, *id, "mixedish").as_ref(),
            Some(expected),
            "node {id}"
        );
    }
}

/// Uniform typed properties retain both their values and useful metadata.
#[test]
fn single_typed_properties_keep_their_types_through_the_frame() {
    use chrono::NaiveDate;
    let props = vec![
        ("i", Value::Int64(7)),
        ("f", Value::Float64(0.5)),
        ("s", Value::String("x".into())),
        ("b", Value::Boolean(true)),
        (
            "d",
            Value::DateTime(NaiveDate::from_ymd_opt(2020, 1, 2).unwrap()),
        ),
        ("l", Value::List(vec![Value::Int64(1), Value::Int64(2)])),
    ];
    let mut g = DirGraph::new();
    apply_frames(
        &mut g,
        &[frame(1, vec![upsert_node(1, "a", props.clone())])],
        0,
    )
    .unwrap();
    for (key, expected) in props {
        assert_eq!(prop(&mut g, 1, key).as_ref(), Some(&expected), "{key}");
    }
    // Uniform values must not become mixed merely because replay is typed.
    let meta = g.get_node_type_metadata("Person").cloned().unwrap();
    assert!(
        !meta.values().any(|t| t == "mixed"),
        "single-typed metadata must stay precise: {meta:?}"
    );
}

/// The narrower numeric case: an int and a float under one property must
/// not promote the int to a float.
#[test]
fn int_and_float_under_one_property_do_not_promote() {
    let mut g = DirGraph::new();
    apply_frames(
        &mut g,
        &[frame(
            1,
            vec![
                upsert_node(1, "a", vec![("n", Value::Int64(2))]),
                upsert_node(2, "b", vec![("n", Value::Float64(2.5))]),
            ],
        )],
        0,
    )
    .unwrap();
    assert_eq!(prop(&mut g, 1, "n"), Some(Value::Int64(2)));
    assert_eq!(prop(&mut g, 2, "n"), Some(Value::Float64(2.5)));
}

/// Import conversion once turned Point properties into WKT text.
#[test]
fn point_property_survives_replay_as_a_point() {
    let mut g = DirGraph::new();
    apply_frames(
        &mut g,
        &[frame(
            1,
            vec![upsert_node(
                1,
                "a",
                vec![(
                    "loc",
                    Value::Point {
                        lat: 59.9,
                        lon: 10.7,
                    },
                )],
            )],
        )],
        0,
    )
    .unwrap();
    assert_eq!(
        prop(&mut g, 1, "loc"),
        Some(Value::Point {
            lat: 59.9,
            lon: 10.7
        })
    );
}

/// Last-write folding must preserve both the mixed value and the rest of its row.
#[test]
fn mixed_property_folds_with_later_ops_on_the_same_node() {
    let mut g = DirGraph::new();
    apply_frames(
        &mut g,
        &[
            frame(
                1,
                vec![
                    upsert_node(1, "a", vec![("m", Value::Int64(1))]),
                    upsert_node(2, "b", vec![("m", Value::String("two".into()))]),
                ],
            ),
            frame(
                2,
                vec![upsert_node(
                    1,
                    "a",
                    vec![("m", Value::Boolean(false)), ("age", Value::Int64(41))],
                )],
            ),
        ],
        0,
    )
    .unwrap();
    assert_eq!(prop(&mut g, 1, "m"), Some(Value::Boolean(false)));
    assert_eq!(prop(&mut g, 1, "age"), Some(Value::Int64(41)));
    assert_eq!(prop(&mut g, 2, "m"), Some(Value::String("two".into())));
    assert_eq!(
        g.get_node_type_metadata("Person").unwrap().get("m"),
        Some(&"mixed".to_string()),
        "a heterogeneous property is declared 'mixed', not left undeclared"
    );
}

/// Different ID variants remain distinct through a single recovery batch.
#[test]
fn nodes_whose_ids_differ_in_type_keep_their_ids() {
    let mut g = DirGraph::new();
    let string_id = MutationOp::UpsertNode {
        node_type: "Person".into(),
        id: Value::String("x".into()),
        title: Value::String("b".into()),
        properties: vec![("tag".to_string(), Value::String("str-id".into()))],
    };
    apply_frames(
        &mut g,
        &[frame(
            1,
            vec![
                upsert_node(1, "a", vec![("tag", Value::String("int-id".into()))]),
                string_id,
            ],
        )],
        0,
    )
    .unwrap();
    assert_eq!(g.graph.node_count(), 2);
    let idx = g
        .lookup_by_id("Person", &Value::Int64(1))
        .expect("the integer id must still be an integer");
    assert_eq!(g.graph.get_node_id(idx), Some(Value::Int64(1)));
    let idx = g
        .lookup_by_id("Person", &Value::String("x".into()))
        .expect("the string id must survive alongside it");
    assert_eq!(g.graph.get_node_id(idx), Some(Value::String("x".into())));
}

/// Titles must retain their variants alongside heterogeneous IDs.
#[test]
fn nodes_whose_titles_differ_in_type_keep_their_titles() {
    let mut g = DirGraph::new();
    let numeric_title = MutationOp::UpsertNode {
        node_type: "Person".into(),
        id: Value::Int64(2),
        title: Value::Int64(5),
        properties: vec![],
    };
    apply_frames(
        &mut g,
        &[frame(1, vec![upsert_node(1, "a", vec![]), numeric_title])],
        0,
    )
    .unwrap();
    let title = |g: &mut DirGraph, id: i64| {
        let idx = g.lookup_by_id("Person", &Value::Int64(id)).unwrap();
        g.graph.get_node_title(idx)
    };
    assert_eq!(title(&mut g, 1), Some(Value::String("a".into())));
    assert_eq!(title(&mut g, 2), Some(Value::Int64(5)));
}

/// An edge's endpoints are addressed by those same ids, so a mixed-id
/// node type must not cost the edges that reach it.
#[test]
fn edges_reach_endpoints_whose_ids_differ_in_type() {
    let mut g = DirGraph::new();
    let string_node = MutationOp::UpsertNode {
        node_type: "Person".into(),
        id: Value::String("x".into()),
        title: Value::String("b".into()),
        properties: vec![],
    };
    let edge = MutationOp::UpsertEdge {
        conn_type: "KNOWS".into(),
        src_type: "Person".into(),
        src_id: Value::Int64(1),
        tgt_type: "Person".into(),
        tgt_id: Value::String("x".into()),
        properties: vec![],
    };
    apply_frames(
        &mut g,
        &[frame(
            1,
            vec![upsert_node(1, "a", vec![]), string_node, knows(1, 2), edge],
        )],
        0,
    )
    .unwrap();
    // Nodes: 1, "x", and the id-2 stub `knows(1, 2)` vivifies — three,
    // not four. A `tgt_id` column holding both `2` and `"x"` renders the
    // integer endpoint as `"2"`, which matches nothing and vivifies a
    // *second* stub under a string id.
    assert_eq!(g.graph.node_count(), 3, "no stub under a stringified id");
    assert_eq!(g.graph.edge_count(), 2, "both edges land");
    let src = g.lookup_by_id("Person", &Value::Int64(1)).unwrap();
    for tgt_id in [Value::Int64(2), Value::String("x".into())] {
        let tgt = g
            .lookup_by_id("Person", &tgt_id)
            .unwrap_or_else(|| panic!("endpoint {tgt_id:?} must exist"));
        assert!(
            g.graph.find_edge(src, tgt).is_some(),
            "the edge to {tgt_id:?} must connect that node"
        );
    }
}

/// Mapped replay must retain the same mixed values as memory replay.
#[test]
fn mixed_typed_property_keeps_its_types_on_a_mapped_graph() {
    use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};
    let mut g = new_dir_graph_in_mode(StorageMode::Mapped, None).unwrap();
    assert!(g.graph.is_mapped(), "fixture must really be mapped");
    apply_frames(
        &mut g,
        &[frame(
            1,
            vec![
                upsert_node(1, "a", vec![("m", Value::Int64(1))]),
                upsert_node(2, "b", vec![("m", Value::String("two".into()))]),
            ],
        )],
        0,
    )
    .unwrap();
    assert!(g.graph.is_mapped(), "replay must not switch the backend");
    assert_eq!(prop(&mut g, 1, "m"), Some(Value::Int64(1)));
    assert_eq!(prop(&mut g, 2, "m"), Some(Value::String("two".into())));
}

/// Relationship properties once shared the lossy import conversion.
#[test]
fn mixed_typed_edge_property_keeps_every_value_type() {
    let mut g = DirGraph::new();
    let knows_with = |src: i64, tgt: i64, v: Value| MutationOp::UpsertEdge {
        conn_type: "KNOWS".into(),
        src_type: "Person".into(),
        src_id: Value::Int64(src),
        tgt_type: "Person".into(),
        tgt_id: Value::Int64(tgt),
        properties: vec![("w".to_string(), v)],
    };
    apply_frames(
        &mut g,
        &[frame(
            1,
            vec![
                upsert_node(1, "a", vec![]),
                upsert_node(2, "b", vec![]),
                upsert_node(3, "c", vec![]),
                knows_with(1, 2, Value::Int64(7)),
                knows_with(1, 3, Value::String("heavy".into())),
            ],
        )],
        0,
    )
    .unwrap();
    let w = |g: &mut DirGraph, src: i64, tgt: i64| -> Option<Value> {
        let s = g.lookup_by_id("Person", &Value::Int64(src))?;
        let t = g.lookup_by_id("Person", &Value::Int64(tgt))?;
        let e = g.graph.find_edge(s, t)?;
        g.graph
            .edge_weight(e)?
            .properties
            .iter()
            .find(|(k, _)| *k == InternedKey::from_str("w"))
            .map(|(_, v)| v.clone())
    };
    assert_eq!(w(&mut g, 1, 2), Some(Value::Int64(7)));
    assert_eq!(w(&mut g, 1, 3), Some(Value::String("heavy".into())));
}

#[test]
fn replaying_twice_is_idempotent() {
    let frames = vec![frame(
        1,
        vec![
            upsert_node(1, "Alice", vec![("age", Value::Int64(30))]),
            upsert_node(2, "Bob", vec![]),
            knows(1, 2),
        ],
    )];
    let mut g = DirGraph::new();
    apply_frames(&mut g, &frames, 0).unwrap();
    apply_frames(&mut g, &frames, 0).unwrap();
    assert_eq!(g.graph.node_count(), 2, "idempotent — no duplicate nodes");
    assert_eq!(g.graph.edge_count(), 1, "idempotent — no duplicate edge");
}

// ── type-level identity-field declarations ───────────────────────────

fn declare(node_type: &str, id_field: Option<&str>, title_field: Option<&str>) -> MutationOp {
    MutationOp::SetTypeFieldAliases {
        node_type: node_type.into(),
        id_field: id_field.map(str::to_string),
        title_field: title_field.map(str::to_string),
    }
}

fn aliases(g: &DirGraph, node_type: &str) -> (Option<String>, Option<String>) {
    (
        g.id_field_aliases.get(node_type).cloned(),
        g.title_field_aliases.get(node_type).cloned(),
    )
}

#[test]
fn declaration_replays_beside_the_rows_it_describes() {
    let mut g = DirGraph::new();
    let frames = vec![frame(
        1,
        vec![
            declare("Person", Some("uid"), Some("name")),
            upsert_node(1, "Alice", vec![("age", Value::Int64(30))]),
        ],
    )];
    apply_frames(&mut g, &frames, 0).unwrap();
    assert_eq!(
        aliases(&g, "Person"),
        (Some("uid".into()), Some("name".into()))
    );
    assert_eq!(prop(&mut g, 1, "age"), Some(Value::Int64(30)));
}

#[test]
fn a_declaration_only_frame_is_not_folded_away_as_empty() {
    // The plan has no node and no edge slot, so an emptiness test that only
    // counted those would skip the whole replay and silently drop the op.
    let mut g = DirGraph::new();
    let frames = vec![frame(1, vec![declare("Person", Some("uid"), None)])];
    assert_eq!(apply_frames(&mut g, &frames, 0).unwrap(), 1);
    assert_eq!(aliases(&g, "Person"), (Some("uid".into()), None));
}

#[test]
fn a_later_declaration_wins_per_field() {
    let mut g = DirGraph::new();
    let frames = vec![
        frame(1, vec![declare("Person", Some("uid"), Some("name"))]),
        frame(2, vec![declare("Person", Some("pid"), Some("label"))]),
    ];
    apply_frames(&mut g, &frames, 0).unwrap();
    assert_eq!(
        aliases(&g, "Person"),
        (Some("pid".into()), Some("label".into()))
    );
}

#[test]
fn an_undeclared_field_leaves_an_earlier_declaration_standing() {
    // `None` is "this call named no spelling", never "clear it". Reading it
    // as a clear would rebind the title spelling to the id column on the
    // very next chunk of a chunked load — the bug `should_update_title`
    // prevents on the live path.
    let mut g = DirGraph::new();
    let frames = vec![
        frame(1, vec![declare("Person", Some("uid"), Some("name"))]),
        frame(2, vec![declare("Person", Some("uid"), None)]),
    ];
    apply_frames(&mut g, &frames, 0).unwrap();
    assert_eq!(
        aliases(&g, "Person"),
        (Some("uid".into()), Some("name".into()))
    );

    // Same within one frame, where the fold sees both ops back to back.
    let mut g = DirGraph::new();
    apply_frames(
        &mut g,
        &[frame(
            1,
            vec![
                declare("Person", Some("uid"), Some("name")),
                declare("Person", None, None),
            ],
        )],
        0,
    )
    .unwrap();
    assert_eq!(
        aliases(&g, "Person"),
        (Some("uid".into()), Some("name".into()))
    );
}

#[test]
fn declarations_for_several_types_are_independent() {
    let mut g = DirGraph::new();
    let frames = vec![frame(
        1,
        vec![
            declare("Person", Some("uid"), None),
            declare("Company", Some("orgnr"), Some("nm")),
        ],
    )];
    apply_frames(&mut g, &frames, 0).unwrap();
    assert_eq!(aliases(&g, "Person"), (Some("uid".into()), None));
    assert_eq!(
        aliases(&g, "Company"),
        (Some("orgnr".into()), Some("nm".into()))
    );
}

#[test]
fn a_frame_at_or_below_the_checkpoint_lsn_declares_nothing() {
    // The checkpoint already holds the declaration; a frame it consumed must
    // not be re-applied over a spelling a later frame changed.
    let mut g = DirGraph::new();
    let frames = vec![
        frame(1, vec![declare("Person", Some("stale"), None)]),
        frame(2, vec![declare("Person", Some("uid"), None)]),
    ];
    apply_frames(&mut g, &frames, 1).unwrap();
    assert_eq!(aliases(&g, "Person"), (Some("uid".into()), None));
}

// ── schema, index and constraint declarations ────────────────────────
//
// Everything below lived only in the `.kgl` checkpoint before v6: a crash
// before the first `save()` recovered every row and none of what had been
// declared about them. Each case asserts the declaration is back, because the
// live-vs-recovered divergence is what the Python crash tests measure.

use crate::graph::constraints::{ConstraintKind, EntityKind};
use crate::graph::wal::PropertyIndexKind;

fn index_op(properties: &[&str], kind: PropertyIndexKind, present: bool) -> MutationOp {
    MutationOp::SetPropertyIndex {
        node_type: "Person".into(),
        properties: properties.iter().map(|p| (*p).to_string()).collect(),
        kind,
        present,
    }
}

fn not_null(properties: &[&str], present: bool) -> MutationOp {
    MutationOp::SetConstraint {
        name: Some("nn".into()),
        entity: EntityKind::Node,
        kind: ConstraintKind::NotNull,
        entity_type: "Person".into(),
        properties: properties.iter().map(|p| (*p).to_string()).collect(),
        declared_type: None,
        present,
    }
}

#[test]
fn metadata_declarations_replay_from_a_declaration_only_frame() {
    // No node and no edge slot: an emptiness test counting only those would
    // skip the whole replay and drop every op here.
    let mut g = DirGraph::new();
    let frames = vec![frame(
        1,
        vec![
            MutationOp::SetTypeParent {
                node_type: "Well".into(),
                parent_type: Some("Field".into()),
            },
            MutationOp::SetSchemaVersion { version: 7 },
            MutationOp::SetSpatialConfig {
                node_type: "Well".into(),
                config: r#"{"location":["lat","lon"]}"#.into(),
            },
            MutationOp::SetOntology {
                document: r#"{"classes":{"Thing":{"abstract":true}}}"#.into(),
            },
        ],
    )];
    assert_eq!(apply_frames(&mut g, &frames, 0).unwrap(), 1);
    assert_eq!(g.parent_types.get("Well"), Some(&"Field".to_string()));
    assert_eq!(g.user_schema_version, 7);
    assert_eq!(
        g.get_spatial_config("Well")
            .and_then(|c| c.location.clone()),
        Some(("lat".into(), "lon".into()))
    );
    assert!(g.ontology.classes.contains_key("Thing"));
}

#[test]
fn a_withdrawn_parent_type_replays_as_a_withdrawal() {
    let mut g = DirGraph::new();
    let frames = vec![
        frame(
            1,
            vec![MutationOp::SetTypeParent {
                node_type: "Well".into(),
                parent_type: Some("Field".into()),
            }],
        ),
        frame(
            2,
            vec![MutationOp::SetTypeParent {
                node_type: "Well".into(),
                parent_type: None,
            }],
        ),
    ];
    apply_frames(&mut g, &frames, 0).unwrap();
    assert_eq!(g.parent_types.get("Well"), None);
}

#[test]
fn a_declared_index_is_rebuilt_from_the_replayed_rows() {
    let mut g = DirGraph::new();
    let frames = vec![frame(
        1,
        vec![
            upsert_node(1, "Alice", vec![("age", Value::Int64(30))]),
            upsert_node(2, "Bob", vec![("age", Value::Int64(31))]),
            index_op(&["age"], PropertyIndexKind::Equality, true),
            index_op(&["age"], PropertyIndexKind::Range, true),
        ],
    )];
    apply_frames(&mut g, &frames, 0).unwrap();
    // Rebuilt from the rows the same frame carried, so the structure must be
    // populated rather than merely declared.
    assert!(g.has_index("Person", "age"));
    assert!(g
        .range_indices
        .contains_key(&("Person".to_string(), "age".to_string())));
}

#[test]
fn an_index_created_and_dropped_replays_as_dropped() {
    // The fold keeps the last decision about one index, not both events: an
    // ordered replay of create-then-drop that lost the ordering would leave a
    // structure the writer had removed.
    let mut g = DirGraph::new();
    let frames = vec![
        frame(
            1,
            vec![
                upsert_node(1, "Alice", vec![("age", Value::Int64(30))]),
                index_op(&["age"], PropertyIndexKind::Equality, true),
            ],
        ),
        frame(
            2,
            vec![index_op(&["age"], PropertyIndexKind::Equality, false)],
        ),
    ];
    apply_frames(&mut g, &frames, 0).unwrap();
    assert!(!g.has_index("Person", "age"));

    // …and the reverse order is a live index, not a dropped one.
    let mut g = DirGraph::new();
    let frames = vec![
        frame(
            1,
            vec![
                upsert_node(1, "Alice", vec![("age", Value::Int64(30))]),
                index_op(&["age"], PropertyIndexKind::Equality, false),
            ],
        ),
        frame(
            2,
            vec![index_op(&["age"], PropertyIndexKind::Equality, true)],
        ),
    ];
    apply_frames(&mut g, &frames, 0).unwrap();
    assert!(g.has_index("Person", "age"));
}

#[test]
fn a_declared_constraint_replays_over_the_rows_it_constrains() {
    let mut g = DirGraph::new();
    let frames = vec![frame(
        1,
        vec![
            upsert_node(1, "Alice", vec![("email", Value::String("a@x".into()))]),
            upsert_node(2, "Bob", vec![("email", Value::String("b@x".into()))]),
            not_null(&["email"], true),
            MutationOp::SetConstraint {
                name: Some("u".into()),
                entity: EntityKind::Node,
                kind: ConstraintKind::Unique,
                entity_type: "Person".into(),
                properties: vec!["email".into()],
                declared_type: None,
                present: true,
            },
        ],
    )];
    apply_frames(&mut g, &frames, 0).unwrap();
    // Declared *and* named: `DROP CONSTRAINT u` on a recovered graph has to
    // resolve, which needs the registry as well as the enforcement structure.
    assert!(g.constraint_by_name("u").is_some());
    assert!(g.constraint_by_name("nn").is_some());
    assert!(g.has_ddl_unique_declaration("Person", &["email".to_string()]));
}

#[test]
fn a_dropped_constraint_replays_as_a_withdrawal() {
    let mut g = DirGraph::new();
    let frames = vec![
        frame(
            1,
            vec![
                upsert_node(1, "Alice", vec![("email", Value::String("a@x".into()))]),
                not_null(&["email"], true),
            ],
        ),
        frame(2, vec![not_null(&["email"], false)]),
    ];
    apply_frames(&mut g, &frames, 0).unwrap();
    assert!(g.constraint_by_name("nn").is_none());
}

#[test]
fn a_constraint_the_recovered_rows_violate_refuses_replay_loudly() {
    // The writer that logged the declaration had it satisfied. Recovering rows
    // that violate it means the replayed state is not the committed state, and
    // a silently unenforced rule is the worse of the two failures.
    let mut g = DirGraph::new();
    let frames = vec![frame(
        1,
        vec![upsert_node(1, "Alice", vec![]), not_null(&["email"], true)],
    )];
    let error = apply_frames(&mut g, &frames, 0).unwrap_err();
    assert!(
        error.contains("could not reinstate a logged constraint"),
        "{error}"
    );
    assert_eq!(
        g.graph.node_count(),
        0,
        "a refused replay publishes nothing"
    );
}

// ── timeseries and embedding payloads ────────────────────────────────
//
// Bulk payloads rather than declarations, but the same blind spot before v7:
// rows recovered while `timeseries()` and `list_embeddings()` answered as if
// the load had never happened. Design:
// dev-docs/designs/timeseries-embeddings-wal-2026-09.md.

use crate::graph::features::timeseries::{NodeTimeseries, TimeseriesConfig};
use crate::graph::wal::EmbeddingWrite;

fn series(values: &[f64]) -> NodeTimeseries {
    NodeTimeseries {
        keys: (1..=values.len())
            .map(|month| chrono::NaiveDate::from_ymd_opt(2020, month as u32, 1).unwrap())
            .collect(),
        channels: std::collections::HashMap::from([("oil".to_string(), values.to_vec())]),
    }
}

fn set_series(id: i64, values: &[f64]) -> MutationOp {
    MutationOp::SetNodeTimeseries {
        node_type: "Person".into(),
        id: Value::Int64(id),
        timeseries: series(values),
    }
}

fn vectors(entries: Vec<(i64, Vec<f32>, Option<u64>)>, mode: EmbeddingWrite) -> MutationOp {
    MutationOp::SetEmbeddings {
        node_type: "Person".into(),
        text_column: "title".into(),
        dimension: 2,
        metric: Some("cosine".into()),
        model_id: Some("stub/v1".into()),
        entries: entries
            .into_iter()
            .map(|(id, vector, hash)| (Value::Int64(id), vector, hash))
            .collect(),
        mode,
    }
}

fn stored_series(g: &mut DirGraph, id: i64) -> Option<NodeTimeseries> {
    let idx = g.lookup_by_id("Person", &Value::Int64(id))?;
    g.get_node_timeseries(idx.index()).cloned()
}

fn store(g: &DirGraph) -> Option<&crate::graph::schema::EmbeddingStore> {
    g.embeddings
        .get(&("Person".to_string(), "title_emb".to_string()))
}

#[test]
fn timeseries_payload_replays_onto_its_logical_node() {
    let mut g = DirGraph::new();
    let frames = vec![frame(
        1,
        vec![
            upsert_node(1, "Alice", vec![]),
            upsert_node(2, "Bob", vec![]),
            set_series(2, &[1.0, 2.0]),
            MutationOp::SetTimeseriesConfig {
                node_type: "Person".into(),
                config: serde_json::to_string(&TimeseriesConfig {
                    resolution: "month".into(),
                    channels: vec!["oil".into()],
                    units: std::collections::HashMap::new(),
                    bin_type: None,
                })
                .unwrap(),
            },
        ],
    )];
    apply_frames(&mut g, &frames, 0).unwrap();
    assert_eq!(stored_series(&mut g, 2), Some(series(&[1.0, 2.0])));
    assert_eq!(stored_series(&mut g, 1), None);
    assert_eq!(
        g.timeseries_configs
            .get("Person")
            .map(|c| c.resolution.clone()),
        Some("month".to_string())
    );
}

#[test]
fn a_replayed_timeseries_is_last_writer_wins_and_idempotent() {
    let mut g = DirGraph::new();
    let frames = vec![
        frame(
            1,
            vec![upsert_node(1, "Alice", vec![]), set_series(1, &[1.0])],
        ),
        frame(2, vec![set_series(1, &[9.0, 8.0])]),
    ];
    apply_frames(&mut g, &frames, 0).unwrap();
    assert_eq!(stored_series(&mut g, 1), Some(series(&[9.0, 8.0])));
    // Replaying the same log over the state it produced converges.
    apply_frames(&mut g, &frames, 0).unwrap();
    assert_eq!(stored_series(&mut g, 1), Some(series(&[9.0, 8.0])));
}

#[test]
fn a_deleted_node_does_not_hand_its_payloads_to_the_slot_that_replaces_it() {
    // The logical-key case. The stores address `NodeIndex.index()`, which a
    // delete frees and the next create takes, so a physically-keyed payload
    // would surface on a node that never had one.
    let mut g = DirGraph::new();
    let frames = vec![
        frame(
            1,
            vec![
                upsert_node(1, "Alice", vec![]),
                set_series(1, &[1.0, 2.0]),
                vectors(vec![(1, vec![1.0, 0.0], Some(7))], EmbeddingWrite::Replace),
            ],
        ),
        frame(
            2,
            vec![MutationOp::RemoveNode {
                node_type: "Person".into(),
                id: Value::Int64(1),
            }],
        ),
        frame(3, vec![upsert_node(2, "Bob", vec![])]),
    ];
    apply_frames(&mut g, &frames, 0).unwrap();
    assert_eq!(
        stored_series(&mut g, 2),
        None,
        "Bob inherited a dead series"
    );
    assert_eq!(
        store(&g).map(|s| s.len()),
        Some(0),
        "Bob inherited a dead vector"
    );
}

#[test]
fn embedding_payloads_replay_with_their_provenance() {
    let mut g = DirGraph::new();
    let frames = vec![frame(
        1,
        vec![
            upsert_node(1, "Alice", vec![]),
            upsert_node(2, "Bob", vec![]),
            vectors(
                vec![(1, vec![1.0, 0.0], Some(11)), (2, vec![0.0, 1.0], Some(22))],
                EmbeddingWrite::Replace,
            ),
        ],
    )];
    apply_frames(&mut g, &frames, 0).unwrap();
    let idx = g.lookup_by_id("Person", &Value::Int64(1)).unwrap();
    let store = store(&g).expect("the store must exist");
    assert_eq!(store.len(), 2);
    assert_eq!(store.dimension, 2);
    assert_eq!(store.metric.as_deref(), Some("cosine"));
    assert_eq!(
        store.model_id.as_deref(),
        Some("stub/v1"),
        "the model stamp rides with the vectors, it is not re-derived"
    );
    assert_eq!(
        store.text_hashes.len(),
        2,
        "without the hashes, embed_texts(mode='changed') re-embeds the corpus"
    );
    assert_eq!(store.get_embedding(idx.index()), Some(&[1.0f32, 0.0][..]));
}

#[test]
fn upserted_embedding_batches_accumulate_across_frames() {
    // `add_embeddings` logs its own batch, not the whole store — so the fold
    // has to add them up rather than keep the last one.
    let mut g = DirGraph::new();
    let frames = vec![
        frame(
            1,
            vec![
                upsert_node(1, "Alice", vec![]),
                upsert_node(2, "Bob", vec![]),
                vectors(vec![(1, vec![1.0, 0.0], None)], EmbeddingWrite::Upsert),
            ],
        ),
        frame(
            2,
            vec![vectors(
                vec![(2, vec![0.0, 1.0], None)],
                EmbeddingWrite::Upsert,
            )],
        ),
    ];
    apply_frames(&mut g, &frames, 0).unwrap();
    assert_eq!(store(&g).map(|s| s.len()), Some(2));
}

#[test]
fn a_replaced_store_discards_the_batches_logged_before_it() {
    let mut g = DirGraph::new();
    let frames = vec![
        frame(
            1,
            vec![
                upsert_node(1, "Alice", vec![]),
                upsert_node(2, "Bob", vec![]),
                vectors(vec![(1, vec![1.0, 0.0], None)], EmbeddingWrite::Upsert),
            ],
        ),
        frame(
            2,
            vec![vectors(
                vec![(2, vec![0.0, 1.0], None)],
                EmbeddingWrite::Replace,
            )],
        ),
    ];
    apply_frames(&mut g, &frames, 0).unwrap();
    assert_eq!(store(&g).map(|s| s.len()), Some(1));
    let idx = g.lookup_by_id("Person", &Value::Int64(2)).unwrap();
    assert_eq!(
        store(&g).unwrap().get_embedding(idx.index()),
        Some(&[0.0f32, 1.0][..])
    );
}

#[test]
fn a_withdrawn_store_replays_as_absent() {
    let mut g = DirGraph::new();
    let frames = vec![
        frame(
            1,
            vec![
                upsert_node(1, "Alice", vec![]),
                vectors(vec![(1, vec![1.0, 0.0], None)], EmbeddingWrite::Replace),
            ],
        ),
        frame(
            2,
            vec![MutationOp::SetEmbeddings {
                node_type: "Person".into(),
                text_column: "title".into(),
                dimension: 0,
                metric: None,
                model_id: None,
                entries: vec![],
                mode: EmbeddingWrite::Withdraw,
            }],
        ),
    ];
    apply_frames(&mut g, &frames, 0).unwrap();
    assert!(store(&g).is_none());
}

#[test]
fn a_vector_index_declaration_rebuilds_from_the_replayed_vectors() {
    // Only the declaration is logged: the HNSW topology addresses store slots,
    // which this replay renumbers, so it has to be rebuilt after the vectors
    // land.
    let mut g = DirGraph::new();
    let frames = vec![frame(
        1,
        vec![
            upsert_node(1, "Alice", vec![]),
            upsert_node(2, "Bob", vec![]),
            vectors(
                vec![(1, vec![1.0, 0.0], None), (2, vec![0.0, 1.0], None)],
                EmbeddingWrite::Replace,
            ),
            MutationOp::SetVectorIndex {
                node_type: "Person".into(),
                text_column: "title".into(),
                metric: Some("cosine".into()),
                m: Some(16),
                ef_construction: None,
                ef_search: None,
                auto_refresh_limit: None,
                present: true,
            },
        ],
    )];
    apply_frames(&mut g, &frames, 0).unwrap();
    assert!(crate::graph::embeddings::has_vector_index(
        &g, "Person", "title"
    ));
}

#[test]
fn a_withdrawn_vector_index_replays_as_dropped() {
    let mut g = DirGraph::new();
    let declare = |present| MutationOp::SetVectorIndex {
        node_type: "Person".into(),
        text_column: "title".into(),
        metric: Some("cosine".into()),
        m: None,
        ef_construction: None,
        ef_search: None,
        auto_refresh_limit: None,
        present,
    };
    let frames = vec![
        frame(
            1,
            vec![
                upsert_node(1, "Alice", vec![]),
                vectors(vec![(1, vec![1.0, 0.0], None)], EmbeddingWrite::Replace),
                declare(true),
            ],
        ),
        frame(2, vec![declare(false)]),
    ];
    apply_frames(&mut g, &frames, 0).unwrap();
    assert!(!crate::graph::embeddings::has_vector_index(
        &g, "Person", "title"
    ));
    assert_eq!(store(&g).map(|s| s.len()), Some(1), "the vectors stay");
}

#[test]
fn payloads_replay_from_a_payload_only_frame() {
    // No node and no edge slot in the plan: an emptiness test counting only
    // those would skip the whole replay and drop every payload here.
    let mut g = DirGraph::new();
    apply_frames(
        &mut g,
        &[frame(
            1,
            vec![
                upsert_node(1, "Alice", vec![]),
                upsert_node(2, "Bob", vec![]),
            ],
        )],
        0,
    )
    .unwrap();
    let frames = vec![frame(
        2,
        vec![
            set_series(1, &[3.0]),
            vectors(vec![(1, vec![1.0, 0.0], None)], EmbeddingWrite::Replace),
        ],
    )];
    apply_frames(&mut g, &frames, 1).unwrap();
    assert_eq!(stored_series(&mut g, 1), Some(series(&[3.0])));
    assert_eq!(store(&g).map(|s| s.len()), Some(1));
}
