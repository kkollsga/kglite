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
