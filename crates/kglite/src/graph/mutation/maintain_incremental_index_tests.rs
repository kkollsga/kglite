//! Incremental id-index maintenance on the bulk append path (`add_nodes`).
//!
//! Split out of `maintain.rs` to keep that file under the source-quality line
//! ceiling, matching `maintain_delete_id_index_tests.rs`.
//!
//! `add_nodes` used to resolve its per-row conflict check by *materializing*
//! the whole type's id index into an owned map, and then to invalidate and
//! rebuild that index from scratch at return time — two O(N_type) passes per
//! call, whatever the batch size. These tests pin the replacement: the index
//! is built once (when absent) and afterwards only the call's own creations
//! are folded in.

use super::*;

/// `rows` frames of a single `id` column.
fn id_frame(ids: impl IntoIterator<Item = i64>) -> DataFrame {
    let rows: Vec<Vec<Value>> = ids.into_iter().map(|i| vec![Value::Int64(i)]).collect();
    DataFrame::from_cypher_rows(vec!["id".to_string()], rows).unwrap()
}

fn load(graph: &mut DirGraph, node_type: &str, ids: impl IntoIterator<Item = i64>) {
    add_nodes(
        graph,
        id_frame(ids),
        node_type.to_string(),
        "id".to_string(),
        Some("id".to_string()),
        None,
    )
    .unwrap();
}

/// **The cost pin.** A second `add_nodes` into an already-indexed type must
/// fold in its own rows, not re-derive the map.
///
/// Pinned structurally rather than by a clock: a sentinel entry that no live
/// node backs is planted in the index, and a rebuild — which re-derives every
/// entry from the type's members — is the only thing that can remove it. The
/// bench cell `test_bench_incremental_add_nodes_append` carries the timing
/// half (10.2 ms -> sub-ms at 200k existing rows); this carries the mechanism,
/// where a unit-level timing assertion would be pure flake.
#[test]
fn appending_to_an_indexed_type_does_not_rebuild_the_index() {
    let mut graph = DirGraph::new();
    load(&mut graph, "Person", 0..50);
    assert!(graph.id_indices.contains_key("Person"));

    // A mapping no member of the type can produce.
    let planted = NodeIndex::new(0);
    graph
        .id_indices
        .entry_or_default("Person".to_string())
        .insert(Value::Int64(9_999), planted);

    load(&mut graph, "Person", 50..60);

    assert_eq!(
        graph.id_indices.lookup("Person", &Value::Int64(9_999)),
        Some(planted),
        "the append rebuilt the whole id index instead of folding in its own rows"
    );
}

/// Correctness of that fold: every id — pre-existing and appended — resolves
/// through the **non-building** read, so a missing or misapplied delta shows
/// up as a `None` rather than being masked by the self-healing rebuild.
#[test]
fn appended_ids_resolve_without_a_rebuild() {
    let mut graph = DirGraph::new();
    load(&mut graph, "Person", 0..20);
    load(&mut graph, "Person", 20..30);

    for id in 0..30i64 {
        let idx = graph
            .id_indices
            .lookup("Person", &Value::Int64(id))
            .unwrap_or_else(|| panic!("id {id} lost from the index"));
        let stored = {
            let _guard = graph.graph.begin_query();
            graph
                .graph
                .node_view(idx)
                .map(|view| view.id().into_owned())
                .expect("index must point at a live node")
        };
        assert_eq!(stored, Value::Int64(id), "id {id} points at the wrong node");
    }
    assert_eq!(graph.id_indices.lookup("Person", &Value::Int64(30)), None);
    assert_eq!(graph.type_indices.get("Person").map(|m| m.len()), Some(30));
}

/// An upsert row must not add an index entry — it re-points nothing.
#[test]
fn an_upserting_batch_leaves_the_index_length_alone() {
    let mut graph = DirGraph::new();
    load(&mut graph, "Person", 0..10);
    let before = graph
        .lookup_by_id_readonly("Person", &Value::Int64(5))
        .unwrap();

    // Five updates (0..5) and five creations (10..15).
    load(&mut graph, "Person", (0..5).chain(10..15));

    assert_eq!(
        graph.id_indices.overlay_len("Person"),
        Some(15),
        "an upsert must not double-count the rows it updated"
    );
    assert_eq!(
        graph.id_indices.lookup("Person", &Value::Int64(5)),
        Some(before),
        "an updated row must keep pointing at the node it updated"
    );
    assert_eq!(graph.graph.node_count(), 15);
}

/// A deleted id re-loaded by `add_nodes` must resolve to the *new* node.
///
/// The delete evicts the entry in place and the append folds a fresh one in;
/// if the fold reused a stale position — or skipped it because the id was
/// "already there" — the id would resolve to a tombstoned slot.
#[test]
fn deleting_then_recreating_an_id_repoints_the_index() {
    let mut graph = DirGraph::new();
    load(&mut graph, "Person", 0..5);
    let doomed = graph
        .lookup_by_id_readonly("Person", &Value::Int64(2))
        .unwrap();

    let mut to_delete = HashSet::new();
    to_delete.insert(doomed);
    assert_eq!(detach_delete_nodes(&mut graph, &to_delete), (1, 0));
    assert_eq!(graph.id_indices.lookup("Person", &Value::Int64(2)), None);

    load(&mut graph, "Person", [2]);

    let reborn = graph
        .id_indices
        .lookup("Person", &Value::Int64(2))
        .expect("the recreated id must be indexed");
    let stored = {
        let _guard = graph.graph.begin_query();
        graph
            .graph
            .node_view(reborn)
            .map(|view| view.id().into_owned())
            .expect("the index must point at a live node")
    };
    assert_eq!(stored, Value::Int64(2));
    assert_eq!(graph.graph.node_count(), 5);
}

/// Two rows of one batch carrying the same id (no primary key declared) create
/// two nodes, and the index must resolve to the same one a full rebuild picks —
/// the later member of the type bucket.
#[test]
fn a_within_batch_duplicate_id_collapses_the_way_a_rebuild_would() {
    let mut graph = DirGraph::new();
    load(&mut graph, "Person", [1, 2, 2, 3]);

    assert_eq!(graph.type_indices.get("Person").map(|m| m.len()), Some(4));
    let resolved = graph
        .id_indices
        .lookup("Person", &Value::Int64(2))
        .expect("id 2 must resolve");

    // What the pre-fix full rebuild produced: the last member of the bucket
    // carrying that id wins.
    let expected = {
        let _guard = graph.graph.begin_query();
        let members = graph.type_indices.get("Person").unwrap().to_vec();
        members
            .into_iter()
            .rfind(|idx| {
                graph
                    .graph
                    .node_view(*idx)
                    .map(|view| view.id().into_owned() == Value::Int64(2))
                    .unwrap_or(false)
            })
            .expect("two members carry id 2")
    };
    assert_eq!(resolved, expected);
}

/// The first `add_nodes` into a fresh type still leaves a complete index
/// (issue #20's contract), and it is complete for *every* row — the partial
/// -entry hazard the batch path documents at `batch.rs`.
#[test]
fn the_first_append_leaves_a_complete_index() {
    let mut graph = DirGraph::new();
    load(&mut graph, "Person", 0..100);

    assert!(graph.id_indices.contains_key("Person"));
    assert_eq!(graph.id_indices.overlay_len("Person"), Some(100));
    for id in 0..100i64 {
        assert!(
            graph
                .id_indices
                .lookup("Person", &Value::Int64(id))
                .is_some(),
            "id {id} missing from a supposedly complete index"
        );
    }
}

/// **The partial-index hazard, closed.** Appending to a type whose index was
/// invalidated must rebuild it, never fold this call's rows into an absent
/// entry: `build_id_index` short-circuits on any entry that exists, so a
/// ten-row fold would be trusted forever as the whole type's index and every
/// older id would resolve to `None` — a silent wrong answer for
/// `MATCH (n {id: …})`, not a slow one. The hazard is documented at the batch
/// funnel (`mutation/batch.rs`); this is its test.
#[test]
fn appending_to_an_invalidated_index_rebuilds_it_whole() {
    let mut graph = DirGraph::new();
    load(&mut graph, "Person", 0..100);

    // The state every CREATE-then-DELETE leaves behind.
    graph.id_indices.remove("Person");
    assert!(!graph.id_indices.contains_key("Person"));

    load(&mut graph, "Person", 100..110);

    assert_eq!(
        graph.id_indices.overlay_len("Person"),
        Some(110),
        "the append folded into an absent entry instead of rebuilding"
    );
    for id in 0..110i64 {
        assert!(
            graph
                .id_indices
                .lookup("Person", &Value::Int64(id))
                .is_some(),
            "id {id} unreachable after an append onto an invalidated index"
        );
    }
}

/// **The fold must be indistinguishable from the rebuild it replaced.**
///
/// Compared wholesale rather than property by property: same variant — the
/// compact `Integer` map is ~8 bytes an entry against `General`'s ~60, and a
/// demotion would silently inflate a large type's index — and the same
/// `(id, node)` set, over a mixed batch of creations and upserts. Any
/// divergence the individual tests above did not think to ask about lands
/// here.
#[test]
fn the_folded_index_equals_the_rebuilt_one() {
    fn snapshot(graph: &DirGraph) -> (bool, Vec<(String, usize)>) {
        let (_, index) = graph
            .id_indices
            .iter()
            .into_iter()
            .find(|(name, _)| name == "Person")
            .expect("Person must be indexed");
        let compact = matches!(index, crate::graph::schema::TypeIdIndex::Integer(_));
        let mut entries: Vec<(String, usize)> = index
            .iter()
            .map(|(id, idx)| (format!("{id:?}"), idx.index()))
            .collect();
        entries.sort();
        (compact, entries)
    }

    let mut graph = DirGraph::new();
    load(&mut graph, "Person", 0..40);
    load(&mut graph, "Person", 40..55);
    // A mixed batch: fifteen upserts and ten creations.
    load(&mut graph, "Person", (40..55).chain(55..65));
    let folded = snapshot(&graph);

    graph.id_indices.remove("Person");
    graph.build_id_index("Person");
    let rebuilt = snapshot(&graph);

    assert_eq!(folded.0, rebuilt.0, "the fold changed the index variant");
    assert_eq!(folded.1, rebuilt.1, "the fold and the rebuild disagree");
    assert_eq!(folded.1.len(), 65);
}

/// **The same equivalence for the user indexes.** A creation-only append gives
/// each new node the per-node maintenance a `CREATE` would, instead of
/// rebuilding every covering index from every member of the type (measured
/// 2026-08-14: +6.1 ms per property index per call at 200k rows). The two must
/// leave byte-identical buckets, *in order* — `lookup_by_index` hands the
/// bucket straight to the matcher, so bucket order is the row order an indexed
/// `MATCH` without `ORDER BY` returns.
#[test]
fn folded_user_indexes_equal_the_rebuilt_ones() {
    fn buckets(graph: &DirGraph) -> Vec<(String, String, Vec<usize>)> {
        let mut out = Vec::new();
        for ((node_type, property), index) in &graph.property_indices {
            for (value, members) in index.iter() {
                out.push((
                    format!("prop:{node_type}.{property}"),
                    format!("{value:?}"),
                    members.iter().map(|idx| idx.index()).collect(),
                ));
            }
        }
        for ((node_type, property), index) in &graph.range_indices {
            for (value, members) in index.iter() {
                out.push((
                    format!("range:{node_type}.{property}"),
                    format!("{value:?}"),
                    members.iter().map(|idx| idx.index()).collect(),
                ));
            }
        }
        for ((node_type, properties), index) in &graph.composite_indices {
            for (value, members) in index.iter() {
                out.push((
                    format!("comp:{node_type}.{}", properties.join("+")),
                    format!("{value:?}"),
                    members.iter().map(|idx| idx.index()).collect(),
                ));
            }
        }
        out.sort();
        out
    }
    fn load_bucketed(graph: &mut DirGraph, ids: std::ops::Range<i64>) {
        let rows: Vec<Vec<Value>> = ids
            .map(|id| {
                vec![
                    Value::Int64(id),
                    Value::Int64(id % 3),
                    Value::String(format!("g{}", id % 2)),
                ]
            })
            .collect();
        let frame = DataFrame::from_cypher_rows(
            vec!["id".to_string(), "bucket".to_string(), "group".to_string()],
            rows,
        )
        .unwrap();
        add_nodes(
            graph,
            frame,
            "Person".to_string(),
            "id".to_string(),
            Some("id".to_string()),
            None,
        )
        .unwrap();
    }

    let mut graph = DirGraph::new();
    load_bucketed(&mut graph, 0..30);
    graph.create_index("Person", "bucket");
    graph.create_range_index("Person", "bucket");
    graph.create_composite_index("Person", &["bucket", "group"]);

    load_bucketed(&mut graph, 30..45);
    let folded = buckets(&graph);
    assert!(!folded.is_empty(), "the fixture must have indexed buckets");

    graph.refresh_indexes_for_type("Person");
    assert_eq!(
        folded,
        buckets(&graph),
        "the appended rows landed differently from the rebuild"
    );
    // And the appended rows are actually reachable through the index.
    let hits = graph
        .lookup_by_index("Person", "bucket", &Value::Int64(0))
        .expect("indexed lookup must resolve");
    assert_eq!(hits.len(), 15);
}

/// An upsert batch must rebuild rather than append: a moved value has to
/// vacate its old bucket, which the per-node append path never does.
#[test]
fn an_upserting_batch_keeps_the_user_index_correct() {
    let frame = |rows: Vec<(i64, i64)>| {
        DataFrame::from_cypher_rows(
            vec!["id".to_string(), "bucket".to_string()],
            rows.into_iter()
                .map(|(id, bucket)| vec![Value::Int64(id), Value::Int64(bucket)])
                .collect(),
        )
        .unwrap()
    };
    let load = |graph: &mut DirGraph, rows: Vec<(i64, i64)>| {
        add_nodes(
            graph,
            frame(rows),
            "Person".to_string(),
            "id".to_string(),
            Some("id".to_string()),
            None,
        )
        .unwrap();
    };

    let mut graph = DirGraph::new();
    load(&mut graph, vec![(1, 10), (2, 20)]);
    graph.create_index("Person", "bucket");
    let moved = graph
        .lookup_by_id_readonly("Person", &Value::Int64(1))
        .unwrap();

    // Row 1 moves from bucket 10 to 30; row 3 is new.
    load(&mut graph, vec![(1, 30), (3, 10)]);

    assert!(
        graph
            .lookup_by_index("Person", "bucket", &Value::Int64(10))
            .unwrap_or_default()
            .iter()
            .all(|idx| *idx != moved),
        "the vacated bucket still holds the moved node"
    );
    assert_eq!(
        graph
            .lookup_by_index("Person", "bucket", &Value::Int64(30))
            .unwrap_or_default(),
        vec![moved]
    );
    assert_eq!(
        graph
            .lookup_by_index("Person", "bucket", &Value::Int64(10))
            .unwrap_or_default()
            .len(),
        1,
        "the new row must be indexed"
    );
}

/// String ids take the `General` variant of the index; the fold must not
/// demote or lose them.
#[test]
fn string_ids_survive_an_incremental_append() {
    let mut graph = DirGraph::new();
    let frame = |ids: &[&str]| {
        let rows: Vec<Vec<Value>> = ids
            .iter()
            .map(|s| vec![Value::String((*s).to_string())])
            .collect();
        DataFrame::from_cypher_rows(vec!["id".to_string()], rows).unwrap()
    };
    for batch in [&["a", "b"][..], &["c"][..]] {
        add_nodes(
            &mut graph,
            frame(batch),
            "Doc".to_string(),
            "id".to_string(),
            Some("id".to_string()),
            None,
        )
        .unwrap();
    }

    for id in ["a", "b", "c"] {
        assert!(
            graph
                .id_indices
                .lookup("Doc", &Value::String(id.to_string()))
                .is_some(),
            "string id {id} lost"
        );
    }
}
