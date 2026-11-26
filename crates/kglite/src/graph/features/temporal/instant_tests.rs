//! The whole-graph filter at an instant: the Disk instant mask agrees with
//! the evaluator, is cached per instant and refused over its cap; the slice
//! it gives is cached per instant.

use std::collections::{BTreeSet, HashMap};

use petgraph::graph::NodeIndex;

use super::*;
use crate::datatypes::values::Value;
use crate::graph::features::temporal::declarations::{declare, TemporalTarget};
use crate::graph::features::temporal::eval::IntervalConvention;
use crate::graph::session::execute::{execute_mut, ExecuteOptions};
use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};

/// Wells (declared, closed) on a field; `IN` declared half-open. At
/// 2005-06-01 wells 1 and 3 are visible, well 2 has not started, and only
/// the relationship from well 3 runs (well 1's ended in 2004).
fn fixture(mode: StorageMode, dir: &tempfile::TempDir) -> DirGraph {
    let path = matches!(mode, StorageMode::Disk).then(|| dir.path());
    let mut g = new_dir_graph_in_mode(mode, path).expect("graph");
    let params: HashMap<String, Value> = HashMap::new();
    execute_mut(
        &mut g,
        "CREATE (w1:Well {id: 1, vf: date('2000-01-01'), vt: date('2009-12-31')}),
                (w2:Well {id: 2, vf: date('2008-01-01')}),
                (w3:Well {id: 3, vf: date('2001-01-01')}),
                (f:Project {id: 10}),
                (w1)-[:IN {f: date('2000-01-01'), t: date('2004-01-01')}]->(f),
                (w2)-[:IN {f: date('2000-01-01')}]->(f),
                (w3)-[:IN {f: date('2000-01-01')}]->(f)",
        &ExecuteOptions::eager(&params),
    )
    .expect("load");
    let closed = IntervalConvention::Closed;
    declare(
        &mut g,
        &TemporalTarget::Node("Well".into()),
        "vf",
        "vt",
        closed,
    )
    .unwrap();
    let rel = TemporalTarget::Relationship {
        rel_type: "IN".into(),
        source_type: None,
    };
    declare(&mut g, &rel, "f", "t", IntervalConvention::HalfOpen).unwrap();
    g
}

fn at(text: &str) -> Instant {
    Instant::Date(chrono::NaiveDate::parse_from_str(text, "%Y-%m-%d").unwrap())
}

/// The ids of the visible nodes, and the (source id, target id) of the
/// visible relationships, under `filter`.
fn visible(g: &DirGraph, filter: &ElementFilter) -> (BTreeSet<i64>, BTreeSet<(i64, i64)>) {
    let _arena_guard = g.graph.begin_query();
    let id = |idx: NodeIndex| match g.graph.get_node_id(idx) {
        Some(Value::Int64(id)) => id,
        other => panic!("id {other:?}"),
    };
    let nodes = g
        .graph
        .node_indices()
        .filter(|&idx| filter.admits_node(g, idx))
        .map(id)
        .collect();
    let mut edges = BTreeSet::new();
    for idx in g.graph.node_indices() {
        for edge in g.graph.edges(idx) {
            let weight = edge.weight();
            if filter.admits_relationship(g, edge.id(), weight.connection_type, idx, edge.target())
            {
                edges.insert((id(idx), id(edge.target())));
            }
        }
    }
    (nodes, edges)
}

#[test]
fn the_disk_instant_mask_agrees_with_the_in_memory_filter_and_is_cached() {
    let dir = tempfile::tempdir().unwrap();
    let memory = fixture(StorageMode::Memory, &dir);
    let disk_dir = tempfile::tempdir().unwrap();
    let disk = fixture(StorageMode::Disk, &disk_dir);
    for t in ["2005-06-01", "1999-01-01", "2011-06-01"] {
        let (_, reference) = instant_filter(&memory, at(t)).unwrap();
        let (key, masked) = instant_filter(&disk, at(t)).unwrap();
        assert_eq!(key.instant, Some(at(t)), "Disk slices key on the instant");
        let (reference, masked) = (reference.unwrap(), masked.unwrap());
        assert_eq!(visible(&memory, &reference), visible(&disk, &masked), "{t}");
    }
    let (nodes, edges) = visible(
        &disk,
        &instant_filter(&disk, at("2005-06-01")).unwrap().1.unwrap(),
    );
    assert_eq!(nodes, BTreeSet::from([1, 3, 10]));
    assert_eq!(edges, BTreeSet::from([(3, 10)]));
    let first = endpoint_index::cached_disk_masks(&disk, at("2005-06-01")).expect("cached");
    let (_, again) = instant_filter(&disk, at("2005-06-01")).unwrap();
    drop(again);
    let second = endpoint_index::cached_disk_masks(&disk, at("2005-06-01")).unwrap();
    assert!(Arc::ptr_eq(&first, &second), "one pass per instant");
}

#[test]
fn a_disk_mask_over_its_cap_is_refused_before_it_is_built() {
    let dir = tempfile::tempdir().unwrap();
    let disk = fixture(StorageMode::Disk, &dir);
    let err = instant_filter_capped(&disk, at("2005-06-01"), 1).unwrap_err();
    assert!(err.contains("Disk mask cap"), "{err}");
    assert!(err.contains(DISK_MASK_CAP_ENV), "{err}");
    assert!(endpoint_index::cached_disk_masks(&disk, at("2005-06-01")).is_none());
    // The default cap holds it.
    assert!(instant_filter(&disk, at("2005-06-01")).unwrap().1.is_some());
}

#[test]
fn memory_mode_builds_no_instant_mask_and_caches_one_slice_per_instant() {
    let dir = tempfile::tempdir().unwrap();
    let g = fixture(StorageMode::Memory, &dir);
    let (key, filter) = instant_filter(&g, at("2005-06-01")).unwrap();
    assert!(filter.is_some());
    assert!(
        key.instant.is_none(),
        "indexed targets key on their segments"
    );
    assert!(endpoint_index::cached_disk_masks(&g, at("2005-06-01")).is_none());
    let first = slice_for(&g, at("2005-06-01")).unwrap();
    let second = slice_for(&g, at("2006-06-01")).unwrap();
    assert!(Arc::ptr_eq(&first, &second), "one segment, one slice");
    assert_eq!(first.graph().graph.node_count(), 3);
    let other = slice_for(&g, at("2011-06-01")).unwrap();
    assert!(!Arc::ptr_eq(&first, &other));
    assert_eq!(endpoint_index::cached_slice_count(&g), 2);
}

/// The node vector entry on Disk tests candidates against the instant mask
/// over the declared node labels, so an ambiguous declaration of a
/// relationship type the statement never reaches refuses nothing: the answer
/// is the in-memory one.
#[test]
fn an_unreached_ambiguous_relationship_type_does_not_refuse_node_retrieval() {
    use crate::graph::embeddings::set_embeddings;
    use crate::graph::schema::TemporalConfig;
    use crate::graph::session::execute::execute_read;

    let query = "FOR VALID_TIME AS OF date('2005-06-01') MATCH (w:Well) \
                 RETURN w.id AS id, vector_score(w, 'vf_emb', $v) AS s ORDER BY s DESC LIMIT 5";
    let params = HashMap::from([(
        "v".to_string(),
        Value::List(vec![Value::Float64(1.0), Value::Float64(0.0)]),
    )]);
    let answer = |mode: StorageMode| {
        let dir = tempfile::tempdir().unwrap();
        let mut g = fixture(mode, &dir);
        let vectors = [(1, [1.0f32, 0.0]), (2, [0.9, 0.1]), (3, [0.2, 0.8])];
        set_embeddings(
            &mut g,
            "Well",
            "vf",
            None,
            vectors.map(|(id, v)| (Value::Int64(id), v)),
        )
        .unwrap();
        for from in ["a", "b"] {
            let config = TemporalConfig {
                valid_from: from.to_string(),
                valid_to: format!("{from}_to"),
                convention: IntervalConvention::Closed,
                source_type: None,
                empty_when: None,
            };
            let unreached = TemporalTarget::Relationship {
                rel_type: "LEGACY".into(),
                source_type: None,
            };
            g.temporal.insert(&unreached, config, None);
        }
        assert!(g.temporal.is_ambiguous("LEGACY"));
        execute_read(&g, query, &ExecuteOptions::eager(&params))
            .unwrap_or_else(|e| panic!("{mode:?}: {e}"))
            .result
            .rows
    };
    let memory = answer(StorageMode::Memory);
    assert_eq!(memory.len(), 2, "wells 1 and 3 are valid: {memory:?}");
    assert_eq!(answer(StorageMode::Disk), memory);
}

/// The mask walk and the slice walk read each relationship's type and bounds
/// without materialising it: on Disk, `weight()` builds an owned copy of the
/// relationship's properties, so a walk that called it per relationship would
/// build one record for every relationship it walked. The mask build must
/// build none, and the slice one per kept relationship for its property count
/// plus one for the induced copy — never one per walked relationship.
#[test]
fn the_disk_mask_and_slice_walks_materialize_no_unkept_relationship() {
    const WELLS: usize = 2_000;
    let dir = tempfile::tempdir().unwrap();
    let mut g = new_dir_graph_in_mode(StorageMode::Disk, Some(dir.path())).expect("graph");
    let params: HashMap<String, Value> = HashMap::new();
    for query in [
        "CREATE (:Project {id: 0})".to_string(),
        format!(
            "MATCH (f:Project {{id: 0}}) UNWIND range(1, {WELLS}) AS i \
             CREATE (w:Well {{id: i, vf: date('2000-01-01')}})\
                    -[:IN {{f: date('2000-01-01'), t: CASE WHEN i % 100 = 0 \
                      THEN date('2030-01-01') ELSE date('2004-01-01') END}}]->(f)"
        ),
    ] {
        execute_mut(&mut g, &query, &ExecuteOptions::eager(&params)).expect("load");
    }
    let well = TemporalTarget::Node("Well".into());
    declare(&mut g, &well, "vf", "vt", IntervalConvention::Closed).unwrap();
    let rel = TemporalTarget::Relationship {
        rel_type: "IN".into(),
        source_type: None,
    };
    declare(&mut g, &rel, "f", "t", IntervalConvention::HalfOpen).unwrap();
    assert_eq!(g.graph.edge_count(), WELLS);

    let disk = g.graph.as_disk().expect("disk backend");
    disk.reset_materialization_counts();
    let (_, filter) = instant_filter(&g, at("2010-06-01")).unwrap();
    let filter = filter.expect("the declarations hide the ended relationships");
    assert_eq!(
        disk.edge_materializations(),
        0,
        "the mask walk over {WELLS} relationships built relationship records"
    );

    let slice = slice_at(
        &g,
        Some(&filter),
        SliceCaps {
            bytes: usize::MAX,
            disk_elements: None,
        },
    )
    .unwrap();
    let kept = slice.graph().graph.edge_count();
    assert_eq!(
        kept,
        WELLS / 100,
        "non-vacuity: the slice keeps the running ones"
    );
    assert!(
        disk.edge_materializations() <= 2 * kept as u64,
        "the slice walk built {} relationship records for {kept} kept relationships",
        disk.edge_materializations()
    );
}
