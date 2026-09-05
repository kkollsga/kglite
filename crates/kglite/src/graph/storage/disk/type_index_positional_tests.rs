//! Positional removal from a type bucket.
//!
//! A single-node `DETACH DELETE` used to walk the whole bucket with a hashed
//! set probe per member (`retain_in_type`), so deleting one node from a 1M-row
//! type cost milliseconds. These pin the replacement: locate the doomed
//! members by verified bucket coordinates, then close the gaps with a
//! memmove — **preserving bucket order**, which is the scan order an
//! un-`ORDER BY`'d `MATCH` returns and the coordinate system the statement
//! journal's `BucketRemoved` entries are recorded in.

use super::TypeIndexStore;
use petgraph::graph::NodeIndex;

fn bucket(store: &TypeIndexStore, name: &str) -> Vec<NodeIndex> {
    store
        .get(name)
        .map(|members| members.to_vec())
        .unwrap_or_default()
}

fn seeded(members: &[usize]) -> TypeIndexStore {
    let mut store = TypeIndexStore::new();
    for idx in members {
        store.push_to_type("T", NodeIndex::new(*idx));
    }
    store
}

/// The found positions are the bucket's own coordinates, and removal keeps
/// every survivor in place.
#[test]
fn removing_a_middle_member_preserves_bucket_order() {
    let mut store = seeded(&[0, 1, 2, 3, 4]);
    let hits = store
        .positions_of("T", &[NodeIndex::new(2)])
        .expect("a sorted bucket resolves by binary search");
    assert_eq!(hits, vec![(2, NodeIndex::new(2))]);

    store.remove_positions("T", &hits);
    assert_eq!(
        bucket(&store, "T"),
        vec![0, 1, 3, 4]
            .into_iter()
            .map(NodeIndex::new)
            .collect::<Vec<_>>()
    );
}

/// Several members at once, including the first and the last.
#[test]
fn removing_several_members_closes_every_gap_once() {
    let mut store = seeded(&[10, 11, 12, 13, 14, 15]);
    let doomed = [NodeIndex::new(14), NodeIndex::new(10), NodeIndex::new(12)];
    let hits = store
        .positions_of("T", &doomed)
        .expect("all three are present");
    assert_eq!(
        hits,
        vec![
            (0, NodeIndex::new(10)),
            (2, NodeIndex::new(12)),
            (4, NodeIndex::new(14))
        ],
        "positions must come back ascending, paired with their member"
    );

    store.remove_positions("T", &hits);
    assert_eq!(
        bucket(&store, "T"),
        vec![11, 13, 15]
            .into_iter()
            .map(NodeIndex::new)
            .collect::<Vec<_>>()
    );
}

/// Removing every member empties the bucket rather than leaving debris.
#[test]
fn removing_every_member_empties_the_bucket() {
    let mut store = seeded(&[7, 8]);
    let hits = store
        .positions_of("T", &[NodeIndex::new(7), NodeIndex::new(8)])
        .unwrap();
    store.remove_positions("T", &hits);
    assert!(bucket(&store, "T").is_empty());
}

/// A misplaced member outside the last slot still uses the full retain.
#[test]
fn an_out_of_order_interior_member_declines_the_positional_path() {
    let mut store = seeded(&[1, 2, 3, 0, 4, 5, 6, 7]);
    assert!(bucket(&store, "T")
        .binary_search(&NodeIndex::new(0))
        .is_err());
    assert!(store.positions_of("T", &[NodeIndex::new(0)]).is_none());
    store.retain_in_type("T", |member| *member != NodeIndex::new(0));
    assert_eq!(
        bucket(&store, "T"),
        (1..8).map(NodeIndex::new).collect::<Vec<_>>()
    );
}

/// A reused low slot at the tail is found without sorting live scan order.
#[test]
fn a_reused_tail_resolves_exact_coordinates_and_preserves_a_reader() {
    let mut store = seeded(&[10, 11, 12, 0]);
    let reader = store.clone();
    assert!(bucket(&store, "T")
        .binary_search(&NodeIndex::new(0))
        .is_err());
    let hits = store
        .positions_of("T", &[NodeIndex::new(0)])
        .expect("verified tail hit");
    assert_eq!(hits, vec![(3, NodeIndex::new(0))]);
    store.remove_positions("T", &hits);
    assert_eq!(
        bucket(&store, "T"),
        (10..13).map(NodeIndex::new).collect::<Vec<_>>()
    );
    assert_eq!(bucket(&reader, "T"), [10, 11, 12, 0].map(NodeIndex::new));
}

#[test]
fn a_tail_hit_combines_with_sorted_hits_without_reordering_survivors() {
    let mut store = seeded(&[10, 11, 12, 13, 14, 0]);
    let hits = store
        .positions_of("T", &[NodeIndex::new(0), NodeIndex::new(11)])
        .unwrap();
    assert_eq!(hits, vec![(1, NodeIndex::new(11)), (5, NodeIndex::new(0))]);
    store.remove_positions("T", &hits);
    assert_eq!(bucket(&store, "T"), [10, 12, 13, 14].map(NodeIndex::new));
}

#[test]
fn tail_search_preserves_missing_empty_and_duplicate_request_fallbacks() {
    let mut store = seeded(&[10, 11, 12, 0]);
    let before = bucket(&store, "T");
    for requested in [vec![0, 0], vec![0, 99], vec![99]] {
        let requested: Vec<_> = requested.into_iter().map(NodeIndex::new).collect();
        assert!(store.positions_of("T", &requested).is_none());
        assert_eq!(bucket(&store, "T"), before);
    }
    store.entry_or_default("Empty".into());
    assert!(store.positions_of("Empty", &[NodeIndex::new(0)]).is_none());
    let mut singleton = seeded(&[0]);
    assert!(singleton
        .positions_of("T", &[NodeIndex::new(0), NodeIndex::new(0)])
        .is_none());
}

/// A type with no bucket declines; it must not be materialized as an empty
/// one by the attempt.
#[test]
fn an_absent_type_declines_without_materializing() {
    let mut store = TypeIndexStore::new();
    assert!(store.positions_of("Ghost", &[NodeIndex::new(0)]).is_none());
    assert!(!store.contains_key("Ghost"));
    assert!(store.is_empty());
}

/// An empty doomed list is a no-op, not a decline — the caller must not fall
/// back to a full-bucket walk for it.
#[test]
fn no_doomed_members_is_a_no_op() {
    let mut store = seeded(&[0, 1]);
    let hits = store.positions_of("T", &[]).expect("nothing to find");
    assert!(hits.is_empty());
    store.remove_positions("T", &hits);
    assert_eq!(bucket(&store, "T").len(), 2);
}

/// A bucket shared with a fork must be flattened into this graph's own copy
/// before it is edited — never mutated in place under a reader.
#[test]
fn a_forked_bucket_is_copied_before_removal() {
    let mut store = seeded(&[0, 1, 2]);
    let reader = store.clone();

    let hits = store.positions_of("T", &[NodeIndex::new(1)]).unwrap();
    store.remove_positions("T", &hits);

    assert_eq!(
        bucket(&store, "T"),
        vec![0, 2]
            .into_iter()
            .map(NodeIndex::new)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        bucket(&reader, "T"),
        vec![0, 1, 2]
            .into_iter()
            .map(NodeIndex::new)
            .collect::<Vec<_>>(),
        "the fork must still see the pre-delete bucket"
    );
}
