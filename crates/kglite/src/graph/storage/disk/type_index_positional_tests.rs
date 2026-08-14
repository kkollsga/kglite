//! Positional removal from a type bucket.
//!
//! A single-node `DETACH DELETE` used to walk the whole bucket with a hashed
//! set probe per member (`retain_in_type`), so deleting one node from a 1M-row
//! type cost milliseconds. These pin the replacement: locate the doomed
//! members by the store's sortedness invariant, then close the gaps with a
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

/// A bucket that lost the sortedness invariant — which happens as soon as a
/// freed `NodeIndex` slot is reused by a later create — must decline the fast
/// path rather than remove the wrong member. The caller falls back to the
/// full-bucket retain.
#[test]
fn an_out_of_order_bucket_declines_the_positional_path() {
    // Slot 1 was freed and reused, so it was appended after 2.
    let mut store = seeded(&[0, 2, 1]);
    assert!(
        store.positions_of("T", &[NodeIndex::new(1)]).is_none(),
        "a member the binary search cannot locate must decline, not guess"
    );
    // The members the search *can* still locate are handled normally.
    let hits = store.positions_of("T", &[NodeIndex::new(0)]).unwrap();
    assert_eq!(hits, vec![(0, NodeIndex::new(0))]);
    store.remove_positions("T", &hits);
    assert_eq!(
        bucket(&store, "T"),
        vec![2, 1]
            .into_iter()
            .map(NodeIndex::new)
            .collect::<Vec<_>>()
    );
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
