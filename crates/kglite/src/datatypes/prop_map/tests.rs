//! Unit contract for [`PropMap`] — the invariants the rest of the engine is
//! allowed to assume, asserted directly rather than inferred from callers.

use super::*;
use crate::serde_codec::{decode_exact_with, encode_versioned, DecodeLimits, CURRENT_CODEC};

fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> T {
    decode_exact_with(
        CURRENT_CODEC,
        bytes,
        u64::MAX,
        DecodeLimits::new(u64::MAX, u64::MAX),
    )
    .expect("decode fixture")
}

fn s(v: &str) -> Value {
    Value::String(v.to_string())
}

fn btree(pairs: &[(&str, Value)]) -> BTreeMap<String, Value> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect()
}

#[test]
fn from_pairs_sorts_and_keeps_the_last_write() {
    let m = PropMap::from_pairs(vec![
        (PropKey::from("z"), Value::Int64(1)),
        (PropKey::from("a"), Value::Int64(2)),
        // Same key twice: the LAST push must win, matching `BTreeMap::insert`.
        (PropKey::from("m"), Value::Int64(3)),
        (PropKey::from("m"), Value::Int64(4)),
        (PropKey::from("a"), Value::Int64(5)),
    ]);
    assert_eq!(m.keys().collect::<Vec<_>>(), vec!["a", "m", "z"]);
    assert_eq!(m.get("a"), Some(&Value::Int64(5)));
    assert_eq!(m.get("m"), Some(&Value::Int64(4)));
    assert_eq!(m.get("z"), Some(&Value::Int64(1)));
    assert_eq!(m.len(), 3);
}

/// Three-in-a-row duplicates and a duplicate run at the very end are the two
/// shapes a hand-rolled dedup gets wrong.
#[test]
fn dedup_keep_last_handles_runs_and_tail_duplicates() {
    let m = PropMap::from_pairs(vec![
        (PropKey::from("a"), Value::Int64(1)),
        (PropKey::from("a"), Value::Int64(2)),
        (PropKey::from("a"), Value::Int64(3)),
        (PropKey::from("b"), Value::Int64(4)),
        (PropKey::from("z"), Value::Int64(5)),
        (PropKey::from("z"), Value::Int64(6)),
    ]);
    assert_eq!(m.keys().collect::<Vec<_>>(), vec!["a", "b", "z"]);
    assert_eq!(m.get("a"), Some(&Value::Int64(3)));
    assert_eq!(m.get("b"), Some(&Value::Int64(4)));
    assert_eq!(m.get("z"), Some(&Value::Int64(6)));
}

#[test]
fn empty_and_single_entry_edge_cases() {
    let empty = PropMap::new();
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.get("anything"), None);
    assert_eq!(empty.keys().count(), 0);

    let one = PropMap::from_pairs(vec![(PropKey::from("k"), s("v"))]);
    assert_eq!(one.len(), 1);
    assert_eq!(one.get("k"), Some(&s("v")));
    assert_eq!(one.get("j"), None);
    assert_eq!(one.get("l"), None);
}

#[test]
fn insert_replaces_in_place_and_keeps_sort() {
    let mut m = PropMap::from_pairs(vec![
        (PropKey::from("b"), Value::Int64(2)),
        (PropKey::from("d"), Value::Int64(4)),
    ]);
    assert_eq!(m.insert("a", Value::Int64(1)), None);
    assert_eq!(m.insert("c", Value::Int64(3)), None);
    assert_eq!(m.insert("e", Value::Int64(5)), None);
    assert_eq!(m.insert("c", Value::Int64(30)), Some(Value::Int64(3)));
    assert_eq!(m.keys().collect::<Vec<_>>(), vec!["a", "b", "c", "d", "e"]);
    assert_eq!(m.get("c"), Some(&Value::Int64(30)));
}

/// `Arc::make_mut` must copy-on-write: mutating one handle cannot be visible
/// through another. This is the property the whole "cheap clone" story rests
/// on, and it is exactly what a naive `Arc<Vec<_>>` + `unsafe` shortcut breaks.
#[test]
fn mutation_through_one_handle_does_not_leak_into_a_clone() {
    let original = PropMap::from_pairs(vec![(PropKey::from("a"), Value::Int64(1))]);
    let mut copy = original.clone();
    copy.insert("b", Value::Int64(2));
    copy.remove("a");

    assert_eq!(original.keys().collect::<Vec<_>>(), vec!["a"]);
    assert_eq!(original.get("a"), Some(&Value::Int64(1)));
    assert_eq!(copy.keys().collect::<Vec<_>>(), vec!["b"]);
}

#[test]
fn remove_and_retain() {
    let mut m = PropMap::from(btree(&[
        ("a", Value::Int64(1)),
        ("b", Value::Int64(2)),
        ("c", Value::Int64(3)),
    ]));
    assert_eq!(m.remove("b"), Some(Value::Int64(2)));
    assert_eq!(m.remove("b"), None);
    m.retain(|k, _| k != "a");
    assert_eq!(m.keys().collect::<Vec<_>>(), vec!["c"]);
}

// ============================================================================
// The contracts that outrank everything else: order and bytes
// ============================================================================

/// `Ord` must be `BTreeMap`'s `Ord` — lexicographic over `(key, value)` pairs,
/// then by length. Asserted against a `BTreeMap` computing the same
/// comparisons, because `ORDER BY n` reaches `properties` as its tie-break.
#[test]
fn ord_matches_btree_map_ord_pairwise() {
    let fixtures: Vec<Vec<(&str, Value)>> = vec![
        vec![],
        vec![("a", Value::Int64(1))],
        vec![("a", Value::Int64(2))],
        vec![("a", Value::Int64(1)), ("b", Value::Int64(1))],
        vec![("a", Value::Int64(1)), ("z", Value::Int64(1))],
        vec![("b", Value::Int64(0))],
        vec![("tag", s("pair")), ("zz", s("a"))],
        vec![("tag", s("pair")), ("zz", s("b"))],
        vec![("tag", s("solo"))],
    ];
    for a in &fixtures {
        for b in &fixtures {
            let (ba, bb) = (btree(a), btree(b));
            let (pa, pb) = (PropMap::from(ba.clone()), PropMap::from(bb.clone()));
            assert_eq!(
                pa.cmp(&pb),
                ba.cmp(&bb),
                "PropMap Ord diverged from BTreeMap Ord for {a:?} vs {b:?} — \
                 `ORDER BY n` results would move"
            );
            assert_eq!(pa == pb, ba == bb, "PropMap Eq diverged for {a:?} / {b:?}");
        }
    }
}

/// The byte contract, at the unit level. The system-level proof is
/// `graph::value_byte_identity_tests` (pinned hex + `.kgl` digest); this one
/// localises a failure to the container rather than the whole pipeline.
#[test]
fn postcard_bytes_are_identical_to_btree_map() {
    let cases: Vec<Vec<(&str, Value)>> = vec![
        vec![],
        vec![("k", s("v"))],
        vec![
            ("age", Value::Int64(30)),
            ("city", s("Oslo")),
            ("nested", Value::Map(PropMap::from(btree(&[("k", s("v"))])))),
            ("tags", Value::List(vec![s("a"), Value::Null])),
            ("title", s("Alice")),
        ],
        // A unicode key, and a key that is a prefix of another — the two
        // places a length-then-bytes comparison can disagree with a byte-wise
        // one.
        vec![
            ("Ålesund", Value::Int64(1)),
            ("ab", s("x")),
            ("abc", s("y")),
        ],
    ];
    for case in cases {
        let bt = btree(&case);
        let pm = PropMap::from(bt.clone());
        let want = encode_versioned(CURRENT_CODEC, &bt, u64::MAX).unwrap();
        let got = encode_versioned(CURRENT_CODEC, &pm, u64::MAX).unwrap();
        assert_eq!(
            got, want,
            "PropMap postcard bytes diverged from BTreeMap for {case:?} — the \
             representation change is no longer byte-invisible"
        );
        let back: PropMap = decode(&got);
        assert_eq!(back, pm, "PropMap did not survive its own round-trip");
    }
}

/// Postcard writes what `BTreeMap` wrote, so postcard must also *read* what
/// `BTreeMap` wrote — the `.kgl`/WAL read-compat direction.
#[test]
fn deserializes_bytes_written_by_btree_map() {
    let bt = btree(&[("a", Value::Int64(1)), ("b", s("two"))]);
    let bytes = encode_versioned(CURRENT_CODEC, &bt, u64::MAX).unwrap();
    let pm: PropMap = decode(&bytes);
    assert_eq!(pm.len(), 2);
    assert_eq!(pm.get("a"), Some(&Value::Int64(1)));
    assert_eq!(pm.get("b"), Some(&s("two")));
}

/// Foreign input (a JSON payload, a hand-built map) arrives in arbitrary key
/// order. The binary-search invariant is not optional, so deserialization
/// normalises rather than trusting the wire order.
///
/// `Value` is an externally-tagged enum in a self-describing format, hence the
/// `{"Int64": n}` payloads — that is `Value`'s existing JSON shape, unrelated
/// to the container, and the reason the C ABI converts through
/// `param::json_value_to_kglite_value` rather than deserializing `Value`
/// directly.
#[test]
fn deserializing_unsorted_json_normalises_the_order() {
    let json = r#"{"z": {"Int64": 1}, "a": {"Int64": 2}, "m": {"Int64": 3}}"#;
    let pm: PropMap = serde_json::from_str(json).unwrap();
    assert_eq!(pm.keys().collect::<Vec<_>>(), vec!["a", "m", "z"]);
    assert_eq!(pm.get("z"), Some(&Value::Int64(1)));
    // And it serializes back as a JSON object, in sorted key order.
    assert_eq!(
        serde_json::to_string(&pm).unwrap(),
        r#"{"a":{"Int64":2},"m":{"Int64":3},"z":{"Int64":1}}"#
    );
}

/// The map itself is shared on clone — one refcount, no per-entry copying.
/// This is what the `Arc` buys and what `collect(n)` / `ORDER BY n` cash in;
/// the *keys* are deliberately owned (see the module docs' measurement table).
#[test]
fn cloning_shares_the_backing_allocation() {
    let a = PropMap::from_sorted_pairs(vec![
        (PropKey::from("k1"), Value::Int64(1)),
        (PropKey::from("k2"), Value::Int64(2)),
    ]);
    let b = a.clone();
    assert!(
        std::ptr::eq(a.0.as_ptr(), b.0.as_ptr()),
        "cloning a PropMap deep-copied its entries — the whole point of the \
         Arc is that a node's properties travel between rows for a refcount"
    );
    assert_eq!(a, b);
}

#[test]
fn into_pairs_avoids_the_copy_when_unshared() {
    let m = PropMap::from_sorted_pairs(vec![(PropKey::from("a"), Value::Int64(1))]);
    let ptr = m.0.as_ptr();
    let pairs = m.into_pairs();
    assert_eq!(
        pairs.as_ptr(),
        ptr,
        "into_pairs copied a uniquely-owned map"
    );
}
