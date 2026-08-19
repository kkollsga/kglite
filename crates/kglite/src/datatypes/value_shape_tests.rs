//! Derived-`Ord` goldens for the projection-boundary value structs.
//!
//! # Why this file exists (Part N safety net, phase N1)
//!
//! `NodeValue`, `RelValue` and `PathValue` all derive `PartialOrd`/`Ord`. A
//! derived `Ord` is **field-declaration order**, so the sort key of every
//! `ORDER BY n` in the engine is a silent consequence of how the struct
//! happens to be written down. Part N (`dev-docs/plans/arc-values-rel-constraints-cdc2.md`)
//! replaces `properties: BTreeMap<String, Value>` with an `Arc`'d sorted flat
//! map. That change touches the `properties` field's *type* and its position
//! in the struct — either of which can reorder results without a single test
//! noticing, because nothing else in the suite pins the comparison order.
//!
//! These tests are **snapshot-style**: the expected ordering is written down
//! as a checked-in literal, never re-derived from the same `Ord` under test.
//! A test that sorts a vector and then asserts it is sorted proves nothing.
//!
//! **N2 must pass against these orderings unchanged.** If a representation
//! change makes one of them go red, the change altered a user-visible sort
//! order and needs a documented decision, not a refreshed expectation.

use super::{NodeValue, PathValue, RelValue, Value};
use std::cmp::Ordering;
use std::collections::BTreeMap;

/// Build a `NodeValue` from primitive parts, so the fixtures below read as
/// data rather than struct-literal noise.
fn node(id: u32, labels: &[&str], props: &[(&str, Value)]) -> NodeValue {
    NodeValue {
        id,
        labels: labels.iter().map(|s| (*s).to_string()).collect(),
        properties: props
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect::<BTreeMap<String, Value>>()
            .into(),
    }
}

fn rel(id: u32, start_id: u32, end_id: u32, rel_type: &str, props: &[(&str, Value)]) -> RelValue {
    RelValue {
        id,
        start_id,
        end_id,
        rel_type: rel_type.to_string(),
        properties: props
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect::<BTreeMap<String, Value>>()
            .into(),
    }
}

fn s(v: &str) -> Value {
    Value::String(v.to_string())
}

// ============================================================================
// NodeValue — field order is (id, labels, properties)
// ============================================================================

/// The whole point of the golden: `id` outranks `labels`, which outranks
/// `properties`. Each fixture below differs from its neighbour in exactly one
/// field, and the checked-in expectation is the *labels* of the sorted result
/// — a literal, not a re-derivation.
#[test]
fn node_value_ord_ranks_id_then_labels_then_properties() {
    // Deliberately shuffled input. Every entry carries a unique tag in the
    // `tag` property so the expected order can be written down by name.
    let unsorted = vec![
        // id 2, but a label that sorts FIRST and properties that sort FIRST.
        // If `labels` or `properties` outranked `id` this would lead.
        node(2, &["Aaa"], &[("tag", s("id2-aaa"))]),
        // id 1 with the highest-sorting label + properties. Must still lead,
        // because `id` is the first field.
        node(1, &["Zzz"], &[("tag", s("id1-zzz"))]),
        // id 1, lower label — decides against the entry above.
        node(1, &["Mmm"], &[("tag", s("id1-mmm-solo"))]),
        // id 1, same label as the previous, properties decide. `pair` sorts
        // before `solo` and is not a prefix of it, so the expected order below
        // reads off the tag alphabet directly.
        node(1, &["Mmm"], &[("tag", s("id1-mmm-pair")), ("zz", s("a"))]),
        node(1, &["Mmm"], &[("tag", s("id1-mmm-pair")), ("zz", s("b"))]),
    ];

    let mut sorted = unsorted.clone();
    sorted.sort();

    // ---- CHECKED-IN EXPECTATION (do not regenerate from `sorted`) ----------
    let expected_tags = [
        // id 1 group, ordered by labels: Mmm < Zzz. Within [Mmm], the
        // BTreeMap compares as a pair sequence: ("tag","...pair") beats
        // ("tag","...solo"), then the second pair ("zz", ..) breaks the tie.
        "id1-mmm-pair", // {tag: "id1-mmm-pair", zz: "a"}
        "id1-mmm-pair", // {tag: "id1-mmm-pair", zz: "b"}
        "id1-mmm-solo", // {tag: "id1-mmm-solo"}
        "id1-zzz",      // labels=[Zzz]
        // id 2 group, last despite the lowest-sorting label
        "id2-aaa",
    ];
    // ------------------------------------------------------------------------

    let got_tags: Vec<&str> = sorted
        .iter()
        .map(|n| match n.properties.get("tag") {
            Some(Value::String(t)) => t.as_str(),
            other => panic!("fixture lost its tag: {other:?}"),
        })
        .collect();
    assert_eq!(
        got_tags, expected_tags,
        "NodeValue Ord changed. Field declaration order (id, labels, properties) \
         is the sort key of every `ORDER BY n`; a reorder is a user-visible \
         semantic change, not a refactor."
    );

    // The two `id1-mmm-aaa` rows differ only in a property value, which proves
    // the third field is actually reached rather than the sort stopping early.
    assert_eq!(
        sorted[0].properties.get("zz"),
        Some(&s("a")),
        "properties must break the (id, labels) tie in key-then-value order"
    );
    assert_eq!(sorted[1].properties.get("zz"), Some(&s("b")));

    // And the pairwise statements the ordering above encodes, asserted
    // directly so a failure names the responsible field.
    assert_eq!(
        node(1, &["Zzz"], &[("z", s("z"))]).cmp(&node(2, &["Aaa"], &[("a", s("a"))])),
        Ordering::Less,
        "`id` must outrank `labels` and `properties`"
    );
    assert_eq!(
        node(1, &["Aaa"], &[("z", s("z"))]).cmp(&node(1, &["Bbb"], &[("a", s("a"))])),
        Ordering::Less,
        "`labels` must outrank `properties`"
    );
    assert_eq!(
        node(1, &["Aaa"], &[("a", s("a"))]).cmp(&node(1, &["Aaa"], &[("a", s("b"))])),
        Ordering::Less,
        "`properties` must be the final tie-break"
    );
}

/// Identical nodes compare `Equal`, and a sort leaves them adjacent in input
/// order. This is the tie case: it makes a tie-break *change* visible (a new
/// discriminating field would turn `Equal` into `Less`/`Greater`).
#[test]
fn node_value_ord_ties_are_equal_and_sort_is_stable() {
    let a = node(7, &["Person"], &[("age", Value::Int64(30))]);
    let b = node(7, &["Person"], &[("age", Value::Int64(30))]);
    assert_eq!(a.cmp(&b), Ordering::Equal, "identical NodeValues must tie");
    assert_eq!(a, b);

    // Two distinct-but-tying entries, distinguishable only by their position.
    // `sort` is stable, so a tie preserves input order; if a future field made
    // them comparable this assertion flips.
    let mut v = [(a.clone(), "first"), (b.clone(), "second")];
    v.sort_by(|l, r| l.0.cmp(&r.0));
    assert_eq!(
        v.iter().map(|(_, tag)| *tag).collect::<Vec<_>>(),
        ["first", "second"],
        "a NodeValue tie must stay a tie — sort stability is the only thing \
         separating equal records"
    );

    // The one field that *does* separate two otherwise-identical nodes in the
    // engine is `id` (the petgraph index), so distinct graph nodes never tie.
    let other_index = node(8, &["Person"], &[("age", Value::Int64(30))]);
    assert_eq!(a.cmp(&other_index), Ordering::Less);
}

/// Label-list comparison is lexicographic **element-wise, then by length** —
/// the `Vec<String>` derive. A secondary label appended to a node changes its
/// sort position; pinned so a labels-representation change is visible.
#[test]
fn node_value_ord_compares_label_lists_elementwise_then_by_length() {
    let mut v = [
        node(1, &["Person", "Employee"], &[]),
        node(1, &["Person"], &[]),
        node(1, &["Employee", "Person"], &[]),
        node(1, &["Employee"], &[]),
    ];
    v.sort();

    // ---- CHECKED-IN EXPECTATION -------------------------------------------
    let expected: Vec<Vec<&str>> = vec![
        vec!["Employee"],
        vec!["Employee", "Person"],
        vec!["Person"],
        vec!["Person", "Employee"],
    ];
    // ------------------------------------------------------------------------

    let got: Vec<Vec<&str>> = v
        .iter()
        .map(|n| n.labels.iter().map(String::as_str).collect())
        .collect();
    assert_eq!(got, expected);
}

// ============================================================================
// RelValue — field order is (id, start_id, end_id, rel_type, properties)
// ============================================================================

#[test]
fn rel_value_ord_ranks_id_start_end_type_then_properties() {
    let unsorted = vec![
        rel(2, 0, 0, "AAA", &[("tag", s("id2"))]),
        rel(1, 9, 9, "ZZZ", &[("tag", s("id1-s9"))]),
        rel(1, 5, 9, "ZZZ", &[("tag", s("id1-s5-e9"))]),
        rel(1, 5, 1, "ZZZ", &[("tag", s("id1-s5-e1"))]),
        rel(1, 5, 1, "AAA", &[("tag", s("id1-s5-e1-aaa"))]),
        rel(
            1,
            5,
            1,
            "AAA",
            &[("tag", s("id1-s5-e1-aaa2")), ("z", s("z"))],
        ),
    ];
    let mut sorted = unsorted.clone();
    sorted.sort();

    // ---- CHECKED-IN EXPECTATION -------------------------------------------
    let expected_tags = [
        "id1-s5-e1-aaa",  // id1, start5, end1, AAA, {tag: "...aaa"}
        "id1-s5-e1-aaa2", // id1, start5, end1, AAA, {tag: "...aaa2", z}
        "id1-s5-e1",      // id1, start5, end1, ZZZ
        "id1-s5-e9",      // id1, start5, end9
        "id1-s9",         // id1, start9
        "id2",            // id2 last, despite the lowest rel_type
    ];
    // ------------------------------------------------------------------------

    let got_tags: Vec<&str> = sorted
        .iter()
        .map(|r| match r.properties.get("tag") {
            Some(Value::String(t)) => t.as_str(),
            other => panic!("fixture lost its tag: {other:?}"),
        })
        .collect();
    assert_eq!(
        got_tags, expected_tags,
        "RelValue Ord changed — field order (id, start_id, end_id, rel_type, properties)"
    );
}

// ============================================================================
// PathValue — field order is (nodes, rels)
// ============================================================================

#[test]
fn path_value_ord_ranks_nodes_before_rels() {
    let n1 = node(1, &["Person"], &[]);
    let n2 = node(2, &["Person"], &[]);
    let r_low = rel(1, 1, 2, "AAA", &[]);
    let r_high = rel(9, 1, 2, "ZZZ", &[]);

    // Same rels, differing nodes → nodes decide.
    let a = PathValue {
        nodes: vec![n1.clone()],
        rels: vec![r_high.clone()],
    };
    let b = PathValue {
        nodes: vec![n2.clone()],
        rels: vec![r_low.clone()],
    };
    assert_eq!(
        a.cmp(&b),
        Ordering::Less,
        "`nodes` must outrank `rels` in PathValue's derived Ord"
    );

    // Same nodes, differing rels → rels decide.
    let c = PathValue {
        nodes: vec![n1.clone()],
        rels: vec![r_low.clone()],
    };
    let d = PathValue {
        nodes: vec![n1.clone()],
        rels: vec![r_high.clone()],
    };
    assert_eq!(c.cmp(&d), Ordering::Less);

    // A shorter node list sorts before a longer one that shares its prefix.
    let short = PathValue {
        nodes: vec![n1.clone()],
        rels: vec![],
    };
    let long = PathValue {
        nodes: vec![n1.clone(), n2.clone()],
        rels: vec![],
    };
    assert_eq!(short.cmp(&long), Ordering::Less);

    // Identical paths tie.
    assert_eq!(c.cmp(&c.clone()), Ordering::Equal);
}

// ============================================================================
// Value — the `Node`/`Relationship`/`Path` variants participate in `Value`'s
// own ordering, so a struct-level change reaches `ORDER BY` on any column.
// ============================================================================

/// `Value`'s ordering across the graph-entity variants is
/// **rank-first** — the hand-written `disc()` ladder in `Value::cmp`
/// (Node=12 < Relationship=13 < Path=14), which is deliberately *not* the
/// serde discriminant — then by the inner struct's `Ord`. Pinned because
/// `ORDER BY n` on a mixed-type column compares whole `Value`s, not
/// `NodeValue`s.
#[test]
fn value_ord_places_node_before_relationship_before_path() {
    let v_node = Value::Node(Box::new(node(9, &["Zzz"], &[])));
    let v_rel = Value::Relationship(Box::new(rel(0, 0, 0, "AAA", &[])));
    let v_path = Value::Path(Box::new(PathValue {
        nodes: vec![],
        rels: vec![],
    }));

    let mut v = [v_path.clone(), v_rel.clone(), v_node.clone()];
    v.sort();

    // ---- CHECKED-IN EXPECTATION -------------------------------------------
    let expected = ["Node", "Relationship", "Path"];
    // ------------------------------------------------------------------------

    let got: Vec<&str> = v
        .iter()
        .map(|x| match x {
            Value::Node(_) => "Node",
            Value::Relationship(_) => "Relationship",
            Value::Path(_) => "Path",
            other => panic!("unexpected variant in fixture: {other:?}"),
        })
        .collect();
    assert_eq!(
        got, expected,
        "Value variant order reaches user-visible ORDER BY; Node/Relationship/Path \
         discriminants are on-disk format (see the layout note on `Value`)"
    );

    // Within one variant, the inner struct's Ord decides.
    let low = Value::Node(Box::new(node(1, &["Aaa"], &[])));
    let high = Value::Node(Box::new(node(2, &["Aaa"], &[])));
    assert!(low < high);
}

/// The complete cross-variant rank ladder, as a checked-in literal.
///
/// `Value::cmp`'s `disc()` is the sort order every mixed-type `ORDER BY`
/// column sees. `Value::Map` sits at rank 11 — N2 decides `Value::Map`'s
/// container against the same byte-identity bar, and this is the ordering half
/// of that bar.
#[test]
fn value_cross_variant_rank_ladder_golden() {
    use chrono::NaiveDate;

    // Deliberately constructed in reverse of the expected order.
    let mut v = vec![
        Value::Path(Box::new(PathValue {
            nodes: vec![],
            rels: vec![],
        })),
        Value::Relationship(Box::new(rel(0, 0, 0, "R", &[]))),
        Value::Node(Box::new(node(0, &["N"], &[]))),
        Value::Map(super::PropMap::new()),
        Value::List(vec![]),
        Value::NodeRef(0),
        Value::Point { lat: 0.0, lon: 0.0 },
        Value::Duration {
            months: 0,
            days: 0,
            seconds: 0,
        },
        Value::DateTime(NaiveDate::from_ymd_opt(2020, 1, 1).unwrap()),
        s(""),
        Value::Float64(0.0),
        Value::Int64(0),
        Value::UniqueId(0),
        Value::Boolean(false),
        Value::Null,
        // Appended last to the enum (serde discriminant 15) AND ranked last
        // by `disc` — the two are independent facts and both are pinned.
        Value::Timestamp(
            NaiveDate::from_ymd_opt(2020, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
        ),
    ];
    v.sort();

    // ---- CHECKED-IN EXPECTATION (rank order, not re-derived) ---------------
    let expected = [
        "Null",
        "Boolean",
        "UniqueId",
        "Int64",
        "Float64",
        "String",
        "DateTime",
        "Duration",
        "Point",
        "NodeRef",
        "List",
        "Map",
        "Node",
        "Relationship",
        "Path",
        "Timestamp",
    ];
    // ------------------------------------------------------------------------

    let got: Vec<&str> = v.iter().map(|x| x.type_name()).collect();
    assert_eq!(
        got, expected,
        "Value's ORDER BY rank ladder changed (Value::cmp's `disc`). This is a \
         user-visible sort contract, independent of the serde discriminant."
    );
}
