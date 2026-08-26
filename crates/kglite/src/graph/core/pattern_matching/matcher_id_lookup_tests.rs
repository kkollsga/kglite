//! Regression tests for the id-anchored point lookup in
//! [`PatternExecutor::try_index_lookup`].
//!
//! Split out of `matcher.rs` to keep that file under the source-quality line
//! ceiling, matching the existing `maintain_delete_id_index_tests.rs`.
//!
//! The contract under test is the anchor's *return shape*, which is what the
//! caller ([`PatternExecutor::find_matching_nodes`]) reads as "answered" vs
//! "scan the whole type":
//!
//! * `Some(v)` — the index answered; the caller uses `v` verbatim (unioned
//!   with a secondary-label scan when the queried label has carriers).
//! * `None`   — no index covers the pattern; the caller scans every node of
//!   the type.
//!
//! So `Some(vec![])` for an absent key *is* the structural assertion that no
//! scan happens: there is no other way for the caller to reach a scan.

use super::*;
use crate::graph::session::execute::{execute_mut, execute_read, ExecuteOptions};

fn run(graph: &mut DirGraph, query: &str) {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("setup query failed: {query}: {e}"));
}

/// Row count for a read query.
fn count_rows(graph: &DirGraph, query: &str) -> usize {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    let outcome = execute_read(graph, query, &opts)
        .unwrap_or_else(|e| panic!("read query failed: {query}: {e}"));
    outcome.result.rows.len()
}

fn eq_props(name: &str, value: Value) -> HashMap<String, PropertyMatcher> {
    HashMap::from([(name.to_string(), PropertyMatcher::Equals(value))])
}

/// Three `Doc` nodes with ids 1..3 and a warm id index.
fn seeded_docs() -> DirGraph {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "CREATE (:Doc {id: 1, name: 'a'}), (:Doc {id: 2, name: 'b'}), (:Doc {id: 3, name: 'c'})",
    );
    graph.build_id_index("Doc");
    assert!(graph.id_indices.contains_key("Doc"));
    graph
}

fn lookup(
    graph: &DirGraph,
    node_type: &str,
    props: &HashMap<String, PropertyMatcher>,
) -> Option<Vec<NodeIndex>> {
    PatternExecutor::new(graph, None).try_index_lookup(node_type, props)
}

/// **The fix.** A key that is absent from a built id index resolves to an
/// empty answer, not to a full-type scan.
///
/// `Some(vec![])` is the no-scan proof: `find_matching_nodes` scans exactly
/// when this returns `None`. Before the fix the anchor fell through on every
/// miss, which could only ever return nothing — at O(V) cost (0.39 ms at 50k
/// nodes, 1.56 ms at 200k, versus 2.5 µs for a hit).
#[test]
fn absent_id_with_index_answers_empty_without_scanning() {
    let graph = seeded_docs();
    assert_eq!(
        lookup(&graph, "Doc", &eq_props("id", Value::Int64(999))),
        Some(Vec::new()),
        "a missing key with a built index must answer empty, not fall through to a scan"
    );
    // The hit path is untouched.
    assert_eq!(
        lookup(&graph, "Doc", &eq_props("id", Value::Int64(2)))
            .expect("present key answers")
            .len(),
        1
    );
}

/// The same, through the user-declared per-type id alias.
#[test]
fn absent_alias_key_with_index_answers_empty() {
    let mut graph = DirGraph::new();
    run(&mut graph, "CREATE (:Star {id: 7, name: 'Sol'})");
    // `add_nodes(df, "Star", "starId", ...)` records this: the alias column
    // *is* the node's id, so `{starId: X}` routes to the id index.
    graph
        .id_field_aliases_mut()
        .insert("Star".to_string(), "starId".to_string());
    graph.build_id_index("Star");

    assert_eq!(
        lookup(&graph, "Star", &eq_props("starId", Value::Int64(8))),
        Some(Vec::new())
    );
    assert_eq!(
        lookup(&graph, "Star", &eq_props("starId", Value::Int64(7)))
            .expect("present alias key answers")
            .len(),
        1
    );
}

/// **The other direction of the pair.** With the id index *not* built, the
/// anchor must still be correct: present keys are found and absent keys are
/// not invented.
///
/// This is what makes the fast-empty safe — `lookup_by_id_readonly` is
/// self-healing (`IdIndexStore::lookup_or_build` builds and caches the type's
/// index on a miss, issue #20), so by the time it returns `None` the index
/// exists and is authoritative. Mutation check: make that build a no-op
/// (return `None` from `lookup_or_build` when the type is unindexed) and this
/// test goes red on the *present*-key assertions, because the anchor would
/// then answer empty for a node that exists.
#[test]
fn absent_id_without_index_still_correct() {
    let mut graph = seeded_docs();
    graph.id_indices.remove("Doc");
    assert!(
        !graph.id_indices.contains_key("Doc"),
        "precondition: the id index is not built"
    );

    // Present key: found (the read path self-heals the index).
    assert_eq!(
        lookup(&graph, "Doc", &eq_props("id", Value::Int64(2)))
            .expect("present key answers")
            .len(),
        1,
        "an unbuilt id index must not make an existing node unreachable"
    );

    // Absent key: empty, and no node is invented.
    let mut graph = seeded_docs();
    graph.id_indices.remove("Doc");
    assert_eq!(
        lookup(&graph, "Doc", &eq_props("id", Value::Int64(999)))
            .map(|v| v.len())
            .unwrap_or(0),
        0
    );

    // End-to-end, from an unbuilt index, both directions.
    let mut graph = seeded_docs();
    graph.id_indices.remove("Doc");
    assert_eq!(count_rows(&graph, "MATCH (n:Doc {id: 2}) RETURN n.name"), 1);
    assert_eq!(
        count_rows(&graph, "MATCH (n:Doc {id: 999}) RETURN n.name"),
        0
    );
}

/// A node reachable only through a *secondary* label must survive the
/// fast-empty: the queried label's id index covers primary members only, and
/// the caller unions a filtered scan of the secondary carriers.
#[test]
fn secondary_label_carrier_survives_the_fast_empty() {
    let mut graph = DirGraph::new();
    run(&mut graph, "CREATE (:Person:Director {id: 1, name: 'Ada'})");
    assert!(graph.has_secondary_labels);

    // The anchor itself finds nothing under :Director — its id index is over
    // primary members, of which there are none.
    assert_eq!(
        lookup(&graph, "Director", &eq_props("id", Value::Int64(1)))
            .map(|v| v.len())
            .unwrap_or(0),
        0
    );
    // …but the query still returns the node, because `find_matching_nodes`
    // unions the secondary-label scan with whatever the anchor answered.
    assert_eq!(
        count_rows(&graph, "MATCH (n:Director {id: 1}) RETURN n.name"),
        1
    );
    assert_eq!(
        count_rows(&graph, "MATCH (n:Director {id: 2}) RETURN n.name"),
        0
    );
}

/// Numeric coercion must agree with `values_equal`, which is what the scan
/// the anchor replaces would have used: a `Float64`-stored id is matchable by
/// an integer literal. Without this the fast-empty would turn a working
/// lookup into an empty result.
#[test]
fn integral_float_id_is_reachable_by_int_query() {
    let mut graph = DirGraph::new();
    run(&mut graph, "CREATE (:Doc {id: 5.0, name: 'float-id'})");
    graph.build_id_index("Doc");

    assert_eq!(
        lookup(&graph, "Doc", &eq_props("id", Value::Int64(5)))
            .expect("the anchor answers")
            .len(),
        1,
        "Int64(5) must resolve a Float64(5.0) id — values_equal says they are equal"
    );
    assert_eq!(count_rows(&graph, "MATCH (n:Doc {id: 5}) RETURN n.name"), 1);
    // The IN-on-id anchor shares the same index lookup and must agree.
    assert_eq!(
        count_rows(&graph, "MATCH (n:Doc) WHERE n.id IN [5] RETURN n.name"),
        1
    );
    assert_eq!(count_rows(&graph, "MATCH (n:Doc {id: 6}) RETURN n.name"), 0);
}

/// An UNWIND over a mixed hit/miss id list returns exactly the hits — the
/// shape whose miss half cost 6.7 s over 16k absent ids.
#[test]
fn unwind_mixed_hit_and_miss_returns_only_hits() {
    let graph = seeded_docs();
    assert_eq!(
        count_rows(
            &graph,
            "UNWIND [1, 99, 2, 98, 3] AS i MATCH (n:Doc {id: i}) RETURN n.name",
        ),
        3
    );
    assert_eq!(
        count_rows(
            &graph,
            "UNWIND [97, 98, 99] AS i MATCH (n:Doc {id: i}) RETURN n.name",
        ),
        0
    );
}

/// String ids: the miss must be empty and the hit unaffected.
#[test]
fn string_id_hit_and_miss() {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "CREATE (:Doc {id: 's1', v: 1}), (:Doc {id: 's2', v: 2})",
    );
    graph.build_id_index("Doc");

    assert_eq!(
        lookup(&graph, "Doc", &eq_props("id", Value::String("nope".into()))),
        Some(Vec::new())
    );
    assert_eq!(
        lookup(&graph, "Doc", &eq_props("id", Value::String("s2".into())))
            .expect("present key answers")
            .len(),
        1
    );
}

// ── The untyped `{id: X}` anchor: cross-type union, literal == param ──────
//
// `MATCH (n {id: 2})` has no label to scope the id space, so it means "every
// node whose id is 2" — one per type that carries the key. Two defects lived
// on that path until 2026-08-15:
//
//   * it returned on the FIRST type whose index answered, so a cross-type id
//     collision collapsed to one arbitrary node — arbitrary because
//     `type_indices` is a `HashMap`, so which one survived was not stable; and
//   * it read only `PropertyMatcher::Equals`, so `{id: $x}` fell past the
//     anchor into the exhaustive scan and answered a *different* (complete)
//     set than the literal.

/// Two nodes per label under three labels, ids {1, 2} reused across all of
/// them and unique within each. Every id index is warm.
fn cross_type_docs() -> DirGraph {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "CREATE (:Alpha {id: 1, name: 'a1'}), (:Alpha {id: 2, name: 'a2'}), \
         (:Beta {id: 1, name: 'b1'}), (:Beta {id: 2, name: 'b2'}), \
         (:Gamma {id: 1, name: 'g1'}), (:Gamma {id: 2, name: 'g2'})",
    );
    for node_type in ["Alpha", "Beta", "Gamma"] {
        graph.build_id_index(node_type);
        assert!(graph.id_indices.contains_key(node_type));
    }
    graph
}

/// Run an untyped node pattern through the public matcher entry point.
fn untyped_lookup(
    graph: &DirGraph,
    props: HashMap<String, PropertyMatcher>,
    params: &HashMap<String, Value>,
) -> Vec<NodeIndex> {
    let pattern = NodePattern {
        variable: None,
        node_type: None,
        extra_labels: Vec::new(),
        alt_labels: None,
        properties: Some(props),
        label_params: Vec::new(),
    };
    PatternExecutor::new_lightweight_with_params(graph, None, params)
        .find_matching_nodes_pub(&pattern)
        .expect("untyped id anchor must not error")
}

/// The union: one node per type carrying the id, in ascending `NodeIndex`
/// order. Ascending order is the contract because the source of the hits is a
/// `HashMap` iteration — without the sort the *row order* would change between
/// processes even once the set became right.
#[test]
fn untyped_id_anchor_unions_every_type() {
    let graph = cross_type_docs();
    let params = HashMap::new();

    let hits = untyped_lookup(&graph, eq_props("id", Value::Int64(2)), &params);
    assert_eq!(
        hits.len(),
        3,
        "one node per type carrying id 2, not the first"
    );
    let mut sorted = hits.clone();
    sorted.sort_unstable();
    assert_eq!(
        hits, sorted,
        "hits must be emitted in ascending NodeIndex order"
    );

    // The names prove the union spans all three types rather than repeating
    // one type's node.
    assert_eq!(count_rows(&graph, "MATCH (n {id: 2}) RETURN n.name"), 3);
    assert_eq!(count_rows(&graph, "MATCH (n {id: 1}) RETURN n.name"), 3);
}

/// The parameter spelling resolves through the same anchor and answers the
/// identical set — not merely the same count.
#[test]
fn untyped_id_anchor_param_equals_literal() {
    let graph = cross_type_docs();

    let literal = untyped_lookup(&graph, eq_props("id", Value::Int64(2)), &HashMap::new());

    let params: HashMap<String, Value> = HashMap::from([("x".to_string(), Value::Int64(2))]);
    let param_props = HashMap::from([(
        "id".to_string(),
        PropertyMatcher::EqualsParam("x".to_string()),
    )]);
    let parameterized = untyped_lookup(&graph, param_props, &params);

    assert_eq!(
        parameterized, literal,
        "`{{id: $x}}` and `{{id: 2}}` denote the same nodes and must answer identically"
    );
    assert_eq!(literal.len(), 3);
}

/// The absent-id contract holds on the untyped path for both spellings: empty,
/// and — because every type's index is built and authoritative — without a
/// scan inventing anything.
#[test]
fn untyped_absent_id_is_empty_for_literal_and_param() {
    let graph = cross_type_docs();

    assert!(untyped_lookup(&graph, eq_props("id", Value::Int64(999)), &HashMap::new()).is_empty());

    let params: HashMap<String, Value> = HashMap::from([("x".to_string(), Value::Int64(999))]);
    let param_props = HashMap::from([(
        "id".to_string(),
        PropertyMatcher::EqualsParam("x".to_string()),
    )]);
    assert!(untyped_lookup(&graph, param_props, &params).is_empty());

    // …and the typed anchor's own fast-empty is untouched by the union change.
    assert_eq!(
        lookup(&graph, "Alpha", &eq_props("id", Value::Int64(999))),
        Some(Vec::new()),
        "the typed fast-empty must still answer without falling through to a scan"
    );
}

/// A param naming nothing in the bound set must not be *treated* as a literal
/// id: the anchor declines and the ordinary property scan answers (nothing, as
/// no node stores a property called `id`).
#[test]
fn untyped_unbound_param_does_not_anchor() {
    let graph = cross_type_docs();
    let param_props = HashMap::from([(
        "id".to_string(),
        PropertyMatcher::EqualsParam("x".to_string()),
    )]);
    assert!(untyped_lookup(&graph, param_props, &HashMap::new()).is_empty());
}

/// A conjunction whose id predicate misses is empty regardless of the other
/// predicates, and one whose id predicate hits still honours them.
#[test]
fn id_miss_with_extra_predicates_is_empty() {
    let graph = seeded_docs();
    assert_eq!(
        count_rows(
            &graph,
            "MATCH (n:Doc {id: 999}) WHERE n.name = 'a' RETURN n.name"
        ),
        0
    );
    assert_eq!(
        count_rows(
            &graph,
            "MATCH (n:Doc {id: 1}) WHERE n.name = 'a' RETURN n.name"
        ),
        1
    );
    assert_eq!(
        count_rows(
            &graph,
            "MATCH (n:Doc {id: 1}) WHERE n.name = 'zzz' RETURN n.name"
        ),
        0
    );
}

// ── The IN-list anchors bind each node once ──────────────────────────────

use crate::graph::core::membership::MembershipSet;

fn in_props(name: &str, values: Vec<Value>) -> HashMap<String, PropertyMatcher> {
    HashMap::from([(
        name.to_string(),
        PropertyMatcher::In(MembershipSet::new(values)),
    )])
}

/// **The fix.** The id anchor is driven by the list — one index probe per
/// element — so a repeated element used to emit the same node once per
/// occurrence. A `MATCH` binds each node once.
///
/// Mutation check: drop `dedup_candidates` from the id arm and this goes red
/// with `[1, 1, 2]`.
#[test]
fn id_in_list_with_duplicate_entries_binds_each_node_once() {
    let graph = seeded_docs();
    let one =
        lookup(&graph, "Doc", &eq_props("id", Value::Int64(1))).expect("present key answers")[0];
    let two =
        lookup(&graph, "Doc", &eq_props("id", Value::Int64(2))).expect("present key answers")[0];

    assert_eq!(
        lookup(
            &graph,
            "Doc",
            &in_props(
                "id",
                vec![Value::Int64(1), Value::Int64(1), Value::Int64(2)]
            )
        ),
        Some(vec![one, two]),
        "a duplicated list entry must not bind its node twice"
    );
}

/// Dedup keeps the **first** occurrence, so the anchor still answers in list
/// order — the order it has always returned, and the one a caller reading
/// rows without an `ORDER BY` sees.
#[test]
fn id_in_list_dedup_preserves_first_occurrence_order() {
    let graph = seeded_docs();
    let idx = |v: i64| {
        lookup(&graph, "Doc", &eq_props("id", Value::Int64(v))).expect("present key answers")[0]
    };
    assert_eq!(
        lookup(
            &graph,
            "Doc",
            &in_props(
                "id",
                vec![
                    Value::Int64(3),
                    Value::Int64(1),
                    Value::Int64(3),
                    Value::Int64(2),
                ]
            )
        ),
        Some(vec![idx(3), idx(1), idx(2)]),
        "first occurrence wins, so the surviving order is the list's"
    );
}

/// Equal *values* are not the only way two elements land on one node: the id
/// index coerces across the numeric family (`values_equal`), so `1` and `1.0`
/// are two distinct list elements resolving to one node. Deduping the list
/// instead of the candidates would miss this.
#[test]
fn id_in_list_coercion_equal_spellings_bind_once() {
    let graph = seeded_docs();
    let one =
        lookup(&graph, "Doc", &eq_props("id", Value::Int64(1))).expect("present key answers")[0];
    assert_eq!(
        lookup(
            &graph,
            "Doc",
            &in_props("id", vec![Value::Int64(1), Value::Float64(1.0)])
        ),
        Some(vec![one])
    );
}

/// Past [`MembershipSet`]'s linear/indexed crossover the list is probed
/// through its hash index; the anchor still walks it element-wise, so the
/// duplicate must be dropped on the same terms.
#[test]
fn id_in_long_list_with_duplicates_binds_each_node_once() {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "UNWIND range(1, 200) AS i CREATE (:Doc {id: i, name: 'n'})",
    );
    graph.build_id_index("Doc");

    // 100 distinct ids, each named twice — 200 elements, well past the
    // 8-element crossover and past the 64-element list size that broke the
    // bare point lookup in 0.11.2.
    let values: Vec<Value> = (1..=100)
        .chain(1..=100)
        .map(|i| Value::Int64(i as i64))
        .collect();
    let hits = lookup(&graph, "Doc", &in_props("id", values)).expect("the anchor answers");
    assert_eq!(hits.len(), 100);
    let distinct: HashSet<NodeIndex> = hits.iter().copied().collect();
    assert_eq!(distinct.len(), 100, "no node may appear twice");
}

/// The same defect lived in the sibling arm: `IN` on a **non-id** property
/// that carries a per-type index probes the index once per element and
/// concatenated the answers.
///
/// The control is the same query without the index — that path filters each
/// node once and always answered 2.
#[test]
fn indexed_property_in_list_with_duplicates_binds_each_node_once() {
    let mut graph = seeded_docs();
    let unindexed = lookup(
        &graph,
        "Doc",
        &in_props(
            "name",
            vec![
                Value::String("a".to_string()),
                Value::String("a".to_string()),
                Value::String("b".to_string()),
            ],
        ),
    );
    assert!(
        unindexed.is_none(),
        "without an index the property IN has no anchor — the caller scans"
    );

    graph.create_index("Doc", "name");
    let hits = lookup(
        &graph,
        "Doc",
        &in_props(
            "name",
            vec![
                Value::String("a".to_string()),
                Value::String("a".to_string()),
                Value::String("b".to_string()),
            ],
        ),
    )
    .expect("the property index answers");
    assert_eq!(hits.len(), 2, "a duplicated list entry binds its node once");
    let distinct: HashSet<NodeIndex> = hits.iter().copied().collect();
    assert_eq!(distinct.len(), 2);
}

/// End-to-end through the executor: the row count a user sees.
#[test]
fn duplicate_id_in_list_returns_one_row_per_node() {
    let graph = seeded_docs();
    assert_eq!(
        count_rows(&graph, "MATCH (n:Doc) WHERE n.id IN [1, 1, 2] RETURN n.id"),
        2
    );
    assert_eq!(
        count_rows(&graph, "MATCH (n:Doc {id: 1}) RETURN n.id"),
        1,
        "the equality anchor is the control — it never duplicated"
    );
}
