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
