//! Regression tests for the composite-index probe in
//! [`PatternExecutor::try_index_lookup`].
//!
//! Split out of `matcher.rs` to keep that file under the source-quality line
//! ceiling, matching `matcher_id_lookup_tests.rs`.
//!
//! The probe sorts the pattern's equality properties alphabetically and looks
//! the key up exactly, so the *stored* key has to be sorted too. It was not:
//! `create_composite_index` keyed by the caller's order, and a composite index
//! declared `ON (n.city, n.age)` was therefore unreachable from every MATCH —
//! `try_index_lookup` returned `None` and the caller scanned the whole type.
//! `None` vs `Some` is the assertion here for the same reason as in
//! `matcher_id_lookup_tests`: `find_matching_nodes` scans exactly when the
//! probe answers `None`.

use super::*;
use crate::graph::session::execute::{execute_mut, ExecuteOptions};

fn run(graph: &mut DirGraph, query: &str) {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("setup query failed: {query}: {e}"));
}

/// Two `Doc` nodes sharing an `age` and differing in `city`.
fn seeded_docs() -> DirGraph {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "CREATE (:Doc {id: 1, city: 'Oslo', age: 30}), (:Doc {id: 2, city: 'Bergen', age: 30})",
    );
    graph
}

fn city_and_age(city: &str, age: i64) -> HashMap<String, PropertyMatcher> {
    HashMap::from([
        (
            "city".to_string(),
            PropertyMatcher::Equals(Value::String(city.to_string())),
        ),
        (
            "age".to_string(),
            PropertyMatcher::Equals(Value::Int64(age)),
        ),
    ])
}

fn lookup(
    graph: &DirGraph,
    node_type: &str,
    props: &HashMap<String, PropertyMatcher>,
) -> Option<Vec<NodeIndex>> {
    PatternExecutor::new(graph, None).try_index_lookup(node_type, props)
}

/// **The fix.** A composite index declared in non-alphabetical property order
/// is still the one a two-equality MATCH resolves through.
#[test]
fn composite_index_declared_out_of_order_still_answers_the_probe() {
    let mut graph = seeded_docs();
    graph.create_composite_index("Doc", &["city", "age"]);

    assert_eq!(
        lookup(&graph, "Doc", &city_and_age("Oslo", 30)).map(|hits| hits.len()),
        Some(1),
        "a composite index must answer regardless of the order it was declared in"
    );
    assert_eq!(
        lookup(&graph, "Doc", &city_and_age("Bergen", 30)).map(|hits| hits.len()),
        Some(1),
        "the other tuple resolves through the same key"
    );
}

/// The control: the alphabetically-declared spelling of the same index, which
/// worked before the fix. Without it the test above cannot show that the
/// declaration order — rather than the probe itself — was the broken part.
#[test]
fn composite_index_declared_in_order_answers_the_probe() {
    let mut graph = seeded_docs();
    graph.create_composite_index("Doc", &["age", "city"]);

    assert_eq!(
        lookup(&graph, "Doc", &city_and_age("Oslo", 30)).map(|hits| hits.len()),
        Some(1)
    );
}

/// Both spellings name the same index: declaring one and asking about the
/// other must agree, or `CREATE INDEX … IF NOT EXISTS` builds a duplicate and
/// `DROP INDEX` misses the index it was pointed at.
#[test]
fn composite_index_identity_ignores_property_order() {
    let mut graph = seeded_docs();
    graph.create_composite_index("Doc", &["city", "age"]);

    let declared = vec!["city".to_string(), "age".to_string()];
    let sorted = vec!["age".to_string(), "city".to_string()];
    assert!(graph.has_composite_index("Doc", &declared));
    assert!(graph.has_composite_index("Doc", &sorted));
    assert!(graph.get_composite_index_stats("Doc", &declared).is_some());
    assert!(graph.get_composite_index_stats("Doc", &sorted).is_some());
    assert_eq!(
        graph.list_composite_indexes(),
        vec![("Doc".to_string(), sorted.clone())],
        "the stored key is the canonical, sorted spelling"
    );

    assert!(graph.drop_composite_index("Doc", &declared));
    assert!(!graph.has_composite_index("Doc", &sorted));
}

/// The value tuple has to follow the names through canonicalization: a probe
/// built in the stored key's order must find the tuple the build wrote.
#[test]
fn canonicalized_values_stay_paired_with_their_names() {
    let mut graph = seeded_docs();
    graph.create_composite_index("Doc", &["city", "age"]);

    assert_eq!(
        graph
            .lookup_by_composite_index(
                "Doc",
                &["age".to_string(), "city".to_string()],
                &[Value::Int64(30), Value::String("Oslo".to_string())],
            )
            .map(|hits| hits.len()),
        Some(1),
        "values are stored in the canonical name order, not the declared one"
    );
}
