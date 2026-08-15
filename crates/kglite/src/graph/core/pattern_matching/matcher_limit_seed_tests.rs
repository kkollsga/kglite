//! Regression tests for the capped-expansion start-node seeding in
//! [`PatternExecutor::execute`].
//!
//! Split out of `matcher.rs` to keep that file under the source-quality line
//! ceiling, matching `matcher_id_lookup_tests.rs`.
//!
//! Under `max_matches` the expansion stops as soon as the cap is met, so the
//! start nodes past that point are never read and their `PatternMatch` seeds
//! are built lazily. What that must not change is *which* start node each
//! surviving row was seeded from — a seed built at the wrong position, or
//! built from the wrong index, produces rows that still look well-formed.
//! These tests place the only matching start nodes *behind* non-matching ones
//! so an off-by-anything in the lazy seed changes the answer instead of just
//! the count.

use super::*;
use crate::graph::core::pattern_matching::parser::parse_pattern;
use crate::graph::session::execute::{execute_mut, ExecuteOptions};

fn run(graph: &mut DirGraph, query: &str) {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("setup query failed: {query}: {e}"));
}

/// Six `P` nodes named a..f. Only `c->d` and `e->f` carry an `R` edge, so the
/// first two start nodes in index order expand to nothing.
fn seeded_chain() -> DirGraph {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "CREATE (a:P {id: 1, name: 'a'}), (b:P {id: 2, name: 'b'}),
                (c:P {id: 3, name: 'c'}), (d:P {id: 4, name: 'd'}),
                (e:P {id: 5, name: 'e'}), (f:P {id: 6, name: 'f'})
         CREATE (c)-[:R]->(d), (e)-[:R]->(f)",
    );
    graph
}

/// The `(name_of_a, name_of_b)` pairs a pattern produced, in match order.
fn names(graph: &DirGraph, pattern: &str, max_matches: Option<usize>) -> Vec<(String, String)> {
    let params = HashMap::new();
    let pattern = parse_pattern(pattern).expect("pattern parses");
    let executor = PatternExecutor::new_lightweight_with_params(graph, max_matches, &params);
    let name_of = |binding: &MatchBinding| -> String {
        let idx = match binding {
            MatchBinding::Node { index, .. } | MatchBinding::NodeRef(index) => *index,
            other => panic!("expected a node binding, got {other:?}"),
        };
        match &*graph
            .graph
            .node_view(idx)
            .expect("bound node exists")
            .get_property("name")
            .expect("node has a name")
        {
            Value::String(s) => s.clone(),
            other => panic!("unexpected name value {other:?}"),
        }
    };
    executor
        .execute(&pattern)
        .expect("pattern executes")
        .iter()
        .map(|m| {
            let lookup = |var: &str| {
                m.bindings
                    .iter()
                    .find(|(name, _)| name == var)
                    .map(|(_, b)| name_of(b))
                    .unwrap_or_else(|| panic!("no binding for {var}"))
            };
            (lookup("a"), lookup("b"))
        })
        .collect()
}

/// **The contract.** A capped expansion binds the same start node the
/// uncapped one does — the cap changes how many rows come back, never which
/// start node a row came from.
#[test]
fn capped_expansion_seeds_from_the_reached_start_node() {
    let graph = seeded_chain();
    let uncapped = names(&graph, "(a:P)-[r:R]->(b:P)", None);
    assert_eq!(
        uncapped,
        vec![
            ("c".to_string(), "d".to_string()),
            ("e".to_string(), "f".to_string()),
        ],
        "the two R edges, seeded from their real sources"
    );

    // The cap is met by the FIRST matching start node, which sits behind two
    // non-matching ones: a seed built at the loop position rather than at the
    // reached node would bind `a` to 'a'.
    assert_eq!(
        names(&graph, "(a:P)-[r:R]->(b:P)", Some(1)),
        vec![("c".to_string(), "d".to_string())]
    );
    assert_eq!(names(&graph, "(a:P)-[r:R]->(b:P)", Some(2)), uncapped);
    // A cap above the available matches must not truncate or duplicate.
    assert_eq!(names(&graph, "(a:P)-[r:R]->(b:P)", Some(50)), uncapped);
}

/// A cap on a start-node set with no matches at all must not leave a
/// half-built seed behind as a row.
#[test]
fn capped_expansion_over_a_non_matching_type_returns_nothing() {
    let graph = seeded_chain();
    assert!(names(&graph, "(a:P)-[r:MISSING]->(b:P)", Some(1)).is_empty());
    assert!(names(&graph, "(a:P)-[r:MISSING]->(b:P)", None).is_empty());
}
