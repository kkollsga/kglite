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
    rows(graph, pattern, max_matches, &["a", "b"])
        .into_iter()
        .map(|row| {
            let mut it = row.into_iter();
            (it.next().unwrap(), it.next().unwrap())
        })
        .collect()
}

/// The `name` of each requested variable, per match, in match order.
fn rows(
    graph: &DirGraph,
    pattern: &str,
    max_matches: Option<usize>,
    vars: &[&str],
) -> Vec<Vec<String>> {
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
            vars.iter()
                .map(|var| {
                    m.bindings
                        .iter()
                        .find(|(name, _)| name == var)
                        .map(|(_, b)| name_of(b))
                        .unwrap_or_else(|| panic!("no binding for {var}"))
                })
                .collect()
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

// ───────────────────────── advisory caps ─────────────────────────────
//
// The seeding above is only half the contract. The candidate caps that make
// the lazy seeding worth doing — `max(max_matches * 100, 1000)` on the start
// nodes, `max(max_matches * 50, 1000)` on an intermediate hop — are a
// *selectivity heuristic*: neither knows anything about the relationship type
// being matched, so a start node whose only matching edge sits past the cap
// used to be dropped and the query answered with silence. Every fixture below
// deliberately crosses those thresholds; a fixture under 1 000 nodes cannot
// see either cap at all, which is why the six-node one above never caught this.

/// Number of leading nodes that match nothing — comfortably past the 1 000
/// start-node floor, so the only rows in the graph sit behind the cap.
const LEADING_FILLER: usize = 1_500;

/// `LEADING_FILLER` edgeless nodes, then the only two edges in the graph:
/// `x-[:R]->y` and `p-[:S]->q`. Every start node that can produce a row is
/// past the start-node cap.
fn late_sources() -> DirGraph {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        &format!(
            "UNWIND range(1, {LEADING_FILLER}) AS i \
             CREATE (:Filler {{id: i, name: 'filler-' + toString(i)}})"
        ),
    );
    run(
        &mut graph,
        "CREATE (x:P {id: 9001, name: 'x'})-[:R]->(y:P {id: 9002, name: 'y'})
         CREATE (p:P {id: 9003, name: 'p'})-[:S]->(q:P {id: 9004, name: 'q'})",
    );
    graph
}

/// `LEADING_FILLER` `:Symbol` nodes with no edges, then two more `:Symbol`
/// nodes joined by the graph's only `:RARE` edge. A *labeled* start is capped
/// exactly like an unlabeled one, so the type index does not save it.
fn sparse_label() -> DirGraph {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        &format!(
            "UNWIND range(1, {LEADING_FILLER}) AS i \
             CREATE (:Symbol {{id: i, name: 'sym-' + toString(i)}})"
        ),
    );
    run(
        &mut graph,
        "CREATE (a:Symbol {id: 9001, name: 'rare-src'})-[:RARE]->
                (b:Symbol {id: 9002, name: 'rare-dst'})",
    );
    graph
}

/// One `:Root` with an `:E1` edge to each of 2 000 `:Mid` nodes, of which only
/// `mid-1` carries the `:E2` edge on to the single `:Leaf`. The start-node cap
/// is irrelevant here (the root is the first node); the *intermediate-hop* cap
/// is what drops the one Mid that matters. `mid-1` rather than `mid-2000`
/// because the expansion walks a node's edges newest-first, so the
/// first-created edge is the *last* intermediate the hop produces — the test
/// below asserts that position rather than trusting it.
fn sparse_middle_hop() -> DirGraph {
    let mut graph = DirGraph::new();
    run(&mut graph, "CREATE (:Root {id: 1, name: 'root'})");
    run(
        &mut graph,
        "UNWIND range(1, 2000) AS i \
         CREATE (:Mid {id: 1000 + i, name: 'mid-' + toString(i)})",
    );
    run(&mut graph, "MATCH (r:Root), (m:Mid) CREATE (r)-[:E1]->(m)");
    run(&mut graph, "CREATE (:Leaf {id: 90001, name: 'leaf'})");
    run(
        &mut graph,
        "MATCH (m:Mid {id: 1001}), (l:Leaf) CREATE (m)-[:E2]->(l)",
    );
    graph
}

/// **The contract.** A cap is advisory: a capped run returns a prefix of the
/// uncapped answer, never a *shorter* answer than the pattern has rows for.
#[test]
fn start_node_cap_does_not_hide_late_sources() {
    let graph = late_sources();
    let uncapped = names(&graph, "(a)-[r:R]->(b)", None);
    assert_eq!(
        uncapped,
        vec![("x".to_string(), "y".to_string())],
        "the graph's only R edge"
    );
    // 1 * 100 -> floor 1000 < 1502 nodes: the cap bites, and every start node
    // it keeps is a filler.
    assert_eq!(names(&graph, "(a)-[r:R]->(b)", Some(1)), uncapped);
    // The LIMIT cliff: cap arithmetic used to decide the answer.
    assert_eq!(names(&graph, "(a)-[r:R]->(b)", Some(10)), uncapped);
    assert_eq!(names(&graph, "(a)-[r:R]->(b)", Some(11)), uncapped);
}

#[test]
fn start_node_cap_does_not_hide_late_labeled_sources() {
    let graph = sparse_label();
    let uncapped = names(&graph, "(a:Symbol)-[r:RARE]->(b:Symbol)", None);
    assert_eq!(
        uncapped,
        vec![("rare-src".to_string(), "rare-dst".to_string())]
    );
    assert_eq!(
        names(&graph, "(a:Symbol)-[r:RARE]->(b:Symbol)", Some(1)),
        uncapped
    );
    assert_eq!(
        names(&graph, "(a:Symbol)-[r:RARE]->(b:Symbol)", Some(3)),
        uncapped
    );
}

#[test]
fn start_node_cap_does_not_hide_late_sources_for_undirected_reversed_or_var_length() {
    let graph = late_sources();
    for pattern in [
        "(a)-[r:R]-(b)",       // undirected
        "(a)<-[r:R]-(b)",      // reversed
        "(a)-[r:R*1..2]->(b)", // variable length
        "(a)-[r:R|S]->(b)",    // type alternation
    ] {
        let uncapped = names(&graph, pattern, None);
        assert!(
            !uncapped.is_empty(),
            "fixture must produce rows for {pattern}"
        );
        assert_eq!(
            names(&graph, pattern, Some(1)),
            uncapped[..1].to_vec(),
            "capped run lost the late source for {pattern}"
        );
    }
}

#[test]
fn intermediate_hop_cap_does_not_hide_a_sparse_middle() {
    let graph = sparse_middle_hop();
    let pattern = "(a:Root)-[:E1]->(m:Mid)-[:E2]->(b:Leaf)";
    let uncapped = rows(&graph, pattern, None, &["a", "m", "b"]);
    assert_eq!(
        uncapped,
        vec![vec![
            "root".to_string(),
            "mid-1".to_string(),
            "leaf".to_string()
        ]],
        "only mid-1 carries an E2 edge"
    );
    // Non-vacuity: the surviving intermediate must really sit past the 1 000
    // intermediate-hop floor, or this fixture proves nothing.
    let mids = names(&graph, "(a:Root)-[r:E1]->(b:Mid)", None);
    let position = mids
        .iter()
        .position(|(_, mid)| mid == "mid-1")
        .expect("mid-1 is expanded");
    assert!(
        position >= 1000,
        "fixture does not cross the intermediate-hop cap: mid-1 at {position}"
    );
    // 1 * 50 -> floor 1000 intermediates kept, out of 2 000.
    assert_eq!(rows(&graph, pattern, Some(1), &["a", "m", "b"]), uncapped);
    assert_eq!(rows(&graph, pattern, Some(10), &["a", "m", "b"]), uncapped);
}

/// A cap the candidate pool never reaches must not truncate or duplicate, and
/// a cap a *dense* pattern does reach still bounds the result exactly.
#[test]
fn caps_still_bound_a_pattern_that_has_the_rows() {
    let graph = late_sources();
    let uncapped = names(&graph, "(a)-[r:R|S]->(b)", None);
    assert_eq!(
        uncapped,
        vec![
            ("x".to_string(), "y".to_string()),
            ("p".to_string(), "q".to_string()),
        ]
    );
    // 20 * 100 = 2000 > 1502 candidates: no truncation, no duplication.
    assert_eq!(names(&graph, "(a)-[r:R|S]->(b)", Some(20)), uncapped);

    let dense = sparse_middle_hop();
    let all = names(&dense, "(a:Root)-[r:E1]->(b:Mid)", None);
    assert_eq!(all.len(), 2000);
    assert_eq!(
        names(&dense, "(a:Root)-[r:E1]->(b:Mid)", Some(3)),
        all[..3].to_vec()
    );
}

/// A pattern with no rows at all still returns nothing — the retry must not
/// invent rows, and must stay bounded when it finds none.
#[test]
fn a_missing_relationship_type_stays_empty_under_a_cap() {
    let graph = late_sources();
    assert!(names(&graph, "(a)-[r:MISSING]->(b)", Some(1)).is_empty());
    assert!(names(&graph, "(a)-[r:MISSING]->(b)", None).is_empty());
    let two_hop = sparse_middle_hop();
    assert!(rows(
        &two_hop,
        "(a:Root)-[:E1]->(m:Mid)-[:MISSING]->(b:Leaf)",
        Some(1),
        &["a", "m", "b"]
    )
    .is_empty());
}
