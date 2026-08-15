//! **Absolute goldens for the `dot` / `cosine` / `norm` scalar functions.**
//!
//! These read *list-valued data* — a stored list property, a list literal, a
//! `collect()` — as opposed to the embedding store `vector_score` /
//! `embedding_norm` read. Every expected value below is hand-computed and
//! written next to its assertion; the contract they pin (null propagation,
//! length mismatch as an error, non-numeric element as an error, zero-norm
//! cosine as null) is stated in `scalar_functions/vector.rs`'s module header.
//!
//! Red proof: before the functions existed, every query here failed with
//! `Unknown function: dot` / `cosine` / `norm`.

use super::*;
use std::borrow::Cow;

/// Run a read query and return its single row's single cell.
fn cell(graph: &DirGraph, query: &str) -> Value {
    let parsed = parser::parse_cypher(query)
        .unwrap_or_else(|e| panic!("query failed to parse: {query}\n  error: {e}"));
    let no_params = HashMap::new();
    let result = CypherExecutor::with_params(graph, &no_params, None)
        .execute(&parsed)
        .unwrap_or_else(|e| panic!("query failed: {query}\n  error: {e}"));
    assert_eq!(result.rows.len(), 1, "expected one row from: {query}");
    assert_eq!(result.rows[0].len(), 1, "expected one column from: {query}");
    result.rows[0][0].clone()
}

/// Run a read query for its error, or panic if it unexpectedly succeeded.
fn error(graph: &DirGraph, query: &str) -> String {
    let parsed = parser::parse_cypher(query)
        .unwrap_or_else(|e| panic!("query failed to parse: {query}\n  error: {e}"));
    let no_params = HashMap::new();
    match CypherExecutor::with_params(graph, &no_params, None).execute(&parsed) {
        Ok(result) => panic!("query unexpectedly succeeded: {query}\n  rows: {result:?}"),
        Err(e) => e,
    }
}

fn float(graph: &DirGraph, query: &str) -> f64 {
    match cell(graph, query) {
        Value::Float64(f) => f,
        other => panic!("expected Float64 from {query}, got {other:?}"),
    }
}

// ========================================================================
// The arithmetic
// ========================================================================

#[test]
fn dot_is_the_sum_of_elementwise_products() {
    let graph = DirGraph::new();
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert_eq!(float(&graph, "RETURN dot([1, 2, 3], [4, 5, 6])"), 32.0);
    // Mixed Int64/Float64 elements: 0.5*4 + 2*0.25 = 2 + 0.5 = 2.5
    assert_eq!(float(&graph, "RETURN dot([0.5, 2], [4, 0.25])"), 2.5);
    // Orthogonal.
    assert_eq!(float(&graph, "RETURN dot([1, 0], [0, 1])"), 0.0);
    // Negative components: 1*-1 + -2*3 = -1 - 6 = -7
    assert_eq!(float(&graph, "RETURN dot([1, -2], [-1, 3])"), -7.0);
    // The empty sum.
    assert_eq!(float(&graph, "RETURN dot([], [])"), 0.0);
    // Longer than the 8-wide chunk the f32 kernels use — this path has no
    // chunking, so the value must simply stay exact: Σ i*i for i in 1..=10
    // = 385.
    assert_eq!(
        float(
            &graph,
            "RETURN dot([1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10])"
        ),
        385.0
    );
}

#[test]
fn norm_is_the_euclidean_length() {
    let graph = DirGraph::new();
    // sqrt(9 + 16) = 5
    assert_eq!(float(&graph, "RETURN norm([3, 4])"), 5.0);
    // sqrt(1) = 1
    assert_eq!(float(&graph, "RETURN norm([1])"), 1.0);
    // sqrt(25 + 144) = 13
    assert_eq!(float(&graph, "RETURN norm([5, 12])"), 13.0);
    // The zero vector has length 0 — not null; only *cosine* is undefined there.
    assert_eq!(float(&graph, "RETURN norm([0, 0, 0])"), 0.0);
    // The empty sum.
    assert_eq!(float(&graph, "RETURN norm([])"), 0.0);
    // Sign-independent: sqrt(9 + 16) = 5 again.
    assert_eq!(float(&graph, "RETURN norm([-3, -4])"), 5.0);
}

#[test]
fn cosine_is_the_normalised_dot_product() {
    let graph = DirGraph::new();
    // Identical direction → 1.
    assert_eq!(float(&graph, "RETURN cosine([1, 2, 3], [1, 2, 3])"), 1.0);
    // Same direction, different magnitude → still 1.
    assert_eq!(float(&graph, "RETURN cosine([1, 2, 3], [2, 4, 6])"), 1.0);
    // Orthogonal → 0.
    assert_eq!(float(&graph, "RETURN cosine([1, 0], [0, 1])"), 0.0);
    // Opposite → -1.
    assert_eq!(float(&graph, "RETURN cosine([1, 2], [-1, -2])"), -1.0);
    // 45°: dot = 1, |a| = 1, |b| = sqrt(2) → 1/sqrt(2) = 0.70710678…
    let got = float(&graph, "RETURN cosine([1, 0], [1, 1])");
    assert!(
        (got - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12,
        "cosine([1,0],[1,1]) = {got}"
    );
    // 3-4-5 against the x axis: dot = 3, |a| = 1, |b| = 5 → 0.6.
    assert!((float(&graph, "RETURN cosine([1, 0], [3, 4])") - 0.6).abs() < 1e-12);
}

// ========================================================================
// Null propagation
// ========================================================================

#[test]
fn a_null_argument_makes_the_whole_call_null() {
    let graph = DirGraph::new();
    for query in [
        "RETURN dot(null, [1, 2])",
        "RETURN dot([1, 2], null)",
        "RETURN dot(null, null)",
        "RETURN cosine(null, [1, 2])",
        "RETURN cosine([1, 2], null)",
        "RETURN norm(null)",
    ] {
        assert_eq!(cell(&graph, query), Value::Null, "{query}");
    }
}

#[test]
fn a_missing_property_reads_as_null_not_as_an_error() {
    // The row's node has no `vec` property at all — the shape a partially
    // embedded corpus produces. That must be null, not a failed query, so a
    // scored projection over a mixed corpus still returns its rows.
    let graph = build_test_graph();
    assert_eq!(
        cell(
            &graph,
            "MATCH (n:Person) WHERE n.name = 'Alice' RETURN norm(n.vec)"
        ),
        Value::Null
    );
    assert_eq!(
        cell(
            &graph,
            "MATCH (n:Person) WHERE n.name = 'Alice' RETURN cosine(n.vec, [1, 2])"
        ),
        Value::Null
    );
}

// ========================================================================
// The error arms
// ========================================================================

#[test]
fn a_length_mismatch_is_an_error_naming_both_lengths() {
    // Not null: a 2-vector meeting a 3-vector is a data bug, and a null would
    // sit in a column of plausible scores without saying so. Neo4j's
    // vector.similarity.* family likewise only compares equal dimensions.
    let graph = DirGraph::new();
    let msg = error(&graph, "RETURN dot([1, 2], [1, 2, 3])");
    assert!(
        msg.contains("same length") && msg.contains('2') && msg.contains('3'),
        "message must name both lengths: {msg}"
    );
    let msg = error(&graph, "RETURN cosine([1, 2, 3], [])");
    assert!(msg.contains("same length"), "{msg}");
}

#[test]
fn a_non_numeric_element_is_an_error_naming_its_position() {
    let graph = DirGraph::new();
    let msg = error(&graph, "RETURN dot([1, 'x'], [1, 2])");
    assert!(
        msg.contains("element 1") && msg.contains("first"),
        "message must name the vector and the position: {msg}"
    );
    let msg = error(&graph, "RETURN dot([1, 2], [1, 'x'])");
    assert!(msg.contains("element 1") && msg.contains("second"), "{msg}");
    let msg = error(&graph, "RETURN norm([1, true])");
    assert!(msg.contains("element 1"), "{msg}");
    // A null *element* is not silently zeroed (Neo4j's GDS does substitute
    // 0.0); a zeroed component changes the answer without changing its shape.
    let msg = error(&graph, "RETURN norm([1, null])");
    assert!(msg.contains("element 1") && msg.contains("Null"), "{msg}");
}

#[test]
fn a_non_list_argument_is_an_error_even_opposite_a_null() {
    let graph = DirGraph::new();
    let msg = error(&graph, "RETURN dot(7, [1, 2])");
    assert!(msg.contains("list of numbers"), "{msg}");
    // Both arguments are type-checked before the null short-circuit, so this
    // reports the 7 rather than answering null.
    let msg = error(&graph, "RETURN dot(null, 7)");
    assert!(msg.contains("list of numbers"), "{msg}");
    let msg = error(&graph, "RETURN norm('not a list')");
    assert!(msg.contains("list of numbers"), "{msg}");
}

#[test]
fn the_wrong_arity_is_an_error() {
    let graph = DirGraph::new();
    assert!(error(&graph, "RETURN dot([1, 2])").contains("2 arguments"));
    assert!(error(&graph, "RETURN cosine([1], [1], [1])").contains("2 arguments"));
    assert!(error(&graph, "RETURN norm([1], [2])").contains("1 argument"));
}

// ========================================================================
// Zero norm
// ========================================================================

#[test]
fn cosine_of_a_zero_length_vector_is_null_not_nan() {
    // 0/0 is undefined, and null is Cypher's word for that. (The vector-search
    // Scorer answers 0.0 for the same input because a top-k ranking needs a
    // total order; a scalar function carries no such constraint.)
    let graph = DirGraph::new();
    for query in [
        "RETURN cosine([0, 0], [1, 2])",
        "RETURN cosine([1, 2], [0, 0])",
        "RETURN cosine([0, 0], [0, 0])",
        "RETURN cosine([], [])",
    ] {
        assert_eq!(cell(&graph, query), Value::Null, "{query}");
    }
    // dot and norm are defined there and stay numbers.
    assert_eq!(float(&graph, "RETURN dot([0, 0], [1, 2])"), 0.0);
    assert_eq!(float(&graph, "RETURN norm([0, 0])"), 0.0);
}

// ========================================================================
// Stored list properties (the actual use case)
// ========================================================================

/// A graph whose `Doc` nodes carry a real list-valued `vec` property, plus one
/// that stores the same vector as bracketed text (the legacy JSON-string shape
/// `size()` / `head()` also accept).
fn docs_with_vectors() -> DirGraph {
    let mut graph = DirGraph::new();
    for (id, title, vec) in [
        (
            1u32,
            "native",
            Value::List(vec![Value::Float64(3.0), Value::Float64(4.0)]),
        ),
        (
            2,
            "ints",
            Value::List(vec![Value::Int64(1), Value::Int64(0)]),
        ),
        (3, "text", Value::String("[3.0, 4.0]".to_string())),
    ] {
        let node = NodeData::new(
            Value::UniqueId(id),
            Value::String(title.to_string()),
            "Doc".to_string(),
            HashMap::from([("vec".to_string(), vec)]),
            &mut graph.interner,
        );
        let idx = graph.graph.add_node(node);
        graph
            .type_indices
            .entry_or_default("Doc".to_string())
            .push(idx);
    }
    graph
}

#[test]
fn the_functions_read_stored_list_properties() {
    let graph = docs_with_vectors();
    // norm([3,4]) = 5
    assert_eq!(
        float(
            &graph,
            "MATCH (d:Doc) WHERE d.title = 'native' RETURN norm(d.vec)"
        ),
        5.0
    );
    // dot([3,4],[1,0]) = 3
    assert_eq!(
        float(
            &graph,
            "MATCH (d:Doc) WHERE d.title = 'native' RETURN dot(d.vec, [1, 0])"
        ),
        3.0
    );
    // cosine([3,4],[1,0]) = 3/5 = 0.6
    assert!(
        (float(
            &graph,
            "MATCH (d:Doc) WHERE d.title = 'native' RETURN cosine(d.vec, [1, 0])"
        ) - 0.6)
            .abs()
            < 1e-12
    );
    // An integer-typed stored list works the same.
    assert_eq!(
        float(
            &graph,
            "MATCH (d:Doc) WHERE d.title = 'ints' RETURN norm(d.vec)"
        ),
        1.0
    );
    // The bracketed-text shape answers identically to the native one.
    assert_eq!(
        float(
            &graph,
            "MATCH (d:Doc) WHERE d.title = 'text' RETURN norm(d.vec)"
        ),
        5.0
    );
}

#[test]
fn two_stored_vectors_can_be_compared_against_each_other() {
    // The cross-join shape a similarity query actually takes.
    let graph = docs_with_vectors();
    let query = "MATCH (a:Doc) MATCH (b:Doc) \
                 WHERE a.title = 'native' AND b.title = 'ints' \
                 RETURN cosine(a.vec, b.vec)";
    // cosine([3,4],[1,0]) = 3/5 = 0.6
    assert!((float(&graph, query) - 0.6).abs() < 1e-12);
}

// ========================================================================
// The borrow pin (T13)
// ========================================================================

#[test]
fn a_stored_list_reaches_the_kernel_without_being_cloned() {
    // The kernels read the row's own elements: the in-memory column read
    // borrows (`ColumnStore::get_cow` returns `Cow::Borrowed` for the Mixed
    // arm — T13), and `vector_arg` *moves* the evaluated `Value::List` rather
    // than cloning it. This pins the storage half directly, which is the half
    // that regressed once already (0.16.0 undid the 0.15.11 borrow fix).
    use crate::graph::storage::GraphRead;
    let graph = docs_with_vectors();
    let idx = graph
        .type_indices
        .get("Doc")
        .expect("Doc type index")
        .get(0)
        .expect("at least one Doc");
    let view = graph.graph.node_view(idx).expect("node view");
    let key = crate::graph::schema::InternedKey::from_str("vec");
    let got = view.get(key).expect("stored vec property");
    assert!(
        matches!(got, Cow::Borrowed(_)),
        "an in-memory list property must be borrowed, not cloned"
    );
}
