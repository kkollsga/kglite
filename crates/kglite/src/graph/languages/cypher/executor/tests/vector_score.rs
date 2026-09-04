//! The `vector_score()` scalar's per-query argument cache.
//!
//! What is pinned here is *call identity*: a query that scores two different
//! things — two query vectors, two embedding columns, two metrics — must get
//! two answers. Until 0.16.9 the cache was a single unkeyed slot, so the first
//! call site's property and query vector answered every later call in the same
//! query, silently.
//!
//! Red proof: on 0.16.9 every assertion below that compares two columns of one
//! row fails with the two columns equal.

use super::*;
use crate::graph::embeddings::set_embeddings;

/// A `Doc` graph with a `summary_emb` store, one 2-d vector per node.
fn docs(vectors: &[(&str, [f32; 2])]) -> DirGraph {
    let mut graph = DirGraph::new();
    for (index, (title, _)) in vectors.iter().enumerate() {
        let node = NodeData::new(
            Value::Int64(index as i64 + 1),
            Value::String((*title).to_string()),
            "Doc".to_string(),
            // The embedding columns' source properties: `set_embeddings`
            // resolves a column against real node properties, and the stores
            // below are named after them (`summary` → `summary_emb`).
            HashMap::from([
                ("summary".to_string(), Value::String((*title).to_string())),
                ("abstract".to_string(), Value::String((*title).to_string())),
                // The row-dependent query-vector case reads this one.
                (
                    "vec".to_string(),
                    Value::List(
                        vectors[index]
                            .1
                            .iter()
                            .map(|component| Value::Float64(*component as f64))
                            .collect(),
                    ),
                ),
            ]),
            &mut graph.interner,
        );
        let idx = graph.graph.add_node(node);
        graph
            .type_indices
            .entry_or_default("Doc".to_string())
            .push(idx);
    }
    graph.build_id_index("Doc");
    embed(&mut graph, "summary", vectors);
    graph
}

/// Add a second store over the same nodes, under `{column}_emb`.
fn embed(graph: &mut DirGraph, column: &str, vectors: &[(&str, [f32; 2])]) {
    let entries: Vec<(Value, Vec<f32>)> = vectors
        .iter()
        .enumerate()
        .map(|(index, (_, vector))| (Value::Int64(index as i64 + 1), vector.to_vec()))
        .collect();
    set_embeddings(graph, "Doc", column, None, entries).unwrap();
}

fn rows_with(graph: &DirGraph, query: &str, params: HashMap<String, Value>) -> Vec<Vec<Value>> {
    let parsed = parser::parse_cypher(query)
        .unwrap_or_else(|e| panic!("query failed to parse: {query}\n  error: {e}"));
    CypherExecutor::with_params(graph, &params, None)
        .execute(&parsed)
        .unwrap_or_else(|e| panic!("query failed: {query}\n  error: {e}"))
        .rows
}

fn rows(graph: &DirGraph, query: &str) -> Vec<Vec<Value>> {
    rows_with(graph, query, HashMap::new())
}

fn float(cell: &Value) -> f64 {
    match cell {
        Value::Float64(f) => *f,
        other => panic!("expected a score, got {other:?}"),
    }
}

#[test]
fn two_query_vectors_in_one_query_do_not_share_a_cache() {
    // The shipped bug, in its smallest form: one document on the x axis scored
    // against x and against y. Cosine says 1.0 and 0.0; the unkeyed cache said
    // 1.0 twice.
    let graph = docs(&[("a", [1.0, 0.0])]);

    let rows = rows(
        &graph,
        "MATCH (d:Doc) RETURN vector_score(d, 'summary_emb', [1.0, 0.0]) AS a, \
         vector_score(d, 'summary_emb', [0.0, 1.0]) AS c",
    );
    assert_eq!(float(&rows[0][0]), 1.0, "{rows:?}");
    assert_eq!(float(&rows[0][1]), 0.0, "{rows:?}");
}

#[test]
fn two_embedding_columns_in_one_query_do_not_share_a_cache() {
    // The hybrid shape: the same query vector against two stores. The
    // `abstract` store is the axis flip of `summary`, so one column must score
    // 1.0 and the other 0.0 — the unkeyed cache read both out of
    // `summary_emb`.
    let mut graph = docs(&[("a", [1.0, 0.0])]);
    embed(&mut graph, "abstract", &[("a", [0.0, 1.0])]);

    let rows = rows(
        &graph,
        "MATCH (d:Doc) RETURN vector_score(d, 'summary_emb', [1.0, 0.0]) AS s, \
         vector_score(d, 'abstract_emb', [1.0, 0.0]) AS t",
    );
    assert_eq!(float(&rows[0][0]), 1.0, "{rows:?}");
    assert_eq!(float(&rows[0][1]), 0.0, "{rows:?}");
}

#[test]
fn two_parameters_in_one_query_do_not_share_a_cache() {
    // This is the shape `text_score(n, 'summary', '<text>')` becomes: the
    // planner rewrites each distinct query text to its own `$__ts_N` parameter
    // (`simplification::param_for_text`), so two text_score calls arrive here
    // as two parameter arguments. The parameter *name* is the key — a
    // parameter is bound once for the execution.
    let graph = docs(&[("a", [1.0, 0.0])]);
    let params = HashMap::from([
        (
            "__ts_0".to_string(),
            Value::List(vec![Value::Float64(1.0), Value::Float64(0.0)]),
        ),
        (
            "__ts_1".to_string(),
            Value::List(vec![Value::Float64(0.0), Value::Float64(1.0)]),
        ),
    ]);

    let rows = rows_with(
        &graph,
        "MATCH (d:Doc) RETURN vector_score(d, 'summary_emb', $__ts_0) AS a, \
         vector_score(d, 'summary_emb', $__ts_1) AS c",
        params,
    );
    assert_eq!(float(&rows[0][0]), 1.0, "{rows:?}");
    assert_eq!(float(&rows[0][1]), 0.0, "{rows:?}");
}

#[test]
fn two_metrics_in_one_query_do_not_share_a_cache() {
    // The metric is the fourth argument and part of the same identity: a
    // dot_product call reading a cosine call's scorer is the same silent wrong
    // answer. |[2,0]| = 2, so cosine says 1.0 where dot_product says 2.0.
    let graph = docs(&[("a", [2.0, 0.0])]);

    let rows = rows(
        &graph,
        "MATCH (d:Doc) RETURN vector_score(d, 'summary_emb', [1.0, 0.0], 'cosine') AS c, \
         vector_score(d, 'summary_emb', [1.0, 0.0], 'dot_product') AS d",
    );
    assert_eq!(float(&rows[0][0]), 1.0, "{rows:?}");
    assert_eq!(float(&rows[0][1]), 2.0, "{rows:?}");
}

// ========================================================================
// Prepared once per call site, not once per row
// ========================================================================
//
// The cached and uncached paths compute the same number, so a wrong answer
// cannot report a cache miss. These read the preparation counter instead.

/// Run `query` and report how many times `vector_score` parsed its arguments.
fn prepares(graph: &DirGraph, query: &str) -> usize {
    VECTOR_SCORE_PREPARES.with(|count| count.set(0));
    rows(graph, query);
    VECTOR_SCORE_PREPARES.with(|count| count.get())
}

#[test]
fn a_list_literal_query_vector_is_prepared_once_for_the_whole_scan() {
    // A bare `[1.0, 0.0]` is a ListLiteral, not a Literal, and needs its own
    // key: without one it has no identity, falls out of the cache, and every
    // row re-parses the vector and rebuilds the scorer.
    //
    // The call is wrapped in a CASE deliberately. A RETURN item is
    // constant-folded before execution (`fold_constants_expr`), which turns a
    // literal list argument into `Literal(Value::List)` and hides the question;
    // CASE is one of the shapes folding steps over, so this is the argument the
    // scalar actually receives. Drop the `ArgKey::LiteralList` arm and this
    // reads 5.
    let graph = docs(&[
        ("a", [1.0, 0.0]),
        ("b", [0.0, 1.0]),
        ("c", [1.0, 1.0]),
        ("d", [2.0, 0.0]),
        ("e", [0.0, 2.0]),
    ]);

    assert_eq!(
        prepares(
            &graph,
            "MATCH (d:Doc) RETURN CASE WHEN d.summary IS NOT NULL \
             THEN vector_score(d, 'summary_emb', [1.0, 0.0]) ELSE 0.0 END AS s"
        ),
        1,
        "five rows, one call site"
    );
    // The folded form is cached too — under `ArgKey::Literal`, since folding
    // has already turned the list into one value.
    assert_eq!(
        prepares(
            &graph,
            "MATCH (d:Doc) RETURN vector_score(d, 'summary_emb', [1.0, 0.0]) AS s"
        ),
        1,
        "five rows, one call site"
    );
}

#[test]
fn two_call_sites_are_prepared_once_each() {
    // Both answers are cached, so neither call site pays the other's miss —
    // the hybrid query's whole point.
    let graph = docs(&[("a", [1.0, 0.0]), ("b", [0.0, 1.0]), ("c", [1.0, 1.0])]);

    assert_eq!(
        prepares(
            &graph,
            "MATCH (d:Doc) RETURN vector_score(d, 'summary_emb', [1.0, 0.0]) AS a, \
             vector_score(d, 'summary_emb', [0.0, 1.0]) AS c"
        ),
        2,
        "three rows, two call sites"
    );
}

#[test]
fn a_row_dependent_query_vector_is_prepared_per_row() {
    // The complement of the cache: a vector read out of the row has no
    // key, so it is never parked and each row is scored against its own.
    let graph = docs(&[("a", [1.0, 0.0]), ("b", [0.0, 1.0]), ("c", [1.0, 1.0])]);

    let query = "MATCH (d:Doc) RETURN vector_score(d, 'summary_emb', d.vec) AS s";
    assert_eq!(prepares(&graph, query), 3, "three rows, prepared per row");
    // Each row scores against itself: cosine(v, v) = 1.0 (f32 normalisation,
    // hence the tolerance).
    for row in rows(&graph, query) {
        assert!((float(&row[0]) - 1.0).abs() < 1e-6, "{row:?}");
    }
}

// ── The fused scans must refuse an unknown embedding store ───────────────────
//
// The sibling of the `text_bm25` case in `text_bm25.rs`: `WHERE
// vector_score(…) > 0 … ORDER BY … LIMIT k` reaches `FusedNodeScanTopK`, whose
// WHERE filter drops a row it cannot evaluate. "No embedding of that name on
// this type" is wrong for every row, so the fused plan answered zero rows where
// the scalar raised.

/// Optimise `query`, assert the planner really produced the fused clause the
/// test is about, and return the error.
fn fused_error(graph: &DirGraph, query: &str, claimed: fn(&Clause) -> bool) -> String {
    let params = HashMap::new();
    let mut parsed = parser::parse_cypher(query).expect("parses");
    crate::graph::languages::cypher::planner::optimize(&mut parsed, graph, &params);
    assert!(
        parsed.clauses.iter().any(claimed),
        "the pass did not claim this shape, so the assertion would be vacuous: {query}"
    );
    match CypherExecutor::with_params(graph, &params, None).execute(&parsed) {
        Ok(result) => panic!("query unexpectedly succeeded: {query}\n  rows: {result:?}"),
        Err(e) => e,
    }
}

#[test]
fn the_fused_top_k_scan_refuses_an_unknown_embedding_store() {
    let graph = docs(&[("a", [1.0, 0.0]), ("b", [0.0, 1.0])]);

    let message = fused_error(
        &graph,
        "MATCH (d:Doc) WHERE vector_score(d, 'nope_emb', [1.0, 0.0]) > 0 RETURN d.title AS t \
         ORDER BY vector_score(d, 'nope_emb', [1.0, 0.0]) DESC LIMIT 5",
        |clause| matches!(clause, Clause::FusedNodeScanTopK { .. }),
    );

    assert!(
        message.contains("no embedding 'nope_emb' found for node type 'Doc'"),
        "the fast path must raise what the scalar raises: {message}"
    );
}

#[test]
fn the_fused_scan_aggregate_refuses_an_unknown_embedding_store() {
    let graph = docs(&[("a", [1.0, 0.0]), ("b", [0.0, 1.0])]);

    let message = fused_error(
        &graph,
        "MATCH (d:Doc) WHERE vector_score(d, 'nope_emb', [1.0, 0.0]) > 0 RETURN count(d) AS c",
        |clause| matches!(clause, Clause::FusedNodeScanAggregate { .. }),
    );

    assert!(
        message.contains("no embedding 'nope_emb' found for node type 'Doc'"),
        "a count must not answer zero where the scalar raises: {message}"
    );
}

#[test]
fn literal_options_maps_keep_one_preparation_inside_case() {
    let graph = docs(&[("a", [1.0, 0.0]), ("b", [0.0, 1.0]), ("c", [1.0, 1.0])]);
    assert_eq!(prepares(&graph,
        "MATCH (d:Doc) RETURN CASE WHEN d.title = 'a' THEN 0 ELSE vector_score(d, 'summary_emb', [1.0,0.0], {exact:true}) END AS s"
    ), 1);
    VECTOR_SCORE_PREPARES.with(|count| count.set(0));
    rows_with(
        &graph,
        "MATCH (d:Doc) RETURN vector_score(d, 'summary_emb', [1.0,0.0], $options) AS s",
        HashMap::from([(
            "options".to_string(),
            Value::Map(
                [("exact".to_string(), Value::Boolean(true))]
                    .into_iter()
                    .collect(),
            ),
        )]),
    );
    assert_eq!(VECTOR_SCORE_PREPARES.with(|count| count.get()), 1);
}

#[test]
fn fused_scans_propagate_vector_argument_errors() {
    let graph = docs(&[("a", [1.0, 0.0]), ("b", [0.0, 1.0])]);
    for (arguments, expected) in [
        ("[1.0]", "dimension"),
        ("[1.0, 'bad']", "numeric"),
        ("'[oops]'", "vector_score"),
        ("[1.0,0.0], 'bad_metric'", "unknown metric"),
        ("[1.0,0.0], {exact:1}", "exact"),
    ] {
        let predicate =
            format!("MATCH (d:Doc) WHERE vector_score(d, 'summary_emb', {arguments}) > 0 ");
        let error = fused_error(&graph, &(predicate.clone() + "RETURN count(d) AS c"), |c| {
            matches!(c, Clause::FusedNodeScanAggregate { .. })
        });
        assert!(error.contains(expected), "{error}");
        let error = fused_error(
            &graph,
            &(predicate + "RETURN d.title AS title ORDER BY title LIMIT 2"),
            |c| matches!(c, Clause::FusedNodeScanTopK { .. }),
        );
        assert!(error.contains(expected), "{error}");
    }
}
