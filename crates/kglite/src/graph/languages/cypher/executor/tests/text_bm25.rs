//! The `text_bm25()` scalar: what a row scores, what an unindexed row scores,
//! what a stale index serves, and the one hazard the per-query cache carries.
//!
//! Ranking *values* are pinned in `tests/golden/` from Python — human-readable
//! rankings the oracles cannot express. What is pinned here is the contract
//! around the number: null versus zero, the query-entry freshness policy
//! (release-train-0-16-10, decision 11a), and the cache's term-id staleness.
//!
//! Red proof: before the scalar existed every query here failed with
//! `Unknown function: text_bm25`.

use super::*;
use crate::graph::text_indexes::{build_text_index, refresh_text_index};

/// A `Doc` graph, one node per `(title, body)` pair, in the order given.
fn docs(bodies: &[(&str, &str)]) -> DirGraph {
    let mut graph = DirGraph::new();
    for (index, (title, body)) in bodies.iter().enumerate() {
        let node = NodeData::new(
            Value::UniqueId(index as u32 + 1),
            Value::String((*title).to_string()),
            "Doc".to_string(),
            HashMap::from([("body".to_string(), Value::String((*body).to_string()))]),
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

fn run(graph: &DirGraph, query: &str) -> CypherResult {
    let parsed = parser::parse_cypher(query)
        .unwrap_or_else(|e| panic!("query failed to parse: {query}\n  error: {e}"));
    let no_params = HashMap::new();
    CypherExecutor::with_params(graph, &no_params, None)
        .execute(&parsed)
        .unwrap_or_else(|e| panic!("query failed: {query}\n  error: {e}"))
}

/// `(title, score)` per row, in row order.
fn scored(graph: &DirGraph, query: &str) -> Vec<(String, Value)> {
    run(graph, query)
        .rows
        .iter()
        .map(|row| match (&row[0], &row[1]) {
            (Value::String(title), score) => (title.clone(), score.clone()),
            other => panic!("unexpected row shape: {other:?}"),
        })
        .collect()
}

fn error(graph: &DirGraph, query: &str) -> String {
    let parsed = parser::parse_cypher(query).unwrap();
    let no_params = HashMap::new();
    match CypherExecutor::with_params(graph, &no_params, None).execute(&parsed) {
        Ok(result) => panic!("query unexpectedly succeeded: {query}\n  rows: {result:?}"),
        Err(e) => e,
    }
}

fn warnings(result: &CypherResult) -> Vec<String> {
    result
        .diagnostics
        .as_ref()
        .map(|d| d.warnings.clone())
        .unwrap_or_default()
}

const QUERY: &str =
    "MATCH (d:Doc) RETURN d.title AS t, text_bm25(d, 'body', 'quick fox') AS s ORDER BY t";

#[test]
fn an_indexed_document_sharing_no_query_term_scores_zero_not_null() {
    // The whole null-versus-zero split: "indexed, no match" is evidence, and
    // collapsing it into "not searchable" would hide a working index.
    let mut graph = docs(&[("a", "the quick brown fox"), ("b", "slow green turtles")]);
    build_text_index(&mut graph, "Doc", "body", None).unwrap();

    let rows = scored(&graph, QUERY);
    assert_eq!(rows[0].0, "a");
    assert!(
        matches!(rows[0].1, Value::Float64(s) if s > 0.0),
        "{rows:?}"
    );
    assert_eq!(rows[1], ("b".to_string(), Value::Float64(0.0)));
}

#[test]
fn a_node_the_index_never_saw_scores_null() {
    // Non-string property: skipped at build, so it has no document at all.
    let mut graph = docs(&[("a", "the quick brown fox")]);
    let node = NodeData::new(
        Value::UniqueId(99),
        Value::String("b".to_string()),
        "Doc".to_string(),
        HashMap::from([("body".to_string(), Value::Int64(42))]),
        &mut graph.interner,
    );
    let idx = graph.graph.add_node(node);
    graph
        .type_indices
        .entry_or_default("Doc".to_string())
        .push(idx);
    build_text_index(&mut graph, "Doc", "body", None).unwrap();

    let rows = scored(&graph, QUERY);
    assert_eq!(rows[1], ("b".to_string(), Value::Null));
}

#[test]
fn no_index_is_an_error_naming_the_call_that_builds_one() {
    let graph = docs(&[("a", "the quick brown fox")]);

    let message = error(&graph, QUERY);
    assert!(message.contains("no text index on 'Doc.body'"), "{message}");
    assert!(
        message.contains("build_text_index('Doc', 'body')"),
        "{message}"
    );
}

#[test]
fn the_error_names_the_properties_that_are_indexed() {
    let mut graph = docs(&[("a", "the quick brown fox")]);
    build_text_index(&mut graph, "Doc", "title", None).unwrap();

    let message = error(&graph, QUERY);
    assert!(
        message.contains("Indexed on 'Doc' today: title."),
        "{message}"
    );
}

#[test]
fn a_query_folds_in_a_small_delta_before_it_scores() {
    // Decision 11a's end-to-end: a document written after the build scores
    // without anyone calling build_text_index again.
    let mut graph = docs(&[("a", "the quick brown fox")]);
    build_text_index(&mut graph, "Doc", "body", None).unwrap();
    let create =
        parser::parse_cypher("CREATE (:Doc {title: 'b', body: 'a quick fox appears'})").unwrap();
    execute_mutable(
        &mut graph,
        &create,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let result = run(&graph, QUERY);
    let rows: Vec<_> = result
        .rows
        .iter()
        .map(|row| (row[0].clone(), row[1].clone()))
        .collect();
    assert!(
        matches!(rows[1].1, Value::Float64(s) if s > 0.0),
        "the new document should have been folded in: {rows:?}"
    );
    assert!(warnings(&result).is_empty(), "{:?}", warnings(&result));
}

#[test]
fn a_delta_over_the_limit_serves_stale_rows_as_null_and_warns() {
    let mut graph = docs(&[("a", "the quick brown fox")]);
    build_text_index(&mut graph, "Doc", "body", Some(0)).unwrap();
    let create =
        parser::parse_cypher("CREATE (:Doc {title: 'b', body: 'a quick fox appears'})").unwrap();
    execute_mutable(
        &mut graph,
        &create,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let result = run(&graph, QUERY);
    assert_eq!(result.rows[1][1], Value::Null, "an unindexed row is null");
    let warnings = warnings(&result);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(
        warnings[0].contains("text index 'Doc.body' is stale"),
        "{warnings:?}"
    );
    assert!(warnings[0].contains("up to 1 documents"), "{warnings:?}");
    assert!(
        warnings[0].contains("auto_refresh_limit of 0"),
        "{warnings:?}"
    );
    assert!(
        warnings[0].contains("build_text_index('Doc', 'body')"),
        "{warnings:?}"
    );
}

#[test]
fn a_read_only_graph_is_never_caught_up_by_a_query() {
    let mut graph = docs(&[("a", "the quick brown fox")]);
    build_text_index(&mut graph, "Doc", "body", None).unwrap();
    let create =
        parser::parse_cypher("CREATE (:Doc {title: 'b', body: 'a quick fox appears'})").unwrap();
    execute_mutable(
        &mut graph,
        &create,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();
    graph.read_only = true;

    let result = run(&graph, QUERY);
    assert_eq!(result.rows[1][1], Value::Null);
    let warnings = warnings(&result);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("read-only"), "{warnings:?}");
    assert!(
        graph
            .text_indexes
            .values()
            .all(|store| store.is_stale(&graph)),
        "a read-only query must not have refreshed the index"
    );
}

#[test]
fn a_refresh_between_two_queries_on_one_executor_invalidates_the_prepared_query() {
    // The cache hazard, made observable. Term ids are recycled: after the
    // refresh below, `alpha`'s freed id belongs to `beta`, so a query still
    // holding the old ids would score the rewritten document as if it still
    // said `alpha`. The generation stamp is what stops that.
    let mut graph = docs(&[("a", "alpha")]);
    build_text_index(&mut graph, "Doc", "body", Some(0)).unwrap();
    let set = parser::parse_cypher("MATCH (d:Doc) SET d.body = 'beta'").unwrap();
    execute_mutable(
        &mut graph,
        &set,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap();

    let parsed =
        parser::parse_cypher("MATCH (d:Doc) RETURN text_bm25(d, 'body', 'alpha') AS s").unwrap();
    let no_params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &no_params, None);

    // Over the limit, so the index still holds the pre-SET text.
    let before = executor.execute(&parsed).unwrap();
    assert!(
        matches!(before.rows[0][0], Value::Float64(s) if s > 0.0),
        "the stale index still says 'alpha': {before:?}"
    );

    assert_eq!(refresh_text_index(&graph, "Doc", "body"), Some(1));

    let after = executor.execute(&parsed).unwrap();
    assert_eq!(
        after.rows[0][0],
        Value::Float64(0.0),
        "the document says 'beta' now, and 'alpha' is no longer in the corpus"
    );
}

#[test]
fn two_call_sites_in_one_query_do_not_share_a_prepared_query() {
    // `vector_score`'s single-slot cache would serve the first call's arguments
    // to the second. A hybrid query scoring a title and a body is that shape.
    let mut graph = docs(&[("quick", "slow green turtles")]);
    build_text_index(&mut graph, "Doc", "body", None).unwrap();
    build_text_index(&mut graph, "Doc", "title", None).unwrap();

    let rows = run(
        &graph,
        "MATCH (d:Doc) RETURN text_bm25(d, 'body', 'turtles') AS b, \
         text_bm25(d, 'title', 'quick') AS t",
    )
    .rows;
    assert!(
        matches!(rows[0][0], Value::Float64(s) if s > 0.0),
        "{rows:?}"
    );
    assert!(
        matches!(rows[0][1], Value::Float64(s) if s > 0.0),
        "{rows:?}"
    );
}

#[test]
fn a_null_query_is_null_for_every_row() {
    let mut graph = docs(&[("a", "the quick brown fox")]);
    build_text_index(&mut graph, "Doc", "body", None).unwrap();

    let rows = run(
        &graph,
        "MATCH (d:Doc) RETURN text_bm25(d, 'body', null) AS s",
    )
    .rows;
    assert_eq!(rows[0][0], Value::Null);
}

#[test]
fn the_scalar_composes_with_where_and_order_by_limit() {
    let mut graph = docs(&[
        ("a", "the quick brown fox"),
        ("b", "a quick quick fox and another quick fox"),
        ("c", "slow green turtles"),
    ]);
    build_text_index(&mut graph, "Doc", "body", None).unwrap();

    let filtered = run(
        &graph,
        "MATCH (d:Doc) WHERE text_bm25(d, 'body', 'quick fox') > 0.0 RETURN d.title AS t ORDER BY t",
    );
    assert_eq!(filtered.rows.len(), 2, "{filtered:?}");

    let top = run(
        &graph,
        "MATCH (d:Doc) RETURN d.title AS t, text_bm25(d, 'body', 'quick fox') AS s \
         ORDER BY s DESC LIMIT 1",
    );
    assert_eq!(top.rows.len(), 1);
    assert_eq!(top.rows[0][0], Value::String("b".to_string()));
}

#[test]
fn a_row_dependent_query_argument_is_prepared_per_row() {
    // The argument key exists to stop one row's query answering another's. A
    // query text read out of the row is the case that proves it: each row must
    // be scored against its own words.
    let mut graph = docs(&[("alpha", "alpha alpha alpha"), ("beta", "beta beta beta")]);
    build_text_index(&mut graph, "Doc", "body", None).unwrap();

    let rows = run(
        &graph,
        "MATCH (d:Doc) RETURN d.title AS t, text_bm25(d, 'body', d.title) AS s ORDER BY t",
    )
    .rows;
    // Both documents match their own title, and neither scores the other's.
    assert!(
        matches!(rows[0][1], Value::Float64(s) if s > 0.0),
        "{rows:?}"
    );
    assert!(
        matches!(rows[1][1], Value::Float64(s) if s > 0.0),
        "{rows:?}"
    );
    let cross = run(
        &graph,
        "MATCH (d:Doc) WHERE d.title = 'alpha' RETURN text_bm25(d, 'body', 'beta') AS s",
    )
    .rows;
    assert_eq!(cross[0][0], Value::Float64(0.0));
}
