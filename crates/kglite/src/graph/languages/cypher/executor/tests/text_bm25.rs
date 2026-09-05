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

// ── The postings-driven top-k operator ──────────────────────────────────────
//
// `fuse_text_bm25_order_limit` replaces `RETURN text_bm25(...) AS s ORDER BY s
// DESC LIMIT k` with `FusedTextBm25TopK`, which asks the index for its own
// top-k instead of scoring every row. These pin what the differential corpus
// structurally cannot see: it normalises row *order* away before comparing, and
// tie order is exactly where an index-driven ranking and a row-driven one come
// apart.

/// `(title, score)` per row, in row order — what `scored` returns.
type Ranking = Vec<(String, Value)>;

/// The same query's rows from the fully optimised plan and from the
/// unoptimised one, in row order.
fn ranked_both_ways(graph: &DirGraph, query: &str) -> (Ranking, Ranking) {
    let params = HashMap::new();
    let unoptimized = parser::parse_cypher(query).expect("parses");
    let mut optimized = unoptimized.clone();
    crate::graph::languages::cypher::planner::optimize(&mut optimized, graph, &params);
    assert!(
        optimized.clauses.iter().any(|c| matches!(
            c,
            crate::graph::languages::cypher::ast::Clause::FusedTextBm25TopK { .. }
        )),
        "the pass did not claim this shape, so the comparison would be vacuous: {query}"
    );
    let rows = |query: &_| -> Ranking {
        CypherExecutor::with_params(graph, &params, None)
            .execute(query)
            .unwrap_or_else(|e| panic!("query failed: {e}"))
            .rows
            .iter()
            .map(|row| match (&row[0], &row[1]) {
                (Value::String(title), score) => (title.clone(), score.clone()),
                other => panic!("unexpected row shape: {other:?}"),
            })
            .collect()
    };
    (rows(&optimized), rows(&unoptimized))
}

#[test]
fn the_fused_top_k_returns_the_same_rows_in_the_same_order_as_the_scan() {
    // Two documents share a body, so two scores are equal to the last bit and
    // the tie-break is what decides their order.
    let mut graph = docs(&[
        ("a", "alpha beta gamma"),
        ("b", "alpha alpha beta"),
        ("c", "beta gamma delta"),
        ("d", "alpha"),
        ("e", "epsilon"),
        ("f", "alpha beta"),
        ("g", "alpha beta"),
    ]);
    build_text_index(&mut graph, "Doc", "body", None).unwrap();
    for limit in [1, 2, 3, 5, 7] {
        let query = format!(
            "MATCH (d:Doc) RETURN d.title AS t, text_bm25(d, 'body', 'alpha beta') AS s \
             ORDER BY s DESC LIMIT {limit}"
        );
        let (fused, scan) = ranked_both_ways(&graph, &query);
        assert_eq!(fused, scan, "LIMIT {limit}");
    }
}

#[test]
fn the_fused_top_k_declines_when_fewer_documents_match_than_the_limit_asks_for() {
    // Only one document shares a term with the query. The scan fills the rest
    // of the top-5 with documents scoring exactly 0.0; the postings never yield
    // those, so answering from the index alone would return one row where the
    // unoptimised pipeline returns five.
    let mut graph = docs(&[
        ("a", "alpha"),
        ("b", "beta"),
        ("c", "gamma"),
        ("d", "delta"),
        ("e", "epsilon"),
    ]);
    build_text_index(&mut graph, "Doc", "body", None).unwrap();
    let (fused, scan) = ranked_both_ways(
        &graph,
        "MATCH (d:Doc) RETURN d.title AS t, text_bm25(d, 'body', 'alpha') AS s \
         ORDER BY s DESC LIMIT 5",
    );
    assert_eq!(
        fused.len(),
        5,
        "the zero-scoring documents must still be returned"
    );
    assert_eq!(fused, scan);
}

#[test]
fn a_stale_index_ranks_its_unindexed_rows_the_way_the_unoptimised_plan_does() {
    // The regression this operator shipped with: an over-limit delta scores the
    // un-caught-up rows null, `ORDER BY ... DESC` places nulls *first*, and the
    // first fallback written for this path dropped null rows instead — so the
    // fused plan answered with scored documents where the unoptimised one
    // answered with nulls.
    let mut graph = docs(&[("a", "alpha beta"), ("b", "alpha"), ("c", "beta")]);
    build_text_index(&mut graph, "Doc", "body", Some(1)).unwrap();
    for title in ["d", "e"] {
        let create = parser::parse_cypher(&format!(
            "CREATE (:Doc {{title: '{title}', body: 'alpha alpha'}})"
        ))
        .unwrap();
        execute_mutable(
            &mut graph,
            &create,
            HashMap::new(),
            crate::graph::algorithms::Interrupt::default(),
        )
        .unwrap();
    }

    let (fused, scan) = ranked_both_ways(
        &graph,
        "MATCH (d:Doc) RETURN d.title AS t, text_bm25(d, 'body', 'alpha beta') AS s \
         ORDER BY s DESC LIMIT 3",
    );
    assert!(
        fused.iter().any(|(_, score)| *score == Value::Null),
        "the stale rows must reach the answer as nulls: {fused:?}"
    );
    assert_eq!(fused, scan);
}

// ── The fused scans must refuse an unindexed property ────────────────────────
//
// `WHERE text_bm25(…) > 0 … ORDER BY … LIMIT k` — the ranked-retrieval shape the
// docs recommend — is claimed by `FusedNodeScanTopK`, whose WHERE filter drops
// any row whose predicate cannot be evaluated. "No text index on this type" is
// wrong for every row and no row can make it right, so dropping answered the
// recommended query with zero rows and no error while the bare scalar raised.
// Reported downstream on 0.16.21.

/// Optimise `query`, assert the planner really produced the fused clause the
/// test is about (an unfused plan would prove nothing), and return the error.
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
fn the_fused_top_k_scan_refuses_an_unindexed_property() {
    let graph = docs(&[("a", "the quick brown fox"), ("b", "slow green turtles")]);

    let message = fused_error(
        &graph,
        "MATCH (d:Doc) WHERE text_bm25(d, 'body', 'quick') > 0 RETURN d.title AS t \
         ORDER BY text_bm25(d, 'body', 'quick') DESC LIMIT 5",
        |clause| matches!(clause, Clause::FusedNodeScanTopK { .. }),
    );

    assert!(
        message.contains("no text index on 'Doc.body'"),
        "the fast path must raise what the scalar raises: {message}"
    );
}

#[test]
fn the_fused_scan_aggregate_refuses_an_unindexed_property() {
    // The same swallow one clause over: a count of the matching rows came back
    // as a confident zero.
    let graph = docs(&[("a", "the quick brown fox"), ("b", "slow green turtles")]);

    let message = fused_error(
        &graph,
        "MATCH (d:Doc) WHERE text_bm25(d, 'body', 'quick') > 0 RETURN count(d) AS c",
        |clause| matches!(clause, Clause::FusedNodeScanAggregate { .. }),
    );

    assert!(
        message.contains("no text index on 'Doc.body'"),
        "a count must not answer zero where the scalar raises: {message}"
    );
}

#[test]
fn fused_bm25_equal_cardinality_requires_actual_index_membership() {
    let mut graph = docs(&[("positive", "needle"), ("excluded", "other")]);
    let node = NodeData::new(
        Value::UniqueId(99),
        Value::String("missing".to_owned()),
        "Doc".to_owned(),
        HashMap::new(),
        &mut graph.interner,
    );
    let index = graph.graph.add_node(node);
    graph
        .type_indices
        .entry_or_default("Doc".to_owned())
        .push(index);
    build_text_index(&mut graph, "Doc", "body", None).unwrap();
    let (fused, scalar) = ranked_both_ways(
        &graph,
        "MATCH (d:Doc) WHERE d.title <> 'excluded' RETURN d.title AS t, \
         text_bm25(d, 'body', 'needle') AS score ORDER BY score DESC LIMIT 1",
    );
    let expected = vec![("missing".to_owned(), Value::Null)];
    assert_eq!(scalar, expected);
    assert_eq!(fused, expected);
}
