//! `vector_score(r, …) ORDER BY … DESC LIMIT k` over relationships: the fused
//! relationship route must return exactly what the unfused pipeline returns —
//! rows, order, endpoint projection — on every shape, with and without an
//! index, and including ties, NULL scores and filtered populations.
use super::*;
use crate::graph::languages::cypher::result::{CypherResult, RetrievalDiagnostics};
use crate::graph::session::execute::{execute_mut, execute_read, ExecuteOptions};
use std::collections::HashSet;

const PASS: &str = "fuse_vector_score_order_limit";

fn run(graph: &mut DirGraph, query: &str) {
    let params = HashMap::new();
    execute_mut(graph, query, &ExecuteOptions::eager(&params))
        .unwrap_or_else(|e| panic!("{query}: {e}"));
}

fn read(graph: &DirGraph, query: &str, disabled: bool) -> CypherResult {
    let params = HashMap::new();
    let passes: HashSet<String> = HashSet::from([PASS.to_string()]);
    let mut opts = ExecuteOptions::eager(&params);
    if disabled {
        opts.disabled_passes = Some(&passes);
    }
    execute_read(graph, query, &opts)
        .unwrap_or_else(|e| panic!("{query}: {e}"))
        .result
}

/// A hub with six `C` relationships to six docs (inserted in an order unlike
/// the vector order), plus one `C` between two docs and one `T` relationship.
/// k=1 and k=4 carry the *same* vector, a tie.
fn corpus(embed_all: bool) -> DirGraph {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "CREATE (h:Hub {id: 0}), (a:Doc {id: 1}), (b:Doc {id: 2}), (c:Doc {id: 3}), \
         (d:Doc {id: 4}), (e:Doc {id: 5}), (f:Doc {id: 6}), \
         (h)-[:C {k: 3, text: 't3'}]->(c), (h)-[:C {k: 1, text: 't1'}]->(a), (h)-[:C {k: 5, text: 't5'}]->(e), \
         (h)-[:C {k: 2, text: 't2'}]->(b), (h)-[:C {k: 6, text: 't6'}]->(f), (h)-[:C {k: 4, text: 't4'}]->(d), \
         (a)-[:C {k: 7, text: 't7'}]->(b), (a)-[:T {k: 8, text: 't8'}]->(c)",
    );
    let vectors = [
        (1, "[1.0, 0.2]"),
        (2, "[0.1, 1.0]"),
        (3, "[0.9, 0.5]"),
        (4, "[1.0, 0.2]"),
        (5, "[-1.0, 0.1]"),
        (6, "[0.6, 0.6]"),
        (7, "[0.3, 0.9]"),
    ];
    for (k, vector) in vectors {
        if !embed_all && k == 6 {
            continue;
        }
        run(
            &mut graph,
            &format!(
                "MATCH ()-[r:C {{k: {k}}}]->() CALL db.relationship_embeddings.set({{type:'C', \
                 text_column:'text', entries:[{{relationship:r, vector:{vector}}}]}}) \
                 YIELD stored RETURN stored"
            ),
        );
    }
    graph
}

fn assert_same(graph: &DirGraph, query: &str) -> CypherResult {
    let fused = read(graph, query, false);
    let unfused = read(graph, query, true);
    assert_eq!(fused.columns, unfused.columns, "{query}");
    assert_eq!(fused.rows, unfused.rows, "{query}");
    fused
}

fn retrieval(result: &CypherResult) -> Vec<RetrievalDiagnostics> {
    result
        .diagnostics
        .as_ref()
        .map(|d| d.retrieval.clone())
        .unwrap_or_default()
}

const SCAN: &str = "MATCH (h:Hub)-[r:C]->(d:Doc) \
                    RETURN h.id AS h, d.id AS d, r.k AS k, vector_score(r, 'text_emb', [1.0, 0.0]) AS s \
                    ORDER BY s DESC LIMIT ";

#[test]
fn plain_scan_is_served_from_the_store_and_equals_the_unfused_answer() {
    // Hub-only pattern: seven C relationships exist, one is doc-to-doc, so
    // the labelled pattern is not the store — the entry must decline.
    let graph = corpus(true);
    assert_same(&graph, &format!("{SCAN}2"));
    // The unlabelled whole-type pattern *is* the store: served by the entry.
    let query = "MATCH ()-[r:C]->() RETURN r.k AS k, vector_score(r, 'text_emb', [0.0, 1.0]) AS s \
                 ORDER BY s DESC LIMIT 3";
    let fused = assert_same(&graph, query);
    assert_eq!(
        fused
            .rows
            .iter()
            .map(|row| row[0].clone())
            .collect::<Vec<_>>(),
        vec![Value::Int64(2), Value::Int64(7), Value::Int64(6)]
    );
    let records = retrieval(&fused);
    assert!(
        records.iter().any(
            |record| record.store.as_deref() == Some("relationship:C.text_emb")
                && record.actual_mode == "exact"
                && record.fallback_reason.as_deref() == Some("no_index")
        ),
        "{records:?}"
    );
}

#[test]
fn endpoints_resolve_for_the_winners() {
    let graph = corpus(true);
    let fused = assert_same(
        &graph,
        "MATCH (a)-[r:C]->(b) RETURN a.id AS a, b.id AS b, r.k AS k, \
         vector_score(r, 'text_emb', [0.0, 1.0]) AS s ORDER BY s DESC LIMIT 3",
    );
    assert_eq!(
        fused.rows[0][..3],
        [Value::Int64(0), Value::Int64(2), Value::Int64(2)]
    );
    assert_eq!(
        fused.rows[1][..3],
        [Value::Int64(1), Value::Int64(2), Value::Int64(7)]
    );
    assert_same(
        &graph,
        "MATCH (b)<-[r:C]-(a) RETURN a.id AS a, b.id AS b, \
         vector_score(r, 'text_emb', [0.0, 1.0]) AS s ORDER BY s DESC LIMIT 3",
    );
}

#[test]
fn ties_nulls_where_and_asc_keep_the_unfused_answer() {
    let graph = corpus(true);
    // k=1 and k=4 tie for first: the entry declines, order is the matcher's.
    assert_same(
        &graph,
        "MATCH ()-[r:C]->() RETURN r.k AS k, vector_score(r, 'text_emb', [1.0, 0.0]) AS s \
         ORDER BY s DESC LIMIT 1",
    );
    assert_same(
        &graph,
        "MATCH ()-[r:C]->() RETURN r.k AS k, vector_score(r, 'text_emb', [1.0, 0.0]) AS s \
         ORDER BY s DESC LIMIT 5",
    );
    assert_same(
        &graph,
        "MATCH (a)-[r:C]->(b) WHERE b.id > 2 RETURN r.k AS k, \
         vector_score(r, 'text_emb', [0.0, 1.0]) AS s ORDER BY s DESC LIMIT 2",
    );
    assert_same(
        &graph,
        "MATCH ()-[r:C]->() RETURN r.k AS k, vector_score(r, 'text_emb', [0.0, 1.0]) AS s \
         ORDER BY s ASC LIMIT 2",
    );
    // One relationship of the type is unembedded: it scores NULL and ranks
    // first under DESC, which the store alone cannot know.
    let sparse = corpus(false);
    let fused = assert_same(
        &sparse,
        "MATCH ()-[r:C]->() RETURN r.k AS k, vector_score(r, 'text_emb', [0.0, 1.0]) AS s \
         ORDER BY s DESC LIMIT 2",
    );
    assert_eq!(fused.rows[0], vec![Value::Int64(6), Value::Null]);
}

#[test]
fn an_indexed_store_is_served_through_hnsw() {
    let mut graph = corpus(true);
    run(
        &mut graph,
        "CALL db.relationship_embeddings.build_index({type:'C', text_column:'text'}) YIELD indexed RETURN indexed",
    );
    let fused = assert_same(
        &graph,
        "MATCH ()-[r:C]->() RETURN r.k AS k, vector_score(r, 'text_emb', [0.0, 1.0]) AS s \
         ORDER BY s DESC LIMIT 3",
    );
    assert!(
        retrieval(&fused)
            .iter()
            .any(|record| record.actual_mode == "hnsw"
                && record.store.as_deref() == Some("relationship:C.text_emb")),
        "{:?}",
        retrieval(&fused)
    );
    // The rows route (a WHERE) with an index: HNSW over-fetch, filtered.
    let filtered = assert_same(
        &graph,
        "MATCH (a)-[r:C]->(b) WHERE a.id = 0 RETURN r.k AS k, \
         vector_score(r, 'text_emb', [0.0, 1.0]) AS s ORDER BY s DESC LIMIT 2",
    );
    assert!(
        retrieval(&filtered)
            .iter()
            .any(|record| record.store.as_deref() == Some("relationship:C.text_emb")),
        "{:?}",
        retrieval(&filtered)
    );
    // Forced exact bypasses the index.
    let forced = assert_same(
        &graph,
        "MATCH ()-[r:C]->() RETURN r.k AS k, \
         vector_score(r, 'text_emb', [0.0, 1.0], {exact: true}) AS s ORDER BY s DESC LIMIT 3",
    );
    assert!(retrieval(&forced)
        .iter()
        .any(|record| record.fallback_reason.as_deref() == Some("forced_exact")));
}

#[test]
fn argument_errors_keep_the_scalar_message() {
    let graph = corpus(true);
    let params = HashMap::new();
    let Err(error) = execute_read(
        &graph,
        "MATCH ()-[r:C]->() RETURN vector_score(r, 'text_emb', [0.0, 1.0, 0.0]) AS s \
         ORDER BY s DESC LIMIT 2",
        &ExecuteOptions::eager(&params),
    ) else {
        panic!("a wrong-dimension query vector must fail");
    };
    let error = error.to_string();
    assert!(
        error.contains("vector_score(): query vector dimension"),
        "{error}"
    );
}

/// Which shapes the store entry itself serves — the equality tests above hold
/// whichever route answers, so the route is pinned here directly.
#[test]
fn the_store_entry_serves_exactly_the_store_shaped_scans() {
    let graph = corpus(true);
    let sparse = corpus(false);
    let entry = |graph: &DirGraph, source: &str| {
        let params = HashMap::new();
        let mut query = parser::parse_cypher(source).unwrap();
        crate::graph::languages::cypher::planner::optimize(&mut query, graph, &params);
        CypherExecutor::with_params(graph, &params, None)
            .try_retrieval_entry(&query.clauses)
            .unwrap()
            .map(|result| result.rows.len())
    };
    let q = |pattern: &str, vector: &str, limit: usize| {
        format!(
            "MATCH {pattern} RETURN r.k AS k, vector_score(r, 'text_emb', {vector}) AS s \
             ORDER BY s DESC LIMIT {limit}"
        )
    };
    assert_eq!(entry(&graph, &q("()-[r:C]->()", "[0.0, 1.0]", 3)), Some(3));
    assert_eq!(
        entry(&graph, &q("(a)<-[r:C]-(b)", "[0.0, 1.0]", 3)),
        Some(3)
    );
    // Tie at the boundary, an unembedded relationship, a label that excludes
    // one relationship of the type, a WHERE: all left to the pipeline.
    assert_eq!(entry(&graph, &q("()-[r:C]->()", "[1.0, 0.0]", 1)), None);
    assert_eq!(entry(&sparse, &q("()-[r:C]->()", "[0.0, 1.0]", 3)), None);
    assert_eq!(entry(&graph, &q("(:Hub)-[r:C]->()", "[0.0, 1.0]", 3)), None);
    assert_eq!(
        entry(
            &graph,
            "MATCH (a)-[r:C]->() WHERE a.id = 0 RETURN vector_score(r, 'text_emb', [0.0, 1.0]) AS s \
             ORDER BY s DESC LIMIT 3"
        ),
        None
    );
}

// ── several relationship types: alternation and untyped patterns ──────────

/// Three types `A`, `B`, `D`, each with its own `text` store, and eight
/// relationships at distinct angles, so no two scores tie for any query here.
/// `mixed_metric` declares `B`'s store euclidean.
fn cross_type_corpus(mixed_metric: bool) -> DirGraph {
    let mut graph = DirGraph::new();
    run(&mut graph, "CREATE (:Hub {id: 0})");
    let edges: [(&str, i64, f64); 8] = [
        ("A", 1, 0.10),
        ("B", 2, 0.35),
        ("D", 3, 0.60),
        ("A", 4, 0.85),
        ("B", 5, 1.10),
        ("D", 6, 1.35),
        ("A", 7, 1.60),
        ("B", 8, 2.20),
    ];
    for (rel_type, k, angle) in edges {
        let metric = if mixed_metric && rel_type == "B" {
            ", metric:'euclidean'"
        } else {
            ""
        };
        run(
            &mut graph,
            &format!(
                "MATCH (h:Hub) CREATE (h)-[r:{rel_type} {{k: {k}, text: 't'}}]->(:Doc {{id: {k}}}) \
                 WITH r CALL db.relationship_embeddings.set({{type:'{rel_type}', text_column:'text', \
                 entries:[{{relationship:r, vector:[{}, {}]}}]{metric}}}) YIELD stored RETURN stored",
                angle.cos(),
                angle.sin()
            ),
        );
    }
    graph
}

/// The brute-force oracle: every relationship's cosine against `query`,
/// best first, as `k` values.
fn cross_type_oracle(query: (f64, f64), limit: usize) -> Vec<Value> {
    let angles = [0.10, 0.35, 0.60, 0.85, 1.10, 1.35, 1.60, 2.20];
    let norm = (query.0 * query.0 + query.1 * query.1).sqrt();
    let mut scored: Vec<(i64, f64)> = angles
        .iter()
        .enumerate()
        .map(|(at, angle): (usize, &f64)| {
            (
                at as i64 + 1,
                (angle.cos() * query.0 + angle.sin() * query.1) / norm,
            )
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored
        .into_iter()
        .take(limit)
        .map(|(k, _)| Value::Int64(k))
        .collect()
}

const ALL_THREE: &str = "relationship:A.text_emb,relationship:B.text_emb,relationship:D.text_emb";

fn cross_query(pattern: &str, limit: usize) -> String {
    format!(
        "MATCH {pattern} RETURN r.k AS k, vector_score(r, 'text_emb', [0.3, 1.0]) AS s \
         ORDER BY s DESC LIMIT {limit}"
    )
}

fn ks(result: &CypherResult) -> Vec<Value> {
    result.rows.iter().map(|row| row[0].clone()).collect()
}

#[test]
fn alternation_and_untyped_scans_merge_every_store() {
    let graph = cross_type_corpus(false);
    for pattern in ["()-[r:A|B|D]->()", "()-[r]->()", "(:Hub)-[r:D|A|B]->(:Doc)"] {
        let fused = assert_same(&graph, &cross_query(pattern, 4));
        assert_eq!(ks(&fused), cross_type_oracle((0.3, 1.0), 4), "{pattern}");
        let records = retrieval(&fused);
        assert!(
            records
                .iter()
                .any(|record| record.store.as_deref() == Some(ALL_THREE)
                    && record.actual_mode == "exact"
                    && record.fallback_reason.as_deref() == Some("no_index")),
            "{pattern}: {records:?}"
        );
    }
    // A two-type alternation reads only those two stores.
    let two = assert_same(&graph, &cross_query("()-[r:B|A]->()", 3));
    assert!(retrieval(&two)
        .iter()
        .any(|record| record.store.as_deref()
            == Some("relationship:A.text_emb,relationship:B.text_emb")));
}

#[test]
fn indexed_stores_merge_through_hnsw_on_both_routes() {
    let mut graph = cross_type_corpus(false);
    for rel_type in ["A", "B"] {
        run(
            &mut graph,
            &format!(
                "CALL db.relationship_embeddings.build_index({{type:'{rel_type}', text_column:'text'}}) \
                 YIELD indexed RETURN indexed"
            ),
        );
    }
    // `D` has no index yet: the merge stays exact rather than mixing routes.
    let partial = assert_same(&graph, &cross_query("()-[r:A|B|D]->()", 4));
    assert!(retrieval(&partial)
        .iter()
        .any(|record| record.actual_mode == "exact"));
    run(
        &mut graph,
        "CALL db.relationship_embeddings.build_index({type:'D', text_column:'text'}) YIELD indexed RETURN indexed",
    );
    for pattern in ["()-[r:A|B|D]->()", "()-[r]->()"] {
        let fused = assert_same(&graph, &cross_query(pattern, 4));
        assert_eq!(ks(&fused), cross_type_oracle((0.3, 1.0), 4), "{pattern}");
        assert!(
            retrieval(&fused)
                .iter()
                .any(|record| record.actual_mode == "hnsw"
                    && record.store.as_deref() == Some(ALL_THREE)),
            "{pattern}: {:?}",
            retrieval(&fused)
        );
    }
    // The rows route: a WHERE keeps every store whole, so HNSW serves it.
    let rows = assert_same(
        &graph,
        "MATCH (h:Hub)-[r:A|B|D]->() WHERE h.id = 0 RETURN r.k AS k, \
         vector_score(r, 'text_emb', [0.3, 1.0]) AS s ORDER BY s DESC LIMIT 4",
    );
    assert_eq!(ks(&rows), cross_type_oracle((0.3, 1.0), 4));
    assert!(
        retrieval(&rows).iter().any(
            |record| record.actual_mode == "hnsw" && record.store.as_deref() == Some(ALL_THREE)
        ),
        "{:?}",
        retrieval(&rows)
    );
    // A filter that leaves one store's rows short of k falls back to exact.
    let underfilled = assert_same(
        &graph,
        "MATCH ()-[r:A|B|D]->(d) WHERE d.id <> 3 RETURN r.k AS k, \
         vector_score(r, 'text_emb', [0.3, 1.0]) AS s ORDER BY s DESC LIMIT 4",
    );
    assert!(
        retrieval(&underfilled)
            .iter()
            .any(|record| record.fallback_reason.as_deref() == Some("filtered_underfill")),
        "{:?}",
        retrieval(&underfilled)
    );
}

#[test]
fn a_type_in_play_without_the_store_keeps_the_scalar_error() {
    let mut graph = cross_type_corpus(false);
    run(
        &mut graph,
        "MATCH (h:Hub), (d:Doc {id: 1}) CREATE (h)-[:PLAIN]->(d)",
    );
    for disabled in [false, true] {
        let params = HashMap::new();
        let passes: HashSet<String> = HashSet::from([PASS.to_string()]);
        let mut opts = ExecuteOptions::eager(&params);
        if disabled {
            opts.disabled_passes = Some(&passes);
        }
        for pattern in ["()-[r]->()", "()-[r:A|PLAIN]->()"] {
            let Err(error) = execute_read(&graph, &cross_query(pattern, 3), &opts) else {
                panic!("{pattern}: a type without the store must fail");
            };
            let error = error.to_string();
            assert!(
                error.contains("no embedding 'text_emb' found for relationship type 'PLAIN'"),
                "{pattern} disabled={disabled}: {error}"
            );
        }
    }
    // The alternation that names only stored types is still served.
    let fused = assert_same(&graph, &cross_query("()-[r:A|B|D]->()", 3));
    assert_eq!(ks(&fused), cross_type_oracle((0.3, 1.0), 3));
}

#[test]
fn stores_under_different_metrics_rank_as_the_scalar_scores() {
    // The fused route ranks what the written query ranks: each relationship
    // scored under its own store's metric, exactly as row-by-row scoring does.
    let graph = cross_type_corpus(true);
    let fused = assert_same(&graph, &cross_query("()-[r:A|B|D]->()", 5));
    assert_eq!(fused.rows.len(), 5);
}

#[test]
fn the_store_entry_serves_alternation_and_untyped_scans() {
    let graph = cross_type_corpus(false);
    let entry = |source: &str| {
        let params = HashMap::new();
        let mut query = parser::parse_cypher(source).unwrap();
        crate::graph::languages::cypher::planner::optimize(&mut query, &graph, &params);
        CypherExecutor::with_params(&graph, &params, None)
            .try_retrieval_entry(&query.clauses)
            .unwrap()
            .map(|result| result.rows.len())
    };
    assert_eq!(entry(&cross_query("()-[r:A|B|D]->()", 3)), Some(3));
    assert_eq!(entry(&cross_query("()-[r]->()", 3)), Some(3));
    assert_eq!(entry(&cross_query("(:Hub)<-[r:A|B]-()", 3)), None);
}

// ── embedding(x, 'col_emb'): the stored vector, both entities ──────────

fn read_rows(graph: &DirGraph, query: &str) -> Vec<Vec<Value>> {
    read(graph, query, false).rows
}

fn floats(value: &Value) -> Vec<f64> {
    let Value::List(items) = value else {
        panic!("expected a list, got {value:?}");
    };
    items
        .iter()
        .map(|item| match item {
            Value::Float64(x) => *x,
            other => panic!("expected floats, got {other:?}"),
        })
        .collect()
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
fn embedding_reads_a_relationship_vector_from_a_binding_or_a_value() {
    let graph = cross_type_corpus(false);
    // k=4 is stored at angle 0.85.
    let expected = [0.85f64.cos() as f32 as f64, 0.85f64.sin() as f32 as f64];
    for query in [
        "MATCH ()-[r:A {k: 4}]->() RETURN embedding(r, 'text_emb') AS v",
        "MATCH ()-[r:A {k: 4}]->() WITH collect(r) AS rs RETURN embedding(rs[0], 'text_emb') AS v",
        "MATCH ()-[r:A {k: 4}]->() WITH collect(r) AS rs UNWIND rs AS x RETURN embedding(x, 'text_emb') AS v",
    ] {
        let rows = read_rows(&graph, query);
        assert_eq!(floats(&rows[0][0]), expected, "{query}");
    }
}

#[test]
fn edge_to_edge_similarity_composes_with_vector_score() {
    let graph = cross_type_corpus(false);
    let rows = read_rows(
        &graph,
        "MATCH ()-[a:A {k: 1}]->(), ()-[b:B {k: 5}]->() \
         RETURN vector_score(b, 'text_emb', embedding(a, 'text_emb')) AS s, \
                embedding(a, 'text_emb') AS va, embedding(b, 'text_emb') AS vb",
    );
    let (va, vb) = (floats(&rows[0][1]), floats(&rows[0][2]));
    let Value::Float64(score) = rows[0][0] else {
        panic!("{:?}", rows[0][0]);
    };
    assert!((score - cosine(&va, &vb)).abs() < 1e-6, "{score} vs oracle");
    assert!((score - (1.10f64 - 0.10).cos()).abs() < 1e-5);
}

#[test]
fn embedding_reads_node_vectors_and_is_null_without_one() {
    let mut graph = cross_type_corpus(false);
    run(&mut graph, "MATCH (d:Doc) SET d.summary = 'text'");
    crate::graph::embeddings::set_embeddings(
        &mut graph,
        "Doc",
        "summary",
        None,
        [
            (Value::Int64(1), vec![1.0, 0.0]),
            (Value::Int64(2), vec![0.6, 0.8]),
        ],
    )
    .unwrap();
    let rows = read_rows(
        &graph,
        "MATCH (a:Doc {id: 1}), (b:Doc {id: 2}) \
         RETURN embedding(a, 'summary_emb') AS va, \
                vector_score(b, 'summary_emb', embedding(a, 'summary_emb')) AS s",
    );
    assert_eq!(floats(&rows[0][0]), vec![1.0, 0.0]);
    assert_eq!(rows[0][1], Value::Float64(0.6f32 as f64));
    // A node of the type without a vector, and a node value, read the same way.
    let rows = read_rows(
        &graph,
        "MATCH (d:Doc) WITH d ORDER BY d.id WITH collect(d) AS ds \
         RETURN embedding(ds[1], 'summary_emb') AS second, embedding(ds[2], 'summary_emb') AS third",
    );
    assert_eq!(floats(&rows[0][0]), vec![0.6f32 as f64, 0.8f32 as f64]);
    assert_eq!(rows[0][1], Value::Null);
}

#[test]
fn embedding_without_the_store_is_refused_naming_type_and_property() {
    let mut graph = cross_type_corpus(false);
    run(
        &mut graph,
        "MATCH (h:Hub), (d:Doc {id: 1}) CREATE (h)-[:PLAIN]->(d)",
    );
    let params = HashMap::new();
    let refuse = |query: &str| {
        execute_read(&graph, query, &ExecuteOptions::eager(&params))
            .err()
            .unwrap_or_else(|| panic!("{query} must fail"))
            .to_string()
    };
    let error = refuse("MATCH ()-[r:PLAIN]->() RETURN embedding(r, 'text_emb') AS v");
    assert!(
        error.contains("embedding(): no embedding 'text_emb' found for relationship type 'PLAIN'"),
        "{error}"
    );
    let error = refuse("MATCH ()-[r:A]->() RETURN embedding(r, 'text') AS v");
    assert!(error.contains("Did you mean 'text_emb'?"), "{error}");
    let error = refuse("MATCH (h:Hub) RETURN embedding(h, 'text_emb') AS v");
    assert!(error.contains("for node type 'Hub'"), "{error}");
    let error = refuse("RETURN embedding(1, 'text_emb') AS v");
    assert!(
        error.contains("first argument must be a node or a relationship"),
        "{error}"
    );
    assert_eq!(
        read_rows(
            &graph,
            "OPTIONAL MATCH (n:Missing) RETURN embedding(n, 'x_emb') AS v"
        )[0][0],
        Value::Null
    );
}

/// A missing store is an error in a fused `WHERE` too, never a row that
/// "does not match": the fused filters once swallowed per-row evaluation
/// errors, and counted `0` for `embedding()` / `embedding_norm()` where
/// `vector_score` raised.
#[test]
fn a_missing_store_raises_inside_a_fused_where() {
    let mut graph = cross_type_corpus(false);
    run(
        &mut graph,
        "MATCH (h:Hub), (d:Doc {id: 1}) CREATE (h)-[:PLAIN]->(d)",
    );
    let params = HashMap::new();
    let cases = [
        (
            "MATCH (h:Hub) WHERE size(embedding(h, 'text_emb')) > 0 RETURN count(h) AS n",
            "embedding(): no embedding 'text_emb' found for node type 'Hub'",
        ),
        (
            "MATCH (h:Hub) WHERE embedding_norm(h, 'text_emb') > 0 RETURN count(h) AS n",
            "embedding_norm(): no embedding 'text_emb' found for node type 'Hub'",
        ),
        (
            "MATCH ()-[r:PLAIN]->() WHERE size(embedding(r, 'text_emb')) > 0 RETURN count(r) AS n",
            "embedding(): no embedding 'text_emb' found for relationship type 'PLAIN'",
        ),
        (
            "MATCH ()-[r:PLAIN]->() WHERE embedding_norm(r, 'text_emb') > 0 RETURN count(r) AS n",
            "embedding_norm(): no embedding 'text_emb' found for relationship type 'PLAIN'",
        ),
        (
            "MATCH (h:Hub) WHERE text_score(h, 'text', [1.0, 0.0]) > 0 RETURN count(h) AS n",
            "text_score(): no embedding for property 'text' on node type 'Hub'",
        ),
    ];
    for (query, expected) in cases {
        let Err(error) = execute_read(&graph, query, &ExecuteOptions::eager(&params)) else {
            panic!("{query}: a missing store must raise, not count 0");
        };
        let error = error.to_string();
        assert!(error.contains(expected), "{query}: {error}");
    }
}
