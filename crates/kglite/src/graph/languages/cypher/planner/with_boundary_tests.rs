//! Plan-shape tests for the WITH-boundary rewrites.
//!
//! The differential corpus proves the rewritten plan answers what the
//! unoptimised plan answers; these prove the rewrite *happened* on each
//! trigger shape and *did not happen* on each bail. A gate that silently
//! stops firing is a lost optimisation the corpus cannot see, and a gate
//! that silently starts firing on a bail shape is a wrong answer.

use super::*;
use crate::graph::languages::cypher::parser::parse_cypher;

/// Four `P` nodes over two cities and four ages, plus four `K` edges so a
/// multi-row driving pattern (and therefore a hideable variable) exists.
fn with_boundary_graph() -> DirGraph {
    let nodes = crate::datatypes::DataFrame::from_cypher_rows(
        vec!["id".into(), "title".into(), "city".into(), "age".into()],
        vec![
            vec![
                Value::Int64(1),
                Value::String("a".into()),
                Value::String("X".into()),
                Value::Int64(25),
            ],
            vec![
                Value::Int64(2),
                Value::String("b".into()),
                Value::String("X".into()),
                Value::Int64(35),
            ],
            vec![
                Value::Int64(3),
                Value::String("c".into()),
                Value::String("Y".into()),
                Value::Int64(45),
            ],
            vec![
                Value::Int64(4),
                Value::String("d".into()),
                Value::String("Y".into()),
                Value::Int64(55),
            ],
        ],
    )
    .unwrap();
    let edges = crate::datatypes::DataFrame::from_cypher_rows(
        vec!["src".into(), "tgt".into()],
        vec![
            vec![Value::Int64(1), Value::Int64(2)],
            vec![Value::Int64(1), Value::Int64(3)],
            vec![Value::Int64(2), Value::Int64(3)],
            vec![Value::Int64(3), Value::Int64(4)],
        ],
    )
    .unwrap();

    let mut graph = DirGraph::new();
    crate::graph::mutation::maintain::add_nodes(
        &mut graph,
        nodes,
        "P".to_string(),
        "id".to_string(),
        Some("title".to_string()),
        None,
    )
    .unwrap();
    crate::graph::mutation::maintain::add_connections(
        &mut graph,
        edges,
        "K".to_string(),
        "P".to_string(),
        "src".to_string(),
        "P".to_string(),
        "tgt".to_string(),
        None,
        None,
        None,
    )
    .unwrap();
    graph
}

/// True when a `WITH` still carries its own `WHERE` after optimisation —
/// i.e. `hoist_with_where` declined this query.
fn with_where_survives(graph: &DirGraph, text: &str) -> bool {
    let params = HashMap::new();
    let mut query = parse_cypher(text).unwrap();
    optimize(&mut query, graph, &params);
    query
        .clauses
        .iter()
        .any(|c| matches!(c, Clause::With(w) if w.where_clause.is_some()))
}

/// Rows from the fully optimised plan.
fn rows(graph: &DirGraph, text: &str) -> Vec<Vec<Value>> {
    let params = HashMap::new();
    let mut query = parse_cypher(text).unwrap();
    optimize(&mut query, graph, &params);
    crate::graph::languages::cypher::executor::CypherExecutor::with_params(graph, &params, None)
        .execute(&query)
        .unwrap()
        .rows
}

#[test]
fn the_hoist_fires_on_every_trigger_shape() {
    let graph = with_boundary_graph();
    for text in [
        // The measured cell: the predicate reaches the scan instead of a
        // per-row filter behind a whole-node projection.
        "MATCH (p:P) WITH p WHERE p.age > 30 RETURN count(*) AS n",
        // Rows delivered rather than counted.
        "MATCH (p:P) WITH p WHERE p.city = 'X' RETURN p.title AS t",
        // H5 satisfied while the WITH drops `b`: the predicate reads only
        // the projected variable.
        "MATCH (a:P)-[:K]->(b:P) WITH a WHERE a.age > 30 RETURN a.title AS t",
        // `WITH *` hides nothing, so the scope half of H5 is satisfied by
        // construction. The star keeps `fold_pass_through_with` from
        // removing the WITH afterwards — the hoist is still the win.
        "MATCH (p:P) WITH * WHERE p.city = 'X' RETURN p.title AS t",
    ] {
        assert!(
            !with_where_survives(&graph, text),
            "expected the WITH-attached WHERE to be hoisted: {text}"
        );
    }
}

#[test]
fn the_hoist_declines_every_bail_shape() {
    let graph = with_boundary_graph();
    for (text, why) in [
        (
            "MATCH (p:P) WITH p.city AS c, count(*) AS k WHERE k > 1 RETURN c, k",
            "H2 — the predicate is a HAVING over groups",
        ),
        (
            "MATCH (p:P) WITH p WHERE count(*) > 1 RETURN p.title AS t",
            "H2 — an aggregate inside the predicate itself",
        ),
        (
            "MATCH (p:P) WITH DISTINCT p.city AS c WHERE c <> 'X' RETURN c",
            "H3 — DISTINCT",
        ),
        (
            "MATCH (p:P) WITH p.age AS a WHERE a > 30 RETURN count(*) AS n",
            "H6 — the predicate reads an alias the WITH introduces",
        ),
        (
            "MATCH (p:P) UNWIND [25, 35] AS k WITH p, k WHERE p.age > k RETURN count(*) AS n",
            "H1 — a clause between the MATCH and the WITH",
        ),
        (
            "MATCH (p:P) OPTIONAL MATCH (p)-[:K]->(q) WITH p, q WHERE q.age > 30 RETURN count(*) AS n",
            "H7 — an OPTIONAL MATCH before the WITH",
        ),
        (
            "MATCH (a:P)-[:K]->(b:P) WITH a WHERE b.city = 'X' RETURN a.title AS t",
            "H5 — the predicate reads a variable the WITH hides (a scope error)",
        ),
    ] {
        assert!(
            with_where_survives(&graph, text),
            "expected the WITH-attached WHERE to stand ({why}): {text}"
        );
    }
}

/// Absolute answers through the rewritten plan. The corpus compares the two
/// profiles against each other; these say what the number is.
#[test]
fn the_hoisted_plan_answers_absolute_values() {
    let graph = with_boundary_graph();
    assert_eq!(
        rows(
            &graph,
            "MATCH (p:P) WITH p WHERE p.age > 30 RETURN count(*) AS n"
        ),
        vec![vec![Value::Int64(3)]]
    );
    assert_eq!(
        rows(
            &graph,
            "MATCH (p:P) WITH p WHERE p.city = 'X' RETURN p.title AS t"
        ),
        vec![
            vec![Value::String("a".into())],
            vec![Value::String("b".into())]
        ]
    );
    // The dropped `b` binding stays dropped: one row per (a, b) pair whose
    // `a` qualifies, not one row per `a`.
    assert_eq!(
        rows(
            &graph,
            "MATCH (a:P)-[:K]->(b:P) WITH a WHERE a.age > 30 RETURN a.title AS t"
        ),
        vec![
            vec![Value::String("b".into())],
            vec![Value::String("c".into())]
        ]
    );
}

// ============================================================================
// fold_aliasing_with / hoist_terminal_return_over_with_top_k
// ============================================================================

/// True when a `WITH` is still standing after optimisation.
fn with_survives(graph: &DirGraph, text: &str) -> bool {
    let params = HashMap::new();
    let mut query = parse_cypher(text).unwrap();
    optimize(&mut query, graph, &params);
    query.clauses.iter().any(|c| matches!(c, Clause::With(_)))
}

/// True when the optimised plan reached a fused top-K operator — the whole
/// point of removing the barrier.
fn fuses_top_k(graph: &DirGraph, text: &str) -> bool {
    let params = HashMap::new();
    let mut query = parse_cypher(text).unwrap();
    optimize(&mut query, graph, &params);
    query.clauses.iter().any(|c| {
        matches!(
            c,
            Clause::FusedNodeScanTopK { .. } | Clause::FusedOrderByTopK { .. }
        )
    })
}

#[test]
fn an_aliasing_with_folds_and_reaches_the_top_k_operator() {
    let graph = with_boundary_graph();
    for text in [
        // Form B — the node plus a derived sort key.
        "MATCH (p:P) WITH p, p.age AS a RETURN p.title AS t ORDER BY a DESC LIMIT 2",
        // Form C — all-scalar projection, column name carried by the fold.
        "MATCH (p:P) WITH p.title AS t, p.age AS a RETURN t ORDER BY a DESC LIMIT 2",
        // Form E — the openCypher clause order, which needs the reorder.
        "MATCH (p:P) WITH p, p.age AS a ORDER BY a DESC LIMIT 2 RETURN p.title AS t",
    ] {
        assert!(
            !with_survives(&graph, text),
            "WITH should be folded: {text}"
        );
        assert!(fuses_top_k(&graph, text), "top-K should fuse: {text}");
    }
    // No ORDER BY: the fold still fires, and the result is the shape
    // `push_limit_into_match`'s own single-MATCH guard already covers.
    assert!(!with_survives(
        &graph,
        "MATCH (p:P) WITH p, p.age AS a RETURN p.title AS t LIMIT 2"
    ));
}

#[test]
fn the_aliasing_fold_declines_every_bail_shape() {
    let graph = with_boundary_graph();
    for (text, why) in [
        (
            "MATCH (p:P) WITH p.city AS c, count(*) AS k RETURN c, k ORDER BY k DESC LIMIT 2",
            "F1 — an aggregating WITH is not 1:1",
        ),
        (
            "MATCH (p:P) WITH DISTINCT p.city AS c RETURN c ORDER BY c LIMIT 2",
            "F1 — DISTINCT",
        ),
        (
            "MATCH (p:P) WITH p, p.age AS a WHERE a > 30 RETURN p.title AS t ORDER BY a LIMIT 2",
            "F1 — a WITH-attached WHERE this pass does not own",
        ),
        (
            "MATCH (p:P) WITH p, p.title AS t MATCH (p)-[:K]->(q:P) RETURN t, q.title AS u",
            "F5 — a MATCH downstream",
        ),
        (
            "MATCH (p:P) WITH p.title AS p RETURN p LIMIT 2",
            "F7 — the alias shadows a pre-WITH variable",
        ),
        (
            "MATCH (p:P) WITH p, p.age AS a RETURN p.title AS a ORDER BY a DESC LIMIT 2",
            "F6 — the RETURN re-binds the WITH's name, shadowing it downstream",
        ),
        (
            "MATCH (p:P) WITH p, p.age AS a RETURN * ORDER BY a DESC LIMIT 2",
            "F6 — `RETURN *` reads the runtime row the fold changes",
        ),
        (
            "MATCH (p:P) WITH p, p.city AS c ORDER BY c LIMIT 2 RETURN DISTINCT c",
            "E2 — a DISTINCT terminal RETURN is not 1:1",
        ),
        (
            "MATCH (p:P) WITH p, p.city AS c ORDER BY c LIMIT 2 RETURN c, count(*) AS k",
            "E2 — an aggregating terminal RETURN changes the row count",
        ),
        (
            "MATCH (p:P) WITH p ORDER BY p.age DESC LIMIT 2 MATCH (p)-[:K]->(q) RETURN count(*) AS n",
            "E1 — a clause other than the terminal RETURN follows the window",
        ),
    ] {
        assert!(
            with_survives(&graph, text),
            "the WITH should stand ({why}): {text}"
        );
    }
}

/// Absolute answers through the folded plans, including the tie order and the
/// NULL placement the reorder has to preserve.
#[test]
fn the_folded_plans_answer_absolute_values() {
    let graph = with_boundary_graph();
    assert_eq!(
        rows(
            &graph,
            "MATCH (p:P) WITH p.title AS t, p.age AS a RETURN t ORDER BY a DESC LIMIT 2"
        ),
        vec![
            vec![Value::String("d".into())],
            vec![Value::String("c".into())]
        ]
    );
    assert_eq!(
        rows(
            &graph,
            "MATCH (p:P) WITH p, p.age AS a ORDER BY a ASC LIMIT 2 RETURN p.title AS t"
        ),
        vec![
            vec![Value::String("a".into())],
            vec![Value::String("b".into())]
        ]
    );
    // SKIP is carried, not dropped.
    assert_eq!(
        rows(
            &graph,
            "MATCH (p:P) WITH p, p.age AS a ORDER BY a DESC SKIP 1 LIMIT 2 RETURN p.title AS t"
        ),
        vec![
            vec![Value::String("c".into())],
            vec![Value::String("b".into())]
        ]
    );
}

fn fuses_vector_top_k(graph: &DirGraph, text: &str) -> bool {
    let params = HashMap::new();
    let mut query = parse_cypher(text).unwrap();
    optimize(&mut query, graph, &params);
    query
        .clauses
        .iter()
        .any(|c| matches!(c, Clause::FusedVectorScoreTopK { .. }))
}

#[test]
fn retrieval_with_alias_reaches_vector_fusion() {
    let graph = DirGraph::new();
    for text in [
        "MATCH (n:Doc) WITH n, vector_score(n, 'summary_emb', $q) AS s \
         ORDER BY s DESC LIMIT 10 RETURN n.id AS id, s",
        "MATCH (n:Doc) WITH n, vector_score(n, 'summary_emb', $q) AS s \
         ORDER BY s DESC LIMIT 10 RETURN n.id AS id",
    ] {
        assert!(
            !with_survives(&graph, text),
            "WITH should be folded: {text}"
        );
        assert!(
            fuses_vector_top_k(&graph, text),
            "vector top-K should fuse: {text}"
        );
    }
}

#[test]
fn rand_with_alias_does_not_fold() {
    let graph = with_boundary_graph();
    let text = "MATCH (p:P) WITH p, rand() AS s ORDER BY s LIMIT 2 RETURN p.title AS t";
    assert!(
        with_survives(&graph, text),
        "rand() is not substitutable, so the WITH must stand"
    );
}
