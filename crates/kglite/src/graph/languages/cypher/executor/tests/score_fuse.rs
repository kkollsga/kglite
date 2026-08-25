//! The `score_fuse()` scalar: which signals reach the average, which weights
//! are refused, and what a call made entirely of constants folds to.
//!
//! The absent-signal rule is the whole function — a lane that could not see a
//! row reports `null` (or `NaN`, or `±inf`), and the row must still be ranked
//! on the lanes that did run. Most of the cases below are expression-level
//! rather than query-level because `NaN` and `±inf` have no Cypher literal.
//!
//! Red proof: before the scalar existed every case here failed with
//! `Unknown function: score_fuse`.

use super::*;

fn lit(v: Value) -> Expression {
    Expression::Literal(v)
}

fn f(x: f64) -> Expression {
    lit(Value::Float64(x))
}

fn list(items: Vec<Value>) -> Expression {
    lit(Value::List(items))
}

/// Evaluate `score_fuse` on constant arguments through the real dispatcher.
fn fuse(args: &[Expression]) -> Result<Value, String> {
    let graph = DirGraph::new();
    let params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &params, None);
    executor.test_evaluate_scalar_function("score_fuse", args, &ResultRow::new())
}

fn fused(args: &[Expression]) -> Value {
    fuse(args).unwrap_or_else(|e| panic!("score_fuse unexpectedly failed: {e}"))
}

fn refused(args: &[Expression]) -> String {
    match fuse(args) {
        Ok(v) => panic!("score_fuse unexpectedly succeeded with {v:?}"),
        Err(e) => e,
    }
}

fn float(value: &Value) -> f64 {
    match value {
        Value::Float64(f) => *f,
        other => panic!("expected a float, got {other:?}"),
    }
}

// ── the average, and what is in it ────────────────────────────────────────

#[test]
fn two_present_signals_average_with_equal_weight() {
    assert_eq!(float(&fused(&[f(1.0), f(3.0)])), 2.0);
}

#[test]
fn integers_score_as_numbers() {
    // A lane can hand back an Int64 (a count, a rank arithmetic result); the
    // fusion is float arithmetic either way.
    assert_eq!(
        float(&fused(&[lit(Value::Int64(1)), lit(Value::Int64(4))])),
        2.5
    );
}

#[test]
fn an_absent_signal_leaves_the_average_instead_of_scoring_zero() {
    // THE contract. Treating null as 0.0 would give 0.5 here and rank this row
    // below a document both lanes searched and disliked.
    assert_eq!(float(&fused(&[f(1.0), lit(Value::Null)])), 1.0);
    assert_eq!(float(&fused(&[lit(Value::Null), f(0.25), f(0.75)])), 0.5);
}

#[test]
fn nan_and_infinity_are_absent_too() {
    // A lane with no answer for this row may report either instead of null —
    // an empty-corpus IDF, a zero-norm cosine. Neither carries a rank position.
    assert_eq!(float(&fused(&[f(2.0), f(f64::NAN)])), 2.0);
    assert_eq!(float(&fused(&[f(2.0), f(f64::INFINITY)])), 2.0);
    assert_eq!(float(&fused(&[f(2.0), f(f64::NEG_INFINITY)])), 2.0);
}

#[test]
fn every_signal_absent_is_null() {
    assert_eq!(fused(&[lit(Value::Null), lit(Value::Null)]), Value::Null);
    assert_eq!(fused(&[lit(Value::Null), f(f64::NAN)]), Value::Null);
}

#[test]
fn a_non_numeric_signal_is_an_error_naming_its_position() {
    let err = refused(&[f(1.0), lit(Value::String("high".into()))]);
    assert!(err.contains("argument 2"), "{err}");
    assert!(err.contains("number or null"), "{err}");
}

// ── weights ───────────────────────────────────────────────────────────────

#[test]
fn a_trailing_list_is_the_weight_vector() {
    // 3:1 in favour of the first lane: (3*1 + 1*3) / 4.
    let weighted = fused(&[
        f(1.0),
        f(3.0),
        list(vec![Value::Float64(3.0), Value::Float64(1.0)]),
    ]);
    assert_eq!(float(&weighted), 1.5);
}

#[test]
fn an_absent_signals_weight_leaves_the_denominator_with_it() {
    // Not 4.0 * 1/4: the missing lane's 3.0 weight is not evidence for a zero.
    let weighted = fused(&[
        lit(Value::Null),
        f(4.0),
        list(vec![Value::Float64(3.0), Value::Float64(1.0)]),
    ]);
    assert_eq!(float(&weighted), 4.0);
}

#[test]
fn all_weights_zero_is_null_not_a_division_by_zero() {
    let weighted = fused(&[
        f(1.0),
        f(3.0),
        list(vec![Value::Float64(0.0), Value::Float64(0.0)]),
    ]);
    assert_eq!(weighted, Value::Null);
}

#[test]
fn a_weights_list_of_the_wrong_length_is_refused() {
    let err = refused(&[
        f(1.0),
        f(2.0),
        f(3.0),
        list(vec![Value::Float64(1.0), Value::Float64(1.0)]),
    ]);
    assert!(err.contains("2 weights for 3 scores"), "{err}");
}

#[test]
fn a_negative_weight_is_refused() {
    let err = refused(&[
        f(1.0),
        f(3.0),
        list(vec![Value::Float64(1.0), Value::Float64(-1.0)]),
    ]);
    assert!(err.contains("weight 2"), "{err}");
    assert!(err.contains("≥ 0"), "{err}");
}

#[test]
fn a_non_numeric_weight_is_refused_even_when_its_signal_is_absent() {
    // The weights list is a query bug wherever it appears; which rows happen
    // to have an absent signal must not decide whether it is reported.
    let err = refused(&[
        lit(Value::Null),
        f(3.0),
        list(vec![Value::String("heavy".into()), Value::Float64(1.0)]),
    ]);
    assert!(err.contains("weight 1"), "{err}");
}

#[test]
fn an_infinite_weight_is_refused() {
    let err = refused(&[
        f(1.0),
        f(3.0),
        list(vec![Value::Float64(f64::INFINITY), Value::Float64(1.0)]),
    ]);
    assert!(err.contains("weight 1"), "{err}");
}

// ── arity ─────────────────────────────────────────────────────────────────

#[test]
fn fewer_than_two_scores_is_refused() {
    assert!(refused(&[f(1.0)]).contains("2 or more scores"));
    // One score plus weights is the same shortage wearing a longer arg list.
    assert!(refused(&[f(1.0), list(vec![Value::Float64(1.0)])]).contains("2 or more scores"));
}

// ── constant folding ──────────────────────────────────────────────────────

#[test]
fn an_all_constant_call_folds_to_the_right_constant() {
    // `score_fuse` reads no row state, so `is_row_independent` folds a
    // constant call to a literal once per query instead of per row. The value
    // has to survive that: a fold to the wrong constant is a wrong answer for
    // every row at once.
    let call = Expression::FunctionCall {
        name: "score_fuse".to_string(),
        args: vec![
            f(1.0),
            f(3.0),
            list(vec![Value::Float64(3.0), Value::Float64(1.0)]),
        ],
        distinct: false,
    };
    assert!(CypherExecutor::is_row_independent(&call));

    let graph = DirGraph::new();
    let params = HashMap::new();
    let executor = CypherExecutor::with_params(&graph, &params, None);
    match executor.fold_constants_expr(&call) {
        Expression::Literal(Value::Float64(v)) => assert_eq!(v, 1.5),
        other => panic!("expected a folded literal, got {other:?}"),
    }
}

// ── end to end ────────────────────────────────────────────────────────────

#[test]
fn a_query_fuses_two_row_dependent_lanes_and_orders_by_the_result() {
    // The marquee shape in miniature: two per-row scores, fused, ranked. `b`
    // wins on the fusion despite losing the first lane, and `c` — which one
    // lane could not score at all — is ranked on the other rather than dropped.
    let mut graph = DirGraph::new();
    for (id, title, a, b) in [
        (1u32, "a", Value::Float64(0.9), Value::Float64(0.1)),
        (2, "b", Value::Float64(0.6), Value::Float64(0.9)),
        (3, "c", Value::Null, Value::Float64(0.7)),
    ] {
        let node = NodeData::new(
            Value::UniqueId(id),
            Value::String(title.to_string()),
            "Doc".to_string(),
            HashMap::from([("lex".to_string(), a), ("vec".to_string(), b)]),
            &mut graph.interner,
        );
        let idx = graph.graph.add_node(node);
        graph
            .type_indices
            .entry_or_default("Doc".to_string())
            .push(idx);
    }

    let parsed = parser::parse_cypher(
        "MATCH (d:Doc) RETURN d.title AS t, score_fuse(d.lex, d.vec) AS s ORDER BY s DESC",
    )
    .unwrap();
    let params = HashMap::new();
    let result = CypherExecutor::with_params(&graph, &params, None)
        .execute(&parsed)
        .unwrap();

    let ranked: Vec<(String, f64)> = result
        .rows
        .iter()
        .map(|row| match (&row[0], &row[1]) {
            (Value::String(t), score) => (t.clone(), float(score)),
            other => panic!("unexpected row shape: {other:?}"),
        })
        .collect();
    assert_eq!(ranked[0], ("b".to_string(), 0.75));
    // `c` scores its one present lane, not half of it.
    assert_eq!(ranked[1], ("c".to_string(), 0.7));
    assert_eq!(ranked[2], ("a".to_string(), 0.5));
}

#[test]
fn rrf_is_expressible_as_window_ranks_fused() {
    // The documented Reciprocal Rank Fusion recipe (CYPHER.md), pinned because
    // it is the answer to "where is rrf()?": RRF needs each lane's *rank* over
    // the whole result, which no per-row scalar can see — `rank() OVER` can,
    // and `score_fuse` combines what it produces.
    //
    // The corpus is the case RRF exists for: an unbounded lexical score
    // (BM25) beside a bounded one (cosine). Fusing the raw numbers lets `a`'s
    // 20.0 dominate the whole average despite `a` being *last* on the other
    // lane; fusing the ranks puts the consistently-good `b` first.
    let mut graph = DirGraph::new();
    for (id, title, lex, vector) in [
        (1u32, "a", 20.0, 0.10),
        (2, "b", 2.0, 0.95),
        (3, "c", 1.5, 0.90),
        (4, "d", 1.0, 0.99),
    ] {
        let node = NodeData::new(
            Value::UniqueId(id),
            Value::String(title.to_string()),
            "Doc".to_string(),
            HashMap::from([
                ("lex".to_string(), Value::Float64(lex)),
                ("vec".to_string(), Value::Float64(vector)),
            ]),
            &mut graph.interner,
        );
        let idx = graph.graph.add_node(node);
        graph
            .type_indices
            .entry_or_default("Doc".to_string())
            .push(idx);
    }

    let order = |query: &str| -> Vec<String> {
        let parsed = parser::parse_cypher(query).unwrap();
        let params = HashMap::new();
        CypherExecutor::with_params(&graph, &params, None)
            .execute(&parsed)
            .unwrap()
            .rows
            .iter()
            .map(|row| match &row[0] {
                Value::String(t) => t.clone(),
                other => panic!("unexpected row shape: {other:?}"),
            })
            .collect()
    };

    let raw =
        order("MATCH (d:Doc) RETURN d.title AS t, score_fuse(d.lex, d.vec) AS s ORDER BY s DESC");
    assert_eq!(raw[0], "a", "{raw:?}");

    let rrf = order(
        "MATCH (d:Doc) \
         WITH d, rank() OVER (ORDER BY d.lex DESC) AS lex_rank, \
              rank() OVER (ORDER BY d.vec DESC) AS vec_rank \
         RETURN d.title AS t, \
                score_fuse(1.0 / (60 + lex_rank), 1.0 / (60 + vec_rank)) AS s \
         ORDER BY s DESC",
    );
    assert_eq!(rrf[0], "b", "{rrf:?}");
    // Reciprocal rank is convex, so a first-and-last pair (`a`, `d`) still
    // outscores a steady third place: `c` is last, not merely mid-table.
    assert_eq!(rrf[3], "c", "{rrf:?}");
}
