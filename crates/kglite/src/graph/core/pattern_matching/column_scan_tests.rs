//! The column kernel against the generic fused scan: every decline-matrix row
//! that can be built through the public write paths is constructed, and for
//! each one the answer with the pass on must equal the answer with
//! `columnar_scan_aggregate` disabled, with the meters naming which route ran.

use super::column_scan::{kernel_answers, kernel_declines, reset_kernel_meters, Decline};
use crate::datatypes::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::session::{execute_mut, execute_read, ExecuteOptions, Session};
use std::collections::{HashMap, HashSet};

const OFF: &str = "columnar_scan_aggregate";

fn write(graph: &mut DirGraph, query: &str) {
    execute_mut(graph, query, &ExecuteOptions::eager(&HashMap::new()))
        .unwrap_or_else(|e| panic!("{query}: {e}"));
}

fn rows_with(graph: &DirGraph, query: &str, disabled: &HashSet<String>) -> Vec<Vec<Value>> {
    let params = HashMap::new();
    let mut opts = ExecuteOptions::eager(&params);
    opts.disabled_passes = Some(disabled);
    execute_read(graph, query, &opts)
        .unwrap_or_else(|e| panic!("{query}: {e}"))
        .result
        .rows
}

/// What a run did: its rows, and what the meters read around it.
struct Run {
    rows: Vec<Vec<Value>>,
    answered: usize,
    declined: Vec<Decline>,
}

/// Run `query` with the pass on, then off; the rows must be identical, and the
/// off run must never touch the kernel.
fn both(graph: &DirGraph, query: &str) -> Run {
    reset_kernel_meters();
    let on = rows_with(graph, query, &HashSet::new());
    let (answered, declined) = (kernel_answers(), kernel_declines());
    reset_kernel_meters();
    let off = rows_with(graph, query, &HashSet::from([OFF.to_string()]));
    // Debug text, not `==`: it tells NaN apart from nothing and -0.0 from 0.0.
    assert_eq!(
        format!("{on:?}"),
        format!("{off:?}"),
        "kernel diverged from the generic scan on {query}"
    );
    assert_eq!(kernel_answers(), 0, "the pass was off but the kernel ran");
    assert!(
        kernel_declines().is_empty(),
        "the pass was off but a decline was recorded"
    );
    Run {
        rows: on,
        answered,
        declined,
    }
}

fn int(run: &Run) -> i64 {
    match &run.rows[0][0] {
        Value::Int64(i) => *i,
        other => panic!("{other:?}"),
    }
}

const COUNT_SHAPES: &[&str] = &[
    "MATCH (n:Item) WHERE n.score > 10 RETURN count(n) AS c",
    "MATCH (n:Item) WHERE n.score >= 10 AND n.score < 40 RETURN count(*) AS c",
    "MATCH (n:Item) WHERE n.ratio <= 12.5 RETURN count(n) AS c",
    "MATCH (n:Item) WHERE n.score <= 30 AND n.ratio > 2.5 RETURN count(n.score) AS c",
    "MATCH (n:Item {score: 20}) RETURN count(n) AS c",
    "MATCH (n:Item) WHERE n.score > 0 RETURN count(n) AS a, count(n.rank) AS b",
];

fn seeded(rows: i64) -> DirGraph {
    let mut graph = DirGraph::new();
    write(
        &mut graph,
        &format!(
            "UNWIND range(0, {}) AS i CREATE (:Item {{id: i, score: i, ratio: i / 2.0, rank: i * 2, tag: 'x'}})",
            rows - 1
        ),
    );
    graph
}

/// Every count shape must be answered by the kernel on `graph`, equal to the
/// generic scan; returns the answers.
fn assert_kernel_answers_all(graph: &DirGraph) -> Vec<Vec<Value>> {
    COUNT_SHAPES
        .iter()
        .map(|q| {
            let run = both(graph, q);
            assert_eq!(
                run.answered, 1,
                "kernel did not answer {q}: {:?}",
                run.declined
            );
            assert!(run.declined.is_empty(), "{q}: {:?}", run.declined);
            run.rows[0].clone()
        })
        .collect()
}

#[test]
fn a_fresh_store_is_answered_by_the_kernel() {
    let graph = seeded(60);
    let answers = assert_kernel_answers_all(&graph);
    // Hand-computed: score > 10 over 0..=59 is 49 rows.
    assert_eq!(answers[0], vec![Value::Int64(49)]);
    assert_eq!(answers[1], vec![Value::Int64(30)]);
    assert_eq!(answers[4], vec![Value::Int64(1)]);
}

#[test]
fn tombstones_and_delete_then_create_are_answered_and_correct() {
    let mut graph = seeded(60);
    write(&mut graph, "MATCH (n:Item) WHERE n.score >= 50 DELETE n");
    let after_delete = assert_kernel_answers_all(&graph);
    assert_eq!(after_delete[0], vec![Value::Int64(39)]);

    // New rows land past the tombstones; the bucket is no longer in row order.
    write(
        &mut graph,
        "CREATE (:Item {id: 900, score: 55, ratio: 1.0, tag: 'y'}), \
         (:Item {id: 901, score: 12, ratio: 99.0, tag: 'y'})",
    );
    let after_create = assert_kernel_answers_all(&graph);
    assert_eq!(after_create[0], vec![Value::Int64(41)]);
    assert_eq!(
        both(&graph, COUNT_SHAPES[0]).rows[0],
        vec![Value::Int64(41)]
    );
}

#[test]
fn null_cells_never_count_and_count_of_a_property_skips_them() {
    let mut graph = seeded(10);
    write(&mut graph, "CREATE (:Item {id: 100, tag: 'no-numbers'})");
    // The new node has no `score` (never matches) and no `rank`.
    let run = both(
        &graph,
        "MATCH (n:Item) WHERE n.score >= 0 RETURN count(n) AS a, count(n.rank) AS b",
    );
    assert_eq!(run.answered, 1);
    assert_eq!(run.rows[0], vec![Value::Int64(10), Value::Int64(10)]);
}

#[test]
fn a_heap_tail_declines() {
    // Above the store's heap-tail floor (16_384 rows); the has_heap_tail
    // assertion below fails loudly if the floor ever moves past this.
    let rows = 17_500;
    let session = Session::new(seeded(rows));
    let mut tx = session.begin();
    let working = tx.working_mut().unwrap();
    write(
        working,
        "CREATE (:Item {id: 5000000, score: 7, ratio: 1.0, tag: 't'})",
    );
    assert!(working.column_store("Item").unwrap().has_heap_tail());
    let q = "MATCH (n:Item) WHERE n.score < 8 RETURN count(n) AS c";
    let run = both(working, q);
    assert_eq!(run.answered, 0);
    assert_eq!(run.declined, vec![Decline::HeapTail]);
    assert_eq!(int(&run), 9, "scores 0..=7 plus the tail row");
}

#[test]
fn a_mixed_column_declines() {
    let mut graph = DirGraph::new();
    write(
        &mut graph,
        "CREATE (:M {id: 1, x: 1}), (:M {id: 2, x: 2.5}), (:M {id: 3, x: 4})",
    );
    let run = both(&graph, "MATCH (n:M) WHERE n.x > 1 RETURN count(n) AS c");
    assert_eq!(run.answered, 0);
    assert_eq!(run.declined, vec![Decline::NonNumericColumn]);
    assert_eq!(int(&run), 2);
}

#[test]
fn secondary_label_carriers_decline() {
    let mut graph = DirGraph::new();
    write(
        &mut graph,
        "CREATE (:A {id: 1, x: 5}), (:A {id: 2, x: 1}), (:B {id: 3, x: 9})",
    );
    write(&mut graph, "MATCH (n:B {id: 3}) SET n:A");
    let run = both(&graph, "MATCH (n:A) WHERE n.x > 2 RETURN count(n) AS c");
    assert_eq!(run.answered, 0);
    assert_eq!(run.declined, vec![Decline::SecondaryLabel]);
    assert_eq!(int(&run), 2, "the B node carrying :A is a candidate");
}

#[test]
fn a_declared_valid_time_type_declines_and_an_undeclared_one_does_not() {
    let mut graph = DirGraph::new();
    write(
        &mut graph,
        "CREATE (:Well {id: 1, x: 5, vf: '2000-01-01', vt: '2010-01-01'}), \
         (:Well {id: 2, x: 9, vf: '2005-01-01', vt: null}), \
         (:Plain {id: 1, x: 5}), (:Plain {id: 2, x: 9})",
    );
    write(
        &mut graph,
        "CALL db.temporal.declare({node: 'Well', from: 'vf', to: 'vt', convention: 'closed'}) \
         YIELD declared RETURN declared",
    );
    // At 2012 the first well has ended, so the filter hides something.
    let as_of = "FOR VALID_TIME AS OF date('2012-01-01') ";
    let run = both(
        &graph,
        &format!("{as_of}MATCH (n:Well) WHERE n.x > 1 RETURN count(n) AS c"),
    );
    assert_eq!(run.answered, 0);
    assert_eq!(run.declined, vec![Decline::ValidTime]);
    assert_eq!(int(&run), 1, "only the open-ended well is valid in 2012");
    let run = both(
        &graph,
        &format!("{as_of}MATCH (n:Plain) WHERE n.x > 1 RETURN count(n) AS c"),
    );
    assert_eq!(run.answered, 1, "{:?}", run.declined);
    assert_eq!(int(&run), 2);
}

#[test]
fn parallel_requested_declines() {
    let graph = seeded(40);
    let params = HashMap::new();
    let mut opts = ExecuteOptions::eager(&params);
    opts.parallel = true;
    reset_kernel_meters();
    let out = execute_read(&graph, COUNT_SHAPES[0], &opts).unwrap();
    assert_eq!(out.result.rows[0], vec![Value::Int64(29)]);
    assert_eq!(kernel_answers(), 0);
    assert_eq!(kernel_declines(), vec![Decline::Parallel]);
}

#[test]
fn a_row_without_an_owning_node_declines_on_cardinality() {
    let mut graph = seeded(30);
    // Tombstone a row while its node stays in the bucket: live rows (29) no
    // longer equal the bucket (30).
    let store = graph.column_store_mut("Item").unwrap();
    std::sync::Arc::make_mut(store).tombstone(3);
    reset_kernel_meters();
    let on = rows_with(&graph, COUNT_SHAPES[0], &HashSet::new());
    assert_eq!(kernel_answers(), 0);
    assert_eq!(kernel_declines(), vec![Decline::Cardinality]);
    let off = rows_with(&graph, COUNT_SHAPES[0], &HashSet::from([OFF.to_string()]));
    assert_eq!(on, off);
}

#[test]
fn an_overflow_bag_declines() {
    use crate::graph::storage::mapped::mmap_vec::{MmapBytes, MmapOrVec};
    let mut graph = seeded(20);
    let rows = graph.column_store("Item").unwrap().row_count() as usize;
    let store = std::sync::Arc::make_mut(graph.column_store_mut("Item").unwrap());
    store.replace_overflow_bag(MmapOrVec::from_vec(vec![0u64; rows + 1]), MmapBytes::new());
    assert!(store.has_overflow());
    let run = both(&graph, COUNT_SHAPES[0]);
    assert_eq!(run.answered, 0);
    assert_eq!(run.declined, vec![Decline::Overflow]);
    assert_eq!(int(&run), 9);
}

#[test]
fn a_displaced_column_log_declines() {
    let mut graph = seeded(20);
    let store = std::sync::Arc::make_mut(graph.column_store_mut("Item").unwrap());
    store.begin_displaced_log();
    // A string into an Int64 column displaces it while the undo log records.
    write(&mut graph, "MATCH (n:Item {id: 3}) SET n.ratio = 'text'");
    let store = graph.column_store("Item").unwrap();
    assert!(store.has_displaced(), "the type change was not logged");
    let run = both(
        &graph,
        "MATCH (n:Item) WHERE n.score > 5 RETURN count(n) AS c",
    );
    assert_eq!(run.answered, 0);
    assert_eq!(run.declined, vec![Decline::Displaced]);
    assert_eq!(int(&run), 14);
}

#[test]
fn shapes_outside_the_kernel_are_not_marked() {
    let graph = seeded(20);
    for q in [
        "MATCH (n:Item) WHERE n.tag = 'x' RETURN count(n) AS c",
        "MATCH (n:Item) WHERE n.score > 3 OR n.score < 1 RETURN count(n) AS c",
        "MATCH (n:Item) WHERE n.score > 3 RETURN sum(DISTINCT n.score) AS s",
        "MATCH (n:Item) WHERE n.score > 3 RETURN sum(n) AS s",
        "MATCH (n:Item) WHERE n.score > 3 RETURN max(n.score + 1) AS s",
        "MATCH (n:Item) WHERE n.score > 3 RETURN collect(n.score) AS s",
        "MATCH (n:Item) WHERE n.score > 3 RETURN sum(n.score) AS s, collect(n.rank) AS r",
        "MATCH (n:Item) WHERE n.score > 3 RETURN n.tag AS t, count(n) AS c",
        "MATCH (n:Item) WHERE n.score > 3 RETURN count(DISTINCT n.score) AS c",
        "MATCH (n:Item) WHERE n.id > 3 RETURN count(n) AS c",
        "MATCH (n:Item) WHERE n.score IN [1, 2, 3] RETURN count(n) AS c",
    ] {
        let run = both(&graph, q);
        assert_eq!((run.answered, run.declined.len()), (0, 0), "{q}");
    }
}

#[test]
fn a_float_count_target_declines() {
    let graph = seeded(20);
    let run = both(
        &graph,
        "MATCH (n:Item) WHERE n.score > 5 RETURN count(n.ratio) AS c",
    );
    assert_eq!(run.answered, 0);
    assert_eq!(run.declined, vec![Decline::FloatCountTarget]);
    assert_eq!(int(&run), 14);
}

#[test]
fn an_indexed_predicate_declines_so_the_index_still_serves_it() {
    let mut graph = seeded(40);
    write(&mut graph, "CREATE INDEX FOR (n:Item) ON (n.score)");
    let run = both(&graph, "MATCH (n:Item {score: 20}) RETURN count(n) AS c");
    assert_eq!(run.answered, 0);
    assert_eq!(run.declined, vec![Decline::Indexed]);
    assert_eq!(int(&run), 1);
    // A column without an index is still the kernel's.
    let run = both(
        &graph,
        "MATCH (n:Item) WHERE n.rank > 10 RETURN count(n) AS c",
    );
    assert_eq!(run.answered, 1);
}

// --- K2: sum, avg, min, max ---------------------------------------------------

const AGG_SHAPES: &[&str] = &[
    "MATCH (n:Item) WHERE n.score > 10 \
     RETURN count(n) AS c, sum(n.score) AS s, min(n.score) AS lo, max(n.score) AS hi",
    "MATCH (n:Item) WHERE n.score >= 10 AND n.ratio < 20.5 \
     RETURN sum(n.rank) AS s, avg(n.rank) AS a, min(n.ratio) AS lo, max(n.ratio) AS hi",
    "MATCH (n:Item) WHERE n.score > 0 RETURN avg(n.score) AS a, max(n.rank) AS m, count(n.rank) AS k",
    "MATCH (n:Item {score: 20}) RETURN sum(n.score) AS s, min(n.ratio) AS lo",
    "MATCH (n:Item) RETURN count(n) AS c, sum(n.rank) AS s, min(n.rank) AS lo, max(n.score) AS hi",
    "MATCH (n:Item) WHERE n.score > 1000 \
     RETURN count(n) AS c, sum(n.score) AS s, min(n.score) AS lo, max(n.ratio) AS hi, avg(n.rank) AS a",
];

fn assert_agg_shapes_answered(graph: &DirGraph) -> Vec<Vec<Value>> {
    AGG_SHAPES
        .iter()
        .map(|q| {
            let run = both(graph, q);
            assert_eq!(
                run.answered, 1,
                "kernel did not answer {q}: {:?}",
                run.declined
            );
            assert!(run.declined.is_empty(), "{q}: {:?}", run.declined);
            run.rows[0].clone()
        })
        .collect()
}

#[test]
fn multi_aggregate_scans_are_answered_and_correct() {
    let graph = seeded(60);
    let answers = assert_agg_shapes_answered(&graph);
    // score 11..=59: 49 rows, sum 1715.
    assert_eq!(
        answers[0],
        vec![
            Value::Int64(49),
            Value::Int64(1715),
            Value::Int64(11),
            Value::Int64(59)
        ]
    );
    // Empty selection: count 0, sum 0, everything else null.
    assert_eq!(
        answers[5],
        vec![
            Value::Int64(0),
            Value::Int64(0),
            Value::Null,
            Value::Null,
            Value::Null
        ]
    );
    // 60 rows, rank = 2 * score: sum 3540, min 0, max score 59.
    assert_eq!(
        answers[4],
        vec![
            Value::Int64(60),
            Value::Int64(3540),
            Value::Int64(0),
            Value::Int64(59)
        ]
    );
}

#[test]
fn aggregates_survive_tombstones_creates_and_null_cells() {
    let mut graph = seeded(60);
    write(&mut graph, "MATCH (n:Item) WHERE n.score >= 50 DELETE n");
    assert_agg_shapes_answered(&graph);
    // Two rows land past the tombstones; neither has a `rank`.
    write(
        &mut graph,
        "CREATE (:Item {id: 900, score: 55, ratio: 1.0, tag: 'y'}), \
         (:Item {id: 901, score: 12, ratio: 99.0, tag: 'y'})",
    );
    let answers = assert_agg_shapes_answered(&graph);
    // score 11..=49 plus 55 and 12: 41 rows, sum 1170 + 67.
    assert_eq!(
        answers[0],
        vec![
            Value::Int64(41),
            Value::Int64(1237),
            Value::Int64(11),
            Value::Int64(55)
        ]
    );
    // rank is null on the new rows: the rank aggregates skip them.
    assert_eq!(answers[4][1], Value::Int64(2 * (0..50).sum::<i64>()));
}

#[test]
fn float_sum_and_avg_decline_and_take_every_aggregate_with_them() {
    let graph = seeded(40);
    for q in [
        "MATCH (n:Item) WHERE n.score > 5 RETURN sum(n.ratio) AS s",
        "MATCH (n:Item) WHERE n.score > 5 RETURN avg(n.ratio) AS a",
        "MATCH (n:Item) WHERE n.score > 5 RETURN count(n) AS c, sum(n.score) AS s, avg(n.ratio) AS a",
    ] {
        let run = both(&graph, q);
        assert_eq!(run.answered, 0, "{q}");
        assert_eq!(run.declined, vec![Decline::FloatOrder], "{q}");
    }
}

#[test]
fn a_zero_float_extreme_declines_but_a_nonzero_one_is_served() {
    let graph = seeded(20);
    // ratio = score / 2.0, so the unfiltered minimum is 0.0.
    let run = both(&graph, "MATCH (n:Item) RETURN min(n.ratio) AS lo");
    assert_eq!((run.answered, run.declined), (0, vec![Decline::SignedZero]));
    let run = both(&graph, "MATCH (n:Item) RETURN max(n.ratio) AS hi");
    assert_eq!(run.answered, 1);
    assert_eq!(run.rows[0], vec![Value::Float64(9.5)]);
    let run = both(
        &graph,
        "MATCH (n:Item) WHERE n.score > 0 RETURN min(n.ratio) AS lo",
    );
    assert_eq!(run.answered, 1);
    assert_eq!(run.rows[0], vec![Value::Float64(0.5)]);
}

#[test]
fn an_integer_average_past_f64_precision_declines() {
    let mut graph = DirGraph::new();
    write(
        &mut graph,
        "CREATE (:Big {id: 1, v: 1152921504606846976}), (:Big {id: 2, v: 1152921504606846977}), \
         (:Big {id: 3, v: 5})",
    );
    let run = both(&graph, "MATCH (n:Big) RETURN avg(n.v) AS a");
    assert_eq!((run.answered, run.declined), (0, vec![Decline::FloatOrder]));
    // The same column sums exactly, so sum/min/max are served.
    let run = both(
        &graph,
        "MATCH (n:Big) RETURN sum(n.v) AS s, min(n.v) AS lo, max(n.v) AS hi",
    );
    assert_eq!(run.answered, 1);
    assert_eq!(
        run.rows[0],
        vec![
            Value::Int64(2_305_843_009_213_693_958),
            Value::Int64(5),
            Value::Int64(1_152_921_504_606_846_977)
        ]
    );
}

#[test]
fn an_integer_sum_is_exact_across_cancelling_extremes_and_raises_on_overflow() {
    let mut graph = DirGraph::new();
    write(
        &mut graph,
        "CREATE (:W {id: 1, v: 9223372036854775807}), (:W {id: 2, v: 5}), \
         (:W {id: 3, v: -9223372036854775807})",
    );
    let run = both(&graph, "MATCH (n:W) RETURN sum(n.v) AS s");
    assert_eq!(run.answered, 1);
    assert_eq!(run.rows[0], vec![Value::Int64(5)]);

    write(&mut graph, "CREATE (:W {id: 4, v: 9223372036854775807})");
    let q = "MATCH (n:W) RETURN sum(n.v) AS s";
    let params = HashMap::new();
    reset_kernel_meters();
    let on = execute_read(&graph, q, &ExecuteOptions::eager(&params));
    assert_eq!(
        kernel_answers(),
        1,
        "the overflow must come from the kernel"
    );
    let mut off_opts = ExecuteOptions::eager(&params);
    let off_set = HashSet::from([OFF.to_string()]);
    off_opts.disabled_passes = Some(&off_set);
    let off = execute_read(&graph, q, &off_opts);
    for result in [
        on.err().map(|e| e.to_string()),
        off.err().map(|e| e.to_string()),
    ] {
        let message = result.expect("a sum outside the i64 range must raise");
        assert!(message.contains("Integer overflow in sum"), "{message}");
    }
}

#[test]
fn nan_cells_follow_the_total_order() {
    let mut graph = DirGraph::new();
    write(
        &mut graph,
        "CREATE (:Q {id: 1, k: 1, f: 1.5}), (:Q {id: 2, k: 2, f: 2.5}), (:Q {id: 3, k: 3, f: 4.0})",
    );
    let nan = HashMap::from([("x".to_string(), Value::Float64(f64::NAN))]);
    execute_mut(
        &mut graph,
        "MATCH (n:Q {id: 2}) SET n.f = $x",
        &ExecuteOptions::eager(&nan),
    )
    .unwrap();
    let run = both(&graph, "MATCH (n:Q) RETURN min(n.f) AS lo, max(n.f) AS hi");
    assert_eq!(run.answered, 1, "{:?}", run.declined);
    assert_eq!(run.rows[0][0], Value::Float64(1.5), "min ignores NaN");
    assert!(matches!(run.rows[0][1], Value::Float64(f) if f.is_nan()));
    // Only NaN left: min is NaN too.
    let run = both(
        &graph,
        "MATCH (n:Q) WHERE n.k = 2 RETURN min(n.f) AS lo, max(n.f) AS hi",
    );
    assert_eq!(run.answered, 1);
    assert!(run.rows[0]
        .iter()
        .all(|v| matches!(v, Value::Float64(f) if f.is_nan())));
}

#[test]
fn aggregates_over_a_non_numeric_property_decline() {
    let graph = seeded(20);
    let run = both(
        &graph,
        "MATCH (n:Item) WHERE n.score > 3 RETURN min(n.tag) AS t",
    );
    assert_eq!(
        (run.answered, run.declined),
        (0, vec![Decline::NonNumericColumn])
    );
}
