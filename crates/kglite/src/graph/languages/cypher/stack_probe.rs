//! How much stack does one level of AST nesting actually cost, and does the
//! deepest query the parser accepts still fit the stack it runs on?
//!
//! `MAX_EXPRESSION_DEPTH` bounds AST *depth*, and only the parser grows its
//! stack on demand (`stacker`). The planner's expression walkers, the
//! executor's `evaluate_expression` and the AST's recursive `Drop` run on
//! whatever stack the caller owns, so "at most 512 levels" only becomes
//! "cannot overflow" once someone measures the bytes per level. This module
//! is that measurement, plus the regression test that keeps the answer true.
//!
//! Two things live here:
//!
//! 1. [`budget_ceiling_query_fits_the_query_thread_stack`] — an always-on
//!    test. It runs the deepest accepted query end-to-end on a thread sized
//!    exactly [`QUERY_THREAD_STACK_SIZE`], which is what the Bolt and MCP
//!    servers give their query threads and what the CLI and Python wheel get
//!    from the main thread. No `unsafe`, no platform assumptions: it either
//!    completes or the process aborts, and an abort is the finding.
//! 2. [`stack_probe`] — an exploratory harness, inert unless
//!    `KGL_STACK_PROBE` is set, that reports the exact bytes a chosen stage
//!    consumes at a chosen depth. Re-run it when frame costs might have moved.
//!
//! # Method used by the exploratory harness
//!
//! Paint the unused stack below the current stack pointer with a known byte
//! pattern, run the stage, then scan upward from the bottom for the first
//! word that no longer matches — that address is the deepest point the stage
//! reached. This yields the real figure in one run without ever overflowing
//! anything, so there is no crash-and-binary-search loop.
//!
//! Painting walks *downward one page at a time*: Windows commits stack pages
//! through a guard page that must be touched in descending order, and a
//! straight ascending `memset` from the bottom of the window would fault
//! there. The numbers on record were taken on macOS/aarch64.
//!
//! ```text
//! KGL_STACK_PROBE=1 KGL_PROBE_STAGE=exec KGL_PROBE_SHAPE=or \
//! KGL_PROBE_DEPTH=408 <lib-test-binary> \
//!     --exact graph::languages::cypher::stack_probe::stack_probe --nocapture
//! ```
//!
//! Stages: `parse`, `plan`, `exec`, `drop`, `full`, plus `calibrate` (a
//! recursion of known frame size, to prove the method) and `bench` (a
//! release-profile timing of the per-row expression path, for costing any
//! proposed guard). `KGL_PROBE_STACK_KIB` sizes the measured thread.

use super::ast::CypherQuery;
use super::executor::CypherExecutor;
use super::parser::{parse_cypher, MAX_EXPRESSION_DEPTH};
use super::planner::optimize;
use crate::datatypes::Value;
use crate::graph::algorithms::Interrupt;
use crate::graph::dir_graph::DirGraph;
use crate::graph::session::QUERY_THREAD_STACK_SIZE;
use std::collections::HashMap;

/// Stack for the preparation threads, which are never measured.
const PREP_STACK: usize = 512 * 1024 * 1024;
const PAINT: u8 = 0xA5;
const PAINT_WORD: u64 = u64::from_ne_bytes([PAINT; 8]);
const PAGE: usize = 4096;

fn probe_stack() -> usize {
    std::env::var("KGL_PROBE_STACK_KIB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|kib| kib * 1024)
        .unwrap_or(512 * 1024 * 1024)
}

fn paint_len() -> usize {
    (probe_stack() / 4 * 3).min(64 * 1024 * 1024) / PAGE * PAGE
}

/// Query text for `shape` producing `depth` levels of AST nesting.
fn query(shape: &str, depth: usize) -> String {
    match shape {
        // Recursively parsed shapes.
        "parens" => format!("RETURN {}1{} AS x", "(".repeat(depth), ")".repeat(depth)),
        "lists" => format!("RETURN {}1{} AS x", "[".repeat(depth), "]".repeat(depth)),
        "not" => format!("RETURN {}false AS x", "NOT ".repeat(depth)),
        "neg" => format!("RETURN {}5 AS x", "-".repeat(depth)),
        // Iteratively parsed left-associative chains (one AST level each).
        "or" => format!("RETURN {} AS x", vec!["false"; depth + 1].join(" OR ")),
        "and" => format!("RETURN {} AS x", vec!["true"; depth + 1].join(" AND ")),
        "add" => format!("RETURN {} AS x", vec!["1"; depth + 1].join(" + ")),
        "concat" => format!("RETURN {} AS x", vec!["'a'"; depth + 1].join(" || ")),
        "subscript" => format!("RETURN [1]{} AS x", "[0]".repeat(depth)),
        // Distinct values on purpose: identical disjuncts get collapsed, which
        // would hide the recursion this exists to measure. This is also the
        // literal shape a filter/facet builder emits for a multi-select — and
        // the planner's `fold_or_to_in` pass rewrites it to `IN [...]`, so the
        // executor never sees the chain.
        "where_or" => format!(
            "MATCH (n:T) WHERE {} RETURN count(n) AS c",
            (0..=depth)
                .map(|i| format!("n.id = {i}"))
                .collect::<Vec<_>>()
                .join(" OR ")
        ),
        // ORs across *different* properties: `fold_or_to_in` only folds
        // same-property equalities, so this shape survives into the executor
        // as a genuine one-level-per-term predicate tree.
        "where_or_mixed" => format!(
            "MATCH (n:T) WHERE {} RETURN count(n) AS c",
            (0..=depth)
                .map(|i| format!("n.p{i} = {i}"))
                .collect::<Vec<_>>()
                .join(" OR ")
        ),
        // The rewrite a user should reach for instead: one AST level no
        // matter how many values the list holds.
        "where_in" => format!(
            "MATCH (n:T) WHERE n.id IN [{}] RETURN count(n) AS c",
            (0..=depth)
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => panic!("unknown shape {other}"),
    }
}

/// Every nesting shape under test. The always-on stack test walks this list,
/// so a new nesting construct gets stack coverage by being added in one place.
const ALL_SHAPES: &[&str] = &[
    "or",
    "and",
    "not",
    "neg",
    "lists",
    "add",
    "concat",
    "subscript",
    "parens",
    "where_or",
    "where_or_mixed",
    "where_in",
];

/// A one-node `:T` graph, so `MATCH (n:T) WHERE …` actually produces a row
/// and the executor really walks the predicate tree. On an empty graph the
/// WHERE shapes measure zero — no rows, no evaluation.
fn seeded_graph() -> DirGraph {
    let mut graph = DirGraph::new();
    let create = parse_cypher("CREATE (:T {id: 1})").expect("seed parses");
    super::executor::write::execute_mutable(
        &mut graph,
        &create,
        HashMap::new(),
        Interrupt::default(),
    )
    .expect("seed executes");
    graph
}

/// Run the whole pipeline — parse, plan, execute, and drop the AST.
fn run_full_pipeline(graph: &DirGraph, text: &str) {
    let params: HashMap<String, Value> = HashMap::new();
    let mut q = parse_cypher(text).expect("query must parse within the budget");
    optimize(&mut q, graph, &params);
    let exec = CypherExecutor::with_params(graph, &params, None);
    exec.execute(&q).expect("query must execute");
}

/// The load-bearing invariant: the deepest AST the parser accepts still fits
/// the stack the engine actually runs queries on.
///
/// [`QUERY_THREAD_STACK_SIZE`] is what the Bolt and MCP servers hand their
/// query threads, and it matches the main-thread default the CLI and the
/// Python wheel get. Only the parser grows its stack on demand; the planner
/// walkers, `evaluate_expression` and the recursive `Drop` do not, so this is
/// the only thing standing between a budget-ceiling query and a process
/// abort. A stack overflow here aborts the test binary rather than failing
/// cleanly — that is the intended signal, and it is why the bound is checked
/// on every platform CI runs rather than reasoned about.
///
/// Measured headroom when this was written (macOS/aarch64, worst shape `or`,
/// executor path): 3.7 MiB of 8 MiB in debug, 0.54 MiB in release.
#[test]
fn budget_ceiling_query_fits_the_query_thread_stack() {
    let depth = MAX_EXPRESSION_DEPTH - 1;
    std::thread::Builder::new()
        .stack_size(QUERY_THREAD_STACK_SIZE)
        .spawn(move || {
            let graph = seeded_graph();
            for shape in ALL_SHAPES {
                run_full_pipeline(&graph, &query(shape, depth));
            }
        })
        .expect("spawn query-sized thread")
        .join()
        .expect("budget-ceiling query overflowed the query-thread stack");
}

/// A query past the budget is refused by the parser, so no downstream walker
/// ever sees it — and the refusal names the rewrite that actually works.
#[test]
fn past_budget_is_refused_with_the_in_rewrite_named() {
    let err = parse_cypher(&query("or", MAX_EXPRESSION_DEPTH + 50))
        .expect_err("past-budget query must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("nesting exceeds"),
        "unexpected error text: {msg}"
    );
    assert!(
        msg.contains("IN ["),
        "the budget error must name the IN [...] rewrite, got: {msg}"
    );
}

// ── Exploratory harness ─────────────────────────────────────────────────────

/// Parse (and optionally plan) on a separate generously-sized thread, so the
/// measured thread only pays for the stage under test.
fn prepared_off_thread(text: String, plan: bool) -> CypherQuery {
    std::thread::Builder::new()
        .stack_size(PREP_STACK)
        .spawn(move || {
            let mut q = parse_cypher(&text).expect("probe query must parse");
            if plan {
                let graph = seeded_graph();
                let params: HashMap<String, Value> = HashMap::new();
                optimize(&mut q, &graph, &params);
            }
            q
        })
        .expect("spawn")
        .join()
        .expect("prep thread")
}

/// Paint the free stack below the current frame, run `body`, then report how
/// far down it reached.
///
/// # Safety
/// Every write lies strictly below the current frame and strictly inside the
/// thread's own reserved stack, and pages are touched in descending order so
/// the Windows guard-page protocol is respected. Rust installs its
/// stack-overflow handler on a `sigaltstack`, so no signal handler runs in
/// the painted window.
fn measure(body: impl FnOnce()) -> usize {
    let anchor = 0u64;
    // `&anchor` is an address *inside* the current frame, not the true stack
    // pointer: the compiler may place other live locals of this same frame
    // (notably `body`) below it, and painting over them corrupts live data.
    // Skip a margin that clears the whole frame. It is a constant offset in
    // every reading, so it cancels in the depth-to-depth deltas that matter.
    const MARGIN: usize = 4 * 1024;
    let reference = ((&anchor as *const u64 as usize) & !7) - MARGIN;
    let len = paint_len();
    let bottom = reference - len;

    let mut page = reference - PAGE;
    while page >= bottom {
        // Descending, one page at a time: that is what Windows requires to
        // commit stack pages through the guard page.
        //
        // SAFETY: `page` walks `[bottom, reference)`, which lies below this
        // frame and inside this thread's own reserved stack. Nothing live is
        // there — the callee frames that will use it do not exist yet, and
        // `MARGIN` keeps the write clear of this frame's own locals.
        unsafe { std::ptr::write_bytes(page as *mut u8, PAINT, PAGE) };
        page -= PAGE;
    }

    body();

    let mut deepest = reference;
    for i in 0..len / 8 {
        let addr = bottom + i * 8;
        // SAFETY: `addr` is an 8-byte-aligned address inside the window just
        // painted, so it is mapped, committed and readable. The read is
        // volatile so the scan is not optimised against the `write_bytes`
        // above, whose effect the compiler cannot otherwise see.
        if unsafe { std::ptr::read_volatile(addr as *const u64) } != PAINT_WORD {
            deepest = addr;
            break;
        }
    }
    assert!(
        deepest > bottom,
        "stage reached the bottom of the paint window; raise KGL_PROBE_STACK_KIB"
    );
    reference - deepest
}

/// Self-check of the measurement method: a recursion whose frame cost is
/// knowable independently, so a wrong number here invalidates the rest.
#[inline(never)]
fn calibration_recurse(depth: usize, sink: &mut u64) {
    let mut pad = [0u64; 16]; // 128 bytes, must not be optimised away
    pad[depth % 16] = depth as u64;
    if depth > 0 {
        calibration_recurse(depth - 1, sink);
    }
    *sink = sink.wrapping_add(std::hint::black_box(pad)[depth % 16]);
}

fn on_probe_thread<R: Send + 'static>(body: impl FnOnce() -> R + Send + 'static) -> R {
    std::thread::Builder::new()
        .stack_size(probe_stack())
        .spawn(body)
        .expect("spawn probe thread")
        .join()
        .expect("probe thread panicked")
}

/// Release-profile timing of the per-row expression path — the cost centre
/// any `stacker::maybe_grow` inside `evaluate_expression` would tax. Reports
/// the minimum over rounds (CLAUDE.md: trust min over median for
/// sub-millisecond work).
fn run_bench() {
    let rounds: usize = std::env::var("KGL_PROBE_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let nodes: usize = std::env::var("KGL_PROBE_NODES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000);
    let mut graph = DirGraph::new();
    for i in 0..nodes {
        let create =
            parse_cypher(&format!("CREATE (:T {{id: {i}, v: {}}})", i * 3)).expect("seed parses");
        super::executor::write::execute_mutable(
            &mut graph,
            &create,
            HashMap::new(),
            Interrupt::default(),
        )
        .expect("seed executes");
    }
    let params: HashMap<String, Value> = HashMap::new();
    // Arithmetic-heavy projection: several `evaluate_expression` calls per
    // row, which is precisely the hot path under discussion.
    let mut q = parse_cypher("MATCH (n:T) RETURN sum(n.id * 2 + n.v * 3 - n.id + 1) AS s")
        .expect("bench query parses");
    optimize(&mut q, &graph, &params);
    let mut best = std::time::Duration::from_secs(3600);
    for _ in 0..rounds {
        let exec = CypherExecutor::with_params(&graph, &params, None);
        let t0 = std::time::Instant::now();
        let r = exec.execute(&q).expect("bench executes");
        let dt = t0.elapsed();
        std::hint::black_box(&r);
        best = best.min(dt);
    }
    println!(
        "PROBE-BENCH nodes={nodes} rounds={rounds} min_us={}",
        best.as_micros()
    );
}

#[test]
fn stack_probe() {
    if std::env::var_os("KGL_STACK_PROBE").is_none() {
        return;
    }
    let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} must be set"));
    let stage = env("KGL_PROBE_STAGE");

    match stage.as_str() {
        "calibrate" => {
            for depth in [1usize, 101, 1101] {
                let used = on_probe_thread(move || {
                    measure(|| {
                        let mut sink = 0u64;
                        calibration_recurse(depth, &mut sink);
                        std::hint::black_box(sink);
                    })
                });
                println!("PROBE-RESULT stage=calibrate shape=none depth={depth} bytes={used}");
            }
            println!("PROBE-OK");
            return;
        }
        "bench" => {
            run_bench();
            println!("PROBE-OK");
            return;
        }
        _ => {}
    }

    let shape = env("KGL_PROBE_SHAPE");
    let depth: usize = env("KGL_PROBE_DEPTH").parse().expect("depth");
    let text = query(&shape, depth);

    let used = match stage.as_str() {
        // Parser only; the AST is leaked so the recursive Drop never runs
        // here. On the huge default probe stack `stacker` never fires, so
        // this reports what the parser *would* need unguarded.
        "parse" => on_probe_thread(move || {
            measure(|| {
                let q = parse_cypher(&text).expect("parse");
                std::mem::forget(q);
            })
        }),
        "drop" => {
            let q = prepared_off_thread(text, false);
            on_probe_thread(move || measure(move || drop(q)))
        }
        "plan" => {
            let q = prepared_off_thread(text, false);
            on_probe_thread(move || {
                let mut q = q;
                let graph = seeded_graph();
                let params: HashMap<String, Value> = HashMap::new();
                let used = measure(|| optimize(&mut q, &graph, &params));
                std::mem::forget(q);
                used
            })
        }
        "exec" => {
            let q = prepared_off_thread(text, true);
            on_probe_thread(move || {
                let graph = seeded_graph();
                let params: HashMap<String, Value> = HashMap::new();
                let used = measure(|| {
                    let exec = CypherExecutor::with_params(&graph, &params, None);
                    let result = exec.execute(&q).expect("execute");
                    std::mem::forget(result);
                });
                std::mem::forget(q);
                used
            })
        }
        // Set KGL_PROBE_STACK_KIB to a realistic thread stack to see the
        // parser's `stacker` guard actually engage.
        "full" => on_probe_thread(move || {
            let graph = seeded_graph();
            measure(|| run_full_pipeline(&graph, &text))
        }),
        other => panic!("unknown stage {other}"),
    };

    println!("PROBE-RESULT stage={stage} shape={shape} depth={depth} bytes={used}");
    println!("PROBE-OK");
}
