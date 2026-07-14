//! code_tree accuracy harness — the measurement substrate for the
//! re-resolution phases.
//!
//! Builds the code graph for a repo and reports, as a single JSON object:
//! build wall-time, node/edge counts, and the CALLS-resolution breakdown
//! (`total_calls` → `excluded_noise` / `no_candidate` / `ambiguous_dropped`
//! / `resolved_call_sites`, plus the de-duplicated `resolved_edges`).
//!
//! The headline metric is `resolution_rate` = `resolved_call_sites` /
//! (`total_calls` - `excluded_noise`): of every call site that wasn't stdlib
//! noise, the fraction we pinned to at least one in-project symbol. Track it
//! across phases; re-resolution should push it up without moving build-time
//! on the default path.
//!
//! Usage:
//!   cargo run -p kglite --bin code_tree_stats --release -- <path>
//!   cargo run -p kglite --bin code_tree_stats --release -- <path> --include-tests
//!   cargo run -p kglite --bin code_tree_stats --release -- <path> --function-metrics

use kglite::api::GraphRead;
use kglite::api::{session, Value};
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

#[derive(Serialize)]
struct FunctionMetric {
    path: String,
    qualified_name: String,
    start_line: i64,
    end_line: i64,
    branch_count: i64,
    max_nesting: i64,
    is_test: bool,
}

fn string_value(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Null => String::new(),
        other => panic!("expected string metric, got {other:?}"),
    }
}

fn int_value(value: &Value) -> i64 {
    match value {
        Value::Int64(value) => *value,
        Value::UniqueId(value) => i64::from(*value),
        Value::Null => 0,
        other => panic!("expected integer metric, got {other:?}"),
    }
}

fn bool_value(value: &Value) -> bool {
    match value {
        Value::Boolean(value) => *value,
        Value::Null => false,
        other => panic!("expected boolean metric, got {other:?}"),
    }
}

fn function_metrics(graph: &kglite::api::DirGraph) -> Vec<FunctionMetric> {
    let params = HashMap::new();
    let options = session::ExecuteOptions::eager(&params);
    let query = "MATCH (f:Function) RETURN f.file_path, f.qualified_name, \
                 f.line_number, f.end_line, f.branch_count, f.max_nesting, f.is_test";
    let outcome = session::execute_read(graph, query, &options).expect("query function metrics");
    let mut metrics: Vec<_> = outcome
        .result
        .rows
        .iter()
        .map(|row| FunctionMetric {
            path: string_value(&row[0]),
            qualified_name: string_value(&row[1]),
            start_line: int_value(&row[2]),
            end_line: int_value(&row[3]),
            branch_count: int_value(&row[4]),
            max_nesting: int_value(&row[5]),
            is_test: bool_value(&row[6]),
        })
        .collect();
    metrics.sort_by(|a, b| {
        (&a.path, &a.qualified_name, a.start_line).cmp(&(&b.path, &b.qualified_name, b.start_line))
    });
    metrics
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: code_tree_stats <path> [--include-tests]");
            std::process::exit(2);
        }
    };
    let options: Vec<_> = args.collect();
    let include_tests = options.iter().any(|a| a == "--include-tests");
    let emit_function_metrics = options.iter().any(|a| a == "--function-metrics");

    let t = Instant::now();
    let (graph, stats) = match kglite::code_tree::builder::run_with_options_stats(
        Path::new(&path),
        false,
        include_tests,
        None,
        None,
        false,
    ) {
        Ok(pair) => pair,
        Err(e) => {
            eprintln!("build failed: {e}");
            std::process::exit(1);
        }
    };
    let build_secs = t.elapsed().as_secs_f64();

    if emit_function_metrics {
        println!(
            "{}",
            serde_json::to_string_pretty(&function_metrics(&graph)).unwrap()
        );
        return;
    }

    let denom = stats.total_calls.saturating_sub(stats.excluded_noise);
    let resolution_rate = if denom > 0 {
        stats.resolved_call_sites as f64 / denom as f64
    } else {
        0.0
    };

    let out = serde_json::json!({
        "path": path,
        "include_tests": include_tests,
        "build_secs": (build_secs * 1000.0).round() / 1000.0,
        "nodes": graph.graph.node_count(),
        "edges": graph.graph.edge_count(),
        "total_calls": stats.total_calls,
        "excluded_noise": stats.excluded_noise,
        "no_candidate": stats.no_candidate,
        "ambiguous_dropped": stats.ambiguous_dropped,
        "resolved_call_sites": stats.resolved_call_sites,
        "resolved_via_inheritance": stats.resolved_via_inheritance,
        "resolved_edges": stats.resolved_edges,
        "resolution_rate": (resolution_rate * 10000.0).round() / 10000.0,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
