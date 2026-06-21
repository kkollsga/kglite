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

use kglite::api::GraphRead;
use std::path::Path;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: code_tree_stats <path> [--include-tests]");
            std::process::exit(2);
        }
    };
    let include_tests = args.any(|a| a == "--include-tests");

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
