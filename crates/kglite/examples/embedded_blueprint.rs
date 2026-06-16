//! Build a graph programmatically and save it to disk — the
//! "import data into kglite from Rust" path.
//!
//! Demonstrates the `code_tree` builder as the foundation pattern:
//! parse a (very small) Rust directory tree into a `DirGraph`,
//! then query the resulting graph via Cypher.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p kglite --example embedded_blueprint
//! ```

use kglite::api::{build_code_tree, session, Value};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Build a graph from the current crate's source ──────────
    //
    // `build_code_tree(src_dir, verbose, include_tests, save_to,
    // max_loc_per_file)` returns `Arc<DirGraph>` after Phase G.3-pre
    // — no KnowledgeGraph wrapping. Same engine the Python
    // `kglite.code_tree.build(...)` pyfunction calls.
    let src_dir = std::env::current_dir()?.join("crates/kglite/src");
    if !src_dir.exists() {
        eprintln!(
            "warning: expected to run from workspace root; src dir at {:?} not found",
            src_dir
        );
        return Ok(());
    }
    println!("Parsing {:?} ...", src_dir);
    let graph = build_code_tree(
        &src_dir, /* verbose = */ false, /* include_tests = */ true, None, None,
        /* include_docs = */ false,
    )?;

    // ── 2. Inspect the graph schema via Cypher ────────────────────
    let params = HashMap::new();
    let opts = session::ExecuteOptions::eager(&params);

    let outcome = session::execute_read(
        &graph,
        "MATCH (n) RETURN labels(n) AS type, count(n) AS n ORDER BY n DESC",
        &opts,
    )?;
    println!("\nNode counts by type:");
    for row in &outcome.result.rows {
        let type_name = row.first().map(|v| format!("{:?}", v)).unwrap_or_default();
        let count = match row.get(1) {
            Some(Value::Int64(n)) => *n,
            _ => 0,
        };
        println!("  {:6}  {}", count, type_name);
    }

    // ── 3. Find the top 5 functions by signature length (a proxy
    //      for "complex API") ────────────────────────────────────
    let outcome = session::execute_read(
        &graph,
        "MATCH (f:Function) \
         WHERE f.signature IS NOT NULL \
         RETURN f.title AS name, f.signature AS sig \
         ORDER BY size(f.signature) DESC \
         LIMIT 5",
        &opts,
    )?;
    println!("\nTop 5 functions by signature length:");
    for row in &outcome.result.rows {
        let name = row
            .first()
            .and_then(|v| {
                if let Value::String(s) = v {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .unwrap_or("?");
        let sig_len = row
            .get(1)
            .and_then(|v| {
                if let Value::String(s) = v {
                    Some(s.len())
                } else {
                    None
                }
            })
            .unwrap_or(0);
        println!("  ({:4} chars)  {}", sig_len, name);
    }

    Ok(())
}
