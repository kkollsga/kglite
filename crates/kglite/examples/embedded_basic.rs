//! Smallest possible kglite embedder: load a `.kgl` file from disk
//! and run a Cypher query against it. Zero PyO3 in the dep tree.
//!
//! Run with:
//!
//! ```bash
//! # From the workspace root, against any kgl in your environment:
//! cargo run -p kglite --example embedded_basic -- path/to/graph.kgl
//!
//! # Verify the dep tree is pyo3-free:
//! cargo tree -p kglite --example embedded_basic | grep pyo3
//! # → (empty)
//! ```
//!
//! The single .kgl file produced by Python (`kg.save("graph.kgl")`)
//! is the same file this Rust binary reads. The on-disk format is
//! the engine's portable contract; it travels across any kglite
//! binding.

use kglite::api::{load_file, session, Value};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("Usage: embedded_basic <path/to/graph.kgl>")?;

    // ── 1. Load the graph from disk ───────────────────────────────
    //
    // `load_file` returns an `Arc<DirGraph>` — the engine type. No
    // pyo3 wrapping (`KnowledgeGraph` is a pyo3 concern; it lives
    // in the kglite-py wrapper crate, not here).
    let graph = load_file(&path)?;
    println!(
        "Loaded {}: {} bytes resident",
        path,
        std::mem::size_of_val(&*graph)
    );

    // ── 2. Count nodes via Cypher ─────────────────────────────────
    //
    // The session module is the canonical query pipeline — same
    // path Python, Bolt, and MCP all flow through (Phase E).
    let params = HashMap::new();
    let opts = session::ExecuteOptions {
        params: &params,
        deadline: None,
        max_rows: None,
        lazy_eligible: false,
        disabled_passes: None,
        embedder: None,
    };
    let outcome = session::execute_read(&graph, "MATCH (n) RETURN count(n) AS total", &opts)?;
    for row in &outcome.result.rows {
        if let Some(Value::Int64(n)) = row.first() {
            println!("Total nodes: {}", n);
        }
    }

    // ── 3. Sample a few node titles ───────────────────────────────
    let outcome =
        session::execute_read(&graph, "MATCH (n) RETURN n.title AS title LIMIT 5", &opts)?;
    println!("\nSample nodes:");
    for row in &outcome.result.rows {
        if let Some(Value::String(s)) = row.first() {
            println!("  - {}", s);
        }
    }

    Ok(())
}
