//! `kglite` — an interactive Cypher shell for `.kgl` knowledge graphs, in the
//! spirit of the `sqlite3` CLI: open a single file, run queries and
//! dot-commands from the terminal, no Python or server required.
//!
//! Pure-Rust binary over `kglite::api::*` (no libpython link), mirroring the
//! kglite-bolt-server / kglite-mcp-server crate pattern.

mod format;
mod repl;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use kglite::api::io::load_file;
use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};
use kglite::api::DirGraph;

/// Interactive Cypher shell for kglite `.kgl` graphs.
#[derive(Parser, Debug)]
#[command(name = "kglite", version, about)]
struct Cli {
    /// Path to a `.kgl` file to open. If omitted (or the file does not exist
    /// yet), the shell starts with a fresh in-memory graph.
    graph: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let (graph, source): (Arc<DirGraph>, Option<String>) = match &cli.graph {
        Some(path) if path.exists() => {
            let p = path.to_string_lossy().to_string();
            let g = load_file(&p).with_context(|| format!("failed to open {p}"))?;
            (g, Some(p))
        }
        Some(path) => {
            // Named but missing: start fresh; `.save <path>` (Phase 5) will
            // write here. Tell the user so a typo'd path isn't silently empty.
            let p = path.to_string_lossy().to_string();
            eprintln!("note: {p} does not exist — starting an empty in-memory graph");
            (Arc::new(fresh_graph()?), None)
        }
        None => (Arc::new(fresh_graph()?), None),
    };

    repl::run(graph, source.as_deref())
}

/// A fresh in-memory graph. `new_dir_graph_in_mode` returns `Result<_, String>`
/// (not an `Error`), so adapt it into `anyhow` explicitly.
fn fresh_graph() -> Result<DirGraph> {
    new_dir_graph_in_mode(StorageMode::Memory, None)
        .map_err(|e| anyhow::anyhow!("failed to create an in-memory graph: {e}"))
}
