//! `kglite` — an interactive Cypher shell for `.kgl` knowledge graphs, in the
//! spirit of the `sqlite3` CLI: open a single file, run queries and
//! dot-commands from the terminal, no Python or server required.
//!
//! Pure-Rust binary over `kglite::api::*` (no libpython link), mirroring the
//! kglite-bolt-server / kglite-mcp-server crate pattern.

mod exec;
mod format;
mod helper;
mod repl;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use kglite::api::io::{load_file, save_graph};
use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};
use kglite::api::{DirGraph, Value};

use crate::exec::QueryOptions;
use crate::format::Mode;

/// Interactive Cypher shell for kglite `.kgl` graphs.
#[derive(Parser, Debug)]
#[command(name = "kglite", version, about)]
#[command(args_conflicts_with_subcommands = true)]
struct Cli {
    /// Path to a `.kgl` file to open. If omitted (or the file does not exist
    /// yet), the shell starts with a fresh in-memory graph.
    graph: Option<PathBuf>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run a read-only Cypher query against a `.kgl` graph and print the result.
    Query {
        /// Path to the `.kgl` file.
        graph: PathBuf,
        /// Cypher query string.
        query: String,
        /// Output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },
    /// Run a write-capable Cypher statement against a `.kgl` graph.
    Write {
        /// Path to the `.kgl` file.
        graph: PathBuf,
        /// Cypher statement.
        query: String,
        /// Output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
        /// Persist the graph after a successful statement.
        #[arg(long)]
        save: bool,
        /// Comma-separated node-type whitelist for CREATE/SET mutations.
        #[arg(long)]
        write_scope: Option<String>,
        /// Git SHA to stamp on auto_timestamp types.
        #[arg(long)]
        git_sha: Option<String>,
        /// Actor id to stamp on auto_timestamp types.
        #[arg(long)]
        modified_by: Option<String>,
    },
    /// Print a deterministic, human-readable text projection of a `.kgl` to
    /// stdout — the canonical form for a git `textconv` diff filter. Set up:
    /// `git config diff.kglite.textconv "kglite export-text"` +
    /// `echo '*.kgl diff=kglite' >> .gitattributes`.
    ExportText {
        /// Path to the `.kgl` file.
        file: PathBuf,
    },
    /// Show what changed between two `.kgl` graphs — a structural delta over the
    /// deterministic text projection: `-` lines dropped from A, `+` lines added
    /// in B (a node/edge whose properties changed shows as a `-`/`+` pair).
    Diff {
        /// The "before" `.kgl`.
        a: PathBuf,
        /// The "after" `.kgl`.
        b: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum OutputFormat {
    #[default]
    Table,
    Csv,
    Json,
}

impl From<OutputFormat> for Mode {
    fn from(value: OutputFormat) -> Self {
        match value {
            OutputFormat::Table => Mode::Table,
            OutputFormat::Csv => Mode::Csv,
            OutputFormat::Json => Mode::Json,
        }
    }
}

fn open_text(path: &Path) -> Result<String> {
    let p = path.to_string_lossy().to_string();
    let g = load_file(&p).with_context(|| format!("failed to open {p}"))?;
    Ok(kglite::api::io::to_text(&g))
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(Command::Query {
        graph,
        query,
        format,
    }) = &cli.command
    {
        run_query(graph, query, (*format).into())?;
        return Ok(());
    }
    if let Some(Command::Write {
        graph,
        query,
        format,
        save,
        write_scope,
        git_sha,
        modified_by,
    }) = &cli.command
    {
        run_write(
            graph,
            query,
            (*format).into(),
            *save,
            write_scope.as_deref(),
            git_sha.clone(),
            modified_by.clone(),
        )?;
        return Ok(());
    }
    if let Some(Command::ExportText { file }) = &cli.command {
        print!("{}", open_text(file)?);
        return Ok(());
    }
    if let Some(Command::Diff { a, b }) = &cli.command {
        let (ta, tb) = (open_text(a)?, open_text(b)?);
        let a_lines: std::collections::BTreeSet<&str> =
            ta.lines().filter(|l| !l.trim().is_empty()).collect();
        let b_lines: std::collections::BTreeSet<&str> =
            tb.lines().filter(|l| !l.trim().is_empty()).collect();
        for l in a_lines.difference(&b_lines) {
            println!("-{}", l.trim_start());
        }
        for l in b_lines.difference(&a_lines) {
            println!("+{}", l.trim_start());
        }
        return Ok(());
    }

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

fn load_graph(path: &Path) -> Result<Arc<DirGraph>> {
    let p = path.to_string_lossy().to_string();
    load_file(&p).with_context(|| format!("failed to open {p}"))
}

fn run_query(path: &Path, query: &str, mode: Mode) -> Result<()> {
    let graph = load_graph(path)?;
    let (_, is_mutation) = kglite::api::cypher::parse_with_mutation_check(query)
        .map_err(|e| anyhow::anyhow!("Cypher parse error: {e}"))?;
    if is_mutation {
        anyhow::bail!("query is read-only; use `kglite write` for mutations");
    }
    let params: HashMap<String, Value> = HashMap::new();
    let outcome = exec::execute_readonly(&graph, query, &params)
        .with_context(|| "Cypher execution failed")?;
    exec::write_stdout(&exec::render_outcome(mode, &outcome))?;
    Ok(())
}

fn run_write(
    path: &Path,
    query: &str,
    mode: Mode,
    persist: bool,
    write_scope: Option<&str>,
    git_sha: Option<String>,
    modified_by: Option<String>,
) -> Result<()> {
    let mut graph = if path.exists() {
        load_graph(path)?
    } else if persist {
        Arc::new(fresh_graph()?)
    } else {
        anyhow::bail!(
            "{} does not exist; pass --save to create it after a successful write",
            path.display()
        );
    };
    let params: HashMap<String, Value> = HashMap::new();
    let options = QueryOptions {
        write_scope: exec::parse_write_scope(write_scope),
        git_sha,
        modified_by,
        ..QueryOptions::default()
    };
    let outcome = exec::execute(&mut graph, query, &params, &options)
        .with_context(|| "Cypher execution failed")?;
    if persist {
        let p = path.to_string_lossy().to_string();
        save_graph(&mut graph, &p).map_err(|e| anyhow::anyhow!("failed to save {p}: {e}"))?;
    }
    exec::write_stdout(&exec::render_outcome(mode, &outcome))?;
    Ok(())
}
