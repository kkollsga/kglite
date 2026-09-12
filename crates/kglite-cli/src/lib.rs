//! Shared implementation of the `kglite` CLI.
//!
//! `kglite` is an interactive Cypher shell for `.kgl` knowledge graphs, in the
//! spirit of the `sqlite3` CLI: open a single file, run queries and
//! dot-commands from the terminal, no Python or server required.
//!
//! Pure-Rust binary over `kglite::api::*` (no libpython link), mirroring the
//! kglite-bolt-server / kglite-mcp-server crate pattern.

mod agent_response;
mod exec;
mod format;
mod helper;
mod migrate;
mod repl;

use std::collections::HashMap;
use std::ffi::OsString;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use kglite::api::introspection::{
    compute_description, ConnectionDetail, CypherDetail, DescribeRequest, DescribeSurface,
    FluentDetail,
};
use kglite::api::io::{
    load_file, open_or_create_graph, GraphWriterLease, OpenDisposition, WriteOwnership,
    WriteRefusal,
};
use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};
use kglite::api::{DirGraph, Value};

use crate::exec::QueryOptions;
use crate::format::Mode;

#[derive(Debug)]
struct ReportedAgentFailure;

impl std::fmt::Display for ReportedAgentFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("agent operation failed")
    }
}

impl std::error::Error for ReportedAgentFailure {}

/// Whether the CLI already emitted a bounded structured failure for this error.
pub fn is_reported_agent_failure(error: &anyhow::Error) -> bool {
    error.downcast_ref::<ReportedAgentFailure>().is_some()
}

/// How long a save-capable invocation waits for a peer to release the graph.
///
/// Taken *before* the graph is read, and deliberately long: two `kglite write
/// --save` runs against one file are expected to serialize into one after the
/// other, and the second one has nothing useful to do with a snapshot the
/// first is about to invalidate.
const WRITE_LOCK_TIMEOUT: Duration = Duration::from_secs(30);

/// Render a core write refusal in the CLI's own vocabulary. The lost-update
/// wording is the one users have seen since the guard shipped.
fn write_refusal(path: &Path, refusal: WriteRefusal) -> anyhow::Error {
    match refusal {
        WriteRefusal::Stale { .. } => anyhow::anyhow!(
            "refusing to overwrite {}: it changed since this session loaded it",
            path.display()
        ),
        other => anyhow::anyhow!("failed to save {}: {other}", path.display()),
    }
}

/// Open `path` for a run that may write it back, holding the writer lease
/// across the whole read/modify/publish interval.
///
/// The lease is taken *before* the open, which is the ordering that makes two
/// concurrent `--save` runs serialize: the loser waits, then reads what the
/// winner published, instead of reading first and finding its snapshot stale.
fn open_owned(
    path: &Path,
    create_mode: Option<StorageMode>,
) -> Result<(Arc<DirGraph>, WriteOwnership)> {
    let lease = GraphWriterLease::acquire(path, WRITE_LOCK_TIMEOUT)?;
    let opened = open_or_create_graph(path, create_mode)
        .with_context(|| format!("failed to open or create {}", path.display()))?;
    let graph = opened.graph;
    let mut ownership = WriteOwnership::new(
        path.to_path_buf(),
        opened.identity,
        &graph,
        None,
        // The CLI has no rollback path: a failed statement ends the process,
        // and holding a second `Arc` would fork the graph on the next write
        // for a snapshot nothing would ever read.
        false,
    );
    ownership.adopt_lease(lease, &graph, true);
    Ok((graph, ownership))
}

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
        /// Opt in to the parallel runtime for this query. A hint: operators
        /// that cannot partition deterministically, and queries below the
        /// engine's runtime size gate, still run sequentially.
        #[arg(long)]
        parallel: bool,
        /// Deadline for this statement, in milliseconds. Omitted means no
        /// deadline, which is this CLI's declared default: Ctrl-C is the
        /// interactive cancel, and a batch query over a very large graph may
        /// legitimately run for hours. 0 is the same as omitting it.
        #[arg(long)]
        timeout_ms: Option<u64>,
        /// Serialized-byte budget for agent output (minimum 4096).
        #[arg(long, conflicts_with = "response_full")]
        response_max_bytes: Option<usize>,
        /// Return the complete agent envelope inline.
        #[arg(long)]
        response_full: bool,
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
        /// Comma-separated node-type whitelist. Node writes (CREATE, MERGE, SET,
        /// REMOVE, DELETE, DETACH DELETE, node-type DDL) are judged by the node's
        /// stored type; a relationship write needs at least one endpoint in the list.
        #[arg(long)]
        write_scope: Option<String>,
        /// Git SHA to stamp on auto_timestamp types.
        #[arg(long)]
        git_sha: Option<String>,
        /// Actor id to stamp on auto_timestamp types.
        #[arg(long)]
        modified_by: Option<String>,
        /// Deadline for this statement, in milliseconds. Omitted means no
        /// deadline, which is this CLI's declared default: Ctrl-C is the
        /// interactive cancel, and a batch query over a very large graph may
        /// legitimately run for hours. 0 is the same as omitting it.
        #[arg(long)]
        timeout_ms: Option<u64>,
        /// Serialized-byte budget for agent output (minimum 4096).
        #[arg(long, conflicts_with = "response_full")]
        response_max_bytes: Option<usize>,
        /// Return the complete agent envelope inline.
        #[arg(long)]
        response_full: bool,
    },
    /// Print the dependency frontier from `CALL ready_set(...)`.
    ReadySet {
        /// Path to the `.kgl` file.
        graph: PathBuf,
        /// Dependency relationship type.
        #[arg(long, default_value = "DEPENDS_ON")]
        relationship: String,
        /// Done predicate over `n`, for example: `n.status = "done"`.
        #[arg(long)]
        done: String,
        /// Optional node type to include in the frontier.
        #[arg(long)]
        node_type: Option<String>,
        /// Output format.
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },
    /// Print the XML graph description used by agents for structure discovery.
    Describe {
        /// Path to the `.kgl` file.
        graph: PathBuf,
        /// Comma-separated node types for focused detail.
        #[arg(long)]
        types: Option<String>,
        /// Search node types by name.
        #[arg(long)]
        type_search: Option<String>,
        /// Include connection overview.
        #[arg(long)]
        connections: bool,
        /// Comma-separated connection types for deep-dive detail.
        #[arg(long)]
        connection_types: Option<String>,
        /// Include compact Cypher reference.
        #[arg(long)]
        cypher: bool,
        /// Comma-separated Cypher topics for detailed docs.
        #[arg(long)]
        cypher_topics: Option<String>,
        /// Include compact fluent API reference.
        #[arg(long)]
        fluent: bool,
        /// Comma-separated fluent API topics for detailed docs.
        #[arg(long)]
        fluent_topics: Option<String>,
        /// Max `(source_type, target_type)` pairs for connection deep-dives.
        #[arg(long)]
        max_pairs: Option<usize>,
        /// Truncate long sample strings to this many characters.
        #[arg(long, default_value_t = 40)]
        sample_truncate: usize,
    },
    /// Keep one graph loaded and process JSONL requests on stdin.
    Session {
        /// Path to the `.kgl` file.
        graph: PathBuf,
        /// Default output format for query/write responses.
        #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
        format: OutputFormat,
        /// Save the graph when the session exits successfully.
        #[arg(long)]
        save_on_exit: bool,
        /// Comma-separated node-type whitelist for write requests. Node writes are
        /// judged by the node's stored type; a relationship write needs at least one
        /// endpoint in the list.
        #[arg(long)]
        write_scope: Option<String>,
        /// Git SHA to stamp on auto_timestamp types for write requests.
        #[arg(long)]
        git_sha: Option<String>,
        /// Actor id to stamp on auto_timestamp types for write requests.
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
    /// Export a `.kgl` as a SQLite-dialect SQL script — the no-lock-in exit.
    /// Node types become tables, connection types become link tables. Ingest
    /// with: `sqlite3 target.db < dump.sql`.
    ExportSqlite {
        /// Path to the `.kgl` file.
        graph: PathBuf,
        /// Where to write the SQL script. Omit to write to stdout.
        output: Option<PathBuf>,
    },
    /// Apply pending Cypher migrations and advance the graph's user-schema
    /// version. Migrations are `<version>_<name>.cypher` files in one
    /// directory, applied in ascending version order; a re-run is a no-op.
    Migrate {
        /// Path to the `.kgl` file.
        graph: PathBuf,
        /// Directory holding the `<version>_<name>.cypher` migrations.
        directory: PathBuf,
        /// Print the plan without applying or saving anything.
        #[arg(long)]
        dry_run: bool,
    },
    /// Print the graph's user-schema version (the caller's own data-model
    /// revision, distinct from the `.kgl` format version).
    SchemaVersion {
        /// Path to the `.kgl` file.
        graph: PathBuf,
        /// Stamp this version instead of printing — the "adopt migrations on an
        /// existing graph" operation. Runs nothing; it asserts that the data
        /// already has the shape migrations up to this version produce.
        #[arg(long)]
        set: Option<u32>,
    },
    /// Retrieve or purge retained agent-response evidence.
    #[command(subcommand)]
    Response(ResponseCommand),
}

#[derive(Subcommand, Debug)]
enum ResponseCommand {
    /// Expand retained evidence without opening a graph or rerunning a query.
    Expand {
        /// Opaque handle emitted by an earlier agent response.
        handle: String,
        /// JSON Pointer into the retained canonical envelope.
        #[arg(long, default_value = "")]
        path: String,
        /// Array item, object field, or Unicode-character offset.
        #[arg(long, default_value_t = 0)]
        offset: usize,
        /// Serialized-byte budget for this expansion (minimum 4096).
        #[arg(long, conflicts_with = "response_full")]
        response_max_bytes: Option<usize>,
        /// Return the complete selected value inline; offset must be zero.
        #[arg(long)]
        response_full: bool,
    },
    /// Remove retained evidence across every workspace namespace.
    Purge {
        /// Confirm that the complete per-user response cache is removed.
        #[arg(long, required = true)]
        all: bool,
    },
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum OutputFormat {
    #[default]
    Table,
    Csv,
    Json,
    Agent,
}

fn validate_agent_cli(cli: &Cli) -> Result<()> {
    let controls = match &cli.command {
        Some(Command::Query {
            format,
            response_max_bytes,
            response_full,
            ..
        })
        | Some(Command::Write {
            format,
            response_max_bytes,
            response_full,
            ..
        }) => Some((*format, *response_max_bytes, *response_full)),
        _ => None,
    };
    if let Some((format, max_bytes, full)) = controls {
        if format != OutputFormat::Agent && (max_bytes.is_some() || full) {
            anyhow::bail!("--response-max-bytes and --response-full require --format agent");
        }
        if format == OutputFormat::Agent {
            agent_response::AgentOptions { max_bytes, full }.validate()?;
        }
    }
    Ok(())
}

impl From<OutputFormat> for Mode {
    fn from(value: OutputFormat) -> Self {
        match value {
            OutputFormat::Table => Mode::Table,
            OutputFormat::Csv => Mode::Csv,
            OutputFormat::Json => Mode::Json,
            OutputFormat::Agent => unreachable!("agent output uses its structured renderer"),
        }
    }
}

fn open_text(path: &Path) -> Result<String> {
    let p = path.to_string_lossy().to_string();
    let g = load_file(&p).with_context(|| format!("failed to open {p}"))?;
    Ok(kglite::api::io::to_text(&g))
}

/// Run the CLI over an explicit argument vector, including the program name.
///
/// The standalone binary and the `pip install kglite` wheel shim both call
/// this entry point, so command parsing and behavior cannot drift.
pub fn run<I, T>(args: I) -> Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let cli = Cli::parse_from(args);

    validate_agent_cli(&cli)?;

    if let Some(Command::Query {
        graph,
        query,
        format,
        parallel,
        timeout_ms,
        response_max_bytes,
        response_full,
    }) = &cli.command
    {
        if *format == OutputFormat::Agent {
            run_agent_query(
                graph,
                query,
                *parallel,
                *timeout_ms,
                agent_response::AgentOptions {
                    max_bytes: *response_max_bytes,
                    full: *response_full,
                },
            )?;
        } else {
            run_query(graph, query, (*format).into(), *parallel, *timeout_ms)?;
        }
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
        timeout_ms,
        response_max_bytes,
        response_full,
    }) = &cli.command
    {
        let options = exec::QueryOptions {
            write_scope: exec::parse_write_scope(write_scope.as_deref()),
            git_sha: git_sha.clone(),
            modified_by: modified_by.clone(),
            timeout_ms: *timeout_ms,
            ..exec::QueryOptions::default()
        };
        if *format == OutputFormat::Agent {
            run_agent_write(
                graph,
                query,
                *save,
                options,
                agent_response::AgentOptions {
                    max_bytes: *response_max_bytes,
                    full: *response_full,
                },
            )?;
        } else {
            run_write(graph, query, (*format).into(), *save, options)?;
        }
        return Ok(());
    }
    if let Some(Command::ReadySet {
        graph,
        relationship,
        done,
        node_type,
        format,
    }) = &cli.command
    {
        run_ready_set(
            graph,
            relationship,
            done,
            node_type.as_deref(),
            (*format).into(),
        )?;
        return Ok(());
    }
    if let Some(Command::Describe {
        graph,
        types,
        type_search,
        connections,
        connection_types,
        cypher,
        cypher_topics,
        fluent,
        fluent_topics,
        max_pairs,
        sample_truncate,
    }) = &cli.command
    {
        run_describe(
            graph,
            DescribeOptions {
                types: parse_csv(types.as_deref()),
                type_search: type_search.clone(),
                connections: detail_connections(*connections, connection_types.as_deref()),
                cypher: detail_cypher(*cypher, cypher_topics.as_deref()),
                fluent: detail_fluent(*fluent, fluent_topics.as_deref()),
                max_pairs: *max_pairs,
                sample_truncate: Some(*sample_truncate),
            },
        )?;
        return Ok(());
    }
    if let Some(Command::Session {
        graph,
        format,
        save_on_exit,
        write_scope,
        git_sha,
        modified_by,
    }) = &cli.command
    {
        run_session(
            graph,
            *format,
            *save_on_exit,
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
    if let Some(Command::ExportSqlite { graph, output }) = &cli.command {
        return run_export_sqlite(graph, output.as_deref());
    }
    if let Some(Command::Migrate {
        graph,
        directory,
        dry_run,
    }) = &cli.command
    {
        return migrate::run(graph, directory, *dry_run);
    }
    if let Some(Command::SchemaVersion { graph, set }) = &cli.command {
        return match set {
            Some(version) => migrate::set_version(graph, *version),
            None => migrate::print_version(graph),
        };
    }
    if let Some(Command::Response(command)) = &cli.command {
        return run_response(command);
    }

    let (graph, ownership) = match &cli.graph {
        Some(path) => {
            let p = path.to_string_lossy().to_string();
            let opened = open_or_create_graph(path, Some(StorageMode::Memory))
                .with_context(|| format!("failed to open or create {}", path.display()))?;
            if opened.disposition == OpenDisposition::Created {
                eprintln!("note: {p} does not exist — starting an empty in-memory graph");
            }
            // A path that did not exist is not this session's file yet: `.save`
            // with no argument still asks for one, as it always has.
            let ownership = (opened.disposition == OpenDisposition::Opened).then(|| {
                WriteOwnership::new(path.clone(), opened.identity, &opened.graph, None, false)
            });
            (opened.graph, ownership)
        }
        None => (Arc::new(fresh_graph()?), None),
    };

    repl::run(graph, ownership)
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

/// Write the SQL projection of a graph to a file, or to stdout when no output
/// path is given — the same shape as `export-text`, so the dump can be piped
/// straight into `sqlite3`.
fn run_export_sqlite(graph_path: &Path, output: Option<&Path>) -> Result<()> {
    let graph = load_graph(graph_path)?;
    let sql = kglite::api::io::to_sqlite_dump(&graph, None)
        .map_err(|e| anyhow::anyhow!("SQL export failed: {e}"))?;
    match output {
        Some(path) => {
            std::fs::write(path, &sql)
                .with_context(|| format!("failed to write {}", path.display()))?;
            eprintln!("wrote {} ({} bytes)", path.display(), sql.len());
        }
        None => exec::write_stdout(&sql)?,
    }
    Ok(())
}

fn run_query(
    path: &Path,
    query: &str,
    mode: Mode,
    parallel: bool,
    timeout_ms: Option<u64>,
) -> Result<()> {
    let graph = load_graph(path)?;
    let (_, is_mutation) = kglite::api::cypher::parse_with_mutation_check(query)
        .map_err(|e| anyhow::anyhow!("Cypher parse error: {e}"))?;
    if is_mutation {
        anyhow::bail!("query is read-only; use `kglite write` for mutations");
    }
    let params: HashMap<String, Value> = HashMap::new();
    let options = QueryOptions {
        parallel,
        timeout_ms,
        ..QueryOptions::default()
    };
    let outcome = exec::execute_readonly(&graph, query, &params, &options)
        .with_context(|| "Cypher execution failed")?;
    exec::write_stdout(&exec::render_outcome(
        mode,
        &outcome,
        format::stdout_cell_cap(),
    ))?;
    Ok(())
}

fn run_agent_query(
    path: &Path,
    query: &str,
    parallel: bool,
    timeout_ms: Option<u64>,
    response: agent_response::AgentOptions,
) -> Result<()> {
    let operation = kglite::api::cypher::with_query_warning_sink(
        kglite::api::cypher::QueryWarningSink::Silent,
        || {
            let graph = load_graph(path)?;
            let (_, is_mutation) = kglite::api::cypher::parse_with_mutation_check(query)
                .map_err(|error| anyhow::anyhow!("Cypher parse error: {error}"))?;
            if is_mutation {
                anyhow::bail!("query is read-only; use `kglite write` for mutations");
            }
            let params = HashMap::new();
            exec::execute_readonly(
                &graph,
                query,
                &params,
                &QueryOptions {
                    parallel,
                    timeout_ms,
                    ..QueryOptions::default()
                },
            )
            .context("Cypher execution failed")
        },
    );
    emit_agent_operation(path, query, operation, response)
}

fn run_agent_write(
    path: &Path,
    query: &str,
    persist: bool,
    options: exec::QueryOptions,
    response: agent_response::AgentOptions,
) -> Result<()> {
    let operation = kglite::api::cypher::with_query_warning_sink(
        kglite::api::cypher::QueryWarningSink::Silent,
        || {
            let (mut graph, mut ownership) = if persist {
                let (graph, ownership) = open_owned(path, Some(StorageMode::Memory))?;
                (graph, Some(ownership))
            } else {
                let graph = open_or_create_graph(path, None)
                    .with_context(|| format!("failed to open or create {}", path.display()))?
                    .graph;
                (graph, None)
            };
            let params = HashMap::new();
            let outcome = exec::execute(&mut graph, query, &params, &options)
                .context("Cypher execution failed")?;
            if let Some(ownership) = ownership.as_mut() {
                ownership
                    .publish(&mut graph)
                    .map_err(|refusal| write_refusal(path, refusal))?;
            }
            Ok(outcome)
        },
    );
    emit_agent_operation(path, query, operation, response)
}

fn emit_agent_operation(
    path: &Path,
    query: &str,
    operation: Result<kglite::api::session::ExecuteOutcome>,
    response: agent_response::AgentOptions,
) -> Result<()> {
    let (envelope, error) = match operation {
        Ok(outcome) => (
            agent_response::result_envelope(&outcome, query, path),
            false,
        ),
        Err(error) => (agent_response::error_envelope(&error, query, path), true),
    };
    let root = agent_response::cache_root()?;
    let namespace = agent_response::namespace(path);
    let rendered = agent_response::present(root, &namespace, envelope, error, response);
    exec::write_stdout(&serde_json::to_string(&rendered)?)?;
    if error {
        Err(ReportedAgentFailure.into())
    } else {
        Ok(())
    }
}

fn run_response(command: &ResponseCommand) -> Result<()> {
    let root = agent_response::cache_root()?;
    match command {
        ResponseCommand::Expand {
            handle,
            path,
            offset,
            response_max_bytes,
            response_full,
        } => {
            let expanded = match agent_response::expand(
                root,
                handle,
                path.clone(),
                *offset,
                agent_response::AgentOptions {
                    max_bytes: *response_max_bytes,
                    full: *response_full,
                },
            ) {
                Ok(value) => value,
                Err(error) => {
                    let value = serde_json::json!({
                        "content": [{"type":"text","text":"Retained response expansion failed"}],
                        "structuredContent": {"schema_version":1,"kind":"response_error","diagnostics":{"errors":[{"message":format!("{error:#}")}]}},
                        "isError": true
                    });
                    exec::write_stdout(&serde_json::to_string(&value)?)?;
                    return Err(ReportedAgentFailure.into());
                }
            };
            exec::write_stdout(&serde_json::to_string(&expanded)?)?;
        }
        ResponseCommand::Purge { all } => {
            debug_assert!(*all, "clap requires --all");
            agent_response::purge(root)?;
            exec::write_stdout("{\"ok\":true,\"purged\":\"all\"}")?;
        }
    }
    Ok(())
}

/// `options` rather than four more positional arguments: the write knobs
/// (scope, provenance, deadline) are exactly the `QueryOptions` the executor
/// already takes, so threading them one by one only creates a place to drop
/// one silently.
fn run_write(
    path: &Path,
    query: &str,
    mode: Mode,
    persist: bool,
    options: exec::QueryOptions,
) -> Result<()> {
    let (mut graph, mut ownership) = if persist {
        let (graph, ownership) = open_owned(path, Some(StorageMode::Memory))?;
        (graph, Some(ownership))
    } else {
        let graph = open_or_create_graph(path, None)
            .with_context(|| format!("failed to open or create {}", path.display()))?
            .graph;
        (graph, None)
    };
    let params: HashMap<String, Value> = HashMap::new();
    let outcome = exec::execute(&mut graph, query, &params, &options)
        .with_context(|| "Cypher execution failed")?;
    if let Some(ownership) = ownership.as_mut() {
        ownership
            .publish(&mut graph)
            .map_err(|refusal| write_refusal(path, refusal))?;
    }
    exec::write_stdout(&exec::render_outcome(
        mode,
        &outcome,
        format::stdout_cell_cap(),
    ))?;
    Ok(())
}

fn run_ready_set(
    path: &Path,
    relationship: &str,
    done: &str,
    node_type: Option<&str>,
    mode: Mode,
) -> Result<()> {
    let mut config = vec![
        format!("relationship: '{}'", cypher_string(relationship)),
        format!("done: '{}'", cypher_string(done)),
    ];
    if let Some(node_type) = node_type {
        config.push(format!("node_type: '{}'", cypher_string(node_type)));
    }
    let query = format!(
        "CALL ready_set({{{}}}) YIELD node, dependency_count \
         RETURN node.id AS id, node.title AS title, dependency_count \
         ORDER BY dependency_count, id",
        config.join(", ")
    );
    run_query(path, &query, mode, false, None)
}

struct DescribeOptions {
    types: Option<Vec<String>>,
    type_search: Option<String>,
    connections: ConnectionDetail,
    cypher: CypherDetail,
    fluent: FluentDetail,
    max_pairs: Option<usize>,
    sample_truncate: Option<usize>,
}

fn run_describe(path: &Path, options: DescribeOptions) -> Result<()> {
    let graph = load_graph(path)?;
    let description = describe_graph(&graph, &options)?;
    exec::write_stdout(&description)?;
    Ok(())
}

fn describe_graph(graph: &Arc<DirGraph>, options: &DescribeOptions) -> Result<String> {
    compute_description(
        graph,
        &DescribeRequest {
            types: options.types.as_deref(),
            connections: &options.connections,
            cypher: &options.cypher,
            fluent: &options.fluent,
            type_search: options.type_search.as_deref(),
            max_pairs: options.max_pairs,
            sample_truncate: options.sample_truncate,
            ..DescribeRequest::new(DescribeSurface::Cli)
        },
    )
    .map_err(|e| anyhow::anyhow!("describe failed: {e}"))
}

fn run_session(
    path: &Path,
    default_format: OutputFormat,
    save_on_exit: bool,
    write_scope: Option<&str>,
    git_sha: Option<String>,
    modified_by: Option<String>,
) -> Result<()> {
    let (mut graph, mut ownership) = if save_on_exit {
        open_owned(path, Some(StorageMode::Memory))?
    } else {
        let opened = open_or_create_graph(path, None)
            .with_context(|| format!("failed to open or create {}", path.display()))?;
        let ownership = WriteOwnership::new(
            path.to_path_buf(),
            opened.identity,
            &opened.graph,
            None,
            false,
        );
        (opened.graph, ownership)
    };
    let base_options = QueryOptions {
        write_scope: exec::parse_write_scope(write_scope),
        git_sha,
        modified_by,
        ..QueryOptions::default()
    };
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match handle_session_line(
            &mut graph,
            path,
            line,
            default_format,
            &base_options,
            &mut ownership,
        ) {
            SessionAction::Continue(value) => write_json_line(value)?,
            SessionAction::Exit(value) => {
                write_json_line(value)?;
                if save_on_exit {
                    session_save(&mut graph, &mut ownership)?;
                }
                return Ok(());
            }
        }
    }
    if save_on_exit {
        session_save(&mut graph, &mut ownership)?;
    }
    Ok(())
}

enum SessionAction {
    Continue(serde_json::Value),
    Exit(serde_json::Value),
}

fn handle_session_line(
    graph: &mut Arc<DirGraph>,
    graph_path: &Path,
    line: &str,
    default_format: OutputFormat,
    base_options: &QueryOptions,
    ownership: &mut WriteOwnership,
) -> SessionAction {
    let request: serde_json::Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => {
            return SessionAction::Continue(json_error("parse", format!("invalid JSON: {e}")));
        }
    };
    let op = request
        .get("op")
        .and_then(|v| v.as_str())
        .unwrap_or("query");
    let request_id = request.get("id").cloned();
    let agent_intent = session_agent_intent(op, &request, default_format);
    let result = match op {
        "query" => session_query(graph, graph_path, &request, default_format),
        "write" => session_write(graph, graph_path, &request, default_format, base_options),
        "response_expand" => session_response_expand(&request),
        "describe" => session_describe(graph, &request),
        "save" => {
            session_save(graph, ownership).map(|()| serde_json::json!({"ok": true, "op": "save"}))
        }
        "help" => Ok(session_help()),
        "exit" | "quit" => {
            let mut value = serde_json::json!({"ok": true, "op": op});
            insert_request_id(&mut value, request_id);
            return SessionAction::Exit(value);
        }
        other => Err(anyhow::anyhow!(
            "unknown op {other:?}; valid ops: {} — send {{\"op\":\"help\"}} for details",
            session_op_names()
        )),
    };
    SessionAction::Continue(match result {
        Ok(value) if agent_intent => agent_response::finalize_session(
            value,
            op,
            request_id,
            session_effective_agent_options(&request),
        ),
        Ok(mut value) => {
            if let Some(obj) = value.as_object_mut() {
                obj.entry("op").or_insert_with(|| serde_json::json!(op));
            }
            insert_request_id(&mut value, request_id);
            value
        }
        Err(e) if agent_intent => {
            let query = request
                .get("query")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            let envelope = agent_response::error_envelope(&e, query, graph_path);
            let options = session_effective_agent_options(&request);
            let value = match agent_response::cache_root() {
                Ok(root) => agent_response::present(
                    root,
                    &agent_response::namespace(graph_path),
                    envelope,
                    true,
                    options,
                ),
                Err(cache_error) => serde_json::json!({
                    "content":[{"type":"text","text":"Agent session operation failed"}],
                    "structuredContent":agent_response::error_envelope(
                        &cache_error,
                        query,
                        graph_path,
                    ),
                    "isError":true
                }),
            };
            agent_response::finalize_session(value, op, request_id, options)
        }
        Err(e) => {
            let mut value = json_error(op, e.to_string());
            insert_request_id(&mut value, request_id);
            value
        }
    })
}

/// The JSONL session protocol's op table: name + a one-line description of the
/// request shape. Served by `{"op":"help"}` and named in the unknown-op error,
/// so the protocol is discoverable from inside the protocol — an agent driving
/// the session over a pipe has no other way to learn it.
const SESSION_OPS: &[(&str, &str)] = &[
    (
        "query",
        "run a read-only Cypher query — {\"op\":\"query\",\"query\":\"MATCH …\",\"format\":\"json|table|csv|agent\"} \
         (agent optional: \"response\":{\"max_bytes\":4096}|{\"mode\":\"full\"})",
    ),
    (
        "write",
        "run a write Cypher statement — {\"op\":\"write\",\"query\":\"CREATE …\"} \
         (optional: \"format\":\"agent\", \"response\", \"write_scope\":[\"Type\"], \"git_sha\", \"modified_by\")",
    ),
    (
        "response_expand",
        "expand retained agent evidence without rerunning — {\"op\":\"response_expand\",\"handle\":\"…\"} \
         (optional: \"path\", \"offset\", \"response\":{\"max_bytes\":4096}|{\"mode\":\"full\"})",
    ),
    (
        "describe",
        "describe the graph for agents — {\"op\":\"describe\"} \
         (optional: \"types\", \"type_search\", \"connections\", \"cypher\", \"fluent\", \"max_pairs\")",
    ),
    (
        "save",
        "save the loaded graph back to its file — {\"op\":\"save\"}",
    ),
    ("help", "list these ops — {\"op\":\"help\"}"),
    (
        "exit",
        "end the session (alias: \"quit\") — {\"op\":\"exit\"}",
    ),
];

/// Comma-separated op names for the unknown-op error, `quit` included because
/// it is accepted even though `exit` is the documented spelling.
fn session_op_names() -> String {
    let mut names: Vec<&str> = SESSION_OPS.iter().map(|(name, _)| *name).collect();
    names.push("quit");
    names.join(", ")
}

/// `{"op":"help"}` — the op table as a normal `ok:true` response.
fn session_help() -> serde_json::Value {
    let ops: Vec<serde_json::Value> = SESSION_OPS
        .iter()
        .map(|(name, description)| serde_json::json!({"op": name, "description": description}))
        .collect();
    serde_json::json!({
        "ok": true,
        "protocol": "one JSON request object per line on stdin, one JSON response per line on stdout; \
                     every response echoes \"op\" and, when the request carried one, its \"id\". \
                     A response is {\"ok\":true, …} or {\"ok\":false,\"error\":\"…\"}.",
        "ops": ops,
    })
}

fn session_query(
    graph: &Arc<DirGraph>,
    graph_path: &Path,
    request: &serde_json::Value,
    default_format: OutputFormat,
) -> Result<serde_json::Value> {
    let query = request_string(request, "query")?;
    let agent = session_agent_options(request, default_format)?;
    if let Some(response) = agent {
        let operation = kglite::api::cypher::with_query_warning_sink(
            kglite::api::cypher::QueryWarningSink::Silent,
            || {
                let (_, is_mutation) = kglite::api::cypher::parse_with_mutation_check(&query)
                    .map_err(|e| anyhow::anyhow!("Cypher parse error: {e}"))?;
                if is_mutation {
                    anyhow::bail!("query is read-only; use op=write for mutations");
                }
                exec::execute_readonly(graph, &query, &HashMap::new(), &QueryOptions::default())
            },
        );
        return session_agent_operation(graph_path, &query, operation, response);
    }
    let mode = session_mode(request, default_format)?;
    let (_, is_mutation) = kglite::api::cypher::parse_with_mutation_check(&query)
        .map_err(|e| anyhow::anyhow!("Cypher parse error: {e}"))?;
    if is_mutation {
        anyhow::bail!("query is read-only; use op=write for mutations");
    }
    let params = HashMap::new();
    let outcome = exec::execute_readonly(graph, &query, &params, &QueryOptions::default())?;
    Ok(session_outcome_response(mode, &outcome))
}

fn session_write(
    graph: &mut Arc<DirGraph>,
    graph_path: &Path,
    request: &serde_json::Value,
    default_format: OutputFormat,
    base_options: &QueryOptions,
) -> Result<serde_json::Value> {
    let query = request_string(request, "query")?;
    let agent = session_agent_options(request, default_format)?;
    let params = HashMap::new();
    let options = QueryOptions {
        write_scope: request
            .get("write_scope")
            .and_then(json_string_vec)
            .map(|v| v.into_iter().collect())
            .or_else(|| base_options.write_scope.clone()),
        git_sha: request
            .get("git_sha")
            .and_then(|v| v.as_str().map(str::to_string))
            .or_else(|| base_options.git_sha.clone()),
        modified_by: request
            .get("modified_by")
            .and_then(|v| v.as_str().map(str::to_string))
            .or_else(|| base_options.modified_by.clone()),
        ..QueryOptions::default()
    };
    if let Some(response) = agent {
        let operation = kglite::api::cypher::with_query_warning_sink(
            kglite::api::cypher::QueryWarningSink::Silent,
            || exec::execute(graph, &query, &params, &options),
        );
        return session_agent_operation(graph_path, &query, operation, response);
    }
    let mode = session_mode(request, default_format)?;
    let outcome = exec::execute(graph, &query, &params, &options)?;
    Ok(session_outcome_response(mode, &outcome))
}

fn session_agent_operation(
    graph_path: &Path,
    query: &str,
    operation: Result<kglite::api::session::ExecuteOutcome>,
    response: agent_response::AgentOptions,
) -> Result<serde_json::Value> {
    let (envelope, error) = match operation {
        Ok(outcome) => (
            agent_response::result_envelope(&outcome, query, graph_path),
            false,
        ),
        Err(error) => (
            agent_response::error_envelope(&error, query, graph_path),
            true,
        ),
    };
    Ok(agent_response::present(
        agent_response::cache_root()?,
        &agent_response::namespace(graph_path),
        envelope,
        error,
        response,
    ))
}

fn session_response_expand(request: &serde_json::Value) -> Result<serde_json::Value> {
    let handle = request_string(request, "handle")?;
    let path = request
        .get("path")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_owned();
    let offset = request
        .get("offset")
        .map(|value| {
            value
                .as_u64()
                .and_then(|number| usize::try_from(number).ok())
                .ok_or_else(|| anyhow::anyhow!("offset must be a non-negative integer"))
        })
        .transpose()?
        .unwrap_or(0);
    let response = response_options_from_json(request.get("response"))?;
    Ok(
        match agent_response::expand(
            agent_response::cache_root()?,
            &handle,
            path,
            offset,
            response,
        ) {
            Ok(value) => value,
            Err(error) => serde_json::json!({
                "content": [{"type":"text","text":"Retained response expansion failed"}],
                "structuredContent": {
                    "schema_version":1,
                    "kind":"response_error",
                    "diagnostics":{"errors":[{"message":format!("{error:#}")}]}
                },
                "isError": true
            }),
        },
    )
}

fn session_describe(
    graph: &Arc<DirGraph>,
    request: &serde_json::Value,
) -> Result<serde_json::Value> {
    let options = describe_options_from_json(request)?;
    Ok(serde_json::json!({
        "ok": true,
        "description": describe_graph(graph, &options)?,
    }))
}

fn session_save(graph: &mut Arc<DirGraph>, ownership: &mut WriteOwnership) -> Result<()> {
    let path = ownership.path().to_path_buf();
    ownership
        .publish(graph)
        .map_err(|refusal| write_refusal(&path, refusal))
}

fn write_json_line(value: serde_json::Value) -> Result<()> {
    let mut stdout = io::stdout().lock();
    serde_json::to_writer(&mut stdout, &value)?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

fn session_outcome_response(
    mode: Mode,
    outcome: &kglite::api::session::ExecuteOutcome,
) -> serde_json::Value {
    if mode == Mode::Json {
        serde_json::json!({
            "ok": true,
            "rows": exec::outcome_rows_json(outcome),
        })
    } else {
        serde_json::json!({
            "ok": true,
            // The session speaks a machine protocol: a rendered table here is
            // still data a caller parses, so it is never width-truncated.
            "output": exec::render_outcome(mode, outcome, None),
        })
    }
}

fn insert_request_id(value: &mut serde_json::Value, request_id: Option<serde_json::Value>) {
    let Some(id) = request_id else {
        return;
    };
    if let Some(obj) = value.as_object_mut() {
        obj.entry("id").or_insert(id);
    }
}

fn json_error(op: &str, message: String) -> serde_json::Value {
    serde_json::json!({"ok": false, "op": op, "error": message})
}

fn request_string(request: &serde_json::Value, key: &str) -> Result<String> {
    request
        .get(key)
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("missing string field {key:?}"))
}

fn session_agent_options(
    request: &serde_json::Value,
    default_format: OutputFormat,
) -> Result<Option<agent_response::AgentOptions>> {
    let response = request.get("response");
    let explicit_format = request.get("format");
    if response.is_none() {
        if explicit_format.and_then(serde_json::Value::as_str) == Some("agent") {
            return response_options_from_json(None).map(Some);
        }
        if explicit_format.is_some() || default_format != OutputFormat::Agent {
            return Ok(None);
        }
        return response_options_from_json(None).map(Some);
    }
    let format = explicit_format
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("format must be a string"))
        })
        .transpose()?
        .map(parse_session_format)
        .transpose()?
        .unwrap_or(default_format);
    if format != OutputFormat::Agent {
        if response.is_some() {
            anyhow::bail!("response controls require format=agent");
        }
        return Ok(None);
    }
    response_options_from_json(response).map(Some)
}

fn session_agent_intent(
    op: &str,
    request: &serde_json::Value,
    default_format: OutputFormat,
) -> bool {
    if op == "response_expand" {
        return true;
    }
    matches!(op, "query" | "write")
        && (default_format == OutputFormat::Agent
            || request.get("response").is_some()
            || request.get("format").and_then(serde_json::Value::as_str) == Some("agent"))
}

fn session_effective_agent_options(request: &serde_json::Value) -> agent_response::AgentOptions {
    response_options_from_json(request.get("response")).unwrap_or_default()
}

fn response_options_from_json(
    value: Option<&serde_json::Value>,
) -> Result<agent_response::AgentOptions> {
    let Some(value) = value else {
        return Ok(agent_response::AgentOptions::default());
    };
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("response must be an object"))?;
    let mode = object
        .get("mode")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("bounded");
    let max_bytes = object
        .get("max_bytes")
        .map(|value| {
            value
                .as_u64()
                .and_then(|number| usize::try_from(number).ok())
                .ok_or_else(|| anyhow::anyhow!("response.max_bytes must be a non-negative integer"))
        })
        .transpose()?;
    let full = match mode {
        "bounded" => false,
        "full" => true,
        other => anyhow::bail!("unknown response mode {other:?}; use bounded or full"),
    };
    if full && max_bytes.is_some() {
        anyhow::bail!("response.max_bytes cannot be combined with response mode full");
    }
    let options = agent_response::AgentOptions { max_bytes, full };
    options.validate()?;
    Ok(options)
}

fn parse_session_format(value: &str) -> Result<OutputFormat> {
    match value {
        "table" => Ok(OutputFormat::Table),
        "csv" => Ok(OutputFormat::Csv),
        "json" => Ok(OutputFormat::Json),
        "agent" => Ok(OutputFormat::Agent),
        other => anyhow::bail!("unknown format {other:?}; use table, csv, json, or agent"),
    }
}

fn session_mode(request: &serde_json::Value, default_format: OutputFormat) -> Result<Mode> {
    let fallback = match default_format {
        OutputFormat::Agent => Mode::Json,
        other => other.into(),
    };
    Ok(request
        .get("format")
        .and_then(serde_json::Value::as_str)
        .and_then(Mode::parse)
        .unwrap_or(fallback))
}

fn describe_options_from_json(request: &serde_json::Value) -> Result<DescribeOptions> {
    Ok(DescribeOptions {
        types: request.get("types").and_then(json_string_vec),
        type_search: request
            .get("type_search")
            .and_then(|v| v.as_str().map(str::to_string)),
        connections: detail_from_json(request.get("connections"), detail_connections(false, None))?,
        cypher: detail_from_json(request.get("cypher"), detail_cypher(false, None))?,
        fluent: detail_from_json(request.get("fluent"), detail_fluent(false, None))?,
        max_pairs: request
            .get("max_pairs")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize),
        sample_truncate: request
            .get("sample_truncate")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .or(Some(40)),
    })
}

fn json_string_vec(value: &serde_json::Value) -> Option<Vec<String>> {
    if let Some(s) = value.as_str() {
        return parse_csv(Some(s));
    }
    value.as_array().map(|items| {
        items
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect()
    })
}

trait DetailFromTopics: Sized {
    fn off() -> Self;
    fn overview() -> Self;
    fn topics(topics: Vec<String>) -> Self;
}

impl DetailFromTopics for ConnectionDetail {
    fn off() -> Self {
        ConnectionDetail::Off
    }
    fn overview() -> Self {
        ConnectionDetail::Overview
    }
    fn topics(topics: Vec<String>) -> Self {
        ConnectionDetail::Topics(topics)
    }
}

impl DetailFromTopics for CypherDetail {
    fn off() -> Self {
        CypherDetail::Off
    }
    fn overview() -> Self {
        CypherDetail::Overview
    }
    fn topics(topics: Vec<String>) -> Self {
        CypherDetail::Topics(topics)
    }
}

impl DetailFromTopics for FluentDetail {
    fn off() -> Self {
        FluentDetail::Off
    }
    fn overview() -> Self {
        FluentDetail::Overview
    }
    fn topics(topics: Vec<String>) -> Self {
        FluentDetail::Topics(topics)
    }
}

fn detail_from_json<T: DetailFromTopics>(
    value: Option<&serde_json::Value>,
    default: T,
) -> Result<T> {
    match value {
        None | Some(serde_json::Value::Null) => Ok(default),
        Some(serde_json::Value::Bool(false)) => Ok(T::off()),
        Some(serde_json::Value::Bool(true)) => Ok(T::overview()),
        Some(serde_json::Value::Object(obj)) => detail_from_object(obj),
        Some(v) => json_string_vec(v)
            .map(T::topics)
            .ok_or_else(|| anyhow::anyhow!("detail must be bool, string, string array, or object")),
    }
}

fn detail_from_object<T: DetailFromTopics>(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> Result<T> {
    if let Some(types) = obj
        .get("types")
        .or_else(|| obj.get("topics"))
        .or_else(|| obj.get("names"))
    {
        return json_string_vec(types)
            .map(T::topics)
            .ok_or_else(|| anyhow::anyhow!("detail topics must be string or string array"));
    }

    let detail = obj
        .get("detail")
        .or_else(|| obj.get("mode"))
        .and_then(|v| v.as_str())
        .unwrap_or("overview");
    match detail {
        "off" | "none" | "false" => Ok(T::off()),
        "overview" | "true" => Ok(T::overview()),
        "topics" | "types" => obj
            .get("value")
            .or_else(|| obj.get("values"))
            .and_then(json_string_vec)
            .map(T::topics)
            .ok_or_else(|| {
                anyhow::anyhow!("detail='{detail}' requires value as string or string array")
            }),
        other => Err(anyhow::anyhow!(
            "unknown detail {other:?}; use off, overview, or topics"
        )),
    }
}

fn parse_csv(raw: Option<&str>) -> Option<Vec<String>> {
    raw.map(|s| {
        s.split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(str::to_string)
            .collect()
    })
    .filter(|v: &Vec<String>| !v.is_empty())
}

fn detail_connections(overview: bool, topics: Option<&str>) -> ConnectionDetail {
    match parse_csv(topics) {
        Some(v) => ConnectionDetail::Topics(v),
        None if overview => ConnectionDetail::Overview,
        None => ConnectionDetail::Off,
    }
}

fn detail_cypher(overview: bool, topics: Option<&str>) -> CypherDetail {
    match parse_csv(topics) {
        Some(v) => CypherDetail::Topics(v),
        None if overview => CypherDetail::Overview,
        None => CypherDetail::Off,
    }
}

fn detail_fluent(overview: bool, topics: Option<&str>) -> FluentDetail {
    match parse_csv(topics) {
        Some(v) => FluentDetail::Topics(v),
        None if overview => FluentDetail::Overview,
        None => FluentDetail::Off,
    }
}

fn cypher_string(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\'', "\\'")
}

#[cfg(test)]
mod tests {
    use super::{session_save, validate_agent_cli, Cli};
    use clap::Parser;
    use kglite::api::io::{GraphFileIdentity, WriteOwnership};
    use kglite::api::DirGraph;
    use std::fs;
    use std::sync::Arc;

    #[test]
    fn ad_hoc_save_rejects_lost_update() {
        let tmp = tempfile::tempdir().unwrap();
        let graph = tmp.path().join("demo.kgl");
        let mut initial = Arc::new(DirGraph::new());
        kglite::api::io::save_graph(&mut initial, &graph.to_string_lossy()).unwrap();
        let mut working = initial.clone();
        let mut ownership = WriteOwnership::new(
            graph.clone(),
            GraphFileIdentity::capture(&graph).unwrap(),
            &working,
            None,
            false,
        );

        fs::write(&graph, b"competing writer").unwrap();
        let error = session_save(&mut working, &mut ownership).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("changed since this session loaded"),
            "unexpected refusal text: {error}"
        );
        assert_eq!(fs::read(&graph).unwrap(), b"competing writer");
    }

    #[test]
    fn agent_response_controls_validate_before_command_execution() {
        let valid = Cli::try_parse_from([
            "kglite",
            "query",
            "missing.kgl",
            "RETURN 1",
            "--format",
            "agent",
            "--response-max-bytes",
            "4096",
        ])
        .unwrap();
        validate_agent_cli(&valid).unwrap();

        let below_minimum = Cli::try_parse_from([
            "kglite",
            "write",
            "missing.kgl",
            "CREATE (:N)",
            "--format",
            "agent",
            "--response-max-bytes",
            "4095",
        ])
        .unwrap();
        assert!(validate_agent_cli(&below_minimum).is_err());

        let wrong_format = Cli::try_parse_from([
            "kglite",
            "write",
            "missing.kgl",
            "CREATE (:N)",
            "--response-full",
        ])
        .unwrap();
        assert!(validate_agent_cli(&wrong_format).is_err());
    }

    #[test]
    fn clap_rejects_conflicting_agent_controls_and_requires_purge_all() {
        assert!(Cli::try_parse_from([
            "kglite",
            "query",
            "missing.kgl",
            "RETURN 1",
            "--format",
            "agent",
            "--response-full",
            "--response-max-bytes",
            "8192",
        ])
        .is_err());
        assert!(Cli::try_parse_from(["kglite", "response", "purge"]).is_err());
        assert!(Cli::try_parse_from(["kglite", "response", "purge", "--all"]).is_ok());
    }
}
