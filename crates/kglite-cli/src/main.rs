//! `kglite` — an interactive Cypher shell for `.kgl` knowledge graphs, in the
//! spirit of the `sqlite3` CLI: open a single file, run queries and
//! dot-commands from the terminal, no Python or server required.
//!
//! Pure-Rust binary over `kglite::api::*` (no libpython link), mirroring the
//! kglite-bolt-server / kglite-mcp-server crate pattern.

mod code_tree_cli;
mod exec;
mod format;
mod helper;
mod repl;
mod skill;

use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use kglite::api::introspection::{
    compute_description, ConnectionDetail, CypherDetail, FluentDetail,
};
use kglite::api::io::{
    load_file, open_or_create_graph, save_graph, GraphFileIdentity, GraphWriterLease,
    OpenDisposition,
};
use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};
use kglite::api::{DirGraph, Value};

use crate::exec::QueryOptions;
use crate::format::Mode;

const WRITE_LOCK_TIMEOUT: Duration = Duration::from_secs(30);

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
    /// Build and inspect code-graph artifacts without Python or MCP.
    CodeTree {
        #[command(subcommand)]
        command: code_tree_cli::CodeTreeCommand,
    },
    /// Install or remove the bundled code-review skill for an agent host.
    Skill {
        #[command(subcommand)]
        command: skill::SkillCommand,
    },
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
        /// Comma-separated node-type whitelist for write requests.
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

    if let Some(Command::CodeTree { command }) = &cli.command {
        return code_tree_cli::run(command);
    }
    if let Some(Command::Skill { command }) = &cli.command {
        return skill::run(command);
    }
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
            (*format).into(),
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

    let (graph, source, source_identity) = match &cli.graph {
        Some(path) => {
            let p = path.to_string_lossy().to_string();
            let opened = open_or_create_graph(path, Some(StorageMode::Memory))
                .with_context(|| format!("failed to open or create {}", path.display()))?;
            if opened.disposition == OpenDisposition::Created {
                eprintln!("note: {p} does not exist — starting an empty in-memory graph");
            }
            let source = (opened.disposition == OpenDisposition::Opened).then_some(p);
            let identity = source.as_ref().map(|_| opened.identity);
            (opened.graph, source, identity)
        }
        None => (Arc::new(fresh_graph()?), None, None),
    };

    repl::run(graph, source.as_deref(), source_identity)
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
    let _lease = if persist {
        Some(GraphWriterLease::acquire(path, WRITE_LOCK_TIMEOUT)?)
    } else {
        None
    };
    let mut graph = open_or_create_graph(path, persist.then_some(StorageMode::Memory))
        .with_context(|| format!("failed to open or create {}", path.display()))?
        .graph;
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
    run_query(path, &query, mode)
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
        options.types.as_deref(),
        &options.connections,
        &options.cypher,
        &options.fluent,
        options.type_search.as_deref(),
        options.max_pairs,
        options.sample_truncate,
    )
    .map_err(|e| anyhow::anyhow!("describe failed: {e}"))
}

fn run_session(
    path: &Path,
    default_mode: Mode,
    save_on_exit: bool,
    write_scope: Option<&str>,
    git_sha: Option<String>,
    modified_by: Option<String>,
) -> Result<()> {
    let lease = if save_on_exit {
        Some(GraphWriterLease::acquire(path, WRITE_LOCK_TIMEOUT)?)
    } else {
        None
    };
    let mut graph = open_or_create_graph(path, save_on_exit.then_some(StorageMode::Memory))
        .with_context(|| format!("failed to open or create {}", path.display()))?
        .graph;
    let mut source_identity = GraphFileIdentity::capture(path)?;
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
            default_mode,
            &base_options,
            &mut source_identity,
            lease.is_some(),
        ) {
            SessionAction::Continue(value) => write_json_line(value)?,
            SessionAction::Exit(value) => {
                write_json_line(value)?;
                if save_on_exit {
                    save_loaded_graph(&mut graph, path, &mut source_identity, lease.is_some())?;
                }
                return Ok(());
            }
        }
    }
    if save_on_exit {
        save_loaded_graph(&mut graph, path, &mut source_identity, lease.is_some())?;
    }
    Ok(())
}

enum SessionAction {
    Continue(serde_json::Value),
    Exit(serde_json::Value),
}

fn handle_session_line(
    graph: &mut Arc<DirGraph>,
    path: &Path,
    line: &str,
    default_mode: Mode,
    base_options: &QueryOptions,
    source_identity: &mut GraphFileIdentity,
    lease_held: bool,
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
    let result = match op {
        "query" => session_query(graph, &request, mode_from_request(&request, default_mode)),
        "write" => session_write(
            graph,
            &request,
            mode_from_request(&request, default_mode),
            base_options,
        ),
        "describe" => session_describe(graph, &request),
        "save" => save_loaded_graph(graph, path, source_identity, lease_held)
            .map(|()| serde_json::json!({"ok": true, "op": "save"})),
        "exit" | "quit" => {
            let mut value = serde_json::json!({"ok": true, "op": op});
            insert_request_id(&mut value, request_id);
            return SessionAction::Exit(value);
        }
        other => Err(anyhow::anyhow!("unknown op {other:?}")),
    };
    SessionAction::Continue(match result {
        Ok(mut value) => {
            if let Some(obj) = value.as_object_mut() {
                obj.entry("op").or_insert_with(|| serde_json::json!(op));
            }
            insert_request_id(&mut value, request_id);
            value
        }
        Err(e) => {
            let mut value = json_error(op, e.to_string());
            insert_request_id(&mut value, request_id);
            value
        }
    })
}

fn session_query(
    graph: &Arc<DirGraph>,
    request: &serde_json::Value,
    mode: Mode,
) -> Result<serde_json::Value> {
    let query = request_string(request, "query")?;
    let (_, is_mutation) = kglite::api::cypher::parse_with_mutation_check(&query)
        .map_err(|e| anyhow::anyhow!("Cypher parse error: {e}"))?;
    if is_mutation {
        anyhow::bail!("query is read-only; use op=write for mutations");
    }
    let params = HashMap::new();
    let outcome = exec::execute_readonly(graph, &query, &params)?;
    Ok(session_outcome_response(mode, &outcome))
}

fn session_write(
    graph: &mut Arc<DirGraph>,
    request: &serde_json::Value,
    mode: Mode,
    base_options: &QueryOptions,
) -> Result<serde_json::Value> {
    let query = request_string(request, "query")?;
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
    let outcome = exec::execute(graph, &query, &params, &options)?;
    Ok(session_outcome_response(mode, &outcome))
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

fn save_loaded_graph(
    graph: &mut Arc<DirGraph>,
    path: &Path,
    source_identity: &mut GraphFileIdentity,
    lease_held: bool,
) -> Result<()> {
    let _lease = (!lease_held)
        .then(|| GraphWriterLease::acquire(path, WRITE_LOCK_TIMEOUT))
        .transpose()?;
    let current = GraphFileIdentity::capture(path)?;
    if current != *source_identity {
        anyhow::bail!(
            "refusing to overwrite {}: it changed since this session loaded it",
            path.display()
        );
    }
    let p = path.to_string_lossy().to_string();
    save_graph(graph, &p).map_err(|e| anyhow::anyhow!("failed to save {p}: {e}"))?;
    *source_identity = GraphFileIdentity::capture(path)?;
    Ok(())
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
            "output": exec::render_outcome(mode, outcome),
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

fn mode_from_request(request: &serde_json::Value, default_mode: Mode) -> Mode {
    request
        .get("format")
        .and_then(|v| v.as_str())
        .and_then(Mode::parse)
        .unwrap_or(default_mode)
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
    use super::save_loaded_graph;
    use kglite::api::io::GraphFileIdentity;
    use kglite::api::DirGraph;
    use std::fs;
    use std::sync::Arc;

    #[test]
    fn ad_hoc_save_rejects_lost_update() {
        let tmp = tempfile::tempdir().unwrap();
        let graph = tmp.path().join("demo.kgl");
        let mut initial = Arc::new(DirGraph::new());
        kglite::api::io::save_graph(&mut initial, &graph.to_string_lossy()).unwrap();
        let mut identity = GraphFileIdentity::capture(&graph).unwrap();
        let mut working = initial.clone();

        fs::write(&graph, b"competing writer").unwrap();
        let error = save_loaded_graph(&mut working, &graph, &mut identity, false).unwrap_err();

        assert!(error
            .to_string()
            .contains("changed since this session loaded"));
        assert_eq!(fs::read(&graph).unwrap(), b"competing writer");
    }
}
