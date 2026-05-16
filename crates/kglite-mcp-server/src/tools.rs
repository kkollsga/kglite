//! KGLite-specific MCP tools: `cypher_query`, `graph_overview`, `save_graph`.
//!
//! All three close over a [`GraphState`] holding the active
//! [`kglite::api::KnowledgeGraph`] behind an `Arc<RwLock<…>>`. Wired
//! into the framework's tool router via `register_typed_tool` so they
//! sit alongside the built-in source / GitHub tools.
//!
//! 0.9.18: rewritten against the pure-Rust `kglite::api` surface.
//! There is no `Python::attach` anywhere in this module — the binary
//! has no libpython link at all.

use std::path::Path;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use kglite::api::cypher;
use kglite::api::{
    compute_description, compute_schema, load_file, ConnectionDetail, CypherDetail, Embedder,
    FluentDetail, KnowledgeGraph, Value,
};
use mcp_methods::server::McpServer;
use serde::{Deserialize, Serialize};

const NO_GRAPH: &str =
    "No active graph. Pass --graph X.kgl, or activate one via repo_management('org/repo').";

/// Shared active-graph state. Cloning is cheap (Arc).
#[derive(Clone, Default)]
pub struct GraphState {
    inner: Arc<RwLock<Option<ActiveGraph>>>,
}

struct ActiveGraph {
    kg: KnowledgeGraph,
    source_path: Option<std::path::PathBuf>,
}

impl GraphState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load_kgl(&self, path: &Path) -> Result<()> {
        let kg = load_file(&path.to_string_lossy())
            .map_err(|e| anyhow::anyhow!("kglite::load_file failed: {}", e))?;
        *self.inner.write().unwrap() = Some(ActiveGraph {
            kg,
            source_path: Some(path.to_path_buf()),
        });
        Ok(())
    }

    pub fn build_code_tree(&self, dir: &Path) -> Result<()> {
        let kg = kglite::api::build_code_tree(dir, false, true, None, None)
            .map_err(|e| anyhow::anyhow!("kglite::build_code_tree failed: {}", e))?;
        *self.inner.write().unwrap() = Some(ActiveGraph {
            kg,
            source_path: None,
        });
        Ok(())
    }

    pub fn bind_embedder(&self, embedder: Arc<dyn Embedder>) -> Result<()> {
        let mut guard = self.inner.write().unwrap();
        let Some(active) = guard.as_mut() else {
            tracing::warn!("embedder loaded before any graph is active; binding deferred");
            return Ok(());
        };
        active.kg.set_embedder_native(embedder);
        Ok(())
    }

    pub fn schema(&self) -> Option<(u64, u64)> {
        let guard = self.inner.read().unwrap();
        let active = guard.as_ref()?;
        let overview = compute_schema(active.kg.dir());
        Some((overview.node_count as u64, overview.edge_count as u64))
    }

    /// Whether the active graph has at least one node of the named
    /// type. Returns `false` when no graph is active. Backs the
    /// `graph_has_node_type:` predicate for skill `applies_when:`
    /// gating (0.9.31 / mcp-methods 0.3.36).
    pub fn has_node_type(&self, node_type: &str) -> bool {
        let guard = self.inner.read().unwrap();
        guard
            .as_ref()
            .map(|active| active.kg.dir().has_node_type(node_type))
            .unwrap_or(false)
    }

    /// Whether the active graph's node-type metadata for `node_type`
    /// contains an entry for `prop_name`. Returns `false` when no
    /// graph is active or the type doesn't exist. Backs the
    /// `graph_has_property:` predicate for skill `applies_when:`
    /// gating.
    pub fn has_property(&self, node_type: &str, prop_name: &str) -> bool {
        let guard = self.inner.read().unwrap();
        guard
            .as_ref()
            .map(|active| {
                active
                    .kg
                    .dir()
                    .get_node_type_metadata(node_type)
                    .map(|meta| meta.contains_key(prop_name))
                    .unwrap_or(false)
            })
            .unwrap_or(false)
    }

    fn with_active<F>(&self, f: F) -> String
    where
        F: FnOnce(&ActiveGraph) -> String,
    {
        let guard = self.inner.read().unwrap();
        match guard.as_ref() {
            Some(active) => f(active),
            None => NO_GRAPH.to_string(),
        }
    }

    /// Borrow the active `KnowledgeGraph` for read-only inspection.
    /// Returns `None` when no graph is loaded — callers format their
    /// own "no graph active" message so the surrounding tool can give
    /// a tool-specific hint.
    pub fn with_kg<F, T>(&self, f: F) -> Option<T>
    where
        F: FnOnce(&kglite::api::KnowledgeGraph) -> T,
    {
        let guard = self.inner.read().unwrap();
        guard.as_ref().map(|active| f(&active.kg))
    }

    /// Resolve a code-entity qualified name to its source location via
    /// `KnowledgeGraph::source_location`. Used by the `read_code_source`
    /// tool to bridge the qualified-name → file path lookup.
    pub fn source_lookup(
        &self,
        qualified_name: &str,
        node_type: Option<&str>,
    ) -> Result<crate::code_source::SourceLookup, String> {
        let guard = self.inner.read().unwrap();
        let Some(active) = guard.as_ref() else {
            return Err(NO_GRAPH.to_string());
        };
        match active.kg.source_location(qualified_name, node_type) {
            kglite::api::SourceLookup::Found(loc) => {
                let file_path = loc.file_path.ok_or_else(|| {
                    format!("graph.source({qualified_name:?}) returned no file_path")
                })?;
                let line_number = loc.line_number.unwrap_or(1).max(1) as usize;
                let end_line = loc.end_line.unwrap_or(loc.line_number.unwrap_or(1)).max(1) as usize;
                Ok(crate::code_source::SourceLookup {
                    file_path,
                    line_number,
                    end_line,
                })
            }
            kglite::api::SourceLookup::Ambiguous(matches) => Err(format!(
                "ambiguous qualified_name {qualified_name:?}; matches: {matches:?}. \
                 Pass `node_type` to narrow."
            )),
            kglite::api::SourceLookup::NotFound => Err(format!(
                "graph.source({qualified_name:?}) returned no match. \
                 Try passing `node_type` or using a different qualified name."
            )),
        }
    }

    /// Run a parameterised Cypher template against the active graph.
    /// Used by the YAML-declared `tools[].cypher` registration path
    /// (see [`crate::cypher_tools::register_cypher_tools`]).
    pub fn run_cypher_template(
        &self,
        template: &str,
        args: &serde_json::Map<String, serde_json::Value>,
        csv_http: Option<&crate::csv_http::CsvHttpConfig>,
    ) -> String {
        let guard = self.inner.read().unwrap();
        let Some(active) = guard.as_ref() else {
            return NO_GRAPH.to_string();
        };
        let mut params = std::collections::HashMap::new();
        for (k, v) in args {
            params.insert(k.clone(), json_to_value(v));
        }
        match run_cypher_inner(&active.kg, template, params, csv_http) {
            Ok(body) => body,
            Err(e) => format!("Cypher error: {e}"),
        }
    }
}

/// Convert a `serde_json::Value` into a Cypher param `Value`. Mirrors
/// the Python boundary's `py_value_to_value` for the JSON subset.
fn json_to_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Boolean(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int64(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float64(f)
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => Value::String(s.clone()),
        // Arrays and objects flow through as JSON-serialised strings; the
        // Cypher engine doesn't have a first-class list/map Value variant
        // at the param boundary, so this matches the existing behaviour
        // for non-scalar JSON inputs from MCP tool calls.
        other => Value::String(other.to_string()),
    }
}

/// Run a Cypher query against the given KnowledgeGraph snapshot. Picks
/// between read and write paths based on `is_mutation_query`; on success
/// returns the rendered tool body (CSV when `FORMAT CSV` is in the
/// query, inline 15-row preview otherwise).
fn run_cypher_inner(
    kg: &KnowledgeGraph,
    query: &str,
    params: std::collections::HashMap<String, Value>,
    csv_http: Option<&crate::csv_http::CsvHttpConfig>,
) -> Result<String, String> {
    let mut parsed =
        cypher::parse_cypher(query).map_err(|e| format!("Cypher syntax error: {e}"))?;
    let mut params = params;

    let rewrite = cypher::rewrite_text_score(&mut parsed, &params)?;
    if !rewrite.texts_to_embed.is_empty() && !parsed.explain {
        let model = kg
            .embedder()
            .ok_or_else(|| {
                "text_score() requires a registered embedding model. \
                 Configure `extensions.embedder:` in the manifest."
                    .to_string()
            })?
            .clone();
        model.load()?;
        let texts: Vec<String> = rewrite
            .texts_to_embed
            .iter()
            .map(|(_, t)| t.clone())
            .collect();
        let embeddings = model.embed(&texts);
        model.unload();
        let embeddings = embeddings?;
        if embeddings.len() != texts.len() {
            return Err(format!(
                "text_score: model.embed() returned {} vectors for {} texts",
                embeddings.len(),
                texts.len()
            ));
        }
        for (i, (param_name, _)) in rewrite.texts_to_embed.iter().enumerate() {
            let json = format!(
                "[{}]",
                embeddings[i]
                    .iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            params.insert(param_name.clone(), Value::String(json));
        }
    }

    cypher::planner::optimize_with_disabled(
        &mut parsed,
        kg.dir(),
        &params,
        cypher::planner::empty_disabled_set(),
    );
    cypher::mark_lazy_eligibility(&mut parsed);

    let output_csv = parsed.output_format == cypher::OutputFormat::Csv;

    let result = if cypher::is_mutation_query(&parsed) {
        // Mutation requires &mut DirGraph; the shim holds an
        // Arc<DirGraph> snapshot via the KnowledgeGraph, so we cannot
        // mutate it here without a write lock. The shim today only
        // exposes mutating queries when explicitly invoked through
        // `cypher_query`, and the surrounding RwLock<Option<ActiveGraph>>
        // is held read-only across the call. For 0.9.18 we forbid
        // mutations through the MCP tool surface — agents that need to
        // edit the graph use a CLI shell instead.
        return Err(
            "mutation Cypher (CREATE/SET/DELETE/REMOVE/MERGE) is not allowed through \
             the MCP cypher_query tool. Use the kglite CLI for graph edits."
                .to_string(),
        );
    } else {
        let executor = cypher::CypherExecutor::with_params(kg.dir(), &params, None);
        executor
            .execute(&parsed)
            .map_err(|e| format!("Cypher execution error: {e}"))?
    };

    if output_csv {
        let csv = result.to_csv();
        if let Some(cfg) = csv_http {
            match crate::csv_http::write_csv(cfg, &csv) {
                Ok(name) => {
                    let url = cfg.url_for(&name);
                    // 0.9.19 fix: count rows from the CSV body, not from
                    // `result.rows.len()`. The planner's lazy_eligible
                    // pass leaves `rows` empty for simple
                    // MATCH-RETURN-LIMIT queries and materialises through
                    // the lazy descriptor (or streaming pipeline) — the
                    // CSV is correct but `rows.len()` reads 0 and the
                    // operator-facing status says "0 row(s) written".
                    // Counting newlines in the CSV agrees with what the
                    // file actually contains.
                    let row_count = count_csv_rows(&csv);
                    Ok(format!(
                        "FORMAT CSV: {row_count} row(s) written to {url}\n\
                         Fetch with: curl {url}"
                    ))
                }
                Err(e) => {
                    tracing::warn!(error = %e, "csv_http write_csv failed; falling back to inline");
                    Ok(csv)
                }
            }
        } else {
            Ok(csv)
        }
    } else {
        Ok(format_cypher_inline(&result))
    }
}

/// Render a CypherResult as an inline 15-row preview (header + repr per
/// row). Matches the format the pre-0.9.18 Python shim produced via
/// `format_cypher_result`.
fn format_cypher_inline(result: &cypher::CypherResult) -> String {
    let len = result.rows.len();
    if len == 0 {
        return "No results.".to_string();
    }
    let header = if len > 15 {
        format!("{len} row(s) (showing first 15):\n")
    } else {
        format!("{len} row(s):\n")
    };
    let mut out = header;
    out.push_str(&result.columns.join("\t"));
    out.push('\n');
    for row in result.rows.iter().take(15) {
        for (i, val) in row.iter().enumerate() {
            if i > 0 {
                out.push('\t');
            }
            push_value_repr(&mut out, val);
        }
        out.push('\n');
    }
    out
}

/// Count data rows in a CSV string, defined as (newline-terminated lines) - 1
/// for the header. Trailing newlines after the last row don't add to the
/// count. Handles the edge cases: empty string → 0, header-only → 0,
/// header + N rows → N. Quoted newlines inside cells aren't recognised
/// here — kglite's `csv_value` doesn't emit Value variants that contain
/// embedded newlines, so a plain `lines()` count agrees with row count.
fn count_csv_rows(csv: &str) -> usize {
    let line_count = csv.lines().count();
    line_count.saturating_sub(1)
}

fn push_value_repr(out: &mut String, val: &Value) {
    use std::fmt::Write;
    match val {
        Value::Null => out.push_str("null"),
        Value::String(s) => {
            let _ = write!(out, "{s:?}");
        }
        Value::Int64(n) => {
            let _ = write!(out, "{n}");
        }
        Value::Float64(f) => {
            let _ = write!(out, "{f}");
        }
        Value::Boolean(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::UniqueId(u) => {
            let _ = write!(out, "{u}");
        }
        Value::DateTime(d) => out.push_str(&d.format("%Y-%m-%d").to_string()),
        Value::Point { lat, lon } => {
            let _ = write!(out, "POINT({lon} {lat})");
        }
        Value::Duration {
            months,
            days,
            seconds,
        } => {
            let _ = write!(out, "duration(M={months}, D={days}, S={seconds})");
        }
        Value::NodeRef(idx) => {
            let _ = write!(out, "node[{idx}]");
        }
    }
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
struct CypherArgs {
    /// Cypher query string. Append `FORMAT CSV` for CSV-encoded output.
    pub query: String,
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
struct OverviewArgs {
    /// Drill into specific node types (e.g. `["Person", "Document"]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub types: Option<Vec<String>>,
    /// `true` for all connection types; or `["CALLS"]` for a deep-dive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connections: Option<serde_json::Value>,
    /// `true` for the Cypher language reference; or `["MATCH","WHERE"]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cypher: Option<serde_json::Value>,
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
struct SaveGraphArgs {}

/// Builtins toggled by the manifest's `builtins:` block.
#[derive(Clone, Debug, Default)]
pub struct Builtins {
    pub save_graph: bool,
    pub temp_cleanup_on_overview: bool,
    /// Directory wiped by `temp_cleanup: on_overview`. Resolved against
    /// the manifest's parent in `main.rs` — when csv_http_server is
    /// configured we reuse its directory (so the same place CSVs are
    /// written is also the place they get swept). Falls back to
    /// `<manifest_dir>/temp/` when csv_http_server isn't set.
    pub temp_dir: Option<std::path::PathBuf>,
}

pub fn register(
    server: &mut McpServer,
    state: GraphState,
    builtins: Builtins,
    csv_http: Option<Arc<crate::csv_http::CsvHttpConfig>>,
) {
    let s = state.clone();
    let csv = csv_http.clone();
    let cypher_desc: &'static str = if csv.is_some() {
        "Run a Cypher query against the active knowledge graph. Returns up to 15 rows \
         inline; append FORMAT CSV to export results — large CSVs are written to the \
         csv_http_server directory and returned as a fetch URL."
    } else {
        "Run a Cypher query against the active knowledge graph. Returns up to 15 rows \
         inline; append FORMAT CSV to export full results to a CSV string."
    };
    server.register_typed_tool::<CypherArgs, _>("cypher_query", cypher_desc, move |args| {
        let csv = csv.clone();
        s.with_active(|g| run_cypher_tool(g, &args.query, csv.as_deref()))
    });
    let s = state.clone();
    let cleanup_temp = builtins.temp_cleanup_on_overview;
    let temp_dir = builtins.temp_dir.clone();
    server.register_typed_tool::<OverviewArgs, _>(
        "graph_overview",
        "Inspect the active graph's schema. With no args returns the inventory; pass \
         types=[...] / connections=true|[...] / cypher=true|[...] for drill-down.",
        move |args| {
            if cleanup_temp
                && args.types.is_none()
                && args.connections.is_none()
                && args.cypher.is_none()
            {
                if let Some(dir) = temp_dir.as_deref() {
                    wipe_temp_dir(dir);
                }
            }
            s.with_active(|g| run_overview(g, &args))
        },
    );
    if builtins.save_graph {
        let s = state;
        server.register_typed_tool::<SaveGraphArgs, _>(
            "save_graph",
            "Persist the active graph to its source .kgl file (single-graph mode only).",
            move |_| s.with_active(run_save),
        );
    }
}

fn wipe_temp_dir(dir: &std::path::Path) {
    if !dir.is_dir() {
        tracing::debug!(dir = %dir.display(), "temp_cleanup: directory does not exist; nothing to wipe");
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            tracing::debug!(error = %e, dir = %dir.display(), "temp_cleanup: read_dir failed");
            return;
        }
    };
    let mut wiped = 0usize;
    for entry in entries.flatten() {
        let path = entry.path();
        let res = if path.is_dir() {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
        match res {
            Ok(()) => wiped += 1,
            Err(e) => {
                tracing::debug!(path = %path.display(), error = %e, "temp_cleanup: remove failed");
            }
        }
    }
    if wiped > 0 {
        tracing::info!(count = wiped, dir = %dir.display(), "temp_cleanup: wiped entries");
    }
}

fn run_cypher_tool(
    graph: &ActiveGraph,
    query: &str,
    csv_http: Option<&crate::csv_http::CsvHttpConfig>,
) -> String {
    match run_cypher_inner(&graph.kg, query, std::collections::HashMap::new(), csv_http) {
        Ok(s) => s,
        Err(e) => format!("Cypher error: {e}"),
    }
}

fn run_overview(graph: &ActiveGraph, args: &OverviewArgs) -> String {
    let conn = parse_connection_detail(args.connections.as_ref());
    let cy = parse_cypher_detail(args.cypher.as_ref());
    let fluent = FluentDetail::Off;
    match compute_description(
        graph.kg.dir(),
        args.types.as_deref(),
        &conn,
        &cy,
        &fluent,
        None,
        None,
        None,
    ) {
        Ok(s) => s,
        Err(e) => format!("graph_overview error: {e}"),
    }
}

fn parse_connection_detail(v: Option<&serde_json::Value>) -> ConnectionDetail {
    use serde_json::Value;
    match v {
        None | Some(Value::Null) => ConnectionDetail::Off,
        Some(Value::Bool(false)) => ConnectionDetail::Off,
        Some(Value::Bool(true)) => ConnectionDetail::Overview,
        Some(Value::Array(items)) => {
            let names: Vec<String> = items
                .iter()
                .filter_map(|i| i.as_str().map(String::from))
                .collect();
            if names.is_empty() {
                ConnectionDetail::Overview
            } else {
                ConnectionDetail::Topics(names)
            }
        }
        Some(_) => ConnectionDetail::Overview,
    }
}

fn parse_cypher_detail(v: Option<&serde_json::Value>) -> CypherDetail {
    use serde_json::Value;
    match v {
        None | Some(Value::Null) => CypherDetail::Off,
        Some(Value::Bool(false)) => CypherDetail::Off,
        Some(Value::Bool(true)) => CypherDetail::Overview,
        Some(Value::Array(items)) => {
            let names: Vec<String> = items
                .iter()
                .filter_map(|i| i.as_str().map(String::from))
                .collect();
            if names.is_empty() {
                CypherDetail::Overview
            } else {
                CypherDetail::Topics(names)
            }
        }
        Some(_) => CypherDetail::Overview,
    }
}

fn run_save(graph: &ActiveGraph) -> String {
    let Some(path) = graph.source_path.as_ref() else {
        return "save_graph requires --graph mode (no source path bound).".to_string();
    };
    let path_str = path.to_string_lossy().into_owned();
    // save_disk needs &mut DirGraph; we hold an Arc<DirGraph>. Clone the
    // Arc and unwrap to get exclusive access (cheap when the Arc is
    // already unique, otherwise we do a deep clone of the storage —
    // acceptable for save since it's an explicit operator action).
    let mut dir_arc = graph.kg.dir().clone();
    let dir = std::sync::Arc::make_mut(&mut dir_arc);
    match dir.save_disk(&path_str) {
        Ok(()) => {
            let overview = compute_schema(dir);
            format!(
                "Saved {path_str} ({} nodes, {} edges).",
                overview.node_count, overview.edge_count
            )
        }
        Err(e) => format!("save_graph error: {e}"),
    }
}
