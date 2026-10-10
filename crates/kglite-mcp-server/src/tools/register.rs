//! Tool registration: the manifest-driven builtin toggles, the
//! `graph_overview` decorations, and the router wiring for every KGLite
//! MCP route.

use crate::output_schema::ObjectOutputSchema;
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use kglite::api::storage::StorageMode;
use mcp_methods::server::McpServer;
use rmcp::handler::server::router::tool::ToolRoute;
use rmcp::handler::server::tool::ToolCallContext;
use rmcp::model::{CallToolResponse, CallToolResult, ContentBlock, Tool, ToolAnnotations};
use rmcp::ErrorData as McpError;
use serde::de::DeserializeOwned;

use crate::recipe_queries::CatalogHint;
use crate::tools::*;

type DynFut<'a, T> = Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

fn register_cypher_tool<A>(
    server: &mut McpServer,
    description: &'static str,
    writable: bool,
    handler: impl Fn(A) -> Result<CypherToolOutput, String> + Send + Sync + 'static,
) where
    A: DeserializeOwned + schemars::JsonSchema + Send + 'static,
{
    let tool = Tool::new_with_raw(
        "cypher_query",
        Some(description.into()),
        Arc::new(serde_json::Map::new()),
    )
    .with_input_schema::<A>()
    .with_object_output_schema::<CypherResultEnvelope>()
    .with_annotations(
        ToolAnnotations::new()
            .read_only(!writable)
            .destructive(writable)
            .idempotent(false)
            .open_world(false),
    );
    let handler = Arc::new(handler);
    server.tool_router_mut().add_route(ToolRoute::new_dyn(
        tool,
        move |ctx: ToolCallContext<'_, McpServer>| -> DynFut<'_, Result<CallToolResponse, McpError>> {
            let handler = handler.clone();
            let arguments = ctx.arguments.clone();
            Box::pin(async move {
                let args = serde_json::from_value(serde_json::Value::Object(
                    arguments.unwrap_or_default(),
                ));
                let result = match args {
                    Ok(args) => match handler(args) {
                        Ok(output) => output.into_call_tool_result(),
                        Err(text) => CallToolResult::error(vec![ContentBlock::text(text)]),
                    },
                    Err(error) => CallToolResult::error(vec![ContentBlock::text(format!(
                        "Invalid arguments: {error}"
                    ))]),
                };
                Ok(result.into())
            })
        },
    ));
}

/// Builtins toggled by the manifest's `builtins:` block.
#[derive(Clone, Debug, Default)]
pub struct Builtins {
    /// The `save_graph` route. Wider than [`Self::writable`]: a manifest that
    /// sets `builtins.save_graph: true` registers it on an otherwise read-only
    /// server (`server_run::owns_graph_file`), which is how a boot-time
    /// ontology materialization gets persisted.
    pub save_graph: bool,
    /// Write-enabled "agent graph workbench" mode — `--writable` or
    /// `extensions.writable: true`, resolved once in
    /// `server_run::boot_mutations_enabled`. When true, `cypher_query` accepts
    /// mutations (routed through the write-lock) and the runtime
    /// graph-lifecycle tools (`load_graph` / `create_graph` / `save_graph_as`)
    /// are registered. Off by default — read-only is the safe default for
    /// code-review / analysis deployments.
    pub writable: bool,
    /// Operator-pinned `write_scope` (CLI `--write-scope` /
    /// `extensions.write_scope`, intersected in `server_run::boot_write_scope`).
    /// `None` = the operator pinned nothing and the agent's own `write_scope`
    /// argument is the whole story. `Some(..)` is the ceiling the agent can
    /// only narrow — including when it supplies no scope at all. See
    /// [`resolve_write_scope`].
    pub write_scope: Option<Vec<String>>,
    pub temp_cleanup_on_overview: bool,
    /// Directory wiped by `temp_cleanup: on_overview`. Resolved against
    /// the manifest's parent in `main.rs` — when csv_http_server is
    /// configured we reuse its directory (so the same place CSVs are
    /// written is also the place they get swept). Falls back to
    /// `<manifest_dir>/temp/` when csv_http_server isn't set.
    pub temp_dir: Option<std::path::PathBuf>,
}

/// `save_graph`'s description on a write-enabled server. The two spellings
/// differ only in the `force` sentence: `run_save` refuses `force=true` when
/// mutations are off, and a description that advertised it anyway would send
/// the agent at a route this deployment does not have.
const SAVE_GRAPH_DESCRIPTION_WRITABLE: &str =
    "Persist the active graph to its source .kgl file (single-graph mode only). With \
     nothing unsaved to write this is a no-op that reports \"Nothing to save\" and leaves \
     the file untouched, so other servers reading the same graph are not made to re-read \
     it; pass force=true to rewrite it anyway.";

/// [`SAVE_GRAPH_DESCRIPTION_WRITABLE`] for a server registered through
/// `builtins.save_graph` alone — it may still publish unsaved changes and boot
/// configuration, but not re-encode an unchanged file.
const SAVE_GRAPH_DESCRIPTION_READ_ONLY: &str =
    "Persist the active graph to its source .kgl file (single-graph mode only). With \
     nothing unsaved to write this is a no-op that reports \"Nothing to save\" and leaves \
     the file untouched, so other servers reading the same graph are not made to re-read \
     it. force=true is refused on this server, which is not write-enabled.";

/// The bare-overview skills index, shared between the route closure that
/// renders it and the boot step that computes it.
///
/// A slot rather than a value because [`OverviewDecorations`] is built inside
/// `register_kglite_tools` and moved into the `graph_overview` closure, while
/// the index can only be computed after `install_skills` — which needs the
/// final tool surface the closed router describes.
pub(crate) type SkillsIndexSlot = Arc<RwLock<Option<String>>>;

/// The connected client's peer handle, published once `serve` has returned it.
///
/// A slot for the same reason [`SkillsIndexSlot`] is one, inverted in time: a
/// tool handler needs the peer to announce a rebuilt tool list, and the peer
/// only exists *after* every handler has been registered. Empty until then,
/// and empty forever on a transport that never connects.
pub(crate) type PeerSlot = Arc<RwLock<Option<rmcp::service::Peer<rmcp::RoleServer>>>>;

/// MCP-layer additions to the bare `graph_overview` response.
///
/// These describe the deployment, not the active graph, so they are captured
/// by the route at boot instead of entering [`GraphState`] or core
/// `describe()`.
#[derive(Clone, Debug, Default)]
pub(crate) struct OverviewDecorations {
    pub(crate) prefix: Option<String>,
    pub(crate) catalog: Option<CatalogHint>,
    /// Index of the skills this session actually serves, filled by
    /// `install_skills` at boot and **refreshed on every graph swap** by
    /// [`crate::skills::SkillRefresher`].
    ///
    /// Refreshing it is only honest because mcp-methods 0.4.11 can rebuild the
    /// prompt plane after `serve`: before that the served set was frozen at
    /// boot and re-rendering the index would have advertised skills the
    /// session could not serve. The tool *descriptions* move with it now, so
    /// the index and `prompts/list` still agree.
    pub(crate) skills: SkillsIndexSlot,
}

impl OverviewDecorations {
    pub(crate) fn render(&self, body: String, is_bare: bool) -> String {
        if !is_bare {
            return body;
        }

        let mut rendered = String::new();
        if let Some(prefix) = self.prefix.as_deref() {
            append_overview_section(&mut rendered, prefix);
        }
        append_overview_section(&mut rendered, &body);
        if let Some(hint) = self.catalog.as_ref() {
            // `names` needs no XML escaping: a recipe and a query name are
            // both `^[A-Za-z_][A-Za-z0-9_]*$` catalogue identifiers, checked
            // before a catalogue compiles.
            append_overview_section(
                &mut rendered,
                &format!(
                    "<query-catalog recipes=\"{}\" queries=\"{}\" \
                     list-tool=\"list_recipe_queries\" run-tool=\"run_recipe_query\" \
                     names=\"{}\"/>",
                    hint.summary.recipe_count,
                    hint.summary.query_count,
                    hint.names.join(", ")
                ),
            );
        }
        if let Some(index) = read_lock(&self.skills).as_deref() {
            append_overview_section(&mut rendered, index);
        }
        rendered
    }
}

/// Apply a body decoration to whichever arm carries the text.
///
/// Every decoration this module applies — the rebuild-staleness warning, the
/// bare-overview prefix and catalog hint — describes the *deployment*, not the
/// outcome, so a failed call needs it exactly as much as a successful one: an
/// error read against a graph the agent believes is fresh, or without the
/// discovery hint that would let it recover, is the wrong kind of unhelpful.
/// Decorating both arms is also what keeps the response text byte-identical to
/// the pre-`isError` shape, where an error *was* the body.
fn map_body(
    body: Result<String, String>,
    decorate: impl FnOnce(String) -> String,
) -> Result<String, String> {
    match body {
        Ok(body) => Ok(decorate(body)),
        Err(error) => Err(decorate(error)),
    }
}

pub(crate) fn append_overview_section(rendered: &mut String, section: &str) {
    if section.is_empty() {
        return;
    }
    if !rendered.is_empty() && !rendered.ends_with('\n') {
        rendered.push('\n');
    }
    rendered.push_str(section);
}

/// Apply bare-overview side effects and return the shared bare predicate for
/// response decoration. Keeping both decisions behind this function prevents
/// cleanup and sticky discovery from drifting onto different call shapes.
pub(crate) fn prepare_overview(
    args: &OverviewArgs,
    cleanup_temp: bool,
    temp_dir: Option<&std::path::Path>,
) -> bool {
    let is_bare = args.is_bare();
    if cleanup_temp && is_bare {
        if let Some(dir) = temp_dir {
            wipe_temp_dir(dir);
        }
    }
    is_bare
}

/// Register the tools that only make sense when the server was pointed at one
/// graph file (`--graph`): currently `reload_graph`.
///
/// Separate from [`register`] because the mode is a boot fact this module does
/// not otherwise see, and every other mode either has no source file to
/// re-read (bare) or owns its own freshness lifecycle (workspace modes rebuild
/// lazily from a producer). Called from `run_async`, which already does
/// mode-conditional router work.
///
/// Registered for read-only servers too — a read-only deployment is precisely
/// the one whose graph is rebuilt by *someone else*, so it needs the refresh
/// affordance most.
pub fn register_graph_mode_tools(
    server: &mut McpServer,
    state: GraphState,
    skills: crate::skills::SkillRefresher,
) {
    server.register_typed_tool_fallible::<ReloadGraphArgs, _>(
        "reload_graph",
        "Re-read the served graph file from disk, replacing the in-memory graph — use this \
         when the file has been rebuilt by another process and queries are returning stale \
         results. The path is the one this server was started on. If this server has unsaved \
         in-memory changes the reload is refused, because it would discard them: call \
         save_graph first to keep them, or pass discard_unsaved=true to drop them and serve \
         the file as it is on disk. If the re-read fails, the current graph stays active and \
         the error is returned.",
        move |args| match state.source_path() {
            // `open_or_create(path, None)`: no storage mode is requested, so a
            // reload never re-runs the boot `--storage` conversion — it serves
            // whatever the (possibly newly written) checkpoint records, exactly
            // as `load_graph` does. A load failure returns before the write
            // lock is taken, so the old graph provably stays active.
            Some(path) => match reload_disposition(&state, args.discard_unsaved) {
                Err(refusal) => Err(refusal),
                Ok(()) => match state.open_or_create(&path, None) {
                    Ok(_) => {
                        // The reloaded file carries its own `KgliteSkill`
                        // records; the ones injected at boot describe the graph
                        // that was just replaced.
                        skills.refresh();
                        let path = path.display();
                        let load = state
                            .load_count()
                            .map(|n| format!(" Load {n} on this server."))
                            .unwrap_or_default();
                        Ok(match state.schema() {
                            Some((n, e)) => {
                                format!("Reloaded {path} ({n} nodes, {e} edges).{load}")
                            }
                            None => format!("Reloaded {path}.{load}"),
                        })
                    }
                    Err(e) => Err(format!("reload_graph error: {e}")),
                },
            },
            None => Err(format!("reload_graph error: {NO_GRAPH}")),
        },
    );
}

/// Register the tools that only make sense when this binary builds the graph
/// from a directory it watches (`--vault`, and `--watch` with an injected
/// producer): currently `rebuild_graph`.
///
/// `reload_graph`'s counterpart, and deliberately not the same route: a
/// producer-backed graph has no served file to re-read, and the two failure
/// modes an agent hits are different — a stale `.kgl` on disk versus a watcher
/// that never saw an edit (a network mount, an editor that writes through a
/// temp file, a change made before the server booted).
pub fn register_vault_mode_tools(
    server: &mut McpServer,
    state: GraphState,
    skills: crate::skills::SkillRefresher,
    root: std::path::PathBuf,
    last_report: crate::vault::VaultReportSlot,
) {
    server.register_typed_tool_fallible::<RebuildGraphArgs, _>(
        "rebuild_graph",
        "Rebuild the served graph from the vault directory now, and report what the build \
         saw — use it after editing notes if queries still return the old content (the \
         watcher normally rebuilds on the next tool call by itself), or after editing \
         `.kglite/vault.yaml`, `.kglite/skills/` or `.kglite/recipes/`. Takes no arguments: \
         the directory is the one this server was started on. The reply is the build report \
         — notes scanned, nodes by label, edges by type, and any errors or warnings such as \
         dangling links or missing images. If the build fails the previous graph stays \
         active and the error is returned.",
        move |_args| {
            state
                .build_workspace_graph(&root, None)
                .map_err(|e| format!("rebuild_graph error: {e}"))?;
            // The rebuilt graph carries its own `KgliteSkill` records — the
            // ones resolved at boot describe the vault as it was. Same reason
            // `reload_graph` refreshes, and the same placement: after the
            // build call has returned every lock it took.
            skills.refresh();
            let report = last_report
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .map(|report| report.render());
            Ok(match report {
                Some(text) => text,
                // The slot *is* empty after a boot served from the vault's
                // cache — nothing was read, so there is no report — but not
                // here: this handler reaches the producer with a graph
                // already in hand, which is the case the producer always
                // rebuilds. Still not an `expect`: an agent that asked for a
                // rebuild got one, and a missing summary is no reason to
                // fail the call.
                None => "Rebuilt the vault graph.".to_string(),
            })
        },
    );
}

/// Settle what an incoming `reload_graph` does about unsaved changes, before
/// the re-read that would silently drop them.
///
/// `Ok(())` means the re-read may proceed — either nothing was unsaved, or the
/// caller asked for the discard and it has now happened. The discard is a
/// snapshot restore rather than a re-read, so it also releases the writer
/// lease: the peer waiting on it gets the file back even if the reload that
/// follows fails.
fn reload_disposition(state: &GraphState, discard_unsaved: bool) -> Result<(), String> {
    if !state.is_dirty() {
        return Ok(());
    }
    if !discard_unsaved {
        return Err(refused_while_dirty("reload_graph"));
    }
    state.discard_unsaved_changes();
    Ok(())
}

/// Refuse a route that would replace the active graph while this server holds
/// unsaved changes. `load_graph` and `create_graph` grow no flag of their own —
/// the one spelling for "throw my work away" is
/// `reload_graph(discard_unsaved=true)`, so an agent cannot reach it by
/// accident on a route whose purpose is something else.
fn refuse_swap_while_dirty(state: &GraphState, tool: &str) -> Result<(), String> {
    if state.is_dirty() {
        return Err(refused_while_dirty(tool));
    }
    Ok(())
}

/// Extend the write-enabled `cypher_query` description with the operator's
/// pinned write scope, naming the types so an agent can plan inside the
/// ceiling instead of discovering it one refusal at a time.
///
/// The framework's `register_typed_tool` takes a `&'static str`, and this
/// string is only knowable at boot. Leaking it is exact rather than merely
/// convenient: there is one per process, built once, and it must live as long
/// as the router that holds it — which is the whole process.
fn pinned_cypher_description(base: &str, pin: &[String]) -> &'static str {
    let scope = if pin.is_empty() {
        "an empty list — this server permits NO writes at all".to_string()
    } else {
        format!("[{}]", pin.join(", "))
    };
    format!(
        "{base} This server's operator has pinned write_scope to {scope}: a write_scope you \
         pass is intersected with it and can only narrow it, omitting write_scope leaves the \
         pinned scope in force, and a write with nothing left in scope is refused."
    )
    .leak()
}

/// Pick the `cypher_query` description for this server's actual capabilities.
///
/// Every arm leads with the code-exploration vocabulary agents search for
/// (explore, understand, "how does", call graph, "where defined", structure,
/// navigate) so lazy-tool-discovery clients (Codex / code_mode) surface
/// `cypher_query` on their first broad tool search instead of falling back to
/// grep. (mcp-servers inbox 2026-07-01.)
///
/// `csv_enabled` is `config().is_some()` — "a fetch URL is actually available",
/// not "the operator asked for one". A configured-but-failed listener reads as
/// absent here; its failure reason reaches the agent on the result itself
/// (`inline_csv_reason`), which a description loaded once at boot could not do.
/// An operator `pin` is part of the contract the agent plans against, so it is
/// stated here rather than left to surface as a refusal.
fn cypher_description(csv_enabled: bool, writable: bool, pin: Option<&[String]>) -> &'static str {
    let base: &'static str = match (csv_enabled, writable) {
        (_, true) => {
            "Query, explore, and understand the active knowledge graph with Cypher, and \
             modify it — reads AND mutations are accepted; this is a \
             write-enabled graph. The primary tool for structural questions: how things \
             relate, where an entity/function/type is defined, what references or calls what, \
             counts, and multi-hop paths (for code graphs: call graphs, definitions, imports — \
             navigate the codebase structure). Pass write_scope=[...] to restrict mutations \
             to those node types: every node write (CREATE, INSERT, MERGE, SET, REMOVE, DELETE, \
             NODETACH DELETE, DETACH DELETE, and node-type DDL) is judged by the node's stored \
             type, and a relationship write (edge CREATE/INSERT, DELETE r, SET r.p, REMOVE r.p) needs at least \
             one endpoint's type in the list. Pass params={...} to bind $placeholders — both \
             `{prop: $p}` inside a pattern and `WHERE x.prop = $p` read from it, and a \
             $name with no value is an error rather than an empty result. Mutations are in-memory; call \
             save_graph to persist. Returns up to 15 rows inline; append FORMAT CSV for a CSV body \
             capped at 200 rows with a notice naming the true total — narrow the query to fit."
        }
        (true, false) => {
            "Query, explore, and understand the active knowledge graph with Cypher — the \
             primary tool for structural questions: how things relate, where an \
             entity/function/type is defined, what references or calls what, counts, and \
             multi-hop paths (for code graphs: call graphs, definitions, imports — navigate the \
             codebase structure). Pass params={...} to bind $placeholders — both \
             `{prop: $p}` inside a pattern and `WHERE x.prop = $p` read from it, and a \
             $name with no value is an error rather than an empty result. Returns up to 15 rows inline; append FORMAT \
             CSV for a CSV body — this server has csv_http_server enabled, so the full result is \
             written to its directory and returned as a fetch URL."
        }
        (false, false) => {
            "Query, explore, and understand the active knowledge graph with Cypher — the \
             primary tool for structural questions: how things relate, where an \
             entity/function/type is defined, what references or calls what, counts, and \
             multi-hop paths (for code graphs: call graphs, definitions, imports — navigate the \
             codebase structure). Pass params={...} to bind $placeholders — both \
             `{prop: $p}` inside a pattern and `WHERE x.prop = $p` read from it, and a \
             $name with no value is an error rather than an empty result. Returns up to 15 rows inline; append FORMAT \
             CSV for a CSV body capped at 200 rows, with a notice naming the true total — narrow \
             the query to fit, or ask the operator to enable extensions.csv_http_server for a \
             fetch URL carrying the complete result."
        }
    };
    match pin {
        Some(pin) => pinned_cypher_description(base, pin),
        None => base,
    }
}

/// Register the runtime graph-lifecycle tools — `load_graph`, `create_graph`
/// and `save_graph_as` — that turn a write-enabled server into a workbench an
/// agent can re-point at another graph mid-session.
///
/// Write-enabled only: each one replaces or rebinds the served graph, so on a
/// read-only deployment they would offer a mutation the rest of the surface
/// refuses. They reuse the existing `GraphState` swap methods, which take the
/// write lock internally, so a swap cannot race a query in flight.
fn register_graph_lifecycle_tools(
    server: &mut McpServer,
    state: GraphState,
    skills: crate::skills::SkillRefresher,
) {
    let s = state.clone();
    let refresher = skills.clone();
    server.register_typed_tool_fallible::<LoadGraphArgs, _>(
        "load_graph",
        "Load a .kgl file as the new active graph (replaces the current one). Refused \
         while this server has unsaved changes — call save_graph first to keep them, or \
         reload_graph(discard_unsaved=true) to drop them. Write-enabled servers only.",
        move |args| {
            refuse_swap_while_dirty(&s, "load_graph")?;
            match s.load_kgl(Path::new(&args.path)) {
                Ok(()) => {
                    refresher.refresh();
                    Ok(match s.schema() {
                        Some((n, e)) => format!("Loaded {} ({n} nodes, {e} edges).", args.path),
                        None => format!("Loaded {}.", args.path),
                    })
                }
                Err(e) => Err(format!("load_graph error: {e}")),
            }
        },
    );
    let s = state.clone();
    let refresher = skills;
    server.register_typed_tool_fallible::<CreateGraphArgs, _>(
        "create_graph",
        "Create a fresh, empty graph bound to a path (its save_graph target) and \
         make it active. storage = memory (default) | mapped | disk. Refused while this \
         server has unsaved changes — call save_graph first to keep them, or \
         reload_graph(discard_unsaved=true) to drop them. Write-enabled servers only.",
        move |args| {
            refuse_swap_while_dirty(&s, "create_graph")?;
            let mode = args
                .storage
                .as_ref()
                .map_or(StorageMode::Memory, StorageArg::mode);
            match s.create_in_mode(Path::new(&args.path), mode) {
                Ok(()) => {
                    // An empty graph carries no skills, so this is what drops
                    // the previous graph's from the surface.
                    refresher.refresh();
                    Ok(format!("Created empty graph at {} (active).", args.path))
                }
                Err(e) => Err(format!("create_graph error: {e}")),
            }
        },
    );
    let s = state;
    server.register_typed_tool_fallible::<SaveGraphAsArgs, _>(
        "save_graph_as",
        "Save the active graph to an explicit path and rebind the save target there. \
         Write-enabled servers only.",
        move |args| {
            s.ensure_graph_fresh();
            s.save_as(Path::new(&args.path))
        },
    );
}

pub fn register(
    server: &mut McpServer,
    state: GraphState,
    builtins: Builtins,
    overview_decorations: OverviewDecorations,
    csv_http: Arc<crate::csv_http::CsvHttpState>,
    skills: crate::skills::SkillRefresher,
) {
    let s = state.clone();
    let csv = csv_http.clone();
    let writable = builtins.writable;
    let operator_scope = builtins.write_scope.clone();
    let cypher_desc =
        cypher_description(csv.config().is_some(), writable, operator_scope.as_deref());
    if writable {
        register_cypher_tool::<CypherArgs>(server, cypher_desc, true, move |args| {
            let raw_args = serde_json::to_value(&args).expect("Cypher arguments serialize");
            let csv = csv.clone();
            s.ensure_graph_fresh();
            let policy = s.exec_policy().with_timeout_ms(args.timeout_ms);
            let scope = args.write_scope.clone();
            let git_sha = args.git_sha.clone();
            let modified_by = args.modified_by.clone();
            let authz = WriteAuthz {
                operator_scope: operator_scope.as_deref(),
                agent_scope: scope.as_deref(),
                git_sha: git_sha.as_deref(),
                modified_by: modified_by.as_deref(),
            };
            let params = match params_from_json(args.params.as_ref()) {
                Ok(params) => params,
                Err(error) => return Err(s.with_rebuild_warning(error)),
            };
            let query = match query_with_valid_at(&args.query, args.valid_at.as_deref()) {
                Ok(query) => query,
                Err(error) => return Err(s.with_rebuild_warning(error)),
            };
            let body = s
                .with_active_mut(|active| {
                    run_cypher_write_output(active, &query, params, authz, policy, &csv)
                        .map_err(|e| cypher_tool_error(&e))
                })
                .unwrap_or_else(|| Err(NO_GRAPH.to_string()));
            body.map(|output| {
                output
                    .with_rebuild_warning(&s)
                    .with_result_steering(&s, &raw_args)
            })
            .map_err(|error| s.with_rebuild_warning(error))
        });
    } else {
        register_cypher_tool::<ReadCypherArgs>(server, cypher_desc, false, move |args| {
            let raw_args = serde_json::to_value(&args).expect("Cypher arguments serialize");
            let csv = csv.clone();
            s.ensure_graph_fresh();
            let policy = s.exec_policy().with_timeout_ms(args.timeout_ms);
            let params = match params_from_json(args.params.as_ref()) {
                Ok(params) => params,
                Err(error) => return Err(s.with_rebuild_warning(error)),
            };
            let query = match query_with_valid_at(&args.query, args.valid_at.as_deref()) {
                Ok(query) => query,
                Err(error) => return Err(s.with_rebuild_warning(error)),
            };
            let body = s
                .with_active(|g| run_cypher_tool_output(g, &query, params, policy, &csv))
                .unwrap_or_else(|| Err(NO_GRAPH.to_string()));
            body.map(|output| {
                output
                    .with_rebuild_warning(&s)
                    .with_result_steering(&s, &raw_args)
            })
            .map_err(|error| s.with_rebuild_warning(error))
        });
    }
    crate::raw_query_routes::protect_query_route(
        server,
        "cypher_query",
        crate::raw_query_routes::CYPHER_QUERY_POINTER,
    );
    let s = state.clone();
    let cleanup_temp = builtins.temp_cleanup_on_overview;
    let temp_dir = builtins.temp_dir.clone();
    server.register_typed_tool_fallible::<OverviewArgs, _>(
        "graph_overview",
        "Inspect and explore the active graph's schema — start here to understand a codebase \
         or dataset: node types, properties, connections, sample values, and a per-type \
         example query (anchored on each type's real identifier property). With no args \
         returns the inventory; pass types=[...] / connections=true|[...] / \
         cypher=true|[...] for drill-down.",
        move |args| {
            let is_bare = prepare_overview(&args, cleanup_temp, temp_dir.as_deref());
            s.ensure_graph_fresh();
            let body = s
                .with_active(|g| run_overview(g, &args))
                .unwrap_or_else(|| Err(NO_GRAPH.to_string()));
            let body = map_body(body, |body| s.with_rebuild_warning(body));
            map_body(body, |body| overview_decorations.render(body, is_bare))
        },
    );
    if builtins.save_graph {
        let s = state.clone();
        // `force` re-encodes the served file, which `run_save` offers only
        // where mutations are — so the description advertises it only there
        // too, rather than naming a route this deployment refuses.
        let mutations_enabled = builtins.writable;
        let description = if mutations_enabled {
            SAVE_GRAPH_DESCRIPTION_WRITABLE
        } else {
            SAVE_GRAPH_DESCRIPTION_READ_ONLY
        };
        server.register_typed_tool_fallible::<SaveGraphArgs, _>(
            "save_graph",
            description,
            move |args: SaveGraphArgs| {
                s.ensure_graph_fresh();
                // Mutable access: the save must go through the active
                // graph's own Arc so `prepare_save`'s `Arc::make_mut` sees
                // refcount 1 (no whole-graph deep copy per save).
                s.with_active_mut(|g| run_save(g, args.force.unwrap_or(false), mutations_enabled))
                    .unwrap_or_else(|| Err(NO_GRAPH.to_string()))
            },
        );
    }

    if builtins.writable {
        register_graph_lifecycle_tools(server, state, skills);
    }
}
