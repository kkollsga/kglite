//! Tool registration: the manifest-driven builtin toggles, the
//! `graph_overview` decorations, and the router wiring for every KGLite
//! MCP route.

use std::path::Path;
use std::sync::Arc;

use kglite::api::storage::StorageMode;
use mcp_methods::server::McpServer;

use crate::recipe_queries::CatalogSummary;
use crate::tools::*;

/// Builtins toggled by the manifest's `builtins:` block.
#[derive(Clone, Debug, Default)]
pub struct Builtins {
    pub save_graph: bool,
    /// Write-enabled "agent graph workbench" mode (CLI `--writable`). When
    /// true, `cypher_query` accepts mutations (routed through the write-lock)
    /// and the runtime graph-lifecycle tools (`load_graph` / `create_graph` /
    /// `save_graph_as`) are registered. Off by default — read-only is the safe
    /// default for code-review / analysis deployments.
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

/// Immutable MCP-layer additions to the bare `graph_overview` response.
///
/// These describe the deployment, not the active graph, so they are captured
/// by the route at boot instead of entering [`GraphState`] or core
/// `describe()`.
#[derive(Clone, Debug, Default)]
pub(crate) struct OverviewDecorations {
    pub(crate) prefix: Option<String>,
    pub(crate) catalog: Option<CatalogSummary>,
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
        if let Some(summary) = self.catalog {
            append_overview_section(
                &mut rendered,
                &format!(
                    "<query-catalog recipes=\"{}\" queries=\"{}\" \
                     list-tool=\"list_recipe_queries\" run-tool=\"run_recipe_query\"/>",
                    summary.recipe_count, summary.query_count
                ),
            );
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
pub fn register_graph_mode_tools(server: &mut McpServer, state: GraphState) {
    server.register_typed_tool_fallible::<ReloadGraphArgs, _>(
        "reload_graph",
        "Re-read the served graph file from disk, replacing the in-memory graph — use this \
         when the file has been rebuilt by another process and queries are returning stale \
         results. Takes no arguments; the path is the one this server was started on. On a \
         write-enabled server any unsaved in-memory changes are discarded (call save_graph \
         first to keep them). If the re-read fails, the current graph stays active and the \
         error is returned.",
        move |_| match state.source_path() {
            // `open_or_create(path, None)`: no storage mode is requested, so a
            // reload never re-runs the boot `--storage` conversion — it serves
            // whatever the (possibly newly written) checkpoint records, exactly
            // as `load_graph` does. A load failure returns before the write
            // lock is taken, so the old graph provably stays active.
            Some(path) => match state.open_or_create(&path, None) {
                Ok(_) => {
                    let path = path.display();
                    let generation = state
                        .generation()
                        .map(|g| format!(" Graph generation {g}."))
                        .unwrap_or_default();
                    Ok(match state.schema() {
                        Some((n, e)) => {
                            format!("Reloaded {path} ({n} nodes, {e} edges).{generation}")
                        }
                        None => format!("Reloaded {path}.{generation}"),
                    })
                }
                Err(e) => Err(format!("reload_graph error: {e}")),
            },
            None => Err(format!("reload_graph error: {NO_GRAPH}")),
        },
    );
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

pub fn register(
    server: &mut McpServer,
    state: GraphState,
    builtins: Builtins,
    overview_decorations: OverviewDecorations,
    csv_http: Option<Arc<crate::csv_http::CsvHttpConfig>>,
) {
    let s = state.clone();
    let csv = csv_http.clone();
    let writable = builtins.writable;
    let operator_scope = builtins.write_scope.clone();
    // Descriptions lead with the code-exploration vocabulary agents actually
    // search for (explore, understand, "how does", call graph, "where defined",
    // structure, navigate) so lazy-tool-discovery clients (Codex / code_mode)
    // surface cypher_query on their first broad tool search instead of falling
    // back to grep. (mcp-servers inbox 2026-07-01.)
    let cypher_desc: &'static str = match (csv.is_some(), writable) {
        (_, true) => {
            "Query, explore, and understand the active knowledge graph with Cypher, and \
             modify it — reads AND writes (CREATE/SET/DELETE/MERGE) are accepted; this is a \
             write-enabled graph. The primary tool for structural questions: how things \
             relate, where an entity/function/type is defined, what references or calls what, \
             counts, and multi-hop paths (for code graphs: call graphs, definitions, imports — \
             navigate the codebase structure). Pass write_scope=[...] to restrict mutations \
             to those node types: every node write (CREATE/MERGE/SET/REMOVE/DELETE/DETACH \
             DELETE and node-type DDL) is judged by the node's stored type, and a \
             relationship write (edge CREATE, DELETE r, SET r.p, REMOVE r.p) needs at least \
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
    // An operator pin is part of the contract the agent plans against, so it
    // is stated in the description rather than left to surface as a refusal.
    let cypher_desc = match operator_scope.as_deref() {
        Some(pin) => pinned_cypher_description(cypher_desc, pin),
        None => cypher_desc,
    };
    if writable {
        server.register_typed_tool_fallible::<CypherArgs, _>(
            "cypher_query",
            cypher_desc,
            move |args| {
                let csv = csv.clone();
                s.ensure_graph_fresh();
                let policy = s.exec_policy();
                let scope = args.write_scope.clone();
                let git_sha = args.git_sha.clone();
                let modified_by = args.modified_by.clone();
                let authz = WriteAuthz {
                    operator_scope: operator_scope.as_deref(),
                    agent_scope: scope.as_deref(),
                    git_sha: git_sha.as_deref(),
                    modified_by: modified_by.as_deref(),
                };
                let params = params_from_json(args.params.as_ref());
                let body = s
                    .with_active_mut(|active| {
                        run_cypher_write(active, &args.query, params, authz, policy, csv.as_deref())
                            .map_err(|e| cypher_tool_error(&e))
                    })
                    .unwrap_or_else(|| Err(NO_GRAPH.to_string()));
                map_body(body, |body| s.with_rebuild_warning(body))
            },
        );
    } else {
        server.register_typed_tool_fallible::<ReadCypherArgs, _>(
            "cypher_query",
            cypher_desc,
            move |args| {
                let csv = csv.clone();
                s.ensure_graph_fresh();
                let policy = s.exec_policy();
                let params = params_from_json(args.params.as_ref());
                let body = s
                    .with_active(|g| {
                        run_cypher_tool(g, &args.query, params, policy, csv.as_deref())
                    })
                    .unwrap_or_else(|| Err(NO_GRAPH.to_string()));
                map_body(body, |body| s.with_rebuild_warning(body))
            },
        );
    }
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
        server.register_typed_tool_fallible::<SaveGraphArgs, _>(
            "save_graph",
            "Persist the active graph to its source .kgl file (single-graph mode only).",
            move |_| {
                s.ensure_graph_fresh();
                // Mutable access: the save must go through the active
                // graph's own Arc so `prepare_save`'s `Arc::make_mut` sees
                // refcount 1 (no whole-graph deep copy per save).
                s.with_active_mut(run_save)
                    .unwrap_or_else(|| Err(NO_GRAPH.to_string()))
            },
        );
    }

    // Runtime graph-lifecycle tools — only on a write-enabled workbench server.
    // They reuse the existing GraphState swap methods (which take the write-lock
    // internally), so an agent can load/create/save graphs and switch between
    // them within one session.
    if builtins.writable {
        let s = state.clone();
        server.register_typed_tool_fallible::<LoadGraphArgs, _>(
            "load_graph",
            "Load a .kgl file as the new active graph (replaces the current one — \
             save_graph first to keep unsaved changes). Write-enabled servers only.",
            move |args| match s.load_kgl(Path::new(&args.path)) {
                Ok(()) => Ok(match s.schema() {
                    Some((n, e)) => format!("Loaded {} ({n} nodes, {e} edges).", args.path),
                    None => format!("Loaded {}.", args.path),
                }),
                Err(e) => Err(format!("load_graph error: {e}")),
            },
        );
        let s = state.clone();
        server.register_typed_tool_fallible::<CreateGraphArgs, _>(
            "create_graph",
            "Create a fresh, empty graph bound to a path (its save_graph target) and \
             make it active. storage = memory (default) | mapped | disk. Write-enabled \
             servers only.",
            move |args| {
                let mode = args
                    .storage
                    .as_ref()
                    .map_or(StorageMode::Memory, StorageArg::mode);
                match s.create_in_mode(Path::new(&args.path), mode) {
                    Ok(()) => Ok(format!("Created empty graph at {} (active).", args.path)),
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
}
