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
    server.register_typed_tool::<ReloadGraphArgs, _>(
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
                    match state.schema() {
                        Some((n, e)) => {
                            format!("Reloaded {path} ({n} nodes, {e} edges).{generation}")
                        }
                        None => format!("Reloaded {path}.{generation}"),
                    }
                }
                Err(e) => format!("reload_graph error: {e}"),
            },
            None => format!("reload_graph error: {NO_GRAPH}"),
        },
    );
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
             one endpoint's type in the list. Mutations are in-memory; call save_graph to \
             persist. Append FORMAT CSV to export results."
        }
        (true, false) => {
            "Query, explore, and understand the active knowledge graph with Cypher — the \
             primary tool for structural questions: how things relate, where an \
             entity/function/type is defined, what references or calls what, counts, and \
             multi-hop paths (for code graphs: call graphs, definitions, imports — navigate the \
             codebase structure). Returns up to 15 rows inline; append FORMAT CSV to export \
             results — large CSVs are written to the csv_http_server directory and returned as \
             a fetch URL."
        }
        (false, false) => {
            "Query, explore, and understand the active knowledge graph with Cypher — the \
             primary tool for structural questions: how things relate, where an \
             entity/function/type is defined, what references or calls what, counts, and \
             multi-hop paths (for code graphs: call graphs, definitions, imports — navigate the \
             codebase structure). Returns up to 15 rows inline; append FORMAT CSV to export \
             full results to a CSV string."
        }
    };
    if writable {
        server.register_typed_tool::<CypherArgs, _>("cypher_query", cypher_desc, move |args| {
            let csv = csv.clone();
            s.ensure_graph_fresh();
            let codecs = s.value_codecs();
            let scope = args.write_scope.clone();
            let git_sha = args.git_sha.clone();
            let modified_by = args.modified_by.clone();
            let body = s
                .with_active_mut(|active| {
                    run_cypher_write(
                        active,
                        &args.query,
                        scope.as_deref(),
                        git_sha.as_deref(),
                        modified_by.as_deref(),
                        codecs,
                        csv.as_deref(),
                    )
                    .unwrap_or_else(|e| cypher_tool_error(&e))
                })
                .unwrap_or_else(|| NO_GRAPH.to_string());
            s.with_rebuild_warning(body)
        });
    } else {
        server.register_typed_tool::<ReadCypherArgs, _>("cypher_query", cypher_desc, move |args| {
            let csv = csv.clone();
            s.ensure_graph_fresh();
            let codecs = s.value_codecs();
            let body = s.with_active(|g| run_cypher_tool(g, &args.query, codecs, csv.as_deref()));
            s.with_rebuild_warning(body)
        });
    }
    let s = state.clone();
    let cleanup_temp = builtins.temp_cleanup_on_overview;
    let temp_dir = builtins.temp_dir.clone();
    server.register_typed_tool::<OverviewArgs, _>(
        "graph_overview",
        "Inspect and explore the active graph's schema — start here to understand a codebase \
         or dataset: node types, properties, connections, sample values, and a per-type \
         example query (anchored on each type's real identifier property). With no args \
         returns the inventory; pass types=[...] / connections=true|[...] / \
         cypher=true|[...] for drill-down.",
        move |args| {
            let is_bare = prepare_overview(&args, cleanup_temp, temp_dir.as_deref());
            s.ensure_graph_fresh();
            let body = s.with_active(|g| run_overview(g, &args));
            let body = s.with_rebuild_warning(body);
            overview_decorations.render(body, is_bare)
        },
    );
    if builtins.save_graph {
        let s = state.clone();
        server.register_typed_tool::<SaveGraphArgs, _>(
            "save_graph",
            "Persist the active graph to its source .kgl file (single-graph mode only).",
            move |_| {
                s.ensure_graph_fresh();
                // Mutable access: the save must go through the active
                // graph's own Arc so `prepare_save`'s `Arc::make_mut` sees
                // refcount 1 (no whole-graph deep copy per save).
                s.with_active_mut(run_save)
                    .unwrap_or_else(|| NO_GRAPH.to_string())
            },
        );
    }

    // Runtime graph-lifecycle tools — only on a write-enabled workbench server.
    // They reuse the existing GraphState swap methods (which take the write-lock
    // internally), so an agent can load/create/save graphs and switch between
    // them within one session.
    if builtins.writable {
        let s = state.clone();
        server.register_typed_tool::<LoadGraphArgs, _>(
            "load_graph",
            "Load a .kgl file as the new active graph (replaces the current one — \
             save_graph first to keep unsaved changes). Write-enabled servers only.",
            move |args| match s.load_kgl(Path::new(&args.path)) {
                Ok(()) => match s.schema() {
                    Some((n, e)) => format!("Loaded {} ({n} nodes, {e} edges).", args.path),
                    None => format!("Loaded {}.", args.path),
                },
                Err(e) => format!("load_graph error: {e}"),
            },
        );
        let s = state.clone();
        server.register_typed_tool::<CreateGraphArgs, _>(
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
                    Ok(()) => format!("Created empty graph at {} (active).", args.path),
                    Err(e) => format!("create_graph error: {e}"),
                }
            },
        );
        let s = state;
        server.register_typed_tool::<SaveGraphAsArgs, _>(
            "save_graph_as",
            "Save the active graph to an explicit path and rebind the save target there. \
             Write-enabled servers only.",
            move |args| {
                s.ensure_graph_fresh();
                match s.save_as(Path::new(&args.path)) {
                    Ok(msg) => msg,
                    Err(e) => e,
                }
            },
        );
    }
}
