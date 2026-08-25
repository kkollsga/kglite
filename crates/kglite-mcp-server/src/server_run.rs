//! The async server body: one named boot phase after another, ending in
//! the stdio MCP service.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use mcp_methods::server::{init_tracing, load_env_for_mode, McpServer, ResultCtx, ServerOptions};
use rmcp::transport::stdio;
use rmcp::ServiceExt;

use crate::tools::GraphState;
use crate::*;

/// Compile and validate the manifest's stored Cypher recipes once at boot.
///
/// Recipe definitions are configuration, not per-call input, so the
/// deliberately small schema subset is checked here and later route handlers
/// share one immutable catalog.
fn boot_recipe_catalog(
    manifest: Option<&mcp_methods::server::Manifest>,
) -> Result<Arc<recipe_queries::RecipeCatalog>> {
    let catalog = Arc::new(match manifest {
        Some(manifest) => recipe_queries::RecipeCatalog::from_manifest_value(
            manifest.extensions.get("cypher_recipes"),
        )
        .context("extensions.cypher_recipes parse failed")?,
        None => recipe_queries::RecipeCatalog::default(),
    });
    if let Some(summary) = catalog.discovery_summary() {
        tracing::info!(
            recipes = summary.recipe_count,
            queries = summary.query_count,
            "Cypher recipe catalog validated"
        );
    }
    Ok(catalog)
}

/// Build the manifest-declared, position-scoped literal codecs
/// (`extensions.value_codecs`: prefix / map / regex).
///
/// Passed to the engine via `ExecuteOptions` per query — decode on the way in,
/// encode on the way out. Empty when absent; a malformed block errors at boot,
/// not per-query.
fn boot_value_codecs(
    manifest: Option<&mcp_methods::server::Manifest>,
) -> Result<Option<Arc<Vec<kglite::api::cypher::ValueCodec>>>> {
    let codecs = match manifest {
        Some(m) => value_codecs::from_manifest(m.extensions.get("value_codecs"))
            .context("extensions.value_codecs parse failed")?,
        None => Vec::new(),
    };
    Ok((!codecs.is_empty()).then(|| Arc::new(codecs)))
}

/// `extensions.graph_watch: true` — opt-in filesystem watch on the served
/// `.kgl` in `--graph` mode.
///
/// Off by default: watching costs an OS watch registration and makes the served
/// graph change under a client that never asked for it, so it stays an operator
/// decision. When on, an external rewrite of the file marks the graph for
/// reload and the next graph tool call re-reads it (`tools::graph_reload`).
/// Applies only to `--graph` mode; `resolve_graph_watch_target` warns and
/// declines everywhere else. A non-boolean value is a boot error, never a
/// silently ignored key (see [`boot_tools_allow`]).
fn boot_graph_watch(manifest: Option<&mcp_methods::server::Manifest>) -> Result<bool> {
    let Some(raw) = manifest.and_then(|m| m.extensions.get("graph_watch")) else {
        return Ok(false);
    };
    raw.as_bool()
        .context("extensions.graph_watch must be a boolean (true or false)")
}

/// `extensions.parallel: true` — the manifest half of the parallel opt-in.
///
/// Boolean-key handling matches [`boot_graph_watch`]: absent is off, a
/// non-boolean is a boot error rather than a silently dropped key.
fn boot_manifest_parallel(manifest: Option<&mcp_methods::server::Manifest>) -> Result<bool> {
    let Some(raw) = manifest.and_then(|m| m.extensions.get("parallel")) else {
        return Ok(false);
    };
    raw.as_bool()
        .context("extensions.parallel must be a boolean (true or false)")
}

/// Resolve the parallel-runtime opt-in from `--parallel` and
/// `extensions.parallel`.
///
/// **OR, not intersection** — the opposite of [`boot_write_scope`], and for
/// the reason that makes the two opposite: a write scope is a *perimeter*, so
/// combining two of them may only narrow, while this is a *resource
/// permission* whose worst case is a query using more cores than it needed.
/// Either surface alone turns it on, so a wrapper that owns the manifest but
/// not argv, and a bare binary launched with no manifest at all, each have a
/// working way to say yes.
///
/// Applies to the read seam only ([`tools::ExecPolicy`]); mutations stay
/// sequential.
fn boot_parallel(manifest: Option<&mcp_methods::server::Manifest>, cli: &Cli) -> Result<bool> {
    Ok(boot_manifest_parallel(manifest)? || cli.parallel)
}

/// The width the engine's query pool will be built at, for the boot log.
///
/// Recomputed here rather than read from the engine: the pool is built lazily
/// on the first query that crosses a fan-out threshold, so at boot there is
/// nothing to ask. Mirrors `kglite::graph::parallel::configured_width` — same
/// env var, same `available_parallelism` fallback — and is reported as
/// observed configuration, never as a promise about a pool that does not
/// exist yet.
fn query_pool_width() -> (usize, bool) {
    let override_width = std::env::var_os("KGLITE_QUERY_THREADS")
        .and_then(|raw| raw.to_str()?.trim().parse::<usize>().ok())
        .filter(|n| *n > 0);
    match override_width {
        Some(n) => (n, true),
        None => (
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            false,
        ),
    }
}

/// `extensions.tools_allow: [name, ...]` — the closed-by-default tool surface.
///
/// Absent key = no allowlist = every registered route stays visible. Present,
/// it names the *final* tool surface: see [`apply_tool_allowlist`] for what
/// that means and what it deliberately does not do. An explicit empty list is
/// honoured literally — a server with no tools — because the alternative
/// (silently treating `[]` as "no allowlist") would widen a surface an
/// operator asked to close.
///
/// A non-list value, or a non-string element, is a boot error rather than a
/// dropped key: an allowlist that silently fails open is worse than no
/// allowlist, and "parsed and then ignored" is this crate's recurring defect
/// shape (`graph_watch`, `temp_cleanup`).
fn boot_tools_allow(
    manifest: Option<&mcp_methods::server::Manifest>,
) -> Result<Option<Vec<String>>> {
    let Some(raw) = manifest.and_then(|m| m.extensions.get("tools_allow")) else {
        return Ok(None);
    };
    let items = raw
        .as_array()
        .context("extensions.tools_allow must be a list of tool names")?;
    let mut names = Vec::with_capacity(items.len());
    for item in items {
        let name = item.as_str().with_context(|| {
            format!("extensions.tools_allow entries must be tool names (strings); found {item}")
        })?;
        names.push(name.to_owned());
    }
    Ok(Some(names))
}

/// `extensions.write_scope: [NodeType, ...]` — the operator-pinned write scope.
///
/// Absent key = no pin = the agent's own `write_scope` argument decides (which
/// may be nothing at all). Present, it is the ceiling the agent can only narrow:
/// see [`resolve_write_scope`] for how the two combine. An explicit empty list
/// is honoured literally — a write-enabled server that permits no writes —
/// because reading `[]` as "no pin" would widen a perimeter the operator asked
/// to close.
///
/// A non-list value, or a non-string element, is a boot error rather than a
/// dropped key, for the same reason [`boot_tools_allow`] fails that way.
fn boot_manifest_write_scope(
    manifest: Option<&mcp_methods::server::Manifest>,
) -> Result<Option<Vec<String>>> {
    let Some(raw) = manifest.and_then(|m| m.extensions.get("write_scope")) else {
        return Ok(None);
    };
    let items = raw
        .as_array()
        .context("extensions.write_scope must be a list of node types")?;
    let mut names = Vec::with_capacity(items.len());
    for item in items {
        let name = item.as_str().with_context(|| {
            format!("extensions.write_scope entries must be node types (strings); found {item}")
        })?;
        names.push(name.to_owned());
    }
    Ok(Some(names))
}

/// Resolve the operator-pinned write scope from the manifest key and the
/// `--write-scope` flag.
///
/// Both set is not an error: the two are **intersected**, the same rule the
/// agent's own scope obeys, so neither surface can widen what the other pinned.
/// Refusing the ambiguity at boot would only push the operator into deleting
/// one of them. The intersection is logged, so the effective scope shows in the
/// boot output rather than having to be inferred.
fn boot_write_scope(
    manifest: Option<&mcp_methods::server::Manifest>,
    cli: &Cli,
) -> Result<Option<Vec<String>>> {
    let from_manifest = boot_manifest_write_scope(manifest)?;
    let from_flag = cli.write_scope.as_deref().map(parse_write_scope_flag);
    let resolved = match (from_manifest, from_flag) {
        (None, None) => None,
        (Some(only), None) | (None, Some(only)) => Some(only),
        (Some(m), Some(f)) => {
            let both: Vec<String> = f.iter().filter(|t| m.contains(t)).cloned().collect();
            tracing::info!(
                manifest = m.join(","),
                flag = f.join(","),
                effective = both.join(","),
                "write_scope pinned by both --write-scope and extensions.write_scope; \
                 the effective scope is their intersection"
            );
            Some(both)
        }
    };
    // `cli.writable`, not `graph_writes_enabled`: a manifest that only enables
    // `builtins.save_graph` registers `save_graph` but still serves the
    // read-only `cypher_query`, so the pin has nothing to apply to there
    // either.
    if resolved.is_some() && !cli.writable {
        tracing::warn!(
            "--write-scope / extensions.write_scope is set on a server whose cypher_query is \
             read-only; every mutation is refused already. Add --writable to serve scoped writes."
        );
    }
    Ok(resolved)
}

/// Directory the manifest was loaded from — the base both `csv_http_server`
/// (resolving `dir:`) and `temp_cleanup` (finding the directory to wipe)
/// resolve against. Falls back to cwd when there's no manifest.
fn manifest_base_dir(manifest: Option<&mcp_methods::server::Manifest>) -> PathBuf {
    manifest
        .and_then(|m| m.yaml_path.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

/// Start the opt-in `extensions.csv_http_server:` CSV-over-HTTP listener.
///
/// When configured we spawn a tokio task to serve files out of the directory;
/// the `cypher_query` tool sees the same config and writes `FORMAT CSV` results
/// to that directory, returning a URL instead of an inline CSV blob.
async fn boot_csv_http(
    manifest: Option<&mcp_methods::server::Manifest>,
    manifest_base: &Path,
) -> Result<Option<csv_http::CsvHttpConfig>> {
    let cfg = match manifest.and_then(|m| m.extensions.get("csv_http_server")) {
        Some(raw) => csv_http::CsvHttpConfig::from_manifest_value(raw, manifest_base)
            .context("extensions.csv_http_server parse failed")?,
        None => None,
    };
    if let Some(cfg) = cfg.as_ref() {
        csv_http::spawn(cfg.clone())
            .await
            .context("csv_http_server failed to bind")?;
    }
    Ok(cfg)
}

/// Whether this deployment can write the graph file it serves — `--writable`
/// (which implies `save_graph`: an agent that mutates needs to persist) or a
/// manifest that enables `save_graph` on its own.
///
/// Read twice, once for each thing that follows from it: which write tools get
/// registered ([`tools::Builtins`]) and whether the server takes the path's
/// cross-process writer lease ([`writer_lease_policy`]). One function so those
/// two can never disagree about what "this server writes the graph" means.
fn graph_writes_enabled(manifest: Option<&mcp_methods::server::Manifest>, cli: &Cli) -> bool {
    manifest.map(|m| m.builtins.save_graph).unwrap_or(false) || cli.writable
}

/// Whether to take the single-writer lease on the served path.
///
/// A read-only server never rewrites the file, so holding its lease refuses
/// every external rebuilder (`kglite.open(path)` fails fast) for the server's
/// whole lifetime and buys nothing back — and it stops two read-only servers
/// from serving one `.kgl`. Write-enabled deployments keep the exclusive lease;
/// so do disk-graph directories, whose retained mmaps make an external writer
/// genuinely unsafe rather than merely stale (see
/// `GraphState::takes_writer_lease`).
fn writer_lease_policy(
    manifest: Option<&mcp_methods::server::Manifest>,
    cli: &Cli,
) -> tools::WriterLeasePolicy {
    if graph_writes_enabled(manifest, cli) {
        tools::WriterLeasePolicy::Exclusive
    } else {
        tools::WriterLeasePolicy::ReadOnly
    }
}

/// Builtin toggles from the manifest.
///   - `save_graph`: registration gated on [`graph_writes_enabled`], so a
///     destructive operation is not exposed to the agent on every graph
///     regardless of intent.
///   - `temp_cleanup: on_overview`: wipe `temp/` on every bare
///     `graph_overview()`.
fn boot_builtins(
    manifest: Option<&mcp_methods::server::Manifest>,
    cli: &Cli,
    csv_http_cfg: Option<&csv_http::CsvHttpConfig>,
    manifest_base: &Path,
    write_scope: Option<Vec<String>>,
) -> tools::Builtins {
    tools::Builtins {
        save_graph: graph_writes_enabled(manifest, cli),
        writable: cli.writable,
        write_scope,
        temp_cleanup_on_overview: manifest
            .map(|m| {
                matches!(
                    m.builtins.temp_cleanup,
                    mcp_methods::server::manifest::TempCleanup::OnOverview
                )
            })
            .unwrap_or(false),
        // Resolved against the manifest base, not the cwd — the server's cwd
        // need not match the manifest's parent. Reuse the csv_http_server
        // directory when configured, so both sides of the CSV pipeline agree
        // on what counts as "the temp dir".
        temp_dir: Some(
            csv_http_cfg
                .map(|c| c.dir.clone())
                .unwrap_or_else(|| manifest_base.join("temp")),
        ),
    }
}

fn register_kglite_tools(
    server: &mut McpServer,
    graph_state: &GraphState,
    manifest: Option<&mcp_methods::server::Manifest>,
    builtins: tools::Builtins,
    recipe_catalog_summary: Option<crate::recipe_queries::CatalogSummary>,
    csv_http_arc: Option<Arc<csv_http::CsvHttpConfig>>,
    source_roots_provider: Option<mcp_methods::server::source::SourceRootsProvider>,
) -> Result<()> {
    tools::register(
        server,
        graph_state.clone(),
        builtins,
        tools::OverviewDecorations {
            prefix: manifest.and_then(|manifest| manifest.overview_prefix.clone()),
            catalog: recipe_catalog_summary,
        },
        csv_http_arc,
    );
    code_source::register(server, graph_state.clone(), source_roots_provider.clone())
        .context("read_code_source registration failed")?;
    explore::register(server, graph_state.clone(), source_roots_provider)
        .context("explore registration failed")?;
    Ok(())
}

/// Whether `explore` and `read_code_source` should be hidden for this boot.
///
/// `explore` pins its entry types and its traversal edge whitelist to code node
/// types, so on a graph without `Function`/`Class` it can only ever return "no
/// match" — it is structurally dead, and its presence in `tools/list` is pure
/// misdirection. Worse, `read_code_source`'s optional `node_type` argument makes
/// it a general-purpose reader of whatever `file_path` properties the graph
/// happens to carry, so on a non-code graph it is a way to pull arbitrary files
/// off disk with no code-graph purpose to justify it.
///
/// Three constraints are deliberate and settled:
///
/// 1. **This is a boot-time snapshot of a runtime property.** Every other
///    `has_node_type` consumer (skill `applies_when:` predicates, the result
///    footers) re-evaluates per call; this one cannot, because tool handlers
///    have no access to the router and so nothing can re-enable a route once
///    the server is running. A same-path graph reload rarely changes the
///    graph's *class* (a code graph stays a code graph), so the staleness
///    window is accepted. If handlers ever gain router access, make it live.
/// 2. **Writable servers are exempt.** `load_graph` can swap a code graph in
///    at any time, and by (1) there would be no way to re-enable the routes
///    afterwards — so a writable server keeps both tools regardless of what
///    it booted with.
/// 3. **Graph mode only.** Every other mode has no graph at boot, so
///    `has_node_type` is uniformly `false` there and a mode-blind gate would
///    strip `explore` from exactly the deployments that need it most
///    (local-workspace and GitHub-repo code servers, whose graphs are built
///    after registration).
fn code_tools_are_dead(mode: &Mode, builtins: &tools::Builtins, graph_state: &GraphState) -> bool {
    matches!(mode, Mode::Graph { .. })
        && !builtins.writable
        && !(graph_state.has_node_type("Function") || graph_state.has_node_type("Class"))
}

/// Bind the manifest's `extensions.embedder:` as the embedder `text_score()`
/// uses.
///
/// Selection is by `library:` — `fastembed-rs` for the Rust-native adapter (the
/// only option on the standalone libpython-free binary), anything else for a
/// Python library built by the wheel-supplied factory. The rules, and the
/// `trust.allow_embedder` gate they sit behind, live in
/// [`build_embedder_from_manifest`].
fn bind_manifest_embedder(
    manifest: Option<&mcp_methods::server::Manifest>,
    py_embedder_factory: Option<&PyEmbedderFactory>,
    graph_state: &GraphState,
) -> Result<()> {
    if let Some(m) = manifest {
        if let Some(embedder) = build_embedder_from_manifest(m, py_embedder_factory)? {
            graph_state
                .bind_embedder(embedder)
                .context("graph.set_embedder_native failed")?;
        }
    }
    Ok(())
}

pub(crate) async fn run_async(
    cli: Cli,
    py_embedder_factory: Option<PyEmbedderFactory>,
    extensions: ServerExtensions,
) -> Result<()> {
    let ServerExtensions {
        workspace_graph,
        domain_tools,
    } = extensions;
    init_tracing();
    let mode = pick_mode(&cli);
    validate_mode_paths(&mode, &cli)?;

    let manifest = load_manifest(&cli, &mode).context("manifest load failed")?;
    let recipe_catalog = boot_recipe_catalog(manifest.as_ref())?;
    let recipe_catalog_summary = recipe_catalog.discovery_summary();
    // Parsed here, at the top of boot, so a malformed value fails the server
    // rather than surfacing as a watcher that quietly never fires.
    let graph_watch = boot_graph_watch(manifest.as_ref())?;
    let tools_allow = boot_tools_allow(manifest.as_ref())?;
    let write_scope = boot_write_scope(manifest.as_ref(), &cli)?;
    let parallel = boot_parallel(manifest.as_ref(), &cli)?;
    if parallel {
        let (width, overridden) = query_pool_width();
        tracing::info!(
            pool_threads = width,
            width_source = if overridden {
                "KGLITE_QUERY_THREADS"
            } else {
                "available_parallelism"
            },
            "parallel runtime enabled for reads (--parallel / extensions.parallel); \
             mutations stay sequential"
        );
    }

    // Manifest `workspace.kind: local` wins over CLI flags — promote before
    // mode-specific binding so the rest of boot sees `Mode::LocalWorkspace`.
    let mode = promote_local_workspace(mode, manifest.as_ref())?;

    // Load `.env` before anything reads env vars (notably the GitHub tools'
    // `GITHUB_TOKEN` auth check). The walk-up start point matches the framework
    // binary's choice in `mcp-server`'s own main: the mode's directory for
    // source-aware modes, cwd for bare. An explicit manifest `env_file:`
    // overrides walk-up. Returns the path actually loaded, so the boot summary
    // can name it.
    let env_start_dir = resolve_env_start_dir(&mode);
    let env_file_loaded = load_env_for_mode(manifest.as_ref(), &env_start_dir)
        .context("manifest env_file load failed")?;

    let mut options = ServerOptions::from_manifest(manifest.as_ref(), fallback_name(&mode));
    if cli.name.is_some() {
        options.name = cli.name.clone();
    }
    // Fold the lazy-tool-discovery steer into workspace-mode instructions so
    // code-mode / tool-search clients get the "search the registry for cypher"
    // guidance by default, without every deployment copy-pasting it.
    let options = apply_discovery_steer(&mode, options);

    let graph_state = GraphState::new(workspace_graph_mode(&mode))
        .with_value_codecs(boot_value_codecs(manifest.as_ref())?)
        .with_parallel(parallel)
        .with_workspace_graph(workspace_graph.map(Arc::new))
        // Declared here rather than from `builtins` (built below, after the
        // csv_http boot it needs) because `bind_mode` performs the boot open on
        // the very next line — a policy set later would arrive after the lease
        // decision it governs. Both read the same `graph_writes_enabled`.
        .with_writer_lease_policy(writer_lease_policy(manifest.as_ref(), &cli));

    let options = bind_mode(&mode, &cli, manifest.as_ref(), &graph_state, options)?;

    // Runtime graph-over-grep steering (mcp-methods result-postprocess hook):
    // append a one-line footer to a builtin tool result at the moment of a
    // likely misuse — a definition-shaped or zero-match `grep`, or a
    // `cypher_query` result carrying `qualified_name`. Delivered on the RESULT
    // (read every call), it corrects course where the load-once tool
    // description could not (petekSuite field report 2026-07-02). See
    // `graph_result_footer` for when it declines and leaves the result
    // untouched.
    let options = {
        let gs = graph_state.clone();
        options.with_result_postprocess(Arc::new(
            move |tool: &str, args: &serde_json::Value, body: &str, _ctx: &ResultCtx| {
                graph_result_footer(&gs, tool, args, body)
            },
        ))
    };

    // Snapshot the dynamic source-roots provider before `options` moves into
    // the McpServer. `read_code_source` queries it on every call, so
    // workspace-mode active-repo swaps immediately re-target file resolution.
    let source_roots_provider = options.source_roots.clone();

    let manifest_base = manifest_base_dir(manifest.as_ref());
    let csv_http_cfg = boot_csv_http(manifest.as_ref(), &manifest_base).await?;
    let builtins = boot_builtins(
        manifest.as_ref(),
        &cli,
        csv_http_cfg.as_ref(),
        &manifest_base,
        write_scope,
    );
    let csv_http_arc = csv_http_cfg.map(Arc::new);

    let mut server = McpServer::new(options);
    if matches!(mode, Mode::LocalWorkspace { .. }) {
        // Local workspaces activate a directory with `set_root_dir`; the
        // GitHub clone-oriented `repo_management` tool is mutually exclusive
        // and would steer agents toward the wrong activation protocol.
        //
        // Disable, never remove: `remove_route` drops the entry from
        // `router.map`, and `apply_bundled_tool_overrides` hard-errors on any
        // manifest override naming a route that is not in that map — so a
        // local-workspace manifest carrying `bundled: repo_management` (hide,
        // rename, or description) would fail every boot with "unknown route".
        // `disable_route` keeps the entry and hides it the same way (unlisted,
        // rejected on call) while leaving overrides resolvable.
        server.tool_router_mut().disable_route("repo_management");
    }
    // Read before `builtins` moves into `register_kglite_tools` below.
    let gate_code_tools = code_tools_are_dead(&mode, &builtins, &graph_state);
    register_kglite_tools(
        &mut server,
        &graph_state,
        manifest.as_ref(),
        builtins,
        recipe_catalog_summary,
        csv_http_arc.clone(),
        source_roots_provider,
    )?;
    if matches!(mode, Mode::Graph { .. }) {
        // `reload_graph` is graph-mode-only: it re-reads *the* served file, an
        // identity no other mode has. Registered from here rather than inside
        // `tools::register`, because the mode is not otherwise visible there.
        tools::register_graph_mode_tools(&mut server, graph_state.clone());
    }
    if gate_code_tools {
        // Disable, never skip registration: an unregistered name is absent from
        // `router.map`, which hard-errors any manifest override naming it (see
        // the `repo_management` comment above for the full mechanism).
        server.tool_router_mut().disable_route("explore");
        server.tool_router_mut().disable_route("read_code_source");
    }
    bind_manifest_embedder(
        manifest.as_ref(),
        py_embedder_factory.as_ref(),
        &graph_state,
    )?;

    // Register YAML Cypher tools, then downstream domain routes. Keeping this
    // before skill finalisation makes every route visible to predicates.
    register_extension_tools(
        &mut server,
        &graph_state,
        manifest.as_ref(),
        &csv_http_arc,
        recipe_catalog,
        domain_tools,
    )?;

    // Last, and only here: the allowlist closes the surface once *every* route
    // source has registered and `apply_bundled_tool_overrides` (at the tail of
    // `register_extension_tools`) has settled the final names. Applying it any
    // earlier would let a later registration re-widen the surface, and would
    // match against pre-rename names the agent never sees. Before skills, so a
    // `tool_registered:` predicate sees the closed surface.
    if let Some(allow) = tools_allow.as_deref() {
        apply_tool_allowlist(&mut server, allow, recipe_catalog_summary.is_some())
            .context("extensions.tools_allow could not be applied")?;
    }

    let _watch_handle = spawn_mode_watcher(&mode, &graph_state, graph_watch)?;

    // Bare-mode (no manifest) deployments get no skills — the `skills:`
    // declaration lives in the manifest.
    if let Some(m) = manifest.as_ref() {
        install_skills(&mut server, m, &graph_state, recipe_catalog_summary);
    }

    print_boot_summary(
        &mode,
        manifest.as_ref(),
        &graph_state,
        env_file_loaded.as_deref(),
    );

    let service = server
        .serve(stdio())
        .await
        .context("failed to start MCP service over stdio")?;
    service.waiting().await?;
    Ok(())
}

/// Every boot knob read here must survive the whole chain: spelled in a YAML
/// file, accepted by mcp-methods' loader (which validates nothing inside
/// `extensions:`), and read back here as the type the wiring expects. Loading
/// through the real loader rather than constructing a `Manifest` is deliberate
/// — a struct literal would skip the two steps most likely to break.
#[cfg(test)]
mod boot_manifest_tests {
    use super::*;

    fn manifest_with(dir: &Path, body: &str) -> mcp_methods::server::Manifest {
        let path = dir.join("graph_watch_mcp.yaml");
        std::fs::write(&path, body).expect("write manifest");
        mcp_methods::server::load_manifest(&path).expect("manifest loads")
    }

    #[test]
    fn graph_watch_defaults_off_and_reads_the_manifest_key() {
        let tmp = tempfile::tempdir().expect("tempdir");
        assert!(
            !boot_graph_watch(None).expect("no manifest parses"),
            "no manifest must leave the watcher off"
        );
        let bare = manifest_with(tmp.path(), "name: bare\n");
        assert!(
            !boot_graph_watch(Some(&bare)).expect("bare manifest parses"),
            "an absent key must leave the watcher off — otherwise the on-arm proves nothing"
        );
        let off = manifest_with(tmp.path(), "name: off\nextensions:\n  graph_watch: false\n");
        assert!(!boot_graph_watch(Some(&off)).expect("explicit false parses"));
        let on = manifest_with(tmp.path(), "name: on\nextensions:\n  graph_watch: true\n");
        assert!(
            boot_graph_watch(Some(&on)).expect("explicit true parses"),
            "extensions.graph_watch: true must reach the watcher wiring; if this fails \
             the key is parsed and then dropped"
        );
    }

    /// Boot wiring for the writer-lease policy: which deployments own the
    /// served path. The state-level consequences (lease taken or not, disk
    /// directories exempt) are pinned in `tools::tests::lifecycle`; this pins
    /// only the mapping from CLI/manifest to policy.
    #[test]
    fn only_write_enabled_deployments_take_the_writer_lease() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let read_only = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);
        assert_eq!(
            writer_lease_policy(None, &read_only),
            tools::WriterLeasePolicy::ReadOnly,
            "a read-only --graph server must leave the file lockable by a rebuilder"
        );

        let writable = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl", "--writable"]);
        assert_eq!(
            writer_lease_policy(None, &writable),
            tools::WriterLeasePolicy::Exclusive,
            "--writable owns the file it serves"
        );

        // `save_graph` without `--writable`: still a writer of that file.
        let saver = manifest_with(tmp.path(), "name: saver\nbuiltins:\n  save_graph: true\n");
        assert_eq!(
            writer_lease_policy(Some(&saver), &read_only),
            tools::WriterLeasePolicy::Exclusive
        );
        let plain = manifest_with(tmp.path(), "name: plain\n");
        assert_eq!(
            writer_lease_policy(Some(&plain), &read_only),
            tools::WriterLeasePolicy::ReadOnly,
            "a manifest that enables nothing must not re-arm the lease"
        );
    }

    /// The absent-key arm is what makes the present-key arm mean anything.
    #[test]
    fn tools_allow_defaults_to_absent_and_reads_the_manifest_list() {
        let tmp = tempfile::tempdir().expect("tempdir");
        assert!(boot_tools_allow(None)
            .expect("no manifest parses")
            .is_none());
        let bare = manifest_with(tmp.path(), "name: bare\n");
        assert!(
            boot_tools_allow(Some(&bare))
                .expect("bare manifest parses")
                .is_none(),
            "an absent key must leave the surface open"
        );

        let listed = manifest_with(
            tmp.path(),
            "name: listed\nextensions:\n  tools_allow:\n    - cypher_query\n    - graph_overview\n    - ping\n",
        );
        assert_eq!(
            boot_tools_allow(Some(&listed)).expect("list parses"),
            Some(vec![
                "cypher_query".to_string(),
                "graph_overview".to_string(),
                "ping".to_string(),
            ]),
            "extensions.tools_allow must reach the router wiring; if this fails the key is \
             parsed and then dropped"
        );

        let empty = manifest_with(tmp.path(), "name: empty\nextensions:\n  tools_allow: []\n");
        assert_eq!(
            boot_tools_allow(Some(&empty)).expect("empty list parses"),
            Some(Vec::new()),
            "an explicit empty list closes the surface completely — it is not 'no allowlist'"
        );
    }

    #[test]
    fn a_malformed_tools_allow_fails_boot() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let scalar = manifest_with(
            tmp.path(),
            "name: scalar\nextensions:\n  tools_allow: cypher_query\n",
        );
        let error = boot_tools_allow(Some(&scalar))
            .expect_err("a scalar must fail boot, not be ignored")
            .to_string();
        assert!(error.contains("must be a list"), "{error}");

        let null = manifest_with(tmp.path(), "name: null\nextensions:\n  tools_allow:\n");
        assert!(
            boot_tools_allow(Some(&null)).is_err(),
            "an empty (null) value is a typo, not an empty list"
        );

        let mistyped = manifest_with(
            tmp.path(),
            "name: mistyped\nextensions:\n  tools_allow:\n    - ping\n    - 7\n",
        );
        let error = boot_tools_allow(Some(&mistyped))
            .expect_err("a non-string element must fail boot")
            .to_string();
        assert!(error.contains("must be tool names"), "{error}");
    }

    #[test]
    fn write_scope_reads_the_flag_the_manifest_and_their_intersection() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let bare_cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl", "--writable"]);
        assert!(
            boot_write_scope(None, &bare_cli)
                .expect("no pin parses")
                .is_none(),
            "no flag and no manifest must leave writes unpinned"
        );

        let flag = Cli::parse_from([
            "kglite-mcp-server",
            "--graph",
            "g.kgl",
            "--writable",
            "--write-scope",
            "Plan,Task",
        ]);
        assert_eq!(
            boot_write_scope(None, &flag).expect("flag parses"),
            Some(vec!["Plan".to_string(), "Task".to_string()])
        );

        let listed = manifest_with(
            tmp.path(),
            "name: listed\nextensions:\n  write_scope:\n    - Plan\n    - Task\n",
        );
        assert_eq!(
            boot_write_scope(Some(&listed), &bare_cli).expect("list parses"),
            Some(vec!["Plan".to_string(), "Task".to_string()]),
            "extensions.write_scope must reach the tool wiring; if this fails the key is \
             parsed and then dropped"
        );

        // Both surfaces set: neither can widen the other.
        let narrower = Cli::parse_from([
            "kglite-mcp-server",
            "--graph",
            "g.kgl",
            "--writable",
            "--write-scope",
            "Task,Algorithm",
        ]);
        assert_eq!(
            boot_write_scope(Some(&listed), &narrower).expect("both parse"),
            Some(vec!["Task".to_string()]),
            "flag + manifest is their intersection, not either one alone"
        );

        let empty = manifest_with(tmp.path(), "name: empty\nextensions:\n  write_scope: []\n");
        assert_eq!(
            boot_write_scope(Some(&empty), &bare_cli).expect("empty list parses"),
            Some(Vec::new()),
            "an explicit empty list pins 'no writes' — it is not 'no pin'"
        );
    }

    #[test]
    fn a_malformed_write_scope_fails_boot() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl", "--writable"]);

        let scalar = manifest_with(
            tmp.path(),
            "name: scalar\nextensions:\n  write_scope: Task\n",
        );
        let error = boot_write_scope(Some(&scalar), &cli)
            .expect_err("a scalar must fail boot, not be ignored")
            .to_string();
        assert!(error.contains("must be a list"), "{error}");

        let null = manifest_with(tmp.path(), "name: null\nextensions:\n  write_scope:\n");
        assert!(
            boot_write_scope(Some(&null), &cli).is_err(),
            "an empty (null) value is a typo, not an empty list"
        );

        let mistyped = manifest_with(
            tmp.path(),
            "name: mistyped\nextensions:\n  write_scope:\n    - Task\n    - 7\n",
        );
        let error = boot_write_scope(Some(&mistyped), &cli)
            .expect_err("a non-string element must fail boot")
            .to_string();
        assert!(error.contains("must be node types"), "{error}");
    }

    #[test]
    fn a_non_boolean_graph_watch_fails_boot() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let bad = manifest_with(
            tmp.path(),
            "name: bad\nextensions:\n  graph_watch: yes-please\n",
        );
        let error = boot_graph_watch(Some(&bad))
            .expect_err("a non-boolean value must fail boot, not be ignored")
            .to_string();
        assert!(error.contains("must be a boolean"), "{error}");
    }

    /// The parallel opt-in is an operator *pair* — `--parallel` and
    /// `extensions.parallel` — so the boot resolution is the OR of the two.
    /// Pinned here because a manifest-only deployment (a wrapper that never
    /// controls argv) and a flag-only one (the bare binary, no manifest) are
    /// both real, and either surface silently losing its half looks exactly
    /// like "the knob does nothing".
    #[test]
    fn parallel_reads_the_flag_and_the_manifest() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let bare_cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);
        let flag_cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl", "--parallel"]);

        assert!(
            !boot_parallel(None, &bare_cli).expect("no manifest parses"),
            "neither surface set must leave the engine sequential"
        );
        let bare = manifest_with(tmp.path(), "name: bare\n");
        assert!(
            !boot_parallel(Some(&bare), &bare_cli).expect("bare manifest parses"),
            "an absent key must leave the engine sequential — otherwise the on-arm proves nothing"
        );
        let off = manifest_with(tmp.path(), "name: off\nextensions:\n  parallel: false\n");
        assert!(!boot_parallel(Some(&off), &bare_cli).expect("explicit false parses"));

        assert!(
            boot_parallel(None, &flag_cli).expect("flag alone parses"),
            "--parallel must reach the execution wiring; if this fails the flag is parsed \
             and then dropped"
        );
        let on = manifest_with(tmp.path(), "name: on\nextensions:\n  parallel: true\n");
        assert!(
            boot_parallel(Some(&on), &bare_cli).expect("explicit true parses"),
            "extensions.parallel: true must reach the execution wiring; if this fails the \
             key is parsed and then dropped"
        );
        assert!(
            boot_parallel(Some(&on), &flag_cli).expect("both parse"),
            "flag + manifest is their OR — neither surface can switch the other off"
        );
        assert!(
            boot_parallel(Some(&off), &flag_cli).expect("both parse"),
            "an explicit manifest `false` does not veto the flag; the pair is an OR, and a \
             flag the operator typed on this launch is the more specific statement"
        );
    }

    #[test]
    fn a_malformed_parallel_fails_boot() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);

        let word = manifest_with(
            tmp.path(),
            "name: word\nextensions:\n  parallel: yes-please\n",
        );
        let error = boot_parallel(Some(&word), &cli)
            .expect_err("a non-boolean value must fail boot, not be ignored")
            .to_string();
        assert!(error.contains("must be a boolean"), "{error}");

        let null = manifest_with(tmp.path(), "name: null\nextensions:\n  parallel:\n");
        assert!(
            boot_parallel(Some(&null), &cli).is_err(),
            "an empty (null) value is a typo, not `false`"
        );

        let listed = manifest_with(
            tmp.path(),
            "name: listed\nextensions:\n  parallel:\n    - true\n",
        );
        assert!(
            boot_parallel(Some(&listed), &cli).is_err(),
            "a list is a typo, not a truthy value"
        );
    }
}
