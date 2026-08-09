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

/// P4 + P5 (operator feedback): builtin toggles from the manifest.
///   - P5 `save_graph`: gate registration on `builtins.save_graph: true`.
///     Historically always-on, exposing a destructive operation to the agent on
///     every graph regardless of intent.
///   - P4 `temp_cleanup: on_overview`: wipe `temp/` on every bare
///     `graph_overview()`. Historically parsed-but-ignored.
fn boot_builtins(
    manifest: Option<&mcp_methods::server::Manifest>,
    cli: &Cli,
    csv_http_cfg: Option<&csv_http::CsvHttpConfig>,
    manifest_base: &Path,
) -> tools::Builtins {
    tools::Builtins {
        save_graph: manifest.map(|m| m.builtins.save_graph).unwrap_or(false)
            // Write mode implies save_graph (an agent that mutates needs to
            // persist) so `--writable` alone gives the full workbench.
            || cli.writable,
        writable: cli.writable,
        temp_cleanup_on_overview: manifest
            .map(|m| {
                matches!(
                    m.builtins.temp_cleanup,
                    mcp_methods::server::manifest::TempCleanup::OnOverview
                )
            })
            .unwrap_or(false),
        // 0.9.19 fix: temp_cleanup target dir was hardcoded to `./temp`
        // (cwd-relative) — that's the wrong place to look when the
        // server's cwd doesn't match the manifest's parent. Resolve
        // against the manifest base, reusing the csv_http_server
        // directory when configured so both sides of the CSV pipeline
        // agree on what counts as "the temp dir".
        temp_dir: Some(
            csv_http_cfg
                .map(|c| c.dir.clone())
                .unwrap_or_else(|| manifest_base.join("temp")),
        ),
    }
}

/// Register KGLite's own routes: the graph/Cypher tools plus the two
/// source-reading tools that follow the dynamic source-roots provider.
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

/// `extensions.embedder:` in the manifest selects the embedding backend for
/// `text_score()`:
///   - `backend: fastembed` — the Rust-native fastembed-rs adapter (cargo
///     `--features fastembed`). The only option for the standalone
///     libpython-free binary.
///   - `backend: python` — a fastembed-py model, built by the wheel's
///     `_run_mcp_server` factory and wrapped in a `PyEmbedderAdapter`. Only
///     available when a factory is supplied (the pip-hosted server); the cargo
///     binary rejects it with a clear message.
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

    // Manifest `workspace.kind: local` wins over CLI flags — promote before
    // mode-specific binding so the rest of boot sees `Mode::LocalWorkspace`.
    let mode = promote_local_workspace(mode, manifest.as_ref())?;

    // Load `.env` before anything reads env vars (notably the GitHub
    // tools' `GITHUB_TOKEN` auth check). Walk-up start point matches
    // the framework binary's choice in `mcp-server`'s own main: the
    // mode's directory for source-aware modes, cwd for bare. Explicit
    // `env_file:` in the manifest overrides walk-up. Returns the path
    // actually loaded so the boot summary can name it.
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
        .with_workspace_graph(workspace_graph.map(Arc::new));

    // Mode-specific bindings: source roots, workspace handle, initial graph
    // build. Extracted to `bind_mode` so this boot fn reads as a sequence of
    // named phases.
    let options = bind_mode(&mode, &cli, manifest.as_ref(), &graph_state, options)?;

    // Runtime graph-over-grep steering (mcp-methods 0.3.46 result-postprocess
    // hook): append a one-line footer to a builtin tool result at the moment of
    // a likely misuse — a definition-shaped or zero-match `grep`, or a
    // `cypher_query` result carrying `qualified_name`. Delivered on the RESULT
    // (read every call), it corrects course where the load-once tool
    // description could not (petekSuite field report 2026-07-02). Returns `None`
    // — leaving the result byte-for-byte unchanged — unless a code graph is
    // active and the shape matches, so non-code deployments are untouched.
    let options = {
        let gs = graph_state.clone();
        options.with_result_postprocess(Arc::new(
            move |tool: &str, args: &serde_json::Value, body: &str, _ctx: &ResultCtx| {
                graph_result_footer(&gs, tool, args, body)
            },
        ))
    };

    // Snapshot the dynamic source-roots provider before we move
    // `options` into the McpServer. The `read_code_source` tool
    // queries it on every call so workspace-mode active-repo swaps
    // immediately re-target file resolution.
    let source_roots_provider = options.source_roots.clone();

    let manifest_base = manifest_base_dir(manifest.as_ref());
    let csv_http_cfg = boot_csv_http(manifest.as_ref(), &manifest_base).await?;
    let builtins = boot_builtins(
        manifest.as_ref(),
        &cli,
        csv_http_cfg.as_ref(),
        &manifest_base,
    );
    let csv_http_arc = csv_http_cfg.map(Arc::new);

    let mut server = McpServer::new(options);
    if matches!(mode, Mode::LocalWorkspace { .. }) {
        // Local workspaces activate a directory with `set_root_dir`; the
        // GitHub clone-oriented `repo_management` tool is mutually exclusive
        // and would steer agents toward the wrong activation protocol.
        server.tool_router_mut().remove_route("repo_management");
    }
    register_kglite_tools(
        &mut server,
        &graph_state,
        manifest.as_ref(),
        builtins,
        recipe_catalog_summary,
        csv_http_arc.clone(),
        source_roots_provider,
    )?;
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

    let _watch_handle = spawn_mode_watcher(&mode, &graph_state)?;

    // Bare-mode (no manifest) deployments don't get skills — the
    // `skills:` declaration lives in the manifest. Operators who want
    // skills must declare them in YAML.
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
