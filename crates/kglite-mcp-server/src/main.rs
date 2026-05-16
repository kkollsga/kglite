//! `kglite-mcp-server` — single-binary MCP server for KGLite knowledge graphs.
//!
//! Layers three kglite-specific tools on top of the generic
//! `mcp-server` framework: `cypher_query`, `graph_overview`, and
//! `save_graph`. All three close over a [`GraphState`] holding the
//! active `KnowledgeGraph` Python object.
//!
//! Modes:
//! - `--graph X.kgl` — load a pre-built graph file at boot.
//! - `--workspace DIR` — multi-repo. Post-activate hook runs
//!   `kglite.code_tree.build()` on each cloned repo.
//! - `--watch DIR` — file-watcher mode. Change handler rebuilds the
//!   code-tree graph and atomic-swaps the active slot.
//! - `--source-root DIR` — generic file-tree mode (no graph).
//! - bare — framework + manifest tools only.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use mcp_methods::server::manifest::{
    find_sibling_manifest, find_workspace_manifest, ManifestError,
};
use mcp_methods::server::{
    init_tracing, load_env_for_mode, maybe_watch, resolve_source_roots, serve_prompts, watch,
    workspace, BundledSkill, Manifest, McpServer, PredicateClause, ServerOptions,
    SkillPredicateEvaluator, SkillRegistry, WorkspaceKind,
};
use rmcp::transport::stdio;
use rmcp::ServiceExt;

mod code_source;
mod csv_http;
mod cypher_tools;
mod explore;
mod tools;
use crate::tools::GraphState;

#[derive(Parser, Debug)]
#[command(
    name = "kglite-mcp-server",
    about = "MCP server for KGLite knowledge graphs (Rust-native)"
)]
struct Cli {
    /// Path to a .kgl knowledge graph file. Loaded at boot.
    #[arg(long, conflicts_with_all = ["workspace", "watch", "source_root"])]
    graph: Option<PathBuf>,

    /// Source-root mode (no graph).
    #[arg(long = "source-root", conflicts_with_all = ["graph", "workspace", "watch"])]
    source_root: Option<PathBuf>,

    /// Workspace mode: clone GitHub repos and build code-tree graphs.
    #[arg(long, conflicts_with_all = ["graph", "source_root", "watch"])]
    workspace: Option<PathBuf>,

    /// Watch mode: rebuild the code-tree graph on file changes.
    #[arg(long, conflicts_with_all = ["graph", "source_root", "workspace"])]
    watch: Option<PathBuf>,

    #[arg(long = "mcp-config")]
    mcp_config: Option<PathBuf>,
    #[arg(long)]
    name: Option<String>,
    #[arg(long = "trust-tools")]
    #[allow(dead_code)]
    trust_tools: bool,
    #[arg(long = "stale-after-days", default_value_t = 7)]
    stale_after_days: u32,
}

#[derive(Debug, Clone)]
enum Mode {
    Graph {
        path: PathBuf,
    },
    SourceRoot {
        dir: PathBuf,
    },
    Workspace {
        dir: PathBuf,
    },
    /// `manifest.workspace.kind: local`. Equivalent to `--workspace`
    /// but bound to a fixed local directory (no clone) and with
    /// `set_root_dir` registered for runtime root swap. Manifest
    /// declaration wins over the `--workspace` CLI flag.
    LocalWorkspace {
        root: PathBuf,
        watch: bool,
    },
    Watch {
        dir: PathBuf,
    },
    Bare,
}

fn pick_mode(cli: &Cli) -> Mode {
    if let Some(p) = &cli.graph {
        Mode::Graph { path: p.clone() }
    } else if let Some(d) = &cli.source_root {
        Mode::SourceRoot { dir: d.clone() }
    } else if let Some(d) = &cli.workspace {
        Mode::Workspace { dir: d.clone() }
    } else if let Some(d) = &cli.watch {
        Mode::Watch { dir: d.clone() }
    } else {
        Mode::Bare
    }
}

fn fallback_name(mode: &Mode) -> &'static str {
    match mode {
        Mode::Graph { .. } => "KGLite (single-graph)",
        Mode::SourceRoot { .. } => "KGLite (source-root)",
        Mode::Workspace { .. } => "KGLite (workspace)",
        Mode::LocalWorkspace { .. } => "KGLite (local-workspace)",
        Mode::Watch { .. } => "KGLite (watch)",
        Mode::Bare => "KGLite",
    }
}

fn default_manifest_path(mode: &Mode) -> Option<PathBuf> {
    match mode {
        Mode::Graph { path } => find_sibling_manifest(path),
        Mode::Workspace { dir } | Mode::Watch { dir } => find_workspace_manifest(dir),
        Mode::LocalWorkspace { root, .. } => find_workspace_manifest(root),
        Mode::SourceRoot { .. } | Mode::Bare => None,
    }
}

fn load_manifest(cli: &Cli, mode: &Mode) -> Result<Option<Manifest>, ManifestError> {
    let path = match &cli.mcp_config {
        Some(p) if !p.is_file() => {
            return Err(ManifestError::bare(format!(
                "--mcp-config path does not exist: {}",
                p.display()
            )))
        }
        Some(p) => Some(p.clone()),
        None => default_manifest_path(mode),
    };
    match path {
        Some(p) => Ok(Some(mcp_methods::server::load_manifest(&p)?)),
        None => Ok(None),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    let mut mode = pick_mode(&cli);

    if let Mode::Graph { path } = &mode {
        if !path.is_file() {
            anyhow::bail!("--graph path does not exist: {}", path.display());
        }
    }
    if let Mode::SourceRoot { dir } | Mode::Watch { dir } = &mode {
        if !dir.is_dir() {
            anyhow::bail!(
                "path does not exist or is not a directory: {}",
                dir.display()
            );
        }
    }

    let manifest = load_manifest(&cli, &mode).context("manifest load failed")?;

    // Manifest `workspace.kind: local` wins over CLI flags — promote
    // before mode-specific binding runs so the rest of the boot path
    // sees `Mode::LocalWorkspace`. Mirrors the framework's own
    // `mcp-server` binary (`crates/mcp-server/src/main.rs` in 0.3.23+).
    if let Some(m) = manifest.as_ref() {
        if let Some(wcfg) = m.workspace.as_ref() {
            if wcfg.kind == WorkspaceKind::Local {
                let raw_root = wcfg.root.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("manifest.workspace.kind=local is missing required `root`")
                })?;
                let base = m
                    .yaml_path
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from("."));
                let resolved = base.join(raw_root).canonicalize().with_context(|| {
                    format!("workspace.root {raw_root:?} resolves to a path that does not exist")
                })?;
                mode = Mode::LocalWorkspace {
                    root: resolved,
                    watch: wcfg.watch,
                };
            }
        }
    }

    // Load `.env` before anything reads env vars (notably the GitHub
    // tools' `GITHUB_TOKEN` auth check). Walk-up start point matches
    // the framework binary's choice in `mcp-server`'s own main: the
    // mode's directory for source-aware modes, cwd for bare. Explicit
    // `env_file:` in the manifest overrides walk-up. Returns the path
    // actually loaded so the boot summary can name it.
    let env_start_dir: PathBuf = match &mode {
        Mode::Graph { path } => path
            .canonicalize()
            .ok()
            .and_then(|p| p.parent().map(PathBuf::from))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))),
        Mode::SourceRoot { dir } | Mode::Workspace { dir } | Mode::Watch { dir } => dir.clone(),
        Mode::LocalWorkspace { root, .. } => root.clone(),
        Mode::Bare => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    };
    let env_file_loaded = load_env_for_mode(manifest.as_ref(), &env_start_dir)
        .context("manifest env_file load failed")?;

    let mut options = ServerOptions::from_manifest(manifest.as_ref(), fallback_name(&mode));
    if cli.name.is_some() {
        options.name = cli.name.clone();
    }

    let graph_state = GraphState::new();

    // Mode-specific bindings: source roots, workspace handle, initial graph build.
    match &mode {
        Mode::Graph { path } => {
            let canon = path.canonicalize()?;
            graph_state.load_kgl(&canon).context("kglite.load failed")?;
            // P1 (operator feedback): honor the manifest's explicit
            // `source_root:` / `source_roots:` declaration in `--graph`
            // mode. The historical behaviour auto-bound the parent of
            // the `.kgl` file as the source root, which silently
            // overrode operators who declared a different root in
            // YAML (e.g. when the .kgl lives in a build dir but the
            // source files are elsewhere). Now: explicit YAML wins,
            // auto-bind only when the manifest doesn't declare one.
            let manifest_roots = manifest
                .as_ref()
                .filter(|m| !m.source_roots.is_empty())
                .map(resolve_source_roots)
                .transpose()
                .context("manifest source_root resolution failed")?;
            let roots = if let Some(rs) = manifest_roots {
                rs
            } else if let Some(parent) = canon.parent() {
                vec![parent.to_string_lossy().into_owned()]
            } else {
                Vec::new()
            };
            if !roots.is_empty() {
                options = options.with_static_source_roots(roots);
            }
        }
        Mode::SourceRoot { dir } | Mode::Watch { dir } => {
            let canon = dir.canonicalize()?;
            options = options.with_static_source_roots(vec![canon.to_string_lossy().into_owned()]);
            if matches!(mode, Mode::Watch { .. }) {
                graph_state
                    .build_code_tree(&canon)
                    .context("initial code_tree build failed")?;
            }
        }
        Mode::Workspace { dir } => {
            let canon = dir.canonicalize().unwrap_or_else(|_| dir.clone());
            let gs = graph_state.clone();
            let hook: workspace::PostActivateHook = Arc::new(move |path, name| {
                tracing::info!(repo = name, "code_tree::build on activate");
                gs.build_code_tree(path)
            });
            let ws = workspace::Workspace::open(canon, cli.stale_after_days, Some(hook))
                .context("workspace init failed")?;
            options = options.with_workspace(ws);
        }
        Mode::LocalWorkspace { root, .. } => {
            let gs = graph_state.clone();
            let hook: workspace::PostActivateHook = Arc::new(move |path, name| {
                tracing::info!(repo = name, "code_tree::build on local-workspace activate");
                gs.build_code_tree(path)
            });
            let ws = workspace::Workspace::open_local(root.clone(), Some(hook))
                .context("local-workspace init failed")?;
            options = options.with_workspace(ws);
        }
        Mode::Bare => {
            if let Some(m) = manifest.as_ref() {
                if !m.source_roots.is_empty() {
                    let resolved =
                        resolve_source_roots(m).context("source root resolution failed")?;
                    options = options.with_static_source_roots(resolved);
                }
            }
        }
    }

    // Snapshot the dynamic source-roots provider before we move
    // `options` into the McpServer. The `read_code_source` tool
    // queries it on every call so workspace-mode active-repo swaps
    // immediately re-target file resolution.
    let source_roots_provider = options.source_roots.clone();

    // P4 + P5 (operator feedback): builtin toggles from the manifest.
    //   - P5 `save_graph`: gate registration on
    //     `builtins.save_graph: true`. Historically always-on,
    //     exposing a destructive operation to the agent on every
    //     graph regardless of intent.
    //   - P4 `temp_cleanup: on_overview`: wipe `temp/` on every bare
    //     `graph_overview()`. Historically parsed-but-ignored.
    // Manifest base dir — used by both csv_http_server (to resolve
    // `dir:` against the YAML location) and temp_cleanup (to find the
    // directory to wipe). Falls back to cwd when there's no manifest.
    let manifest_base: PathBuf = manifest
        .as_ref()
        .and_then(|m| m.yaml_path.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    // `extensions.csv_http_server:` opt-in CSV-over-HTTP listener.
    // When configured we spawn a tokio task to serve files out of
    // the directory; the `cypher_query` tool sees the same config
    // and writes `FORMAT CSV` results to that directory, returning
    // a URL instead of an inline CSV blob.
    let csv_http_cfg = match manifest.as_ref() {
        Some(m) => match m.extensions.get("csv_http_server") {
            Some(raw) => csv_http::CsvHttpConfig::from_manifest_value(raw, &manifest_base)
                .context("extensions.csv_http_server parse failed")?,
            None => None,
        },
        None => None,
    };
    if let Some(cfg) = csv_http_cfg.as_ref() {
        csv_http::spawn(cfg.clone())
            .await
            .context("csv_http_server failed to bind")?;
    }

    let builtins = tools::Builtins {
        save_graph: manifest
            .as_ref()
            .map(|m| m.builtins.save_graph)
            .unwrap_or(false),
        temp_cleanup_on_overview: manifest
            .as_ref()
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
                .as_ref()
                .map(|c| c.dir.clone())
                .unwrap_or_else(|| manifest_base.join("temp")),
        ),
    };

    let csv_http_arc = csv_http_cfg.map(Arc::new);

    let mut server = McpServer::new(options);
    tools::register(
        &mut server,
        graph_state.clone(),
        builtins,
        csv_http_arc.clone(),
    );
    code_source::register(
        &mut server,
        graph_state.clone(),
        source_roots_provider.clone(),
    )
    .context("read_code_source registration failed")?;
    explore::register(
        &mut server,
        graph_state.clone(),
        source_roots_provider.clone(),
    )
    .context("explore registration failed")?;

    // `extensions.embedder:` in the manifest selects a Rust-native
    // embedder backend (fastembed-rs in 0.9.18). The 0.9.17-and-prior
    // `embedder:` block + Python embedder factories are gone — the
    // binary has no libpython link at all and so cannot host a Python
    // embedder class. See the 0.9.18 migration note in
    // `docs/guides/mcp-servers.md` for the YAML schema change.
    if let Some(m) = manifest.as_ref() {
        if let Some(embedder) = build_embedder_from_manifest(m)? {
            graph_state
                .bind_embedder(embedder)
                .context("graph.set_embedder_native failed")?;
        }
    }

    // YAML-declared `tools[].cypher` entries. The mcp-methods framework
    // parses them into `manifest.tools` but stays domain-agnostic and
    // doesn't know how to run Cypher — so the kglite shim owns the
    // registration loop, using the framework's now-public
    // `build_tool_attr` plus an in-shim runner that dispatches into the
    // active graph's `cypher(template, params=args)` method.
    if let Some(m) = manifest.as_ref() {
        let runner = cypher_tools::make_runner(graph_state.clone(), csv_http_arc.clone());
        let registered = cypher_tools::register_cypher_tools(&mut server, m, runner)
            .context("YAML cypher tool registration failed")?;
        if registered > 0 {
            tracing::info!(count = registered, "manifest cypher tools registered");
        }
    }

    // Watch handler: rebuild on every debounced change batch. Both
    // explicit `--watch DIR` and `manifest.workspace.kind: local` with
    // `watch: true` wire the same change-handler shape.
    let watch_handle = match &mode {
        Mode::Watch { dir } => {
            let canon = dir.canonicalize()?;
            let gs = graph_state.clone();
            let cb: watch::ChangeHandler = Arc::new(move |_paths| {
                if let Err(e) = gs.build_code_tree(&canon) {
                    tracing::warn!(error = %e, "code_tree rebuild failed");
                }
            });
            maybe_watch(Some(dir), Some(cb))?
        }
        Mode::LocalWorkspace { root, watch: true } => {
            let canon = root.clone();
            let gs = graph_state.clone();
            let cb: watch::ChangeHandler = Arc::new(move |_paths| {
                if let Err(e) = gs.build_code_tree(&canon) {
                    tracing::warn!(error = %e, "code_tree rebuild failed (local workspace)");
                }
            });
            maybe_watch(Some(root), Some(cb))?
        }
        _ => None,
    };
    let _watch_handle = watch_handle;

    // 0.9.31: SkillRegistry wiring. Bundled methodology for kglite's
    // four custom tools (cypher_query / graph_overview / save_graph /
    // read_code_source) plus framework defaults (grep / read_source /
    // list_source / github_issues / repo_management), composed with
    // the operator-side project layer and any operator-declared
    // domain skill packs. The predicate evaluator gates
    // `read_code_source` on `graph_has_node_type: [Function, Class]`
    // so it stays out of prompts/list when the active graph isn't a
    // code-tree (legal-corpus / o&g / etc. deployments).
    if let Some(m) = manifest.as_ref() {
        let registry_result = SkillRegistry::new()
            .add_bundled(BundledSkill {
                name: "cypher_query",
                body: include_str!("../../../kglite/mcp_server/skills/cypher_query.md"),
            })
            .add_bundled(BundledSkill {
                name: "graph_overview",
                body: include_str!("../../../kglite/mcp_server/skills/graph_overview.md"),
            })
            .add_bundled(BundledSkill {
                name: "save_graph",
                body: include_str!("../../../kglite/mcp_server/skills/save_graph.md"),
            })
            .add_bundled(BundledSkill {
                name: "read_code_source",
                body: include_str!("../../../kglite/mcp_server/skills/read_code_source.md"),
            })
            .add_bundled(BundledSkill {
                name: "explore",
                body: include_str!("../../../kglite/mcp_server/skills/explore.md"),
            })
            .merge_framework_defaults()
            .auto_detect_project_layer(&m.yaml_path)
            .layer_dirs(&m.skills, &m.yaml_path)
            .and_then(|r| {
                r.with_predicate_evaluator(KglitePredicateEvaluator {
                    state: graph_state.clone(),
                })
                .finalise()
            });
        match registry_result {
            Ok(registry) => serve_prompts(&registry, &mut server),
            Err(e) => {
                tracing::warn!(error = %e, "skills registry build failed; skills disabled for this session");
            }
        }
    }
    // Bare-mode (no manifest) deployments don't get skills — the
    // `skills:` declaration lives in the manifest. Operators who want
    // skills must declare them in YAML.

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

/// Evaluates `applies_when:` predicates that depend on kglite's
/// runtime graph state. The framework dispatches `tool_registered:`
/// and `extension_enabled:` itself; this evaluator only handles the
/// two domain predicates that require knowing what node types and
/// properties the active graph carries.
///
/// Returning `None` for an unrecognised clause marks the predicate
/// `Unknown` upstream — the framework's safe default suppresses the
/// skill when any clause is `Unknown`, which prevents a typo'd
/// predicate from silently activating a skill against the wrong
/// domain.
struct KglitePredicateEvaluator {
    state: GraphState,
}

impl SkillPredicateEvaluator for KglitePredicateEvaluator {
    fn evaluate(&self, clause: &PredicateClause<'_>) -> Option<bool> {
        match clause {
            PredicateClause::GraphHasNodeType(types) => {
                Some(types.iter().any(|t| self.state.has_node_type(t)))
            }
            PredicateClause::GraphHasProperty {
                node_type,
                prop_name,
            } => Some(self.state.has_property(node_type, prop_name)),
            // Framework-internal predicates — `tool_registered` and
            // `extension_enabled` are dispatched against ServerOptions
            // by the framework itself, not via this evaluator.
            _ => None,
        }
    }
}

/// Read `manifest.extensions.embedder.{backend, model, cooldown}` and
/// build the corresponding Rust-native [`kglite::api::Embedder`]
/// implementation. Returns `Ok(None)` when no `embedder:` is declared,
/// `Err` on validation failures (unknown backend, missing fields).
///
/// 0.9.18 supports a single backend (`fastembed`) — a future release
/// may add `candle` or `onnx-runtime` here without changing the YAML
/// shape.
fn build_embedder_from_manifest(
    manifest: &Manifest,
) -> Result<Option<Arc<dyn kglite::api::Embedder>>> {
    let Some(raw) = manifest.extensions.get("embedder") else {
        return Ok(None);
    };
    let obj = raw
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("extensions.embedder must be a mapping (got: {raw:?})"))?;
    let backend = obj
        .get("backend")
        .and_then(|v| v.as_str())
        .unwrap_or("fastembed");
    match backend {
        "fastembed" => {
            let model = obj.get("model").and_then(|v| v.as_str()).ok_or_else(|| {
                anyhow::anyhow!("extensions.embedder.model is required for the fastembed backend")
            })?;
            let adapter = kglite::api::FastEmbedAdapter::new(model)
                .map_err(|e| anyhow::anyhow!("fastembed init failed: {e}"))?;
            tracing::info!(model, backend, "registered Rust-native embedder");
            Ok(Some(Arc::new(adapter)))
        }
        other => anyhow::bail!(
            "extensions.embedder.backend = {other:?} is not supported. \
             Known: fastembed."
        ),
    }
}

fn print_boot_summary(
    mode: &Mode,
    manifest: Option<&Manifest>,
    graph_state: &GraphState,
    env_file_loaded: Option<&std::path::Path>,
) {
    let label = match mode {
        Mode::Graph { path } => format!("graph [{}]", path.display()),
        Mode::SourceRoot { dir } => format!("source-root [{}]", dir.display()),
        Mode::Workspace { dir } => format!("workspace [{}]", dir.display()),
        Mode::LocalWorkspace { root, watch } => format!(
            "local-workspace [{}{}]",
            root.display(),
            if *watch { " +watch" } else { "" }
        ),
        Mode::Watch { dir } => format!("watch [{}]", dir.display()),
        Mode::Bare => "bare".to_string(),
    };
    let mut parts = vec![format!("mode: {label}")];
    if let Some(p) = env_file_loaded {
        parts.push(format!("env: {}", p.display()));
    } else {
        parts.push("env: (no .env found)".to_string());
    }
    if let Some(m) = manifest {
        parts.push(format!("manifest: {}", m.yaml_path.display()));
    }
    if let Some((nodes, edges)) = graph_state.schema() {
        parts.push(format!("graph: {nodes} nodes, {edges} edges"));
    }
    eprintln!("kglite-mcp-server: {}", parts.join("; "));
}
