//! `kglite-mcp-server` — single MCP server for KGLite knowledge graphs.
//!
//! Layers three kglite-specific tools on top of the generic
//! `mcp-server` framework: `cypher_query`, `graph_overview`, and
//! `save_graph`. All three close over a [`GraphState`] holding the
//! active `KnowledgeGraph` Python object.
//!
//! ## Two frontends, one library
//!
//! The server body lives in [`run`] (sync, builds its own tokio
//! runtime) so it can be driven from two places without duplication:
//!
//! - the thin `src/main.rs` binary (`cargo install kglite-mcp-server`),
//!   which is libpython-free; and
//! - the `kglite` Python wheel, whose PyO3 wrapper calls [`run`]
//!   inside `py.detach(...)` (GIL released) so `pip install kglite` ships
//!   the exact same server with no separate wheel and no duplicated engine.
//!
//! The library never links libpython — it depends only on the pure-Rust
//! `kglite` core. The wheel's `.so` and the standalone binary share this
//! one engine build.
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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

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
mod preprocessor;
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

/// Whether a changed path should tag a code_tree rebuild: any file a parser
/// handles, plus `.md` files when the docs pass is enabled (so editing a
/// README re-links it to the code).
fn is_graph_relevant(p: &std::path::Path, include_docs: bool) -> bool {
    kglite::api::language_for_path(p).is_some()
        || (include_docs
            && p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("md")))
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

/// Builds a graph embedder from a model name, on demand.
///
/// This is the seam that lets the **pip-hosted** server use a Python
/// embedder (`extensions.embedder.backend: python`) without the
/// libpython-free library knowing anything about Python: the kglite-py
/// wrapper supplies a factory that wraps a fastembed-py model in a
/// `PyEmbedderAdapter` (which re-acquires the GIL only for the embed
/// call). The standalone cargo binary passes no factory, so
/// `backend: python` errors there with a clear message; it uses
/// `backend: fastembed` (the Rust `FastEmbedAdapter`) instead.
///
/// `Send` because `run_with_embedder_factory` may move it into the
/// tokio runtime's future.
pub type PyEmbedderFactory =
    Box<dyn Fn(&str) -> Result<Arc<dyn kglite::api::Embedder>, String> + Send>;

/// Run the MCP server to completion over stdio.
///
/// `args` is a full argv vector (program name in `args[0]`, as clap
/// expects). The binary passes `std::env::args()`; the PyO3 wrapper
/// passes a synthesised `["kglite-mcp-server", ...sys.argv[1:]]`.
///
/// Synchronous by design (see CLAUDE.md "core is sync, bindings own
/// async"): this builds its own multi-thread tokio runtime and blocks
/// on the stdio serve loop. The PyO3 wrapper calls it inside
/// `py.detach(...)`, so the GIL is released for the server's entire
/// lifetime and the Python process simply *becomes* the server.
pub fn run<I, T>(args: I) -> Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    run_with_embedder_factory(args, None)
}

/// Like [`run`], but with an optional Python-embedder factory for the
/// `extensions.embedder.backend: python` path. The standalone binary
/// calls [`run`] (no factory); the kglite wheel passes a factory that
/// builds a fastembed-py embedder.
pub fn run_with_embedder_factory<I, T>(args: I, factory: Option<PyEmbedderFactory>) -> Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let cli = Cli::parse_from(args);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build tokio runtime")?;
    runtime.block_on(run_async(cli, factory))
}

async fn run_async(cli: Cli, py_embedder_factory: Option<PyEmbedderFactory>) -> Result<()> {
    init_tracing();
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

    // The github-workspace (open-source) mode ingests each cloned repo's
    // markdown as `:Doc` nodes and links them to code (MENTIONS/DOCUMENTS) —
    // a repo's prose is part of its intelligence. Local / file / graph modes
    // keep the lean code-only graph.
    let include_docs = matches!(mode, Mode::Workspace { .. });
    // extensions.cypher_preprocessor: build the manifest-declared query
    // rewriter (regex rules and/or a subprocess command), trust-gated. None
    // when absent. Errors here surface at boot rather than per-query.
    let preprocessor = match manifest.as_ref() {
        Some(m) => {
            let base_dir = m
                .yaml_path
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."));
            preprocessor::Preprocessor::from_manifest(
                m.extensions.get("cypher_preprocessor"),
                m.trust.allow_query_preprocessor,
                base_dir,
            )
            .context("extensions.cypher_preprocessor parse failed")?
            .map(Arc::new)
        }
        None => None,
    };
    let graph_state = GraphState::new(include_docs).with_preprocessor(preprocessor);

    // Shared "active root" slot for local-workspace mode. Populated
    // by the post-activate hook on each `set_root_dir`; read by the
    // watch callback to scope rebuilds. Stays `None` until the first
    // `set_root_dir` (the boot-time activate is intentionally
    // deferred — see `Mode::LocalWorkspace` arm below for why).
    // Other modes never write to it.
    let local_active_root: Arc<RwLock<Option<PathBuf>>> = Arc::new(RwLock::new(None));

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
            // Operator inbox 2026-05-25: with a wide `workspace.root`
            // (the documented "lateral swap" pattern,
            // e.g. `/Volumes/EksternalHome/Koding` ≈ 360k files), the
            // mcp-methods `Workspace::open_local(root, hook)` fires
            // `hook(workspace.root, ...)` synchronously inside the
            // open. The hook here calls `gs.build_code_tree(root)`
            // which parses every source file under the wide root
            // before returning — blowing past Claude Desktop's 60s
            // MCP `initialize` window. The user sees only
            // "Could not attach to MCP server."
            //
            // Fix: defer the very first hook invocation (the boot-
            // time activate). The first `set_root_dir(path)` becomes
            // the first real activate; we build the code-tree for
            // exactly the one repo the operator picked. Boot returns
            // in ms. mcp-methods unchanged.
            //
            // Active root is also captured in a shared RwLock so the
            // watch callback (later in this fn) can scope its
            // rebuilds.
            let gs = graph_state.clone();
            let initial_activate_seen = Arc::new(AtomicBool::new(false));
            let initial_flag = initial_activate_seen.clone();
            let active_root_for_hook = local_active_root.clone();
            let hook: workspace::PostActivateHook = Arc::new(move |path, name| {
                if !initial_flag.swap(true, Ordering::SeqCst) {
                    tracing::info!(
                        root = %path.display(),
                        "deferring local-workspace code_tree build until first set_root_dir"
                    );
                    return Ok(());
                }
                if let Ok(mut guard) = active_root_for_hook.write() {
                    *guard = Some(path.to_path_buf());
                }
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

    // `extensions.embedder:` in the manifest selects the embedding
    // backend for `text_score()`:
    //   - `backend: fastembed` — the Rust-native fastembed-rs adapter
    //     (cargo `--features fastembed`). The only option for the
    //     standalone libpython-free binary.
    //   - `backend: python` — a fastembed-py model, built by the wheel's
    //     `_run_mcp_server` factory and wrapped in a `PyEmbedderAdapter`.
    //     Only available when a factory is supplied (the pip-hosted
    //     server); the cargo binary rejects it with a clear message.
    if let Some(m) = manifest.as_ref() {
        if let Some(embedder) = build_embedder_from_manifest(m, py_embedder_factory.as_ref())? {
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
            let cb: watch::ChangeHandler = Arc::new(move |paths| {
                // Skip when no changed path is a file code_tree parses
                // — a `cargo build` / `npm install` storm of `.rlib` /
                // `node_modules/` events would otherwise needlessly
                // tag a rebuild. Cheap predicate (just an ext lookup).
                let any_code = paths
                    .iter()
                    .any(|p| is_graph_relevant(p, gs.include_docs()));
                if !any_code {
                    return;
                }
                // Tag for rebuild — the actual work happens on the
                // next MCP tool call via ensure_code_tree_fresh.
                // Cheap: no rebuild on the watcher thread, ms-scale.
                gs.tag_code_tree_dirty(canon.clone());
            });
            maybe_watch(Some(dir), Some(cb))?
        }
        Mode::LocalWorkspace { root, watch: true } => {
            // Hand mcp-methods the wide `workspace.root` to monitor —
            // FSEvents/inotify only emit events for files inside the
            // subtree, so watching wide is cheap. Filtering happens
            // in the callback.
            //
            // Operator inbox 2026-05-25: pre-fix this captured
            // `workspace.root` and rebuilt the entire wide tree on
            // every event (build storm on any `cargo build` /
            // editor save anywhere in the sandbox). Fix: read the
            // shared `local_active_root` (populated by the
            // post-activate hook above on each `set_root_dir`),
            // skip when nothing changed under the active root,
            // rebuild against the active root only.
            let gs = graph_state.clone();
            let active_root_for_watch = local_active_root.clone();
            let cb: watch::ChangeHandler = Arc::new(move |paths| {
                let active = match active_root_for_watch.read() {
                    Ok(g) => g.clone(),
                    Err(_) => return,
                };
                let Some(active) = active else {
                    // No `set_root_dir` yet; nothing to rebuild.
                    return;
                };
                // Two-stage filter: (1) inside the active root,
                // (2) is a code file `code_tree` would parse. The
                // second filter skips `cargo build` /
                // `node_modules/` storms that otherwise tag a
                // rebuild for changes the parser doesn't see.
                let any_under_active_and_code = paths
                    .iter()
                    .any(|p| p.starts_with(&active) && is_graph_relevant(p, gs.include_docs()));
                if !any_under_active_and_code {
                    return;
                }
                // Tag for rebuild; the actual rebuild fires on the
                // next MCP tool call (ensure_code_tree_fresh).
                gs.tag_code_tree_dirty(active.clone());
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
        // Skill `.md` bodies live at `crates/kglite-mcp-server/skills/` — the
        // single canonical home since 0.10.25, when the Python MCP server (and
        // its duplicate `kglite/mcp_server/skills/`) was retired and this Rust
        // binary became the one MCP server. `cargo publish` only packages
        // files inside the crate dir, so they must live here (not behind a
        // `../../../kglite/...` `include_str!` path).
        let registry_result = SkillRegistry::new()
            .add_bundled(BundledSkill {
                name: "cypher_query",
                body: include_str!("../skills/cypher_query.md"),
            })
            .add_bundled(BundledSkill {
                name: "graph_overview",
                body: include_str!("../skills/graph_overview.md"),
            })
            .add_bundled(BundledSkill {
                name: "save_graph",
                body: include_str!("../skills/save_graph.md"),
            })
            .add_bundled(BundledSkill {
                name: "read_code_source",
                body: include_str!("../skills/read_code_source.md"),
            })
            .add_bundled(BundledSkill {
                name: "explore",
                body: include_str!("../skills/explore.md"),
            })
            // Cross-tool skills: named after no tool, they attach via
            // `references_tools` and lead with the `description` routing —
            // both rely on the serve_prompts injection added in mcp-methods
            // 0.3.42 (## When to use + references_tools), so they only became
            // active with that pin bump.
            .add_bundled(BundledSkill {
                name: "code_graph_analysis",
                body: include_str!("../skills/code_graph_analysis.md"),
            })
            .add_bundled(BundledSkill {
                name: "code_graph_views",
                body: include_str!("../skills/code_graph_views.md"),
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

/// Read `manifest.extensions.embedder.{backend, model, …}` and build the
/// corresponding [`kglite::api::Embedder`]. Returns `Ok(None)` when no
/// `embedder:` is declared, `Err` on validation failures (unknown
/// backend, missing fields, `python` without a factory).
///
/// Two backends:
/// - `fastembed` — the Rust-native fastembed-rs adapter (cargo
///   `--features fastembed`). The only option for the standalone binary.
/// - `python` — a fastembed-py model built by `py_embedder_factory`
///   (supplied only by the pip-hosted server). Lets `pip install
///   kglite[embed]` power `text_score()` with no Rust toolchain and no
///   `ort-sys` download.
fn build_embedder_from_manifest(
    manifest: &Manifest,
    py_embedder_factory: Option<&PyEmbedderFactory>,
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
        // The pip-hosted server's path: build a fastembed-py model via the
        // wrapper-supplied factory and wrap it in a PyEmbedderAdapter. The
        // factory acquires the GIL only for the (per-query) embed call.
        "python" => {
            let model = obj.get("model").and_then(|v| v.as_str()).ok_or_else(|| {
                anyhow::anyhow!("extensions.embedder.model is required for the python backend")
            })?;
            let factory = py_embedder_factory.ok_or_else(|| {
                anyhow::anyhow!(
                    "extensions.embedder.backend = \"python\" requires the pip-hosted \
                     server (the kglite wheel). The standalone `cargo install \
                     kglite-mcp-server` binary has no Python interpreter, so it cannot \
                     host a fastembed-py model. Either run the server from the wheel \
                     (`pip install 'kglite[embed]'`, then `kglite-mcp-server …`), or use \
                     `backend: fastembed` with `cargo install kglite-mcp-server \
                     --features fastembed`."
                )
            })?;
            let embedder = factory(model)
                .map_err(|e| anyhow::anyhow!("python embedder construction failed: {e}"))?;
            tracing::info!(model, backend, "registered python (fastembed-py) embedder");
            Ok(Some(embedder))
        }
        #[cfg(feature = "fastembed")]
        "fastembed" => {
            let model = obj.get("model").and_then(|v| v.as_str()).ok_or_else(|| {
                anyhow::anyhow!("extensions.embedder.model is required for the fastembed backend")
            })?;
            let adapter = kglite::api::FastEmbedAdapter::new(model)
                .map_err(|e| anyhow::anyhow!("fastembed init failed: {e}"))?;
            tracing::info!(model, backend, "registered Rust-native embedder");
            Ok(Some(Arc::new(adapter)))
        }
        #[cfg(not(feature = "fastembed"))]
        "fastembed" => anyhow::bail!(
            "extensions.embedder.backend = \"fastembed\" requires this binary \
             to be built with the `fastembed` feature enabled. Rebuild with: \
             `cargo install kglite-mcp-server --features fastembed`. The \
             default build excludes fastembed because its ort-sys dependency \
             has a flaky upstream binary download — opt in only when you need \
             text_score() semantic search. (If you are running the pip wheel, \
             use `backend: python` instead, backed by `pip install \
             'kglite[embed]'`.)"
        ),
        other => anyhow::bail!(
            "extensions.embedder.backend = {other:?} is not supported. Known: \
             `python` (pip wheel, fastembed-py) and `fastembed` (cargo \
             `--features fastembed`, fastembed-rs)."
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
