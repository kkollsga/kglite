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
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use clap::Parser;
use kglite::api::io::OpenDisposition;
use mcp_methods::server::manifest::{
    find_sibling_manifest, find_workspace_manifest, ManifestError,
};
use mcp_methods::server::{
    init_tracing, load_env_for_mode, maybe_watch, resolve_source_roots, serve_prompts, watch,
    workspace, BundledSkill, Manifest, McpServer, PostActivateRevsHook, PredicateClause, ResultCtx,
    ServerOptions, SkillPredicateEvaluator, SkillRegistry, WorkspaceKind,
};
use rmcp::transport::stdio;
use rmcp::ServiceExt;

mod code_source;
mod csv_http;
mod cypher_tools;
mod explore;
mod selftest;
mod tools;
mod value_codecs;
use crate::tools::GraphState;
use kglite::api::storage::StorageMode;

#[derive(Parser, Debug)]
#[command(
    name = "kglite-mcp-server",
    about = "MCP server for KGLite knowledge graphs (Rust-native)"
)]
struct Cli {
    /// Path to a knowledge graph. An existing `.kgl` file or disk-graph
    /// directory is loaded at boot (mode auto-detected); a path that does
    /// not exist is an error unless `--storage` is given, in which case a
    /// fresh, empty graph is created (build-and-serve via the mutation tools,
    /// then `save_graph`).
    #[arg(long, conflicts_with_all = ["workspace", "watch", "source_root"])]
    graph: Option<PathBuf>,

    /// Create a fresh, empty `--graph` in this storage mode (`memory`,
    /// `mapped`, or `disk`) when its path does not exist — opt-in, so a typo'd
    /// path fails fast instead of silently serving an empty graph. Ignored
    /// when the graph already exists (its saved mode is auto-detected).
    #[arg(long)]
    storage: Option<String>,

    /// Source-root mode (no graph).
    #[arg(long = "source-root", conflicts_with_all = ["graph", "workspace", "watch"])]
    source_root: Option<PathBuf>,

    /// Workspace mode: clone GitHub repos and build code-tree graphs.
    #[arg(long, conflicts_with_all = ["graph", "source_root", "watch"])]
    workspace: Option<PathBuf>,

    /// Watch mode: rebuild the code-tree graph on file changes.
    #[arg(long, conflicts_with_all = ["graph", "source_root", "workspace"])]
    watch: Option<PathBuf>,

    /// Enable the write-mode "agent graph workbench" (single-graph mode):
    /// `cypher_query` accepts mutations (CREATE/SET/DELETE/MERGE, optionally
    /// `write_scope`-restricted) and the runtime graph-lifecycle tools
    /// (`load_graph` / `create_graph` / `save_graph_as`) are registered.
    /// Off by default — read-only is the safe default for analysis servers.
    #[arg(long)]
    writable: bool,

    /// Run a configuration self-test instead of serving: re-spawn this binary
    /// with the same flags, drive a live MCP handshake (initialize →
    /// tools/list → activate → cypher_query), and print green/red per
    /// capability (tools present, graph hydrates, github tools when a token is
    /// set). Exits non-zero if any check fails, so it doubles as a deployment
    /// smoke gate.
    #[arg(long)]
    selftest: bool,

    /// (`--selftest` only) Activate this directory for the handshake instead of
    /// building the whole `workspace.root`. For `workspace.kind: local` the root
    /// is a wide sandbox that agents narrow with `set_root_dir` and is never
    /// built as a unit, so `--selftest` does registration-only checks by
    /// default; pass a small representative subdir here to also verify a real
    /// code_tree build + `cypher_query` hydration.
    #[arg(long = "selftest-path")]
    selftest_path: Option<PathBuf>,

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
    kglite::api::code_tree::language_for_path(p).is_some()
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

/// Builds a graph embedder from the manifest's `extensions.embedder` config
/// (passed as a JSON string), on demand.
///
/// This is the seam that lets the **pip-hosted** server use *any* Python
/// embedding library (`extensions.embedder.library: sentence-transformers`,
/// `fastembed`, or a `factory:` escape) without the libpython-free library
/// knowing anything about Python: the kglite-py wrapper hands the config JSON
/// to a Python factory (`kglite._mcp_embed`) which picks the library, builds
/// the model, and wraps it in a `PyEmbedderAdapter` (GIL re-acquired only for
/// the embed call). The standalone cargo binary passes no factory, so a Python
/// library errors there with a clear message; it uses `library: fastembed-rs`
/// (the Rust `FastEmbedAdapter`) instead.
///
/// The argument is the whole `extensions.embedder` JSON object, so new fields
/// (library / model / factory / kwargs / …) flow through to Python without any
/// Rust change. `Send` because `run_with_embedder_factory` may move it into the
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
/// `extensions.embedder.library` Python path (`sentence-transformers` /
/// `fastembed`). The standalone binary calls [`run`] (no factory); the
/// kglite wheel passes a factory that builds the named Python embedder.
pub fn run_with_embedder_factory<I, T>(args: I, factory: Option<PyEmbedderFactory>) -> Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let argv: Vec<std::ffi::OsString> = args.into_iter().map(Into::into).collect();
    let cli = Cli::parse_from(argv.iter().cloned());
    // `--selftest` is a diagnostic mode, not a serve mode: it re-spawns this
    // binary with the operator's other flags and drives a real handshake. It
    // runs before the tokio runtime is built (the child owns the async serve
    // loop; the parent's RPC client is plain blocking I/O).
    if cli.selftest {
        return selftest::run_selftest(&cli, &argv);
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build tokio runtime")?;
    runtime.block_on(run_async(cli, factory))
}

/// Fail fast on bad mode-specific path arguments before any expensive setup.
fn validate_mode_paths(mode: &Mode, cli: &Cli) -> Result<()> {
    if let Mode::Graph { .. } = mode {
        // Validate --storage up front (only used when creating a new graph).
        if let Some(s) = &cli.storage {
            StorageMode::parse(s).map_err(|e| anyhow::anyhow!(e))?;
        }
        // Existence and open-vs-create are resolved exactly once by
        // `open_or_create_graph` in bind_mode; checking here would introduce a
        // second TOCTOU decision.
    }
    if let Mode::SourceRoot { dir } | Mode::Watch { dir } = mode {
        if !dir.is_dir() {
            anyhow::bail!(
                "path does not exist or is not a directory: {}",
                dir.display()
            );
        }
    }
    Ok(())
}

/// Manifest `workspace.kind: local` wins over CLI flags — promote `mode` to
/// `LocalWorkspace` so the rest of boot sees it. Mirrors the framework's own
/// `mcp-server` binary (`crates/mcp-server/src/main.rs` in 0.3.23+). Returns
/// `mode` unchanged when no local-workspace manifest is in play.
fn promote_local_workspace(mode: Mode, manifest: Option<&Manifest>) -> Result<Mode> {
    let Some(wcfg) = manifest.and_then(|m| m.workspace.as_ref()) else {
        return Ok(mode);
    };
    if wcfg.kind != WorkspaceKind::Local {
        return Ok(mode);
    }
    let m = manifest.expect("manifest present when wcfg is");
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
    Ok(Mode::LocalWorkspace {
        root: resolved,
        watch: wcfg.watch,
    })
}

/// The directory to start the `.env` walk-up from, per mode (the mode's own
/// directory for source-aware modes, cwd for bare).
fn resolve_env_start_dir(mode: &Mode) -> PathBuf {
    match mode {
        Mode::Graph { path } => path
            .canonicalize()
            .ok()
            .and_then(|p| p.parent().map(PathBuf::from))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))),
        Mode::SourceRoot { dir } | Mode::Workspace { dir } | Mode::Watch { dir } => dir.clone(),
        Mode::LocalWorkspace { root, .. } => root.clone(),
        Mode::Bare => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    }
}

/// Graph-aware steering footer for a builtin tool result — the content behind
/// the `with_result_postprocess` hook wired in [`run`]. Only fires against an
/// active code graph (Function/Class present); everything else returns `None`,
/// so the framework leaves the result untouched. Cheap: at most two
/// `has_node_type` read-locks (grep) or a substring test (cypher). The tool has
/// already released its lock by the time this runs, so there is no re-entrancy.
fn graph_result_footer(
    gs: &GraphState,
    tool: &str,
    args: &serde_json::Value,
    body: &str,
) -> Option<String> {
    match tool {
        "grep" => {
            if !(gs.has_node_type("Function") || gs.has_node_type("Class")) {
                return None;
            }
            // Zero-match: the framework returns "No matches for pattern '…'."
            if body.starts_with("No matches for pattern") {
                return Some(
                    "No grep matches — but the active code graph indexes the layout, so a \
                     wrong glob won't hide results there. Try `graph_overview()` then \
                     `cypher_query`."
                        .to_string(),
                );
            }
            // Definition-shaped pattern → a structural question grep answers poorly.
            let p = args
                .get("pattern")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim_start_matches(['^', '\\', '('])
                .trim_start();
            let definition_shaped = ["fn ", "def ", "class ", "impl ", "func ", "function "]
                .iter()
                .any(|kw| p.starts_with(kw));
            if definition_shaped {
                Some(
                    "Tip: that looks like a definition search. The active code graph resolves \
                     definitions and callers exactly — e.g. `cypher_query(\"MATCH (f:Function \
                     {title:'NAME'}) RETURN f.file_path, f.line_number\")`, and CALLS edges give \
                     callers. Reserve grep for literal text (log strings, comments, config keys)."
                        .to_string(),
                )
            } else {
                None
            }
        }
        "cypher_query" if body.contains("qualified_name") => Some(
            "Tip: `read_code_source(qualified_name=…)` pulls a matched symbol's source body."
                .to_string(),
        ),
        _ => None,
    }
}

/// Apply mode-specific bindings — source roots, workspace handle, initial
/// graph load/build — onto `options`, returning the transformed value.
fn bind_mode(
    mode: &Mode,
    cli: &Cli,
    manifest: Option<&Manifest>,
    graph_state: &GraphState,
    local_active_root: &Arc<RwLock<Option<PathBuf>>>,
    mut options: ServerOptions,
) -> Result<ServerOptions> {
    match mode {
        Mode::Graph { path } => {
            let create_mode = cli
                .storage
                .as_deref()
                .map(StorageMode::parse)
                .transpose()
                .map_err(|e| anyhow::anyhow!(e))?;
            let disposition = graph_state
                .open_or_create(path, create_mode)
                .context("kglite graph open/create failed")?;
            tracing::info!(
                path = %path.display(),
                disposition = match disposition {
                    OpenDisposition::Opened => "opened",
                    OpenDisposition::Created => "created",
                },
                "graph ready"
            );
            let base = if disposition == OpenDisposition::Opened {
                path.canonicalize().unwrap_or_else(|_| path.clone())
            } else {
                path.clone()
            };
            // P1 (operator feedback): honor the manifest's explicit
            // `source_root:` / `source_roots:` declaration in `--graph`
            // mode. The historical behaviour auto-bound the parent of
            // the `.kgl` file as the source root, which silently
            // overrode operators who declared a different root in
            // YAML (e.g. when the .kgl lives in a build dir but the
            // source files are elsewhere). Now: explicit YAML wins,
            // auto-bind only when the manifest doesn't declare one.
            let manifest_roots = manifest
                .filter(|m| !m.source_roots.is_empty())
                .map(resolve_source_roots)
                .transpose()
                .context("manifest source_root resolution failed")?;
            let roots = if let Some(rs) = manifest_roots {
                rs
            } else if let Some(parent) = base.parent() {
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
            // Revs-aware hook (mcp-methods 0.3.49 `with_post_activate_revs`):
            // fired instead of the plain hook when `repo_management(revs=…)` is
            // called, with the resolved revspecs (oldest→newest, HEAD last).
            // Builds ONE multi-rev graph via `build_code_tree_revs` (B.2b). The
            // plain hook above stays the HEAD-only default; only a `revs`
            // request reaches this path.
            let gs_revs = graph_state.clone();
            let revs_hook: PostActivateRevsHook = Arc::new(move |path, name, revs| {
                tracing::info!(repo = name, revs = ?revs, "code_tree::build_revs on activate");
                gs_revs.build_code_tree_revs(path, revs)
            });
            // Opening steer: append the graph mini-map to the activation
            // message. The post-activate hook above builds the graph first, so
            // counts are live when this runs. Chained before the workspace is
            // cloned into `options` (framework caveat). Correctness of the
            // single active-graph slot across A→B→A repo swaps is owned by
            // mcp-methods ≥0.3.47 (its skip gate tracks the currently-active
            // built root, so a re-bind of a different root always rebuilds).
            let gs_sum = graph_state.clone();
            let summary: workspace::ActivationSummaryHook =
                Arc::new(move |_path, _name| gs_sum.activation_summary());
            let ws = workspace::Workspace::open(canon, cli.stale_after_days, Some(hook))
                .context("workspace init failed")?
                .with_post_activate_revs(revs_hook)
                .with_activation_summary(summary);
            options = options.with_workspace(ws);
        }
        Mode::LocalWorkspace { root, .. } => {
            // mcp-methods `Workspace::open_local(root, hook)` STORES the
            // post-activate hook but does NOT fire it at open — verified
            // against mcp-methods 0.3.42 `src/server/workspace.rs`:
            // `open_local` (`:145`) just stores `post_activate`; the hook
            // fires only inside `activate()` (`:491-492`), which
            // `set_root_dir` calls on every swap. So every fire reaching this
            // closure is a real user activate — build the code-tree on each.
            //
            // History (do not re-add): an older mcp-methods contract fired
            // the hook synchronously inside `open` with the wide
            // `workspace.root` (~360k files), parsing everything before
            // returning and blowing past Claude Desktop's 60s `initialize`
            // window. We added an `initial_activate_seen` deferral to swallow
            // that one boot fire. mcp-methods has since removed the boot fire
            // (`open_local` no longer calls the hook), so the deferral was
            // instead swallowing the user's FIRST `set_root_dir` — leaving the
            // graph permanently unbuilt in local mode ("No active graph" on
            // every graph tool). Operator inbox 2026-06-23 (+ original
            // 2026-06-06 repro). Deferral removed; building eagerly is safe
            // because mcp-methods no longer fires at boot.
            //
            // Active root is captured in a shared RwLock so the watch
            // callback (later in this fn) can scope its rebuilds.
            let gs = graph_state.clone();
            let active_root_for_hook = local_active_root.clone();
            let hook: workspace::PostActivateHook = Arc::new(move |path, name| {
                // Poison-recovering lock policy (see tools::read_lock) — a
                // panicked holder must not silently stop root updates.
                *tools::write_lock(&active_root_for_hook) = Some(path.to_path_buf());
                tracing::info!(repo = name, "code_tree::build on local-workspace activate");
                // Surface a build failure instead of leaving the tools to
                // report a bare "No active graph" (operator ask, 2026-06-23).
                if let Err(e) = gs.build_code_tree(path) {
                    tracing::error!(
                        repo = name,
                        root = %path.display(),
                        error = %e,
                        "local-workspace code_tree build failed"
                    );
                    return Err(e);
                }
                Ok(())
            });
            // Revs-aware hook (mcp-methods 0.3.49): fired instead of the plain
            // hook when `set_root_dir(revs=…)` is called, with the resolved
            // revspecs (oldest→newest, HEAD last). Builds ONE multi-rev graph
            // via `build_code_tree_revs` (B.2b) and updates the shared active
            // root the same way the plain hook does, so the watch callback can
            // still scope its rebuilds. The plain hook stays the single-rev
            // default; only a `revs` request reaches this path.
            let gs_revs = graph_state.clone();
            let active_root_for_revs_hook = local_active_root.clone();
            let revs_hook: PostActivateRevsHook = Arc::new(move |path, name, revs| {
                *tools::write_lock(&active_root_for_revs_hook) = Some(path.to_path_buf());
                tracing::info!(
                    repo = name,
                    revs = ?revs,
                    "code_tree::build_revs on local-workspace activate"
                );
                if let Err(e) = gs_revs.build_code_tree_revs(path, revs) {
                    tracing::error!(
                        repo = name,
                        root = %path.display(),
                        error = %e,
                        "local-workspace multi-rev code_tree build failed"
                    );
                    return Err(e);
                }
                Ok(())
            });
            // Opening steer: mini-map on the activation message (see the
            // github-workspace arm above). Chained before the clone into
            // `options`. Correctness of the single active-graph slot across
            // A→B→A `set_root_dir` swaps is owned by mcp-methods ≥0.3.47: its
            // skip gate tracks the currently-active built root (not every root
            // ever hydrated), so re-binding a different root always re-fires
            // the build hook, and a same-root re-bind still cheap-skips.
            let gs_sum = graph_state.clone();
            let summary: workspace::ActivationSummaryHook =
                Arc::new(move |_path, _name| gs_sum.activation_summary());
            let ws = workspace::Workspace::open_local(root.clone(), Some(hook))
                .context("local-workspace init failed")?
                .with_post_activate_revs(revs_hook)
                .with_activation_summary(summary);
            options = options.with_workspace(ws);
        }
        Mode::Bare => {
            if let Some(m) = manifest {
                if !m.source_roots.is_empty() {
                    let resolved =
                        resolve_source_roots(m).context("source root resolution failed")?;
                    options = options.with_static_source_roots(resolved);
                }
            }
        }
    }
    Ok(options)
}

/// Client-side tool-discovery steer, folded into workspace-mode
/// `instructions` so every `--workspace` / `workspace.kind: local`
/// deployment emits it on `initialize` without copy-pasting it into each
/// manifest. It complements the 0.12.6 in-band steering (graph-over-grep
/// vocabulary in tool descriptions, the activation mini-map, the result
/// footer) by making the *"search the registry"* instruction explicit for
/// lazy-tool-discovery clients (Codex / code_mode / tool-search), which can
/// surface only `grep`/`read_source` on a broad first query and miss the
/// always-registered graph tools. Skipped when the manifest already carries
/// equivalent guidance (see the dedup check in `run_async`).
const DISCOVERY_STEER: &str = "Tool discovery: graph_overview and cypher_query are ALWAYS registered. \
If a broad first tool-search surfaces only grep/read_source, search your tool registry for 'cypher' or \
'graph_overview' and load those before falling back to grep — a discovery miss does not mean the graph \
path is unavailable.";

/// Fold [`DISCOVERY_STEER`] into `options.instructions` for the two
/// workspace modes. Appends (preserving any manifest `instructions:`) or
/// sets it when none exists; bails when the text already mentions the
/// always-registered graph tools so an opted-in manifest isn't duplicated.
fn apply_discovery_steer(mode: &Mode, mut options: ServerOptions) -> ServerOptions {
    if !matches!(mode, Mode::Workspace { .. } | Mode::LocalWorkspace { .. }) {
        return options;
    }
    let already = options
        .instructions
        .as_deref()
        .is_some_and(|s| s.to_lowercase().contains("always registered"));
    if already {
        return options;
    }
    options.instructions = Some(match options.instructions.take() {
        Some(existing) if !existing.trim().is_empty() => format!("{existing}\n\n{DISCOVERY_STEER}"),
        _ => DISCOVERY_STEER.to_string(),
    });
    options
}

async fn run_async(cli: Cli, py_embedder_factory: Option<PyEmbedderFactory>) -> Result<()> {
    init_tracing();
    let mode = pick_mode(&cli);
    validate_mode_paths(&mode, &cli)?;

    let manifest = load_manifest(&cli, &mode).context("manifest load failed")?;

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

    // The github-workspace (open-source) mode ingests each cloned repo's
    // markdown as `:Doc` nodes and links them to code (MENTIONS/DOCUMENTS) —
    // a repo's prose is part of its intelligence. Local / file / graph modes
    // keep the lean code-only graph.
    let include_docs = matches!(mode, Mode::Workspace { .. });
    // extensions.value_codecs: build the manifest-declared, position-scoped
    // literal codecs (prefix / map / regex). Passed to the engine via
    // ExecuteOptions per query — decode on the way in, encode on the way out.
    // Empty when absent; a malformed block errors at boot, not per-query.
    let value_codecs = match manifest.as_ref() {
        Some(m) => value_codecs::from_manifest(m.extensions.get("value_codecs"))
            .context("extensions.value_codecs parse failed")?,
        None => Vec::new(),
    };
    let value_codecs = if value_codecs.is_empty() {
        None
    } else {
        Some(Arc::new(value_codecs))
    };
    let graph_state = GraphState::new(include_docs).with_value_codecs(value_codecs);

    // Shared "active root" slot for local-workspace mode. Populated
    // by the post-activate hook on each `set_root_dir`; read by the
    // watch callback to scope rebuilds. Stays `None` until the first
    // `set_root_dir` (the boot-time activate is intentionally
    // deferred — see `Mode::LocalWorkspace` arm below for why).
    // Other modes never write to it.
    let local_active_root: Arc<RwLock<Option<PathBuf>>> = Arc::new(RwLock::new(None));

    // Mode-specific bindings: source roots, workspace handle, initial graph
    // build. Extracted to `bind_mode` so this boot fn reads as a sequence of
    // named phases.
    let options = bind_mode(
        &mode,
        &cli,
        manifest.as_ref(),
        &graph_state,
        &local_active_root,
        options,
    )?;

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
            .unwrap_or(false)
            // Write mode implies save_graph (an agent that mutates needs to
            // persist) so `--writable` alone gives the full workbench.
            || cli.writable,
        writable: cli.writable,
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
    if matches!(mode, Mode::LocalWorkspace { .. }) {
        // Local workspaces activate a directory with `set_root_dir`; the
        // GitHub clone-oriented `repo_management` tool is mutually exclusive
        // and would steer agents toward the wrong activation protocol.
        server.tool_router_mut().remove_route("repo_management");
    }
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
                let active = tools::read_lock(&active_root_for_watch).clone();
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

/// Read `manifest.extensions.embedder.{library, model, …}` and build the
/// corresponding [`kglite::api::Embedder`]. Returns `Ok(None)` when no
/// `embedder:` is declared, `Err` on validation failures.
///
/// The `library` field names the embedding engine; the host (Rust vs Python)
/// is inferred from it:
/// - `fastembed-rs` — the Rust-native fastembed-rs adapter (cargo
///   `--features fastembed`; the only option on the standalone binary).
/// - any other value, or a `factory:` escape — a Python embedding library
///   (`fastembed`, `sentence-transformers`, …) built by `py_embedder_factory`
///   (supplied only by the pip-hosted server). The whole config object is
///   handed to Python as JSON, so the library set + its options live entirely
///   on the Python side (`kglite._mcp_embed`) — adding a library never touches
///   this function.
fn build_embedder_from_manifest(
    manifest: &Manifest,
    py_embedder_factory: Option<&PyEmbedderFactory>,
) -> Result<Option<Arc<dyn kglite::api::Embedder>>> {
    let Some(raw) = manifest.extensions.get("embedder") else {
        return Ok(None);
    };
    if !manifest.trust.allow_embedder {
        anyhow::bail!(
            "extensions.embedder is disabled unless the manifest explicitly sets \
             trust.allow_embedder: true"
        );
    }
    let obj = raw
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("extensions.embedder must be a mapping (got: {raw:?})"))?;
    // `fastembed-rs` is the only Rust-hosted engine; everything else (and any
    // `factory:`) is a Python library hosted by the wheel. Default to a Python
    // library so the common pip case needs only `library: + model:`.
    let library = obj.get("library").and_then(|v| v.as_str());
    let is_rust = library == Some("fastembed-rs");

    if is_rust {
        let model = obj.get("model").and_then(|v| v.as_str()).ok_or_else(|| {
            anyhow::anyhow!("extensions.embedder.model is required for library: fastembed-rs")
        })?;
        return build_rust_embedder(model);
    }

    // Python-hosted: hand the whole config object to the Python factory, which
    // picks the library, builds the model, and wraps it. The cargo binary
    // supplies no factory.
    let factory = py_embedder_factory.ok_or_else(|| {
        let lib = library.unwrap_or("<a Python library>");
        anyhow::anyhow!(
            "extensions.embedder.library = {lib:?} is a Python embedding library, but the \
             standalone `cargo install kglite-mcp-server` binary has no Python interpreter to \
             host it. Either run the server from the kglite wheel (`pip install kglite`, then \
             `pip install {lib}`), or use `library: fastembed-rs` with `cargo install \
             kglite-mcp-server --features fastembed`."
        )
    })?;
    let config_json = serde_json::to_string(raw)
        .map_err(|e| anyhow::anyhow!("serializing extensions.embedder failed: {e}"))?;
    let embedder = factory(&config_json)
        .map_err(|e| anyhow::anyhow!("python embedder construction failed: {e}"))?;
    tracing::info!(library = ?library, "registered python embedder");
    Ok(Some(embedder))
}

/// Build the Rust-native fastembed-rs embedder (`library: fastembed-rs`).
/// Gated on the `fastembed` cargo feature; the default build errors with a
/// rebuild hint (the feature is off by default because ort-sys has a flaky
/// upstream binary download).
#[cfg(feature = "fastembed")]
fn build_rust_embedder(model: &str) -> Result<Option<Arc<dyn kglite::api::Embedder>>> {
    let adapter = kglite::api::FastEmbedAdapter::new(model)
        .map_err(|e| anyhow::anyhow!("fastembed-rs init failed: {e}"))?;
    tracing::info!(model, "registered Rust-native (fastembed-rs) embedder");
    Ok(Some(Arc::new(adapter)))
}

#[cfg(not(feature = "fastembed"))]
fn build_rust_embedder(_model: &str) -> Result<Option<Arc<dyn kglite::api::Embedder>>> {
    anyhow::bail!(
        "extensions.embedder.library = \"fastembed-rs\" requires this binary to be built with \
         the `fastembed` feature: `cargo install kglite-mcp-server --features fastembed`. The \
         default build excludes it because its ort-sys dependency has a flaky upstream binary \
         download. (If you are running the pip wheel, use a Python library instead — e.g. \
         `library: sentence-transformers` with `pip install sentence-transformers`.)"
    )
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

#[cfg(test)]
mod discovery_steer_tests {
    use super::*;
    use std::path::PathBuf;

    fn ws_mode() -> Mode {
        Mode::LocalWorkspace {
            root: PathBuf::from("/tmp/ws"),
            watch: false,
        }
    }

    #[test]
    fn appends_to_workspace_modes() {
        let out = apply_discovery_steer(&ws_mode(), ServerOptions::default());
        let text = out.instructions.expect("instructions set");
        assert!(text.contains("ALWAYS registered"));
        assert!(text.contains("cypher"));
    }

    #[test]
    fn preserves_manifest_instructions() {
        let opts = ServerOptions {
            instructions: Some("Domain guidance here.".to_string()),
            ..Default::default()
        };
        let out = apply_discovery_steer(&ws_mode(), opts);
        let text = out.instructions.expect("instructions set");
        assert!(text.starts_with("Domain guidance here."));
        assert!(text.contains("ALWAYS registered"));
    }

    #[test]
    fn dedupes_when_already_present() {
        let opts = ServerOptions {
            instructions: Some(
                "graph_overview and cypher_query are ALWAYS registered.".to_string(),
            ),
            ..Default::default()
        };
        let out = apply_discovery_steer(&ws_mode(), opts);
        let text = out.instructions.expect("instructions set");
        // Only the manifest's own copy — not appended a second time.
        assert_eq!(text.matches("ALWAYS registered").count(), 1);
    }

    #[test]
    fn skips_non_workspace_modes() {
        let mode = Mode::Graph {
            path: PathBuf::from("/tmp/g.kgl"),
        };
        let out = apply_discovery_steer(&mode, ServerOptions::default());
        assert!(out.instructions.is_none());
    }
}

#[cfg(test)]
mod embedder_trust_tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn load_embedder_manifest(allow_embedder: bool) -> (tempfile::TempDir, Manifest) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("mcp.yaml");
        fs::write(
            &path,
            format!(
                "name: trust-test\ntrust:\n  allow_embedder: {allow_embedder}\n\
                 extensions:\n  embedder:\n    library: test\n    model: test\n"
            ),
        )
        .expect("write manifest");
        let manifest = mcp_methods::server::load_manifest(&path).expect("load manifest");
        (dir, manifest)
    }

    #[test]
    fn untrusted_embedder_never_invokes_factory() {
        let (_dir, manifest) = load_embedder_manifest(false);
        let called = Arc::new(AtomicBool::new(false));
        let called_by_factory = called.clone();
        let factory: PyEmbedderFactory = Box::new(move |_| {
            called_by_factory.store(true, Ordering::SeqCst);
            Err("factory sentinel".to_string())
        });

        let error = match build_embedder_from_manifest(&manifest, Some(&factory)) {
            Ok(_) => panic!("untrusted embedder must be rejected"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("trust.allow_embedder: true"));
        assert!(!called.load(Ordering::SeqCst));
    }

    #[test]
    fn trusted_embedder_reaches_factory() {
        let (_dir, manifest) = load_embedder_manifest(true);
        let called = Arc::new(AtomicBool::new(false));
        let called_by_factory = called.clone();
        let factory: PyEmbedderFactory = Box::new(move |_| {
            called_by_factory.store(true, Ordering::SeqCst);
            Err("factory sentinel".to_string())
        });

        let error = match build_embedder_from_manifest(&manifest, Some(&factory)) {
            Ok(_) => panic!("sentinel factory must fail"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("factory sentinel"));
        assert!(called.load(Ordering::SeqCst));
    }
}
