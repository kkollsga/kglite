//! Command-line surface and the server mode it resolves to, plus the
//! mode-derived path/manifest/env decisions boot makes before wiring.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use kglite::api::storage::StorageMode;
use kglite::okf::CachePolicy;
use mcp_methods::server::manifest::{
    find_sibling_manifest, find_workspace_manifest, ManifestError,
};
use mcp_methods::server::{Manifest, WorkspaceKind};

use crate::*;

/// Resolve a path written in a manifest against the addressed manifest's
/// directory. `PathBuf::join` preserves absolute inputs by replacing the base.
pub(crate) fn manifest_relative_path(
    manifest: &Manifest,
    raw: impl AsRef<std::path::Path>,
) -> PathBuf {
    manifest
        .yaml_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join(raw)
}

#[derive(Parser, Debug)]
#[command(
    name = "kglite-mcp-server",
    version,
    about = "MCP server for KGLite knowledge graphs (Rust-native)"
)]
pub(crate) struct Cli {
    /// Path to a knowledge graph. An existing `.kgl` file or disk-graph
    /// directory is loaded at boot (mode auto-detected); a path that does
    /// not exist is an error unless `--storage` is given, in which case a
    /// fresh, empty graph is created (build-and-serve via the mutation tools,
    /// then `save_graph`).
    #[arg(long, conflicts_with_all = ["workspace", "watch", "source_root", "vault"])]
    pub(crate) graph: Option<PathBuf>,

    /// Storage mode (`memory`, `mapped`, or `disk`), applied whether or not
    /// `--graph` exists. A missing path is created in this mode — opt-in, so a
    /// typo'd path fails fast instead of silently serving an empty graph — and
    /// an existing graph saved in a different mode is *converted* to it
    /// (memory ⇄ mapped). A disk graph is a directory rather than a file, so
    /// converting into or out of disk mode has no in-place form and is refused
    /// at boot. Omit the flag to serve whatever mode the graph recorded.
    #[arg(long)]
    pub(crate) storage: Option<String>,

    /// Source-root mode (no graph).
    #[arg(long = "source-root", conflicts_with_all = ["graph", "workspace", "watch", "vault"])]
    pub(crate) source_root: Option<PathBuf>,

    /// Workspace mode: clone GitHub repos and build workspace graphs.
    #[arg(long, conflicts_with_all = ["graph", "source_root", "watch", "vault"])]
    pub(crate) workspace: Option<PathBuf>,

    /// Watch mode: rebuild the workspace graph on file changes.
    #[arg(long, conflicts_with_all = ["graph", "source_root", "workspace", "vault"])]
    pub(crate) watch: Option<PathBuf>,

    /// Vault mode: serve a directory of frontmatter-markdown notes (the
    /// format `VAULT.md` specifies) as a knowledge graph, rebuilt from the
    /// files whenever they change. Unlike `--watch`, the producer is built
    /// in — no embedding binary is required. `.kglite/vault.yaml` configures
    /// the build; `.kglite/skills/` and `.kglite/recipes/` are served as the
    /// graph's own agent guidance.
    #[arg(long, conflicts_with_all = ["graph", "source_root", "workspace", "watch"])]
    pub(crate) vault: Option<PathBuf>,

    /// Where vault mode keeps the graph it builds, so a restart serves the
    /// vault without rebuilding it. Defaults to `<VAULT>/.kglite/graph.kgl`,
    /// which travels with the vault — ship a pre-built vault and the first
    /// boot on a new machine costs a `stat` pass instead of a build. Give a
    /// path to keep it elsewhere (a read-only vault, or a cache the vault's
    /// own git history should not carry), or `none` to switch it off and
    /// rebuild at every boot. A cache that cannot be read or written never
    /// fails the boot: the server rebuilds and says so.
    #[arg(long = "vault-cache", value_name = "PATH|none", requires = "vault")]
    pub(crate) vault_cache: Option<String>,

    /// Enable the write-mode "agent graph workbench" (single-graph mode):
    /// `cypher_query` accepts mutations (optionally `write_scope`-restricted)
    /// and the runtime graph-lifecycle tools
    /// (`load_graph` / `create_graph` / `save_graph_as`) are registered.
    /// Off by default — read-only is the safe default for analysis servers.
    /// Equivalent to `extensions.writable: true` in the manifest; either
    /// surface alone enables writes (`builtins.save_graph: true` does not —
    /// it registers `save_graph` and nothing else).
    #[arg(long)]
    pub(crate) writable: bool,

    /// Operator-pinned write scope: a comma-separated node-type list
    /// (`--write-scope Plan,Task`) the agent's writes may never widen. When
    /// set, an agent that omits `write_scope` gets *this* scope rather than
    /// unrestricted writes, and an agent that supplies one gets the
    /// **intersection** of the two. An empty intersection — or an explicitly
    /// empty pin — refuses every mutation. Combines with
    /// `extensions.write_scope:` in the manifest by intersection. Only
    /// meaningful on a write-enabled server (`--writable` /
    /// `extensions.writable: true`); a read-only one refuses every mutation
    /// already.
    #[arg(long = "write-scope")]
    pub(crate) write_scope: Option<String>,

    /// Operator-facing name this server publishes while it holds the graph's
    /// writer lease, so a peer refused a write is told *which* client is
    /// mid-write ("Claude Desktop") instead of a bare pid. Falls back to
    /// `KGLITE_LEASE_LABEL`, then to the parent process's name — which is
    /// usually the MCP client that spawned this server, and is why four
    /// clients sharing one manifest still name themselves apart.
    #[arg(long = "lease-label")]
    pub(crate) lease_label: Option<String>,

    /// Let this server's Cypher *reads* use the engine's parallel runtime.
    /// Off by default: a server's cores belong to its clients, so nothing
    /// turns this on by omission. It is a permission, not an instruction —
    /// operators that cannot partition deterministically, and queries below
    /// the engine's runtime size gate, still run sequentially. Mutations are
    /// deliberately excluded, so a `--writable` server keeps writing
    /// sequentially. Equivalent to `extensions.parallel: true` in the
    /// manifest (either surface alone turns it on). Pool width is
    /// `available_parallelism`, overridable with `KGLITE_QUERY_THREADS`.
    #[arg(long)]
    pub(crate) parallel: bool,

    /// The valid-time instant a statement, recipe tool or fluent step reads
    /// when it names none, on a graph with validity declarations: `today`
    /// (the built-in default, UTC), `all` (every version) or a fixed
    /// `YYYY-MM-DD` day. Overrides `extensions.valid_time.default` in the
    /// manifest. Runtime only — never written into a `.kgl` file.
    #[arg(long = "valid-time-default", value_name = "today|all|YYYY-MM-DD",
          value_parser = |text: &str| kglite::api::temporal::ValidTimeDefault::parse(text))]
    pub(crate) valid_time_default: Option<kglite::api::temporal::ValidTimeDefault>,

    /// Run a configuration self-test instead of serving: re-spawn this binary
    /// with the same flags, drive a live MCP handshake (initialize →
    /// tools/list → activate → cypher_query), and print green/red per
    /// capability (tools present, graph hydrates, github tools when the
    /// manifest opts in and a token is set). Exits non-zero if any check
    /// fails, so it doubles as a deployment smoke gate.
    #[arg(long)]
    pub(crate) selftest: bool,

    /// (`--selftest` only) Activate this directory for the handshake instead of
    /// building the whole `workspace.root`. For `workspace.kind: local` the root
    /// is a wide sandbox that agents narrow with `set_root_dir` and is never
    /// built as a unit, so `--selftest` does registration-only checks by
    /// default; pass a small representative subdir here to also verify a real
    /// workspace-graph build + `cypher_query` hydration.
    #[arg(long = "selftest-path")]
    pub(crate) selftest_path: Option<PathBuf>,

    #[arg(long = "mcp-config")]
    pub(crate) mcp_config: Option<PathBuf>,
    #[arg(long)]
    pub(crate) name: Option<String>,
    #[arg(long = "stale-after-days", default_value_t = 7)]
    pub(crate) stale_after_days: u32,
}

#[derive(Debug, Clone)]
pub(crate) enum Mode {
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
    /// `--vault DIR`. Structurally identical to [`Mode::Watch`] — eager boot
    /// build, static source root, watcher over the root, dirty-tag plus lazy
    /// rebuild — and deliberately aliased onto its arm everywhere, because a
    /// parallel lifecycle is exactly what would drift. What the separate
    /// variant buys is the three things that are *not* the same: the boot
    /// banner, the first-party okf producer boot installs for it (`--watch`
    /// has none), and the `rebuild_graph` route.
    Vault {
        dir: PathBuf,
    },
    Bare,
}

pub(crate) fn pick_mode(cli: &Cli) -> Mode {
    if let Some(p) = &cli.graph {
        Mode::Graph { path: p.clone() }
    } else if let Some(d) = &cli.source_root {
        Mode::SourceRoot { dir: d.clone() }
    } else if let Some(d) = &cli.workspace {
        Mode::Workspace { dir: d.clone() }
    } else if let Some(d) = &cli.watch {
        Mode::Watch { dir: d.clone() }
    } else if let Some(d) = &cli.vault {
        Mode::Vault { dir: d.clone() }
    } else {
        Mode::Bare
    }
}

/// The spelling `--vault-cache none` reserves, so a vault cannot be cached
/// into a file of that name by accident.
const CACHE_OFF: &str = "none";

/// Resolve `--vault-cache` into the policy the vault producer opens with.
///
/// An omitted flag is the in-vault default; `none` switches the cache off;
/// anything else is a path. Relative paths resolve against the working
/// directory the server was started in, like every other path argument —
/// an MCP client's working directory is its own, so an operator writing one
/// into a manifest should write it absolute.
pub(crate) fn vault_cache_policy(cli: &Cli) -> CachePolicy {
    match cli.vault_cache.as_deref() {
        None => CachePolicy::Default,
        Some(CACHE_OFF) => CachePolicy::Disabled,
        Some(path) => CachePolicy::At(PathBuf::from(path)),
    }
}

pub(crate) fn fallback_name(mode: &Mode) -> &'static str {
    match mode {
        Mode::Graph { .. } => "KGLite (single-graph)",
        Mode::SourceRoot { .. } => "KGLite (source-root)",
        Mode::Workspace { .. } => "KGLite (workspace)",
        Mode::LocalWorkspace { .. } => "KGLite (local-workspace)",
        Mode::Watch { .. } => "KGLite (watch)",
        Mode::Vault { .. } => "KGLite (vault)",
        Mode::Bare => "KGLite",
    }
}

pub(crate) fn workspace_graph_mode(mode: &Mode) -> Option<WorkspaceGraphMode> {
    match mode {
        Mode::Workspace { .. } => Some(WorkspaceGraphMode::Workspace),
        Mode::LocalWorkspace { .. } => Some(WorkspaceGraphMode::LocalWorkspace),
        // `WorkspaceGraphMode` is the *producer's* view of the request, and
        // the vault producer is constructed knowing its own root — it has
        // nothing to learn from a `Vault` discriminant that a `Watch` one does
        // not already tell it. Aliasing keeps the public enum (and every
        // downstream match on it) unchanged.
        Mode::Watch { .. } | Mode::Vault { .. } => Some(WorkspaceGraphMode::Watch),
        _ => None,
    }
}

pub(crate) fn default_manifest_path(mode: &Mode) -> Option<PathBuf> {
    match mode {
        Mode::Graph { path } => find_sibling_manifest(path),
        Mode::Workspace { dir } | Mode::Watch { dir } | Mode::Vault { dir } => {
            find_workspace_manifest(dir)
        }
        Mode::LocalWorkspace { root, .. } => find_workspace_manifest(root),
        Mode::SourceRoot { .. } | Mode::Bare => None,
    }
}

pub(crate) fn load_manifest(cli: &Cli, mode: &Mode) -> Result<Option<Manifest>, ManifestError> {
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

/// Fail fast on bad mode-specific path arguments before any expensive setup.
pub(crate) fn validate_mode_paths(mode: &Mode, cli: &Cli) -> Result<()> {
    if let Mode::Graph { .. } = mode {
        // Validate --storage up front. It applies to both branches — creating a
        // missing graph in that mode, or converting an existing one to it — so
        // an unknown spelling must fail before any of that, not inside it.
        if let Some(s) = &cli.storage {
            StorageMode::parse(s).map_err(|e| anyhow::anyhow!(e))?;
        }
        // Existence and open-vs-create are resolved exactly once by
        // `open_or_create_graph` in bind_mode; checking here would introduce a
        // second TOCTOU decision.
    }
    if let Mode::SourceRoot { dir } | Mode::Watch { dir } | Mode::Vault { dir } = mode {
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
/// `mcp-server` binary (`crates/mcp-server/src/main.rs`). Returns `mode`
/// unchanged when no local-workspace manifest is in play.
pub(crate) fn promote_local_workspace(mode: Mode, manifest: Option<&Manifest>) -> Result<Mode> {
    let Some(wcfg) = manifest.and_then(|m| m.workspace.as_ref()) else {
        return Ok(mode);
    };
    if wcfg.kind != WorkspaceKind::Local {
        return Ok(mode);
    }
    // Upstream, `adopt_client_roots` is a fallback for when no root is
    // configured, and it lets `root` be omitted. Local mode here always binds
    // an explicit root, so the key can never take effect: refuse it instead of
    // booting with it inert, or failing a rootless manifest on "missing root".
    if wcfg.adopt_client_roots {
        anyhow::bail!(
            "manifest.workspace.adopt_client_roots is not supported by kglite-mcp-server: \
             local-workspace mode always binds the explicit `workspace.root`, so a \
             client-advertised root would never be adopted. Remove the key and set \
             `workspace.root` (switch roots at runtime with `set_root_dir`)."
        );
    }
    let m = manifest.expect("manifest present when wcfg is");
    let raw_root = wcfg.root.as_ref().ok_or_else(|| {
        anyhow::anyhow!("manifest.workspace.kind=local is missing required `root`")
    })?;
    let resolved = manifest_relative_path(m, raw_root)
        .canonicalize()
        .with_context(|| {
            format!("workspace.root {raw_root:?} resolves to a path that does not exist")
        })?;
    Ok(Mode::LocalWorkspace {
        root: resolved,
        watch: wcfg.watch,
    })
}

/// Split the comma-separated `--write-scope` value into node types.
///
/// Whitespace around a name is dropped and empty segments are ignored, so
/// `"Plan, Task,"` is `["Plan", "Task"]`. `--write-scope ""` therefore yields
/// an **empty** list, which is a deliberate fail-closed configuration ("this
/// server permits no writes") rather than an absent pin — the absent pin is
/// the flag not being passed at all.
pub(crate) fn parse_write_scope_flag(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

/// The directory to start the `.env` walk-up from, per mode (the mode's own
/// directory for source-aware modes, cwd for bare).
pub(crate) fn resolve_env_start_dir(mode: &Mode) -> PathBuf {
    match mode {
        Mode::Graph { path } => path
            .canonicalize()
            .ok()
            .and_then(|p| p.parent().map(PathBuf::from))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))),
        Mode::SourceRoot { dir }
        | Mode::Workspace { dir }
        | Mode::Watch { dir }
        | Mode::Vault { dir } => dir.clone(),
        Mode::LocalWorkspace { root, .. } => root.clone(),
        Mode::Bare => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    }
}

#[cfg(test)]
mod cli_contract_tests {
    use super::{parse_write_scope_flag, Cli};
    use clap::Parser;

    #[test]
    fn version_flag_displays_version() {
        let error = Cli::try_parse_from(["kglite-mcp-server", "--version"])
            .expect_err("--version exits through clap");
        assert_eq!(error.kind(), clap::error::ErrorKind::DisplayVersion);
        assert!(error.to_string().contains(env!("CARGO_PKG_VERSION")));
    }

    #[test]
    fn retired_trust_tools_flag_is_rejected() {
        let error = Cli::try_parse_from(["kglite-mcp-server", "--trust-tools"])
            .expect_err("removed no-op flag must not parse");
        assert!(error.to_string().contains("--trust-tools"));
    }

    #[test]
    fn write_scope_flag_parses_a_comma_separated_list() {
        let cli = Cli::parse_from([
            "kglite-mcp-server",
            "--graph",
            "g.kgl",
            "--writable",
            "--write-scope",
            "Plan,Task",
        ]);
        assert_eq!(cli.write_scope.as_deref(), Some("Plan,Task"));
        assert_eq!(
            parse_write_scope_flag(cli.write_scope.as_deref().unwrap()),
            vec!["Plan".to_string(), "Task".to_string()]
        );
        // Absent flag = no operator pin at all (distinct from an empty pin).
        let bare = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);
        assert!(bare.write_scope.is_none());
    }

    #[test]
    fn lease_label_flag_parses_and_defaults_absent() {
        let named = Cli::parse_from([
            "kglite-mcp-server",
            "--graph",
            "g.kgl",
            "--lease-label",
            "Claude Desktop",
        ]);
        assert_eq!(named.lease_label.as_deref(), Some("Claude Desktop"));
        // Absent flag = fall through to KGLITE_LEASE_LABEL, then to the parent
        // process name; `boot_lease_label` owns that precedence.
        let bare = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);
        assert!(bare.lease_label.is_none());
    }

    #[test]
    fn parallel_flag_parses_and_defaults_off() {
        let on = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl", "--parallel"]);
        assert!(on.parallel);
        let off = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);
        assert!(
            !off.parallel,
            "a server's cores belong to its clients — the runtime is never opted in by omission"
        );
    }

    #[test]
    fn valid_time_default_flag_parses_three_spellings_and_refuses_the_rest() {
        use kglite::api::temporal::ValidTimeDefault;
        let parse = |value: &str| {
            Cli::try_parse_from([
                "kglite-mcp-server",
                "--graph",
                "g.kgl",
                "--valid-time-default",
                value,
            ])
        };
        assert_eq!(
            parse("all").unwrap().valid_time_default,
            Some(ValidTimeDefault::All)
        );
        assert_eq!(
            parse("today").unwrap().valid_time_default,
            Some(ValidTimeDefault::Today)
        );
        assert_eq!(
            parse("2015-06-15")
                .unwrap()
                .valid_time_default
                .map(|d| d.to_string()),
            Some("2015-06-15".to_string())
        );
        assert!(parse("yesterday").is_err());
        let bare = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);
        assert_eq!(bare.valid_time_default, None);
    }

    #[test]
    fn write_scope_flag_trims_and_drops_empty_segments() {
        assert_eq!(
            parse_write_scope_flag(" Plan , Task ,"),
            vec!["Plan".to_string(), "Task".to_string()]
        );
        // Explicitly empty: a fail-closed pin, not an absent one.
        assert!(parse_write_scope_flag("").is_empty());
        assert!(parse_write_scope_flag(" , ").is_empty());
    }
}
