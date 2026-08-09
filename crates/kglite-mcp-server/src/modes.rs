//! Mode-specific boot wiring: workspace boundaries, github and local
//! workspace handles, and the per-mode source-root/graph binding.

use std::path::PathBuf;

use anyhow::{Context, Result};
use kglite::api::io::OpenDisposition;
use kglite::api::storage::StorageMode;
use mcp_methods::server::{resolve_source_roots, workspace, Manifest, ServerOptions};

use crate::tools::GraphState;
use crate::*;

/// Github-workspace wiring: clone-and-track activation builds the workspace
/// graph through the injected producer ([`WorkspaceGraphHooks`]); without one,
/// activation still binds source tools and the activation summary carries
/// the builder-unavailable note.
/// Apply the manifest's workspace-boundary keys to a freshly-opened workspace.
///
/// `sandbox_root` and `adopt_client_roots` are **manifest** keys parsed by
/// mcp-methods' loader, but they take effect only as *builder* calls on the
/// `Workspace`. Nothing carries them across on its own. Opening a workspace
/// without this step leaves both at their defaults, so a manifest that sets
/// them would be read, validated as known keys, and then silently ignored —
/// the exact shape of two defects this project has already had to fix
/// (`storage=` on `open`, `from_blueprint(save=True)`). If you add a third
/// boundary key upstream, wire it here in the same change.
///
/// Both keys are applied at every construction site, but the manifest loader
/// refuses either unless `workspace.kind: local` (and `kind: local` in turn
/// requires `workspace.root`), so in practice only the local path ever sees
/// them. Applying them uniformly anyway costs nothing and means a future
/// upstream relaxation of that constraint does not silently skip a mode.
pub(crate) fn apply_workspace_boundaries(
    ws: workspace::Workspace,
    manifest: Option<&Manifest>,
) -> Result<workspace::Workspace> {
    let Some(cfg) = manifest.and_then(|m| m.workspace.as_ref()) else {
        return Ok(ws);
    };
    let mut ws = ws;
    if let Some(boundary) = cfg.sandbox_root.as_deref() {
        let path = PathBuf::from(boundary);
        ws = ws
            .with_sandbox_root(&path)
            .with_context(|| format!("workspace.sandbox_root is not usable: {boundary}"))?;
    }
    if cfg.adopt_client_roots {
        ws = ws.with_adopt_client_roots();
    }
    Ok(ws)
}

pub(crate) fn github_workspace(
    canon: PathBuf,
    stale_after_days: u32,
    graph_state: &GraphState,
    manifest: Option<&Manifest>,
) -> Result<workspace::Workspace> {
    let ws = workspace::Workspace::open(canon, stale_after_days, None)
        .context("workspace init failed")?
        .with_activation_transaction(workspace_activation_transaction(graph_state));
    apply_workspace_boundaries(ws, manifest)
}

/// Local-workspace wiring (`set_root_dir` root swaps).
///
/// `Workspace::open_local` stores the activation transaction but does not fire
/// it at boot. Each `set_root_dir` prepares a graph off-lock; mcp-methods 0.4.1
/// commits it only if that request remains the latest activation intent.
///
/// History (do not re-add): an older mcp-methods contract fired the hook
/// synchronously inside `open` with the wide `workspace.root` (~360k
/// files), parsing everything before returning and blowing past Claude
/// Desktop's 60s `initialize` window. We added an `initial_activate_seen`
/// deferral to swallow that one boot fire. mcp-methods has since removed
/// the boot fire (`open_local` no longer calls the hook), so the deferral
/// was instead swallowing the user's FIRST `set_root_dir` — leaving the
/// graph permanently unbuilt in local mode ("No active graph" on every
/// graph tool). Operator inbox 2026-06-23 (+ original 2026-06-06 repro).
/// Deferral removed; preparation is safe because mcp-methods no longer fires
/// any activation callback at boot.
pub(crate) fn local_workspace(
    root: PathBuf,
    graph_state: &GraphState,
    manifest: Option<&Manifest>,
) -> Result<workspace::Workspace> {
    let ws = workspace::Workspace::open_local(root, None)
        .context("local-workspace init failed")?
        .with_activation_transaction(workspace_activation_transaction(graph_state));
    apply_workspace_boundaries(ws, manifest)
}

/// Apply mode-specific bindings — source roots, workspace handle, initial
/// graph load/build — onto `options`, returning the transformed value.
pub(crate) fn bind_mode(
    mode: &Mode,
    cli: &Cli,
    manifest: Option<&Manifest>,
    graph_state: &GraphState,
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
                    .build_workspace_graph(&canon, None)
                    .context("initial workspace graph build failed")?;
            }
        }
        Mode::Workspace { dir } => {
            let canon = dir.canonicalize().unwrap_or_else(|_| dir.clone());
            let ws = github_workspace(canon, cli.stale_after_days, graph_state, manifest)?;
            options = options.with_workspace(ws);
        }
        Mode::LocalWorkspace { root, .. } => {
            let ws = local_workspace(root.clone(), graph_state, manifest)?;
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

/// The manifest's workspace-boundary keys must actually reach the `Workspace`.
///
/// mcp-methods parses `workspace.sandbox_root` and `workspace.adopt_client_roots`
/// in its manifest loader, but they only take effect as builder calls on the
/// `Workspace` we construct. Nothing carries them across implicitly. Before
/// 0.4.3 was wired in, opening a workspace ignored both — a manifest setting
/// them would be read, accepted as valid, and then do nothing.
///
/// That is the same shape as two defects this project already shipped and had
/// to fix: `storage=` silently ignored by `open()`, and
/// `from_blueprint(save=True)` silently doing nothing. Both were "the kwarg is
/// parsed, then dropped". These tests exist so a third one cannot happen here
/// quietly.
#[cfg(test)]
mod workspace_boundary_tests {
    use super::*;

    use mcp_methods::server::{workspace, Manifest};

    use crate::tools::GraphState;

    /// Build a manifest by writing TOML and loading it through mcp-methods'
    /// own loader — deliberately, rather than constructing the struct. That
    /// exercises the whole chain the user actually traverses: the key is
    /// spelled in a file, recognised by the loader, lands in the struct, and
    /// reaches the builder. A struct-literal test would skip the two steps
    /// most likely to break on an upstream bump.
    fn manifest_with(dir: &std::path::Path, body: &str) -> Manifest {
        let path = dir.join("mcp-manifest.yaml");
        std::fs::write(&path, body).expect("write manifest");
        mcp_methods::server::load_manifest(&path).expect("manifest loads")
    }

    /// Go through `local_workspace` — the REAL construction path — not through
    /// `apply_workspace_boundaries` directly.
    ///
    /// This helper called the applier directly for one revision, and both tests
    /// below passed with the plumbing deleted from `local_workspace`. They were
    /// testing the applier in isolation, which is the one thing that was never
    /// in doubt; the regression that matters is a construction site forgetting
    /// to call it. Mutation caught it. Route through the caller or the tests
    /// prove nothing about wiring.
    fn open(dir: &std::path::Path, manifest: Option<&Manifest>) -> workspace::Workspace {
        let state = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace));
        local_workspace(dir.to_path_buf(), &state, manifest).expect("local workspace opens")
    }

    #[test]
    fn adopt_client_roots_reaches_the_workspace() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().canonicalize().expect("canonicalize");

        // The control arm is what makes the positive arm mean anything: if the
        // default were already `true`, the assertion below would pass without
        // the plumbing existing at all.
        let without = open(&root, None);
        assert!(
            !without.adopts_client_roots(),
            "no manifest must leave adoption off — otherwise the positive arm proves nothing"
        );
        let m_off = manifest_with(
            tmp.path(),
            &format!(
                "workspace:\n  kind: local\n  root: {:?}\n  adopt_client_roots: false\n",
                root.to_string_lossy()
            ),
        );
        let off = open(&root, Some(&m_off));
        assert!(
            !off.adopts_client_roots(),
            "a manifest that does not opt in must leave adoption off"
        );

        let m_on = manifest_with(
            tmp.path(),
            &format!(
                "workspace:\n  kind: local\n  root: {:?}\n  adopt_client_roots: true\n",
                root.to_string_lossy()
            ),
        );
        let on = open(&root, Some(&m_on));
        assert!(
            on.adopts_client_roots(),
            "workspace.adopt_client_roots = true must reach the Workspace; if this \
             fails the manifest key is being parsed and then dropped"
        );
    }

    /// `sandbox_root` has no public reader, so it is asserted through the
    /// behaviour it exists for: the ACTIVE ROOT must not move when a swap
    /// outside the boundary is attempted.
    ///
    /// An earlier version compared the two response strings and passed with the
    /// plumbing deleted — they differ for reasons unrelated to the boundary, so
    /// the comparison proved nothing. Assert the state, not the message.
    #[test]
    fn sandbox_root_reaches_the_workspace_and_bounds_a_swap() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let base = tmp.path().canonicalize().expect("canonicalize");
        let inside = base.join("inside");
        let outside = base.join("outside");
        std::fs::create_dir_all(&inside).expect("mkdir inside");
        std::fs::create_dir_all(&outside).expect("mkdir outside");

        let m = manifest_with(
            tmp.path(),
            &format!(
                "workspace:\n  kind: local\n  root: {:?}\n  sandbox_root: {:?}\n",
                inside.to_string_lossy(),
                inside.to_string_lossy()
            ),
        );
        let bounded = open(&inside, Some(&m));
        let before = bounded.active_repo_path().expect("an active root");
        let _ = bounded.set_root_dir(&outside, None);
        assert_eq!(
            bounded.active_repo_path().expect("still an active root"),
            before,
            "a swap outside workspace.sandbox_root must leave the active root \
             unchanged; if it moved, sandbox_root never reached the Workspace"
        );

        // Control: without the key the identical swap MUST move the root.
        // Without this arm the assertion above would also pass on a build where
        // set_root_dir is broken for every input.
        let unbounded = open(&inside, None);
        let _ = unbounded.set_root_dir(&outside, None);
        assert_eq!(
            unbounded.active_repo_path().expect("an active root"),
            outside.canonicalize().expect("canonicalize outside"),
            "with no sandbox_root the same swap must succeed — otherwise the \
             refusal above cannot be attributed to the boundary"
        );
    }
}
