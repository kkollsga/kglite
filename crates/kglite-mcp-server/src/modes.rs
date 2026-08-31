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

/// Resolve a manifest's declared `source_root:` / `source_roots:` without
/// making a stale path fatal.
///
/// A declared root that no longer exists costs exactly three tools —
/// `read_source` / `grep` / `list_source`. The graph tools never read it, so
/// aborting boot over one is disproportionate: the operator report that
/// prompted this (0.16.17) was a deployment directory copied without its
/// `source/` subdir, where the process exited before `initialize` and the
/// client could not distinguish a misconfigured server from a crashed binary.
/// On failure this warns, records the detail for the boot summary, and returns
/// `None` — the caller then binds no source roots at all.
///
/// **No fallback.** In `--graph` mode the caller must NOT fall back to the
/// auto-bound parent of the `.kgl` when a declared root fails: substituting a
/// different directory for the one the operator asked for is the silent-wrong-
/// root failure the explicit-YAML-wins rule below exists to prevent.
///
/// mcp-methods' `resolve_source_roots` is all-or-nothing — it returns on the
/// first entry that does not canonicalise — so a multi-root manifest with one
/// missing entry degrades the whole set. Serving the survivors would need a
/// per-root upstream API that does not exist.
fn resolve_declared_source_roots(
    manifest: &Manifest,
    unavailable: &mut Option<String>,
) -> Option<Vec<String>> {
    match resolve_source_roots(manifest) {
        Ok(roots) => Some(roots),
        Err(e) => {
            tracing::warn!(
                manifest = %manifest.yaml_path.display(),
                detail = %e.message,
                "manifest source_root does not resolve — serving without source tools \
                 (read_source / grep / list_source); graph tools are unaffected"
            );
            *unavailable = Some(e.message);
            None
        }
    }
}

/// Apply mode-specific bindings — source roots, workspace handle, initial
/// graph load/build — onto `options`, returning the transformed value plus,
/// when the manifest declared source roots that did not resolve, the reason
/// the source tools are unavailable (for the boot summary and the degraded
/// tool-result footer).
pub(crate) fn bind_mode(
    mode: &Mode,
    cli: &Cli,
    manifest: Option<&Manifest>,
    graph_state: &GraphState,
    mut options: ServerOptions,
) -> Result<(ServerOptions, Option<String>)> {
    let mut source_tools_unavailable: Option<String> = None;
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
            // auto-bind only when the manifest doesn't declare one —
            // and a declaration that fails to resolve keeps winning:
            // it degrades to no source tools rather than falling back
            // to the parent directory (see
            // `resolve_declared_source_roots`).
            let roots = match manifest.filter(|m| !m.source_roots.is_empty()) {
                Some(m) => resolve_declared_source_roots(m, &mut source_tools_unavailable)
                    .unwrap_or_default(),
                None => base
                    .parent()
                    .map(|parent| vec![parent.to_string_lossy().into_owned()])
                    .unwrap_or_default(),
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
            if let Some(m) = manifest.filter(|m| !m.source_roots.is_empty()) {
                if let Some(resolved) =
                    resolve_declared_source_roots(m, &mut source_tools_unavailable)
                {
                    options = options.with_static_source_roots(resolved);
                }
            }
        }
    }
    Ok((options, source_tools_unavailable))
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

/// A manifest-declared source root that no longer exists must not take the
/// server down.
///
/// The operator report (0.16.17): a deployment directory copied without its
/// `source/` subdir exited before `initialize`, so every graph tool was lost
/// to a configuration problem that only affects `read_source` / `grep` /
/// `list_source`. From the client side that is indistinguishable from a
/// crashed binary. These tests pin both halves of the contract — boot
/// survives, and the source roots stay unbound rather than silently falling
/// back to some other directory.
#[cfg(test)]
mod source_root_degradation_tests {
    use super::*;

    use clap::Parser;
    use mcp_methods::server::Manifest;

    use crate::tools::GraphState;

    fn manifest_with(dir: &std::path::Path, body: &str) -> Manifest {
        let path = dir.join("fixture_mcp.yaml");
        std::fs::write(&path, body).expect("write manifest");
        mcp_methods::server::load_manifest(&path).expect("manifest loads")
    }

    /// The bound roots, or `None` when the source tools have nothing to serve.
    fn bound_roots(options: &ServerOptions) -> Option<Vec<String>> {
        options.source_roots.as_ref().map(|provider| provider())
    }

    fn bind_graph_mode(
        dir: &std::path::Path,
        manifest: Option<&Manifest>,
    ) -> Result<(ServerOptions, Option<String>)> {
        let cli = Cli::parse_from([
            "kglite-mcp-server",
            "--graph",
            "g.kgl",
            "--storage",
            "memory",
        ]);
        let state = GraphState::new(None);
        let mode = Mode::Graph {
            path: dir.join("g.kgl"),
        };
        bind_mode(&mode, &cli, manifest, &state, ServerOptions::default())
    }

    #[test]
    fn graph_mode_serves_on_without_a_missing_declared_root() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().canonicalize().expect("canonicalize");
        let m = manifest_with(&dir, "name: fixture\nsource_root: source\n");

        let (options, unavailable) = bind_graph_mode(&dir, Some(&m)).expect(
            "a manifest source_root that does not exist must not abort boot — the graph \
             tools do not read it",
        );
        assert_eq!(
            bound_roots(&options),
            None,
            "a declared root that failed to resolve must leave the source tools unbound; \
             falling back to the .kgl's parent would serve a directory the operator never \
             asked for"
        );
        let detail = unavailable.expect("the degraded state must be reported to the operator");
        assert!(
            detail.contains("source") && detail.contains(&dir.join("source").display().to_string()),
            "the boot-summary detail must name the declaration and the path it resolved \
             to, got: {detail}"
        );
    }

    #[test]
    fn graph_mode_binds_a_declared_root_that_exists() {
        // Control for the test above: without this arm, "roots are unbound"
        // would also pass on a build where a declared root never binds at all.
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().canonicalize().expect("canonicalize");
        std::fs::create_dir(dir.join("source")).expect("mkdir source");
        let m = manifest_with(&dir, "name: fixture\nsource_root: source\n");

        let (options, unavailable) = bind_graph_mode(&dir, Some(&m)).expect("binds");
        assert_eq!(
            bound_roots(&options),
            Some(vec![dir.join("source").to_string_lossy().into_owned()]),
            "an existing declared root must still win over the .kgl parent auto-bind"
        );
        assert_eq!(unavailable, None, "nothing is degraded here");
    }

    #[test]
    fn graph_mode_auto_binds_the_parent_when_nothing_is_declared() {
        // Second control: the auto-bind path is what the missing-root case must
        // NOT fall back to, so it has to be demonstrably alive.
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().canonicalize().expect("canonicalize");
        let m = manifest_with(&dir, "name: fixture\n");

        let (options, unavailable) = bind_graph_mode(&dir, Some(&m)).expect("binds");
        assert_eq!(
            bound_roots(&options),
            Some(vec![dir.to_string_lossy().into_owned()]),
            "with no declaration the .kgl's parent is auto-bound"
        );
        assert_eq!(unavailable, None);
    }

    #[test]
    fn bare_mode_serves_on_without_a_missing_declared_root() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().canonicalize().expect("canonicalize");
        let m = manifest_with(&dir, "name: fixture\nsource_root: source\n");
        let cli = Cli::parse_from(["kglite-mcp-server"]);
        let state = GraphState::new(None);

        let (options, unavailable) = bind_mode(
            &Mode::Bare,
            &cli,
            Some(&m),
            &state,
            ServerOptions::default(),
        )
        .expect("a manifest source_root that does not exist must not abort bare boot");
        assert_eq!(bound_roots(&options), None);
        let detail = unavailable.expect("the degraded state must be reported to the operator");
        assert!(
            detail.contains(&dir.join("source").display().to_string()),
            "detail must name the missing path, got: {detail}"
        );
    }

    #[test]
    fn bare_mode_binds_a_declared_root_that_exists() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().canonicalize().expect("canonicalize");
        std::fs::create_dir(dir.join("source")).expect("mkdir source");
        let m = manifest_with(&dir, "name: fixture\nsource_root: source\n");
        let cli = Cli::parse_from(["kglite-mcp-server"]);
        let state = GraphState::new(None);

        let (options, unavailable) = bind_mode(
            &Mode::Bare,
            &cli,
            Some(&m),
            &state,
            ServerOptions::default(),
        )
        .expect("binds");
        assert_eq!(
            bound_roots(&options),
            Some(vec![dir.join("source").to_string_lossy().into_owned()])
        );
        assert_eq!(unavailable, None);
    }

    /// One missing entry degrades the whole set: mcp-methods'
    /// `resolve_source_roots` returns on the first failure, so there is no
    /// per-root survivor list to bind. Pinned so an upstream bump that gains a
    /// per-root API surfaces here as a failure rather than passing unnoticed.
    #[test]
    fn one_missing_entry_degrades_every_declared_root() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().canonicalize().expect("canonicalize");
        std::fs::create_dir(dir.join("present")).expect("mkdir present");
        let m = manifest_with(
            &dir,
            "name: fixture\nsource_roots:\n  - present\n  - absent\n",
        );

        let (options, unavailable) = bind_graph_mode(&dir, Some(&m)).expect("boots");
        assert_eq!(bound_roots(&options), None);
        assert!(unavailable.is_some_and(|d| d.contains("absent")));
    }
}
