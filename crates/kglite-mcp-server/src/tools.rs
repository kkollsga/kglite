//! KGLite-specific MCP tools: `cypher_query`, `graph_overview`, `save_graph`.
//!
//! All three close over a [`GraphState`] holding the active
//! [`kglite::api::KnowledgeGraph`] behind an `Arc<RwLock<…>>`. Wired
//! into the framework's tool router via `register_typed_tool` so they
//! sit alongside the built-in source / GitHub tools.
//!
//! 0.9.18: rewritten against the pure-Rust `kglite::api` surface.
//! There is no `Python::attach` anywhere in this module — the binary
//! has no libpython link at all.

use std::path::Path;
use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::{Duration, SystemTime};

use anyhow::Result;
use kglite::api::cypher;
use kglite::api::cypher::ValueCodec;
use kglite::api::introspection::{
    compute_description, compute_schema, ConnectionDetail, CypherDetail, FluentDetail,
};
use kglite::api::io::{open_or_create_graph, GraphWriterLease, OpenDisposition};
use kglite::api::storage::StorageMode;
use kglite::api::{Embedder, KnowledgeGraph, Value};
use mcp_methods::server::McpServer;
use serde::{Deserialize, Serialize};

const NO_GRAPH: &str =
    "No active graph. Pass --graph X.kgl, or activate one via repo_management('org/repo').";

/// Hot-fail guard for the lazy workspace-graph rebuild: after this many
/// consecutive failures for the same target, [`GraphState::ensure_workspace_graph_fresh`]
/// stops restoring the dirty marker (no more per-tool-call retries) and
/// keeps serving the stale graph — with the failure surfaced in tool
/// output — until a new FS event re-tags the target.
const MAX_CONSECUTIVE_REBUILD_FAILURES: u32 = 3;

/// Lock the `RwLock` for reading, recovering a poisoned lock instead of
/// propagating the panic. This is the mcp-server-wide lock policy: every
/// guarded value in this crate (the active-graph slot, the pending-rebuild
/// slot, the rebuild status, the watch root) is a swap-in-place
/// `Option`/`Arc` with no multi-step invariants, so the state a panicking
/// holder left behind is always coherent. Without recovery, one panic
/// while holding a lock poisons it and every later MCP request panics —
/// wedging the server until restart.
pub(crate) fn read_lock<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(PoisonError::into_inner)
}

/// Write-lock companion to [`read_lock`] — same poison-recovery policy.
pub(crate) fn write_lock<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(PoisonError::into_inner)
}

/// Refusal surfaced when a workspace mode has no graph producer configured.
const NO_BUILDER_MSG: &str = "workspace-graph building is not configured in this binary. \
Embed kglite-mcp-server and inject WorkspaceGraphHooks through \
ServerExtensions::with_workspace_graph. For source-code graphs, use codingest-mcp. \
Reading existing .kgl graphs with --graph remains available.";

/// Server mode that requested a workspace graph.
///
/// Producers own all domain policy derived from this value, including which
/// files to ingest. KGLite does not assume source languages or documentation
/// behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum WorkspaceGraphMode {
    /// Clone-backed `--workspace` mode.
    Workspace,
    /// Manifest-declared local workspace activated through `set_root_dir`.
    LocalWorkspace,
    /// Fixed-directory `--watch` mode.
    Watch,
}

/// One producer request for a workspace graph.
pub struct WorkspaceGraphRequest {
    root: std::path::PathBuf,
    revisions: Option<Vec<String>>,
    mode: WorkspaceGraphMode,
}

impl WorkspaceGraphRequest {
    pub(crate) fn new(
        root: std::path::PathBuf,
        revisions: Option<Vec<String>>,
        mode: WorkspaceGraphMode,
    ) -> Self {
        Self {
            root,
            revisions,
            mode,
        }
    }

    /// Canonical source root to build.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolved revisions requested by activation, oldest to newest.
    /// `None` means the producer should build its ordinary working-tree view.
    pub fn revisions(&self) -> Option<&[String]> {
        self.revisions.as_deref()
    }

    /// Workspace mode that originated the request.
    pub fn mode(&self) -> WorkspaceGraphMode {
        self.mode
    }
}

/// Completed graph plus the canonical revision labels represented by it.
pub struct WorkspaceGraphResult {
    graph: Arc<kglite::api::DirGraph>,
    revisions: Option<Vec<String>>,
}

impl WorkspaceGraphResult {
    /// Return a normal working-tree graph.
    pub fn new(graph: Arc<kglite::api::DirGraph>) -> Self {
        Self {
            graph,
            revisions: None,
        }
    }

    /// Return a graph spanning canonicalized revision labels.
    pub fn with_revisions(graph: Arc<kglite::api::DirGraph>, revisions: Vec<String>) -> Self {
        Self {
            graph,
            revisions: Some(revisions),
        }
    }

    fn into_parts(self) -> (Arc<kglite::api::DirGraph>, Option<Vec<String>>) {
        (self.graph, self.revisions)
    }
}

/// Borrowed watch-change context passed to the producer's relevance policy.
pub struct WorkspaceGraphRelevance<'a> {
    path: &'a Path,
    mode: WorkspaceGraphMode,
}

impl<'a> WorkspaceGraphRelevance<'a> {
    pub(crate) fn new(path: &'a Path, mode: WorkspaceGraphMode) -> Self {
        Self { path, mode }
    }

    /// Changed path reported by the watcher.
    pub fn path(&self) -> &'a Path {
        self.path
    }

    /// Workspace mode whose active graph would be rebuilt.
    pub fn mode(&self) -> WorkspaceGraphMode {
        self.mode
    }
}

/// Unified plain/revision-set workspace graph build closure.
pub type WorkspaceGraphBuildFn =
    dyn Fn(WorkspaceGraphRequest) -> Result<WorkspaceGraphResult, String> + Send + Sync;

/// Producer-owned watch relevance policy.
pub type WorkspaceGraphRelevanceFn =
    dyn for<'a> Fn(WorkspaceGraphRelevance<'a>) -> bool + Send + Sync;

/// Generic workspace-graph lifecycle extension for embedding binaries.
pub struct WorkspaceGraphHooks {
    /// Build the graph requested by KGLite. The producer owns revision
    /// canonicalization and all domain-specific ingestion policy.
    pub build: Box<WorkspaceGraphBuildFn>,
    /// Return whether a changed path can affect the active graph.
    pub is_relevant: Box<WorkspaceGraphRelevanceFn>,
}

/// Shared active-graph state. Cloning is cheap (Arc).
#[derive(Clone, Default)]
pub struct GraphState {
    inner: Arc<RwLock<Option<ActiveGraph>>>,
    /// Deferred-rebuild slot. The watcher tags the active root here
    /// (cheap, microseconds — sets the slot, drops the lock); each
    /// MCP tool entry calls [`ensure_workspace_graph_fresh`] which atomically
    /// `take()`s the slot and rebuilds. Pattern: do the actual work
    /// lazily, never on the watcher thread. N FS events between two
    /// tool calls → 1 rebuild (the slot just holds the latest target).
    pending_rebuild: Arc<RwLock<Option<WorkspaceGraphTarget>>>,
    /// Outcome bookkeeping for the lazy rebuild: the last failure (kept
    /// until the next successful build and surfaced in tool output next
    /// to the built-at identity) plus a consecutive-failure counter
    /// implementing the [`MAX_CONSECUTIVE_REBUILD_FAILURES`] hot-fail
    /// guard.
    rebuild_status: Arc<RwLock<RebuildStatus>>,
    /// Workspace mode used to build request/relevance context. `None` for
    /// graph/source-root/bare modes that never ask a producer to build.
    workspace_mode: Option<WorkspaceGraphMode>,
    /// Manifest-declared value codecs (`extensions.value_codecs`). Server-
    /// config, set once at boot via [`with_value_codecs`] and carried by every
    /// clone; passed to `ExecuteOptions::value_codecs` on each `cypher_query` /
    /// `tools[].cypher` run so the engine decodes query-side literals and
    /// encodes result columns (`'Q42'` ↔ `42`) — safely, after parsing.
    value_codecs: Option<Arc<Vec<ValueCodec>>>,
    /// External workspace-graph lifecycle extension. Set once at boot and
    /// carried by every clone so lazy watch rebuilds use the same producer.
    workspace_graph_hooks: Option<Arc<WorkspaceGraphHooks>>,
}

/// Bookkeeping for lazy workspace-graph rebuild failures. Reset to default on
/// the next successful build.
#[derive(Default)]
struct RebuildStatus {
    /// Human-readable description of the last failed rebuild.
    last_error: Option<String>,
    /// When that failure happened (for age display).
    failed_at: Option<SystemTime>,
    /// Consecutive failures for `failed_target` with no intervening
    /// success.
    consecutive_failures: u32,
    /// The target whose rebuilds keep failing.
    failed_target: Option<WorkspaceGraphTarget>,
}

/// Exact installed workspace product a watcher event observed. The generation
/// prevents a slow rebuild prepared for an older activation from overwriting a
/// newer graph, even when the root/revision labels later repeat.
#[derive(Clone, Debug, Eq, PartialEq)]
struct WorkspaceGraphTarget {
    root: std::path::PathBuf,
    revisions: Option<Vec<String>>,
    generation: u64,
}

struct ActiveGraph {
    kg: KnowledgeGraph,
    source_path: Option<std::path::PathBuf>,
    /// Held for every path-backed graph because this MCP surface can publish
    /// mutations through `save_graph` / `save_graph_as`.
    writer_lease: Option<GraphWriterLease>,
    /// The source root this graph was built/loaded from — a code-tree
    /// directory or a `.kgl` file path. Stamped into agent-facing output
    /// (the `<active_graph/>` header, the `cypher_query` footer, and the
    /// activation message) so an agent can see which root it is querying and
    /// spot a stale graph. `None` for an in-memory graph created without a
    /// path.
    root: Option<std::path::PathBuf>,
    /// The resolved git revisions this graph spans, when it was built as a
    /// revision-set graph — oldest → newest, HEAD
    /// last. `None` for a plain single-rev / loaded graph. Surfaced in the
    /// `<active_graph …>` header (`revs="…"`) and the activation summary so an
    /// agent knows unscoped queries span all these revs (the over-count trap)
    /// and can scope with `WHERE '<rev>' IN n.revs`.
    revs: Option<Vec<String>>,
    /// Wall-clock time this graph was built/loaded. Surfaced next to `root`
    /// so an agent can tell how fresh the active graph is.
    built_at: SystemTime,
    /// Monotonic identity of this installed graph within the server process.
    generation: u64,
}

/// Producer output prepared off the workspace activation lock. Publication is
/// deliberately separate so mcp-methods can discard a superseded request
/// without this graph ever becoming visible.
pub(crate) struct PreparedWorkspaceGraph {
    active: ActiveGraph,
    summary: Option<String>,
}

/// Format a `SystemTime` as a second-precision UTC ISO-8601 timestamp.
fn iso8601(t: SystemTime) -> String {
    chrono::DateTime::<chrono::Utc>::from(t)
        .format("%Y-%m-%dT%H:%M:%SZ")
        .to_string()
}

/// Human-readable age of `t` relative to now (e.g. `3s`, `4m`, `2h 5m`,
/// `1d 3h`). Saturates to `0s` if `t` is somehow in the future.
fn humanize_age(t: SystemTime) -> String {
    let secs = SystemTime::now()
        .duration_since(t)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m", secs / 60)
    } else if secs < 86_400 {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    } else {
        format!("{}d {}h", secs / 86_400, (secs % 86_400) / 3600)
    }
}

impl ActiveGraph {
    fn workspace_target(&self) -> Option<WorkspaceGraphTarget> {
        Some(WorkspaceGraphTarget {
            root: self.root.clone()?,
            revisions: self.revs.clone(),
            generation: self.generation,
        })
    }

    /// `root="…" built_at="…" age="…"` attributes for the `<active_graph/>`
    /// header injected above the `graph_overview` schema. Omits `root` when
    /// no path is recorded.
    fn identity_attrs(&self) -> String {
        let time = format!(
            " built_at=\"{}\" age=\"{}\"",
            iso8601(self.built_at),
            humanize_age(self.built_at)
        );
        // A multi-rev graph names the loaded rev-set on the header so an agent
        // sees at a glance that unscoped queries span all these revs.
        let revs = match &self.revs {
            Some(revs) if !revs.is_empty() => format!(" revs=\"{}\"", revs.join(",")),
            _ => String::new(),
        };
        match &self.root {
            Some(r) => format!(" root={:?}{time}{revs}", r.display().to_string()),
            None => format!("{time}{revs}"),
        }
    }

    /// Compact one-line identity footer appended to `cypher_query` results so
    /// every query self-identifies which graph (and how fresh) it ran against.
    fn identity_footer(&self) -> String {
        let root = match &self.root {
            Some(r) => r.display().to_string(),
            None => "(in-memory)".to_string(),
        };
        format!(
            "\n\n— active graph: {root} · built {} ({} ago)",
            iso8601(self.built_at),
            humanize_age(self.built_at)
        )
    }
}

fn activation_summary_for_active(active: &ActiveGraph) -> Option<String> {
    let overview = compute_schema(active.kg.dir());
    if overview.node_count == 0 {
        return None;
    }
    let mut types: Vec<(&str, usize)> = overview
        .node_types
        .iter()
        .map(|(name, detail)| (name.as_str(), detail.count))
        .collect();
    types.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let top: Vec<String> = types
        .iter()
        .take(4)
        .map(|(name, count)| format!("{count} {name}"))
        .collect();
    let root_note = match &active.root {
        Some(root) => format!(
            " · root {} · built {} ago.",
            root.display(),
            humanize_age(active.built_at)
        ),
        None => format!(" · built {} ago.", humanize_age(active.built_at)),
    };
    let mut message = format!(
        "Graph ready: {} nodes ({}) · {} edges.{root_note} Start with graph_overview() \
         → cypher_query for structure (definitions, callers, types, counts, \
         paths); use grep for literal text only. If graph_overview/cypher_query aren't \
         in your loaded tools, search your tool registry for 'cypher' or 'graph_overview' \
         and load them before falling back to grep — they are always registered.",
        overview.node_count,
        top.join(", "),
        overview.edge_count,
    );
    if let Some(revisions) = active
        .revs
        .as_ref()
        .filter(|revisions| !revisions.is_empty())
    {
        if revisions.len() == 1 {
            message.push_str(&format!(
                " Code graph at revision '{}' (a committed snapshot, not the working tree).",
                revisions[0],
            ));
        } else {
            let newest = revisions.last().map(String::as_str).unwrap_or("");
            message.push_str(&format!(
                " Multi-rev graph spanning {} revisions: {}. UNSCOPED queries span ALL revs \
                 (they over-count) — scope with `WHERE '<rev>' IN n.revs` (head only: `WHERE \
                 '{newest}' IN n.revs`); for deltas use `CALL rev_diff({{from: '<rev>', \
                 to: '<rev>'}})`.",
                revisions.len(),
                revisions.join(", "),
            ));
        }
    }
    Some(message)
}

impl GraphState {
    /// Create state for an optional workspace-graph-producing mode.
    pub fn new(workspace_mode: Option<WorkspaceGraphMode>) -> Self {
        Self {
            workspace_mode,
            ..Self::default()
        }
    }

    /// Attach the manifest-declared value codecs. Builder form so they're
    /// set once at boot, before the tool closures clone the state.
    pub fn with_value_codecs(mut self, codecs: Option<Arc<Vec<ValueCodec>>>) -> Self {
        self.value_codecs = codecs;
        self
    }

    /// Attach an external workspace-graph producer. Builder form, set once at
    /// boot like [`Self::with_value_codecs`].
    pub fn with_workspace_graph(mut self, hooks: Option<Arc<WorkspaceGraphHooks>>) -> Self {
        self.workspace_graph_hooks = hooks;
        self
    }

    /// Whether an external builder is injected. Activation hooks branch on
    /// this: without a builder, "no graph after activate" is a permanent
    /// configuration state (surfaced via the activation summary), not a
    /// build failure worth erroring the activation for.
    pub fn has_workspace_graph_builder(&self) -> bool {
        self.workspace_graph_hooks.is_some()
    }

    /// Whether the configured producer considers a changed path relevant.
    pub fn is_graph_relevant(&self, p: &Path) -> bool {
        let (Some(hooks), Some(mode)) = (&self.workspace_graph_hooks, self.workspace_mode) else {
            return false;
        };
        (hooks.is_relevant)(WorkspaceGraphRelevance::new(p, mode))
    }

    /// The configured value codecs as a slice for `ExecuteOptions::value_codecs`
    /// (`None` when unconfigured — the common case).
    pub fn value_codecs(&self) -> Option<&[ValueCodec]> {
        self.value_codecs.as_deref().map(|v| v.as_slice())
    }

    /// Tag the installed workspace graph as needing rebuild. Called from the
    /// watch callback; non-blocking (two short lock-protected reads/writes).
    /// Capturing the installed revision set and generation here prevents a
    /// deferred rebuild from silently collapsing a multi-revision graph or
    /// overwriting a newer activation.
    /// The actual rebuild happens lazily on the next tool call via
    /// [`ensure_workspace_graph_fresh`].
    pub fn tag_workspace_graph_dirty(&self) {
        // Keep the active read lock through the pending write so a rebuild
        // cannot publish between capturing this receipt and enqueueing it.
        let active = read_lock(&self.inner);
        let Some(target) = active.as_ref().and_then(ActiveGraph::workspace_target) else {
            return;
        };
        tracing::debug!(
            target = %target.root.display(),
            revisions = ?target.revisions,
            generation = target.generation,
            "workspace graph tagged for rebuild"
        );
        *write_lock(&self.pending_rebuild) = Some(target);
        drop(active);
    }

    /// Rebuild the workspace graph if the watcher tagged it dirty since the
    /// last call. Called by each MCP tool entry that reads the graph
    /// (cypher_query / graph_overview / save_graph / read_code_source
    /// / explore). No-op when nothing's pending.
    ///
    /// **Failure policy.** A failed rebuild must not silently serve a
    /// stale graph forever: the dirty marker is restored so the next
    /// tool call retries, and the error is recorded on `rebuild_status`
    /// (surfaced next to the built-at identity in graph_overview /
    /// cypher_query output). To avoid a hot retry loop when the source
    /// dir is permanently broken, after
    /// [`MAX_CONSECUTIVE_REBUILD_FAILURES`] consecutive failures for the
    /// same target the marker is NOT restored — the stale graph keeps
    /// being served (error still surfaced) and the next retry happens
    /// only when a fresh FS event re-tags the target.
    pub fn ensure_workspace_graph_fresh(&self) {
        let target = write_lock(&self.pending_rebuild).take();
        let Some(target) = target else { return };
        tracing::info!(
            target = %target.root.display(),
            revisions = ?target.revisions,
            generation = target.generation,
            "rebuilding workspace graph (lazy, FS changed)"
        );
        match self.prepare_workspace_graph(&target.root, target.revisions.as_deref()) {
            Ok(prepared) => {
                if self.commit_workspace_rebuild(prepared, &target) {
                    *write_lock(&self.rebuild_status) = RebuildStatus::default();
                } else {
                    tracing::debug!(
                        target = %target.root.display(),
                        generation = target.generation,
                        "discarding workspace rebuild prepared for a superseded graph"
                    );
                }
            }
            Err(e) => {
                let Some(failures) = self.record_current_rebuild_failure(&target, &e) else {
                    tracing::debug!(
                        target = %target.root.display(),
                        generation = target.generation,
                        "discarding workspace rebuild failure for a superseded graph"
                    );
                    return;
                };
                tracing::warn!(error = %e, "lazy workspace graph rebuild failed");
                if failures < MAX_CONSECUTIVE_REBUILD_FAILURES {
                    // Restore the marker so the next tool call retries —
                    // without clobbering a newer target the watcher may
                    // have tagged while this build was running.
                    self.restore_current_rebuild_target(target);
                } else {
                    tracing::warn!(
                        target = %target.root.display(),
                        failures,
                        "workspace graph rebuild keeps failing — serving the stale \
                         graph; retrying only on the next FS event"
                    );
                }
            }
        }
    }

    /// A one-line warning describing the last failed lazy rebuild, or
    /// `None` when the last rebuild succeeded (the common case).
    /// Appended to tool output wherever the graph's built-at identity
    /// appears, so an agent knows the graph it queries is staler than
    /// the filesystem.
    pub fn rebuild_error_note(&self) -> Option<String> {
        let active = read_lock(&self.inner);
        let active_target = active.as_ref().and_then(ActiveGraph::workspace_target)?;
        let status = read_lock(&self.rebuild_status);
        if status.failed_target.as_ref() != Some(&active_target) {
            return None;
        }
        let err = status.last_error.as_ref()?;
        let age = status
            .failed_at
            .map(humanize_age)
            .unwrap_or_else(|| "?".to_string());
        let note = format!(
            "WARNING: workspace graph rebuild failed {age} ago ({} consecutive \
             failure(s)) — the active graph is STALE relative to the \
             filesystem. Error: {err}",
            status.consecutive_failures
        );
        drop(status);
        drop(active);
        Some(note)
    }

    /// Append the rebuild-failure warning (if any) to a tool response.
    fn with_rebuild_warning(&self, body: String) -> String {
        match self.rebuild_error_note() {
            Some(note) => format!("{body}\n\n{note}"),
            None => body,
        }
    }

    pub fn load_kgl(&self, path: &Path) -> Result<()> {
        // Phase G.3-pre: load_file now returns Arc<DirGraph>;
        // wrap into KnowledgeGraph here to preserve ActiveGraph's
        // existing shape (kg.set_embedder_native, kg.source_location,
        // kg.cypher, etc. are still used downstream).
        self.open_or_create(path, None).map(|_| ())
    }

    /// Create a fresh, empty graph in `mode` bound to `path` (so `save_graph`
    /// later writes back here). The create/ingest counterpart of
    /// [`Self::load_kgl`]: route through the shared core builder
    /// (`new_dir_graph_in_mode`) so the server speaks the same
    /// memory/mapped/disk vocabulary as the wheel and C ABI.
    pub fn create_in_mode(&self, path: &Path, mode: StorageMode) -> Result<()> {
        self.open_or_create(path, Some(mode)).map(|_| ())
    }

    pub fn open_or_create(
        &self,
        path: &Path,
        create_mode: Option<StorageMode>,
    ) -> Result<OpenDisposition> {
        let reuse_existing = read_lock(&self.inner)
            .as_ref()
            .is_some_and(|active| active.source_path.as_deref() == Some(path));
        let mut writer_lease = if reuse_existing {
            None
        } else {
            Some(
                GraphWriterLease::acquire(path, Duration::from_secs(30))
                    .map_err(|e| anyhow::anyhow!("kglite writer lease failed: {e}"))?,
            )
        };
        let opened = open_or_create_graph(path, create_mode)
            .map_err(|e| anyhow::anyhow!("kglite graph open/create failed: {e}"))?;
        let kg = KnowledgeGraph::from_arc(opened.graph);
        let mut guard = write_lock(&self.inner);
        if reuse_existing {
            writer_lease = guard.as_mut().and_then(|active| active.writer_lease.take());
        }
        let generation = guard
            .as_ref()
            .map_or(1, |active| active.generation.saturating_add(1));
        *guard = Some(ActiveGraph {
            kg,
            source_path: Some(path.to_path_buf()),
            writer_lease,
            root: Some(path.to_path_buf()),
            revs: None,
            built_at: SystemTime::now(),
            generation,
        });
        Ok(opened.disposition)
    }

    /// Save the active graph to an explicit `path` and rebind the active
    /// graph's `source_path` to it, so subsequent `save_graph` calls target
    /// the new location. Backs the `save_graph_as` workbench tool. Returns a
    /// human-readable status (node/edge counts) or an error string.
    fn save_as(&self, path: &Path) -> std::result::Result<String, String> {
        let mut guard = write_lock(&self.inner);
        let Some(active) = guard.as_mut() else {
            return Err(NO_GRAPH.to_string());
        };
        let replacing_target = active.source_path.as_deref() != Some(path);
        let new_lease = replacing_target
            .then(|| GraphWriterLease::acquire(path, Duration::from_secs(30)))
            .transpose()
            .map_err(|e| format!("save_graph_as writer lease error: {e}"))?;
        let path_str = path.to_string_lossy().into_owned();
        // Save through the active graph's own Arc (write lock held) so
        // `prepare_save`'s `Arc::make_mut` sees refcount 1 — no deep copy,
        // and the columnar consolidation lands on the live graph instead
        // of a discarded clone. `compute_schema` only needs `&DirGraph`.
        kglite::api::io::save_graph(active.kg.dir_mut(), &path_str)
            .map_err(|e| format!("save_graph_as error: {e}"))?;
        active.source_path = Some(path.to_path_buf());
        if let Some(lease) = new_lease {
            active.writer_lease = Some(lease);
        }
        let overview = compute_schema(active.kg.dir());
        Ok(format!(
            "Saved {path_str} ({} nodes, {} edges); save target rebound here.",
            overview.node_count, overview.edge_count
        ))
    }

    /// Ask the configured producer for a workspace graph without publishing
    /// it. Expensive parsing and summary generation happen here, outside the
    /// mcp-methods activation commit lock.
    pub(crate) fn prepare_workspace_graph(
        &self,
        root: &Path,
        revisions: Option<&[String]>,
    ) -> Result<PreparedWorkspaceGraph> {
        let Some(hooks) = &self.workspace_graph_hooks else {
            anyhow::bail!(NO_BUILDER_MSG);
        };
        let Some(mode) = self.workspace_mode else {
            anyhow::bail!("workspace-graph build requested outside a workspace/watch mode");
        };
        let request = WorkspaceGraphRequest::new(
            root.to_path_buf(),
            revisions.map(|revs| revs.to_vec()),
            mode,
        );
        let result = (hooks.build)(request)
            .map_err(|e| anyhow::anyhow!("workspace-graph build hook failed: {e}"))?;
        let (graph, revisions) = result.into_parts();
        let active = ActiveGraph {
            kg: KnowledgeGraph::from_arc(graph),
            source_path: None,
            writer_lease: None,
            root: Some(root.to_path_buf()),
            revs: revisions,
            built_at: SystemTime::now(),
            generation: 0,
        };
        let summary = activation_summary_for_active(&active);
        Ok(PreparedWorkspaceGraph { active, summary })
    }

    /// Publish one already-prepared graph and return the summary computed from
    /// that exact artifact. Keep this to a single slot swap: it may run inside
    /// mcp-methods' generation commit boundary.
    pub(crate) fn commit_workspace_graph(
        &self,
        mut prepared: PreparedWorkspaceGraph,
    ) -> Option<String> {
        let mut slot = write_lock(&self.inner);
        prepared.active.generation = slot
            .as_ref()
            .map_or(1, |active| active.generation.saturating_add(1));
        *slot = Some(prepared.active);
        prepared.summary
    }

    /// Publish a lazy rebuild only if the exact graph observed by the watcher
    /// is still installed. A watcher event that arrived during the rebuild is
    /// retargeted to the newly installed generation so it still gets one
    /// follow-up rebuild instead of being mistaken for stale work.
    fn commit_workspace_rebuild(
        &self,
        mut prepared: PreparedWorkspaceGraph,
        expected: &WorkspaceGraphTarget,
    ) -> bool {
        let mut active_slot = write_lock(&self.inner);
        if active_slot
            .as_ref()
            .and_then(ActiveGraph::workspace_target)
            .as_ref()
            != Some(expected)
        {
            return false;
        }
        prepared.active.generation = expected.generation.saturating_add(1);
        let installed_target = prepared
            .active
            .workspace_target()
            .expect("workspace rebuild always has a root");
        *active_slot = Some(prepared.active);

        let mut pending = write_lock(&self.pending_rebuild);
        if pending.as_ref() == Some(expected) {
            *pending = Some(installed_target);
        }
        true
    }

    #[cfg(test)]
    fn active_workspace_target(&self) -> Option<WorkspaceGraphTarget> {
        read_lock(&self.inner)
            .as_ref()
            .and_then(ActiveGraph::workspace_target)
    }

    /// Record a rebuild failure only while its source graph remains current.
    /// Holding the active-graph read lock through the status write closes the
    /// activation race between the identity check and failure publication.
    fn record_current_rebuild_failure(
        &self,
        target: &WorkspaceGraphTarget,
        error: &anyhow::Error,
    ) -> Option<u32> {
        let active = read_lock(&self.inner);
        if active
            .as_ref()
            .and_then(ActiveGraph::workspace_target)
            .as_ref()
            != Some(target)
        {
            return None;
        }
        let mut status = write_lock(&self.rebuild_status);
        if status.failed_target.as_ref() == Some(target) {
            status.consecutive_failures += 1;
        } else {
            status.consecutive_failures = 1;
            status.failed_target = Some(target.clone());
        }
        status.last_error = Some(error.to_string());
        status.failed_at = Some(SystemTime::now());
        Some(status.consecutive_failures)
    }

    /// Restore a failed target only if it remains installed and no newer
    /// watcher event has already occupied the pending slot.
    fn restore_current_rebuild_target(&self, target: WorkspaceGraphTarget) {
        let active = read_lock(&self.inner);
        if active
            .as_ref()
            .and_then(ActiveGraph::workspace_target)
            .as_ref()
            != Some(&target)
        {
            return;
        }
        let mut pending = write_lock(&self.pending_rebuild);
        if pending.is_none() {
            *pending = Some(target);
        }
    }

    /// Build and publish outside activation transactions (boot and lazy-watch
    /// paths). Activation uses the prepare/commit pair directly.
    pub fn build_workspace_graph(&self, root: &Path, revisions: Option<&[String]>) -> Result<()> {
        let prepared = self.prepare_workspace_graph(root, revisions)?;
        self.commit_workspace_graph(prepared);
        Ok(())
    }

    /// Root of the exact workspace graph currently installed. Watchers use
    /// this committed identity instead of a separately-published root slot.
    pub(crate) fn active_workspace_root(&self) -> Option<std::path::PathBuf> {
        read_lock(&self.inner)
            .as_ref()
            .and_then(|active| active.root.clone())
    }

    #[cfg(test)]
    pub(crate) fn active_workspace_revisions(&self) -> Option<Vec<String>> {
        read_lock(&self.inner)
            .as_ref()
            .and_then(|active| active.revs.clone())
    }

    pub fn bind_embedder(&self, embedder: Arc<dyn Embedder>) -> Result<()> {
        let mut guard = write_lock(&self.inner);
        let Some(active) = guard.as_mut() else {
            tracing::warn!("embedder loaded before any graph is active; binding deferred");
            return Ok(());
        };
        active.kg.set_embedder_native(embedder);
        Ok(())
    }

    pub fn schema(&self) -> Option<(u64, u64)> {
        let guard = read_lock(&self.inner);
        let active = guard.as_ref()?;
        let overview = compute_schema(active.kg.dir());
        Some((overview.node_count as u64, overview.edge_count as u64))
    }

    /// A one-line schema mini-map for the workspace activation message
    /// (the mcp-methods 0.3.46 activation-summary hook). Steers an agent's
    /// FIRST move toward the graph before it defaults to grep — the
    /// activation result is the one message read before any tool choice.
    /// Also carries a lazy-discovery escape hatch: a client that loads MCP
    /// tools lazily (Codex / code-mode / tool-search) can surface only
    /// grep/read_source on a broad first search and miss the always-registered
    /// graph tools, so the message tells it to search its registry for
    /// `cypher`/`graph_overview` rather than conclude the graph is unavailable
    /// (petekSuite report, 2026-07-08). The `instructions`-block `DISCOVERY_STEER`
    /// says the same thing, but a tool-call *result* is read more reliably than
    /// the handshake `instructions`. `None` when no graph is active.
    #[cfg(test)]
    pub fn activation_summary(&self) -> Option<String> {
        let guard = read_lock(&self.inner);
        let Some(active) = guard.as_ref() else {
            // Activation ran but no graph landed. Without a producer
            // that's expected, not silent: the framework swallows the
            // post-activate hook's error, so this summary is the only
            // channel that reaches the activation message.
            if self.workspace_graph_hooks.is_none() {
                return Some(NO_BUILDER_MSG.to_string());
            }
            return None;
        };
        activation_summary_for_active(active)
    }

    /// Snapshot the summary only when the installed graph is the exact plain
    /// target mcp-methods asked to reuse. A stale framework identity must not
    /// describe an unrelated graph that another lifecycle route installed.
    pub(crate) fn reusable_activation_summary(&self, root: &Path) -> Option<String> {
        let guard = read_lock(&self.inner);
        let active = guard.as_ref()?;
        if active.root.as_deref() != Some(root) || active.revs.is_some() {
            return None;
        }
        activation_summary_for_active(active)
    }

    pub(crate) fn no_builder_summary(&self) -> Option<String> {
        self.workspace_graph_hooks
            .is_none()
            .then(|| NO_BUILDER_MSG.to_string())
    }

    /// Whether the active graph has at least one node of the named
    /// type. Returns `false` when no graph is active. Backs the
    /// `graph_has_node_type:` predicate for skill `applies_when:`
    /// gating (0.9.31 / mcp-methods 0.3.36).
    pub fn has_node_type(&self, node_type: &str) -> bool {
        let guard = read_lock(&self.inner);
        guard
            .as_ref()
            .map(|active| active.kg.dir().has_node_type(node_type))
            .unwrap_or(false)
    }

    /// Whether the active graph's node-type metadata for `node_type`
    /// contains an entry for `prop_name`. Returns `false` when no
    /// graph is active or the type doesn't exist. Backs the
    /// `graph_has_property:` predicate for skill `applies_when:`
    /// gating.
    pub fn has_property(&self, node_type: &str, prop_name: &str) -> bool {
        let guard = read_lock(&self.inner);
        guard
            .as_ref()
            .map(|active| {
                active
                    .kg
                    .dir()
                    .get_node_type_metadata(node_type)
                    .map(|meta| meta.contains_key(prop_name))
                    .unwrap_or(false)
            })
            .unwrap_or(false)
    }

    fn with_active<F>(&self, f: F) -> String
    where
        F: FnOnce(&ActiveGraph) -> String,
    {
        let guard = read_lock(&self.inner);
        match guard.as_ref() {
            Some(active) => f(active),
            None => NO_GRAPH.to_string(),
        }
    }

    /// Borrow the active `KnowledgeGraph` for read-only inspection.
    /// Returns `None` when no graph is loaded — callers format their
    /// own "no graph active" message so the surrounding tool can give
    /// a tool-specific hint.
    pub fn with_kg<F, T>(&self, f: F) -> Option<T>
    where
        F: FnOnce(&kglite::api::KnowledgeGraph) -> T,
    {
        let guard = read_lock(&self.inner);
        guard.as_ref().map(|active| f(&active.kg))
    }

    /// Borrow the active graph and both path identities under one read lock.
    pub(crate) fn with_kg_context<F, T>(&self, f: F) -> Option<T>
    where
        F: FnOnce(&KnowledgeGraph, Option<&Path>, Option<&Path>) -> T,
    {
        let guard = read_lock(&self.inner);
        guard.as_ref().map(|active| {
            f(
                &active.kg,
                active.source_path.as_deref(),
                active.root.as_deref(),
            )
        })
    }

    /// Exclusive (write-locked) access to the active graph, for the
    /// write-enabled `cypher_query` path. The `RwLock` write-lock
    /// serializes mutations and excludes concurrent readers for the
    /// duration of the mutation — correct under any MCP dispatch model
    /// (serial or concurrent). Returns `None` when no graph is active.
    fn with_active_mut<F, T>(&self, f: F) -> Option<T>
    where
        F: FnOnce(&mut ActiveGraph) -> T,
    {
        let mut guard = write_lock(&self.inner);
        guard.as_mut().map(f)
    }

    /// Resolve a code-entity qualified name to its source location via
    /// `KnowledgeGraph::source_location`. Used by the `read_code_source`
    /// tool to bridge the qualified-name → file path lookup.
    pub(crate) fn source_lookup(
        &self,
        qualified_name: &str,
        node_type: Option<&str>,
    ) -> Result<crate::code_source::SourceLookup, String> {
        let guard = read_lock(&self.inner);
        let Some(active) = guard.as_ref() else {
            return Err(NO_GRAPH.to_string());
        };
        match active.kg.source_location(qualified_name, node_type) {
            kglite::api::code_entities::SourceLookup::Found(loc) => {
                let file_path = loc.file_path.ok_or_else(|| {
                    format!("graph.source({qualified_name:?}) returned no file_path")
                })?;
                let line_number = loc.line_number.unwrap_or(1).max(1) as usize;
                let end_line = loc.end_line.unwrap_or(loc.line_number.unwrap_or(1)).max(1) as usize;
                Ok(crate::code_source::SourceLookup {
                    file_path,
                    line_number,
                    end_line,
                })
            }
            kglite::api::code_entities::SourceLookup::Ambiguous(matches) => Err(format!(
                "ambiguous qualified_name {qualified_name:?}; matches: {matches:?}. \
                 Pass `node_type` to narrow."
            )),
            kglite::api::code_entities::SourceLookup::NotFound => Err(format!(
                "graph.source({qualified_name:?}) returned no match. \
                 Try passing `node_type` or using a different qualified name."
            )),
        }
    }

    /// Run a parameterised Cypher template against the active graph.
    /// Used by the YAML-declared `tools[].cypher` registration path
    /// (see [`crate::cypher_tools::register_cypher_tools`]).
    pub fn run_cypher_template(
        &self,
        template: &str,
        args: &serde_json::Map<String, serde_json::Value>,
        csv_http: Option<&crate::csv_http::CsvHttpConfig>,
    ) -> String {
        let guard = read_lock(&self.inner);
        let Some(active) = guard.as_ref() else {
            return NO_GRAPH.to_string();
        };
        let mut params = std::collections::HashMap::new();
        for (k, v) in args {
            params.insert(k.clone(), json_to_value(v));
        }
        // extensions.value_codecs apply to manifest cypher tools too (passed
        // through ExecuteOptions, not by rewriting the template text).
        match run_cypher_inner(&active.kg, template, params, self.value_codecs(), csv_http) {
            Ok(body) => body,
            Err(e) => cypher_tool_error(&e),
        }
    }
}

/// Convert a `serde_json::Value` into a Cypher param `Value`. Mirrors
/// the Python boundary's `py_value_to_value` for the JSON subset.
///
/// As of the 2026-05-25 binding-framework lift this is a 1-line
/// delegate to `kglite::api::param::json_value_to_kglite_value`,
/// which any REST/gRPC binding can call directly.
fn json_to_value(v: &serde_json::Value) -> Value {
    kglite::api::param::json_value_to_kglite_value(v)
}

/// Attach the tool-level `Cypher error:` prefix to a surfaced error — unless
/// the message already self-identifies as a Cypher error (`KgError`'s Display
/// emits `Cypher execution error: …` / `Cypher syntax error: …`). Prefixing
/// those again stutters (`Cypher error: Cypher execution error: …`); this keeps
/// every surfaced Cypher error reading once, whichever layer produced it.
fn cypher_tool_error(e: &str) -> String {
    if e.starts_with("Cypher ") {
        e.to_string()
    } else {
        format!("Cypher error: {e}")
    }
}

/// Run a Cypher query against the given KnowledgeGraph snapshot. Picks
/// between read and write paths based on `is_mutation_query`; on success
/// returns the rendered tool body (CSV when `FORMAT CSV` is in the
/// query, inline 15-row preview otherwise).
fn run_cypher_inner(
    kg: &KnowledgeGraph,
    query: &str,
    params: std::collections::HashMap<String, Value>,
    value_codecs: Option<&[ValueCodec]>,
    csv_http: Option<&crate::csv_http::CsvHttpConfig>,
) -> Result<String, String> {
    // Phase E.3 — delegate to kglite::api::session for the canonical
    // pipeline (parse → validate → rewrite_text_score (+embed) →
    // optimize → mutation-gate → execute). The mcp-server still
    // owns mutation policy (reject) + CSV output formatting.

    // MCP rejects mutations regardless of read-only graph mode:
    // mutation Cypher through the MCP surface is a deliberate policy
    // restriction (agents should use the CLI for graph edits). Pre-
    // parse to catch this cleanly before session::execute_read errors.
    let (pre_parsed, is_mutation) =
        kglite::api::cypher::parse_with_mutation_check(query).map_err(|e| e.to_string())?;
    if is_mutation {
        return Err(
            "mutation Cypher (CREATE/SET/DELETE/REMOVE/MERGE) is not allowed through \
             the MCP cypher_query tool. Use the kglite CLI for graph edits."
                .to_string(),
        );
    }
    let output_csv = pre_parsed.output_format == kglite::api::cypher::OutputFormat::Csv;

    // Eager rows — MCP output formatters (CSV / 15-row preview)
    // need materialized results; no lazy materializer at this layer.
    // Embedder is plumbed when the active graph has one wired (for
    // `text_score()` queries); otherwise None.
    let mut opts = kglite::api::session::ExecuteOptions::eager(&params);
    opts.embedder = kg.embedder().cloned();
    // extensions.value_codecs: decode query-side literals bound to a codec'd
    // property + encode result columns. None/empty → no transform.
    opts.value_codecs = value_codecs;
    // `KgError`'s Display already prefixes `Cypher execution error: …`; take it
    // verbatim (a second `Cypher execution error:` here is the reported stutter).
    let outcome =
        kglite::api::session::execute_read(kg.dir(), query, &opts).map_err(|e| e.to_string())?;
    render_cypher_output(&outcome.result, output_csv, csv_http)
}

/// Render a `CypherResult` for the MCP text surface: CSV (inline or via the
/// csv_http server) or a 15-row inline preview. Shared by the read path and
/// the write path so both format results identically.
fn render_cypher_output(
    result: &cypher::CypherResult,
    output_csv: bool,
    csv_http: Option<&crate::csv_http::CsvHttpConfig>,
) -> Result<String, String> {
    if output_csv {
        let csv = result.to_csv();
        if let Some(cfg) = csv_http {
            match crate::csv_http::write_csv(cfg, &csv) {
                Ok(name) => {
                    let url = cfg.url_for(&name);
                    // 0.9.19 fix: count rows from the CSV body, not from
                    // `result.rows.len()`. The planner's lazy_eligible
                    // pass leaves `rows` empty for simple
                    // MATCH-RETURN-LIMIT queries and materialises through
                    // the lazy descriptor (or streaming pipeline) — the
                    // CSV is correct but `rows.len()` reads 0 and the
                    // operator-facing status says "0 row(s) written".
                    // Counting newlines in the CSV agrees with what the
                    // file actually contains.
                    let row_count = count_csv_rows(&csv);
                    Ok(format!(
                        "FORMAT CSV: {row_count} row(s) written to {url}\n\
                         Fetch with: curl {url}"
                    ))
                }
                Err(e) => {
                    tracing::warn!(error = %e, "csv_http write_csv failed; falling back to inline");
                    Ok(csv)
                }
            }
        } else {
            Ok(csv)
        }
    } else {
        Ok(format_cypher_inline(result))
    }
}

/// Render a CypherResult as an inline 15-row preview (header + repr per
/// row). Matches the format the pre-0.9.18 Python shim produced via
/// `format_cypher_result`.
fn format_cypher_inline(result: &cypher::CypherResult) -> String {
    let len = result.rows.len();
    if len == 0 {
        return "No results.".to_string();
    }
    let header = if len > 15 {
        format!("{len} row(s) (showing first 15):\n")
    } else {
        format!("{len} row(s):\n")
    };
    let mut out = header;
    out.push_str(&result.columns.join("\t"));
    out.push('\n');
    for row in result.rows.iter().take(15) {
        for (i, val) in row.iter().enumerate() {
            if i > 0 {
                out.push('\t');
            }
            push_value_repr(&mut out, val);
        }
        out.push('\n');
    }
    out
}

/// Count data rows in a CSV string, defined as (newline-terminated lines) - 1
/// for the header. Trailing newlines after the last row don't add to the
/// count. Handles the edge cases: empty string → 0, header-only → 0,
/// header + N rows → N. Quoted newlines inside cells aren't recognised
/// here — kglite's `csv_value` doesn't emit Value variants that contain
/// embedded newlines, so a plain `lines()` count agrees with row count.
fn count_csv_rows(csv: &str) -> usize {
    let line_count = csv.lines().count();
    line_count.saturating_sub(1)
}

fn push_value_repr(out: &mut String, val: &Value) {
    use std::fmt::Write;
    match val {
        Value::Null => out.push_str("null"),
        Value::String(s) => {
            let _ = write!(out, "{s:?}");
        }
        Value::Int64(n) => {
            let _ = write!(out, "{n}");
        }
        Value::Float64(f) => {
            let _ = write!(out, "{f}");
        }
        Value::Boolean(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::UniqueId(u) => {
            let _ = write!(out, "{u}");
        }
        Value::DateTime(d) => out.push_str(&d.format("%Y-%m-%d").to_string()),
        Value::Timestamp(dt) => out.push_str(&dt.format("%Y-%m-%dT%H:%M:%S").to_string()),
        Value::Point { lat, lon } => {
            let _ = write!(out, "POINT({lon} {lat})");
        }
        Value::Duration {
            months,
            days,
            seconds,
        } => {
            let _ = write!(out, "duration(M={months}, D={days}, S={seconds})");
        }
        Value::NodeRef(idx) => {
            let _ = write!(out, "node[{idx}]");
        }
        // Phase A.1 / C5 — collection / graph-entity variants. Render
        // as compact JSON for the MCP text surface; the structured
        // form is already what agents consume via `to_dicts()` /
        // `to_list()`. Falls back to `?` on serialisation failure
        // (shouldn't happen — these all derive Serialize).
        Value::List(_)
        | Value::Map(_)
        | Value::Node(_)
        | Value::Relationship(_)
        | Value::Path(_) => {
            let _ = write!(
                out,
                "{}",
                serde_json::to_string(val).unwrap_or_else(|_| "?".to_string())
            );
        }
    }
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
struct ReadCypherArgs {
    /// Cypher query string. Append `FORMAT CSV` for CSV-encoded output.
    pub query: String,
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
struct CypherArgs {
    /// Cypher query string. Append `FORMAT CSV` for CSV-encoded output.
    pub query: String,
    /// Role-scoped write whitelist (write-enabled servers only). When set, a
    /// `CREATE`/`SET` whose node type is not in this list is rejected — so an
    /// agent can plan in its own types (`["Plan","Task"]`) without touching
    /// research-owned ones. Ignored on read-only servers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub write_scope: Option<Vec<String>>,
    /// Freshness provenance for this write (write-enabled servers only): the git
    /// commit SHA the agent is working against, stamped as `updated_at`'s
    /// companion on writes to `auto_timestamp` node/edge types — so a node can
    /// record "describes the code as of sha X". Optional; ignored on reads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_sha: Option<String>,
    /// Optional actor id stamped alongside `git_sha` (e.g. the agent/session
    /// name). Same gating as `git_sha`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modified_by: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
enum DetailSelection {
    Enabled(bool),
    Topics(Vec<String>),
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
struct OverviewArgs {
    /// Drill into specific node types (e.g. `["Person", "Document"]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub types: Option<Vec<String>>,
    /// `true` for all connection types; or `["CALLS"]` for a deep-dive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connections: Option<DetailSelection>,
    /// `true` for the Cypher language reference; or `["MATCH","WHERE"]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cypher: Option<DetailSelection>,
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
struct SaveGraphArgs {}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
struct LoadGraphArgs {
    /// Path to a `.kgl` file (or disk-graph directory) to load as the new
    /// active graph, replacing the current one. Unsaved in-memory changes to
    /// the previous graph are discarded — call `save_graph` first to keep them.
    pub path: String,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
enum StorageArg {
    Memory,
    Mapped,
    Disk,
}

impl StorageArg {
    fn mode(&self) -> StorageMode {
        match self {
            Self::Memory => StorageMode::Memory,
            Self::Mapped => StorageMode::Mapped,
            Self::Disk => StorageMode::Disk,
        }
    }
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
struct CreateGraphArgs {
    /// Path the new empty graph is bound to (its `save_graph` target).
    pub path: String,
    /// Storage mode: `memory` (default), `mapped`, or `disk`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage: Option<StorageArg>,
}

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
struct SaveGraphAsArgs {
    /// Path to save the active graph to; also becomes the new `save_graph`
    /// target.
    pub path: String,
}

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

pub fn register(
    server: &mut McpServer,
    state: GraphState,
    builtins: Builtins,
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
             navigate the codebase structure). Pass write_scope=[...] to restrict mutations to \
             those node types. Mutations are in-memory; call save_graph to persist. Append \
             FORMAT CSV to export results."
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
            s.ensure_workspace_graph_fresh();
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
            s.ensure_workspace_graph_fresh();
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
            if cleanup_temp
                && args.types.is_none()
                && args.connections.is_none()
                && args.cypher.is_none()
            {
                if let Some(dir) = temp_dir.as_deref() {
                    wipe_temp_dir(dir);
                }
            }
            s.ensure_workspace_graph_fresh();
            let body = s.with_active(|g| run_overview(g, &args));
            s.with_rebuild_warning(body)
        },
    );
    if builtins.save_graph {
        let s = state.clone();
        server.register_typed_tool::<SaveGraphArgs, _>(
            "save_graph",
            "Persist the active graph to its source .kgl file (single-graph mode only).",
            move |_| {
                s.ensure_workspace_graph_fresh();
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
                s.ensure_workspace_graph_fresh();
                match s.save_as(Path::new(&args.path)) {
                    Ok(msg) => msg,
                    Err(e) => e,
                }
            },
        );
    }
}

fn wipe_temp_dir(dir: &std::path::Path) {
    if !dir.is_dir() {
        tracing::debug!(dir = %dir.display(), "temp_cleanup: directory does not exist; nothing to wipe");
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            tracing::debug!(error = %e, dir = %dir.display(), "temp_cleanup: read_dir failed");
            return;
        }
    };
    let mut wiped = 0usize;
    for entry in entries.flatten() {
        let path = entry.path();
        let res = if path.is_dir() {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
        match res {
            Ok(()) => wiped += 1,
            Err(e) => {
                tracing::debug!(path = %path.display(), error = %e, "temp_cleanup: remove failed");
            }
        }
    }
    if wiped > 0 {
        tracing::info!(count = wiped, dir = %dir.display(), "temp_cleanup: wiped entries");
    }
}

fn run_cypher_tool(
    graph: &ActiveGraph,
    query: &str,
    value_codecs: Option<&[ValueCodec]>,
    csv_http: Option<&crate::csv_http::CsvHttpConfig>,
) -> String {
    match run_cypher_inner(
        &graph.kg,
        query,
        std::collections::HashMap::new(),
        value_codecs,
        csv_http,
    ) {
        // Compact identity footer so a query result self-identifies its
        // graph (agents often go straight to cypher_query without a prior
        // graph_overview, where a stale active root would otherwise hide).
        Ok(s) => format!("{s}{}", graph.identity_footer()),
        Err(e) => cypher_tool_error(&e),
    }
}

/// Write-enabled Cypher path (only reachable when the server is `--writable`).
/// A read query delegates to the read path; a mutation routes through
/// `execute_mut` against a `&mut DirGraph` obtained under the active graph's
/// write-lock, with an optional role-scoped `write_scope`. Mutations land on
/// the live active graph (in-memory) so subsequent queries observe them;
/// persistence is the separate `save_graph` step.
#[allow(clippy::too_many_arguments)]
fn run_cypher_write(
    active: &mut ActiveGraph,
    query: &str,
    write_scope: Option<&[String]>,
    git_sha: Option<&str>,
    modified_by: Option<&str>,
    value_codecs: Option<&[ValueCodec]>,
    csv_http: Option<&crate::csv_http::CsvHttpConfig>,
) -> Result<String, String> {
    let (pre_parsed, is_mutation) =
        kglite::api::cypher::parse_with_mutation_check(query).map_err(|e| e.to_string())?;
    if !is_mutation {
        // Read on a writable server — same path as the read-only tool.
        return run_cypher_inner(
            &active.kg,
            query,
            std::collections::HashMap::new(),
            value_codecs,
            csv_http,
        );
    }
    let output_csv = pre_parsed.output_format == kglite::api::cypher::OutputFormat::Csv;
    let params = std::collections::HashMap::new();
    let scope: Option<std::collections::HashSet<String>> =
        write_scope.map(|v| v.iter().cloned().collect());
    // Snapshot the embedder Arc before the mutable borrow of `kg`.
    let embedder = active.kg.embedder().cloned();
    let dir = kglite::api::make_dir_graph_mut(active.kg.dir_mut());
    let mut opts = kglite::api::session::ExecuteOptions::eager(&params);
    opts.embedder = embedder;
    opts.value_codecs = value_codecs;
    opts.write_scope = scope.as_ref();
    opts.git_sha = git_sha;
    opts.modified_by = modified_by;
    // `KgError`'s Display already prefixes `Cypher execution error: …` — pass it
    // through verbatim rather than re-prefixing (which produced the triple wrap).
    let outcome =
        kglite::api::session::execute_mut(dir, query, &opts).map_err(|e| e.to_string())?;
    // A mutation with no RETURN yields no rows — acknowledge with a write
    // summary (nodes/edges/props changed) instead of the bare "No results."
    // that a *read* matching nothing returns, so an agent can tell a
    // successful write apart from a no-op match. (A mutation that does RETURN
    // falls through to the normal row rendering.)
    if !output_csv && outcome.result.rows.is_empty() {
        return Ok(format_mutation_ack(&outcome.result));
    }
    render_cypher_output(&outcome.result, output_csv, csv_http)
}

/// One-line acknowledgement of a write that returned no rows, summarising the
/// mutation stats (e.g. `OK: 1 node(s) created, 1 relationship(s) created.`).
fn format_mutation_ack(result: &cypher::CypherResult) -> String {
    let Some(st) = result.stats.as_ref() else {
        return "OK (write applied).".to_string();
    };
    let mut parts: Vec<String> = Vec::new();
    let mut push = |n: usize, label: &str| {
        if n > 0 {
            parts.push(format!("{n} {label}"));
        }
    };
    push(st.nodes_created, "node(s) created");
    push(st.relationships_created, "relationship(s) created");
    push(st.properties_set, "property(ies) set");
    push(st.nodes_deleted, "node(s) deleted");
    push(st.relationships_deleted, "relationship(s) deleted");
    push(st.properties_removed, "property(ies) removed");
    // Stamp the running engine version on every write ack. A long-running
    // server pins its engine; after a venv upgrade the *running* binary may lag,
    // so writes silently stop honouring a newer feature (e.g. auto_timestamp
    // stamping) until restart. Surfacing the version makes that visible.
    let engine = env!("CARGO_PKG_VERSION");
    if parts.is_empty() {
        format!("OK (no changes). [engine {engine}]")
    } else {
        format!("OK: {}. [engine {engine}]", parts.join(", "))
    }
}

fn run_overview(graph: &ActiveGraph, args: &OverviewArgs) -> String {
    let conn = parse_connection_detail(args.connections.as_ref());
    let cy = parse_cypher_detail(args.cypher.as_ref());
    let fluent = FluentDetail::Off;
    match compute_description(
        graph.kg.dir(),
        args.types.as_deref(),
        &conn,
        &cy,
        &fluent,
        None,
        None,
        None,
    ) {
        // Prepend a server-level identity header so the active root + build
        // time are the first thing an agent reads — staleness after a root
        // swap is visible before any structural claim is trusted.
        Ok(s) => format!("<active_graph{}/>\n{s}", graph.identity_attrs()),
        Err(e) => format!("graph_overview error: {e}"),
    }
}

fn parse_connection_detail(v: Option<&DetailSelection>) -> ConnectionDetail {
    match v {
        None | Some(DetailSelection::Enabled(false)) => ConnectionDetail::Off,
        Some(DetailSelection::Enabled(true)) => ConnectionDetail::Overview,
        Some(DetailSelection::Topics(items)) => {
            let names = items.clone();
            if names.is_empty() {
                ConnectionDetail::Overview
            } else {
                ConnectionDetail::Topics(names)
            }
        }
    }
}

fn parse_cypher_detail(v: Option<&DetailSelection>) -> CypherDetail {
    match v {
        None | Some(DetailSelection::Enabled(false)) => CypherDetail::Off,
        Some(DetailSelection::Enabled(true)) => CypherDetail::Overview,
        Some(DetailSelection::Topics(items)) => {
            let names = items.clone();
            if names.is_empty() {
                CypherDetail::Overview
            } else {
                CypherDetail::Topics(names)
            }
        }
    }
}

fn run_save(graph: &mut ActiveGraph) -> String {
    let Some(path) = graph.source_path.as_ref() else {
        return "save_graph requires --graph mode (no source path bound).".to_string();
    };
    let path_str = path.to_string_lossy().into_owned();
    // `kglite::api::io::save_graph` dispatches on storage mode (mirrors
    // `KnowledgeGraph::save` at `src/graph/pyapi/kg_core.rs`):
    //   - disk-backed → `save_disk(path)` (the folder IS the graph)
    //   - in-memory  → `prepare_save` → `enable_columnar` → `write_kgl`
    // The pre-0.9.45 inline `save_disk` call errored "save_disk requires
    // disk mode" for in-memory `.kgl` graphs — see CHANGELOG [0.9.45].
    //
    // Save through the active graph's OWN Arc (`dir_mut`), under the
    // caller's write lock. The previous `graph.kg.dir().clone()` bumped
    // the refcount to ≥2, so `prepare_save`'s `Arc::make_mut` deep-copied
    // the entire graph on EVERY save — and the columnar consolidation
    // landed in the discarded clone, so the next save paid it all again.
    match kglite::api::io::save_graph(graph.kg.dir_mut(), &path_str) {
        Ok(()) => {
            // `compute_schema` only needs `&DirGraph` — no second make_mut.
            let overview = compute_schema(graph.kg.dir());
            format!(
                "Saved {path_str} ({} nodes, {} edges).",
                overview.node_count, overview.edge_count
            )
        }
        Err(e) => format!("save_graph error: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kglite::api::storage::{new_dir_graph_in_mode, StorageMode};

    #[test]
    fn workspace_graph_hooks_unify_builds_and_own_relevance() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let build_calls = Arc::new(AtomicUsize::new(0));
        let calls = build_calls.clone();
        let hooks = WorkspaceGraphHooks {
            build: Box::new(move |request| {
                calls.fetch_add(1, Ordering::SeqCst);
                new_dir_graph_in_mode(StorageMode::Memory, None)
                    .map(Arc::new)
                    .map(|graph| match request.revisions() {
                        Some(revisions) => {
                            WorkspaceGraphResult::with_revisions(graph, revisions.to_vec())
                        }
                        None => WorkspaceGraphResult::new(graph),
                    })
                    .map_err(|e| e.to_string())
            }),
            is_relevant: Box::new(|change| change.path().extension().is_some_and(|e| e == "zig")),
        };
        let gs = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
            .with_workspace_graph(Some(Arc::new(hooks)));

        // Watch predicate comes from the hook, not language_for_path
        // (in-tree has no zig parser; hook says only zig is code).
        assert!(gs.is_graph_relevant(Path::new("a.zig")));
        assert!(!gs.is_graph_relevant(Path::new("a.rs")));

        // build goes through the hook and swaps in the returned graph.
        gs.build_workspace_graph(Path::new("/nonexistent-is-fine-for-hook"), None)
            .expect("hook build");
        assert_eq!(build_calls.load(Ordering::SeqCst), 1);
        assert!(gs.schema().is_some(), "hook-built graph became active");

        // revs path records the hook's canonical rev list.
        gs.build_workspace_graph(
            Path::new("/nonexistent-is-fine-for-hook"),
            Some(&["a".into(), "b".into()]),
        )
        .expect("revision-set hook build");
    }

    #[test]
    fn without_hooks_nothing_is_relevant_and_builds_refuse() {
        // The in-tree builder is gone: a hook-less state can't rebuild, so
        // no path is graph-relevant and build requests refuse with a
        // pointer at codingest.
        let gs = GraphState::new(Some(WorkspaceGraphMode::Workspace));
        assert!(!gs.is_graph_relevant(Path::new("a.rs")));
        assert!(!gs.is_graph_relevant(Path::new("README.md")));
        let err = gs
            .build_workspace_graph(Path::new("/tmp"), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("codingest"), "refusal names codingest: {err}");
        let err = gs
            .build_workspace_graph(Path::new("/tmp"), Some(&["r1".into()]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("codingest"), "refusal names codingest: {err}");
        // The producer, not KGLite, chooses that markdown is relevant in
        // clone-backed workspace mode.
        let gs_docs = GraphState::new(Some(WorkspaceGraphMode::Workspace))
            .with_workspace_graph(Some(test_hooks()));
        assert!(gs_docs.is_graph_relevant(Path::new("README.md")));
    }

    fn fresh_active() -> ActiveGraph {
        let dir = new_dir_graph_in_mode(StorageMode::Memory, None).expect("create graph");
        ActiveGraph {
            kg: KnowledgeGraph::from_arc(Arc::new(dir)),
            source_path: None,
            writer_lease: None,
            root: None,
            revs: None,
            built_at: SystemTime::now(),
            generation: 0,
        }
    }

    /// Stub builder hooks: the real builder lives in codingest, so these
    /// tests exercise GraphState's machinery (activation summaries, rev
    /// recording, rebuild backoff) against a minimal hand-built
    /// code-schema graph. Fails on a missing dir like a real builder;
    /// The unified build closure dedups labels and stamps a `revs` list prop.
    fn test_hooks() -> Arc<WorkspaceGraphHooks> {
        fn mini_graph(revs: Option<&[String]>) -> Result<Arc<kglite::api::DirGraph>, String> {
            let mut dir =
                new_dir_graph_in_mode(StorageMode::Memory, None).map_err(|e| e.to_string())?;
            let params = std::collections::HashMap::new();
            let opts = kglite::api::session::ExecuteOptions::eager(&params);
            kglite::api::session::execute_mut(
                &mut dir,
                "CREATE (f:File {id:'m.py'})-[:DEFINES]->\
                 (g:Function {id:'m.hub', name:'hub', file_path:'m.py', line:1})",
                &opts,
            )
            .map_err(|e| e.to_string())?;
            if let Some(revs) = revs {
                let list = revs
                    .iter()
                    .map(|r| format!("'{r}'"))
                    .collect::<Vec<_>>()
                    .join(", ");
                kglite::api::session::execute_mut(
                    &mut dir,
                    &format!("MATCH (n:Function) SET n.revs = [{list}]"),
                    &opts,
                )
                .map_err(|e| e.to_string())?;
            }
            Ok(Arc::new(dir))
        }
        Arc::new(WorkspaceGraphHooks {
            build: Box::new(|request| {
                if !request.root().is_dir() {
                    return Err(format!("no such directory: {}", request.root().display()));
                }
                let Some(revisions) = request.revisions() else {
                    return mini_graph(None).map(WorkspaceGraphResult::new);
                };
                let mut canonical = Vec::new();
                for revision in revisions {
                    if !canonical.contains(revision) {
                        canonical.push(revision.clone());
                    }
                }
                mini_graph(Some(&canonical))
                    .map(|graph| WorkspaceGraphResult::with_revisions(graph, canonical))
            }),
            is_relevant: Box::new(|change| {
                change
                    .path()
                    .extension()
                    .is_some_and(|e| e == "py" || e == "rs")
                    || (change.mode() == WorkspaceGraphMode::Workspace
                        && change
                            .path()
                            .extension()
                            .is_some_and(|e| e.eq_ignore_ascii_case("md")))
            }),
        })
    }

    #[test]
    fn save_does_not_deep_copy_the_active_graph() {
        // `run_save` / `save_as` must save through the active graph's OWN
        // Arc so `prepare_save`'s `Arc::make_mut` sees refcount 1. The old
        // `kg.dir().clone()` route deep-copied the entire graph on every
        // save (and threw the columnar consolidation away with the clone).
        // Pin the fix by asserting the DirGraph allocation is pointer-
        // identical across the save.
        let mut active = fresh_active();
        {
            // Put something in the graph so the save isn't trivially empty.
            let dir = kglite::api::make_dir_graph_mut(active.kg.dir_mut());
            let opts_params = std::collections::HashMap::new();
            let opts = kglite::api::session::ExecuteOptions::eager(&opts_params);
            kglite::api::session::execute_mut(
                dir,
                "CREATE (a:Thing {id: 1, name: 'a'})-[:REL]->(b:Thing {id: 2, name: 'b'})",
                &opts,
            )
            .expect("seed mutation");
        }
        let path = std::env::temp_dir().join(format!(
            "kglite-mcp-save-noclone-{}.kgl",
            std::process::id()
        ));
        active.source_path = Some(path.clone());

        let before = Arc::as_ptr(active.kg.dir());
        let msg = run_save(&mut active);
        assert!(msg.starts_with("Saved"), "save must succeed: {msg}");
        assert_eq!(
            Arc::as_ptr(active.kg.dir()),
            before,
            "save must not deep-copy the active graph (refcount must be 1 \
             at prepare_save's Arc::make_mut)"
        );

        // save_as: same invariant, and the save target must rebind.
        let path2 = std::env::temp_dir().join(format!(
            "kglite-mcp-saveas-noclone-{}.kgl",
            std::process::id()
        ));
        let state = GraphState::new(None);
        *write_lock(&state.inner) = Some(active);
        let msg = state.save_as(&path2).expect("save_as succeeds");
        assert!(msg.starts_with("Saved"), "{msg}");
        {
            let guard = read_lock(&state.inner);
            let active = guard.as_ref().expect("still active");
            assert_eq!(
                Arc::as_ptr(active.kg.dir()),
                before,
                "save_as must not deep-copy the active graph"
            );
            assert_eq!(active.source_path.as_deref(), Some(path2.as_path()));
        }

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&path2);
    }

    #[test]
    fn failed_rebuild_restores_marker_then_backs_off_after_cap() {
        let state = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
            .with_workspace_graph(Some(test_hooks()));
        let workspace = tempfile::tempdir().expect("workspace tempdir");
        let root = workspace.path().to_path_buf();
        std::fs::write(root.join("m.py"), "def stale():\n    return 1\n").unwrap();
        state
            .build_workspace_graph(&root, None)
            .expect("install the graph that later becomes stale");
        let target = state.active_workspace_target().expect("workspace target");
        std::fs::remove_dir_all(&root).expect("make the current target fail rebuilding");

        // A failed rebuild must restore the dirty marker (so the next tool
        // call retries) and record the error — up to the hot-fail cap.
        state.tag_workspace_graph_dirty();
        for failure in 1..MAX_CONSECUTIVE_REBUILD_FAILURES {
            state.ensure_workspace_graph_fresh();
            assert_eq!(
                read_lock(&state.pending_rebuild).as_ref(),
                Some(&target),
                "failure {failure} must restore the marker for a retry"
            );
            let note = state.rebuild_error_note().expect("error recorded");
            assert!(note.contains("STALE"), "note flags staleness: {note}");
        }

        // Failure #cap: stop retrying (marker not restored) — the stale
        // graph keeps being served with the error surfaced.
        state.ensure_workspace_graph_fresh();
        assert!(
            read_lock(&state.pending_rebuild).is_none(),
            "after {MAX_CONSECUTIVE_REBUILD_FAILURES} consecutive failures \
             the marker must NOT be restored (no hot-fail loop)"
        );
        let note = state.rebuild_error_note().expect("error still surfaced");
        assert!(
            note.contains(&format!("{MAX_CONSECUTIVE_REBUILD_FAILURES} consecutive")),
            "note reports the failure count: {note}"
        );
        // With no marker, further tool calls are no-ops (no retry storm).
        state.ensure_workspace_graph_fresh();

        // A fresh FS event re-tags → exactly one more retry; still failing,
        // so the marker again stays cleared.
        state.tag_workspace_graph_dirty();
        state.ensure_workspace_graph_fresh();
        assert!(read_lock(&state.pending_rebuild).is_none());
        assert!(state.rebuild_error_note().is_some());

        // A successful rebuild clears the error and resets the counter.
        std::fs::create_dir_all(&root).expect("restore workspace directory");
        std::fs::write(root.join("m.py"), "def ok():\n    return 1\n").unwrap();
        state.tag_workspace_graph_dirty();
        state.ensure_workspace_graph_fresh();
        assert!(
            state.rebuild_error_note().is_none(),
            "successful rebuild must clear the recorded failure"
        );
        assert!(read_lock(&state.pending_rebuild).is_none());
    }

    #[test]
    fn workspace_rebuild_preserves_multi_revision_target() {
        let workspace = tempfile::tempdir().expect("workspace tempdir");
        std::fs::write(
            workspace.path().join("m.py"),
            "def changed():\n    return 1\n",
        )
        .unwrap();
        let state = GraphState::new(Some(WorkspaceGraphMode::Workspace))
            .with_workspace_graph(Some(test_hooks()));
        let revisions = vec!["base".to_string(), "head".to_string()];
        state
            .build_workspace_graph(workspace.path(), Some(&revisions))
            .expect("initial revision-set build");

        state.tag_workspace_graph_dirty();
        state.ensure_workspace_graph_fresh();

        assert_eq!(state.active_workspace_revisions(), Some(revisions));
        assert!(read_lock(&state.pending_rebuild).is_none());
    }

    #[test]
    fn workspace_rebuild_cannot_overwrite_newer_activation() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Barrier,
        };

        let calls = Arc::new(AtomicUsize::new(0));
        let rebuild_started = Arc::new(Barrier::new(2));
        let release_rebuild = Arc::new(Barrier::new(2));
        let hooks = Arc::new(WorkspaceGraphHooks {
            build: Box::new({
                let calls = calls.clone();
                let rebuild_started = rebuild_started.clone();
                let release_rebuild = release_rebuild.clone();
                move |request| {
                    let call = calls.fetch_add(1, Ordering::SeqCst) + 1;
                    if call == 2 {
                        rebuild_started.wait();
                        release_rebuild.wait();
                    }
                    let graph = new_dir_graph_in_mode(StorageMode::Memory, None)
                        .map(Arc::new)
                        .map_err(|e| e.to_string())?;
                    Ok(match request.revisions() {
                        Some(revisions) => {
                            WorkspaceGraphResult::with_revisions(graph, revisions.to_vec())
                        }
                        None => WorkspaceGraphResult::new(graph),
                    })
                }
            }),
            is_relevant: Box::new(|_| true),
        });
        let state = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
            .with_workspace_graph(Some(hooks));
        let root_a = Path::new("/workspace/a");
        let root_b = Path::new("/workspace/b");
        state
            .build_workspace_graph(root_a, None)
            .expect("install initial graph");
        state.tag_workspace_graph_dirty();

        let rebuilding = state.clone();
        let rebuild_thread = std::thread::spawn(move || rebuilding.ensure_workspace_graph_fresh());
        rebuild_started.wait();

        let newer = state
            .prepare_workspace_graph(root_b, None)
            .expect("prepare newer activation");
        state.commit_workspace_graph(newer);
        release_rebuild.wait();
        rebuild_thread.join().expect("rebuild thread");

        assert_eq!(state.active_workspace_root().as_deref(), Some(root_b));
        assert!(read_lock(&state.pending_rebuild).is_none());
        assert!(state.rebuild_error_note().is_none());
    }

    #[test]
    fn activation_summary_reports_node_types_or_none() {
        let gs = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
            .with_workspace_graph(Some(test_hooks()));
        assert!(
            gs.activation_summary().is_none(),
            "no active graph → terse activation (no mini-map)"
        );
        let dir = std::env::temp_dir().join(format!("kgl_actsum_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(
            dir.join("m.py"),
            "def hub():\n    return leaf()\n\ndef leaf():\n    return 1\n\nclass Bar:\n    pass\n",
        )
        .unwrap();
        gs.build_workspace_graph(&dir, None)
            .expect("build workspace graph");
        let summary = gs
            .activation_summary()
            .expect("summary present once a graph is active");
        assert!(summary.contains("Function"), "names node types: {summary}");
        assert!(
            summary.contains("graph_overview()"),
            "steers to the graph: {summary}"
        );
        assert!(
            summary.contains("search your tool registry"),
            "carries the lazy-discovery escape hatch: {summary}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn revision_build_swaps_slot_and_records_revisions() {
        let dir = std::env::temp_dir().join(format!("kgl_slot_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let (s1, s2) = ("r1".to_string(), "r2".to_string());
        let gs = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
            .with_workspace_graph(Some(test_hooks()));
        let revs = vec![s1.clone(), s2.clone()];
        gs.build_workspace_graph(&dir, Some(&revs))
            .expect("multi-rev build");
        // The slot is active with nodes.
        let (nodes, _edges) = gs.schema().expect("schema after multi-rev build");
        assert!(nodes > 0, "multi-rev graph should have nodes");
        // `bar` exists only in the second rev → its `revs` list is a subset.
        // `foo` exists in both. Assert the rev list props landed on the merged
        // graph (the B.2b merge stamps `revs` on every node).
        let has_revs_prop = gs.has_property("Function", "revs");
        assert!(
            has_revs_prop,
            "merged multi-rev Function nodes should carry a `revs` list prop"
        );
        // The active slot records the resolved rev-set for the identity surfaces.
        let attrs = gs.with_active(|a| a.identity_attrs());
        assert!(
            attrs.contains(&format!("revs=\"{},{}\"", s1, s2)),
            "identity header should name the loaded revs; got: {attrs}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn activation_summary_teaches_rev_scoping_for_multi_rev() {
        let dir = std::env::temp_dir().join(format!("kgl_actsumrev_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let (s1, s2) = ("r1".to_string(), "r2".to_string());
        let gs = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
            .with_workspace_graph(Some(test_hooks()));
        gs.build_workspace_graph(&dir, Some(&[s1.clone(), s2.clone()]))
            .expect("multi-rev build");
        let summary = gs.activation_summary().expect("summary present");
        // Still carries the base mini-map + discovery hatch.
        assert!(summary.contains("Function"), "names node types: {summary}");
        // Multi-rev steer: names the revs, warns about over-count, teaches the
        // scoping idiom + rev_diff (matching the describe() provenance text).
        assert!(
            summary.contains("Multi-rev graph spanning 2"),
            "names the rev span: {summary}"
        );
        assert!(
            summary.contains("IN n.revs"),
            "teaches the `WHERE '<rev>' IN n.revs` scoping idiom: {summary}"
        );
        assert!(
            summary.contains("rev_diff"),
            "points at CALL rev_diff for deltas: {summary}"
        );
        // The newest rev (HEAD-equivalent, last in the list) is surfaced for
        // head-only scoping.
        assert!(
            summary.contains(&format!("'{s2}' IN n.revs")),
            "surfaces the newest rev for head-only scoping: {summary}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn single_rev_build_carries_no_revs_attr_or_steer() {
        // The plain build path leaves `revs = None`, so neither the header attr
        // nor the multi-rev steer appears (no regression for single-rev graphs).
        let gs = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
            .with_workspace_graph(Some(test_hooks()));
        let dir = std::env::temp_dir().join(format!("kgl_single_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("m.py"), "def foo():\n    return 1\n").unwrap();
        gs.build_workspace_graph(&dir, None)
            .expect("single-rev build");
        let attrs = gs.with_active(|a| a.identity_attrs());
        assert!(
            !attrs.contains("revs="),
            "no revs attr for single-rev: {attrs}"
        );
        let summary = gs.activation_summary().expect("summary");
        assert!(
            !summary.contains("Multi-rev graph"),
            "no multi-rev steer for single-rev: {summary}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cypher_tool_error_reads_once() {
        // A KgError-derived message already self-identifies (`Cypher execution
        // error: …` / `Cypher syntax error: …`); the tool prefix must not stutter
        // it — the reported triple `Cypher error: Cypher execution error: Cypher
        // execution error: …` collapses to the engine message read once.
        assert_eq!(
            cypher_tool_error("Cypher execution error: CALL rev_diff: boom"),
            "Cypher execution error: CALL rev_diff: boom"
        );
        assert_eq!(
            cypher_tool_error("Cypher syntax error: bad token"),
            "Cypher syntax error: bad token"
        );
        // A message that does NOT self-identify still gets the single tool prefix.
        assert_eq!(
            cypher_tool_error("mutation Cypher is not allowed"),
            "Cypher error: mutation Cypher is not allowed"
        );
    }

    #[test]
    fn single_rev_via_revs_reads_as_snapshot_and_dedups() {
        let dir = std::env::temp_dir().join(format!("kgl_singlerev_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let s2 = "r2".to_string();
        let gs = GraphState::new(Some(WorkspaceGraphMode::LocalWorkspace))
            .with_workspace_graph(Some(test_hooks()));
        // Duplicate labels for one commit → deduped to a single rev (defect B).
        gs.build_workspace_graph(&dir, Some(&[s2.clone(), s2.clone()]))
            .expect("single-rev-via-revs build");
        // Header carries the rev once, not "s2,s2".
        let attrs = gs.with_active(|a| a.identity_attrs());
        assert!(
            attrs.contains(&format!("revs=\"{s2}\"")) && !attrs.contains(&format!("{s2},{s2}")),
            "duplicate labels collapse to one in the header: {attrs}"
        );
        // Summary reads as a plain snapshot: no over-count warning, no rev_diff,
        // and NOT "Multi-rev … spanning 1" (defect E).
        let summary = gs.activation_summary().expect("summary");
        assert!(
            !summary.contains("Multi-rev graph"),
            "a single rev is not a multi-rev graph: {summary}"
        );
        assert!(
            !summary.contains("over-count") && !summary.contains("rev_diff"),
            "a single rev has nothing to over-count or diff: {summary}"
        );
        assert!(
            summary.contains(&format!("revision '{s2}'")),
            "reads as a committed snapshot at the rev: {summary}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn tmp_kgl(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("kglmcp_{}_{}.kgl", std::process::id(), tag));
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn lifecycle_create_mutate_save_load() {
        let p = tmp_kgl("lifecycle");
        let s = GraphState::default();
        // create empty → mutate via the write path → save_as
        s.create_in_mode(&p, StorageMode::Memory).unwrap();
        let r = s
            .with_active_mut(|a| write(a, "CREATE (:Task {id:'t1', status:'todo'})", None))
            .unwrap();
        assert!(r.is_ok(), "{r:?}");
        s.save_as(&p).unwrap();
        drop(s);
        // load into a *fresh* state → the node survived (the 0.12.2 fix path too)
        let s2 = GraphState::default();
        s2.load_kgl(&p).unwrap();
        assert_eq!(s2.schema().unwrap().0, 1, "expected 1 node after reload");
        drop(s2);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn path_backed_active_graph_retains_writer_lease() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("retained_lease.kgl");
        let state = GraphState::default();
        state.create_in_mode(&p, StorageMode::Memory).unwrap();
        assert_eq!(Arc::strong_count(&state.inner), 1);
        assert!(kglite::api::io::GraphWriterLease::acquire(&p, Duration::ZERO).is_err());
        drop(state);
        kglite::api::io::GraphWriterLease::acquire(&p, Duration::ZERO).unwrap();
    }

    #[test]
    fn load_graph_swaps_active() {
        let pa = tmp_kgl("swapA");
        let pb = tmp_kgl("swapB");
        // build two distinct graphs on disk
        for (p, n) in [(&pa, 1u64), (&pb, 3u64)] {
            let s = GraphState::default();
            s.create_in_mode(p, StorageMode::Memory).unwrap();
            for i in 0..n {
                s.with_active_mut(|a| write(a, &format!("CREATE (:N {{id:'{i}'}})"), None))
                    .unwrap()
                    .unwrap();
            }
            s.save_as(p).unwrap();
        }
        // one state loads A then B → active reflects B
        let s = GraphState::default();
        s.load_kgl(&pa).unwrap();
        assert_eq!(s.schema().unwrap().0, 1);
        s.load_kgl(&pb).unwrap();
        assert_eq!(s.schema().unwrap().0, 3, "load_graph should swap to B");
        drop(s);
        let _ = std::fs::remove_file(&pa);
        let _ = std::fs::remove_file(&pb);
    }

    fn write(
        active: &mut ActiveGraph,
        q: &str,
        scope: Option<&[String]>,
    ) -> Result<String, String> {
        run_cypher_write(active, q, scope, None, None, None, None)
    }

    #[test]
    fn write_path_creates_and_reads_back() {
        let mut a = fresh_active();
        write(&mut a, "CREATE (:Task {id:'t1', status:'todo'})", None).unwrap();
        // A subsequent read on the writable path observes the mutation.
        let out = write(&mut a, "MATCH (t:Task) RETURN count(t) AS c", None).unwrap();
        assert!(out.contains('1'), "expected 1 task, got: {out}");
    }

    #[test]
    fn write_with_no_return_acknowledges_stats() {
        // A CREATE/SET/MERGE with no RETURN must NOT read back "No results"
        // (indistinguishable from a no-op match) — it acknowledges the write.
        let mut a = fresh_active();
        let out = write(&mut a, "CREATE (:Task {id:'t1'})", None).unwrap();
        assert!(out.starts_with("OK:"), "expected write ack, got: {out}");
        assert!(out.contains("node(s) created"), "got: {out}");
        // The ack stamps the running engine version (stale-server footgun).
        assert!(
            out.contains(&format!("[engine {}]", env!("CARGO_PKG_VERSION"))),
            "ack should carry the engine version, got: {out}"
        );
        // SET acks too.
        let out = write(&mut a, "MATCH (t:Task{id:'t1'}) SET t.status='done'", None).unwrap();
        assert!(out.contains("property(ies) set"), "got: {out}");
        // A read that matches nothing still says "No results" (distinct signal).
        let out = write(&mut a, "MATCH (x:Nope) RETURN x", None).unwrap();
        assert!(out.contains("No results"), "got: {out}");
    }

    #[test]
    fn write_scope_blocks_out_of_scope_create() {
        let mut a = fresh_active();
        let scope = vec!["Plan".to_string(), "Task".to_string()];
        // In-scope is allowed.
        write(&mut a, "CREATE (:Task {id:'t1'})", Some(&scope)).unwrap();
        // Out-of-scope is rejected.
        let err = write(&mut a, "CREATE (:Algorithm {id:'a1'})", Some(&scope)).unwrap_err();
        assert!(
            err.contains("write scope"),
            "expected scope error, got: {err}"
        );
        // The rejected CREATE did not land.
        let out = write(&mut a, "MATCH (n:Algorithm) RETURN count(n) AS c", None).unwrap();
        assert!(
            out.contains('0') || out.contains("No results"),
            "got: {out}"
        );
    }

    #[test]
    fn new_edge_type_via_write_path_registers() {
        // The 0.12.2 edge-persistence fix in action through the MCP write path:
        // a brand-new relationship type is registered (queryable, would persist).
        let mut a = fresh_active();
        write(&mut a, "CREATE (:Task {id:'t'})", None).unwrap();
        write(&mut a, "CREATE (:Spec {id:'s'})", None).unwrap();
        write(
            &mut a,
            "MATCH (t:Task{id:'t'}),(s:Spec{id:'s'}) CREATE (t)-[:IMPLEMENTS_SPEC]->(s)",
            None,
        )
        .unwrap();
        let out = write(
            &mut a,
            "MATCH (:Task)-[:IMPLEMENTS_SPEC]->() RETURN count(*) AS c",
            None,
        )
        .unwrap();
        assert!(out.contains('1'), "expected 1 edge, got: {out}");
    }
}
