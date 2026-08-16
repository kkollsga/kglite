//! [`GraphState`] — the clone-shared active-graph slot behind every KGLite
//! MCP tool — plus its construction, graph lifecycle and read accessors.
//! Workspace rebuild lifecycle lives in `state_workspace`; the Cypher
//! execution seam lives in `cypher_exec`.

use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};

use anyhow::Result;
use kglite::api::cypher::ValueCodec;
use kglite::api::durable::DurabilityLevel;
use kglite::api::introspection::compute_schema;
use kglite::api::io::{open_or_create_graph_in_mode, GraphWriterLease, OpenDisposition};
use kglite::api::storage::StorageMode;
use kglite::api::{Embedder, KnowledgeGraph};

use crate::tools::*;

/// Shared active-graph state. Cloning is cheap (Arc).
#[derive(Clone, Default)]
pub struct GraphState {
    pub(crate) inner: Arc<RwLock<Option<ActiveGraph>>>,
    /// Clone-shared ownership gate for lazy workspace rebuilds. A rebuild is
    /// prepared without holding the active-graph lock, so readers that also
    /// passed through freshness handling must wait here until publication (or
    /// failure bookkeeping) completes instead of querying the old generation.
    /// Non-workspace states bypass this gate entirely.
    pub(crate) rebuild_gate: Arc<WorkspaceRebuildGate>,
    /// Deferred-rebuild slot. The watcher tags the active root here
    /// (cheap, microseconds — sets the slot, drops the lock); each
    /// MCP tool entry calls [`ensure_workspace_graph_fresh`] which atomically
    /// `take()`s the slot and rebuilds. Pattern: do the actual work
    /// lazily, never on the watcher thread. N FS events between two
    /// tool calls → 1 rebuild with one target-bound union of accepted paths.
    pub(crate) pending_rebuild: Arc<RwLock<Option<PendingWorkspaceRebuild>>>,
    /// Outcome bookkeeping for the lazy rebuild: the last failure (kept
    /// until the next successful build and surfaced in tool output next
    /// to the built-at identity) plus a consecutive-failure counter
    /// implementing the [`MAX_CONSECUTIVE_REBUILD_FAILURES`] hot-fail
    /// guard.
    pub(crate) rebuild_status: Arc<RwLock<RebuildStatus>>,
    /// Workspace mode used to build request/relevance context. `None` for
    /// graph/source-root/bare modes that never ask a producer to build.
    pub(crate) workspace_mode: Option<WorkspaceGraphMode>,
    /// Manifest-declared value codecs (`extensions.value_codecs`). Server-
    /// config, set once at boot via [`with_value_codecs`] and carried by every
    /// clone; passed to `ExecuteOptions::value_codecs` on each `cypher_query` /
    /// `tools[].cypher` run so the engine decodes query-side literals and
    /// encodes result columns (`'Q42'` ↔ `42`) — safely, after parsing.
    pub(crate) value_codecs: Option<Arc<Vec<ValueCodec>>>,
    /// External workspace-graph lifecycle extension. Set once at boot and
    /// carried by every clone so lazy watch rebuilds use the same producer.
    pub(crate) workspace_graph_hooks: Option<Arc<WorkspaceGraphHooks>>,
    /// The manifest-declared embedder (`extensions.embedder`), bound once at
    /// boot and re-applied to every graph this state installs.
    ///
    /// It lives here rather than only on the active `KnowledgeGraph` because a
    /// handle's embedder does not survive a swap: `KnowledgeGraph::from_arc`
    /// starts at `embedder: None`, so `load_graph` / `create_graph` / a
    /// workspace rebuild used to leave `text_score()` dead for the rest of the
    /// process. Interior mutability rather than a `with_*` builder field:
    /// [`bind_embedder`](Self::bind_embedder) runs *after*
    /// `register_kglite_tools` has cloned the state into every tool closure,
    /// so a plain field would only ever reach the boot clone.
    pub(crate) embedder: Arc<RwLock<Option<Arc<dyn Embedder>>>>,
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

    /// A one-line warning describing the last failed lazy rebuild, or
    /// `None` when the last rebuild succeeded (the common case).
    /// Appended to tool output wherever the graph's built-at identity
    /// appears, so an agent knows the graph it queries is staler than
    /// the filesystem.
    pub fn rebuild_error_note(&self) -> Option<String> {
        let failure = self.workspace_rebuild_failure()?;
        let age = humanize_age(failure.failed_at);
        let note = format!(
            "WARNING: workspace graph rebuild failed {age} ago ({} consecutive \
             failure(s)) — the active graph is STALE relative to the \
             filesystem. Error: {}",
            failure.consecutive_failures, failure.message
        );
        Some(note)
    }

    /// Append the rebuild-failure warning (if any) to a tool response.
    pub(crate) fn with_rebuild_warning(&self, body: String) -> String {
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

    /// Open the graph at `path` — or create it when the path is absent — and
    /// bind it as the active graph.
    ///
    /// `requested_mode` means the same thing on both branches: create a missing
    /// graph in it, and convert an existing one to it. `None` — no `--storage`,
    /// and the `load_graph` tool's case — means the checkpoint decides, exactly
    /// as `kglite.open(path)` does. A request with no conversion (either disk
    /// direction) fails with the core reason rather than binding a graph in a
    /// mode nobody asked for: a flag that is parsed and then dropped is the
    /// defect shape this crate has already had to fix twice.
    pub fn open_or_create(
        &self,
        path: &Path,
        requested_mode: Option<StorageMode>,
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
        // `DurabilityLevel::Off`: this server attaches no write-ahead log, so
        // it takes the unrecovered-sidecar refusal rather than opening a graph
        // that is silently missing another writer's committed frames.
        let opened = open_or_create_graph_in_mode(path, requested_mode, DurabilityLevel::Off)
            .map_err(|e| anyhow::anyhow!("kglite graph open/create failed: {e}"))?;
        let mut kg = KnowledgeGraph::from_arc(opened.graph);
        // Off-lock, before publication: the new handle carries no embedder of
        // its own, so the boot-bound one is re-applied here.
        self.apply_bound_embedder(&mut kg);
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
    pub(crate) fn save_as(&self, path: &Path) -> std::result::Result<String, String> {
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

    /// Bind the embedder `text_score()` uses. Held on the state, so it applies
    /// to the active graph *and* to every graph installed afterwards.
    pub fn bind_embedder(&self, embedder: Arc<dyn Embedder>) -> Result<()> {
        *write_lock(&self.embedder) = Some(Arc::clone(&embedder));
        let mut guard = write_lock(&self.inner);
        let Some(active) = guard.as_mut() else {
            tracing::debug!("embedder loaded before any graph is active; binding deferred");
            return Ok(());
        };
        active.kg.set_embedder_native(embedder);
        Ok(())
    }

    /// Apply the state's bound embedder (if any) to a freshly built graph
    /// handle. Every path that installs a new [`ActiveGraph`] must pass its
    /// handle through here before publication — `KnowledgeGraph::from_arc`
    /// yields `embedder: None`, and a swap that skips this step silently
    /// disables `text_score()` for the rest of the process.
    pub(crate) fn apply_bound_embedder(&self, kg: &mut KnowledgeGraph) {
        let bound = read_lock(&self.embedder).as_ref().map(Arc::clone);
        if let Some(embedder) = bound {
            kg.set_embedder_native(embedder);
        }
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

    pub(crate) fn with_active<F>(&self, f: F) -> String
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
    pub(crate) fn with_active_mut<F, T>(&self, f: F) -> Option<T>
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
}
