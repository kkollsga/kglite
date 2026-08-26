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

/// Whether this server takes the cross-process single-writer lease
/// (`<path>.lock`, `GraphWriterLease`) on the graph path it opens.
///
/// The lease is what stops two writers from each building a full snapshot and
/// having the last `save()` win. A server that can never write the file it
/// serves buys nothing with it and costs the operator a great deal: while it is
/// held, `kglite.open(path)` — the default, locking open an external rebuilder
/// uses — is refused outright for the server's whole lifetime, with an error
/// that never mentions this server.
///
/// [`Self::Exclusive`] is the [`Default`] on purpose: a construction path that
/// forgets to declare a policy keeps the historical, conservative behaviour.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WriterLeasePolicy {
    /// Own the path: take the lease on every open. What a `--writable` /
    /// `save_graph`-enabled server does, because it really can rewrite the file.
    #[default]
    Exclusive,
    /// Read-only deployment: skip the lease for a regular-file graph, so
    /// external rebuilders (and other read-only servers) can lock the same
    /// `.kgl`. A torn in-place rewrite arriving mid-load is already caught by
    /// the load path's `GraphFileIdentity` before/after check, which fails the
    /// load and leaves the previously served graph installed.
    ///
    /// Not unconditional — see [`GraphState::takes_writer_lease`].
    ReadOnly,
}

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
    /// MCP tool entry calls [`GraphState::ensure_graph_fresh`], which for a
    /// workspace mode dispatches to `ensure_workspace_graph_fresh` — atomically
    /// `take()`s the slot and rebuilds. N FS events between two tool calls →
    /// 1 rebuild with one target-bound union of accepted paths.
    pub(crate) pending_rebuild: Arc<RwLock<Option<PendingWorkspaceRebuild>>>,
    /// Outcome bookkeeping for the lazy rebuild: the last failure (kept
    /// until the next successful build and surfaced in tool output next
    /// to the built-at identity) plus a consecutive-failure counter
    /// implementing the [`MAX_CONSECUTIVE_REBUILD_FAILURES`] hot-fail
    /// guard.
    pub(crate) rebuild_status: Arc<RwLock<RebuildStatus>>,
    /// `--graph` mode's counterpart to `pending_rebuild` + `rebuild_status`:
    /// the flag an armed `extensions.graph_watch` watcher sets when the served
    /// file is rewritten, plus that reload's failure bookkeeping. Untouched
    /// (and never contended) in every other mode — see `graph_reload`.
    pub(crate) graph_reload: Arc<RwLock<GraphReloadStatus>>,
    /// Workspace mode used to build request/relevance context. `None` for
    /// graph/source-root/bare modes that never ask a producer to build.
    pub(crate) workspace_mode: Option<WorkspaceGraphMode>,
    /// Manifest-declared value codecs (`extensions.value_codecs`). Server-
    /// config, set once at boot via [`with_value_codecs`] and carried by every
    /// clone; passed to `ExecuteOptions::value_codecs` on each `cypher_query` /
    /// `tools[].cypher` run so the engine decodes query-side literals and
    /// encodes result columns (`'Q42'` ↔ `42`) — safely, after parsing.
    pub(crate) value_codecs: Option<Arc<Vec<ValueCodec>>>,
    /// The operator's parallel-runtime opt-in (`--parallel` /
    /// `extensions.parallel`). Server-config, set once at boot via
    /// [`with_parallel`](Self::with_parallel) and carried by every clone;
    /// reaches the engine through [`ExecPolicy`] on the read seam only.
    pub(crate) parallel: bool,
    /// External workspace-graph lifecycle extension. Set once at boot and
    /// carried by every clone so lazy watch rebuilds use the same producer.
    pub(crate) workspace_graph_hooks: Option<Arc<WorkspaceGraphHooks>>,
    /// The manifest-declared embedder (`extensions.embedder`), bound once at
    /// boot and re-applied to every graph this state installs.
    ///
    /// It lives here because a handle's embedder does not survive a swap — see
    /// [`apply_bound_embedder`](Self::apply_bound_embedder). Interior
    /// mutability rather than a `with_*` builder field:
    /// [`bind_embedder`](Self::bind_embedder) runs *after*
    /// `register_kglite_tools` has cloned the state into every tool closure,
    /// so a plain field would only ever reach the boot clone.
    pub(crate) embedder: Arc<RwLock<Option<Arc<dyn Embedder>>>>,
    /// The manifest-declared ontology (`extensions.ontology`), parsed once
    /// at boot and re-applied — **memory-only** — to every graph this state
    /// installs. Nothing in this server auto-saves, and boot opens use
    /// `DurabilityLevel::Off`, so the declarations never reach the source
    /// file unless an agent explicitly calls the save_graph tool (which then
    /// correctly persists them). Same interior-mutability shape and reason
    /// as `embedder` above.
    pub(crate) ontology: Arc<RwLock<Option<Arc<kglite::api::OntologyStore>>>>,
    /// Whether opens take the cross-process writer lease. Server-config, set
    /// once at boot via [`with_writer_lease_policy`](Self::with_writer_lease_policy)
    /// — before `bind_mode` performs the boot open — and carried by every clone.
    pub(crate) writer_lease_policy: WriterLeasePolicy,
}

impl GraphState {
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

    /// Let this server's reads use the engine's parallel runtime. Builder
    /// form, set once at boot like [`Self::with_value_codecs`].
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Attach an external workspace-graph producer. Builder form, set once at
    /// boot like [`Self::with_value_codecs`].
    pub fn with_workspace_graph(mut self, hooks: Option<Arc<WorkspaceGraphHooks>>) -> Self {
        self.workspace_graph_hooks = hooks;
        self
    }

    /// Declare whether this server owns the graph file it opens. Builder form,
    /// set once at boot like [`Self::with_value_codecs`] — and, unlike the
    /// embedder, it must be set *before* `bind_mode` opens the boot graph,
    /// which is why it is a builder field rather than interior mutability.
    pub fn with_writer_lease_policy(mut self, policy: WriterLeasePolicy) -> Self {
        self.writer_lease_policy = policy;
        self
    }

    /// Whether opening `path` should take the writer lease.
    ///
    /// [`WriterLeasePolicy::ReadOnly`] skips it only for a **regular file** —
    /// the atomically-republished `.kgl` case. Two targets keep the lease even
    /// there:
    ///
    /// - a **disk-graph directory**: a tree of retained mmaps behind a `CURRENT`
    ///   pointer, where an external writer mutating a column under our live
    ///   mapping is memory corruption, not a stale read;
    /// - a **path that does not exist yet**: this open is about to *create* the
    ///   graph, which is a write regardless of what the tool surface allows
    ///   afterwards.
    fn takes_writer_lease(&self, path: &Path) -> bool {
        match self.writer_lease_policy {
            WriterLeasePolicy::Exclusive => true,
            WriterLeasePolicy::ReadOnly => !path.is_file(),
        }
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

    pub fn value_codecs(&self) -> Option<&[ValueCodec]> {
        self.value_codecs.as_deref().map(|v| v.as_slice())
    }

    /// The boot-decided engine policy every Cypher route applies.
    ///
    /// One accessor rather than one per setting, so a route cannot pick up
    /// the codecs and miss the parallel pin (or the next setting added here).
    pub(crate) fn exec_policy(&self) -> ExecPolicy<'_> {
        ExecPolicy {
            value_codecs: self.value_codecs(),
            parallel: self.parallel,
        }
    }

    /// A one-line warning describing the last failed lazy refresh, or
    /// `None` when the last one succeeded (the common case).
    /// Appended to tool output wherever the graph's built-at identity
    /// appears, so an agent knows the graph it queries is staler than
    /// the filesystem.
    ///
    /// Two producers, one channel: workspace modes refresh by rebuilding from
    /// their producer, `--graph` mode by re-reading its file. At most one can
    /// have a failure recorded — a state is only ever in one of those modes.
    pub fn rebuild_error_note(&self) -> Option<String> {
        let Some(failure) = self.workspace_rebuild_failure() else {
            return self.graph_reload_error_note();
        };
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
    ///
    /// Applied to the error arm too (see `register::map_body`): a stale graph is
    /// exactly as relevant to a failed call as to a successful one, and
    /// keeping the note on both arms is what makes the text byte-identical to
    /// the pre-`isError` responses, which carried the error inside the body.
    pub(crate) fn with_rebuild_warning(&self, body: String) -> String {
        match self.rebuild_error_note() {
            Some(note) => format!("{body}\n\n{note}"),
            None => body,
        }
    }

    pub fn load_kgl(&self, path: &Path) -> Result<()> {
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
        let mut writer_lease = if reuse_existing || !self.takes_writer_lease(path) {
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
        // Take the outgoing graph *out* of the slot rather than letting the
        // assignment free it in place: dropping a large graph (petgraph arenas,
        // columnar buffers, mmap teardown) is not free, and doing it under the
        // write lock stalls every reader for the duration.
        let previous = guard.replace(ActiveGraph {
            kg,
            source_path: Some(path.to_path_buf()),
            writer_lease,
            root: Some(path.to_path_buf()),
            revs: None,
            built_at: SystemTime::now(),
            generation,
        });
        drop(guard);
        drop(previous);
        // A graph is installed from this path, so whatever made previous
        // reloads fail is behind us: clear the failure counter (and with it any
        // watcher dormancy) for every open route — boot, `load_graph`,
        // `create_graph`, `reload_graph`, and the lazy watch reload alike.
        self.clear_graph_reload_failures();
        Ok(opened.disposition)
    }

    /// The file (or disk-graph directory) the active graph was opened from —
    /// the path `save_graph` writes to and `reload_graph` re-reads. `None` when
    /// no graph is active, or when it was built without a backing path.
    pub(crate) fn source_path(&self) -> Option<std::path::PathBuf> {
        read_lock(&self.inner)
            .as_ref()
            .and_then(|active| active.source_path.clone())
    }

    /// Monotonic identity of the installed graph — bumped by every swap.
    /// Reported by `reload_graph` so a caller can tell a completed re-read
    /// apart from a response about the graph it already had.
    pub(crate) fn generation(&self) -> Option<u64> {
        read_lock(&self.inner)
            .as_ref()
            .map(|active| active.generation)
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
        // The manifest ontology rides the same seam: every install path
        // already routes through here before publication.
        let ontology = read_lock(&self.ontology).as_ref().map(Arc::clone);
        if let Some(store) = ontology {
            match std::sync::Arc::make_mut(kg.dir_mut()).define_ontology((*store).clone()) {
                Ok(warnings) => {
                    for w in warnings {
                        tracing::warn!("manifest ontology: {w}");
                    }
                }
                Err(e) => tracing::error!("manifest ontology rejected for this graph: {e}"),
            }
        }
    }

    /// Bind the manifest-declared ontology; ["apply_bound_embedder"] installs
    /// it on every subsequently published graph.
    pub fn bind_ontology(&self, store: Arc<kglite::api::OntologyStore>) {
        *write_lock(&self.ontology) = Some(store);
    }

    pub fn schema(&self) -> Option<(u64, u64)> {
        let guard = read_lock(&self.inner);
        let active = guard.as_ref()?;
        let overview = compute_schema(active.kg.dir());
        Some((overview.node_count as u64, overview.edge_count as u64))
    }

    /// Test-only accessor for the one-line schema mini-map the workspace
    /// activation message carries (the mcp-methods 0.3.46 activation-summary
    /// hook); production reaches it through
    /// [`Self::reusable_activation_summary`] and the workspace rebuild path.
    /// Steers an agent's FIRST move toward the graph before it defaults to
    /// grep — the activation result is the one message read before any tool
    /// choice.
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

    /// Borrow the active graph for a read, or `None` when no graph is loaded.
    ///
    /// The no-graph substitution belongs to the caller: a tool route turns it
    /// into an `Err(NO_GRAPH)` error envelope, while an identity or
    /// introspection read wants the `Option` itself. Read-side twin of
    /// [`Self::with_active_mut`].
    pub(crate) fn with_active<F, T>(&self, f: F) -> Option<T>
    where
        F: FnOnce(&ActiveGraph) -> T,
    {
        let guard = read_lock(&self.inner);
        guard.as_ref().map(f)
    }

    /// Borrow the active `KnowledgeGraph` for read-only inspection, or `None`
    /// when no graph is loaded — the no-graph substitution belongs to the
    /// caller, as in [`Self::with_active`].
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
    /// write-enabled `cypher_query` path: the write lock is what keeps
    /// mutation correct under any MCP dispatch model, serial or concurrent.
    /// `None` when no graph is active.
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
