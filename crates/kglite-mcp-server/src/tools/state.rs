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
use kglite::api::io::{
    open_or_create_graph_in_mode, GraphFileIdentity, GraphWriterLease, OpenDisposition,
    WriteOwnership,
};
use kglite::api::storage::StorageMode;
use kglite::api::{Embedder, KnowledgeGraph};

use crate::tools::*;

/// Whether this server may ever write back the graph file it opens — and so
/// whether it takes the cross-process single-writer lease
/// (`<path>.lock`, `GraphWriterLease`) on it at all.
///
/// The lease is what stops two writers from each building a full snapshot and
/// having the last `save()` win. It is taken **lazily**: a server that may
/// write holds it from its first unsaved change until that change is saved or
/// discarded, not for its lifetime. That window is exactly the lost-update
/// window, and nothing else — four MCP clients can serve one `.kgl` and only
/// the one actually mid-write excludes the others. While a lease is held,
/// `kglite.open(path)` — the default, locking open an external rebuilder uses
/// — is refused, with an error that names the holder.
///
/// [`Self::Exclusive`] is the [`Default`] on purpose: a construction path that
/// forgets to declare a policy keeps the writable, conservative behaviour.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WriterLeasePolicy {
    /// May write the path: carry write ownership and take the lease at the
    /// first unsaved change. What a `--writable` / `save_graph`-enabled server
    /// does, because it really can rewrite the file.
    #[default]
    Exclusive,
    /// Read-only deployment: never writes a regular-file graph back, so it
    /// carries no ownership and never locks — external rebuilders (and other
    /// read-only servers) can lock the same `.kgl`. A torn in-place rewrite
    /// arriving mid-load is already caught by the load path's
    /// `GraphFileIdentity` before/after check, which fails the load and leaves
    /// the previously served graph installed.
    ///
    /// Not unconditional — see [`GraphState::owns_writes`].
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
    /// what the per-call `stat` of the served file learned — the last failed
    /// re-read (with the bytes it failed on, so the same failure is not
    /// retried every call) and whether the file has moved away from a server
    /// holding unsaved changes. Untouched (and never contended) in every other
    /// mode — see `graph_reload`.
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
    pub(crate) ontology: Arc<RwLock<Option<BoundOntology>>>,
    /// Whether opens carry write ownership at all. Server-config, set
    /// once at boot via [`with_writer_lease_policy`](Self::with_writer_lease_policy)
    /// — before `bind_mode` performs the boot open — and carried by every clone.
    pub(crate) writer_lease_policy: WriterLeasePolicy,
    /// Operator-facing name this server publishes in the `<path>.lock-owner`
    /// record, so a peer refused the lease is told *which* client is holding
    /// it rather than a bare pid. Server-config, set once at boot via
    /// [`with_lease_label`](Self::with_lease_label).
    pub(crate) lease_label: Option<Arc<str>>,
}

/// The manifest-declared ontology plus its boot-time materialization flag.
#[derive(Clone)]
pub struct BoundOntology {
    pub store: Arc<kglite::api::OntologyStore>,
    pub materialize: bool,
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

    /// Name this server publishes in the owner record it writes while holding
    /// the writer lease. Builder form, set once at boot like
    /// [`Self::with_value_codecs`].
    pub fn with_lease_label(mut self, label: Option<String>) -> Self {
        self.lease_label = label.map(Arc::from);
        self
    }

    /// Whether opening `path` carries write ownership of it.
    ///
    /// [`WriterLeasePolicy::ReadOnly`] declines it only for a **regular file** —
    /// the atomically-republished `.kgl` case. Two targets keep ownership even
    /// there:
    ///
    /// - a **disk-graph directory**: a tree of retained mmaps behind a `CURRENT`
    ///   pointer, where an external writer mutating a column under our live
    ///   mapping is memory corruption, not a stale read;
    /// - a **path that does not exist yet**: this open is about to *create* the
    ///   graph, which is a write regardless of what the tool surface allows
    ///   afterwards.
    fn owns_writes(&self, path: &Path) -> bool {
        match self.writer_lease_policy {
            WriterLeasePolicy::Exclusive => true,
            WriterLeasePolicy::ReadOnly => !path.is_file(),
        }
    }

    /// Whether the lease has to be held from the open rather than from the
    /// first unsaved change.
    ///
    /// The lazy lease answers "who may overwrite this file", and waiting for a
    /// first mutation is safe for a `.kgl` that is replaced atomically. The two
    /// non-regular-file cases cannot wait: a path that does not exist yet is
    /// being *created* by this very open, and a disk-graph directory is a tree
    /// of live mmaps an external writer would corrupt rather than merely make
    /// stale. Both are decided before the open, while `path` still describes
    /// what was there when we looked.
    fn leases_at_open(path: &Path) -> bool {
        !path.is_file()
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
        let owns_writes = self.owns_writes(path);
        // Decided here, once, rather than per tool call: `GraphFileIdentity`
        // is a cheap `stat` for a regular file but an open + read of `CURRENT`
        // for a disk-graph directory, and the answer cannot change without a
        // new open landing here anyway. A workspace-backed state never reaches
        // the stat path at all (`ensure_graph_fresh` dispatches on the mode),
        // so this only has to separate "one republished file" from "a
        // directory of live mmaps" and "a path that is not there".
        let freshness_path =
            (self.workspace_mode.is_none() && path.is_file()).then(|| path.to_path_buf());
        // A same-path reload inherits the lease the previous ownership already
        // holds, so it must not try to take a second one: `flock` is per
        // open-file-description, and this process would contend with itself.
        // Taken *before* the open, not after: for a path this call creates,
        // lock-then-create is what stops two servers booting on the same
        // missing path from both creating it. Adopted unpinned below, so the
        // first save hands it to the lazy lifecycle like any other lease.
        let eager_lease = (owns_writes && !reuse_existing && Self::leases_at_open(path))
            .then(|| {
                GraphWriterLease::acquire_labeled(
                    path,
                    Duration::from_secs(30),
                    self.lease_label.as_deref(),
                )
            })
            .transpose()
            .map_err(|e| anyhow::anyhow!("kglite writer lease failed: {}", e.error))?;
        // `DurabilityLevel::Off`: this server attaches no write-ahead log, so
        // it takes the unrecovered-sidecar refusal rather than opening a graph
        // that is silently missing another writer's committed frames.
        let opened = open_or_create_graph_in_mode(path, requested_mode, DurabilityLevel::Off)
            .map_err(|e| anyhow::anyhow!("kglite graph open/create failed: {e}"))?;
        let identity = opened.identity;
        let loaded_identity = identity.clone();
        let mut kg = KnowledgeGraph::from_arc(opened.graph);
        // Off-lock, before publication: the new handle carries no embedder of
        // its own, so the boot-bound one is re-applied here. The version
        // baseline below is taken *after* it, so a manifest ontology installed
        // (or materialized) on this graph is part of what "clean" means rather
        // than reading as an unsaved change nobody made.
        self.apply_bound_embedder(&mut kg);
        let mut lease_since = eager_lease.is_some().then(SystemTime::now);
        let mut ownership = owns_writes.then(|| {
            WriteOwnership::new(
                path.to_path_buf(),
                identity.clone(),
                kg.dir(),
                self.lease_label.as_deref().map(str::to_owned),
                // The pristine snapshot is what makes a failed first write —
                // and `reload_graph(discard_unsaved=true)` — recoverable
                // without re-reading the file. `Arc::clone` is O(1); the fork
                // it may later cost is paid once per dirty window.
                true,
            )
        });
        if let (Some(ownership), Some(lease)) = (ownership.as_mut(), eager_lease) {
            ownership.adopt_lease(lease, kg.dir(), false);
        }
        let mut guard = write_lock(&self.inner);
        if reuse_existing {
            // Carry the previous ownership across the swap rather than the
            // fresh one: it holds the lease (if this server was mid-write) and
            // knows the high-water version, neither of which survives being
            // rebuilt. `resynced` re-points it at what was just read.
            if let Some(mut previous) = guard.as_mut().and_then(|active| active.ownership.take()) {
                previous.resynced(identity, kg.dir());
                lease_since = previous
                    .holds_lease()
                    .then(|| guard.as_ref().and_then(|active| active.lease_since))
                    .flatten()
                    .or(lease_since);
                ownership = Some(previous);
            }
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
            ownership,
            lease_since,
            freshness_path,
            loaded_identity: Some(loaded_identity),
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
    ///
    /// Retargeting **releases the source file's lease**: this graph is not
    /// going back there, and it is `save_graph_as` an agent reaches for when a
    /// peer holds the original — leaving the old path locked would keep the
    /// jam it exists to escape.
    pub(crate) fn save_as(&self, path: &Path) -> std::result::Result<String, String> {
        let mut guard = write_lock(&self.inner);
        let Some(active) = guard.as_mut() else {
            return Err(NO_GRAPH.to_string());
        };
        let replacing_target = active.source_path.as_deref() != Some(path);
        let identity = GraphFileIdentity::capture(path)
            .map_err(|e| format!("save_graph_as error: cannot read {}: {e}", path.display()))?;
        match active.ownership.as_mut() {
            Some(ownership) if replacing_target => ownership.retarget(path.to_path_buf(), identity),
            // Same target as the bound one: `save_graph` under another name,
            // lost-update check included — publishing over a file somebody
            // else replaced is refused here exactly as it is there.
            Some(_) => {}
            // A graph with no file behind it (a workspace build, a bare
            // in-memory session) acquires ownership by being given a path.
            None => {
                active.ownership = Some(WriteOwnership::new(
                    path.to_path_buf(),
                    identity,
                    active.kg.dir(),
                    self.lease_label.as_deref().map(str::to_owned),
                    true,
                ));
            }
        }
        let ownership = active
            .ownership
            .as_mut()
            .expect("ownership is present on every branch above");
        // Publish through the active graph's own Arc (write lock held) so
        // `prepare_save`'s `Arc::make_mut` sees refcount 1 — no deep copy,
        // and the columnar consolidation lands on the live graph instead
        // of a discarded clone. `compute_schema` only needs `&DirGraph`.
        ownership
            .publish(active.kg.dir_mut())
            .map_err(|refusal| refused_save("save_graph_as", &refusal))?;
        // A successful publish hands the lease back, so the status this graph
        // reports on every later response must stop claiming to hold one.
        active.lease_since = None;
        active.source_path = Some(path.to_path_buf());
        let path_str = path.to_string_lossy().into_owned();
        let overview = compute_schema(active.kg.dir());
        Ok(format!(
            "Saved {path_str} ({} nodes, {} edges); save target rebound here.",
            overview.node_count, overview.edge_count
        ))
    }

    /// Whether the active graph carries changes its file does not.
    pub(crate) fn is_dirty(&self) -> bool {
        read_lock(&self.inner)
            .as_ref()
            .is_some_and(ActiveGraph::is_dirty)
    }

    /// Throw away every unsaved change and release the writer lease, leaving
    /// the graph exactly as the file last had it.
    ///
    /// The rollback is a snapshot restore, not a re-read: nothing touches the
    /// disk, so it works while the file is unreadable or held by somebody
    /// else. Returns whether a snapshot was there to restore — `false` means
    /// the mutations are still installed and only a reload can clear them.
    pub(crate) fn discard_unsaved_changes(&self) -> bool {
        let mut guard = write_lock(&self.inner);
        let Some(active) = guard.as_mut() else {
            return false;
        };
        let Some(ownership) = active.ownership.as_mut() else {
            return false;
        };
        let discarded = ownership.discard(active.kg.dir_mut());
        active.lease_since = None;
        discarded.restored
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
        let ontology = read_lock(&self.ontology).clone();
        if let Some(bound) = ontology {
            // Lineage-preserving rather than a raw `Arc::make_mut`: a forced
            // clone here must re-adopt the disk writer authority and fold the
            // copy-on-write overlays back, which `Arc::make_mut` alone skips.
            // And deliberately *not* the version-bumping `make_dir_graph_mut`:
            // a manifest ontology is server configuration, not an agent's
            // change, so a materializing server must still boot clean.
            let dir = kglite::api::make_dir_graph_mut_preserving_lineage(kg.dir_mut());
            match dir.define_ontology((*bound.store).clone()) {
                Ok(warnings) => {
                    for w in warnings {
                        tracing::warn!("manifest ontology: {w}");
                    }
                    if bound.materialize {
                        // adopt=true: a served file's pre-existing manual
                        // labels must not fail the boot — the label goes
                        // Open, which only disables optimizations.
                        match dir.materialize_ontology(true) {
                            Ok(report) => {
                                let stamped: usize = report.iter().map(|r| r.stamped).sum();
                                tracing::info!(
                                    labels = report.len(),
                                    stamped,
                                    "manifest ontology materialized (memory-only)"
                                );
                            }
                            Err(e) => tracing::error!("manifest ontology materialize failed: {e}"),
                        }
                    }
                }
                Err(e) => tracing::error!("manifest ontology rejected for this graph: {e}"),
            }
        }
    }

    /// Bind the manifest-declared ontology; ["apply_bound_embedder"] installs
    /// it on every subsequently published graph.
    pub fn bind_ontology(&self, bound: BoundOntology) {
        *write_lock(&self.ontology) = Some(bound);
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
