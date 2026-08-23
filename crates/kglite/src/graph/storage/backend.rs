//! GraphBackend enum + per-backend dispatcher + GraphRead / GraphWrite impls.
//!
//! The `GraphBackend` enum is the runtime variant of all storage backends
//! (Memory / Mapped / Disk / Recording). Its trait impls forward to the
//! inner backend via enum match. This is the central enum-dispatch boundary
//! captured by the source-quality whitelist.

use crate::graph::schema::{EdgeData, InternedKey, NodeData};
use crate::graph::storage::column_store::ColumnStore;
use crate::graph::storage::forked::{can_fork, ForkedGraph};
use crate::graph::storage::recording::RecordingGraph;
use crate::graph::storage::undo::UndoJournal;
use crate::graph::storage::{GraphRead, GraphWrite, MappedGraph, MemoryGraph};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableDiGraph;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::sync::Arc;

use crate::graph::storage::disk::graph::{DiskGraph, DiskQueryGuard};

#[cfg(test)]
thread_local! {
    static BACKEND_CLONE_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Nodes copied by backend clones, summed. Distinguishes a genuinely
    /// expensive whole-graph clone from the O(1) clone of an intentionally
    /// emptied backend (the statement checkpoint's schema shell), which the
    /// bare count cannot tell apart.
    static BACKEND_CLONE_NODES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_backend_clone_count() {
    BACKEND_CLONE_COUNT.set(0);
    BACKEND_CLONE_NODES.set(0);
}

#[cfg(test)]
pub(crate) fn backend_clone_count() -> usize {
    BACKEND_CLONE_COUNT.get()
}

/// Total nodes copied by backend clones since the last reset.
#[cfg(test)]
pub(crate) fn backend_clone_nodes() -> usize {
    BACKEND_CLONE_NODES.get()
}

/// Record `n` nodes genuinely copied.
///
/// **Which paths call this is load-bearing.** `impl Clone for
/// GraphBackend` used to bump this by `node_count()` on *every* clone; once the
/// `Memory` arm started producing a shallow `Forked` overlay, that would have
/// gone on reporting a whole-graph copy that no longer happens — the oracle
/// becoming a liar, and `held_reader_copies_no_nodes` unable to distinguish the
/// fix from the defect. Now only the paths that actually duplicate node storage
/// call this: the deep-clone fallback in `Clone`, and `ForkedGraph::materialise`
/// (the adjacency-write escape hatch, which is a genuine copy).
#[cfg(test)]
pub(crate) fn note_nodes_copied(n: usize) {
    BACKEND_CLONE_NODES.set(BACKEND_CLONE_NODES.get() + n);
}

// ============================================================================
// Graph Backend Abstraction
// ============================================================================

/// `&mut T` from a heap backend handle, copying it first if it is shared.
///
/// **Phase 1 made this an `expect` and Phase 2 had to soften it**, and the
/// reason is worth keeping: in Phase 1 a `Memory` handle was provably unique,
/// because the only producer was a `Clone` that deep-copied into a fresh `Arc`.
/// Phase 2 introduced a second holder — the `base` of somebody else's overlay —
/// so the assertion started firing on five real Python tests
/// (`g.copy()` then writing through the *original*).
///
/// [`GraphBackend::ensure_writable`] is the fix and it runs at write entry, so
/// by the time any caller reaches here the handle is unique again and this is a
/// plain `Arc::get_mut`. `Arc::make_mut` is the fallback for a path that
/// somehow bypasses write entry: it deep-copies rather than mutating a backend
/// someone is reading, i.e. it fails **slow, never wrong** — the same direction
/// `rollback::journal_covers` and `forked::can_fork` take. A panic here would
/// have been a user-visible crash for a cost problem.
#[inline(always)]
pub(crate) fn unique_heap_backend<T: Clone>(handle: &mut Arc<T>) -> &mut T {
    Arc::make_mut(handle)
}

/// Graph storage backend. Four variants — heap-resident memory,
/// mmap-columnar-spilled mapped, CSR-on-disk, and a Phase 6 validation
/// wrapper that logs reads. Phase 5 promoted `MappedGraph` from a type
/// alias to a distinct struct so each backend owns its own
/// [`GraphRead`] / [`GraphWrite`] impl in
/// [`crate::graph::storage::impls`]; Phase 6 added
/// [`RecordingGraph`] as a live test of the trait surface. This
/// enum is now a 4-arm dumb dispatcher.
///
/// The `Recording` variant wraps any other `GraphBackend` — including,
/// in principle, another `Recording` — via
/// `Box<RecordingGraph<GraphBackend>>`. Its `is_memory` / `is_mapped`
/// / `is_disk` predicates forward to the inner backend so consumers
/// that switch on "what's the underlying storage" keep working
/// unchanged when wrapped.
pub enum GraphBackend {
    /// The `Arc` indirection is the whole point: it is what lets a fork share
    /// the heap graph with a reader instead of deep-copying it. A write regains
    /// a uniquely-owned handle first — see [`unique_heap_backend`].
    ///
    /// Reads pay one pointer deref (`&**g`) per backend dispatch. That single
    /// cost is what the ≤5% dispatch gate measures.
    Memory(Arc<MemoryGraph>),
    /// **A writer's copy-on-write overlay over a base a reader still holds.**
    /// Produced by `Clone` in place of a deep copy whenever the base qualifies
    /// (`forked::can_fork`), and collapsed back to `Memory` by
    /// [`try_compact`](Self::try_compact) the moment the reader drops.
    ///
    /// Every predicate below treats it as memory storage, because it *is* one —
    /// `is_memory`, `supports_undo_journal` and
    /// `supports_checkpoint_free_mutation` all answer exactly as the `Memory`
    /// arm it was forked from would. That is not a convenience: answering
    /// `false` to `supports_undo_journal` would send every statement taken
    /// while a view is held to `StatementCheckpoint::Clone`, i.e. an O(V+E)
    /// clone *per statement* — a worse cliff than the defect this variant
    /// removes (D2 risk R2).
    Forked(Box<ForkedGraph>),
    Mapped(Arc<MappedGraph>),
    Disk(Box<DiskGraph>),
    // Write-capture wrapper for the WAL. Introduced as a Phase 6
    // test-only validation wrapper, now the production backend of every
    // graph opened with `durable=True`: the binding wraps the loaded
    // backend via `wrap_backend_for_durability`, so each mutation that
    // passes the `GraphWrite` seam is buffered as a `RawOp` and flushed
    // to the log. Because it wraps the enum itself, it is
    // storage-agnostic — the capture layer is identical for a memory,
    // mapped, or disk graph underneath.
    Recording(Box<RecordingGraph<GraphBackend>>),
}

impl GraphBackend {
    /// An **owned** node record, for a caller that finishes with it inside its
    /// own frame — scans and filters, which drop each record immediately.
    ///
    /// This exists for the disk backend, where [`GraphRead::node_weight`]
    /// parks every record it builds in the per-query arena: a scan walking a
    /// million nodes retains a million records until its query ends
    /// (`storage/disk/query_arena.rs`). Materializing into the caller's frame
    /// keeps such a scan flat in memory and skips the arena mutex entirely.
    ///
    /// **Gate on [`GraphRead::is_disk`] before calling.** The heap backends
    /// already own their records and can only answer by *cloning*, which is
    /// pure loss for them — hence the branch at the call sites rather than a
    /// wholesale switch.
    #[inline]
    pub(crate) fn owned_node_data(&self, idx: NodeIndex) -> Option<NodeData> {
        match self {
            Self::Disk(g) => g.owned_node_data(idx),
            Self::Recording(rg) => rg.inner().owned_node_data(idx),
            _ => GraphRead::node_weight(self, idx).cloned(),
        }
    }

    #[inline]
    // Keep the established constructor-only backend API stable in this hardening pass.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        GraphBackend::Memory(Arc::new(MemoryGraph::new()))
    }

    /// Whether this backend is wrapped in the WAL write-capture layer, i.e.
    /// whether some owner is logging its mutations.
    ///
    /// The durable machinery reads and reaches into the capture layer from
    /// several places; those questions are answered here, on the dispatcher,
    /// rather than by re-matching the variant at each site.
    #[inline]
    pub(crate) fn is_recording(&self) -> bool {
        matches!(self, GraphBackend::Recording(_))
    }

    /// The write-capture layer, if this backend is wrapped in one.
    ///
    /// Read-side only, and therefore test-only: the production durable paths
    /// all *drain* the buffer and use [`recording_mut`](Self::recording_mut).
    /// This exists so the durable session's replay-before-wrap ordering has an
    /// observable — a graph wrapped before its WAL replay carries every
    /// replayed op in this buffer, which nothing else can see.
    #[cfg(test)]
    #[inline]
    pub(crate) fn recording(&self) -> Option<&RecordingGraph<GraphBackend>> {
        match self {
            GraphBackend::Recording(rg) => Some(rg),
            _ => None,
        }
    }

    /// Mutable access to the write-capture layer, if this backend is wrapped
    /// in one. The durable commit and checkpoint paths drain the op buffer
    /// through this.
    #[inline]
    pub(crate) fn recording_mut(&mut self) -> Option<&mut RecordingGraph<GraphBackend>> {
        match self {
            GraphBackend::Recording(rg) => Some(rg),
            _ => None,
        }
    }

    /// Wrap this backend in the write-capture layer, idempotently. See
    /// [`crate::graph::storage::recording::wrap_for_durability`] for the
    /// `DirGraph`-shaped entry point every binding calls.
    pub(crate) fn wrap_for_durability(&mut self) {
        self.wrap_for_capture();
        if let GraphBackend::Recording(rg) = self {
            rg.claim_wal_ownership();
        }
    }

    /// Wrap this backend in the write-capture layer **without** claiming
    /// write-ahead-log ownership, idempotently — the change-data-capture
    /// entry point (`graph::cdc::enable`).
    ///
    /// Same seam, different consumer: CDC derives events from the buffer the
    /// wrapper fills, but keeps no log, so it must not present itself as a
    /// durable owner. See [`RecordingGraph::is_wal_owner`].
    pub(crate) fn wrap_for_capture(&mut self) {
        if self.is_recording() {
            return;
        }
        let inner = std::mem::replace(self, GraphBackend::new());
        *self = GraphBackend::Recording(Box::new(RecordingGraph::new(inner)));
    }

    /// Remove the write-capture layer, **unless** a write-ahead log owns it.
    ///
    /// The inverse of [`wrap_for_capture`](Self::wrap_for_capture), for
    /// `cdc::disable`: capture is not free — a wrapped backend buffers a
    /// `RawOp` per mutation and gives up the checkpoint-free mutation fast
    /// path ([`supports_checkpoint_free_mutation`](Self::supports_checkpoint_free_mutation))
    /// — so turning capture off has to actually turn it off, or "disable"
    /// would leave a permanent tax behind.
    ///
    /// Refuses on a WAL-owned wrapper because unwrapping one silently stops
    /// logging: the graph would keep committing and the log would keep
    /// claiming to describe it. Any ops still buffered are dropped with the
    /// wrapper, which is correct precisely because nothing owns them — a
    /// WAL-owned buffer is never reached here.
    pub(crate) fn unwrap_capture_if_unowned(&mut self) {
        let GraphBackend::Recording(recording) = self else {
            return;
        };
        if recording.is_wal_owner() {
            return;
        }
        let inner = std::mem::replace(recording.inner_mut(), GraphBackend::new());
        *self = inner;
    }

    /// Whether this backend's capture layer is owned by a write-ahead log
    /// (as opposed to being installed for change data capture alone, or
    /// absent). See [`RecordingGraph::is_wal_owner`].
    #[inline]
    pub(crate) fn is_wal_owner(&self) -> bool {
        matches!(self, GraphBackend::Recording(rg) if rg.is_wal_owner())
    }

    /// Whether a proven-infallible mutation may commit without a full rollback
    /// checkpoint. Recording/durable wrappers deliberately return false even
    /// when their inner backend is memory: their post-write WAL lifecycle is a
    /// distinct boundary and keeps the conservative checkpoint path.
    #[inline]
    pub(crate) fn supports_checkpoint_free_mutation(&self) -> bool {
        // `Forked` answers exactly as the `Memory` it forked from: before D2
        // Phase 2 a held view produced a deep-cloned `Memory` here, which
        // returned true, so answering false would *newly* charge a checkpoint
        // to every single-node CREATE taken while a view is held. Pinned by
        // `rollback_tests::forked_statements_copy_zero_nodes`.
        matches!(self, GraphBackend::Memory(_) | GraphBackend::Forked(_))
    }

    /// Whether this backend can capture inverse operations for a
    /// statement-scoped undo journal, i.e. whether rollback can avoid the
    /// whole-graph clone.
    ///
    /// Every petgraph-backed backend can. `Memory` and `Mapped` both hold a
    /// heap `StableDiGraph<NodeData, EdgeData>` as `inner`, and every
    /// `UndoEntry` variant is keyed on the `NodeIndex`/`EdgeIndex` that graph
    /// hands out, so the capture seam is the same one for both. "Larger than
    /// RAM" is loose for `Mapped`: what `StorageMode::Mapped` changes is where
    /// *properties* live (`memory_limit = Some(0)` spills the columnar store
    /// to mmap), not the node/edge graph, which stays heap-resident.
    ///
    /// `Disk` cannot, and that is the whole of the remaining veto. It has no
    /// petgraph at all — it mutates a CSR + mmap layout through generation
    /// overlays and arena-staged writes, its slots carry no `NodeIndex`
    /// identity for an entry to name, and it has no free list whose LIFO
    /// ordering reverse replay could exploit to restore slot identity. So a
    /// disk graph keeps the clone checkpoint (see `dir_graph/rollback.rs`):
    /// every mutating statement on it still opens an O(V+E)
    /// `fork_transaction()` whole-graph checkpoint, the pre-journal behaviour,
    /// and its statement-rollback cost scales with graph size. Mirrored in the
    /// user-facing storage-mode guide (`docs/python/core-concepts.md`).
    ///
    /// `Recording` forwards to whatever it wraps, so a durable graph
    /// participates exactly when its underlying backend would:
    /// `Recording(Memory)` and `Recording(Mapped)` journal,
    /// `Recording(Disk)` keeps the clone. Durability and rollback strategy are
    /// independent concerns.
    ///
    /// **This is the only remaining veto term in `journal_covers`** — every
    /// other term was retired as the journal grew to cover saved, indexed and
    /// columnar graphs.
    #[inline]
    pub(crate) fn supports_undo_journal(&self) -> bool {
        match self {
            GraphBackend::Memory(_) | GraphBackend::Mapped(_) => true,
            // MUST be true — see the `Forked` variant doc (D2 risk R2). Every
            // `UndoEntry` is keyed on a `NodeIndex`/`EdgeIndex`, which the
            // overlay still hands out, and reversal goes through
            // `ForkedGraph`'s own `GraphWrite`, so entries land in the overlay
            // and never touch the shared base (D2 risk R3).
            GraphBackend::Forked(_) => true,
            GraphBackend::Recording(rg) => rg.inner().supports_undo_journal(),
            GraphBackend::Disk(_) => false,
        }
    }

    /// `true` while this backend is a copy-on-write overlay over a base a
    /// reader still holds.
    ///
    /// Public as a **diagnostic**: it is the one cheap, non-timing observable
    /// that distinguishes D2's copy-on-write fork from the whole-graph clone it
    /// replaced, and from a compaction that failed to fold back. Bindings
    /// expose it for regression tests (`kglite._backend_is_forked`); nothing in
    /// the engine's behaviour depends on a caller reading it.
    #[inline]
    pub fn is_forked(&self) -> bool {
        match self {
            GraphBackend::Forked(_) => true,
            GraphBackend::Recording(rg) => rg.inner().is_forked(),
            _ => false,
        }
    }

    /// Make this backend safe to mutate in place, given that a `Memory` arm's
    /// `Arc` can be shared — as the *base* of somebody else's overlay.
    ///
    /// The shape that needs it: `g.copy()` (or a transaction snapshot) forks
    /// **from** `g`, so the fork's `base` and `g`'s own `Memory(_)` are now the
    /// same allocation. `g` is still a uniquely-owned `Arc<DirGraph>`, so
    /// `Arc::make_mut` at the `DirGraph` level does nothing and the write would
    /// go straight into a backend the fork is reading. Before Phase 2 that could
    /// not happen — a `Memory` handle was always unique — which is why the whole
    /// Python suite passed Phase 1 with `unique_heap_backend`'s assertion armed,
    /// and why these five tests are what found it.
    ///
    /// The resolution is symmetric with the fork itself: `g` becomes an overlay
    /// over the shared base too. Both graphs then read the same untouched base
    /// and write their own deltas, and whichever outlives the other compacts it.
    /// A base that cannot be forked cheaply falls back to the deep copy.
    ///
    /// Called at write entry alongside [`try_compact`](Self::try_compact); one
    /// `Arc::get_mut` probe when nothing is shared, which is the steady state.
    pub(crate) fn ensure_writable(&mut self) {
        // `Arc::strong_count`, not `Arc::get_mut`: a match guard borrows the
        // scrutinee immutably, and this probe has to run *before* the arm that
        // would move out of `self`.
        let shared = match self {
            GraphBackend::Recording(rg) => {
                rg.inner_mut().ensure_writable();
                return;
            }
            GraphBackend::Memory(g) => Arc::strong_count(g) > 1 || Arc::weak_count(g) > 0,
            GraphBackend::Mapped(g) => Arc::strong_count(g) > 1 || Arc::weak_count(g) > 0,
            GraphBackend::Forked(_) | GraphBackend::Disk(_) => return,
        };
        if !shared {
            return;
        }
        match std::mem::replace(self, GraphBackend::new()) {
            GraphBackend::Memory(base) => {
                *self = if can_fork(&base) {
                    GraphBackend::Forked(Box::new(ForkedGraph::new(base)))
                } else {
                    #[cfg(test)]
                    note_nodes_copied(base.inner().node_count());
                    GraphBackend::Memory(Arc::new(base.deep_clone()))
                };
            }
            GraphBackend::Mapped(base) => {
                // Mapped keeps the deep copy this phase — D2 R5 says do Mapped
                // with Memory or leave it explicitly on the old path with a
                // named test, not ambiguously half-done. This is that choice,
                // and `mapped_statements_copy_zero_nodes` still pins its cost.
                #[cfg(test)]
                note_nodes_copied(base.inner().node_count());
                *self = GraphBackend::Mapped(Arc::new(base.deep_clone()));
            }
            other => *self = other,
        }
    }

    /// Fold an overlay back into its base when this writer is the last holder.
    ///
    /// **Called at write entry** (`handle::make_dir_graph_mut_preserving_lineage`),
    /// which is the earliest moment a writer can observe that the reader has
    /// gone — `Arc::get_mut` succeeding *is* that observation. So "hold a view,
    /// write, drop the view, write again" returns to the flat representation on
    /// the very next write, with no timer and no bookkeeping. A no-op on every
    /// other variant, and a no-op on `Forked` while a reader is still live.
    pub(crate) fn try_compact(&mut self) {
        if let GraphBackend::Recording(rg) = self {
            rg.inner_mut().try_compact();
            return;
        }
        if !matches!(self, GraphBackend::Forked(_)) {
            return;
        }
        let GraphBackend::Forked(forked) = std::mem::replace(self, GraphBackend::new()) else {
            unreachable!("just matched Forked")
        };
        *self = match forked.try_compact() {
            Ok(memory) => GraphBackend::Memory(Arc::new(memory)),
            Err(still_forked) => GraphBackend::Forked(still_forked),
        };
    }

    /// Collapse an overlay to a plain `Memory` backend **unconditionally**,
    /// deep-copying the base if a reader still holds it.
    ///
    /// The escape hatch for the three writes an overlay cannot express
    /// (`add_edge` / `remove_node` / `remove_edge`, which rewrite existing
    /// nodes' petgraph adjacency) and for the handful of whole-graph operations
    /// that need one concrete `StableDiGraph`. Cost is the pre-D2 fork, paid on
    /// that write only; every other write stays O(changes).
    pub(crate) fn flatten_fork(&mut self) {
        if let GraphBackend::Recording(rg) = self {
            rg.inner_mut().flatten_fork();
            return;
        }
        // Prefer the free path: if the reader has already gone, this is a fold
        // rather than a copy.
        self.try_compact();
        if !matches!(self, GraphBackend::Forked(_)) {
            return;
        }
        let GraphBackend::Forked(mut forked) = std::mem::replace(self, GraphBackend::new()) else {
            unreachable!("just matched Forked")
        };
        *self = GraphBackend::Memory(Arc::new(forked.materialise()));
    }

    /// Install a fresh undo journal on a petgraph-backed backend. No-op on
    /// backends that do not support one — callers gate on
    /// [`Self::supports_undo_journal`] first.
    #[inline]
    pub(crate) fn begin_undo(&mut self) {
        match self {
            GraphBackend::Memory(g) => unique_heap_backend(g).begin_undo(),
            GraphBackend::Forked(g) => g.begin_undo(),
            GraphBackend::Mapped(g) => unique_heap_backend(g).begin_undo(),
            GraphBackend::Recording(rg) => rg.inner_mut().begin_undo(),
            GraphBackend::Disk(_) => {}
        }
    }

    /// Uninstall and return the undo journal, ending capture.
    #[inline]
    pub(crate) fn take_undo(&mut self) -> Option<Box<UndoJournal>> {
        match self {
            GraphBackend::Memory(g) => unique_heap_backend(g).take_undo(),
            GraphBackend::Forked(g) => g.take_undo(),
            GraphBackend::Mapped(g) => unique_heap_backend(g).take_undo(),
            GraphBackend::Recording(rg) => rg.inner_mut().take_undo(),
            GraphBackend::Disk(_) => None,
        }
    }

    /// Mutable access to the active undo journal, for the `DirGraph`-level
    /// capture seam (inverted-index and timeseries edits, which live above
    /// storage and so cannot be seen from a `GraphWrite` impl).
    #[inline]
    pub(crate) fn undo_journal_mut(&mut self) -> Option<&mut UndoJournal> {
        match self {
            GraphBackend::Memory(g) => unique_heap_backend(g).undo_journal_mut(),
            GraphBackend::Forked(g) => g.undo_journal_mut(),
            GraphBackend::Mapped(g) => unique_heap_backend(g).undo_journal_mut(),
            GraphBackend::Recording(rg) => rg.inner_mut().undo_journal_mut(),
            GraphBackend::Disk(_) => None,
        }
    }

    /// Number of raw WAL-capture ops buffered by a `Recording` wrapper, or
    /// `None` for a backend that captures nothing. Paired with
    /// [`Self::truncate_recorded_ops`] so a rolled-back statement's writes
    /// never reach the write-ahead log.
    #[inline]
    pub(crate) fn recorded_ops_len(&self) -> Option<usize> {
        match self {
            GraphBackend::Recording(rg) => Some(rg.ops_len()),
            _ => None,
        }
    }

    /// Drop buffered WAL-capture ops past `len`, discarding the ops a
    /// rolled-back statement produced while keeping any earlier, still-unflushed
    /// ones.
    #[inline]
    pub(crate) fn truncate_recorded_ops(&mut self, len: usize) {
        if let GraphBackend::Recording(rg) = self {
            rg.truncate_ops(len);
        }
    }

    /// Transfer writer-lineage authority to an already-cloned child that keeps
    /// the parent's runtime identity (transaction or Arc copy-on-write view).
    /// Generic `Clone` never transfers that authority on its own.
    pub(crate) fn adopt_shared_writer_lineage(&mut self, parent: &Self) {
        if let (GraphBackend::Disk(child), GraphBackend::Disk(parent)) = (self, parent) {
            child.adopt_writer_lineage(parent);
        }
    }

    /// Give an explicit copy private writer authority while retaining any
    /// mutation-workspace files needed to reproduce the parent's current
    /// logical state.
    pub(crate) fn detach_independent_copy(&mut self, parent: &Self) {
        if let (GraphBackend::Disk(child), GraphBackend::Disk(parent)) = (self, parent) {
            child.detach_for_independent_copy(parent);
        }
    }

    /// Edge-storage observability for `graph_info()`: `(edges_mapped,
    /// edge_property_overlay_rows)`.
    ///
    /// Only the disk backend has a CSR or an edge-property overlay; every
    /// other backend keeps its edges in the heap `StableDiGraph` and reports
    /// `(false, 0)`.
    pub(crate) fn edge_storage_info(&self) -> (bool, usize) {
        match self {
            GraphBackend::Disk(g) => (g.csr_is_mapped(), g.edge_property_overlay_len()),
            GraphBackend::Recording(rg) => rg.inner().edge_storage_info(),
            GraphBackend::Memory(_) | GraphBackend::Mapped(_) | GraphBackend::Forked(_) => {
                (false, 0)
            }
        }
    }

    /// Hold the disk materialization arenas for one read-query lifetime.
    /// Heap/mapped backends do not materialize through shared arenas.
    pub(crate) fn begin_query(&self) -> Option<DiskQueryGuard> {
        match self {
            GraphBackend::Disk(graph) => Some(graph.begin_query()),
            GraphBackend::Recording(graph) => graph.inner().begin_query(),
            GraphBackend::Memory(_) | GraphBackend::Mapped(_) | GraphBackend::Forked(_) => None,
        }
    }

    /// Record that node `idx` was upserted, for the WAL capture wrapper.
    /// No-op unless this is the [`GraphBackend::Recording`] backend. Used by
    /// mutation paths that write through a side channel (the columnar master
    /// `ColumnStore`) and so bypass the recorded
    /// [`GraphWrite::node_weight_mut`] — without this the mutation would not
    /// be captured for the WAL at all (the recorded path only sees the
    /// silent handle-refresh sweep).
    #[inline]
    pub fn note_recorded_node_upsert(&mut self, idx: NodeIndex) {
        if let GraphBackend::Recording(rg) = self {
            rg.note_node_upsert(idx);
        }
    }

    /// Turn before-image capture on or off on the write-capture wrapper.
    /// No-op when there is no wrapper — a graph with no capture layer has
    /// nothing to enrich.
    #[inline]
    pub(crate) fn set_capture_before(&mut self, on: bool) {
        if let GraphBackend::Recording(rg) = self {
            rg.set_capture_before(on);
        }
    }

    /// Whether writes on this backend capture before-images.
    ///
    /// The gate the two **side-channel choke points** test before doing any
    /// work: both sit on hot write paths and must cost a bool read, not a
    /// state read, when enrichment is off (which is the default).
    #[inline]
    pub fn captures_before_images(&self) -> bool {
        match self {
            GraphBackend::Recording(rg) => rg.captures_before(),
            _ => false,
        }
    }

    /// Whether node `idx` still needs its first-touch before-image.
    ///
    /// Lets a choke point skip the whole-entity read for every write after the
    /// first to the same entity in one commit.
    #[inline]
    pub fn needs_node_before_image(&self, idx: NodeIndex) -> bool {
        match self {
            GraphBackend::Recording(rg) => rg.needs_node_before(idx),
            _ => false,
        }
    }

    /// Hand a node's pre-write state to the capture wrapper, from a site that
    /// mutates outside the `GraphWrite` seam. Must be called **before** that
    /// site's write. See
    /// [`RecordingGraph::note_node_before`](crate::graph::storage::recording::RecordingGraph::note_node_before).
    #[inline]
    pub fn note_node_before_image(
        &mut self,
        idx: NodeIndex,
        image: crate::graph::storage::recording::BeforeImage,
    ) {
        if let GraphBackend::Recording(rg) = self {
            rg.note_node_before(idx, image);
        }
    }

    /// Fill in the label half of a node's already-captured before-image.
    #[inline]
    pub fn backfill_node_before_labels(&mut self, idx: NodeIndex, labels: Vec<String>) {
        if let GraphBackend::Recording(rg) = self {
            rg.backfill_node_before_labels(idx, labels);
        }
    }

    /// Record that node `idx`'s secondary labels changed, for the WAL
    /// capture wrapper. No-op unless this is the
    /// [`GraphBackend::Recording`] backend.
    ///
    /// Secondary labels live in `DirGraph::secondary_label_index`, above
    /// this backend — `NodeData` carries none — so *no* `GraphWrite` call
    /// describes a label change and the recorded seam cannot infer one.
    /// `DirGraph`'s label choke points call this instead; without it a
    /// durable graph silently lost every `CREATE (n:A:B)` / `SET n:B` on
    /// WAL replay while keeping the node's properties.
    #[inline]
    pub fn note_recorded_node_labels(&mut self, idx: NodeIndex) {
        if let GraphBackend::Recording(rg) = self {
            rg.note_node_labels(idx);
        }
    }

    /// Swap in a rebuilt heap petgraph, **preserving this backend's variant
    /// and any write-capture wrapper around it**. Returns `false` for `Disk`,
    /// whose CSR arrays are not a `StableDiGraph` and cannot be replaced this
    /// way; the caller must treat that as "not rebuilt".
    ///
    /// Exists because `DirGraph::vacuum` rebuilds the graph with contiguous
    /// indices and used to assign `GraphBackend::Memory(...)` unconditionally.
    /// That silently did two damaging things: it downgraded a `Mapped` graph
    /// to heap storage, and — worse — it *dropped the `Recording` wrapper*, so
    /// a durable graph stopped write-ahead logging for the rest of the
    /// session with no error. Rebuilding through this method keeps both
    /// properties.
    ///
    /// Note the wrapper is preserved but its op buffer is not meaningful
    /// across a rebuild: buffered ops are keyed by `NodeIndex`, and a vacuum
    /// remaps every index. Callers must flush the log *before* vacuuming.
    pub(crate) fn replace_heap_graph(&mut self, new: StableDiGraph<NodeData, EdgeData>) -> bool {
        match self {
            // The column stores are *owned* state, not a lazy index: carry
            // them across the swap or every columnar node's properties vanish
            // (`vacuum` is the caller that would otherwise lose them).
            GraphBackend::Memory(g) => {
                let g = unique_heap_backend(g);
                let stores = std::mem::take(&mut g.column_stores);
                *g = MemoryGraph::from_graph(new);
                g.column_stores = stores;
                true
            }
            GraphBackend::Mapped(g) => {
                let g = unique_heap_backend(g);
                let stores = std::mem::take(&mut g.column_stores);
                *g = MappedGraph::from_graph(new);
                g.column_stores = stores;
                true
            }
            GraphBackend::Recording(rg) => rg.inner_mut().replace_heap_graph(new),
            // Reached only through `vacuum`, which holds `&mut Arc<DirGraph>` and
            // therefore compacts at write entry before it gets here.
            GraphBackend::Forked(_) => unreachable!("replace_heap_graph on a forked backend"),
            GraphBackend::Disk(_) => false,
        }
    }

    /// Move the heap `StableDiGraph` out of the backend, leaving an empty one
    /// in its place. `None` on **disk**, whose CSR arrays are not a
    /// `StableDiGraph` — the same "not rebuilt" signal
    /// [`replace_heap_graph`](Self::replace_heap_graph) returns `false` for.
    ///
    /// Exists for `DirGraph::vacuum`, which used to deep-clone every node and
    /// edge weight into the compacted graph and then drop the originals when
    /// the backend was replaced. Owning the old graph lets it *relocate* the
    /// weights instead. The backend is left holding an empty graph and its
    /// now-stale derived caches; the sole caller replaces it with the
    /// compacted graph a few statements later and nothing reads it in
    /// between, exactly as during the old clone loop.
    pub(crate) fn take_heap_graph(&mut self) -> Option<StableDiGraph<NodeData, EdgeData>> {
        match self {
            GraphBackend::Memory(g) => Some(std::mem::take(&mut unique_heap_backend(g).inner)),
            GraphBackend::Mapped(g) => Some(std::mem::take(&mut unique_heap_backend(g).inner)),
            GraphBackend::Recording(rg) => rg.inner_mut().take_heap_graph(),
            GraphBackend::Forked(_) => unreachable!("take_heap_graph on a forked backend"),
            GraphBackend::Disk(_) => None,
        }
    }

    /// The inner `StableDiGraph` when this is a plain heap `Memory` backend,
    /// and `None` for every other variant — including a D2 copy-on-write
    /// overlay, whose nodes are base⊕overlay and so are not one petgraph.
    ///
    /// A *fast-path* probe: callers must have a correct generic fallback for
    /// `None`. Exhaustive by construction, so a new variant has to opt in here
    /// rather than silently joining the fast path.
    #[inline]
    pub(crate) fn plain_memory_digraph(&self) -> Option<&StableDiGraph<NodeData, EdgeData>> {
        match self {
            GraphBackend::Memory(g) => Some(g.inner()),
            GraphBackend::Forked(_)
            | GraphBackend::Mapped(_)
            | GraphBackend::Recording(_)
            | GraphBackend::Disk(_) => None,
        }
    }

    /// Borrow the inner heap `StableDiGraph` for petgraph algorithms
    /// (e.g. `kosaraju_scc`) that require concrete petgraph types.
    /// Disk panics — callers must gate on [`GraphRead::is_disk`].
    /// `Recording` forwards to the wrapped backend.
    #[inline]
    pub fn as_stable_digraph(&self) -> &StableDiGraph<NodeData, EdgeData> {
        match self {
            GraphBackend::Memory(g) => g.inner(),
            GraphBackend::Mapped(g) => g.inner(),
            // Same contract as Disk: callers gate first. `connected_components`
            // routes a forked graph to the generic `GraphRead` traversal, which is
            // the same fallback the disk backend already uses.
            GraphBackend::Forked(_) => {
                unimplemented!("Forked backend: as_stable_digraph — gate on is_forked()")
            }
            GraphBackend::Disk(_) => unimplemented!("Disk backend: as_stable_digraph"),
            GraphBackend::Recording(rg) => rg.inner().as_stable_digraph(),
        }
    }

    /// Closure-based hot-path iteration over all live edges, yielding
    /// `(source, target, connection_type)` per edge.
    ///
    /// Avoids the `Box<dyn Iterator>` + virtual `.next()` dispatch that
    /// [`GraphRead::edge_endpoint_keys`] requires. Monomorphises per
    /// backend at the call site, so the compiler fully inlines the hot
    /// loop. Use this from any code path that walks every edge on a
    /// large graph (e.g. `compute_type_connectivity`, cache rebuild,
    /// bulk index builds). 863M-edge benchmarks show ~40–90 s savings
    /// per sweep vs the boxed-iterator path on Wikidata-scale graphs.
    ///
    /// The `Recording` variant forwards to its wrapped backend without
    /// recording — it logs only the trait-path methods.
    #[inline(always)]
    pub fn for_each_edge_endpoint_key<F>(&self, mut f: F)
    where
        F: FnMut(NodeIndex, NodeIndex, InternedKey),
    {
        use petgraph::visit::{EdgeRef, IntoEdgeReferences};
        match self {
            GraphBackend::Memory(g) => {
                for er in g.inner().edge_references() {
                    let w = er.weight();
                    f(er.source(), er.target(), w.connection_type);
                }
            }
            GraphBackend::Mapped(g) => {
                for er in g.inner().edge_references() {
                    let w = er.weight();
                    f(er.source(), er.target(), w.connection_type);
                }
            }
            GraphBackend::Disk(g) => {
                let dg = g.as_ref();
                for i in 0..dg.next_edge_idx {
                    let ep = dg.edge_endpoint(i as usize);
                    if ep.source == crate::graph::storage::disk::csr::TOMBSTONE_EDGE {
                        continue;
                    }
                    f(
                        NodeIndex::new(ep.source as usize),
                        NodeIndex::new(ep.target as usize),
                        InternedKey::from_u64(ep.connection_type),
                    );
                }
            }
            // No overlay edge exists (module doc), so the base carries every edge.
            GraphBackend::Forked(g) => {
                for er in g.base_stable_digraph().edge_references() {
                    let w = er.weight();
                    f(er.source(), er.target(), w.connection_type);
                }
            }
            GraphBackend::Recording(rg) => {
                rg.inner().for_each_edge_endpoint_key(f);
            }
        }
    }

    /// Iterate only edges whose connection type matches `conn_type`,
    /// yielding `(src, tgt, edge_idx, properties)` per match.
    ///
    /// The callback returns `true` to continue or `false` to stop — so
    /// callers collecting a bounded prefix (sample edges, first match)
    /// don't pay for the rest of the matches.
    ///
    /// Avoids the disk backend's per-edge `Box<EdgeData>` arena push by
    /// reading `edge_endpoints` + `edge_properties` directly. On
    /// Memory/Mapped the petgraph iterator already hands out `&EdgeData`
    /// references into the resident storage, so there is no arena cost.
    ///
    /// On the disk backend, complexity is O(matching edges) thanks to
    /// the persisted `conn_type_index_*` inverted index — not O(all
    /// edges) as a filtered `edge_references()` sweep would be.
    ///
    /// `properties` is the empty slice when the edge has no custom
    /// properties (common case for topology-heavy graphs).
    #[inline(always)]
    pub fn for_each_edge_of_conn_type<F>(&self, conn_type: InternedKey, mut f: F)
    where
        F: FnMut(NodeIndex, NodeIndex, u32, &[(InternedKey, Value)]) -> bool,
    {
        use petgraph::visit::{EdgeRef, IntoEdgeReferences};
        let ct_u64 = conn_type.as_u64();
        match self {
            GraphBackend::Memory(g) => {
                for er in g.inner().edge_references() {
                    let w = er.weight();
                    if w.connection_type == conn_type
                        && !f(
                            er.source(),
                            er.target(),
                            er.id().index() as u32,
                            w.properties.as_slice(),
                        )
                    {
                        return;
                    }
                }
            }
            GraphBackend::Mapped(g) => {
                for er in g.inner().edge_references() {
                    let w = er.weight();
                    if w.connection_type == conn_type
                        && !f(
                            er.source(),
                            er.target(),
                            er.id().index() as u32,
                            w.properties.as_slice(),
                        )
                    {
                        return;
                    }
                }
            }
            GraphBackend::Disk(g) => {
                let dg = g.as_ref();
                dg.for_each_edge_of_conn_type(ct_u64, |src, tgt, edge_idx| {
                    // edge_properties_at returns Cow; bind to extend its
                    // lifetime across the callback, then deref to a slice.
                    let props_cow = dg.edge_properties_at(edge_idx);
                    let props: &[(
                        crate::graph::schema::InternedKey,
                        crate::datatypes::values::Value,
                    )] = props_cow.as_deref().unwrap_or(&[]);
                    f(src, tgt, edge_idx, props)
                });
            }
            GraphBackend::Forked(g) => {
                for er in g.base_stable_digraph().edge_references() {
                    let w = er.weight();
                    if w.connection_type == conn_type
                        && !f(
                            er.source(),
                            er.target(),
                            er.id().index() as u32,
                            w.properties.as_slice(),
                        )
                    {
                        return;
                    }
                }
            }
            GraphBackend::Recording(rg) => {
                rg.inner().for_each_edge_of_conn_type(conn_type, f);
            }
        }
    }

    /// Borrow the default heap backend's immutable peer-count histogram.
    /// Other backends keep their existing owned-result fallback.
    pub(crate) fn cached_edge_counts_grouped_by_peer(
        &self,
        conn_type: InternedKey,
        dir: petgraph::Direction,
        deadline: Option<std::time::Instant>,
    ) -> Result<Option<Arc<HashMap<u32, i64>>>, String> {
        Ok(match self {
            GraphBackend::Memory(graph) => {
                let counts = graph.ensure_peer_counts_with_deadline(conn_type, deadline)?;
                Some(match dir {
                    petgraph::Direction::Outgoing => Arc::clone(&counts.by_target),
                    petgraph::Direction::Incoming => Arc::clone(&counts.by_source),
                })
            }
            GraphBackend::Recording(graph) => graph
                .inner()
                .cached_edge_counts_grouped_by_peer(conn_type, dir, deadline)?,
            // Cold by design: the fork resets peer counts rather than sharing the
            // base's, so the writer never publishes a count into a reader's
            // snapshot (D2 R4). The owned fallback is correct, just uncached.
            GraphBackend::Forked(_) | GraphBackend::Mapped(_) | GraphBackend::Disk(_) => None,
        })
    }
}

// -- Index traits --

impl std::ops::Index<NodeIndex> for GraphBackend {
    type Output = NodeData;
    #[inline]
    fn index(&self, index: NodeIndex) -> &NodeData {
        match self {
            GraphBackend::Memory(g) => &g.inner()[index],
            GraphBackend::Mapped(g) => &g.inner()[index],
            GraphBackend::Forked(g) => {
                GraphRead::node_weight(g.as_ref(), index).expect("Index on a missing node")
            }
            GraphBackend::Disk(dg) => &dg[index],
            GraphBackend::Recording(rg) => &rg.inner()[index],
        }
    }
}

impl std::ops::Index<EdgeIndex> for GraphBackend {
    type Output = EdgeData;
    #[inline]
    fn index(&self, index: EdgeIndex) -> &EdgeData {
        match self {
            GraphBackend::Memory(g) => &g.inner()[index],
            GraphBackend::Mapped(g) => &g.inner()[index],
            GraphBackend::Forked(g) => {
                GraphRead::edge_weight(g.as_ref(), index).expect("Index on a missing edge")
            }
            GraphBackend::Disk(dg) => &dg[index],
            GraphBackend::Recording(rg) => &rg.inner()[index],
        }
    }
}

// -- Clone --

impl Clone for GraphBackend {
    fn clone(&self) -> Self {
        #[cfg(test)]
        BACKEND_CLONE_COUNT.set(BACKEND_CLONE_COUNT.get() + 1);
        match self {
            // **The fork site.** Instead of deep-copying every node and edge,
            // hand the writer an overlay over the same base. The reader's
            // `Arc<MemoryGraph>` is left byte-for-byte untouched.
            //
            // `can_fork` is the slot-identity precondition (free lists provably
            // empty, so the fold-back reproduces the overlay's indices); a base
            // that fails it keeps the deep copy — slower, never wrong.
            //
            // ⚠ `deep_clone()` on the fallback, never `g.clone()`: `g` is an
            // `Arc` handle, so `.clone()` on it is a refcount bump — one
            // character away, and it would share a backend that every later
            // write mutates in place under the reader.
            GraphBackend::Memory(g) if can_fork(g) => {
                GraphBackend::Forked(Box::new(ForkedGraph::new(Arc::clone(g))))
            }
            GraphBackend::Memory(g) => {
                #[cfg(test)]
                note_nodes_copied(g.inner().node_count());
                GraphBackend::Memory(Arc::new(g.deep_clone()))
            }
            GraphBackend::Mapped(g) => {
                #[cfg(test)]
                note_nodes_copied(g.inner().node_count());
                GraphBackend::Mapped(Arc::new(g.deep_clone()))
            }
            // Forking a fork: same base, only the delta is duplicated.
            GraphBackend::Forked(g) => {
                #[cfg(test)]
                note_nodes_copied(g.overlay_node_count());
                GraphBackend::Forked(Box::new((**g).clone()))
            }
            GraphBackend::Disk(dg) => {
                #[cfg(test)]
                note_nodes_copied(dg.node_count());
                GraphBackend::Disk(dg.clone())
            }
            GraphBackend::Recording(rg) => GraphBackend::Recording(Box::new((**rg).clone())),
        }
    }
}

// -- Serialize / Deserialize --
// Delegates to StableDiGraph so the binary format is identical to before.

impl Serialize for GraphBackend {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            GraphBackend::Memory(g) => g.serialize(serializer),
            GraphBackend::Mapped(g) => g.serialize(serializer),
            // Serialization needs one concrete `StableDiGraph`, so the overlay is
            // folded into a throwaway copy. O(V+E) — but so is writing the file,
            // and the bytes are identical to the unforked graph's: this variant is
            // a pure in-memory representation with **no** `.kgl` format impact.
            GraphBackend::Forked(g) => g.to_memory_graph().serialize(serializer),
            GraphBackend::Disk(_) => Err(serde::ser::Error::custom(
                "Disk backend does not support serialization",
            )),
            // Validation wrapper is transparent — serialize as the
            // wrapped backend. Recursively hits the Disk error arm
            // if the wrapped backend is Disk.
            GraphBackend::Recording(rg) => rg.inner().serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for GraphBackend {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let g = StableDiGraph::<NodeData, EdgeData>::deserialize(deserializer)?;
        Ok(GraphBackend::Memory(Arc::new(MemoryGraph::from_graph(g))))
    }
}

// -- Debug --

impl std::fmt::Debug for GraphBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphBackend::Memory(g) => write!(
                f,
                "Memory({} nodes, {} edges)",
                g.node_count(),
                g.edge_count()
            ),
            GraphBackend::Mapped(g) => write!(
                f,
                "Mapped({} nodes, {} edges)",
                g.node_count(),
                g.edge_count()
            ),
            GraphBackend::Forked(g) => write!(f, "{g:?}"),
            GraphBackend::Disk(_) => write!(f, "Disk(placeholder)"),
            GraphBackend::Recording(rg) => write!(f, "Recording({:?})", rg.inner()),
        }
    }
}

// ============================================================================
// GraphRead / GraphWrite dispatcher impls
//
// Phase 5 shrank these to dumb 3-arm dispatchers. The real impls live
// on each backend in `src/graph/storage/impls.rs`. The inherent
// `impl GraphBackend` method blocks that used to host these bodies are
// deleted — every caller either uses the trait (most of the codebase)
// or goes through the per-backend impl directly.
// ============================================================================

use crate::datatypes::values::Value;
use std::collections::HashMap;

impl GraphRead for GraphBackend {
    type NodeIndicesIter<'a> = crate::graph::core::iterators::GraphNodeIndices<'a>;
    type EdgeIndicesIter<'a> = crate::graph::core::iterators::GraphEdgeIndices<'a>;
    type EdgesIter<'a> = crate::graph::core::iterators::GraphEdges<'a>;
    type EdgeReferencesIter<'a> = crate::graph::core::iterators::GraphEdgeReferences<'a>;
    type EdgesConnectingIter<'a> = crate::graph::core::iterators::GraphEdgesConnecting<'a>;
    type NeighborsIter<'a> = crate::graph::core::iterators::GraphNeighbors<'a>;

    /// Dispatch **once**, not twice.
    ///
    /// The trait default calls `self.node_weight(idx)` and
    /// `self.column_store(..)`, each of which is a four-arm match on this enum
    /// — so a scan paid two dispatches per node on top of the store probe.
    /// Matching here and delegating to the concrete backend's `node_view` lets
    /// the probe inline into the same call.
    #[inline]
    fn node_view(&self, idx: NodeIndex) -> Option<crate::graph::storage::NodeView<'_>> {
        match self {
            Self::Memory(g) => GraphRead::node_view(&**g, idx),
            Self::Forked(g) => GraphRead::node_view(g.as_ref(), idx),
            Self::Mapped(g) => GraphRead::node_view(&**g, idx),
            Self::Disk(g) => GraphRead::node_view(g.as_ref(), idx),
            Self::Recording(rg) => GraphRead::node_view(rg.as_ref(), idx),
        }
    }

    #[inline]
    fn column_store(&self, type_key: InternedKey) -> Option<&std::sync::Arc<ColumnStore>> {
        match self {
            Self::Memory(g) => GraphRead::column_store(&**g, type_key),
            Self::Forked(g) => GraphRead::column_store(g.as_ref(), type_key),
            Self::Mapped(g) => GraphRead::column_store(&**g, type_key),
            Self::Disk(g) => GraphRead::column_store(g.as_ref(), type_key),
            Self::Recording(rg) => GraphRead::column_store(rg.as_ref(), type_key),
        }
    }

    fn column_stores_iter(
        &self,
    ) -> Box<dyn Iterator<Item = (InternedKey, &std::sync::Arc<ColumnStore>)> + '_> {
        match self {
            Self::Memory(g) => GraphRead::column_stores_iter(&**g),
            Self::Forked(g) => GraphRead::column_stores_iter(g.as_ref()),
            Self::Mapped(g) => GraphRead::column_stores_iter(&**g),
            Self::Disk(g) => GraphRead::column_stores_iter(g.as_ref()),
            Self::Recording(rg) => GraphRead::column_stores_iter(rg.as_ref()),
        }
    }

    #[inline]
    fn node_count(&self) -> usize {
        match self {
            Self::Memory(g) => GraphRead::node_count(&**g),
            Self::Forked(g) => GraphRead::node_count(g.as_ref()),
            Self::Mapped(g) => GraphRead::node_count(&**g),
            Self::Disk(g) => GraphRead::node_count(g.as_ref()),
            Self::Recording(rg) => GraphRead::node_count(rg.as_ref()),
        }
    }

    #[inline]
    fn edge_count(&self) -> usize {
        match self {
            Self::Memory(g) => GraphRead::edge_count(&**g),
            Self::Forked(g) => GraphRead::edge_count(g.as_ref()),
            Self::Mapped(g) => GraphRead::edge_count(&**g),
            Self::Disk(g) => GraphRead::edge_count(g.as_ref()),
            Self::Recording(rg) => GraphRead::edge_count(rg.as_ref()),
        }
    }

    #[inline]
    fn node_bound(&self) -> usize {
        match self {
            Self::Memory(g) => GraphRead::node_bound(&**g),
            Self::Forked(g) => GraphRead::node_bound(g.as_ref()),
            Self::Mapped(g) => GraphRead::node_bound(&**g),
            Self::Disk(g) => GraphRead::node_bound(g.as_ref()),
            Self::Recording(rg) => GraphRead::node_bound(rg.as_ref()),
        }
    }

    #[inline]
    fn edge_bound(&self) -> usize {
        match self {
            Self::Memory(g) => GraphRead::edge_bound(&**g),
            Self::Forked(g) => GraphRead::edge_bound(g.as_ref()),
            Self::Mapped(g) => GraphRead::edge_bound(&**g),
            Self::Disk(g) => GraphRead::edge_bound(g.as_ref()),
            Self::Recording(rg) => GraphRead::edge_bound(rg.as_ref()),
        }
    }

    #[inline]
    fn is_memory(&self) -> bool {
        match self {
            Self::Memory(_) => true,
            Self::Recording(rg) => GraphRead::is_memory(rg.as_ref()),
            _ => false,
        }
    }

    #[inline]
    fn is_mapped(&self) -> bool {
        match self {
            Self::Mapped(_) => true,
            Self::Recording(rg) => GraphRead::is_mapped(rg.as_ref()),
            _ => false,
        }
    }

    #[inline]
    fn is_disk(&self) -> bool {
        match self {
            Self::Disk(_) => true,
            Self::Recording(rg) => GraphRead::is_disk(rg.as_ref()),
            _ => false,
        }
    }

    #[inline(always)]
    fn node_type_of(&self, idx: NodeIndex) -> Option<InternedKey> {
        match self {
            Self::Memory(g) => GraphRead::node_type_of(&**g, idx),
            Self::Forked(g) => GraphRead::node_type_of(g.as_ref(), idx),
            Self::Mapped(g) => GraphRead::node_type_of(&**g, idx),
            Self::Disk(g) => GraphRead::node_type_of(g.as_ref(), idx),
            Self::Recording(rg) => GraphRead::node_type_of(rg.as_ref(), idx),
        }
    }

    #[inline(always)]
    fn node_labels_of(&self, idx: NodeIndex) -> Vec<InternedKey> {
        match self {
            Self::Memory(g) => GraphRead::node_labels_of(&**g, idx),
            Self::Forked(g) => GraphRead::node_labels_of(g.as_ref(), idx),
            Self::Mapped(g) => GraphRead::node_labels_of(&**g, idx),
            Self::Disk(g) => GraphRead::node_labels_of(g.as_ref(), idx),
            Self::Recording(rg) => GraphRead::node_labels_of(rg.as_ref(), idx),
        }
    }

    #[inline(always)]
    fn node_weight(&self, idx: NodeIndex) -> Option<&NodeData> {
        match self {
            Self::Memory(g) => GraphRead::node_weight(&**g, idx),
            Self::Forked(g) => GraphRead::node_weight(g.as_ref(), idx),
            Self::Mapped(g) => GraphRead::node_weight(&**g, idx),
            Self::Disk(g) => GraphRead::node_weight(g.as_ref(), idx),
            Self::Recording(rg) => GraphRead::node_weight(rg.as_ref(), idx),
        }
    }

    #[inline]
    fn get_node_property(&self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
        match self {
            Self::Memory(g) => GraphRead::get_node_property(&**g, idx, key),
            Self::Forked(g) => GraphRead::get_node_property(g.as_ref(), idx, key),
            Self::Mapped(g) => GraphRead::get_node_property(&**g, idx, key),
            Self::Disk(g) => GraphRead::get_node_property(g.as_ref(), idx, key),
            Self::Recording(rg) => GraphRead::get_node_property(rg.as_ref(), idx, key),
        }
    }

    #[inline]
    fn get_node_id(&self, idx: NodeIndex) -> Option<Value> {
        match self {
            Self::Memory(g) => GraphRead::get_node_id(&**g, idx),
            Self::Forked(g) => GraphRead::get_node_id(g.as_ref(), idx),
            Self::Mapped(g) => GraphRead::get_node_id(&**g, idx),
            Self::Disk(g) => GraphRead::get_node_id(g.as_ref(), idx),
            Self::Recording(rg) => GraphRead::get_node_id(rg.as_ref(), idx),
        }
    }

    #[inline]
    fn get_node_title(&self, idx: NodeIndex) -> Option<Value> {
        match self {
            Self::Memory(g) => GraphRead::get_node_title(&**g, idx),
            Self::Forked(g) => GraphRead::get_node_title(g.as_ref(), idx),
            Self::Mapped(g) => GraphRead::get_node_title(&**g, idx),
            Self::Disk(g) => GraphRead::get_node_title(g.as_ref(), idx),
            Self::Recording(rg) => GraphRead::get_node_title(rg.as_ref(), idx),
        }
    }

    #[inline]
    fn str_prop_eq(&self, idx: NodeIndex, key: InternedKey, target: &str) -> Option<bool> {
        match self {
            Self::Memory(g) => GraphRead::str_prop_eq(&**g, idx, key, target),
            Self::Forked(g) => GraphRead::str_prop_eq(g.as_ref(), idx, key, target),
            Self::Mapped(g) => GraphRead::str_prop_eq(&**g, idx, key, target),
            Self::Disk(g) => GraphRead::str_prop_eq(g.as_ref(), idx, key, target),
            Self::Recording(rg) => GraphRead::str_prop_eq(rg.as_ref(), idx, key, target),
        }
    }

    #[inline]
    fn node_indices(&self) -> crate::graph::core::iterators::GraphNodeIndices<'_> {
        match self {
            Self::Memory(g) => GraphRead::node_indices(&**g),
            Self::Forked(g) => GraphRead::node_indices(g.as_ref()),
            Self::Mapped(g) => GraphRead::node_indices(&**g),
            Self::Disk(g) => GraphRead::node_indices(g.as_ref()),
            Self::Recording(rg) => GraphRead::node_indices(rg.as_ref()),
        }
    }

    #[inline]
    fn edge_indices(&self) -> crate::graph::core::iterators::GraphEdgeIndices<'_> {
        match self {
            Self::Memory(g) => GraphRead::edge_indices(&**g),
            Self::Forked(g) => GraphRead::edge_indices(g.as_ref()),
            Self::Mapped(g) => GraphRead::edge_indices(&**g),
            Self::Disk(g) => GraphRead::edge_indices(g.as_ref()),
            Self::Recording(rg) => GraphRead::edge_indices(rg.as_ref()),
        }
    }

    #[inline]
    fn edge_references(&self) -> crate::graph::core::iterators::GraphEdgeReferences<'_> {
        match self {
            Self::Memory(g) => GraphRead::edge_references(&**g),
            Self::Forked(g) => GraphRead::edge_references(g.as_ref()),
            Self::Mapped(g) => GraphRead::edge_references(&**g),
            Self::Disk(g) => GraphRead::edge_references(g.as_ref()),
            Self::Recording(rg) => GraphRead::edge_references(rg.as_ref()),
        }
    }

    #[inline]
    fn edge_weights<'a>(&'a self) -> Box<dyn Iterator<Item = &'a EdgeData> + 'a> {
        match self {
            Self::Memory(g) => GraphRead::edge_weights(&**g),
            Self::Forked(g) => GraphRead::edge_weights(g.as_ref()),
            Self::Mapped(g) => GraphRead::edge_weights(&**g),
            Self::Disk(g) => GraphRead::edge_weights(g.as_ref()),
            Self::Recording(rg) => GraphRead::edge_weights(rg.as_ref()),
        }
    }

    #[inline]
    fn edges_directed(
        &self,
        idx: NodeIndex,
        dir: petgraph::Direction,
    ) -> crate::graph::core::iterators::GraphEdges<'_> {
        match self {
            Self::Memory(g) => GraphRead::edges_directed(&**g, idx, dir),
            Self::Forked(g) => GraphRead::edges_directed(g.as_ref(), idx, dir),
            Self::Mapped(g) => GraphRead::edges_directed(&**g, idx, dir),
            Self::Disk(g) => GraphRead::edges_directed(g.as_ref(), idx, dir),
            Self::Recording(rg) => GraphRead::edges_directed(rg.as_ref(), idx, dir),
        }
    }

    #[inline]
    fn edges(&self, idx: NodeIndex) -> crate::graph::core::iterators::GraphEdges<'_> {
        match self {
            Self::Memory(g) => GraphRead::edges(&**g, idx),
            Self::Forked(g) => GraphRead::edges(g.as_ref(), idx),
            Self::Mapped(g) => GraphRead::edges(&**g, idx),
            Self::Disk(g) => GraphRead::edges(g.as_ref(), idx),
            Self::Recording(rg) => GraphRead::edges(rg.as_ref(), idx),
        }
    }

    #[inline]
    fn edges_directed_filtered(
        &self,
        idx: NodeIndex,
        dir: petgraph::Direction,
        conn_type_filter: Option<InternedKey>,
    ) -> crate::graph::core::iterators::GraphEdges<'_> {
        match self {
            Self::Memory(g) => GraphRead::edges_directed_filtered(&**g, idx, dir, conn_type_filter),
            Self::Forked(g) => {
                GraphRead::edges_directed_filtered(g.as_ref(), idx, dir, conn_type_filter)
            }
            Self::Mapped(g) => GraphRead::edges_directed_filtered(&**g, idx, dir, conn_type_filter),
            Self::Disk(g) => {
                GraphRead::edges_directed_filtered(g.as_ref(), idx, dir, conn_type_filter)
            }
            Self::Recording(rg) => {
                GraphRead::edges_directed_filtered(rg.as_ref(), idx, dir, conn_type_filter)
            }
        }
    }

    #[inline]
    fn edges_connecting(
        &self,
        a: NodeIndex,
        b: NodeIndex,
    ) -> crate::graph::core::iterators::GraphEdgesConnecting<'_> {
        match self {
            Self::Memory(g) => GraphRead::edges_connecting(&**g, a, b),
            Self::Forked(g) => GraphRead::edges_connecting(g.as_ref(), a, b),
            Self::Mapped(g) => GraphRead::edges_connecting(&**g, a, b),
            Self::Disk(g) => GraphRead::edges_connecting(g.as_ref(), a, b),
            Self::Recording(rg) => GraphRead::edges_connecting(rg.as_ref(), a, b),
        }
    }

    #[inline]
    fn edge_weight(&self, idx: EdgeIndex) -> Option<&EdgeData> {
        match self {
            Self::Memory(g) => GraphRead::edge_weight(&**g, idx),
            Self::Forked(g) => GraphRead::edge_weight(g.as_ref(), idx),
            Self::Mapped(g) => GraphRead::edge_weight(&**g, idx),
            Self::Disk(g) => GraphRead::edge_weight(g.as_ref(), idx),
            Self::Recording(rg) => GraphRead::edge_weight(rg.as_ref(), idx),
        }
    }

    #[inline]
    fn find_edge(&self, a: NodeIndex, b: NodeIndex) -> Option<EdgeIndex> {
        match self {
            Self::Memory(g) => GraphRead::find_edge(&**g, a, b),
            Self::Forked(g) => GraphRead::find_edge(g.as_ref(), a, b),
            Self::Mapped(g) => GraphRead::find_edge(&**g, a, b),
            Self::Disk(g) => GraphRead::find_edge(g.as_ref(), a, b),
            Self::Recording(rg) => GraphRead::find_edge(rg.as_ref(), a, b),
        }
    }

    #[inline(always)]
    fn edge_endpoints(&self, idx: EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        match self {
            Self::Memory(g) => GraphRead::edge_endpoints(&**g, idx),
            Self::Forked(g) => GraphRead::edge_endpoints(g.as_ref(), idx),
            Self::Mapped(g) => GraphRead::edge_endpoints(&**g, idx),
            Self::Disk(g) => GraphRead::edge_endpoints(g.as_ref(), idx),
            Self::Recording(rg) => GraphRead::edge_endpoints(rg.as_ref(), idx),
        }
    }

    #[inline(always)]
    fn edge_endpoint_keys<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (NodeIndex, NodeIndex, InternedKey)> + 'a> {
        match self {
            Self::Memory(g) => GraphRead::edge_endpoint_keys(&**g),
            Self::Forked(g) => GraphRead::edge_endpoint_keys(g.as_ref()),
            Self::Mapped(g) => GraphRead::edge_endpoint_keys(&**g),
            Self::Disk(g) => GraphRead::edge_endpoint_keys(g.as_ref()),
            Self::Recording(rg) => GraphRead::edge_endpoint_keys(rg.as_ref()),
        }
    }

    #[inline]
    fn neighbors_directed(
        &self,
        idx: NodeIndex,
        dir: petgraph::Direction,
    ) -> crate::graph::core::iterators::GraphNeighbors<'_> {
        match self {
            Self::Memory(g) => GraphRead::neighbors_directed(&**g, idx, dir),
            Self::Forked(g) => GraphRead::neighbors_directed(g.as_ref(), idx, dir),
            Self::Mapped(g) => GraphRead::neighbors_directed(&**g, idx, dir),
            Self::Disk(g) => GraphRead::neighbors_directed(g.as_ref(), idx, dir),
            Self::Recording(rg) => GraphRead::neighbors_directed(rg.as_ref(), idx, dir),
        }
    }

    #[inline]
    fn neighbors_undirected(
        &self,
        idx: NodeIndex,
    ) -> crate::graph::core::iterators::GraphNeighbors<'_> {
        match self {
            Self::Memory(g) => GraphRead::neighbors_undirected(&**g, idx),
            Self::Forked(g) => GraphRead::neighbors_undirected(g.as_ref(), idx),
            Self::Mapped(g) => GraphRead::neighbors_undirected(&**g, idx),
            Self::Disk(g) => GraphRead::neighbors_undirected(g.as_ref(), idx),
            Self::Recording(rg) => GraphRead::neighbors_undirected(rg.as_ref(), idx),
        }
    }

    #[inline]
    fn sources_for_conn_type_bounded(
        &self,
        conn_type: InternedKey,
        max: Option<usize>,
    ) -> Option<Vec<u32>> {
        match self {
            Self::Memory(g) => GraphRead::sources_for_conn_type_bounded(&**g, conn_type, max),
            Self::Forked(g) => GraphRead::sources_for_conn_type_bounded(g.as_ref(), conn_type, max),
            Self::Mapped(g) => GraphRead::sources_for_conn_type_bounded(&**g, conn_type, max),
            Self::Disk(g) => GraphRead::sources_for_conn_type_bounded(g.as_ref(), conn_type, max),
            Self::Recording(rg) => {
                GraphRead::sources_for_conn_type_bounded(rg.as_ref(), conn_type, max)
            }
        }
    }

    #[inline]
    fn lookup_peer_counts(&self, conn_type: InternedKey) -> Option<HashMap<u32, i64>> {
        match self {
            Self::Memory(g) => GraphRead::lookup_peer_counts(&**g, conn_type),
            Self::Forked(g) => GraphRead::lookup_peer_counts(g.as_ref(), conn_type),
            Self::Mapped(g) => GraphRead::lookup_peer_counts(&**g, conn_type),
            Self::Disk(g) => GraphRead::lookup_peer_counts(g.as_ref(), conn_type),
            Self::Recording(rg) => GraphRead::lookup_peer_counts(rg.as_ref(), conn_type),
        }
    }

    #[inline]
    fn lookup_by_property_eq(
        &self,
        node_type: &str,
        property: &str,
        value: &str,
    ) -> Option<Vec<NodeIndex>> {
        match self {
            Self::Memory(g) => GraphRead::lookup_by_property_eq(&**g, node_type, property, value),
            Self::Forked(g) => {
                GraphRead::lookup_by_property_eq(g.as_ref(), node_type, property, value)
            }
            Self::Mapped(g) => GraphRead::lookup_by_property_eq(&**g, node_type, property, value),
            Self::Disk(g) => {
                GraphRead::lookup_by_property_eq(g.as_ref(), node_type, property, value)
            }
            Self::Recording(rg) => {
                GraphRead::lookup_by_property_eq(rg.as_ref(), node_type, property, value)
            }
        }
    }

    #[inline]
    fn lookup_by_property_prefix(
        &self,
        node_type: &str,
        property: &str,
        prefix: &str,
        limit: usize,
    ) -> Option<Vec<NodeIndex>> {
        match self {
            Self::Memory(g) => {
                GraphRead::lookup_by_property_prefix(&**g, node_type, property, prefix, limit)
            }
            Self::Forked(g) => {
                GraphRead::lookup_by_property_prefix(g.as_ref(), node_type, property, prefix, limit)
            }
            Self::Mapped(g) => {
                GraphRead::lookup_by_property_prefix(&**g, node_type, property, prefix, limit)
            }
            Self::Disk(g) => {
                GraphRead::lookup_by_property_prefix(g.as_ref(), node_type, property, prefix, limit)
            }
            Self::Recording(rg) => GraphRead::lookup_by_property_prefix(
                rg.as_ref(),
                node_type,
                property,
                prefix,
                limit,
            ),
        }
    }

    #[inline]
    fn lookup_by_property_eq_any_type(
        &self,
        property: &str,
        value: &str,
    ) -> Option<Vec<NodeIndex>> {
        match self {
            Self::Memory(g) => GraphRead::lookup_by_property_eq_any_type(&**g, property, value),
            Self::Forked(g) => {
                GraphRead::lookup_by_property_eq_any_type(g.as_ref(), property, value)
            }
            Self::Mapped(g) => GraphRead::lookup_by_property_eq_any_type(&**g, property, value),
            Self::Disk(g) => GraphRead::lookup_by_property_eq_any_type(g.as_ref(), property, value),
            Self::Recording(rg) => {
                GraphRead::lookup_by_property_eq_any_type(rg.as_ref(), property, value)
            }
        }
    }

    #[inline]
    fn lookup_by_property_prefix_any_type(
        &self,
        property: &str,
        prefix: &str,
        limit: usize,
    ) -> Option<Vec<NodeIndex>> {
        match self {
            Self::Memory(g) => {
                GraphRead::lookup_by_property_prefix_any_type(&**g, property, prefix, limit)
            }
            Self::Forked(g) => {
                GraphRead::lookup_by_property_prefix_any_type(g.as_ref(), property, prefix, limit)
            }
            Self::Mapped(g) => {
                GraphRead::lookup_by_property_prefix_any_type(&**g, property, prefix, limit)
            }
            Self::Disk(g) => {
                GraphRead::lookup_by_property_prefix_any_type(g.as_ref(), property, prefix, limit)
            }
            Self::Recording(rg) => {
                GraphRead::lookup_by_property_prefix_any_type(rg.as_ref(), property, prefix, limit)
            }
        }
    }

    #[inline]
    fn count_edges_grouped_by_peer(
        &self,
        conn_type: InternedKey,
        dir: petgraph::Direction,
        deadline: Option<std::time::Instant>,
    ) -> Result<HashMap<u32, i64>, String> {
        match self {
            Self::Memory(g) => {
                GraphRead::count_edges_grouped_by_peer(&**g, conn_type, dir, deadline)
            }
            Self::Forked(g) => {
                GraphRead::count_edges_grouped_by_peer(g.as_ref(), conn_type, dir, deadline)
            }
            Self::Mapped(g) => {
                GraphRead::count_edges_grouped_by_peer(&**g, conn_type, dir, deadline)
            }
            Self::Disk(g) => {
                GraphRead::count_edges_grouped_by_peer(g.as_ref(), conn_type, dir, deadline)
            }
            Self::Recording(rg) => {
                GraphRead::count_edges_grouped_by_peer(rg.as_ref(), conn_type, dir, deadline)
            }
        }
    }

    #[inline]
    fn count_edges_filtered(
        &self,
        node: NodeIndex,
        dir: petgraph::Direction,
        conn_type: Option<InternedKey>,
        other_node_type: Option<InternedKey>,
        deadline: Option<std::time::Instant>,
    ) -> Result<usize, String> {
        match self {
            Self::Memory(g) => GraphRead::count_edges_filtered(
                &**g,
                node,
                dir,
                conn_type,
                other_node_type,
                deadline,
            ),
            Self::Forked(g) => GraphRead::count_edges_filtered(
                g.as_ref(),
                node,
                dir,
                conn_type,
                other_node_type,
                deadline,
            ),
            Self::Mapped(g) => GraphRead::count_edges_filtered(
                &**g,
                node,
                dir,
                conn_type,
                other_node_type,
                deadline,
            ),
            Self::Disk(g) => GraphRead::count_edges_filtered(
                g.as_ref(),
                node,
                dir,
                conn_type,
                other_node_type,
                deadline,
            ),
            Self::Recording(rg) => GraphRead::count_edges_filtered(
                rg.as_ref(),
                node,
                dir,
                conn_type,
                other_node_type,
                deadline,
            ),
        }
    }

    #[inline]
    fn iter_peers_filtered<'a>(
        &'a self,
        node: NodeIndex,
        dir: petgraph::Direction,
        conn_type: Option<u64>,
    ) -> Box<dyn Iterator<Item = (NodeIndex, EdgeIndex)> + 'a> {
        match self {
            Self::Memory(g) => GraphRead::iter_peers_filtered(&**g, node, dir, conn_type),
            Self::Forked(g) => GraphRead::iter_peers_filtered(g.as_ref(), node, dir, conn_type),
            Self::Mapped(g) => GraphRead::iter_peers_filtered(&**g, node, dir, conn_type),
            Self::Disk(g) => GraphRead::iter_peers_filtered(g.as_ref(), node, dir, conn_type),
            Self::Recording(rg) => {
                GraphRead::iter_peers_filtered(rg.as_ref(), node, dir, conn_type)
            }
        }
    }

    #[inline]
    fn reset_arenas(&self) {
        match self {
            Self::Disk(g) => GraphRead::reset_arenas(g.as_ref()),
            Self::Recording(rg) => GraphRead::reset_arenas(rg.as_ref()),
            _ => {}
        }
    }
}

impl GraphWrite for GraphBackend {
    #[inline]
    fn install_column_store(&mut self, type_key: InternedKey, store: std::sync::Arc<ColumnStore>) {
        match self {
            Self::Memory(g) => {
                GraphWrite::install_column_store(unique_heap_backend(g), type_key, store)
            }
            Self::Forked(g) => GraphWrite::install_column_store(g.as_mut(), type_key, store),
            Self::Mapped(g) => {
                GraphWrite::install_column_store(unique_heap_backend(g), type_key, store)
            }
            Self::Disk(g) => GraphWrite::install_column_store(g.as_mut(), type_key, store),
            Self::Recording(rg) => GraphWrite::install_column_store(rg.as_mut(), type_key, store),
        }
    }

    #[inline]
    fn column_store_mut(
        &mut self,
        type_key: InternedKey,
    ) -> Option<&mut std::sync::Arc<ColumnStore>> {
        match self {
            Self::Memory(g) => GraphWrite::column_store_mut(unique_heap_backend(g), type_key),
            Self::Forked(g) => GraphWrite::column_store_mut(g.as_mut(), type_key),
            Self::Mapped(g) => GraphWrite::column_store_mut(unique_heap_backend(g), type_key),
            Self::Disk(g) => GraphWrite::column_store_mut(g.as_mut(), type_key),
            Self::Recording(rg) => GraphWrite::column_store_mut(rg.as_mut(), type_key),
        }
    }

    #[inline]
    fn take_column_store(&mut self, type_key: InternedKey) -> Option<std::sync::Arc<ColumnStore>> {
        match self {
            Self::Memory(g) => GraphWrite::take_column_store(unique_heap_backend(g), type_key),
            Self::Forked(g) => GraphWrite::take_column_store(g.as_mut(), type_key),
            Self::Mapped(g) => GraphWrite::take_column_store(unique_heap_backend(g), type_key),
            Self::Disk(g) => GraphWrite::take_column_store(g.as_mut(), type_key),
            Self::Recording(rg) => GraphWrite::take_column_store(rg.as_mut(), type_key),
        }
    }

    #[inline]
    fn clear_column_stores(&mut self) {
        match self {
            Self::Memory(g) => GraphWrite::clear_column_stores(unique_heap_backend(g)),
            Self::Forked(g) => GraphWrite::clear_column_stores(g.as_mut()),
            Self::Mapped(g) => GraphWrite::clear_column_stores(unique_heap_backend(g)),
            Self::Disk(g) => GraphWrite::clear_column_stores(g.as_mut()),
            Self::Recording(rg) => GraphWrite::clear_column_stores(rg.as_mut()),
        }
    }

    #[inline]
    fn set_node_property(&mut self, idx: NodeIndex, key: InternedKey, value: Value) {
        match self {
            Self::Memory(g) => {
                GraphWrite::set_node_property(unique_heap_backend(g), idx, key, value)
            }
            Self::Forked(g) => GraphWrite::set_node_property(g.as_mut(), idx, key, value),
            Self::Mapped(g) => {
                GraphWrite::set_node_property(unique_heap_backend(g), idx, key, value)
            }
            Self::Disk(g) => GraphWrite::set_node_property(g.as_mut(), idx, key, value),
            Self::Recording(rg) => GraphWrite::set_node_property(rg.as_mut(), idx, key, value),
        }
    }

    #[inline]
    fn set_node_title(&mut self, idx: NodeIndex, value: Value) {
        match self {
            Self::Memory(g) => GraphWrite::set_node_title(unique_heap_backend(g), idx, value),
            Self::Forked(g) => GraphWrite::set_node_title(g.as_mut(), idx, value),
            Self::Mapped(g) => GraphWrite::set_node_title(unique_heap_backend(g), idx, value),
            Self::Disk(g) => GraphWrite::set_node_title(g.as_mut(), idx, value),
            Self::Recording(rg) => GraphWrite::set_node_title(rg.as_mut(), idx, value),
        }
    }

    #[inline]
    fn set_node_property_if_absent(&mut self, idx: NodeIndex, key: InternedKey, value: Value) {
        match self {
            Self::Memory(g) => {
                GraphWrite::set_node_property_if_absent(unique_heap_backend(g), idx, key, value)
            }
            Self::Forked(g) => GraphWrite::set_node_property_if_absent(g.as_mut(), idx, key, value),
            Self::Mapped(g) => {
                GraphWrite::set_node_property_if_absent(unique_heap_backend(g), idx, key, value)
            }
            Self::Disk(g) => GraphWrite::set_node_property_if_absent(g.as_mut(), idx, key, value),
            Self::Recording(rg) => {
                GraphWrite::set_node_property_if_absent(rg.as_mut(), idx, key, value)
            }
        }
    }

    #[inline]
    fn remove_node_property(&mut self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
        match self {
            Self::Memory(g) => GraphWrite::remove_node_property(unique_heap_backend(g), idx, key),
            Self::Forked(g) => GraphWrite::remove_node_property(g.as_mut(), idx, key),
            Self::Mapped(g) => GraphWrite::remove_node_property(unique_heap_backend(g), idx, key),
            Self::Disk(g) => GraphWrite::remove_node_property(g.as_mut(), idx, key),
            Self::Recording(rg) => GraphWrite::remove_node_property(rg.as_mut(), idx, key),
        }
    }

    #[inline]
    fn clear_node_property(&mut self, idx: NodeIndex, key: InternedKey) -> Option<Value> {
        match self {
            Self::Memory(g) => GraphWrite::clear_node_property(unique_heap_backend(g), idx, key),
            Self::Forked(g) => GraphWrite::clear_node_property(g.as_mut(), idx, key),
            Self::Mapped(g) => GraphWrite::clear_node_property(unique_heap_backend(g), idx, key),
            Self::Disk(g) => GraphWrite::clear_node_property(g.as_mut(), idx, key),
            Self::Recording(rg) => GraphWrite::clear_node_property(rg.as_mut(), idx, key),
        }
    }

    #[inline]
    fn replace_node_properties(&mut self, idx: NodeIndex, pairs: Vec<(InternedKey, Value)>) {
        match self {
            Self::Memory(g) => {
                GraphWrite::replace_node_properties(unique_heap_backend(g), idx, pairs)
            }
            Self::Forked(g) => GraphWrite::replace_node_properties(g.as_mut(), idx, pairs),
            Self::Mapped(g) => {
                GraphWrite::replace_node_properties(unique_heap_backend(g), idx, pairs)
            }
            Self::Disk(g) => GraphWrite::replace_node_properties(g.as_mut(), idx, pairs),
            Self::Recording(rg) => GraphWrite::replace_node_properties(rg.as_mut(), idx, pairs),
        }
    }

    #[inline]
    fn node_weight_mut(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        match self {
            Self::Memory(g) => GraphWrite::node_weight_mut(unique_heap_backend(g), idx),
            Self::Forked(g) => GraphWrite::node_weight_mut(g.as_mut(), idx),
            Self::Mapped(g) => GraphWrite::node_weight_mut(unique_heap_backend(g), idx),
            Self::Disk(g) => GraphWrite::node_weight_mut(g.as_mut(), idx),
            Self::Recording(rg) => GraphWrite::node_weight_mut(rg.as_mut(), idx),
        }
    }

    #[inline]
    fn node_weight_mut_silent(&mut self, idx: NodeIndex) -> Option<&mut NodeData> {
        match self {
            Self::Memory(g) => GraphWrite::node_weight_mut_silent(unique_heap_backend(g), idx),
            Self::Forked(g) => GraphWrite::node_weight_mut_silent(g.as_mut(), idx),
            Self::Mapped(g) => GraphWrite::node_weight_mut_silent(unique_heap_backend(g), idx),
            Self::Disk(g) => GraphWrite::node_weight_mut_silent(g.as_mut(), idx),
            // The whole point: route to the wrapper's *silent* override so the
            // columnar handle-refresh sweep isn't captured as N mutations.
            Self::Recording(rg) => GraphWrite::node_weight_mut_silent(rg.as_mut(), idx),
        }
    }

    #[inline]
    fn edge_weight_mut(&mut self, idx: EdgeIndex) -> Option<&mut EdgeData> {
        match self {
            Self::Memory(g) => GraphWrite::edge_weight_mut(unique_heap_backend(g), idx),
            Self::Forked(g) => GraphWrite::edge_weight_mut(g.as_mut(), idx),
            Self::Mapped(g) => GraphWrite::edge_weight_mut(unique_heap_backend(g), idx),
            Self::Disk(g) => GraphWrite::edge_weight_mut(g.as_mut(), idx),
            Self::Recording(rg) => GraphWrite::edge_weight_mut(rg.as_mut(), idx),
        }
    }

    #[inline]
    fn add_node(&mut self, data: NodeData) -> NodeIndex {
        match self {
            Self::Memory(g) => GraphWrite::add_node(unique_heap_backend(g), data),
            Self::Forked(g) => GraphWrite::add_node(g.as_mut(), data),
            Self::Mapped(g) => GraphWrite::add_node(unique_heap_backend(g), data),
            Self::Disk(g) => GraphWrite::add_node(g.as_mut(), data),
            Self::Recording(rg) => GraphWrite::add_node(rg.as_mut(), data),
        }
    }

    #[inline]
    fn remove_node(&mut self, idx: NodeIndex) -> Option<NodeData> {
        // An overlay cannot express an adjacency edit — `StableDiGraph` threads
        // adjacency through per-node linked lists, so this rewrites *existing*
        // nodes. Collapse to a plain backend first (free if the reader already
        // dropped, a deep copy otherwise), which is why the `Forked` arm below
        // is unreachable rather than implemented.
        self.flatten_fork();
        match self {
            Self::Memory(g) => GraphWrite::remove_node(unique_heap_backend(g), idx),
            Self::Forked(_) => unreachable!("flatten_fork above collapsed the overlay"),
            Self::Mapped(g) => GraphWrite::remove_node(unique_heap_backend(g), idx),
            Self::Disk(g) => GraphWrite::remove_node(g.as_mut(), idx),
            Self::Recording(rg) => GraphWrite::remove_node(rg.as_mut(), idx),
        }
    }

    #[inline]
    fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, data: EdgeData) -> EdgeIndex {
        // An overlay cannot express an adjacency edit — `StableDiGraph` threads
        // adjacency through per-node linked lists, so this rewrites *existing*
        // nodes. Collapse to a plain backend first (free if the reader already
        // dropped, a deep copy otherwise), which is why the `Forked` arm below
        // is unreachable rather than implemented.
        self.flatten_fork();
        match self {
            Self::Memory(g) => GraphWrite::add_edge(unique_heap_backend(g), a, b, data),
            Self::Forked(_) => unreachable!("flatten_fork above collapsed the overlay"),
            Self::Mapped(g) => GraphWrite::add_edge(unique_heap_backend(g), a, b, data),
            Self::Disk(g) => GraphWrite::add_edge(g.as_mut(), a, b, data),
            Self::Recording(rg) => GraphWrite::add_edge(rg.as_mut(), a, b, data),
        }
    }

    #[inline]
    fn remove_edge(&mut self, idx: EdgeIndex) -> Option<EdgeData> {
        // An overlay cannot express an adjacency edit — `StableDiGraph` threads
        // adjacency through per-node linked lists, so this rewrites *existing*
        // nodes. Collapse to a plain backend first (free if the reader already
        // dropped, a deep copy otherwise), which is why the `Forked` arm below
        // is unreachable rather than implemented.
        self.flatten_fork();
        match self {
            Self::Memory(g) => GraphWrite::remove_edge(unique_heap_backend(g), idx),
            Self::Forked(_) => unreachable!("flatten_fork above collapsed the overlay"),
            Self::Mapped(g) => GraphWrite::remove_edge(unique_heap_backend(g), idx),
            Self::Disk(g) => GraphWrite::remove_edge(g.as_mut(), idx),
            Self::Recording(rg) => GraphWrite::remove_edge(rg.as_mut(), idx),
        }
    }

    #[inline]
    fn update_row_id(&mut self, node_idx: NodeIndex, row_id: u32) {
        match self {
            Self::Disk(g) => GraphWrite::update_row_id(g.as_mut(), node_idx, row_id),
            Self::Recording(rg) => GraphWrite::update_row_id(rg.as_mut(), node_idx, row_id),
            _ => {}
        }
    }

    #[inline]
    fn flush_pending_writes(&mut self) {
        match self {
            Self::Memory(g) => GraphWrite::flush_pending_writes(unique_heap_backend(g)),
            Self::Forked(g) => GraphWrite::flush_pending_writes(g.as_mut()),
            Self::Mapped(g) => GraphWrite::flush_pending_writes(unique_heap_backend(g)),
            Self::Disk(g) => GraphWrite::flush_pending_writes(g.as_mut()),
            Self::Recording(rg) => GraphWrite::flush_pending_writes(rg.as_mut()),
        }
    }
}
