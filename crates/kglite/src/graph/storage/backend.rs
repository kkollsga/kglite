//! GraphBackend enum + per-backend dispatcher + GraphRead / GraphWrite impls.
//!
//! Trait impls forward to the inner backend (Memory / Forked / Mapped / Disk /
//! Recording) via enum match. This is the central enum-dispatch boundary
//! captured by the source-quality whitelist.

use crate::graph::schema::{EdgeData, InternedKey, NodeData};
use crate::graph::storage::column_store::ColumnStore;
use crate::graph::storage::fork_base::{can_fork, Folded};
use crate::graph::storage::forked::ForkedGraph;
use crate::graph::storage::frozen::FrozenGraph;
use crate::graph::storage::record_ref::RecordRef;
use crate::graph::storage::recording::{RecordingGraph, RecordingState};
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

#[cfg(test)]
pub(crate) fn backend_clone_nodes() -> usize {
    BACKEND_CLONE_NODES.get()
}

/// Record `n` nodes genuinely copied.
///
/// **Which paths call this is load-bearing.** `Clone` used to bump it by
/// `node_count()` on *every* clone; once the `Memory` arm began producing a
/// shallow `Forked` overlay that would report a whole-graph copy that no longer
/// happens, leaving `held_reader_copies_no_nodes` unable to tell the fix from
/// the defect. Only the paths that actually duplicate node storage call it: the
/// deep-clone fallback in `Clone`, and `ForkedGraph::materialise`.
#[cfg(test)]
pub(crate) fn note_nodes_copied(n: usize) {
    BACKEND_CLONE_NODES.set(BACKEND_CLONE_NODES.get() + n);
}

/// `&mut T` from a heap backend handle, copying it first if it is shared.
///
/// **This was once an `expect` and had to be softened.** A `Memory` handle was
/// provably unique until the copy-on-write overlay introduced a second holder —
/// the `base` of somebody else's overlay (see
/// [`GraphBackend::ensure_writable`], which documents the shape).
///
/// `ensure_writable` runs at write entry, so by the time any caller reaches
/// here the handle is unique again and this is a plain `Arc::get_mut`.
/// `Arc::make_mut` is the fallback for a path that bypasses write entry: it
/// deep-copies rather than mutating a backend someone is reading, i.e. it fails
/// **slow, never wrong** — the direction `rollback::journal_covers` and
/// `fork_base::can_fork` also take. A panic here would have been a user-visible
/// crash for a cost problem.
#[inline(always)]
pub(crate) fn unique_heap_backend<T: Clone>(handle: &mut Arc<T>) -> &mut T {
    Arc::make_mut(handle)
}

/// Graph storage backend. Six variants — heap-resident memory, a
/// copy-on-write overlay over a shared memory or mapped base, mmap-columnar-spilled
/// mapped, a read-only topology served from a `.kgl`'s raw sections, CSR-on-disk,
/// and a write-capture wrapper. `MappedGraph` is a
/// distinct struct rather than a type alias, so each backend owns its own
/// [`GraphRead`] / [`GraphWrite`] impl — Memory/Mapped/Disk in
/// [`crate::graph::storage::impls`], `ForkedGraph` and `RecordingGraph` in
/// their own modules. This enum is a dumb dispatcher.
///
/// The `Recording` variant wraps any other `GraphBackend` — including, in
/// principle, another `Recording`. Its `is_memory` / `is_mapped` / `is_disk`
/// predicates forward to the inner backend so consumers that switch on "what's
/// the underlying storage" keep working unchanged when wrapped.
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
    /// (`fork_base::can_fork`), over a `Memory` or a `Mapped` base, and
    /// collapsed back to the variant it forked from by
    /// [`try_compact`](Self::try_compact) the moment the reader drops.
    ///
    /// The same variant serves a **frozen graph that has taken a write**: the
    /// base is the `Frozen` topology, no reader is involved, and
    /// [`try_compact`](Self::try_compact) never folds it (there is no petgraph
    /// to fold into). It ends at a save, which rebases the graph onto the file
    /// it wrote ([`rebase_frozen`](Self::rebase_frozen)), or a whole-graph
    /// operation ([`flatten_fork`](Self::flatten_fork)); its size never ends it.
    ///
    /// `is_mapped`, `supports_undo_journal` and
    /// `supports_checkpoint_free_mutation` answer exactly as the `Memory`,
    /// `Mapped` or `Frozen` arm it was forked from would. `is_memory` answers `false` for
    /// every fork, so memory-only executor routes take their generic path
    /// while a view is held. Answering `false` to `supports_undo_journal` would
    /// send every statement taken while a view is held to
    /// `StatementCheckpoint::Clone` — an O(V+E) clone *per statement*, a worse
    /// cliff than the defect this variant removes.
    Forked(Box<ForkedGraph>),
    Mapped(Arc<MappedGraph>),
    /// **A read-only topology served from the raw sections of a v9 `.kgl`**, for
    /// a graph loaded into mapped storage from a file that recorded them. Resident
    /// memory is the pages a query touches, not the size of the graph.
    ///
    /// Immutable and shared by `Arc`, so a clone is a refcount bump. The first
    /// write (any [`GraphWrite`] method, or the start of an undo journal) turns
    /// the backend into a [`Forked`](Self::Forked) overlay over this topology
    /// ([`overlay_frozen`](Self::overlay_frozen)): nothing is copied, and the
    /// heap holds only what is written. The overlay never folds back; a save
    /// writes its merge out and serves the graph from that file again, and a
    /// whole-graph operation collapses it into a heap [`Mapped`](Self::Mapped)
    /// graph. See [`crate::graph::storage::frozen`].
    ///
    /// `is_mapped` answers `true`, like the overlay and the heap variant it
    /// becomes.
    Frozen(Arc<FrozenGraph>),
    Disk(Box<DiskGraph>),
    // Write-capture wrapper: the production backend of every graph opened with
    // `durable=True`. `graph::durability` wraps the loaded backend (via
    // `recording::wrap_for_durability`) once its WAL replay has finished, so
    // each mutation crossing the `GraphWrite` seam is buffered as a `RawOp` and
    // flushed to the log. Because it wraps the enum itself, the capture layer is
    // identical for a memory, mapped, or disk graph underneath.
    Recording(Box<RecordingGraph<GraphBackend>>),
}

impl GraphBackend {
    /// Incoming relationship bindings excluding the second incidence of a self-loop.
    /// This private count route preserves backend indexes without materializing edges.
    pub(crate) fn count_incoming_nonself_edges_filtered(
        &self,
        node: NodeIndex,
        conn_type: Option<InternedKey>,
        other_node_type: Option<InternedKey>,
        deadline: Option<std::time::Instant>,
    ) -> Result<usize, String> {
        match self {
            Self::Memory(g) => g.count_edges_filtered_impl(
                node,
                petgraph::Direction::Incoming,
                conn_type,
                other_node_type,
                deadline,
                true,
            ),
            Self::Mapped(g) => g.count_edges_filtered_impl(
                node,
                petgraph::Direction::Incoming,
                conn_type,
                other_node_type,
                deadline,
                true,
            ),
            Self::Forked(g) => {
                g.count_incoming_nonself_edges_filtered(node, conn_type, other_node_type, deadline)
            }
            Self::Frozen(g) => g.count_edges_filtered_impl(
                node,
                petgraph::Direction::Incoming,
                conn_type,
                other_node_type,
                deadline,
                true,
            ),
            Self::Disk(g) => g.count_edges_filtered_impl(
                node,
                petgraph::Direction::Incoming,
                conn_type.map(|key| key.as_u64()),
                other_node_type,
                deadline,
                true,
            ),
            Self::Recording(g) => g.inner().count_incoming_nonself_edges_filtered(
                node,
                conn_type,
                other_node_type,
                deadline,
            ),
        }
    }

    /// An **owned** node record, for a caller that finishes with it inside its
    /// own frame — scans and filters, which drop each record immediately.
    ///
    /// Exists for the disk backend, where [`GraphRead::node_weight`] parks
    /// every record it builds in the per-query arena: a scan walking a million
    /// nodes retains a million records until its query ends
    /// (`storage/disk/query_arena.rs`). Materializing into the caller's frame
    /// keeps such a scan flat in memory and skips the arena mutex.
    ///
    /// **Gate on [`GraphRead::is_disk`] before calling** — the heap backends
    /// already own their records and can only answer by *cloning*.
    #[inline]
    pub(crate) fn owned_node_data(&self, idx: NodeIndex) -> Option<NodeData> {
        match self {
            Self::Disk(g) => g.owned_node_data(idx),
            Self::Recording(rg) => rg.inner().owned_node_data(idx),
            _ => GraphRead::node_weight(self, idx).map(|r| (*r).clone()),
        }
    }

    #[inline]
    // Keep the established constructor-only backend API stable.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        GraphBackend::Memory(Arc::new(MemoryGraph::new()))
    }

    /// Whether this backend is wrapped in the write-capture layer.
    #[inline]
    pub(crate) fn is_recording(&self) -> bool {
        matches!(self, GraphBackend::Recording(_))
    }

    /// The write-capture layer, if any. Test-only: the production durable paths
    /// *drain* the buffer through [`recording_mut`](Self::recording_mut). This
    /// exists so the durable session's replay-before-wrap ordering has an
    /// observable — a graph wrapped before its WAL replay carries every replayed
    /// op in this buffer, which nothing else can see.
    #[cfg(test)]
    #[inline]
    pub(crate) fn recording(&self) -> Option<&RecordingGraph<GraphBackend>> {
        match self {
            GraphBackend::Recording(rg) => Some(rg),
            _ => None,
        }
    }

    /// Mutable access to the write-capture layer; the durable commit and
    /// checkpoint paths drain the op buffer through it.
    #[inline]
    pub(crate) fn recording_mut(&mut self) -> Option<&mut RecordingGraph<GraphBackend>> {
        match self {
            GraphBackend::Recording(rg) => Some(rg),
            _ => None,
        }
    }

    pub(crate) fn records_edge_embedding_bases(&self) -> Option<bool> {
        match self {
            GraphBackend::Recording(recording) => Some(recording.captures_edge_embedding_bases()),
            _ => None,
        }
    }

    /// Snapshot the write-capture wrapper without cloning its inner graph.
    pub(crate) fn recording_state(&self) -> Option<RecordingState> {
        match self {
            GraphBackend::Recording(recording) => Some(recording.checkpoint_state()),
            _ => None,
        }
    }

    /// Restore the exact wrapper shape and transient state around the current
    /// inner graph. Used when a failed CDC lifecycle statement installed or
    /// removed the wrapper after the ordinary graph checkpoint opened.
    pub(crate) fn restore_recording_state(&mut self, state: Option<RecordingState>) {
        match state {
            Some(state) => {
                self.wrap_for_capture();
                let GraphBackend::Recording(recording) = self else {
                    unreachable!("wrap_for_capture must install a recording wrapper")
                };
                recording.restore_state(state);
            }
            None => {
                debug_assert!(
                    !self.is_wal_owner(),
                    "CDC lifecycle rollback cannot remove a pre-existing WAL wrapper"
                );
                self.unwrap_capture_if_unowned();
            }
        }
    }

    /// The `DiskGraph` underneath, looking **through** any write-capture
    /// wrapper. `durable=True` and `cdc::enable` both replace the backend with
    /// `Recording`, so a bare `matches!(.., Disk(_))` answers "not a disk
    /// graph" for every such graph — and a caller routing by storage backend
    /// then takes the memory path on a graph that cannot afford it.
    #[inline]
    pub(crate) fn as_disk(&self) -> Option<&DiskGraph> {
        match self {
            GraphBackend::Disk(dg) => Some(dg),
            GraphBackend::Recording(rg) => rg.inner().as_disk(),
            _ => None,
        }
    }

    /// Mutable counterpart of [`as_disk`](Self::as_disk).
    #[inline]
    pub(crate) fn as_disk_mut(&mut self) -> Option<&mut DiskGraph> {
        match self {
            GraphBackend::Disk(dg) => Some(dg),
            GraphBackend::Recording(rg) => rg.inner_mut().as_disk_mut(),
            _ => None,
        }
    }

    /// Claim capture after DirGraph identity admission. The public entry is
    /// [`crate::graph::storage::recording::wrap_for_durability`]; durable-open
    /// publication also calls here after its pre-file/pre-publication checks.
    pub(crate) fn wrap_for_durability(&mut self) {
        self.wrap_for_capture();
        if let GraphBackend::Recording(rg) = self {
            rg.claim_wal_ownership();
        }
    }

    /// Wrap this backend in the write-capture layer **without** claiming
    /// write-ahead-log ownership, idempotently — the change-data-capture
    /// entry point (`graph::cdc::enable`). CDC derives events from the buffer
    /// the wrapper fills but keeps no log, so it must not present itself as a
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
    /// `cdc::disable`. Capture is not free — a wrapped backend buffers a
    /// `RawOp` per mutation and gives up the checkpoint-free mutation fast path
    /// ([`supports_checkpoint_free_mutation`](Self::supports_checkpoint_free_mutation),
    /// except for a terminal `DELETE`) — so "disable" has to actually unwrap, or
    /// it leaves a permanent tax.
    ///
    /// Refuses on a WAL-owned wrapper: unwrapping one silently stops logging,
    /// leaving the graph committing and the log still claiming to describe it.
    /// Buffered ops are dropped with the wrapper, which is correct precisely
    /// because nothing owns them — a WAL-owned buffer never reaches here.
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

    pub(crate) fn release_wal_ownership(&mut self) {
        if let GraphBackend::Recording(recording) = self {
            recording.release_wal_ownership();
        }
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
    /// distinct boundary and keeps the conservative checkpoint path. Terminal
    /// `DELETE` is the exception, see
    /// [`supports_checkpoint_free_delete`](Self::supports_checkpoint_free_delete).
    #[inline]
    pub(crate) fn supports_checkpoint_free_mutation(&self) -> bool {
        // `Forked` answers as the backend it forked from — see the variant
        // doc — so an overlay over `Mapped` keeps `Mapped`'s full checkpoint.
        // Pinned by
        // `rollback_tests::forked_statements_copy_zero_nodes_except_one_flatten`.
        match self {
            GraphBackend::Memory(_) => true,
            GraphBackend::Forked(g) => !g.base_is_mapped(),
            _ => false,
        }
    }

    /// [`supports_checkpoint_free_mutation`](Self::supports_checkpoint_free_mutation)
    /// for a terminal `DELETE`, which also holds under the capture wrapper.
    ///
    /// The checkpoint's undo journal keeps the full `NodeData` of every node
    /// the statement removes until it commits, which for a million-node delete
    /// is the largest allocation of the statement (~1.7 KB per node measured,
    /// against ~120 bytes for the capture op the log needs). A delete that
    /// cannot fail after its first removal has nothing to roll back, and the
    /// wrapper only appends to its op buffer, so a durable graph may skip the
    /// journal exactly as a plain in-memory one does. `CREATE` keeps the
    /// conservative path: the durable create refuses duplicate identities
    /// inside the write.
    #[inline]
    pub(crate) fn supports_checkpoint_free_delete(&self) -> bool {
        match self {
            GraphBackend::Recording(rg) => rg.inner().supports_checkpoint_free_mutation(),
            other => other.supports_checkpoint_free_mutation(),
        }
    }

    /// Whether this backend can capture inverse operations for a
    /// statement-scoped undo journal, i.e. whether rollback can avoid the
    /// whole-graph clone.
    ///
    /// Every petgraph-backed backend can. `Memory` and `Mapped` both hold a
    /// heap `StableDiGraph<NodeData, EdgeData>` as `inner` and every
    /// `UndoEntry` is keyed on the `NodeIndex`/`EdgeIndex` it hands out, so the
    /// capture seam is the same for both — `StorageMode::Mapped` spills
    /// *properties* to mmap (`memory_limit = Some(0)`), not the node/edge graph,
    /// which stays heap-resident. A `Frozen` topology is the one mapped graph
    /// whose nodes are not: it answers `true` because its first write, which is
    /// `begin_undo`, makes it an overlay (`Forked`, below) whose journal reverses
    /// through the overlay's own writes.
    ///
    /// `Disk` cannot: it has no petgraph at all — it mutates a CSR + mmap layout
    /// through generation overlays and arena-staged writes, its slots carry no
    /// `NodeIndex` identity for an entry to name, and it has no free list whose
    /// LIFO ordering reverse replay could exploit to restore slot identity. So a
    /// disk graph keeps the clone checkpoint (see `dir_graph/rollback.rs`):
    /// every mutating statement opens an O(V+E) `fork_transaction()` whole-graph
    /// checkpoint and its statement-rollback cost scales with graph size.
    /// Mirrored in the user-facing storage-mode guide
    /// (`docs/python/core-concepts.md`).
    ///
    /// `Recording` forwards to whatever it wraps — durability and rollback
    /// strategy are independent concerns.
    ///
    /// **This is the only remaining veto term in `journal_covers`.**
    #[inline]
    pub(crate) fn supports_undo_journal(&self) -> bool {
        match self {
            GraphBackend::Memory(_) | GraphBackend::Mapped(_) => true,
            // A frozen graph becomes an overlay before its first write, and
            // `begin_undo` is that write, so the journal it opens is `Forked`'s.
            GraphBackend::Frozen(_) => true,
            // MUST be true — see the `Forked` variant doc. `UndoEntry` keys are
            // the `NodeIndex`/`EdgeIndex` the overlay still hands out, and
            // reversal goes through `ForkedGraph`'s own `GraphWrite`, so entries
            // land in the overlay and never touch the shared base.
            GraphBackend::Forked(_) => true,
            GraphBackend::Recording(rg) => rg.inner().supports_undo_journal(),
            GraphBackend::Disk(_) => false,
        }
    }

    /// `true` while this backend is a copy-on-write overlay over a base a
    /// reader still holds.
    ///
    /// Public as a **diagnostic**: the one cheap, non-timing observable that
    /// distinguishes the fork from the whole-graph clone it replaced, and from a
    /// compaction that failed to fold back. Bindings expose it for regression
    /// tests (`kglite._backend_is_forked`). The one engine behaviour that
    /// depends on it: a store this overlay appends to may take a heap tail
    /// (`ColumnStore::allow_append_tail`), which the overlay's fold removes.
    #[inline]
    pub fn is_forked(&self) -> bool {
        match self {
            GraphBackend::Forked(_) => true,
            GraphBackend::Recording(rg) => rg.inner().is_forked(),
            _ => false,
        }
    }

    /// Make this backend safe to mutate in place, given that a `Memory` or
    /// `Mapped` arm's `Arc` can be shared — as the *base* of somebody else's overlay.
    ///
    /// The shape that needs it: `g.copy()` (or a transaction snapshot) forks
    /// **from** `g`, so the fork's `base` and `g`'s own `Memory(_)` are now the
    /// same allocation. `g` is still a uniquely-owned `Arc<DirGraph>`, so
    /// `Arc::make_mut` at the `DirGraph` level does nothing and the write would
    /// go straight into a backend the fork is reading. Before the copy-on-write
    /// overlay a `Memory` handle was always unique, so this could not happen.
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
            GraphBackend::Forked(_) | GraphBackend::Frozen(_) | GraphBackend::Disk(_) => return,
        };
        if !shared {
            return;
        }
        match std::mem::replace(self, GraphBackend::new()) {
            GraphBackend::Memory(base) => {
                *self = if can_fork(&*base) {
                    GraphBackend::Forked(Box::new(ForkedGraph::new(base)))
                } else {
                    #[cfg(test)]
                    note_nodes_copied(base.inner().node_count());
                    GraphBackend::Memory(Arc::new(base.deep_clone()))
                };
            }
            GraphBackend::Mapped(base) => {
                *self = if can_fork(&*base) {
                    GraphBackend::Forked(Box::new(ForkedGraph::new(base)))
                } else {
                    #[cfg(test)]
                    note_nodes_copied(base.inner().node_count());
                    GraphBackend::Mapped(Arc::new(base.deep_clone()))
                };
            }
            other => *self = other,
        }
    }

    /// Fold an overlay back into its base when this writer is the last holder.
    ///
    /// Called at handle write entry and after Session publication drops its old
    /// owner. A no-op on other variants and while a reader still holds the base.
    ///
    /// The fold borrows the overlay, as [`Self::flatten_fork`]'s does, and the
    /// backend is replaced only by its result: a refused fold leaves the
    /// overlay serving, never an empty placeholder.
    pub(crate) fn try_compact(&mut self) {
        match self {
            GraphBackend::Recording(rg) => rg.inner_mut().try_compact(),
            GraphBackend::Forked(forked) => {
                if let Some(folded) = forked.fold_in_place() {
                    *self = GraphBackend::from(folded);
                }
            }
            _ => {}
        }
    }

    /// Collapse an overlay to a plain `Memory` or `Mapped` backend **unconditionally**,
    /// deep-copying the base if a reader still holds it.
    ///
    /// No write needs it: node and adjacency edits alike land in the overlay.
    /// Two kinds of caller collapse it: the whole-graph operations that need
    /// one concrete `StableDiGraph` (vacuum, a disk conversion, an N-Triples
    /// load, the column rebuild), and an adjacency write whose delta has passed
    /// [`ForkedGraph::delta_exceeds_write_cap`]. Cost is the whole-graph deep
    /// copy the overlay normally avoids, paid by the caller that asked.
    ///
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
        let GraphBackend::Forked(forked) = self else {
            unreachable!("just matched Forked")
        };
        // `materialise` borrows the overlay and folds into a copy, so a refusal
        // leaves this backend serving the intact overlay rather than an empty
        // placeholder whose indexes point at nothing (issue #195).
        match forked.materialise() {
            Ok(folded) => *self = GraphBackend::from(folded),
            Err(reason) => panic!(
                "cannot collapse the copy-on-write overlay: {reason}; the overlay \
                 is left intact (see storage/forked.rs)"
            ),
        }
    }

    /// A backend serving `frozen`, the topology a mapped load read from the
    /// raw sections of its `.kgl`.
    pub(crate) fn from_frozen(frozen: FrozenGraph) -> Self {
        GraphBackend::Frozen(Arc::new(frozen))
    }

    /// Replace a frozen topology with the heap `Mapped` backend a write needs.
    ///
    /// Idempotent and a no-op on every other variant, so a write path may call
    /// it unconditionally. Only this handle is promoted: a clone still holding
    /// the frozen topology keeps serving it. O(V+E) once, the cost mapped
    /// storage always paid at load; see [`FrozenGraph::to_mapped`].
    pub(crate) fn promote_frozen(&mut self) {
        match self {
            GraphBackend::Frozen(frozen) => {
                let mapped = frozen.to_mapped();
                *self = GraphBackend::Mapped(Arc::new(mapped));
            }
            GraphBackend::Recording(rg) => rg.inner_mut().promote_frozen(),
            _ => {}
        }
    }

    /// Turn a frozen topology into a copy-on-write overlay over it, which is
    /// what every write to a frozen graph does. O(1): nothing is copied, the
    /// overlay holds only what is written and reads the rest from the mapped
    /// sections. Idempotent and a no-op on every other variant.
    ///
    /// The overlay does not fold back (a frozen base has no petgraph to fold
    /// into). It ends at a save, where its merge is written out and the graph
    /// is rebased onto that file, or at a whole-graph operation, which collapses
    /// it into a heap `Mapped` graph once.
    pub(crate) fn overlay_frozen(&mut self) {
        match self {
            GraphBackend::Frozen(frozen) => {
                let overlay = ForkedGraph::new(Arc::clone(frozen));
                *self = GraphBackend::Forked(Box::new(overlay));
            }
            GraphBackend::Recording(rg) => rg.inner_mut().overlay_frozen(),
            _ => {}
        }
    }

    /// Entries the write overlay holds (see [`ForkedGraph::delta_len`]), looking
    /// through a write-capture wrapper; `None` without an overlay.
    pub(crate) fn overlay_entries(&self) -> Option<usize> {
        match self {
            GraphBackend::Forked(forked) => Some(forked.delta_len()),
            GraphBackend::Recording(rg) => rg.inner().overlay_entries(),
            _ => None,
        }
    }

    /// Clear what a rolled-back statement left in the write overlay over a
    /// frozen base; see [`super::undo::OverlayResidue`]. Looks through a
    /// write-capture wrapper.
    pub(crate) fn clear_overlay_residue(&mut self, residue: super::undo::OverlayResidue) {
        match self {
            GraphBackend::Forked(forked) => forked.clear_residue(residue),
            GraphBackend::Recording(rg) => rg.inner_mut().clear_overlay_residue(residue),
            _ => {}
        }
    }

    /// `true` while the topology is served from a `.kgl`'s raw sections.
    ///
    /// Public as a **diagnostic**, like [`is_forked`](Self::is_forked): the
    /// observable that tells a mapped load that kept the file's topology from
    /// one that decoded it. Strict: it turns false at the first write, when the
    /// graph becomes an overlay over that topology ([`Self::has_frozen_base`]).
    #[inline]
    pub fn is_frozen(&self) -> bool {
        match self {
            GraphBackend::Frozen(_) => true,
            GraphBackend::Recording(rg) => rg.inner().is_frozen(),
            _ => false,
        }
    }

    /// Replace the frozen topology this graph reads, and the overlay over it,
    /// with `fresh`: the topology and column stores of the file the graph was
    /// just saved to, which describe the same graph. `false` (and nothing
    /// changed) when `fresh` is not a frozen backend or this graph has no
    /// frozen base. A write-capture wrapper keeps its state; only what it wraps
    /// is replaced.
    pub(crate) fn rebase_frozen(&mut self, fresh: GraphBackend) -> bool {
        if !matches!(fresh, GraphBackend::Frozen(_)) {
            return false;
        }
        match self {
            GraphBackend::Recording(rg) => rg.inner_mut().rebase_frozen(fresh),
            GraphBackend::Frozen(_) => {
                *self = fresh;
                true
            }
            GraphBackend::Forked(forked) if forked.frozen_base().is_some() => {
                *self = fresh;
                true
            }
            _ => false,
        }
    }

    /// The frozen topology this graph reads, whether it is served directly or
    /// sits under a write overlay, looking through a write-capture wrapper.
    /// Public as a **diagnostic**: `is_frozen` turns false at the first write,
    /// this stays true until a save rebases the graph (which makes it
    /// [`is_frozen`](Self::is_frozen) again) or a whole-graph operation
    /// collapses the overlay.
    pub fn has_frozen_base(&self) -> bool {
        self.frozen_base().is_some()
    }

    /// [`Self::has_frozen_base`], with the topology.
    pub(crate) fn frozen_base(&self) -> Option<&Arc<FrozenGraph>> {
        match self {
            GraphBackend::Frozen(frozen) => Some(frozen),
            GraphBackend::Forked(forked) => forked.frozen_base(),
            GraphBackend::Recording(rg) => rg.inner().frozen_base(),
            _ => None,
        }
    }

    /// The frozen topology underneath, looking through a write-capture wrapper.
    pub(crate) fn as_frozen(&self) -> Option<&Arc<FrozenGraph>> {
        match self {
            GraphBackend::Frozen(frozen) => Some(frozen),
            GraphBackend::Recording(rg) => rg.inner().as_frozen(),
            _ => None,
        }
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
            GraphBackend::Frozen(_) => {
                self.overlay_frozen();
                self.begin_undo();
            }
            GraphBackend::Recording(rg) => rg.inner_mut().begin_undo(),
            GraphBackend::Disk(_) => {}
        }
    }

    #[inline]
    pub(crate) fn take_undo(&mut self) -> Option<Box<UndoJournal>> {
        match self {
            GraphBackend::Memory(g) => unique_heap_backend(g).take_undo(),
            GraphBackend::Forked(g) => g.take_undo(),
            GraphBackend::Mapped(g) => unique_heap_backend(g).take_undo(),
            // A frozen graph opens no journal, so there is none to take.
            GraphBackend::Frozen(_) => None,
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
            GraphBackend::Frozen(_) => None,
            GraphBackend::Recording(rg) => rg.inner_mut().undo_journal_mut(),
            GraphBackend::Disk(_) => None,
        }
    }

    /// The journal that records a statement's writes to the per-type column
    /// stores: [`Self::undo_journal_mut`] on the heap backends, and on a disk
    /// graph the cell journal its `DiskCells` checkpoint opened.
    ///
    /// Separate from `undo_journal_mut` because a disk graph's journal carries
    /// the columnar entries alone — everything else of its statement is
    /// restored from a snapshot — so the call sites that capture anything else
    /// must keep seeing `None` there.
    #[inline]
    pub(crate) fn columnar_undo_mut(&mut self) -> Option<&mut UndoJournal> {
        match self {
            GraphBackend::Disk(g) => g.statement_undo.as_deref_mut(),
            GraphBackend::Recording(rg) => rg.inner_mut().columnar_undo_mut(),
            other => other.undo_journal_mut(),
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
        if let (Some(child), Some(parent)) = (self.as_disk_mut(), parent.as_disk()) {
            child.adopt_writer_lineage(parent);
        }
    }

    /// Give an explicit copy private writer authority while retaining any
    /// mutation-workspace files needed to reproduce the parent's current
    /// logical state.
    pub(crate) fn detach_independent_copy(&mut self, parent: &Self) {
        if let (Some(child), Some(parent)) = (self.as_disk_mut(), parent.as_disk()) {
            child.detach_for_independent_copy(parent);
        }
    }

    /// Edge-storage observability for `graph_info()`: `(edges_mapped,
    /// edge_property_overlay_rows)`. Only the disk backend has a CSR or an
    /// edge-property overlay; every other backend reports `(false, 0)`.
    pub(crate) fn edge_storage_info(&self) -> (bool, usize) {
        match self {
            GraphBackend::Disk(g) => (g.csr_is_mapped(), g.edge_property_overlay_len()),
            GraphBackend::Recording(rg) => rg.inner().edge_storage_info(),
            GraphBackend::Memory(_)
            | GraphBackend::Mapped(_)
            | GraphBackend::Frozen(_)
            | GraphBackend::Forked(_) => (false, 0),
        }
    }

    /// Hand a freshly loaded mapped backend the peer counts its `.kgl` recorded.
    /// A no-op for every other backend.
    pub(crate) fn install_peer_base(
        &mut self,
        base: crate::graph::storage::mapped::peer_base::PeerBase,
    ) {
        match self {
            GraphBackend::Mapped(mapped) => unique_heap_backend(mapped).install_peer_base(base),
            GraphBackend::Frozen(frozen) => Arc::make_mut(frozen).install_peer_base(base),
            _ => {}
        }
    }

    /// Hold the disk materialization arenas for one read-query lifetime.
    /// Heap/mapped backends do not materialize through shared arenas.
    pub(crate) fn begin_query(&self) -> Option<DiskQueryGuard> {
        match self {
            GraphBackend::Disk(graph) => Some(graph.begin_query()),
            GraphBackend::Recording(graph) => graph.inner().begin_query(),
            GraphBackend::Memory(_)
            | GraphBackend::Mapped(_)
            | GraphBackend::Frozen(_)
            | GraphBackend::Forked(_) => None,
        }
    }

    /// Record that node `idx` was upserted, for the WAL capture wrapper. Used
    /// by mutation paths that write through a side channel (the columnar master
    /// `ColumnStore`) and so bypass the recorded
    /// [`GraphWrite::node_weight_mut`] — without this the mutation is never
    /// captured for the WAL (the recorded path only sees the silent
    /// handle-refresh sweep).
    #[inline]
    pub fn note_recorded_node_upsert(&mut self, idx: NodeIndex) {
        if let GraphBackend::Recording(rg) = self {
            rg.note_node_upsert(idx);
        }
    }

    /// Turn before-image capture on or off on the write-capture wrapper.
    #[inline]
    pub(crate) fn set_capture_before(&mut self, on: bool) {
        if let GraphBackend::Recording(rg) = self {
            rg.set_capture_before(on);
        }
    }

    /// Whether writes on this backend capture before-images.
    ///
    /// The gate the **side-channel choke points** test before doing any work:
    /// they sit on hot write paths and must cost a bool read, not a state read,
    /// when enrichment is off (which is the default).
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

    /// Record that node `idx`'s secondary labels changed, for the WAL capture
    /// wrapper.
    ///
    /// Secondary labels live in `DirGraph::secondary_label_index`, above this
    /// backend — `NodeData` carries none — so *no* `GraphWrite` call describes a
    /// label change and the recorded seam cannot infer one. `DirGraph`'s label
    /// choke points call this instead; without it a durable graph silently lost
    /// every `CREATE (n:A:B)` / `SET n:B` on WAL replay while keeping the node's
    /// properties.
    #[inline]
    pub fn note_recorded_node_labels(&mut self, idx: NodeIndex) {
        if let GraphBackend::Recording(rg) = self {
            rg.note_node_labels(idx);
        }
    }

    /// Record a declaration made above this backend for the WAL capture
    /// wrapper, for the same reason as the labels above: identity-field
    /// spellings, parent types, the ontology, the schema stamp, spatial
    /// configs, user indexes and constraints all live in `DirGraph`, not in
    /// this backend, so no `GraphWrite` call carries them and a crash before
    /// the first checkpoint recovered the rows without what was declared
    /// about them.
    #[inline]
    pub fn note_recorded_declaration(&mut self, op: crate::graph::wal::MutationOp) {
        if let GraphBackend::Recording(rg) = self {
            rg.note_declaration(op);
        }
    }

    /// Swap in a rebuilt heap petgraph, **preserving this backend's variant
    /// and any write-capture wrapper around it**. Returns `false` for `Disk`,
    /// whose CSR arrays are not a `StableDiGraph`; the caller must treat that
    /// as "not rebuilt".
    ///
    /// Exists because `DirGraph::vacuum` used to assign
    /// `GraphBackend::Memory(...)` unconditionally, which silently downgraded a
    /// `Mapped` graph to heap storage and — worse — *dropped the `Recording`
    /// wrapper*, so a durable graph stopped write-ahead logging for the rest of
    /// the session with no error.
    ///
    /// The wrapper survives but its op buffer does not: buffered ops are keyed
    /// by `NodeIndex` and a vacuum remaps every index, so callers must flush the
    /// log *before* vacuuming.
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
            GraphBackend::Frozen(_) => {
                self.promote_frozen();
                self.replace_heap_graph(new)
            }
            // Reached only through `vacuum`, which collapses an overlay first.
            GraphBackend::Forked(_) => unreachable!("replace_heap_graph on a forked backend"),
            GraphBackend::Disk(_) => false,
        }
    }

    /// Move the heap `StableDiGraph` out of the backend, leaving an empty one
    /// in its place. `None` on **disk**, whose CSR arrays are not a
    /// `StableDiGraph` — the same "not rebuilt" signal
    /// [`replace_heap_graph`](Self::replace_heap_graph) returns `false` for.
    ///
    /// Exists so `DirGraph::vacuum` can *relocate* node and edge weights into
    /// the compacted graph instead of deep-cloning them. The backend is left
    /// holding an empty graph and now-stale derived caches; the sole caller
    /// replaces it a few statements later and nothing reads it in between.
    pub(crate) fn take_heap_graph(&mut self) -> Option<StableDiGraph<NodeData, EdgeData>> {
        match self {
            GraphBackend::Memory(g) => Some(std::mem::take(&mut unique_heap_backend(g).inner)),
            GraphBackend::Mapped(g) => Some(std::mem::take(&mut unique_heap_backend(g).inner)),
            GraphBackend::Recording(rg) => rg.inner_mut().take_heap_graph(),
            GraphBackend::Frozen(_) => {
                self.promote_frozen();
                self.take_heap_graph()
            }
            GraphBackend::Forked(_) => unreachable!("take_heap_graph on a forked backend"),
            GraphBackend::Disk(_) => None,
        }
    }

    /// The inner `StableDiGraph` when this is a plain heap `Memory` backend,
    /// and `None` for every other variant — including a copy-on-write
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
            | GraphBackend::Frozen(_)
            | GraphBackend::Recording(_)
            | GraphBackend::Disk(_) => None,
        }
    }

    /// Borrow the inner heap `StableDiGraph` for petgraph algorithms
    /// (e.g. `kosaraju_scc`) that require concrete petgraph types.
    /// Disk **and** `Forked` panic — callers must gate on
    /// [`GraphRead::is_disk`] and [`is_forked`](Self::is_forked) first.
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
            GraphBackend::Frozen(_) => {
                unimplemented!("Frozen backend: as_stable_digraph — gate on is_frozen()")
            }
            GraphBackend::Recording(rg) => rg.inner().as_stable_digraph(),
        }
    }

    /// Closure-based hot-path iteration over all live edges, yielding
    /// `(source, target, connection_type)` per edge.
    ///
    /// Avoids the `Box<dyn Iterator>` + virtual `.next()` dispatch that
    /// [`GraphRead::edge_endpoint_keys`] requires: monomorphises per backend at
    /// the call site, so the compiler fully inlines the hot loop. 863M-edge
    /// benchmarks show ~40–90 s savings per sweep vs the boxed-iterator path, so
    /// prefer it in any path that walks every edge of a large graph
    /// (`compute_type_connectivity`, cache rebuild, bulk index builds).
    ///
    /// The `Recording` variant forwards without recording — it logs only the
    /// trait-path methods.
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
            GraphBackend::Frozen(g) => g.for_each_endpoint_key(f),
            GraphBackend::Forked(g) => g.for_each_edge_endpoint_key(f),
            GraphBackend::Recording(rg) => {
                rg.inner().for_each_edge_endpoint_key(f);
            }
        }
    }

    /// Iterate only edges whose connection type matches `conn_type`, yielding
    /// `(src, tgt, edge_idx, properties)` per match. The callback returns `true`
    /// to continue or `false` to stop, so a caller collecting a bounded prefix
    /// doesn't pay for the remaining matches; `properties` is the empty slice
    /// when the edge has no custom ones.
    ///
    /// Avoids the disk backend's per-edge `Box<EdgeData>` arena push by reading
    /// `edge_endpoints` + `edge_properties` directly, and is O(matching edges)
    /// there when the persisted `conn_type_index_*` inverted index is present
    /// (a graph converted by `enable_disk_mode` has none and falls back to an
    /// O(CSR edges) sweep — still every matching edge, just not indexed). On
    /// Memory/Mapped the petgraph iterator already hands out `&EdgeData` into
    /// resident storage, so there is no arena cost either way.
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
            GraphBackend::Frozen(g) => g.for_each_edge_of_type(conn_type, f),
            GraphBackend::Forked(g) => g.for_each_edge_of_conn_type(conn_type, f),
            GraphBackend::Recording(rg) => {
                rg.inner().for_each_edge_of_conn_type(conn_type, f);
            }
        }
    }

    /// Borrow the in-memory and mapped backends' immutable peer-count
    /// histogram. Other backends keep their existing owned-result fallback.
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
            GraphBackend::Mapped(graph) => Some(graph.peer_counts(conn_type, dir)),
            GraphBackend::Frozen(graph) => Some(graph.peer_counts(conn_type, dir)),
            // Cold by design: a fork keeps no peer-count cache of its own, so
            // the writer can never publish a count into a reader's snapshot.
            // The owned fallback is correct, just uncached. An overlay over a
            // frozen file with no edge write shares the file's own cache,
            // which a write cannot reach.
            GraphBackend::Forked(forked) => forked.untouched_peer_counts(conn_type, dir),
            GraphBackend::Disk(_) => None,
        })
    }
}

/// The backend variant that matches the base an overlay was folded from.
impl From<Folded> for GraphBackend {
    fn from(folded: Folded) -> Self {
        match folded {
            Folded::Memory(g) => GraphBackend::Memory(Arc::new(g)),
            Folded::Mapped(g) => GraphBackend::Mapped(Arc::new(g)),
        }
    }
}

impl Clone for GraphBackend {
    fn clone(&self) -> Self {
        #[cfg(test)]
        BACKEND_CLONE_COUNT.set(BACKEND_CLONE_COUNT.get() + 1);
        match self {
            // **The fork site.** Instead of deep-copying every node and edge,
            // hand the writer an overlay over the same base; the reader's
            // `Arc<MemoryGraph>` (or `Arc<MappedGraph>`) is left byte-for-byte
            // untouched. `can_fork` is
            // the slot-identity precondition (a slot mirror that knows the free
            // list, so the overlay allocates what the fold-back's `add_node`
            // will); a base that fails it keeps the deep copy — slower, never
            // wrong.
            //
            // ⚠ `deep_clone()` on the fallback, never `g.clone()`: `g` is an
            // `Arc` handle, so `.clone()` on it is a refcount bump — one
            // character away, and it would share a backend that every later
            // write mutates in place under the reader.
            GraphBackend::Memory(g) if can_fork(&**g) => {
                GraphBackend::Forked(Box::new(ForkedGraph::new(Arc::clone(g))))
            }
            GraphBackend::Memory(g) => {
                #[cfg(test)]
                note_nodes_copied(g.inner().node_count());
                GraphBackend::Memory(Arc::new(g.deep_clone()))
            }
            GraphBackend::Mapped(g) if can_fork(&**g) => {
                GraphBackend::Forked(Box::new(ForkedGraph::new(Arc::clone(g))))
            }
            GraphBackend::Mapped(g) => {
                #[cfg(test)]
                note_nodes_copied(g.inner().node_count());
                GraphBackend::Mapped(Arc::new(g.deep_clone()))
            }
            // Forking a fork: same base, only the delta is duplicated — until
            // the delta is large enough that one collapse is cheaper than
            // copying it again at every later fork.
            GraphBackend::Forked(g) => {
                if g.delta_exceeds_clone_cap() {
                    if let Ok(folded) = g.to_folded_graph() {
                        #[cfg(test)]
                        note_nodes_copied(folded.node_count());
                        return GraphBackend::from(folded);
                    }
                }
                #[cfg(test)]
                note_nodes_copied(g.overlay_node_count());
                GraphBackend::Forked(Box::new((**g).clone()))
            }
            // Immutable, so a reader and a writer can share it: the writer
            // promotes its own handle at its first write.
            GraphBackend::Frozen(g) => GraphBackend::Frozen(Arc::clone(g)),
            GraphBackend::Disk(dg) => {
                #[cfg(test)]
                note_nodes_copied(dg.node_count());
                GraphBackend::Disk(dg.clone())
            }
            GraphBackend::Recording(rg) => GraphBackend::Recording(Box::new((**rg).clone())),
        }
    }
}

// Every serializable variant emits the bare inner `StableDiGraph`, so which
// backend holds a graph has no effect on the persisted format.
impl Serialize for GraphBackend {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            GraphBackend::Memory(g) => g.serialize(serializer),
            GraphBackend::Mapped(g) => g.serialize(serializer),
            // Serialization needs one concrete `StableDiGraph`, so a frozen
            // topology is copied into a throwaway heap graph first. O(V+E)
            // transient memory, paid by a save of a loaded frozen graph.
            // A section that stayed mapped has not been checksummed yet, and
            // the copy below would write it into a fresh file as if it were good.
            GraphBackend::Frozen(g) => {
                g.verify_sections().map_err(serde::ser::Error::custom)?;
                g.to_stable_graph().serialize(serializer)
            }
            // Serialization needs one concrete `StableDiGraph`, so the overlay is
            // folded into a throwaway copy. O(V+E) — but so is writing the file,
            // and the bytes are identical to the unforked graph's. Over a frozen
            // base the copy is the same transient heap graph the `Frozen` arm
            // builds (its sections are checksummed first), plus the overlay.
            GraphBackend::Forked(g) => g
                .to_folded_graph()
                .map_err(serde::ser::Error::custom)?
                .serialize(serializer),
            GraphBackend::Disk(_) => Err(serde::ser::Error::custom(
                "Disk backend does not support serialization",
            )),
            // Capture wrapper is transparent — serialize as the wrapped
            // backend, recursively hitting the Disk error arm if it is Disk.
            GraphBackend::Recording(rg) => rg.inner().serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for GraphBackend {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let g = StableDiGraph::<NodeData, EdgeData>::deserialize(deserializer)?;
        Ok(GraphBackend::Memory(Arc::new(
            MemoryGraph::from_deserialized(g),
        )))
    }
}

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
            GraphBackend::Frozen(g) => write!(f, "{g:?}"),
            GraphBackend::Forked(g) => write!(f, "{g:?}"),
            GraphBackend::Disk(_) => write!(f, "Disk(placeholder)"),
            GraphBackend::Recording(rg) => write!(f, "Recording({:?})", rg.inner()),
        }
    }
}

// GraphRead / GraphWrite dispatcher impls — per-variant forwarding. The real
// impls live on each backend: `impls.rs` (Memory/Mapped/Disk), `forked.rs`,
// `recording.rs`.

use crate::datatypes::values::Value;
use std::collections::HashMap;

/// Out-of-line non-`Memory` arms of the hottest `GraphRead` adjacency methods: the trait method keeps
/// only the heap arm inline, so its size does not grow with the number of backends.
/// The overlay's arm of `node_type_of`, a direct call: inlined it grows every site
/// that reads a node's type, and the dispatch above it is hot in the matcher.
#[inline(never)]
fn forked_node_type_of(graph: &ForkedGraph, idx: NodeIndex) -> Option<InternedKey> {
    GraphRead::node_type_of(graph, idx)
}

impl GraphBackend {
    /// The arms of [`GraphRead::node_type_of`] that are not a plain petgraph,
    /// out of line: with the overlay's and the frozen base's bodies inlined at
    /// every call, the matcher's per-node reads of a heap graph measured +5-9 %.
    #[inline(never)]
    fn node_type_of_other(&self, idx: NodeIndex) -> Option<InternedKey> {
        match self {
            Self::Memory(g) => GraphRead::node_type_of(&**g, idx),
            Self::Mapped(g) => GraphRead::node_type_of(&**g, idx),
            Self::Forked(g) => GraphRead::node_type_of(g.as_ref(), idx),
            Self::Frozen(g) => GraphRead::node_type_of(&**g, idx),
            Self::Disk(g) => GraphRead::node_type_of(g.as_ref(), idx),
            Self::Recording(rg) => GraphRead::node_type_of(rg.as_ref(), idx),
        }
    }

    #[inline(never)]
    fn edges_directed_filtered_other(
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
            Self::Frozen(g) => GraphRead::edges_directed_filtered(&**g, idx, dir, conn_type_filter),
            Self::Disk(g) => {
                GraphRead::edges_directed_filtered(g.as_ref(), idx, dir, conn_type_filter)
            }
            Self::Recording(rg) => {
                GraphRead::edges_directed_filtered(rg.as_ref(), idx, dir, conn_type_filter)
            }
        }
    }

    #[inline(never)]
    fn edges_directed_other(
        &self,
        idx: NodeIndex,
        dir: petgraph::Direction,
    ) -> crate::graph::core::iterators::GraphEdges<'_> {
        match self {
            Self::Memory(g) => GraphRead::edges_directed(&**g, idx, dir),
            Self::Forked(g) => GraphRead::edges_directed(g.as_ref(), idx, dir),
            Self::Mapped(g) => GraphRead::edges_directed(&**g, idx, dir),
            Self::Frozen(g) => GraphRead::edges_directed(&**g, idx, dir),
            Self::Disk(g) => GraphRead::edges_directed(g.as_ref(), idx, dir),
            Self::Recording(rg) => GraphRead::edges_directed(rg.as_ref(), idx, dir),
        }
    }

    #[inline(never)]
    fn neighbors_directed_other(
        &self,
        idx: NodeIndex,
        dir: petgraph::Direction,
    ) -> crate::graph::core::iterators::GraphNeighbors<'_> {
        match self {
            Self::Memory(g) => GraphRead::neighbors_directed(&**g, idx, dir),
            Self::Forked(g) => GraphRead::neighbors_directed(g.as_ref(), idx, dir),
            Self::Mapped(g) => GraphRead::neighbors_directed(&**g, idx, dir),
            Self::Frozen(g) => GraphRead::neighbors_directed(&**g, idx, dir),
            Self::Disk(g) => GraphRead::neighbors_directed(g.as_ref(), idx, dir),
            Self::Recording(rg) => GraphRead::neighbors_directed(rg.as_ref(), idx, dir),
        }
    }

    #[inline(never)]
    fn neighbors_undirected_other(
        &self,
        idx: NodeIndex,
    ) -> crate::graph::core::iterators::GraphNeighbors<'_> {
        match self {
            Self::Memory(g) => GraphRead::neighbors_undirected(&**g, idx),
            Self::Forked(g) => GraphRead::neighbors_undirected(g.as_ref(), idx),
            Self::Mapped(g) => GraphRead::neighbors_undirected(&**g, idx),
            Self::Frozen(g) => GraphRead::neighbors_undirected(&**g, idx),
            Self::Disk(g) => GraphRead::neighbors_undirected(g.as_ref(), idx),
            Self::Recording(rg) => GraphRead::neighbors_undirected(rg.as_ref(), idx),
        }
    }
}

impl GraphRead for GraphBackend {
    type NodeIndicesIter<'a> = crate::graph::core::iterators::GraphNodeIndices<'a>;
    type EdgeIndicesIter<'a> = crate::graph::core::iterators::GraphEdgeIndices<'a>;
    type EdgesIter<'a> = crate::graph::core::iterators::GraphEdges<'a>;
    type EdgeReferencesIter<'a> = crate::graph::core::iterators::GraphEdgeReferences<'a>;
    type EdgesConnectingIter<'a> = crate::graph::core::iterators::GraphEdgesConnecting<'a>;
    type NeighborsIter<'a> = crate::graph::core::iterators::GraphNeighbors<'a>;

    /// Dispatch **once**, not twice: the trait default calls
    /// `self.node_weight(idx)` and `self.column_store(..)`, each a full match on
    /// this six-variant enum, so a scan paid two dispatches per node. Matching
    /// here and delegating to the concrete backend's `node_view` inlines the
    /// store probe into the same call.
    #[inline]
    fn node_view(&self, idx: NodeIndex) -> Option<crate::graph::storage::NodeView<'_>> {
        match self {
            Self::Memory(g) => GraphRead::node_view(&**g, idx),
            Self::Forked(g) => GraphRead::node_view(g.as_ref(), idx),
            Self::Mapped(g) => GraphRead::node_view(&**g, idx),
            Self::Frozen(g) => GraphRead::node_view(&**g, idx),
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
            Self::Frozen(g) => GraphRead::column_store(&**g, type_key),
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
            Self::Frozen(g) => GraphRead::column_stores_iter(&**g),
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
            Self::Frozen(g) => GraphRead::node_count(&**g),
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
            Self::Frozen(g) => GraphRead::edge_count(&**g),
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
            Self::Frozen(g) => GraphRead::node_bound(&**g),
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
            Self::Frozen(g) => GraphRead::edge_bound(&**g),
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
            Self::Mapped(_) | Self::Frozen(_) => true,
            Self::Forked(g) => g.base_is_mapped(),
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
            Self::Mapped(g) => GraphRead::node_type_of(&**g, idx),
            Self::Forked(g) => forked_node_type_of(g, idx),
            other => other.node_type_of_other(idx),
        }
    }

    #[inline(always)]
    fn node_labels_of(&self, idx: NodeIndex) -> Vec<InternedKey> {
        match self {
            Self::Memory(g) => GraphRead::node_labels_of(&**g, idx),
            Self::Forked(g) => GraphRead::node_labels_of(g.as_ref(), idx),
            Self::Mapped(g) => GraphRead::node_labels_of(&**g, idx),
            Self::Frozen(g) => GraphRead::node_labels_of(&**g, idx),
            Self::Disk(g) => GraphRead::node_labels_of(g.as_ref(), idx),
            Self::Recording(rg) => GraphRead::node_labels_of(rg.as_ref(), idx),
        }
    }

    #[inline(always)]
    fn node_weight(&self, idx: NodeIndex) -> Option<RecordRef<'_, NodeData>> {
        match self {
            Self::Memory(g) => GraphRead::node_weight(&**g, idx),
            Self::Forked(g) => GraphRead::node_weight(g.as_ref(), idx),
            Self::Mapped(g) => GraphRead::node_weight(&**g, idx),
            Self::Frozen(g) => GraphRead::node_weight(&**g, idx),
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
            Self::Frozen(g) => GraphRead::get_node_property(&**g, idx, key),
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
            Self::Frozen(g) => GraphRead::get_node_id(&**g, idx),
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
            Self::Frozen(g) => GraphRead::get_node_title(&**g, idx),
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
            Self::Frozen(g) => GraphRead::str_prop_eq(&**g, idx, key, target),
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
            Self::Frozen(g) => GraphRead::node_indices(&**g),
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
            Self::Frozen(g) => GraphRead::edge_indices(&**g),
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
            Self::Frozen(g) => GraphRead::edge_references(&**g),
            Self::Disk(g) => GraphRead::edge_references(g.as_ref()),
            Self::Recording(rg) => GraphRead::edge_references(rg.as_ref()),
        }
    }

    #[inline]
    fn edge_weights<'a>(&'a self) -> Box<dyn Iterator<Item = RecordRef<'a, EdgeData>> + 'a> {
        match self {
            Self::Memory(g) => GraphRead::edge_weights(&**g),
            Self::Forked(g) => GraphRead::edge_weights(g.as_ref()),
            Self::Mapped(g) => GraphRead::edge_weights(&**g),
            Self::Frozen(g) => GraphRead::edge_weights(&**g),
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
        if let Self::Memory(g) = self {
            return GraphRead::edges_directed(&**g, idx, dir);
        }
        self.edges_directed_other(idx, dir)
    }

    #[inline]
    fn edges(&self, idx: NodeIndex) -> crate::graph::core::iterators::GraphEdges<'_> {
        match self {
            Self::Memory(g) => GraphRead::edges(&**g, idx),
            Self::Forked(g) => GraphRead::edges(g.as_ref(), idx),
            Self::Mapped(g) => GraphRead::edges(&**g, idx),
            Self::Frozen(g) => GraphRead::edges(&**g, idx),
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
        if let Self::Memory(g) = self {
            return GraphRead::edges_directed_filtered(&**g, idx, dir, conn_type_filter);
        }
        self.edges_directed_filtered_other(idx, dir, conn_type_filter)
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
            Self::Frozen(g) => GraphRead::edges_connecting(&**g, a, b),
            Self::Disk(g) => GraphRead::edges_connecting(g.as_ref(), a, b),
            Self::Recording(rg) => GraphRead::edges_connecting(rg.as_ref(), a, b),
        }
    }

    #[inline]
    fn edge_weight(&self, idx: EdgeIndex) -> Option<RecordRef<'_, EdgeData>> {
        match self {
            Self::Memory(g) => GraphRead::edge_weight(&**g, idx),
            Self::Forked(g) => GraphRead::edge_weight(g.as_ref(), idx),
            Self::Mapped(g) => GraphRead::edge_weight(&**g, idx),
            Self::Frozen(g) => GraphRead::edge_weight(&**g, idx),
            Self::Disk(g) => GraphRead::edge_weight(g.as_ref(), idx),
            Self::Recording(rg) => GraphRead::edge_weight(rg.as_ref(), idx),
        }
    }

    #[inline]
    fn get_edge_property(&self, idx: EdgeIndex, key: InternedKey) -> Option<Value> {
        match self {
            Self::Memory(g) => GraphRead::get_edge_property(&**g, idx, key),
            Self::Forked(g) => GraphRead::get_edge_property(g.as_ref(), idx, key),
            Self::Mapped(g) => GraphRead::get_edge_property(&**g, idx, key),
            Self::Frozen(g) => GraphRead::get_edge_property(&**g, idx, key),
            Self::Disk(g) => GraphRead::get_edge_property(g.as_ref(), idx, key),
            Self::Recording(rg) => GraphRead::get_edge_property(rg.as_ref(), idx, key),
        }
    }

    #[inline]
    fn edge_property_list(&self, idx: EdgeIndex) -> Option<Vec<(InternedKey, Value)>> {
        match self {
            Self::Memory(g) => GraphRead::edge_property_list(&**g, idx),
            Self::Forked(g) => GraphRead::edge_property_list(g.as_ref(), idx),
            Self::Mapped(g) => GraphRead::edge_property_list(&**g, idx),
            Self::Frozen(g) => GraphRead::edge_property_list(&**g, idx),
            Self::Disk(g) => GraphRead::edge_property_list(g.as_ref(), idx),
            Self::Recording(rg) => GraphRead::edge_property_list(rg.as_ref(), idx),
        }
    }

    #[inline]
    fn find_edge(&self, a: NodeIndex, b: NodeIndex) -> Option<EdgeIndex> {
        match self {
            Self::Memory(g) => GraphRead::find_edge(&**g, a, b),
            Self::Forked(g) => GraphRead::find_edge(g.as_ref(), a, b),
            Self::Mapped(g) => GraphRead::find_edge(&**g, a, b),
            Self::Frozen(g) => GraphRead::find_edge(&**g, a, b),
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
            Self::Frozen(g) => GraphRead::edge_endpoints(&**g, idx),
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
            Self::Frozen(g) => GraphRead::edge_endpoint_keys(&**g),
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
        if let Self::Memory(g) = self {
            return GraphRead::neighbors_directed(&**g, idx, dir);
        }
        self.neighbors_directed_other(idx, dir)
    }

    #[inline]
    fn neighbors_undirected(
        &self,
        idx: NodeIndex,
    ) -> crate::graph::core::iterators::GraphNeighbors<'_> {
        if let Self::Memory(g) = self {
            return GraphRead::neighbors_undirected(&**g, idx);
        }
        self.neighbors_undirected_other(idx)
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
            Self::Frozen(g) => GraphRead::sources_for_conn_type_bounded(&**g, conn_type, max),
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
            Self::Frozen(g) => GraphRead::lookup_peer_counts(&**g, conn_type),
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
            Self::Frozen(g) => GraphRead::lookup_by_property_eq(&**g, node_type, property, value),
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
            Self::Frozen(g) => {
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
            Self::Frozen(g) => GraphRead::lookup_by_property_eq_any_type(&**g, property, value),
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
            Self::Frozen(g) => {
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
            Self::Frozen(g) => {
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
            Self::Frozen(g) => GraphRead::count_edges_filtered(
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
            Self::Frozen(g) => GraphRead::iter_peers_filtered(&**g, node, dir, conn_type),
            Self::Disk(g) => GraphRead::iter_peers_filtered(g.as_ref(), node, dir, conn_type),
            Self::Recording(rg) => {
                GraphRead::iter_peers_filtered(rg.as_ref(), node, dir, conn_type)
            }
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
            Self::Frozen(g) => Arc::make_mut(g).install_column_store(type_key, store),
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
            // A column store is property data the backend owns, not topology:
            // reaching it for a spill or a cell write needs no heap graph. A
            // statement that journals its writes has promoted at `begin_undo`.
            Self::Frozen(g) => Arc::make_mut(g).column_store_mut(type_key),
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
            Self::Frozen(g) => Arc::make_mut(g).take_column_store(type_key),
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
            Self::Frozen(g) => Arc::make_mut(g).clear_column_stores(),
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
            Self::Frozen(_) => {
                self.overlay_frozen();
                GraphWrite::set_node_property(self, idx, key, value)
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
            Self::Frozen(_) => {
                self.overlay_frozen();
                GraphWrite::set_node_title(self, idx, value)
            }
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
            Self::Frozen(_) => {
                self.overlay_frozen();
                GraphWrite::set_node_property_if_absent(self, idx, key, value)
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
            Self::Frozen(_) => {
                self.overlay_frozen();
                GraphWrite::remove_node_property(self, idx, key)
            }
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
            Self::Frozen(_) => {
                self.overlay_frozen();
                GraphWrite::clear_node_property(self, idx, key)
            }
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
            Self::Frozen(_) => {
                self.overlay_frozen();
                GraphWrite::replace_node_properties(self, idx, pairs)
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
            Self::Frozen(_) => {
                self.overlay_frozen();
                GraphWrite::node_weight_mut(self, idx)
            }
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
            Self::Frozen(_) => {
                self.overlay_frozen();
                GraphWrite::node_weight_mut_silent(self, idx)
            }
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
            Self::Frozen(_) => {
                self.overlay_frozen();
                GraphWrite::edge_weight_mut(self, idx)
            }
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
            Self::Frozen(_) => {
                self.overlay_frozen();
                GraphWrite::add_node(self, data)
            }
            Self::Disk(g) => GraphWrite::add_node(g.as_mut(), data),
            Self::Recording(rg) => GraphWrite::add_node(rg.as_mut(), data),
        }
    }

    #[inline]
    fn remove_node(&mut self, idx: NodeIndex) -> Option<NodeData> {
        match self {
            Self::Memory(g) => GraphWrite::remove_node(unique_heap_backend(g), idx),
            Self::Forked(g) => {
                let result = GraphWrite::remove_node(g.as_mut(), idx);
                if g.delta_exceeds_write_cap() {
                    self.flatten_fork();
                }
                result
            }
            Self::Mapped(g) => GraphWrite::remove_node(unique_heap_backend(g), idx),
            Self::Frozen(_) => {
                self.overlay_frozen();
                GraphWrite::remove_node(self, idx)
            }
            Self::Disk(g) => GraphWrite::remove_node(g.as_mut(), idx),
            Self::Recording(rg) => GraphWrite::remove_node(rg.as_mut(), idx),
        }
    }

    #[inline]
    fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, data: EdgeData) -> EdgeIndex {
        match self {
            Self::Memory(g) => GraphWrite::add_edge(unique_heap_backend(g), a, b, data),
            Self::Forked(g) => {
                let result = GraphWrite::add_edge(g.as_mut(), a, b, data);
                if g.delta_exceeds_write_cap() {
                    self.flatten_fork();
                }
                result
            }
            Self::Mapped(g) => GraphWrite::add_edge(unique_heap_backend(g), a, b, data),
            Self::Frozen(_) => {
                self.overlay_frozen();
                GraphWrite::add_edge(self, a, b, data)
            }
            Self::Disk(g) => GraphWrite::add_edge(g.as_mut(), a, b, data),
            Self::Recording(rg) => GraphWrite::add_edge(rg.as_mut(), a, b, data),
        }
    }

    #[inline]
    fn remove_edge(&mut self, idx: EdgeIndex) -> Option<EdgeData> {
        match self {
            Self::Memory(g) => GraphWrite::remove_edge(unique_heap_backend(g), idx),
            Self::Forked(g) => {
                let result = GraphWrite::remove_edge(g.as_mut(), idx);
                if g.delta_exceeds_write_cap() {
                    self.flatten_fork();
                }
                result
            }
            Self::Mapped(g) => GraphWrite::remove_edge(unique_heap_backend(g), idx),
            Self::Frozen(_) => {
                self.overlay_frozen();
                GraphWrite::remove_edge(self, idx)
            }
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
            // Nothing is staged on a frozen graph, so there is nothing to flush.
            Self::Frozen(_) => {}
            Self::Disk(g) => GraphWrite::flush_pending_writes(g.as_mut()),
            Self::Recording(rg) => GraphWrite::flush_pending_writes(rg.as_mut()),
        }
    }
}
