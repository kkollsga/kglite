//! Query-lifetime materialization arenas for the disk backend.
//!
//! Disk-backed reads have no in-memory `NodeData`/`EdgeData` to borrow: a read
//! that needs one builds it from the node slot + column store and must park it
//! somewhere that outlives the call, because `node_weight` hands back a
//! reference. That parking space is this module's two arenas.
//!
//! # Reclamation protocol
//!
//! Every read that may materialize runs under a [`DiskQueryGuard`]. Guards take
//! ids from a monotonic counter, and every materialized record is stamped with
//! the counter's *current* value — an epoch strictly greater than the id of any
//! query that was already running when the record was created.
//!
//! That stamp is what makes reclamation decidable per record rather than only
//! at global quiescence:
//!
//! * A record can only be reached through the query that materialized it (the
//!   guard on that call stack), and that guard was opened *before* the record
//!   existed, so its id is strictly below the record's epoch.
//! * Therefore a record stamped `E` is unreachable once every guard with an id
//!   below `E` has been dropped — i.e. once the **oldest live query**'s id is
//!   `>= E`. Records are dropped in exactly that moment, by the guard whose
//!   release advances the oldest-live id.
//! * With no live queries at all, every stamp qualifies and both arenas are
//!   emptied — the quiescent case, which this protocol contains as a special
//!   case of the same rule.
//!
//! The bound this buys: retained records are only those materialized since the
//! **currently oldest** query started, instead of since the last moment the
//! graph was completely idle. Sustained overlapping reads — a Bolt server, a
//! thread pool, any concurrent session — never reach global quiescence, so
//! under the old rule their arenas grew for as long as the load lasted (60 MB →
//! 10.4 GB on a 43 MB graph, measured 2026-08-12). Under this one the retained
//! set is bounded by what the concurrent queries in flight have themselves
//! materialized, and it shrinks the moment they finish.
//!
//! Reclamation is disk-mode only: the heap backends borrow their records
//! directly out of the graph and never enter this module.

use std::collections::{BTreeSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::graph::schema::{EdgeData, NodeData};

/// A materialized record plus the epoch that decides when it may be dropped.
type Stamped<T> = (u64, Box<T>);

/// The disk backend's materialization arenas and their epoch bookkeeping.
///
/// Held behind an `Arc` so a [`DiskQueryGuard`] can outlive the borrow of the
/// graph that created it (mutation paths hold `&mut DirGraph` while a guard is
/// live) and still reclaim on drop.
pub(crate) struct QueryArenas {
    /// Ids of the queries currently reading. Ordered, so the oldest live query
    /// — the one that bounds reclamation — is `first()`.
    active: Mutex<BTreeSet<u64>>,
    /// Id the next query will take, and the epoch stamped on records created
    /// right now. Bumped under the `active` lock so ids are handed out in the
    /// same order queries register.
    next_id: AtomicU64,
    nodes: Mutex<VecDeque<Stamped<NodeData>>>,
    edges: Mutex<VecDeque<Stamped<EdgeData>>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl QueryArenas {
    pub(crate) fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            active: Mutex::new(BTreeSet::new()),
            next_id: AtomicU64::new(1),
            nodes: Mutex::new(VecDeque::with_capacity(capacity)),
            edges: Mutex::new(VecDeque::with_capacity(capacity)),
        })
    }

    /// Open a read query. The returned guard keeps every record materialized
    /// during its lifetime reachable; dropping it releases them.
    pub(crate) fn begin(self: &Arc<Self>) -> DiskQueryGuard {
        let mut active = lock(&self.active);
        if active.is_empty() {
            // Nothing is live, so nothing can be holding a reference: take the
            // opportunity to drop anything a previous generation left behind.
            self.clear_records();
        }
        let id = self.next_id.fetch_add(1, Ordering::AcqRel);
        active.insert(id);
        DiskQueryGuard {
            arenas: Arc::clone(self),
            id,
        }
    }

    /// Number of queries currently holding a guard. Read by the debug-only
    /// guard assertion in `disk/graph.rs` and by the arena tests.
    #[cfg(any(test, debug_assertions))]
    pub(crate) fn active_count(&self) -> usize {
        lock(&self.active).len()
    }

    /// Park a materialized node record and return a pointer that stays valid
    /// for as long as the calling query's guard is alive.
    pub(crate) fn push_node(&self, data: NodeData) -> *const NodeData {
        let boxed = Box::new(data);
        let ptr: *const NodeData = &*boxed;
        // `Acquire` pairs with `begin`'s `AcqRel` bump: whatever id the calling
        // query took is already visible here, so the stamp is strictly above it.
        let epoch = self.next_id.load(Ordering::Acquire);
        lock(&self.nodes).push_back((epoch, boxed));
        ptr
    }

    /// Edge counterpart of [`Self::push_node`].
    pub(crate) fn push_edge(&self, data: EdgeData) -> *const EdgeData {
        let boxed = Box::new(data);
        let ptr: *const EdgeData = &*boxed;
        let epoch = self.next_id.load(Ordering::Acquire);
        lock(&self.edges).push_back((epoch, boxed));
        ptr
    }

    #[cfg(test)]
    pub(crate) fn node_len(&self) -> usize {
        lock(&self.nodes).len()
    }

    #[cfg(test)]
    pub(crate) fn edge_len(&self) -> usize {
        lock(&self.edges).len()
    }

    /// Drop every record unconditionally. Callers must hold `&mut DiskGraph`,
    /// which the borrow checker already orders after any outstanding `&self`
    /// materialization borrow.
    pub(crate) fn clear_all(&self) {
        let _active = lock(&self.active);
        self.clear_records();
    }

    /// Drop every record if no query is reading. Safe to call from any `&self`
    /// path: with a query live it is a no-op.
    pub(crate) fn reclaim_if_idle(&self) {
        let active = lock(&self.active);
        if active.is_empty() {
            self.clear_records();
        }
    }

    /// Close a query and drop the records that its departure made unreachable.
    fn release(&self, id: u64) {
        let mut active = lock(&self.active);
        let was_oldest = active.first().copied() == Some(id);
        let removed = active.remove(&id);
        debug_assert!(removed, "DiskQueryGuard released an unregistered query id");
        match active.first().copied() {
            // No reader left: every stamp qualifies.
            None => self.clear_records(),
            // The oldest live query advanced, so everything stamped at or below
            // its id belongs to queries that have all finished.
            Some(oldest) if was_oldest => self.drain_through(oldest),
            _ => {}
        }
    }

    /// Drop records stamped at or below `oldest`. Stamps are assigned in
    /// increasing order, so the reclaimable records sit at the front; a record
    /// whose stamp raced ahead of an older one simply waits for the next
    /// release, which costs retention, never correctness.
    fn drain_through(&self, oldest: u64) {
        let mut nodes = lock(&self.nodes);
        while nodes.front().is_some_and(|(epoch, _)| *epoch <= oldest) {
            nodes.pop_front();
        }
        drop(nodes);
        let mut edges = lock(&self.edges);
        while edges.front().is_some_and(|(epoch, _)| *epoch <= oldest) {
            edges.pop_front();
        }
    }

    /// Arena locks are always taken *after* `active`, so every caller here
    /// already holds it — one consistent order, no deadlock.
    fn clear_records(&self) {
        lock(&self.nodes).clear();
        lock(&self.edges).clear();
    }
}

/// Query-lifetime token for the disk materialization arenas.
///
/// The token is intentionally small and non-cloneable. It owns a handle to the
/// arenas rather than borrowing the graph, so mutation paths (which hold
/// `&mut DirGraph`) and direct readers outside the executor can hold one
/// without freezing an immutable borrow. Dropping it releases every record the
/// query materialized, unless an older query is still running.
pub struct DiskQueryGuard {
    arenas: Arc<QueryArenas>,
    id: u64,
}

impl Drop for DiskQueryGuard {
    fn drop(&mut self) {
        self.arenas.release(self.id);
    }
}
